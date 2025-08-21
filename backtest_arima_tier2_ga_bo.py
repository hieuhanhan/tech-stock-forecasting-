import warnings
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# -----------------------------
# Default constants 
# -----------------------------
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015
EPS = 1e-8

# -----------------------------
# Logging
# -----------------------------
def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Non-invertible")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
    warnings.filterwarnings("ignore", message="Too few observations")
    warnings.filterwarnings("ignore", category=FutureWarning)
# -----------------------------
# Metrics
# -----------------------------
def sharpe_ratio(simple_returns: np.ndarray) -> float:
    r = np.asarray(simple_returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    return 0.0 if std == 0.0 else (float(np.mean(r)) / std) * np.sqrt(252.0)

def max_drawdown(data: np.ndarray, input_type: str = "simple") -> float:
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return 0.0
    if input_type == "price":
        C = x / (x[0] + np.finfo(float).eps)
    elif input_type == "simple":
        C = np.cumprod(1.0 + x)
    elif input_type == "log":
        C = np.exp(np.cumsum(x))
    elif input_type == "auto":
        C = np.exp(np.cumsum(x)) if np.min(x) <= -1.0 else np.cumprod(1.0 + x)
    else:
        raise ValueError("input_type must be one of {'price','simple','log','auto'}")
    peak = np.maximum.accumulate(C)
    dd = (peak - C) / (peak + np.finfo(float).eps)
    return float(np.max(dd))

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def knee_index(F: np.ndarray) -> int:
    # F minimized (f0=-Sharpe, f1=MDD)
    f = F.copy()
    f = (f - f.min(axis=0)) / (np.ptp(f, axis=0) + 1e-12)
    d = np.sqrt((f**2).sum(axis=1))
    return int(np.argmin(d))

# -----------------------------
# Backtest engine (ARIMA + GARCH continuous sizing)
# -----------------------------
def backtest_continuous(
    test_log: np.ndarray,
    p: int, q: int, thr: float,
    retrain_interval: int,
    cost_per_turnover: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    scale_factor: float = 1000.0,
    arch_rescale_flag: bool = False,
    pos_clip: float = 1.0,
    warmup_len: int = 252,
    external_history: Optional[np.ndarray] = None,
    debug: bool = False
) -> Dict:
    """
    Mimics Tier-2 objective logic but WITHOUT penalties; evaluates on test.
    - Uses `external_history` if provided; otherwise uses the first `warmup_len` points of test as history,
      and starts forecasting from index warmup_len.
    """
    test_log = np.asarray(test_log, dtype=float)
    n_total = len(test_log)
    if external_history is not None:
        history = np.asarray(external_history, dtype=float).copy()
        start_idx = 0
    else:
        if n_total <= warmup_len + 1:
            raise ValueError(f"test length {n_total} too small for warmup_len={warmup_len}")
        history = test_log[:warmup_len].copy()
        start_idx = warmup_len

    simple_all, log_all = [], []
    total_turnover = 0.0
    prev_pos_last = 0.0

    # keep full position & per-step turnover traces
    pos_all = []
    turnover_all = []

    for start in range(start_idx, n_total, retrain_interval):
        end = min(start + retrain_interval, n_total)
        block_log = test_log[start:end]
        h = end - start
        if h <= 0:
            continue

        # Filter out ultra-low volatility blocks (consistent with objective)
        block_vol = float(np.std(block_log))
        if block_vol < float(min_block_vol):
            if debug:
                logging.debug(f"[TEST BLK {start:04d}-{end:04d}] SKIP vol={block_vol:.3e} < {min_block_vol:.3e}")
            history = np.concatenate([history, block_log])
            continue

        # Fit ARIMA + GARCH
        try:
            arima_fit = ARIMA(history, order=(int(p), 0, int(q))).fit()
            resid = np.asarray(arima_fit.resid, dtype=float).ravel()

            garch = arch_model(resid * scale_factor, mean="Zero", vol="Garch",
                               p=1, q=1, dist="normal", rescale=arch_rescale_flag)
            g_res = garch.fit(disp="off")

            f_var = np.asarray(g_res.forecast(horizon=h).variance.iloc[-1]).ravel()
            f_mu  = np.asarray(arima_fit.forecast(steps=h)).ravel()
            f_sig = np.sqrt(np.maximum(f_var / (scale_factor**2 + EPS), EPS))
        except Exception as e:
            if debug:
                logging.debug(f"[TEST FIT_FAIL] p={p}, q={q}, err={e}")
            history = np.concatenate([history, block_log])
            continue

        # Continuous sizing
        z = f_mu / (f_sig + EPS)
        pos_now = np.clip(z / (thr + EPS), -pos_clip, pos_clip)

        # Turnover costs
        signals_full = np.concatenate([[prev_pos_last], pos_now])
        turnover = np.abs(np.diff(signals_full))
        cost_vec = cost_per_turnover * turnover
        total_turnover += float(turnover.sum())

        pos_all.append(pos_now)
        turnover_all.append(turnover)

        # PnL
        block_simple = np.exp(block_log) - 1.0
        ret_simple = block_simple * pos_now - cost_vec
        ret_simple = np.clip(ret_simple, -0.9999, None)
        ret_log = np.log1p(ret_simple)

        simple_all.append(ret_simple)
        log_all.append(ret_log)

        history = np.concatenate([history, block_log])
        prev_pos_last = float(pos_now[-1])

    if not simple_all:
        return dict(
            sharpe=0.0, mdd=1.0, ann_return=0.0, ann_vol=0.0,
            turnover=0.0, n_trades=0, n_points=0
        )

    net_simple = np.concatenate(simple_all)
    net_log    = np.concatenate(log_all)

    # build full series and counts
    pos_series = np.concatenate(pos_all) if pos_all else np.array([])
    turn_series = np.concatenate(turnover_all) if turnover_all else np.array([])
    eps_tr = 1e-6  # tolerance to ignore numerical dust
    n_rebalances = int(np.count_nonzero(turn_series > eps_tr))
    approx_round_turns = float(np.sum(turn_series) / 2.0)  # optional metric

    sr = sharpe_ratio(net_simple)
    mdd = max_drawdown(net_log, input_type="log")
    ann_ret = float(np.mean(net_simple) * 252.0)
    ann_vol = float(np.std(net_simple, ddof=1) * np.sqrt(252.0))

    return dict(
        sharpe=float(sr),
        mdd=float(mdd),
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        turnover=float(total_turnover),
        n_rebalances=n_rebalances,
        approx_round_turns=approx_round_turns,
        n_points=int(net_simple.size)
    )

# -----------------------------
# Seed selection for GA vs GA+BO (from *_front.csv)
# -----------------------------
def pick_knee_from_front(front_df: pd.DataFrame, fold_id: int, interval: int, front_type: str) -> Optional[Dict]:
    sub = front_df[(front_df["fold_id"] == fold_id) &
                   (front_df["retrain_interval"] == interval) &
                   (front_df["front_type"] == front_type)]
    if sub.empty:
        return None
    # Build F for knee: f0=-Sharpe, f1=MDD (both minimized)
    F = np.vstack([[-row["sharpe"], row["mdd"]] for _, row in sub.iterrows()])
    idx = knee_index(F)
    row = sub.iloc[idx]
    return dict(p=int(row["p"]), q=int(row["q"]), threshold=float(row["threshold"]),
                sharpe=float(row["sharpe"]), mdd=float(row["mdd"]))

# -----------------------------
# Decision rule for final interval
# -----------------------------
def choose_final_interval(
    hv_df: Optional[pd.DataFrame],
    test_table: pd.DataFrame,
    intervals: List[int]
) -> Tuple[int, Dict]:
    """
    Prefer interval with large and stable HV improvement.
    Fallback: interval with highest median test Sharpe (GA+BO knee).
    Returns: (best_interval, diagnostics_dict)
    """
    diag = {}

    if hv_df is not None and not hv_df.empty:
        # Expect rows with stage in {"final_ga","final_union"}
        piv = hv_df.pivot_table(index=["fold_id", "retrain_interval"],
                                columns="stage", values="hv", aggfunc="first").reset_index()
        if {"final_ga", "final_union"}.issubset(piv.columns):
            piv["hv_gain"] = piv["final_union"] - piv["final_ga"]
            by_int = piv.groupby("retrain_interval")["hv_gain"].agg(["median", "std", "count"]).reset_index()
            by_int["score"] = by_int["median"] / (by_int["std"].replace(0, np.nan) + 1e-9)
            # Pick max score; tie-breaker on higher median, then larger count
            by_int = by_int.sort_values(["score", "median", "count"], ascending=[False, False, False])
            best_int = int(by_int.iloc[0]["retrain_interval"])
            diag["hv_summary"] = by_int.to_dict("records")
            diag["rule_used"] = "hv_gain_stability"
            return best_int, diag

    # Fallback: median test Sharpe (GA+BO knee only)
    sub = test_table[test_table["source"] == "GA+BO_knee"]
    by_int = sub.groupby("retrain_interval")["test_sharpe"].median().reset_index()
    by_int = by_int.sort_values(["test_sharpe"], ascending=False)
    best_int = int(by_int.iloc[0]["retrain_interval"])
    diag["sharpe_median_by_interval"] = by_int.to_dict("records")
    diag["rule_used"] = "median_test_sharpe"
    return best_int, diag

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Backtest Tier-2 GA vs GA+BO across intervals on test set")
    ap.add_argument("--tier2-front-csv", required=True,
                    help="Path to *_front.csv produced by Tier-2 (contains GA and GA+BO fronts).")
    ap.add_argument("--hv-csv", default="",
                    help="Optional path to *_hv.csv with final_ga/final_union hypervolumes.")
    ap.add_argument("--test-csv", required=True,
                    help="Path to TEST CSV with column 'Log_Returns' (global test set).")
    ap.add_argument("--intervals", default="10,20,42",
                    help="Comma-separated retrain intervals to evaluate.")
    ap.add_argument("--cost-per-turnover", type=float, default=BASE_COST + SLIPPAGE)
    ap.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--arch-rescale-flag", action="store_true")
    ap.add_argument("--pos-clip", type=float, default=1.0)
    ap.add_argument("--warmup-len", type=int, default=252,
                    help="If no external history is provided, use this many test points to warm-start.")
    ap.add_argument("--outdir", default="data/backtest_tier2",
                    help="Directory to write reports and charts.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    setup_logging(args.debug)
    suppress_warnings()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # --- Resume/checkpoint setup ---
    result_csv = outdir / "backtest_results_ga_vs_gabo.csv"
    completed = set()
    if result_csv.exists():
        try:
            _prev = pd.read_csv(result_csv)
            if not _prev.empty:
                for _, r in _prev.iterrows():
                    completed.add((int(r["fold_id"]), int(r["retrain_interval"]), str(r["source"])))
                logging.info(f"[RESUME] Found {len(_prev)} rows. Will skip completed jobs.")
        except Exception as e:
            logging.warning(f"[RESUME] Could not read existing result CSV ({e}). Resume disabled.")

    # --- Load inputs (directly with pandas) ---
    front_df = pd.read_csv(args.tier2_front_csv)
    required_cols = {"fold_id", "retrain_interval", "front_type", "p", "q", "threshold", "sharpe", "mdd"}
    if not required_cols.issubset(front_df.columns):
        raise ValueError(f"tier2 front CSV missing required columns: {required_cols - set(front_df.columns)}")

    hv_df = None
    if args.hv_csv:
        try:
            hv_df = pd.read_csv(args.hv_csv)
        except FileNotFoundError:
            logging.warning(f"[WARN] HV CSV not found: {args.hv_csv}")

    test_df = pd.read_csv(args.test_csv)
    if "Log_Returns" not in test_df.columns:
        raise ValueError("TEST CSV must contain column 'Log_Returns'.")
    test_log = test_df["Log_Returns"].fillna(0).to_numpy(dtype=float)

    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]

    fold_ids = sorted(front_df["fold_id"].unique().tolist())

    # Backtest table rows
    rows: List[Dict] = []

    pairs = [(fid, interval) for fid in fold_ids for interval in intervals]
    if not pairs:
        raise RuntimeError("No (fold, interval) pairs to backtest.")

    for fid, interval in tqdm(pairs, desc="Backtesting folds × intervals", unit="job"):
        if args.debug:
            tqdm.write(f"Running fold_id={fid}, interval={interval}")

        # If both sources completed for this pair -> skip fast
        if ((fid, interval, "GA_knee") in completed) and ((fid, interval, "GA+BO_knee") in completed):
            if args.debug:
                logging.debug(f"[SKIP] Already completed: fold={fid}, interval={interval}")
            continue

        ga_pick = pick_knee_from_front(front_df, fid, interval, "GA")
        u_pick = pick_knee_from_front(front_df, fid, interval, "GA+BO")

        for label, pick in [("GA_knee", ga_pick), ("GA+BO_knee", u_pick)]:
            if (fid, interval, label) in completed:
                if args.debug:
                    logging.debug(f"[SKIP] Already completed: fold={fid}, interval={interval}, src={label}")
                continue

            if pick is None:
                if args.debug:
                    logging.debug(f"[MISS] fold={fid}, int={interval}, source={label} not found.")
                continue

            bt = backtest_continuous(
                test_log=test_log,
                p=int(pick["p"]), q=int(pick["q"]), thr=float(pick["threshold"]),
                retrain_interval=interval,
                cost_per_turnover=args.cost_per_turnover,
                min_block_vol=args.min_block_vol,
                scale_factor=args.scale_factor,
                arch_rescale_flag=args.arch_rescale_flag,
                pos_clip=args.pos_clip,
                warmup_len=args.warmup_len,
                external_history=None,
                debug=args.debug
            )
            result_row = {
                "fold_id": int(fid),
                "retrain_interval": int(interval),
                "source": label,
                "p": int(pick["p"]),
                "q": int(pick["q"]),
                "threshold": float(pick["threshold"]),
                "val_sharpe_front": float(pick["sharpe"]),
                "val_mdd_front": float(pick["mdd"]),
                "test_sharpe": float(bt["sharpe"]),
                "test_mdd": float(bt["mdd"]),
                "test_ann_return": float(bt["ann_return"]),
                "test_ann_vol": float(bt["ann_vol"]),
                "test_turnover": float(bt["turnover"]),
                "test_n_points": int(bt["n_points"])
            }

            # 1) Cache in-memory for summary at the end
            rows.append(result_row)

            # 2) Append to CSV immediately (checkpoint)
            row_df = pd.DataFrame([result_row])
            header_needed = (not result_csv.exists()) or os.path.getsize(result_csv) == 0
            row_df.to_csv(result_csv, mode="a", index=False, header=header_needed)

            # 3) Mark completed (for resume)
            completed.add((fid, interval, label))

    # Build final result_df
    if rows:
        result_df = pd.DataFrame(rows)
    else:
        if not result_csv.exists():
            raise RuntimeError("No backtest rows generated and no existing result CSV found.")
        result_df = pd.read_csv(result_csv)

    # Normalize & save a clean final table (overwrite to ensure consistent ordering)
    result_df = result_df[
        ["fold_id","retrain_interval","source","p","q","threshold","val_sharpe_front","val_mdd_front",
         "test_sharpe","test_mdd","test_ann_return","test_ann_vol","test_turnover","test_n_points"]
    ].sort_values(["retrain_interval","fold_id","source"]).reset_index(drop=True)
    result_df.to_csv(result_csv, index=False)
    logging.info(f"[OK] Saved backtest table -> {result_csv}")

    # Summary table (by interval & source)
    summary = (result_df
               .groupby(["retrain_interval", "source"], as_index=False)
               .agg(test_sharpe_median=("test_sharpe", "median"),
                    test_mdd_median=("test_mdd", "median"),
                    test_sharpe_mean=("test_sharpe", "mean"),
                    test_mdd_mean=("test_mdd", "mean"),
                    n=("fold_id", "count")))
    summary_csv = outdir / "backtest_summary_by_interval.csv"
    summary.to_csv(summary_csv, index=False)
    logging.info(f"[OK] Saved summary -> {summary_csv}")

    # Choose final interval
    best_interval, diag = choose_final_interval(hv_df, result_df, intervals)
    choice_json = outdir / "final_interval_choice.json"
    with open(choice_json, "w") as f:
        json.dump({"best_interval": best_interval, "diagnostics": diag}, f, indent=2)
    logging.info(f"[OK] Final interval = {best_interval} (details -> {choice_json})")

    # Small human-readable printout
    print("\n=== Backtest complete ===")
    print(f"Results table : {result_csv}")
    print(f"Summary table : {summary_csv}")
    print(f"Final choice  : interval={best_interval}")
    if "rule_used" in diag:
        print(f"Decision rule : {diag['rule_used']}")

if __name__ == "__main__":
    main()