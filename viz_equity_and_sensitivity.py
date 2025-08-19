#!/usr/bin/env python3
# tier2_tools.py
"""
Utilities for Tier-2 post-analysis:
- Subcommands:
  1) viz         : Plot equity curve + rolling metrics for a chosen config.
  2) sensitivity : Sweep cost & threshold multipliers around a chosen config.

Parameter sources (choose one):
  A) --front-csv + (fold_id, interval, source) -> auto-pick knee from front.
  B) --backtest-csv + (fold_id, interval, source_label) -> read p,q,thr from backtest results.

Notes:
- We recompute the test trace (positions & PnL) to draw charts or run sensitivity.
- If you want zero recomputation, modify your main backtest to dump traces and point these tools to those files.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# -------- Defaults --------
EPS = 1e-8
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015

# -------- Logging --------
def setup_logging(debug: bool):
    logging.basicConfig(level=(logging.DEBUG if debug else logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

# -------- IO helpers --------
def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

# -------- Metrics --------
def sharpe_ratio(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if r.size < 2: return 0.0
    sd = float(np.std(r, ddof=1))
    return 0.0 if sd == 0.0 else float(np.mean(r))/sd*np.sqrt(252.0)

def max_drawdown_from_log(logr: np.ndarray) -> float:
    C = np.exp(np.cumsum(np.asarray(logr, dtype=float)))
    peak = np.maximum.accumulate(C)
    dd = (peak - C) / (peak + EPS)
    return float(np.max(dd))

# -------- Knee picker from front --------
def knee_index(F: np.ndarray) -> int:
    f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    d = np.sqrt((f**2).sum(axis=1))
    return int(np.argmin(d))

def pick_knee_from_front(front_df: pd.DataFrame, fold_id: int, interval: int, front_type: str) -> Optional[Dict]:
    sub = front_df[(front_df["fold_id"]==fold_id) &
                   (front_df["retrain_interval"]==interval) &
                   (front_df["front_type"]==front_type)]
    if sub.empty: return None
    F = np.vstack([[-row["sharpe"], row["mdd"]] for _, row in sub.iterrows()])
    idx = knee_index(F)
    row = sub.iloc[idx]
    return dict(p=int(row["p"]), q=int(row["q"]), threshold=float(row["threshold"]),
                val_sharpe=float(row["sharpe"]), val_mdd=float(row["mdd"]))

# -------- Read params from backtest table --------
def pick_from_backtest(backtest_df: pd.DataFrame, fold_id: int, interval: int, source_label: str) -> Optional[Dict]:
    """
    backtest_df rows should include: fold_id, retrain_interval, source, p,q,threshold
    Typical sources: "GA_knee" or "GA+BO_knee".
    """
    sub = backtest_df[(backtest_df["fold_id"]==fold_id) &
                      (backtest_df["retrain_interval"]==interval) &
                      (backtest_df["source"]==source_label)]
    if sub.empty: return None
    row = sub.iloc[0]
    return dict(p=int(row["p"]), q=int(row["q"]), threshold=float(row["threshold"]))

# -------- Backtest engines (trace & single-run) --------
def backtest_continuous_trace(
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
    Returns full traces to plot: positions, ret_simple, ret_log, plus summary stats.
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

    pos_trace: List[float] = []
    ret_s_trace: List[float] = []
    ret_l_trace: List[float] = []
    prev_pos_last = 0.0
    total_turnover = 0.0

    for start in range(start_idx, n_total, retrain_interval):
        end = min(start + retrain_interval, n_total)
        block_log = test_log[start:end]
        h = end - start
        if h <= 0:
            continue

        block_vol = float(np.std(block_log))
        if block_vol < float(min_block_vol):
            if debug:
                logging.debug(f"[TEST BLK {start:04d}-{end:04d}] SKIP vol={block_vol:.3e} < {min_block_vol:.3e}")
            history = np.concatenate([history, block_log])
            continue

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

        z = f_mu / (f_sig + EPS)
        pos_now = np.clip(z / (thr + EPS), -pos_clip, pos_clip)

        signals_full = np.concatenate([[prev_pos_last], pos_now])
        turnover = np.abs(np.diff(signals_full))
        cost_vec = cost_per_turnover * turnover
        total_turnover += float(turnover.sum())

        block_simple = np.exp(block_log) - 1.0
        ret_simple = np.clip(block_simple * pos_now - cost_vec, -0.9999, None)
        ret_log = np.log1p(ret_simple)

        pos_trace.extend(pos_now.tolist())
        ret_s_trace.extend(ret_simple.tolist())
        ret_l_trace.extend(ret_log.tolist())

        history = np.concatenate([history, block_log])
        prev_pos_last = float(pos_now[-1])

    pos_trace = np.array(pos_trace, dtype=float)
    ret_s_trace = np.array(ret_s_trace, dtype=float)
    ret_l_trace = np.array(ret_l_trace, dtype=float)

    return dict(
        positions=pos_trace,
        ret_simple=ret_s_trace,
        ret_log=ret_l_trace,
        sharpe=sharpe_ratio(ret_s_trace) if ret_s_trace.size else 0.0,
        mdd=max_drawdown_from_log(ret_l_trace) if ret_l_trace.size else 1.0,
        turnover=float(total_turnover)
    )

def backtest_once(
    test_log: np.ndarray,
    p: int, q: int, thr: float,
    retrain_interval: int,
    cost_per_turnover: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    scale_factor: float = 1000.0,
    arch_rescale_flag: bool = False,
    pos_clip: float = 1.0,
    warmup_len: int = 252,
) -> Dict:
    trace = backtest_continuous_trace(
        test_log, p,q,thr, retrain_interval,
        cost_per_turnover, min_block_vol, scale_factor, arch_rescale_flag,
        pos_clip, warmup_len, external_history=None, debug=False
    )
    return dict(sharpe=trace["sharpe"], mdd=trace["mdd"], turnover=trace["turnover"])

# -------- Rolling helpers --------
def rolling_apply(x: np.ndarray, window: int, fn) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window, len(x)+1):
        out[i-1] = fn(x[i-window:i])
    return out

# =================================
# Subcommand: viz
# =================================
def cmd_viz(args):
    setup_logging(args.debug)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve params (p,q,thr)
    if args.backtest_csv:
        bt = read_csv_required(Path(args.backtest_csv))
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None:
            raise RuntimeError("Params not found in backtest CSV. Check fold/interval/source_label.")
    else:
        front = read_csv_required(Path(args.front_csv))
        if not {"fold_id","retrain_interval","front_type","p","q","threshold","sharpe","mdd"}.issubset(front.columns):
            raise ValueError("front CSV missing required columns.")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None:
            raise RuntimeError("Knee not found in front CSV. Check fold/interval/front_type.")

    # 2) Load test
    test = read_csv_required(Path(args.test_csv))
    if "Log_Returns" not in test.columns:
        raise ValueError("test CSV must contain Log_Returns")
    test_log = test["Log_Returns"].fillna(0).to_numpy(float)

    # 3) Run trace
    trace = backtest_continuous_trace(
        test_log=test_log,
        p=pick["p"], q=pick["q"], thr=pick["threshold"],
        retrain_interval=args.interval,
        cost_per_turnover=args.cost_per_turnover,
        min_block_vol=args.min_block_vol,
        scale_factor=args.scale_factor,
        arch_rescale_flag=args.arch_rescale_flag,
        pos_clip=args.pos_clip,
        warmup_len=args.warmup_len,
        external_history=None,
        debug=args.debug
    )

    # 4) Plots
    equity = np.exp(np.cumsum(trace["ret_log"]))
    fig, ax = plt.subplots(figsize=(8,4), dpi=140)
    ax.plot(equity)
    title_src = args.source_label if args.backtest_csv else args.front_type
    ax.set_title(f"Equity — fold {args.fold_id}, int {args.interval}, src {title_src}")
    ax.set_ylabel("Equity (norm.)"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "equity_curve.png"); plt.close(fig)

    roll_sharpe = rolling_apply(trace["ret_simple"], args.window, sharpe_ratio)
    fig, ax = plt.subplots(figsize=(8,3.5), dpi=140)
    ax.plot(roll_sharpe)
    ax.set_title(f"Rolling Sharpe (win={args.window})")
    ax.set_ylabel("Sharpe"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "rolling_sharpe.png"); plt.close(fig)

    cum_log = np.cumsum(trace["ret_log"])
    equity_series = np.exp(cum_log)
    peak = np.maximum.accumulate(equity_series)
    dd_series = (peak - equity_series) / (peak + EPS)
    fig, ax = plt.subplots(figsize=(8,3.5), dpi=140)
    ax.plot(dd_series)
    ax.set_title("Drawdown (path)")
    ax.set_ylabel("Drawdown"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "rolling_drawdown.png"); plt.close(fig)

    meta = dict(
        fold_id=args.fold_id,
        interval=args.interval,
        source=(args.source_label if args.backtest_csv else args.front_type),
        params=dict(p=pick["p"], q=pick["q"], threshold=pick["threshold"]),
        test_stats=dict(sharpe=trace["sharpe"], mdd=trace["mdd"], turnover=trace["turnover"])
    )
    with open(outdir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved viz to {outdir}")

# =================================
# Subcommand: sensitivity
# =================================
def cmd_sensitivity(args):
    setup_logging(args.debug)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve params
    if args.backtest_csv:
        bt = read_csv_required(Path(args.backtest_csv))
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None:
            raise RuntimeError("Params not found in backtest CSV.")
        src_name = args.source_label
    else:
        front = read_csv_required(Path(args.front_csv))
        if not {"fold_id","retrain_interval","front_type","p","q","threshold","sharpe","mdd"}.issubset(front.columns):
            raise ValueError("front CSV missing required columns.")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None:
            raise RuntimeError("Knee not found in front CSV.")
        src_name = args.front_type

    # 2) Load test
    test = read_csv_required(Path(args.test_csv))
    if "Log_Returns" not in test.columns:
        raise ValueError("test CSV must contain Log_Returns")
    test_log = test["Log_Returns"].fillna(0).to_numpy(float)

    cost_mults = [float(x) for x in args.cost_mults.split(",") if x.strip()]
    thr_mults  = [float(x) for x in args.thr_mults.split(",") if x.strip()]

    rows = []
    for cm in cost_mults:
        for tm in thr_mults:
            res = backtest_once(
                test_log=test_log,
                p=pick["p"], q=pick["q"], thr=pick["threshold"] * tm,
                retrain_interval=args.interval,
                cost_per_turnover=args.cost_per_turnover * cm,
                min_block_vol=args.min_block_vol,
                scale_factor=args.scale_factor,
                arch_rescale_flag=args.arch_rescale_flag,
                pos_clip=args.pos_clip,
                warmup_len=args.warmup_len
            )
            rows.append({
                "fold_id": args.fold_id,
                "interval": args.interval,
                "source": src_name,
                "p": pick["p"], "q": pick["q"],
                "threshold_base": pick["threshold"],
                "cost_mult": cm,
                "thr_mult": tm,
                "test_sharpe": float(res["sharpe"]),
                "test_mdd": float(res["mdd"]),
                "test_turnover": float(res["turnover"])
            })

    df = pd.DataFrame(rows)
    csv_path = outdir / "sensitivity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved sensitivity table -> {csv_path}")

    if args.plot:
        # Heatmap Sharpe by (thr_mult, cost_mult)
        pv = df.pivot(index="thr_mult", columns="cost_mult", values="test_sharpe")
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
        im = ax.imshow(pv.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index)
        ax.set_xlabel("Cost multiplier"); ax.set_ylabel("Threshold multiplier")
        ax.set_title("Sensitivity: Test Sharpe"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(outdir / "heatmap_sharpe.png"); plt.close(fig)

        pv2 = df.pivot(index="thr_mult", columns="cost_mult", values="test_mdd")
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
        im = ax.imshow(pv2.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pv2.columns))); ax.set_xticklabels(pv2.columns)
        ax.set_yticks(range(len(pv2.index)));   ax.set_yticklabels(pv2.index)
        ax.set_xlabel("Cost multiplier"); ax.set_ylabel("Threshold multiplier")
        ax.set_title("Sensitivity: Test MDD"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(outdir / "heatmap_mdd.png"); plt.close(fig)
        print(f"[OK] Saved heatmaps -> {outdir}")

# =================================
# CLI
# =================================
def build_parser():
    ap = argparse.ArgumentParser(description="Tier-2 viz & sensitivity tools (GA/GA+BO)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- Common base options ---
    base_common = argparse.ArgumentParser(add_help=False)
    base_common.add_argument("--test-csv", required=True, help="CSV with column 'Log_Returns'")
    base_common.add_argument("--interval", type=int, required=True)
    base_common.add_argument("--fold-id", type=int, required=True)
    base_common.add_argument("--cost-per-turnover", type=float, default=BASE_COST+SLIPPAGE)
    base_common.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    base_common.add_argument("--pos-clip", type=float, default=1.0)
    base_common.add_argument("--scale-factor", type=float, default=1000.0)
    base_common.add_argument("--arch-rescale-flag", action="store_true")
    base_common.add_argument("--warmup-len", type=int, default=252)
    base_common.add_argument("--debug", action="store_true")

    # --- Two ways to supply params ---
    # A) from *_front.csv (auto knee)
    from_front = argparse.ArgumentParser(add_help=False)
    from_front.add_argument("--front-csv", help="Tier-2 *_front.csv (for knee picking)")
    from_front.add_argument("--front-type", default="GA+BO", choices=["GA","GA+BO"])
    # B) from backtest results (already picked)
    from_backtest = argparse.ArgumentParser(add_help=False)
    from_backtest.add_argument("--backtest-csv", help="backtest_results_ga_vs_gabo.csv (read p,q,thr)")
    from_backtest.add_argument("--source-label", default="GA+BO_knee", help="Row label in backtest CSV, e.g., GA_knee or GA+BO_knee")

    # --- viz subcommand ---
    p_viz = sub.add_parser("viz", parents=[base_common], help="Plot equity curve & rolling metrics")
    p_viz.add_argument("--outdir", default="viz_outputs")
    p_viz.add_argument("--window", type=int, default=63)
    # param source (front OR backtest)
    p_viz.add_argument("--use-backtest", action="store_true",
                       help="If set, use --backtest-csv & --source-label; else use --front-csv & --front-type")
    p_viz.add_argument("--front-csv", help="(if not --use-backtest) Path to *_front.csv")
    p_viz.add_argument("--front-type", default="GA+BO", choices=["GA","GA+BO"])
    p_viz.add_argument("--backtest-csv", help="(if --use-backtest) Path to backtest_results_ga_vs_gabo.csv")
    p_viz.add_argument("--source-label", default="GA+BO_knee")

    # --- sensitivity subcommand ---
    p_sens = sub.add_parser("sensitivity", parents=[base_common], help="Cost & threshold sensitivity")
    p_sens.add_argument("--outdir", default="sensitivity_outputs")
    p_sens.add_argument("--cost-mults", default="0.5,1.0,1.5")
    p_sens.add_argument("--thr-mults", default="0.8,1.0,1.2")
    p_sens.add_argument("--plot", action="store_true")
    p_sens.add_argument("--use-backtest", action="store_true",
                        help="If set, use --backtest-csv & --source-label; else use --front-csv & --front-type")
    p_sens.add_argument("--front-csv", help="(if not --use-backtest) Path to *_front.csv")
    p_sens.add_argument("--front-type", default="GA+BO", choices=["GA","GA+BO"])
    p_sens.add_argument("--backtest-csv", help="(if --use-backtest) Path to backtest_results_ga_vs_gabo.csv")
    p_sens.add_argument("--source-label", default="GA+BO_knee")

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "viz":
        # check param source
        if args.use_backtest:
            if not args.backtest_csv:
                ap.error("viz: --use-backtest requires --backtest-csv")
        else:
            if not args.front_csv:
                ap.error("viz: without --use-backtest you must provide --front-csv")
        cmd_viz(args)

    elif args.cmd == "sensitivity":
        if args.use_backtest:
            if not args.backtest_csv:
                ap.error("sensitivity: --use-backtest requires --backtest-csv")
        else:
            if not args.front_csv:
                ap.error("sensitivity: without --use-backtest you must provide --front-csv")
        cmd_sensitivity(args)

if __name__ == "__main__":
    main()