#!/usr/bin/env python3
# tier2_aggregate_plots.py
"""
Aggregate plots for Tier-2 results:
- Boxplots of Sharpe/MDD/Ann Return across folds by retrain interval (optionally by source).
- Mean equity curve ± std for each (interval, source) by recomputing traces.

Inputs:
  --backtest-csv : CSV from backtest (must include columns: fold_id,retrain_interval,source,p,q,threshold, test_*).
  --test-csv     : Global TEST file with column 'Log_Returns' (used to rebuild traces).
  --intervals    : Comma-separated list (e.g. 10,20,42)
  --source-label : GA_knee | GA+BO_knee | all
Outputs:
  - PNG figures saved to --outdir
  - CSV summary saved to --outdir/aggregate_summary.csv
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# ---- Defaults ----
EPS = 1e-8
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015

# ---- Small utils ----
def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(fig, outpath: Path):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {outpath}")

# ---- Backtest trace (same logic as tools) ----
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
) -> Dict:
    test_log = np.asarray(test_log, dtype=float)
    n_total = len(test_log)
    if n_total <= warmup_len + 1:
        raise ValueError(f"test length {n_total} too small for warmup_len={warmup_len}")

    history = test_log[:warmup_len].copy()
    start_idx = warmup_len

    positions = []
    ret_s = []
    ret_l = []
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
        except Exception:
            history = np.concatenate([history, block_log])
            continue

        z = f_mu / (f_sig + EPS)
        pos_now = np.clip(z / (thr + EPS), -pos_clip, pos_clip)

        signals_full = np.concatenate([[prev_pos_last], pos_now])
        turnover = np.abs(np.diff(signals_full))
        cost_vec = cost_per_turnover * turnover
        total_turnover += float(turnover.sum())

        block_simple = np.exp(block_log) - 1.0
        r_simple = np.clip(block_simple * pos_now - cost_vec, -0.9999, None)
        r_log = np.log1p(r_simple)

        positions.extend(pos_now.tolist())
        ret_s.extend(r_simple.tolist())
        ret_l.extend(r_log.tolist())

        history = np.concatenate([history, block_log])
        prev_pos_last = float(pos_now[-1])

    pos = np.asarray(positions, dtype=float)
    rs  = np.asarray(ret_s, dtype=float)
    rl  = np.asarray(ret_l, dtype=float)

    return dict(
        positions=pos,
        ret_simple=rs,
        ret_log=rl,
        sharpe=sharpe_ratio(rs) if rs.size else 0.0,
        mdd=max_drawdown_from_log(rl) if rl.size else 1.0,
        turnover=float(total_turnover)
    )

# ---- Aggregation plotting ----
def plot_box_by_interval(df: pd.DataFrame, metric: str, outdir: Path, title: str, split_by_source: bool):
    ensure_outdir(outdir)
    if split_by_source:
        # one box per (interval, source), grouped by interval on x
        fig, ax = plt.subplots(figsize=(8,4.5), dpi=150)
        intervals = sorted(df["retrain_interval"].unique())
        sources = sorted(df["source"].unique())
        data = []
        labels = []
        for k in intervals:
            for s in sources:
                vals = df[(df["retrain_interval"]==k) & (df["source"]==s)][metric].values
                if vals.size:
                    data.append(vals)
                    labels.append(f"{k} — {s}")
        ax.boxplot(data, labels=labels, vert=True, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        savefig(fig, outdir / f"box_{metric}_by_interval_source.png")
    else:
        fig, ax = plt.subplots(figsize=(7,4.5), dpi=150)
        by = [df[df["retrain_interval"]==k][metric].values for k in sorted(df["retrain_interval"].unique())]
        ax.boxplot(by, labels=sorted(df["retrain_interval"].unique()), vert=True, showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("retrain_interval")
        ax.set_ylabel(metric)
        savefig(fig, outdir / f"box_{metric}_by_interval.png")

def pad_and_stack(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """Pad arrays with NaN to the same length and stack as 2D [n_series, T]."""
    if not list_of_arrays:
        return np.zeros((0,0))
    T = max(arr.size for arr in list_of_arrays)
    out = np.full((len(list_of_arrays), T), np.nan, dtype=float)
    for i, arr in enumerate(list_of_arrays):
        out[i,:arr.size] = arr
    return out

def plot_mean_equity(test_log: np.ndarray,
                     bt_rows: pd.DataFrame,
                     interval: int,
                     source_label: str,
                     outdir: Path,
                     cost_per_turnover: float,
                     min_block_vol: float,
                     scale_factor: float,
                     arch_rescale_flag: bool,
                     pos_clip: float,
                     warmup_len: int):
    ensure_outdir(outdir)
    # build traces per fold
    traces = []
    for _, r in bt_rows.iterrows():
        trace = backtest_continuous_trace(
            test_log=test_log,
            p=int(r["p"]),
            q=int(r["q"]),
            thr=float(r["threshold"]),
            retrain_interval=interval,
            cost_per_turnover=cost_per_turnover,
            min_block_vol=min_block_vol,
            scale_factor=scale_factor,
            arch_rescale_flag=arch_rescale_flag,
            pos_clip=pos_clip,
            warmup_len=warmup_len,
        )
        eq = np.exp(np.cumsum(trace["ret_log"]))  # equity path (normalized)
        traces.append(eq)

    if not traces:
        print(f"[WARN] No traces for interval={interval}, source={source_label}")
        return

    M = pad_and_stack(traces)  # [N, T] with NaN padding
    mean_eq = np.nanmean(M, axis=0)
    std_eq  = np.nanstd(M, axis=0)

    # plot mean ± std
    fig, ax = plt.subplots(figsize=(8,4.2), dpi=150)
    t = np.arange(mean_eq.size)
    ax.plot(t, mean_eq, label="Mean equity")
    ax.fill_between(t, mean_eq-std_eq, mean_eq+std_eq, alpha=0.25, label="±1 std")
    ax.set_title(f"Mean equity ± std — interval {interval}, source {source_label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (norm.)")
    ax.legend()
    savefig(fig, outdir / f"mean_equity_int{interval}_{source_label}.png")

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Aggregate Tier-2 plots")
    ap.add_argument("--backtest-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--intervals", default="10,20,42")
    ap.add_argument("--source-label", default="GA+BO_knee",
                    choices=["GA_knee","GA+BO_knee","all"])
    ap.add_argument("--outdir", default="agg_plots")

    # params for rebuilding traces
    ap.add_argument("--cost-per-turnover", type=float, default=BASE_COST+SLIPPAGE)
    ap.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    ap.add_argument("--pos-clip", type=float, default=1.0)
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--arch-rescale-flag", action="store_true")
    ap.add_argument("--warmup-len", type=int, default=252)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # load
    bt = pd.read_csv(args.backtest_csv)
    required = {"fold_id","retrain_interval","source","p","q","threshold",
                "test_sharpe","test_mdd","test_ann_return"}
    if not required.issubset(bt.columns):
        missing = required - set(bt.columns)
        raise ValueError(f"Backtest CSV missing columns: {missing}")

    test_df = pd.read_csv(args.test_csv)
    if "Log_Returns" not in test_df.columns:
        raise ValueError("test CSV must contain 'Log_Returns'")
    test_log = test_df["Log_Returns"].fillna(0).to_numpy(float)

    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]

    # filter by source
    if args.source_label != "all":
        bt_f = bt[bt["source"] == args.source_label].copy()
        split_by_source = False
    else:
        bt_f = bt.copy()
        split_by_source = True

    # ---- Boxplots across intervals ----
    plot_box_by_interval(
        df=bt_f,
        metric="test_sharpe",
        outdir=outdir,
        title="Test Sharpe by interval",
        split_by_source=split_by_source
    )
    plot_box_by_interval(
        df=bt_f,
        metric="test_mdd",
        outdir=outdir,
        title="Test MDD by interval",
        split_by_source=split_by_source
    )
    plot_box_by_interval(
        df=bt_f,
        metric="test_ann_return",
        outdir=outdir,
        title="Test Annualized Return by interval",
        split_by_source=split_by_source
    )

    # ---- Mean equity ± std per (interval, source) ----
    if args.source_label == "all":
        sources = sorted(bt["source"].unique())
    else:
        sources = [args.source_label]

    for k in intervals:
        for s in sources:
            rows = bt[(bt["retrain_interval"]==k) & (bt["source"]==s)]
            if rows.empty:
                print(f"[WARN] No rows for interval={k}, source={s}")
                continue
            plot_mean_equity(
                test_log=test_log,
                bt_rows=rows[["p","q","threshold"]],
                interval=k,
                source_label=s,
                outdir=outdir / "mean_equity",
                cost_per_turnover=args.cost_per_turnover,
                min_block_vol=args.min_block_vol,
                scale_factor=args.scale_factor,
                arch_rescale_flag=args.arch_rescale_flag,
                pos_clip=args.pos_clip,
                warmup_len=args.warmup_len
            )

    # ---- Summary CSV ----
    if args.source_label == "all":
        grp_cols = ["retrain_interval","source"]
    else:
        grp_cols = ["retrain_interval"]

    summary = (bt_f.groupby(grp_cols, as_index=False)
               .agg(test_sharpe_median=("test_sharpe","median"),
                    test_sharpe_mean=("test_sharpe","mean"),
                    test_mdd_median=("test_mdd","median"),
                    test_mdd_mean=("test_mdd","mean"),
                    test_ann_return_median=("test_ann_return","median"),
                    test_ann_return_mean=("test_ann_return","mean"),
                    n=("fold_id","count")))
    summary_path = outdir / "aggregate_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()