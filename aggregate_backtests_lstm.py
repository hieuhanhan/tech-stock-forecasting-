#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate Tier-2 LSTM backtests → Figures + Tables

Inputs
------
- --backtest-csv : CSV kết quả backtest LSTM (per fold × interval × source)
- --test-csv     : TEST set CSV chứa PC1..PC7, target, target_log_returns (hoặc Log_Returns)
- --tier1-json   : JSON champions Tier-1 (list hoặc {"results":[...]}), mỗi fold có {layers,batch_size,dropout,patience}

Outputs (under --outdir)
------------------------
- box_test_sharpe_by_interval(_source).png
- box_test_mdd_by_interval(_source).png
- box_test_ann_return_by_interval(_source).png
- mean_equity/mean_equity_int{K}_{SRC}.png           (nếu không tắt)
- aggregate_summary.csv
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== TF/Keras ==============
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from numpy.lib.stride_tricks import sliding_window_view

# ============== Defaults ==============
EPS = 1e-8
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015

PC_FEATURES = [f"PC{i}" for i in range(1, 8)]
TARGET_COL  = "target"
RETN_COL    = "target_log_returns"

# ============== Utils IO/plot ==============
def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(fig, outpath: Path):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved → {outpath}")

def smart_read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

# ============== Stats & helpers ==============
def sharpe_ratio(simple_returns: np.ndarray) -> float:
    r = np.asarray(simple_returns, dtype=float)
    if r.size < 2: return 0.0
    sd = float(np.std(r, ddof=1))
    return 0.0 if sd == 0.0 else float(np.mean(r))/sd*np.sqrt(252.0)

def max_drawdown_from_log(logr: np.ndarray) -> float:
    C = np.exp(np.cumsum(np.asarray(logr, dtype=float)))
    peak = np.maximum.accumulate(C)
    dd = (peak - C) / (peak + EPS)
    return float(np.max(dd)) if dd.size else 0.0

def create_windows(X: np.ndarray, y: np.ndarray, lookback: int):
    """Return (N,L,F) windows and (N,) targets; None if not enough history."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    N = X.shape[0]
    if N <= lookback:
        return None, None
    wins = sliding_window_view(X, window_shape=lookback, axis=0)   # (N-L+1,1,L,F) or (N-L+1,L,F)
    if wins.ndim == 4:
        wins = wins.squeeze(1)
    Xw = wins[:-1]
    yw = y[lookback:]
    if Xw.shape[1] != lookback:
        Xw = np.transpose(Xw, (0,2,1))
    return Xw.astype(np.float32), yw.astype(np.float32)

def build_lstm(window: int, units: int, lr: float, num_features: int,
               layers: int, dropout: float, norm_type: str) -> Sequential:
    def _norm():
        return LayerNormalization() if norm_type == "layer" else BatchNormalization()
    m = Sequential()
    m.add(Input((int(window), int(num_features))))
    m.add(LSTM(int(units), return_sequences=(layers > 1)))
    m.add(_norm()); m.add(Dropout(float(dropout)))
    if layers > 1:
        m.add(LSTM(max(8, int(units//2))))
        m.add(_norm()); m.add(Dropout(float(dropout)))
    m.add(Dense(1))
    m.compile(optimizer=Adam(learning_rate=float(lr)), loss="mse")
    return m

def get_champ_map(tier1_json: Path) -> Dict[int, Dict]:
    obj = smart_read_json(tier1_json)
    if isinstance(obj, dict) and "results" in obj:
        obj = obj["results"]
    cmap: Dict[int, Dict] = {}
    for r in obj:
        fid = int(r.get("fold_id", r.get("global_fold_id", -1)))
        if fid == -1: continue
        ch = r.get("champion") or r.get("best_params") or r
        cmap[fid] = dict(
            layers=int(ch.get("layers", ch.get("num_layers", 1))),
            batch_size=int(ch.get("batch_size", 32)),
            dropout=float(ch.get("dropout", 0.2)),
            patience=int(ch.get("patience", 5)),
        )
    return cmap

def pick_lr(row: pd.Series) -> float:
    if "learning_rate" in row.index: return float(row["learning_rate"])
    if "lr" in row.index:           return float(row["lr"])
    raise KeyError("Row must contain 'learning_rate' or 'lr'.")

# ============== LSTM backtest trace ==============
def backtest_lstm_trace(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    champion: Dict,          # layers,batch_size,dropout,patience
    window: int, units: int, lr: float, epochs: int, threshold: float,
    retrain_interval: int,
    cost_per_turnover: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    hysteresis: float = 0.05,
    mad_k: float = 0.5,
    warmup_len: int = 252,
) -> Dict:
    assert set(feature_cols).issubset(test_df.columns), "Missing PCA feature columns."

    metric_col = RETN_COL if RETN_COL in test_df.columns else ("Log_Returns" if "Log_Returns" in test_df.columns else None)
    if metric_col is None:
        raise ValueError("Need 'target_log_returns' or 'Log_Returns' in TEST CSV.")

    Xall  = test_df[feature_cols].to_numpy(np.float32)
    yall  = test_df[TARGET_COL].to_numpy(np.float32)
    r_log = test_df[metric_col].to_numpy(np.float64)
    n_tot = Xall.shape[0]
    if n_tot <= warmup_len + 1:
        raise ValueError(f"TEST length {n_tot} too small (warmup={warmup_len}).")

    layers     = int(champion["layers"])
    batch_size = int(champion["batch_size"])
    dropout    = float(champion["dropout"])
    patience   = int(champion["patience"])
    norm_type  = "layer" if batch_size <= 32 else "batch"

    hist_X = Xall[:warmup_len].copy()
    hist_y = yall[:warmup_len].copy()

    start_idx = warmup_len
    simple_all: List[np.ndarray] = []
    log_all:    List[np.ndarray] = []

    prev_sig_last = 0.0
    total_turnover = 0.0

    for start in range(start_idx, n_tot, int(retrain_interval)):
        end = min(start + int(retrain_interval), n_tot)
        if end - start <= 0: continue

        block_log = r_log[start:end]
        block_X   = Xall[start:end]
        block_y   = yall[start:end]

        block_vol = float(np.std(block_log))
        if block_vol < float(min_block_vol):
            hist_X = np.vstack([hist_X, block_X])
            hist_y = np.concatenate([hist_y, block_y])
            continue

        Xw, yw = create_windows(hist_X, hist_y, int(window))
        Xw_val, yw_val = create_windows(
            np.concatenate([hist_X[-int(window):], block_X], axis=0),
            np.concatenate([hist_y[-int(window):], block_y], axis=0),
            int(window)
        )
        if Xw is None or Xw_val is None:
            hist_X = np.vstack([hist_X, block_X]); hist_y = np.concatenate([hist_y, block_y]); continue

        try:
            K.clear_session()
            model = build_lstm(int(window), int(units), float(lr), hist_X.shape[1],
                               layers, dropout, norm_type)
            es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
            model.fit(Xw, yw, validation_data=(Xw_val, yw_val),
                      epochs=int(epochs), batch_size=batch_size, callbacks=[es], verbose=0)
            preds = model.predict(Xw_val, verbose=0).ravel().astype(float)
        except Exception:
            hist_X = np.vstack([hist_X, block_X]); hist_y = np.concatenate([hist_y, block_y]); continue
        finally:
            K.clear_session()

        if preds.size == 0 or not np.isfinite(preds).all():
            hist_X = np.vstack([hist_X, block_X]); hist_y = np.concatenate([hist_y, block_y]); continue

        # hysteresis + MAD (nhẹ)
        q_rel = float(threshold)
        thr_enter = np.percentile(preds, q_rel * 100.0)
        thr_exit  = np.percentile(preds, max(0.0, q_rel - hysteresis) * 100.0)
        med = float(np.median(preds))
        mad = float(np.median(np.abs(preds - med))) + 1e-12
        strong = (np.abs(preds - med) >= mad_k * mad)

        sig = np.zeros_like(preds, dtype=float)
        state = int(prev_sig_last > 0.5)
        held  = 0
        for t in range(preds.size):
            if state == 0:
                if (preds[t] >= thr_enter) and strong[t]:
                    state = 1; held = 1
                else:
                    held = 0
            else:
                if (held >= 0) and ((preds[t] < thr_exit) or (not strong[t])):
                    state = 0; held = 0
                else:
                    held += 1
            sig[t] = float(state)

        sig_full = np.concatenate([[prev_sig_last], sig])
        turnover = np.abs(np.diff(sig_full))
        total_turnover += float(turnover.sum())
        cost_vec = (BASE_COST + SLIPPAGE) * turnover  # cost_per_turnover passed below anyway

        block_simple = np.exp(block_log) - 1.0
        ret_simple = np.clip(block_simple * sig - cost_vec, -0.9999, None)
        ret_log    = np.log1p(ret_simple)

        simple_all.append(ret_simple)
        log_all.append(ret_log)

        hist_X = np.vstack([hist_X, block_X])
        hist_y = np.concatenate([hist_y, block_y])
        prev_sig_last = float(sig[-1])

    if not simple_all:
        return dict(positions=np.zeros(0), ret_simple=np.zeros(0), ret_log=np.zeros(0),
                    sharpe=0.0, mdd=1.0, turnover=0.0)

    rs = np.concatenate(simple_all)
    rl = np.concatenate(log_all)
    return dict(
        positions=np.zeros_like(rs),  # (không dùng ở aggregate)
        ret_simple=rs,
        ret_log=rl,
        sharpe=sharpe_ratio(rs),
        mdd=max_drawdown_from_log(rl),
        turnover=float(total_turnover),
    )

# ============== Aggregation plotting ==============
def box_by_interval(df: pd.DataFrame, metric: str, outdir: Path, split_by_source: bool):
    ensure_outdir(outdir)
    if split_by_source:
        fig, ax = plt.subplots(figsize=(9,4.8), dpi=150)
        intervals = sorted(df["retrain_interval"].unique())
        sources   = sorted(df["source"].unique())
        data, labels = [], []
        for k in intervals:
            for s in sources:
                vals = df[(df["retrain_interval"]==k) & (df["source"]==s)][metric].values
                if vals.size:
                    data.append(vals); labels.append(f"{k}—{s}")
        ax.boxplot(data, labels=labels, vert=True, showfliers=False)
        ax.set_title(f"{metric} by interval × source"); ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        savefig(fig, outdir / f"box_{metric}_by_interval_source.png")
    else:
        fig, ax = plt.subplots(figsize=(7.5,4.6), dpi=150)
        uniq = sorted(df["retrain_interval"].unique())
        by = [df[df["retrain_interval"]==k][metric].values for k in uniq]
        ax.boxplot(by, labels=uniq, vert=True, showfliers=False)
        ax.set_title(f"{metric} by interval"); ax.set_xlabel("retrain_interval"); ax.set_ylabel(metric)
        savefig(fig, outdir / f"box_{metric}_by_interval.png")

def pad_and_stack(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    if not list_of_arrays:
        return np.zeros((0,0))
    T = max(arr.size for arr in list_of_arrays)
    out = np.full((len(list_of_arrays), T), np.nan, dtype=float)
    for i, arr in enumerate(list_of_arrays):
        out[i, :arr.size] = arr
    return out

def plot_mean_equity(
    test_df: pd.DataFrame,
    rows_params: pd.DataFrame,
    champ_map: Dict[int, Dict],
    interval: int,
    source_label: str,
    outdir: Path,
    cost_per_turnover: float,
    min_block_vol: float,
    warmup_len: int
):
    ensure_outdir(outdir)
    traces = []
    for _, r in rows_params.iterrows():
        fid = int(r["fold_id"])
        if fid not in champ_map:  # skip if missing Tier-1
            continue
        champion = champ_map[fid]
        lr = pick_lr(r)
        trace = backtest_lstm_trace(
            test_df=test_df,
            feature_cols=PC_FEATURES,
            champion=champion,
            window=int(r["window"]),
            units=int(r["units"]),
            lr=float(lr),
            epochs=int(r["epochs"]),
            threshold=float(r["threshold"]),
            retrain_interval=int(interval),
            cost_per_turnover=float(cost_per_turnover),
            min_block_vol=float(min_block_vol),
            hysteresis=0.05,
            mad_k=0.5,
            warmup_len=int(warmup_len),
        )
        eq = np.exp(np.cumsum(trace["ret_log"]))
        traces.append(eq)

    if not traces:
        print(f"[WARN] No traces for int={interval}, src={source_label}")
        return

    M = pad_and_stack(traces)
    mean_eq = np.nanmean(M, axis=0)
    std_eq  = np.nanstd(M, axis=0)

    fig, ax = plt.subplots(figsize=(8.2,4.2), dpi=150)
    t = np.arange(mean_eq.size)
    ax.plot(t, mean_eq, label="Mean equity")
    ax.fill_between(t, mean_eq-std_eq, mean_eq+std_eq, alpha=0.25, label="±1 std")
    ax.set_title(f"Mean equity ± std — interval {interval}, source {source_label}")
    ax.set_xlabel("Time"); ax.set_ylabel("Equity (norm.)"); ax.legend()
    savefig(fig, outdir / f"mean_equity_int{interval}_{source_label}.png")

# ============== CLI ==============
def get_args():
    ap = argparse.ArgumentParser("Aggregate plots for LSTM Tier-2 backtests")
    ap.add_argument("--backtest-csv", required=True, help="CSV from LSTM backtest (per fold×interval×source).")
    ap.add_argument("--test-csv", required=True, help="TEST CSV with PC1..PC7, target, and target_log_returns or Log_Returns.")
    ap.add_argument("--tier1-json", required=True, help="Tier-1 champions JSON (per fold).")
    ap.add_argument("--intervals", default="10,20,42")
    ap.add_argument("--source-label", default="GA+BO_knee", choices=["GA_knee","GA+BO_knee","all"])
    ap.add_argument("--outdir", default="agg_plots_lstm")

    # knobs for LSTM trace
    ap.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    ap.add_argument("--cost-per-turnover", type=float, default=BASE_COST + SLIPPAGE)
    ap.add_argument("--warmup-len", type=int, default=252)

    # perf / options
    ap.add_argument("--no-mean-equity", action="store_true", help="Skip mean equity reconstruction (faster).")
    ap.add_argument("--cpu-only", action="store_true", help="Disable mixed precision / XLA for CPU machines.")
    ap.add_argument("--max-rows-per-plot", type=int, default=200, help="(Soft) limit folds per mean-equity plot for speed.")

    return ap.parse_args()

def setup_tf(cpu_only: bool):
    try:
        if cpu_only:
            # turn off mixed precision / XLA
            tf.keras.mixed_precision.set_global_policy("float32")
            tf.config.optimizer.set_jit(False)
        else:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            tf.config.optimizer.set_jit(True)
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    np.random.seed(42); tf.random.set_seed(42)

def main():
    args = get_args()
    setup_tf(args.cpu_only)

    outdir = Path(args.outdir); ensure_outdir(outdir)

    bt = pd.read_csv(args.backtest_csv)
    need_cols = {"fold_id","retrain_interval","source","window","units","epochs","threshold",
                 "test_sharpe","test_mdd","test_ann_return"}
    if not (need_cols.issubset(bt.columns) or ({"learning_rate"}|need_cols).issubset(bt.columns) or ({"lr"}|need_cols).issubset(bt.columns)):
        missing = (need_cols - set(bt.columns))
        raise ValueError(f"[backtest-csv] missing columns: {missing} (+ learning_rate/lr)")

    test_df = pd.read_csv(args.test_csv)
    # Ensure required columns
    miss = [c for c in (PC_FEATURES+[TARGET_COL]) if c not in test_df.columns]
    if miss:
        raise ValueError(f"[test-csv] missing columns: {miss}")
    if (RETN_COL not in test_df.columns) and ("Log_Returns" not in test_df.columns):
        raise ValueError("[test-csv] need 'target_log_returns' or 'Log_Returns'.")

    champ_map = get_champ_map(Path(args.tier1_json))

    intervals = [int(x) for x in str(args.intervals).split(",") if x.strip()]
    # filter by source
    if args.source_label != "all":
        bt_f = bt[bt["source"] == args.source_label].copy()
        split_by_source = False
    else:
        bt_f = bt.copy()
        split_by_source = True

    # ---- Boxplots
    box_by_interval(bt_f, "test_sharpe", outdir, split_by_source)
    box_by_interval(bt_f, "test_mdd", outdir, split_by_source)
    box_by_interval(bt_f, "test_ann_return", outdir, split_by_source)

    # ---- Mean equity ± std (optional)
    if not args.no_mean_equity:
        me_dir = outdir / "mean_equity"
        for k in intervals:
            if args.source_label == "all":
                sources = sorted(bt["source"].unique())
            else:
                sources = [args.source_label]
            for s in sources:
                rows = bt[(bt["retrain_interval"]==k) & (bt["source"]==s)].copy()
                if rows.empty:
                    print(f"[WARN] No rows for interval={k}, source={s}"); continue
                # soft limit to speed up
                if args.max_rows_per_plot and len(rows) > args.max_rows_per_plot:
                    rows = rows.sample(args.max_rows_per_plot, random_state=42)
                # ensure lr column
                if "learning_rate" not in rows.columns and "lr" not in rows.columns:
                    raise ValueError("backtest-csv needs 'learning_rate' or 'lr' for LSTM.")
                plot_mean_equity(
                    test_df=test_df,
                    rows_params=rows[["fold_id","window","units","epochs","threshold","learning_rate"]]
                                if "learning_rate" in rows.columns else
                                rows[["fold_id","window","units","epochs","threshold","lr"]],
                    champ_map=champ_map,
                    interval=int(k),
                    source_label=s,
                    outdir=me_dir,
                    cost_per_turnover=args.cost_per_turnover,
                    min_block_vol=args.min_block_vol,
                    warmup_len=args.warmup_len
                )

    # ---- Summary table
    grp_cols = ["retrain_interval"] if args.source_label!="all" else ["retrain_interval","source"]
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
    print(f"[OK] Saved summary → {summary_path}")

if __name__ == "__main__":
    main()