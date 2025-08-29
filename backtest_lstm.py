#!/usr/bin/env python3
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import tensorflow as tf
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Defaults
# -----------------------------
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015
EPS = 1e-8

PC_FEATURES = [f"PC{i}" for i in range(1, 8)]  # PC1..PC7
TARGET_COL = "target"                          # used to build windows
RETN_COL   = "target_log_returns"              # realized returns for PnL/MDD

# -----------------------------
# Logging / TF setup
# -----------------------------
def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

def setup_tf(seed: int = 42):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(seed)
    np.random.seed(seed)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        tf.config.optimizer.set_jit(True)  # XLA
    except Exception:
        pass

# -----------------------------
# Metrics
# -----------------------------
def sharpe_ratio(simple_returns: np.ndarray) -> float:
    r = np.asarray(simple_returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return 0.0 if sd == 0.0 else (float(np.mean(r))/sd) * np.sqrt(252.0)

def max_drawdown(data: np.ndarray, input_type: str = "log") -> float:
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return 0.0
    if input_type == "price":
        C = x / (x[0] + np.finfo(float).eps)
    elif input_type == "simple":
        C = np.cumprod(1.0 + x)
    elif input_type == "log":
        C = np.exp(np.cumsum(x))
    else:
        C = np.exp(np.cumsum(x)) if np.min(x) <= -1.0 else np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(C)
    dd = (peak - C) / (peak + np.finfo(float).eps)
    return float(np.max(dd))

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def create_windows(X: np.ndarray, y: np.ndarray, lookback: int):
    """Return (N,L,F) windows and (N,) targets; None if not enough history."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    N = X.shape[0]
    if N <= lookback:
        return None, None
    wins = sliding_window_view(X, window_shape=lookback, axis=0)  # (N-L+1, 1, L, F)
    if wins.ndim == 4:
        wins = wins.squeeze(1)
    Xw = wins[:-1]                 # up to the penultimate window
    yw = y[lookback:]              # aligned next-step target
    if Xw.shape[1] != lookback:    # safety
        Xw = np.transpose(Xw, (0, 2, 1))
    return Xw.astype(np.float32), yw.astype(np.float32)

def build_lstm(window: int, units: int, lr: float, num_features: int,
               layers: int, dropout_rate: float, norm_type: str) -> Sequential:
    def _norm():
        return LayerNormalization() if norm_type == "layer" else BatchNormalization()
    m = Sequential()
    m.add(Input((int(window), int(num_features))))
    m.add(LSTM(int(units), return_sequences=(layers > 1)))
    m.add(_norm()); m.add(Dropout(float(dropout_rate)))
    if layers > 1:
        m.add(LSTM(max(8, int(units//2))))
        m.add(_norm()); m.add(Dropout(float(dropout_rate)))
    m.add(Dense(1))
    m.compile(optimizer=Adam(learning_rate=float(lr)), loss="mse")
    return m

def pick_knee_from_front(front_df: pd.DataFrame, fold_id: int, interval: int, front_type: str) -> Optional[Dict]:
    sub = front_df[(front_df["fold_id"] == fold_id) &
                   (front_df["retrain_interval"] == interval) &
                   (front_df["front_type"] == front_type)]
    if sub.empty:
        return None
    # Use their pre-computed knee if present; otherwise nearest to origin in normalized (−Sharpe, MDD)
    if {"is_knee"}.issubset(sub.columns) and sub["is_knee"].any():
        row = sub[sub["is_knee"]].iloc[0]
    else:
        F = np.c_[ -sub["sharpe"].to_numpy(float), sub["mdd"].to_numpy(float) ]
        f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
        idx = int(np.argmin(np.sqrt((f**2).sum(axis=1))))
        row = sub.iloc[idx]
    # Column names may vary (lr vs learning_rate)
    lr_key = "learning_rate" if "learning_rate" in sub.columns else ("lr" if "lr" in sub.columns else None)
    if lr_key is None:
        raise ValueError("Front CSV must contain 'learning_rate' or 'lr'.")
    return dict(
        window=int(row["window"]),
        units=int(row["units"]),
        lr=float(row[lr_key]),
        epochs=int(row.get("epochs", 20)),
        threshold=float(row.get("threshold", row.get("rel_thresh", 0.6))),
        sharpe=float(row["sharpe"]),
        mdd=float(row["mdd"])
    )

# -----------------------------
# LSTM backtest engine (block re-train, 0/1 signals)
# -----------------------------
def backtest_lstm(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    champion: Dict,                 # from Tier-1: layers, batch_size, dropout
    t2_vars: Dict,                  # from Tier-2 knee: window, units, lr, epochs, threshold
    retrain_interval: int,
    cost_per_turnover: float = BASE_COST + SLIPPAGE,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    hysteresis: float = 0.05,
    mad_k: float = 0.5,
    warmup_len: int = 252,
    debug: bool = False,
) -> Dict:

    assert set(feature_cols).issubset(test_df.columns), "Missing PCA feature columns in test CSV."
    assert TARGET_COL in test_df.columns, f"Missing '{TARGET_COL}' in test CSV."
    metric_col = RETN_COL if RETN_COL in test_df.columns else ("Log_Returns" if "Log_Returns" in test_df.columns else None)
    if metric_col is None:
        raise ValueError("Test CSV must contain 'target_log_returns' or 'Log_Returns' for realized PnL.")
    arr_feat  = test_df[feature_cols].to_numpy(np.float32)
    arr_tgt   = test_df[TARGET_COL].to_numpy(np.float32)
    arr_r_log = test_df[metric_col].to_numpy(np.float64)

    n_total = arr_feat.shape[0]
    if n_total <= warmup_len + 1:
        raise ValueError(f"Test length {n_total} too small for warmup_len={warmup_len}")

    # fixed hyperparams from T1 champion
    layers     = int(champion.get("layers", 1))
    batch_size = int(champion.get("batch_size", 32))
    dropout    = float(champion.get("dropout", 0.2))
    patience   = int(champion.get("patience", 5))

    # vars from T2 knee
    window  = int(t2_vars["window"])
    units   = int(t2_vars["units"])
    lr      = float(t2_vars["lr"])
    epochs  = int(t2_vars["epochs"])
    q_rel   = float(t2_vars["threshold"])

    # Walk-forward
    hist_X = arr_feat[:warmup_len].copy()
    hist_y = arr_tgt[:warmup_len].copy()

    start_idx = warmup_len
    simple_all, log_all = [], []
    prev_sig_last = 0.0
    total_turnover = 0.0

    # traces (optional)
    sig_trace = []
    turn_trace = []

    for start in range(start_idx, n_total, int(retrain_interval)):
        end = min(start + int(retrain_interval), n_total)
        if end - start <= 0:
            continue

        block_log = arr_r_log[start:end]
        block_feat = arr_feat[start:end]
        block_tgt  = arr_tgt[start:end]

        block_vol = float(np.std(block_log))
        if block_vol < float(min_block_vol):
            if debug:
                logging.debug(f"[BLK {start:04d}-{end:04d}] SKIP (low vol={block_vol:.3e})")
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

        # windows on expanded history
        Xw, yw = create_windows(hist_X, hist_y, window)
        if Xw is None:
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

        # windows for block with warm-start (tail of history)
        Xw_val, yw_val = create_windows(
            np.concatenate([hist_X[-window:], block_feat], axis=0),
            np.concatenate([hist_y[-window:], block_tgt], axis=0),
            window
        )
        if Xw_val is None:
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

        # train & predict block
        try:
            K.clear_session()
            norm_type = "layer" if batch_size <= 32 else "batch"
            model = build_lstm(window, units, lr, hist_X.shape[1], layers, dropout, norm_type)
            es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
            model.fit(Xw, yw, validation_data=(Xw_val, yw_val),
                      epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
            preds = model.predict(Xw_val, verbose=0).ravel().astype(float)
        except Exception as e:
            if debug:
                logging.debug(f"[FIT_FAIL] w={window},u={units},lr={lr:.2e},ep={epochs}: {e}")
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue
        finally:
            K.clear_session()

        if preds.size == 0 or not np.isfinite(preds).all():
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

        # thresholding + hysteresis + MAD filter
        thr_enter = np.percentile(preds, q_rel * 100.0)
        thr_exit  = np.percentile(preds, max(0.0, (q_rel - hysteresis)) * 100.0)
        med = float(np.median(preds))
        mad = float(np.median(np.abs(preds - med))) + 1e-12
        strong = (np.abs(preds - med) >= mad_k * mad)

        sig = np.zeros_like(preds, dtype=float)
        state = int(prev_sig_last > 0.5)
        for t in range(preds.size):
            if state == 0:
                if (preds[t] >= thr_enter) and strong[t]:
                    state = 1
            else:
                if (preds[t] < thr_exit) or (not strong[t]):
                    state = 0
            sig[t] = float(state)

        sig_full = np.concatenate([[prev_sig_last], sig])
        turnover = np.abs(np.diff(sig_full))
        cost_vec = cost_per_turnover * turnover
        total_turnover += float(turnover.sum())

        # realized PnL
        block_simple = np.exp(block_log) - 1.0
        ret_simple = np.clip(block_simple * sig - cost_vec, -0.9999, None)
        ret_log = np.log1p(ret_simple)

        simple_all.append(ret_simple)
        log_all.append(ret_log)
        sig_trace.append(sig)
        turn_trace.append(turnover)

        # extend history & carry state
        hist_X = np.vstack([hist_X, block_feat])
        hist_y = np.concatenate([hist_y, block_tgt])
        prev_sig_last = float(sig[-1])

    if not simple_all:
        return dict(sharpe=0.0, mdd=1.0, ann_return=0.0, ann_vol=0.0, turnover=0.0, n_points=0)

    net_simple = np.concatenate(simple_all)
    net_log    = np.concatenate(log_all)

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
        n_points=int(net_simple.size)
    )

# -----------------------------
# CLI driver (read fronts → pick knees → backtest → checkpoint)
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Backtest Tier-2 LSTM (PC1..PC7)")
    ap.add_argument("--tier2-front-csv", required=True,
                    help="Tier-2 fronts CSV for LSTM (must have window, units, learning_rate/lr, epochs, threshold, sharpe, mdd).")
    ap.add_argument("--tier1-json", required=True,
                    help="Tier-1 champions JSON (list or {'results': [...]}) with per-fold {layers,batch_size,dropout}.")
    ap.add_argument("--test-csv", required=True,
                    help="TEST CSV containing PC1..PC7, 'target', and 'target_log_returns' (or 'Log_Returns').")
    ap.add_argument("--intervals", default="10,20,42",
                    help="Comma-separated retrain intervals to evaluate.")
    ap.add_argument("--outdir", default="data/backtest_lstm", help="Where to save results.")
    ap.add_argument("--cost-per-turnover", type=float, default=BASE_COST + SLIPPAGE)
    ap.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    ap.add_argument("--hysteresis", type=float, default=0.05)
    ap.add_argument("--mad-k", type=float, default=0.5)
    ap.add_argument("--warmup-len", type=int, default=252)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    setup_logging(args.debug)
    setup_tf(seed=42)

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    result_csv = outdir / "backtest_lstm_results.csv"

    # --- Load Tier-2 fronts ---
    fronts = pd.read_csv(args.tier2_front_csv)
    need_cols = {"fold_id","retrain_interval","front_type","window","units","sharpe","mdd"}
    if not need_cols.issubset(fronts.columns):
        missing = need_cols - set(fronts.columns)
        raise ValueError(f"Tier-2 fronts CSV missing columns: {missing}")

    # --- Load Tier-1 champions ---
    with open(args.tier1_json, "r") as f:
        t1 = json.load(f)
    if isinstance(t1, dict) and "results" in t1:
        t1 = t1["results"]
    # infer key names: champion dict might be under 'champion' or 'best_params'
    champ_map: Dict[int, Dict] = {}
    for rec in t1:
        fid = int(rec.get("fold_id"))
        ch  = rec.get("champion") or rec.get("best_params") or {}
        # normalize field names
        champ_map[fid] = dict(
            layers=int(ch.get("layers", ch.get("num_layers", 1))),
            batch_size=int(ch.get("batch_size", 32)),
            dropout=float(ch.get("dropout", 0.2)),
            patience=int(ch.get("patience", 5)),
        )

    # --- Load TEST set ---
    test_df = pd.read_csv(args.test_csv)
    for c in PC_FEATURES + [TARGET_COL]:
        if c not in test_df.columns:
            raise ValueError(f"Test CSV must contain column '{c}'")
    if (RETN_COL not in test_df.columns) and ("Log_Returns" not in test_df.columns):
        raise ValueError("Test CSV must contain 'target_log_returns' or 'Log_Returns' for realized returns.")

    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]
    fold_ids = sorted(fronts["fold_id"].unique().tolist())

    # --- Resume (checkpoint) ---
    completed = set()
    if result_csv.exists():
        try:
            prev = pd.read_csv(result_csv)
            for _, r in prev.iterrows():
                completed.add((int(r["fold_id"]), int(r["retrain_interval"]), str(r["source"])))
            logging.info(f"[RESUME] Found {len(prev)} rows. Already done pairs will be skipped.")
        except Exception as e:
            logging.warning(f"[RESUME] Could not read existing results ({e}).")

    rows = []
    for fid in fold_ids:
        if fid not in champ_map:
            logging.warning(f"[WARN] Tier-1 champion not found for fold {fid}; skipping.")
            continue
        champion = champ_map[fid]

        for interval in intervals:
            for src in ("GA_knee", "GA+BO_knee"):
                if (fid, interval, src) in completed:
                    if args.debug:
                        logging.debug(f"[SKIP] Done: fold={fid}, int={interval}, src={src}")
                    continue

                pick = pick_knee_from_front(fronts, fid, interval, "GA" if src=="GA_knee" else "GA+BO")
                if pick is None:
                    if args.debug:
                        logging.debug(f"[MISS] Knee not found for fold={fid}, int={interval}, src={src}")
                    continue

                metrics = backtest_lstm(
                    test_df=test_df,
                    feature_cols=PC_FEATURES,
                    champion=champion,
                    t2_vars=pick,
                    retrain_interval=interval,
                    cost_per_turnover=args.cost_per_turnover,
                    min_block_vol=args.min_block_vol,
                    hysteresis=args.hysteresis,
                    mad_k=args.mad_k,
                    warmup_len=args.warmup_len,
                    debug=args.debug
                )

                row = {
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "source": src,
                    "window": int(pick["window"]),
                    "units": int(pick["units"]),
                    "learning_rate": float(pick["lr"]),
                    "epochs": int(pick["epochs"]),
                    "threshold": float(pick["threshold"]),
                    "val_sharpe_front": float(pick["sharpe"]),
                    "val_mdd_front": float(pick["mdd"]),
                    "test_sharpe": float(metrics["sharpe"]),
                    "test_mdd": float(metrics["mdd"]),
                    "test_ann_return": float(metrics["ann_return"]),
                    "test_ann_vol": float(metrics["ann_vol"]),
                    "test_turnover": float(metrics["turnover"]),
                    "test_n_points": int(metrics["n_points"])
                }
                rows.append(row)

                # append immediately (checkpoint)
                df1 = pd.DataFrame([row])
                header = (not result_csv.exists()) or os.path.getsize(result_csv) == 0
                df1.to_csv(result_csv, mode="a", index=False, header=header)
                completed.add((fid, interval, src))

    # Final tidy table
    if rows or result_csv.exists():
        df_all = (pd.DataFrame(rows) if rows else pd.read_csv(result_csv))
        df_all = df_all[[
            "fold_id","retrain_interval","source",
            "window","units","learning_rate","epochs","threshold",
            "val_sharpe_front","val_mdd_front",
            "test_sharpe","test_mdd","test_ann_return","test_ann_vol","test_turnover","test_n_points"
        ]].sort_values(["retrain_interval","fold_id","source"])
        df_all.to_csv(result_csv, index=False)
        logging.info(f"[OK] Saved backtest table -> {result_csv}")

        # quick summary by interval & source
        summary = (df_all.groupby(["retrain_interval","source"], as_index=False)
                   .agg(test_sharpe_median=("test_sharpe","median"),
                        test_mdd_median=("test_mdd","median"),
                        test_sharpe_mean=("test_sharpe","mean"),
                        test_mdd_mean=("test_mdd","mean"),
                        n=("fold_id","count")))
        summary_csv = outdir / "backtest_lstm_summary_by_interval.csv"
        summary.to_csv(summary_csv, index=False)
        logging.info(f"[OK] Saved summary -> {summary_csv}")

    print("\n=== LSTM Backtest complete ===")
    print(f"Results table : {result_csv}")

if __name__ == "__main__":
    main()