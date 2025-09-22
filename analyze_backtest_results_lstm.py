import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------- Defaults ----------
EPS = 1e-8
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015
PC_FEATURES = [f"PC{i}" for i in range(1, 8)]  # PC1..PC7
TARGET_COL = "target"
RETN_COL   = "target_log_returns"

# ---------- Logging / TF ----------
def setup_logging(debug: bool):
    logging.basicConfig(level=(logging.DEBUG if debug else logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

def setup_tf(seed: int = 42, use_mixed_precision: bool = True):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            tf.config.optimizer.set_jit(True)
    except Exception:
        pass

# ---------- IO ----------
def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

# ---------- Metrics ----------
def sharpe_ratio(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if r.size < 2: return 0.0
    sd = float(np.std(r, ddof=1))
    return 0.0 if sd == 0.0 else (float(np.mean(r))/sd) * np.sqrt(252.0)

def max_drawdown_from_log(logr: np.ndarray) -> float:
    C = np.exp(np.cumsum(np.asarray(logr, dtype=float)))
    peak = np.maximum.accumulate(C)
    dd = (peak - C) / (peak + EPS)
    return float(np.max(dd))

def daily_volatility(ret_simple: np.ndarray) -> float:
    r = np.asarray(ret_simple, dtype=float)
    if r.size < 2: return 0.0
    return float(np.std(r, ddof=1))

def annualized_volatility(ret_simple: np.ndarray, periods_per_year: int = 252) -> float:
    return daily_volatility(ret_simple) * np.sqrt(periods_per_year)

# ---------- Helpers ----------
def ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL not in df.columns:
        if "Log_Returns" in df.columns:
            df[TARGET_COL] = df["Log_Returns"].shift(-1)
        elif RETN_COL in df.columns:
            df[TARGET_COL] = df[RETN_COL]
        else:
            raise ValueError("Missing 'target' and 'Log_Returns' to build target.")
    if RETN_COL not in df.columns:
        if "Log_Returns" in df.columns:
            df[RETN_COL] = df["Log_Returns"].shift(-1)
        else:
            df[RETN_COL] = df[TARGET_COL]
    df = df.dropna(subset=[TARGET_COL, RETN_COL]).reset_index(drop=True)
    return df

def create_windows(X: np.ndarray, y: np.ndarray, lookback: int):
    from numpy.lib.stride_tricks import sliding_window_view
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.shape[0] <= lookback:
        return None, None
    wins = sliding_window_view(X, window_shape=lookback, axis=0)
    if wins.ndim == 4:
        wins = wins.squeeze(1)
    Xw = wins[:-1]
    yw = y[lookback:]
    if Xw.shape[1] != lookback:
        Xw = np.transpose(Xw, (0,2,1))
    return Xw.astype(np.float32), yw.astype(np.float32)

def build_lstm(window: int, units: int, lr: float, num_features: int,
               layers: int, dropout: float, norm_type: str = "auto") -> Sequential:
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

# ---------- Front/Backtest picking ----------
def pick_knee_from_front(front_df: pd.DataFrame, fold_id: int, interval: int, front_type: str) -> Optional[Dict]:
    sub = front_df[(front_df["fold_id"]==fold_id) &
                   (front_df["retrain_interval"]==interval) &
                   (front_df["front_type"]==front_type)]
    if sub.empty:
        return None
    lr_key = "learning_rate" if "learning_rate" in sub.columns else ("lr" if "lr" in sub.columns else None)
    if lr_key is None: return None

    F = np.c_[ -sub["sharpe"].to_numpy(float), sub["mdd"].to_numpy(float) ]
    f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    idx = int(np.argmin(np.sqrt((f**2).sum(axis=1))))
    row = sub.iloc[idx]
    return dict(window=int(row["window"]), units=int(row["units"]),
                lr=float(row[lr_key]), epochs=int(row.get("epochs", 20)),
                threshold=float(row.get("threshold", row.get("rel_thresh", 0.6))),
                val_sharpe=float(row["sharpe"]), val_mdd=float(row["mdd"]))

def pick_from_backtest(backtest_df: pd.DataFrame, fold_id: int, interval: int, source_label: str) -> Optional[Dict]:
    sub = backtest_df[(backtest_df["fold_id"]==fold_id) &
                      (backtest_df["retrain_interval"]==interval) &
                      (backtest_df["source"]==source_label)]
    if sub.empty: return None
    row = sub.iloc[0]
    lr_key = "learning_rate" if "learning_rate" in sub.columns else ("lr" if "lr" in sub.columns else None)
    if lr_key is None: return None
    return dict(window=int(row["window"]), units=int(row["units"]),
                lr=float(row[lr_key]), epochs=int(row.get("epochs", 20)),
                threshold=float(row["threshold"]))

# ---------- Backtest engine (walk-forward, 0/1 signals) ----------
def backtest_lstm_trace(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    champion: Dict,                 # {"layers","batch_size","dropout","patience"?}
    t2_vars: Dict,                  # {"window","units","lr","epochs","threshold"}
    retrain_interval: int,
    cost_per_turnover: float = BASE_COST + SLIPPAGE,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    hysteresis: float = 0.05,
    mad_k: float = 0.5,
    warmup_len: int = 252,
    min_holding_days: int = 0,
    enter_k: int = 1,
    exit_k: int = 1,
    debug: bool = False
) -> Dict:
    assert set(feature_cols).issubset(test_df.columns), "Missing PCA feature(s)."
    assert TARGET_COL in test_df.columns, "Missing 'target'."
    metric_col = RETN_COL if RETN_COL in test_df.columns else ("Log_Returns" if "Log_Returns" in test_df.columns else None)
    if metric_col is None:
        raise ValueError("Need 'target_log_returns' or 'Log_Returns'.")

    X = test_df[feature_cols].to_numpy(np.float32)
    y = test_df[TARGET_COL].to_numpy(np.float32)
    rlog = test_df[metric_col].to_numpy(np.float64)
    n = X.shape[0]
    if n <= warmup_len + 1:
        raise ValueError(f"test length {n} too small for warmup_len={warmup_len}")

    layers     = int(champion.get("layers", 1))
    batch_size = int(champion.get("batch_size", 32))
    dropout    = float(champion.get("dropout", 0.2))
    patience   = int(champion.get("patience", 5))

    window  = int(t2_vars["window"])
    units   = int(t2_vars["units"])
    lr      = float(t2_vars["lr"])
    epochs  = int(t2_vars["epochs"])
    q_rel   = float(t2_vars["threshold"])

    hist_X = X[:warmup_len].copy()
    hist_y = y[:warmup_len].copy()

    start_idx = warmup_len
    pos_all, ret_s_all, ret_l_all, tov_all = [], [], [], []
    prev_sig_last = 0.0

    for start in range(start_idx, n, int(retrain_interval)):
        end = min(start + int(retrain_interval), n)
        if end - start <= 0: continue
        block_log = rlog[start:end]
        block_feat = X[start:end]
        block_tgt  = y[start:end]

        vol = float(np.std(block_log))
        if vol < float(min_block_vol):
            if debug:
                logging.debug(f"[BLK {start}-{end}] skip low vol {vol:.3e}")
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

        Xw, yw = create_windows(hist_X, hist_y, window)
        Xw_val, yw_val = create_windows(
            np.concatenate([hist_X[-window:], block_feat], axis=0),
            np.concatenate([hist_y[-window:], block_tgt], axis=0),
            window
        )
        if Xw is None or Xw_val is None:
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

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
                logging.debug(f"[FIT_FAIL] {e}")
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue
        finally:
            K.clear_session()

        if preds.size == 0 or not np.isfinite(preds).all():
            hist_X = np.vstack([hist_X, block_feat])
            hist_y = np.concatenate([hist_y, block_tgt])
            continue

        thr_enter = np.percentile(preds, q_rel * 100.0)
        thr_exit  = np.percentile(preds, max(0.0, q_rel - hysteresis) * 100.0)
        med = float(np.median(preds))
        mad = float(np.median(np.abs(preds - med))) + 1e-12
        strong = (np.abs(preds - med) >= mad_k * mad)

        sig = np.zeros_like(preds, dtype=float)
        state = int(prev_sig_last > 0.5)
        held = 0
        enter_streak = 0
        exit_streak  = 0
        for t in range(preds.size):
            if (preds[t] >= thr_enter) and strong[t]:
                enter_streak += 1
            else:
                enter_streak = 0
            if (preds[t] < thr_exit) or (not strong[t]):
                exit_streak += 1
            else:
                exit_streak = 0

            if state == 0:
                if enter_streak >= enter_k:
                    state = 1
                    enter_streak = 0
                    held = 1
                else:
                    held = 0
            else:
                if (held >= min_holding_days) and (exit_streak >= exit_k):
                    state = 0
                    exit_streak = 0
                    held = 0
                else:
                    held += 1
            sig[t] = float(state)

        sig_full = np.concatenate([[prev_sig_last], sig])
        turnover = np.abs(np.diff(sig_full))
        cost_vec = (BASE_COST + SLIPPAGE) * turnover if cost_per_turnover is None else cost_per_turnover * turnover

        block_simple = np.exp(block_log) - 1.0
        ret_simple = np.clip(block_simple * sig - cost_vec, -0.9999, None)
        ret_log = np.log1p(ret_simple)

        pos_all.append(sig)
        ret_s_all.append(ret_simple)
        ret_l_all.append(ret_log)
        tov_all.append(turnover)

        hist_X = np.vstack([hist_X, block_feat])
        hist_y = np.concatenate([hist_y, block_tgt])
        prev_sig_last = float(sig[-1])

    if not ret_s_all:
        return dict(positions=np.array([]), ret_simple=np.array([]), ret_log=np.array([]),
                    turnover_series=np.array([]), sharpe=0.0, mdd=1.0, turnover=0.0)

    pos = np.concatenate(pos_all)
    rs  = np.concatenate(ret_s_all)
    rl  = np.concatenate(ret_l_all)
    tov = np.concatenate(tov_all)

    return dict(
        positions=pos,
        ret_simple=rs,
        ret_log=rl,
        turnover_series=tov,
        sharpe=sharpe_ratio(rs),
        mdd=max_drawdown_from_log(rl),
        turnover=float(tov.sum())
    )

# ---------- Subcommands ----------
def rolling_apply(x: np.ndarray, window: int, fn) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window, len(x)+1):
        out[i-1] = fn(x[i-window:i])
    return out

def cmd_viz(args):
    setup_logging(args.debug)
    setup_tf(use_mixed_precision=not args.no_mixed_precision)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Resolve params
    if args.use_backtest:
        bt = read_csv_required(Path(args.backtest_csv))
        # >>> FIX: thêm args.source_label <<<
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None:
            raise RuntimeError(
                f"Params not found in backtest CSV for fold={args.fold_id}, "
                f"interval={args.interval}, source={args.source_label}."
            )
    else:
        front = read_csv_required(Path(args.front_csv))
        need = {"fold_id","retrain_interval","front_type","window","units","sharpe","mdd"}
        if not need.issubset(front.columns):
            raise ValueError(f"front CSV missing columns: {need - set(front.columns)}")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None:
            raise RuntimeError(
                f"Knee not found in front CSV for fold={args.fold_id}, "
                f"interval={args.interval}, front_type={args.front_type}."
            )

    # Load test
    test = read_csv_required(Path(args.test_csv))
    test = ensure_targets(test)
    for c in PC_FEATURES + [TARGET_COL]:
        if c not in test.columns:
            raise ValueError(f"Test CSV must contain column '{c}'")

    champion = dict(layers=args.layers, batch_size=args.batch_size, dropout=args.dropout, patience=args.patience)

    # Run trace
    trace = backtest_lstm_trace(
        test_df=test, feature_cols=PC_FEATURES, champion=champion, t2_vars=pick,
        retrain_interval=args.interval, cost_per_turnover=args.cost_per_turnover,
        min_block_vol=args.min_block_vol, hysteresis=args.hysteresis, mad_k=args.mad_k,
        warmup_len=args.warmup_len, min_holding_days=args.min_holding_days,
        enter_k=args.enter_k, exit_k=args.exit_k, debug=args.debug
    )

    # Plots
    equity = np.exp(np.cumsum(trace["ret_log"]))
    fig, ax = plt.subplots(figsize=(8,4), dpi=140)
    ax.plot(equity); ax.set_title(f"Equity — fold {args.fold_id}, int {args.interval}, {args.source_label}")
    ax.set_ylabel("Equity (norm.)"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "equity_curve.png"); plt.close(fig)

    roll_sharpe = rolling_apply(trace["ret_simple"], args.window, sharpe_ratio)
    fig, ax = plt.subplots(figsize=(8,3.5), dpi=140)
    ax.plot(roll_sharpe); ax.set_title(f"Rolling Sharpe (win={args.window})")
    ax.set_ylabel("Sharpe"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "rolling_sharpe.png"); plt.close(fig)

    cum_log = np.cumsum(trace["ret_log"])
    eq = np.exp(cum_log); peak = np.maximum.accumulate(eq)
    dd_series = (peak - eq) / (peak + EPS)
    fig, ax = plt.subplots(figsize=(8,3.5), dpi=140)
    ax.plot(dd_series); ax.set_title("Drawdown path"); ax.set_ylabel("Drawdown"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "rolling_drawdown.png"); plt.close(fig)

    # Turnover rolling mean
    def _mean(x): x=np.asarray(x,dtype=float); return float(np.mean(x)) if x.size else np.nan
    roll_turn = rolling_apply(trace["turnover_series"], args.turnover_window, _mean)
    if args.annualize_turnover:
        roll_turn = roll_turn * (252.0 / float(args.turnover_window))
    fig, ax = plt.subplots(figsize=(8,3.5), dpi=140)
    ax.plot(roll_turn)
    ttl = f"Rolling Turnover (win={args.turnover_window})"
    if args.annualize_turnover: ttl += " — annualized"
    ax.set_title(ttl); ax.set_ylabel("Turnover"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "rolling_turnover.png"); plt.close(fig)

    meta = dict(
        fold_id=args.fold_id, interval=args.interval,
        params=pick, test_stats=dict(sharpe=trace["sharpe"], mdd=trace["mdd"], turnover=trace["turnover"])
    )
    with open(outdir/"run_meta.json","w") as f: json.dump(meta, f, indent=2)
    print(f"[OK] Saved viz to {outdir}")

def cmd_sensitivity(args):
    setup_logging(args.debug)
    setup_tf(use_mixed_precision=not args.no_mixed_precision)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.use_backtest:
        bt = read_csv_required(Path(args.backtest_csv))
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None: raise RuntimeError("Params not found in backtest CSV.")
    else:
        front = read_csv_required(Path(args.front_csv))
        need = {"fold_id","retrain_interval","front_type","window","units","sharpe","mdd"}
        if not need.issubset(front.columns):
            raise ValueError(f"front CSV missing columns: {need - set(front.columns)}")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None: raise RuntimeError("Knee not found in front CSV.")

    test = read_csv_required(Path(args.test_csv))
    test = ensure_targets(test)
    for c in PC_FEATURES + [TARGET_COL]:
        if c not in test.columns:
            raise ValueError(f"Test CSV must contain column '{c}'")

    champion = dict(layers=args.layers, batch_size=args.batch_size, dropout=args.dropout, patience=args.patience)

    cost_mults = [float(x) for x in args.cost_mults.split(",") if x.strip()]
    thr_mults  = [float(x) for x in args.thr_mults.split(",") if x.strip()]

    rows = []
    for cm in cost_mults:
        for tm in thr_mults:
            pick2 = dict(pick)
            pick2["threshold"] = pick["threshold"] * tm
            res = backtest_lstm_trace(
                test_df=test, feature_cols=PC_FEATURES, champion=champion, t2_vars=pick2,
                retrain_interval=args.interval, cost_per_turnover=args.cost_per_turnover * cm,
                min_block_vol=args.min_block_vol, hysteresis=args.hysteresis, mad_k=args.mad_k,
                warmup_len=args.warmup_len, min_holding_days=args.min_holding_days,
                enter_k=args.enter_k, exit_k=args.exit_k, debug=args.debug
            )
            rows.append({
                "fold_id": args.fold_id, "interval": args.interval,
                "p_window": pick["window"], "p_units": pick["units"],
                "cost_mult": cm, "thr_mult": tm,
                "test_sharpe": float(res["sharpe"]),
                "test_mdd": float(res["mdd"]),
                "test_turnover": float(res["turnover"])
            })

    df = pd.DataFrame(rows)
    csv_path = outdir / "sensitivity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved sensitivity table -> {csv_path}")

    if args.plot:
        # Heatmap Sharpe
        pv = df.pivot(index="thr_mult", columns="cost_mult", values="test_sharpe")
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
        im = ax.imshow(pv.to_numpy(), aspect="auto"); fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index)
        ax.set_xlabel("Cost multiplier"); ax.set_ylabel("Threshold multiplier")
        ax.set_title("Sensitivity: Test Sharpe")
        fig.tight_layout(); fig.savefig(outdir/"heatmap_sharpe.png"); plt.close(fig)

        pv2 = df.pivot(index="thr_mult", columns="cost_mult", values="test_mdd")
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
        im = ax.imshow(pv2.to_numpy(), aspect="auto"); fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pv2.columns))); ax.set_xticklabels(pv2.columns)
        ax.set_yticks(range(len(pv2.index)));   ax.set_yticklabels(pv2.index)
        ax.set_xlabel("Cost multiplier"); ax.set_ylabel("Threshold multiplier")
        ax.set_title("Sensitivity: Test MDD")
        fig.tight_layout(); fig.savefig(outdir/"heatmap_mdd.png"); plt.close(fig)

        pv3 = df.pivot(index="thr_mult", columns="cost_mult", values="test_turnover")
        fig, ax = plt.subplots(figsize=(6,4), dpi=140)
        im = ax.imshow(pv3.to_numpy(), aspect="auto"); fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pv3.columns))); ax.set_xticklabels(pv3.columns)
        ax.set_yticks(range(len(pv3.index)));   ax.set_yticklabels(pv3.index)
        ax.set_xlabel("Cost multiplier"); ax.set_ylabel("Threshold multiplier")
        ax.set_title("Sensitivity: Turnover")
        fig.tight_layout(); fig.savefig(outdir/"heatmap_turnover.png"); plt.close(fig)
        print(f"[OK] Saved heatmaps -> {outdir}")

def cmd_cumret(args):
    setup_logging(args.debug)
    setup_tf(use_mixed_precision=not args.no_mixed_precision)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.use_backtest:
        bt = read_csv_required(Path(args.backtest_csv))
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None: raise RuntimeError("Params not found in backtest CSV.")
    else:
        front = read_csv_required(Path(args.front_csv))
        need = {"fold_id","retrain_interval","front_type","window","units","sharpe","mdd"}
        if not need.issubset(front.columns):
            raise ValueError(f"front CSV missing columns: {need - set(front.columns)}")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None: raise RuntimeError("Knee not found in front CSV.")

    test = read_csv_required(Path(args.test_csv))
    test = ensure_targets(test)
    for c in PC_FEATURES + [TARGET_COL]:
        if c not in test.columns:
            raise ValueError(f"Test CSV must contain column '{c}'")

    champion = dict(layers=args.layers, batch_size=args.batch_size, dropout=args.dropout, patience=args.patience)

    trace = backtest_lstm_trace(
        test_df=test, feature_cols=PC_FEATURES, champion=champion, t2_vars=pick,
        retrain_interval=args.interval, cost_per_turnover=args.cost_per_turnover,
        min_block_vol=args.min_block_vol, hysteresis=args.hysteresis, mad_k=args.mad_k,
        warmup_len=args.warmup_len, min_holding_days=args.min_holding_days,
        enter_k=args.enter_k, exit_k=args.exit_k, debug=args.debug
    )
    cumret = np.cumprod(1.0 + trace["ret_simple"]) - 1.0

    fig, ax = plt.subplots(figsize=(8,4), dpi=140)
    ax.plot(100.0 * cumret)
    ax.set_title(f"Cumulative Return — fold {args.fold_id}, int {args.interval}")
    ax.set_ylabel("Return (%)"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir / "cumulative_return.png"); plt.close(fig)

    if args.write_csv:
        pd.DataFrame({"cum_return": cumret}).to_csv(outdir/"cumulative_return_series.csv", index=False)

    with open(outdir/"cumret_meta.json","w") as f:
        json.dump(dict(fold_id=args.fold_id, interval=args.interval,
                       params=pick, cumulative_return=float(cumret[-1]) if cumret.size else 0.0), f, indent=2)
    print(f"[OK] Saved cumulative_return.png to {outdir}")

def cmd_vol(args):
    setup_logging(args.debug)
    setup_tf(use_mixed_precision=not args.no_mixed_precision)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.use_backtest:
        bt = read_csv_required(Path(args.backtest_csv))
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None: raise RuntimeError("Params not found in backtest CSV.")
    else:
        front = read_csv_required(Path(args.front_csv))
        need = {"fold_id","retrain_interval","front_type","window","units","sharpe","mdd"}
        if not need.issubset(front.columns):
            raise ValueError(f"front CSV missing columns: {need - set(front.columns)}")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None: raise RuntimeError("Knee not found in front CSV.")

    test = read_csv_required(Path(args.test_csv))
    test = ensure_targets(test)
    for c in PC_FEATURES + [TARGET_COL]:
        if c not in test.columns:
            raise ValueError(f"Test CSV must contain column '{c}'")

    champion = dict(layers=args.layers, batch_size=args.batch_size, dropout=args.dropout, patience=args.patience)

    trace = backtest_lstm_trace(
        test_df=test, feature_cols=PC_FEATURES, champion=champion, t2_vars=pick,
        retrain_interval=args.interval, cost_per_turnover=args.cost_per_turnover,
        min_block_vol=args.min_block_vol, hysteresis=args.hysteresis, mad_k=args.mad_k,
        warmup_len=args.warmup_len, min_holding_days=args.min_holding_days,
        enter_k=args.enter_k, exit_k=args.exit_k, debug=args.debug
    )
    rs = trace["ret_simple"]

    vol_d = daily_volatility(rs)
    vol_a = annualized_volatility(rs, periods_per_year=args.periods_per_year)

    roll_fn = lambda x: np.std(x, ddof=1) * np.sqrt(args.periods_per_year)
    roll_vol = rolling_apply(rs, args.window, roll_fn)

    fig, ax = plt.subplots(figsize=(8,3.8), dpi=140)
    ax.plot(roll_vol)
    ax.set_title(f"Rolling Annualized Volatility (win={args.window}) — fold {args.fold_id}, int {args.interval}")
    ax.set_ylabel("Annualized Vol"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir/"rolling_volatility.png"); plt.close(fig)

    if args.write_csv:
        pd.DataFrame({"rolling_annualized_vol": roll_vol}).to_csv(outdir/"rolling_volatility.csv", index=False)
        pd.DataFrame({"ret_simple": rs}).to_csv(outdir/"ret_simple_series.csv", index=False)

    with open(outdir/"volatility_meta.json","w") as f:
        json.dump(dict(fold_id=args.fold_id, interval=args.interval, params=pick,
                       daily_volatility=vol_d, annualized_volatility=vol_a,
                       window=args.window, periods_per_year=args.periods_per_year), f, indent=2)
    print(f"[OK] Saved rolling_volatility.png to {outdir}")
    print(f"[OK] Daily vol  = {vol_d:.6f}")
    print(f"[OK] Annual vol = {vol_a:.6f}")

def cmd_turnover(args):
    setup_logging(args.debug)
    setup_tf(use_mixed_precision=not args.no_mixed_precision)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.use_backtest:
        bt = read_csv_required(Path(args.backtest_csv))
        pick = pick_from_backtest(bt, args.fold_id, args.interval, args.source_label)
        if pick is None: raise RuntimeError("Params not found in backtest CSV.")
    else:
        front = read_csv_required(Path(args.front_csv))
        need = {"fold_id","retrain_interval","front_type","window","units","sharpe","mdd"}
        if not need.issubset(front.columns):
            raise ValueError(f"front CSV missing columns: {need - set(front.columns)}")
        pick = pick_knee_from_front(front, args.fold_id, args.interval, args.front_type)
        if pick is None: raise RuntimeError("Knee not found in front CSV.")

    test = read_csv_required(Path(args.test_csv))
    test = ensure_targets(test)
    for c in PC_FEATURES + [TARGET_COL]:
        if c not in test.columns:
            raise ValueError(f"Test CSV must contain column '{c}'")

    champion = dict(layers=args.layers, batch_size=args.batch_size, dropout=args.dropout, patience=args.patience)

    trace = backtest_lstm_trace(
        test_df=test, feature_cols=PC_FEATURES, champion=champion, t2_vars=pick,
        retrain_interval=args.interval, cost_per_turnover=args.cost_per_turnover,
        min_block_vol=args.min_block_vol, hysteresis=args.hysteresis, mad_k=args.mad_k,
        warmup_len=args.warmup_len, min_holding_days=args.min_holding_days,
        enter_k=args.enter_k, exit_k=args.exit_k, debug=args.debug
    )

    def _mean(x): x=np.asarray(x,dtype=float); return float(np.mean(x)) if x.size else np.nan
    roll_turn = rolling_apply(trace["turnover_series"], args.turnover_window, _mean)
    if args.annualize_turnover:
        roll_turn = roll_turn * (252.0 / float(args.turnover_window))

    fig, ax = plt.subplots(figsize=(8,3.5), dpi=140)
    ax.plot(roll_turn)
    ttl = f"Rolling Turnover (win={args.turnover_window})"
    if args.annualize_turnover: ttl += " (annualized)"
    ax.set_title(ttl); ax.set_ylabel("Turnover"); ax.set_xlabel("Time")
    fig.tight_layout(); fig.savefig(outdir/"rolling_turnover.png"); plt.close(fig)

    pd.DataFrame({"turnover_series": trace["turnover_series"], "rolling_mean": roll_turn}).to_csv(outdir/"turnover_series.csv", index=False)
    with open(outdir/"turnover_meta.json","w") as f:
        json.dump(dict(fold_id=args.fold_id, interval=args.interval, total_turnover=float(trace["turnover"]),
                       params=pick), f, indent=2)
    print(f"[OK] Saved turnover plots/CSV to {outdir}")

# ---------- CLI ----------
def build_parser():
    ap = argparse.ArgumentParser(description="Tier-2 LSTM viz & sensitivity tools (GA/GA+BO)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common options
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--test-csv", required=True, help="CSV with PC1..PC7, target, and target_log_returns/Log_Returns")
    base.add_argument("--interval", type=int, required=True)
    base.add_argument("--fold-id", type=int, required=True)
    base.add_argument("--cost-per-turnover", type=float, default=BASE_COST+SLIPPAGE)
    base.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    base.add_argument("--hysteresis", type=float, default=0.05)
    base.add_argument("--mad-k", type=float, default=0.5)
    base.add_argument("--warmup-len", type=int, default=252)
    base.add_argument("--min-holding-days", type=int, default=0)
    base.add_argument("--enter-k", type=int, default=1)
    base.add_argument("--exit-k", type=int, default=1)
    base.add_argument("--layers", type=int, default=1, help="Tier-1 champion layers (or default)")
    base.add_argument("--batch-size", type=int, default=32, help="Tier-1 champion batch_size (or default)")
    base.add_argument("--dropout", type=float, default=0.2, help="Tier-1 champion dropout (or default)")
    base.add_argument("--patience", type=int, default=5, help="EarlyStopping patience")
    base.add_argument("--no-mixed-precision", action="store_true")
    base.add_argument("--debug", action="store_true")

    # Param sources
    # from front
    from_front = argparse.ArgumentParser(add_help=False)
    from_front.add_argument("--front-csv", help="Tier-2 *_front.csv (for knee)")
    from_front.add_argument("--front-type", default="GA+BO", choices=["GA","GA+BO"])
    # from backtest
    from_backtest = argparse.ArgumentParser(add_help=False)
    from_backtest.add_argument("--backtest-csv", help="backtest_lstm_results.csv (has window/units/lr/epochs/threshold)")
    from_backtest.add_argument("--source-label", default="GA+BO_knee")

    # viz
    p_viz = sub.add_parser("viz", parents=[base], help="Plot equity / rolling metrics")
    p_viz.add_argument("--outdir", default="viz_outputs/lstm")
    p_viz.add_argument("--window", type=int, default=63)
    p_viz.add_argument("--turnover-window", type=int, default=21)
    p_viz.add_argument("--annualize-turnover", action="store_true")
    p_viz.add_argument("--use-backtest", action="store_true")
    p_viz.add_argument("--front-csv"); p_viz.add_argument("--front-type", choices=["GA","GA+BO"], default="GA+BO")
    p_viz.add_argument("--backtest-csv"); p_viz.add_argument("--source-label", default="GA+BO_knee")

    # sensitivity
    p_sens = sub.add_parser("sensitivity", parents=[base], help="Cost & threshold sensitivity")
    p_sens.add_argument("--outdir", default="sensitivity_outputs/lstm")
    p_sens.add_argument("--cost-mults", default="0.5,1.0,1.5")
    p_sens.add_argument("--thr-mults", default="0.9,1.0,1.1")
    p_sens.add_argument("--plot", action="store_true")
    p_sens.add_argument("--use-backtest", action="store_true")
    p_sens.add_argument("--front-csv"); p_sens.add_argument("--front-type", choices=["GA","GA+BO"], default="GA+BO")
    p_sens.add_argument("--backtest-csv"); p_sens.add_argument("--source-label", default="GA+BO_knee")

    # cumret
    p_cum = sub.add_parser("cumret", parents=[base], help="Cumulative Return only")
    p_cum.add_argument("--outdir", default="cumret_outputs/lstm")
    p_cum.add_argument("--write-csv", action="store_true")
    p_cum.add_argument("--use-backtest", action="store_true")
    p_cum.add_argument("--front-csv"); p_cum.add_argument("--front-type", choices=["GA","GA+BO"], default="GA+BO")
    p_cum.add_argument("--backtest-csv"); p_cum.add_argument("--source-label", default="GA+BO_knee")

    # vol
    p_vol = sub.add_parser("vol", parents=[base], help="Realized & rolling volatility")
    p_vol.add_argument("--outdir", default="volatility/lstm")
    p_vol.add_argument("--window", type=int, default=63)
    p_vol.add_argument("--periods-per-year", type=int, default=252)
    p_vol.add_argument("--write-csv", action="store_true")
    p_vol.add_argument("--use-backtest", action="store_true")
    p_vol.add_argument("--front-csv"); p_vol.add_argument("--front-type", choices=["GA","GA+BO"], default="GA+BO")
    p_vol.add_argument("--backtest-csv"); p_vol.add_argument("--source-label", default="GA+BO_knee")

    # turnover
    p_tov = sub.add_parser("turnover", parents=[base], help="Turnover plots only")
    p_tov.add_argument("--outdir", default="turnover_outputs/lstm")
    p_tov.add_argument("--turnover-window", type=int, default=21)
    p_tov.add_argument("--annualize-turnover", action="store_true")
    p_tov.add_argument("--use-backtest", action="store_true")
    p_tov.add_argument("--front-csv"); p_tov.add_argument("--front-type", choices=["GA","GA+BO"], default="GA+BO")
    p_tov.add_argument("--backtest-csv"); p_tov.add_argument("--source-label", default="GA+BO_knee")

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()

    # guard param sources
    def _need_front():
        if not args.front_csv:
            ap.error(f"{args.cmd}: without --use-backtest you must provide --front-csv")
    def _need_backtest():
        if not args.backtest_csv:
            ap.error(f"{args.cmd}: --use-backtest requires --backtest-csv")

    if args.cmd == "viz":
        if args.use_backtest: _need_backtest()
        else: _need_front()
        cmd_viz(args)
    elif args.cmd == "sensitivity":
        if args.use_backtest: _need_backtest()
        else: _need_front()
        cmd_sensitivity(args)
    elif args.cmd == "cumret":
        if args.use_backtest: _need_backtest()
        else: _need_front()
        cmd_cumret(args)
    elif args.cmd == "vol":
        if args.use_backtest: _need_backtest()
        else: _need_front()
        cmd_vol(args)
    elif args.cmd == "turnover":
        if args.use_backtest: _need_backtest()
        else: _need_front()
        cmd_turnover(args)

if __name__ == "__main__":
    main()
