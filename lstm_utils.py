# lstm_utils_new.py
# ==========================================
# Utilities for 2-tier LSTM pipeline (NEW STRATEGY)
# Tier-1: backbone only (layers, batch_size, dropout) via RMSE
# Tier-2: trading/time-scale knobs (window, units, lr, epochs, rel_thresh) via (-Sharpe, MDD)
# ==========================================

from __future__ import annotations
import os
import gc
import json
import logging
import random
import warnings
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import norm

# --- TensorFlow / Keras ---
import tensorflow as tf
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

# --- pymoo wrappers ---
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

# ==================== Global Config / Defaults ====================
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015
TRADING_DAYS = 252
EPS = 1e-8

# Tier-1 fixed (neutral) knobs used *only* during Tier-1 objective
# (You can override these in Tier-1 script via constructor args)
WINDOW_T1_DEFAULT = 25   # ~1 trading month
UNITS_T1_DEFAULT  = 48
LR_T1_DEFAULT     = 1e-3

# Tier-2 decision variable bounds: [window, units, lr, epochs, rel_thresh]
T2_XL = np.array([10, 32, 1e-4, 10, 0.05], dtype=float)
T2_XU = np.array([40, 96, 1e-2,  80, 0.95], dtype=float)

# Columns you normally exclude from the feature set
NON_FEATURE_KEEP = ["Date", "Ticker", "target", "target_log_returns", "Log_Returns", "Close_raw"]

# ==================== Reproducibility & TF setup ====================
def set_global_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Best-effort TF perf setup
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        try:
            tf.config.optimizer.set_jit(True)  # XLA
        except Exception:
            pass
        try:
            mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            pass
    except Exception:
        pass

set_global_seeds(42)
suppress_warnings()

# ==================== IO helpers ====================
def load_json(path: str | os.PathLike) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)

# ==================== Metrics ====================
def sharpe_ratio(simple_returns: np.ndarray) -> float:
    r = np.asarray(simple_returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return 0.0 if sd == 0.0 else (float(np.mean(r)) / sd) * np.sqrt(TRADING_DAYS)

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

# ==================== Data prep ====================
def ensure_cols(df: pd.DataFrame, cols: List[str], ctx: str = "") -> bool:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        logging.warning(f"[{ctx}] Missing columns: {miss}")
        return False
    return True

def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    # numeric, and not in the drop list
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_KEEP and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols

def create_windows(X: np.ndarray, y: np.ndarray, lookback: int):
    """Build sliding windows (N,L,F) and targets (N,)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    N, F = X.shape
    if N <= lookback:
        return None, None
    all_w = sliding_window_view(X, window_shape=lookback, axis=0)
    wins = all_w[:-1]
    tars = y[lookback:]
    if wins.ndim == 4:  # (N-L+1, 1, L, F)
        wins = wins.squeeze(1)
    if wins.shape[1] != lookback:  # safety
        wins = np.transpose(wins, (0, 2, 1))
    return wins.astype(np.float32), tars.astype(np.float32)

# ==================== LSTM model ====================
def build_lstm(window: int, units: int, lr: float, num_features: int,
               layers: int, dropout_rate: float = 0.2, norm_type: str = "auto") -> Sequential:
    def _norm_layer():
        if norm_type == "layer":
            return LayerNormalization()
        if norm_type == "batch":
            return BatchNormalization()
        return LayerNormalization()
    model = Sequential()
    model.add(Input((int(window), int(num_features))))
    model.add(LSTM(int(units), return_sequences=(layers > 1)))
    model.add(_norm_layer()); model.add(Dropout(float(dropout_rate)))
    if layers > 1:
        model.add(LSTM(max(8, int(units // 2))))
        model.add(_norm_layer()); model.add(Dropout(float(dropout_rate)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss="mse")
    return model

# ==================== Tier-1: BACKBONE-ONLY objective ====================
def lstm_rmse_backbone(
    params: List[float],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    window_t1: int = WINDOW_T1_DEFAULT,
    units_t1: int = UNITS_T1_DEFAULT,
    lr_t1: float = LR_T1_DEFAULT,
    t1_epochs: int = 10,
    patience: int = 3,
) -> float:
    """
    Tier-1 RMSE (validation) while holding *neutral* window/units/lr fixed.
    Params: [layers, batch_size, dropout]
    """
    layers, batch_size, dropout = params
    layers     = int(np.clip(np.rint(layers), 1, 3))
    batch_size = int(np.clip(np.rint(batch_size), 16, 256))
    dropout    = float(np.clip(dropout, 0.0, 0.6))

    ok = all([
        ensure_cols(train_df, feature_cols + ["target"], "T1-TRAIN"),
        ensure_cols(val_df,   feature_cols + ["target"], "T1-VAL"),
    ])
    if not ok:
        return np.inf

    X_tr = train_df[feature_cols].to_numpy(np.float32)
    y_tr = train_df["target"].to_numpy(np.float32)
    X_val = val_df[feature_cols].to_numpy(np.float32)
    y_val = val_df["target"].to_numpy(np.float32)

    Xw, yw = create_windows(X_tr, y_tr, window_t1)
    if Xw is None: return np.inf
    Xw_val, yw_val = create_windows(
        np.concatenate([X_tr[-window_t1:], X_val], axis=0),
        np.concatenate([y_tr[-window_t1:], y_val], axis=0),
        window_t1
    )
    if Xw_val is None: return np.inf

    try:
        K.clear_session()
        norm_type = "layer" if batch_size <= 32 else "batch"
        model = build_lstm(window_t1, units_t1, lr_t1, X_tr.shape[1],
                           layers, dropout_rate=dropout, norm_type=norm_type)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
        model.fit(Xw, yw, validation_data=(Xw_val, yw_val),
                  epochs=t1_epochs, batch_size=batch_size, callbacks=[es], verbose=0)
        preds = model.predict(Xw_val, verbose=0).ravel().astype(np.float32)
        rmse = float(np.sqrt(np.mean((preds - yw_val) ** 2)))
        return rmse
    except Exception as e:
        logging.warning(f"[Tier1] RMSE eval error: {e}")
        return np.inf
    finally:
        K.clear_session(); gc.collect()

class Tier1LSTMBackboneProblem(ElementwiseProblem):
    """
    Decision vars (Tier-1): [layers, batch_size, dropout]
    Objective: minimize RMSE on validation (with neutral window/units/lr).
    """
    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 feature_cols: List[str],
                 window_t1: int = WINDOW_T1_DEFAULT,
                 units_t1: int = UNITS_T1_DEFAULT,
                 lr_t1: float = LR_T1_DEFAULT,
                 t1_epochs: int = 10,
                 patience: int = 3):
        super().__init__(n_var=3, n_obj=1,
                         xl=np.array([1,  16, 0.0], dtype=float),
                         xu=np.array([3, 256, 0.6], dtype=float))
        self.train_df   = train_df
        self.val_df     = val_df
        self.feature_cols = list(feature_cols)
        self.window_t1  = int(window_t1)
        self.units_t1   = int(units_t1)
        self.lr_t1      = float(lr_t1)
        self.t1_epochs  = int(t1_epochs)
        self.patience   = int(patience)

    def _evaluate(self, x, out, *_, **__):
        layers     = float(x[0])
        batch_size = float(x[1])
        dropout    = float(x[2])
        rmse = lstm_rmse_backbone(
            [layers, batch_size, dropout],
            self.train_df, self.val_df, self.feature_cols,
            window_t1=self.window_t1, units_t1=self.units_t1, lr_t1=self.lr_t1,
            t1_epochs=self.t1_epochs, patience=self.patience
        )
        out["F"] = [rmse]

# ==================== Tier-2: Problem & Objective ====================
class Tier2MOGAProblem(ElementwiseProblem):
    """
    Decision vars: [window, units, lr, epochs, rel_thresh]
    Objectives (min): f0 = -Sharpe(simple PnL), f1 = MDD(log PnL)
    """
    def __init__(self, obj_func: Callable[[np.ndarray], Tuple[float, float]]):
        super().__init__(n_var=5, n_obj=2, xl=T2_XL.copy(), xu=T2_XU.copy())
        self.obj = obj_func

    def _evaluate(self, x, out, *_, **__):
        window     = int(np.clip(np.rint(float(x[0])), T2_XL[0], T2_XU[0]))
        units      = int(np.clip(np.rint(float(x[1])), T2_XL[1], T2_XU[1]))
        lr         = float(x[2])
        epochs     = int(np.clip(np.rint(float(x[3])), T2_XL[3], T2_XU[3]))
        rel_thresh = float(np.clip(float(x[4]),      T2_XL[4], T2_XU[4]))
        f0, f1 = self.obj(np.array([window, units, lr, epochs, rel_thresh], dtype=float))
        out["F"] = [f0, f1]

def create_periodic_lstm_objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    champ: Dict,                      # Tier-1 best: {'layers','batch_size','dropout'}
    retrain_interval: int,
    cost_per_turnover: float = BASE_COST + SLIPPAGE,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    metric_col: str = "target_log_returns",   # realized log-returns for PnL/MDD
    debug: bool = False,
):
    """
    Multi-objective (minimize): f0 = -Sharpe(simple PnL), f1 = MDD(log PnL).
    Signals: 0/1 via percentile threshold (with hysteresis) & MAD amplitude guard.
    Costs charged on turnover (|sig_t - sig_{t-1}|).
    Exposes objective.stats_map[(w,u,round(lr,6),ep,round(thr,6))] = {turnover, raw_sharpe, raw_mdd, penalty}.
    """
    req_ok = all([
        ensure_cols(train_df, feature_cols + ["target"], "T2-TRAIN"),
        ensure_cols(val_df,   feature_cols + ["target", metric_col], "T2-VAL")
    ])
    if not req_ok:
        raise ValueError("[Tier2-LSTM] Missing required columns.")

    X_tr = train_df[feature_cols].to_numpy(np.float32)
    y_tr = train_df["target"].to_numpy(np.float32)
    X_val = val_df[feature_cols].to_numpy(np.float32)
    y_val_for_train = val_df["target"].to_numpy(np.float32)
    r_log_val = val_df[metric_col].to_numpy(np.float64)

    layers    = int(champ.get("layers", 1))
    batch_sz  = int(champ.get("batch_size", 32))
    dropout   = float(champ.get("dropout", 0.2))
    patience  = int(champ.get("patience", 5))

    # Thresholding & robustness settings
    HYSTERESIS = 0.05   # exit threshold = q - 0.05 (in percentile units)
    MAD_K = 0.5         # minimum deviation from median (MAD units) to be actionable

    # caches
    win_cache_tr: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray]] = {}
    win_cache_val: Dict[Tuple[int,int,int,int], Tuple[np.ndarray, np.ndarray]] = {}

    set_global_seeds(42)

    # penalties
    W_NO_TRADE = 20.0
    W_COLLAPSE = 20.0
    W_LOW_VOL  =  2.0

    stats_map: Dict[Tuple[int,int,float,int,float], Dict] = {}

    def objective(x: np.ndarray) -> Tuple[float, float]:
        nonlocal stats_map
        w   = int(np.clip(np.rint(x[0]), T2_XL[0], T2_XU[0]))
        u   = int(np.clip(np.rint(x[1]), T2_XL[1], T2_XU[1]))
        lr  = float(x[2])
        ep  = int(np.clip(np.rint(x[3]), T2_XL[3], T2_XU[3]))
        thr = float(x[4])

        penalty = 0.0
        hist_X, hist_y = X_tr.copy(), y_tr.copy()

        simple_all, log_all = [], []
        total_turnover = 0.0
        prev_sig_last = 0.0

        n_val = len(r_log_val)
        if n_val != len(y_val_for_train):
            return (1e3, 1e3)

        for start in range(0, n_val, int(retrain_interval)):
            end = min(start + int(retrain_interval), n_val)
            if end - start <= 0:
                continue

            block_log = r_log_val[start:end]
            block_vol = float(np.std(block_log))
            if block_vol < float(min_block_vol):
                penalty += W_LOW_VOL * (min_block_vol - block_vol)
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                if debug:
                    logging.debug(f"[BLK {start:04d}-{end:04d}] SKIP vol={block_vol:.3e} < {min_block_vol:.3e}")
                continue

            # Train windows
            cache_key_tr = (hist_X.shape[0], w)
            if cache_key_tr in win_cache_tr:
                Xw, yw = win_cache_tr[cache_key_tr]
            else:
                Xw, yw = create_windows(hist_X, hist_y, w)
                if Xw is not None:
                    win_cache_tr[cache_key_tr] = (Xw, yw)
            if Xw is None:
                penalty += W_COLLAPSE
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                continue

            # Val windows (warm-start with history)
            cache_key_val = (hist_X.shape[0], start, end, w)
            if cache_key_val in win_cache_val:
                Xw_val, yw_val = win_cache_val[cache_key_val]
            else:
                Xw_val, yw_val = create_windows(
                    np.concatenate([hist_X[-w:], X_val[start:end]], axis=0),
                    np.concatenate([hist_y[-w:], y_val_for_train[start:end]], axis=0),
                    w
                )
                if Xw_val is not None:
                    win_cache_val[cache_key_val] = (Xw_val, yw_val)
            if Xw_val is None:
                penalty += W_COLLAPSE
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                continue

            try:
                K.clear_session()
                norm_type = "layer" if batch_sz <= 32 else "batch"
                model = build_lstm(w, u, lr, hist_X.shape[1], layers,
                                   dropout_rate=dropout, norm_type=norm_type)
                es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
                model.fit(Xw, yw, validation_data=(Xw_val, yw_val),
                          epochs=ep, batch_size=batch_sz, callbacks=[es], verbose=0)

                # Predict entire block (many-to-one over sliding windows)
                preds = model.predict(Xw_val, verbose=0).ravel().astype(float)
                preds = np.asarray(preds, dtype=float)
                if preds.size == 0 or not np.isfinite(preds).all():
                    penalty += W_COLLAPSE
                    hist_X = np.vstack([hist_X, X_val[start:end]])
                    hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                    continue

                # Percentile-based thresholds with hysteresis
                q = float(np.clip(thr, 0.5, 0.95))
                thr_enter = np.percentile(preds, q*100.0)
                thr_exit  = np.percentile(preds, max(0.0, (q - HYSTERESIS))*100.0)

                # MAD guardrail
                med = float(np.median(preds))
                mad = float(np.median(np.abs(preds - med))) + 1e-12
                strong = (np.abs(preds - med) >= MAD_K * mad)

                # Stateful signal with hysteresis
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

                if sig.sum() == 0:
                    penalty += W_NO_TRADE

                # Turnover & costs
                sig_full = np.concatenate([[prev_sig_last], sig])
                turnover = np.abs(np.diff(sig_full))
                cost_vec = (cost_per_turnover) * turnover
                total_turnover += float(turnover.sum())

                # PnL
                block_simple = np.exp(block_log) - 1.0
                ret_simple = np.clip(block_simple * sig - cost_vec, -0.9999, None)
                ret_log    = np.log1p(ret_simple)

                simple_all.append(ret_simple)
                log_all.append(ret_log)

                if debug:
                    logging.debug(
                        f"[BLK {start:04d}-{end:04d}] w={w},u={u},lr={lr:.4g},ep={ep},thr={thr:.3f} | "
                        f"vol={block_vol:.3e}, turnover={turnover.sum():.2f}, sig_rate={sig.mean():.2f}"
                    )

                # extend history
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                prev_sig_last = float(sig[-1])

            except Exception as e:
                if debug:
                    logging.debug(f"[FIT_FAIL] w={w},u={u},lr={lr:.3g},ep={ep},thr={thr:.3f} | err={e}")
                penalty += W_COLLAPSE
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                K.clear_session()
                continue
            finally:
                K.clear_session(); gc.collect()

        # No valid blocks
        if not simple_all:
            stats_map[(w, u, round(lr,6), ep, round(thr,6))] = dict(
                turnover=float(total_turnover), raw_sharpe=0.0, raw_mdd=1.0, penalty=float(penalty)
            )
            return float(1.0 + penalty), float(1.0 + penalty)

        net_simple = np.concatenate(simple_all)
        net_log    = np.concatenate(log_all)
        raw_sharpe = sharpe_ratio(net_simple)
        raw_mdd    = max_drawdown(net_log, input_type="log")

        f0, f1 = -raw_sharpe, raw_mdd
        if not np.isfinite(f0): penalty += 10.0; f0 = 1.0
        if not np.isfinite(f1): penalty += 10.0; f1 = 1.0

        stats_map[(w, u, round(lr,6), ep, round(thr,6))] = dict(
            turnover=float(total_turnover),
            raw_sharpe=float(raw_sharpe),
            raw_mdd=float(raw_mdd),
            penalty=float(penalty),
        )
        return float(f0 + penalty), float(f1 + penalty)

    objective.stats_map = stats_map
    return objective

# ==================== Pareto / BO helpers ====================
def is_nondominated(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        fi = F[i]
        dom = np.all(F <= fi + 1e-12, axis=1) & np.any(F < fi - 1e-12, axis=1)
        dom[i] = False
        if np.any(dom):
            nd[i] = False
    return nd

def knee_index(F: np.ndarray) -> int:
    f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    d = np.sqrt((f ** 2).sum(axis=1))
    return int(np.argmin(d))

def parego_scalarize(F: np.ndarray, lam: np.ndarray, rho: float = 0.05) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    g = np.max(lam * Fn, axis=1) + rho * np.sum(lam * Fn, axis=1)
    return g

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

class FromTier1SeedSampling(Sampling):
    """
    Seeding + random completion for pymoo.
    For Tier-1 backbone: int indices = [0,1]  (layers, batch_size); dropout is real.
    """
    def __init__(self, seeds: List[List[float]], n_var: int, int_indices: Optional[List[int]] = None):
        super().__init__()
        self.seeds = np.array(seeds, dtype=float) if len(seeds) else np.empty((0, n_var), dtype=float)
        self.n_var = int(n_var)
        self.int_indices = set(int_indices or [])

    def _do(self, problem, n_samples, **kwargs):
        n0 = min(len(self.seeds), n_samples)
        pop = self.seeds[:n0].copy()
        if n_samples > n0:
            rand = np.random.uniform(problem.xl, problem.xu, size=(n_samples - n0, self.n_var))
            pop = np.vstack([pop, rand])
        if pop.size and self.int_indices:
            for idx in self.int_indices:
                pop[:, idx] = np.rint(pop[:, idx])
        pop = np.clip(pop, problem.xl, problem.xu)
        return pop

def run_bo_parego(
    objective: Callable[[np.ndarray], Tuple[float, float]],
    bounds: Tuple[np.ndarray, np.ndarray],     # (lb, ub)
    init_X: List[np.ndarray],
    init_F: List[np.ndarray],
    n_iter: int = 40,
    n_pool: int = 2000,
    random_state: int = 42,
    int_indices: Optional[List[int]] = None,
    round_digits: Optional[Dict[int, int]] = None,
    eval_logger: Optional[Callable[[np.ndarray, Tuple[float,float], int], None]] = None,
) -> List[Dict]:
    """
    Generic ParEGO loop for multi-objective BO.
    - int_indices: dims to round-to-int
    - round_digits: {dim: k} rounding precision for continuous dims
    """
    rng = np.random.default_rng(random_state)
    lb, ub = np.array(bounds[0], float), np.array(bounds[1], float)
    int_idx_set = set(int_indices or [])
    round_digits = round_digits or {}

    X_hist = [np.array(x, dtype=float) for x in init_X]
    F_hist = [np.array(f, dtype=float) for f in init_F]
    rows: List[Dict] = []

    def _round_clip(x: np.ndarray) -> np.ndarray:
        xx = np.array(x, dtype=float)
        for i in range(xx.size):
            if i in int_idx_set:
                xx[i] = np.rint(xx[i])
            if i in round_digits:
                xx[i] = np.round(xx[i], int(round_digits[i]))
        return np.clip(xx, lb, ub)

    for it in range(1, n_iter + 1):
        w = float(rng.uniform(0.05, 0.95))
        lam = np.array([w, 1.0 - w], dtype=float)

        F_arr = np.vstack(F_hist)
        y = parego_scalarize(F_arr, lam)

        X_arr = np.vstack(X_hist)
        Xs = (X_arr - lb) / (ub - lb + 1e-12)
        
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state)
        gp.fit(Xs, y)

        cand = rng.uniform(lb, ub, size=(n_pool, lb.size))
        cand = np.unique(cand, axis=0)

        if len(X_hist) > 0:
            hist_unique = np.unique(np.vstack(X_hist), axis=0)
            def not_in_hist(x):
                return not np.any(np.all(np.isclose(hist_unique, x, atol=1e-8), axis=1))
            mask = np.array([not_in_hist(x) for x in cand])
            cand = cand[mask]

        if cand.shape[0] == 0:
            # jitter around current best in scalarized space
            best_idx = int(np.argmin(y))
            center = X_arr[best_idx]
            jitter = rng.normal(scale=0.05, size=(max(256, lb.size * 64), lb.size)) * (ub - lb)
            cand = np.clip(center + jitter, lb, ub)

        cand_proc = np.vstack([_round_clip(x) for x in cand])
        Xc = (cand_proc - lb) / (ub - lb + 1e-12)
        mu, std = gp.predict(Xc, return_std=True)
        y_best = float(np.min(y))
        ei = expected_improvement(mu, std, y_best)

        idx = int(np.argmax(ei))
        x_next = cand_proc[idx]
        f_next = objective(x_next)

        if eval_logger is not None:
            try:
                eval_logger(x_next, f_next, it)
            except Exception:
                pass

        meta = getattr(objective, "stats_map", {})
        key = tuple([
            int(np.rint(x_next[i])) if i in (int_indices or []) else float(x_next[i])
            for i in range(x_next.size)
        ])
        row = {"stage": "BO", "iter": int(it), "x": x_next.tolist(), "f0": float(f_next[0]), "f1": float(f_next[1])}
        md = meta.get(key, {})
        row.update({
            "turnover": float(md.get("turnover", np.nan)),
            "raw_sharpe": float(md.get("raw_sharpe", np.nan)),
            "raw_mdd": float(md.get("raw_mdd", np.nan)),
            "penalty": float(md.get("penalty", np.nan)),
        })
        rows.append(row)

        X_hist.append(x_next)
        F_hist.append(np.array(f_next, dtype=float))

    return rows