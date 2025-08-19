import os
import gc
import json
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Callable
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import norm

import tensorflow as tf
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

# ==================== CONFIG ====================
BASE_COST = 0.0005
SLIPPAGE  = 0.0002
DEFAULT_MIN_BLOCK_VOL = 0.0015
TRADING_DAYS = 252
EPS = 1e-8

# Tier-1 search bounds: [window, layers, units, lr, batch_size]
T1_XL = np.array([10, 1, 32, 1e-4, 16], dtype=float)
T1_XU = np.array([40, 3, 128, 1e-2, 128], dtype=float)

# Tier-2 decision variable bounds: [window, units, lr, epochs, rel_thresh]
# (layers, batch_size held from Tier-1 champion)
T2_XL = np.array([10, 32, 1e-4, 10, 0.05], dtype=float)
T2_XU = np.array([40, 64, 1e-2,  60, 0.95], dtype=float)

# Feature inference exclusions
NON_FEATURE_KEEP = ["Date", "Ticker", "target", "target_log_returns", "Log_Returns", "Close_raw"]

# ==================== Warnings & session handling ====================
def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# ==================== METRICS ====================
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

# ==================== DATA PREP ====================
def ensure_cols(df: pd.DataFrame, cols: List[str], ctx: str = "") -> bool:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        logging.warning(f"[{ctx}] Missing columns: {miss}")
        return False
    return True

def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_KEEP and not str(c).lower().startswith("target")]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

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
    if wins.ndim == 4:  # (N- L +1, 1, L, F)
        wins = wins.squeeze(1)
    if wins.shape[1] != lookback:  # safety
        wins = np.transpose(wins, (0, 2, 1))
    return wins.astype(np.float32), tars.astype(np.float32)

# ==================== LSTM model ====================
def build_lstm(window: int, units: int, lr: float, num_features: int, layers: int, dropout_rate: float = 0.2) -> Sequential:
    model = Sequential()
    model.add(Input((int(window), int(num_features))))
    model.add(LSTM(int(units), return_sequences=(layers > 1)))
    model.add(BatchNormalization())
    model.add(Dropout(float(dropout_rate)))
    if layers > 1:
        model.add(LSTM(max(8, int(units // 2))))
        model.add(BatchNormalization())
        model.add(Dropout(float(dropout_rate)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss="mse")
    return model

# ==================== Tier-1 objective (RMSE on validation) ====================
def lstm_rmse(params: List[float],
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              feature_cols: List[str],
              t1_epochs: int = 10,
              patience: int = 3) -> float:
    """
    Params: [window, layers, units, lr, batch_size]
    Train on TRAIN windows; validate on VAL with warm started context (train tail + val).
    Returns RMSE on VAL windows.
    """
    w, layers, units, lr, batch_size = params
    w, layers, units, batch_size = map(int, [round(w), round(layers), round(units), round(batch_size)])
    lr = float(lr)

    # checks
    req_ok = all([
        ensure_cols(train_df, feature_cols + ["target"], "T1-TRAIN"),
        ensure_cols(val_df,   feature_cols + ["target"], "T1-VAL")
    ])
    if not req_ok:
        return np.inf

    X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr = train_df["target"].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.float32)

    Xw, yw = create_windows(X_tr, y_tr, w)
    if Xw is None:
        return np.inf

    Xw_val, yw_val = create_windows(
        np.concatenate([X_tr[-w:], X_val], axis=0),
        np.concatenate([y_tr[-w:], y_val], axis=0),
        w
    )
    if Xw_val is None:
        return np.inf

    try:
        K.clear_session()
        model = build_lstm(w, units, lr, X_tr.shape[1], layers, dropout_rate=0.2)
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
        model.fit(Xw, yw, validation_data=(Xw_val, yw_val),
                  epochs=t1_epochs, batch_size=batch_size, callbacks=[es], verbose=0)
        preds = model.predict(Xw_val, verbose=0).ravel().astype(np.float32)
        rmse = float(np.sqrt(np.mean((preds - yw_val) ** 2)))
        return rmse
    except Exception as e:
        logging.warning(f"[Tier1] Eval error: {e}")
        return np.inf
    finally:
        K.clear_session()
        gc.collect()

class Tier1LSTMProblem(ElementwiseProblem):
    """
    pymoo wrapper for Tier-1: minimize RMSE on validation windows.
    Decision vars: [window, layers, units, lr, batch_size] with bounds T1_XL/T1_XU.
    """
    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 feature_cols: List[str],
                 t1_epochs: int = 10,
                 patience: int = 3):
        super().__init__(n_var=5, n_obj=1, xl=T1_XL.copy(), xu=T1_XU.copy())
        self.train_df = train_df
        self.val_df = val_df
        self.feature_cols = list(feature_cols)
        self.t1_epochs = int(t1_epochs)
        self.patience = int(patience)

    def _evaluate(self, x, out, *_, **__):
        # unpack & coerce with explicit clipping/rounding (no "xx")
        window      = int(np.clip(np.rint(float(x[0])), T1_XL[0], T1_XU[0]))
        layers      = int(np.clip(np.rint(float(x[1])), T1_XL[1], T1_XU[1]))
        units       = int(np.clip(np.rint(float(x[2])), T1_XL[2], T1_XU[2]))
        lr          = float(x[3])  # learning-rate is real; no clip here (kernel bounds already enforce it)
        batch_size  = int(np.clip(np.rint(float(x[4])), T1_XL[4], T1_XU[4]))

        params = [window, layers, units, lr, batch_size]
        rmse = lstm_rmse(params,
                         train_df=self.train_df,
                         val_df=self.val_df,
                         feature_cols=self.feature_cols,
                         t1_epochs=self.t1_epochs,
                         patience=self.patience)
        out["F"] = [rmse]

class FromTier1SeedSampling(Sampling):
    """
    Seeding + random completion for pymoo. Rounds integer-like vars.
    For Tier-1: int indices = [0,1,2,4] (window, layers, units, batch).
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
        # round integer vars
        if pop.size and self.int_indices:
            for idx in self.int_indices:
                pop[:, idx] = np.rint(pop[:, idx])
        # clip to bounds
        pop = np.clip(pop, problem.xl, problem.xu)
        return pop

# ==================== Tier-2: multi-objective problem ====================
class Tier2MOGAProblem(ElementwiseProblem):
    """
    Decision vars: [window, units, lr, epochs, rel_thresh]  (layers & batch from T1 champion)
    Objectives (min): f0 = -Sharpe(simple PnL), f1 = MDD(log PnL)
    """
    def __init__(self, obj_func: Callable[[np.ndarray], Tuple[float, float]]):
        super().__init__(n_var=5, n_obj=2, xl=T2_XL.copy(), xu=T2_XU.copy())
        self.obj = obj_func

    def _evaluate(self, x, out, *_, **__):
        window     = int(np.clip(np.rint(float(x[0])), T2_XL[0], T2_XU[0]))
        units      = int(np.clip(np.rint(float(x[1])), T2_XL[1], T2_XU[1]))
        lr         = float(x[2])  # real
        epochs     = int(np.clip(np.rint(float(x[3])), T2_XL[3], T2_XU[3]))
        rel_thresh = float(np.clip(float(x[4]),      T2_XL[4], T2_XU[4]))

        f0, f1 = self.obj(np.array([window, units, lr, epochs, rel_thresh], dtype=float))
        out["F"] = [f0, f1]

# ==================== Tier-2 walk-forward objective (continuous sizing via 0/1 signals) ====================
def create_periodic_lstm_objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    champ: Dict,                      # Tier-1 best params: layers, batch_size, dropout, etc.
    retrain_interval: int,
    cost_per_turnover: float = BASE_COST + SLIPPAGE,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    metric_col: str = "target_log_returns",   # realized log-returns for PnL/MDD
    debug: bool = False,
):
    """
    Multi-objective (minimize): f0 = -Sharpe(simple PnL), f1 = MDD(log PnL).
    Signals are 0/1 via dynamic threshold on predictions per block.
    Costs charged on turnover (|sig_t - sig_{t-1}|).
    Exposes objective.stats_map[(w,u,round(lr,6),ep,round(thr,6))] = {turnover, raw_sharpe, raw_mdd, penalty}.
    """
    req_ok = all([
        ensure_cols(train_df, feature_cols + ["target"], "T2-TRAIN"),
        ensure_cols(val_df,   feature_cols + ["target", metric_col], "T2-VAL")
    ])
    if not req_ok:
        raise ValueError("[Tier2-LSTM] Missing required columns.")

    X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr = train_df["target"].to_numpy(dtype=np.float32)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val_for_train = val_df["target"].to_numpy(dtype=np.float32)
    r_log_val = val_df[metric_col].to_numpy(dtype=np.float64)

    layers    = int(champ.get("layers", 1))
    batch_sz  = int(champ.get("batch_size", 32))
    dropout   = float(champ.get("dropout", 0.2))
    patience  = int(champ.get("patience", 5))

    # penalties
    W_NO_TRADE = 20.0
    W_COLLAPSE = 20.0
    W_LOW_VOL  = 2.0

    stats_map: Dict[Tuple[int,int,float,int,float], Dict] = {}

    def objective(x: np.ndarray) -> Tuple[float, float]:
        nonlocal stats_map
        w  = int(np.clip(np.rint(x[0]), T2_XL[0], T2_XU[0]))
        u  = int(np.clip(np.rint(x[1]), T2_XL[1], T2_XU[1]))
        lr = float(x[2])
        ep = int(np.clip(np.rint(x[3]), T2_XL[3], T2_XU[3]))
        thr= float(x[4])

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
            Xw, yw = create_windows(hist_X, hist_y, w)
            if Xw is None:
                penalty += W_COLLAPSE
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                continue

            # Val windows (warm-start with history)
            Xw_val, yw_val = create_windows(
                np.concatenate([hist_X[-w:], X_val[start:end]], axis=0),
                np.concatenate([hist_y[-w:], y_val_for_train[start:end]], axis=0),
                w
            )
            if Xw_val is None:
                penalty += W_COLLAPSE
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                continue

            try:
                K.clear_session()
                model = build_lstm(w, u, lr, hist_X.shape[1], layers, dropout_rate=dropout)
                es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
                model.fit(Xw, yw, validation_data=(Xw_val, yw_val),
                          epochs=ep, batch_size=batch_sz, callbacks=[es], verbose=0)

                # Rolling predictions on raw block (autoregressive on features)
                preds, buf = [], [x.copy() for x in hist_X[-w:]]
                for i in range(start, end):
                    window_slice = np.array(buf[-w:], dtype=np.float32)
                    if window_slice.shape != (w, hist_X.shape[1]):
                        penalty += W_COLLAPSE
                        preds = None
                        break
                    p = float(model.predict(window_slice[None, ...], verbose=0)[0, 0])
                    preds.append(p)
                    buf.append(X_val[i].copy())

                if preds is None:
                    hist_X = np.vstack([hist_X, X_val[start:end]])
                    hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                    continue

                preds = np.asarray(preds, dtype=float)
                mn, mx = float(preds.min()), float(preds.max())
                if (mx - mn) < 1e-8:
                    penalty += W_COLLAPSE
                    hist_X = np.vstack([hist_X, X_val[start:end]])
                    hist_y = np.concatenate([hist_y, y_val_for_train[start:end]])
                    continue

                thresh_val = mn + (mx - mn) * float(thr)
                sig = (preds > thresh_val).astype(float)
                if sig.sum() == 0:
                    penalty += W_NO_TRADE

                # Turnover & cost
                sig_full = np.concatenate([[prev_sig_last], sig])
                turnover = np.abs(np.diff(sig_full))
                cost_vec = (BASE_COST + SLIPPAGE) * turnover if cost_per_turnover is None else cost_per_turnover * turnover
                total_turnover += float(turnover.sum())

                # PnL (convert log-returns -> simple)
                block_simple = np.exp(block_log) - 1.0
                ret_simple = block_simple * sig - cost_vec
                ret_simple = np.clip(ret_simple, -0.9999, None)
                ret_log = np.log1p(ret_simple)

                simple_all.append(ret_simple)
                log_all.append(ret_log)

                if debug:
                    logging.debug(
                        f"[BLK {start:04d}-{end:04d}] w={w},u={u},lr={lr:.4g},ep={ep},thr={thr:.3f} | "
                        f"vol={block_vol:.3e}, turnover={turnover.sum():.2f}, sig_rate={sig.mean():.2f}"
                    )

                # Extend history
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
                K.clear_session()
                gc.collect()

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

# ==================== Pareto helpers ====================
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
    f = F.copy()
    f = (f - f.min(axis=0)) / (np.ptp(f, axis=0) + 1e-12)
    d = np.sqrt((f ** 2).sum(axis=1))
    return int(np.argmin(d))

# ==================== BO / ParEGO helpers (generic) ====================
def parego_scalarize(F: np.ndarray, lam: np.ndarray, rho: float = 0.05) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    g = np.max(lam * Fn, axis=1) + rho * np.sum(lam * Fn, axis=1)
    return g

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def run_bo_parego(
    objective: Callable[[np.ndarray], Tuple[float, float]],
    bounds: Tuple[np.ndarray, np.ndarray],       # (lb, ub) arrays for each var
    init_X: List[np.ndarray], init_F: List[np.ndarray],
    n_iter: int = 40,
    n_pool: int = 2000,
    random_state: int = 42,
    int_indices: Optional[List[int]] = None,
    round_digits: Optional[Dict[int, int]] = None,
    eval_logger: Optional[Callable[[np.ndarray, Tuple[float,float], int], None]] = None,
) -> List[Dict]:
    """
    Generic ParEGO loop for any dimensionality.
    - int_indices: which dims to round to nearest int
    - round_digits: dict {dim: k} to round float dims to k decimals
    - eval_logger: optional callback(x, f, iter) -> None to record rows externally
    Returns a list of dict rows (stage='BO', params + Sharpe/MDD-like fields if objective fills stats_map).
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
                k = int(round_digits[i])
                xx[i] = np.round(xx[i], k)
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

        # remove already-evaluated points
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

        # round/clip candidates for int dims and specified precision
        cand_proc = np.vstack([_round_clip(x) for x in cand])
        # predict
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

        # try to record stats_map if provided
        meta = getattr(objective, "stats_map", {})
        key = tuple([
            int(np.rint(x_next[i])) if i in int_idx_set else float(x_next[i])
            for i in range(x_next.size)
        ])
        row = {"stage": "BO", "iter": int(it)}
        # Attempt to attach common fields if present
        md = meta.get(key, {})
        row.update({
            "x": x_next.tolist(),
            "f0": float(f_next[0]),
            "f1": float(f_next[1]),
            "turnover": float(md.get("turnover", np.nan)),
            "raw_sharpe": float(md.get("raw_sharpe", np.nan)),
            "raw_mdd": float(md.get("raw_mdd", np.nan)),
            "penalty": float(md.get("penalty", np.nan)),
        })
        rows.append(row)

        X_hist.append(x_next)
        F_hist.append(np.array(f_next, dtype=float))

    return rows