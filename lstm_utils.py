import numpy as np
import pandas as pd
import gc
import logging
import tensorflow as tf
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error
from math import sqrt

from skopt.space import Integer, Real
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==================== CONFIG ====================
LOG_DIR = Path("logs")
DEFAULT_T1_EPOCHS = 10
DEFAULT_T2_EPOCHS = 10
DEFAULT_T2_NGEN_LSTM = 10
PATIENCE_T1 = 3
PATIENCE_T2 = 5
TRADING_DAYS = 252
BASE_COST = 0.0005
SLIPPAGE = 0.0001
GA_POP_SIZE = 20
GA_N_GEN = 15
BO_N_CALLS = 30
TOP_N_SEEDS = 5

# Model-specific search space
LSTM_SEARCH_SPACE = [
    Integer(10, 40, name='window'),
    Integer(1, 3, name='layers'),
    Integer(32, 64, name='units'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
    Integer(16, 64, name='batch_size')
]

# ==================== METRICS ====================
def sharpe_ratio(returns):
    std = np.std(returns)
    return 0.0 if std == 0 else (np.mean(returns) / std) * np.sqrt(TRADING_DAYS)

def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    return np.max((peak - cum) / (peak + np.finfo(float).eps))

# ==================== DATA PREP ====================
def create_windows(X, y, lookback):
    """
    Create sliding windows for sequence modeling.
    Early exit if window is larger than dataset length.
    """
    N, F = X.shape
    if N <= lookback:
        logging.warning(f"[create_windows] Lookback {lookback} exceeds data length {N}.")
        return None, None
    all_w = sliding_window_view(X, window_shape=lookback, axis=0)
    wins = all_w[:-1]
    tars = y[lookback:]
    if wins.shape[1] != lookback or wins.shape[2] != F:
        wins = wins.transpose(0, 2, 1)
    return wins, tars

# ==================== MODEL ====================
def build_lstm(window, units, lr, num_features, layers, dropout_rate=0.2):
    """
    Build LSTM model.
    Dropout is fixed for Tier 1, tunable for Tier 2.
    """
    model = Sequential()
    model.add(Input((window, num_features)))
    model.add(LSTM(units, return_sequences=(layers > 1)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    if layers > 1:
        model.add(LSTM(units // 2))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# ==================== TIER 1 ====================
class Tier1LSTMProblem(ElementwiseProblem):
    def __init__(self, X_tr, y_tr, X_val, y_val, num_features, patience=PATIENCE_T1):
        xl = np.array([dim.low for dim in LSTM_SEARCH_SPACE])
        xu = np.array([dim.high for dim in LSTM_SEARCH_SPACE])
        super().__init__(n_var=len(LSTM_SEARCH_SPACE), n_obj=1, xl=xl, xu=xu)
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val
        self.num_features = num_features
        self.patience = patience

    def _evaluate(self, x, out, *_, **__):
        w, layers, units, lr, batch_size = x
        w, layers, units, batch_size = map(int, [w, layers, units, batch_size])

        Xtr, ytr = create_windows(self.X_tr, self.y_tr, w)
        if Xtr is None:
            out['F'] = [np.inf]
            return

        Xvl, yvl = create_windows(
            np.concatenate([self.X_tr[-w:], self.X_val], axis=0),
            np.concatenate([self.y_tr[-w:], self.y_val], axis=0),
            w
        )
        if Xvl is None:
            out['F'] = [np.inf]
            return

        try:
            K.clear_session()
            model = build_lstm(w, units, lr, self.num_features, layers, dropout_rate=0.2)
            es = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=0)

            model.fit(Xtr, ytr,
                      validation_data=(Xvl, yvl),
                      epochs=DEFAULT_T1_EPOCHS,
                      batch_size=batch_size,
                      callbacks=[es],
                      verbose=0)

            preds = model.predict(Xvl, verbose=0).ravel()
            rmse = sqrt(mean_squared_error(yvl, preds))
        except Exception as e:
            logging.warning(f"[Tier1] Eval error: {e}")
            rmse = np.inf
        finally:
            K.clear_session()
            gc.collect()
        out['F'] = [rmse]

# ==================== TIER 2 ====================
class FromTier1SeedSampling(Sampling):
    def __init__(self, seeds, n_var):
        super().__init__()
        self.seeds = np.array(seeds)
        self.n_var = n_var

    def _do(self, problem, n_samples, **kwargs):
        n0 = min(len(self.seeds), n_samples)
        pop = self.seeds[:n0].copy()
        if n_samples > n0:
            rand = np.random.uniform(problem.xl, problem.xu,
                                     size=(n_samples-n0, self.n_var))
            pop = np.vstack([pop, rand])
        return pop

class Tier2MOGAProblem(ElementwiseProblem):
    def __init__(self, obj):
        xl = np.array([10, 32, 1e-4, 10, 0.0])  # window, units, lr, epochs, rel_thresh
        xu = np.array([40, 64, 1e-2, 50, 1.0])
        super().__init__(n_var=5, n_obj=2, xl=xl, xu=xu)
        self.obj = obj

    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

# ==================== WALK-FORWARD OBJ ====================
def create_periodic_lstm_objective(tr_df, val_df, feats, champ,
                                   retrain_interval, patience=PATIENCE_T2,
                                   penalty_no_trade=100.0):
    """
    Walk-forward LSTM backtest with periodic retraining.
    Soft penalties for collapse or no-trade cases.
    """
    X_full, y_full = tr_df[feats].values, tr_df['target'].values
    X_val, y_val_train = val_df[feats].values, val_df['target'].values
    y_val_metrics = val_df['target_log_returns'].values

    cost = BASE_COST + SLIPPAGE
    n_val = len(y_val_train)

    def obj(x):
        w, units, lr, epochs, rel_thresh = x
        w, units, epochs = map(int, (w, units, epochs))
        lr, rel_thresh = float(lr), float(rel_thresh)

        if w > len(X_full):
            logging.warning(f"[Tier2] Window {w} > train length {len(X_full)}")
            return 1e3, 1e3
        if not 0 < rel_thresh < 1:
            return 1e3, 1e3

        hist_X, hist_y = X_full.copy(), y_full.copy()
        all_returns = []

        try:
            for start in range(0, n_val, retrain_interval):
                end = min(start + retrain_interval, n_val)
                K.clear_session()
                batch_size = champ.get('batch_size')
                layers = champ.get('layers')

                win, tar = create_windows(hist_X, hist_y, w)
                if win is None:
                    return 1e3, 1e3
                ds_tr = tf.data.Dataset.from_tensor_slices((win, tar)) \
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE)

                val_X_block = X_val[start:end]
                val_y_block = y_val_train[start:end]

                win_val, tar_val = create_windows(
                    np.concatenate([hist_X[-w:], val_X_block], axis=0),
                    np.concatenate([hist_y[-w:], val_y_block], axis=0),
                    w
                )
                if win_val is None:
                    return 1e3, 1e3

                val_ds = tf.data.Dataset.from_tensor_slices((win_val, tar_val)) \
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE)

                model = build_lstm(w, units, lr, hist_X.shape[1], layers, dropout_rate=champ.get('dropout', 0.2))
                es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
                model.fit(ds_tr, validation_data=val_ds, epochs=epochs, callbacks=[es], verbose=0)

                # Predict
                preds, buf = [], [x.copy() for x in hist_X[-w:]]
                for i in range(start, end):
                    window_slice = np.array(buf[-w:])
                    if window_slice.ndim != 2 or window_slice.shape[0] != w:
                        logging.warning(f"[Tier2] Bad input window shape: {window_slice.shape}")
                        return 1e3, 1e3
                    p = model.predict(window_slice[None, ...], verbose=0)[0, 0]
                    preds.append(p)
                    buf.append(X_val[i].copy())

                preds = np.array(preds)
                mn, mx = preds.min(), preds.max()
                if mx - mn < 1e-8:
                    logging.warning(f"[Tier2] Prediction collapse in block {start}:{end}")
                    return penalty_no_trade, penalty_no_trade

                thresh_val = mn + (mx - mn) * rel_thresh
                sig = (preds > thresh_val).astype(int)
                if sig.sum() == 0:
                    logging.warning(f"[Tier2] No trades in block {start}:{end}")
                    return penalty_no_trade, penalty_no_trade

                ret_block = y_val_metrics[start:end] * sig - cost * sig
                all_returns.append(ret_block)

                # Expand history
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val_train[start:end]])

            net_returns = np.concatenate(all_returns)
            return -sharpe_ratio(net_returns), max_drawdown(net_returns)

        except Exception as e:
            logging.warning(f"[Tier2] Eval failed: {e}")
            return 1e3, 1e3
        finally:
            K.clear_session()
            gc.collect()

    return obj