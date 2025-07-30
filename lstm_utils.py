import numpy as np
import pandas as pd
import gc
import logging
import tensorflow as tf
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

# Configs
LOG_DIR = "logs"
DEFAULT_T1_EPOCHS = 10
DEFAULT_T2_EPOCHS = 15
DEFAULT_T2_NGEN_LSTM = 8
BATCH_SIZE = 32
PATIENCE = 3
TRADING_DAYS = 252
BASE_COST = 0.0005       
SLIPPAGE = 0.0001  
GA_POP_SIZE = 20
GA_N_GEN = 15
BO_N_CALLS = 30
TOP_N_SEEDS = 5

LSTM_SEARCH_SPACE = [
    Integer(10, 40, name='window'),
    Integer(1, 3, name='layers'),
    Integer(32, 64, name='units'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
    Integer(16, 64, name='batch_size')
]

# Evaluation Metrics
def sharpe_ratio(returns):
    std = np.std(returns)
    return 0.0 if std == 0 else (np.mean(returns) / std) * np.sqrt(TRADING_DAYS)

def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    return np.max((peak - cum) / (peak + np.finfo(float).eps))

# data windows
def create_windows(X, y, lookback):
    N, F = X.shape
    if N <= lookback:
        raise ValueError(f"Not enough rows {N} for lookback {lookback}")
    all_w = sliding_window_view(X, window_shape=lookback, axis=0)
    wins = all_w[:-1]
    tars = y[lookback:]
    if wins.shape[1] != lookback or wins.shape[2] != F:
        wins = wins.transpose(0, 2, 1)
    assert wins.shape == (N - lookback, lookback, F)
    assert len(tars) == N - lookback
    return wins, tars

# Model
def build_lstm(window, units, lr, num_features, layers, dropout_rate=0.2):
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

# -------------------- TIER 1 -------------------------
class Tier1LSTMProblem(ElementwiseProblem):
    def __init__(self, X_tr, y_tr, X_val, y_val, num_features):
        xl = np.array([dim.low for dim in LSTM_SEARCH_SPACE])
        xu = np.array([dim.high for dim in LSTM_SEARCH_SPACE])
        super().__init__(n_var=len(LSTM_SEARCH_SPACE), n_obj=1, xl=xl, xu=xu)
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val
        self.num_features = num_features

    def _evaluate(self, x, out, *_, **__):
        w, layers, units, lr, batch_size = x
        w, layers, units, batch_size = map(int, [w, layers, units, batch_size])
        dropout = 0.2

        try:
            Xtr, ytr = create_windows(self.X_tr, self.y_tr, w)
            Xvl, yvl = create_windows(
                np.concatenate([self.X_tr[-w:], self.X_val], axis=0),
                np.concatenate([self.y_tr[-w:], self.y_val], axis=0),
                w
            )

            K.clear_session()      
            model = build_lstm(w, units, lr, self.num_features, layers, dropout)
            es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)

            model.fit(Xtr, ytr,
                      validation_data=(Xvl, yvl),
                      epochs=DEFAULT_T1_EPOCHS,
                      batch_size=batch_size,
                      callbacks=[es],
                      verbose=0)
            preds = model.predict(Xvl, verbose=0).ravel()
            rmse = sqrt(mean_squared_error(yvl, preds))

        except Exception as e:
            logging.warning(f"Tier1 eval error: {e}")
            rmse = np.inf
        finally:
            K.clear_session()        
            gc.collect()
        out['F'] = [rmse]

# -------------------- TIER 2 -------------------------
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
        xl = np.array([10, 32, 1e-4, 10, 0.0])
        xu = np.array([40, 64, 1e-2, 50, 1.0])
        super().__init__(n_var=5, n_obj=2, xl=xl, xu=xu)
        self.obj = obj
    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

# objective for tier2 with walk-forward validation

def create_periodic_lstm_objective(tr_df, val_df, feats, champ,
                                   retrain_interval, base_cost=BASE_COST):
    """
    Walk-forward LSTM backtest with periodic retraining every `retrain_interval` steps.
    - tr_df: in-sample dataframe
    - val_df: out-of-sample dataframe
    - feats: list of feature column Hnames
    - champ: dict of Tier 1 champion hyperparams (includes 'layers')
    - retrain_interval: steps between refits
    - base_cost: per-trade cost (we’ll add SLIPPAGE internally)
    Returns f(x)->(-Sharpe, MaxDrawdown) for x=(window,units,lr,epochs,thresh_rel).
    """
    X_full, y_full = tr_df[feats].values, tr_df['target'].values
    X_val,   y_val   = val_df[feats].values, val_df['target'].values
    cost = base_cost + SLIPPAGE
    n_val = len(y_val)

    def obj(x):
        w, units, lr, epochs, rel_thresh = x
        w, units, epochs = map(int, (w, units, epochs))
        lr, rel_thresh = float(lr), float(rel_thresh)
        if not 0 < rel_thresh < 1:
            return 1e3, 1e3

        hist_X, hist_y = X_full.copy(), y_full.copy()
        all_returns = []

        try:
            for start in range(0, n_val, retrain_interval):
                end = min(start + retrain_interval, n_val)

                K.clear_session()
                batch_size = champ['batch_size']
                # 1) retrain on history
                win, tar = create_windows(hist_X, hist_y, w)
                ds_tr = tf.data.Dataset.from_tensor_slices((win, tar)) \
                       .batch(batch_size) \
                       .prefetch(tf.data.AUTOTUNE)

                val_ds = tf.data.Dataset.from_tensor_slices((X_val[start:end], y_val[start:end])) \
                                        .batch(batch_size) \
                                        .prefetch(tf.data.AUTOTUNE)

                model = build_lstm(w, units, lr, hist_X.shape[1], layers=champ['layers'])
                es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
                model.fit(ds_tr,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[es],
                    verbose=0)

                # 2) forecast this block one-step at a time
                preds = []
                buf = list(hist_X[-w:])
                for i in range(start, end):
                    inp = np.array(buf[-w:])[None, ...]  # shape (1,w,features)
                    p = model.predict(inp, verbose=0)[0,0]
                    preds.append(p)
                    buf.append(X_val[i])  # append true features for next window

                preds = np.array(preds)

                # threshold
                mn, mx = preds.min(), preds.max()
                if mx - mn < 1e-8:
                    return 1e3, 1e3
                thresh_val = mn + (mx - mn) * rel_thresh
                sig = (preds > thresh_val).astype(int)
                if sig.sum() == 0:
                    return 1e3, 1e3

                # block returns
                ret_block = y_val[start:end] * sig - cost * sig
                all_returns.append(ret_block)

                # 3) expand history with actual outcomes
                hist_X = np.vstack([hist_X, X_val[start:end]])
                hist_y = np.concatenate([hist_y, y_val[start:end]])

            # flatten
            net_returns = np.concatenate(all_returns)
            sr  = sharpe_ratio(net_returns)
            mdd = max_drawdown(net_returns)
            return -sr, mdd

        except Exception as e:
                logging.warning(f"Periodic LSTM eval failed: {e}")
                return 1e3, 1e3
        finally:
            K.clear_session()
            gc.collect()

    return obj