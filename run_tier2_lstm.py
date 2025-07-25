import os
import json
import time
import argparse
import logging
import gc
from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------- CONFIG & LOGGING ----------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/tier2_lstm.log"), logging.StreamHandler()]
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logging.info(f"GPUs: {gpus}")
else:
    logging.info("Using CPU")

np.random.seed(42)
tf.random.set_seed(42)

TRADING_DAYS = 252
BATCH_SIZE = 32
PATIENCE = 3

# -------------------- METRICS ---------------------------
def sharpe_ratio(returns):
    if np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(TRADING_DAYS)

def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / (peak + np.finfo(float).eps)
    return np.max(dd)

# -------------------- WINDOW ----------------------------
def create_windows(X, y, lookback_window):
    """
    Build windows: shape (n_samples, lookback_window, n_features)
    X: np.ndarray (n_timesteps, n_features)
    y: np.ndarray (n_timesteps,)
    """
    N, F = X.shape
    W = lookback_window
    if N <= W:
        raise ValueError(f"Not enough rows ({N}) for lookback {W}")

    # sliding window over time axis
    all_windows = sliding_window_view(X, window_shape=W, axis=0)  # shape: (N-W+1, W, F)
    windows = all_windows[:-1]   # remove last (since we predict y[W:] with windows[:-1])
    targets = y[W:]              # target aligned with last timestep of each window

    # fix shape mismatch if needed (some numpy versions swap axes)
    if windows.shape[1] != W or windows.shape[2] != F:
        windows = windows.transpose(0, 2, 1)

    # sanity checks
    assert windows.shape == (N - W, W, F), f"Expected shape {(N - W, W, F)}, got {windows.shape}"
    assert len(targets) == N - W

    return windows, targets

# -------------------- MODEL -----------------------------
def build_lstm(window, units, lr, num_features, layers=2, dropout_rate=0.2):
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

# -------------------- OBJECTIVE -------------------------
def create_objective(train_df, val_df, features, target_col='target', cost=0.0005):
    X, y = train_df[features].values, train_df[target_col].values
    Xv, yv = val_df[features].values, val_df[target_col].values

    def obj(params):
        w, units, lr, epochs, thresh = map(float, params)
        w, units, epochs = int(w), int(units), int(epochs)
        logging.info(f"Evaluating params - lookback={w}, units={units}, lr={lr:.5f}, epochs={epochs}, thresh={thresh:.3f}")
        try:
            win, targ = create_windows(X, y, w)
        except Exception as e:
            logging.error(f"Window creation failed: {e}")
            return 1e3, 1e3

        split = int(0.8 * len(win))
        X_tr, X_val = win[:split], win[split:]
        y_tr, y_val = targ[:split], targ[split:]

        ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        K.clear_session()
        model = build_lstm(w, units, lr, num_features=X.shape[1])
        es = EarlyStopping('val_loss', PATIENCE, restore_best_weights=True, verbose=0)

        try:
            history = model.fit(ds_tr, validation_data=ds_val, epochs=epochs, callbacks=[es], verbose=0)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            K.clear_session()
            return 1e3, 1e3

        # rolling prediction
        buf = list(X[-int(w):])
        preds = []
        for t in Xv:
            inp = np.array(buf[-int(w):])[None, :]
            p_ = model.predict(inp, verbose=0)[0, 0]
            preds.append(p_)
            buf.append(t)

        preds = np.array(preds)
        mn, mx = preds.min(), preds.max()
        if mx - mn < 1e-8:
            return 1e3, 1e3
        
        threshold = mn + (mx - mn) * thresh
        signals = (preds > threshold).astype(int)
        if signals.sum() == 0:
            K.clear_session()
            return 1e3, 1e3

        returns = yv * signals - cost * signals
        sr = sharpe_ratio(returns)
        mdd = max_drawdown(returns)
        K.clear_session()
        return -sr, mdd

    return obj

# -------------------- SEED SAMPLING ----------------------
class FromTier1SeedSampling(Sampling):
    def __init__(self, seeds, n_var):
        super().__init__()
        self.seeds = np.array(seeds)
        self.n_var = n_var

    def _do(self, problem, n_samples, **kwargs):
        n_seeds = min(len(self.seeds), n_samples)
        remaining = n_samples - n_seeds
        pop = self.seeds[:n_seeds].copy()
        if remaining > 0:
            rand = np.random.uniform(problem.xl, problem.xu, size=(remaining, self.n_var))
            pop = np.vstack([pop, rand])
        return pop

# -------------------- MOGA PROBLEM ------------------------
class MOGAProblem(ElementwiseProblem):
    def __init__(self, obj, seed):
        super().__init__(n_var=5, n_obj=2,
                         xl=np.array([10, 32, 1e-4, 10, 0.0]),
                         xu=np.array([40, 64, 1e-2, 50, 1.0]))
        self.obj = obj

    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

# -------------------- MAIN ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop-size', type=int, default=30)
    parser.add_argument('--n-gen', type=int, default=15)
    parser.add_argument('--data-dir', type=str, default='data/processed_folds')
    parser.add_argument('--tier1-file', type=str, default='data/tuning_results/tier1_lstm.json')
    parser.add_argument('--out-file', type=str, default='data/tuning_results/tier2_lstm.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    tier1 = json.load(open(args.tier1_file))
    summary = {r['fold_id']: r for r in tier1}
    folds_info = json.load(open(f"{args.data_dir}/folds_summary.json"))

    all_out = []
    for fid in tqdm(summary, desc="Tier2 LSTM"):
        start = time.time()
        meta = next(f for f in folds_info if f['global_fold_id'] == fid)

        tr = pd.read_csv(f"{args.data_dir}/{meta['train_path_lstm_gru']}")
        vl = pd.read_csv(f"{args.data_dir}/{meta['val_path_lstm_gru']}")
        feats = [c for c in tr.columns if c not in ['Date', 'Ticker', 'Log_Returns', 'target']]

        obj = create_objective(tr, vl, feats, target_col='target')
        seed = summary[fid]['best_params']
        seed_vector = [seed['window'], seed['units'], seed['lr'], seed['epochs'], seed['threshold']] 

        problem = MOGAProblem(objective)
        sampling = FromTier1SeedSampling([seed_vector], n_var=5)
        algorithm = NSGA2(pop_size=args.pop_size, sampling=sampling)

        res = minimize(problem, algorithm, ('n_gen', args.n_gen), seed=42, verbose=False)

        front = [{'params': x.tolist(), 'obj': F.tolist()} for x, F in zip(res.X, res.F)]
        all_out.append({'fold_id': fid,
                        'tier1_seed': seed_vector,
                        'pareto_front': front})
        
        elapsed = timedelta(seconds=int(time.time() - start))
        logging.info(f"Fold {fid} done in {elapsed}; Pareto size={len(front)}")
        gc.collect()

    # write outputs once
    with open(args.out_file, 'w') as f:
        json.dump(all_out, f, indent=2)
    df_out = pd.DataFrame([
        {'fold_id': o['fold_id'],
         'sharpe': -min(f['obj'][0] for f in o['pareto_front']),
         'mdd': min(f['obj'][1] for f in o['pareto_front'])}
        for o in all_out
    ])
    df_out.to_csv(args.out_file.replace('.json', '.csv'), index=False)

    logging.info("=== Tier 2 complete ===")