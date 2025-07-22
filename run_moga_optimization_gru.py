import os
import json
import logging
import time
import datetime
import argparse
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view 

from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# ---------------------- CONFIG & LOGGING ----------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logging.info(f"GPUs: {gpus}")
else:
    logging.info("Using CPU")

# reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ------------------------ CONSTANTS ---------------------------
TRADING_DAYS = 252
BATCH_SIZE = 32
PATIENCE = 3

# ----------------------- METRICS ------------------------------
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

# --------------------- MODEL CREATION -------------------------
def make_gru(lookback_window, n_features, n_units, learning_rate):
    model = Sequential([
        Input((lookback_window, n_features)),
        GRU(
            n_units,
            return_sequences=True,
            kernel_initializer=Orthogonal(),
            recurrent_initializer=Orthogonal(),
        ),
        LayerNormalization(),
        Dropout(0.2),

        GRU(
            n_units // 2,
            kernel_initializer=Orthogonal(),
            recurrent_initializer=Orthogonal(),
        ),
        LayerNormalization(),
        Dropout(0.2),

        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate), loss='huber')
    return model

# --------------------- WINDOW GENERATION ----------------------
from numpy.lib.stride_tricks import sliding_window_view

def create_windows(X, y, lookback_window):
    """
    Build (n_samples, lookback_window, n_features) windows from
    X.shape == (n_timesteps, n_features), with targets y[lookback_window:].
    """
    N, F = X.shape
    W = lookback_window

    if N <= W:
        raise ValueError(f"Not enough rows ({N}) for lookback {W}")

    # 1) Sliding over time axis → shape (N-W+1, W, F)
    all_windows = sliding_window_view(X, window_shape=W, axis=0)

    # 2) Chop off the final window so that windows[i] predicts y[i+W]
    windows = all_windows[:-1]   # now (N-W, W, F)
    targets = y[W:]              # length N-W

    # 3) Sanity‐check & auto‐transpose if needed
    if windows.shape[1] != W or windows.shape[2] != F:
        # axes got reversed somewhere: swap axis 1↔2
        windows = windows.transpose(0, 2, 1)

    assert windows.shape == (N - W, W, F), (
        f"windows {windows.shape} ≠ expected ({N-W},{W},{F})"
    )
    assert len(targets) == N - W

    return windows, targets

# --------------------- OBJECTIVE SETUP ------------------------
def create_objective(train_df, val_df, features, target='Log_Returns', cost=0.0005):
    train_X = train_df[features].values
    train_y = train_df[target].values
    val_X   = val_df[features].values
    val_y   = val_df[target].values
    n_features = train_X.shape[1]

    def objective_function(params):
        lookback_window = int(params[0])
        n_units = int(params[1])
        learning_rate = float(params[2])
        epochs = int(params[3])
        relative_thresh = float(params[4])

        logging.info(f"Params → lookback={lookback_window}, units={n_units}, lr={learning_rate:.5f}, epochs={epochs}, thresh={relative_thresh:.4f}")

        try:
            windows, targets = create_windows(train_X, train_y, lookback_window)
        except Exception as e:
            logging.error(f"Window creation failed: {e}")
            return 1e3, 1e3

        if windows.shape[0] < 2:
            logging.warning("Too few windows. Skipping.")
            return 1e3, 1e3

        split = int(0.9 * len(windows))
        X_tr, X_valsplit = windows[:split], windows[split:]
        y_tr, y_valsplit = targets[:split], targets[split:]

        ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_v  = tf.data.Dataset.from_tensor_slices((X_valsplit, y_valsplit)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        K.clear_session()
        model = make_gru(lookback_window, n_features, n_units, learning_rate)
        es = EarlyStopping('val_loss', PATIENCE, restore_best_weights=True, verbose=0)

        def clr(epoch):
            base, max_lr = learning_rate/10, learning_rate*10
            step = max(1, epochs//2)
            cycle = np.floor(1 + epoch/(2*step))
            x = abs(epoch/step - 2*cycle + 1)
            return float(base + (max_lr-base) * max(0, 1-x))
        lr_cb = LearningRateScheduler(clr, verbose=0)

        try:
            model.fit(ds_tr, validation_data=ds_v, epochs=epochs, callbacks=[es, lr_cb], verbose=0)
            val_loss = model.evaluate(ds_v, verbose=0)
            logging.info(f"Val loss: {val_loss:.6f}")
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return 1e3, 1e3

        # walk-forward
        buf, preds = list(train_X[-lookback_window:]), []
        for true_next in val_X:
            inp = np.array(buf[-lookback_window:]).reshape(1, lookback_window, n_features)
            p   = model.predict(inp, verbose=0)[0,0]
            preds.append(p)
            buf.append(true_next)
        preds = np.array(preds)

        min_p, max_p = preds.min(), preds.max()
        if max_p - min_p < 1e-8:
            logging.warning("Constant preds. Skipping.")
            return 1e3, 1e3
        thresh = min_p + (max_p - min_p) * relative_thresh
        signals = (preds > thresh).astype(int)
        if signals.sum() == 0:
            logging.warning("No signals. Skipping.")
            return 1e3, 1e3

        returns = val_y * signals - signals * cost
        sharpe = sharpe_ratio(returns)
        mdd    = max_drawdown(returns)
        logging.info(f"Sharpe={sharpe:.4f}, MDD={mdd:.4f}")

        return -sharpe, mdd

    return objective_function

# ---------------------- PYMOO PROBLEM ------------------------
class GRUProblem(ElementwiseProblem):
    def __init__(self, obj_func):
        super().__init__(n_var=5, n_obj=2,
                         xl=np.array([10, 20, 1e-5, 10, 0.0]),
                         xu=np.array([40, 64, 1e-3, 50, 1.0]))
        self.obj = obj_func

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.obj(x)

# ------------------------ MAIN LOOP --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-folds', type=int, default=None,
                        help='Max number of new folds to run before exiting')
    args = parser.parse_args()

    logging.info("Starting MOGA GRU tuning")
    out_dir  = 'data/tuning_results'
    fold_dir = 'data/processed_folds'
    os.makedirs(out_dir, exist_ok=True)
    res_file = os.path.join(out_dir, 'final_moga_gru_results.json')

    # load checkpoint
    try:
        with open(res_file, 'r') as f:
            res_list = json.load(f)
        done = { r['fold_id'] for r in res_list if r.get('status') in ['success','error'] }
    except Exception:
        res_list, done = [], set()

    summary = json.load(open(os.path.join(fold_dir,'folds_summary.json')))
    reps    = json.load(open(os.path.join(fold_dir,'shared_meta','representative_fold_ids.json')))
    smap    = { f['global_fold_id']: f for f in summary }

    new_runs = 0
    for fid in tqdm(reps, total=len(reps), desc='Processing folds'):
        if fid in done:
            logging.info(f"Skipping fold {fid} (already done)")
            continue
        if args.max_new_folds and new_runs >= args.max_new_folds:
            logging.info(f"Reached limit of {args.max_new_folds} new folds; exiting.")
            break

        logging.info(f"Running fold {fid}...")
        start = time.time()
        try:
            train = pd.read_csv(os.path.join(fold_dir, smap[fid]['train_path_lstm_gru']))
            val   = pd.read_csv(os.path.join(fold_dir, smap[fid]['val_path_lstm_gru']))
            feats = [c for c in train.columns if c not in ['Date','Ticker','Log_Returns','target']]

            obj  = create_objective(train, val, feats)
            prob = GRUProblem(obj)
            alg  = NSGA2(pop_size=20)
            res  = minimize(prob, alg, ('n_gen', 15), seed=42, verbose=False)

            solutions = []
            for x,f in zip(res.X, res.F):
                solutions.append({
                    'lookback_window': int(x[0]),
                    'n_units':         int(x[1]),
                    'learning_rate':   float(x[2]),
                    'epochs':          int(x[3]),
                    'action_threshold':float(x[4]),
                    'sharpe':          -f[0],
                    'max_drawdown':    f[1]
                })
            res_list.append({'fold_id': fid, 'solutions': solutions, 'status': 'success'})
            with open(res_file, 'w') as f:
                json.dump(res_list, f, indent=2)

            logging.info(f"Completed fold {fid} with {len(solutions)} solutions")
            new_runs += 1

        except Exception as e:
            logging.error(f"Error in fold {fid}: {e}")
            res_list.append({'fold_id': fid, 'status': 'error'})
            with open(res_file, 'w') as f:
                json.dump(res_list, f, indent=2)

        # cleanup
        K.clear_session()
        gc.collect()

        duration = time.time() - start
        logging.info(f"Fold {fid} took {datetime.timedelta(seconds=int(duration))}")

    logging.info("All done")

if __name__ == '__main__':
    main()