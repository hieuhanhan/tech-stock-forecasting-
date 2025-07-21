import os
import json
import logging
import time
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view 


from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, LayerNormalization
)
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
    """
    - Orthogonal init on GRU kernels
    - Huber loss
    - AdamW with small weight decay
    """
    model = Sequential([
        Input((lookback_window, n_features)),
        GRU(
            n_units,
            return_sequences=True,
            kernel_initializer=Orthogonal(),
            recurrent_initializer=Orthogonal()
        ),
        LayerNormalization(),
        Dropout(0.2),

        GRU(
            n_units // 2,
            kernel_initializer=Orthogonal(),
            recurrent_initializer=Orthogonal()
        ),
        LayerNormalization(),
        Dropout(0.2),

        Dense(1)
    ])

    model.compile(Adam(learning_rate), loss='huber')
    return model

# --------------------- OBJECTIVE SETUP ------------------------
def create_objective(train_df, val_df, features, target='Log_Returns', cost=0.0005):
    """
    This function is called only once per fold. Its job is to set up the data and return
    the actual function that the optimizer will use.
    """
    train_X = train_df[features].values
    train_y = train_df[target].values
    val_X = val_df[features].values
    val_y = val_df[target].values
    n_features = train_X.shape[1]

    # --- Inner helper for input generation ---
    def create_windows(train_X, train_y, lookback_window, use_sliding_view=True, debug=False):
        if len(train_X) <= lookback_window:
            raise ValueError("Training data is too short for the specified lookback window.")

        if use_sliding_view:
            try:
                windows = sliding_window_view(train_X, lookback_window, axis=0)[:-1]
            except Exception as e:
                if debug:
                    print(f"sliding_window_view failed: {e}")
                raise
        else:
            windows = np.stack([
                train_X[i:i + lookback_window] for i in range(len(train_X) - lookback_window)
            ])

        targets = train_y[lookback_window:]
        assert windows.shape[0] == targets.shape[0], (
            f"Shape mismatch: windows={windows.shape}, targets={targets.shape}"
        )
        return windows, targets
    
    def objective_function(params):
        """
        This is the core function return by create_objective. The optimizer call this
        function hundreds or thousands of times. It has access to the variables defined 
        in its parent function (train_X, train_y, n_features, etc.)
        """
        lookback_window = int(params[0])    
        n_units = int(params[1])
        learning_rate = float(params[2])
        epochs = int(params[3])
        relative_threshold = float(params[4])

        logging.info(f"Trying params: lookback={lookback_window}, units={n_units}, "
                 f"learning_rate={learning_rate:.5f}, epochs={epochs}, threshold={relative_threshold:.4f}")

        try:
            windows, targets = create_windows(train_X, train_y, lookback_window,
                                              use_sliding_view=False,  
                                              debug=True)
        except Exception as e:
            logging.error(f"Failed to create windows for lookback={lookback_window}: {e}")
            return 1e3, 1e3

        if windows.shape[0] < 2:
            logging.warning("Too few windows after slicing. Skipping.")
            return 1e3, 1e3
        
        split = int(0.9 * len(windows))
        X_train, X_valsplit = windows[:split], windows[split:]
        y_train, y_valsplit = targets[:split], targets[split:]

        logging.info(f"Train samples: {X_train.shape[0]}, Val samples: {X_valsplit.shape[0]}")

        ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        ds_train = ds_train.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
        ds_val = tf.data.Dataset.from_tensor_slices((X_valsplit, y_valsplit))
        ds_val = ds_val.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

        K.clear_session()
        model = make_gru(lookback_window, n_features, n_units, learning_rate)
        es = EarlyStopping('val_loss', PATIENCE, restore_best_weights=True, verbose=0)

        def clr(epoch):
            base, max_lr = learning_rate/10, learning_rate*10
            step = max(1, epochs//2)
            cycle = np.floor(1 + epoch/(2*step))
            x = abs(epoch/step - 2*cycle + 1)
            return float(base + (max_lr-base) * max(0, 1-x))
        lr_cb = LearningRateScheduler(lambda e: clr(e), verbose=0)

        try:
            logging.info(f"Fitting model on {X_train.shape[0]} samples")
            model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[es, lr_cb], verbose=0)
            val_loss = model.evaluate(ds_val, verbose=0)
            logging.info(f"Final val_loss: {val_loss:.6f}")
        except Exception as e:
            logging.error(f"Model fitting failed for params {params}: {e}")
            return 1e3, 1e3
        
        # walk-forward prediction and evaluation
        buf = list(train_X[-lookback_window:])
        preds = []

        for i in range(len(val_y)):
            inp = np.array(buf[-lookback_window:]).reshape(1, lookback_window, n_features)
            p = model.predict(inp, verbose=0)[0,0]
            preds.append(p)
            buf.append(val_X[i])

        preds = np.array(preds)
        # ==== NEW RELATIVE THRESHOLDING LOGIC ====
        min_pred, max_pred = preds.min(), preds.max()
        if max_pred - min_pred < 1e-8:
            logging.warning("Preds nearly constant. Skipping.")
            return 1e3, 1e3
        
        dynamic_threshold = min_pred + (max_pred - min_pred) * relative_threshold
        signals = (preds > dynamic_threshold).astype(int)

        logging.info(f"Sample preds: {preds[:5]}")
        logging.info(f"Dynamic threshold: {dynamic_threshold:.6f}")
        logging.info(f"Signals distribution: {np.unique(signals, return_counts=True)}")

        if signals.sum() == 0:
            logging.warning("All signals are zero, skipping.")
            return 1e3, 1e3
        
        returns = val_y * signals - signals * cost
        logging.info(f"Returns: mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")

        sharpe = sharpe_ratio(returns)
        mdd = max_drawdown(returns)
        logging.info(f"Sharpe: {sharpe:.4f}, Max Drawdown: {mdd:.4f}")

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
    logging.info("MOGA GRU tuning start")
    out_dir = 'data/tuning_results'
    fold_dir = 'data/processed_folds'
    os.makedirs(out_dir, exist_ok=True)
    res_file = os.path.join(out_dir, 'final_moga_gru_results.json')

    try:
        res_list = json.load(open(res_file))
        done = {r['fold_id'] for r in res_list if r['status'] in ['success','error']}
    except:
        res_list, done = [], set()

    summary = json.load(open(os.path.join(fold_dir,'folds_summary.json')))
    reps = json.load(open(os.path.join(fold_dir,'shared_meta','representative_fold_ids.json')))
    smap = {f['global_fold_id']:f for f in summary}

    for fid in tqdm(reps[:3]):
        if fid in done:
            logging.info(f"Skipping fold {fid} (already done)")
            continue

        logging.info(f"Running fold {fid} ...")
        start = time.time() 

        try:
            train = pd.read_csv(os.path.join(fold_dir, smap[fid]['train_path_lstm_gru']))
            val   = pd.read_csv(os.path.join(fold_dir, smap[fid]['val_path_lstm_gru']))
            feats = [c for c in train.columns if c not in ['Date','Ticker','Log_Returns','target']]

            obj = create_objective(train, val, feats)
            prob = GRUProblem(obj)
            alg  = NSGA2(pop_size=20)

            res  = minimize(prob, alg, ('n_gen',15), seed=42, verbose=False)
            solutions = []
            for x,f in zip(res.X,res.F):
                solutions.append({
                    'lookback_window':int(x[0]),
                    'n_units':int(x[1]),
                    'learning_rate':x[2],
                    'epochs':int(x[3]),
                    'action_threshold':x[4],
                    'sharpe':-f[0],
                    'max_drawdown':f[1]
                    })
            res_list.append({'fold_id':fid,'solutions':solutions,'status':'success'})
            json.dump(res_list, open(res_file,'w'), indent=2)
            logging.info(f"Completed fold {fid} with {len(solutions)} solutions")

        except Exception as e:
            logging.error(f"Error while running fold {fid}: {e}")
            res_list.append({'fold_id': fid, 'status': 'error'})
            json.dump(res_list, open(res_file, 'w'), indent=2)

        duration = time.time() - start
        duration_td = datetime.timedelta(seconds=int(duration))
        logging.info(f"Fold {fid} took {duration_td}")
    logging.info("Done all folds")

if __name__=='__main__':
    main()