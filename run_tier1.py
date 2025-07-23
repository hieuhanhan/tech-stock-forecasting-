import os
import argparse
import json
import logging
import time
import gc
import signal
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer, Real
from pymoo.algorithms.soo.genetic_algorithm import GA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from numpy.lib.stride_tricks import sliding_window_view

import tensorflow as tf
from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------- CONFIG & LOGGING ----------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/tier1_lstm.log"), logging.StreamHandler()]
)
tf.random.set_seed(42); np.random.seed(42)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# ------------------------ SIGNAL HANDLER FOR GRACEFUL EXIT --------------------------
tier1_results = []
def _on_term(sig, frame):
    logging.info("SIGTERM caught, saving progress…")
    json.dump(tier1_results, open(ARGS.out_file,'w'), indent=2)
    sys.exit(0)
signal.signal(signal.SIGTERM, _on_term)

# ---------------------- ARGUMENTS ----------------------
parser = argparse.ArgumentParser("Tier-1 GA→BO tuning for LSTM")
parser.add_argument('--ga-pop',     type=int,   default=20)
parser.add_argument('--ga-gen',     type=int,   default=10)
parser.add_argument('--top-n',      type=int,   default=5)
parser.add_argument('--bo-calls',   type=int,   default=30)
parser.add_argument('--data-dir',   type=str,   default='data/processed_folds')
parser.add_argument('--out-file',   type=str,   default='data/tuning_results/tier1_lstm.json')
ARGS = parser.parse_args()
os.makedirs(os.path.dirname(ARGS.out_file), exist_ok=True)

# ---------------------- PARAMETER SPACE ----------------------
dl_space = [
    Integer(10,60,name='window'),
    Integer(1,3,  name='layers'),
    Integer(32,128,name='units'),
    Real(0.1,0.5,name='dropout'),
    Real(1e-4,1e-2,'log-uniform',name='lr'),
    Integer(16,128,name='batch')
]
xl = np.array([10, 1, 32, 0.1, 1e-4, 16])
xu = np.array([60, 3, 128, 0.5, 1e-2, 128])

# ------------------------ HELPERS ----------------------------
def create_windows(arr, w):
    X,y = [],[]
    for i in range(len(arr)-w):
        X.append(arr[i:i+w]); y.append(arr[i+w])
    return np.array(X)[...,None], np.array(y)

def build_lstm(window, units, lr, layers, dropout):
    m = Sequential([ Input((window,1)),
                     LSTM(units, return_sequences=(layers>1)),
                     BatchNormalization(), Dropout(dropout) ] +
                   sum([[ LSTM(units//2 if i==layers-1 else units, return_sequences=(i<layers-1)),
                          BatchNormalization(), Dropout(dropout)]
                        for i in range(1, layers)], []) +
                   [Dense(1)])
    m.compile(tf.keras.optimizers.Adam(lr), 'mse')
    return m

class RMSEProblem(ElementwiseProblem):
    def __init__(self, train, val):
        super().__init__(n_var=6, n_obj=1, xl=xl, xu=xu)
        self.t, self.v = train.values, val.values

    def _evaluate(self, x, out, *_, **__):
        try:
            w, layers, units = map(int, x[:3])
            do, lr, bs = x[3], float(x[4]), int(x[5])
            Xtr, ytr = create_windows(self.t, w)
            Xvl, yvl = create_windows(np.concatenate([self.t[-w:], self.v]), w)
            m = build_lstm(w, units, lr, layers, do)
            m.fit(Xtr, ytr, epochs=20, batch_size=bs,
                  callbacks=[EarlyStopping('loss',5,restore_best_weights=True)],
                  verbose=0)
            preds = m.predict(Xvl, verbose=0).ravel()
            rmse = float(np.sqrt(np.mean((preds-yvl)**2)))
        except Exception as e:
            logging.warning(f"GA eval error: {e}")
            rmse = 1e6
        out['F'] = [rmse]

def run_ga(train, val, pop, gen):
    return minimize(RMSEProblem(train, val), GA(pop_size=pop), ('n_gen', gen), verbose=False)

def bo_obj(train, val):
    def f(x):
        out = {}
        RMSEProblem(train, val)._evaluate(x, out)
        return out['F'][0]
    return f

# ------------------------ MAIN ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Tier-1 GA→BO tuning for LSTM")
    parser.add_argument('--ga-pop', type=int, default=20)
    parser.add_argument('--ga-gen', type=int, default=10)
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--bo-calls', type=int, default=30)
    parser.add_argument('--data-dir', type=str, default='data/processed_folds')
    parser.add_argument('--out-file', type=str, default='data/tuning_results/json/tier1_lstm.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    try:
        tier1_results = json.load(open(args.out_file))
        done = {r['fold_id'] for r in tier1_results}
        logging.info(f"Resuming {len(done)} folds")
    except:
        tier1_results, done = [], set()

    summary = {f['global_fold_id']: f for f in json.load(open(f"{args.data_dir}/folds_summary.json"))}
    reps = json.load(open(f"{args.data_dir}/shared_meta/representative_fold_ids.json"))

    for fid in tqdm(reps, desc="Tier1 LSTM"):
        if fid in done: continue
        t0 = time.time()

        df = summary[fid]
        train = pd.read_csv(f"{args.data_dir}/{df['train_path_lstm_gru']}", parse_dates=['Date'])['Log_Returns']
        val   = pd.read_csv(f"{args.data_dir}/{df['val_path_lstm_gru']}",   parse_dates=['Date'])['Log_Returns']

        logging.info(f"Fold {fid}: GA(pop={args.ga_pop}, gen={args.ga_gen})")
        res_ga = run_ga(train, val, args.ga_pop, args.ga_gen)
        top   = sorted(res_ga.pop, key=lambda i: i.F[0])[:args.top_n]
        x0, y0 = [i.X for i in top], [i.F[0] for i in top]

        logging.info(f"Fold {fid}: BO(n_calls={args.bo_calls})")
        bo = gp_minimize(bo_obj(train, val),
                         dimensions=dl_space,
                         n_calls=args.bo_calls,
                         x0=x0, y0=y0,
                         random_state=42)

        tier1_results.append({
            'fold_id': fid,
            'best_rmse': float(bo.fun),
            'best_params': bo.x,
            'top_ga': [list(ind.X) for ind in top]
        })

        # checkpoint
        json.dump(tier1_results, open(args.out_file,'w'), indent=2)
        pd.DataFrame(tier1_results).to_csv(args.out_file.replace('.json','.csv'), index=False)

        dt = timedelta(seconds=int(time.time()-t0))
        logging.info(f"→ Fold {fid} done in {dt}, RMSE={bo.fun:.4f}")

        K.clear_session()
        gc.collect()

    logging.info("=== Tier 1 complete ===")