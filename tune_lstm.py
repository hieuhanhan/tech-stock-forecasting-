#!/usr/bin/env python3
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
from sklearn.metrics import mean_squared_error
from math import sqrt

from skopt import gp_minimize
from skopt.space import Integer, Real
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------- CONFIG & LOGGING ----------------------
LOG_DIR = "logs"
DEFAULT_T1_EPOCHS = 10
DEFAULT_T2_EPOCHS = 15
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 32
PATIENCE = 3
TRADING_DAYS = 252
BASE_COST = 0.0005       
SLIPPAGE = 0.0001        

# -------------------- HELPERS ---------------------------

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

# model builder
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

# -------------------- TIER 1 PROBLEM -------------------------
class RMSEProblem(ElementwiseProblem):
    def __init__(self, X_tr, y_tr, X_val, y_val, num_features):
        xl = np.array([10, 1, 32, 1e-4, 16])
        xu = np.array([40, 3, 64, 1e-2, 64])
        super().__init__(n_var=5, n_obj=1, xl=xl, xu=xu)
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val
        self.num_features = num_features

    def _evaluate(self, x, out, *_, **__):
        w, layers, units, lr, batch = x
        w, layers, units, batch = map(int, [w, layers, units, batch])
        dropout = 0.2

        try:
            Xtr, ytr = create_windows(self.X_tr, self.y_tr, w)
            Xvl, yvl = create_windows(
                np.concatenate([self.X_tr[-w:], self.X_val], axis=0),
                np.concatenate([self.y_tr[-w:], self.y_val], axis=0),
                w
            )

            K.clear_session()       # clear before building model to ensure not accumulating leftover graphs or layers from previous evaluations
            model = build_lstm(w, units, lr, self.num_features, layers, dropout)
            es = EarlyStopping('loss', PATIENCE, restore_best_weights=True, verbose=0)

            model.fit(Xtr, ytr,
                      epochs=DEFAULT_T1_EPOCHS,
                      batch_size=batch,
                      callbacks=[es],
                      verbose=0)
            preds = model.predict(Xvl, verbose=0).ravel()
            rmse = sqrt(mean_squared_error(yvl, preds))

        except Exception as e:
            logging.warning(f"Tier1 eval error: {e}")
            rmse = np.inf
        finally:
            K.clear_session()        # clear once at the end to clean up after training and prediction
            gc.collect()
        out['F'] = [rmse]

# -------------------- TIER 2 PROBLEM -------------------------
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

class MOGAProblem(ElementwiseProblem):
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
    - feats: list of feature column names
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

                # 1) retrain on history
                win, tar = create_windows(hist_X, hist_y, w)
                ds_tr = tf.data.Dataset.from_tensor_slices((win, tar)) \
                                       .batch(BATCH_SIZE) \
                                       .prefetch(tf.data.AUTOTUNE)

                model = build_lstm(w, units, lr, hist_X.shape[1], layers=champ['layers'])
                es    = EarlyStopping('loss', PATIENCE, restore_best_weights=True, verbose=0)
                model.fit(ds_tr, epochs=epochs, callbacks=[es], verbose=0)

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

# -------------------- MAIN ------------------------------
def main():
    parser = argparse.ArgumentParser("LSTM tuning pipeline")
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True)
    parser.add_argument('--data-dir', default='data/processed_folds')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--tier1-json', dest='tier1_json',
                    default='data/tuning_results/jsons/tier1_lstm.json')
    parser.add_argument('--tier1-csv',  dest='tier1_csv',
                    default='data/tuning_results/csv/tier1_lstm.csv')
    parser.add_argument('--tier2-json', dest='tier2_json',
                    default='data/tuning_results/jsons/tier2_lstm.json')
    parser.add_argument('--tier2-csv',  dest='tier2_csv',
                    default='data/tuning_results/csv/tier2_lstm.csv')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(f"{LOG_DIR}/phase{args.phase}.log"), logging.StreamHandler()]
    )
    np.random.seed(42); tf.random.set_seed(42)

    # load fold metadata
    folds_summary = os.path.join(args.data_dir, 'folds_summary.json')
    summary = {f['global_fold_id']: f for f in json.load(open(folds_summary))}
    rep_path = os.path.join(args.data_dir, 'shared_meta/representative_fold_ids.json')
    reps = json.load(open(rep_path))
    vol_df = pd.read_csv(os.path.join(args.data_dir, 'shared_meta/fold_volatility_categorized.csv'))
    vol_map = dict(zip(vol_df['global_fold_id'], vol_df['volatility_category']))

    if args.phase == 1:
        results, done = [], set()
        tier1_json = args.tier1_json
        tier1_csv  = args.tier1_csv
        os.makedirs(os.path.dirname(tier1_json), exist_ok=True)
        os.makedirs(os.path.dirname(tier1_csv),  exist_ok=True)

        if os.path.exists(tier1_json):
            prev = json.load(open(tier1_json))
            done = {r['fold_id'] for r in prev}
            results = prev

        pending = [fid for fid in reps if fid not in done]

        if args.max_folds: 
            pending = pending[:args.max_folds]

        for fid in tqdm(pending, desc="Tier1 GA-BO"):
            info = summary[fid]
            tr = pd.read_csv(os.path.join(args.data_dir, info['train_path_lstm_gru']))
            vl = pd.read_csv(os.path.join(args.data_dir, info['val_path_lstm_gru']))
            feats = [c for c in tr.columns if c not in ['Date','Ticker','Log_Returns','target']]

            X_tr, y_tr = tr[feats].values, tr['target'].values
            X_vl, y_vl = vl[feats].values, vl['target'].values
            num_f = X_tr.shape[1]

            ga_pop, ga_gen, bo_calls, top_n = get_params_for_volatility(vol_map.get(fid,'medium'))

            ga_res = pymoo_minimize(
                RMSEProblem(X_tr,y_tr,X_vl,y_vl,num_f),
                GA(pop_size=ga_pop), ('n_gen',ga_gen), verbose=False)

            top = sorted(ga_res.pop, key=lambda ind: ind.F[0])[:top_n]
            x0 = [ind.X.tolist() for ind in top]; y0 = [float(ind.F[0]) for ind in top]
            def bo_fun(x): out={}; RMSEProblem(X_tr,y_tr,X_vl,y_vl,num_f)._evaluate(x,out); return out['F'][0]

            bo = gp_minimize(bo_fun,
                             dimensions=[Integer(10,40), Integer(1,3), Integer(32,64),
                             Real(1e-4, 1e-2, 'log-uniform'),
                             Integer(16,64)],
                             n_calls=bo_calls, x0=x0, y0=y0, random_state=42)
            
            best = bo.x
            params = ['window', 'layers', 'units', 'lr', 'batch']
            top_ga = []
            for ind in top:
                params_dict = dict(zip(params, ind.X))
                ga_dict = {
                    'window': int(params_dict['window']),
                    'layers': int(params_dict['layers']),
                    'units': int(params_dict['units']),
                    'lr': round(params_dict['lr'], 6),
                    'batch': int(params_dict['batch'])
                }
                top_ga.append(ga_dict)

            results.append({
                'fold_id': fid, 
                'champion': {
                    'window': int(best[0]), 
                    'layers': int(best[1]),
                    'units': int(best[2]), 
                    'lr': float(best[3]),
                    'batch': int(best[4])
                }, 
                'rmse': float(bo.fun),
                'top_ga': top_ga 
            })

            with open(tier1_json,'w') as f: 
                json.dump(results,f,indent=2)

        pd.DataFrame(results).to_csv(tier1_csv, index=False)
        logging.info("=== Tier 1 complete ===")


    else:
        tier1 = {r['fold_id']: r['champion'] for r in json.load(open(args.tier1_json))}
        all_out = []

        for fid in tqdm(tier1, desc="Tier2 MOGA"):
            info = summary[fid]
            tr = pd.read_csv(os.path.join(args.data_dir, info['train_path_lstm_gru']))
            vl = pd.read_csv(os.path.join(args.data_dir, info['val_path_lstm_gru']))
            feats = [c for c in tr.columns if c not in ['Date','Ticker','Log_Returns','target']]
            champ = tier1[fid]
            obj = create_periodic_lstm_objective(tr, vl, feats, champ, retrain_interval, 
                                                cost=BASE_COST + SLIPPAGE)

            # ensemble seeds with small perturbations
            multipliers = [0.9, 0.95, 1.0, 1.05, 1.1]
            seeds = []

            for m in multipliers:
                seeds.append([
                    max(10, min(40, int(champ['window'] * m))),
                    max(32, min(64, int(champ['units'] * m))),
                    float(champ['lr'] * m),
                    DEFAULT_T2_EPOCHS,
                    DEFAULT_THRESHOLD
                ])

            sampling = FromTier1SeedSampling(seeds, n_var=5)

            res = pymoo_minimize(
                MOGAProblem(obj),
                NSGA2(pop_size=20, sampling=sampling),
                ('n_gen',10), seed=42, verbose=False
            )

            front = [{'params': x.tolist(), 'obj': F.tolist()} for x,F in zip(res.X, res.F)]
            all_out.append({'fold_id': fid, 'tier1_seeds': seeds, 'pareto': front})

        tier2_json = args.tier2_json
        tier2_csv  = args.tier2_csv
        os.makedirs(os.path.dirname(tier2_json), exist_ok=True)
        os.makedirs(os.path.dirname(tier2_csv),  exist_ok=True)

        with open(tier2_json,'w') as f:
            json.dump(all_out, f, indent=2)
        
        pd.DataFrame([
        {'fold_id':o['fold_id'], 'sharpe':-min(p[0] for p in o['pareto']),
         'mdd':min(p[1] for p in o['pareto'])}
        for o in all_out]).to_csv(tier2_csv, index=False)

        logging.info("=== Tier 2 complete ===")

if __name__ == '__main__':
    main()