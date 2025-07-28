#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from skopt import gp_minimize
from skopt.space import Integer
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

# ─── CONFIG & LOGGING ─────────────────────────────────────
LOG_DIR = "logs"
GA_POP_SIZE = 35
GA_N_GEN = 25
BO_N_CALLS = 50
TOP_N_SEEDS = 5
BASE_COST = 0.0005
SLIPPAGE = 0.0002
DEFAULT_T2_NGEN = 40

# ─── HELPERS ─────────────────────────────────────────
def arima_rmse(params, train, val):
    p, q = map(int, params)
    try:
        model = ARIMA(train, order=(p, 0, q)).fit()
        forecast = model.forecast(steps=len(val))
        return sqrt(mean_squared_error(val, forecast))
    except Exception as e:
        logging.warning(f"ARIMA({p},{q}) eval error: {e}")
        return 1e3

def sharpe_ratio(r):
    std = np.std(r)
    return 0.0 if std == 0 else (np.mean(r) / std) * np.sqrt(252)

def max_drawdown(r):
    if len(r) == 0:
        return 0.0
    cum = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(cum)
    return np.max((peak - cum) / (peak + np.finfo(float).eps))

def create_periodic_arima_objective(train, val, retrain_interval=20, cost=BASE_COST):
    """
    Walk‐forward ARIMA backtest with periodic retraining:
      - train: 1D np.array of in‐sample log‐returns
      - val: 1D np.array of out‐of‐sample log‐returns
      - retrain_interval: number of steps between refits
      - cost:  transaction cost per trade
    Returns f(x) -> (-Sharpe, MaxDrawdown) for x = (p, q, thresh_rel)
    """
    def objective(x):
        p, q, rel_thresh = int(x[0]), int(x[1]), float(x[2])
        if not (0.0 < rel_thresh < 1.0):
            return 1e3, 1e3

        history = train.copy()
        all_returns = []
        n = len(val)

        # step through val in blocks of retrain_interval
        for start in range(0, n, retrain_interval):
            end = min(start + retrain_interval, n)
            try:
                model = ARIMA(history, order=(p, 0, q)).fit()
            except Exception as e:
                logging.warning(f"ARIMA fit error p={p},q={q}: {e}")
                return 1e3, 1e3

            h = end - start
            forecast = model.forecast(steps=h)
            threshold_value = np.quantile(forecast, rel_thresh)
            signals = (forecast > threshold_value).astype(int)
            if signals.sum() == 0:
                return 1e3, 1e3
            
            block_returns = val[start:end] * signals - cost * signals
            all_returns.append(block_returns)
            history = np.concatenate([history, val[start:end]])

        net_returns = np.concatenate(all_returns)
        return -sharpe_ratio(net_returns), max_drawdown(net_returns)
        
    return objective

# ─── PYMOO PROBLEM DEFINITIONS ────────────────────────────────
class Tier1ARIMAProblem(ElementwiseProblem):
    def __init__(self, train, val):
        super().__init__(n_var=2, n_obj=1, xl=np.array([1, 1]), xu=np.array([7, 7]))
        self.train = train
        self.val = val

    def _evaluate(self, x, out, *_, **__):
        out['F'] = [arima_rmse(x, self.train, self.val)]

class Tier2MOGAProblem(ElementwiseProblem):
    def __init__(self, obj_func):
        super().__init__(n_var=3, n_obj=2,
                         xl=np.array([1, 1, 0.0]), xu=np.array([7, 7, 1.0]))
        self.obj = obj_func

    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

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
                                     size=(n_samples - n0, self.n_var))
            pop = np.vstack([pop, rand])
        return pop

# ─── MAIN SCRIPT ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("ARIMA tuning pipeline")
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True)
    parser.add_argument('--data-dir', default='data/scaled_folds')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--tier1-json', dest='tier1_json',
                    default='data/tuning_results/jsons/tier1_arima.json')
    parser.add_argument('--tier1-csv',  dest='tier1_csv',
                    default='data/tuning_results/csv/tier1_arima.csv')
    parser.add_argument('--tier2-json', dest='tier2_json',
                    default='data/tuning_results/jsons/tier2_arima.json')
    parser.add_argument('--tier2-csv',  dest='tier2_csv',
                    default='data/tuning_results/csv/tier2_arima.csv')
    args = parser.parse_args()

    # Setup logging
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"phase{args.phase}_arima.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    np.random.seed(42)

    # Load fold metadata
    folds_summary = os.path.join(args.data_dir, 'folds_summary_rescaled.json')
    with open(folds_summary, 'r') as f:
        folds_data = json.load(f)
    summary = {f['fold_id']: f for f in folds_data['arima']}
    arima_reps_path = os.path.join(args.data_dir, 'arima', 'arima_tuning_folds.json')
    reps = [r['fold_id'] for r in json.load(open(arima_reps_path))]

    if args.phase == 1:
        # --- Tier 1: GA -> BO to find best (p,q) by RMSE ---
        tier1_json = args.tier1_json
        tier1_csv  = args.tier1_csv
        os.makedirs(os.path.dirname(tier1_json), exist_ok=True)
        os.makedirs(os.path.dirname(tier1_csv),  exist_ok=True)

        try:
            results = json.load(open(tier1_json))
            done = {r['fold_id'] for r in results}
        except FileNotFoundError:
            results, done = [], set()

        pending = [fid for fid in reps if fid not in done]
        if args.max_folds:
            pending = pending[:args.max_folds]

        for fid in tqdm(pending, desc='Tier1 ARIMA'):
            info = summary[fid]
            train = pd.read_csv(os.path.join(args.data_dir, info['train_path']), parse_dates=['Date'])
            val = pd.read_csv(os.path.join(args.data_dir, info['val_path']), parse_dates=['Date'])
            train_ret = train['Log_Returns'].dropna().values
            val_ret = val['Log_Returns'].values

            logging.info(f"Fold {fid}: GA(pop={GA_POP_SIZE}, gen={GA_N_GEN}), BO(calls={BO_N_CALLS})")

            # GA global search
            algorithm = GA(pop_size=GA_POP_SIZE,
                           eliminate_duplicates=True,
                           save_history=True)
            
            res_ga = pymoo_minimize(
                Tier1ARIMAProblem(train_ret, val_ret),
                algorithm,
                ('n_gen', GA_N_GEN),
                verbose=False,
                return_algorithm=True
            )
            algorithm = res_ga.algorithm

            logging.info(f"Collected {len(algorithm.history)} generations in GA history")

            # Pick top seeds from GA
            all_individuals = []
            for entry in algorithm.history:
                all_individuals.extend(entry.pop)

            seen = set()
            unique_seeds = []
            for ind in all_individuals:
                key = (int(ind.X[0]), int(ind.X[1]))
                if key not in seen:
                    seen.add(key)
                    unique_seeds.append(ind)

            unique_seeds.sort(key=lambda ind: ind.F[0])
            top_seeds_for_bo = unique_seeds[:TOP_N_SEEDS]

            if len(top_seeds_for_bo) == 0:
                best = res_ga.pop.get("F").argmin()
                best_ind = res_ga.pop[best]
                x0 = [[int(best_ind.X[0]), int(best_ind.X[1])]]
                y0 = [best_ind.F[0]]
            else:
                x0 = [[int(ind.X[0]), int(ind.X[1])] for ind in top_seeds_for_bo]
                y0 = [float(ind.F[0]) for ind in top_seeds_for_bo]

            # BO refinement
            res_bo = gp_minimize(
                lambda x: arima_rmse(x, train_ret, val_ret),
                dimensions=[Integer(1, 7), Integer(1, 7)],
                n_calls=BO_N_CALLS,
                x0=x0,
                y0=y0,
                random_state=42
            )

            best_p, best_q = map(int, res_bo.x)

            top_ga = [
                {'p': int(ind.X[0]), 'q': int(ind.X[1]), 'rmse': float(ind.F[0])}
                for ind in unique_seeds
            ]
            results.append({'fold_id': fid,
                            'best_params': {'p': best_p, 'd': 0, 'q': best_q},
                            'rmse': float(res_bo.fun),
                            'top_ga': top_ga})
            
            with open(tier1_json, 'w') as f:
                json.dump(results, f, indent=2)

        pd.DataFrame(results).to_csv(tier1_csv, index=False)
        logging.info('=== Tier 1 ARIMA complete ===')

    else:
        # --- Tier 2: NSGA-II to find Pareto front for (Sharpe, MDD) ---
        tier1_json = args.tier1_json
        tier2_json = args.tier2_json
        tier2_csv  = args.tier2_csv

        os.makedirs(os.path.dirname(tier2_json), exist_ok=True)
        os.makedirs(os.path.dirname(tier2_csv), exist_ok=True)

        champions = {r['fold_id']: r['best_params'] for r in json.load(open(tier1_json))}

        try:
            all_results = json.load(open(tier2_json))
            done = {r['fold_id'] for r in all_results}
        except FileNotFoundError:
            all_results, done = [], set()
        
        pending = [fid for fid in champions if fid not in done]

        for fid in tqdm(pending, desc='Tier2 ARIMA MOGA'):
            params = champions[fid]
            info = summary[fid]
            train = pd.read_csv(os.path.join(args.data_dir, info['train_path']))
            val = pd.read_csv(os.path.join(args.data_dir, info['val_path']))
            train_ret = train['Log_Returns'].dropna().values
            val_ret = val['Log_Returns'].values
            
            # build objective
            obj = create_periodic_arima_objective(
                    train_ret, val_ret,
                    retrain_interval=20,
                    cost=BASE_COST + SLIPPAGE
                    )
            prob = Tier2MOGAProblem(obj)
            
            # seeds with threshold perturbations
            base_thr = 0.5
            delta = 0.1
            thr_vals = [base_thr - delta, base_thr, base_thr + delta]
            seeds = [[params['p'], params['q'], t] for t in thr_vals]
            sampling = FromTier1SeedSampling(seeds, n_var=3)

            res_moga = pymoo_minimize(
                prob,
                NSGA2(pop_size=30, sampling=sampling),
                ('n_gen', DEFAULT_T2_NGEN),
                seed=42,
                verbose=False
            )

            # Pareto front
            front = []
            for x, F in zip(res_moga.X, res_moga.F):
                front.append({
                    'p': int(x[0]),
                    'q': int(x[1]),
                    'threshold': float(x[2]),
                    'sharpe': -F[0],
                    'mdd': F[1]
                })

            all_results.append({'fold_id': fid, 'pareto_front': front})
            with open(tier2_json, 'w') as f:
                json.dump(all_results, f, indent=2)

        # summary CSV
        records = []
        for r in all_results:
            fid = r['fold_id']
            front = r['pareto_front']
            if not front:
                continue
            best_sh = max(p['sharpe'] for p in front)
            mdds = [p['mdd'] for p in front if p['sharpe'] == best_sh]
            best_mdd = min(mdds) if mdds else None
            records.append({'fold_id': fid, 'best_sharpe': best_sh, 'mdd_at_best_sharpe': best_mdd})
        pd.DataFrame(records).to_csv(tier2_csv, index=False)
        logging.info('=== Tier 2 ARIMA complete ===')

if __name__ == '__main__':
    main()
