import os
import json
import argparse
import logging
import time
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as ga_minimize
from pymoo.core.problem import ElementwiseProblem

from skopt import gp_minimize
from skopt.space import Integer


# ─── CONFIG & LOGGING ─────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/tier1_arima.log"),
        logging.StreamHandler()
    ]
)
np.random.seed(42)

# ─── ARGUMENTS ────────────────────────────────────────────────
parser = argparse.ArgumentParser("Tier-1 GA→BO tuning for ARIMA")
parser.add_argument('--max-folds', type=int, default=None, help="Debug: only run first N folds")
parser.add_argument('--data-dir', type=str, default='data/processed_folds',
                    help="Folder with processed_folds/{train,val}_arima_prophet")
parser.add_argument('--out-file',  type=str, default='data/tuning_results/jsons/tier1_arima.json',
                    help="Where to write tier-1 ARIMA results")
args = parser.parse_args()
out_dir = os.path.dirname(args.out_file)
os.makedirs(out_dir, exist_ok=True)
print("Saving to:", args.out_file)

# ─── PARAMETER SPACES ─────────────────────────────────────────
arima_space = [ Integer(1, 7, name='p'),
                Integer(1, 7, name='q') ]
xl = np.array([1,1]); xu = np.array([7,7])

# ─── GA PROBLEM ───────────────────────────────────────────────
def arima_rmse(params, train, val):
    p, q = params
    try:
        m = ARIMA(train, order=(p,0,q)).fit()
        f = m.forecast(steps=len(val))
        return sqrt(mean_squared_error(val, f))
    except:
        return 1e3

class ARIMAProblem(ElementwiseProblem):
    def __init__(self, train, val):
        super().__init__(n_var=2, n_obj=1, xl=xl, xu=xu)
        self.t, self.v = train, val

    def _evaluate(self, x, out, *_, **__):
        rmse = arima_rmse(x, self.t, self.v)
        out['F'] = [rmse]

# ─── HELPERS ─────────────────────────────────────────
def bo_obj(params, train, val):
    return arima_rmse(params, train, val)

def get_params_for_volatility(category: str):
    category = category.lower()
    if category == 'low':
        return 20, 10, 30, 3
    elif category == 'medium':
        return 30, 15, 40, 5
    elif category == 'high':
        return 50, 25, 60, 7
    else:
        return 30, 15, 40, 5

# ─── MAIN LOOP ────────────────────────────────────────────────
if __name__ == '__main__':
    # load / resume checkpoint
    try:
        results = json.load(open(args.out_file))
        done    = {r['fold_id'] for r in results}
        logging.info(f"Resuming {len(done)} completed folds")
    except:
        results, done = [], set()

    # load fold IDs & summary
    rep_path = os.path.join(args.data_dir, 'shared_meta/representative_fold_ids.json')
    sum_path = os.path.join(args.data_dir, 'folds_summary.json')
    reps = json.load(open(rep_path))
    summary = {f['global_fold_id']: f for f in json.load(open(sum_path))}

    # Load volatility info
    vol_path = os.path.join(args.data_dir, 'shared_meta/fold_volatility_categorized.csv')
    vol_df = pd.read_csv(vol_path)
    vol_map = dict(zip(vol_df['global_fold_id'], vol_df['volatility_category']))

    max_folds = args.max_folds
    pending = [fid for fid in reps if fid not in done]
    subset = pending if max_folds is None else pending[:max_folds]

    for fid in tqdm(subset, desc="Tier1 ARIMA"):
        if fid in done:
            continue

        info = summary[fid]
        train_df = pd.read_csv(os.path.join(args.data_dir, info['train_path_arima_prophet']),
                                 parse_dates=['Date'])
        val_df = pd.read_csv(os.path.join(args.data_dir, info['val_path_arima_prophet']),
                                 parse_dates=['Date'])
        train_ret = train_df['Log_Returns'].dropna().values
        val_ret = val_df['Log_Returns'].values

        t0 = time.time()

        category = vol_map.get(fid, "medium")
        ga_pop, ga_gen, bo_calls, top_n = get_params_for_volatility(category)
        logging.info(f"Fold {fid}: Volatility={category}, GA(pop={ga_pop}, gen={ga_gen}), BO(calls={bo_calls})")

        # Run GA
        ga_res = ga_minimize(
            ARIMAProblem(train_ret, val_ret),
            GA(pop_size=ga_pop),
            ('n_gen', ga_gen),
            verbose=False   
            )

        # debug
        print("GA result object:", type(ga_res))
        print("Attributes on result:", dir(ga_res))
        print("Population type:", type(ga_res.pop))
        if hasattr(ga_res, 'pop'):
            print("First two individuals in ga_res.pop:")
            for ind in ga_res.pop[:2]:
                print("  -", type(ind), dir(ind))
        else:
            print("No attribute 'pop' on ga_res!")

        # top-N individuals
        top_inds = sorted(ga_res.pop, key=lambda i: i.F[0])[:top_n]
        x0, y0 = [], []

        for ind in top_inds:
            vec = getattr(ind, 'X', None)
            if vec is None:
                vec = getattr(ind, 'x', None)
            if vec is None:
                raise AttributeError("Individual has neither .X nor .x")
            
            vec = [int(v) for v in vec]
            x0.append(vec)
            y0.append(float(ind.F[0]))
        
        top_ga = [vec for vec in x0]

        # now sanity‐check
        for i, xi in enumerate(x0):
            if len(xi) != len(arima_space):
                raise ValueError(f"x0[{i}] length {len(xi)} != space dim {len(arima_space)}")

        logging.info(f"Fold {fid}: starting BO(n_calls={bo_calls})")
        bo = gp_minimize(
            func=lambda x: bo_obj(x, train_ret, val_ret),
            dimensions=arima_space,
            n_calls=bo_calls,
            x0=x0, y0=y0,
            random_state=42
        )

        best_p, best_q = int(bo.x[0]), int(bo.x[1])
        best_rmse = float(bo.fun)

        results.append({
            'fold_id': fid,
            'best_params': {'p': best_p, 'd': 0, 'q': best_q},
            'best_rmse': best_rmse,                  
            'top_ga': top_ga 
        })

        # write checkpoint
        with open(args.out_file, 'w') as f:
            json.dump(results, f, indent=2)
        # CSV for easy plotting
        pd.DataFrame(results).to_csv(args.out_file.replace('.json','.csv'), index=False)

        dt = time.time() - t0
        logging.info(f"→ Fold {fid} done in {time.time()-t0:.1f}s: "
                     f"p={best_p},q={best_q}, RMSE={best_rmse:.4f}")
        gc.collect()

    logging.info("=== Tier-1 ARIMA complete ===")
    print(f"Saved results to {args.out_file}")