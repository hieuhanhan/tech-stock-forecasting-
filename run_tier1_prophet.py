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

from prophet import Prophet
from skopt.space import Real, Categorical
from skopt import gp_minimize

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as ga_minimize
from pymoo.core.problem import ElementwiseProblem

# ─── CONFIG & LOGGING ─────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/tier1_prophet.log"),
        logging.StreamHandler()
    ]
)
np.random.seed(42)

# ─── ARGUMENTS ────────────────────────────────────────────────
parser = argparse.ArgumentParser("Tier-1 GA→BO tuning for PROPHET")
parser.add_argument('--max-folds', type=int, default=None, help="Debug: only run first N folds")
parser.add_argument('--data-dir', type=str, default='data/processed_folds',
                    help="Folder with processed_folds/{train,val}_arima_prophet")
parser.add_argument('--out-file',  type=str, default='data/tuning_results/jsons/tier1_prophet.json',
                    help="Where to write tier-1 PROPHET results")
args = parser.parse_args()
out_dir = os.path.dirname(args.out_file)
os.makedirs(out_dir, exist_ok=True)
print("Saving to:", args.out_file)

# ─── PARAMETER SPACES ─────────────────────────────────────────
prophet_space = [
    Real(0.001, 0.5, name='changepoint_prior_scale', prior='log-uniform'),
    Real(0.01, 10.0, name='seasonality_prior_scale', prior='log-uniform'),
    Categorical(['additive','multiplicative'], name='seasonality_mode')
]
xl = np.array([0.001, 0.01, 0])
xu = np.array([0.5, 10.0, 1])

# ─── OBJECTIVE FUNCTION ───────────────────────────────────────────────
def prophet_rmse(params, train_df, val_df, fid=None):
    cps, sps, mode = params
    try:
        df = train_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            seasonality_mode=mode
        )
        model.fit(df)
        future = model.make_future_dataframe(periods=len(val_df), freq='D')
        fc = model.predict(future)['yhat'][-len(val_df):].values
        return sqrt(mean_squared_error(val_df['Close'].values, fc))
    except Exception as e:
        logging.warning(f"Prophet fold {fid} params={params} error: {e}")
        return 1e3

class ProphetProblem(ElementwiseProblem):
    def __init__(self, train_df, val_df, fid):
        super().__init__(n_var=3, n_obj=1, xl=xl, xu=xu)
        self.train = train_df
        self.val = val_df
        self.fid = fid

    def _evaluate(self, x, out, *_, **__):
        # Convert mode from float to category index
        mode = 'additive' if x[2] < 0.5 else 'multiplicative'
        out['F'] = [prophet_rmse([x[0], x[1], mode], self.train, self.val, self.fid)]

# ─── PARAMETER SELECTION BY VOLATILITY ─────────────────────────────────────────
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
    try:
        results = json.load(open(args.out_file))
        done = {r['fold_id'] for r in results}
        logging.info(f"Resuming from {len(done)} completed folds")
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

    for fid in tqdm(subset, desc="Tier1 Prophet"):
        if fid in done:
            continue

        info = summary[fid]
        train_df = pd.read_csv(os.path.join(args.data_dir, info['train_path_arima_prophet']),
                                 parse_dates=['Date'])
        val_df = pd.read_csv(os.path.join(args.data_dir, info['val_path_arima_prophet']),
                                 parse_dates=['Date'])

        category = vol_map.get(fid, "medium")
        ga_pop, ga_gen, bo_calls, top_n = get_params_for_volatility(category)
        logging.info(f"Fold {fid}: Volatility={category}, GA(pop={ga_pop}, gen={ga_gen}), BO(calls={bo_calls})")

        t0 = time.time()

        # Run GA
        ga_res = ga_minimize(
            ProphetProblem(train_df, val_df, fid),
            GA(pop_size=ga_pop),
            ('n_gen', ga_gen),
            verbose=False
        )

        # Extract top-N
        top_inds = sorted(ga_res.pop, key=lambda i: i.F[0])[:top_n]
        x0, y0 = [], []
        for ind in top_inds:
            vec = getattr(ind, 'X', getattr(ind, 'x', None))
            if vec is None:
                raise AttributeError("Individual missing both .X and .x")
            cps, sps, mode_raw = vec
            mode = 'additive' if mode_raw < 0.5 else 'multiplicative'
            x0.append([float(cps), float(sps), mode])
            y0.append(float(ind.F[0]))

        top_ga = x0


        # Run BO
        logging.info(f"Fold {fid}: starting BO(n_calls={bo_calls})")
        bo = gp_minimize(
            func=lambda x: prophet_rmse(x, train_df, val_df, fid),
            dimensions=prophet_space,
            n_calls=bo_calls,
            x0=x0,
            y0=y0,
            random_state=42
        )

        best_rmse = float(bo.fun)
        best_cps, best_sps, best_mode = bo.x

        results.append({
            'fold_id': fid,
            'best_params': {
                'changepoint_prior_scale': best_cps,
                'seasonality_prior_scale': best_sps,
                'seasonality_mode': best_mode
            },
            'best_rmse': best_rmse,
            'top_ga': top_ga
        })

        # write checkpoint
        with open(args.out_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        csv_path = os.path.join("data/tuning_results/csv", os.path.basename(args.out_file).replace(".json", ".csv"))
        pd.DataFrame(results).to_csv(csv_path, index=False)

        dt = time.time() - t0
        logging.info(f"→ Fold {fid} done in {dt:.1f}s | RMSE={best_rmse:.4f}")
        gc.collect()

    logging.info("=== Tier-1 Prophet complete ===")
    print(f"Results saved to {args.out_file}")
    



    

        

 