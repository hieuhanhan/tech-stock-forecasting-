import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize

from arima_utils import (
    Tier1ARIMAProblem,
    arima_rmse,
    LOG_DIR,
    GA_POP_SIZE,
    GA_N_GEN,
    BO_N_CALLS,
    TOP_N_SEEDS
)

def main():
    parser = argparse.ArgumentParser("Tier 1 ARIMA Tuning")
    parser.add_argument('--data-dir', default='data/scaled_folds')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--tier1-json', default='data/tuning_results/jsons/tier1_arima.json')
    parser.add_argument('--tier1-csv',  default='data/tuning_results/csv/tier1_arima.csv')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "tier1_arima.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    np.random.seed(42)

    folds_summary = os.path.join(args.data_dir, 'folds_summary_rescaled.json')
    reps_path = os.path.join(args.data_dir, 'arima', 'arima_tuning_folds.json')
    summary = {f['fold_id']: f for f in json.load(open(folds_summary))['arima']}
    reps = [r['fold_id'] for r in json.load(open(reps_path))]

    os.makedirs(os.path.dirname(args.tier1_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.tier1_csv), exist_ok=True)

    try:
        results = json.load(open(args.tier1_json))
        done = {r['fold_id'] for r in results}
    except FileNotFoundError:
        results, done = [], set()

    pending = [fid for fid in reps if fid not in done]
    if args.max_folds:
        pending = pending[:args.max_folds]

    for fid in tqdm(pending, desc='Tier1 ARIMA'):
        info = summary[fid]
        train = pd.read_csv(os.path.join(args.data_dir, info['train_path']))
        val = pd.read_csv(os.path.join(args.data_dir, info['val_path']))
        train_ret = train['Log_Returns'].dropna().values
        val_ret = val['Log_Returns'].values

        algorithm = GA(pop_size=GA_POP_SIZE, eliminate_duplicates=True, save_history=True)
        res_ga = pymoo_minimize(
            Tier1ARIMAProblem(train_ret, val_ret),
            algorithm,
            ('n_gen', GA_N_GEN),
            verbose=False,
            return_algorithm=True
        )

        all_individuals = []
        for entry in res_ga.algorithm.history:
            all_individuals.extend(entry.pop)

        seen, unique_seeds = set(), []
        for ind in all_individuals:
            key = (int(ind.X[0]), int(ind.X[1]))
            if key not in seen:
                seen.add(key)
                unique_seeds.append(ind)

        unique_seeds.sort(key=lambda ind: ind.F[0])
        top_seeds = unique_seeds[:TOP_N_SEEDS]
        x0 = [[int(ind.X[0]), int(ind.X[1])] for ind in top_seeds]
        y0 = [float(ind.F[0]) for ind in top_seeds]

        res_bo = gp_minimize(
            lambda x: arima_rmse(x, train_ret, val_ret),
            dimensions=[Integer(1, 7), Integer(1, 7)],
            n_calls=BO_N_CALLS,
            x0=x0, y0=y0,
            random_state=42
        )

        results.append({
            'fold_id': fid,
            'best_params': {'p': int(res_bo.x[0]), 'd': 0, 'q': int(res_bo.x[1])},
            'rmse': float(res_bo.fun),
            'top_ga': [
                {'p': int(ind.X[0]), 'q': int(ind.X[1]), 'rmse': float(ind.F[0])}
                for ind in unique_seeds
            ]
        })
        json.dump(results, open(args.tier1_json, 'w'), indent=2)

    pd.DataFrame(results).to_csv(args.tier1_csv, index=False)
    logging.info('=== Tier 1 ARIMA complete ===')


if __name__ == '__main__':
    main()
