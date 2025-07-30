import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

from arima_utils import (
    create_periodic_arima_objective,
    Tier2MOGAProblem,
    FromTier1SeedSampling,
    BASE_COST,
    SLIPPAGE,
    DEFAULT_T2_NGEN_ARIMA
)

def main():
    parser = argparse.ArgumentParser("Tier-2 ARIMA Script")
    parser.add_argument('--data-dir', default='data/scaled_folds')
    parser.add_argument('--tier1-json', default='data/tuning_results/jsons/tier1_arima.json')
    parser.add_argument('--tier2-json', default='data/tuning_results/jsons/tier2_arima.json')
    parser.add_argument('--tier2-csv', default='data/tuning_results/csv/tier2_arima.csv')
    parser.add_argument('--retrain-intervals', type=str, default='10,20,42')
    parser.add_argument('--max-folds', type=int, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    retrain_intervals = [int(x) for x in args.retrain_intervals.split(',')]
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("logs/phase2_arima.log"),
            logging.StreamHandler()
        ]
    )

    np.random.seed(42)
    logging.info(f"Running sensitivity analysis for retrain_intervals={retrain_intervals}")

    # Load fold summary
    with open(os.path.join(data_dir, 'folds_summary_rescaled.json')) as f:
        summary = {f['fold_id']: f for f in json.load(f)['arima']}

    # Load representative folds
    with open(os.path.join(data_dir, 'arima', 'arima_tuning_folds.json')) as f:
        reps = [r['fold_id'] for r in json.load(f)]

    # Load best p, q from Phase 1
    with open(args.tier1_json) as f:
        champions_raw = json.load(f)
        champions = {r['fold_id']: r['best_params'] for r in champions_raw if r['fold_id'] in reps}

    # Initialize result container
    processed_folds = set()
    all_results = []
    champion_items = list(champions.items())

    if os.path.exists(args.tier2_csv):
        logging.info(f"Found existing results file at {args.tier2_csv}. Resuming...")
        df_existing = pd.read_csv(args.tier2_csv)
        processed_folds = set(df_existing['fold_id'].unique())
        all_results = df_existing.to_dict('records')
        logging.info(f"Already processed {len(processed_folds)} folds. Skipping them.")

    if args.max_folds:
        champions = dict(list(champions.items())[:args.max_folds])
    # Filter out the folds that have already been processed
    champion_items = [(fid, params) for fid, params in champions.items() if fid not in processed_folds]

    for fid, params in tqdm(champion_items, desc="Sensitivity Analysis"):
        info = summary[fid]
        train = pd.read_csv(os.path.join(data_dir, info['train_path']))
        val = pd.read_csv(os.path.join(data_dir, info['val_path']))
        train_ret = train['Log_Returns'].dropna().values
        val_ret = val['Log_Returns'].values

        for interval in retrain_intervals:
            logging.info(f"Fold {fid}, Retrain Interval {interval}")

            obj = create_periodic_arima_objective(
                train_ret, val_ret,
                retrain_interval=interval,
                cost=BASE_COST + SLIPPAGE
            )

            prob = Tier2MOGAProblem(obj)
            thr_vals = [0.4, 0.5, 0.6]
            seeds = [[params['p'], params['q'], t] for t in thr_vals]
            sampling = FromTier1SeedSampling(seeds, n_var=3)

            res = pymoo_minimize(
                prob,
                NSGA2(pop_size=25, sampling=sampling),
                ('n_gen', DEFAULT_T2_NGEN_ARIMA),
                seed=42,
                verbose=False
            )

            for x, F in zip(res.X, res.F):
                all_results.append({
                    'fold_id': fid,
                    'retrain_interval': interval,
                    'p': int(x[0]),
                    'q': int(x[1]),
                    'threshold': float(x[2]),
                    'sharpe': -F[0],
                    'mdd': F[1]
                })
        # Save results immediately after processing this fold
        pd.DataFrame(all_results).to_csv(args.tier2_csv, index=False)
        with open(args.tier2_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Checkpoint saved for fold {fid}")

if __name__ == '__main__':
    main()
