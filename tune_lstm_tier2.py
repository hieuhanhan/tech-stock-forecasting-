import numpy as np
import pandas as pd
import argparse
import json
import os
import logging
import tensorflow as tf
from tqdm import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

from lstm_utils import(
    FromTier1SeedSampling,
    Tier2MOGAProblem,
    create_periodic_lstm_objective,
    BASE_COST,
    SLIPPAGE,
    DEFAULT_T2_EPOCHS,
    DEFAULT_T2_NGEN_LSTM
)

def main():
    parser = argparse.ArgumentParser("Tier-2 LSTM")
    parser.add_argument('--data-dir', default='data/scaled_folds')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--tier1-json', default='data/tuning_results/jsons/tier1_lstm.json')
    parser.add_argument('--tier2-json', default='data/tuning_results/jsons/tier2_lstm.json')
    parser.add_argument('--tier2-csv', default='data/tuning_results/csv/tier2_lstm.csv')
    parser.add_argument('--retrain-intervals', type=str, default='10,20,42')
    args = parser.parse_args()

    data_dir = args.data_dir
    retrain_intervals = [int(x) for x in args.retrain_intervals.split(',')]

    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs(os.path.dirname(args.tier2_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.tier2_csv), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("logs/phase2_lstm.log"),
            logging.StreamHandler()
        ]
    )
    np.random.seed(42); tf.random.set_seed(42)
    logging.info(f"Running sensitivity analysis for retrain_intervals={retrain_intervals}")

    with open(os.path.join(data_dir, 'folds_summary_rescaled.json')) as f:
        summary = {f['fold_id']: f for f in json.load(f)['lstm']}  
    with open(os.path.join(data_dir, 'lstm', 'lstm_tuning_folds.json')) as f:
        reps = [r['fold_id'] for r in json.load(f)]
    with open(args.tier1_json) as f:
        champions_raw = json.load(f)
        champions = {r['fold_id']: r['champion'] for r in champions_raw if r['fold_id'] in reps}

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

    for fid, champ in tqdm(champion_items, desc="Sensitivity Analysis"):
        if fid not in summary:
            logging.warning(f"Fold {fid} not found in summary. Skipping.")
            continue
        info = summary[fid]
        train = pd.read_csv(os.path.join(data_dir, info['train_path']))
        val = pd.read_csv(os.path.join(data_dir, info['val_path']))
        features = [c for c in train.columns if c not in ['Date','Ticker','Log_Returns','target']]

        for interval in retrain_intervals:
            logging.info(f"Fold {fid}, Retrain Interval {interval}")

            obj = create_periodic_lstm_objective(
                train, val, 
                features, champ, 
                retrain_interval=interval, 
                cost=BASE_COST + SLIPPAGE)

            prob = Tier2MOGAProblem(obj)
            multipliers = [0.9, 0.95, 1.0, 1.05, 1.1]
            thresholds = [0.4, 0.5, 0.6]  
            seeds = []

            for m in multipliers:
                for t in thresholds:
                    # log-scale adjustment for learning rate
                    log_lr = np.log(champ['lr'])
                    adjusted_log_lr = log_lr * m
                    adjusted_lr = float(np.clip(np.exp(adjusted_log_lr), 1e-4, 1e-2))

                    seeds.append([
                        max(10, min(40, int(champ['window'] * m))),
                        max(32, min(64, int(champ['units'] * m))),
                        adjusted_lr,
                        DEFAULT_T2_EPOCHS,
                        float(t)
                    ])

            sampling = FromTier1SeedSampling(seeds, n_var=len(seeds[0]))

            res = pymoo_minimize(
                prob,
                NSGA2(pop_size=15, sampling=sampling),
                ('n_gen', DEFAULT_T2_NGEN_LSTM), 
                seed=42, 
                verbose=False
            )
            logging.info(f"NSGA2 result: Pareto front size = {len(res.F)}")

            for x, F in zip(res.X, res.F):
                w, units, lr, epochs, threshold = x
                all_results.append({
                    'fold_id': fid,
                    'ticker': info['ticker'],  
                    'retrain_interval': interval,
                    'window': int(w),
                    'units': int(units),
                    'learning_rate': float(f"{lr:.6f}"),
                    'epochs': int(epochs),
                    'threshold': float(threshold),
                    'sharpe': round(-F[0], 4),
                    'mdd': round(F[1], 6)
                })

            pd.DataFrame(all_results).to_csv(args.tier2_csv, index=False)
            with open(args.tier2_json + ".tmp", "w") as f:
                json.dump(all_results, f, indent=2)
            os.replace(args.tier2_json + ".tmp", args.tier2_json)

            logging.info(f"Checkpoint saved for fold {fid}, retrain_interval {interval}")

if __name__ == '__main__':
    main()