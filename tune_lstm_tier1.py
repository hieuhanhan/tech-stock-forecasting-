import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import Path

from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from skopt.space import Integer, Real
from skopt import gp_minimize

from lstm_utils import (
    LOG_DIR,
    LSTM_SEARCH_SPACE,
    Tier1LSTMProblem,
    GA_POP_SIZE,
    GA_N_GEN,
    BO_N_CALLS,
    TOP_N_SEEDS
)

# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def load_lstm_features(path):
    with open(path) as f:
        feats = json.load(f)
    assert isinstance(feats, list) and all(isinstance(c, str) for c in feats)
    return feats

def prepare_X(df, features, strict=True, fill_value=0.0):
    missing = [c for c in features if c not in df.columns]
    extra   = [c for c in df.columns if c not in features]
    if missing and strict:
        raise KeyError(f"Missing LSTM features: {missing}")
    for c in missing:
        df[c] = fill_value
    X = df[features].to_numpy(dtype=np.float32, copy=False)
    return X, missing, extra

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser("Tier 1 LSTM Tuning")
    parser.add_argument('--meta-path', default='data/processed_folds/final/lstm/selected_lstm_final_paths.json',
                        help='JSON with train/val CSV paths per fold')
    parser.add_argument('--folds-path', default='data/processed_folds/final/lstm/lstm_tuning_folds.json',
                        help='JSON with list of fold_ids to tune')
    parser.add_argument('--feature-path', default='data/processed_folds/lstm_feature_columns.json',
                        help='JSON with list of features to tune')
    parser.add_argument('--output-json', default='data/tuning_results/jsons/tier1_lstm.json')
    parser.add_argument('--output-csv',  default='data/tuning_results/csv/tier1_lstm.csv')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--strict-paths', action='store_true')
    args = parser.parse_args()

    # Logging
    ensure_dir(Path(LOG_DIR))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Path(LOG_DIR) / "tier1_lstm.log"),
            logging.StreamHandler()
        ]
    )
    np.random.seed(42)

    ensure_dir(Path(args.output_json).parent)
    ensure_dir(Path(args.output_csv).parent)

    # Load metadata & folds
    summary_raw = load_json(Path(args.meta_path))
    if isinstance(summary_raw, dict) and "lstm" in summary_raw:
        summary_list = summary_raw["lstm"]
    elif isinstance(summary_raw, list):
        summary_list = summary_raw
    else:
        raise ValueError(f"Unexpected format in {args.meta_path}")
    summary = {f["global_fold_id"]: f for f in summary_list}

    reps_ids = load_json(Path(args.folds_path))  
    if args.max_folds:
        reps_ids = reps_ids[:args.max_folds]

    # Resume if exists
    try:
        results = load_json(Path(args.output_json))
        if isinstance(results, dict):
            results = results.get("results", [])
        done = {r['fold_id'] for r in results}
    except FileNotFoundError:
        results, done = [], set()

    pending = [fid for fid in reps_ids if fid not in done]

    # Tune each fold
    root_dir = Path(args.meta_path).parents[2]
    logging.info("[PATH] CWD=%s", Path.cwd().resolve())
    logging.info("[PATH] meta_path=%s", Path(args.meta_path).resolve())
    logging.info("[PATH] root_dir=%s", root_dir.resolve())


    for fid in tqdm(pending, desc="Tier1 LSTM"):
        logging.info(f"Running fold {fid} for ticker {summary[fid]['ticker']}")
        info = summary[fid]
        train = pd.read_csv(os.path.join(args.data_dir, info['train_path']))
        val = pd.read_csv(os.path.join(args.data_dir, info['val_path']))
        features = [c for c in train.columns if c not in ['Date','Ticker','Log_Returns','target']]

        X_train, y_train = train[features].values, train['target'].values
        X_val, y_val = val[features].values, val['target'].values
        num_features = X_train.shape[1]

        algorithm = GA(pop_size=GA_POP_SIZE, eliminate_duplicates=True, save_history=True)
        res_ga = pymoo_minimize(
            Tier1LSTMProblem(X_train, y_train, X_val, y_val, num_features),
            algorithm, ('n_gen',GA_N_GEN), 
            verbose=False,
            return_algorithm=True)
        
        all_individuals = []
        for entry in res_ga.algorithm.history:
            all_individuals.extend(entry.pop)

        seen, unique_seeds = set(), []
        for ind in all_individuals:
            key = tuple(ind.X.tolist())
            if key not in seen:
                seen.add(key)
                unique_seeds.append(ind)

        unique_seeds.sort(key=lambda ind: ind.F[0])
        top_seeds = unique_seeds[:TOP_N_SEEDS]
        x0 = [ind.X.tolist() for ind in top_seeds]
        y0 = [float(ind.F[0]) for ind in top_seeds]

        problem_instance = Tier1LSTMProblem(X_train, y_train, X_val, y_val, num_features)
        def bo_objective(x):
            out = {}
            problem_instance._evaluate(x, out)
            return out['F'][0]

        res_bo = gp_minimize(
            bo_objective,
            dimensions=LSTM_SEARCH_SPACE,
            n_calls=BO_N_CALLS, x0=x0, y0=y0, random_state=42)
        
        def log_best_params(best, search_space):
            logging.info("Best BO Parameters:")
            for name, val in zip([dim.name for dim in search_space], best):
                logging.info(f"{name}: {val}")

        best = res_bo.x
        log_best_params(best, LSTM_SEARCH_SPACE)
        params = ['window', 'layers', 'units', 'lr', 'batch_size']
        top_ga = []
        for ind in top_seeds:
            params_dict = dict(zip(params, ind.X))
            ga_dict = {
                'window': int(params_dict['window']),
                'layers': int(params_dict['layers']),
                'units': int(params_dict['units']),
                'lr': round(params_dict['lr'], 6),
                'batch_size': int(params_dict['batch_size'])
            }
            top_ga.append(ga_dict)

        results.append({
            'fold_id': fid, 
            'champion': {
                'window': int(best[0]), 
                'layers': int(best[1]),
                'units': int(best[2]), 
                'lr': float(best[3]),
                'batch_size': int(best[4])
            }, 
            'rmse': float(res_bo.fun),
            'top_ga': top_ga ,
            'ticker': summary[fid]['ticker']
        })
        logging.info(f"Completed fold {fid} with RMSE: {res_bo.fun:.4f}")
        with open(args.tier1_json, 'w') as f:
            json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(args.tier1_csv, index=False)
    logging.info("=== Tier 1 complete ===")

if __name__ == '__main__':
    main()

