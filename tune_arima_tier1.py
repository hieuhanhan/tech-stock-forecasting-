# tier1_arima.py
import json
import logging
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
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
    TOP_N_SEEDS,
)

# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def pick_arima_feature(df: pd.DataFrame) -> str:
    """Always use Log_Returns for ARIMA."""
    if "Log_Returns" not in df.columns:
        raise ValueError("Log_Returns not found")
    return "Log_Returns"

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser("Tier 1 ARIMA Tuning (Log_Returns only)")
    parser.add_argument('--meta-path', default='data/processed_folds/final/arima/selected_arima_final_paths.json',
                        help='JSON with train/val CSV paths per fold')
    parser.add_argument('--folds-path', default='data/processed_folds/final/arima/arima_tuning_folds.json',
                        help='JSON with list of fold_ids to tune')
    parser.add_argument('--output-json', default='data/tuning_results/jsons/tier1_arima.json')
    parser.add_argument('--output-csv',  default='data/tuning_results/csv/tier1_arima.csv')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--strict-paths', action='store_true')
    args = parser.parse_args()

    # Logging setup
    ensure_dir(Path(LOG_DIR))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Path(LOG_DIR) / "tier1_arima.log"),
            logging.StreamHandler()
        ]
    )
    np.random.seed(42)

    ensure_dir(Path(args.output_json).parent)
    ensure_dir(Path(args.output_csv).parent)

    # Load metadata & folds
    summary_raw = load_json(Path(args.meta_path))
    if isinstance(summary_raw, dict) and "arima" in summary_raw:
        summary_list = summary_raw["arima"]
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

    root_dir = Path(args.meta_path).parents[2]
    logging.info("[PATH] CWD=%s", Path.cwd().resolve())
    logging.info("[PATH] meta_path=%s", Path(args.meta_path).resolve())
    logging.info("[PATH] root_dir=%s", root_dir.resolve())

    # Tune each fold
    for fid in tqdm(pending, desc='Tier1 ARIMA', leave=True, position=0, file=sys.stdout):
        info = summary.get(fid)
        if not info:
            logging.warning(f"[WARN] Missing metadata for fid={fid}")
            continue

        train_rel = info.get("final_train_path") or info.get("train_path_arima")
        val_rel   = info.get("final_val_path")   or info.get("val_path_arima")

        if not train_rel or not val_rel:
            msg = f"Missing train/val paths in metadata for fid={fid} (keys: final_* or *_arima)"
            if args.strict_paths:
                raise FileNotFoundError(msg)
            logging.warning("[WARN] " + msg)
            continue

        train_csv = (root_dir / train_rel).resolve()
        val_csv   = (root_dir / val_rel).resolve()

        if not (train_csv.exists() and val_csv.exists()):
            logging.warning(f"[WARN] Missing CSV for fid={fid}")
            continue

        train = pd.read_csv(train_csv)
        val   = pd.read_csv(val_csv)

        try:
            feat = pick_arima_feature(train)
        except ValueError as e:
            if args.strict_paths: raise
            logging.warning(f"[WARN] {e} -> skip fid={fid}")
            continue

        train_ret = train[feat].dropna().to_numpy(float)
        val_ret   = val[feat].to_numpy(float)
        if train_ret.size == 0 or val_ret.size == 0:
            logging.warning(f"[WARN] Empty series for fid={fid}")
            continue

        # GA Search
        algorithm = GA(pop_size=GA_POP_SIZE, eliminate_duplicates=True, save_history=True)
        res_ga = pymoo_minimize(
            Tier1ARIMAProblem(train_ret, val_ret),
            algorithm,
            ('n_gen', GA_N_GEN),
            verbose=False,
            return_algorithm=True,
            seed=42
        )

        all_inds = [ind for gen in res_ga.algorithm.history for ind in getattr(gen, 'pop', [])]
        seen, unique_seeds = set(), []
        for ind in all_inds:
            key = (int(ind.X[0]), int(ind.X[1]))
            if key not in seen:
                seen.add(key)
                unique_seeds.append(ind)
        unique_seeds.sort(key=lambda ind: ind.F[0])

        if unique_seeds:
            x0 = [[int(ind.X[0]), int(ind.X[1])] for ind in unique_seeds[:TOP_N_SEEDS]]
            y0 = [float(ind.F[0]) for ind in unique_seeds[:TOP_N_SEEDS]]
        else:
            x0 = [[1, 1]]
            y0 = [arima_rmse([1, 1], train_ret, val_ret)]

        # BO Refinement
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
            'feature': feat,
            'top_ga': [
                {'p': int(ind.X[0]), 'q': int(ind.X[1]), 'rmse': float(ind.F[0])}
                for ind in unique_seeds
            ]
        })

        with Path(args.output_json).open('w') as f:
            json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(Path(args.output_csv), index=False)
    logging.info('=== Tier 1 ARIMA complete ===')

if __name__ == '__main__':
    main()