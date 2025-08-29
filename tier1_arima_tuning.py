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
    LOG_DIR
)

# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def append_rows_csv(rows: list[dict], csv_path: Path):
    if not rows:
        return
    ensure_dir(csv_path.parent)
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)

def pick_arima_feature(df: pd.DataFrame) -> str:
    if "Log_Returns" not in df.columns:
        raise ValueError("Log_Returns not found")
    return "Log_Returns"

def load_tuning_folds(folds_path: Path) -> list[dict]:
    raw = load_json(folds_path)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "arima" in raw and isinstance(raw["arima"], list):
        return raw["arima"]
    raise ValueError(f"Unexpected JSON format in {folds_path}. Expected a list or a dict with key 'arima'.")

def infer_base_dir_for_csvs(folds_path: Path) -> Path:
    try:
        return folds_path.parents[2].resolve() 
    except IndexError:
        return Path("data/processed_folds").resolve()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser("Tier 1 ARIMA Tuning (Log_Returns only)")
    parser.add_argument('--folds-path', default='data/processed_folds/final/arima/arima_tuning_folds.json',
                        help='JSON with train/val CSV paths per fold (list or {"arima":[...]})')
    parser.add_argument('--output-json', default='data/tuning_results/jsons/tier1_arima.json')
    parser.add_argument('--output-csv',  default='data/tuning_results/csv/tier1_arima.csv')
    parser.add_argument('--ga-history-csv', default='data/tuning_results/csv/tier1_ga_history.csv')
    parser.add_argument('--bo-history-csv', default='data/tuning_results/csv/tier1_bo_history.csv')

    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--strict-paths', action='store_true', help='Raise on missing CSVs/columns')

    # search hyperparams (your flags)
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--ngen", type=int, default=40)
    parser.add_argument("--n-calls", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=5)

    # normalization flag
    parser.add_argument("--rmse-normalize", action="store_true",
                        help="Normalize RMSE by std(val) for scale-invariant scoring")

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

    # Load tuning folds
    folds_path = Path(args.folds_path).resolve()
    folds_list = load_tuning_folds(folds_path)
    summary = {int(f["global_fold_id"]): f for f in folds_list if "global_fold_id" in f}

    # Resume if exists
    try:
        results = load_json(Path(args.output_json))
        if isinstance(results, dict):
            results = results.get("results", [])
        done = {r['fold_id'] for r in results}
    except FileNotFoundError:
        results, done = [], set()

    all_fold_ids = sorted(summary.keys())
    if args.max_folds:
        all_fold_ids = all_fold_ids[:args.max_folds]
    pending = [fid for fid in all_fold_ids if fid not in done]

    # Base dir for CSVs (pointing to data/processed_folds)
    base_dir = infer_base_dir_for_csvs(folds_path)

    # Tune each fold
    for fid in tqdm(pending, desc='Tier1 ARIMA', leave=True, position=0, file=sys.stdout):
        info = summary.get(fid)
        if not info:
            logging.warning(f"[WARN] Missing metadata for fid={fid}")
            continue

        train_rel = info.get("final_train_path") or info.get("train_path_arima")
        val_rel = info.get("final_val_path") or info.get("val_path_arima")

        if not train_rel or not val_rel:
            msg = f"Missing train/val paths in metadata for fid={fid} (keys: final_* or *_arima)"
            if args.strict_paths:
                raise FileNotFoundError(msg)
            logging.warning("[WARN] " + msg)
            continue

        train_csv = (base_dir / train_rel).resolve()
        val_csv = (base_dir / val_rel).resolve()

        if not train_csv.exists() or not val_csv.exists():
            logging.warning(
                "[WARN] Missing CSV for fid=%s | train_exists=%s | val_exists=%s | train=%s | val=%s",
                fid, train_csv.exists(), val_csv.exists(), str(train_csv), str(val_csv)
            )
            if args.strict_paths:
                raise FileNotFoundError(f"Missing CSV(s) for fid={fid}")
            continue

        try:
            train = pd.read_csv(train_csv)
            val   = pd.read_csv(val_csv)
        except Exception as e:
            logging.warning(f"[WARN] Read CSV failed for fid={fid}: {e}")
            if args.strict_paths:
                raise
            continue

        try:
            feat = pick_arima_feature(train)
        except ValueError as e:
            if args.strict_paths: raise
            logging.warning(f"[WARN] {e} -> skip fid={fid}")
            continue

        train_ret = pd.to_numeric(train[feat], errors="coerce").dropna().to_numpy()
        val_ret   = pd.to_numeric(val[feat], errors="coerce").dropna().to_numpy()
        if train_ret.size == 0 or val_ret.size == 0:
            logging.warning(f"[WARN] Empty series for fid={fid} | train_len={train_ret.size} | val_len={val_ret.size}")
            if args.strict_paths:
                raise ValueError(f"Empty series for fid={fid}")
            continue

        # -------- GA Search (single-objective: RMSE) --------
        algorithm = GA(pop_size=args.pop_size, eliminate_duplicates=True, save_history=True)
        ga_rows = []
        try:
            res_ga = pymoo_minimize(
                Tier1ARIMAProblem(train_ret, val_ret, normalize=args.rmse_normalize),
                algorithm,
                ('n_gen', args.ngen),
                verbose=False,
                return_algorithm=True,
                seed=42
            )
            # dump all individuals per generation
            best_rmse_so_far = np.inf
            for gen_idx, gen in enumerate(res_ga.algorithm.history, start=1):
                pop = getattr(gen, 'pop', [])
                if not pop:
                    continue
                # best-of-generation for flag
                gen_rmses = [float(ind.F[0]) for ind in pop]
                gen_best = float(np.min(gen_rmses)) if len(gen_rmses) else np.inf
                for ind in pop:
                    p_, q_ = int(ind.X[0]), int(ind.X[1])
                    rmse_ = float(ind.F[0])
                    is_best_gen = (rmse_ == gen_best)
                    ga_rows.append({
                        "stage": "GA",
                        "fold_id": int(fid),
                        "gen": int(gen_idx),
                        "p": p_,
                        "q": q_,
                        "rmse": rmse_,
                        "is_best_gen": bool(is_best_gen),
                    })
                    best_rmse_so_far = min(best_rmse_so_far, rmse_)
        except Exception as e:
            logging.warning(f"[WARN] GA failed for fid={fid}: {e}")
            ga_rows = []

        append_rows_csv(ga_rows, Path(args.ga_history_csv))

        seen, unique_seeds = set(), []
        for row in sorted(ga_rows, key=lambda r: r["rmse"]):
            key = (int(row["p"]), int(row["q"]))
            if key not in seen:
                seen.add(key)
                unique_seeds.append(row)
        top_inds = unique_seeds[:args.n_seeds]
        if top_inds:
            x0 = [[int(r["p"]), int(r["q"])] for r in top_inds]
            y0 = [float(r["rmse"]) for r in top_inds]
        else:
            x0 = [[1, 1]]
            y0 = [arima_rmse([1, 1], train_ret, val_ret, normalize=args.rmse_normalize)]

        # -------- BO Refinement (still RMSE) --------
        bo_res = gp_minimize(
            lambda x: arima_rmse(x, train_ret, val_ret, normalize=args.rmse_normalize),
            dimensions=[Integer(1, 7), Integer(1, 7)],
            n_calls=args.n_calls,
            x0=x0, y0=y0,
            random_state=42
        )

        # log BO trajectory
        bo_rows = []
        best_so_far = np.inf
        for it, (x_iter, f_iter) in enumerate(zip(bo_res.x_iters, bo_res.func_vals), start=1):
            p_, q_ = int(x_iter[0]), int(x_iter[1])
            rmse_ = float(f_iter)
            best_so_far = min(best_so_far, rmse_)
            bo_rows.append({
                "stage": "BO",
                "fold_id": int(fid),
                "iter": int(it),
                "p": p_,
                "q": q_,
                "rmse": rmse_,
                "is_best_so_far": bool(rmse_ <= best_so_far + 1e-12),
            })
        append_rows_csv(bo_rows, Path(args.bo_history_csv))

        # -------- Final result for this fold --------
        results.append({
            'fold_id': int(fid),
            'best_params': {'p': int(bo_res.x[0]), 'd': 0, 'q': int(bo_res.x[1])},
            'rmse': float(bo_res.fun),
            'rmse_normalized': bool(args.rmse_normalize),
            'feature': feat,
            'top_ga': [
                {'p': int(r["p"]), 'q': int(r["q"]), 'rmse': float(r["rmse"]), 'gen': int(r["gen"])}
                for r in top_inds
            ],
            'pop_size': int(args.pop_size),
            'n_gen': int(args.ngen),
            'n_calls': int(args.n_calls),
            'n_seeds': int(args.n_seeds),
        })

        with Path(args.output_json).open('w') as f:
            json.dump(results, f, indent=2)

    # Final CSV (summary champions)
    pd.DataFrame(results).to_csv(Path(args.output_csv), index=False)
    logging.info('=== Tier 1 ARIMA complete ===')

if __name__ == '__main__':
    main()