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
#!/usr/bin/env python3
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV

from tier_utils import (
    suppress_warnings,
    create_continuous_arima_objective,
    Tier2MOGAProblem,
    FromTier1SeedSampling,
    is_nondominated,
    knee_index,
    dedup_results,
    BASE_COST,
    SLIPPAGE,
    DEFAULT_T2_NGEN_ARIMA,
)


# ======================================================
# Helpers
# ======================================================
def append_rows_csv(rows: list[dict], path: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


# ======================================================
# Recorder for GA generations
# ======================================================
class GARecorder(Callback):
    def __init__(self, fold_id, retrain_interval):
        super().__init__()
        self.fold_id = fold_id
        self.retrain_interval = retrain_interval
        self.hv_rows = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        if F is None or F.size == 0:
            return
        ref_point = np.array([F[:, 0].max() * 1.1 + 1e-6,
                              F[:, 1].max() * 1.1 + 1e-6])
        hv = HV(ref_point=ref_point)(F)
        self.hv_rows.append({
            "fold_id": int(self.fold_id),
            "retrain_interval": int(self.retrain_interval),
            "stage": "GA",
            "gen": int(gen),
            "hv": float(hv),
            "ref0": float(ref_point[0]),
            "ref1": float(ref_point[1]),
            "n_front": int(F.shape[0]),
        })


# ======================================================
# Main
# ======================================================
def main(args):
    suppress_warnings()
    np.random.seed(42)

    # load folds
    with open(args.folds_json, "r") as f:
        folds = json.load(f)

    results = []
    tier1_df = pd.read_csv(args.tier1_csv)

    hv_path = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_hv.csv")

    for fold in folds:
        fid = fold["global_fold_id"]
        train_path, val_path = fold["train_path_arima"], fold["val_path_arima"]
        if not (Path(train_path).exists() and Path(val_path).exists()):
            continue
        train = pd.read_csv(train_path)["Log_Returns"].values
        val = pd.read_csv(val_path)["Log_Returns"].values

        # seeds from Tier-1
        seeds = tier1_df[tier1_df.fold_id == fid][["p", "q"]].drop_duplicates().values
        if seeds.size == 0:
            continue

        for interval in args.retrain_intervals:
            obj_func = create_continuous_arima_objective(
                train, val,
                retrain_interval=interval,
                cost_per_turnover=(BASE_COST + SLIPPAGE),
            )

            problem = Tier2MOGAProblem(obj_func)
            sampling = FromTier1SeedSampling(seeds, n_var=3)
            callback = GARecorder(fid, interval)

            algorithm = NSGA2(
                pop_size=args.pop_size,
                sampling=sampling,
                eliminate_duplicates=True,
            )
            termination = get_termination("n_gen", args.ngen)

            res = pymoo_minimize(problem,
                                 algorithm,
                                 termination,
                                 seed=42,
                                 callback=callback,
                                 verbose=False)

            if res.F is None or res.X is None or res.F.shape[0] == 0:
                continue

            # --- append GA HV log for this fold×interval ---
            append_rows_csv(callback.hv_rows, hv_path)
            callback.hv_rows.clear()

            # Pareto's final GA
            F_ga = res.F    # objectives (minimize): [f0=-Sharpe, f1=MDD]
            X_ga = res.X    # decision vars: [p,q,thr]

            nd = is_nondominated(F_ga)
            F_ga_front = F_ga[nd]
            X_ga_front = X_ga[nd]

            # BO refine
            seeds_bo = []
            for row in res.X:
                p, q, thr = row
                for dx in [-0.05, 0.0, 0.05]:
                    seeds_bo.append([p, q, max(0.01, min(0.6, thr + dx))])
            seeds_bo = np.array(seeds_bo)

            sampling_bo = FromTier1SeedSampling(seeds_bo, n_var=3)
            algorithm_bo = NSGA2(
                pop_size=args.pop_size,
                sampling=sampling_bo,
                eliminate_duplicates=True,
            )
            res_bo = pymoo_minimize(problem,
                                    algorithm_bo,
                                    termination,
                                    seed=123,
                                    verbose=False)

            F_bo = res_bo.F if res_bo.F is not None else np.empty((0, 2))
            X_bo = res_bo.X if res_bo.X is not None else np.empty((0, 3))

            F_union = np.vstack([F_ga_front, F_bo])
            X_union = np.vstack([X_ga_front, X_bo]) if X_bo.size else X_ga_front
            nd_union = is_nondominated(F_union)
            F_union_front = F_union[nd_union]
            X_union_front = X_union[nd_union]

            kidx = knee_index(F_union_front)
            F_knee = F_union_front[kidx]
            X_knee = X_union_front[kidx]

            results.append({
                "fold_id": fid,
                "retrain_interval": interval,
                "p": int(X_knee[0]),
                "q": int(X_knee[1]),
                "thr": float(X_knee[2]),
                "f0": float(F_knee[0]),
                "f1": float(F_knee[1]),
            })

            # save fronts
            front_df = pd.DataFrame(np.hstack([X_union_front, F_union_front]),
                                    columns=["p", "q", "thr", "f0", "f1"])
            out_front = Path(args.tier2_csv).with_name(
                f"tier2_front_fold{fid}_int{interval}.csv")
            front_df.to_csv(out_front, index=False)

            # save knee
            knee_df = pd.DataFrame([{
                "p": int(X_knee[0]),
                "q": int(X_knee[1]),
                "thr": float(X_knee[2]),
                "f0": float(F_knee[0]),
                "f1": float(F_knee[1]),
            }])
            out_knee = Path(args.tier2_csv).with_name(
                f"tier2_knee_fold{fid}_int{interval}.csv")
            knee_df.to_csv(out_knee, index=False)

            # ----- Hypervolume: GA-only vs GA∪BO -----
            ref0 = float(F_ga[:, 0].max()) * 1.1 + 1e-6
            ref1 = float(F_ga[:, 1].max()) * 1.1 + 1e-6
            ref_point = np.array([ref0, ref1], dtype=float)
            hv_metric = HV(ref_point=ref_point)

            hv_final_ga = float(hv_metric(F_ga_front)) if F_ga_front.size else np.nan
            hv_final_union = float(hv_metric(F_union_front)) if F_union_front.size else np.nan

            hv_final_rows = [
                {
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "stage": "final_ga",
                    "gen": -1,
                    "hv": hv_final_ga,
                    "ref0": ref0,
                    "ref1": ref1,
                    "n_front": int(F_ga_front.shape[0]),
                },
                {
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "stage": "final_union",
                    "gen": -1,
                    "hv": hv_final_union,
                    "ref0": ref0,
                    "ref1": ref1,
                    "n_front": int(F_union_front.shape[0]),
                },
            ]
            append_rows_csv(hv_final_rows, hv_path)

    # final results (knees)
    if results:
        df = pd.DataFrame(results)
        df = dedup_results(df)
        df.to_csv(args.tier2_csv, index=False)


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tier-2 MOGA for ARIMA+GARCH")
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--tier1-csv", required=True)
    parser.add_argument("--tier2-csv", required=True)
    parser.add_argument("--retrain-intervals", type=int, nargs="+", default=[20])
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--ngen", type=int, default=DEFAULT_T2_NGEN_ARIMA)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    main(args)

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
        processed_combinations = set(zip(df_existing['fold_id'], df_existing['retrain_interval']))
        all_results = df_existing.to_dict('records')
        logging.info(f"Already processed {len(processed_folds)} folds. Skipping them.")
    else:
        processed_combinations = set()

    if args.max_folds:
        champions = dict(list(champions.items())[:args.max_folds])
    # Filter out the folds that have already been processed
    champion_items = list(champions.items())  
    for fid, champ in tqdm(champion_items, desc="Sensitivity Analysis"):
        if fid not in summary:
            logging.warning(f"Fold {fid} not found in summary. Skipping.")
            continue
        info = summary[fid]
        train = pd.read_csv(os.path.join(data_dir, info['train_path']))
        val = pd.read_csv(os.path.join(data_dir, info['val_path']))
        features = [c for c in train.columns if c not in ['Date','Ticker','Log_Returns','target']]

        for interval in retrain_intervals:
            if (fid, interval) in processed_combinations:
                logging.info(f"Fold {fid}, interval {interval} already processed. Skipping.")
                continue

            logging.info(f"Processing Fold {fid}, Interval {interval}")

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