import os
import json
import logging
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

from arima_utils import (
    create_periodic_arima_objective,
    Tier2MOGAProblem,
    FromTier1SeedSampling,
    BASE_COST,
    SLIPPAGE,
    DEFAULT_T2_NGEN_ARIMA,
    DEFAULT_MIN_BLOCK_VOL,
)


# -------- Helpers --------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def require_log_returns(train_df: pd.DataFrame, val_df: pd.DataFrame, fid: int) -> bool:
    ok = ("Log_Returns" in train_df.columns) and ("Log_Returns" in val_df.columns)
    if not ok:
        logging.warning(f"[WARN] Missing 'Log_Returns' for fid={fid}")
    return ok

# Main
def main():
    parser = argparse.ArgumentParser("Tier-2 ARIMA (MOGA) — synced with Tier-1 utils")
    parser.add_argument("--meta-path", default="data/processed_folds/final/arima/selected_arima_final_paths.json",
                        help="Meta JSON used in Tier-1 (contains final_* paths)")
    parser.add_argument("--folds-path", default="data/processed_folds/final/arima/arima_tuning_folds.json",
                        help="Folds JSON from Tier-1 (list of representative fold_ids)")
    parser.add_argument("--tier1-json", default="data/tuning_results/jsons/tier1_arima.json",
                        help="Tier-1 results JSON with best p,q per fold")
    parser.add_argument("--tier2-json", default="data/tuning_results/jsons/tier2_arima.json")
    parser.add_argument("--tier2-csv", default="data/tuning_results/csv/tier2_arima.csv")

    parser.add_argument("--retrain-intervals", type=str, default="10,20,42", 
                        help="Comma-separated block sizes on val, e.g. 10,20,42")
    parser.add_argument("--pop-size", type=int, default=25)
    parser.add_argument("--ngen", type=int, default=DEFAULT_T2_NGEN_ARIMA)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--strict-paths", action="store_true")
    parser.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL, 
                        help="Minimum std of val block (log-returns) to accept")
    args = parser.parse_args()

    # Logging
    ensure_dir(Path("logs"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/tier2_arima.log"),
            logging.StreamHandler()
        ]
    )

    np.random.seed(42)
    retrain_intervals = [int(x) for x in args.retrain_intervals.split(",") if x.strip()]

    # Ensure output dirs
    ensure_dir(Path(args.tier2_json).parent)
    ensure_dir(Path(args.tier2_csv).parent)

    # ----- Load meta & folds -----
    meta = load_json(Path(args.meta_path))
    if isinstance(meta, dict) and "arima" in meta:
        meta_list = meta["arima"]
    elif isinstance(meta, list):
        meta_list = meta
    else:
        raise ValueError(f"Unexpected format in {args.meta_path}")
    summary = {item["global_fold_id"]: item for item in meta_list}

    reps_raw = load_json(Path(args.folds_path))
    reps_ids = [r["fold_id"] if isinstance(r, dict) else r for r in reps_raw]
    if args.max_folds:
        reps_ids = reps_ids[:args.max_folds]

    root_dir = Path(args.meta_path).parents[2]
    logging.info("[PATH] CWD=%s", Path.cwd().resolve())
    logging.info("[PATH] meta_path=%s", Path(args.meta_path).resolve())
    logging.info("[PATH] root_dir=%s", root_dir.resolve())

    tier1_results = load_json(Path(args.tier1_json))
    champions = {
        r["fold_id"]: r["best_params"]
        for r in tier1_results
        if r.get("fold_id") in reps_ids and "best_params" in r
    }
    # ----- Resume if exists -----
    all_results = []
    processed = set()
    if Path(args.tier2_csv).exists():
        logging.info(f"Found existing results at {args.tier2_csv} — resuming.")
        df_exist = pd.read_csv(args.tier2_csv)
        all_results = df_exist.to_dict("records")
        processed = set(df_exist["fold_id"].unique())
        logging.info(f"Already processed {len(processed)} folds.")

    pending = [fid for fid in reps_ids if fid in champions and fid not in processed]
    

    # ----- Main loop -----
    for fid in tqdm(pending, desc="Tier-2 ARIMA"):
        info = summary.get(fid)
        if not info:
            logging.warning(f"[WARN] Missing metadata for fid={fid}")
            continue

        train_rel = info.get("final_train_path") or info.get("train_path_arima")
        val_rel   = info.get("final_val_path")   or info.get("val_path_arima")
        if not train_rel or not val_rel:
            msg = f"Missing train/val paths in metadata for fid={fid}"
            if args.strict_paths:
                raise FileNotFoundError(msg)
            logging.warning("[WARN] " + msg)
            continue

        train_csv = (root_dir / train_rel).resolve()
        val_csv   = (root_dir / val_rel).resolve()
        if not (train_csv.exists() and val_csv.exists()):
            logging.warning(f"[WARN] Missing CSV for fid={fid}")
            continue

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        if not require_log_returns(train_df, val_df, fid):
            continue

        train_ret = train_df["Log_Returns"].dropna().to_numpy(float)
        val_ret   = val_df["Log_Returns"].fillna(0).to_numpy(float)
        if train_ret.size == 0 or val_ret.size == 0:
            logging.warning(f"[WARN] Empty series for fid={fid}")
            continue
        
        params = champions.get(fid)
        if not params:
            logging.warning(f"[WARN] No Tier-1 best_params for fid={fid}")
            continue
        
        # --- Seed threshold from volatility ---
        pred_range = np.nanmax(np.abs(val_ret))
        upper_thr = max(0.01, min(pred_range, 2.0))
        thr_vals = np.linspace(0.01, upper_thr, num=3)
        p0 = max(1, int(params["p"]))
        q0 = max(1, int(params["q"]))
        seeds = [[p0, q0, t] for t in thr_vals]
        
        for interval in retrain_intervals:
            logging.info(f"Fold {fid}, Retrain Interval {interval}")

            obj = create_periodic_arima_objective(
                train=train_ret,
                val=val_ret,
                retrain_interval=interval,
                cost_per_trade=BASE_COST + SLIPPAGE,
                min_block_vol=args.min_block_vol,
            )

            problem = Tier2MOGAProblem(obj)
            sampling = FromTier1SeedSampling(seeds, n_var=3)

            res = pymoo_minimize(
                problem,
                NSGA2(pop_size=args.pop_size, sampling=sampling),
                ("n_gen", args.ngen),
                seed=42,
                verbose=False
            )

            for x, F in zip(res.X, res.F):
                all_results.append({
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "p": int(round(x[0])),
                    "q": int(round(x[1])),
                    "threshold": float(x[2]),
                    "sharpe": float(-F[0]),
                    "mdd": float(F[1]),
                })
        # Save after each fold
        pd.DataFrame(all_results).to_csv(args.tier2_csv, index=False)
        with open(args.tier2_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"[CKPT] Saved after fold {fid}")

    logging.info("=== Tier-2 ARIMA complete ===")

if __name__ == '__main__':
    main()
