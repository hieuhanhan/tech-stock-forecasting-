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
    evaluate_strategy, 
    eval_cache,
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

def neighbor_params(p: int, q: int):
    cands = []
    for dp in (-1, 0, 1):
        for dq in (-1, 0, 1):
            pp = int(np.clip(p + dp, 1, 7))
            qq = int(np.clip(q + dq, 1, 7))
            cands.append((pp, qq))
    return sorted(set(cands))

def dedup_results(rows: list[dict]) -> list[dict]:
    seen = set()
    uniq = []
    for r in rows:
        key = (
            int(r["fold_id"]),
            int(r["retrain_interval"]),
            int(r["p"]),
            int(r["q"]),
            round(float(r["threshold"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser("Tier-2 ARIMA (NSGA-II) — adaptive + soft rebalance")
    parser.add_argument("--folds-path", default="data/processed_folds/final/arima/arima_tuning_folds.json")
    parser.add_argument("--tier1-json",  default="data/tuning_results/jsons/tier1_arima.json")
    parser.add_argument("--tier2-json",  default="data/tuning_results/jsons/tier2_arima_soft.json")
    parser.add_argument("--tier2-csv",   default="data/tuning_results/csv/tier2_arima_soft.csv")
    parser.add_argument("--retrain-intervals", type=str, default="10,20,42")
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--ngen", type=int, default=DEFAULT_T2_NGEN_ARIMA)

    # Adaptive modes
    parser.add_argument("--adaptive-mode", choices=["block","power","quantile"], default="block")
    parser.add_argument("--thr-min", type=float, default=0.005)
    parser.add_argument("--thr-max", type=float, default=0.4)
    parser.add_argument("--soft-factor", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=-0.2)
    parser.add_argument("--ref-vol-ema-span", type=int, default=60)
    parser.add_argument("--q-window", type=int, default=120)
    parser.add_argument("--q-level", type=float, default=0.95)

    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--strict-paths", action="store_true")
    parser.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Logging
    ensure_dir(Path("logs"))
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/tier2_arima.log"), logging.StreamHandler()]
    )

    np.random.seed(42)
    rng = np.random.default_rng(42)
    retrain_intervals = [int(x) for x in args.retrain_intervals.split(",") if x.strip()]

    # Ensure output dirs
    ensure_dir(Path(args.tier2_json).parent)
    ensure_dir(Path(args.tier2_csv).parent)

    # ----- Load meta & folds -----
    reps_ids = load_json(Path(args.folds_path))
    if isinstance(reps_ids, dict) and "arima" in reps_ids:
        summary_list = reps_ids["arima"]
    elif isinstance(reps_ids, list):
        summary_list = reps_ids
    else:
        raise ValueError(f"Unexpected format in {args.folds_path}")
    summary = {f["global_fold_id"]: f for f in summary_list}
    all_fold_ids = list(summary.keys())

    folds_path = Path(args.folds_path).resolve()
    base_dir = folds_path.parents[2] 

    # Tier-1 champions
    tier1_results = load_json(Path(args.tier1_json))
    if isinstance(tier1_results, dict):
        tier1_results = tier1_results.get("results", [])
    champions = {
        int(r["fold_id"]): r["best_params"]
        for r in tier1_results
        if r.get("fold_id") in all_fold_ids and "best_params" in r
    }

    # ----- Resume if exists -----
    all_results = []
    processed = set()
    if Path(args.tier2_csv).exists():
        df_exist = pd.read_csv(args.tier2_csv)
        all_results = df_exist.to_dict("records")
        processed = set(int(x) for x in df_exist["fold_id"].unique())

    pending = [fid for fid in all_fold_ids if fid in champions and fid not in processed]
    if args.max_folds is not None:
        pending = pending[: int(args.max_folds)]
    
    # ----- Main loop -----
    for fid in tqdm(pending, desc="Tier-2 ARIMA"):
        info = summary.get(fid)
        if not info:
            continue

        eval_cache.clear()
        logging.debug(f"[Fold {fid}] Cache cleared, starting optimization.")

        train_rel = info.get("final_train_path") or info.get("train_path_arima")
        val_rel   = info.get("final_val_path")   or info.get("val_path_arima")
        if not train_rel or not val_rel:
            if args.strict_paths:
                raise FileNotFoundError(f"Missing train/val for fid={fid}")
            continue 

        train_csv = (base_dir / train_rel).resolve()
        val_csv   = (base_dir / val_rel).resolve()
        if not (train_csv.exists() and val_csv.exists()):
            if args.strict_paths:
                raise FileNotFoundError(f"Missing CSV for fid={fid}")
            continue

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        if not require_log_returns(train_df, val_df, fid):
            continue

        train_ret = train_df["Log_Returns"].dropna().to_numpy(float)
        val_ret   = val_df["Log_Returns"].fillna(0).to_numpy(float)
        if train_ret.size == 0 or val_ret.size == 0:
            continue

        if args.debug:
            logging.debug(f"[Fold {fid}] train_len={train_ret.size}, val_len={val_ret.size}")
            logging.debug(f"[Fold {fid}] train_ret: mean={np.mean(train_ret):.6e}, std={np.std(train_ret):.6e}, "
                          f"min={np.min(train_ret):.6e}, max={np.max(train_ret):.6e}")
            logging.debug(f"[Fold {fid}] val_ret  : mean={np.mean(val_ret):.6e}, std={np.std(val_ret):.6e}, "
                          f"min={np.min(val_ret):.6e}, max={np.max(val_ret):.6e}")
            if (not np.isfinite(train_ret).all()) or (not np.isfinite(val_ret).all()):
                logging.debug(f"[Fold {fid}] WARNING: non-finite values detected in returns!")
        
        params = champions.get(fid)
        if not params:
            continue
        
        # (p, q) neighbor seeds
        p0 = max(1, int(params["p"]))
        q0 = max(1, int(params["q"]))
        pq_seeds = neighbor_params(p0, q0)

        rng = np.random.default_rng(42)

        for interval in retrain_intervals:
            if args.debug:
                logging.debug(f"[Fold {fid}] interval={interval}, p0={p0}, q0={q0}, seeds={pq_seeds}, "
                              f"thr_range=[{args.thr_min}, {args.thr_max}], adaptive_mode={args.adaptive_mode}")
                
            obj = create_periodic_arima_objective(
                train=train_ret,
                val=val_ret,
                retrain_interval=interval,
                cost_per_trade=BASE_COST + SLIPPAGE,
                min_block_vol=args.min_block_vol,
                scale_factor=1000.0,
                arch_rescale_flag=False,
                debug=args.debug,
                adaptive_mode=args.adaptive_mode,
                soft_factor=args.soft_factor,
                alpha=args.alpha,
                ref_vol_ema_span=args.ref_vol_ema_span,
                q_window=args.q_window,
                q_level=args.q_level,
                use_arima_residuals=True,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
            )  

            if args.debug:
                logging.debug(f"[Fold {fid}] Objective created: cost={BASE_COST + SLIPPAGE}, "
                              f"min_block_vol={args.min_block_vol}, scale_factor=1000.0")

            # Seed: random uniform threshold per (p,q)
            seed_points = [[pp, qq, float(rng.uniform(args.thr_min, args.thr_max))] for (pp, qq) in pq_seeds]
            logging.debug(f"[Fold {fid}] Seed points: {seed_points[:5]}... (total={len(seed_points)})")

            sampling = FromTier1SeedSampling(seed_points, n_var=3)
            problem = Tier2MOGAProblem(obj)
            # Update thr bounds dynamically
            problem.xl[2] = args.thr_min
            problem.xu[2] = args.thr_max

            if args.debug:
                logging.debug(f"[Fold {fid}] Starting NSGA-II: pop_size={args.pop_size}, ngen={args.ngen}, "
                              f"#seed_points={len(seed_points)}; first_seeds={seed_points[:3]}")

            res = pymoo_minimize(
                problem,
                NSGA2(pop_size=args.pop_size, sampling=sampling),
                ("n_gen", args.ngen),
                seed=42,
                verbose=False
            )

            n_sol = 0 if (res.F is None) else res.F.shape[0]
            logging.debug(f"[Fold {fid}] NSGA-II done. Num solutions={n_sol}")

            if res.X is None or res.F is None or n_sol == 0:
                logging.debug(f"[Fold {fid}] No solutions returned by NSGA-II.")
                continue

            logging.debug(f"[Fold {fid}] Example solution[0]: X={res.X[0]}, F={res.F[0]}")
            best_idx = int(np.argmin(res.F[:, 0]))
            logging.debug(f"[Fold {fid}] Best-by-F0 idx={best_idx}, X={res.X[best_idx]}, F={res.F[best_idx]}")
            
            fold_results = [{
                "fold_id": int(fid),
                "retrain_interval": int(interval),
                "p": int(round(x[0])),
                "q": int(round(x[1])),
                "threshold": float(x[2]),
                "sharpe": float(-F[0]),
                "mdd": float(F[1]),
                "soft_rebalance": True
            } for x, F in zip(res.X, res.F)]

            if args.debug:
                sharpe_vals = [float(-F[0]) for F in res.F] if res.F is not None else []
                mdd_vals    = [float(F[1])  for F in res.F] if res.F is not None else []
                if sharpe_vals:
                    logging.debug(f"[Fold {fid}] Interval={interval} Sharpe: "
                                  f"min={np.min(sharpe_vals):.6f}, median={np.median(sharpe_vals):.6f}, "
                                  f"max={np.max(sharpe_vals):.6f}")
                    logging.debug(f"[Fold {fid}] Interval={interval} MDD   : "
                                  f"min={np.min(mdd_vals):.6f}, median={np.median(mdd_vals):.6f}, "
                                  f"max={np.max(mdd_vals):.6f}")

            all_results.extend(dedup_results(fold_results))

        all_results = dedup_results(all_results)
        pd.DataFrame(all_results).to_csv(args.tier2_csv, index=False)
        with open(args.tier2_json, "w") as f:
            json.dump(all_results, f, indent=2)

    logging.info("=== Tier-2 ARIMA complete ===")

if __name__ == '__main__':
    main()