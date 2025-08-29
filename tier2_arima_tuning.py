"""
Tier-2 ARIMA (NSGA-II + BO/ParEGO) — Continuous sizing
- Multi-objective: maximize Sharpe, minimize MDD (we minimize [-Sharpe, MDD])
- Seeds from Tier-1 (best p, q) -> NSGA-II exploration -> BO/ParEGO refinement
- Auto/forced GARCH innovations (normal vs student-t) with IC selection (AIC/BIC/HQIC)
- Full tracing of distribution choice + ICs into CSV logs
"""
import os, json, logging
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.stats import norm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.indicators.hv import HV

from arima_utils import (
    suppress_warnings,
    create_continuous_arima_objective,
    Tier2MOGAProblem,
    FromTier1SeedSampling,
    is_nondominated, knee_index,
    BASE_COST, SLIPPAGE, 
    DEFAULT_T2_NGEN_ARIMA, 
    DEFAULT_MIN_BLOCK_VOL
)
suppress_warnings()

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
            str(r.get("stage", "")),
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

def append_rows_csv(rows: list[dict], path: Path):
    if not rows:
        return
    ensure_dir(path.parent)
    write_header = not path.exists()
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)

# -------- GA callback recorder --------
class GARecorder:
    def __init__(self, fold_id, interval, objective, hv_log_list, ga_rows_list, seed=42,
                 garch_mode="auto", garch_ic="aic"):
        self.fold_id = int(fold_id)
        self.interval = int(interval)
        self.obj = objective      # callable with .stats_map
        self.hv_log = hv_log_list # list to append
        self.ga_rows = ga_rows_list
        self.seed = seed
        self._ref = None          # hypervolume ref-point
        self.garch_mode = str(garch_mode)
        self.garch_ic = str(garch_ic)       # hypervolume ref-point

    def __call__(self, algo):
        gen = int(algo.n_gen)
        pop = algo.pop
        X = pop.get("X"); F = pop.get("F")
        # Pareto front of current pop
        nd_mask = is_nondominated(F)
        F_nd = F[nd_mask]; X_nd = X[nd_mask]

        # set/update ref-point once (stable across generations)
        if self._ref is None and F.shape[0] > 0:
            ref0 = float(F[:,0].max()) * 1.1 + 1e-6
            ref1 = float(F[:,1].max()) * 1.1 + 1e-6
            self._ref = np.array([ref0, ref1], dtype=float)

        if self._ref is not None and F_nd.shape[0] > 0:
            hv = HV(ref_point=self._ref)(F_nd)
            self.hv_log.append({
                "fold_id": self.fold_id,
                "retrain_interval": self.interval,
                "stage": "GA",
                "gen": gen,
                "hv": float(hv),
                "ref0": float(self._ref[0]),
                "ref1": float(self._ref[1]),
                "n_front": int(F_nd.shape[0]),
            })

        # dump ND points with metadata (turnover + GARCH dist trace if available)
        stats_map = getattr(self.obj, "stats_map", {})
        for x, f in zip(X_nd, F_nd):
            p = int(np.clip(np.rint(x[0]), 1, 7))
            q = int(np.clip(np.rint(x[1]), 1, 7))
            thr = float(x[2])
            key = (p, q, round(thr, 6))
            meta = stats_map.get(key, {})

            self.ga_rows.append({
                "stage": "GA",
                "seed": int(self.seed),
                "gen": int(gen),
                "iter": -1,
                "fold_id": int(self.fold_id),
                "retrain_interval": int(self.interval),
                "p": p, "q": q, "threshold": thr,
                "sharpe": float(-f[0]),        # f0 = -Sharpe
                "mdd": float(f[1]),
                "turnover": float(meta.get("turnover", np.nan)),
                "penalty": float(meta.get("penalty", np.nan)),
                "strategy": "continuous",
                # --- GARCH tracing ---
                "garch_mode": self.garch_mode,
                "garch_ic": self.garch_ic,
                "dist": meta.get("dist_majority", None),
                "dist_last": meta.get("dist_last", None),
                "aic_norm": meta.get("aic_norm", np.nan),
                "aic_t": meta.get("aic_t", np.nan),
            })

# -------- BO --------
def parego_scalarize(F, lam, rho=0.05):
    F = np.asarray(F, dtype=float)
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    g = np.max(lam * Fn, axis=1) + rho * np.sum(lam * Fn, axis=1)
    return g

def expected_improvement(mu, sigma, y_best):
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def run_bo_parego(
    objective,
    bounds,                # ([1,1,thr_min], [7,7,thr_max])
    init_X, init_F,        # seeds from GA front
    n_iter=40,
    n_pool=2000,
    random_state=42,
    fold_id=0,
    interval=10,
    garch_mode="auto",
    garch_ic="aic",
):
    rng = np.random.default_rng(random_state)
    X_hist = [np.array([int(round(x[0])), int(round(x[1])), float(x[2])], dtype=float) for x in init_X]
    F_hist = [np.array(f, dtype=float) for f in init_F]
    rows = []

    # utility to eval and record one point
    def eval_and_log(x, it):
        f = objective(x)
        p = int(np.clip(np.rint(x[0]), 1, 7))
        q = int(np.clip(np.rint(x[1]), 1, 7))
        thr = float(x[2])
        meta = getattr(objective, "stats_map", {}).get((p, q, round(thr,6)), {})
        rows.append({
            "stage": "BO",
            "seed": int(random_state),
            "gen": -1,
            "iter": int(it),
            "fold_id": int(fold_id),
            "retrain_interval": int(interval),
            "p": p, "q": q, "threshold": thr,
            "sharpe": float(-(f[0])),
            "mdd": float(f[1]),
            "turnover": float(meta.get("turnover", np.nan)),
            "penalty": float(meta.get("penalty", np.nan)),
            "strategy": "continuous",
            # --- GARCH tracing ---
            "garch_mode": str(garch_mode),
            "garch_ic": str(garch_ic),
            "dist": meta.get("dist_majority", None),
            "dist_last": meta.get("dist_last", None),
            "aic_norm": meta.get("aic_norm", np.nan),
            "aic_t": meta.get("aic_t", np.nan),
        })
        return f

    # BO loop
    lb, ub = np.array(bounds[0], float), np.array(bounds[1], float)
    for it in range(1, n_iter + 1):
        w = float(rng.uniform(0.05, 0.95))
        lam = np.array([w, 1.0 - w], dtype=float)

        F_arr = np.vstack(F_hist)
        y = parego_scalarize(F_arr, lam)  # minimize

        X_arr = np.vstack(X_hist)
        Xs = (X_arr - lb) / (ub - lb + 1e-12)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state)
        gp.fit(Xs, y)

        cand = rng.uniform(lb, ub, size=(n_pool, 3))
        cand[:,0] = np.clip(np.rint(cand[:,0]), 1, 7)
        cand[:,1] = np.clip(np.rint(cand[:,1]), 1, 7)
        cand_unique = np.unique(cand, axis=0)

        if len(X_hist) > 0:
            hist_unique = np.unique(np.vstack(X_hist), axis=0)
            # set difference approx: keep ones not in hist within tolerance
            def not_in_hist(x):
                return not np.any(np.all(np.isclose(hist_unique, x, atol=1e-8), axis=1))
            mask = np.array([not_in_hist(x) for x in cand_unique])
            cand_unique = cand_unique[mask]

        if cand_unique.shape[0] == 0:
            best_idx = int(np.argmin(y))
            center = X_arr[best_idx]
            jitter = rng.normal(scale=[0.5,0.5,0.02], size=(512,3))
            cand_unique = np.clip(center + jitter, lb, ub)
            cand_unique[:,0] = np.clip(np.rint(cand_unique[:,0]), 1, 7)
            cand_unique[:,1] = np.clip(np.rint(cand_unique[:,1]), 1, 7)

        # EI
        Xc = (cand_unique - lb) / (ub - lb + 1e-12)
        mu, std = gp.predict(Xc, return_std=True)
        y_best = float(np.min(y))
        ei = expected_improvement(mu, std, y_best)

        # pick best EI
        idx = int(np.argmax(ei))
        x_next = cand_unique[idx]
        f_next = eval_and_log(x_next, it)

        X_hist.append(x_next)
        F_hist.append(np.array(f_next, dtype=float))

    return rows

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser("Tier-2 ARIMA (NSGA-II + BO/ParEGO) — CONTINUOUS sizing")
    parser.add_argument("--folds-path", default="data/processed_folds/final/arima/arima_tuning_folds_final_paths.json")
    parser.add_argument("--tier1-json",  default="data/tuning_results/jsons/tier1_arima.json")
    parser.add_argument("--tier2-json",  default="data/tuning_results/jsons/tier2_arima_cont_gabo.json")
    parser.add_argument("--tier2-csv",   default="data/tuning_results/csv/tier2_arima_cont_gabo.csv")
    parser.add_argument("--retrain-intervals", type=str, default="10,20,42")
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--ngen", type=int, default=DEFAULT_T2_NGEN_ARIMA)

    # Continuous params
    parser.add_argument("--thr-min", type=float, default=0.01)
    parser.add_argument("--thr-max", type=float, default=0.6)
    parser.add_argument("--pos-clip", type=float, default=1.0)
    parser.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    parser.add_argument("--scale-factor", type=float, default=1000.0)
    parser.add_argument("--arch-rescale-flag", action="store_true")

    # BO params
    parser.add_argument("--bo-iters", type=int, default=40)
    parser.add_argument("--bo-pool", type=int, default=2000)
    parser.add_argument("--bo-kmeans", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=None)

    # GARCH options
    parser.add_argument(
        "--garch-dist", default="auto",
        choices=["auto", "normal", "t"],
        help="Innovations for GARCH(1,1): 'auto' selects by IC per block; otherwise force 'normal' or 't'."
    )
    parser.add_argument(
        "--garch-ic", default="aic",
        choices=["aic", "bic", "hqic"],
        help="Information criterion for --garch-dist=auto (default=aic)."
    )

    parser.add_argument("--strict-paths", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Logging
    ensure_dir(Path("logs"))
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/tier2_arima_cont_gabo.log"), logging.StreamHandler()]
    )

    np.random.seed(42)
    retrain_intervals = [int(x) for x in args.retrain_intervals.split(",") if x.strip()]

    # Ensure output dirs
    ensure_dir(Path(args.tier2_json).parent)
    ensure_dir(Path(args.tier2_csv).parent)
    hv_path = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_hv.csv")
    front_path = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_front.csv")
    knee_path = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_knee.csv")

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
    all_rows = []
    processed = set()
    if Path(args.tier2_csv).exists():
        df_exist = pd.read_csv(args.tier2_csv)
        all_rows = df_exist.to_dict("records")
        processed = set(int(x) for x in df_exist["fold_id"].unique())

    pending = [fid for fid in all_fold_ids if fid in champions and fid not in processed]
    if args.max_folds is not None:
        pending = pending[: int(args.max_folds)]
    
    hv_rows = []  # hypervolume log per gen

    for fid in tqdm(pending, desc="Tier-2 ARIMA (continuous, GA+BO)"):
        info = summary.get(fid)
        if not info:
            continue

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

        params = champions.get(fid)
        if not params:
            continue

        # (p, q) neighbor seeds
        p0 = max(1, int(params["p"]))
        q0 = max(1, int(params["q"]))
        pq_seeds = neighbor_params(p0, q0)

        for interval in retrain_intervals:
            # build objective (pass garch_dist + garch_ic)
            obj = create_continuous_arima_objective(
                train=train_ret,
                val=val_ret,
                retrain_interval=interval,
                cost_per_turnover=BASE_COST + SLIPPAGE,
                min_block_vol=args.min_block_vol,
                scale_factor=args.scale_factor,
                arch_rescale_flag=args.arch_rescale_flag,
                debug=args.debug,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
                pos_clip=args.pos_clip,
                garch_dist=args.garch_dist,
                garch_ic=args.garch_ic,
            )

            # Seeds: random threshold per (p,q)
            rng = np.random.default_rng(42)
            seed_points = [[pp, qq, float(rng.uniform(args.thr_min, args.thr_max))] for (pp, qq) in pq_seeds]

            sampling = FromTier1SeedSampling(seed_points, n_var=3)
            problem = Tier2MOGAProblem(obj, thr_bounds=(args.thr_min, args.thr_max))

            ga_rows = []
            recorder = GARecorder(fid, interval, obj, hv_rows, ga_rows, seed=42,
                                  garch_mode=args.garch_dist, garch_ic=args.garch_ic)

            res = pymoo_minimize(
                problem,
                NSGA2(pop_size=args.pop_size, sampling=sampling),
                ("n_gen", args.ngen),
                seed=42,
                verbose=False,
                callback=recorder
            )

            # --- Final GA Pareto front ---
            if res.F is None or res.X is None or res.F.shape[0] == 0:
                logging.debug(f"[Fold {fid}] NSGA-II returned no solutions.")
                continue

            # Append GA rows & log HV per-gen
            all_rows.extend(dedup_results(ga_rows))
            append_rows_csv(hv_rows, hv_path)
            hv_rows.clear()

            # Pareto's final GA
            F_ga = res.F    # objective values (minimize): [-Sharpe, MDD]
            X_ga = res.X    # decision variables: [p, q, threshold]
            nd_mask = is_nondominated(F_ga)
            F_front = F_ga[nd_mask]
            X_front = X_ga[nd_mask]

            if F_front.size == 0:
                logging.debug(f"[Fold {fid}][int {interval}] GA front empty → skip BO & union/HV")
                continue

            # --- BO seed selection: extreme & knee + cluster centers ---
            idx_f0 = int(np.argmin(F_front[:, 0]))   # min f0 = max Sharpe
            idx_f1 = int(np.argmin(F_front[:, 1]))   # min f1 = min MDD
            idx_knee = knee_index(F_front)
            sel_idx = {idx_f0, idx_f1, idx_knee}

            k = min(args.bo_kmeans, F_front.shape[0])
            if k >= 2:
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                labels = km.fit_predict(F_front)
                for lab in range(k):
                    sub = np.where(labels == lab)[0]
                    if sub.size > 0:
                        j = sub[np.argmin(np.linalg.norm(F_front[sub] - km.cluster_centers_[lab], axis=1))]
                        sel_idx.add(int(j))

            seeds_X = [X_front[i] for i in sorted(sel_idx)]
            seeds_F = [F_front[i] for i in sorted(sel_idx)]

            # --- BO (ParEGO) ---
            bounds = ([1, 1, args.thr_min], [7, 7, args.thr_max])
            bo_rows = run_bo_parego(
                objective=obj,
                bounds=bounds,
                init_X=seeds_X,
                init_F=seeds_F,
                n_iter=args.bo_iters,
                n_pool=args.bo_pool,
                random_state=42,
                fold_id=fid,
                interval=interval,
                garch_mode=args.garch_dist,
                garch_ic=args.garch_ic,
            )
            all_rows.extend(dedup_results(bo_rows))

            F_ga_final = res.F
            X_ga_final = res.X
            nd_ga = is_nondominated(F_ga_final)
            F_ga_front = F_ga_final[nd_ga]
            X_ga_front = X_ga_final[nd_ga]

            if len(bo_rows) > 0:
                F_bo_all = np.array([[-row["sharpe"], row["mdd"]] for row in bo_rows], dtype=float)
                X_bo_all = np.array([[row["p"], row["q"], row["threshold"]] for row in bo_rows], dtype=float)
                nd_bo = is_nondominated(F_bo_all)
                F_bo_front = F_bo_all[nd_bo]
                X_bo_front = X_bo_all[nd_bo]
            else:
                F_bo_front = np.empty((0, 2), dtype=float)
                X_bo_front = np.empty((0, 3), dtype=float)

            F_union = np.vstack([F_ga_front, F_bo_front]) if F_bo_front.size else F_ga_front.copy()
            X_union = np.vstack([X_ga_front, X_bo_front]) if X_bo_front.size else X_ga_front.copy()
            nd_union = is_nondominated(F_union)
            F_union_front = F_union[nd_union]
            X_union_front = X_union[nd_union]

            # ----- Save fronts with dist meta (if available) -----
            front_rows = []
            # GA-only
            for x, f in zip(X_ga_front, F_ga_front):
                p_i, q_i, thr_i = int(round(x[0])), int(round(x[1])), float(x[2])
                meta = getattr(obj, "stats_map", {}).get((p_i, q_i, round(thr_i, 6)), {})
                front_rows.append({
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "front_type": "GA",
                    "p": p_i, "q": q_i,
                    "threshold": thr_i,
                    "sharpe": float(-f[0]),
                    "mdd": float(f[1]),
                    "garch_mode": args.garch_dist,
                    "garch_ic": args.garch_ic,
                    "dist": meta.get("dist_majority", None),
                    "dist_last": meta.get("dist_last", None),
                    "aic_norm": meta.get("aic_norm", np.nan),
                    "aic_t": meta.get("aic_t", np.nan),
                })
            # Union GA+BO
            for x, f in zip(X_union_front, F_union_front):
                p_i, q_i, thr_i = int(round(x[0])), int(round(x[1])), float(x[2])
                meta = getattr(obj, "stats_map", {}).get((p_i, q_i, round(thr_i, 6)), {})
                front_rows.append({
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "front_type": "GA+BO",
                    "p": p_i, "q": q_i,
                    "threshold": thr_i,
                    "sharpe": float(-f[0]),
                    "mdd": float(f[1]),
                    "garch_mode": args.garch_dist,
                    "garch_ic": args.garch_ic,
                    "dist": meta.get("dist_majority", None),
                    "dist_last": meta.get("dist_last", None),
                    "aic_norm": meta.get("aic_norm", np.nan),
                    "aic_t": meta.get("aic_t", np.nan),
                })
            append_rows_csv(front_rows, front_path)

            # ----- Knee pick from union -----
            if F_union_front.size:
                k_idx = knee_index(F_union_front)
                p_k, q_k, thr_k = int(round(X_union_front[k_idx, 0])), int(round(X_union_front[k_idx, 1])), float(X_union_front[k_idx, 2])
                k_meta = getattr(obj, "stats_map", {}).get((p_k, q_k, round(thr_k, 6)), {})
                knee = {
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "p": p_k,
                    "q": q_k,
                    "threshold": thr_k,
                    "sharpe": float(-F_union_front[k_idx, 0]),
                    "mdd": float(F_union_front[k_idx, 1]),
                    "pick": "knee_union",
                    "garch_mode": args.garch_dist,
                    "garch_ic": args.garch_ic,
                    "dist": k_meta.get("dist_majority", None),
                    "dist_last": k_meta.get("dist_last", None),
                    "aic_norm": k_meta.get("aic_norm", np.nan),
                    "aic_t": k_meta.get("aic_t", np.nan),
                }
                append_rows_csv([knee], knee_path)

            # ----- Hypervolume: GA-only vs GA∪BO (used with ref-point) -----
            ref0 = float(F_ga_final[:, 0].max()) * 1.1 + 1e-6
            ref1 = float(F_ga_final[:, 1].max()) * 1.1 + 1e-6
            ref_point = np.array([ref0, ref1], dtype=float)
            hv_metric = HV(ref_point=ref_point)

            hv_final_ga = float(hv_metric(F_ga_front)) if F_ga_front.size else np.nan
            hv_final_union = float(hv_metric(F_union_front)) if F_union_front.size else np.nan

            hv_rows = [
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
            append_rows_csv(hv_rows, hv_path)

            # ---- Save all GA/BO rows (dedup) ----
            all_rows = dedup_results(all_rows)
            pd.DataFrame(all_rows).to_csv(args.tier2_csv, index=False)
            with open(args.tier2_json, "w") as f:
                json.dump(all_rows, f, indent=2)

    logging.info("=== Tier-2 ARIMA (continuous, GA+BO) complete ===")

if __name__ == '__main__':
    main()