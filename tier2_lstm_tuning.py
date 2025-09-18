import os
import json
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV

# ---- Project bases ----
PROCESSED_BASE = Path("data/processed_folds")

# ---- LSTM utils (project-local) ----
from lstm_utils import (   # noqa: F401
    FromTier1SeedSampling,
    Tier2MOGAProblem,                 # decision: [window, units, lr, epochs, rel_thresh] -> (f0=-Sharpe, f1=MDD)
    create_periodic_lstm_objective,   # builds objective with backbone (layers,batch,dropout)
    is_nondominated,
    knee_index,
    BASE_COST,
    SLIPPAGE,
)

# ================= Helpers =================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[load_json] File not found: {p}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"[load_json] File is empty: {p}")
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        preview = txt[:200].replace("\n", "\\n")
        raise ValueError(f"[load_json] Invalid JSON in {p}: {e}. Preview: {preview!r}") from e

def resolve_csv(raw_path: str, base_hints: List[Path]) -> Path:
    """Try multiple bases to resolve a relative CSV path."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    # Accept single base or list of bases
    if base_hints is None:
        candidates: List[Path] = []
    elif isinstance(base_hints, (str, Path)):
        candidates = [Path(base_hints)]
    else:
        candidates = [Path(b) for b in base_hints if b is not None]

    for b in candidates:
        cand = (b / raw_path)
        if cand.exists():
            return cand

    cand2 = (PROCESSED_BASE / raw_path)
    if cand2.exists():
        return cand2
    return p

def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def append_rows_csv(rows: list[dict], path: Path):
    """Append rows to CSV (creates dir; header only if new)."""
    if not rows:
        return
    ensure_dir(path.parent)
    write_header = not path.exists()
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)

def ensure_targets_solution_A(df: pd.DataFrame):
    """
    Ensure columns for Solution A:
      - 'target' = Log_Returns.shift(-1)       (supervised label)
      - 'target_log_returns' = Log_Returns.shift(-1)  (PnL/MDD metric)
    Fills NaNs (tail) with 0.0.
    """
    if "Log_Returns" not in df.columns:
        raise ValueError("Missing required column 'Log_Returns' in input CSV.")
    if "target" not in df.columns:
        df["target"] = df["Log_Returns"].shift(-1)
    if "target_log_returns" not in df.columns:
        df["target_log_returns"] = df["Log_Returns"].shift(-1)
    df["target"] = df["target"].fillna(0.0)
    df["target_log_returns"] = df["target_log_returns"].fillna(0.0)

# ================= HV per-gen callback =================
class GARecorder(Callback):
    """
    Record hypervolume on the ND front each generation (GA stage).
    Stores rows in self.hv_rows; caller will append to CSV after a run.
    """
    def __init__(self, fold_id: int, interval: int):
        super().__init__()
        self.fold_id = int(fold_id)
        self.interval = int(interval)
        self.hv_rows: list[dict] = []
        self._ref = None  # stable ref point after first gen

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        if F is None or F.size == 0:
            return
        # ND front for HV logging
        nd = is_nondominated(F)
        F_nd = F[nd]
        if F_nd.size == 0:
            return
        gen = int(algorithm.n_gen)
        if self._ref is None:
            self._ref = np.array([float(F[:, 0].max()) * 1.1 + 1e-6,
                                  float(F[:, 1].max()) * 1.1 + 1e-6], dtype=float)
        hv = HV(ref_point=self._ref)(F_nd)
        self.hv_rows.append({
            "fold_id": self.fold_id,
            "retrain_interval": self.interval,
            "stage": "GA",
            "gen": gen,
            "hv": float(hv),
            "ref0": float(self._ref[0]),
            "ref1": float(self._ref[1]),
            "n_front": int(F_nd.shape[0]),
        })

# ================= Seeds =================
def build_seeds_from_defaults(
    base_window=25, base_units=48, base_lr=1e-3, base_epochs=30, rel_list=(0.50, 0.60, 0.70)
):
    """
    Small, diverse lattice around neutral defaults to kick-start NSGA-II.
    """
    w_mult = [0.8, 1.0, 1.2]
    u_mult = [0.75, 1.0, 1.25]
    lr_mult = [0.5, 1.0, 2.0]
    ep_add = [-10, 0, +10]
    seeds = []
    for wm in w_mult:
        for um in u_mult:
            for lm in lr_mult:
                for ea in ep_add:
                    for r in rel_list:
                        w = int(np.clip(round(base_window * wm), 10, 40))
                        u = int(np.clip(round(base_units * um), 32, 128))
                        lr = float(np.clip(base_lr * lm, 1e-4, 1e-2))
                        ep = int(np.clip(base_epochs + ea, 10, 30))
                        seeds.append([w, u, lr, ep, float(r)])
    uniq = np.unique(np.array(seeds, dtype=float), axis=0)
    return uniq.tolist()

# ================= Main =================
def main():
    ap = argparse.ArgumentParser("Tier-2 LSTM: NSGA-II (+optional ParEGO) on [window, units, lr, epochs, rel_thresh]")
    # inputs
    ap.add_argument("--folds-path", required=True,
                    help="JSON list of folds with train/val paths (same structure as Tier-1 LSTM).")
    ap.add_argument("--tier1-backbone-json", required=True,
                    help="Tier-1 backbone champions JSON (fold_id -> champion: layers,batch_size,dropout).")
    ap.add_argument("--feature-path", default="data/processed_folds/lstm_feature_columns.json",
                    help="Optional JSON list of feature names; otherwise prefer PCA* columns or infer.")
    ap.add_argument("--retrain-intervals", default="10,20,42")
    ap.add_argument("--max-folds", type=int, default=0,
                    help="If >0, process only first N folds (useful for debugging).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip (fold_id, interval) pairs that already exist in tier2_csv.")
    # search
    ap.add_argument("--pop-size", type=int, default=24)
    ap.add_argument("--ngen", type=int, default=40)
    ap.add_argument("--bo-iters", type=int, default=0, help="Optional ParEGO iterations (0 = skip BO).")
    # outputs 
    ap.add_argument("--tier2-json", default="data/tuning_results/jsons/tier2_lstm.json")
    ap.add_argument("--tier2-csv",  default="data/tuning_results/csv/tier2_lstm.csv")
    # seed & misc
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # derived output paths (append-only)
    ensure_dir(Path(args.tier2_json).parent)
    ensure_dir(Path(args.tier2_csv).parent)
    hv_path    = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_hv.csv")
    front_path = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_front.csv")
    knee_path  = Path(args.tier2_csv).with_name(Path(args.tier2_csv).stem + "_knee.csv")

    # ---- Load folds & champions ----
    folds_path = Path(args.folds_path).resolve()
    folds_raw = load_json(folds_path)
    if not isinstance(folds_raw, list):
        raise ValueError("--folds-path must be a JSON list of fold records.")
    # path hints
    base_hints = [
        folds_path.parent,
        folds_path.parents[1] if len(folds_path.parents) > 1 else None,
        folds_path.parents[2] if len(folds_path.parents) > 2 else None,
        PROCESSED_BASE,
        Path.cwd(),
    ]

    champs_raw = load_json(Path(args.tier1_backbone_json).resolve())
    if isinstance(champs_raw, dict) and "results" in champs_raw:
        champs_raw = champs_raw["results"]
    champions = {int(r["fold_id"]): r["champion"] for r in champs_raw}

    # optional features
    feat_list = None
    feat_json = Path(args.feature_path) if args.feature_path else None
    if feat_json and feat_json.exists():
        try:
            fl = load_json(feat_json)
            feat_list = fl if isinstance(fl, list) else None
        except Exception:
            feat_list = None

    intervals = [int(x) for x in str(args.retrain_intervals).split(",") if x.strip()]

    # ---- Resume if exists (main table) ----
    existing_rows = []
    processed_keys = set()  # set of (fold_id, retrain_interval)
    if Path(args.tier2_csv).exists():
        try:
            df_exist = pd.read_csv(args.tier2_csv)
            existing_rows = df_exist.to_dict("records")
            if {"fold_id", "retrain_interval"}.issubset(df_exist.columns):
                processed_keys = set(
                    zip(df_exist["fold_id"].astype(int), df_exist["retrain_interval"].astype(int))
                )
        except Exception:
            pass

    # ---- Iterate folds ----
    folds_count = 0
    for rec in folds_raw:
        fid = int(rec.get("global_fold_id", rec.get("fold_id")))

        # Ensure fold is in Tier-1 champions (avoid KeyError)
        if fid not in champions:
            logging.info("Skip fold %s — not in Tier-1 backbone champions.", fid)
            continue

        train_rel = rec.get("final_train_path") or rec.get("train_path_lstm") or rec.get("train_path")
        val_rel   = rec.get("final_val_path")   or rec.get("val_path_lstm")   or rec.get("val_path")
        if not train_rel or not val_rel:
            logging.info("Skip fold %s — missing train/val paths.", fid)
            continue

        train_csv = resolve_csv(train_rel, base_hints)
        val_csv   = resolve_csv(val_rel,   base_hints)
        if not train_csv.exists() or not val_csv.exists():
            logging.info(
                "Skip fold %s — CSV not found.\n"
                "  train_rel=%s\n  val_rel=%s\n"
                "  train_abs=%s (exists=%s)\n  val_abs=%s (exists=%s)",
                fid, train_rel, val_rel, train_csv, train_csv.exists(), val_csv, val_csv.exists()
            )
            continue

        # Load & ensure Solution A targets
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        try:
            ensure_targets_solution_A(train_df)
            ensure_targets_solution_A(val_df)
        except Exception as e:
            logging.info("Skip fold %s — target creation failed: %s", fid, e)
            continue

        # Features: prefer provided list; else PCs; else let objective infer (pass [])
        if feat_list:
            features = [c for c in feat_list if c in train_df.columns]
        else:
            features = [c for c in train_df.columns if c.startswith("PC")]

        champ = champions[fid]  # {'layers','batch_size','dropout'}

        produced_any = False  # to count only folds that actually run at least one interval

        for interval in intervals:
            if args.skip_existing and (fid, int(interval)) in processed_keys:
                logging.info("Skip fold %s interval %s — already processed.", fid, interval)
                continue

            logging.info("Fold %s | interval=%s", fid, interval)

            # ---- Build objective
            obj = create_periodic_lstm_objective(
                train_df=train_df,
                val_df=val_df,
                feature_cols=features if features else [],
                champ=champ,
                retrain_interval=int(interval),
                cost_per_turnover=(BASE_COST + SLIPPAGE),
                min_block_vol=0.0015,
                metric_col="target_log_returns",
                debug=False
            )

            # ---- Problem
            problem = Tier2MOGAProblem(obj)   # n_var=5, n_obj=2

            # ---- Seeds for NSGA-II
            seeds = build_seeds_from_defaults(
                base_window=25,
                base_units=48,
                base_lr=1e-3,
                base_epochs=30,
                rel_list=(0.50, 0.60, 0.70)
            )
            sampling = FromTier1SeedSampling(seeds, n_var=5, int_indices=[0, 1, 3])  # window, units, epochs are ints

            # ---- GA run (+HV callback)
            callback = GARecorder(fid, interval)
            algo = NSGA2(pop_size=int(args.pop_size), sampling=sampling, eliminate_duplicates=True)

            res = pymoo_minimize(
                problem,
                algo,
                ('n_gen', int(args.ngen)),
                seed=int(args.seed),
                callback=callback,
                verbose=False
            )

            # Flush GA HV per-gen (append mode)
            append_rows_csv(callback.hv_rows, hv_path)

            if res.F is None or res.X is None or res.F.shape[0] == 0:
                logging.info("No solutions for fold %s interval %s.", fid, interval)
                continue

            # ---- GA final front
            F_ga = res.F
            X_ga = res.X
            nd = is_nondominated(F_ga)
            F_ga_front = F_ga[nd]
            X_ga_front = X_ga[nd]

            # ---- Optional ParEGO BO refinement (union front)
            F_union_front, X_union_front = F_ga_front, X_ga_front
            if int(args.bo_iters) > 0:
                try:
                    from lstm_utils import run_bo_parego
                    lb = np.array([10, 32, 1e-4, 10, 0.5], dtype=float)
                    ub = np.array([40, 128, 1e-2, 30, 0.95], dtype=float)
                    bo_rows = run_bo_parego(
                        objective=obj,
                        bounds=(lb, ub),
                        init_X=[x for x in X_ga_front],
                        init_F=[f for f in F_ga_front],
                        n_iter=int(args.bo_iters),
                        n_pool=2000,
                        random_state=int(args.seed),
                        int_indices=[0,1,3],
                        round_digits={2:6, 4:3},
                        eval_logger=None
                    )
                    if bo_rows:
                        # Convert to arrays; keep finite
                        F_bo = np.array([[r["f0"], r["f1"]] for r in bo_rows], float)
                        X_bo = np.array([r["x"] for r in bo_rows], float)
                        mask = np.isfinite(F_bo).all(axis=1)
                        F_bo, X_bo = F_bo[mask], X_bo[mask]
                        if F_bo.size:
                            F_union = np.vstack([F_ga_front, F_bo])
                            X_union = np.vstack([X_ga_front, X_bo])
                            nd_u = is_nondominated(F_union)
                            F_union_front = F_union[nd_u]
                            X_union_front = X_union[nd_u]
                except Exception as e:
                    logging.info("BO skipped (reason: %s)", e)

            # ---- Save union & GA fronts with turnover/penalty meta
            front_rows = []
            stats_map = getattr(obj, "stats_map", {})
            # GA only
            for x, f in zip(X_ga_front, F_ga_front):
                meta = stats_map.get(
                    (int(round(x[0])), int(round(x[1])), round(float(x[2]), 6), int(round(x[3])), round(float(x[4]), 6)),
                    {}
                )
                front_rows.append({
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "front_type": "GA",
                    "window": int(round(x[0])), "units": int(round(x[1])),
                    "lr": float(x[2]), "epochs": int(round(x[3])), "rel_thresh": float(x[4]),
                    "sharpe": float(-f[0]), "mdd": float(f[1]),
                    "turnover": float(meta.get("turnover", np.nan)),
                    "penalty": float(meta.get("penalty", np.nan)),
                })
            # Union (GA+BO)
            for x, f in zip(X_union_front, F_union_front):
                meta = stats_map.get(
                    (int(round(x[0])), int(round(x[1])), round(float(x[2]), 6), int(round(x[3])), round(float(x[4]), 6)),
                    {}
                )
                front_rows.append({
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "front_type": "GA+BO",
                    "window": int(round(x[0])), "units": int(round(x[1])),
                    "lr": float(x[2]), "epochs": int(round(x[3])), "rel_thresh": float(x[4]),
                    "sharpe": float(-f[0]), "mdd": float(f[1]),
                    "turnover": float(meta.get("turnover", np.nan)),
                    "penalty": float(meta.get("penalty", np.nan)),
                })
            append_rows_csv(front_rows, front_path)

            # ---- Knee pick (on union)
            kidx = knee_index(F_union_front)
            xk, fk = X_union_front[kidx], F_union_front[kidx]
            meta_k = stats_map.get(
                (int(round(xk[0])), int(round(xk[1])), round(float(xk[2]), 6), int(round(xk[3])), round(float(xk[4]), 6)),
                {}
            )
            knee_row = {
                "fold_id": int(fid),
                "retrain_interval": int(interval),
                "window": int(round(xk[0])),
                "units": int(round(xk[1])),
                "lr": float(xk[2]),
                "epochs": int(round(xk[3])),
                "rel_thresh": float(xk[4]),
                "f0": float(fk[0]),
                "f1": float(fk[1]),
                "sharpe": float(-fk[0]),
                "mdd": float(fk[1]),
                "turnover": float(meta_k.get("turnover", np.nan)),
                "penalty": float(meta_k.get("penalty", np.nan)),
                "pick": "knee_union"
            }
            append_rows_csv([knee_row], knee_path)

            # ---- Final HV (GA vs Union)
            ref0 = float(F_ga[:, 0].max()) * 1.1 + 1e-6
            ref1 = float(F_ga[:, 1].max()) * 1.1 + 1e-6
            hv_metric = HV(ref_point=np.array([ref0, ref1], float))
            hv_rows = [
                {
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "stage": "final_ga",
                    "gen": -1,
                    "hv": float(hv_metric(F_ga_front)),
                    "ref0": ref0,
                    "ref1": ref1,
                    "n_front": int(F_ga_front.shape[0]),
                },
                {
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "stage": "final_union",
                    "gen": -1,
                    "hv": float(hv_metric(F_union_front)),
                    "ref0": ref0,
                    "ref1": ref1,
                    "n_front": int(F_union_front.shape[0]),
                },
            ]
            append_rows_csv(hv_rows, hv_path)

            # ---- Flat main (append-only): knee summary per fold×interval
            main_row = {
                "fold_id": int(fid),
                "retrain_interval": int(interval),
                "layers": int(champ["layers"]),
                "batch_size": int(champ["batch_size"]),
                "dropout": float(champ["dropout"]),
                "knee_window": int(round(xk[0])),
                "knee_units": int(round(xk[1])),
                "knee_lr": float(xk[2]),
                "knee_epochs": int(round(xk[3])),
                "knee_rel_thresh": float(xk[4]),
                "knee_sharpe": float(-fk[0]),
                "knee_mdd": float(fk[1]),
                "knee_turnover": float(meta_k.get("turnover", np.nan)),
                "knee_penalty": float(meta_k.get("penalty", np.nan)),
            }
            append_rows_csv([main_row], Path(args.tier2_csv))

            # ---- Update JSON checkpoint (append semantics)
            try:
                if Path(args.tier2_json).exists():
                    obj = load_json(Path(args.tier2_json))
                    existing = obj["results"] if isinstance(obj, dict) and "results" in obj else obj
                else:
                    existing = []
                existing.append(main_row)
                with open(args.tier2_json, "w") as f:
                    json.dump(existing, f, indent=2)
            except Exception:
                # best-effort; don't crash tuning
                pass

            produced_any = True  # at least one interval produced output for this fold

        # ---- max-folds logic AFTER finishing a fold ----
        if produced_any:
            folds_count += 1
            if args.max_folds > 0 and folds_count >= args.max_folds:
                break

    logging.info(
        "Done. Outputs:\n  %s\n  %s\n  %s\n  %s",
        args.tier2_csv, front_path, knee_path, hv_path
    )

if __name__ == "__main__":
    main()