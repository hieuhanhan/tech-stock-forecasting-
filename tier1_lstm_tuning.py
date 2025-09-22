#!/usr/bin/env python3
import json
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

from skopt.space import Integer, Real, Space
from skopt import gp_minimize
from skopt.utils import check_x_in_space  # 

# ---- Import your utils (new strategy) ----
from lstm_utils import (
    Tier1LSTMBackboneProblem,   # varies only (layers, batch_size, dropout)
    NON_FEATURE_KEEP,
)

# ----------------- Defaults -----------------
PROCESSED_BASE = Path("data/processed_folds")
LOG_DIR = Path("logs/tier1_lstm_backbone")

# ----------------- Helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def resolve_csv(raw_path: str, base_hint: Path | None) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if base_hint is not None:
        cand = (base_hint / raw_path)
        if cand.exists():
            return cand
    cand2 = (PROCESSED_BASE / raw_path)
    return cand2 if cand2.exists() else p

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_tf_perf():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            tf.config.optimizer.set_jit(True)
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def load_features(feature_path: Path | None, df_train: pd.DataFrame) -> list:
    # Priority 1: PCA columns if exist
    pc_cols = [c for c in df_train.columns if c.startswith("PC")]
    if pc_cols:
        return sorted(pc_cols, key=lambda x: int(x[2:]))
    # Priority 2: feature JSON
    if feature_path and feature_path.exists():
        feats = [c for c in load_json(feature_path) if c in df_train.columns]
        if feats:
            return feats
    # Fallback: infer
    return [c for c in df_train.columns if c not in NON_FEATURE_KEEP]

def read_resume(json_path: Path):
    if not json_path.exists():
        return [], set()
    obj = load_json(json_path)
    results = obj["results"] if isinstance(obj, dict) and "results" in obj else obj
    done = {int(r["fold_id"]) for r in results if "fold_id" in r}
    return results, done

def write_results(json_path: Path, csv_path: Path, results: list):
    ensure_dir(json_path.parent)
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    ensure_dir(csv_path.parent)
    pd.DataFrame(results).to_csv(csv_path, index=False)

def coerce_fold_record(rec: dict) -> dict:
    fid = rec.get("global_fold_id", rec.get("fold_id", None))
    if fid is None:
        raise ValueError("Fold record missing 'global_fold_id' or 'fold_id'.")
    train = rec.get("final_train_path") or rec.get("train_path_lstm") or rec.get("train_path")
    val   = rec.get("final_val_path")   or rec.get("val_path_lstm")   or rec.get("val_path")
    if not train or not val:
        raise ValueError(f"Fold {fid}: missing train/val path fields.")
    return dict(
        fold_id=int(fid),
        train_path=str(train),
        val_path=str(val),
        ticker=rec.get("ticker", None)
    )

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser("Tier-1 LSTM BACKBONE (GA→BO): search layers, batch_size, dropout only")
    ap.add_argument('--folds-path', required=True,
                    help='JSON list of folds, each with train/val CSV paths.')
    ap.add_argument('--feature-path', default='data/processed_folds/lstm_feature_columns.json',
                    help='Optional JSON list of feature names.')
    ap.add_argument('--output-json', default='data/tuning_results/jsons/tier1_lstm_backbone.json')
    ap.add_argument('--output-csv',  default='data/tuning_results/csv/tier1_lstm_backbone.csv')
    ap.add_argument('--max-folds', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)

    # Fixed neutral knobs used INSIDE the problem (Tier-1 only)
    ap.add_argument('--window-t1', type=int, default=25)
    ap.add_argument('--units-t1',  type=int, default=48)
    ap.add_argument('--lr-t1',     type=float, default=1e-3)

    # Objective/training knobs
    ap.add_argument('--t1-epochs', type=int, default=10, help='Epochs per RMSE evaluation.')
    ap.add_argument('--t1-patience', type=int, default=3, help='EarlyStopping patience.')

    # Search knobs
    ap.add_argument('--ga-pop-size', type=int, default=20, help='GA population size.')
    ap.add_argument('--ga-n-gen',    type=int, default=15, help='GA generations.')
    ap.add_argument('--bo-n-calls',  type=int, default=40, help='skopt BO calls.')
    ap.add_argument('--top-n-seeds', type=int, default=10, help='Top GA individuals to warm-start BO.')
    args = ap.parse_args()

    ensure_dir(LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "tier1_lstm_backbone.log"), logging.StreamHandler()],
    )

    set_all_seeds(args.seed)
    setup_tf_perf()

    ensure_dir(Path(args.output_json).parent)
    ensure_dir(Path(args.output_csv).parent)

    # ---- Load folds ----
    folds_path = Path(args.folds_path).resolve()
    folds_raw = load_json(folds_path)
    if not isinstance(folds_raw, list):
        raise ValueError("--folds-path must be a JSON list of fold records.")
    base_hint = folds_path.parent

    folds = []
    for rec in folds_raw:
        try:
            folds.append(coerce_fold_record(rec))
        except Exception as e:
            logging.warning("Skip a fold record: %s", e)

    if args.max_folds:
        folds = folds[:args.max_folds]

    # ---- Resume ----
    results, done = read_resume(Path(args.output_json))
    pending = [f for f in folds if int(f["fold_id"]) not in done]

    logging.info("Total folds: %d | pending: %d", len(folds), len(pending))

    # ---- Search space (ONLY 3 vars) ----
    bo_dims = [
        Integer(1, 3,     name="layers"),
        Integer(16, 128,  name="batch_size"),
        Real(0.0, 0.5,    name="dropout"),
    ]
    space = Space(bo_dims)

    # Helpers to sanitize BO seeds (use skopt's validator)
    def sanitize_bo_seeds(x_list, y_list, logger=logging):
        """
        - Cast to correct python types (int/int/float)
        - Clip to bounds
        - Drop duplicates
        - Validate with check_x_in_space
        - Keep x/y aligned
        """
        clean_x, clean_y, seen = [], [], set()
        bad = 0
        for x, y in zip(x_list, y_list):
            xx = []
            for v, dim in zip(x, space.dimensions):
                lo, hi = dim.low, dim.high
                if isinstance(dim, Integer):
                    vv = int(round(float(v)))
                else:
                    vv = float(v)
                vv = min(max(vv, lo), hi)  # clip
                xx.append(vv)
            try:
                check_x_in_space(xx, space)  # raises ValueError if out-of-bounds
            except ValueError:
                bad += 1
                continue
            key = tuple(xx)
            if key in seen:
                continue
            seen.add(key)
            clean_x.append(xx)
            clean_y.append(float(y))
        if bad and logger:
            logger.debug(f"[BO seeds] dropped {bad} out-of-bounds warm-start points.")
        return clean_x, clean_y

    for item in tqdm(pending, desc="Tier-1 LSTM Backbone"):
        fid = int(item["fold_id"])
        ticker = item.get("ticker")

        train_csv = resolve_csv(item["train_path"], base_hint)
        val_csv   = resolve_csv(item["val_path"],   base_hint)
        if not train_csv.exists() or not val_csv.exists():
            logging.warning("Fold %s CSV missing: %s | %s. Skipping.", fid, train_csv, val_csv)
            continue

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)

        # Features
        try:
            feature_cols = load_features(Path(args.feature_path), train_df)
        except Exception as e:
            logging.warning("Feature load failed (%s). Falling back to inference.", e)
            feature_cols = load_features(None, train_df)

        # ---- Problem: uses FIXED (window_t1, units_t1, lr_t1) internally ----
        problem = Tier1LSTMBackboneProblem(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            window_t1=int(args.window_t1),
            units_t1=int(args.units_t1),
            lr_t1=float(args.lr_t1),
            t1_epochs=int(args.t1_epochs),
            patience=int(args.t1_patience),
        )

        # ---- GA (light) to get diverse warm seeds ----
        ga = GA(pop_size=args.ga_pop_size, eliminate_duplicates=True, save_history=True)
        res_ga = pymoo_minimize(
            problem,
            ga,
            ('n_gen', args.ga_n_gen),
            verbose=False,
            return_algorithm=True,
            seed=args.seed
        )

        # Collect unique GA individuals for BO warm-start
        individuals = []
        try:
            for h in res_ga.algorithm.history:
                if hasattr(h, "pop") and h.pop is not None:
                    individuals.extend(h.pop)
        except Exception:
            pass
        if not individuals:
            individuals = list(getattr(res_ga, "pop", []))

        seen = set()
        uniq = []
        for ind in individuals:
            x = np.array(ind.X, dtype=float)  # [layers, batch_size, dropout]
            f = float(ind.F[0])
            key = tuple(np.round(x, 6).tolist())
            if key not in seen:
                seen.add(key)
                uniq.append((x, f))
        uniq.sort(key=lambda t: t[1])
        top_pairs = uniq[:args.top_n_seeds]

        # If GA found nothing, seed a couple of reasonable defaults
        if not top_pairs:
            top_pairs = [
                (np.array([1, 32, 0.1], dtype=float), 0.0),
                (np.array([2, 64, 0.2], dtype=float), 0.0),
            ]

        x0_raw = [t[0].tolist() for t in top_pairs]
        y0_raw = [t[1]           for t in top_pairs]

        # Sanitize BO seeds
        x0, y0 = sanitize_bo_seeds(x0_raw, y0_raw)

        # Fallback: center of the box
        if not x0:
            center = [(d.low + d.high) / 2.0 for d in space.dimensions]
            # cast center properly and evaluate to get y0
            center_cast = []
            for c, dim in zip(center, space.dimensions):
                center_cast.append(int(round(c)) if isinstance(dim, Integer) else float(c))
            try:
                check_x_in_space(center_cast, space)
            except ValueError:
                # last resort: use strict mids inside bounds
                center_cast = [dim.low if isinstance(dim, Integer) else float(dim.low) for dim in space.dimensions]
            tmp = {}
            problem._evaluate(center_cast, tmp)
            x0 = [center_cast]
            y0 = [float(tmp["F"][0])]

        # ---- BO (skopt) on the same 3 vars ----
        def bo_objective(x):
            tmp = {}
            problem._evaluate(x, tmp)
            return float(tmp["F"][0])  # RMSE

        res_bo = gp_minimize(
            bo_objective,
            dimensions=bo_dims,
            n_calls=args.bo_n_calls,
            x0=x0, y0=y0,
            random_state=args.seed
        )

        best = res_bo.x  # [layers, batch_size, dropout]
        best_rmse = float(res_bo.fun)

        logging.info(
            "Fold %s (%s) | best: layers=%d, batch=%d, dropout=%.3f | RMSE=%.6f | (win_t1=%d, units_t1=%d, lr_t1=%.1e)",
            fid, (ticker or "n/a"),
            int(best[0]), int(best[1]), float(best[2]),
            best_rmse, int(args.window_t1), int(args.units_t1), float(args.lr_t1)
        )

        results.append({
            "fold_id": int(fid),
            "ticker":  ticker,
            "champion": {
                "layers":     int(best[0]),
                "batch_size": int(best[1]),
                "dropout":    float(best[2]),
            },
            "rmse": float(best_rmse),
            "features_used": feature_cols,
            "fixed_neutral": {
                "window_t1": int(args.window_t1),
                "units_t1":  int(args.units_t1),
                "lr_t1":     float(args.lr_t1),
                "epochs":    int(args.t1_epochs),
                "patience":  int(args.t1_patience),
            },
            "search_space": {
                "layers": [1, 3],
                "batch_size": [16, 128],
                "dropout": [0.0, 0.5],
            },
        })

        # checkpoint after each fold
        write_results(Path(args.output_json), Path(args.output_csv), results)

    logging.info("=== Tier-1 LSTM BACKBONE complete. Saved to %s / %s ===", args.output_json, args.output_csv)

if __name__ == "__main__":
    main()
