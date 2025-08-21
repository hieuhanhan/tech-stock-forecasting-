#!/usr/bin/env python3
# tier1_lstm_tuning.py
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
from skopt.space import Integer, Real
from skopt import gp_minimize

from lstm_utils import (
    Tier1LSTMProblem,   # minimize RMSE on validation windows
    infer_feature_cols,
    NON_FEATURE_KEEP,
)

# ----------------- Defaults -----------------
DEFAULT_BASE_DIR = Path("data/processed_folds")
LOG_DIR = Path("logs/tier1_lstm")

# Tier-1 search space for skopt: [window, layers, units, lr, batch_size]
LSTM_SEARCH_SPACE = [
    Integer(10, 40,    name="window"),
    Integer(1, 3,      name="layers"),
    Integer(32, 128,   name="units"),
    Real(1e-4, 1e-2,   name="lr", prior="log-uniform"),
    Integer(16, 128,   name="batch_size"),
]

# ----------------- Helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def resolve_csv(base_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (base_dir / rel_or_abs)

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_tf_perf(enable_mixed_precision: bool = True, enable_xla: bool = True):
    """Enable mixed precision + XLA on GPU and memory growth when available."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            if enable_mixed_precision:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
            if enable_xla:
                tf.config.optimizer.set_jit(True)  # XLA
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def load_features(feature_path: Path, df_train: pd.DataFrame) -> list:
    """Prefer feature list from JSON; otherwise infer numeric features (exclude NON_FEATURE_KEEP)."""
    if feature_path and feature_path.exists():
        feats = load_json(feature_path)
        assert isinstance(feats, list) and all(isinstance(c, str) for c in feats), \
            "feature JSON must be a list of strings"
        feats = [c for c in feats if c in df_train.columns]
        if not feats:
            raise ValueError("No valid features from feature JSON exist in TRAIN columns.")
        return feats

    feats = infer_feature_cols(df_train)
    if not feats:
        cands = [c for c in df_train.columns if c not in NON_FEATURE_KEEP]
        feats = [c for c in cands if pd.api.types.is_numeric_dtype(df_train[c])]
    if not feats:
        raise ValueError("Could not infer any valid numeric features.")
    return feats

def read_resume(json_path: Path):
    """Return (results_list, done_set) to resume cleanly."""
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

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser("Tier-1 LSTM Tuning (paths come directly from lstm_tuning_folds.json)")
    parser.add_argument('--folds-path', default='data/processed_folds/final/lstm/lstm_tuning_folds.json',
                        help='JSON list of {fold_id, train_path, val_path, ticker?}.')
    parser.add_argument('--feature-path', default='data/processed_folds/lstm_feature_columns.json',
                        help='Optional JSON list of features to use; if missing, infer from TRAIN.')
    parser.add_argument('--base-dir', default=str(DEFAULT_BASE_DIR),
                        help='Base directory to resolve relative train/val CSV paths.')
    parser.add_argument('--output-json', default='data/tuning_results/jsons/tier1_lstm.json')
    parser.add_argument('--output-csv',  default='data/tuning_results/csv/tier1_lstm.csv')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    # Per-evaluation knobs
    parser.add_argument('--t1-epochs', type=int, default=10, help='Epochs per RMSE evaluation.')
    parser.add_argument('--t1-patience', type=int, default=3, help='EarlyStopping patience.')

    # Search knobs (flexible)
    parser.add_argument('--ga-pop-size', type=int, default=24, help='Population size for GA.')
    parser.add_argument('--ga-n-gen', type=int, default=25, help='Number of GA generations.')
    parser.add_argument('--bo-n-calls', type=int, default=40, help='Number of BO (skopt) calls.')
    parser.add_argument('--top-n-seeds', type=int, default=12, help='Top GA individuals to warm-start BO.')

    # Performance toggles
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision on GPU.')
    parser.add_argument('--no-xla', action='store_true', help='Disable XLA JIT on GPU.')

    args = parser.parse_args()

    ensure_dir(LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "tier1_lstm.log"),
            logging.StreamHandler()
        ],
    )

    set_all_seeds(args.seed)
    setup_tf_perf(enable_mixed_precision=not args.no_mixed_precision,
                  enable_xla=not args.no_xla)

    ensure_dir(Path(args.output_json).parent)
    ensure_dir(Path(args.output_csv).parent)

    base_dir = Path(args.base_dir)

    # ---- Load folds file (direct paths) ----
    folds = load_json(Path(args.folds_path))
    if not isinstance(folds, list):
        raise ValueError("folds-path must be a JSON list of fold records.")

    if args.max_folds:
        folds = folds[:args.max_folds]

    # ---- Resume support ----
    results, done = read_resume(Path(args.output_json))
    pending = [rec for rec in folds if int(rec.get("fold_id", -1)) not in done]

    logging.info("Total folds: %d | pending: %d", len(folds), len(pending))

    for rec in tqdm(pending, desc="Tier-1 LSTM"):
        fid = int(rec.get("fold_id", -1))
        if fid < 0:
            logging.warning("Bad record (missing fold_id). Skipping: %s", rec)
            continue

        tr_rel = rec.get("train_path")
        va_rel = rec.get("val_path")
        if not tr_rel or not va_rel:
            logging.warning("Fold %s missing train_path/val_path. Skipping.", fid)
            continue

        train_csv = resolve_csv(base_dir, tr_rel)
        val_csv   = resolve_csv(base_dir, va_rel)
        if not train_csv.exists() or not val_csv.exists():
            logging.warning("Fold %s CSV not found: %s | %s. Skipping.", fid, train_csv, val_csv)
            continue

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)

        # Resolve features
        feat_path = Path(args.feature_path)
        try:
            feature_cols = load_features(feat_path, train_df)
        except Exception as e:
            logging.warning("Feature load failed (%s). Falling back to inference.", e)
            feature_cols = load_features(Path(""), train_df)

        # ---- GA (collect seeds): minimize RMSE on validation ----
        problem = Tier1LSTMProblem(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            t1_epochs=args.t1_epochs,
            patience=args.t1_patience,
        )

        algorithm = GA(
            pop_size=args.ga_pop_size,
            eliminate_duplicates=True,
            save_history=True
        )
        res_ga = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', args.ga_n_gen),
            verbose=False,
            return_algorithm=True
        )

        # Gather individuals from history; fallback to final pop if needed
        individuals = []
        try:
            for h in res_ga.algorithm.history:
                if hasattr(h, "pop") and h.pop is not None:
                    individuals.extend(h.pop)
        except Exception:
            pass
        if not individuals:
            individuals = list(getattr(res_ga, "pop", []))

        # Deduplicate by rounded X; keep (x, f) tuples
        seen = set()
        unique_pairs = []
        for ind in individuals:
            x = np.array(ind.X, dtype=float)
            f = float(ind.F[0])
            key = tuple(np.round(x, 6).tolist())
            if key not in seen:
                seen.add(key)
                unique_pairs.append((x, f))

        if not unique_pairs:
            logging.warning("Fold %s: GA produced no individuals. Skipping.", fid)
            continue

        # Sort ascending by RMSE (lower is better) and take top seeds
        unique_pairs.sort(key=lambda t: t[1])
        top_pairs = unique_pairs[:args.top_n_seeds]

        x0 = [t[0].tolist() for t in top_pairs]
        y0 = [t[1] for t in top_pairs]

        # ---- BO (skopt) warm-started by GA ----
        def bo_objective(x):
            out = {}
            # x = [window, layers, units, lr, batch_size]
            problem._evaluate(x, out)
            return float(out["F"][0])

        res_bo = gp_minimize(
            bo_objective,
            dimensions=LSTM_SEARCH_SPACE,
            n_calls=args.bo_n_calls,
            x0=x0, y0=y0,
            random_state=args.seed
        )

        best = res_bo.x
        best_rmse = float(res_bo.fun)

        logging.info(
            "Fold %s | best: window=%d, layers=%d, units=%d, lr=%.6g, batch=%d | RMSE=%.5f",
            fid, int(best[0]), int(best[1]), int(best[2]), float(best[3]), int(best[4]), best_rmse
        )

        # Keep top GA individuals for reference
        top_ga = []
        for x, f in top_pairs:
            d = {
                "window":     int(round(x[0])),
                "layers":     int(round(x[1])),
                "units":      int(round(x[2])),
                "lr":         float(x[3]),
                "batch_size": int(round(x[4])),
                "rmse":       float(f),
            }
            top_ga.append(d)

        results.append({
            "fold_id": int(fid),
            "ticker":  rec.get("ticker"),
            "champion": {
                "window":     int(best[0]),
                "layers":     int(best[1]),
                "units":      int(best[2]),
                "lr":         float(best[3]),
                "batch_size": int(best[4]),
            },
            "rmse": float(best_rmse),
            "top_ga": top_ga,
            "features_used": feature_cols,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
        })

        # Checkpoint after each fold
        write_results(Path(args.output_json), Path(args.output_csv), results)

    logging.info("=== Tier-1 complete. Saved to %s / %s ===", args.output_json, args.output_csv)

if __name__ == "__main__":
    main()