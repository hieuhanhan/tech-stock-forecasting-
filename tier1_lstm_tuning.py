import json
import argparse
import logging
import random
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem

from skopt.space import Integer, Real
from skopt import gp_minimize

from lstm_utils import (
    Tier1LSTMProblem,   
    NON_FEATURE_KEEP,
)

# ----------------- Defaults -----------------
PROCESSED_BASE = Path("data/processed_folds")
LOG_DIR = Path("logs/tier1_lstm")

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
    """Enable mixed precision + XLA on GPU and memory growth when available."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            tf.config.optimizer.set_jit(True)  # XLA
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def load_features(feature_path: Path | None, df_train: pd.DataFrame) -> list:
    # Priority 1: use PCA columns if exist
    pc_cols = [c for c in df_train.columns if c.startswith("PC")]
    if pc_cols:
        return sorted(pc_cols, key=lambda x: int(x[2:]))  # PC1, PC2...
    
    # Priority 2: if feature JSON exists, use intersection
    if feature_path and feature_path.exists():
        feats = [c for c in load_json(feature_path) if c in df_train.columns]
        if feats:
            return feats
    
    # Priority 3: fallback infer
    return [c for c in df_train.columns if c not in NON_FEATURE_KEEP]

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

def coerce_fold_record(rec: dict) -> dict:
    """Flexible record: choose id & train/val from several possible keys."""
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

def have_gpu() -> bool:
    try:
        return len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        return False

# ---------- Dynamic bounds & seeds ----------
def make_dynamic_bounds(n_features: int, n_train: int, gpu: bool):
    """
    Return (xl, xu) arrays and skopt dimensions with dynamic limits.
    """
    # window
    if n_features <= 20:
        w_min, w_max = 10, 30
        units_min, units_max = 16, 48
        batch_min, batch_max = 16, 64
    elif n_features <= 60:
        w_min, w_max = 20, 40
        units_min, units_max = 32, (96 if gpu else 64)
        batch_min, batch_max = 32, 96
    else:
        w_min, w_max = 20, 40
        units_min, units_max = 48, (128 if gpu else 96)
        batch_min, batch_max = 32, 128

    # dataset length heuristic
    if n_train < 10_000:
        units_max = min(units_max, 64)

    # layers & lr fixed range
    layers_min, layers_max = 1, 3
    lr_min, lr_max = 1e-4, 1e-2

    xl = np.array([w_min, layers_min, units_min, lr_min, batch_min], dtype=float)
    xu = np.array([w_max, layers_max, units_max, lr_max, batch_max], dtype=float)

    sk_dims = [
        Integer(w_min, w_max, name="window"),
        Integer(layers_min, layers_max, name="layers"),
        Integer(units_min, units_max, name="units"),
        Real(lr_min, lr_max, name="lr", prior="log-uniform"),
        Integer(batch_min, batch_max, name="batch_size"),
    ]
    return xl, xu, sk_dims

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def make_preferred_bo_seeds(xl, xu):
    """
    Build small set of warm-seeds biased to small/moderate configs.
    """
    w_opts = sorted(set([clamp(20, xl[0], xu[0]), clamp(25, xl[0], xu[0]), clamp(30, xl[0], xu[0])]))
    l_opts = sorted(set([clamp(1, xl[1], xu[1]),  clamp(2, xl[1], xu[1])]))
    u_opts = sorted(set([clamp(32, xl[2], xu[2]), clamp(40, xl[2], xu[2]), clamp(48, xl[2], xu[2])]))
    b_opts = sorted(set([clamp(32, xl[4], xu[4]), clamp(48, xl[4], xu[4]), clamp(64, xl[4], xu[4])]))
    lr0   = clamp(1e-3, xl[3], xu[3])

    seeds = []
    for w, l, u, b in product(w_opts, l_opts, u_opts, b_opts):
        seeds.append([int(w), int(l), int(u), float(lr0), int(b)])
        if len(seeds) >= 12:  # keep compact
            break
    return seeds

# ---------- Constrained wrapper for GA ----------
class Tier1ConstrainedProblem(ElementwiseProblem):
    """
    Wrap Tier1LSTMProblem and apply a mild multiplicative penalty when
    units exceed soft cap: units > min(4*n_features, 2*window).
    """
    def __init__(self, base_problem: Tier1LSTMProblem, xl: np.ndarray, xu: np.ndarray, n_features: int):
        super().__init__(n_var=5, n_obj=1, xl=xl.copy(), xu=xu.copy())
        self.base = base_problem
        self.n_features = int(n_features)

    def _evaluate(self, x, out, *_, **__):
        tmp = {}
        self.base._evaluate(x, tmp)
        rmse = float(tmp["F"][0])

        window = int(round(float(x[0])))
        units  = int(round(float(x[2])))

        soft_cap = int(min(4 * self.n_features, 2 * window))
        overflow = max(0, units - soft_cap)
        mult = 1.0 + 0.25 * (overflow / max(1, soft_cap))  # gentle
        out["F"] = [rmse * mult]

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser("Tier-1 LSTM Tuning (dynamic bounds + soft constraints, no GA/BO history)")
    parser.add_argument('--folds-path', required=True,
                        help='JSON list of folds, each with train/val CSV paths.')
    parser.add_argument('--feature-path', default='data/processed_folds/lstm_feature_columns.json',
                        help='Optional JSON list of feature names.')
    parser.add_argument('--output-json', default='data/tuning_results/jsons/tier1_lstm.json')
    parser.add_argument('--output-csv',  default='data/tuning_results/csv/tier1_lstm.csv')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    # Objective/training knobs
    parser.add_argument('--t1-epochs', type=int, default=10, help='Epochs per RMSE evaluation.')
    parser.add_argument('--t1-patience', type=int, default=3, help='EarlyStopping patience.')

    # Search knobs
    parser.add_argument('--ga-pop-size', type=int, default=25, help='GA population size.')
    parser.add_argument('--ga-n-gen', type=int, default=20, help='GA generations.')
    parser.add_argument('--bo-n-calls', type=int, default=40, help='skopt BO calls.')
    parser.add_argument('--top-n-seeds', type=int, default=12, help='Top GA individuals to warm-start BO.')

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
    setup_tf_perf()
    gpu = have_gpu()

    ensure_dir(Path(args.output_json).parent)
    ensure_dir(Path(args.output_csv).parent)

    # ---- Load folds JSON ----
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

    for item in tqdm(pending, desc="Tier-1 LSTM"):
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

        n_features = len(feature_cols)
        n_train = len(train_df)

        # Base Tier-1 objective (RMSE)
        base_problem = Tier1LSTMProblem(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            t1_epochs=args.t1_epochs,
            patience=args.t1_patience,
        )

        # Dynamic bounds & BO dims
        xl, xu, bo_dims = make_dynamic_bounds(n_features, n_train, gpu)

        # Constrained wrapper for GA (penalty inside)
        problem = Tier1ConstrainedProblem(base_problem, xl, xu, n_features)

        # ---- GA (seed search) ----
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
            return_algorithm=True,
            seed=args.seed
        )

        # Collect unique individuals to warm-start BO
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
            x = np.array(ind.X, dtype=float)
            # keep within bounds (safety)
            x = np.clip(x, xl, xu)
            f = float(ind.F[0])
            key = tuple(np.round(x, 6).tolist())
            if key not in seen:
                seen.add(key)
                uniq.append((x, f))

        uniq.sort(key=lambda t: t[1])
        top_pairs = uniq[:args.top_n_seeds]

        # Preferred BO seeds (bias small/moderate)
        pref_seeds = make_preferred_bo_seeds(xl, xu)

        # Build x0/y0 for BO
        x0 = [t[0].tolist() for t in top_pairs] + pref_seeds
        # Evaluate base RMSE for the preferred seeds (no penalty for y0, BO handles x)
        y0_extra = []
        for s in pref_seeds:
            tmp = {}
            base_problem._evaluate(s, tmp)      # true RMSE on validation
            y0_extra.append(float(tmp["F"][0]))
        y0 = [t[1] for t in top_pairs] + y0_extra

        # ---- BO (skopt) with soft-constraint penalty in objective ----
        def bo_objective(x):
            """
            Apply the same soft penalty here to bias BO away from oversized models.
            """
            tmp = {}
            base_problem._evaluate(x, tmp)
            rmse = float(tmp["F"][0])

            window = int(round(float(x[0])))
            units  = int(round(float(x[2])))
            soft_cap = int(min(4 * n_features, 2 * window))
            overflow = max(0, units - soft_cap)
            mult = 1.0 + 0.25 * (overflow / max(1, soft_cap))
            return rmse * mult

        res_bo = gp_minimize(
            bo_objective,
            dimensions=bo_dims,
            n_calls=args.bo_n_calls,
            x0=x0, y0=y0,
            random_state=args.seed
        )

        best = res_bo.x
        best_rmse = float(res_bo.fun)
        logging.info(
            "Fold %s (%s) | best: window=%d, layers=%d, units=%d, lr=%.6g, batch=%d | RMSE*=%.5f",
            fid, (ticker or "n/a"), int(best[0]), int(best[1]), int(best[2]),
            float(best[3]), int(best[4]), best_rmse
        )

        # Keep top GA individuals for reference (optional, compact)
        top_ga = []
        for x, f in top_pairs[:min(8, len(top_pairs))]:
            d = {
                "window":     int(round(x[0])),
                "layers":     int(round(x[1])),
                "units":      int(round(x[2])),
                "lr":         float(x[3]),
                "batch_size": int(round(x[4])),
                "rmse_penalized": float(f),
            }
            top_ga.append(d)

        results.append({
            "fold_id": int(fid),
            "ticker":  ticker,
            "champion": {
                "window":     int(best[0]),
                "layers":     int(best[1]),
                "units":      int(best[2]),
                "lr":         float(best[3]),
                "batch_size": int(best[4]),
            },
            "rmse_penalized": float(best_rmse),
            "features_used": feature_cols,
            "bounds": {
                "window": [int(xl[0]), int(xu[0])],
                "layers": [int(xl[1]), int(xu[1])],
                "units":  [int(xl[2]), int(xu[2])],
                "lr":     [float(xl[3]), float(xu[3])],
                "batch":  [int(xl[4]), int(xu[4])],
                "gpu": bool(gpu),
                "n_features": int(n_features),
                "n_train": int(n_train),
            },
            "top_ga": top_ga,
        })

        # checkpoint after each fold
        write_results(Path(args.output_json), Path(args.output_csv), results)

    logging.info("=== Tier-1 LSTM complete. Saved to %s / %s ===", args.output_json, args.output_csv)

if __name__ == "__main__":
    main()