import os, json, argparse, logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from lstm_utils import create_periodic_lstm_objective  # builds the Tier-2 objective (Sharpe/MDD)

# ---------- Small helpers (Pareto & knee) ----------
def is_nondominated(F: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows in F (minimization)."""
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        fi = F[i]
        dom = np.all(F <= fi + 1e-12, axis=1) & np.any(F < fi - 1e-12, axis=1)
        dom[i] = False
        if np.any(dom):
            nd[i] = False
    return nd

def knee_index(F: np.ndarray) -> int:
    """Pick knee via min distance to origin in normalized objective space."""
    f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    d = np.sqrt((f**2).sum(axis=1))
    return int(np.argmin(d))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json_any(p: Path):
    with p.open("r") as f:
        return json.load(f)

def normalize_t1_records(rec) -> Dict[str, Any]:
    """Extract LSTM Tier-1 champion fields robustly."""
    ch = rec.get("champion") or rec.get("best_params") or {}
    return dict(
        layers=int(ch.get("layers", ch.get("num_layers", 1))),
        batch_size=int(ch.get("batch_size", 32)),
        dropout=float(ch.get("dropout", 0.2)),
        patience=int(ch.get("patience", 5)),
    )

def unwrap_folds(obj) -> List[dict]:
    """Support both list and nested dict structures."""
    if isinstance(obj, dict):
        for k in ("lstm", "arima", "folds", "data", "results"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unexpected folds JSON structure.")

def get_args():
    ap = argparse.ArgumentParser("Random Search baseline — LSTM Tier-2")
    ap.add_argument("--folds-json", required=True,
                    help="LSTM folds JSON with train/val paths and ticker per fold.")
    ap.add_argument("--tier1-json", required=True,
                    help="Tier-1 champions JSON (list or {'results': [...]}) per fold.")
    ap.add_argument("--intervals", default="10,20,42",
                    help="Comma-separated retraining intervals.")
    ap.add_argument("--n-samples", type=int, default=120,
                    help="Random samples per (fold × interval).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip (fold, interval) pairs that already have >= n-samples in results CSV.")
    ap.add_argument("--cost-per-turnover", type=float, default=0.0005 + 0.0002)
    ap.add_argument("--min-block-vol", type=float, default=0.0015)
    ap.add_argument("--hysteresis", type=float, default=0.05)
    ap.add_argument("--mad-k", type=float, default=0.5)
    ap.add_argument("--warmup-len", type=int, default=252)
    # Tier-2 bounds (window, units, lr, epochs, rel_thresh)
    ap.add_argument("--w-min", type=int, default=10)
    ap.add_argument("--w-max", type=int, default=40)
    ap.add_argument("--u-min", type=int, default=32)
    ap.add_argument("--u-max", type=int, default=96)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--lr-max", type=float, default=1e-2)
    ap.add_argument("--ep-min", type=int, default=10)
    ap.add_argument("--ep-max", type=int, default=80)
    ap.add_argument("--thr-min", type=float, default=0.50)
    ap.add_argument("--thr-max", type=float, default=0.95)

    ap.add_argument("--metric-col", default="target_log_returns",
                    help="Realized returns column in VAL CSV ('target_log_returns' or 'Log_Returns').")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out-prefix", default="results/baselines/rs_lstm",
                    help="Output prefix for CSVs (results/front/knee).")
    return ap.parse_args()

def ensure_targets(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Ensure df has:
      - 'target' (next-step target for supervised training)
      - 'target_log_returns' (realized PnL series; default metric_col)
    If missing and 'Log_Returns' exists, build as shift(-1).
    Drops last row if shift creates NaN.
    """
    df = df.copy()
    # Build target if missing
    if "target" not in df.columns:
        if "Log_Returns" in df.columns:
            df["target"] = df["Log_Returns"].shift(-1)
        elif "target_log_returns" in df.columns:
            df["target"] = df["target_log_returns"]
        else:
            raise ValueError("Missing both 'target' and 'Log_Returns' to create target.")
    # Build target_log_returns if chosen metric is target_log_returns
    if metric_col == "target_log_returns" and "target_log_returns" not in df.columns:
        if "Log_Returns" in df.columns:
            df["target_log_returns"] = df["Log_Returns"].shift(-1)
        elif "target" in df.columns:
            df["target_log_returns"] = df["target"]
        else:
            raise ValueError("Cannot construct 'target_log_returns' without 'Log_Returns' or 'target'.")
    # Clean tail NaNs from shifting
    if df["target"].isna().any():
        df = df.dropna(subset=["target"]).reset_index(drop=True)
    if metric_col in df.columns and df[metric_col].isna().any():
        df = df.dropna(subset=[metric_col]).reset_index(drop=True)
    return df

def pick_pcs(cols: List[str]) -> List[str]:
    """Return existing PCA columns among PC1..PC7, in order."""
    return [f"PC{i}" for i in range(1, 8) if f"PC{i}" in cols]

def _load_progress(res_csv: Path) -> Dict[Tuple[int, int], int]:
    """
    Return dict: (fold_id, retrain_interval) -> max iter completed (count of rows).
    If 'iter' column exists, use max(iter); else use count of rows for the pair.
    """
    progress: Dict[Tuple[int, int], int] = {}
    if not res_csv.exists() or os.path.getsize(res_csv) == 0:
        return progress
    try:
        dfp = pd.read_csv(res_csv, usecols=["fold_id", "retrain_interval", "iter"])
    except Exception:
        # fallback if iter column missing
        dfp = pd.read_csv(res_csv, usecols=["fold_id", "retrain_interval"])
        dfp["iter"] = 1
    grp = dfp.groupby(["fold_id", "retrain_interval"])
    for (fid, itv), g in grp:
        # robust: prefer max(iter) if present; else count
        if "iter" in g.columns and pd.api.types.is_numeric_dtype(g["iter"]):
            progress[(int(fid), int(itv))] = int(g["iter"].max())
        else:
            progress[(int(fid), int(itv))] = int(len(g))
    return progress

def main():
    args = get_args()
    np.random.seed(args.seed)

    # Logging
    ensure_dir(Path(args.out_prefix).parent)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Load folds (LSTM)
    folds_obj = unwrap_folds(load_json_any(Path(args.folds_json)))
    folds_df = pd.DataFrame([{
        "fold_id": int(rec.get("global_fold_id", rec.get("fold_id"))),
        "ticker": rec.get("ticker", "UNK"),
        "train": rec.get("final_train_path", rec.get("train_path")),
        "val":   rec.get("final_val_path", rec.get("val_path")),
        "val_meta": rec.get("val_meta_path_lstm", rec.get("val_meta_path")),
    } for rec in folds_obj])
    if folds_df["fold_id"].isna().any():
        raise ValueError("Some fold records are missing 'global_fold_id'/'fold_id'.")

    # Load Tier-1 champions (LSTM)
    t1 = load_json_any(Path(args.tier1_json))
    if isinstance(t1, dict) and "results" in t1:
        t1 = t1["results"]
    champs = {}
    for rec in t1:
        fid = int(rec.get("fold_id", rec.get("global_fold_id", -1)))
        if fid == -1:
            continue
        champs[fid] = normalize_t1_records(rec)

    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]

    # Outputs
    res_csv   = Path(args.out_prefix + "_results.csv")
    front_csv = Path(args.out_prefix + "_front.csv")
    knee_csv  = Path(args.out_prefix + "_knee.csv")
    for p in (res_csv, front_csv, knee_csv):
        ensure_dir(p.parent)

    # Progress / resume map: (fold_id, interval) -> max_iter_done
    progress = _load_progress(res_csv)
    if progress:
        logging.info(f"[RESUME] Loaded progress for {len(progress)} (fold,interval) pairs.")

    # Base dir to resolve relative CSV paths
    folds_json_path = Path(args.folds_json).resolve()
    base_dir = folds_json_path.parents[2] if len(folds_json_path.parents) >= 2 else folds_json_path.parent

    # Iterate folds
    for _, rec in folds_df.iterrows():
        fid   = int(rec["fold_id"])
        tick  = str(rec["ticker"])
        train = rec["train"]; val = rec["val"]
        if pd.isna(train) or pd.isna(val):
            logging.warning(f"[SKIP] Missing train/val for fold {fid} ({tick})")
            continue
        if fid not in champs:
            logging.warning(f"[SKIP] Tier-1 champion missing for fold {fid}")
            continue

        train_csv = (base_dir / str(train)).resolve()
        val_csv   = (base_dir / str(val)).resolve()
        if not (train_csv.exists() and val_csv.exists()):
            logging.warning(f"[SKIP] CSV not found for fold {fid}: {train_csv} / {val_csv}")
            continue

        # Load dataframes
        try:
            df_tr = pd.read_csv(train_csv)
            df_va = pd.read_csv(val_csv)
        except Exception as e:
            logging.warning(f"[SKIP] Read error fold {fid}: {e}")
            continue

        # Ensure targets from Log_Returns if needed
        try:
            df_tr = ensure_targets(df_tr, metric_col=args.metric_col)
            df_va = ensure_targets(df_va, metric_col=args.metric_col)
        except Exception as e:
            logging.warning(f"[SKIP] Target creation failed for fold {fid}: {e}")
            continue

        # Pick PCA columns PC1..PC7 (only those present)
        pcs_tr = pick_pcs(df_tr.columns.tolist())
        pcs_va = pick_pcs(df_va.columns.tolist())
        pcs = [c for c in pcs_tr if c in pcs_va]  # intersection, keep order
        if len(pcs) == 0:
            logging.warning(f"[SKIP] No PCA columns PC1..PC7 found in fold {fid}")
            continue

        # Final column check
        needed_tr = pcs + ["target"]
        needed_va = pcs + ["target", args.metric_col]
        if any(c not in df_tr.columns for c in needed_tr) or any(c not in df_va.columns for c in needed_va):
            logging.warning(f"[SKIP] Missing required columns in fold {fid}")
            continue

        # Drop residual NaNs in the needed columns
        df_tr = df_tr.dropna(subset=needed_tr).reset_index(drop=True)
        df_va = df_va.dropna(subset=needed_va).reset_index(drop=True)
        if len(df_tr) == 0 or len(df_va) == 0:
            logging.warning(f"[SKIP] Empty after NaN drop in fold {fid}")
            continue

        champion = champs[fid]

        for interval in intervals:
            key = (fid, int(interval))
            already = int(progress.get(key, 0))
            if args.skip_existing and already >= int(args.n_samples):
                logging.info(f"[SKIP-EXISTING] fold={fid} int={interval} already has {already} >= n_samples={args.n_samples}")
                continue

            # Build Tier-2 objective (same as GA/BO)
            obj = create_periodic_lstm_objective(
                train_df=df_tr,
                val_df=df_va,
                feature_cols=pcs,
                champ=champion,
                retrain_interval=int(interval),
                cost_per_turnover=float(args.cost_per_turnover),
                min_block_vol=float(args.min_block_vol),
                metric_col=args.metric_col,
                debug=args.debug
            )

            rng = np.random.default_rng(args.seed + fid + interval)

            # Resume from next iteration
            start_iter = already + 1
            if start_iter > int(args.n_samples):
                start_iter = int(args.n_samples)  # just in case

            eval_rows: List[Dict[str, Any]] = []
            for it in range(start_iter, int(args.n_samples) + 1):
                w   = int(rng.integers(args.w_min, args.w_max + 1))
                u   = int(rng.integers(args.u_min, args.u_max + 1))
                ep  = int(rng.integers(args.ep_min, args.ep_max + 1))
                lr  = float(rng.uniform(args.lr_min, args.lr_max))
                thr = float(rng.uniform(args.thr_min, args.thr_max))

                f0, f1 = obj(np.array([w, u, lr, ep, thr], dtype=float))
                meta = getattr(obj, "stats_map", {}).get((w, u, round(lr, 6), ep, round(thr, 6)), {})

                row = {
                    "fold_id": fid,
                    "ticker": tick,
                    "retrain_interval": int(interval),
                    "stage": "RS",
                    "iter": it,
                    "window": w,
                    "units": u,
                    "learning_rate": lr,
                    "epochs": ep,
                    "threshold": thr,
                    # objectives (min): f0=-Sharpe, f1=MDD  -> report Sharpe/MDD for readability
                    "sharpe": float(-(f0 - meta.get("penalty", 0.0))),  # best-effort undo penalty on display
                    "mdd": float(f1 - meta.get("penalty", 0.0)),
                    # raw stats (if available)
                    "raw_sharpe": float(meta.get("raw_sharpe", np.nan)),
                    "raw_mdd": float(meta.get("raw_mdd", np.nan)),
                    "turnover": float(meta.get("turnover", np.nan)),
                    "penalty": float(meta.get("penalty", np.nan)),
                }
                eval_rows.append(row)

                # checkpoint append (per-iter)
                df1 = pd.DataFrame([row])
                header = (not res_csv.exists()) or os.path.getsize(res_csv) == 0
                df1.to_csv(res_csv, mode="a", index=False, header=header)

            # If we resumed and no new rows were produced, still rebuild front/knee from existing rows?
            # Here, we only compute front/knee from 'eval_rows' produced this run.
            # To be safe, if eval_rows is empty but already >= n_samples, we skip silently (skip-existing covers this).
            if not eval_rows:
                logging.info(f"[RESUME] fold={fid} int={interval} had no new iters to run.")
                continue

            # Pareto front from *this-run* RS evaluations
            df_eval = pd.DataFrame(eval_rows)
            F = np.c_[ -df_eval["sharpe"].astype(float).to_numpy(),
                        df_eval["mdd"].astype(float).to_numpy() ]
            nd = is_nondominated(F)
            df_front = df_eval.loc[nd].copy()
            df_front["front_type"] = "RS"

            header = (not front_csv.exists()) or os.path.getsize(front_csv) == 0
            df_front.to_csv(front_csv, mode="a", index=False, header=header)

            # Knee pick
            if df_front.shape[0] > 0:
                F_front = np.c_[ -df_front["sharpe"].to_numpy(float),
                                   df_front["mdd"].to_numpy(float) ]
                k = knee_index(F_front)
                knee = df_front.iloc[k].copy()
                knee["pick"] = "knee_rs"
                header = (not knee_csv.exists()) or os.path.getsize(knee_csv) == 0
                pd.DataFrame([knee]).to_csv(knee_csv, mode="a", index=False, header=header)

            logging.info(f"[Fold {fid} | int={interval}] RS new_evals={len(df_eval)} | front_added={df_front.shape[0]}")

    print("\n=== Random Search baseline (LSTM Tier-2) — DONE ===")
    print(f"All evals : {res_csv}")
    print(f"RS fronts : {front_csv}")
    print(f"RS knees  : {knee_csv}")

if __name__ == "__main__":
    main()
