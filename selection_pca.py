import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.decomposition import PCA
from joblib import dump

"""
Fit a single global PCA for selection only on the UNION and TRAIN windows of all kept LSTM folds,
then optionally transform TRAIN & VAL of each fold into a common PCA space for downstream meta-
feature & clustering to pick representative folds.
"""

# ---- CONFIG ----
BASE_DIR = "data"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "processed_folds")

LSTM_FEATURE_COLUMNS_PATH = os.path.join(OUTPUT_BASE_DIR, "lstm_feature_columns.json")
DEFAULT_FOLDS_SUMMARY = os.path.join(OUTPUT_BASE_DIR, "folds_summary_lstm_cleaned.json")

SEL_PCA_DIR = os.path.join(OUTPUT_BASE_DIR, "selection_pca")
SEL_PCA_OUT_TRAIN = os.path.join(SEL_PCA_DIR, "train")
SEL_PCA_OUT_VAL = os.path.join(SEL_PCA_DIR, "val")
SEL_PCA_MODEL = os.path.join(OUTPUT_BASE_DIR, "selection_pca_model.pkl")
SEL_PCA_META = os.path.join(OUTPUT_BASE_DIR, "selection_pca_meta.json")

os.makedirs(SEL_PCA_OUT_TRAIN, exist_ok=True)
os.makedirs(SEL_PCA_OUT_VAL, exist_ok=True)

NON_FEATURE_COLUMNS = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns', 'Close_raw']

def load_feature_columns() -> List[str]:
    if os.path.exists(LSTM_FEATURE_COLUMNS_PATH):
        with open(LSTM_FEATURE_COLUMNS_PATH, "r") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or not cols:
            raise ValueError(f"[ERROR] Invalid feature list in {LSTM_FEATURE_COLUMNS_PATH}")
        return cols
    raise FileNotFoundError(f"[ERROR] Missing locked feature columns: {LSTM_FEATURE_COLUMNS_PATH}")

def absolute_path_from_rel(rel_path: str) -> str:
    return os.path.join(OUTPUT_BASE_DIR, rel_path)

def sample_balance_per_ticker(frames: List[pd.DataFrame], tickers: List[str], max_rows_per_ticker: int) -> List[pd.DataFrame]:
    """Down-sample each ticker slice to at most max_rows_per_ticker rows to avoid one
    ticker dominating the PCA fit. Assumes frames[i] corresponds to tickers[i]."""
    balanced = []
    for df, ticker in zip(frames, tickers):
        if df.shape[0] > max_rows_per_ticker:
            balanced.append(df.sample(n=max_rows_per_ticker, random_state=42))
        else:
            balanced.append(df)
    return balanced

def main():
    ap = argparse.ArgumentParser(description="GLOBAL PCA for selection (fit on union of TRAIN folds; optional transform)")
    ap.add_argument("--folds_summary_path", type=str, default=DEFAULT_FOLDS_SUMMARY,
                    help="Path to cleaned LSTM folds summary JSON (kept folds)")
    ap.add_argument("--n_components", type=str, default="0.95",
                    help="PCA n_components. Float variance like 0.95 or integer components, e.g., 12")
    ap.add_argument("--max_pc", type=int, default=20, help="Cap number of PCs after variance threshold")
    ap.add_argument("--balance_per_ticker", action="store_true", help="Down-sample union TRAIN per ticker before PCA fit")
    ap.add_argument("--max_rows_per_ticker", type=int, default=200000, help="Max rows per ticker when balancing")
    ap.add_argument("--no_write", action="store_true", help="Do not write transformed CSVs; only save PCA model/meta and updated summary")
    ap.add_argument("--assume_scaled", action="store_true", help="Skip scale checks (set if inputs are already globally scaled v1)")

    args = ap.parse_args()

    # Parse n_components
    try:
        if "." in args.n_components:
            n_components = float(args.n_components)
        else:
            n_components = int(args.n_components)
    except Exception:
        n_components = 0.95

    # load summary
    with open(args.folds_summary_path, "r") as f:
        folds = json.load(f)
    print(f"[INFO] Loaded cleaned folds: {len(folds)} from {args.folds_summary_path}")

    # Locked feature order
    feature_cols = load_feature_columns()

    # 1) Build UNION of all TRAIN folds (features only) to fit global PCA
    train_frames: List[pd.DataFrame] = []
    train_tickers: List[str] = []
    missing_files = 0
    feat_missing = 0

    for fold in folds:
        train_rel = fold.get("train_path_lstm")
        ticker = fold.get("ticker") or fold.get("Ticker")
        if not train_rel:
            continue
        train_path = absolute_path_from_rel(train_rel)
        if not os.path.exists(train_path):
            missing_files += 1
            continue
        df = pd.read_csv(train_path)
        # ensure columns exist
        if any(col not in df.columns for col in feature_cols):
            feat_missing += 1
            continue
        train_frames.append(df[feature_cols])
        train_tickers.append(str(ticker) if ticker is not None else "UNK")

    if not train_frames:
        raise RuntimeError("[ERROR] No train data found to fit selection PCA.")
    if missing_files:
        print(f"[WARN] Missing {missing_files} train files while building union dataset.")
    if feat_missing:
        print(f"[WARN] {feat_missing} train files skipped due to feature mismatch.")

    if args.balance_per_ticker:
        frames_by_ticker: Dict[str, List[pd.DataFrame]] = {}
        for frame, ticker in zip(train_frames, train_tickers):
            frames_by_ticker.setdefault(ticker, []).append(frame)
        frames_balanced = []
        for ticker, frs in frames_by_ticker.items():
            merged = pd.concat(frs, axis=0, ignore_index=True)
            sampled = sample_balance_per_ticker([merged], [ticker], args.max_rows_per_ticker)[0]
            frames_balanced.append(sampled)
        union_train_df = pd.concat(frames_balanced, axis=0, ignore_index=True)
    else:
        union_train_df = pd.concat(train_frames, axis=0, ignore_index=True)
    
    if not args.assume_scaled:
        # Heuristic: large std/mean magnitude might indicate unscaled features
        means = union_train_df.mean().abs()
        stds = union_train_df.std().abs()
        if (means.max() > 5) or (stds.max() > 5):
            print("[WARN] Feature scale looks large; be sure inputs were globally scaled (v1) before PCA.")

    X = union_train_df.values.astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("[ERROR] Non-finite values in union_train before PCA. Clean your inputs.")

    print(f"[INFO] Union train shape for PCA fit: {X.shape}")

    # --- Determine dims and fit PCA safely ---
    n_samples, n_features = X.shape
    is_variance_target = isinstance(n_components, float) and 0.0 < n_components < 1.0

    if is_variance_target:
        # 1) Fit full SVD to get full EVR, then choose k to reach target variance
        pca = PCA(n_components=None, svd_solver="full", random_state=42)
        pca.fit(X)
        evr_full = pca.explained_variance_ratio_
        k_by_var = int(np.searchsorted(np.cumsum(evr_full), n_components) + 1)

        # 2) Final reduced dimension with caps/sanity
        reduced_dim = min(k_by_var, int(args.max_pc), n_features, n_samples)

        # 3) Trim components/EVR to reduced_dim (no refit needed)
        pca.components_ = pca.components_[:reduced_dim]
        if hasattr(pca, "explained_variance_"):
            pca.explained_variance_ = pca.explained_variance_[:reduced_dim]
        pca.explained_variance_ratio_ = evr_full[:reduced_dim]

    else:
        # Integer target → use randomized for speed, with safe caps
        k_int = int(n_components)
        k_final = min(k_int, int(args.max_pc), n_features, n_samples)
        k_final = max(1, k_final)  # ensure at least 1
        pca = PCA(n_components=k_final, svd_solver="randomized", random_state=42)
        pca.fit(X)
        reduced_dim = int(pca.n_components_)

    evr = pca.explained_variance_ratio_.tolist() if hasattr(pca, "explained_variance_ratio_") else []
    cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1]) if evr else 0.0

    dump(pca, SEL_PCA_MODEL)
    with open(SEL_PCA_META, "w") as f:
        json.dump({
            "purpose": "selection_only",
            "do_not_use_for_final_training": True,
            "fitted_on": "union_of_all_train_folds_for_selection",
            "n_components_type": "var" if isinstance(n_components, float) else "int",
            "n_components_param": float(n_components) if isinstance(n_components, float) else int(n_components),
            "reduced_dim": reduced_dim,
            "explained_variance_ratio": evr,
            "cumulative_variance": cum_var,
            "feature_cols_used": feature_cols,
            "pca_cols": [f"PC{i+1}" for i in range(reduced_dim)],
            "solver": "full" if is_variance_target else "randomized",
            "dtype": "float32",
            "n_rows_fit": int(X.shape[0]),
            "n_features_fit": int(X.shape[1]),
            "balance_per_ticker": bool(args.balance_per_ticker),
            "max_rows_per_ticker": int(args.max_rows_per_ticker) if args.balance_per_ticker else None,
        }, f, indent=2)
    print(f"[INFO] Saved selection PCA model -> {SEL_PCA_MODEL}")
    print(f"[INFO] Selection PCA reduced_dim={reduced_dim}, cumulative_variance={cum_var:.4f}")

    # 2) Optionally transform each fold's TRAIN & VAL with this selection PCA
    folds_with_pca = []
    for fold in folds:
        gid = fold.get("global_fold_id")
        train_rel = fold.get("train_path_lstm")
        val_rel = fold.get("val_path_lstm")
        if train_rel is None or val_rel is None:
            print(f"[WARN] Fold {gid}: missing train/val paths, skip.")
            continue

        train_path = absolute_path_from_rel(train_rel)
        val_path = absolute_path_from_rel(val_rel)
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            print(f"[WARN] Fold {gid}: missing CSV files -> {train_path} / {val_path}, skip.")
            continue

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        if any(col not in train_df.columns for col in feature_cols) or \
           any(col not in val_df.columns for col in feature_cols):
            print(f"[WARN] Fold {gid}: feature mismatch, skip.")
            continue

        keep_cols_train = [c for c in NON_FEATURE_COLUMNS if c in train_df.columns]
        keep_cols_val = [c for c in NON_FEATURE_COLUMNS if c in val_df.columns]

        if not np.isfinite(train_df[feature_cols].to_numpy()).all():
            print(f"[WARN] Fold {gid}: non-finite in TRAIN features (will still transform).")
        if not np.isfinite(val_df[feature_cols].to_numpy()).all():
            print(f"[WARN] Fold {gid}: non-finite in VAL features (will still transform).")

        Xtr = train_df[feature_cols].values.astype(np.float32, copy=False)
        Xva = val_df[feature_cols].values.astype(np.float32, copy=False)

        Xtr_pca = pca.transform(Xtr)[:, :reduced_dim]
        Xva_pca = pca.transform(Xva)[:, :reduced_dim]

        pc_cols = [f"PC{i+1}" for i in range(reduced_dim)]

        train_out_df = pd.concat([train_df[keep_cols_train].reset_index(drop=True),
                                  pd.DataFrame(Xtr_pca, columns=pc_cols)], axis=1)
        val_out_df   = pd.concat([val_df[keep_cols_val].reset_index(drop=True),
                                  pd.DataFrame(Xva_pca, columns=pc_cols)], axis=1)

        if not args.no_write:
            train_out = os.path.join(SEL_PCA_OUT_TRAIN, os.path.basename(train_path))
            val_out = os.path.join(SEL_PCA_OUT_VAL, os.path.basename(val_path))
            train_out_df.to_csv(train_out, index=False)
            val_out_df.to_csv(val_out, index=False)
        else:
            train_out = os.path.basename(train_path)
            val_out = os.path.basename(val_path)

        folds_with_pca.append({
            **fold,
            "train_path_lstm_selection_pca": os.path.relpath(os.path.join(SEL_PCA_OUT_TRAIN, os.path.basename(train_path)), 
                                                             OUTPUT_BASE_DIR) if not args.no_write else None,
            "val_path_lstm_selection_pca": os.path.relpath(os.path.join(SEL_PCA_OUT_VAL, os.path.basename(val_path)), 
                                                             OUTPUT_BASE_DIR) if not args.no_write else None,
            "selection_pca_model_path": os.path.relpath(SEL_PCA_MODEL, OUTPUT_BASE_DIR),
            "selection_pca_meta_path": os.path.relpath(SEL_PCA_META,  OUTPUT_BASE_DIR),
            "selection_reduced_dim": reduced_dim})

        print(f"[OK] Fold {gid}: selection PCs={reduced_dim} -> {train_out} / {val_out}")

    # 3) Save updated fold summary for selection stage
    out_summary = args.folds_summary_path.replace(".json", "_selection_pca.json")
    with open(out_summary, "w") as f:
        json.dump(folds_with_pca, f, indent=2)
    print(f"[INFO] Saved SELECTION PCA fold summary -> {out_summary}")


if __name__ == "__main__":
    main()