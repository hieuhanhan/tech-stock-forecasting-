#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from joblib import dump

# ---- CONFIG ----
BASE_DIR = "data"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "processed_folds")

LSTM_FEATURE_COLUMNS_PATH = os.path.join(OUTPUT_BASE_DIR, "lstm_feature_columns.json")
DEFAULT_FOLDS_SUMMARY = os.path.join(OUTPUT_BASE_DIR, "folds_summary_lstm_cleaned_nvda_cleaned.json")

PCA_OUT_TRAIN = os.path.join(OUTPUT_BASE_DIR, "lstm_pca_global", "train")
PCA_OUT_VAL   = os.path.join(OUTPUT_BASE_DIR, "lstm_pca_global", "val")
PCA_MODEL_PATH = os.path.join(OUTPUT_BASE_DIR, "lstm_pca_global_model.pkl")
PCA_META_PATH  = os.path.join(OUTPUT_BASE_DIR, "lstm_pca_global_meta.json")

os.makedirs(PCA_OUT_TRAIN, exist_ok=True)
os.makedirs(PCA_OUT_VAL, exist_ok=True)


NON_FEATURE_COLS = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns', 'Close_raw']

def load_feature_columns():
    if os.path.exists(LSTM_FEATURE_COLUMNS_PATH):
        with open(LSTM_FEATURE_COLUMNS_PATH, "r") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or not cols:
            raise ValueError(f"[ERROR] Invalid feature list in {LSTM_FEATURE_COLUMNS_PATH}")
        return cols
    raise FileNotFoundError(f"[ERROR] Missing locked feature columns: {LSTM_FEATURE_COLUMNS_PATH}")

def absolute_path_from_rel(rel_path: str) -> str:
    return os.path.join(OUTPUT_BASE_DIR, rel_path)

def main():
    ap = argparse.ArgumentParser(description="Global PCA (fit on union of all TRAIN folds, transform TRAIN & VAL)")
    ap.add_argument("--folds_summary_path", type=str, default=DEFAULT_FOLDS_SUMMARY,
                    help="Path to cleaned LSTM folds summary JSON")
    ap.add_argument("--n_components", type=str, default="0.95",
                    help="PCA n_components. Float (variance) like 0.95 or integer components, e.g., 5")
    args = ap.parse_args()

    # parse n_components
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

    # locked feature order
    feature_cols = load_feature_columns()
    keep_cols = [c for c in NON_FEATURE_COLS if c in ['Date','Ticker','target','Log_Returns','Close_raw']]

    # 1) Build UNION of all TRAIN folds (features only) to fit global PCA
    train_frames = []
    missing_files = 0
    for fold in folds:
        tr_rel = fold.get("train_path_lstm")
        if not tr_rel:
            continue
        tr_path = absolute_path_from_rel(tr_rel)
        if not os.path.exists(tr_path):
            missing_files += 1
            continue
        df = pd.read_csv(tr_path)
        # ensure columns exist
        if any(col not in df.columns for col in feature_cols):
            continue
        train_frames.append(df[feature_cols])

    if not train_frames:
        raise RuntimeError("[ERROR] No train data found to fit global PCA.")
    if missing_files:
        print(f"[WARN] Missing {missing_files} train files while building union dataset.")

    union_train = pd.concat(train_frames, axis=0, ignore_index=True)
    if not np.isfinite(union_train.to_numpy()).all():
        raise ValueError("[ERROR] Non-finite values in union_train before PCA. Clean your inputs.")

    print(f"[INFO] Union train shape for PCA fit: {union_train.shape}")

    # Fit a single global PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(union_train.values)

    reduced_dim = int(pca.n_components_) 
    evr = pca.explained_variance_ratio_.tolist()
    cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])

    dump(pca, PCA_MODEL_PATH)
    with open(PCA_META_PATH, "w") as f:
        json.dump({
            "fitted_on": "union_of_all_train_folds",
            "n_components_type": "var" if isinstance(n_components, float) else "int",
            "n_components_param": n_components if isinstance(n_components, float) else int(n_components),
            "reduced_dim": reduced_dim,
            "explained_variance_ratio": evr,
            "cumulative_variance": cum_var,
            "feature_cols_used": feature_cols,
            "pc_cols": [f"PC{i+1}" for i in range(reduced_dim)]
        }, f, indent=2)
    print(f"[INFO] Saved global PCA model -> {PCA_MODEL_PATH}")
    print(f"[INFO] Global PCA reduced_dim={reduced_dim}, cumulative_variance={cum_var:.4f}")


    # 2) Transform each fold's TRAIN & VAL with this global PCA
    folds_with_pca = []
    for fold in folds:
        gid = fold.get("global_fold_id")
        tr_rel = fold.get("train_path_lstm")
        va_rel = fold.get("val_path_lstm")
        if tr_rel is None or va_rel is None:
            print(f"[WARN] Fold {gid}: missing train/val paths, skip.")
            continue

        tr_path = absolute_path_from_rel(tr_rel)
        va_path = absolute_path_from_rel(va_rel)
        if not (os.path.exists(tr_path) and os.path.exists(va_path)):
            print(f"[WARN] Fold {gid}: missing CSV files -> {tr_path} / {va_path}, skip.")
            continue

        train_df = pd.read_csv(tr_path)
        val_df   = pd.read_csv(va_path)
        if any(col not in train_df.columns for col in feature_cols) or \
           any(col not in val_df.columns for col in feature_cols):
            print(f"[WARN] Fold {gid}: feature mismatch, skip.")
            continue

        Xtr = train_df[feature_cols].values
        Xva = val_df[feature_cols].values
        Xtr_pc = pca.transform(Xtr)
        Xva_pc = pca.transform(Xva)

        pc_cols = [f"PC{i+1}" for i in range(reduced_dim)]
        train_out_df = pd.concat([train_df[keep_cols].reset_index(drop=True),
                                  pd.DataFrame(Xtr_pc, columns=pc_cols)], axis=1)
        val_out_df   = pd.concat([val_df[keep_cols].reset_index(drop=True),
                                  pd.DataFrame(Xva_pc, columns=pc_cols)], axis=1)

        train_out = os.path.join(PCA_OUT_TRAIN, os.path.basename(tr_path))
        val_out   = os.path.join(PCA_OUT_VAL,   os.path.basename(va_path))
        train_out_df.to_csv(train_out, index=False)
        val_out_df.to_csv(val_out, index=False)

        folds_with_pca.append({
            **fold,
            "train_path_lstm_pca": os.path.relpath(train_out, OUTPUT_BASE_DIR),
            "val_path_lstm_pca":   os.path.relpath(val_out,   OUTPUT_BASE_DIR),
            "pca_model_path":      os.path.relpath(PCA_MODEL_PATH, OUTPUT_BASE_DIR),
            "pca_meta_path":       os.path.relpath(PCA_META_PATH,  OUTPUT_BASE_DIR),
            "reduced_dim":         reduced_dim
        })

        print(f"[OK] Fold {gid}: global PCs={reduced_dim} -> {train_out} / {val_out}")

    # 3) Save updated fold summary
    out_summary = args.folds_summary_path.replace(".json", "_global_pca.json")
    with open(out_summary, "w") as f:
        json.dump(folds_with_pca, f, indent=2)
    print(f"[INFO] Saved GLOBAL PCA fold summary -> {out_summary}")

if __name__ == "__main__":
    main()