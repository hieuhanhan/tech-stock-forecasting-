#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_meta(meta_path: str) -> pd.DataFrame:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    df = pd.read_csv(meta_path)
    if df.empty:
        raise ValueError(f"Meta file is empty: {meta_path}")
    if 'fold_id' not in df.columns:
        logging.warning("Column 'fold_id' not found in meta file.")
    return df

def prepare_X(meta_df: pd.DataFrame) -> np.ndarray:
    return meta_df.drop(columns=['fold_id', 'ticker'], errors='ignore').fillna(0).values

def scale_raw(X, **kwargs):
    return X, {"type": "raw"}

def scale_minmax(X, **kwargs):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X), {"type": "minmax"}

def scale_standard(X, **kwargs):
    scaler = StandardScaler()
    return scaler.fit_transform(X), {"type": "standard"}

def scale_standard_pca(X, pca_var=0.9, **kwargs):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=pca_var, random_state=42)
    Xp = pca.fit_transform(Xs)
    return Xp, {"type": "standard+pca", "pca_var": pca_var, "n_components": pca.n_components_}

SCALERS = {
    "raw": scale_raw,
    "minmax": scale_minmax,
    "standard": scale_standard,
    "standard_pca": scale_standard_pca,
}

def try_kmeans(X: np.ndarray, k: int) -> tuple[float, float]:
    """Return (silhouette, inertia). If silhouette can't be computed -> -1."""
    if X.shape[0] <= k or k < 2:
        return -1.0, np.nan
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    if len(np.unique(labels)) < 2:
        return -1.0, km.inertia_
    sil = silhouette_score(X, labels)
    return sil, km.inertia_

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run(meta_path: str,
        out_dir: str,
        k_min: int,
        k_max: int,
        pca_var: float,
        model_name: str):

    os.makedirs(out_dir, exist_ok=True)

    meta_df = load_meta(meta_path)
    X = prepare_X(meta_df)

    results = []

    for scaler_name, scaler_fn in SCALERS.items():
        logging.info(f"[{model_name}] Evaluating scaler='{scaler_name}' ...")
        X_scaled, scaler_info = scaler_fn(X, pca_var=pca_var)

        for k in range(k_min, k_max + 1):
            sil, inertia = try_kmeans(X_scaled, k)
            results.append({
                "model": model_name,
                "scaler": scaler_name,
                "k": k,
                "silhouette": sil,
                "inertia": inertia,
                **scaler_info
            })
            logging.info(f"[{model_name}] scaler={scaler_name:14s} k={k:2d} "
                         f"sil={sil: .4f} inertia={inertia: .2f}")

    res_df = pd.DataFrame(results)

    # choose best by max silhouette, tie-breaker = min inertia, then smaller k
    best = (res_df
            .sort_values(by=["silhouette", "inertia", "k"],
                         ascending=[False, True, True])
            .iloc[0]
            .to_dict())

    # save
    all_csv = os.path.join(out_dir, f"{model_name}_all_silhouette_results.csv")
    best_json = os.path.join(out_dir, f"{model_name}_best_scaler_k.json")
    res_df.to_csv(all_csv, index=False)
    with open(best_json, "w") as f:
        json.dump(best, f, indent=2)

    logging.info(f"[{model_name}] Saved all results -> {all_csv}")
    logging.info(f"[{model_name}] Best config      -> {best_json}")
    logging.info(f"[{model_name}] BEST => scaler={best['scaler']} | k={best['k']} | "
                 f"silhouette={best['silhouette']:.4f}")

    return res_df, best

def main():
    parser = argparse.ArgumentParser("Auto pick (scaler, k) by silhouette")
    parser.add_argument("--meta-file", required=True,
                        help="Path to meta_arima.csv or meta_lstm.csv")
    parser.add_argument("--out-dir", default="data/processed_folds/auto_k",
                        help="Directory to save results")
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--pca-var", type=float, default=0.9,
                        help="Explained variance for StandardScaler+PCA")
    parser.add_argument("--model-name", type=str, default="ARIMA",
                        help="Just a tag for logs/files: ARIMA/LSTM/...")
    args = parser.parse_args()

    run(
        meta_path=args.meta_file,
        out_dir=args.out_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        pca_var=args.pca_var,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()