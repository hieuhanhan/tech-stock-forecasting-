#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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

def scale_robust(X, **kwargs):
    scaler = RobustScaler()
    return scaler.fit_transform(X), {"type": "robust"}

def scale_standard_pca(X, pca_var=0.9, **kwargs):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=pca_var, random_state=42)
    Xp = pca.fit_transform(Xs)
    return Xp, {"type": "standard+pca", "pca_var": pca_var, "n_components": int(pca.n_components_)}

SCALERS = {
    "raw": scale_raw,
    "minmax": scale_minmax,
    "standard": scale_standard,
    "robust": scale_robust,
    "standard_pca": scale_standard_pca,
}

def try_kmeans(X: np.ndarray, k: int, min_cluster_size: int = 3) -> dict:
    """
    Fit KMeans(k) and return a dict of metrics.
    If silhouette can't be computed or cluster sizes are too small, silhouette = -1.
    """
    n = X.shape[0]
    if k < 2 or n <= k:
        return {"silhouette": -1.0, "inertia": np.nan, "db": np.nan, "ch": np.nan, "valid": False}

    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(X)

    uniques, counts = np.unique(labels, return_counts=True)
    if len(uniques) < 2 or counts.min() < min_cluster_size:
        return {"silhouette": -1.0, "inertia": float(km.inertia_), "db": np.nan, "ch": np.nan, "valid": False}

    sil = silhouette_score(X, labels)
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan

    return {"silhouette": float(sil), "inertia": float(km.inertia_), "db": db, "ch": ch, "valid": True}

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
    n = X.shape[0]

    # cap k so average cluster size >= 3
    k_cap = min(k_max, max(k_min, n // 3 if n >= 9 else max(2, n // 2)))
    if k_cap < k_min:
        k_cap = k_min

    results = []

    for scaler_name, scaler_fn in SCALERS.items():
        logging.info(f"[{model_name}] Evaluating scaler='{scaler_name}' ...")
        X_scaled, scaler_info = scaler_fn(X, pca_var=pca_var)

        for k in range(k_min, k_cap + 1):
            mets = try_kmeans(X_scaled, k, min_cluster_size=3)
            row = {
                "model": model_name,
                "scaler": scaler_name,
                "k": k,
                **scaler_info,
                **mets
            }
            results.append(row)
            logging.info(
                f"[{model_name}] scaler={scaler_name:14s} k={k:2d} "
                f"sil={row['silhouette']: .4f} db={row['db'] if not np.isnan(row['db']) else np.nan} "
                f"ch={row['ch'] if not np.isnan(row['ch']) else np.nan} inertia={row['inertia']: .2f}"
            )

    res_df = pd.DataFrame(results)

    # choose best: silhouette desc → DB asc → CH desc → inertia asc → k asc
    best = (res_df
            .sort_values(
                by=["silhouette", "db", "ch", "inertia", "k"],
                ascending=[False, True, False, True, True]
            )
            .iloc[0]
            .to_dict())

    # save
    all_csv = os.path.join(out_dir, f"{model_name}_all_silhouette_results.csv")
    best_json = os.path.join(out_dir, f"{model_name}_best_scaler_k.json")
    res_df.to_csv(all_csv, index=False)
    with open(best_json, "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best.items()}, f, indent=2)

    logging.info(f"[{model_name}] Saved all results -> {all_csv}")
    logging.info(f"[{model_name}] Best config      -> {best_json}")
    logging.info(
        f"[{model_name}] BEST => scaler={best['scaler']} | k={int(best['k'])} | "
        f"silhouette={best['silhouette']:.4f} | db={best['db']} | ch={best['ch']}"
    )

    return res_df, best

def main():
    parser = argparse.ArgumentParser("Auto pick (scaler, k) by multiple cluster metrics")
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