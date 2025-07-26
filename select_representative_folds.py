#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ===================================================================
# 1. CONFIG & LOGGING
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DEFAULT_FOLDS_PATH = 'data/processed_folds/folds_summary.json'
OUTPUT_DIR = 'data/processed_folds'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# 2. META-FEATURE EXTRACTION
# ===================================================================
def compute_acf(series, lag=1):
    return series.autocorr(lag=lag)

def evaluate_kmeans(X, k, model_name):
    """Run KMeans for a given k and return model + silhouette score."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    score = silhouette_score(X, km.labels_) if len(np.unique(km.labels_)) > 1 else -1
    logging.info(f"[{model_name}] k={k}: inertia={km.inertia_:.2f}, silhouette={score:.3f}")
    return km, score

def find_best_k(meta_df, model_name, k_min=5, k_max=25):
    """Find k with best silhouette score."""
    X = meta_df.drop(columns=['fold_id', 'ticker']).fillna(0).values
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        km, score = evaluate_kmeans(X, k, model_name)
        if score > best_score:
            best_k, best_score = k, score
    logging.info(f"[{model_name}] Best k={best_k} with silhouette={best_score:.3f}")
    return best_k

def pick_representatives(meta_df, k, model_name, n_per_cluster=1):
    """Cluster and pick n_per_cluster closest points per cluster."""
    X = meta_df.drop(columns=['fold_id', 'ticker']).fillna(0).values
    km, score = evaluate_kmeans(X, k, model_name)

    if score < 0.3:
        logging.warning(f"[{model_name}] Silhouette < 0.3 → Consider increasing k or n_per_cluster.")

    meta_df['cluster'] = km.labels_
    reps = []
    for cluster_id in range(k):
        cluster_points = meta_df[meta_df['cluster'] == cluster_id]
        reps.append(cluster_points.head(min(n_per_cluster, len(cluster_points))))
    reps_df = pd.concat(reps).reset_index(drop=True)
    return reps_df[['fold_id', 'ticker', 'cluster']]

def build_meta_arima(folds_summary):
    logging.info("[ARIMA] Building meta-features...")
    meta_rows = []
    for fold in folds_summary:
        val_meta_path = os.path.join(OUTPUT_DIR, 'shared_meta',
                                     f"val_meta_fold_{fold['global_fold_id']}.csv")
        if not os.path.exists(val_meta_path):
            continue
        df_val = pd.read_csv(val_meta_path)
        log_ret = df_val['Log_Returns'].dropna()
        if len(log_ret) < 5:
            continue

        meta_rows.append({
            'fold_id': fold['global_fold_id'],
            'ticker': fold['ticker'],
            'volatility_std': log_ret.std(),
            'acf1': compute_acf(log_ret, lag=1),
            'acf2': compute_acf(log_ret, lag=2),
            'trend_slope': np.polyfit(np.arange(len(df_val['Close'])),
                                      np.log1p(df_val['Close']), 1)[0] if len(df_val['Close']) > 1 else 0,
            'skewness': log_ret.skew(),
            'kurtosis': log_ret.kurtosis()
        })
    return pd.DataFrame(meta_rows)

def build_meta_lstm(folds_summary):
    logging.info("[LSTM] Building meta-features...")
    meta_rows = []
    for fold in folds_summary:
        val_path = os.path.join(OUTPUT_DIR, 'lstm', 'val',
                                f"val_fold_{fold['global_fold_id']}.csv")
        if not os.path.exists(val_path):
            continue
        df_val = pd.read_csv(val_path)
        features = df_val.drop(columns=['Date', 'Ticker', 'target', 'target_log_returns'], errors='ignore')
        if features.shape[1] < 2:
            continue

        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(features)
        pca_var1 = PCA(n_components=1, random_state=42).fit(X).explained_variance_ratio_[0]

        meta_rows.append({
            'fold_id': fold['global_fold_id'],
            'ticker': fold['ticker'],
            'feature_corr_mean': np.nanmean(features.corr().abs().values),
            'missing_rate': features.isna().mean().mean(),
            'pca_var1': pca_var1,
            'volatility_std': df_val['Log_Returns'].std()
        })
    return pd.DataFrame(meta_rows)

# ===================================================================
# 3. MAIN LOGIC
# ===================================================================
def main(folds_path, k_arima, k_lstm, evaluate_k, auto_k, n_per_cluster):
    logging.info(f"Loading folds summary from {folds_path}...")
    with open(folds_path, 'r') as f:
        folds_summary = json.load(f)

    meta_arima = build_meta_arima(folds_summary)
    meta_lstm = build_meta_lstm(folds_summary)

    if evaluate_k:
        find_best_k(meta_arima, "ARIMA")
        find_best_k(meta_lstm, "LSTM")
        return

    if auto_k:
        k_arima = find_best_k(meta_arima, "ARIMA")
        k_lstm = find_best_k(meta_lstm, "LSTM")

    arima_reps = pick_representatives(meta_arima, k_arima, "ARIMA", n_per_cluster)
    lstm_reps = pick_representatives(meta_lstm, k_lstm, "LSTM", n_per_cluster)

    arima_path = os.path.join(OUTPUT_DIR, 'arima_tuning_folds.json')
    lstm_path = os.path.join(OUTPUT_DIR, 'lstm_tuning_folds.json')
    arima_reps.to_json(arima_path, orient='records', indent=4)
    lstm_reps.to_json(lstm_path, orient='records', indent=4)
    logging.info(f"[DONE] Saved ARIMA folds -> {arima_path}")
    logging.info(f"[DONE] Saved LSTM folds -> {lstm_path}")

# ===================================================================
# 4. CLI ENTRY POINT
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select representative folds for ARIMA and LSTM.")
    parser.add_argument("--folds_path", type=str, default=DEFAULT_FOLDS_PATH, help="Path to folds_summary.json")
    parser.add_argument("--k_arima", type=int, default=15, help="Number of clusters for ARIMA")
    parser.add_argument("--k_lstm", type=int, default=10, help="Number of clusters for LSTM")
    parser.add_argument("--n_per_cluster", type=int, default=1, help="Number of folds to pick per cluster")
    parser.add_argument("--evaluate_k", action="store_true", help="Evaluate silhouette scores for k=5..25")
    parser.add_argument("--auto_k", action="store_true", help="Automatically pick best k based on silhouette")
    args = parser.parse_args()

    main(args.folds_path, args.k_arima, args.k_lstm, args.evaluate_k, args.auto_k, args.n_per_cluster)