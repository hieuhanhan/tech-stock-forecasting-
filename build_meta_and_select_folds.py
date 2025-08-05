#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import load
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

OUTPUT_DIR = 'data/processed_folds'
ARIMA_FOLDS_PATH = 'data/processed_folds/folds_summary_arima_nvda_cleaned.json'
LSTM_FOLDS_PATH = 'data/processed_folds/folds_summary_lstm_nvda_cleaned.json'
ARIMA_VAL_META_PATH = os.path.join(OUTPUT_DIR, 'arima_meta')
LSTM_VAL_META_PATH = os.path.join(OUTPUT_DIR, 'lstm_meta')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARIMA_VAL_META_PATH, exist_ok=True)
os.makedirs(LSTM_VAL_META_PATH, exist_ok=True)
os.makedirs(os.path.dirname(ARIMA_FOLDS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LSTM_FOLDS_PATH), exist_ok=True)

# META-FEATURE EXTRACTION
def compute_acf(series, lag=1):
    if len(series) <= lag or series.isnull().any():
        return np.nan
    return series.autocorr(lag=lag)

def evaluate_kmeans(X, k, model_name):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    score = silhouette_score(X, km.labels_) if len(np.unique(km.labels_)) > 1 else -1
    logging.info(f"[{model_name}] k={k}: inertia={km.inertia_:.2f}, silhouette={score:.3f}")
    return km, score, X

def find_best_k(meta_df, model_name, k_min=5, k_max=25):
    if meta_df.empty:
        logging.warning(f"[{model_name}] No meta-features available for k search.")
        return k_min
    X = meta_df.drop(columns=['fold_id', 'ticker']).fillna(0).values
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        km, score, _ = evaluate_kmeans(X, k, model_name)
        if score > best_score:
            best_k, best_score = k, score
    logging.info(f"[{model_name}] Best k={best_k} with silhouette={best_score:.3f}")
    return best_k

def pick_representatives(meta_df, k, model_name, n_per_cluster=1):
    if meta_df.empty:
        logging.warning(f"[{model_name}] No meta-features to cluster.")
        return pd.DataFrame(columns=['fold_id', 'ticker', 'cluster', 'cluster_size'])

    X = meta_df.drop(columns=['fold_id', 'ticker']).fillna(0).values
    km, score, X_scaled = evaluate_kmeans(X, k, model_name)

    if score < 0.3:
        logging.warning(f"[{model_name}] Silhouette < 0.3 → Consider increasing k or n_per_cluster.")

    meta_df['cluster'] = km.labels_
    cluster_sizes = meta_df.groupby('cluster')['cluster'].transform('count')
    meta_df['cluster_size'] = cluster_sizes

    reps = []
    for cluster_id in range(k):
        cluster_points = meta_df[meta_df['cluster'] == cluster_id]
        cluster_X = X_scaled[meta_df['cluster'] == cluster_id]

        center = km.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_X - center, axis=1)
        sorted_idx = np.argsort(dists)[:n_per_cluster]
        reps_cluster = cluster_points.iloc[sorted_idx].copy()
        reps_cluster['distance_to_centroid'] = dists[sorted_idx]
        reps.append(reps_cluster)

        logging.info(f"[{model_name}] Cluster {cluster_id}: selected {len(reps_cluster)} fold(s), "
                     f"min_distance={dists.min():.4f}, max_distance={dists.max():.4f}")

    reps_df = pd.concat(reps).reset_index(drop=True)
    return reps_df[['fold_id', 'ticker', 'cluster', 'cluster_size', 'distance_to_centroid']]

def build_meta_arima(folds_summary):
    logging.info("[ARIMA] Building meta-features...")
    meta_rows = []
    excluded_folds = []

    for fold in folds_summary:
        fold_id = fold['global_fold_id']
        val_meta_path = os.path.join(ARIMA_VAL_META_PATH, f"val_meta_fold_{fold_id}.csv")
        if not os.path.exists(val_meta_path):
            continue

        df_val = pd.read_csv(val_meta_path)
        if 'Log_Returns' not in df_val or 'target' not in df_val:
            continue

        log_ret = df_val['Log_Returns'].dropna()
        if len(log_ret) < 30:
            excluded_folds.append((fold_id, fold['ticker'], 'too_short'))
            continue

        target_mean = df_val['target'].mean()
        if target_mean < 0.1 or target_mean > 0.6:
            excluded_folds.append((fold_id, fold['ticker'], 'label_imbalance'))
            continue

        volatility = log_ret.std()
        if volatility < 1e-4:
            excluded_folds.append((fold_id, fold['ticker'], 'flat_volatility'))
            continue

        trend_slope = np.polyfit(np.arange(len(df_val['Close'])),
                                 np.log1p(df_val['Close']), 1)[0]
        if 'Close' not in df_val or df_val['Close'].isnull().any():
            excluded_folds.append((fold_id, fold['ticker'], 'missing_close'))
            continue

        if abs(trend_slope) < 1e-5:
            excluded_folds.append((fold_id, fold['ticker'], 'flat_trend'))
            continue

        meta_rows.append({
            'fold_id': fold_id,
            'ticker': fold['ticker'],
            'positive_ratio': target_mean,
            'volatility_std': volatility,
            'acf1': compute_acf(log_ret, lag=1),
            'acf2': compute_acf(log_ret, lag=2),
            'trend_slope': trend_slope,
            'skewness': log_ret.skew(),
            'kurtosis': log_ret.kurtosis()
        })
    if excluded_folds:
        excluded_df = pd.DataFrame(excluded_folds, columns=['fold_id', 'ticker', 'reason'])
        excluded_path = os.path.join(ARIMA_VAL_META_PATH, 'excluded_arima_folds.csv')
        excluded_df.to_csv(excluded_path, index=False)
        logging.info(f"[ARIMA] Excluded folds saved to {excluded_path}")

    logging.info(f"[ARIMA] Excluded {len(excluded_folds)} folds due to imbalance or flat behavior.")
    return pd.DataFrame(meta_rows)

def build_meta_lstm(folds_summary):
    logging.info("[LSTM] Building meta-features...")

    meta_rows = []
    excluded_folds = []

    for fold in folds_summary:
        fold_id = fold['global_fold_id']
        val_path = os.path.join(OUTPUT_DIR, 'lstm', 'val', f"val_fold_{fold_id}.csv")
        if not os.path.exists(val_path):
            continue

        df_val = pd.read_csv(val_path)
        if not all(col in df_val.columns for col in ['target', 'Log_Returns']):
            continue

        target_mean = df_val['target'].mean()
        if target_mean < 0.1 or target_mean > 0.6:
            excluded_folds.append((fold_id, fold['ticker'], 'label_imbalance'))
            continue

        volatility = df_val['Log_Returns'].std()
        if volatility < 1e-4:
            excluded_folds.append((fold_id, fold['ticker'], 'flat_volatility'))
            continue

        pca_cols = [col for col in df_val.columns if col.startswith('pca')]
        if len(pca_cols) == 0:
            continue

        pca_features = df_val[pca_cols].copy()
        if pca_features.isnull().any().any():
            continue

        if 'Close' in df_val and not df_val['Close'].isnull().any():
            trend_slope = np.polyfit(np.arange(len(df_val['Close'])),
                                    np.log1p(df_val['Close']), 1)[0]
        if abs(trend_slope) < 1e-5:
            excluded_folds.append((fold_id, fold['ticker'], 'flat_trend'))
            continue

        meta = {
            'fold_id': fold_id,
            'ticker': fold['ticker'],
            'positive_ratio': target_mean,
            'volatility_std': volatility
        }
        meta.update({col: pca_features[col].mean() for col in pca_cols})
        meta_rows.append(meta)

    if excluded_folds:
        excluded_df = pd.DataFrame(excluded_folds, columns=['fold_id', 'ticker', 'reason'])
        excluded_path = os.path.join(LSTM_VAL_META_PATH, 'excluded_lstm_folds.csv')
        excluded_df.to_csv(excluded_path, index=False)
        logging.info(f"[LSTM] Excluded folds saved to {excluded_path}")

    logging.info(f"[LSTM] Excluded {len(excluded_folds)} folds due to imbalance or flat behavior.")
    return pd.DataFrame(meta_rows)


# MAIN 
def main(folds_path, k_arima, k_lstm, evaluate_k, auto_k, n_per_cluster_arima, n_per_cluster_lstm):
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

    arima_reps = pick_representatives(meta_arima, k_arima, "ARIMA", n_per_cluster_arima)
    lstm_reps = pick_representatives(meta_lstm, k_lstm, "LSTM", n_per_cluster_lstm)

    # Save representative folds
    arima_path = os.path.join(OUTPUT_DIR, 'arima', 'arima_tuning_folds.json')
    lstm_path = os.path.join(OUTPUT_DIR, 'lstm', 'lstm_tuning_folds.json')
    arima_reps.to_json(arima_path, orient='records', indent=4)
    lstm_reps.to_json(lstm_path, orient='records', indent=4)
    logging.info(f"[DONE] Saved ARIMA folds -> {arima_path}")
    logging.info(f"[DONE] Saved LSTM folds -> {lstm_path}")

    # Save meta-features
    meta_arima_path = os.path.join(ARIMA_VAL_META_PATH, 'meta_arima.csv')
    meta_lstm_path = os.path.join(LSTM_VAL_META_PATH, 'meta_lstm.csv')
    meta_arima.to_csv(meta_arima_path, index=False)
    meta_lstm.to_csv(meta_lstm_path, index=False)
    logging.info(f"[SAVE] Meta ARIMA features -> {meta_arima_path}")
    logging.info(f"[SAVE] Meta LSTM features -> {meta_lstm_path}")

# CLI ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select representative folds for ARIMA and LSTM.")
    parser.add_argument("--folds_path", type=str, required=True)
    parser.add_argument("--k_arima", type=int, default=15, help="Number of clusters for ARIMA")
    parser.add_argument("--k_lstm", type=int, default=10, help="Number of clusters for LSTM")
    parser.add_argument("--n_per_cluster_arima", type=int, default=1, help="Number of folds to pick per ARIMA cluster")
    parser.add_argument("--n_per_cluster_lstm", type=int, default=1, help="Number of folds to pick per LSTM cluster")
    parser.add_argument("--evaluate_k", action="store_true", help="Evaluate silhouette scores for k=5..25")
    parser.add_argument("--auto_k", action="store_true", help="Automatically pick best k based on silhouette")
    args = parser.parse_args()

    main(args.folds_path, args.k_arima, args.k_lstm,
         args.evaluate_k, args.auto_k,
         args.n_per_cluster_arima, args.n_per_cluster_lstm)