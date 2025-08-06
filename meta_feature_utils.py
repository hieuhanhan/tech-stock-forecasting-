import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_trend_slope(close_series):
    if close_series.isnull().any():
        return np.nan
    try:
        return np.polyfit(np.arange(len(close_series)), np.log1p(close_series), 1)[0]
    except Exception:
        return np.nan
    
def compute_acf(series, lag=1):
    if len(series) <= lag or series.isnull().any():
        return np.nan
    return series.autocorr(lag=lag)

def compute_meta_statistics(df_val):
    log_ret = df_val['Log_Returns'].dropna()
    return {
        'positive_ratio': df_val['target'].mean(),
        'volatility_std': log_ret.std(),
        'acf1': log_ret.autocorr(lag=1) if len(log_ret) > 1 else np.nan,
        'acf2': log_ret.autocorr(lag=2) if len(log_ret) > 2 else np.nan,
        'trend_slope': compute_trend_slope(df_val['Close_raw']),
        'skewness': log_ret.skew(),
        'kurtosis': log_ret.kurtosis()
    }

def compute_meta_with_pca(df_val):
    meta = {
        'positive_ratio': df_val['target'].mean(),
        'volatility_std': df_val['Log_Returns'].std(),
        'trend_slope': compute_trend_slope(df_val['Close_raw'])
    }
    for col in df_val.columns:
        if col.startswith("PC") and not df_val[col].isnull().values.any():
            meta[f'{col}_mean'] = df_val[col].mean()
            meta[f'{col}_std'] = df_val[col].std()
    return meta

def evaluate_kmeans(X, k, model_name):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    score = silhouette_score(X, km.labels_) if len(np.unique(km.labels_)) > 1 else -1
    logging.info(f"[{model_name}] k={k}: inertia={km.inertia_:.2f}, silhouette={score:.3f}")
    return km, score, X

def find_best_k(meta_df, model_name, k_min=5, k_max=25):
    if meta_df.empty:
        logging.warning(f"[{model_name}] No meta-features available for k search.")
        return k_min

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(meta_df.drop(columns=['fold_id', 'ticker']))
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
        return pd.DataFrame(columns=['fold_id', 'ticker', 'cluster', 'cluster_size', 'distance_to_centroid']), meta_df

    imputer = SimpleImputer(strategy='mean')
    feature_cols = meta_df.drop(columns=['fold_id', 'ticker'], errors='ignore')
    X_scaled = imputer.fit_transform(feature_cols)

    km, score, _ = evaluate_kmeans(X_scaled, k, model_name)

    if score < 0.3:
        logging.warning(f"[{model_name}] Silhouette < 0.3 → Consider increasing k or n_per_cluster.")

    meta_df = meta_df.copy()
    meta_df['cluster'] = km.labels_
    meta_df['cluster_size'] = meta_df.groupby('cluster')['cluster'].transform('count')

    reps = []
    for cluster_id in range(k):
        cluster_points = meta_df[meta_df['cluster'] == cluster_id]
        cluster_X = X_scaled[meta_df['cluster'] == cluster_id]

        center = km.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_X - center, axis=1)
        cluster_points = cluster_points.copy()
        cluster_points['distance_to_centroid'] = dists

        sorted_idx = np.argsort(dists)[:n_per_cluster]
        reps.append(cluster_points.iloc[sorted_idx])

    reps_df = pd.concat(reps).reset_index(drop=True)
    return reps_df, meta_df

def load_and_filter_val_fold(val_meta_path, model_type):
    if not os.path.exists(val_meta_path):
        return None, 'missing_meta_file'

    try:
        df_val = pd.read_csv(val_meta_path)
    except Exception as e:
        return None, f'read_error: {str(e)}'

    required_cols = ['target', 'Log_Returns', 'Close_raw']
    if not all(col in df_val.columns for col in required_cols):
        return None, 'missing_required_columns'

    if df_val['target'].mean() < 0.1 or df_val['target'].mean() > 0.6:
        return None, 'label_imbalance'

    if df_val['Log_Returns'].std() < 1e-4:
        return None, 'flat_volatility'

    if df_val['Close_raw'].isnull().values.any():
        return None, 'missing_close'

    trend_slope = compute_trend_slope(df_val['Close_raw'])
    if abs(trend_slope) < 1e-5:
        return None, 'flat_trend'

    return df_val, None

def refine_representative_folds(reps_df, meta_df, distance_threshold=0.8, force_distance_limit=50.0):
    desired_tickers = {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}

    reps_df = reps_df[reps_df['distance_to_centroid'] <= distance_threshold].copy()
    reps_df['forced'] = False
    selected_tickers = set(reps_df['ticker'].unique())

    missing_tickers = desired_tickers - selected_tickers
    additions = []

    for ticker in missing_tickers:
        candidates = meta_df[meta_df['ticker'] == ticker].copy()

        if 'distance_to_centroid' not in candidates.columns:
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(candidates.drop(columns=['fold_id', 'ticker'], errors='ignore'))
            center = X.mean(axis=0)
            distances = np.linalg.norm(X - center, axis=1)
            candidates['distance_to_centroid'] = distances

        if not candidates.empty:
            best = candidates.sort_values('distance_to_centroid').iloc[0]
            best = best.copy()

            if best['distance_to_centroid'] > distance_threshold:
                if best['distance_to_centroid'] <= force_distance_limit:
                    logging.warning(f"[forced] Added {ticker} with distance {best['distance_to_centroid']:.3f} > threshold but within force limit")
                    best['forced'] = True
                    additions.append(best)
                else:
                    logging.warning(f"[refine] Skipping {ticker} due to distance {best['distance_to_centroid']:.3f} > force limit ({force_distance_limit})")
            else:
                best['forced'] = False
                additions.append(best)

    if additions:
        reps_df = pd.concat([reps_df, pd.DataFrame(additions)], ignore_index=True)

        reps_df = (
            reps_df.sort_values(by='distance_to_centroid')
                   .groupby('ticker', as_index=False)
                   .head(2)
        )

    if 'cluster' not in meta_df.columns or 'cluster_size' not in meta_df.columns:
        logging.warning("[refine] 'cluster' or 'cluster_size' missing in meta_df → skipping cluster merge")
        cluster_info = pd.DataFrame(columns=['fold_id', 'cluster', 'cluster_size'])
    else:
        cluster_info = meta_df[['fold_id', 'cluster', 'cluster_size']]

    reps_df = reps_df.drop(columns=['cluster', 'cluster_size'], errors='ignore')
    reps_df = reps_df.merge(cluster_info, on='fold_id', how='left')

    if 'forced' not in reps_df.columns:
        reps_df['forced'] = False
    
    max_dist = reps_df['distance_to_centroid'].max()
    min_dist = reps_df['distance_to_centroid'].min()
    reps_df['weight'] = reps_df['distance_to_centroid'].apply(
        lambda d: 1.0 - (d - min_dist) / (max_dist - min_dist + 1e-6))
    
    reps_df.loc[reps_df['forced'], 'weight'] *= 0.5  

    return reps_df[['fold_id', 'ticker', 'cluster', 'cluster_size', 'distance_to_centroid', 'forced', 'weight']].sort_values(by=['ticker'])
