import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict


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

def compute_meta_statistics(df_val, rolling_window=21):
    log_ret = df_val['Log_Returns'].dropna()

    meta = {
        'positive_ratio': df_val['target'].mean(),
        'volatility_std': log_ret.std(),  
        'acf1': log_ret.autocorr(lag=1) if len(log_ret) > 1 else np.nan,
        'acf2': log_ret.autocorr(lag=2) if len(log_ret) > 2 else np.nan,
        'trend_slope': compute_trend_slope(df_val['Close_raw']),
        'skewness': log_ret.skew(),
        'kurtosis': log_ret.kurtosis()
    }

    if len(log_ret) >= rolling_window:
        rolling_vol = log_ret.rolling(window=rolling_window).std()
        meta['rolling_vol_mean'] = rolling_vol.mean()
        meta['rolling_vol_max'] = rolling_vol.max()
    else:
        meta['rolling_vol_mean'] = np.nan
        meta['rolling_vol_max'] = np.nan

    return meta

def compute_meta_with_pca(df_val, rolling_window=21):
    log_ret = df_val['Log_Returns'].dropna()
    
    # Base statistics
    meta = {
        'positive_ratio': df_val['target'].mean(),
        'volatility_std': log_ret.std(),
        'trend_slope': compute_trend_slope(df_val['Close_raw']),
    }

    # Rolling volatility
    if len(log_ret) >= rolling_window:
        rolling_vol = log_ret.rolling(window=rolling_window).std()
        meta['rolling_vol_mean'] = rolling_vol.mean()
        meta['rolling_vol_max'] = rolling_vol.max()
    else:
        meta['rolling_vol_mean'] = np.nan
        meta['rolling_vol_max'] = np.nan

    # PCA-based features
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
        return k_min, None

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(meta_df.drop(columns=['fold_id', 'ticker']))
    best_k, best_score = k_min, -1

    for k in range(k_min, k_max + 1):
        km, score, _ = evaluate_kmeans(X, k, model_name)
        if score > best_score:
            best_k, best_score = k, score
    logging.info(f"[{model_name}] Best k={best_k} with silhouette={best_score:.3f}")
    return best_k, X

def pick_representatives(meta_df, X, k, model_name, n_per_cluster=1):
    if meta_df.empty:
        logging.warning(f"[{model_name}] No meta-features to cluster.")
        return pd.DataFrame(columns=['fold_id', 'ticker', 'cluster', 'cluster_size', 'distance_to_centroid']), meta_df, X, []

    km, score, _ = evaluate_kmeans(X, k, model_name)

    if score < 0.3:
        logging.warning(f"[{model_name}] Silhouette < 0.3 → Consider increasing k or n_per_cluster.")

    meta_df = meta_df.copy()
    meta_df['cluster'] = km.labels_
    meta_df['cluster_size'] = meta_df.groupby('cluster')['cluster'].transform('count')

    meta_dists = []
    for idx, row in meta_df.iterrows():
        fold_id = row['fold_id']
        vec = X[idx]
        centroid = km.cluster_centers_[row['cluster']]
        dist = np.linalg.norm(vec - centroid)
        meta_dists.append(dist)
    meta_df['distance_to_centroid'] = meta_dists

    reps = []
    for cluster_id in range(k):
        cluster_points = meta_df[meta_df['cluster'] == cluster_id].copy()
        sorted_df = cluster_points.sort_values('distance_to_centroid').head(n_per_cluster)
        reps.append(sorted_df)

    reps_df = pd.concat(reps).reset_index(drop=True)
    return reps_df, meta_df, X, km.cluster_centers_

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

def select_top_folds_no_cluster_overlap(df, n_per_ticker=2):
    result_rows = []
    MAX_PER_CLUSTER = {0: 3,  1: 1, 4: 1}

    for ticker, group in df.groupby('ticker'):
        seen_clusters = set()
        ticker_rows = []
        for _, row in group.sort_values('distance_to_centroid').iterrows():
            if row['cluster'] not in seen_clusters:
                ticker_rows.append(row)
                seen_clusters.add(row['cluster'])
            if len(ticker_rows) >= n_per_ticker:
                break
        result_rows.extend(ticker_rows)
    return pd.DataFrame(result_rows)

def refine_representative_folds(
    reps_df,
    meta_df,
    X,
    cluster_centers,
    fold_id_to_index,
    distance_threshold=0.8,
    force_distance_limit=3.0,
    max_forced_per_cluster=1,
    n_per_ticker=2,
    global_weighting=True
):

    desired_tickers = {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}

    meta_df_temp = meta_df.copy()
    meta_dists = []
    for _, row in meta_df_temp.iterrows():
        fold_id = row['fold_id']
        if fold_id in fold_id_to_index:
            vec = X[fold_id_to_index[fold_id]]
            centroid = cluster_centers[row['cluster']]
            dist = np.linalg.norm(vec - centroid)
            meta_dists.append(dist)
        else:
            meta_dists.append(np.nan)
    meta_df_temp['distance_to_centroid'] = meta_dists

    initial_reps = reps_df[reps_df['distance_to_centroid'] <= distance_threshold].copy()
    initial_reps['forced'] = False
    initial_reps['is_initial'] = True

    cluster_initial_count = initial_reps.groupby('cluster').size().to_dict()
    cluster_forced_count = defaultdict(int)

    selected_tickers = set(initial_reps['ticker'].unique())
    all_reps = [initial_reps]

    missing_tickers = desired_tickers - selected_tickers

    for ticker in missing_tickers:
        candidates = meta_df[meta_df['ticker'] == ticker].copy()
        candidates = candidates.sort_values('distance_to_centroid')

        for _, cand in candidates.iterrows():
            cluster_id = cand['cluster']
            dist = cand['distance_to_centroid']

            if dist <= force_distance_limit and cluster_forced_count[cluster_id] < max_forced_per_cluster:
                new_fold = cand.copy()
                new_fold['forced'] = True
                new_fold['is_initial'] = False
                all_reps.append(pd.DataFrame([new_fold]))
                cluster_forced_count[cluster_id] += 1
                logging.warning(f"[forced] Added {ticker} from cluster {cluster_id} with distance {dist:.3f}")
                break

    reps_df = pd.concat(all_reps, ignore_index=True)
    reps_df = select_top_folds_no_cluster_overlap(reps_df, n_per_ticker=n_per_ticker)

    if global_weighting:
        all_dists = meta_df['distance_to_centroid']
        min_dist = all_dists.min()
        max_dist = all_dists.max()
    else:
        min_dist = reps_df['distance_to_centroid'].min()
        max_dist = reps_df['distance_to_centroid'].max()

    if max_dist == min_dist:
        reps_df['weight'] = 1.0
    else:
        reps_df['weight'] = reps_df['distance_to_centroid'].apply(
            lambda d: 1.0 - (d - min_dist) / (max_dist - min_dist + 1e-6)
        )
    reps_df.loc[reps_df['forced'], 'weight'] *= 0.5

    return reps_df[['fold_id', 'ticker', 'cluster', 'cluster_size', 'distance_to_centroid', 'forced', 'weight']].sort_values(by=['ticker', 'fold_id'])
