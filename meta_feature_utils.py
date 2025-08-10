import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict


# ---------- Utility ----------
def compute_trend_slope(close_series):
    if close_series.isnull().any():
        return np.nan
    try:
        return np.polyfit(np.arange(len(close_series)), np.log1p(close_series), 1)[0]
    except Exception:
        return np.nan
    
# ---------- Load & Filter ----------
def load_and_filter_val_fold(
    val_meta_path,
    model_type=None,              
    min_pos=0.05,               
    max_pos=0.70,
    min_vol=5e-5,                
    min_abs_slope=1e-6,           
    soft=False                    
):
    if not os.path.exists(val_meta_path):
        return (None, 'missing_meta_file') if not soft else (pd.DataFrame(), None)

    try:
        df_val = pd.read_csv(val_meta_path)
    except Exception as e:
        return (None, f'read_error: {str(e)}') if not soft else (pd.DataFrame(), None)

    required = ['target', 'Log_Returns', 'Close_raw']
    if not all(c in df_val.columns for c in required):
        return (None, 'missing_required_columns') if not soft else (df_val, None)

    flags = []

    pr = float(df_val['target'].mean())
    if not (min_pos <= pr <= max_pos):
        flags.append('label_imbalance')

    vol = float(df_val['Log_Returns'].std())
    if vol < min_vol:
        flags.append('flat_volatility')

    slope = compute_trend_slope(df_val['Close_raw'])
    if np.isnan(slope) or abs(slope) < min_abs_slope:
        flags.append('flat_trend')

    if model_type == 'lstm':
        pc_cols = [c for c in df_val.columns if c.startswith('PC')]
        if len(pc_cols) == 0:
            flags.append('missing_pca')
        else:
            pc_var = np.nanmean([df_val[c].var() for c in pc_cols])
            if np.isnan(pc_var) or pc_var < 1e-6:
                flags.append('flat_pca')

    if soft:
        df_val = df_val.copy()
        df_val['qc_flags'] = ';'.join(flags) if flags else ''
        return df_val, None

    if flags:
        return None, '+'.join(flags)

    return df_val, None

# ---------- Meta feature computation ----------
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

def compute_meta_with_pca(df_val, rolling_window=21, top_k_pcs=10):
    log_ret = df_val['Log_Returns'].dropna()

    meta = {
        'positive_ratio': float(df_val['target'].mean()),
        'volatility_std': float(log_ret.std()),
        'trend_slope': compute_trend_slope(df_val['Close_raw']),
    }

    if len(log_ret) >= rolling_window:
        rv = log_ret.rolling(window=rolling_window).std()
        meta['rolling_vol_mean'] = float(rv.mean())
        meta['rolling_vol_max']  = float(rv.max())
    else:
        meta['rolling_vol_mean'] = np.nan
        meta['rolling_vol_max']  = np.nan

    pc_cols = sorted([c for c in df_val.columns if c.startswith("PC")],
                     key=lambda x: int(x[2:]))[:top_k_pcs]
    for col in pc_cols:
        s = df_val[col].dropna()
        if len(s):
            meta[f"{col}_mean"] = float(s.mean())
            meta[f"{col}_std"]  = float(s.std())

    if pc_cols:
        meta['PC_mean_all'] = float(np.nanmean([df_val[c].mean() for c in pc_cols]))
        meta['PC_std_all']  = float(np.nanmean([df_val[c].std()  for c in pc_cols]))

    return meta

# ---------- KMeans & clustering ----------
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
    scaler = StandardScaler()
    X = imputer.fit_transform(meta_df.drop(columns=['fold_id', 'ticker']))
    X = scaler.fit_transform(X)

    best_k, best_score, best_model = k_min, -1, None
    for k in range(k_min, k_max + 1):
        km, score, _ = evaluate_kmeans(X, k, model_name)
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    logging.info(f"[{model_name}] Best k={best_k} with silhouette={best_score:.3f}")
    return best_k, X, best_model

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

    # Vectorized distance calculation
    centers_expanded = km.cluster_centers_[meta_df['cluster']]
    meta_df['distance_to_centroid'] = np.linalg.norm(X - centers_expanded, axis=1)

    reps = []
    for cluster_id in range(k):
        cluster_points = meta_df[meta_df['cluster'] == cluster_id].copy()
        sorted_df = cluster_points.sort_values('distance_to_centroid').head(n_per_cluster)
        reps.append(sorted_df)

    reps_df = pd.concat(reps).reset_index(drop=True)
    return reps_df, meta_df, X, km.cluster_centers_

def select_top_folds_no_cluster_overlap(df, n_per_ticker=2):
    selected = []
    for ticker, group in df.groupby('ticker'):
        seen_clusters = set()
        for _, row in group.sort_values('distance_to_centroid').iterrows():
            if row['cluster'] not in seen_clusters and len(selected) < n_per_ticker:
                selected.append(row)
                seen_clusters.add(row['cluster'])
    return pd.DataFrame(selected)

def refine_representative_folds(
    reps_df,
    meta_df,
    X,
    cluster_centers,
    distance_threshold=0.8,
    force_distance_limit=3.0,
    max_forced_per_cluster=1,
    n_per_ticker=2,
    global_weighting=True
):

    desired_tickers = {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}

    meta_df = meta_df.copy()

    # Vectorized distance calculation for all folds
    centers_expanded = cluster_centers[meta_df['cluster']]
    meta_df['distance_to_centroid'] = np.linalg.norm(X - centers_expanded, axis=1)

    initial_reps = reps_df[reps_df['distance_to_centroid'] <= distance_threshold].copy()
    initial_reps['forced'] = False
    initial_reps['is_initial'] = True

    cluster_forced_count = defaultdict(int)
    selected_tickers = set(initial_reps['ticker'].unique())
    all_reps = [initial_reps]

    missing_tickers = desired_tickers - selected_tickers
    for ticker in missing_tickers:
        candidates = meta_df[meta_df['ticker'] == ticker].sort_values('distance_to_centroid')
        for _, cand in candidates.iterrows():
            if cand['distance_to_centroid'] <= force_distance_limit and \
               cluster_forced_count[cand['cluster']] < max_forced_per_cluster:
                new_fold = cand.copy()
                new_fold['forced'] = True
                new_fold['is_initial'] = False
                all_reps.append(pd.DataFrame([new_fold]))
                cluster_forced_count[cand['cluster']] += 1
                logging.warning(f"[forced] Added {ticker} from cluster {cand['cluster']} with distance {cand['distance_to_centroid']:.3f}")
                break

    reps_df = pd.concat(all_reps, ignore_index=True)
    reps_df = select_top_folds_no_cluster_overlap(reps_df, n_per_ticker=n_per_ticker)

    # Weight calculation
    if global_weighting:
        min_dist, max_dist = meta_df['distance_to_centroid'].min(), meta_df['distance_to_centroid'].max()
    else:
        min_dist, max_dist = reps_df['distance_to_centroid'].min(), reps_df['distance_to_centroid'].max()

    if max_dist == min_dist:
        reps_df['weight'] = 1.0
    else:
        reps_df['weight'] = reps_df['distance_to_centroid'].apply(
            lambda d: 1.0 - (d - min_dist) / (max_dist - min_dist + 1e-6)
        )
    reps_df.loc[reps_df['forced'], 'weight'] *= 0.5

    return reps_df[['fold_id', 'ticker', 'cluster', 'cluster_size', 'distance_to_centroid', 'forced', 'weight']].sort_values(by=['ticker', 'fold_id'])