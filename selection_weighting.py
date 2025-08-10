import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Iterable, Tuple
from sklearn.cluster import KMeans

def pick_representatives(meta_df: pd.DataFrame, X: np.ndarray, km: KMeans, n_per_cluster: int = 1) -> pd.DataFrame:
    df = meta_df.copy()
    reps = []
    for cid, g in df.groupby('cluster'):
        reps.append(g.sort_values('distance_to_centroid').head(n_per_cluster))
    return pd.concat(reps, ignore_index=True) if reps else pd.DataFrame(columns=df.columns)


def ensure_minimum_per_cluster(reps_df: pd.DataFrame, meta_df: pd.DataFrame, min_per_cluster: int = 1) -> pd.DataFrame:
    if reps_df.empty:
        return reps_df
    existing = reps_df['cluster'].value_counts().to_dict()
    adds = []
    for cid, g in meta_df.groupby('cluster'):
        need = max(0, min_per_cluster - existing.get(cid, 0))
        if need > 0:
            extra = g.sort_values('distance_to_centroid').head(need)
            extra = extra.assign(forced=True, is_initial=False)
            adds.append(extra)
    if adds:
        reps_df = pd.concat([reps_df] + adds, ignore_index=True).drop_duplicates(subset=['fold_id'])
    return reps_df


def select_top_folds_no_cluster_overlap(df: pd.DataFrame, n_per_ticker: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    for ticker, g in df.groupby('ticker'):
        seen = set()
        picks = []
        for _, r in g.sort_values('distance_to_centroid').iterrows():
            if r['cluster'] not in seen:
                picks.append(r)
                seen.add(r['cluster'])
            if len(picks) >= n_per_ticker:
                break
        if len(picks) < n_per_ticker:
            logging.warning(f"[select] ticker={ticker}: only {len(picks)}/{n_per_ticker} due to cluster overlap.")
        out.extend(picks)
    return pd.DataFrame(out)


def _available_cols(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[str, ...]:
    return tuple(c for c in cols if c in df.columns)


def compute_cluster_stability_weight(meta_df: pd.DataFrame) -> Dict[int, float]:
    pref_cols = ('volatility_std','kurtosis','rolling_vol_max')
    cols = _available_cols(meta_df, pref_cols)
    if not cols:
        w = {cid: 1.0 for cid in meta_df['cluster'].unique()}
        return w

    agg = meta_df.groupby('cluster')[list(cols)].mean().reset_index()
    norm = agg.copy()
    for c in cols:
        v = agg[c].values.astype(float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            norm[c] = 0.5
        else:
            norm[c] = (v - vmin) / (vmax - vmin)
    weights = { 'volatility_std': 0.5, 'kurtosis': 0.3, 'rolling_vol_max': 0.2 }
    total = 0.0
    cw = np.zeros(len(norm))
    for c in cols:
        cw += weights.get(c, 0.0) * norm[c].values
        total += weights.get(c, 0.0)
    if total <= 0:
        cw = np.zeros(len(norm))
        total = 1.0
    raw = 1.0 - (cw / total)
    raw = np.clip(raw, 0.1, 1.0)
    return dict(zip(agg['cluster'].tolist(), raw.tolist()))

def refine_representative_folds(
    meta_df_with_clusters: pd.DataFrame,
    desired_tickers: Iterable[str],
    n_per_ticker: int,
    min_per_cluster: int,
    distance_quantile: float,
    force_quantile: float,
    global_weighting: bool = True,
) -> pd.DataFrame:
    df = meta_df_with_clusters.copy()

    # Quantile-based thresholds
    if df.empty:
        return df
    dist = df['distance_to_centroid'].values
    dist_thr = float(np.quantile(dist, distance_quantile))
    force_thr = float(np.quantile(dist, force_quantile))

    initial = df[df['distance_to_centroid'] <= dist_thr].copy()
    initial['forced'] = False
    initial['is_initial'] = True

    selected_tickers = set(initial['ticker'].unique())
    missing = set(desired_tickers) - selected_tickers

    cluster_forced_count = defaultdict(int)
    forced_rows = []

    for t in missing:
        cand = df[df['ticker'] == t].sort_values('distance_to_centroid')
        for _, r in cand.iterrows():
            if r['distance_to_centroid'] <= force_thr:
                forced_rows.append(r.assign(forced=True, is_initial=False))
                break

    reps = pd.concat([initial] + ([pd.DataFrame(forced_rows)] if forced_rows else []), ignore_index=True)
    reps = ensure_minimum_per_cluster(reps, df, min_per_cluster=min_per_cluster)

    # No-overlap per ticker
    reps = select_top_folds_no_cluster_overlap(reps, n_per_ticker=n_per_ticker)

    # Weights: distance + cluster stability
    if reps.empty:
        return reps

    all_d = df['distance_to_centroid'] if global_weighting else reps['distance_to_centroid']
    dmin, dmax = float(all_d.min()), float(all_d.max())
    if dmax > dmin:
        reps['dist_weight'] = 1.0 - (reps['distance_to_centroid'] - dmin) / (dmax - dmin + 1e-6)
    else:
        reps['dist_weight'] = 1.0

    cweights = compute_cluster_stability_weight(df)
    reps['cluster_weight'] = reps['cluster'].map(cweights)

    # Optional softmax similarity instead of linear
    tau = np.median(df['distance_to_centroid']) + 1e-6
    sims = np.exp(-(reps['distance_to_centroid']) / tau)
    reps['weight'] = (0.5 * reps['dist_weight'] + 0.5 * sims) * reps['cluster_weight']

    cols = ['fold_id','ticker','cluster','cluster_size','distance_to_centroid','forced','weight']
    return reps.sort_values(['ticker','fold_id'])[cols]