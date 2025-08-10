import logging
import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances  # <-- thêm

# -----------------------------
# Feature filtering / alignment
# -----------------------------
def filter_and_align_features(
    meta_df: pd.DataFrame,
    model_type: str,
    min_presence: float = 0.9,
    min_std: float = 1e-8,
    corr_threshold: float = 0.98,
    max_pcs: int = 12,
) -> pd.DataFrame:
    if meta_df.empty:
        return meta_df.copy()

    df = meta_df.copy()
    id_cols = [c for c in ["fold_id", "ticker", "Date"] if c in df.columns]

    # chỉ giữ cột numeric + thêm lại id nếu cần
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_df = df[num_cols].copy()
    for c in id_cols:
        if c not in numeric_df.columns and c in df.columns:
            numeric_df[c] = df[c]

    presence = numeric_df.notna().mean()
    present_cols = presence[presence >= min_presence].index.tolist()

    keep_cols = present_cols[:]
    if model_type.lower() == "lstm":
        fixed_pcs = [f"PC{i}_{suf}" for i in range(1, max_pcs + 1) for suf in ("mean", "std")]
        base_cols = ["positive_ratio", "volatility_std", "trend_slope", "rolling_vol_mean", "rolling_vol_max"]
        forced = [c for c in (fixed_pcs + base_cols) if c in numeric_df.columns]
        keep_cols = sorted(set(keep_cols).union(forced))

    keep_cols = [c for c in keep_cols if c not in id_cols]
    cols_final = id_cols + [c for c in keep_cols if c in numeric_df.columns]
    df = numeric_df[cols_final].copy()

    # drop near-constant
    feat_cols = [c for c in df.columns if c not in id_cols]
    if feat_cols:
        stds = df[feat_cols].std(ddof=0).fillna(0.0)
        good_feats = [c for c in feat_cols if stds.get(c, 0.0) > min_std]
        if len(good_feats) >= 1:
            df = df[id_cols + good_feats]

    # drop highly correlated
    feat_cols = [c for c in df.columns if c not in id_cols]
    if len(feat_cols) >= 3:
        corr = df[feat_cols].corr().abs()
        upper_mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        upper = corr.where(upper_mask)
        drop = set()
        for c in upper.columns:
            if c in drop:
                continue
            high = upper.index[upper[c].fillna(0) >= corr_threshold].tolist()
            drop.update(high)
        keep = [c for c in feat_cols if c not in drop]
        if len(keep) >= 1:
            df = df[id_cols + keep]

    return df


# -----------------------------
# Thin folds (stride + dedup)
# -----------------------------
def thin_folds(
    meta_df: pd.DataFrame,
    per_ticker_stride: int = 2,
    eps: float = 0.02,
    sort_col: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    min_rows: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    1) Lấy mỗi 'per_ticker_stride'-th fold theo thời gian cho từng ticker.
    2) Khử gần-trùng lặp bằng khoảng cách cosine < eps (greedy).

    - meta_df: cần có ['fold_id','ticker'] và các cột feature numeric.
    - eps: ngưỡng khoảng cách cosine; 0 = trùng hệt. Tăng eps để khử mạnh tay hơn.
    """
    if meta_df.empty:
        return meta_df.copy()

    df = meta_df.copy()

    # chọn cột sort
    if sort_col is None:
        sort_col = "Date" if "Date" in df.columns else "fold_id"

    # stride theo ticker
    df = (
        df.sort_values(["ticker", sort_col])
          .groupby("ticker", group_keys=False)
          .apply(lambda g: g.iloc[::max(1, per_ticker_stride)])
          .reset_index(drop=True)
    )

    # chọn cột feature
    if feature_cols is None:
        drop_cols = {"fold_id", "ticker", "Date"}
        feature_cols = [c for c in df.columns if c not in drop_cols]

    if not feature_cols:
        if verbose:
            logging.info("[thin_folds] no feature columns; skip dedup")
        return df

    # impute nhẹ để tính distance ổn định
    imp = SimpleImputer(strategy="mean")
    Z = imp.fit_transform(df[feature_cols].astype(float))

    # ma trận khoảng cách cosine
    D = cosine_distances(Z)

    keep_idx = []
    used = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if used[i]:
            continue
        keep_idx.append(i)
        used |= (D[i] < eps)

    thinned = df.iloc[keep_idx].reset_index(drop=True)

    # đảm bảo không khử quá tay
    if len(thinned) < min_rows:
        thinned = df.iloc[:min_rows].reset_index(drop=True)

    if verbose:
        logging.info(
            f"[thin_folds] input={len(meta_df)} after_stride={len(df)} "
            f"after_dedup={len(thinned)} removed={len(df)-len(thinned)} eps={eps}"
        )
    return thinned


# -----------------------------
# Matrix prep (impute + scale)
# -----------------------------
def prepare_X(
    meta_df: pd.DataFrame,
    l2_normalize: bool = False,
) -> Tuple[np.ndarray, SimpleImputer, StandardScaler]:
    cols = [c for c in meta_df.columns if c not in ("fold_id", "ticker", "Date")]
    if not cols:
        return np.empty((0, 0)), SimpleImputer(), StandardScaler()

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(meta_df[cols].astype(np.float64))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)

    if l2_normalize and X.size:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X = X / norms

    return X, imputer, scaler


# -----------------------------
# KMeans evaluation
# -----------------------------
def evaluate_kmeans(
    X: np.ndarray,
    k: int,
    model_name: str,
    min_cluster_size: int = 3,
) -> Tuple[KMeans, float]:
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    km.fit(X)

    try:
        labels = km.labels_
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) > 1 and counts.min() >= min_cluster_size:
            score = silhouette_score(X, labels)
        else:
            score = -1.0
    except Exception:
        score = -1.0

    logging.info(f"[{model_name}] k={k}: inertia={km.inertia_:.2f}, silhouette={score:.3f}")
    return km, float(score)


def find_best_k(
    meta_df: pd.DataFrame,
    model_name: str,
    k_min: int = 5,
    k_max: int = 25,
) -> Tuple[int, np.ndarray, SimpleImputer, StandardScaler]:
    if meta_df.empty:
        logging.warning(f"[{model_name}] No meta-features available for k search.")
        return k_min, np.empty((0, 0)), SimpleImputer(), StandardScaler()

    X, imputer, scaler = prepare_X(meta_df)
    n = len(X)
    if n < max(2, k_min):
        logging.warning(f"[{model_name}] Not enough points ({n}) for k_min={k_min}. Using k={max(1, n)}.")
        return max(1, n), X, imputer, scaler

    k_cap = min(k_max, max(k_min, n // 3 if n >= 9 else max(2, n // 2)))
    best_k, best_score = k_min, -1.0

    for k in range(k_min, min(k_cap, n) + 1):
        km, score = evaluate_kmeans(X, k, model_name, min_cluster_size=3)
        if score > best_score:
            best_k, best_score = k, score

    logging.info(f"[{model_name}] Best k={best_k} with silhouette={best_score:.3f}")
    return best_k, X, imputer, scaler


# -----------------------------
# Attach clusters & distances
# -----------------------------
def assign_clusters_and_distances(meta_df: pd.DataFrame, X: np.ndarray, km: KMeans) -> pd.DataFrame:
    if meta_df.empty or X.size == 0:
        return meta_df.copy()

    out = meta_df.copy()
    labels = km.labels_
    out["cluster"] = labels
    out["cluster_size"] = pd.Series(labels).map(pd.Series(labels).value_counts()).values

    centers = km.cluster_centers_[labels]
    dists = np.linalg.norm(X - centers, axis=1)
    out["distance_to_centroid"] = dists
    return out