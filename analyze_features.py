#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================================
# CONFIG
# ================================
SCALED_FOLDS_DIR = "data/scaled_folds"
OUTPUT_DIR = "data/feature_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "feature_analysis_report.csv")


def calculate_feature_stats(df, feature_cols):
    stats = []
    for col in feature_cols:
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        std_val = np.std(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr_val = q3 - q1
        stats.append((col, std_val, iqr_val))
    stats_df = pd.DataFrame(stats, columns=['feature', 'std', 'iqr'])
    return stats_df


def calculate_pca_loadings(df, feature_cols, n_components=1):
    X = df[feature_cols].fillna(0).values
    if X.shape[0] < 2:
        return pd.DataFrame(columns=['feature', 'pca_loading'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    loadings = pd.Series(pca.components_[0], index=feature_cols)
    return loadings.abs().sort_values(ascending=False).reset_index().rename(
        columns={'index': 'feature', 0: 'pca_loading'}
    )


def analyze_fold(fold_path):
    df = pd.read_csv(fold_path)
    feature_cols = [c for c in df.columns if c not in ['Date', 'Ticker', 'target', 'target_log_returns']]

    if not feature_cols:
        return pd.DataFrame()

    stats_df = calculate_feature_stats(df, feature_cols)
    pca_df = calculate_pca_loadings(df, feature_cols)

    stats_df['std_rank'] = stats_df['std'].rank(ascending=False)
    stats_df['iqr_rank'] = stats_df['iqr'].rank(ascending=False)

    if not pca_df.empty:
        pca_df['pca_rank'] = pca_df['pca_loading'].rank(ascending=False)
        stats_df = stats_df.merge(pca_df[['feature', 'pca_loading', 'pca_rank']], on='feature', how='left')
    else:
        stats_df['pca_loading'] = np.nan
        stats_df['pca_rank'] = np.nan

    stats_df['total_rank'] = stats_df[['std_rank', 'iqr_rank', 'pca_rank']].mean(axis=1, skipna=True)
    top5 = stats_df.sort_values('total_rank').head(5)
    top5['fold'] = os.path.basename(fold_path)
    return top5


def main():
    if not os.path.exists(SCALED_FOLDS_DIR):
        print(f"[ERROR] Folder {SCALED_FOLDS_DIR} does not exist.")
        return

    all_reports = []
    for file_name in os.listdir(SCALED_FOLDS_DIR):
        if not file_name.endswith(".csv"):
            continue
        fold_path = os.path.join(SCALED_FOLDS_DIR, file_name)
        print(f"[INFO] Analyzing {fold_path} ...")
        top_features = analyze_fold(fold_path)
        if not top_features.empty:
            all_reports.append(top_features)

    if all_reports:
        final_report = pd.concat(all_reports).reset_index(drop=True)
        os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
        final_report.to_csv(OUTPUT_REPORT, index=False)
        print(f"[DONE] Saved feature analysis report -> {OUTPUT_REPORT}")

        # Load lại và in DataFrame
        df_report = pd.read_csv(OUTPUT_REPORT)
        print("\n[INFO] Top 10 rows from feature_analysis_report.csv:")
        print(df_report.head(10))
    else:
        print("[WARNING] No features found to analyze.")


if __name__ == "__main__":
    main()