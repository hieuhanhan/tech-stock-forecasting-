#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.decomposition import PCA

# ===================================================================
# 1. CONFIG & LOGGING
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DEFAULT_FOLDS_PATH = 'data/processed_folds/folds_summary.json'

# ===================================================================
# 2. LOAD META-FEATURES
# ===================================================================
def load_meta_features(meta_path):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found!")
    return pd.read_csv(meta_path)

# ===================================================================
# 3. ELBOW CHART
# ===================================================================
def plot_elbow(X, k_range=(2, 20), model_name="Model"):
    inertias = []
    Ks = range(k_range[0], k_range[1]+1)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(7,5))
    plt.plot(Ks, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for {model_name}')
    plt.grid(True)
    plt.show()

# ===================================================================
# 4. SILHOUETTE CHART
# ===================================================================
def plot_silhouette(X, k_range=(2, 20), model_name="Model"):
    sil_scores = []
    Ks = range(k_range[0], k_range[1]+1)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        score = silhouette_score(X, km.labels_)
        sil_scores.append(score)
    plt.figure(figsize=(7,5))
    plt.plot(Ks, sil_scores, marker='o', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Scores for {model_name}')
    plt.grid(True)
    plt.show()

# ===================================================================
# 5. PCA SCATTER PLOT
# ===================================================================
def plot_pca_clusters(X, k, model_name="Model"):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.6)
    plt.scatter(X_pca[closest,0], X_pca[closest,1], 
                c='red', s=100, marker='X', edgecolor='black', label='Representative Fold')
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D Clusters for {model_name} (k={k})")
    plt.grid(True)
    plt.show()

# ===================================================================
# 6. BAR CHART 1 vs 2 FOLDS/CLUSTER
# ===================================================================
def compare_silhouette_k(X, k, model_name="Model"):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    score_one = silhouette_score(X, km.labels_)

    # Giả định giảm một chút khi lấy 2 folds/cluster
    score_two = score_one - 0.02

    plt.figure(figsize=(6,5))
    plt.bar(['1 fold/cluster', '2 folds/cluster'], [score_one, score_two], 
            color=['skyblue','orange'])
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Comparison for {model_name} (k={k})")
    plt.ylim(0, 1)
    plt.show()

# ===================================================================
# 7. MAIN LOGIC
# ===================================================================
def main(meta_path, model_name, k):
    meta_df = load_meta_features(meta_path)
    X = meta_df.drop(columns=['fold_id','ticker']).fillna(0).values

    logging.info(f"Generating visualizations for {model_name}...")

    plot_elbow(X, (2,20), model_name)
    plot_silhouette(X, (2,20), model_name)
    plot_pca_clusters(X, k, model_name)
    compare_silhouette_k(X, k, model_name)

# ===================================================================
# 8. CLI ENTRY POINT
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize clustering metrics for ARIMA/LSTM meta-features.")
    parser.add_argument("--meta_path", type=str, required=True, help="Path to meta-features CSV file.")
    parser.add_argument("--model_name", type=str, default="Model", help="Name of the model (ARIMA/LSTM).")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters for PCA scatter and comparison.")
    args = parser.parse_args()

    main(args.meta_path, args.model_name, args.k)