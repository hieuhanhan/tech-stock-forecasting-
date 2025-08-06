#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.decomposition import PCA
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
import json

SCALED_CSV_PATH = 'data/scaled/global/train_val_scaled.csv'
PCA_CSV_PATH = 'data/scaled/global/train_val_scaled_pca.csv'
PCA_MODEL_PATH = 'data/scaled/global/pca_model.pkl'
PCA_META_PATH = 'data/scaled/global/pca_meta.json'
CORRELATION_CHART_PATH = 'data/figures/feature_correlation.png'

df = pd.read_csv(SCALED_CSV_PATH)
print(f"[INFO] Loaded scaled dataset with shape: {df.shape}")


non_feature_cols = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns', 'Close_raw']

feature_cols = [col for col in df.columns if col not in non_feature_cols]
X = df[feature_cols].copy()

print("[INFO] Generating correlation heatmap for original features...")
plt.figure(figsize=(20, 15))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Original Features')
plt.tight_layout()
os.makedirs(os.path.dirname(CORRELATION_CHART_PATH), exist_ok=True)
plt.savefig(CORRELATION_CHART_PATH)
print(f"[INFO] Correlation chart saved to: {CORRELATION_CHART_PATH}")
plt.close()

print("[INFO] Applying PCA to reduce dimensions while retaining 95% variance...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X)

pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.concat([df[non_feature_cols].reset_index(drop=True), pd.DataFrame(X_pca, columns=pca_cols)], axis=1)

os.makedirs(os.path.dirname(PCA_CSV_PATH), exist_ok=True)
df_pca.to_csv(PCA_CSV_PATH, index=False)
dump(pca, PCA_MODEL_PATH)

print(f"[INFO] PCA-applied dataset saved to: {PCA_CSV_PATH}")
print(f"[INFO] PCA model saved to: {PCA_MODEL_PATH}")
print(f"[INFO] Reduced to {X_pca.shape[1]} components from {X.shape[1]} original features.")

with open(PCA_META_PATH, 'w') as f:
    json.dump({
        "original_feature_count": X.shape[1],
        "reduced_feature_count": X_pca.shape[1],
        "explained_variance_ratio": list(pca.explained_variance_ratio_),
        "cumulative_variance": float(pca.explained_variance_ratio_.cumsum()[-1])
    }, f, indent=4)
print(f"[INFO] PCA meta saved to: {PCA_META_PATH}")