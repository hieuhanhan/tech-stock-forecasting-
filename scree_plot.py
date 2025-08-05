import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Load scaled data
df = pd.read_csv('data/scaled/global/train_val_scaled.csv')

# Exclude non-feature columns
non_feature_cols = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns']
feature_cols = [c for c in df.columns if c not in non_feature_cols]
X = df[feature_cols]

print("Shape of X:", X.shape)

# Fit full PCA
pca_full = PCA(n_components=None)
pca_full.fit(X)

# Explained variances
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Plot
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
         marker='o', linestyle='--', label='Individual Explained Variance')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
         marker='o', linestyle='-', label='Cumulative Explained Variance')
plt.title('Scree Plot: Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance_ratio) + 1, 1))
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save
scree_path = 'data/figures/pca_scree_plot.png'
os.makedirs(os.path.dirname(scree_path), exist_ok=True)
print("Saving scree plot to:", scree_path)
plt.savefig(scree_path)
plt.show()