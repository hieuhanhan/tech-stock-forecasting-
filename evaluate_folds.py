import matplotlib.pyplot as plt
import pandas as pd
import json
import os

with open('data/processed_folds/folds_summary_global.json') as f:
    folds = json.load(f)

ratios = []
tickers = []

for fold in folds:
    val_path = os.path.join('data/processed_folds', fold['val_path_lstm'])
    df = pd.read_csv(val_path)
    ratio = df['target'].mean()
    ratios.append(ratio)
    tickers.append(fold['ticker'])

plt.figure(figsize=(10, 5))
plt.boxplot(ratios)
plt.title("Distribution of Positive Label Ratio across Validation Folds")
plt.ylabel("Positive Label Ratio")
plt.grid(True)

os.makedirs("figures", exist_ok=True)
output_path = os.path.join("figures", "positive_label_ratio_distribution.png")
plt.savefig(output_path, dpi=300)
plt.close()
print(f"[INFO] Saved plot to '{output_path}'")