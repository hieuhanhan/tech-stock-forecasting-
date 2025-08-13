import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 

TUNING_RESULTS = "data/tuning_results/jsons/tier1_arima.json"

with open(TUNING_RESULTS, "r") as f:
    results = json.load(f)

save_dir = Path("data/tuning_results/figures/arima")
save_dir.mkdir(parents=True, exist_ok=True)

for r in results:
    fold_id = r["fold_id"]
    df = pd.DataFrame(r["top_ga"]) 

    pivot = df.pivot(index="q", columns="p", values="rmse")

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Fold {fold_id} - ARIMA (p,q) RMSE Heatmap")
    plt.xlabel("p")
    plt.ylabel("q")
    plt.tight_layout()

    save_path = save_dir / f"fold_{fold_id}_heatmap.png"
    plt.savefig(save_path)
    plt.close()

print(f"[DONE] Heatmaps saved in '{save_dir}' directory.")