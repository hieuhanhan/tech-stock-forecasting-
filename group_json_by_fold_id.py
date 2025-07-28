import json
import pandas as pd
from collections import defaultdict
import os

INPUT_JSON = 'data/tuning_results/jsons/tier2_arima.json'
OUTPUT_JSON = 'data/tuning_results/jsons/tier2_arima_grouped.json'

with open(INPUT_JSON, 'r') as f:
    raw_data = json.load(f)

grouped = defaultdict(list)

# Convert individual entries to pareto front grouped by fold_id
for entry in raw_data:
    fold_id = entry['fold_id']
    grouped[fold_id].append({
        "p": entry["p"],
        "q": entry["q"],
        "threshold": entry["threshold"],
        "sharpe_ratio": entry["sharpe"],
        "max_drawdown": entry["mdd"],
        "retrain_interval": entry["retrain_interval"],
        "ticker": entry.get("ticker", f"fold_{fold_id}")
    })

# Wrap into list with fold metadata
grouped_result = []
for fid, configs in grouped.items():
    ticker = configs[0].get("ticker", f"fold_{fid}")  
    grouped_result.append({
        "fold_id": fid,
        "ticker": ticker,
        "pareto_front": configs
    })

# Save to new JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(grouped_result, f, indent=2)

print(f"Saved grouped JSON to: {OUTPUT_JSON}")