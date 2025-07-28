import json

tier2_path = "data/tuning_results/jsons/tier2_arima.json"
summary_path = "data/scaled_folds/folds_summary_rescaled.json"

with open(summary_path) as f:
    fold_summaries = json.load(f)['arima']
    fold_id_to_ticker = {entry['fold_id']: entry.get('ticker', f"fold_{entry['fold_id']}") for entry in fold_summaries}

with open(tier2_path) as f:
    tier2_results = json.load(f)

for row in tier2_results:
    fid = row['fold_id']
    row['ticker'] = fold_id_to_ticker.get(fid, f"fold_{fid}")

with open(tier2_path, "w") as f:
    json.dump(tier2_results, f, indent=2)

print("Added 'ticker' to each row in Tier-2 JSON.")