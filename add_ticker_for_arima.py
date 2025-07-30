#!/usr/bin/env python3
import os
import json
import pandas as pd

# ── CONFIGURE YOUR PATHS HERE ──
TIER2_JSON_IN = 'data/tuning_results/jsons/tier2_arima.json'
TIER2_JSON_OUT = 'data/tuning_results/jsons/tier2_arima_with_ticker.json'
TIER2_CSV_IN = 'data/tuning_results/csv/tier2_arima.csv'
TIER2_CSV_OUT = 'data/tuning_results/csv/tier2_arima_with_ticker.csv'
FOLD_SUMMARY_JSON = 'data/processed_folds/folds_summary_global.json' 

# ── LOAD FOLD→TICKER MAPPING ──
with open(FOLD_SUMMARY_JSON, 'r') as f:
    folds = json.load(f)
# assumes each entry in folds is { "global_fold_id": <int>, "ticker": "<symbol>", ... }
fold2ticker = { entry['global_fold_id']: entry['ticker'] for entry in folds }

# ── ENRICH JSON ──
with open(TIER2_JSON_IN, 'r') as f:
    tier2_data = json.load(f)

for row in tier2_data:
    fid = row.get('fold_id')
    row['ticker'] = fold2ticker.get(fid, f"fold_{fid}")

# write enriched JSON
os.makedirs(os.path.dirname(TIER2_JSON_OUT), exist_ok=True)
with open(TIER2_JSON_OUT, 'w') as f:
    json.dump(tier2_data, f, indent=2)
print(f"Wrote enriched Tier-2 JSON to: {TIER2_JSON_OUT}")

# ── ENRICH CSV ──
df = pd.read_csv(TIER2_CSV_IN)
df['ticker'] = df['fold_id'].map(fold2ticker).fillna(df['fold_id'].astype(str).radd('fold_'))

# warn if any unmapped
unmapped = df.loc[df['ticker'].str.startswith('fold_'), 'fold_id'].unique()
if len(unmapped):
    print("Warning: these fold_ids had no ticker in your summary:", unmapped.tolist())

# write enriched CSV
os.makedirs(os.path.dirname(TIER2_CSV_OUT), exist_ok=True)
df.to_csv(TIER2_CSV_OUT, index=False)
print(f"Wrote enriched Tier-2 CSV to:  {TIER2_CSV_OUT}")