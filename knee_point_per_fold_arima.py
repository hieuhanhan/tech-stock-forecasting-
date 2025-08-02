import pandas as pd
import numpy as np
import os

# ── CONFIG ──
tier2_csv = "data/tuning_results/csv/tier2_arima_with_ticker.csv"      
output_csv = "data/backtest_configs/knee_point_per_fold_arima.csv"

# ── LOAD & FILTER ──
df = pd.read_csv(tier2_csv)
df = df[df["retrain_interval"] == 20].copy()

# ── KNEE POINT HELPER ──
def find_knee(df_fold):
    dd = df_fold["mdd"].to_numpy()
    sr = df_fold["sharpe"].to_numpy()
    # normalize each axis
    norm_dd = (dd - dd.min()) / (dd.max() - dd.min()) if dd.max() != dd.min() else np.zeros_like(dd)
    norm_sr = (sr - sr.min()) / (sr.max() - sr.min()) if sr.max() != sr.min() else np.zeros_like(sr)
    # distance to (0 drawdown, 1 sharpe)
    dist = np.sqrt(norm_dd**2 + (1 - norm_sr)**2)
    return df_fold.iloc[np.argmin(dist)]

# ── EXTRACT ONE KNEE PER FOLD ──
records = []
for fold_id, grp in df.groupby("fold_id"):
    knee = find_knee(grp)
    records.append({
        "fold_id": fold_id,
        "ticker": knee["ticker"],
        "p": int(knee["p"]),
        "q": int(knee["q"]),
        "threshold": float(knee["threshold"]),
        "sharpe": float(knee["sharpe"]),
        "max_drawdown": float(knee["mdd"]),
        "retrain_interval": 20
    })

# ── SAVE CSV ──
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(records).to_csv(output_csv, index=False)
print(f"Saved knee-point-per-fold CSV to: {output_csv}")