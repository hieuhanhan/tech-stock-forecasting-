import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ─── CONFIG ────────────────────────────────────────────────
DATA_DIR        = "data/processed_folds"
TIER2_JSON      = "data/tuning_results/jsons/tier2_arima.json"
FOLDS_SUMMARY   = os.path.join(DATA_DIR, "folds_summary.json")
RETRAIN_INTERVAL= 5
TRANSACTION_COST= 0.0005 + 0.0001  # BASE_COST + SLIPPAGE

# ─── UTILS ────────────────────────────────────────────────
def sharpe_ratio(r):
    std = np.std(r)
    return 0.0 if std == 0 else (np.mean(r) / std) * np.sqrt(252)

def max_drawdown(r):
    cum = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(cum)
    return np.max((peak - cum) / (peak + np.finfo(float).eps))

# ─── LOAD RESULTS & METADATA ──────────────────────────────
with open(TIER2_JSON) as f:
    tier2 = json.load(f)
df_t2 = pd.DataFrame(tier2)  # columns: fold_id, pareto_front

df_folds = pd.read_json(FOLDS_SUMMARY)
df_folds = df_folds.rename(columns={"global_fold_id":"fold_id"})

# ─── REBUILD BLOCK RETURNS & DATES ────────────────────────
all_blocks = []  # list of (date, return) pairs

for rec in tier2:
    fid = rec["fold_id"]
    front = rec["pareto_front"]
    # pick champion point = max sharpe, then min mdd
    best = max(front, key=lambda p: p["sharpe"])
    p,q,th = best["p"], best["q"], best["threshold"]

    # load val series with dates
    fold_meta = df_folds[df_folds["fold_id"]==fid].iloc[0]
    val_path = os.path.join(DATA_DIR, fold_meta["val_path_arima_prophet"])
    df_val   = pd.read_csv(val_path, parse_dates=["Date"])
    returns  = df_val["Log_Returns"].values
    dates    = df_val["Date"].values

    history = pd.read_csv(
        os.path.join(DATA_DIR, fold_meta["train_path_arima_prophet"]), 
        parse_dates=["Date"]
    )["Log_Returns"].dropna().values

    n = len(returns)
    for start in range(0, n, RETRAIN_INTERVAL):
        end = min(start+RETRAIN_INTERVAL, n)
        # refit
        model = ARIMA(history, order=(p,0,q)).fit()
        h = end - start
        fc = model.forecast(steps=h)
        # threshold on forecast
        thresh_val = np.quantile(fc, th)
        sig = (fc > thresh_val).astype(int)
        # block returns
        block_r = returns[start:end]*sig - TRANSACTION_COST*sig
        block_dates = dates[start:end]
        all_blocks.append(pd.DataFrame({
            "Date": block_dates,
            "Return": block_r
        }))
        # expand history
        history = np.concatenate([history, returns[start:end]])

# ─── CONCAT & CALC METRICS ────────────────────────────────
df_blocks = pd.concat(all_blocks).sort_values("Date").reset_index(drop=True)
df_blocks["Equity"] = np.exp(np.cumsum(df_blocks["Return"]))  # equity curve

# rolling drawdown
peak = df_blocks["Equity"].cummax()
df_blocks["Drawdown"] = (peak - df_blocks["Equity"]) / peak

# overall Sharpe & max drawdown
overall_sr  = sharpe_ratio(df_blocks["Return"].values)
overall_mdd = df_blocks["Drawdown"].max()

print(f"Overall walk-forward Sharpe: {overall_sr:.3f}")
print(f"Overall walk-forward Max Drawdown: {overall_mdd:.3%}")

# ─── PLOT ─────────────────────────────────────────────────
fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)

# 1) Equity curve
ax[0].plot(df_blocks["Date"], df_blocks["Equity"], label="Equity")
ax[0].set_ylabel("Equity (cum. exp returns)")
ax[0].legend()

# 2) Drawdown
ax[1].plot(df_blocks["Date"], df_blocks["Drawdown"], color="tab:red", label="Drawdown")
ax[1].set_ylabel("Drawdown")
ax[1].set_xlabel("Date")
ax[1].legend()

fig.suptitle("Walk-forward ARIMA: Equity & Drawdown")
fig.tight_layout(rect=[0,0,1,0.95])
plt.show()