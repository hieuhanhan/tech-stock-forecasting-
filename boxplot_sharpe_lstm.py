# Boxplots of realized (backtest) Sharpe Ratios — ARIMA
# - Input: backtest CSV produced by your ARIMA backtest script
#   (columns include: fold_id, retrain_interval, source, test_sharpe, ...)
# - Outputs:
#   1) Fig_5_4a_arima_sharpe_by_interval.png  (per-interval GA vs GA+BO)
#   2) Fig_5_4b_arima_sharpe_overall.png      (overall GA vs GA+BO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- paths ----------------
BACKTEST_CSV = Path("data/backtest_lstm/backtest_lstm_results.csv")
OUT_DIR = Path("results/chap5/backtest_lstm"); OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_A = OUT_DIR/"Fig_5_4a_lstm_sharpe_by_interval.png"
FIG_B = OUT_DIR/"Fig_5_4b_lstm_sharpe_overall.png"

# ---------------- style ----------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# ---------------- load ----------------
df = pd.read_csv(BACKTEST_CSV)

# keep only realized backtest Sharpe and valid rows
need_cols = {"fold_id","retrain_interval","source","test_sharpe"}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"Backtest CSV missing columns: {missing}")

# normalize source tags
df["source"] = df["source"].map({"GA_knee":"GA", "GA+BO_knee":"GA+BO"}).fillna(df["source"])
df = df[df["source"].isin(["GA","GA+BO"])].copy()

# ---------------- Figure A: per-interval GA vs GA+BO ----------------
intervals = sorted(df["retrain_interval"].unique())
# Build a list of arrays: for each interval, Sharpe list for GA, then GA+BO
data = []
labels = []
for interval in intervals:
    sub = df[df["retrain_interval"] == interval]
    ga = sub[sub["source"] == "GA"]["test_sharpe"].dropna().to_numpy()
    gabo = sub[sub["source"] == "GA+BO"]["test_sharpe"].dropna().to_numpy()
    if ga.size > 0:
        data.append(ga); labels.append(f"{interval}\n(GA)")
    if gabo.size > 0:
        data.append(gabo); labels.append(f"{interval}\n(GA+BO)")

fig, ax = plt.subplots(figsize=(7.6, 4.6))
bp = ax.boxplot(data, labels=labels, showmeans=True,
                meanline=False, patch_artist=True)

# light coloring for readability
colors = []
for lab in labels:
    colors.append("tab:blue" if "(GA)" in lab else "tab:orange")
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.35)

ax.set_ylabel("Sharpe Ratio (backtest)")
ax.set_title("Sharpe distribution by retraining interval — LSTM (backtest)", fontsize=12, pad=8)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

# prevent label overlap at bottom
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(FIG_A, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved → {FIG_A}")

# ---------------- Figure B: overall GA vs GA+BO ----------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
overall = [df[df["source"]=="GA"]["test_sharpe"].dropna().to_numpy(),
           df[df["source"]=="GA+BO"]["test_sharpe"].dropna().to_numpy()]
bp2 = ax.boxplot(overall, labels=["GA","GA+BO"], showmeans=True, patch_artist=True)
for patch, c in zip(bp2["boxes"], ["tab:blue","tab:orange"]):
    patch.set_facecolor(c); patch.set_alpha(0.35)

ax.set_ylabel("Sharpe Ratio (backtest)")
ax.set_title("Sharpe distribution — GA vs GA+BO (backtest)", fontsize=12, pad=8)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
plt.tight_layout()
fig.savefig(FIG_B, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved → {FIG_B}")