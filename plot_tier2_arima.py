import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "data/tuning_results/csv/tier2_arima.csv"
df = pd.read_csv(csv_path)

grouped = df.groupby(['fold_id', 'retrain_interval'])
best_by_interval = grouped.agg({'sharpe': 'max', 'mdd': 'min'}).reset_index()

pivot_sharpe = best_by_interval.pivot(index='fold_id', columns='retrain_interval', values='sharpe')
pivot_mdd = best_by_interval.pivot(index='fold_id', columns='retrain_interval', values='mdd')

os.makedirs("figures", exist_ok=True)

# ─── Sharpe Ratio Plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.title("Sharpe Ratio vs. Retrain Interval")
pivot_sharpe.T.plot(marker='o')
plt.xlabel("Retrain Interval")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.legend(title='Fold')
plt.tight_layout()
plt.savefig("figures/sensitivity_sharpe.png")
plt.show()

# ─── Max Drawdown Plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.title("Max Drawdown vs. Retrain Interval")
pivot_mdd.T.plot(marker='o') 
plt.xlabel("Retrain Interval")
plt.ylabel("Max Drawdown")
plt.grid(True)
plt.legend(title='Fold')
plt.tight_layout()
plt.savefig("figures/sensitivity_mdd.png")
plt.show()