import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs("figures", exist_ok=True)

# ========== Load Equity Curve ==========
df = pd.read_csv("data/backtest_curves/portfolio_equity_curve.csv")
df["cumulative_return"] = np.exp(np.cumsum(df["portfolio_return"])) - 1

# ========== Plot Equity Curve ==========
plt.figure(figsize=(10, 5))
plt.plot(df["cumulative_return"], label="Portfolio", color="blue")
plt.title("Portfolio Equity Curve")
plt.xlabel("Day")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/portfolio_equity_curve.png")
plt.show()

# ========== Metrics ==========
sharpe = (df["portfolio_return"].mean() / df["portfolio_return"].std()) * np.sqrt(252)
mdd = ((np.maximum.accumulate(np.exp(np.cumsum(df["portfolio_return"]))) - np.exp(np.cumsum(df["portfolio_return"]))) / np.maximum.accumulate(np.exp(np.cumsum(df["portfolio_return"])))).max()
cum_return = np.exp(df["portfolio_return"].sum()) - 1

print(f"Sharpe Ratio      : {sharpe:.4f}")
print(f"Max Drawdown      : {mdd:.4f}")
print(f"Cumulative Return : {cum_return:.4f}")

# ========== Load Summary ==========
df_summary = pd.read_csv("data/backtest_results/portfolio_backtest_summary.csv")
print(df_summary.sort_values("sharpe", ascending=False)[["ticker", "sharpe", "max_drawdown"]])

# ========== Barplot Sharpe ==========
plt.figure(figsize=(8, 5))
sns.barplot(data=df_summary.sort_values("sharpe", ascending=False), x="ticker", y="sharpe", palette="Blues_d")
plt.title("Sharpe Ratio per Ticker")
plt.xlabel("Ticker")
plt.ylabel("Sharpe Ratio")
plt.tight_layout()
plt.savefig("figures/sharpe_per_ticker.png")
plt.show()