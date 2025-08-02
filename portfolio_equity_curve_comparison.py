import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIG ==========
INPUT_CSV = "data/backtest_curves/portfolio_equity_curve.csv"
OUTPUT_FIG = "figures/portfolio_equity_curve_comparison.png"
MAX_RETURN = 0.1  

# ========== LOAD & CLIP ==========
df = pd.read_csv(INPUT_CSV)

# Clipping extreme returns
df["portfolio_return_clipped"] = df["portfolio_return"].clip(lower=-MAX_RETURN, upper=MAX_RETURN)

# Cumulative returns
df["cumulative_return"] = np.exp(np.cumsum(df["portfolio_return"])) - 1
df["cumulative_return_clipped"] = np.exp(np.cumsum(df["portfolio_return_clipped"])) - 1

# ========== METRICS ==========
returns = df["portfolio_return_clipped"].values
sharpe = returns.mean() / returns.std() * np.sqrt(252)
mdd = ((np.maximum.accumulate(np.exp(np.cumsum(returns))) - np.exp(np.cumsum(returns))) / 
       np.maximum.accumulate(np.exp(np.cumsum(returns)))).max()
cum_return = np.exp(np.sum(returns)) - 1

print(f"Sharpe Ratio (clipped)      : {sharpe:.4f}")
print(f"Max Drawdown (clipped)      : {mdd:.4f}")
print(f"Cumulative Return (clipped) : {cum_return:.4f}")

# ========== PLOT ==========
plt.figure(figsize=(12, 6))
plt.plot(df["cumulative_return"], label="Original", linestyle="--", color="blue")
plt.plot(df["cumulative_return_clipped"], label="Clipped", color="green")
plt.title("Comparison of Original vs Clipped Portfolio Equity Curve")
plt.xlabel("Day")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Create output directory
os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
plt.savefig(OUTPUT_FIG)
plt.show()

print(f"[INFO] Plot saved to: {OUTPUT_FIG}")