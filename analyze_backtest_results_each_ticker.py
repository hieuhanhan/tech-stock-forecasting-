import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

# ========== CONFIG ==========
TRANSACTION_COST = 0.0005
RETRAIN_INTERVAL = 20

CONFIG_PATH = "data/backtest_configs/knee_point_per_fold.csv"
TEST_PATH = "data/global_scaled/test_set_scaled.csv"
SUMMARY_OUTPUT = "data/backtest_results/portfolio_backtest_summary.csv"
CURVE_OUTPUT = "data/backtest_curves/portfolio_equity_curve.csv"

os.makedirs(os.path.dirname(SUMMARY_OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(CURVE_OUTPUT), exist_ok=True)

# ========== METRICS ==========
def calculate_sharpe_ratio(returns):
    if np.std(returns) == 0: return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

def calculate_max_drawdown(returns):
    if len(returns) == 0: return 0.0
    cumulative = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    return np.max(drawdown)

# ========== LOAD DATA ==========
configs = pd.read_csv(CONFIG_PATH).to_dict(orient="records")
df_test = pd.read_csv(TEST_PATH)

summary_records = []
all_curve_chunks = []

for cfg in tqdm(configs):
    ticker = cfg["ticker"]
    p, q, threshold = int(cfg["p"]), int(cfg["q"]), float(cfg["threshold"])

    df_ticker = df_test[df_test["Ticker"] == ticker].copy()
    actual_returns = df_ticker["Log_Returns"].values
    preds = []

    for i in range(len(actual_returns)):
        if i < RETRAIN_INTERVAL:
            preds.append(0.0)
            continue
        try:
            model = ARIMA(actual_returns[i-RETRAIN_INTERVAL:i], order=(p, 0, q)).fit()
            pred = model.forecast(steps=1)[0]
        except:
            pred = 0.0
        preds.append(pred)

    preds = np.array(preds)
    quantile_value = np.quantile(preds, threshold)
    signals = (preds > quantile_value).astype(int)
    net_returns = (actual_returns * signals) - (signals * TRANSACTION_COST)

    # Per-ticker metrics
    sharpe = calculate_sharpe_ratio(net_returns)
    mdd = calculate_max_drawdown(net_returns)
    cum_return = np.exp(np.sum(net_returns)) - 1

    summary = {
        "ticker": ticker,
        "p": p,
        "q": q,
        "threshold": threshold,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "cumulative_return": cum_return,
        "n_trades": int(np.sum(signals))
    }
    summary_records.append(summary)

    curve_df = pd.DataFrame({
        "day": np.arange(len(net_returns)),
        "ticker": ticker,
        "net_return": net_returns
    })
    all_curve_chunks.append(curve_df)

    pd.DataFrame(summary_records).to_csv(SUMMARY_OUTPUT, index=False)
    pd.concat(all_curve_chunks).to_csv(CURVE_OUTPUT, index=False)

# ========== MERGE ALL CURVES ==========
df_curves = pd.concat(all_curve_chunks)

# Avoid pivot error by summing over duplicates
df_grouped = df_curves.groupby(['day', 'ticker']).sum().reset_index()
df_pivot = df_grouped.pivot(index='day', columns='ticker', values='net_return').fillna(0)

# Portfolio-level returns (equal weight)
df_pivot['portfolio_return'] = df_pivot.mean(axis=1)

# ========== Portfolio Metrics ==========
portfolio_returns = df_pivot['portfolio_return'].values
portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
portfolio_mdd = calculate_max_drawdown(portfolio_returns)
portfolio_cum_return = np.exp(np.sum(portfolio_returns)) - 1

print(f"\n[Portfolio Metrics]")
print(f"Sharpe Ratio : {portfolio_sharpe:.4f}")
print(f"Max Drawdown : {portfolio_mdd:.4f}")
print(f"Cumulative Return : {portfolio_cum_return:.4f}")

# ========== SAVE ==========
df_pivot.to_csv(CURVE_OUTPUT, index=True)
print(f"[INFO] Saved summary to : {SUMMARY_OUTPUT}")
print(f"[INFO] Saved equity curve to : {CURVE_OUTPUT}")