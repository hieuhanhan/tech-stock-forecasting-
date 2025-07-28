import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import argparse

# ===================== CONFIG ==========================
TRANSACTION_COST = 0.0005
RETRAIN_INTERVAL = 20

# ===================== ARGUMENT PARSER =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument('--test-path', default='data/global_scaled/test_set_scaled.csv')
    parser.add_argument('--tuned-json', default='data/json/arima_best_config.json')
    parser.add_argument('--summary-path', default='data/backtest_results/arima_final_backtest.csv')
    parser.add_argument('--curve-path', default='data/backtest_curves/arima_equity_curve.csv')
    return parser.parse_args()

args = parse_args()
TICKER = args.ticker
TEST_DATA_PATH = args.test_path
CONFIG_JSON = args.tuned_json
SUMMARY_PATH = args.summary_path
CURVE_PATH = args.curve_path

os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CURVE_PATH), exist_ok=True)

# ===================== METRICS FUNCTIONS ========================
def calculate_sharpe_ratio(daily_returns):
    if np.std(daily_returns) == 0:
        return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    if len(daily_returns) == 0:
        return 0.0
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return np.max(drawdown)

# ====================== MAIN SCRIPT =============================
def main():
    # --- Step 1: Load config ---
    with open(CONFIG_JSON, 'r') as f:
        config = json.load(f)
    p, q, threshold = int(config['p']), int(config['q']), float(config['threshold'])
    print(f"[INFO] Using ARIMA config: p={p}, d=0, q={q}, threshold={threshold}")

    # --- Step 2: Load test set ---
    df_test = pd.read_csv(TEST_DATA_PATH)
    actual_returns = df_test['Log_Returns'].values

    # --- Step 3: Rolling Forecast (chunked retrain) ---
    preds = []
    for i in range(len(actual_returns)):
        if i < RETRAIN_INTERVAL:
            preds.append(0.0)
            continue

        train_returns = actual_returns[i-RETRAIN_INTERVAL:i]
        try:
            model = ARIMA(train_returns, order=(p, 0, q)).fit()
            pred = model.forecast(steps=1)[0]
        except:
            pred = 0.0
        preds.append(pred)

    preds = np.array(preds)
    signals = (preds > threshold).astype(int)
    net_returns = (actual_returns * signals) - (signals * TRANSACTION_COST)

    # --- Step 4: Metrics ---
    sharpe = calculate_sharpe_ratio(net_returns)
    mdd = calculate_max_drawdown(net_returns)
    cumulative_return = np.exp(np.sum(net_returns)) - 1
    n_trades = int(np.sum(signals))
    trade_ratio = n_trades / len(signals)
    avg_return_per_trade = np.mean(net_returns[signals == 1]) if n_trades > 0 else 0.0

    # --- Step 5: Save summary ---
    summary = {
        'ticker': TICKER,
        'p': p,
        'q': q,
        'threshold': threshold,
        'retrain_interval': RETRAIN_INTERVAL,
        'sharpe': sharpe,
        'max_drawdown': mdd,
        'cumulative_return': cumulative_return,
        'n_trades': n_trades,
        'trade_ratio': trade_ratio,
        'avg_return_per_trade': avg_return_per_trade
    }

    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)
    print(f"[INFO] Final summary saved to: {SUMMARY_PATH}")

    # --- Step 6: Save curve ---
    curve_df = pd.DataFrame({
    'actual_returns': actual_returns,
    'predicted_log_return': preds,
    'signal': signals,
    'net_return': net_returns
    })
    curve_df['ticker'] = TICKER  
    curve_df.to_csv(CURVE_PATH, index=False)
    print(f"[INFO] Equity curve saved to: {CURVE_PATH}")

if __name__ == '__main__':
    main()