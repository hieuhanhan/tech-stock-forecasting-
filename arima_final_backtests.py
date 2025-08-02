import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import argparse

# ===================== CONFIG ==========================
TRANSACTION_COST = 0.0005

# ===================== ARGUMENT PARSER =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-csv", default="data/backtest_configs/knee_point_per_fold.csv")
    parser.add_argument("--test-path", default="data/global_scaled/test_set_scaled.csv")
    parser.add_argument("--summary-path", default="data/backtest_results")
    parser.add_argument("--curve-path", default="data/backtest_curves")
    parser.add_argument("--retrain-interval", type=int, default=20)
    parser.add_argument("--threshold-type", default="quantile", choices=["quantile", "fixed"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()

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


def backtest_one_ticker(df_test, p, q, threshold, retrain_interval, threshold_type):
    actual = df_test["Log_Returns"].values
    preds = []

    for i in range(len(actual)):
        if i < retrain_interval:
            preds.append(0.0)
            continue
        try:
            model = ARIMA(actual[i - retrain_interval:i], order=(p, 0, q)).fit()
            preds.append(model.forecast(steps=1)[0])
        except:
            preds.append(0.0)

    preds = np.array(preds)

    if threshold_type == "quantile":
        thr = np.quantile(preds, threshold)
        signals = (preds > thr).astype(int)
    else:
        signals = (preds > threshold).astype(int)

    net = (actual * signals) - (signals * TRANSACTION_COST)

    return {
        "preds": preds,
        "signals": signals,
        "net": net,
        "sharpe": calculate_sharpe_ratio(net),
        "mdd": calculate_max_drawdown(net),
        "cumret": np.exp(np.sum(net)) - 1,
        "n_trades": int(signals.sum())
    }

# ====================== MAIN SCRIPT =============================
def main():
    args = parse_args()
    
    os.makedirs(args.summary_path, exist_ok=True)
    os.makedirs(args.curve_path, exist_ok=True)

    config_df = pd.read_csv(args.config_csv)
    required_columns = {"ticker", "p", "q", "threshold"}
    if not required_columns.issubset(config_df.columns):
        missing = required_columns - set(config_df.columns)
        raise ValueError(f"Missing required columns in config CSV: {missing}")

    df_test = pd.read_csv(args.test_path)

    for _, row in tqdm(config_df.iterrows(), total=len(config_df), desc="Backtesting all tickers"):
        ticker = row["ticker"]
        p, q, thr = int(row["p"]), int(row["q"]), float(row["threshold"])

        out_sum = os.path.join(args.summary_path, f"{ticker}_arima_final_backtest.csv")
        out_curve = os.path.join(args.curve_path, f"{ticker}_equity_curve.csv")

        if (not args.overwrite) and os.path.exists(out_sum) and os.path.exists(out_curve):
            continue

        df_ticker = df_test[df_test["Ticker"] == ticker].copy()
        if df_ticker.empty:
            print(f"[WARN] No test data found for ticker: {ticker}, skipping.")
            continue

        result = backtest_one_ticker(
            df_ticker, p, q, thr,
            retrain_interval=args.retrain_interval,
            threshold_type=args.threshold_type
        )

        # Save summary
        summary = pd.DataFrame([{
            "ticker": ticker,
            "p": p, "q": q, "threshold": thr,
            "retrain_interval": args.retrain_interval,
            "sharpe": result["sharpe"],
            "max_drawdown": result["mdd"],
            "cumulative_return": result["cumret"],
            "n_trades": result["n_trades"],
            "trade_ratio": result["n_trades"] / len(result["signals"]) if len(result["signals"]) else 0.0,
            "avg_return_per_trade": np.mean(result["net"][result["signals"] == 1]) if result["n_trades"] > 0 else 0.0
        }])
        summary.to_csv(out_sum, index=False)

        # Save curve
        curve = pd.DataFrame({
            "day": np.arange(len(result["net"])),
            "ticker": ticker,
            "actual_returns": df_ticker["Log_Returns"].values,
            "predicted_log_return": result["preds"],
            "signal": result["signals"],
            "net_return": result["net"]
        })
        curve.to_csv(out_curve, index=False)

    print("[INFO] Completed backtest for all tickers.")

if __name__ == "__main__":
    main()