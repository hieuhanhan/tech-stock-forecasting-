import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# --- Helper Functions from Previous Steps ---

def calculate_sharpe_ratio(daily_returns):
    """Calculates the annualized 'Simple' Sharpe Ratio."""
    if np.std(daily_returns) == 0:
        return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    """Calculates the maximum drawdown from a series of daily log returns."""
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return np.max(drawdown)

# --- Final Backtesting Script ---

if __name__ == "__main__":
    print("Starting Final Backtest on Unseen Data...")

    # --- 1. Configuration ---
    # The champion parameters and rule we discovered from our MOGA tuning.
    CHAMPION_PROPHET_PARAMS = {'changepoint_prior_scale': 0.4997459185495748, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 0.16644947436717522}
    CHAMPION_ARIMA_ORDER = (1, 1, 1)
    ACTION_THRESHOLD = 0.004362
    TRANSACTION_COST = 0.0005 # 0.05%

    # --- 2. Load Data ---
    # We load the full dataset that has features, as we need the raw Log_Returns for evaluation.
    full_dataset_path = 'data/cleaned/train_val_for_wf_with_features.csv'
    test_set_path = 'data/cleaned/test_set_with_features.csv'
    
    try:
        full_df = pd.read_csv(full_dataset_path)
        test_df = pd.read_csv(test_set_path)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        exit()

    # We will backtest on a single ticker for this example. Let's use NVDA as it showed strong results.
    TICKER_TO_TEST = 'NVDA'
    
    # Combine train and test for a continuous history for walk-forward
    full_df = full_df[full_df['Ticker'] == TICKER_TO_TEST]
    test_df = test_df[test_df['Ticker'] == TICKER_TO_TEST]
    combined_df = pd.concat([full_df, test_df]).sort_values(by='Date').reset_index(drop=True)
    
    # Identify the start index of our test period within the combined data
    test_start_index = len(full_df)

    prophet_forecasts = []
    arima_forecasts = []
    actuals = []

    # --- 3. Walk-Forward Backtesting Loop ---
    # This loop simulates making a prediction each day of the test set.
    print(f"\nRunning walk-forward backtest for {TICKER_TO_TEST} from {test_df['Date'].min()} to {test_df['Date'].max()}...")
    for i in tqdm(range(test_start_index, len(combined_df)), desc=f"Backtesting {TICKER_TO_TEST}"):
        
        # The history includes all data up to the day before our prediction day.
        history_df = combined_df.iloc[:i]
        
        # The true value is the log return of the day we are trying to predict.
        true_value = combined_df.iloc[i]['Log_Returns']
        actuals.append(true_value)

        # Prepare data for Prophet
        prophet_history_df = history_df[['Date', 'Log_Returns']].rename(columns={'Date': 'ds', 'Log_Returns': 'y'})
        
        # --- Prophet Prediction ---
        model_prophet = Prophet(**CHAMPION_PROPHET_PARAMS)
        model_prophet.fit(prophet_history_df)
        future_prophet = model_prophet.make_future_dataframe(periods=1)
        forecast_prophet = model_prophet.predict(future_prophet)
        prophet_forecasts.append(forecast_prophet['yhat'].iloc[-1])

        # --- ARIMA Prediction ---
        # ARIMA works on the series of log returns
        arima_history_series = history_df['Log_Returns']
        try:
            model_arima = ARIMA(arima_history_series, order=CHAMPION_ARIMA_ORDER)
            model_arima_fit = model_arima.fit()
            forecast_arima = model_arima_fit.forecast(steps=1).iloc[0]
            arima_forecasts.append(forecast_arima)
        except Exception:
            # If ARIMA fails for any reason, assume a neutral (0) forecast
            arima_forecasts.append(0.0)
    
    # --- 4. Evaluate the Strategies ---
    
    actual_returns_np = np.array(actuals)
    
    # --- Prophet Strategy Evaluation ---
    prophet_forecasts_np = np.array(prophet_forecasts)
    prophet_signals = (prophet_forecasts_np > ACTION_THRESHOLD).astype(int)
    prophet_net_returns = (actual_returns_np * prophet_signals) - (prophet_signals * TRANSACTION_COST)
    
    prophet_sharpe = calculate_sharpe_ratio(prophet_net_returns)
    prophet_drawdown = calculate_max_drawdown(prophet_net_returns)

    # --- ARIMA Strategy Evaluation ---
    arima_forecasts_np = np.array(arima_forecasts)
    arima_signals = (arima_forecasts_np > ACTION_THRESHOLD).astype(int)
    arima_net_returns = (actual_returns_np * arima_signals) - (arima_signals * TRANSACTION_COST)

    arima_sharpe = calculate_sharpe_ratio(arima_net_returns)
    arima_drawdown = calculate_max_drawdown(arima_net_returns)

    # --- 5. Print Final Results ---
    print("\n\n" + "="*45)
    print("      FINAL BACKTESTING RESULTS")
    print("="*45)
    
    print("\n--- Champion Prophet Strategy ---")
    print(f"  Sharpe Ratio: {prophet_sharpe:.4f}")
    print(f"  Maximum Drawdown: {prophet_drawdown:.2%}")

    print("\n--- Baseline ARIMA(1,1,1) Strategy ---")
    print(f"  Sharpe Ratio: {arima_sharpe:.4f}")
    print(f"  Maximum Drawdown: {arima_drawdown:.2%}")
    
    print("\n" + "="*45)
    if prophet_sharpe > arima_sharpe:
        print("CONCLUSION: The champion Prophet strategy outperformed the ARIMA baseline on the test set.")
    else:
        print("CONCLUSION: The ARIMA baseline strategy outperformed the champion Prophet strategy on the test set.")
