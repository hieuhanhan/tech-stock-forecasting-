import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# --- Import All Model Libraries ---
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# ===================================================================
# 1. HELPER FUNCTIONS
# ===================================================================

def calculate_sharpe_ratio(daily_returns):
    """Calculates the annualized 'Simple' Sharpe Ratio."""
    if np.std(daily_returns) == 0: return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    """Calculates the maximum drawdown from a series of daily log returns."""
    if len(daily_returns) == 0: return 0.0
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return np.max(drawdown)

def create_sequences(data, window_size):
    """Helper function for preparing data for LSTM/GRU."""
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
    return np.array(X)

# ===================================================================
# 2. FINAL BACKTESTING SCRIPT
# ===================================================================

if __name__ == "__main__":
    print("Starting Final Backtest on Unseen Data for All Tickers...")

    # --- 1. Configuration ---
    results_dir = 'data/tuning_results'
    champion_models_path = os.path.join(results_dir, 'champion_models.json')
    
    try:
        with open(champion_models_path, 'r') as f:
            champions_list = json.load(f)
        
        CHAMPION_PROPHET_PARAMS = next(item['best_params'] for item in champions_list if item['model_type'] == 'Prophet')
        arima_params = next(item['best_params'] for item in champions_list if item['model_type'] == 'ARIMA')
        CHAMPION_ARIMA_ORDER = (arima_params['p'], arima_params['d'], arima_params['q'])
        CHAMPION_LSTM_PARAMS = next(item['best_params'] for item in champions_list if item['model_type'] == 'LSTM')
        CHAMPION_GRU_PARAMS = next(item['best_params'] for item in champions_list if item['model_type'] == 'GRU')

    except (FileNotFoundError, StopIteration) as e:
        print(f"Error loading champion parameters: {e}. Please run analyze_tuning_results.py first.")
        exit()

    # NOTE: You will need to determine the optimal thresholds for each model separately
    ACTION_THRESHOLD_PROPHET = 0.005183 
    ACTION_THRESHOLD_ARIMA = 0.005183 # Replace with ARIMA's optimal threshold
    ACTION_THRESHOLD_LSTM = 0.005183  # Replace with LSTM's optimal threshold
    ACTION_THRESHOLD_GRU = 0.005183   # Replace with GRU's optimal threshold
    TRANSACTION_COST = 0.0005

    # --- 2. Load Data ---
    full_dataset_path = 'data/cleaned/train_val_for_wf_with_features.csv'
    test_set_path = 'data/cleaned/test_set_with_features.csv'
    
    full_df_original = pd.read_csv(full_dataset_path, parse_dates=['Date'])
    test_df_original = pd.read_csv(test_set_path, parse_dates=['Date'])

    all_tickers = full_df_original['Ticker'].unique()
    final_backtest_results = []

    # --- 3. Main Loop for Each Ticker ---
    for ticker_to_test in all_tickers:
        print(f"\n--- Starting Full Backtest for {ticker_to_test} ---")
        
        full_df = full_df_original[full_df_original['Ticker'] == ticker_to_test].copy()
        test_df = test_df_original[test_df_original['Ticker'] == ticker_to_test].copy()
        
        if test_df.empty: continue

        combined_df = pd.concat([full_df, test_df]).sort_values(by='Date').reset_index(drop=True)
        test_start_index = len(full_df)

        prophet_forecasts, arima_forecasts, lstm_forecasts, gru_forecasts, actuals = [], [], [], [], []

        # --- 4. Walk-Forward Backtesting Loop ---
        for i in tqdm(range(test_start_index, len(combined_df)), desc=f"Backtesting {ticker_to_test}"):
            history_df = combined_df.iloc[:i]
            actuals.append(combined_df.iloc[i]['Log_Returns'])

            # --- Prophet Prediction ---
            prophet_history = history_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            model_prophet = Prophet(**CHAMPION_PROPHET_PARAMS).fit(prophet_history)
            future = model_prophet.make_future_dataframe(periods=1)
            forecast = model_prophet.predict(future)
            pred_price = forecast['yhat'].iloc[-1]
            last_price = history_df['Close'].iloc[-1]
            prophet_forecasts.append(np.log(pred_price / last_price))

            # --- ARIMA Prediction ---
            arima_history = history_df['Log_Returns'].dropna()
            try:
                model_arima = ARIMA(arima_history, order=CHAMPION_ARIMA_ORDER).fit()
                arima_forecasts.append(model_arima.forecast(steps=1).iloc[0])
            except:
                arima_forecasts.append(0.0)

            # --- LSTM/GRU Predictions ---
            # For DL models, we use the pre-scaled data
            dl_history = history_df['Log_Returns'].values.reshape(-1, 1)
            
            # LSTM
            lstm_params = CHAMPION_LSTM_PARAMS
            window_size = int(lstm_params['window_size'])
            X_test_lstm = dl_history[-window_size:].reshape(1, window_size, 1)
            # (In a real scenario, you'd build and train the model here)
            # This is a placeholder for the complex model training logic
            lstm_forecasts.append(0.0) # Placeholder

            # GRU
            gru_params = CHAMPION_GRU_PARAMS
            window_size = int(gru_params['window_size'])
            X_test_gru = dl_history[-window_size:].reshape(1, window_size, 1)
            # (Placeholder for GRU model training logic)
            gru_forecasts.append(0.0) # Placeholder
        
        # --- 5. Evaluate and Store Results for the Ticker ---
        actual_returns_np = np.array(actuals)
        
        prophet_forecasts_np = np.array(prophet_forecasts)
        prophet_signals = (prophet_forecasts_np > ACTION_THRESHOLD).astype(int)
        prophet_net_returns = (actual_returns_np * prophet_signals) - (prophet_signals * TRANSACTION_COST)
        prophet_sharpe = calculate_sharpe_ratio(prophet_net_returns)
        prophet_drawdown = calculate_max_drawdown(prophet_net_returns)

        arima_forecasts_np = np.array(arima_forecasts)
        arima_signals = (arima_forecasts_np > ACTION_THRESHOLD).astype(int)
        arima_net_returns = (actual_returns_np * arima_signals) - (arima_signals * TRANSACTION_COST)
        arima_sharpe = calculate_sharpe_ratio(arima_net_returns)
        arima_drawdown = calculate_max_drawdown(arima_net_returns)

        final_backtest_results.append({
            'ticker': ticker_to_test,
            'prophet_sharpe': prophet_sharpe,
            'prophet_drawdown': prophet_drawdown,
            'arima_sharpe': arima_sharpe,
            'arima_drawdown': arima_drawdown
        })
        
        curves_df = pd.DataFrame({
            'Date': test_df['Date'],
            'prophet_net_returns': prophet_net_returns,
            'arima_net_returns': arima_net_returns,
            'buy_and_hold_returns': actual_returns_np,
            'prophet_raw_forecast': prophet_forecasts,
            'arima_raw_forecast': arima_forecasts
        })
        
        curves_dir = 'data/backtest_curves'
        os.makedirs(curves_dir, exist_ok=True)
        curves_path = os.path.join(curves_dir, f'backtest_curves_{ticker_to_test}.csv')
        curves_df.to_csv(curves_path, index=False)
        print(f"  Saved backtest curves for {ticker_to_test} to {curves_path}")

    # --- 6. Print Final Summary Table ---
    print("\n\n" + "="*55)
    print("           FINAL BACKTESTING RESULTS SUMMARY")
    print("="*55)
    
    results_summary_df = pd.DataFrame(final_backtest_results)
    print(results_summary_df.round(4))
    
    print("\n" + "="*55)
    avg_prophet_sharpe = results_summary_df['prophet_sharpe'].mean()
    avg_arima_sharpe = results_summary_df['arima_sharpe'].mean()

    print(f"\nAverage Prophet Sharpe Ratio across all tickers: {avg_prophet_sharpe:.4f}")
    print(f"Average ARIMA Sharpe Ratio across all tickers:   {avg_arima_sharpe:.4f}")
    
    if avg_prophet_sharpe > avg_arima_sharpe:
        print("\nOVERALL CONCLUSION: The champion Prophet strategy provided better risk-adjusted returns on average.")
    else:
        print("\nOVERALL CONCLUSION: The champion ARIMA strategy provided better risk-adjusted returns on average.")
