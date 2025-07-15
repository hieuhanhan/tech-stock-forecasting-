import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ===================================================================
# 1. CONFIGURATION
# ===================================================================
# --- Choose which specific fold to plot ---
FOLD_ID_TO_PLOT = 60
TICKER = 'AAPL' # Manually set the ticker for this fold

print(f"Generating diagnostic forecast plots for Fold {FOLD_ID_TO_PLOT} ({TICKER})...")

# --- Define file paths ---
results_dir = 'data/tuning_results'
folds_dir = 'data/processed_folds'

# Path to the MOGA results for our chosen fold
moga_results_path = os.path.join(results_dir, f'moga_results_fold_{FOLD_ID_TO_PLOT}.json')
# Path to the global champion models (we'll use this for ARIMA's params)
champion_models_path = os.path.join(results_dir, 'champion_models.json')
# Path to the summary file to get data paths
folds_summary_path = os.path.join(folds_dir, 'folds_summary.json')

# ===================================================================
# 2. LOAD DATA AND OPTIMAL PARAMETERS
# ===================================================================

# --- Load Fold Data ---
try:
    with open(folds_summary_path, 'r') as f:
        all_folds_summary = json.load(f)
    folds_summary_map = {item['global_fold_id']: item for item in all_folds_summary}
    
    fold_info = folds_summary_map.get(FOLD_ID_TO_PLOT)
    if not fold_info:
        raise FileNotFoundError(f"Fold {FOLD_ID_TO_PLOT} not found in summary.")
        
    train_path = os.path.join(folds_dir, fold_info['train_path_arima_prophet'])
    val_path = os.path.join(folds_dir, fold_info['val_path_arima_prophet'])
    
    train_df = pd.read_csv(train_path, parse_dates=['Date'])
    val_df = pd.read_csv(val_path, parse_dates=['Date'])

except FileNotFoundError as e:
    print(f"ERROR: Could not load data files. Details: {e}")
    exit()

# --- Find Optimal Parameters for This Fold ---
try:
    # For Prophet, find the "knee point" from the MOGA results for this specific fold
    with open(moga_results_path, 'r') as f:
        moga_results = json.load(f)
    
    df_pareto = pd.DataFrame(moga_results['pareto_front'])
    # Normalize values to find the point closest to the ideal (0 drawdown, max sharpe)
    normalized_drawdown = (df_pareto['max_drawdown'] - df_pareto['max_drawdown'].min()) / (df_pareto['max_drawdown'].max() - df_pareto['max_drawdown'].min())
    normalized_sharpe = (df_pareto['sharpe_ratio'] - df_pareto['sharpe_ratio'].min()) / (df_pareto['sharpe_ratio'].max() - df_pareto['sharpe_ratio'].min())
    df_pareto['distance'] = np.sqrt(np.nan_to_num(normalized_drawdown)**2 + (1 - np.nan_to_num(normalized_sharpe))**2)
    knee_point = df_pareto.loc[df_pareto['distance'].idxmin()]
    
    CHAMPION_PROPHET_PARAMS = {
        'changepoint_prior_scale': knee_point['changepoint_prior_scale'],
        'seasonality_prior_scale': knee_point['seasonality_prior_scale'],
        'seasonality_mode': 'multiplicative' # Assuming this was fixed in MOGA
    }

    # For ARIMA, use the global champion from the Tier 1 tuning
    with open(champion_models_path, 'r') as f:
        champions_list = json.load(f)
    arima_params = next(item['best_params'] for item in champions_list if item['model_type'] == 'ARIMA')
    CHAMPION_ARIMA_ORDER = (arima_params['p'], arima_params['d'], arima_params['q'])

except (FileNotFoundError, KeyError, StopIteration) as e:
    print(f"ERROR: Could not load or process parameter files. Details: {e}")
    exit()

print(f"Using Prophet Params for Fold {FOLD_ID_TO_PLOT}: {CHAMPION_PROPHET_PARAMS}")
print(f"Using Global Champion ARIMA Order: {CHAMPION_ARIMA_ORDER}")

# ===================================================================
# 3. RE-TRAIN MODELS AND GENERATE FORECASTS
# ===================================================================

# --- ARIMA Forecast ---
print("Training ARIMA model on Fold 60 training data...")
arima_model = ARIMA(train_df['Log_Returns'].dropna(), order=CHAMPION_ARIMA_ORDER)
arima_fit = arima_model.fit()
arima_forecast_result = arima_fit.get_forecast(steps=len(val_df))
arima_pred_log_returns = arima_forecast_result.predicted_mean
arima_conf_int = arima_forecast_result.conf_int()

# --- Prophet Forecast ---
print("Training Prophet model on Fold 60 training data...")
prophet_train_df = train_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet(**CHAMPION_PROPHET_PARAMS)
prophet_model.fit(prophet_train_df)
future = prophet_model.make_future_dataframe(periods=len(val_df))
prophet_forecast_df = prophet_model.predict(future)

# ===================================================================
# 4. VISUALIZE THE FORECASTS
# ===================================================================

# --- Prepare data for plotting ---
full_actual_df = pd.concat([train_df, val_df])
last_train_price = train_df['Close'].iloc[-1]

# Convert ARIMA log return forecast to price forecast
arima_pred_prices = last_train_price * np.exp(np.cumsum(arima_pred_log_returns))
# Convert Prophet price forecast
prophet_pred_prices = prophet_forecast_df['yhat'][-len(val_df):]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8))

# Plot actual prices
ax.plot(full_actual_df['Date'], full_actual_df['Close'], label='Actual Price', color='black', lw=2)

# Plot Prophet forecast
ax.plot(val_df['Date'], prophet_pred_prices, label='Prophet Forecast', color='blue', linestyle='--')
ax.fill_between(val_df['Date'], 
                prophet_forecast_df['yhat_lower'][-len(val_df):], 
                prophet_forecast_df['yhat_upper'][-len(val_df):], 
                color='blue', alpha=0.1, label='Prophet 95% C.I.')

# Plot ARIMA forecast
ax.plot(val_df['Date'], arima_pred_prices, label='ARIMA Forecast', color='green', linestyle='--')
# Note: Visualizing ARIMA confidence intervals requires similar conversion and is omitted for clarity

# Add vertical line for train/validation split
ax.axvline(train_df['Date'].iloc[-1], color='gray', linestyle=':', label='Train/Validation Split')

ax.set_title(f'Model Forecast vs. Actual Price for {TICKER} (Fold {FOLD_ID_TO_PLOT})', fontsize=18, weight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Close Price', fontsize=12)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
