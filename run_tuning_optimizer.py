import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from math import sqrt
from sklearn.metrics import mean_squared_error

# ===================================================================
# 1. PARAMETER SPACE DEFINITIONS
# ===================================================================

arima_param_space = [
    Integer(1, 7, name='p'),
    Integer(1, 7, name='q')
]

prophet_param_space = [
    Real(0.001, 0.5, name='changepoint_prior_scale', prior='log-uniform'),
    Real(0.01, 10.0, name='seasonality_prior_scale', prior='log-uniform'),
    Categorical(['additive', 'multiplicative'], name='seasonality_mode')
]

# ===================================================================
# 2. OBJECTIVE FUNCTION DEFINITIONS
# ===================================================================

def arima_objective_function(params, train_log_returns, val_log_returns):
    p, q = params
    try:
        model = ARIMA(train_log_returns, order=(p, 0, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(val_log_returns))
        rmse = sqrt(mean_squared_error(val_log_returns, forecast))
        return rmse
    except Exception as e:
        return 1e9

def prophet_objective_function(params, train_df, val_df, fold_id):
    """
    Objective function for Prophet. Includes error printing for debugging.
    """
    changepoint_prior_scale, seasonality_prior_scale, seasonality_mode = params
    try:
        prophet_train_df = train_df[['Date', 'Close']].rename(
            columns={'Date': 'ds', 'Close': 'y'}
        )
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode
        )
        # CORRECTED: Removed the deprecated 'show_forecast_samps' argument.
        model.fit(prophet_train_df)
        
        future = model.make_future_dataframe(periods=len(val_df))
        forecast = model.predict(future)
        y_pred = forecast['yhat'][-len(val_df):]
        y_true = val_df['Close'].values
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        return rmse
    except Exception as e:
        # Print the specific error message for debugging
        print(f"\n--- ERROR in Prophet for Fold ID: {fold_id} ---")
        print(f"  Parameters: {params}")
        print(f"  Error Message: {e}")
        print("-------------------------------------------------")
        return 1e9

# Helper function to handle numpy types for JSON saving
def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    return obj

# ===================================================================
# 3. MAIN EXECUTION SCRIPT
# ===================================================================

# --- Load Setup Files ---
with open('data/processed_folds/shared_meta/representative_fold_ids.json', 'r') as f:
    representative_fold_ids = json.load(f)
with open('data/processed_folds/folds_summary.json', 'r') as f:
    all_folds_summary = json.load(f)

folds_summary_dict = {fold['global_fold_id']: fold for fold in all_folds_summary}

# --- Initialize Result Lists ---
final_arima_results = []
final_prophet_results = []

print(f"Starting tuning process for {len(representative_fold_ids)} representative folds...")

# --- Main Loop Over Representative Folds ---
for fold_id in tqdm(representative_fold_ids, desc="Processing Folds"):
    
    fold_info = folds_summary_dict[fold_id]
    train_path = os.path.join('data/processed_folds', fold_info['train_path_arima_prophet'])
    val_path = os.path.join('data/processed_folds', fold_info['val_path_arima_prophet'])
    
    train_data_fold = pd.read_csv(train_path, parse_dates=['Date'])
    val_data_fold = pd.read_csv(val_path, parse_dates=['Date'])

    # --- ARIMA OPTIMIZATION ---
    print(f"\n  Optimizing ARIMA for Fold {fold_id}...")
    arima_res = gp_minimize(
        func=lambda params: arima_objective_function(params, train_data_fold['Log_Returns'], val_data_fold['Log_Returns']),
        dimensions=arima_param_space,
        n_calls=50,
        n_initial_points=10,
        random_state=42
    )
    final_arima_results.append({
        'fold_id': fold_id,
        'ticker': fold_info['ticker'],
        'best_params': {'p': arima_res.x[0], 'd': 0, 'q': arima_res.x[1]},
        'best_rmse': arima_res.fun
    })
    
    # --- PROPHET OPTIMIZATION ---
    print(f"  Optimizing Prophet for Fold {fold_id}...")
    prophet_res = gp_minimize(
        func=lambda params: prophet_objective_function(params, train_data_fold, val_data_fold, fold_id),
        dimensions=prophet_param_space,
        n_calls=50,
        n_initial_points=10,
        random_state=42
    )
    final_prophet_results.append({
        'fold_id': fold_id,
        'ticker': fold_info['ticker'],
        'best_params': {
            'changepoint_prior_scale': prophet_res.x[0],
            'seasonality_prior_scale': prophet_res.x[1],
            'seasonality_mode': prophet_res.x[2]
        },
        'best_rmse': prophet_res.fun
    })

print("\n--- Tuning Complete! ---")

# --- Save Final Results ---
results_dir = 'data/tuning_results'
os.makedirs(results_dir, exist_ok=True) 

arima_results_path = os.path.join(results_dir, 'final_arima_tuning_results.json')
prophet_results_path = os.path.join(results_dir, 'final_prophet_tuning_results.json')

with open(arima_results_path, 'w') as f:
    json.dump(convert_numpy_types(final_arima_results), f, indent=4)
print(f"ARIMA results saved to: {arima_results_path}")

with open(prophet_results_path, 'w') as f:
    json.dump(convert_numpy_types(final_prophet_results), f, indent=4)
print(f"Prophet results saved to: {prophet_results_path}")



