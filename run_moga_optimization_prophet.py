import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

from prophet import Prophet

# --- Import the MOGA library ---
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# ===================================================================
# 1. HELPER AND OBJECTIVE FUNCTIONS
# ===================================================================

def calculate_sharpe_ratio(daily_returns):
    """Calculates the annualized 'Simple' Sharpe Ratio."""
    if np.std(daily_returns) == 0:
        return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    """Calculates the maximum drawdown from a series of daily log returns."""
    if len(daily_returns) == 0:
        return 0.0
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return np.max(drawdown)

# --- CORRECTED OBJECTIVE FUNCTION FACTORY (WITH DEBUGGING) ---
def create_final_moga_objective(train_df_prophet, val_data_len, last_known_price, actual_log_returns, transaction_cost=0.0005):
    """
    This factory creates the final multi-objective function.
    """
    def moga_objective_function(params):
        changepoint_prior_scale, seasonality_prior_scale, action_threshold = params
        try:
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode='multiplicative'
            )
            model.fit(train_df_prophet)

            future = model.make_future_dataframe(periods=val_data_len)
            forecast_df = model.predict(future)
            
            predicted_prices = forecast_df['yhat'][-val_data_len:].values
            forecast_log_returns = np.log(predicted_prices / last_known_price)

            # --- ADDED FOR DEBUGGING ---
            print(f"Forecast Log Return Stats: Min={np.min(forecast_log_returns):.6f}, Max={np.max(forecast_log_returns):.6f}, Mean={np.mean(forecast_log_returns):.6f}")
            # -----------------------------
            
            signals = (forecast_log_returns > action_threshold).astype(int)
            
            # --- ADDED FOR DEBUGGING ---
            print(f"Threshold={action_threshold:.6f}, Signals Generated={np.sum(signals)}")
            # -----------------------------

            net_returns = (actual_log_returns * signals) - (signals * transaction_cost)

            objective_1_sharpe = calculate_sharpe_ratio(net_returns)
            objective_2_max_drawdown = calculate_max_drawdown(net_returns)

            return (-objective_1_sharpe, objective_2_max_drawdown)
        except Exception as e:
            # Add a print statement here too to catch any other errors
            # print(f"!!! Objective Function Crashed: {e} !!!")
            return (1e9, 1e9)
    
    return moga_objective_function

class AdvancedTradingProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        super().__init__(
            n_var=3,
            n_obj=2,
            xl=np.array([0.0005, 0.1, 0.0005]),
            xu=np.array([0.05,   5.0, 0.0005])
        )
        self.objective_func = objective_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_func(x)

# ===================================================================
# 2. MAIN EXECUTION SCRIPT (WITH CHECKPOINTING)
# ===================================================================

if __name__ == "__main__":
    print("Starting FINAL Multi-Objective Optimization (MOGA) for Prophet...")

    # --- Setup Paths ---
    results_dir = 'data/tuning_results'
    folds_dir = 'data/processed_folds'
    os.makedirs(results_dir, exist_ok=True)
    moga_results_path = os.path.join(results_dir, 'final_moga_prophet_results.json')

    # --- Load existing results to resume progress ---
    try:
        with open(moga_results_path, 'r') as f:
            final_moga_results = json.load(f)
        print(f"Resuming from {len(final_moga_results)} previously completed folds.")
    except FileNotFoundError:
        final_moga_results = []
    
    completed_folds = {res['fold_id'] for res in final_moga_results}

    # --- Load Fold Information ---
    folds_summary_path = os.path.join(folds_dir, 'folds_summary.json')
    representative_folds_path = os.path.join(folds_dir, 'shared_meta', 'representative_fold_ids.json')
    
    with open(folds_summary_path, 'r') as f:
        all_folds_summary = json.load(f)
    with open(representative_folds_path, 'r') as f:
        representative_fold_ids = json.load(f)
    folds_summary_map = {item['global_fold_id']: item for item in all_folds_summary}

    # --- Main Loop with Checkpointing ---
    for fold_id in tqdm(representative_fold_ids, desc="Optimizing Folds"):
        
        if fold_id in completed_folds:
            continue

        fold_info = folds_summary_map.get(fold_id)
        if not fold_info:
            continue

        train_path = os.path.join(folds_dir, fold_info['train_path_arima_prophet'])
        val_path = os.path.join(folds_dir, fold_info['val_path_arima_prophet'])
        train_df = pd.read_csv(train_path, parse_dates=['Date'])
        val_df = pd.read_csv(val_path, parse_dates=['Date'])

        # Prepare data for Prophet
        prophet_train_df = train_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        actual_returns = val_df['Log_Returns'].values
        last_price = train_df['Close'].iloc[-1]
        val_len = len(val_df)

        moga_objective = create_final_moga_objective(prophet_train_df, val_len, last_price, actual_returns)

        problem = AdvancedTradingProblem(moga_objective)
        algorithm = NSGA2(pop_size=40)
        
        res = minimize(problem, algorithm, ('n_gen', 30), seed=42, verbose=False)

        fold_solutions = {
            'fold_id': fold_id,
            'ticker': fold_info['ticker'],
            'pareto_front': [
                {
                    'changepoint_prior_scale': sol[0],
                    'seasonality_prior_scale': sol[1],
                    'threshold': sol[2],
                    'sharpe_ratio': -obj[0],
                    'max_drawdown': obj[1]
                }
                for sol, obj in zip(res.X, res.F)
            ]
        }
        final_moga_results.append(fold_solutions)

        # Save results after every single fold
        with open(moga_results_path, 'w') as f:
            json.dump(final_moga_results, f, indent=4)

    print("\n--- FINAL MOGA Tuning Complete! ---")
    print(f"Final MOGA results saved to: {moga_results_path}")
