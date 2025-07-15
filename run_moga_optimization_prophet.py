import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

from prophet import Prophet

# --- Step 1: Import the MOGA library ---
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Choice

# --- Step 2: Define our Helper and Objective Functions ---

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

# --- OBJECTIVE FUNCTION FACTORY ---
def create_final_moga_objective(train_df_prophet, val_df_prophet, actual_log_returns, transaction_cost=0.0005):
    """
    This factory creates the final multi-objective function. It now re-trains
    the model in every trial.
    """
    def moga_objective_function(params):
        """
        This inner function is what Pymoo will evaluate.
        'params' is a list: [changepoint_prior_scale, seasonality_prior_scale, action_threshold]
        """
        changepoint_prior_scale, seasonality_prior_scale, action_threshold = params

        try:
            # 1. Re-train the Prophet model with the new hyperparameters
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode='multiplicative' # Keep mode fixed for stability or add to tuning
            )
            model.fit(train_df_prophet)

            # 2. Generate a new forecast
            future = model.make_future_dataframe(periods=len(val_df_prophet))
            forecast_df = model.predict(future)
            forecast = forecast_df['yhat'][-len(val_df_prophet):].values
            
            # 3. Apply the action_threshold to generate signals
            signals = (forecast > action_threshold).astype(int)
            
            # 4. Calculate Net Returns
            gross_returns = actual_log_returns * signals
            costs = signals * transaction_cost
            net_returns = gross_returns - costs

            # 5. Calculate the two objectives
            objective_1_sharpe = calculate_sharpe_ratio(net_returns)
            objective_2_max_drawdown = calculate_max_drawdown(net_returns)

            return (-objective_1_sharpe, objective_2_max_drawdown)

        except Exception as e:
            return (1e9, 1e9) # Return a terrible score on error
    
    return moga_objective_function

# --- Step 3: Define the NEW Pymoo Problem Class ---
class AdvancedTradingProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        # We now have 3 variables to tune
        super().__init__(
            n_var=3,
            n_obj=2,
            # Define lower and upper bounds for each variable
            xl=np.array([0.001, 0.01, 0.0]),     # [changepoint_prior, seasonality_prior, threshold]
            xu=np.array([0.5,   10.0, 0.01])     # [changepoint_prior, seasonality_prior, threshold]
        )
        self.objective_func = objective_func

    def _evaluate(self, x, out, *args, **kwargs):
        # x is an array with our 3 variables
        out["F"] = self.objective_func(x)


# --- Step 4: Main Execution Block ---
if __name__ == "__main__":
    print("Starting FINAL Multi-Objective Optimization (MOGA)...")

    # Load prerequisite files (we only need the fold structure now)
    results_dir = 'data/tuning_results'
    folds_dir = 'data/processed_folds'
    folds_summary_path = os.path.join(folds_dir, 'folds_summary.json')
    representative_folds_path = os.path.join(folds_dir, 'shared_meta', 'representative_fold_ids.json')
    
    # ... (code to load files) ...
    # This part can be simplified as we only need the file paths
    with open(folds_summary_path, 'r') as f:
        all_folds_summary = json.load(f)
    with open(representative_folds_path, 'r') as f:
        representative_fold_ids = json.load(f)
    folds_summary_map = {item['global_fold_id']: item for item in all_folds_summary}

    final_moga_results = []

    for fold_id in tqdm(representative_fold_ids, desc="Optimizing Folds"):
        fold_info = folds_summary_map.get(fold_id)
        if not fold_info:
            continue

        # Load the raw data for this fold
        train_path = os.path.join(folds_dir, fold_info['train_path_arima_prophet'])
        val_path = os.path.join(folds_dir, fold_info['val_path_arima_prophet'])
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Prepare data for Prophet
        # CORRECTED: Prophet must be trained on the raw 'Close' price
        prophet_train_df = train_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        actual_returns = val_df['Log_Returns'].values

        # Create the objective function, now without fixed params
        moga_objective = create_final_moga_objective(prophet_train_df, val_df, actual_returns)

        # Set up and run the Pymoo optimization
        problem = AdvancedTradingProblem(moga_objective)
        algorithm = NSGA2(pop_size=50)
        
        res = minimize(problem, algorithm, ('n_gen', 40), seed=42, verbose=False)

        # Store the results, now with 3 parameters
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

    # --- Step 5: Save Final MOGA Results ---
    print("\n--- FINAL MOGA Tuning Complete! ---")
    moga_results_path = os.path.join(results_dir, 'final_moga_prophet_results.json')
    with open(moga_results_path, 'w') as f:
        json.dump(final_moga_results, f, indent=4)
    print(f"Final MOGA results saved to: {moga_results_path}")