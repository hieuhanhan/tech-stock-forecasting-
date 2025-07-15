import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# --- Import ARIMA and MOGA libraries ---
from statsmodels.tsa.arima.model import ARIMA
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# ===================================================================
# 1. HELPER AND OBJECTIVE FUNCTIONS (Adapted for ARIMA)
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

# --- NEW OBJECTIVE FUNCTION FACTORY FOR ARIMA ---
def create_final_moga_objective_arima(train_log_returns, actual_log_returns, transaction_cost=0.0005):
    """
    This factory creates the multi-objective function for ARIMA.
    """
    def moga_objective_function(params):
        """
        'params' is a list: [p, q, action_threshold]
        """
        # Cast p and q to integers, as the optimizer treats them as floats
        p, q, action_threshold = int(params[0]), int(params[1]), params[2]

        try:
            # 1. Re-train the ARIMA model with the new hyperparameters
            # We use d=0 because the data is already stationary (Log_Returns)
            model = ARIMA(train_log_returns, order=(p, 0, q))
            model_fit = model.fit()

            # 2. Generate a new forecast
            forecast = model_fit.forecast(steps=len(actual_log_returns)).values
            
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
            # Return a terrible score on error (e.g., non-stationary parameters)
            return (1e9, 1e9)
    
    return moga_objective_function

# --- NEW Pymoo Problem Class for ARIMA ---
class AdvancedARIMAProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        # We have 3 variables: p, q, and threshold
        super().__init__(
            n_var=3,
            n_obj=2,
            # Define lower and upper bounds for each variable
            # p and q are treated as floats here but will be converted to int
            xl=np.array([1, 1, 0.0]),     # [p, q, threshold]
            xu=np.array([7, 7, 0.01])     # [p, q, threshold]
        )
        self.objective_func = objective_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_func(x)


# ===================================================================
# 3. MAIN EXECUTION SCRIPT (Adapted for ARIMA)
# ===================================================================

if __name__ == "__main__":
    print("Starting FINAL Multi-Objective Optimization (MOGA) for ARIMA...")

    # Load prerequisite files
    results_dir = 'data/tuning_results'
    folds_dir = 'data/processed_folds'
    folds_summary_path = os.path.join(folds_dir, 'folds_summary.json')
    representative_folds_path = os.path.join(folds_dir, 'shared_meta', 'representative_fold_ids.json')
    
    with open(folds_summary_path, 'r') as f:
        all_folds_summary = json.load(f)
    with open(representative_folds_path, 'r') as f:
        representative_fold_ids = json.load(f)
    folds_summary_map = {item['global_fold_id']: item for item in all_folds_summary}

    final_moga_results = []

    # ADDED: tqdm progress bar to the main loop
    for fold_id in tqdm(representative_fold_ids, desc="Optimizing Folds for ARIMA"):
        fold_info = folds_summary_map.get(fold_id)
        if not fold_info:
            continue

        # Load the data for this fold
        train_path = os.path.join(folds_dir, fold_info['train_path_arima_prophet'])
        val_path = os.path.join(folds_dir, fold_info['val_path_arima_prophet'])
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Prepare data for ARIMA
        train_log_returns = train_df['Log_Returns'].dropna()
        actual_returns = val_df['Log_Returns'].values

        # Create the objective function for ARIMA
        moga_objective = create_final_moga_objective_arima(train_log_returns, actual_returns)

        # Set up and run the Pymoo optimization
        problem = AdvancedARIMAProblem(moga_objective)
        algorithm = NSGA2(pop_size=50)
        
        res = minimize(problem, algorithm, ('n_gen', 40), seed=42, verbose=False)

        # Store the results, now with ARIMA parameters
        fold_solutions = {
            'fold_id': fold_id,
            'ticker': fold_info['ticker'],
            'pareto_front': [
                {
                    'p': int(sol[0]),
                    'q': int(sol[1]),
                    'threshold': sol[2],
                    'sharpe_ratio': -obj[0],
                    'max_drawdown': obj[1]
                }
                for sol, obj in zip(res.X, res.F)
            ]
        }
        final_moga_results.append(fold_solutions)

    # --- Save Final MOGA Results for ARIMA ---
    print("\n--- FINAL MOGA Tuning Complete for ARIMA! ---")
    moga_results_path = os.path.join(results_dir, 'final_moga_arima_results.json')
    with open(moga_results_path, 'w') as f:
        json.dump(final_moga_results, f, indent=4)
    print(f"Final ARIMA MOGA results saved to: {moga_results_path}")
