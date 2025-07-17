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
# 1. HELPER AND OBJECTIVE FUNCTIONS
# ===================================================================
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

def create_final_moga_objective_arima(train_log_returns, actual_log_returns, transaction_cost=0.0005):
    def moga_objective_function(params):
        p, q, action_threshold = int(params[0]), int(params[1]), params[2]
        try:
            model = ARIMA(train_log_returns, order=(p, 0, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(actual_log_returns)).values
            
            # Ensure net_returns calculation handles empty signals or all zeros
            signals = (forecast > action_threshold).astype(int)
            
            # Avoid division by zero in Sharpe Ratio if no trades occur or returns are all zero
            if np.sum(signals) == 0: # No trades
                return (1e9, 1e9) 
                
            net_returns = (actual_log_returns * signals) - (signals * transaction_cost)
            
            # Handle cases where std dev might be zero for Sharpe Ratio
            objective_1_sharpe = calculate_sharpe_ratio(net_returns)
            objective_2_max_drawdown = calculate_max_drawdown(net_returns)
            
            return (-objective_1_sharpe, objective_2_max_drawdown)
        
        except ValueError as e:
            # Log the specific error for debugging
            print(f"ValueError for p={p}, q={q}, threshold={action_threshold}: {e}")
            return (1e9, 1e9) # Return a penalty for invalid parameters
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError (e.g., singular matrix) for p={p}, q={q}, threshold={action_threshold}: {e}")
            return (1e9, 1e9)
        except Exception as e:
            print(f"An unexpected error occurred for p={p}, q={q}, threshold={action_threshold}: {e}")
            return (1e9, 1e9)
            
    return moga_objective_function

class AdvancedARIMAProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        super().__init__(
            n_var=3, 
            n_obj=2, 
            xl=np.array([1, 1, 0.0]), 
            xu=np.array([7, 7, 0.01]))
        self.objective_func = objective_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_func(x)

# ===================================================================
# 2. MAIN EXECUTION SCRIPT (WITH CHECKPOINTING)
# ===================================================================
if __name__ == "__main__":
    print("Starting FINAL Multi-Objective Optimization (MOGA) for ARIMA ...")

    # --- Setup Paths ---
    results_dir = 'data/tuning_results'
    folds_dir = 'data/processed_folds'
    os.makedirs(results_dir, exist_ok=True)
    moga_results_path = os.path.join(results_dir, 'final_moga_arima_results.json')

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
    for fold_id in tqdm(representative_fold_ids, desc="Optimizing Folds for ARIMA"):
        
        if fold_id in completed_folds:
            continue

        fold_info = folds_summary_map.get(fold_id)
        if not fold_info:
            continue

        train_path = os.path.join(folds_dir, fold_info['train_path_arima_prophet'])
        val_path = os.path.join(folds_dir, fold_info['val_path_arima_prophet'])
        train_df = pd.read_csv(train_path, parse_dates=['Date'])
        val_df = pd.read_csv(val_path, parse_dates=['Date'])

        train_log_returns = train_df['Log_Returns'].dropna()
        actual_returns = val_df['Log_Returns'].values

    # Create objective and run NSGA2
        moga_objective = create_final_moga_objective_arima(train_log_returns, actual_returns)
        
        problem = AdvancedARIMAProblem(moga_objective)
        algorithm = NSGA2(pop_size=50)

        res = minimize(problem, algorithm, ('n_gen', 40), seed=42, verbose=False)

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

        # Save results after every single fold
        with open(moga_results_path, 'w') as f:
            json.dump(final_moga_results, f, indent=4)

    print("\n--- FINAL MOGA Tuning Complete for ARIMA! ---")
    print(f"Final MOGA results saved to: {moga_results_path}")
