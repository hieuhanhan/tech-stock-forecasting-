import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# --- Import Deep Learning and MOGA libraries ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# ============================================================
# 1. HELPER FUNCTION 
# ============================================================
def calculate_sharpe_ratio(daily_returns):
    """ Calculates the annualized Sharpe Ratio. """
    if np.std(daily_returns) == 0:
        return 0.0
    return (mp.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    """ Calculates the maximum drawdown. """
    if len(daily_returns) == 0:
        return 0.0
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    # Add a small epsilon to avoid division by zero is running_max is ever exactly 0
    drawdown = (running_max - cumulative_returns) / (running_max + np.finfo(float).eps)
    return np.max(drawdown)

# ============================================================
# 2. LSTM SPECIFIC FUNCTION
# ============================================================
def create_sequences(data, lookback_window):
    """
    Creates sequences for LSTM input.
    Input: data (numpy array or list), lookback_window (int)
    Output: X (3D numpy array), y (1D numpy array)
    """
    X, y = [], []
    for i in range(len(data) - lookback_window):
        X.append(data[i:(i + lookback_window)])
        y.append(data[i + lookback_window])
    return np.array(X), np.array(y)

def build_lstm_model(lookback_window, n_units, learning_rate):
    """
    Builds a Sequential LSTM model.
    """
    model = Sequential([
        Input(shape=(lookback_window, 1)),  # Input layer with (timesteps, features)
        LSTM(n_units, activation='tanh', return_sequences=False),    # Only return last output
        Dense(1)    # Output a single forecast value
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')  # Mean Squared Error for regression
    return model

def create_final_moga_objective_lstm(train_log_returns, actual_log_returns, transaction_cost=0.0005):
    """
    Creates the objective function for LSTM optimization.
    """
    def moga_objective_function(params):
        # Hyperparameters to optimize: lookback_window, n_units, learning_rate, epochs, action_threshold
        lookback_window = int(params[0])
        n_units = int(params[1])
        learning_rate = int(params[2])
        epochs = int(params[3])
        action_threshold = params[4]

        # Basic validation for lookback window
        # Ensure training data is long enough to create at least one sequence
        if len(train_log_returns) <= lookback_window:
            print(f"LSTM ValueError: Training data {len(train_log_returns)} too short for lookback_window ({lookback_window}). Penalty applied.")
            return (1e9, 1e9)
        
        if len(actual_log_returns) < 1 or lookback_window == 0:
            return (1e9, 1e9)
        
        try:
            # Data is assumed to be already scaled, just reshape for sequence creation
            # Create sequences for training
            X_train, y_train = create_sequences(train_log_returns.values.reshape(-1, 1), lookback_window)

            # Check if X_train is empty
            if X_train.size == 0:
                print(f"LSTM ValueError: X_train is empty for lookback_window={lookback_window}. Penalty applied.")
                return (1e9, 1e9)
            
            # Build and train the LSTM model
            model = build_lstm_model(lookback_window, n_units, learning_rate)
            # use verbose=0 to suppress training output during optimization
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)

            # Prepare validation data for prediction
            # To forecast the validation period, we need sequences. The forecasts for `actual_returns`
            # require `lookback_window` preceding values. These values come from the end of the
            # training set for the start of `actual_returns`.
            combined_series_for_pred = np.concatenate([train_log_returns.values[-lookback_window:], actual_log_returns])
            
            # X_pred will contain sequences for each point in actual_log_returns
            X_pred, _ = create_sequences(combined_series_for_pred.reshape(-1, 1), lookback_window)

            # Check if X_pred is empty
            if X_pred.size == 0:
                print(f"LSTM ValueError: X_pred is empty for lookback_window={lookback_window}. Penalty applied.")
                return (1e9, 1e9)
            
            # Generate forecasts
            forecast = model.predict(X_pred, verbose=0).flatten()

            signals = (forecast > action_threshold).astype(int)

            # If no trades occur, penalize
            if np.sum(signals) == 0 and len(actual_log_returns) > 0:
                return(1e9, 1e9)
            
            net_returns = (actual_log_returns * signals) - (signals * transaction_cost)

            objective_1_sharpe = calculate_sharpe_ratio(net_returns)
            objective_2_max_drawdown = calculate_max_drawdown(net_returns)

            return (-objective_1_sharpe, objective_2_max_drawdown)
        
        except ValueError as e:
            print(f"LSTM ValueError (params={params}): {e}. Penalty applied.")
            return (1e9, 1e9)
        except tf.errors.InvalidArgumentError as e:
            print(f"LSTM TensorFlow InvalidArgumentError (params={params}): {e}. Penalty applied.")
            return (1e9, 1e9)
        except Exception as e:
            print(f"LSTM Unexpected error (params={params}): {e}. Penalty applied.")
            return (1e9, 1e9)
            
    return moga_objective_function

class AdvancedLSTMProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        super().__init__(
            n_var=5,     # lookback_window, n_units, learning_rate, epochs, action_threshold
            n_obj=2,
            xl=np.array([5, 10, 1e4, 10, 0.0]),
            xu=np.array([30, 100, 1e-2, 100, 0.01])
        )
        self.objective_func = objective_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_func(x)

# ===================================================================
# 3. MAIN EXECUTION SCRIPT FOR LSTM
# ===================================================================
if __name__ == "__main__":
    print("Starting FINAL Multi-Objective Optimization (MOGA) for LSTM ...")

    # --- Configuration ---
    MODEL_TYPE = 'LSTM'
    
    # --- Setup Paths ---
    results_dir = 'data/tuning_results'
    folds_dir = 'data/processed_folds'
    os.makedirs(results_dir, exist_ok=True)
    moga_results_path = os.path.join(results_dir, f'final_moga_{MODEL_TYPE.lower()}_results.json')

    # --- Load existing results to resume progress ---
    try:
        with open(moga_results_path, 'r') as f:
            final_moga_results = json.load(f)
        print(f"Resuming {MODEL_TYPE} optimization from {len(final_moga_results)} previously completed folds.")
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
    # Wrap with tf.device('/CPU:0') to ensure TensorFlow operations run on CPU
    # This is important for stability during hyperparameter tuning, especially without a robust GPU setup.
    with tf.device('/CPU:0'): 
        for fold_id in tqdm(representative_fold_ids, desc=f"Optimizing Folds for {MODEL_TYPE}"):
            
            if fold_id in completed_folds:
                continue

            fold_info = folds_summary_map.get(fold_id)
            if not fold_info:
                continue

            # Assuming the same train/val paths are suitable for deep learning models
            train_path = os.path.join(folds_dir, fold_info['train_path_lstm_gru']) 
            val_path = os.path.join(folds_dir, fold_info['val_path_lstm_gru'])    
            train_df = pd.read_csv(train_path, parse_dates=['Date'])
            val_df = pd.read_csv(val_path, parse_dates=['Date'])

            train_log_returns = train_df['Log_Returns'].dropna()
            actual_returns = val_df['Log_Returns'].
            
            # Create objective and problem for LSTM
            moga_objective = create_final_moga_objective_lstm(train_log_returns, actual_returns)
            problem = AdvancedLSTMProblem(moga_objective)

            # --- MOGA Algorithm Configuration ---
            # These values (pop_size, n_gen) significantly impact runtime.
            # Start small for testing, increase for more thorough optimization.
            algorithm = NSGA2(pop_size=30)
            n_generations = 20

            res = minimize(problem, algorithm, ('n_gen', n_generations), seed=42, verbose=False)

            fold_solutions = {
                'fold_id': fold_id,
                'ticker': fold_info['ticker'],
                'pareto_front': []
            }

            # Process and store results
            if res.X is not None and res.F is not None:
                for sol, obj in zip(res.X, res.F):
                    fold_solutions['pareto_front'].append(
                        {
                            'lookback_window': int(sol[0]),
                            'n_units': int(sol[1]),
                            'learning_rate': float(sol[2]), # Ensure float for JSON
                            'epochs': int(sol[3]),
                            'threshold': float(sol[4]), # Ensure float for JSON
                            'sharpe_ratio': -float(obj[0]), # Convert back and ensure float
                            'max_drawdown': float(obj[1]) # Ensure float for JSON
                        }
                    )
            else:
                print(f"Optimization for fold {fold_id} ({fold_info['ticker']}) did not find any solutions (res.X or res.F was None).")
                
            final_moga_results.append(fold_solutions)

            # Save results after every single fold (checkpointing)
            with open(moga_results_path, 'w') as f:
                json.dump(final_moga_results, f, indent=4)

    print(f"\n--- FINAL MOGA Tuning Complete for {MODEL_TYPE}! ---")
    print(f"Final MOGA results saved to: {moga_results_path}")