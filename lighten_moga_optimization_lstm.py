import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras import Input, Sequential, backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# ============================================================
# CONFIG & LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Limit TensorFlow verbosity, enable GPU growth
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logging.info(f"Using GPU(s): {gpus}")
else:
    logging.info("No GPU found, using CPU only.")

# -----------------------------------------------------------------------------
# CONSTANTS (avoid magic numbers)
# -----------------------------------------------------------------------------
TRADING_DAYS = 252
DEFAULT_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3

# ============================================================
# HELPER FUNCTION 
# ============================================================
def calculate_sharpe_ratio(daily_returns: np.ndarray) -> float:
    """ Calculates the annualized Sharpe Ratio. """
    if np.std(daily_returns) == 0:
        return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(TRADING_DAYS)

def calculate_max_drawdown(daily_returns):
    """ Calculates the maximum drawdown. """
    if len(daily_returns) == 0:
        return 0.0
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    # Add a small epsilon to avoid division by zero is running_max is ever exactly 0
    drawdown = (running_max - cumulative_returns) / (running_max + np.finfo(float).eps)
    return np.max(drawdown)

def create_sequences(series: np.ndarray, lookback_window: int):
    """
    Creates sequences for LSTM input.
    Input: data (numpy array or list), lookback_window (int)
    Output: X (3D numpy array), y (1D numpy array)
    """
    X, y = [], []
    for i in range(len(series) - lookback_window):
        X.append(series[i: i + lookback_window])
        y.append(series[i + lookback_window])
    return np.array(X), np.array(y)

# ============================================================
# MODEL FACTORY
# ============================================================

def build_lstm_model(lookback_window: int, n_units: int, learning_rate: float) -> tf.keras.Model:
    """ Two-layer LSTM w/ dropout & BN """
    model = Sequential([
        Input(shape=(lookback_window, 1)),
        LSTM(n_units, activation='tanh', return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(n_units // 2, activation='tanh'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(1, activation='linear')
        ])

    model.compile(optimizer=Adam(learning_rate), loss='mse')
    return model

# -----------------------------------------------------------------------------
# OBJECTIVE FACTORY
# -----------------------------------------------------------------------------
def create_final_moga_objective_lstm(train_log_returns: pd.Series, actual_log_returns: np.ndarray, transaction_cost: float = 0.0005):
    train_array = train_log_returns.values 
    val_array = actual_log_returns

    def moga_objective_function(params):
        # Hyperparameters to optimize: lookback_window, n_units, learning_rate, epochs, action_threshold
        lookback_window = int(params[0])
        n_units = int(params[1])
        learning_rate = float(params[2])
        epochs = int(params[3])
        action_threshold = float(params[4])

        # Basic validation for lookback window
        # Ensure training data is long enough to create at least one sequence
        if lookback_window < 1 or len(train_array) <= lookback_window or val_array.size == 0:
            return 1e9, 1e9
        
        # prepare training sequences
        X_train, y_train = create_sequences(train_array.reshape(-1, 1), lookback_window)
        if X_train.size == 0:
            return 1e9, 1e9
        
        # Reset graph & build model
        K.clear_session()
        model = build_lstm_model(lookback_window, n_units, learning_rate)

        # Early stopping on training loss
        es = EarlyStopping(monitor='loss', patience=EARLY_STOPPING_PATIENCE, verbose=0, restore_best_weights=True)

        try:
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=DEFAULT_BATCH_SIZE,
                verbose=0,
                shuffle=False,
                callbacks=[es]
            )
        except Exception as e:
            logging.warning(f"Training failed (params={params}): {e}")
            return 1e9, 1e9

        # prepare prediction sequences (use end of train + whole val)
        seq = np.concatenate([train_array[-lookback_window:], val_array])
        X_pred, _ = create_sequences(seq.reshape(-1, 1), lookback_window)
        if X_pred.size == 0:
            return 1e9, 1e9

        forecast = model.predict(X_pred, verbose=0).ravel()
        signals = (forecast > action_threshold).astype(int)
        if signals.sum() == 0:
            return 1e9, 1e9

        net_returns = val_array * signals - signals * transaction_cost
        sharpe_ratio = calculate_sharpe_ratio(net_returns)
        max_drawdown = calculate_max_drawdown(net_returns)
        return -sharpe_ratio, max_drawdown

    return moga_objective_function

# -----------------------------------------------------------------------------
# PYMOO WRAPPER
# -----------------------------------------------------------------------------
class AdvancedLSTMProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        super().__init__(
        n_var=5,
        n_obj=2,
        xl = np.array([10, 32, 1e-4, 10, 0.0001]),
        xu = np.array([60, 128, 1e-3, 50, 0.005]))
        self.objective_func = objective_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_func(x)

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    logging.info("Starting MOGA LSTM tuning in SINGLE FOLD TEST MODE…")

    results_dir = 'data/tuning_results'
    folds_dir   = 'data/processed_folds'
    os.makedirs(results_dir, exist_ok=True)

    moga_path = os.path.join(results_dir, 'final_moga_lstm_results.json')

    all_results = []
    done_fold_ids = set()
    try:
        with open(moga_path, 'r') as f:
            all_results = json.load(f)
        done_fold_ids = {r['fold_id'] for r in all_results if r.get('status') == 'success'}
        logging.info(f"Resuming from {len(all_results)} completed folds")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("Starting fresh.")
        all_results = []
        done_fold_ids = set()

    summary = json.load(open(os.path.join(folds_dir, 'folds_summary.json')))
    reps = json.load(open(os.path.join(folds_dir, 'shared_meta', 'representative_fold_ids.json')))
    summary_map = {f['global_fold_id']: f for f in summary}

    for fid in tqdm(reps, desc="folds"):
        if fid in done_fold_ids: 
            continue
        
        fold_result = {
            'fold_id': fid,
            'ticker': summary_map[fid]['ticker'],
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'running'
        }
        start_time = time.time()

        try:
            train_df = pd.read_csv(os.path.join(folds_dir, summary_map[fid]['train_path_lstm_gru']), parse_dates=['Date'])
            val_df = pd.read_csv(os.path.join(folds_dir, summary_map[fid]['val_path_lstm_gru']), parse_dates=['Date'])
            train_log_returns = train_df['Log_Returns'].dropna()
            actual_returns = val_df['Log_Returns'].dropna().values

            moga_objective = create_final_moga_objective_lstm(train_log_returns, actual_returns)
            problem = AdvancedLSTMProblem(moga_objective)
            algorithm = NSGA2(pop_size=30)
            
            # <<< THAY ĐỔI 1: Bật verbose để xem log tiến trình >>>
            res = minimize(problem, algorithm, ('n_gen', 20), seed=42, verbose=True)

            fold_solutions = []
            for sol, obj in zip(res.X, res.F):
                fold_solutions.append({
                    'lookback_window': int(sol[0]),
                    'n_units': int(sol[1]),
                    'learning_rate': float(sol[2]),
                    'epochs': int(sol[3]),
                    'action_threshold': float(sol[4]),
                    'sharpe_ratio': -float(obj[0]),
                    'max_drawdown': float(obj[1])
                })

            fold_result['pareto_front'] = fold_solutions
            fold_result['status'] = 'success'

        except Exception as e:
            logging.exception(f" Failed on fold {fid}: {e}")
            fold_result['status'] = 'error'
            fold_result['error_message'] = str(e)
            
        finally:
            fold_result['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            fold_result['duration_seconds'] = round(time.time() - start_time, 2)
            all_results.append(fold_result)
            with open(moga_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            
            # <<< THAY ĐỔI 2: Dừng lại sau khi xong fold đầu tiên >>>
            logging.info(f"Test run for fold {fid} finished. Stopping.")
            break

    logging.info(f"Single fold test complete. Results saved to {moga_path}")

