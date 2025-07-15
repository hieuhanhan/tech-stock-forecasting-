import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer, Real
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --- Import TensorFlow and Keras for Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===================================================================
# 1. PARAMETER SPACE DEFINITION
# ===================================================================

dl_param_space = [
    Integer(10, 60, name='window_size'),
    Integer(1, 3, name='n_layers'),
    Integer(32, 128, name='n_neurons'),
    Real(0.1, 0.5, name='dropout'),
    Real(1e-4, 1e-2, 'log-uniform', name='learning_rate'),
    Integer(32, 128, name='batch_size')
]

# ===================================================================
# 2. HELPER & OBJECTIVE FUNCTIONS
# ===================================================================

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def create_dl_objective_function(train_scaled_log_returns, val_scaled_log_returns, model_type, fold_id):
    def objective_function(params):
        try:
            window_size = int(params[0])
            n_layers = int(params[1])
            n_neurons = int(params[2])
            dropout = params[3]
            learning_rate = params[4]
            batch_size = int(params[5])

            train_data = train_scaled_log_returns.values
            X_train, y_train = create_sequences(train_data, window_size)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            model = Sequential()
            for i in range(n_layers):
                return_sequences = (i < n_layers - 1)
                LayerComponent = LSTM if model_type == 'LSTM' else GRU
                if i == 0:
                    model.add(LayerComponent(n_neurons, return_sequences=return_sequences, input_shape=(window_size, 1)))
                else:
                    model.add(LayerComponent(n_neurons, return_sequences=return_sequences))
                model.add(Dropout(dropout))
            model.add(Dense(1))

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=0, callbacks=[early_stopping])

            last_window_of_train = train_data[-window_size:].reshape(-1, 1)
            validation_data = val_scaled_log_returns.values.reshape(-1, 1)
            all_data = np.concatenate((last_window_of_train, validation_data))
            
            X_val, y_val = create_sequences(all_data, window_size)
            X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

            predicted_scaled = model.predict(X_val, verbose=0)
            rmse = sqrt(mean_squared_error(val_scaled_log_returns, predicted_scaled))
            return rmse
        except Exception as e:
            return 1e9
            
    return objective_function

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    return obj

# ===================================================================
# 3. MAIN EXECUTION SCRIPT (WITH CHECKPOINTING)
# ===================================================================

if __name__ == "__main__":
    # --- Setup Paths ---
    results_dir = 'data/tuning_results'
    os.makedirs(results_dir, exist_ok=True)
    lstm_results_path = os.path.join(results_dir, 'final_lstm_tuning_results.json')
    gru_results_path = os.path.join(results_dir, 'final_gru_tuning_results.json')

    # --- NEW: Load existing results to resume progress ---
    try:
        with open(lstm_results_path, 'r') as f:
            final_lstm_results = json.load(f)
        print(f"Resuming from {len(final_lstm_results)} previously completed LSTM folds.")
    except FileNotFoundError:
        final_lstm_results = []

    try:
        with open(gru_results_path, 'r') as f:
            final_gru_results = json.load(f)
        print(f"Resuming from {len(final_gru_results)} previously completed GRU folds.")
    except FileNotFoundError:
        final_gru_results = []
    
    # --- NEW: Determine which folds have already been completed ---
    completed_lstm_folds = {res['fold_id'] for res in final_lstm_results}
    completed_gru_folds = {res['fold_id'] for res in final_gru_results}

    # --- Load Fold Information ---
    with open('data/processed_folds/shared_meta/representative_fold_ids.json', 'r') as f:
        representative_fold_ids = json.load(f)
    with open('data/processed_folds/folds_summary.json', 'r') as f:
        all_folds_summary = json.load(f)
    folds_summary_dict = {fold['global_fold_id']: fold for fold in all_folds_summary}

    print(f"Starting Deep Learning tuning process for {len(representative_fold_ids)} representative folds...")

    # --- Main Loop Over Representative Folds ---
    for fold_id in tqdm(representative_fold_ids, desc="Processing Folds"):
        
        # --- NEW: Skip fold if it's already done ---
        if fold_id in completed_lstm_folds and fold_id in completed_gru_folds:
            print(f"\nSkipping Fold {fold_id} (already completed).")
            continue

        fold_info = folds_summary_dict[fold_id]
        train_path = os.path.join('data/processed_folds', fold_info['train_path_lstm_gru'])
        val_path = os.path.join('data/processed_folds', fold_info['val_path_lstm_gru'])
        
        train_data_fold = pd.read_csv(train_path, parse_dates=['Date'])
        val_data_fold = pd.read_csv(val_path, parse_dates=['Date'])

        # --- LSTM OPTIMIZATION ---
        if fold_id not in completed_lstm_folds:
            print(f"\n  Optimizing LSTM for Fold {fold_id}...")
            lstm_objective = create_dl_objective_function(
                train_data_fold['Log_Returns'], val_data_fold['Log_Returns'], model_type='LSTM', fold_id=fold_id
            )
            lstm_res = gp_minimize(
                func=lstm_objective,
                dimensions=dl_param_space,
                n_calls=25,
                n_initial_points=10,
                random_state=42
            )
            final_lstm_results.append({
                'fold_id': fold_id,
                'ticker': fold_info['ticker'],
                'best_params': {
                    'window_size': lstm_res.x[0], 'n_layers': lstm_res.x[1], 'n_neurons': lstm_res.x[2],
                    'dropout': lstm_res.x[3], 'learning_rate': lstm_res.x[4], 'batch_size': lstm_res.x[5]
                },
                'best_rmse': lstm_res.fun
            })

        # --- GRU OPTIMIZATION ---
        if fold_id not in completed_gru_folds:
            print(f"  Optimizing GRU for Fold {fold_id}...")
            gru_objective = create_dl_objective_function(
                train_data_fold['Log_Returns'], val_data_fold['Log_Returns'], model_type='GRU', fold_id=fold_id
            )
            gru_res = gp_minimize(
                func=gru_objective,
                dimensions=dl_param_space,
                n_calls=25,
                n_initial_points=10,
                random_state=42
            )
            final_gru_results.append({
                'fold_id': fold_id,
                'ticker': fold_info['ticker'],
                'best_params': {
                    'window_size': gru_res.x[0], 'n_layers': gru_res.x[1], 'n_neurons': gru_res.x[2],
                    'dropout': gru_res.x[3], 'learning_rate': gru_res.x[4], 'batch_size': gru_res.x[5]
                },
                'best_rmse': gru_res.fun
            })

        # --- NEW: Save results after every single fold ---
        with open(lstm_results_path, 'w') as f:
            json.dump(convert_numpy_types(final_lstm_results), f, indent=4)
        with open(gru_results_path, 'w') as f:
            json.dump(convert_numpy_types(final_gru_results), f, indent=4)
        print(f"  Checkpoint saved for Fold {fold_id}.")

    print("\n--- Tuning Complete! ---")
