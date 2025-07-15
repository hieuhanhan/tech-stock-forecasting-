import pandas as pd
import json
import os

# ===================================================================
# 1. HELPER FUNCTION (No changes needed here)
# ===================================================================

def find_champion_model(results_path, model_name):
    """
    Loads tuning results, finds the best performing set of hyperparameters
    based on the average RMSE across all folds.
    
    Args:
        results_path (str): Path to the JSON results file.
        model_name (str): Name of the model for printing (e.g., 'ARIMA').
        
    Returns:
        dict: The champion model's info including params and average RMSE, or None on error.
    """
    print(f"--- Analyzing Results for {model_name} ---")
    
    # --- Load the data ---
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Results file not found at {results_path}")
        return None

    if not results_data:
        print(f"ERROR: No data found in {results_path}")
        return None

    df = pd.DataFrame(results_data)

    # --- Process the results ---
    # Convert the 'best_params' dict to a string so we can group by it
    df['params_str'] = df['best_params'].apply(lambda x: json.dumps(x, sort_keys=True))
    
    # Group by each unique set of parameters and calculate the mean RMSE
    performance_summary = df.groupby('params_str')['best_rmse'].mean().reset_index()
    performance_summary.rename(columns={'best_rmse': 'average_rmse'}, inplace=True)
    
    # Find the parameter set with the minimum average RMSE
    champion_idx = performance_summary['average_rmse'].idxmin()
    champion = performance_summary.loc[champion_idx]
    
    # --- Display the champion model ---
    print(f"Champion {model_name} Model:")
    print(f"  Best Parameters: {champion['params_str']}")
    print(f"  Lowest Average RMSE across all folds: {champion['average_rmse']:.6f}\n")
    
    # Prepare final output
    champion_info = {
        'model_type': model_name,
        'best_params': json.loads(champion['params_str']),
        'average_rmse': champion['average_rmse']
    }
    
    return champion_info


# ===================================================================
# 2. MAIN EXECUTION SCRIPT (Revised for all 4 models)
# ===================================================================

# --- Define paths for all four model results ---
results_dir = 'data/tuning_results'
arima_results_path = os.path.join(results_dir, 'final_arima_tuning_results.json')
prophet_results_path = os.path.join(results_dir, 'final_prophet_tuning_results.json')
# REVISED: Add paths for LSTM and GRU results
lstm_results_path = os.path.join(results_dir, 'final_lstm_tuning_results.json')
gru_results_path = os.path.join(results_dir, 'final_gru_tuning_results.json')

# --- Find Champion for each model ---
arima_champion = find_champion_model(arima_results_path, 'ARIMA')
prophet_champion = find_champion_model(prophet_results_path, 'Prophet')
# REVISED: Add calls for LSTM and GRU
lstm_champion = find_champion_model(lstm_results_path, 'LSTM')
gru_champion = find_champion_model(gru_results_path, 'GRU')

# --- Save the Champion Models' Hyperparameters ---
champions_list = []
# REVISED: Add all four champions to the list if they were found successfully
if arima_champion:
    champions_list.append(arima_champion)
if prophet_champion:
    champions_list.append(prophet_champion)
if lstm_champion:
    champions_list.append(lstm_champion)
if gru_champion:
    champions_list.append(gru_champion)

if champions_list:
    # This file will now contain the best parameters for all four models
    champion_models_path = os.path.join(results_dir, 'champion_models.json')
    with open(champion_models_path, 'w') as f:
        json.dump(champions_list, f, indent=4)
    
    print(f"Champion model hyperparameters for all models saved to: {champion_models_path}")
    print("You will use this file for the next phase of MOGA optimization.")
else:
    print("Could not determine any champion models. Please check for errors.")
