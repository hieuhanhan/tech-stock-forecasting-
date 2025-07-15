import pandas as pd
import json
import os
import numpy as np

# --- Calculate Optimal Threshold for a Single Fold ---

print("Calculating the optimal action threshold for a single fold...")

# --- 1. Load the MOGA Results for the single fold ---
results_dir = 'data/tuning_results'
# Make sure this path points to your single-fold result file
moga_results_path = os.path.join(results_dir, 'moga_results_fold_60.json') 

try:
    with open(moga_results_path, 'r') as f:
        # moga_results is now a single dictionary, not a list
        moga_results = json.load(f) 
except FileNotFoundError as e:
    print(f"ERROR: Results file not found. Please ensure the path is correct. Details: {e}")
    exit()

# --- 2. Extract the "Knee Point" Threshold ---
# We no longer need a loop since we're processing one fold's data.

# Access the 'pareto_front' list directly from the loaded dictionary
df_pareto = pd.DataFrame(moga_results['pareto_front'])

knee_point_threshold = None

if not df_pareto.empty and len(df_pareto) > 1:
    # --- Normalize values to find the knee point ---
    # This finds the point closest to the ideal top-left corner (0 drawdown, max sharpe)
    
    # Normalize drawdown to a 0-1 scale
    normalized_drawdown = (df_pareto['max_drawdown'] - df_pareto['max_drawdown'].min()) / \
                          (df_pareto['max_drawdown'].max() - df_pareto['max_drawdown'].min())
    
    # Normalize sharpe ratio to a 0-1 scale
    normalized_sharpe = (df_pareto['sharpe_ratio'] - df_pareto['sharpe_ratio'].min()) / \
                        (df_pareto['sharpe_ratio'].max() - df_pareto['sharpe_ratio'].min())
    
    # Handle cases where normalization might result in NaN (if all values are the same)
    normalized_drawdown = np.nan_to_num(normalized_drawdown, nan=0.5)
    normalized_sharpe = np.nan_to_num(normalized_sharpe, nan=0.5)
    
    # Calculate the distance from the "ideal" point (0 drawdown, 1 sharpe) in normalized space
    df_pareto['distance'] = np.sqrt(normalized_drawdown**2 + (1 - normalized_sharpe)**2)
    
    # The point with the minimum distance is our "knee point"
    knee_point = df_pareto.loc[df_pareto['distance'].idxmin()]
    knee_point_threshold = knee_point['threshold']

# --- 3. Print the Final Result ---
if knee_point_threshold is not None:
    print("\n" + "="*40)
    print("      Optimal Balanced Threshold")
    print("="*40)
    print(f"The 'Balanced (Knee Point)' threshold for Fold {moga_results['fold_id']} is: {knee_point_threshold:.6f}")
    print("\nYou can use this single value as your ACTION_THRESHOLD in the final backtest.")
else:
    print("No valid knee point found. The Pareto front may have too few points.")

