import pandas as pd
import json
import os
import numpy as np

# --- Calculate Average Optimal Threshold ---

print("Calculating the average optimal action threshold...")

# --- 1. Load the MOGA Results ---
results_dir = 'data/tuning_results'
moga_results_path = os.path.join(results_dir, 'moga_results_fold_60.json')
volatility_path = os.path.join('data/processed_folds', 'shared_meta', 'fold_volatility_categorized.csv')

try:
    with open(moga_results_path, 'r') as f:
        moga_results = json.load(f)
    volatility_df = pd.read_csv(volatility_path)
except FileNotFoundError as e:
    print(f"ERROR: Results file not found. Please ensure scripts have been run. Details: {e}")
    exit()

# --- 2. Extract the "Knee Point" Threshold from Each Fold ---
knee_point_thresholds = []
for fold_data in moga_results:
    df_pareto = pd.DataFrame(fold_data['pareto_front'])
    
    if not df_pareto.empty and len(df_pareto) > 1:
        # Normalize values to find the knee point
        normalized_drawdown = (df_pareto['max_drawdown'] - df_pareto['max_drawdown'].min()) / (df_pareto['max_drawdown'].max() - df_pareto['max_drawdown'].min())
        normalized_sharpe = (df_pareto['sharpe_ratio'] - df_pareto['sharpe_ratio'].min()) / (df_pareto['sharpe_ratio'].max() - df_pareto['sharpe_ratio'].min())
        
        normalized_drawdown = np.nan_to_num(normalized_drawdown, nan=0.5)
        normalized_sharpe = np.nan_to_num(normalized_sharpe, nan=0.5)
        
        df_pareto['distance'] = np.sqrt(normalized_drawdown**2 + (1 - normalized_sharpe)**2)
        knee_point = df_pareto.loc[df_pareto['distance'].idxmin()]
        knee_point_thresholds.append(knee_point['threshold'])

# --- 3. Calculate and Print the Global Average ---
if knee_point_thresholds:
    global_average_threshold = np.mean(knee_point_thresholds)
    print("\n" + "="*40)
    print("      Strategy 1: Global Average Rule")
    print("="*40)
    print(f"The average 'Balanced (Knee Point)' threshold across all folds is: {global_average_threshold:.6f}")
    print("You can use this single value as your final, general-purpose trading rule.")
else:
    print("No valid knee points found to calculate an average.")

# --- 4. Calculate Averages Based on Volatility ---
print("\n" + "="*40)
print("  Strategy 2: Adaptive Rule by Volatility")
print("="*40)
# Create a DataFrame of all knee points
all_knee_points = []
for fold_data in moga_results:
    df_pareto = pd.DataFrame(fold_data['pareto_front'])
    if not df_pareto.empty and len(df_pareto) > 1:
        # We recalculate the knee point here to ensure we have the full data
        normalized_drawdown = (df_pareto['max_drawdown'] - df_pareto['max_drawdown'].min()) / (df_pareto['max_drawdown'].max() - df_pareto['max_drawdown'].min())
        normalized_sharpe = (df_pareto['sharpe_ratio'] - df_pareto['sharpe_ratio'].min()) / (df_pareto['sharpe_ratio'].max() - df_pareto['sharpe_ratio'].min())
        normalized_drawdown = np.nan_to_num(normalized_drawdown, nan=0.5)
        normalized_sharpe = np.nan_to_num(normalized_sharpe, nan=0.5)
        df_pareto['distance'] = np.sqrt(normalized_drawdown**2 + (1 - normalized_sharpe)**2)
        knee_point = df_pareto.loc[df_pareto['distance'].idxmin()]
        
        all_knee_points.append({
            'global_fold_id': fold_data['fold_id'],
            'threshold': knee_point['threshold']
        })

if all_knee_points:
    knee_points_df = pd.DataFrame(all_knee_points)
    # Merge with the volatility data
    merged_df = pd.merge(knee_points_df, volatility_df[['global_fold_id', 'volatility_category']], on='global_fold_id')
    
    # Calculate the average threshold for each volatility category
    adaptive_thresholds = merged_df.groupby('volatility_category')['threshold'].mean()
    
    print("Average 'Balanced' threshold for each market condition:")
    print(adaptive_thresholds)
    print("\nThis suggests a more advanced strategy: use a different threshold depending on market volatility.")
else:
    print("No valid data for adaptive threshold analysis.")