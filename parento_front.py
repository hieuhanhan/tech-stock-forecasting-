import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Visualization Script for MOGA Results ---

print("Visualizing the Risk-Return Pareto Front...")

# --- 1. Load the MOGA Results ---
results_dir = 'data/tuning_results'
# CORRECTED: The filename is 'moga_results_fold_60.json'
moga_results_path = os.path.join(results_dir, 'moga_results_fold_60.json')

try:
    with open(moga_results_path, 'r') as f:
        # The file contains a single dictionary, not a list
        fold_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: MOGA results file not found at {moga_results_path}.")
    print("Please run the single-fold MOGA tuning script first.")
    exit()

# --- 2. Extract Data for Plotting ---
fold_id = fold_data['fold_id']
ticker = fold_data['ticker']
pareto_front_data = fold_data['pareto_front']

# Convert to a pandas DataFrame for easy plotting and analysis
df_pareto = pd.DataFrame(pareto_front_data)

# --- 3. Create the Pareto Front Plot ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

if not df_pareto.empty:
    # Create the scatter plot
    # CORRECTED: Plot 'max_drawdown' on the x-axis
    scatter = ax.scatter(
        df_pareto['max_drawdown'], 
        df_pareto['sharpe_ratio'],
        c=df_pareto['threshold'], 
        cmap='viridis_r', # Reversed colormap is good for this
        s=100,
        alpha=0.7,
        edgecolors='k'
    )
    # Add a color bar to explain the point colors
    cbar = fig.colorbar(scatter)
    cbar.set_label('Action Threshold', fontsize=12)

    # --- 4. Highlight Key Strategic Points ---

    # Point with the highest Sharpe Ratio (Highest Return)
    best_sharpe_point = df_pareto.loc[df_pareto['sharpe_ratio'].idxmax()]
    ax.plot(best_sharpe_point['max_drawdown'], best_sharpe_point['sharpe_ratio'], 'r*', markersize=20, label=f"Highest Sharpe ({best_sharpe_point['sharpe_ratio']:.2f})")

    # The "Knee" or "Elbow" point (Balanced Solution)
    if len(df_pareto) > 1:
        # Normalize values to a 0-1 scale to calculate distance
        norm_drawdown = (df_pareto['max_drawdown'] - df_pareto['max_drawdown'].min()) / (df_pareto['max_drawdown'].max() - df_pareto['max_drawdown'].min())
        norm_sharpe = (df_pareto['sharpe_ratio'] - df_pareto['sharpe_ratio'].min()) / (df_pareto['sharpe_ratio'].max() - df_pareto['sharpe_ratio'].min())
        
        # Handle cases where normalization might result in NaN
        norm_drawdown = np.nan_to_num(norm_drawdown, nan=0.5)
        norm_sharpe = np.nan_to_num(norm_sharpe, nan=0.5)

        # Calculate the distance from the "ideal" point (0 drawdown, 1 sharpe)
        df_pareto['distance'] = np.sqrt(norm_drawdown**2 + (1 - norm_sharpe)**2)
        knee_point = df_pareto.loc[df_pareto['distance'].idxmin()]
        ax.plot(knee_point['max_drawdown'], knee_point['sharpe_ratio'], 'gP', markersize=15, label=f"Balanced (Knee Point)")

    ax.legend(fontsize=12)

else:
    ax.text(0.5, 0.5, 'No optimal solutions found for this fold.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15)


# --- 5. Add Labels and Title ---
# CORRECTED: Update axis labels
ax.set_xlabel('Objective 2: Maximum Drawdown (Risk, Lower is Better →)', fontsize=14)
ax.set_ylabel('Objective 1: Sharpe Ratio (Return, Higher is Better →)', fontsize=14)
ax.set_title(f'Risk-Return Pareto Front for Fold {fold_id} ({ticker})', fontsize=18, weight='bold')
plt.show()
