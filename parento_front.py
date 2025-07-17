import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys # Import sys for potential exit if data is empty
import logging # For better logging messages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Helper to find the knee point in a Pareto front DataFrame ---
def find_knee_point_from_df(df_pareto: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame with columns 'max_drawdown' and 'sharpe_ratio',
    normalize both axes, compute distance to ideal (0 drawdown, 1 Sharpe),
    and return the Series of the row at the minimum distance.
    """
    dd = df_pareto['max_drawdown'].to_numpy()
    sr = df_pareto['sharpe_ratio'].to_numpy()
    min_dd, max_dd = dd.min(), dd.max()
    min_sr, max_sr = sr.min(), sr.max()

    norm_dd = np.zeros_like(dd) if max_dd == min_dd else (dd - min_dd) / (max_dd - min_dd)
    norm_sr = np.zeros_like(sr) if max_sr == min_sr else (sr - min_sr) / (max_sr - min_sr)

    dist = np.sqrt(norm_dd**2 + (1 - norm_sr)**2)
    idx = np.argmin(dist)
    return df_pareto.iloc[idx] # Return the entire row (as a Series)

# --- Visualization Script for MOGA Results ---

def main(model_type: str = 'arima', fold_index_to_visualize: int = 0):
    """
    Visualizes the Risk-Return Pareto Front for a specific fold of MOGA results.
    
    :param model_type: string identifier for the model (e.g., 'arima', 'lstm', 'gru')
    :param fold_index_to_visualize: The 0-based index of the fold in the JSON results list to visualize.
                                     Default to 0 (first fold).
    """
    logging.info(f"Starting visualization for {model_type.upper()} - Fold Index {fold_index_to_visualize}...")

    results_dir = 'data/tuning_results'
    
    # Input path
    moga_results_path = os.path.join(results_dir, f'final_moga_{model_type}_results.json')

    # Output figure path
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True) 

    try:
        with open(moga_results_path, 'r') as f:
            # The file contains a LIST of dictionaries, not a single dictionary.
            # We'll pick the first fold as an example for visualization.
            all_folds_data = json.load(f)

            if not all_folds_data:
                print(f"ERROR: MOGA results file at {moga_results_path} is empty. No data to visualize.")
                sys.exit(1)

            if fold_index_to_visualize >= len(all_folds_data) or fold_index_to_visualize < 0:
                logging.error(f"Fold index {fold_index_to_visualize} is out of bounds. Max index is {len(all_folds_data) - 1}.")
                sys.exit(1)

            fold_data_to_visualize = all_folds_data[fold_index_to_visualize]

    except FileNotFoundError:
        logging.error(f"MOGA results file not found at {moga_results_path}. Please run the MOGA tuning script for {model_type} first.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Malformed JSON in {moga_results_path}. Please check the file's integrity.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading MOGA results: {e}")
        sys.exit(1)

    # --- Extract Data for Plotting ---
    fold_id = fold_data_to_visualize['fold_id']
    ticker = fold_data_to_visualize['ticker']
    pareto_front_data = fold_data_to_visualize['pareto_front']

    # Convert to a pandas DataFrame for easy plotting and analysis
    df_pareto = pd.DataFrame(pareto_front_data)

    # --- Create the Pareto Front Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    if not df_pareto.empty:
        # Create the scatter plot
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

        # --- Highlight Key Strategic Points ---

        # Point with the highest Sharpe Ratio (Highest Return)
        best_sharpe_point = df_pareto.loc[df_pareto['sharpe_ratio'].idxmax()]
        ax.plot(best_sharpe_point['max_drawdown'], best_sharpe_point['sharpe_ratio'],
                'rX', markersize=15, label=f"Highest Sharpe ({best_sharpe_point['sharpe_ratio']:.2f})\n(Thr: {best_sharpe_point['threshold']:.4f})")

        # The "Knee" or "Elbow" point (Balanced Solution)
        if len(df_pareto) > 1:
            knee_point = find_knee_point_from_df(df_pareto) # Use the helper function
            ax.plot(knee_point['max_drawdown'], knee_point['sharpe_ratio'],
                    'gP', markersize=15, label=f"Balanced (Knee Point)\n(Thr: {knee_point['threshold']:.4f})")
        else:
            logging.warning(f"Not enough points ({len(df_pareto)}) in Pareto front to calculate knee point for fold {fold_id}.")
            ax.text(0.5, 0.5, 'Not enough optimal solutions to find knee point.', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=10, color='gray')


        ax.legend(fontsize=11)

    else:
        ax.text(0.5, 0.5, 'No optimal solutions found for this fold to plot.',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=15, color='red')
        logging.warning(f"No Pareto front data found for fold {fold_id} ({ticker}). No plot generated.")

    # --- Add Labels and Title ---
    ax.set_xlabel('Maximum Drawdown (Risk, Lower is Better →)', fontsize=14)
    ax.set_ylabel('Sharpe Ratio (Return, Higher is Better →)', fontsize=14)
    ax.set_title(f'Risk-Return Pareto Front for {ticker} (Fold {fold_id}) - {model_type.upper()}', fontsize=18, weight='bold')
    
    # Save the figure
    figure_filename = f'pareto_front_{ticker}_{fold_id}_{model_type}.png'
    figure_path = os.path.join(figures_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight', dpi=300) # Save with tight bounding box and high DPI
    logging.info(f"Pareto front plot saved to: {figure_path}")
    
    plt.show() # Display the plot (optional, can remove for batch processing)
    logging.info("Visualization complete.")

if __name__ == "__main__":
    # Allow optional model_type and fold_index_to_visualize arguments
    # Usage: python visualize_pareto_front.py [model_type] [fold_index]
    # Example: python visualize_pareto_front.py arima 5
    
    model = sys.argv[1] if len(sys.argv) > 1 else 'arima'
    
    fold_idx = 0 
    if len(sys.argv) > 2:
        try:
            fold_idx = int(sys.argv[2])
        except ValueError:
            logging.error(f"Invalid fold index argument: {sys.argv[2]}. Must be an integer.")
            sys.exit(1)

    main(model, fold_idx)