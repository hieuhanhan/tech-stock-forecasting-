import pandas as pd
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_knee_points_data(model_type: str) -> pd.DataFrame:
    """Loads knee point data for a given model type."""
    csv_path = os.path.join('data', 'tuning_results', model_type, 'csv', f'knee_points_{model_type}.csv')
    try:
        df = pd.read_csv(csv_path)
        df['model_type'] = model_type # Add a column to identify the model
        return df
    except FileNotFoundError:
        logging.warning(f"Knee points CSV not found for {model_type}: {csv_path}. Skipping this model.")
        return pd.DataFrame() # Return empty DataFrame if not found
    except pd.errors.EmptyDataError:
        logging.warning(f"Knee points CSV is empty for {model_type}: {csv_path}. Skipping this model.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading knee points for {model_type}: {e}. Skipping.")
        return pd.DataFrame()

def main():
    logging.info("Starting model performance comparison.")

    model_types = ['arima', 'prophet', 'lstm', 'gru'] # List all models to compare
    all_models_knee_points = []

    for model_type in model_types:
        df_knee = load_knee_points_data(model_type)
        if not df_knee.empty:
            all_models_knee_points.append(df_knee)

    if not all_models_knee_points:
        logging.error("No knee point data found for any model. Cannot perform comparison.")
        sys.exit(1)

    combined_df = pd.concat(all_models_knee_points, ignore_index=True)
    
    # Ensure necessary columns are present for merging with volatility
    volatility_path = os.path.join('data/processed_folds', 'shared_meta', 'fold_volatility_categorized.csv')
    try:
        volatility_df = pd.read_csv(volatility_path)
        if 'global_fold_id' not in volatility_df.columns or 'volatility_category' not in volatility_df.columns:
            logging.error("Volatility CSV missing required columns: 'global_fold_id','volatility_category'. Cannot merge.")
            volatility_df = pd.DataFrame() # Make it empty to skip merge
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
        logging.warning(f"Could not load volatility data from {volatility_path}: {e}. Comparison plots will not include volatility categories.")
        volatility_df = pd.DataFrame()
        
    if not volatility_df.empty:
        # Merge combined_df with volatility_df to get volatility_category for each fold_id
        combined_df = pd.merge(combined_df, 
                               volatility_df[['global_fold_id', 'volatility_category']],
                               on='global_fold_id', 
                               how='left') # Use left merge to keep all knee points, even if no volatility match
    
    # --- Visualization Outputs ---
    figures_output_dir = 'data/tuning_results/comparison_figures'
    os.makedirs(figures_output_dir, exist_ok=True)

    # 1. Compare Average Knee Points (Table)
    logging.info("Comparing average knee points across models.")
    avg_knee_points_by_model = combined_df.groupby('model_type')[['sharpe_ratio', 'max_drawdown']].mean().reset_index()
    logging.info("\nAverage Knee Point Performance by Model:")
    logging.info(avg_knee_points_by_model.to_string(float_format="%.4f"))

    # 2. Compare Average Knee Points (Scatter Plot)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=avg_knee_points_by_model,
        x='max_drawdown',
        y='sharpe_ratio',
        hue='model_type',
        s=200, # Marker size
        style='model_type', # Different markers for different models
        edgecolor='black',
        linewidth=1
    )
    # Add text labels for each point
    for i, row in avg_knee_points_by_model.iterrows():
        plt.text(row['max_drawdown'] + 0.0001, row['sharpe_ratio'], # Adjust offset for text
                 row['model_type'].upper(), 
                 horizontalalignment='left', 
                 size='small', 
                 color='black', 
                 weight='semibold')

    plt.title('Average Knee Point Performance Across Models', fontsize=16, weight='bold')
    plt.xlabel('Average Max Drawdown (Lower is Better)', fontsize=12)
    plt.ylabel('Average Sharpe Ratio (Higher is Better)', fontsize=12)
    plt.grid(True)
    plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, 'avg_knee_points_comparison.png'), dpi=300)
    plt.show()

    # 3. Compare Knee Points by Volatility Category (Table)
    if 'volatility_category' in combined_df.columns:
        logging.info("Comparing average knee points by volatility category across models.")
        avg_knee_points_by_model_vol = combined_df.groupby(['model_type', 'volatility_category'])[['sharpe_ratio', 'max_drawdown']].mean().reset_index()
        
        # Pivot table for better readability
        pivot_sr = avg_knee_points_by_model_vol.pivot_table(index='volatility_category', columns='model_type', values='sharpe_ratio')
        pivot_dd = avg_knee_points_by_model_vol.pivot_table(index='volatility_category', columns='model_type', values='max_drawdown')

        logging.info("\nAverage Sharpe Ratio by Volatility Category and Model:")
        logging.info(pivot_sr.to_string(float_format="%.4f"))
        logging.info("\nAverage Max Drawdown by Volatility Category and Model:")
        logging.info(pivot_dd.to_string(float_format="%.4f"))

        # 4. Compare Knee Points by Volatility Category (Plots)
        # Plot Sharpe Ratio by Volatility
        plt.figure(figsize=(12, 7))
        sns.barplot(data=avg_knee_points_by_model_vol, x='volatility_category', y='sharpe_ratio', hue='model_type', palette='viridis')
        plt.title('Average Sharpe Ratio by Volatility Category and Model', fontsize=16, weight='bold')
        plt.xlabel('Volatility Category', fontsize=12)
        plt.ylabel('Average Sharpe Ratio', fontsize=12)
        plt.grid(axis='y')
        plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_output_dir, 'sharpe_by_volatility_comparison.png'), dpi=300)
        plt.show()

        # Plot Max Drawdown by Volatility
        plt.figure(figsize=(12, 7))
        sns.barplot(data=avg_knee_points_by_model_vol, x='volatility_category', y='max_drawdown', hue='model_type', palette='magma')
        plt.title('Average Max Drawdown by Volatility Category and Model', fontsize=16, weight='bold')
        plt.xlabel('Volatility Category', fontsize=12)
        plt.ylabel('Average Max Drawdown', fontsize=12)
        plt.grid(axis='y')
        plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_output_dir, 'drawdown_by_volatility_comparison.png'), dpi=300)
        plt.show()

    else:
        logging.info("Skipping volatility-based comparison as volatility data is not available or merged correctly.")

    logging.info("Model performance comparison complete.")

if __name__ == "__main__":
    main()