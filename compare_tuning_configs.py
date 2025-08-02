import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

# ---------------------- CONFIG ----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='arima')
    parser.add_argument('--input-csv', default=None)
    parser.add_argument('--fig-dir', default=None)
    return parser.parse_args()

args = parse_args()
MODEL_TYPE = args.model
INPUT_CSV = args.input_csv or f"data/backtest_configs/knee_point_and_best_{MODEL_TYPE}.csv"
FIGURE_DIR = args.fig_dir or f"data/tuning_results/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------ MAIN ------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        logging.warning("Input CSV is empty. No plots to generate.")
        return

    # Ensure 'config_type' column exists
    if 'config_type' not in df.columns:
        logging.error("The 'config_type' column is missing in the input CSV. Please run the updated main script first.")
        return

    # Filter for knee and best configs
    df_knee = df[df['config_type'] == 'knee']
    df_best = df[df['config_type'] == 'best']

    if df_knee.empty or df_best.empty:
        logging.warning("No 'knee' or 'best' configurations found in the CSV. Cannot generate plots.")
        return

    # Merge to calculate differences
    df_merged = pd.merge(
        df_knee[['fold_id', 'ticker', 'sharpe_ratio', 'max_drawdown']],
        df_best[['fold_id', 'ticker', 'sharpe_ratio', 'max_drawdown']],
        on=['fold_id', 'ticker'],
        suffixes=('_knee', '_best')
    )

    # Calculate drawdown improvement %
    df_merged['drawdown_improvement_pct'] = (
        (df_merged['max_drawdown_best'] - df_merged['max_drawdown_knee']) / df_merged['max_drawdown_best']
    ) * 100

    # Save CSV with improvement stats
    improved_csv_path = INPUT_CSV.replace('.csv', '_with_improvement.csv')
    df_merged.to_csv(improved_csv_path, index=False)
    logging.info(f"Saved CSV with drawdown improvement to: {improved_csv_path}")

    # --- Plotting Sharpe Ratio Boxplot ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='config_type', y='sharpe_ratio', palette={'knee': 'green', 'best': 'red'})
    plt.title(f'Sharpe Ratio Distribution for {MODEL_TYPE.upper()} - Knee vs. Best')
    plt.xlabel('Configuration Type')
    plt.ylabel('Sharpe Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sharpe_plot_path = os.path.join(FIGURE_DIR, f"sharpe_boxplot_{MODEL_TYPE}.png")
    plt.savefig(sharpe_plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved Sharpe Ratio boxplot to: {sharpe_plot_path}")

    # --- Plotting Max Drawdown Boxplot ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='config_type', y='max_drawdown', palette={'knee': 'green', 'best': 'red'})
    plt.title(f'Max Drawdown Distribution for {MODEL_TYPE.upper()} - Knee vs. Best')
    plt.xlabel('Configuration Type')
    plt.ylabel('Max Drawdown')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    drawdown_plot_path = os.path.join(FIGURE_DIR, f"drawdown_boxplot_{MODEL_TYPE}.png")
    plt.savefig(drawdown_plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved Max Drawdown boxplot to: {drawdown_plot_path}")

    # --- Plotting Drawdown Improvement Percentage ---
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_merged.sort_values(by='drawdown_improvement_pct', ascending=False),
        x='ticker', y='drawdown_improvement_pct', palette='viridis'
    )
    plt.title(f'Drawdown Improvement (%) of Knee vs. Best - {MODEL_TYPE.upper()}')
    plt.xlabel('Ticker')
    plt.ylabel('Drawdown Improvement (%)')
    plt.xticks(rotation=90)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=min(0, df_merged['drawdown_improvement_pct'].min() - 5))
    plt.tight_layout()
    improvement_path = os.path.join(FIGURE_DIR, f"drawdown_improvement_bar_{MODEL_TYPE}.png")
    plt.savefig(improvement_path, dpi=300)
    plt.close()
    logging.info(f"Saved Drawdown Improvement (%) barplot to: {improvement_path}")

    logging.info("Summary plotting complete.")

if __name__ == "__main__":
    main()