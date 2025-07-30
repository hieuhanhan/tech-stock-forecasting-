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
FIGURE_DIR = args.fig_dir or f"data/figures"

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

    if 'config_type' not in df.columns:
        logging.error("Missing 'config_type' column in input CSV.")
        return

    # --- Plot: Sharpe Ratio Distribution ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='config_type', y='sharpe_ratio', palette={'knee': 'green', 'best': 'red'})
    plt.title(f'Sharpe Ratio Distribution ({MODEL_TYPE.upper()})')
    plt.xlabel('Configuration Type')
    plt.ylabel('Sharpe Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    sharpe_path = os.path.join(FIGURE_DIR, f"sharpe_boxplot_{MODEL_TYPE}.png")
    plt.savefig(sharpe_path, dpi=300)
    plt.close()
    logging.info(f"Saved Sharpe Ratio boxplot: {sharpe_path}")

    # --- Plot: Max Drawdown Distribution ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='config_type', y='max_drawdown', palette={'knee': 'green', 'best': 'red'})
    plt.title(f'Max Drawdown Distribution ({MODEL_TYPE.upper()})')
    plt.xlabel('Configuration Type')
    plt.ylabel('Max Drawdown')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    drawdown_path = os.path.join(FIGURE_DIR, f"drawdown_boxplot_{MODEL_TYPE}.png")
    plt.savefig(drawdown_path, dpi=300)
    plt.close()
    logging.info(f"Saved Max Drawdown boxplot: {drawdown_path}")

    logging.info("Summary plotting complete.")

if __name__ == "__main__":
    main()