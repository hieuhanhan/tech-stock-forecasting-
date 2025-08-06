import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def analyze_fold_summary(fold_summary_path, model_type):
    try:
        with open(fold_summary_path) as f:
            folds = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {fold_summary_path}")
        return [], []

    ratios = []
    tickers = []

    for fold in folds:
        fold_id = fold.get("global_fold_id")
        ticker = fold.get("ticker")
        val_meta_path = os.path.join('data/processed_folds', f'{model_type}_meta', f'val_meta_fold_{fold_id}.csv')

        if not os.path.exists(val_meta_path):
            logging.warning(f"[WARN] Missing: {val_meta_path}")
            continue
        try:
            df = pd.read_csv(val_meta_path)
        except Exception as e:
            logging.error(f"Error reading file {val_meta_path}: {e}")
            continue

        if 'target' not in df.columns:
            logging.warning(f"[WARN] No 'target' column in: {val_meta_path}")
            continue
        
        ratio = df['target'].mean()
        ratios.append(ratio)
        tickers.append(ticker)

    return ratios, tickers

def plot_boxplot(ratios, output_path):
    if not ratios:
        logging.warning("No data to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(ratios)
    plt.title("Distribution of Positive Label Ratio across Validation Folds")
    plt.ylabel("Positive Label Ratio")
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved plot to '{output_path}'")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze positive Label Ratio across Validation Folds")
    parser.add_argument("--fold_summary_path", type=str, required=True, help="Path to fold summary JSON file.")
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"], help="Model type to analyze (e.g., 'lstm' or 'arima').")
    args = parser.parse_args()

    ratios, tickers = analyze_fold_summary(
        fold_summary_path=args.fold_summary_path,
        model_type=args.model_type,
    )

    output_dir = "data/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model_type}_positive_label_ratio_distribution.png")
    
    plot_boxplot(ratios, output_path)
