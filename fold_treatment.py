import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import argparse

def analyze_fold_summary(fold_summary_path, model_type, path_key, output_json=True):
    with open(fold_summary_path) as f:
        folds = json.load(f)

    ratios = []
    tickers = []
    fold_ids = []
    cleaned_folds = []

    for i, fold in enumerate(folds):
        val_meta_path = os.path.join('data/processed_folds', f'{model_type}_meta', f'val_meta_fold_{fold["global_fold_id"]}.csv')
        if not os.path.exists(val_meta_path):
            print(f"[WARN] Missing: {val_meta_path}")
            continue

        df = pd.read_csv(val_meta_path)
        if 'target' not in df.columns:
            print(f"[WARN] No 'target' column in: {val_meta_path}")
            continue

        ratio = df['target'].mean()
        ratios.append(ratio)
        tickers.append(fold['ticker'])
        fold_ids.append(i)

        fold['positive_ratio'] = ratio  

    fold_df = pd.DataFrame({
        'fold_id': fold_ids,
        'ticker': tickers,
        'positive_ratio': ratios
    })

    Q1 = fold_df['positive_ratio'].quantile(0.25)
    Q3 = fold_df['positive_ratio'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = fold_df[(fold_df['positive_ratio'] < lower_bound) | (fold_df['positive_ratio'] > upper_bound)]
    filtered_fold_df = fold_df[~fold_df.index.isin(outliers.index)]

    print(f"Total folds: {len(fold_df)}")
    print(f"Outliers removed: {len(outliers)}")
    print(f"Remaining folds: {len(filtered_fold_df)}")

    if output_json:
        cleaned_folds = [fold for i, fold in enumerate(folds) if i in filtered_fold_df['fold_id'].values]
        output_path = fold_summary_path.replace(".json", "_cleaned.json")
        with open(output_path, 'w') as f:
            json.dump(cleaned_folds, f, indent=4)
        print(f"[INFO] Saved cleaned fold summary to: {output_path}")

    # PLOTTING 
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(fold_df['positive_ratio'], bins=15, edgecolor='black')
    plt.title(f'{model_type.upper()} – Histogram of Positive Label Ratio')
    plt.xlabel('Positive Label Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    hist_path = f"data/figures/{model_type}_positive_ratio_hist.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved histogram to: {hist_path}")

    plt.figure(figsize=(15, 6))
    sns.boxplot(data=fold_df, x='ticker', y='positive_ratio')
    plt.xticks(rotation=90)
    plt.title(f'{model_type.upper()} – Boxplot of Positive Label Ratio per Ticker')
    plt.ylabel('Positive Label Ratio')
    plt.xlabel('Ticker')
    plt.grid(True)
    plt.tight_layout()
    boxplot_path = f"data/figures/{model_type}_positive_ratio_boxplot.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved boxplot to: {boxplot_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fold summary and plot label ratios")
    parser.add_argument("--fold_summary_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"])
    parser.add_argument("--path_key", type=str, required=True)
    args = parser.parse_args()

    analyze_fold_summary(
        fold_summary_path=args.fold_summary_path,
        model_type=args.model_type,
        path_key=args.path_key
    )