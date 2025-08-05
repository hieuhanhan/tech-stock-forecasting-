import os
import json
import pandas as pd
import argparse

def analyze_nvda(fold_summary_path, model_type, path_key, output_json=True):
    meta_dir = os.path.join("data", "processed_folds", f"{model_type}_meta")

    with open(fold_summary_path) as f:
        folds = json.load(f)

    filtered_folds = []
    nvda_problem_folds = []

    for fold in folds:
        if fold['ticker'] != 'NVDA':
            filtered_folds.append(fold)
            continue

        meta_path = os.path.join(meta_dir, f'val_meta_fold_{fold["global_fold_id"]}.csv')
        if not os.path.exists(meta_path):
            print(f"[WARN] Missing {meta_path}")
            continue

        df = pd.read_csv(meta_path)
        target_ratio = df['target'].mean()

        if model_type == 'arima':
            close_std = df['Close'].std()
            if target_ratio < 0.1 or target_ratio > 0.9 or close_std < 0.005:
                print(f"[DROP] Fold {fold['global_fold_id']} (NVDA) - target ratio: {target_ratio:.2f}, close std: {close_std:.4f}")
                nvda_problem_folds.append(fold['global_fold_id'])
            else:
                filtered_folds.append(fold)
        
        elif model_type == 'lstm':
            has_pca_cols = all(col in df.columns for col in [f'PC{i}' for i in range(1, 7)])
            if not has_pca_cols:
                print(f"[WARN] Missing PCA columns in fold {fold['global_fold_id']}")
                nvda_problem_folds.append(fold['global_fold_id'])
                continue
            
            pca_std = df[[f'PC{i}' for i in range(1, 7)]].std().mean()
            if target_ratio < 0.1 or target_ratio > 0.9 or pca_std < 0.01:
                print(f"[DROP] Fold {fold['global_fold_id']} (NVDA) - target ratio: {target_ratio:.2f}, PCA std: {pca_std:.4f}")
                nvda_problem_folds.append(fold['global_fold_id'])
            else:
                filtered_folds.append(fold)

    print(f"[INFO] Removed {len(nvda_problem_folds)} NVDA folds due to imbalance or flat price.")
    print(f"[INFO] Remaining folds: {len(filtered_folds)}")

    if output_json:
        cleaned_path = fold_summary_path.replace('.json', '_nvda_cleaned.json')
        with open(cleaned_path, 'w') as f:
            json.dump(filtered_folds, f, indent=4)
        print(f"[INFO] Saved cleaned summary to: {cleaned_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean NVDA folds from summary")
    parser.add_argument("--fold_summary_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"])
    parser.add_argument("--path_key", type=str, required=True)
    args = parser.parse_args()

    analyze_nvda(
        fold_summary_path=args.fold_summary_path,
        model_type=args.model_type,
        path_key=args.path_key
    )
