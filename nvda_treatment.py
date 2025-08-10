import os
import json
import pandas as pd
import argparse

def analyze_nvda(fold_summary_path, model_type, output_json=True,
                 min_pos=0.10, max_pos=0.90, min_vol=0.01, save_dropped=False):
    meta_dir = os.path.join("data", "processed_folds", f"{model_type}_meta")

    with open(fold_summary_path) as f:
        folds = json.load(f)

    filtered_folds = []
    nvda_problem_folds = []
    nvda_dropped = []

    for fold in folds:
        if fold['ticker'] != 'NVDA':
            filtered_folds.append(fold)
            continue

        meta_path = os.path.join(meta_dir, f'val_meta_fold_{fold["global_fold_id"]}.csv')
        if not os.path.exists(meta_path):
            print(f"[WARN] Missing {meta_path}")
            nvda_problem_folds.append(fold['global_fold_id'])
            if save_dropped:
                nvda_dropped.append({**fold, "drop_reason": "missing_meta"})
            continue

        df = pd.read_csv(meta_path)

        if 'target' not in df.columns:
            print(f"[WARN] No target in {meta_path}")
            nvda_problem_folds.append(fold['global_fold_id'])
            if save_dropped:
                nvda_dropped.append({**fold, "drop_reason": "missing_target"})
            continue

        target_ratio = float(df['target'].mean())

        if model_type == 'arima':
            if 'Log_Returns' not in df.columns:
                print(f"[WARN] No Log_Returns in {meta_path}")
                nvda_problem_folds.append(fold['global_fold_id'])
                if save_dropped:
                    nvda_dropped.append({**fold, "drop_reason": "missing_log_returns"})
                continue
            val_vol = float(df['Log_Returns'].std())
            if (target_ratio < min_pos or target_ratio > max_pos) or (val_vol < min_vol):
                print(f"[DROP] Fold {fold['global_fold_id']} (NVDA) - ratio: {target_ratio:.2f}, vol: {val_vol:.4f}")
                nvda_problem_folds.append(fold['global_fold_id'])
                if save_dropped:
                    nvda_dropped.append({**fold, "drop_reason": "imbalance_or_flat_vol"})
            else:
                filtered_folds.append(fold)
        
        elif model_type == 'lstm':
            pc_cols = [f'PC{i}' for i in range(1, 7)]
            has_pca_cols = all(col in df.columns for col in pc_cols)
            if not has_pca_cols:
                print(f"[WARN] Missing PCA columns in fold {fold['global_fold_id']}")
                nvda_problem_folds.append(fold['global_fold_id'])
                if save_dropped:
                    nvda_dropped.append({**fold, "drop_reason": "missing_pca"})
                continue
            pca_var_mean = float(df[pc_cols].var().mean())
            if (target_ratio < min_pos or target_ratio > max_pos) or (pca_var_mean < 0.01):
                print(f"[DROP] Fold {fold['global_fold_id']} (NVDA) - ratio: {target_ratio:.2f}, mean PC var: {pca_var_mean:.4f}")
                nvda_problem_folds.append(fold['global_fold_id'])
                if save_dropped:
                    nvda_dropped.append({**fold, "drop_reason": "imbalance_or_flat_pca"})
            else:
                filtered_folds.append(fold)

    print(f"[INFO] Removed {len(nvda_problem_folds)} NVDA folds due to imbalance or flat price.")
    print(f"[INFO] Remaining folds: {len(filtered_folds)}")

    if output_json:
        cleaned_path = fold_summary_path.replace('.json', '_nvda_cleaned.json')
        with open(cleaned_path, 'w') as f:
            json.dump(filtered_folds, f, indent=4)
        print(f"[INFO] Saved cleaned summary to: {cleaned_path}")

        if save_dropped:
            dropped_path = fold_summary_path.replace('.json', '_nvda_dropped.json')
            with open(dropped_path, 'w') as f:
                json.dump(nvda_dropped, f, indent=4)
            print(f"[INFO] Saved dropped NVDA folds to: {dropped_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean NVDA folds from summary")
    parser.add_argument("--fold_summary_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"])
    parser.add_argument("--min_pos", type=float, default=0.10)
    parser.add_argument("--max_pos", type=float, default=0.90)
    parser.add_argument("--min_vol", type=float, default=0.01) 
    parser.add_argument("--save_dropped", action="store_true")
    args = parser.parse_args()

    analyze_nvda(
        fold_summary_path=args.fold_summary_path,
        model_type=args.model_type,
        output_json=True,
        min_pos=args.min_pos,
        max_pos=args.max_pos,
        min_vol=args.min_vol,
        save_dropped=args.save_dropped
    )
