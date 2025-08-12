import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==== CONFIG ====
BASE_DIR = "data"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "processed_folds")

LSTM_FEATURE_COLUMNS_PATH = os.path.join(OUTPUT_BASE_DIR, "lstm_feature_columns.json")
DEFAULT_FOLDS_SUMMARY = os.path.join(OUTPUT_BASE_DIR, "folds_summary_lstm_cleaned_selection_pca.json")
NON_FEATURE_COLS = ['Date', 'Ticker', 'target_log_returns', 'target', 'Log_Returns', 'Close_raw']

FIG_DIR = "data/figures"
SCREE_PLOT_PATH = os.path.join(FIG_DIR, "pca_scree_plot_lstm_all_train_folds.png")
EVR_CSV_PATH = os.path.join(FIG_DIR, "pca_explained_variance_lstm_all_train_folds.csv")

os.makedirs(FIG_DIR, exist_ok=True)


def load_feature_columns():
    if os.path.exists(LSTM_FEATURE_COLUMNS_PATH):
        with open(LSTM_FEATURE_COLUMNS_PATH, "r") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or not cols:
            raise ValueError(f"[ERROR] Invalid feature list in {LSTM_FEATURE_COLUMNS_PATH}")
        return cols
    raise FileNotFoundError(f"[ERROR] Missing locked feature columns: {LSTM_FEATURE_COLUMNS_PATH}")

def absolute_path_from_rel(rel_path: str) -> str:
    return os.path.join(OUTPUT_BASE_DIR, rel_path)

def main():
    ap = argparse.ArgumentParser(description="Generate Scree Plot from all LSTM train folds")
    ap.add_argument("--folds_summary_path", type=str, default=DEFAULT_FOLDS_SUMMARY,
                    help="Path to cleaned LSTM folds summary JSON")
    args = ap.parse_args()

    feature_cols = load_feature_columns()

    with open(args.folds_summary_path, "r") as f:
        folds = json.load(f)
    print(f"[INFO] Loaded cleaned folds: {len(folds)} from {args.folds_summary_path}")

    train_dfs = []
    for fold in folds:
        tr_rel = fold.get("train_path_lstm")
        if tr_rel is None:
            continue
        tr_path = absolute_path_from_rel(tr_rel)
        if not os.path.exists(tr_path):
            continue
        df_train = pd.read_csv(tr_path)
        train_dfs.append(df_train[feature_cols])
    if not train_dfs:
        raise RuntimeError("[ERROR] No train data found for any fold.")
    
    full_train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
    print(f"[INFO] Combined train shape: {full_train_df.shape}")

    pca = PCA(n_components=None, random_state=42)
    pca.fit(full_train_df.values)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    evr_df = pd.DataFrame({
        "PC": np.arange(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_evr
    })
    evr_df.to_csv(EVR_CSV_PATH, index=False)
    print(f"[INFO] Saved EVR table -> {EVR_CSV_PATH}")

    # Plot scree plot
    plt.figure(figsize=(10, 7))
    xs = np.arange(1, len(evr) + 1)
    plt.plot(xs, evr, marker='o', linestyle='--', label='Individual Explained Variance')
    plt.plot(xs, cum_evr, marker='o', linestyle='-', label='Cumulative Explained Variance')


    for thr in [0.8, 0.9, 0.95]:
        k = np.searchsorted(cum_evr, thr) + 1
        plt.axhline(y=thr, color='r', linestyle='--', alpha=0.4)
        plt.axvline(x=k, color='r', linestyle='--', alpha=0.4)
        plt.text(k, thr, f'  {int(thr*100)}% @ PC{k}', va='bottom', ha='left', fontsize=9)

    plt.title('Scree Plot – LSTM All Train Folds')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(SCREE_PLOT_PATH, dpi=300)
    print(f"[INFO] Saved scree plot -> {SCREE_PLOT_PATH}")

if __name__ == "__main__":
    main()