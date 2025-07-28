import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import cm
from matplotlib.colors import Normalize
import argparse

# ---------------------- CONFIG ----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='arima')
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--json', default=None)
    parser.add_argument('--out-csv', default=None)
    parser.add_argument('--fig-dir', default=None)
    return parser.parse_args()

args = parse_args()
MODEL_TYPE = args.model
RETRAIN_INTERVAL = args.interval
TIER2_JSON_PATH = args.json or f"data/tuning_results/jsons/tier2_{MODEL_TYPE}_grouped.json"
OUTPUT_CSV = args.out_csv or f"data/backtest_configs/knee_point_and_best_{MODEL_TYPE}.csv"
FIGURE_DIR = args.fig_dir or f"data/tuning_results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------ KNEE POINT FINDER ----------------
def find_knee_point_from_df(df_pareto: pd.DataFrame) -> pd.Series:
    dd = df_pareto['max_drawdown'].to_numpy()
    sr = df_pareto['sharpe_ratio'].to_numpy()
    min_dd, max_dd = dd.min(), dd.max()
    min_sr, max_sr = sr.min(), sr.max()

    norm_dd = np.zeros_like(dd) if max_dd == min_dd else (dd - min_dd) / (max_dd - min_dd)
    norm_sr = np.zeros_like(sr) if max_sr == min_sr else (sr - min_sr) / (max_sr - min_sr)

    dist = np.sqrt(norm_dd**2 + (1 - norm_sr)**2)
    idx = np.argmin(dist)
    return df_pareto.iloc[idx]

# ------------------ PARETO PLOTTER -------------------
def plot_pareto_front(df, fold_id, ticker, knee, best, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = Normalize(vmin=df['threshold'].min(), vmax=df['threshold'].max())
    cmap = cm.viridis_r
    sc = ax.scatter(df['max_drawdown'], df['sharpe_ratio'], c=df['threshold'], cmap=cmap, s=80, edgecolors='k', alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Threshold")

    ax.plot(knee['max_drawdown'], knee['sharpe_ratio'], 'gP', label="Knee Point", markersize=12)
    ax.plot(best['max_drawdown'], best['sharpe_ratio'], 'rX', label="Best Sharpe", markersize=12)

    ax.set_xlabel("Max Drawdown", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title(f"Pareto Front - Fold {fold_id} | {ticker}", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ------------------------ MAIN ------------------------
def main():
    if not os.path.exists(TIER2_JSON_PATH):
        logging.error(f"File not found: {TIER2_JSON_PATH}")
        return

    with open(TIER2_JSON_PATH, "r") as f:
        data = json.load(f)

    all_configs = []
    skipped = 0

    for entry in data:
        fold_id = entry.get("fold_id")
        ticker = entry.get("ticker", f"fold_{fold_id}")
        pareto = entry.get("pareto_front", [])
        
        if not pareto or not isinstance(pareto, list) or len(pareto) == 0:
            logging.warning(f"Fold {fold_id}: Empty or missing pareto_front.")
            skipped += 1
            continue

        df = pd.DataFrame(pareto)
        
        expected_cols = {'p', 'q', 'threshold', 'sharpe_ratio', 'max_drawdown', 'retrain_interval'}
        if not expected_cols.issubset(df.columns):
            logging.warning(f"Fold {fold_id}: Missing expected keys in pareto_front.")
            skipped += 1
            continue

        df = df[df['retrain_interval'] == RETRAIN_INTERVAL]

        if df.empty:
            logging.warning(f"Fold {fold_id}: No configs for retrain_interval={RETRAIN_INTERVAL}")
            skipped += 1
            continue

        if df.shape[0] == 1:
            knee = df.iloc[0]
        else:
            knee = find_knee_point_from_df(df)

        best = df.loc[df['sharpe_ratio'].idxmax()]

        # Calculate drawdown_improvement_pct
        drawdown_improvement_pct = 0.0
        if best['max_drawdown'] != 0:
            drawdown_improvement_pct = ((best['max_drawdown'] - knee['max_drawdown']) / best['max_drawdown']) * 100

        all_configs.append({
            'config_type': 'knee',
            'fold_id': fold_id,
            'ticker': ticker,
            'p': int(knee['p']),
            'q': int(knee['q']),
            'threshold': float(knee['threshold']),
            'sharpe_ratio': float(knee['sharpe_ratio']),
            'max_drawdown': float(knee['max_drawdown']),
            'retrain_interval': RETRAIN_INTERVAL,
            'drawdown_improvement_pct': drawdown_improvement_pct
        })

        all_configs.append({
            'config_type': 'best',
            'fold_id': fold_id,
            'ticker': ticker,
            'p': int(best['p']),
            'q': int(best['q']),
            'threshold': float(best['threshold']),
            'sharpe_ratio': float(best['sharpe_ratio']),
            'max_drawdown': float(best['max_drawdown']),
            'retrain_interval': RETRAIN_INTERVAL,
            'drawdown_improvement_pct': 0.0 
        })

        # Save figure
        fig_path = os.path.join(FIGURE_DIR, f"arima_pareto_fold_{fold_id}_{ticker}.png")
        plot_pareto_front(df, fold_id, ticker, knee, best, fig_path)

    # Save all knee configs
    df_out = pd.DataFrame(all_configs)
    df_out.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"\nSaved {len(df_out)} configs to: {OUTPUT_CSV}")
    logging.info(f"Skipped {skipped} folds due to missing data.")

if __name__ == "__main__":
    main()