import pandas as pd
import json
import os
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
from tqdm import tqdm

# -----------------------------------------------------------------------------
#
# Takes the MOGA tuning results for a given model type (e.g., ARIMA, LSTM)
# and fold‐volatility categorization, computes:
#   1) a global “knee point” average threshold,
#   2) knee points per fold,
#   3) volatility‐adaptive average thresholds,
# and writes out CSVs for downstream visualization.
#
# Usage:
#   python post_process_thresholds.py [model_type]
# Default model_type is 'arima'.
# -----------------------------------------------------------------------------

def find_knee(df_pareto: pd.DataFrame) -> float:
    """
    Given a DataFrame with columns 'max_drawdown' and 'sharpe_ratio',
    normalize both axes, compute distance to ideal (0 drawdown, 1 Sharpe),
    and return the 'threshold' at the minimum distance.
    """
    dd = df_pareto['max_drawdown'].to_numpy()
    sr = df_pareto['sharpe_ratio'].to_numpy()
    min_dd, max_dd = dd.min(), dd.max()
    min_sr, max_sr = sr.min(), sr.max()

    norm_dd = np.zeros_like(dd) if max_dd == min_dd else (dd - min_dd) / (max_dd - min_dd)
    norm_sr = np.zeros_like(sr) if max_sr == min_sr else (sr - min_sr) / (max_sr - min_sr)

    dist = np.sqrt(norm_dd**2 + (1 - norm_sr)**2)
    idx = np.argmin(dist)
    return df_pareto.iloc[idx]['threshold']


def main(model_type: str = 'arima'):
    """
    Main entry point for post-processing MOGA results.
    :param model_type: string identifier for the model (e.g., 'arima', 'lstm')
    """
    logging.info(f"Starting post‐processing of {model_type.upper()} MOGA results.")

    results_dir = f'data/tuning_results'
    os.makedirs(results_dir, exist_ok=True)

    # Input paths
    moga_results_path = os.path.join(results_dir, f'final_moga_{model_type}_results.json')
    volatility_path   = os.path.join('data/processed_folds', 'shared_meta', 'fold_volatility_categorized.csv')

    # Output CSV paths
    csv_dir = os.path.join(results_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    global_csv   = os.path.join(csv_dir, f'global_avg_threshold_{model_type}.csv')
    knees_csv    = os.path.join(csv_dir, f'knee_points_{model_type}.csv')
    adaptive_csv = os.path.join(csv_dir, f'adaptive_thresholds_by_volatility_{model_type}.csv')

    # 1) Load MOGA results
    try:
        with open(moga_results_path, 'r') as f:
            moga_results = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {moga_results_path}. Please run the MOGA script first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Malformed JSON in {moga_results_path}: {e}")
        sys.exit(1)

    # 2) Load volatility categories
    try:
        volatility_df = pd.read_csv(volatility_path)
    except FileNotFoundError:
        logging.error(f"File not found: {volatility_path}. Please ensure volatility data is available.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"Volatility CSV is empty or malformed: {volatility_path}")
        sys.exit(1)

    # 3) Extract knee‐point thresholds for global average
    logging.info("Extracting global knee‐point thresholds...")
    knee_thresholds = []
    for fold_data in tqdm(moga_results, desc="Global knee extraction"):
        df_pf = pd.DataFrame(fold_data['pareto_front'])
        if len(df_pf) > 1:
            knee_thresholds.append(find_knee(df_pf))

    # 4) Compute & save global average
    if knee_thresholds:
        avg_thr = np.mean(knee_thresholds)
        logging.info(f"Global average threshold: {avg_thr:.6f}")
        df_global = pd.DataFrame([{
            'Model': model_type,
            'Strategy': 'Global Average Rule',
            'Optimal Threshold': f"{avg_thr:.6f}"
        }])
        df_global.to_csv(global_csv, index=False)
        logging.info(f"Global average saved to {global_csv}")
    else:
        logging.warning("No valid Pareto fronts found for global average.")

    # 5) Extract knee‐points per fold for adaptive rule
    logging.info("Extracting knee‐points for adaptive rule...")
    all_knees = []
    for fold_data in tqdm(moga_results, desc="Adaptive knee extraction"):
        df_pf = pd.DataFrame(fold_data['pareto_front'])
        if len(df_pf) > 1:
            thresh = find_knee(df_pf)
            row = df_pf[df_pf['threshold']==thresh].iloc[0]
            all_knees.append({
            'global_fold_id': fold_data['fold_id'],
            'threshold': thresh,
            'sharpe_ratio': row['sharpe_ratio'],
            'max_drawdown': row['max_drawdown']
            })

    if all_knees:
        kp_df = pd.DataFrame(all_knees)
        kp_df.to_csv(knees_csv, index=False)
        logging.info(f"Knee points per fold saved to {knees_csv}")
    else:
        logging.warning("No knee points extracted for adaptive analysis.")
        return

    # 6) Merge with volatility and compute adaptive thresholds
    if 'global_fold_id' not in volatility_df.columns or 'volatility_category' not in volatility_df.columns:
        logging.error("Volatility CSV missing required columns: 'global_fold_id','volatility_category'")
        sys.exit(1)

    merged = pd.merge(kp_df,
                      volatility_df[['global_fold_id','volatility_category']],
                      on='global_fold_id', how='inner')
    if merged.empty:
        logging.warning("No matching fold IDs between knee points and volatility data.")
        return

    adaptive = merged.groupby('volatility_category')['threshold'].mean().reset_index()
    adaptive['Optimal Threshold'] = adaptive['threshold'].map(lambda x: f"{x:.6f}")
    adaptive = adaptive[['volatility_category','Optimal Threshold']].rename(
        columns={'volatility_category':'Volatility Category'})

    adaptive.to_csv(adaptive_csv, index=False)
    logging.info(f"Adaptive thresholds by volatility saved to {adaptive_csv}")

    logging.info("Post‐processing complete.")

if __name__ == "__main__":
    # allow optional model_type argument
    model = sys.argv[1] if len(sys.argv) > 1 else 'arima'
    main(model)
