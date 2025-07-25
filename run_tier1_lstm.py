import os, json, time, argparse, logging, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error
from math import sqrt

from skopt import gp_minimize
from skopt.space import Integer, Real
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as ga_minimize
from pymoo.core.problem import ElementwiseProblem

from tensorflow.keras import Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ------------------------ HELPERS ----------------------------
def get_params_for_volatility(category: str):
    category = category.lower()
    if category == 'low':
        return 10, 8, 10, 3  # GA pop, gen, BO calls, top_n
    elif category == 'medium':
        return 15, 10, 12, 4
    elif category == 'high':
        return 18, 15, 15, 5
    else:
        return 15, 10, 12, 4
    
def create_windows(X, y, lookback_window):
    """
    Build (n_samples, lookback_window, n_features) windows from
    X.shape == (n_timesteps, n_features), with targets y[lookback_window:].
    """
    N, F = X.shape
    W = lookback_window

    if N <= W:
        raise ValueError(f"Not enough rows ({N}) for lookback {W}")

    all_windows = sliding_window_view(X, window_shape=W, axis=0)
    windows = all_windows[:-1]
    targets = y[W:]

    if windows.shape[1] != W or windows.shape[2] != F:
        windows = windows.transpose(0, 2, 1)

    assert windows.shape == (N - W, W, F), (
        f"windows {windows.shape} ≠ expected ({N-W},{W},{F})"
    )
    assert len(targets) == N - W
    return windows, targets

def build_lstm(window, units, lr, num_features):
    model = Sequential([
        Input((window, num_features)),
        LSTM(units, return_sequences=True),
        BatchNormalization(), Dropout(0.2),
        LSTM(units // 2),
        BatchNormalization(), Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse')
    return model

class RMSEProblem(ElementwiseProblem):
    def __init__(self, X_train, y_train, X_val, y_val, num_features):
        xl=np.array([15, 32, 1e-4, 10, 0.0]),
        xu=np.array([35, 64, 1e-2, 50, 1.0]),
        super().__init__(n_var=6, n_obj=1, xl=xl, xu=xu)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_features = num_features

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            w, units, lr, epochs, threshold = x
            w, units, epochs = map(int, [w, units, epochs])

            logging.info(f"Evaluating: window={w}, layers={layers}, units={units}, "
                     f"dropout={dropout:.3f}, lr={lr:.5f}, batch={batch}")
            
            Xtr, ytr = create_windows(self.X_train, self.y_train, w)
            Xvl, yvl = create_windows(np.concatenate([self.X_train[-w:], self.X_val], axis=0),
                                      np.concatenate([self.y_train[-w:], self.y_val]), w)

            model = build_lstm(w, units, lr, self.num_features)
            model.fit(
                Xtr, ytr,
                epochs=epochs,
                batch_size=32,
                callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)],
                verbose=0
            )
            preds = model.predict(Xvl, verbose=0).ravel()
            rmse = sqrt(mean_squared_error(yvl, preds))

        except Exception as e:
            logging.warning(f"LSTM eval error: {e}")
            rmse = 1e6

        out['F'] = [rmse]
        K.clear_session()
        gc.collect()

# ------------------------ MAIN ----------------------------
def main():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('logs/tier1_lstm.log'),
            logging.StreamHandler()
        ]
    )
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser("Tier-1 GA→BO tuning for LSTM")
    parser.add_argument('--max-folds', type=int, default=None, help="Debug: only run first N folds")
    parser.add_argument('--data-dir', type=str, default='data/processed_folds',
                        help="Folder with processed_folds/{train,val}_lstm_gru")
    parser.add_argument('--out-file', type=str,
                        default='data/tuning_results/jsons/tier1_lstm.json',
                        help="Where to write tier-1 LSTM results")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    logging.info(f"Saving results to {args.out_file}")

    # ─── LOAD PREVIOUSLY COMPLETED ──────────────────────────
    try:
        results = json.load(open(args.out_file))
        done = {r['fold_id'] for r in results}
        logging.info(f"Resuming from {len(done)} completed folds")
    except:
        results, done = [], set()

    # ─── LOAD FOLD IDS & SUMMARY ────────────────────────────
    rep_path = os.path.join(args.data_dir, 'shared_meta/representative_fold_ids.json')
    sum_path = os.path.join(args.data_dir, 'folds_summary.json')
    reps = json.load(open(rep_path))
    summary = {f['global_fold_id']: f for f in json.load(open(sum_path))}

    vol_path = os.path.join(args.data_dir, 'shared_meta/fold_volatility_categorized.csv')
    vol_df = pd.read_csv(vol_path)
    vol_map = dict(zip(vol_df['global_fold_id'], vol_df['volatility_category']))

    # ─── DEFINE SEARCH SPACES ──────────────────────────────
    dl_space = [
        Integer(10, 40, name='window'),
        Integer(1, 3, name='layers'),
        Integer(32, 64, name='units'),
        Real(0.2, 0.4, name='dropout'),
        Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
        Integer(16, 64, name='batch')
    ]

    # ─── ITERATE FOLDS ────────────────────────────────────
    pending = [fid for fid in reps if fid not in done]
    if args.max_folds is not None:
        pending = pending[:args.max_folds]

    for fid in tqdm(pending, desc="Tier1 LSTM"):
        logging.info(f"Starting fold {fid}")
        info = summary[fid]
        train_df = pd.read_csv(os.path.join(args.data_dir, info['train_path_lstm_gru']), parse_dates=['Date'])
        val_df = pd.read_csv(os.path.join(args.data_dir, info['val_path_lstm_gru']), parse_dates=['Date'])
        feats = [c for c in train_df.columns if c not in ['Date', 'Ticker', 'Log_Returns', 'target']]
        X_train, y_train = train_df[feats].values, train_df['target'].values
        X_val, y_val = val_df[feats].values, val_df['target'].values
        num_features = X_train.shape[1]

        category = vol_map.get(fid, 'medium')
        ga_pop, ga_gen, bo_calls, top_n = get_params_for_volatility(category)
        logging.info(f"Fold {fid}: Volatility={category}, GA(pop={ga_pop}, gen={ga_gen}), BO(calls={bo_calls}), top_n={top_n}")
        start = time.time()

        # GA optimization
        ga_res = ga_minimize(
            RMSEProblem(X_train, y_train, X_val, y_val, num_features),
            GA(pop_size=ga_pop),
            ('n_gen', ga_gen),
            verbose=False
        )
        # select top seeds
        top_inds = sorted(ga_res.pop, key=lambda ind: ind.F[0])[:top_n]
        x0 = [list(map(lambda z: int(z) if ind in [0, 1, 2, 5] else float(z), ind.X)) for ind in top_inds]
        y0 = [float(ind.F[0]) for ind in top_inds]
        
        logging.info(f"Fold {fid}: BO(n_calls={bo_calls})")
        def bo_func(x):
            out = {}
            RMSEProblem(X_train, y_train, X_val, y_val, num_features)._evaluate(x, out)
            return out['F'][0]

        bo = gp_minimize(
            func=bo_func,
            dimensions=dl_space,
            n_calls=bo_calls,
            x0=x0,
            y0=y0,
            random_state=42
        )

        best_rmse = float(bo.fun)
        best = bo.x

        results.append({
            'fold_id': fid,
            'best_params': {
                'window': int(best[0]),
                'layers': int(best[1]),
                'units': int(best[2]),
                'dropout': float(best[3]),
                'lr': float(best[4]),
                'batch': int(best[5])
            },
            'best_rmse': best_rmse,
            'top_ga': x0
        })

        # checkpoint
        with open(args.out_file, 'w') as f:
            json.dump(results, f, indent=2)
        csv_path = args.out_file.replace('.json', '.csv').replace('jsons', 'csv')
        pd.DataFrame(results).to_csv(csv_path, index=False)

        elapsed = time.time() - start
        logging.info(f"→ Fold {fid} done in {elapsed:.1f}s | RMSE={best_rmse:.4f}")

    logging.info("=== Tier-1 LSTM complete ===")
    print(f"Results saved to {args.out_file}")

if __name__ == '__main__':
    main()