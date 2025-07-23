import os, json, time, argparse, logging, gc
from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, BatchNormalization, Dense

# ─────────────────── CONFIG & LOGGING ───────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/tier2_lstm.log"), logging.StreamHandler()]
)
np.random.seed(42)
tf.random.set_seed(42)
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

# ─────────────────── HELPERS & METRICS ──────────────────
def create_windows(X, y, W):
    N,F = X.shape
    if N<=W: raise ValueError("Not enough data")
    win = sliding_window_view(X, W, axis=0)[:-1]
    return win, y[W:]

def build_lstm(window, units, lr, layers, do):
    m = Sequential([ Input((window, X.shape[1])),
                     LSTM(units, return_sequences=(layers>1)),
                     BatchNormalization(), Dropout(do)] +
                   sum([[LSTM(units//2 if i==layers-1 else units,
                              return_sequences=(i<layers-1)),
                         BatchNormalization(), Dropout(do)]
                        for i in range(1,layers)], []) +
                   [Dense(1)])
    m.compile(tf.keras.optimizers.Adam(lr), 'mse')
    return m

def sharpe_ratio(r):
    return 0.0 if r.std()==0 else (r.mean()/r.std())*np.sqrt(252)
def max_drawdown(r):
    cum = np.exp(np.cumsum(r))
    dd  = (cum.cummax()-cum)/(cum+1e-9)
    return dd.max()

def create_objective(train_df, val_df, features, cost=0.0005):
    X, y = train_df[features].values, train_df['Log_Returns'].values
    Xv, yv= val_df[features].values,   val_df['Log_Returns'].values
    n_feat = X.shape[1]
    def obj(p):
        w,units,lr,ep,th = map(float,p)
        # 1) fit
        m = build_lstm(int(w), int(units), lr, 2, 0.2)
        win, targ = create_windows(X, y, int(w))
        m.fit(win, targ, epochs=int(ep), verbose=0)
        # 2) walk-forward
        buf = list(X[-int(w):])
        preds=[]
        for t in Xv:
            inp = np.array(buf[-int(w):])[None,:]
            p_ = m.predict(inp, verbose=0)[0,0]
            preds.append(p_)
            buf.append(t)
        preds = np.array(preds)
        mn,mx = preds.min(), preds.max()
        if mx-mn<1e-8: return (1e3,1e3)
        thr = mn + (mx-mn)*th
        sig  = (preds>thr).astype(int)
        if sig.sum()==0: return (1e3,1e3)
        r = yv*sig - cost*sig
        return (-sharpe_ratio(r), max_drawdown(r))
    return obj

class MOGAProblem(ElementwiseProblem):
    def __init__(self, obj):
        super().__init__(n_var=5, n_obj=2,
                         xl=np.array([15, 32, 1e-4, 10, 0.0]),
                         xu=np.array([35, 64, 1e-3, 50, 1.0]))
        self.obj = obj
    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

# ────────────────────── MAIN ────────────────────────────
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--n-gen',    type=int, default=15)
    p.add_argument('--data-dir', type=str, default='data/processed_folds')
    p.add_argument('--tier1-file', type=str, default='data/tuning_results/tier1_lstm.json')
    p.add_argument('--out-file', type=str, default='data/tuning_results/tier2_lstm.json')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    tier1 = json.load(open(args.tier1_file))
    summary = {r['fold_id']: r for r in tier1}

    all_out=[]
    for fid in tqdm(summary, desc="Tier2 LSTM"):
        t0=time.time()
        rec = summary[fid]
        df = json.load(open(f"{args.data_dir}/folds_summary.json"))
        meta = next(x for x in df if x['global_fold_id']==fid)

        tr = pd.read_csv(f"{args.data_dir}/{meta['train_path_lstm_gru']}"); 
        vl = pd.read_csv(f"{args.data_dir}/{meta['val_path_lstm_gru']}")
        feats=[c for c in tr.columns if c not in ['Date','Ticker','Log_Returns','target']]

        obj = create_objective(tr, vl, feats)
        prob= MOGAProblem(obj)
        res = minimize(prob, NSGA2(pop_size=args.pop_size),
                       ('n_gen', args.n_gen), seed=42, verbose=False)

        front = [{'params':x.tolist(), 'obj':F.tolist()} for x,F in zip(res.X,res.F)]
        all_out.append({'fold_id':fid,
                        'top_ga': summary[fid]['top_ga'],
                        'pareto_front': front})

        json.dump(all_out, open(args.out_file,'w'), indent=2)
        pd.DataFrame([
            {'fold_id':o['fold_id'],
             'sharpe':-min(f['obj'][0] for f in o['pareto_front']),
             'mdd':min(f['obj'][1] for f in o['pareto_front'])}
            for o in all_out
        ]).to_csv(args.out_file.replace('.json','.csv'), index=False)

        dt = timedelta(seconds=int(time.time()-t0))
        logging.info(f"Fold {fid} complete in {dt}; Pareto size={len(front)}")

        K.clear_session(); gc.collect()

    logging.info("=== Tier 2 complete ===")