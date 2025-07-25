#!/usr/bin/env python3
import os, json, time, argparse, logging, gc, sys
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.arima.model import ARIMA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

# ─────────────────── CONFIG & LOGGING ───────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/tier2_arima.log"), logging.StreamHandler()]
)
np.random.seed(42)

# ─────────────────── HELPERS & METRICS ──────────────────
def sharpe_ratio(r):
    return 0.0 if r.std()==0 else (r.mean()/r.std())*np.sqrt(252)
def max_drawdown(r):
    cum = np.exp(np.cumsum(r))
    dd  = (cum.cummax()-cum)/(cum+1e-9)
    return dd.max()

def create_moga_obj(train, val, cost=0.0005):
    def f(x):
        p, q, thresh = int(x[0]), int(x[1]), x[2]
        try:
            m = ARIMA(train, order=(p,0,q)).fit()
            fcast = m.forecast(steps=len(val))
            sig   = (fcast>thresh).astype(int)
            if sig.sum()==0:
                return (1e3,1e3)
            ret = val*sig - cost*sig
            return (-sharpe_ratio(ret), max_drawdown(ret))
        except Exception as e:
            logging.warning(f"MOGA eval error p={p},q={q},thr={thresh}: {e}")
            return (1e3,1e3)
    return f

class ARIMAMOGA(ElementwiseProblem):
    def __init__(self, obj):
        super().__init__(n_var=3, n_obj=2,
                         xl=np.array([1,1,0.0]),
                         xu=np.array([7,7,0.01]))
        self.obj = obj
    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

# ────────────────────── MAIN ────────────────────────────
if __name__=='__main__':
    p = argparse.ArgumentParser("Tier 2 MOGA for ARIMA")
    p.add_argument("--data-dir",   default="data/processed_folds")
    p.add_argument("--tier1-file", default="data/tuning_results/tier1_arima_prophet.json")
    p.add_argument("--out-file",   default="data/tuning_results/tier2_arima.json")
    p.add_argument("--pop-size",   type=int, default=50)
    p.add_argument("--n-gen",      type=int, default=40)
    ARGS = p.parse_args()

    os.makedirs(os.path.dirname(ARGS.out_file), exist_ok=True)
    tier1 = json.load(open(ARGS.tier1_file))
    summary = {r["fold_id"]:r["arima"] for r in tier1}

    results = []
    for fid in tqdm(summary, desc="Tier2 ARIMA"):
        t0 = time.time()
        params = summary[fid]["best_params"]
        train_df = pd.read_csv(f"{ARGS.data_dir}/folds_summary.json")  # load your fold meta...
        # — here load train/val same as Tier1 —
        train = ...
        val   = ...

        obj = create_moga_obj(train, val)
        prob=ARIMAMOGA(obj)
        res = minimize(prob, NSGA2(pop_size=ARGS.pop_size),
                       ("n_gen", ARGS.n_gen), seed=42, verbose=False)

        front = [{"p":int(x[0]),"q":int(x[1]),"threshold":x[2],
                  "sharpe":-F[0],"mdd":F[1]} for x,F in zip(res.X,res.F)]

        results.append({"fold_id":fid, "pareto_front":front})
        json.dump(results, open(ARGS.out_file,"w"), indent=2)
        pd.DataFrame([{
            "fold_id":fid,
            "max_sharpe":max(-f["sharpe"] for f in front),
            "min_mdd":min(f["mdd"] for f in front)
        } for _ in results]).to_csv(ARGS.out_file.replace(".json",".csv"), index=False)

        logging.info(f"Fold {fid} done in {timedelta(seconds=int(time.time()-t0))}; Pareto size={len(front)}")
        gc.collect()

    logging.info("=== Tier 2 complete ===")