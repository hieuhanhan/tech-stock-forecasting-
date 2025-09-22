import os, argparse, json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

EPS = 1e-8
DEFAULT_MIN_BLOCK_VOL = 0.0015

# ---------- backtest engine ----------
def sharpe_ratio(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    if r.size < 2: return 0.0
    sd = float(np.std(r, ddof=1))
    return 0.0 if sd==0 else float(np.mean(r))/sd*np.sqrt(252.0)

def backtest_continuous_trace(
    test_log: np.ndarray,
    p: int, q: int, thr: float,
    retrain_interval: int,
    cost_per_turnover: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    scale_factor: float = 1000.0,
    arch_rescale_flag: bool = False,
    pos_clip: float = 1.0,
    warmup_len: int = 252,
) -> Dict:
    test_log = np.asarray(test_log, float)
    n = len(test_log)
    if n <= warmup_len + 1:
        raise ValueError("test too short for warmup_len")
    history = test_log[:warmup_len].copy()
    start_idx = warmup_len

    pos, ret_s, ret_l = [], [], []
    prev_pos_last = 0.0
    total_turn = 0.0

    for st in range(start_idx, n, retrain_interval):
        ed = min(st + retrain_interval, n)
        block_log = test_log[st:ed]
        h = ed - st
        if h <= 0: continue

        if float(np.std(block_log)) < float(min_block_vol):
            history = np.concatenate([history, block_log])
            continue

        try:
            arima_fit = ARIMA(history, order=(int(p),0,int(q))).fit()
            resid = np.asarray(arima_fit.resid, float).ravel()
            garch = arch_model(resid*scale_factor, mean="Zero", vol="Garch",
                               p=1, q=1, dist="normal", rescale=arch_rescale_flag)
            g_res = garch.fit(disp="off")
            f_var = np.asarray(g_res.forecast(horizon=h).variance.iloc[-1]).ravel()
            f_mu  = np.asarray(arima_fit.forecast(steps=h)).ravel()
            f_sig = np.sqrt(np.maximum(f_var/(scale_factor**2+EPS), EPS))
        except Exception:
            history = np.concatenate([history, block_log])
            continue

        z = f_mu/(f_sig+EPS)
        pos_now = np.clip(z/(thr+EPS), -pos_clip, pos_clip)

        signals_full = np.concatenate([[prev_pos_last], pos_now])
        turnover = np.abs(np.diff(signals_full))
        cost_vec = cost_per_turnover * turnover
        total_turn += float(turnover.sum())

        block_simple = np.exp(block_log) - 1.0
        ret_simple = np.clip(block_simple*pos_now - cost_vec, -0.9999, None)
        ret_log    = np.log1p(ret_simple)

        pos.extend(pos_now.tolist())
        ret_s.extend(ret_simple.tolist())
        ret_l.extend(ret_log.tolist())

        history = np.concatenate([history, block_log])
        prev_pos_last = float(pos_now[-1])

    pos = np.array(pos, float)
    ret_s = np.array(ret_s, float)
    ret_l = np.array(ret_l, float)
    return dict(positions=pos, ret_simple=ret_s, ret_log=ret_l,
                sharpe=sharpe_ratio(ret_s) if ret_s.size else 0.0)

# ---------- plotting helpers ----------
def plot_equity_compare(outdir: Path, fold_id: int, interval: int,
                        eq_ga: np.ndarray, eq_bo: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,4), dpi=140)
    ax.plot(eq_ga, label="GA knee")
    ax.plot(eq_bo, label="GA+BO knee", linestyle="--")
    ax.set_title(f"Equity overlay — fold {fold_id}, int {interval}")
    ax.set_ylabel("Equity (norm.)"); ax.set_xlabel("Time")
    ax.legend(); fig.tight_layout()
    fig.savefig(outdir / f"equity_overlay_f{fold_id}_i{interval}.png"); plt.close(fig)

def heatmap(ax, Z, xticks, yticks, title):
    im = ax.imshow(Z, aspect="auto")
    ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks)
    ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks)
    ax.set_xlabel("Cost multiplier"); ax.set_ylabel("Threshold multiplier")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

def run_sensitivity(test_log: np.ndarray, p:int,q:int,thr:float, interval:int,
                    cost:float, thr_mults:List[float], cost_mults:List[float],
                    **kwargs) -> Tuple[np.ndarray,np.ndarray]:
    S = np.zeros((len(thr_mults), len(cost_mults)))
    M = np.zeros_like(S)
    for i, tm in enumerate(thr_mults):
        for j, cm in enumerate(cost_mults):
            tr = backtest_continuous_trace(
                test_log, p,q, thr*tm, interval, cost*cm, **kwargs
            )

            eq = np.exp(np.cumsum(tr["ret_log"]))
            peak = np.maximum.accumulate(eq); dd = (peak-eq)/(peak+EPS)
            S[i,j] = sharpe_ratio(tr["ret_simple"])
            M[i,j] = float(np.nanmax(dd))
    return S, M

def plot_sens_compare(outdir: Path, fold_id:int, interval:int,
                      sens_ga:Dict, sens_bo:Dict,
                      thr_mults:List[float], cost_mults:List[float]):
    outdir.mkdir(parents=True, exist_ok=True)
    # Sharpe
    fig, axs = plt.subplots(1,2, figsize=(10,4), dpi=140, sharey=True)
    heatmap(axs[0], sens_ga["S"], cost_mults, thr_mults, "GA — Sharpe")
    heatmap(axs[1], sens_bo["S"], cost_mults, thr_mults, "GA+BO — Sharpe")
    fig.suptitle(f"Sensitivity (Sharpe) — fold {fold_id}, int {interval}")
    fig.tight_layout(); fig.savefig(outdir / f"sens_sharpe_f{fold_id}_i{interval}.png"); plt.close(fig)
    # MDD
    fig, axs = plt.subplots(1,2, figsize=(10,4), dpi=140, sharey=True)
    heatmap(axs[0], sens_ga["M"], cost_mults, thr_mults, "GA — MDD")
    heatmap(axs[1], sens_bo["M"], cost_mults, thr_mults, "GA+BO — MDD")
    fig.suptitle(f"Sensitivity (MDD) — fold {fold_id}, int {interval}")
    fig.tight_layout(); fig.savefig(outdir / f"sens_mdd_f{fold_id}_i{interval}.png"); plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Highlight 3 GA vs GA+BO differing cases with equity & sensitivity plots")
    ap.add_argument("--backtest-csv", required=True,
                    help="data/backtest_tier2/backtest_results_ga_vs_gabo.csv")
    ap.add_argument("--test-csv", required=True,
                    help="Scaled test CSV with column Log_Returns")
    ap.add_argument("--intervals", default="10,20,42")
    ap.add_argument("--cost-per-turnover", type=float, default=0.0005+0.0002)
    ap.add_argument("--warmup-len", type=int, default=252)
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--pos-clip", type=float, default=1.0)
    ap.add_argument("--min-block-vol", type=float, default=DEFAULT_MIN_BLOCK_VOL)
    ap.add_argument("--outdir", default="compare_ga_vs_gabo")
    ap.add_argument("--thr-mults", default="0.8,1.0,1.2")
    ap.add_argument("--cost-mults", default="0.5,1.0,1.5")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    bt = pd.read_csv(args.backtest_csv)
    test = pd.read_csv(args.test_csv)
    x = test["Log_Returns"].fillna(0).to_numpy(float)

    ints = [int(s) for s in args.intervals.split(",") if s.strip()]
    pairs = []
    for fid, interval in bt.groupby(["fold_id", "retrain_interval"]).groups.keys():
        if interval not in ints: continue
        sub = bt[(bt.fold_id==fid) & (bt.retrain_interval==interval)]
        if {"GA_knee","GA+BO_knee"}.issubset(set(sub["source"])):
            row_ga  = sub[sub.source=="GA_knee"].iloc[0]
            row_bo  = sub[sub.source=="GA+BO_knee"].iloc[0]
            delta_s = abs(float(row_bo["test_sharpe"]) - float(row_ga["test_sharpe"]))
            delta_m = abs(float(row_bo["test_mdd"])    - float(row_ga["test_mdd"]))

            diff = (row_ga["p"]!=row_bo["p"]) or (row_ga["q"]!=row_bo["q"]) or (float(row_ga["threshold"])!=float(row_bo["threshold"])) \
                   or (delta_s>1e-6) or (delta_m>1e-6)
            if diff:
                pairs.append((fid, interval, delta_s, delta_m))
    if not pairs:
        print("[INFO] No differing pairs found.")
        return

    pairs = sorted(pairs, key=lambda t: (t[2], t[3]), reverse=True)[:3] # MDD: pairs = sorted(pairs, key=lambda t: (t[3], t[2]), reverse=True)[:3]
    print("[SELECTED 3] ", pairs)

    thr_mults = [float(v) for v in args.thr_mults.split(",") if v.strip()]
    cost_mults = [float(v) for v in args.cost_mults.split(",") if v.strip()]

    for fid, interval, _, _ in pairs:
        sub = bt[(bt.fold_id==fid) & (bt.retrain_interval==interval)]
        ga  = sub[sub.source=="GA_knee"].iloc[0]
        bo  = sub[sub.source=="GA+BO_knee"].iloc[0]

        # equity overlay
        tr_ga = backtest_continuous_trace(
            x, int(ga["p"]), int(ga["q"]), float(ga["threshold"]),
            interval, args.cost_per_turnover,
            min_block_vol=args.min_block_vol, scale_factor=args.scale_factor,
            arch_rescale_flag=False, pos_clip=args.pos_clip, warmup_len=args.warmup_len
        )
        tr_bo = backtest_continuous_trace(
            x, int(bo["p"]), int(bo["q"]), float(bo["threshold"]),
            interval, args.cost_per_turnover,
            min_block_vol=args.min_block_vol, scale_factor=args.scale_factor,
            arch_rescale_flag=False, pos_clip=args.pos_clip, warmup_len=args.warmup_len
        )
        eq_ga = np.exp(np.cumsum(tr_ga["ret_log"]))
        eq_bo = np.exp(np.cumsum(tr_bo["ret_log"]))
        plot_equity_compare(outdir/ f"f{fid}_i{interval}", fid, interval, eq_ga, eq_bo)

        # sensitivity compare
        S_ga, M_ga = run_sensitivity(
            x, int(ga["p"]), int(ga["q"]), float(ga["threshold"]), interval,
            args.cost_per_turnover, thr_mults, cost_mults,
            min_block_vol=args.min_block_vol, scale_factor=args.scale_factor,
            arch_rescale_flag=False, pos_clip=args.pos_clip, warmup_len=args.warmup_len
        )
        S_bo, M_bo = run_sensitivity(
            x, int(bo["p"]), int(bo["q"]), float(bo["threshold"]), interval,
            args.cost_per_turnover, thr_mults, cost_mults,
            min_block_vol=args.min_block_vol, scale_factor=args.scale_factor,
            arch_rescale_flag=False, pos_clip=args.pos_clip, warmup_len=args.warmup_len
        )
        plot_sens_compare(outdir/ f"f{fid}_i{interval}", fid, interval,
                          {"S":S_ga,"M":M_ga}, {"S":S_bo,"M":M_bo},
                          thr_mults, cost_mults)
    meta = [{"fold_id":int(fid), "interval":int(interval), "delta_sharpe":float(ds), "delta_mdd":float(dm)}
            for fid, interval, ds, dm in pairs]
    with open(outdir/"selected_pairs.json","w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved comparison plots & selected_pairs.json -> {outdir}")
if __name__ == "__main__":
    main()
