#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from arima_utils import (
    suppress_warnings,
    create_continuous_arima_objective,
    is_nondominated
)

def load_json_any(p: Path):
    with p.open("r") as f: return json.load(f)

def main():
    ap = argparse.ArgumentParser("Random Search baseline for Tier-2 ARIMA (tiny budget)")
    ap.add_argument("--folds-json", required=True,
                    help="JSON with final_train_path/final_val_path or train_path_arima/val_path_arima.")
    ap.add_argument("--tier1-json", required=True,
                    help="Tier-1 ARIMA results JSON (to get (p,q) seeds per fold).")
    ap.add_argument("--retrain-intervals", default="10,20,42")
    ap.add_argument("--budget", type=int, default=80, help="Random samples per (fold, interval).")
    ap.add_argument("--thr-min", type=float, default=0.01)
    ap.add_argument("--thr-max", type=float, default=0.6)
    ap.add_argument("--pos-clip", type=float, default=1.0)
    ap.add_argument("--min-block-vol", type=float, default=0.0015)
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--arch-rescale-flag", action="store_true")
    ap.add_argument("--garch-dist", default="auto", choices=["auto","normal","t"])
    ap.add_argument("--garch-ic", default="aic", choices=["aic","bic","hqic"])
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--out-front-csv", default="data/tuning_results/csv/tier2_arima_rs_front.csv")
    args = ap.parse_args()

    suppress_warnings()
    rng = np.random.default_rng(42)

    # --- load folds meta
    meta = load_json_any(Path(args.folds_json))
    if isinstance(meta, dict) and "arima" in meta:
        folds = meta["arima"]
    elif isinstance(meta, list):
        folds = meta
    else:
        raise ValueError("Unexpected folds JSON format.")

    # --- resolve base directory for relative CSV paths
    base_dir = Path(args.folds_json).resolve().parents[2]

    # --- load Tier-1 champions (for p,q sampling centers)
    t1 = load_json_any(Path(args.tier1_json))
    if isinstance(t1, dict): t1 = t1.get("results", t1)
    champions = { int(r["fold_id"]): r["best_params"]
                  for r in t1 if "fold_id" in r and "best_params" in r }

    intervals = [int(x) for x in args.retrain_intervals.split(",") if x.strip()]

    out_rows = []

    # --- iterate folds
    fids = [int(r["global_fold_id"]) for r in folds if "global_fold_id" in r]
    if args.max_folds: fids = fids[:args.max_folds]

    for fid in fids:
        rec = next((r for r in folds if int(r["global_fold_id"]) == fid), None)
        if not rec: continue

        train_rel = rec.get("final_train_path") or rec.get("train_path_arima")
        val_rel   = rec.get("final_val_path")   or rec.get("val_path_arima")
        if not (train_rel and val_rel): continue

        train_csv = (base_dir / train_rel).resolve()
        val_csv   = (base_dir / val_rel).resolve()

        tr = pd.read_csv(train_csv); va = pd.read_csv(val_csv)
        if "Log_Returns" not in tr.columns or "Log_Returns" not in va.columns: continue

        train = tr["Log_Returns"].dropna().to_numpy(float)
        val   = va["Log_Returns"].fillna(0).to_numpy(float)
        if train.size == 0 or val.size == 0: continue

        # center around Tier-1 (p,q) if available
        bp = champions.get(fid, {})
        p0 = int(bp.get("p", 3)); q0 = int(bp.get("q", 3))

        for interval in intervals:
            obj = create_continuous_arima_objective(
                train=train, val=val,
                retrain_interval=interval,
                cost_per_turnover=0.0005+0.0002,
                min_block_vol=args.min_block_vol,
                scale_factor=args.scale_factor,
                arch_rescale_flag=args.arch_rescale_flag,
                debug=False,
                thr_min=args.thr_min, thr_max=args.thr_max,
                pos_clip=args.pos_clip,
                garch_dist=args.garch_dist,
                garch_ic=args.garch_ic
            )

            # random samples around (p0,q0) + uniform thresholds
            P = rng.integers(low=1, high=8, size=args.budget)
            Q = rng.integers(low=1, high=8, size=args.budget)
            # nudge half of them near the champion
            half = args.budget // 2
            P[:half] = np.clip(p0 + rng.integers(-1,2,size=half), 1, 7)
            Q[:half] = np.clip(q0 + rng.integers(-1,2,size=half), 1, 7)
            TH = rng.uniform(args.thr_min, args.thr_max, size=args.budget)

            X = np.c_[P, Q, TH]
            F = np.array([obj(x) for x in X], float)   # minimize [-Sharpe, MDD]
            nd_mask = is_nondominated(F)
            X_nd, F_nd = X[nd_mask], F[nd_mask]

            # write front rows (same schema you already use)
            for x, f in zip(X_nd, F_nd):
                p, q, thr = int(x[0]), int(x[1]), float(x[2])
                out_rows.append({
                    "fold_id": int(fid),
                    "retrain_interval": int(interval),
                    "front_type": "RS",
                    "p": p, "q": q, "threshold": thr,
                    "sharpe": float(-f[0]),
                    "mdd": float(f[1]),
                })

    out = pd.DataFrame(out_rows)
    Path(args.out_front_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_front_csv, index=False)
    print(f"[OK] RS front saved → {args.out_front_csv}  (rows={len(out)})")

if __name__ == "__main__":
    main()