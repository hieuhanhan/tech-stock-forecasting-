#!/usr/bin/env python3
"""
summarize_backtest_lstm.py

Create a summary table for LSTM backtests, similar to Table 5.11 (ARIMA):
- Columns: Sharpe mean/median, MDD mean/median, AnnRet mean/median, AnnVol mean/median, n
- Rows   : grouped by `source` (e.g., GA_knee, GA+BO_knee)
Optional filters:
- --intervals 10,20,42       (only include these retraining intervals)
- --sources GA_knee,GA+BO_knee (only include these sources)

Usage:
python summarize_backtest_lstm.py \
  --csv data/backtest_lstm/backtest_lstm_results.csv \
  --out-prefix data/backtest_lstm/summary_lstm \
  --intervals 10,20,42 \
  --sources GA_knee,GA+BO_knee \
  --round 3
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser("Summarize LSTM backtest results")
    ap.add_argument("--csv", required=True, help="Path to backtest_lstm_results.csv")
    ap.add_argument("--out-prefix", default="summary_lstm",
                    help="Output prefix (will create CSV and LaTeX)")
    ap.add_argument("--intervals", default="", help="Comma-separated intervals to include (optional)")
    ap.add_argument("--sources", default="", help="Comma-separated sources to include (optional)")
    ap.add_argument("--round", type=int, default=3, help="Rounding digits for display")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # required columns (allow either learning_rate or lr to exist, but not required for summary)
    need = {"fold_id","retrain_interval","source",
            "test_sharpe","test_mdd","test_ann_return","test_ann_vol"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # optional filters
    if args.intervals.strip():
        keep_int = [int(x) for x in args.intervals.split(",") if x.strip()]
        df = df[df["retrain_interval"].isin(keep_int)]
    if args.sources.strip():
        keep_src = [x.strip() for x in args.sources.split(",") if x.strip()]
        df = df[df["source"].isin(keep_src)]

    if df.empty:
        raise SystemExit("No rows after filtering. Check --intervals / --sources.")

    # group by source only (to match your ARIMA table)
    grp = (df.groupby("source", as_index=False)
             .agg(Sharpe_mean=("test_sharpe","mean"),
                  Sharpe_median=("test_sharpe","median"),
                  MDD_mean=("test_mdd","mean"),
                  MDD_median=("test_mdd","median"),
                  AnnRet_mean=("test_ann_return","mean"),
                  AnnRet_median=("test_ann_return","median"),
                  AnnVol_mean=("test_ann_vol","mean"),
                  AnnVol_median=("test_ann_vol","median"),
                  n=("fold_id","count"))
           )

    # sort sources in a nice order if both exist
    order = ["GA+BO_knee", "GA_knee", "GA+BO", "GA"]
    cat = pd.Categorical(grp["source"], ordered=True,
                         categories=[s for s in order if s in grp["source"].unique()] +
                                    [s for s in grp["source"].unique() if s not in order])
    grp = grp.assign(source=cat).sort_values("source").rename(columns={"source": ""})

    # round for display
    r = args.round
    disp = grp.copy()
    for c in disp.columns:
        if c == "" or c == "n":  # row label or count
            continue
        disp[c] = disp[c].astype(float).round(r)

    out_csv = Path(f"{args.out_prefix}.csv")
    out_tex = Path(f"{args.out_prefix}.tex")

    disp.to_csv(out_csv, index=False)

    # simple LaTeX tabular (can paste into your table template)
    tex = disp.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.{r}f}")
    out_tex.write_text(tex, encoding="utf-8")

    print(f"[OK] Saved summary CSV -> {out_csv}")
    print(f"[OK] Saved LaTeX table -> {out_tex}")
    print("\nPreview:\n", disp.to_string(index=False))

if __name__ == "__main__":
    main()