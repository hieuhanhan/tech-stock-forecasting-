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

    # group by source only 
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

    print(f"[OK] Saved summary CSV -> {out_csv}")
    print("\nPreview:\n", disp.to_string(index=False))

if __name__ == "__main__":
    main()
