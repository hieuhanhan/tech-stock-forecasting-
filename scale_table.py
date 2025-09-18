#!/usr/bin/env python3
# make_scaling_summary.py
# Create scaling summary tables (min, max, mean, std) before/after global scaling.

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# --------- Defaults (match your pipeline) ----------
RAW_CSV   = "data/cleaned/train_val_for_wf_with_features.csv"
SCALED_CSV= "data/scaled/global/train_val_scaled.csv"
OUT_DIR   = "data/scaled/global"

NON_FEATURE_COLS = [
    "Date", "Ticker", "Close_raw",
    "target", "target_log_returns", "Log_Returns"
]

def infer_feature_groups(df: pd.DataFrame):
    """
    Returns:
      ohlcv_cols: columns that were MinMax-scaled (start with 'Transformed_')
      ti_cols   : other numeric feature columns (z-score standardized)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLS]
    ohlcv_cols = [c for c in feature_cols if c.startswith("Transformed_")]
    ti_cols = [c for c in feature_cols if c not in ohlcv_cols]
    return feature_cols, ohlcv_cols, ti_cols

def summarize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(columns=["feature","min","max","mean","std"])
    s = df[cols].agg(["min","max","mean","std"]).T.reset_index()
    s.columns = ["feature","min","max","mean","std"]
    return s

def add_group_labels(summary_df: pd.DataFrame, ohlcv_cols, ti_cols):
    gmap = {}
    for c in ohlcv_cols: gmap[c] = "OHLCV (MinMax)"
    for c in ti_cols:    gmap[c] = "Indicators (Z-score)"
    summary_df["group"] = summary_df["feature"].map(gmap).fillna("Other")
    return summary_df

def quality_checks(raw_sum: pd.DataFrame, scaled_sum: pd.DataFrame):
    """
    Simple QC flags:
      - MinMax group: min < -1e-9 or max > 1+1e-9
      - Z-score group: |mean|>0.15 or std not in [0.85, 1.15] (tolerant bands)
    """
    qc_rows = []
    for _, r in scaled_sum.iterrows():
        feat = r["feature"]; grp = r["group"]
        vmin, vmax, vmean, vstd = r["min"], r["max"], r["mean"], r["std"]
        flag = ""
        if grp.startswith("OHLCV"):
            if (vmin < -1e-9) or (vmax > 1.0 + 1e-9):
                flag = f"MinMax range off: [{vmin:.4g}, {vmax:.4g}]"
        elif grp.startswith("Indicators"):
            bad_mean = abs(vmean) > 0.15
            bad_std  = (vstd < 0.85) or (vstd > 1.15)
            if bad_mean or bad_std:
                flag = f"Z-score drift: mean={vmean:.3f}, std={vstd:.3f}"
        if flag:
            raw_row = raw_sum[raw_sum["feature"]==feat].iloc[0] if (raw_sum["feature"]==feat).any() else None
            raw_note = ""
            if raw_row is not None:
                raw_note = f"raw[min={raw_row['min']:.3g}, max={raw_row['max']:.3g}, mean={raw_row['mean']:.3g}, std={raw_row['std']:.3g}]"
            qc_rows.append({"feature": feat, "group": grp, "issue": flag, "raw_snapshot": raw_note})
    return pd.DataFrame(qc_rows)

def main():
    ap = argparse.ArgumentParser(description="Create scaling confirmation tables.")
    ap.add_argument("--raw", default=RAW_CSV, help="Path to pre-scaled CSV.")
    ap.add_argument("--scaled", default=SCALED_CSV, help="Path to globally scaled CSV.")
    ap.add_argument("--outdir", default=OUT_DIR, help="Directory to write summary tables.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data
    raw_df = pd.read_csv(args.raw)
    scaled_df = pd.read_csv(args.scaled)

    # Infer groups from RAW column set (same schema as scaled in your pipeline)
    feature_cols, ohlcv_cols, ti_cols = infer_feature_groups(raw_df)

    # Keep only intersecting columns present in both files (safety)
    feature_cols = [c for c in feature_cols if c in scaled_df.columns]

    # --- Summaries
    raw_sum    = summarize(raw_df, feature_cols)
    scaled_sum = summarize(scaled_df, feature_cols)

    # Attach group labels
    raw_sum    = add_group_labels(raw_sum, ohlcv_cols, ti_cols)
    scaled_sum = add_group_labels(scaled_sum, ohlcv_cols, ti_cols)

    # Side-by-side diff
    merged = raw_sum.merge(
        scaled_sum,
        on=["feature","group"],
        suffixes=("_raw","_scaled"),
        how="inner"
    )

    # Tidy ordering
    order_cols = [
        "feature","group",
        "min_raw","max_raw","mean_raw","std_raw",
        "min_scaled","max_scaled","mean_scaled","std_scaled"
    ]
    merged = merged[order_cols].sort_values(["group","feature"]).reset_index(drop=True)

    # --- QC report
    qc_df = quality_checks(raw_sum, scaled_sum)

    # --- Save
    raw_path     = outdir / "scaling_summary_raw.csv"
    scaled_path  = outdir / "scaling_summary_scaled.csv"
    compare_path = outdir / "scaling_summary_compare.csv"
    qc_path      = outdir / "scaling_qc_report.csv"

    raw_sum.to_csv(raw_path, index=False)
    scaled_sum.to_csv(scaled_path, index=False)
    merged.to_csv(compare_path, index=False)
    qc_df.to_csv(qc_path, index=False)

    # Console nudge
    print(f"[OK] Raw summary     -> {raw_path}")
    print(f"[OK] Scaled summary  -> {scaled_path}")
    print(f"[OK] Compare (side-by-side) -> {compare_path}")
    print(f"[OK] QC report       -> {qc_path}")

    # Quick counts for your write-up
    n_ohlcv = (scaled_sum["group"]=="OHLCV (MinMax)").sum()
    n_ti    = (scaled_sum["group"]=="Indicators (Z-score)").sum()
    print(f"[INFO] Features: total={len(feature_cols)} | OHLCV={n_ohlcv} | Indicators={n_ti}")
    if not qc_df.empty:
        print(f"[WARN] QC flags: {len(qc_df)} features. See {qc_path}")

if __name__ == "__main__":
    main()