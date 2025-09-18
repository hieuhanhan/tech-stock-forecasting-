#!/usr/bin/env python3
"""
Make a correlation heatmap for engineered features.

- Automatically EXCLUDES: Date-like columns, Ticker, raw OHLCV (Open/High/Low/Close/Adj Close/Volume),
  common target columns (target, target_log_returns, Log_Returns, etc.), and any obviously non-numeric cols.
- Saves both PNG (hi-DPI) and SVG (vector, best for Word/LaTeX).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: seaborn gives nicer heatmaps; fine for a standalone script
import seaborn as sns

EXCLUDE_EXACT = {
    # identifiers
    "Date", "Datetime", "date", "datetime", "time", "Time", "Index",
    "Ticker", "Symbol",

    # raw OHLCV (untransformed)
    "Open", "High", "Low", "Close","Close_raw" "Adj Close", "Adj_Close", "Volume",

    # typical target columns
    "target", "Target", "y", "Y",
    "target_log_returns", "Target_Log_Returns",
    "Log_Returns", "log_returns", "Returns", "returns",
}

EXCLUDE_CONTAINS = {
    # common labeling/metadata
    "split", "dividend", "isin", "cusip",
    # if you store forward-filled labels/targets with prefixes
    "label", "future_", "next_",
}

def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Choose numeric feature columns by excluding known non-feature fields."""
    # keep only numeric dtypes first
    num_df = df.select_dtypes(include=[np.number]).copy()

    # drop columns with all NaNs or near-constant variance (optional)
    nunique = num_df.nunique(dropna=True)
    num_df = num_df.loc[:, nunique > 1]

    cols = []
    for c in num_df.columns:
        name_low = c.lower()

# Exact exclusions
        if c in EXCLUDE_EXACT:
            continue

        # Exclude raw OHLCV (but keep transformed_*)
        if ("close" in name_low or "open" in name_low or "high" in name_low or 
            "low" in name_low or "volume" in name_low):
            if not name_low.startswith("transformed_"):
                continue

        # Exclude if contains other forbidden substrings
        if any(key in name_low for key in EXCLUDE_CONTAINS):
            continue
        # if you want to exclude RAW "Close" but KEEP transformed (e.g., "Transformed_Close"),
        # the exact name check above already does that.
        cols.append(c)
    return cols

def main():
    ap = argparse.ArgumentParser(description="Plot correlation heatmap for engineered features")
    ap.add_argument("--csv", required=True, help="Path to engineered_features.csv")
    ap.add_argument("--outdir", default="figures", help="Directory to save outputs")
    ap.add_argument("--title", default="Feature Correlation Heatmap", help="Plot title")
    ap.add_argument("--font", default="Arial", help="Matplotlib font family (e.g., Arial or Times New Roman)")
    ap.add_argument("--dpi", type=int, default=400, help="PNG DPI (higher = crisper in Word)")
    ap.add_argument("--max_features", type=int, default=200, help="Safety cap for number of features")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Matplotlib aesthetics
    plt.rcParams["font.family"] = args.font
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = args.dpi

    # Load
    df = pd.read_csv(args.csv)

    # Choose features
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise RuntimeError("No feature columns found after exclusions.")
    if len(feature_cols) > args.max_features:
        # keep the first max_features to avoid unreadable heatmaps
        feature_cols = feature_cols[:args.max_features]

    # Compute correlation
    corr = df[feature_cols].corr(method="pearson")

    # Figure size scales with number of features (rough heuristic)
    n = len(feature_cols)
    fig_w = min(max(8, 0.35 * n), 26)   # cap width to avoid gigantic figures
    fig_h = min(max(6, 0.35 * n), 26)

    # Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # mask upper triangle for readability
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # A publication-friendly diverging palette
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5,
        linecolor="#E6E6E6",
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Pearson correlation"},
        ax=ax,
    )

    ax.set_title(args.title, pad=12)
    # Tighten labels: rotate for space efficiency
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    # Save outputs
    png_path = outdir / "feature_corr_heatmap.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    # Also save the correlation matrix (optional)
    corr.to_csv(outdir / "feature_corr_matrix.csv", index=True)

    print(f"Saved: {png_path}")
    print(f"Matrix CSV: {outdir/'feature_corr_matrix.csv'}")
    print(f"Included features ({len(feature_cols)}): {feature_cols}")

if __name__ == "__main__":
    main()