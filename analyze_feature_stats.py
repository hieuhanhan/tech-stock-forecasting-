#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

BASE_DIR = "data/processed_folds"
FIG_BASE = "data/figures"

NON_FEATURE_COLS = {
    "Date", "Ticker", "target_log_returns", "target",
    "Log_Returns", "Close_raw", "global_fold_id"
}
OHLCV_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def collect_meta_paths(model_type: str):
    pattern = os.path.join(BASE_DIR, f"{model_type}_meta", "val_meta_fold_*.csv")
    paths = sorted(glob.glob(pattern), key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
    return paths

def load_meta_concat(model_type: str, max_folds: int | None = None) -> pd.DataFrame:
    paths = collect_meta_paths(model_type)
    if max_folds: paths = paths[:max_folds]
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["global_fold_id"] = int(os.path.basename(p).split("_")[-1].split(".")[0])
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skip {p}: {e}")
    if not frames: return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    if "Date" in df_all.columns: df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    if "Ticker" in df_all.columns: df_all["Ticker"] = df_all["Ticker"].astype(str)
    return df_all

def get_feature_cols_simple(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c not in OHLCV_COLS
        and not c.startswith("Transformed_")
    ]

def plot_target_distribution(df, figdir, model_type):
    if "target" not in df.columns: return
    plt.figure(figsize=(7,4))
    plt.hist(df["target"].dropna(), bins=3, edgecolor="black")
    plt.title(f"{model_type.upper()} – Target Distribution")
    plt.xlabel("target"); plt.ylabel("Count"); plt.tight_layout()
    out = os.path.join(figdir, f"{model_type}_target_distribution.png")
    plt.savefig(out, dpi=300); plt.close(); print("[PLOT]", out)

def plot_log_returns_distribution(df, figdir, model_type):
    if "Log_Returns" not in df.columns: return
    plt.figure(figsize=(7,4))
    plt.hist(df["Log_Returns"].dropna(), bins=60, edgecolor="black")
    plt.title(f"{model_type.upper()} – Log_Returns Distribution")
    plt.xlabel("Log_Returns"); plt.ylabel("Count"); plt.tight_layout()
    out = os.path.join(figdir, f"{model_type}_log_returns_distribution.png")
    plt.savefig(out, dpi=300); plt.close(); print("[PLOT]", out)

def plot_corr_heatmap(df, feature_cols, figdir, model_type, sample_rows: int):
    if not feature_cols: 
        print("[INFO] No feature columns for correlation."); return
    sample = df[feature_cols].dropna()
    if len(sample) > sample_rows: sample = sample.sample(sample_rows, random_state=42)
    if sample.empty: 
        print("[INFO] Empty sample for correlation; skip."); return
    corr = sample.corr(numeric_only=True)
    plt.figure(figsize=(14,11))
    sns.heatmap(corr, cmap="coolwarm", cbar=True)
    plt.title(f"{model_type.upper()} – Feature Correlation Heatmap")
    plt.tight_layout()
    out = os.path.join(figdir, f"{model_type}_feature_corr_heatmap.png")
    plt.savefig(out, dpi=300); plt.close(); print("[PLOT]", out)

def plot_pca_scatter(df, feature_cols, figdir, model_type, sample_rows: int):
    if len(feature_cols) < 2: 
        print("[INFO] <2 features; skip PCA scatter."); return
    sample = df[feature_cols].dropna()
    if len(sample) > sample_rows: sample = sample.sample(sample_rows, random_state=42)
    if sample.empty: 
        print("[INFO] Empty sample for PCA; skip."); return
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(sample.values)
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(7,6))
    plt.scatter(X2[:,0], X2[:,1], s=8, alpha=0.6)
    plt.title(f"{model_type.upper()} – PCA Scatter (EVR {evr[0]:.2f}/{evr[1]:.2f})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    out = os.path.join(figdir, f"{model_type}_pca_scatter.png")
    plt.savefig(out, dpi=300); plt.close(); print("[PLOT]", out)

def plot_acf_pacf_per_ticker(df, figdir, model_type, max_points=2000, lags=40):
    if not HAS_STATSMODELS or "Log_Returns" not in df.columns:
        return
    outdir = os.path.join(figdir, "acf_pacf"); ensure_dir(outdir)
    for tkr, g in df.dropna(subset=["Log_Returns"]).groupby("Ticker"):
        s = g.sort_values("Date")["Log_Returns"].values
        if len(s) < 10: continue
        s = s[-max_points:]
        fig, axes = plt.subplots(1, 2, figsize=(11,4))
        plot_acf(s, ax=axes[0], lags=lags, title=f"{tkr} ACF")
        plot_pacf(s, ax=axes[1], lags=lags, title=f"{tkr} PACF", method="ywm")
        fig.suptitle(f"{model_type.upper()} – {tkr} Log_Returns ACF/PACF")
        fig.tight_layout()
        out = os.path.join(outdir, f"{model_type}_{tkr}_acf_pacf.png")
        plt.savefig(out, dpi=300); plt.close(fig); print("[PLOT]", out)

def main():
    ap = argparse.ArgumentParser(description="Analyze feature stats BEFORE fold treatment")
    ap.add_argument("--model_type", type=str, required=True, choices=["lstm","arima"])
    ap.add_argument("--max_folds", type=int, default=None, help="Limit number of meta files to load")
    ap.add_argument("--sample_rows", type=int, default=20000, help="Rows to sample for corr/PCA")
    args = ap.parse_args()

    figdir = os.path.join(FIG_BASE, args.model_type); ensure_dir(figdir)
    df = load_meta_concat(args.model_type, max_folds=args.max_folds)
    if df.empty:
        print("[ERROR] No meta data found. Make sure meta CSVs exist."); return

    feature_cols = get_feature_cols_simple(df)
    print(f"[INFO] Using {len(feature_cols)} feature cols (exclude OHLCV & Transformed_*).")

    plot_target_distribution(df, figdir, args.model_type)
    plot_log_returns_distribution(df, figdir, args.model_type)
    plot_corr_heatmap(df, feature_cols, figdir, args.model_type, sample_rows=args.sample_rows)
    plot_pca_scatter(df, feature_cols, figdir, args.model_type, sample_rows=args.sample_rows)
    plot_acf_pacf_per_ticker(df, figdir, args.model_type)

if __name__ == "__main__":
    main()