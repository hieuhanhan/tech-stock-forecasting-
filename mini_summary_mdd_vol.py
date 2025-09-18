import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- Style (tables & figs) -------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

HEADER_COLOR = (0/255, 61/255, 120/255)   # deep blue
HEADER_TEXT  = "white"
ROW_ALT      = (248/255, 251/255, 255/255)
GRID_COLOR   = (210/255, 220/255, 235/255)

def smart_sig(x, sig=4):
    try:
        v = float(x)
    except Exception:
        return x
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e4:
        return f"{v:.2e}"
    return f"{v:.{sig}g}"

def _auto_col_widths(df: pd.DataFrame, min_w=0.60, max_w=1.20, shrink=0.85):
    widths = []
    for c in df.columns:
        lens = [len(str(c))] + [len(str(v)) for v in df[c].head(200)]
        w = min(max(np.mean(lens) * 0.042, min_w), max_w)
        widths.append(w * shrink)
    return widths

def save_table_png(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "",
    max_rows_per_page: int = 18,
    col_width_overrides: dict[str, float] | None = None,
    header_fontsize: float = 16,
    body_fontsize: float = 15,
    header_height: float = 0.22,
    row_height: float = 0.18,
    max_fig_h: float = 4.8,
    sig_digits: int = 4,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # format numeric columns
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda x: smart_sig(x, sig=sig_digits))

    base_widths = _auto_col_widths(df)
    cols = list(df.columns)
    name2idx = {c: i for i, c in enumerate(cols)}
    widths = base_widths[:]

    if col_width_overrides:
        for name, val in col_width_overrides.items():
            if name in name2idx:
                widths[name2idx[name]] = float(val)

    table_w = sum(widths) + 0.5
    n = len(df)
    pages = int(math.ceil(n / max_rows_per_page)) if n else 1

    for p in range(pages):
        start = p * max_rows_per_page
        end   = min((p + 1) * max_rows_per_page, n)
        df_page = df.iloc[start:end] if n else df

        fig_h = header_height + row_height * (len(df_page) + 1) + 0.25
        fig_h = min(fig_h, max_fig_h)

        fig, ax = plt.subplots(figsize=(table_w, fig_h))
        ax.axis("off")
        if title:
            ttl = title if pages == 1 else f"{title} (page {p+1}/{pages})"
            ax.set_title(ttl, fontsize=header_fontsize, fontweight="bold", loc="left", pad=6)

        tbl = ax.table(
            cellText=df_page.values,
            colLabels=cols,
            cellLoc="center",
            colLoc="center",
            loc="upper left",
            bbox=[0, 0, 1, 1]
        )

        # set exact widths
        rel = np.array(widths, float); rel = rel / rel.sum()
        nrows = len(df_page) + 1; ncols = len(cols)
        for j in range(ncols):
            for r in range(nrows):
                tbl[(r, j)].set_width(rel[j])

        # style
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(HEADER_COLOR)
                cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=header_fontsize)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.8)
                cell.set_height(header_height / fig_h)
            else:
                cell.set_facecolor("white" if (r % 2 == 1) else ROW_ALT)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
                cell.set_fontsize(body_fontsize)
                cell.set_height(row_height / fig_h)

        fig.tight_layout()
        outp = out_path if pages == 1 else out_path.with_name(f"{out_path.stem}_p{p+1}.png")
        plt.savefig(outp, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved table → {outp}")

# ---------------------------- Computation ----------------------------
def summarize_by_interval(df: pd.DataFrame, with_source: bool) -> pd.DataFrame:
    """
    Returns a tidy summary:
      [retrain_interval, (source,) N, mean/median MDD, mean/median Vol, mean/median Turnover]
    """
    # NEW: require turnover too (but allow absence gracefully by filling NaN)
    need = {"retrain_interval", "test_mdd", "test_ann_vol"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    has_turnover = "test_turnover" in df.columns  # NEW

    gcols = ["retrain_interval"] + (["source"] if (with_source and "source" in df.columns) else [])
    agg_dict = dict(
        N=("test_mdd", "count"),
        mean_mdd=("test_mdd", "mean"),
        median_mdd=("test_mdd", "median"),
        mean_vol=("test_ann_vol", "mean"),
        median_vol=("test_ann_vol", "median"),
    )
    if has_turnover:  # NEW
        agg_dict.update(
            mean_tov=("test_turnover", "mean"),
            median_tov=("test_turnover", "median"),
        )

    out = (df.groupby(gcols, as_index=False)
             .agg(**agg_dict)
             .sort_values(gcols))

    if not has_turnover:  # NEW: ensure cols exist for downstream rename/order
        out["mean_tov"] = np.nan
        out["median_tov"] = np.nan

    return out

def barplot_by_interval(
    df: pd.DataFrame,
    out_png: Path,
    title: str,
    ylabel: str,
    value_col: str,
    facet_source: bool,
):
    """
    Bar plot (mean MDD, mean Vol, or mean Turnover) by retrain_interval,
    optionally split by source.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if facet_source and "source" in df.columns:
        pivot = df.pivot(index="retrain_interval", columns="source", values=value_col).sort_index()
        pivot = pivot.dropna(how="all")
        x = np.arange(len(pivot.index))
        width = 0.36 if pivot.shape[1] == 2 else 0.25

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for i, col in enumerate(pivot.columns):
            ax.bar(x + (i - (pivot.shape[1]-1)/2)*width, pivot[col].values,
                   width=width, label=str(col), alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("Retraining interval")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved fig → {out_png}")
    else:
        sub = df.groupby("retrain_interval", as_index=False)[value_col].mean().sort_values("retrain_interval")
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(sub["retrain_interval"].astype(str), sub[value_col].values, alpha=0.9)
        ax.set_xlabel("Retraining interval")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved fig → {out_png}")

# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser("Risk summary (MDD, Volatility, Turnover) for ARIMA/LSTM backtests")
    ap.add_argument("--mode", required=True, choices=["arima","lstm"], help="Label outputs for clarity.")
    ap.add_argument("--csv", required=True, help="Backtest CSV with at least retrain_interval, test_mdd, test_ann_vol (and test_turnover if available).")
    ap.add_argument("--outdir", required=True, help="Output directory for tables & figs.")
    ap.add_argument("--split-by-source", action="store_true",
                    help="If provided and 'source' exists, show GA vs GA+BO columns.")
    ap.add_argument("--plot-turnover", action="store_true",  # NEW
                    help="Also export a mean Turnover by interval bar chart if test_turnover is available.")
    args = ap.parse_args()

    mode = args.mode.lower()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    if "retrain_interval" not in df.columns:
        raise ValueError("CSV must contain column 'retrain_interval'.")

    df = df.copy()
    df["retrain_interval"] = pd.to_numeric(df["retrain_interval"], errors="coerce")

    # Keep only needed columns (+ source if exists, + turnover if exists)  # NEW
    keep = ["retrain_interval", "test_mdd", "test_ann_vol"]
    if "test_turnover" in df.columns:
        keep.append("test_turnover")
    if "source" in df.columns:
        keep.append("source")
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["retrain_interval","test_mdd","test_ann_vol"])

    # ---- Summary table
    sum_tab = summarize_by_interval(df, with_source=args.split_by_source)
    sum_tab = sum_tab.rename(columns={
        "retrain_interval": "Interval",
        "mean_mdd": "Mean MDD",
        "median_mdd": "Median MDD",
        "mean_vol": "Mean Ann. Vol",
        "median_vol": "Median Ann. Vol",
        "mean_tov": "Mean Turnover",           # NEW
        "median_tov": "Median Turnover",       # NEW
    })

    cols = ["Interval"]
    if args.split_by_source and "source" in sum_tab.columns:
        cols += ["source"]
    cols += ["N", "Mean MDD", "Median MDD", "Mean Ann. Vol", "Median Ann. Vol",
             "Mean Turnover", "Median Turnover"]  # NEW
    sum_tab = sum_tab[cols]

    # Save CSV + PNG table
    csv_path = outdir / f"Table_risk_summary_by_interval_{mode}.csv"
    sum_tab.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV → {csv_path}")

    png_path = outdir / f"Table_risk_summary_by_interval_{mode}.png"
    col_w = {
        "Interval": 0.75, "N": 0.7,
        "Mean MDD": 0.95, "Median MDD": 0.95,
        "Mean Ann. Vol": 1.05, "Median Ann. Vol": 1.05,
        "Mean Turnover": 1.00, "Median Turnover": 1.00,  # NEW
    }
    if "source" in sum_tab.columns:
        col_w["source"] = 0.9
    save_table_png(
        sum_tab, png_path,
        title=f"Risk Summary by Interval — {mode.upper()}",
        max_rows_per_page=18,
        col_width_overrides=col_w,
        header_fontsize=16,
        body_fontsize=15,
        header_height=0.22,
        row_height=0.18,
        max_fig_h=4.8,
        sig_digits=4,
    )

    # ---- Figures: mean MDD & mean Vol by interval (optionally split by source)
    if args.split_by_source and "source" in df.columns:
        mean_mdd = df.groupby(["retrain_interval","source"], as_index=False)["test_mdd"].mean()
        mean_vol = df.groupby(["retrain_interval","source"], as_index=False)["test_ann_vol"].mean()
    else:
        mean_mdd = df.groupby(["retrain_interval"], as_index=False)["test_mdd"].mean()
        mean_vol = df.groupby(["retrain_interval"], as_index=False)["test_ann_vol"].mean()

    barplot_by_interval(
        mean_mdd, outdir / f"Fig_mean_MDD_by_interval_{mode}.png",
        title=f"Mean Maximum Drawdown by Interval — {mode.upper()}",
        ylabel="Mean MDD",
        value_col="test_mdd",
        facet_source=(args.split_by_source and "source" in df.columns),
    )
    barplot_by_interval(
        mean_vol, outdir / f"Fig_mean_AnnVol_by_interval_{mode}.png",
        title=f"Mean Annualized Volatility by Interval — {mode.upper()}",
        ylabel="Mean Annualized Volatility",
        value_col="test_ann_vol",
        facet_source=(args.split_by_source and "source" in df.columns),
    )

    # ---- Optional: Turnover figure  # NEW
    if args.plot_turnover and "test_turnover" in df.columns:
        if args.split_by_source and "source" in df.columns:
            mean_tov = df.groupby(["retrain_interval","source"], as_index=False)["test_turnover"].mean()
        else:
            mean_tov = df.groupby(["retrain_interval"], as_index=False)["test_turnover"].mean()

        barplot_by_interval(
            mean_tov, outdir / f"Fig_mean_Turnover_by_interval_{mode}.png",
            title=f"Mean Turnover by Interval — {mode.upper()}",
            ylabel="Mean Turnover",
            value_col="test_turnover",
            facet_source=(args.split_by_source and "source" in df.columns),
        )
    else:
        if args.plot_turnover:
            print("[INFO] 'test_turnover' not found in CSV; skipping turnover plot.")

if __name__ == "__main__":
    main()