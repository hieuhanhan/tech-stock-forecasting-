from pathlib import Path
import math, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- Style ----------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

HEADER_COLOR = (0/255, 61/255, 120/255)   # deep blue
HEADER_TEXT  = "white"
ROW_ALT      = (248/255, 251/255, 255/255)  # very light blue
GRID_COLOR   = (210/255, 220/255, 235/255)

# ---------- Helpers ----------
def smart_sig4(x):
    """6 significant digits; switch to scientific if needed."""
    try:
        v = float(x)
    except Exception:
        return x
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e4:
        return f"{v:.2e}"
    # 6 significant digits, strip trailing zeros
    s = f"{v:.4g}"
    return s

def _auto_col_widths(df: pd.DataFrame, min_w=0.75, max_w=1.7, shrink=0.92):
    widths = []
    for c in df.columns:
        lens = [len(str(c))] + [len(str(v)) for v in df[c].head(250)]
        w = min(max(np.mean(lens) * 0.055, min_w), max_w)
        widths.append(w * shrink)
    return widths

def save_table_png(df: pd.DataFrame,
                   out_path: Path,
                   title: str = "",
                   max_rows_per_page: int = 28,
                   numeric_cols: list[str] | None = None):
    """Render DataFrame to one or more PNGs with the thesis table style."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Format numeric columns (or infer)
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        df[c] = df[c].map(smart_sig4)

    n = len(df)
    pages = int(math.ceil(n / max_rows_per_page)) if n else 1
    col_widths = _auto_col_widths(df)
    table_w = sum(col_widths) + 0.6
    header_h, row_h = 0.30, 0.28

    for p in range(pages):
        start, end = p * max_rows_per_page, min((p + 1) * max_rows_per_page, n)
        df_page = df.iloc[start:end] if n else df

        fig_h = header_h + row_h * (len(df_page) + 1) + 0.35
        fig, ax = plt.subplots(figsize=(table_w, fig_h))
        ax.axis("off")
        if title:
            ttl = title if pages == 1 else f"{title} (page {p+1}/{pages})"
            ax.set_title(ttl, fontsize=10, fontweight="bold", loc="left", pad=6)

        tbl = ax.table(
            cellText=df_page.values,
            colLabels=list(df_page.columns),
            cellLoc="center",
            colLoc="center",
            loc="upper left",
            bbox=[0, 0, 1, 1]
        )

        # column widths (relative best-effort)
        for j, _ in enumerate(col_widths):
            tbl.auto_set_column_width(j)

        # style cells
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(HEADER_COLOR)
                cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=9)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.8)
            else:
                cell.set_facecolor("white" if (r % 2 == 1) else ROW_ALT)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
                cell.set_fontsize(8)

        fig.tight_layout()
        stem = out_path.stem if pages == 1 else f"{out_path.stem}_p{p+1}"
        png_path = out_path.with_name(f"{stem}.png")
        plt.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved table → {png_path}")

def csv_to_png(csv_path: Path, out_png: Path | None = None, title: str = ""):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if out_png is None:
        out_png = csv_path.with_suffix(".png")
    save_table_png(df, out_png, title=title)

def batch_render(csv_paths: list[str | Path], out_dir: str | Path | None = None, title_map: dict | None = None):
    out_dir = Path(out_dir) if out_dir else None
    for c in csv_paths:
        c = Path(c)
        if not c.exists():
            print(f"[WARN] Missing CSV: {c}")
            continue
        df = pd.read_csv(c)
        title = (title_map or {}).get(c.name, "")
        out_png = (out_dir / (c.stem + ".png")) if out_dir else c.with_suffix(".png")
        save_table_png(df, out_png, title=title)

csv_list = [
    # HV diagnostics
    "results/chap5/hv_filtered/Table_5_3_hv_improved_only.csv",
    "results/chap5/hv_filtered/Table_5_3_hv_improvement_summary_by_interval.csv",

    # Backtest tables
    "results/chap5/backtest_summaries/Table_backtest_overall_by_source.csv",
    "results/chap5/backtest_summaries/Table_backtest_by_interval_and_source.csv",
    "results/chap5/backtest_summaries/Table_backtest_by_ticker_and_source.csv",
    "results/chap5/backtest_summaries/Table_backtest_summary_by_interval.csv",
]

# 2) Optional pretty titles per file name
"""pretty_titles = {
    "Table_5_3_hv_improved_only.csv": "Folds with HV improvement (GA → GA+BO)",
    "Table_5_3_hv_improvement_summary_by_interval.csv": "HV improvement summary by retraining interval",
    "Table_backtest_overall_by_source.csv": "Summary statistics — Sharpe, MDD, Annualized Return (overall)",
    "Table_backtest_summary_by_interval.csv": "Backtest summary by retraining interval",
    "Table_backtest_by_interval_and_source.csv": "Backtest by interval × source",
    "Table_backtest_by_ticker_and_source.csv": "Backtest by ticker × source",
}"""

batch_render(csv_list, out_dir="results/chap5/tables_png") #title_map=pretty_titles)
