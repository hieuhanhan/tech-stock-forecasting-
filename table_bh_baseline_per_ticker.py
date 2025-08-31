from pathlib import Path
import math, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# --- Style ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

HEADER_COLOR = (0/255, 61/255, 120/255)
HEADER_TEXT  = "white"
ROW_ALT      = (248/255, 251/255, 255/255)
GRID_COLOR   = (210/255, 220/255, 235/255)

def smart_sig4(x):
    try:
        v = float(x)
    except Exception:
        return x
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e4:
        return f"{v:.2e}"
    return f"{v:.4g}"

def _auto_col_widths(df: pd.DataFrame, min_w=0.60, max_w=1.20, shrink=0.85):
    widths = []
    for c in df.columns:
        lens = [len(str(c))] + [len(str(v)) for v in df[c].head(200)]
        w = min(max(np.mean(lens) * 0.042, min_w), max_w)  # narrower coeff
        widths.append(w * shrink)
    return widths

def save_table_png(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "",
    max_rows_per_page: int = 10,          # fewer rows → bigger cells
    col_width_overrides: dict[str, float] | None = None,
    header_fontsize: float = 18,
    body_fontsize: float = 16,
    header_height: float = 0.22,
    row_height: float = 0.18,
    max_fig_h: float = 4.4,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # format numerics
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(smart_sig4)

    # compute widths
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
            ax.set_title(ttl, fontsize=18, fontweight="bold", loc="left", pad=6)

        tbl = ax.table(
            cellText=df_page.values,
            colLabels=cols,
            cellLoc="center",
            colLoc="center",
            loc="upper left",
            bbox=[0, 0, 1, 1]
        )

        # set exact column widths (normalized)
        rel = np.array(widths, dtype=float); rel = rel / rel.sum()
        nrows = len(df_page) + 1; ncols = len(cols)
        for j in range(ncols):
            for r in range(nrows):
                cell = tbl[(r, j)]
                cell.set_width(rel[j])

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
        op = out_path if pages == 1 else out_path.with_name(f"{out_path.stem}_p{p+1}.png")
        plt.savefig(op, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved → {op}")

# -------- Load + tidy BH baseline, then render --------
candidates = list(Path(".").rglob("bh_baseline_per_ticker.csv"))
if not candidates:
    raise FileNotFoundError("Could not find bh_baseline_per_ticker.csv anywhere in project.")
CSV = candidates[0]

df = pd.read_csv(CSV)

# Drop columns that add no information for BH
df = df.drop(columns=[c for c in ["file", "source"] if c in df.columns])

# Rename to thesis-friendly headers
rename_map = {
    "ticker":          "Ticker",
    "test_sharpe":     "Sharpe",
    "test_mdd":        "MDD",
    "test_ann_return": "Ann. return",
    "test_ann_vol":    "Ann. vol",
    "test_n_points":   "N",
}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

# Optional: enforce consistent ticker order
if "Ticker" in df.columns:
    order = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    present = [t for t in order if t in df["Ticker"].tolist()]
    df = (df.set_index("Ticker").reindex(present).reset_index())

# Render
OUT = Path("results/chap5/baselines/Table_BH_per_ticker.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

save_table_png(
    df, OUT,
    #title="Buy-and-Hold Baseline (per ticker)",
    max_rows_per_page=10,
    # compact but readable column widths (in inches)
    col_width_overrides={"Ticker": 0.75, "Sharpe": 0.85, "MDD": 0.85,
                         "Ann. return": 1.00, "Ann. vol": 0.95, "N": 0.65},
    header_fontsize=18,
    body_fontsize=16,
    header_height=0.22,
    row_height=0.18,
    max_fig_h=4.4
)