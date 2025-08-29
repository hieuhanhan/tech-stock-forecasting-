import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- paths ----------------
HV_CSV   = Path("data/tuning_results/csv/tier2_arima_cont_gabo_hv.csv")
OUT_DIR  = Path("results/chap5/hv_filtered"); OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_IMPROVED = OUT_DIR / "Table_5_3_hv_improved_only.csv"
CSV_SUMMARY  = OUT_DIR / "Table_5_3_hv_improvement_summary_by_interval.csv"
PNG_IMPROVED = OUT_DIR / "Fig_5_3_hv_improved_only.png"
PNG_SUMMARY  = OUT_DIR / "Fig_5_3_hv_improvement_summary.png"

# ---------------- style ----------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

HEADER_COLOR = (0/255, 61/255, 120/255)  # light steel/blue
HEADER_TEXT  = "white"
ROW_ALT      = (248/255, 251/255, 255/255)  # very light blue
GRID_COLOR   = (210/255, 220/255, 235/255)

def _auto_col_widths(df: pd.DataFrame, min_w=0.75, max_w=1.6, shrink=0.9):
    """Estimate per-column width in inches; lightly shrink to keep compact."""
    widths = []
    for c in df.columns:
        lens = [len(str(c))] + [len(str(v)) for v in df[c].head(200)]
        w = min(max(np.mean(lens) * 0.055, min_w), max_w)
        widths.append(w * shrink)
    return widths

def save_table_png(df: pd.DataFrame,
                   out_path: str | Path,
                   title: str = "",
                   max_rows_per_page: int = 24,
                   float_fmt_cols: dict[str, str] | None = None):
    """
    Render a DataFrame to one or more PNGs (300 dpi), compact academic style.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Optional numeric formatting
    if float_fmt_cols:
        for col, fmt in float_fmt_cols.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "")

    # Pagination
    n = len(df)
    pages = int(math.ceil(n / max_rows_per_page)) if n else 1

    # Column widths and base size
    col_widths = _auto_col_widths(df)
    table_w = sum(col_widths) + 0.6
    header_h, row_h = 0.3, 0.28

    for p in range(pages):
        start = p * max_rows_per_page
        end   = min((p + 1) * max_rows_per_page, n)
        df_page = df.iloc[start:end] if n else df

        fig_h = header_h + row_h * (len(df_page) + 1) + 0.35
        fig, ax = plt.subplots(figsize=(table_w, fig_h))
        ax.axis("off")

        if title:
            ttl = title if pages == 1 else f"{title} (page {p+1}/{pages})"
            ax.set_title(ttl, fontsize=10, fontweight="bold", loc="left", pad=6)

        # draw table
        table = ax.table(
            cellText=df_page.values,
            colLabels=list(df_page.columns),
            cellLoc="center",
            colLoc="center",
            loc="upper left",
            bbox=[0, 0, 1, 1]
        )

        # column widths (relative)
        maxw = max(col_widths) if col_widths else 1.0
        for j, w in enumerate(col_widths):
            table.auto_set_column_width(j)

        # style cells
        for (r, c), cell in table.get_celld().items():
            if r == 0:  # header
                cell.set_facecolor(HEADER_COLOR)
                cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=9)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.8)
            else:
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
                cell.set_facecolor("white" if (r % 2 == 1) else ROW_ALT)
                cell.set_fontsize(9.0)

        fig.tight_layout()
        stem = out_path.stem if pages == 1 else f"{out_path.stem}_p{p+1}"
        png_path = out_path.with_name(f"{stem}.png")
        plt.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved table → {png_path}")

# ---------------- load & compute deltas ----------------
hv = pd.read_csv(HV_CSV)

# guard: ensure needed columns exist
need_cols = {"fold_id","retrain_interval","stage","hv"}
missing = need_cols - set(hv.columns)
if missing:
    raise ValueError(f"HV CSV missing columns: {missing}")

final = hv[hv["stage"].isin(["final_ga","final_union"])].copy()
if final.empty:
    raise ValueError("No final GA/GA+BO rows found in HV CSV.")

final["stage_clean"] = final["stage"].map({"final_ga":"GA", "final_union":"GA+BO"})

wide = (final.pivot_table(index=["fold_id","retrain_interval"],
                          columns="stage_clean", values="hv", aggfunc="first")
               .reset_index())

if not {"GA","GA+BO"}.issubset(wide.columns):
    # If one of them is missing, fill with NaN to avoid key errors
    for k in ["GA","GA+BO"]:
        if k not in wide.columns: wide[k] = np.nan

wide["ΔHV"] = wide["GA+BO"] - wide["GA"]

# keep only improvements (ΔHV > 0)
EPS = 1e-12
improved = (wide[wide["ΔHV"] > EPS]
            .sort_values(["retrain_interval","ΔHV"], ascending=[True, False])
            .rename(columns={"fold_id":"Fold","retrain_interval":"Interval",
                             "GA":"HV (GA)","GA+BO":"HV (GA+BO)"}))

# save CSV
improved.to_csv(CSV_IMPROVED, index=False)
print(f"[OK] Saved improved-only CSV → {CSV_IMPROVED}")

# summary by interval
by_int = wide.groupby("retrain_interval", as_index=False).agg(
    n_total          = ("fold_id","count"),
    n_improved       = ("ΔHV", lambda s: int((s > EPS).sum())),
    mean_delta_impr  = ("ΔHV", lambda s: float(np.nanmean(s[s > EPS])) if (s > EPS).any() else np.nan),
    median_delta_impr= ("ΔHV", lambda s: float(np.nanmedian(s[s > EPS])) if (s > EPS).any() else np.nan)
)
by_int["n_not_improved"] = by_int["n_total"] - by_int["n_improved"]
by_int["improve_rate_%"] = (by_int["n_improved"] / by_int["n_total"] * 100.0).round(1)

summary_tbl = (by_int
               .rename(columns={"retrain_interval":"Interval",
                                "n_total":"Folds",
                                "n_improved":"Improved",
                                "n_not_improved":"Not improved",
                                "mean_delta_impr":"Mean ΔHV",
                                "median_delta_impr":"Median ΔHV"})
               [["Interval","Folds","Improved","Not improved","improve_rate_%",
                 "Mean ΔHV","Median ΔHV"]])

summary_tbl.to_csv(CSV_SUMMARY, index=False)
print(f"[OK] Saved summary CSV → {CSV_SUMMARY}")

# ---------------- pretty PNG tables ----------------
if not improved.empty:
    df_disp = improved.copy()
    for col in ["HV (GA)", "HV (GA+BO)", "ΔHV"]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "")
    save_table_png(
        df_disp,
        PNG_IMPROVED,
        title="",
        max_rows_per_page=28
    )
else:
    print("[INFO] No folds improved; skipping improved-only PNG.")

summary_tbl_fmt = summary_tbl.copy()
for col in ["Mean ΔHV", "Median ΔHV", "improve_rate_%"]:
    if col in summary_tbl_fmt.columns:
        summary_tbl_fmt[col] = summary_tbl_fmt[col].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "")

save_table_png(
    summary_tbl_fmt,
    PNG_SUMMARY,
    title="",
    max_rows_per_page=20
)

print("\nDone.")