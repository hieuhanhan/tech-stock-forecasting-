#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, math

# ---------- style ----------
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

def save_table_png(df: pd.DataFrame, out_path, title="",
                   max_rows=12, header_fs=16, body_fs=14,
                   row_h=0.22, header_h=0.25, widths=None):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(smart_sig4)

    cols = list(df.columns)
    if not widths:
        widths = [1.0] * len(cols)
    table_w = sum(widths) + 0.5

    n = len(df)
    pages = max(1, math.ceil(n / max_rows))

    for p in range(pages):
        page = df.iloc[p*max_rows : min((p+1)*max_rows, n)]
        fig_h = header_h + row_h*(len(page)+1) + 0.25
        fig, ax = plt.subplots(figsize=(table_w, fig_h))
        ax.axis("off")

        if title:
            ax.set_title(title if pages==1 else f"{title} (p{p+1}/{pages})",
                         fontsize=16, fontweight="bold", loc="left")

        tbl = ax.table(cellText=page.values,
                       colLabels=cols,
                       cellLoc="center", colLoc="center",
                       loc="upper left", bbox=[0,0,1,1])

        rel = np.array(widths) / np.sum(widths)
        nrows, ncols = len(page)+1, len(cols)
        for j in range(ncols):
            for r in range(nrows):
                tbl[(r,j)].set_width(rel[j])

        for (r,c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(HEADER_COLOR)
                cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=header_fs)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.8)
            else:
                cell.set_facecolor("white" if r%2==1 else ROW_ALT)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
                cell.set_fontsize(body_fs)

        fig.tight_layout()
        out_path = Path(out_path)
        op = out_path if pages==1 else out_path.with_name(f"{out_path.stem}_p{p+1}.png")
        plt.savefig(op, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved → {op}")

def main():
    ap = argparse.ArgumentParser("Make nicely formatted PNG table from CSV (ARIMA/LSTM/etc.)")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--fold-col", default="fold_id", help="Column name for fold")
    ap.add_argument("--a-col", required=True, help="Metric A column (e.g., rmse or lstm_rmse)")
    ap.add_argument("--b-col", required=True, help="Metric B column to compare against (e.g., rmse_rwd)")
    ap.add_argument("--a-label", default="A", help="Displayed header for metric A")
    ap.add_argument("--b-label", default="B", help="Displayed header for metric B")
    ap.add_argument("--delta-label", default="Δ", help="Header for difference A - B")
    ap.add_argument("--pct-label", default="Over B (%)", help="Header for percentage (Δ/B * 100)")
    ap.add_argument("--sort-by", default="", help="Optional: column name to sort by (prefix '-' for desc)")
    ap.add_argument("--title", default="", help="Optional table title")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--max-rows", type=int, default=12, help="Max rows per page")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Validate columns
    for col in [args.fold_col, args.a_col, args.b_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Available: {list(df.columns)}")

    # Build display frame
    disp = pd.DataFrame({
        "Fold": df[args.fold_col].values,
        args.a_label: df[args.a_col].values,
        args.b_label: df[args.b_col].values
    })

    # Derived columns
    disp[args.delta_label] = disp[args.a_label] - disp[args.b_label]
    with np.errstate(divide="ignore", invalid="ignore"):
        disp[args.pct_label] = np.where(disp[args.b_label] != 0,
                                        (disp[args.delta_label] / disp[args.b_label]) * 100,
                                        np.nan)

    # Optional sorting
    if args.sort_by:
        col = args.sort_by.lstrip("-")
        ascending = not args.sort_by.startswith("-")
        if col not in disp.columns:
            raise ValueError(f"--sort-by '{args.sort_by}' not found in {list(disp.columns)}")
        disp = disp.sort_values(col, ascending=ascending, kind="mergesort")

    # Save
    save_table_png(
        disp,
        out_path=args.out,
        title=args.title,
        max_rows=args.max_rows
    )

if __name__ == "__main__":
    main()