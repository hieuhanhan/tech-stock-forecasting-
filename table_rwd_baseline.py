from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, math

# --- style setup ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
HEADER_COLOR = (0/255, 61/255, 120/255)
HEADER_TEXT  = "white"
ROW_ALT      = (248/255, 251/255, 255/255)
GRID_COLOR   = (210/255, 220/255, 235/255)

def smart_sig4(x):
    try: v = float(x)
    except: return x
    if v == 0: return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e4: return f"{v:.2e}"
    return f"{v:.4g}"

def save_table_png(df, out_path, title="", max_rows=12,
                   header_fs=16, body_fs=14, row_h=0.22, header_h=0.25):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): df[c] = df[c].map(smart_sig4)

    cols = list(df.columns)
    widths = [1.0]*len(cols)
    table_w = sum(widths) + 0.5

    n = len(df)
    pages = max(1, math.ceil(n/max_rows))

    for p in range(pages):
        page = df.iloc[p*max_rows:min((p+1)*max_rows, n)]
        fig_h = header_h + row_h*(len(page)+1) + 0.25
        fig, ax = plt.subplots(figsize=(table_w, fig_h)); ax.axis("off")

        if title:
            ax.set_title(title if pages==1 else f"{title} (p{p+1}/{pages})",
                         fontsize=16, fontweight="bold", loc="left")

        tbl = ax.table(cellText=page.values, colLabels=cols,
                       cellLoc="center", colLoc="center", loc="upper left",
                       bbox=[0,0,1,1])

        rel = np.array(widths)/np.sum(widths)
        nrows, ncols = len(page)+1, len(cols)
        for j in range(ncols):
            for r in range(nrows): tbl[(r,j)].set_width(rel[j])

        for (r,c),cell in tbl.get_celld().items():
            if r==0:
                cell.set_facecolor(HEADER_COLOR)
                cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=header_fs)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.8)
            else:
                cell.set_facecolor("white" if r%2==1 else ROW_ALT)
                cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.6)
                cell.set_fontsize(body_fs)

        fig.tight_layout()
        op = out_path if pages==1 else Path(out_path).with_name(f"{Path(out_path).stem}_p{p+1}.png")
        plt.savefig(op, bbox_inches="tight"); plt.close(fig)
        print(f"[OK] Saved → {op}")

# ---- main ----
df = pd.read_csv("results/baselines/rwd_arima_rmse.csv")

# chỉ giữ cột quan trọng
df = df[["fold_id","rmse","rmse_rwd"]].rename(columns={
    "fold_id":"Fold", "rmse":"ARIMA RMSE", "rmse_rwd":"RWD RMSE"
})
df["ΔRMSE"] = df["ARIMA RMSE"] - df["RWD RMSE"]
df["Over RWD (%)"] = (df["ΔRMSE"]/df["RWD RMSE"])*100

OUT = Path("results/chap5/baselines/Table_RWD_per_fold_ARIMA.png")
save_table_png(df, OUT) #title="RWD Baseline vs ARIMA (per fold)"