import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= USER CONFIG =========
CSV_PATH = "data/backtest_arima/backtest_arima_results.csv"   # đổi sang file LSTM khi cần
OUT_DIR  = "data/figures/sensitivity_arima"                         # nơi lưu ảnh
FILTER_SOURCE = None  # ví dụ: "GA+BO_knee" hoặc "LSTM_knee"; None = dùng tất cả
AGG_FN = "mean"       # "mean" hoặc "median"
POINT_SIZE_MIN, POINT_SIZE_MAX = 60, 260   # cỡ điểm cho Pareto scatter
# =================================

# Tên cột trong file CSV (giữ đồng nhất giữa ARIMA & LSTM)
COL_FOLD      = "fold_id"
COL_INTERVAL  = "retrain_interval"
COL_SOURCE    = "source"
COL_THRESHOLD = "threshold"
COL_SHARPE    = "test_sharpe"
COL_MDD       = "test_mdd"
COL_TURNOVER  = "test_turnover"

def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_data():
    df = pd.read_csv(CSV_PATH)
    df = _coerce_numeric(
        df,
        [COL_INTERVAL, COL_THRESHOLD, COL_SHARPE, COL_MDD, COL_TURNOVER]
    )
    if FILTER_SOURCE and (COL_SOURCE in df.columns):
        df = df[df[COL_SOURCE] == FILTER_SOURCE].copy()
    # bỏ hàng thiếu số liệu cốt lõi
    df = df.dropna(subset=[COL_INTERVAL, COL_THRESHOLD, COL_SHARPE, COL_MDD, COL_TURNOVER])
    return df

def aggregate_by_threshold_interval(df):
    """
    Trả về 2 bảng: (agg, spread)
      - agg: giá trị trung tâm (mean/median) cho mỗi (interval, threshold)
      - spread: độ lệch chuẩn (std) cho vẽ dải sai số (line profile)
    """
    gb = df.groupby([COL_INTERVAL, COL_THRESHOLD])
    if AGG_FN == "median":
        agg = gb[[COL_SHARPE, COL_MDD, COL_TURNOVER]].median().reset_index()
    else:
        agg = gb[[COL_SHARPE, COL_MDD, COL_TURNOVER]].mean().reset_index()
    spread = gb[[COL_SHARPE, COL_MDD, COL_TURNOVER]].std(ddof=1).reset_index().rename(
        columns={
            COL_SHARPE: f"{COL_SHARPE}_std",
            COL_MDD: f"{COL_MDD}_std",
            COL_TURNOVER: f"{COL_TURNOVER}_std",
        }
    )
    agg = agg.merge(spread, on=[COL_INTERVAL, COL_THRESHOLD], how="left")
    return agg

def plot_line_profiles(agg_df, metric, ylabel, out_path):
    """
    Line profile với dải sai số (±1 std) theo từng interval.
    """
    fig, ax = plt.subplots(figsize=(8,6), dpi=140)
    std_col = f"{metric}_std"
    for interval, sub in agg_df.groupby(COL_INTERVAL):
        sub = sub.sort_values(COL_THRESHOLD)
        x = sub[COL_THRESHOLD].values
        y = sub[metric].values
        s = sub[std_col].values
        ax.plot(x, y, marker="o", label=f"int={interval}")
        # dải sai số (có thể có NaN nếu 1 fold; an toàn thì fillna(0))
        s = np.nan_to_num(s, nan=0.0)
        ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_xlabel("Threshold (multiplier)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Threshold — aggregated across folds for ARIMA")
    ax.legend(title="Retrain interval")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def pareto_frontier(points, x_col, y_col, minimize_x=True, maximize_y=True):
    """
    Trả về chỉ số của các điểm thuộc Pareto frontier (không bị trội bởi điểm khác).
    - Với trường hợp thường gặp: muốn MDD thấp (min x) và Sharpe cao (max y).
    """
    data = points[[x_col, y_col]].values
    idx = np.arange(len(points))
    # sắp xếp theo x (tăng) rồi y (giảm) để duyệt đơn giản
    order = np.lexsort(( -data[:,1] if maximize_y else data[:,1],
                         data[:,0] if minimize_x else -data[:,0]))
    data = data[order]
    idx   = idx[order]

    frontier = []
    best_y = -np.inf if maximize_y else np.inf
    for i, (x, y) in enumerate(data):
        if maximize_y:
            if y > best_y:
                frontier.append(idx[i])
                best_y = y
        else:
            if y < best_y:
                frontier.append(idx[i])
                best_y = y
    return np.array(frontier, dtype=int)

def normalize_sizes(v, vmin=None, vmax=None, smin=POINT_SIZE_MIN, smax=POINT_SIZE_MAX):
    v = np.asarray(v)
    if vmin is None: vmin = np.nanmin(v)
    if vmax is None: vmax = np.nanmax(v)
    if vmax == vmin:
        return np.full_like(v, (smin+smax)/2, dtype=float)
    return smin + (v - vmin) * (smax - smin) / (vmax - vmin)

def plot_pareto_scatter(agg_df, out_path, annotate_best=True):
    """
    Sharpe vs MDD, size ~ Turnover, màu theo interval; vẽ Pareto frontier tổng.
    """
    fig, ax = plt.subplots(figsize=(9,6.5), dpi=140)

    # chuẩn hóa kích thước điểm theo turnover
    sizes = normalize_sizes(agg_df[COL_TURNOVER].values)

    # vẽ theo interval để dễ phân biệt
    for interval, sub in agg_df.groupby(COL_INTERVAL):
        s_sizes = normalize_sizes(sub[COL_TURNOVER].values,
                                  vmin=agg_df[COL_TURNOVER].min(),
                                  vmax=agg_df[COL_TURNOVER].max())
        ax.scatter(sub[COL_MDD], sub[COL_SHARPE], s=s_sizes, alpha=0.8, label=f"int={interval}")

        if annotate_best:
            # đánh dấu best Sharpe & best MDD trong interval
            i_best_sharpe = sub[COL_SHARPE].idxmax()
            i_best_mdd    = sub[COL_MDD].idxmin()
            for i, tag in [(i_best_sharpe, "max Sharpe"), (i_best_mdd, "min MDD")]:
                row = agg_df.loc[i]
                ax.scatter(row[COL_MDD], row[COL_SHARPE], s=260, marker="*", edgecolor="k")
                ax.text(row[COL_MDD], row[COL_SHARPE],
                        f" {tag}\nthr={row[COL_THRESHOLD]:.3g}",
                        fontsize=9, va="bottom")

    # Pareto frontier (min MDD, max Sharpe)
    pf_idx = pareto_frontier(agg_df, x_col=COL_MDD, y_col=COL_SHARPE,
                             minimize_x=True, maximize_y=True)
    pf = agg_df.loc[pf_idx].sort_values([COL_MDD, COL_SHARPE])
    ax.plot(pf[COL_MDD], pf[COL_SHARPE], linewidth=2.2)

    ax.set_xlabel("Max Drawdown (lower is better)")
    ax.set_ylabel("Sharpe Ratio (higher is better)")
    ax.set_title("Sharpe vs MDD — point size ~ Turnover; line = Pareto frontier for ARIMA")
    ax.legend(title="Retrain interval", ncol=3, frameon=True)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    _safe_mkdir(OUT_DIR)
    df = load_data()
    agg = aggregate_by_threshold_interval(df)

    # ===== Line profiles (với dải std) =====
    plot_line_profiles(agg, COL_SHARPE,   "Test Sharpe",   os.path.join(OUT_DIR, "lineprofile_sharpe.png"))
    plot_line_profiles(agg, COL_MDD,      "Test MDD",      os.path.join(OUT_DIR, "lineprofile_mdd.png"))
    plot_line_profiles(agg, COL_TURNOVER, "Test Turnover", os.path.join(OUT_DIR, "lineprofile_turnover.png"))

    # ===== Pareto scatter =====
    plot_pareto_scatter(agg, os.path.join(OUT_DIR, "pareto_scatter.png"))

    print("Saved figures to:", OUT_DIR)

if __name__ == "__main__":
    main()