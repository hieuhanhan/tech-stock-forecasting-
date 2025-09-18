import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================
# CONFIG
# =============================
ARIMA_CSV = "data/tuning_results/csv/tier2_arima_front.csv"  
LSTM_CSV  = "data/tuning_results/csv/tier2_lstm_front.csv"  

OUT_DIR_ARIMA = "figures/pareto_best_vs_knee/arima"
OUT_DIR_LSTM  = "figures/pareto_best_vs_knee/lstm"


RENDER_ARIMA = True
RENDER_LSTM  = True

LIMIT_INTERVALS = None  # e.g., [10, 20, 42]

# =============================
# Helpers
# =============================
def _exists_cols(df, cols):
    return all(c in df.columns for c in cols)

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common variants to canonical names: fold_id, retrain_interval, front_type, sharpe, mdd."""
    rename_map = {}
    if "interval" in df.columns and "retrain_interval" not in df.columns:
        rename_map["interval"] = "retrain_interval"
    if "front" in df.columns and "front_type" not in df.columns:
        rename_map["front"] = "front_type"
    if "SR" in df.columns and "sharpe" not in df.columns:
        rename_map["SR"] = "sharpe"
    if "Sharpe" in df.columns and "sharpe" not in df.columns:
        rename_map["Sharpe"] = "sharpe"
    if "MDD" in df.columns and "mdd" not in df.columns:
        rename_map["MDD"] = "mdd"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required cols exist
    need = {"fold_id", "retrain_interval", "front_type", "sharpe", "mdd"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def _pick_best_point(sub: pd.DataFrame) -> pd.Series:
    """
    'Best' solution = lexicographic: maximize Sharpe, then minimize MDD.
    Operates on a *subset* (e.g., a specific front_type or union).
    """
    best = sub.sort_values(by=["sharpe", "mdd"], ascending=[False, True]).iloc[0]
    return best

def _compute_knee_point(sub: pd.DataFrame) -> pd.Series:
    """
    Knee = point closest to the ideal (max Sharpe, min MDD) in normalized space.
    If `is_knee` column exists and has any True, prefer that.
    """
    if "is_knee" in sub.columns and sub["is_knee"].any():
        return sub[sub["is_knee"]].iloc[0]

    # Ideal in (Sharpe, -MDD) space ⇒ maximize Sharpe, minimize MDD
    s = sub["sharpe"].to_numpy(dtype=float)
    d = sub["mdd"].to_numpy(dtype=float)

    # Normalize each dimension to [0,1]
    s_min, s_max = np.nanmin(s), np.nanmax(s)
    d_min, d_max = np.nanmin(d), np.nanmax(d)

    s_norm = (s - s_min) / (s_max - s_min + 1e-12)
    # for MDD, smaller is better => invert by 1 - norm
    d_norm = 1.0 - (d - d_min) / (d_max - d_min + 1e-12)

    # Ideal point = (1,1) in (s_norm, d_norm)
    dist = np.sqrt((1.0 - s_norm) ** 2 + (1.0 - d_norm) ** 2)
    idx = int(np.argmin(dist))
    return sub.iloc[idx]

def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def _plot_one(ax, df_sub, title):
    """
    df_sub is the filtered data for a single (fold_id, retrain_interval).
    It includes both front_type = GA and GA+BO rows.
    """
    # Split fronts
    ga = df_sub[df_sub["front_type"].str.upper().str.contains("GA", regex=False) &
                ~df_sub["front_type"].str.upper().str.contains("GA+BO")]
    gabo = df_sub[df_sub["front_type"].str.upper().str.contains("GA\\+BO", regex=True)]

    # Draw fronts
    if not ga.empty:
        ax.scatter(ga["mdd"], ga["sharpe"], s=26, alpha=0.7, label="GA", marker="o")
    if not gabo.empty:
        ax.scatter(gabo["mdd"], gabo["sharpe"], s=26, alpha=0.7, label="GA+BO", marker="^")

    # Union for knee/best selection
    union = pd.concat([ga, gabo], ignore_index=True)
    if union.empty:
        ax.set_title(title)
        ax.set_xlabel("Maximum Drawdown (MDD)")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, linestyle="--")
        return

    # Best & Knee
    knee = _compute_knee_point(union)
    best = _pick_best_point(union)

    # Plot markers
    ax.scatter([knee["mdd"]], [knee["sharpe"]], s=90, edgecolor="k", facecolor="none", linewidths=1.6, label="Knee")
    ax.scatter([best["mdd"]], [best["sharpe"]], s=90, edgecolor="k", facecolor="gold", linewidths=1.0, label="Best")

    # Annotate lightly
    ax.annotate("Knee", (knee["mdd"], knee["sharpe"]),
                textcoords="offset points", xytext=(6,6), fontsize=9)
    ax.annotate("Best", (best["mdd"], best["sharpe"]),
                textcoords="offset points", xytext=(6,-12), fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Maximum Drawdown (MDD)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")

def render_all(csv_path: str, out_dir: str, model_tag: str):
    print(f"[INFO] Loading {model_tag} fronts from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _canonicalize_columns(df)

    # Filter intervals if requested
    if LIMIT_INTERVALS is not None:
        df = df[df["retrain_interval"].isin(LIMIT_INTERVALS)].copy()

    _ensure_dir(out_dir)

    # Build list of (fold_id, retrain_interval)
    keys = df[["fold_id", "retrain_interval"]].drop_duplicates().sort_values(["retrain_interval","fold_id"]).values.tolist()

    for (fid, interval) in keys:
        sub = df[(df["fold_id"] == fid) & (df["retrain_interval"] == interval)].copy()
        if sub.empty:
            continue

        title = f"{model_tag} — Fold {fid} | Interval {interval}d"
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
        _plot_one(ax, sub, title)
        fn = f"{model_tag.lower()}_pareto_best_vs_knee_fold{fid}_int{interval}.png"
        out_path = os.path.join(out_dir, fn)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[OK] Saved: {out_path}")

# =============================
# Run
# =============================
if __name__ == "__main__":
    if RENDER_ARIMA and os.path.exists(ARIMA_CSV):
        render_all(ARIMA_CSV, OUT_DIR_ARIMA, model_tag="ARIMA–GARCH")
    if RENDER_LSTM and os.path.exists(LSTM_CSV):
        render_all(LSTM_CSV, OUT_DIR_LSTM, model_tag="LSTM")
    print("\nDone.")