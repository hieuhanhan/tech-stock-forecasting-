import pandas as pd
from pathlib import Path

# --------------------------
# CONFIG – update paths if needed
# --------------------------
PATH_LSTM  = Path("data/backtest_lstm/backtest_lstm_results.csv")
PATH_ARIMA = Path("data/backtest_arima/backtest_arima_results.csv")  # adjust if different

# Optional: filter to a specific "source" (e.g., "GA+BO_knee"); set to None to keep all
SOURCE_FILTER = "GA+BO_knee"   # or None

# --------------------------
# Helpers
# --------------------------
def load_normalize_any(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # common aliases -> canonical names
    rename_map = {
        "test_sharpe": "sharpe",
        "val_sharpe": "sharpe_val",
        "test_mdd": "mdd",
        "val_mdd": "mdd_val",
        "test_ann_return": "ann_return",
        "annualized_return": "ann_return",
        "annualized_ret": "ann_return",
        "retrain": "retrain_interval",
        "interval": "retrain_interval",
    }
    df = df.rename(columns=rename_map)

    # keep only columns we might need; if some are missing, we'll handle gracefully
    keep = [c for c in [
        "fold_id", "retrain_interval", "source",
        "sharpe", "mdd", "ann_return"
    ] if c in df.columns]
    df = df[keep].copy()

    # coerce numerics
    for col in ["fold_id", "retrain_interval"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["sharpe", "mdd", "ann_return"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # optional filter on source (only if the column exists)
    if SOURCE_FILTER is not None and "source" in df.columns:
        df = df[df["source"].isin([SOURCE_FILTER])]

    return df

def summarize(df: pd.DataFrame, model_label: str) -> pd.Series:
    """Compute summary stats without assuming any fixed number of rows."""
    out = {
        "Model": model_label,
        "Avg. Sharpe Ratio": df["sharpe"].mean() if "sharpe" in df.columns else float("nan"),
        "Median MDD": df["mdd"].median() if "mdd" in df.columns else float("nan"),
        "Avg. Annualized Return": df["ann_return"].mean() if "ann_return" in df.columns else float("nan"),
        "n_backtests": len(df)
    }
    return pd.Series(out)

# --------------------------
# Load, normalize, summarize
# --------------------------
arima = load_normalize_any(PATH_ARIMA)
lstm  = load_normalize_any(PATH_LSTM)

sum_arima = summarize(arima, "ARIMA–GARCH")
sum_lstm  = summarize(lstm,  "LSTM")

summary = pd.DataFrame([sum_arima, sum_lstm])

# Nice console view
disp = summary.copy()
if pd.api.types.is_numeric_dtype(disp["Avg. Annualized Return"]):
    disp["Avg. Annualized Return"] = disp["Avg. Annualized Return"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
for c in ["Avg. Sharpe Ratio", "Median MDD"]:
    if pd.api.types.is_numeric_dtype(disp[c]):
        disp[c] = disp[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

print("\n=== Summary comparison (no row limits; NA-safe) ===")
print(disp[["Model","Avg. Sharpe Ratio","Median MDD","Avg. Annualized Return","n_backtests"]].to_string(index=False))

# Save raw numeric summary (better for later processing)
summary.to_csv("table_arima_lstm_summary.csv", index=False)
print("\nSaved CSV -> table_arima_lstm_summary.csv")

# --------------------------
# (Optional) also save per-interval summaries if present (won't fail if not)
# --------------------------
if "retrain_interval" in arima.columns or "retrain_interval" in lstm.columns:
    def per_interval(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
        if "retrain_interval" not in df.columns:
            return pd.DataFrame()
        grp = df.groupby("retrain_interval", dropna=True).agg(
            avg_sharpe=("sharpe","mean"),
            median_mdd=("mdd","median"),
            avg_ann_return=("ann_return","mean"),
            n=("sharpe","count")
        ).reset_index()
        grp.insert(0, "Model", model_label)
        return grp

    pi_arima = per_interval(arima, "ARIMA–GARCH")
    pi_lstm  = per_interval(lstm,  "LSTM")
    per_interval_all = pd.concat([pi_arima, pi_lstm], ignore_index=True)
    if not per_interval_all.empty:
        per_interval_all.to_csv("table_arima_lstm_by_interval.csv", index=False)
        print("Saved CSV -> table_arima_lstm_by_interval.csv")