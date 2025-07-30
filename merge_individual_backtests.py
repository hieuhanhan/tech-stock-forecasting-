# merge_individual_backtests.py
import os
import glob
import argparse
import numpy as np
import pandas as pd

def sharpe_ratio(r):
    if r.std() == 0: return 0.0
    return (r.mean() / r.std()) * np.sqrt(252)

def max_drawdown(r):
    if len(r) == 0: return 0.0
    equity = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max())

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="data/backtest_results")
    ap.add_argument("--curves-dir",  default="data/backtest_curves")
    ap.add_argument("--out-summary", default="data/backtest_results/all_tickers_arima_backtest_summary.csv")
    ap.add_argument("--out-curves",  default="data/backtest_curves/all_tickers_equity_curves.csv")
    ap.add_argument("--out-portfolio", default="data/backtest_curves/all_tickers_portfolio_equity_curve.csv")
    return ap.parse_args()

def infer_ticker_from_filename(path):
    base = os.path.basename(path)
    return base.split("_")[0]

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_curves), exist_ok=True)

    sum_paths = glob.glob(os.path.join(args.summary_dir, "*_arima_final_backtest.csv"))
    summaries = []
    for p in sum_paths:
        df = pd.read_csv(p)
        if "ticker" not in df.columns:
            df["ticker"] = infer_ticker_from_filename(p)
        summaries.append(df)
    if summaries:
        df_sum_all = pd.concat(summaries, ignore_index=True)
        df_sum_all.to_csv(args.out_summary, index=False)
        print(f"[INFO] Saved merged summary to: {args.out_summary}")
    else:
        print("[WARN] No summary files found.")

    # --- Gộp CURVES ---
    curve_paths = glob.glob(os.path.join(args.curves_dir, "*_equity_curve.csv"))
    curves = []
    for p in curve_paths:
        df = pd.read_csv(p)
        if "ticker" not in df.columns:
            df["ticker"] = infer_ticker_from_filename(p)
        curves.append(df[["day","ticker","net_return"]])
    if not curves:
        print("[WARN] No curve files found.")
        return

    df_curves_all = pd.concat(curves, ignore_index=True)

    df_grouped = df_curves_all.groupby(["day","ticker"], as_index=False)["net_return"].sum()
    df_pivot = df_grouped.pivot(index="day", columns="ticker", values="net_return").fillna(0.0)

    df_pivot["portfolio_return"] = df_pivot.mean(axis=1)

    df_pivot.to_csv(args.out_portfolio, index=True)
    print(f"[INFO] Saved portfolio equity curve (wide) to: {args.out_portfolio}")

    df_long = df_pivot.reset_index().melt(id_vars="day", var_name="ticker", value_name="net_return")
    df_long.to_csv(args.out_curves, index=False)
    print(f"[INFO] Saved all-tickers curves (long) to: {args.out_curves}")

    port_ret = df_pivot["portfolio_return"].values
    port_sharpe = sharpe_ratio(pd.Series(port_ret))
    port_mdd = max_drawdown(pd.Series(port_ret))
    port_cum = float(np.exp(np.sum(port_ret)) - 1)
    print("\n[Portfolio Metrics]")
    print(f"Sharpe Ratio      : {port_sharpe:.4f}")
    print(f"Max Drawdown      : {port_mdd:.4f}")
    print(f"Cumulative Return : {port_cum:.4f}")

if __name__ == "__main__":
    main()