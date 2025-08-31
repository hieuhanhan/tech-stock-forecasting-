#!/usr/bin/env python3
# bh_baseline_unified.py
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_DAYS = 252
EPS = 1e-12

# ---------- Metrics ----------
def sharpe_ratio(simple_returns: np.ndarray) -> float:
    if len(simple_returns) < 2:
        return 0.0
    sd = float(np.std(simple_returns, ddof=1))
    return 0.0 if sd == 0.0 else (float(np.mean(simple_returns)) / sd) * math.sqrt(TRADING_DAYS)

def max_drawdown_log(log_rets: np.ndarray) -> float:
    if len(log_rets) == 0:
        return 0.0
    curve = np.exp(np.cumsum(log_rets))
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / (peak + EPS)
    return float(np.max(dd))

# ---------- Helpers ----------
PREF_COLS = [
    "target_log_returns",   # typical LSTM output
    "Log_Returns",          # typical ARIMA output
    "target_Log_Returns",   # alternative casing
    "log_returns",
]
PRICE_COLS = ["Adj Close", "Adj_Close", "Close", "Price"]

def pick_return_column(df: pd.DataFrame) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    # Prefer columns that do not contain "scaled"
    for cand in PREF_COLS:
        lc = cand.lower()
        if lc in cols:
            real = cols[lc]
            if "scaled" not in real.lower():
                return real
    # fallback: any non-scaled column containing "return"
    non_scaled = [c for c in df.columns if "scaled" not in c.lower()]
    for c in non_scaled:
        if "return" in c.lower():
            return c
    return None

def build_log_returns_from_price(df: pd.DataFrame) -> np.ndarray | None:
    for c in PRICE_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            logret = np.log(s / s.shift(1)).dropna()
            return logret.to_numpy(dtype=float)
    return None

def compute_metrics_from_series(logret: np.ndarray) -> dict:
    simple = np.exp(logret) - 1.0
    return {
        "test_sharpe": sharpe_ratio(simple),
        "test_mdd": max_drawdown_log(logret),
        "test_ann_return": float(np.mean(simple) * TRADING_DAYS),
        "test_ann_vol": float(np.std(simple, ddof=1) * math.sqrt(TRADING_DAYS)),
        "test_n_points": int(len(simple)),
    }

def process_csv_grouped_by_ticker(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
    if "Ticker" not in df.columns:
        raise ValueError(f"{csv_path}: missing required column 'Ticker'.")

    ret_col = pick_return_column(df)
    if ret_col is None:
        built = build_log_returns_from_price(df)
        if built is None:
            raise ValueError(f"{csv_path}: no valid returns column found and unable to build from prices.")
        # Build from prices per ticker
        rows = []
        for tck, g in df.groupby("Ticker", dropna=True):
            built_t = build_log_returns_from_price(g)
            if built_t is None or len(built_t) == 0:
                continue
            met = compute_metrics_from_series(built_t)
            rows.append({"source": label, "file": str(csv_path), "ticker": str(tck), **met})
        return pd.DataFrame(rows)

    # Return column exists: compute metrics per ticker
    rows = []
    for tck, g in df.groupby("Ticker", dropna=True):
        logret = pd.to_numeric(g[ret_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(logret) == 0:
            continue
        met = compute_metrics_from_series(logret)
        rows.append({"source": label, "file": str(csv_path), "ticker": str(tck), **met})
    return pd.DataFrame(rows)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Unified Buy-and-Hold baseline (aggregated by Ticker)")
    ap.add_argument("--test-csv", nargs="+", required=True,
                    help="One or more test CSVs. Must contain 'Ticker' and returns (no fold_id required).")
    ap.add_argument("--out-csv", required=True, help="Output CSV path (results aggregated per Ticker).")
    ap.add_argument("--label", default="BH", help="Baseline label (default: BH).")
    args = ap.parse_args()

    out_parts = []
    for p in args.test_csv:
        part = process_csv_grouped_by_ticker(Path(p), label=args.label)
        out_parts.append(part)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(
        columns=["source","file","ticker","test_sharpe","test_mdd","test_ann_return","test_ann_vol","test_n_points"]
    )

    if "fold_id" in out.columns:
        out = out.drop(columns=["fold_id"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved baseline (per Ticker) → {out_path}")

if __name__ == "__main__":
    main()