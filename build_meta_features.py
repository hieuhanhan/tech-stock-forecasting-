#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from joblib import load

# -----------------------------
# IO ROOTS
# -----------------------------
OUTPUT_BASE_DIR = os.path.join("data", "processed_folds")
ARIMA_META_DIR = os.path.join(OUTPUT_BASE_DIR, "arima_meta")
LSTM_META_DIR = os.path.join(OUTPUT_BASE_DIR, "lstm_meta")

os.makedirs(ARIMA_META_DIR, exist_ok=True)
os.makedirs(LSTM_META_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------

def absolute_path_from_relative(relative_path: str) -> str:
    return os.path.join(OUTPUT_BASE_DIR, relative_path)


def read_fold_csv(relative_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not relative_path:
        return None
    path = absolute_path_from_relative(relative_path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def jaccard_time_dates(dates_a: pd.Series, dates_b: pd.Series) -> float:
    if dates_a is None or dates_b is None:
        return 0.0
    try:
        a = pd.to_datetime(dates_a, errors="coerce")
        b = pd.to_datetime(dates_b, errors="coerce")

        if getattr(a.dtype, "tz", None) is not None:
            a = a.dt.tz_convert(None)
        if getattr(b.dtype, "tz", None) is not None:
            b = b.dt.tz_convert(None)

        a_days = a.dt.floor("D").dropna()
        b_days = b.dt.floor("D").dropna()

        set_a = set(a_days.dt.date)
        set_b = set(b_days.dt.date)

        if not set_a and not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def max_drawdown_from_prices(close_prices: pd.Series) -> float:
    if close_prices is None or len(close_prices) == 0:
        return 0.0
    c = pd.Series(close_prices).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if c.empty:
        return 0.0
    running_max = c.cummax()
    drawdown = c / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def downside_volatility(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return 0.0
    negative = r[r < 0]
    return float(negative.std()) if not negative.empty else 0.0


def trend_strength_from_prices(close_prices: pd.Series) -> float:
    """Normalized slope magnitude via simple linear fit on index vs price (z-scored internally)."""
    s = pd.Series(close_prices).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 10:
        return 0.0
    x = np.arange(n, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (s.values - s.mean()) / (s.std() + 1e-8)
    slope = np.polyfit(x, y, 1)[0]
    return float(abs(slope))


def autocorrelation_at_lags(series: pd.Series, lags: List[int]) -> List[float]:
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    for L in lags:
        if len(s) <= L or L <= 0:
            out.append(0.0)
        else:
            out.append(float(s.autocorr(L)))
    return out


def safe_skew_kurtosis(series: pd.Series) -> Tuple[float, float]:
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, 0.0
    return float(s.skew()), float(s.kurtosis())


def infer_regime_from_prices(close: pd.Series) -> pd.Series:
    """
    Simple regime tagging from price: MA20 vs MA50 spread.
    spread > +0.2% -> bull; spread < -0.2% -> bear; else sideways.
    """
    s = pd.Series(close).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if s.empty:
        return pd.Series([], dtype=object)
    ma_fast = s.rolling(20, min_periods=5).mean()
    ma_slow = s.rolling(50, min_periods=10).mean()
    spread = (ma_fast - ma_slow) / (ma_slow.abs() + 1e-8)
    up, down = 0.002, -0.002
    out = np.where(spread > up, "bull", np.where(spread < down, "bear", "sideways"))
    return pd.Series(out, index=s.index, dtype=object)


@dataclass
class MetaBuildConfig:
    model_type: str  # "lstm" or "arima"
    selection_pca_model_path: Optional[str]
    feature_columns: Optional[List[str]]
    use_train_and_val: bool = True


# -----------------------------
# Meta-feature computation per fold
# -----------------------------

def compute_meta_features_for_fold(
    fold: Dict,
    config: MetaBuildConfig,
) -> Optional[Dict]:
    """Compute meta-features for a single fold.
    Common features are computed for both models.
    LSTM-only: optional PC variance shares using selection PCA model (if provided).
    ARIMA-only: autocorrelation at few lags (1, 5, 21) on Log_Returns.
    """
    # Resolve file paths per model
    if config.model_type == "lstm":
        train_relative = fold.get("train_path_lstm")
        val_relative = fold.get("val_path_lstm")
    else:
        train_relative = fold.get("train_path_arima")
        val_relative = fold.get("val_path_arima")

    train_dataframe = read_fold_csv(train_relative)
    validation_dataframe = read_fold_csv(val_relative)
    if train_dataframe is None or validation_dataframe is None:
        return None

    data_for_meta = (
        pd.concat([train_dataframe, validation_dataframe], axis=0, ignore_index=True)
        if config.use_train_and_val else train_dataframe.copy()
    )

    if "regime" not in data_for_meta.columns:
        if "Close_raw" in data_for_meta.columns and data_for_meta["Close_raw"].notna().any():
            try:
                data_for_meta["regime"] = infer_regime_from_prices(data_for_meta["Close_raw"])
            except Exception:
                data_for_meta["regime"] = "sideways"
        else:
            data_for_meta["regime"] = pd.Series(["sideways"] * len(data_for_meta), index=data_for_meta.index, dtype=object)

    regime_series = (data_for_meta["regime"].astype(str).str.strip().str.lower().replace({"nan": np.nan}))
    vc = regime_series.value_counts(normalize=True, dropna=True)
    bull_ratio = float(vc.get("bull", 0.0))
    bear_ratio = float(vc.get("bear", 0.0))
    side_ratio = float(vc.get("sideways", 0.0))

    # Basic columns presence
    dates = data_for_meta["Date"] if "Date" in data_for_meta.columns else None
    ticker = fold.get("ticker") or fold.get("Ticker")
    labels = data_for_meta["target"] if "target" in data_for_meta.columns else None
    log_returns = data_for_meta["Log_Returns"] if "Log_Returns" in data_for_meta.columns else None
    close_prices = data_for_meta["Close_raw"] if "Close_raw" in data_for_meta.columns else None

    # Common meta
    positive_ratio = float(labels.mean()) if labels is not None else 0.0
    volatility = float(log_returns.std()) if log_returns is not None else 0.0
    downside_vol = downside_volatility(log_returns) if log_returns is not None else 0.0
    max_drawdown = max_drawdown_from_prices(close_prices) if close_prices is not None else 0.0
    skewness, kurtosis = safe_skew_kurtosis(log_returns) if log_returns is not None else (0.0, 0.0)
    trend_strength = trend_strength_from_prices(close_prices) if close_prices is not None else 0.0
    
    meta: Dict = {
        "global_fold_id": fold.get("global_fold_id"),
        "ticker": ticker,
        "positive_ratio": positive_ratio,
        "volatility": volatility,
        "downside_volatility": downside_vol,
        "max_drawdown": max_drawdown,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "trend_strength": trend_strength,
        "bull_ratio": bull_ratio,
        "bear_ratio": bear_ratio,
        "side_ratio": side_ratio,
    }

    # Model-specific meta
    if config.model_type == "lstm":
        if config.selection_pca_model_path and config.feature_columns and all(c in data_for_meta.columns for c in config.feature_columns):
            try:
                pca = load(config.selection_pca_model_path)
                features_matrix = data_for_meta[config.feature_columns].values.astype(np.float32, copy=False)
                principal_components = pca.transform(features_matrix)
                variance_per_pc = principal_components.var(axis=0)
                total_variance = variance_per_pc.sum() + 1e-12
                meta.update({
                    "pc1_variance_share": float(variance_per_pc[0] / total_variance) if variance_per_pc.size > 0 else 0.0,
                    "pc2_variance_share": float(variance_per_pc[1] / total_variance) if variance_per_pc.size > 1 else 0.0,
                    "pc1_to_pc5_cumulative_share": float(variance_per_pc[:5].sum() / total_variance) if variance_per_pc.size >= 5 else float(variance_per_pc.sum() / total_variance),
                })
            except Exception:
                meta.update({
                    "pc1_variance_share": 0.0,
                    "pc2_variance_share": 0.0,
                    "pc1_to_pc5_cumulative_share": 0.0,
                })
    else:  # ARIMA
        selected_lags = [1, 5, 21]
        autocorr_values = autocorrelation_at_lags(log_returns if log_returns is not None else pd.Series(dtype=float), selected_lags)
        for lag_value, value in zip(selected_lags, autocorr_values):
            meta[f"acf_lag_{lag_value}"] = float(value)

    # Keep time support (dates list) for overlap calculation downstream
    if dates is not None:
        meta["date_min"] = str(pd.to_datetime(dates).min())
        meta["date_max"] = str(pd.to_datetime(dates).max())
        d = pd.to_datetime(dates, errors="coerce")
        if getattr(d.dtype, "tz", None) is not None:
            d = d.dt.tz_convert(None)
        meta["date_min"] = str(d.min())
        meta["date_max"] = str(d.max())
        meta["date_list"] = d.dt.floor("D").dt.strftime("%Y-%m-%d").dropna().tolist()
    else:
        meta["date_list"] = []

    return meta


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Build meta-features for LSTM/ARIMA folds (selection-only)")
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "arima"], help="Model type")
    parser.add_argument("--folds_summary_path", type=str, required=True, help="Path to folds summary JSON (post-QC; for LSTM you may pass *_selection_pca.json)")
    parser.add_argument("--selection_pca_model", type=str, default="", help="Path to selection_pca_model.pkl (LSTM only)")
    parser.add_argument("--feature_columns_path", type=str, default=os.path.join(OUTPUT_BASE_DIR, "lstm_feature_columns.json"), help="Feature columns JSON for LSTM")
    parser.add_argument("--use_train_and_val", action="store_true", help="Compute meta-features on TRAIN+VAL concatenated (default)")
    parser.add_argument("--out_meta_json", type=str, default="", help="Output JSON path; default derives from input")
    parser.add_argument("--out_meta_csv", type=str, default="", help="Optional CSV output path")

    args = parser.parse_args()

    with open(args.folds_summary_path, "r") as file:
        folds = json.load(file)
    print(f"[INFO] Loaded folds: {len(folds)} from {args.folds_summary_path}")

    feature_columns: Optional[List[str]] = None
    if args.model_type == "lstm" and args.selection_pca_model and os.path.exists(args.feature_columns_path):
        with open(args.feature_columns_path, "r") as file:
            feature_columns = json.load(file)

    config = MetaBuildConfig(
        model_type=args.model_type,
        selection_pca_model_path=(args.selection_pca_model if args.selection_pca_model else None),
        feature_columns=feature_columns,
        use_train_and_val=True if args.use_train_and_val else True  # default True
    )

    meta_rows: List[Dict] = []
    for fold in folds:
        meta = compute_meta_features_for_fold(fold, config)
        if meta is not None:
            meta_rows.append(meta)

    if not meta_rows:
        raise RuntimeError("[ERROR] No meta-features computed. Check inputs.")

    meta_dataframe = pd.DataFrame(meta_rows)

    # Write outputs
    if not args.out_meta_json:
        base_name = os.path.splitext(os.path.basename(args.folds_summary_path))[0]
        if args.model_type.lower() == "arima":
            os.makedirs(ARIMA_META_DIR, exist_ok=True)
            args.out_meta_json = os.path.join(ARIMA_META_DIR, f"{base_name}_meta_arima.json")
        else:
            os.makedirs(LSTM_META_DIR, exist_ok=True)
            args.out_meta_json = os.path.join(LSTM_META_DIR, f"{base_name}_meta_lstm.json")

    with open(args.out_meta_json, "w") as file:
        json.dump(meta_rows, file, indent=2)
    print(f"[INFO] Saved meta-features JSON -> {args.out_meta_json}")

    if not args.out_meta_csv:
        base_name = os.path.splitext(os.path.basename(args.folds_summary_path))[0]
        if args.model_type.lower() == "arima":
            args.out_meta_csv = os.path.join(ARIMA_META_DIR, f"{base_name}_meta_arima.csv")
        else:
            args.out_meta_csv = os.path.join(LSTM_META_DIR, f"{base_name}_meta_lstm.csv")

    meta_dataframe.to_csv(args.out_meta_csv, index=False)
    print(f"[INFO] Saved meta-features CSV -> {args.out_meta_csv}")


if __name__ == "__main__":
    main()
