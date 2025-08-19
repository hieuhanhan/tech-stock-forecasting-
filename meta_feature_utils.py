#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from joblib import load

# -----------------------------
# IO ROOTS
# -----------------------------
OUTPUT_BASE_DIR = os.path.join("data", "processed_folds")

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
    drawdown = c / running_max - 1.0     # <= 0
    return float(-drawdown.min()) if not drawdown.empty else 0.0

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
            # pandas autocorr can return NaN on edge cases
            val = s.autocorr(L)
            out.append(float(val) if pd.notna(val) else 0.0)
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
    selection_scaler_model_path: Optional[str]
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
    Common features for both models; LSTM: optional PCA variance share (with optional scaler);
    ARIMA: autocorrelation at lags (1, 5, 21).
    """
    # -------- Resolve file paths per model --------
    if config.model_type == "lstm":
        train_relative = fold.get("train_path_lstm")
        val_relative   = fold.get("val_path_lstm")
        val_meta_relative = fold.get("val_meta_path_lstm")  
    else:
        train_relative = fold.get("train_path_arima")
        val_relative   = fold.get("val_path_arima")
        val_meta_relative = fold.get("val_meta_path_arima") 

    train_dataframe = read_fold_csv(train_relative)
    validation_dataframe = read_fold_csv(val_relative)
    val_meta_dataframe = read_fold_csv(val_meta_relative) if val_meta_relative else None

    if train_dataframe is None or validation_dataframe is None:
        return None

    # -------- (regime/trend/...) --------
    data_for_meta = (
        pd.concat([train_dataframe, validation_dataframe], axis=0, ignore_index=True)
        if config.use_train_and_val else validation_dataframe.copy()
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

    # -------- Lấy nguồn VAL META cho ratio/vol của ARIMA --------
    dates = data_for_meta["Date"] if "Date" in data_for_meta.columns else None
    ticker = fold.get("ticker") or fold.get("Ticker")

    if config.model_type == "arima":
        # Ưu tiên lấy từ val_meta CSV
        labels_val = val_meta_dataframe["target"] if (val_meta_dataframe is not None and "target" in val_meta_dataframe.columns) else None
        log_returns_val = val_meta_dataframe["Log_Returns"] if (val_meta_dataframe is not None and "Log_Returns" in val_meta_dataframe.columns) else None
        close_val = val_meta_dataframe["Close_raw"] if (val_meta_dataframe is not None and "Close_raw" in val_meta_dataframe.columns) else None

        # Fallback sang val CSV nếu cần
        if labels_val is None and "target" in validation_dataframe.columns:
            labels_val = validation_dataframe["target"]
        if log_returns_val is None and "Log_Returns" in validation_dataframe.columns:
            log_returns_val = validation_dataframe["Log_Returns"]
        if close_val is None and "Close_raw" in validation_dataframe.columns:
            close_val = validation_dataframe["Close_raw"]

        # Tính các chỉ số dựa trên VAL
        positive_ratio = float(pd.to_numeric(labels_val, errors="coerce").mean()) if labels_val is not None else 0.0
        volatility     = float(pd.to_numeric(log_returns_val, errors="coerce").std()) if log_returns_val is not None else 0.0
        downside_vol   = downside_volatility(log_returns_val) if log_returns_val is not None else 0.0
        mdd            = max_drawdown_from_prices(close_val) if close_val is not None else 0.0
        skewness, kurtosis = safe_skew_kurtosis(log_returns_val) if log_returns_val is not None else (0.0, 0.0)
        trend_strength = trend_strength_from_prices(close_val) if close_val is not None else 0.0
    else:
        # LSTM: giữ nguyên nguồn data_for_meta
        labels = data_for_meta["target"] if "target" in data_for_meta.columns else None
        log_returns = data_for_meta["Log_Returns"] if "Log_Returns" in data_for_meta.columns else None
        close_prices = data_for_meta["Close_raw"] if "Close_raw" in data_for_meta.columns else None

        positive_ratio = float(pd.to_numeric(labels, errors="coerce").mean()) if labels is not None else 0.0
        volatility     = float(pd.to_numeric(log_returns, errors="coerce").std()) if log_returns is not None else 0.0
        downside_vol   = downside_volatility(log_returns) if log_returns is not None else 0.0
        mdd            = max_drawdown_from_prices(close_prices) if close_prices is not None else 0.0
        skewness, kurtosis = safe_skew_kurtosis(log_returns) if log_returns is not None else (0.0, 0.0)
        trend_strength = trend_strength_from_prices(close_prices) if close_prices is not None else 0.0

    # -------- Gói meta chung --------
    meta: Dict = {
        "global_fold_id": fold.get("global_fold_id"),
        "ticker": ticker,
        "positive_ratio": positive_ratio,
        "volatility": volatility,
        "downside_volatility": downside_vol,
        "max_drawdown": mdd,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "trend_strength": trend_strength,
        "bull_ratio": bull_ratio,
        "bear_ratio": bear_ratio,
        "side_ratio": side_ratio,
        # tương thích QC/selection cũ:
        "val_pos_ratio": positive_ratio,
        "val_vol": volatility,
    }

    # -------- Model-specific --------
    if config.model_type == "lstm":
        if config.selection_pca_model_path and config.feature_columns and all(c in data_for_meta.columns for c in config.feature_columns):
            try:
                pca = load(config.selection_pca_model_path)
                scaler = load(config.selection_scaler_model_path) if config.selection_scaler_model_path else None
                features_df = (
                    data_for_meta[config.feature_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .ffill().bfill().fillna(0.0)
                )
                X = features_df.values.astype(np.float32, copy=False)
                Xs = scaler.transform(X) if scaler is not None else X
                Xp = pca.transform(Xs)
                evr = getattr(pca, "explained_variance_ratio_", None)
                if evr is not None:
                    evr = np.asarray(evr, dtype=float)
                    meta.update({
                        "pc1_variance_share": float(evr[0]) if evr.size > 0 else 0.0,
                        "pc2_variance_share": float(evr[1]) if evr.size > 1 else 0.0,
                        "pc1_to_pc5_cumulative_share": float(evr[:5].sum()) if evr.size >= 5 else float(evr.sum()),
                    })
                else:
                    var_pc = Xp.var(axis=0)
                    tot = float(var_pc.sum() + 1e-12)
                    meta.update({
                        "pc1_variance_share": float(var_pc[0] / tot) if var_pc.size > 0 else 0.0,
                        "pc2_variance_share": float(var_pc[1] / tot) if var_pc.size > 1 else 0.0,
                        "pc1_to_pc5_cumulative_share": float(var_pc[:5].sum() / tot) if var_pc.size >= 5 else float(var_pc.sum() / tot),
                    })
            except Exception:
                meta.update({
                    "pc1_variance_share": 0.0,
                    "pc2_variance_share": 0.0,
                    "pc1_to_pc5_cumulative_share": 0.0,
                })
    else:  # ARIMA
        # ACF trên VAL log-returns nếu có; fallback rỗng
        selected_lags = [1, 5, 21]
        series_for_acf = (val_meta_dataframe["Log_Returns"]
                          if (val_meta_dataframe is not None and "Log_Returns" in val_meta_dataframe.columns)
                          else (validation_dataframe["Log_Returns"] if "Log_Returns" in validation_dataframe.columns else pd.Series(dtype=float)))
        acfs = autocorrelation_at_lags(series_for_acf, selected_lags)
        for lag_value, value in zip(selected_lags, acfs):
            meta[f"acf_lag_{lag_value}"] = float(value)

    # -------- Date support --------
    if dates is not None:
        d = pd.to_datetime(dates, errors="coerce")
        if getattr(d.dtype, "tz", None) is not None:
            d = d.dt.tz_convert(None)
        meta["date_min"] = str(d.min())
        meta["date_max"] = str(d.max())
        meta["date_list"] = d.dt.floor("D").dt.strftime("%Y-%m-%d").dropna().tolist()
    else:
        meta["date_list"] = []

    return meta