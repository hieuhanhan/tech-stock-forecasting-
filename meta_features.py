import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

REQUIRED_COLS = ['target', 'Log_Returns', 'Close_raw']


def compute_trend_slope(close_series: pd.Series) -> float:
    if close_series.isnull().any() or len(close_series) < 3:
        return np.nan
    try:
        x = np.arange(len(close_series))
        y = np.log1p(close_series.astype(float))
        return np.polyfit(x, y, 1)[0]
    except Exception:
        return np.nan

def load_and_filter_val_fold(val_meta_path: str, model_type: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not os.path.exists(val_meta_path):
        return None, 'missing_meta_file'

    try:
        df_val = pd.read_csv(val_meta_path)
    except Exception as e:
        return None, f'read_error: {str(e)}'

    if not all(col in df_val.columns for col in REQUIRED_COLS):
        return None, 'missing_required_columns'

    target_mean = float(df_val['target'].mean())
    if target_mean < 0.1 or target_mean > 0.6:
        return None, 'label_imbalance'

    logret_std = float(df_val['Log_Returns'].std())
    if logret_std < 1e-4:
        return None, 'flat_volatility'

    if df_val['Close_raw'].isnull().values.any():
        return None, 'missing_close'

    slope = compute_trend_slope(df_val['Close_raw'])
    if not np.isfinite(slope) or abs(slope) < 1e-5:
        return None, 'flat_or_invalid_trend'
    
    if str(model_type).lower() == 'lstm':
        if not any(str(c).startswith('PC') for c in df_val.columns):
            return None, 'missing_pca_features'

    return df_val, None

def compute_meta_statistics(df_val: pd.DataFrame, rolling_window: int = 21) -> Dict[str, float]:
    log_ret = df_val['Log_Returns'].dropna()
    meta = {
        'positive_ratio': float(df_val['target'].mean()),
        'volatility_std': float(log_ret.std()),
        'acf1': float(log_ret.autocorr(lag=1)) if len(log_ret) > 1 else np.nan,
        'acf2': float(log_ret.autocorr(lag=2)) if len(log_ret) > 2 else np.nan,
        'trend_slope': float(compute_trend_slope(df_val['Close_raw'])),
        'skewness': float(log_ret.skew()) if len(log_ret) > 2 else np.nan,
        'kurtosis': float(log_ret.kurtosis()) if len(log_ret) > 3 else np.nan,
    }

    if len(log_ret) >= rolling_window:
        rolling_vol = log_ret.rolling(window=rolling_window).std()
        meta['rolling_vol_mean'] = float(rolling_vol.mean())
        meta['rolling_vol_max']  = float(rolling_vol.max())
    else:
        meta['rolling_vol_mean'] = np.nan
        meta['rolling_vol_max']  = np.nan

    return meta

def compute_meta_with_pca(df_val: pd.DataFrame, rolling_window: int = 21) -> Dict[str, float]:
    log_ret = df_val['Log_Returns'].dropna()
    meta = {
        'positive_ratio': float(df_val['target'].mean()),
        'volatility_std': float(log_ret.std()),
        'trend_slope': float(compute_trend_slope(df_val['Close_raw'])),
    }

    if len(log_ret) >= rolling_window:
        rolling_vol = log_ret.rolling(window=rolling_window).std()
        meta['rolling_vol_mean'] = float(rolling_vol.mean())
        meta['rolling_vol_max']  = float(rolling_vol.max())
    else:
        meta['rolling_vol_mean'] = np.nan
        meta['rolling_vol_max']  = np.nan

    for col in df_val.columns:
        if col.startswith("PC") and not df_val[col].isnull().values.any():
            meta[f'{col}_mean'] = float(df_val[col].mean())
            meta[f'{col}_std']  = float(df_val[col].std())

    return meta

def build_meta_row(df_val: pd.DataFrame, model_type: str, fold_id: str, ticker: str) -> Dict[str, float]:
    if model_type == 'arima':
        meta = compute_meta_statistics(df_val)
    else:
        meta = compute_meta_with_pca(df_val)
    meta['fold_id'] = fold_id
    meta['ticker'] = ticker
    return meta
