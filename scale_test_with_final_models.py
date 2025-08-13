#!/usr/bin/env python3
"""
Scale (and optional PCA-transform) a TEST set using the *final* scaler/PCA
trained per-model (ARIMA vs LSTM) by `final_scaler_and_pca.py`.

This version integrates with `scaling_utils.py` so that:
- Old artifacts that referenced `__main__.HybridScaler` can still be unpickled.
- New artifacts reference `scaling_utils.HybridScaler`, avoiding future errors.
"""

import os
import json
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scaling_utils import install_unpickle_shim

# ensure old pickles that reference __main__.HybridScaler can be loaded
install_unpickle_shim()

DEFAULT_KEEP = [
    "Date", "Ticker", "target_log_returns", "target", "Log_Returns", "Close_raw"
]


def _load_meta(models_root: str, model_type: str) -> dict:
    meta_path = os.path.join(models_root, f"{model_type}_final_models_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"[ERROR] Meta not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def _load_scaler_pca(models_root: str, model_type: str):
    scaler_path = os.path.join(models_root, f"{model_type}_final_scaler.pkl")
    pca_path = os.path.join(models_root, f"{model_type}_final_pca.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")
    scaler = load(scaler_path)
    pca = load(pca_path) if os.path.exists(pca_path) else None
    return scaler, pca


def _parse_keep_cols(keep_cols_arg: Optional[str]) -> List[str]:
    if not keep_cols_arg:
        return DEFAULT_KEEP
    return [c.strip() for c in keep_cols_arg.split(",") if c.strip()]


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            # create missing feature as NaN then fill
            df[c] = np.nan
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # simple gap fill to avoid inf/nan explosions at transform time
    df[cols] = df[cols].ffill().bfill()
    arr = df[cols].to_numpy()
    if not np.isfinite(arr).all():
        raise ValueError("[ERROR] Non-finite values remain in feature columns after fill.")
    return df


def _apply_transform(
    test_df: pd.DataFrame,
    features: List[str],
    scaler,
    pca,
    keep_cols: List[str],
) -> pd.DataFrame:
    """Apply scaler and optional PCA.

    Supports StandardScaler/MinMaxScaler and custom scalers with
    transform(DataFrame)->DataFrame (e.g., HybridScaler from scaling_utils).
    """
    # Transform features
    if isinstance(scaler, (StandardScaler, MinMaxScaler)):
        Xs = scaler.transform(test_df[features].to_numpy(dtype=np.float32, copy=False))
    else:
        # HybridScaler (or any object exposing transform(DataFrame)->DataFrame)
        Xs = scaler.transform(test_df[features]).to_numpy(dtype=np.float32)

    if pca is not None:
        Xp = pca.transform(Xs)
        out_cols = [f"PC{i+1}" for i in range(Xp.shape[1])]
        feat_df = pd.DataFrame(Xp, columns=out_cols, index=test_df.index)
    else:
        out_cols = features
        feat_df = pd.DataFrame(Xs, columns=out_cols, index=test_df.index)

    # Keep metadata columns if present
    keep = [c for c in keep_cols if c in test_df.columns]
    if keep:
        out_df = pd.concat(
            [test_df[keep].reset_index(drop=True), feat_df.reset_index(drop=True)],
            axis=1,
        )
    else:
        out_df = feat_df
    return out_df


def main():
    ap = argparse.ArgumentParser(
        description="Scale a TEST set using per-model FinalScaler/PCA."
    )
    ap.add_argument("--model_type", required=True, choices=["arima", "lstm"])
    ap.add_argument("--test_in", required=True, help="CSV of raw test with features (unscaled)")
    ap.add_argument(
        "--test_out",
        required=True,
        help="CSV path to save the scaled (and PCA'd) test set",
    )
    ap.add_argument(
        "--models_root",
        default=os.path.join("data", "processed_folds", "final", "models"),
    )
    ap.add_argument(
        "--feature_columns_path",
        default="",
        help="Optional JSON list of feature names. If omitted, load from meta",
    )
    ap.add_argument("--keep_cols", default=",".join(DEFAULT_KEEP))

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.test_out), exist_ok=True)

    # 1) Load artifacts & feature names
    meta = _load_meta(args.models_root, args.model_type)
    scaler, pca = _load_scaler_pca(args.models_root, args.model_type)

    if args.feature_columns_path and os.path.exists(args.feature_columns_path):
        with open(args.feature_columns_path, "r") as f:
            features = json.load(f)
        if not isinstance(features, list) or not features:
            raise ValueError("[ERROR] Invalid feature columns JSON provided.")
    else:
        features = meta.get("feature_cols")
        if not features:
            raise ValueError(
                "[ERROR] feature_cols missing in meta; please provide --feature_columns_path"
            )

    # 2) Load TEST
    test_df = pd.read_csv(args.test_in)
    if "Date" in test_df.columns:
        test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")

    # 3) Ensure numeric + no NaN/inf in features
    test_df = _ensure_numeric(test_df, features)

    # 4) Transform
    keep_cols = _parse_keep_cols(args.keep_cols)
    out_df = _apply_transform(test_df, features, scaler, pca, keep_cols)

    # 5) Save
    out_df.to_csv(args.test_out, index=False)
    print(f"[INFO] Saved scaled test set -> {args.test_out}")


if __name__ == "__main__":
    main()
