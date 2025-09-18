#!/usr/bin/env python3
"""
Refit Tier-1 backbone for a single fold to log:
- Effective epochs (before early stopping)
- Best validation RMSE

Usage examples:
  python refit_tier1_fold.py \
    --manifest data/processed_folds/final/lstm/lstm_folds_final_paths.json \
    --base-dir data/processed_folds \
    --fold-id 347 \
    --champ-json data/tuning_results/jsons/tier1_lstm_backbone_results.json

  # Hoặc tự truyền hyperparams nếu không có champ-json:
  python refit_tier1_fold.py \
    --manifest data/.../lstm_folds_final_paths.json \
    --base-dir data/processed_folds \
    --fold-id 347 \
    --layers 2 --batch-size 64 --dropout 0.3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# --- TensorFlow/Keras ---
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ==================== CONFIG (Tier-1 neutral knobs) ====================
WINDOW_T1 = 25       # fixed look-back cho Tier-1 backbone
UNITS_T1  = 48
LR_T1     = 1e-3
MAX_EPOCHS = 10
PATIENCE   = 3

# Các cột KHÔNG dùng làm feature (giống pipeline)
NON_FEATURE_KEEP = ["Date", "Ticker", "target", "target_log_returns", "Log_Returns", "Close_raw"]

# ==================== Helpers ====================
def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_KEEP and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols

def create_windows(X: np.ndarray, y: np.ndarray, lookback: int):
    """Build sliding windows (N,L,F) and targets (N,)."""
    from numpy.lib.stride_tricks import sliding_window_view
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.shape[0] <= lookback:
        return None, None
    all_w = sliding_window_view(X, window_shape=lookback, axis=0)   # (N-L+1, 1, L, F) or (N-L+1, L, F)
    wins = all_w[:-1]
    tars = y[lookback:]
    if wins.ndim == 4:
        wins = wins.squeeze(1)
    if wins.shape[1] != lookback:
        wins = np.transpose(wins, (0, 2, 1))
    return wins.astype(np.float32), tars.astype(np.float32)

def build_lstm(window: int, units: int, lr: float, num_features: int,
               layers: int, dropout: float, norm_type: str = "auto") -> Sequential:
    def _norm():
        if norm_type == "batch":
            return BatchNormalization()
        return LayerNormalization()
    m = Sequential()
    m.add(Input((int(window), int(num_features))))
    m.add(LSTM(int(units), return_sequences=(layers > 1)))
    m.add(_norm()); m.add(Dropout(float(dropout)))
    if layers > 1:
        m.add(LSTM(max(8, int(units // 2))))
        m.add(_norm()); m.add(Dropout(float(dropout)))
    m.add(Dense(1))
    m.compile(optimizer=Adam(learning_rate=float(lr)), loss="mse")
    return m

def refit_fold(train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               feature_cols: List[str],
               champ: Dict[str, float],
               label: str = "Fold") -> Dict[str, float]:
    """Train Tier-1 backbone once for a fold; return stats."""
    Xtr = train_df[feature_cols].to_numpy(np.float32)
    ytr = train_df["target"].to_numpy(np.float32)
    Xva = val_df[feature_cols].to_numpy(np.float32)
    yva = val_df["target"].to_numpy(np.float32)

    # Build windows: validation warm-start với đuôi train để đủ history
    Xw, yw = create_windows(Xtr, ytr, WINDOW_T1)
    Xw_val, yw_val = create_windows(
        np.concatenate([Xtr[-WINDOW_T1:], Xva], axis=0),
        np.concatenate([ytr[-WINDOW_T1:], yva], axis=0),
        WINDOW_T1
    )
    if Xw is None or Xw_val is None:
        raise RuntimeError("Not enough data to form sliding windows for this fold.")

    layers  = int(champ["layers"])
    bsize   = int(champ["batch_size"])
    dropout = float(champ["dropout"])
    normtype = "layer" if bsize <= 32 else "batch"

    # Clear TF graph/session để sạch bộ nhớ giữa các lần refit
    K.clear_session()
    model = build_lstm(WINDOW_T1, UNITS_T1, LR_T1, Xtr.shape[1], layers, dropout, norm_type=normtype)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0)

    hist = model.fit(Xw, yw,
                     validation_data=(Xw_val, yw_val),
                     epochs=MAX_EPOCHS,
                     batch_size=bsize,
                     callbacks=[es],
                     verbose=0)

    effective_epochs = int(len(hist.history["loss"]))
    best_val_rmse = float(np.sqrt(np.min(hist.history["val_loss"])))
    train_rmse_last = float(np.sqrt(hist.history["loss"][-1]))

    print(f"[{label}] Effective epochs: {effective_epochs}/{MAX_EPOCHS} | "
          f"Best val RMSE: {best_val_rmse:.6f} | Last train RMSE: {train_rmse_last:.6f}")

    return {
        "effective_epochs": effective_epochs,
        "best_val_rmse": best_val_rmse,
        "last_train_rmse": train_rmse_last,
        "layers": layers, "batch_size": bsize, "dropout": dropout
    }

def get_champion_from_json(champ_json: str, fold_id: int) -> Optional[Dict[str, float]]:
    """
    Tìm champion Tier-1 cho fold (layers, batch_size, dropout) từ file kết quả Tier-1.
    Hỗ trợ 2 dạng phổ biến:
      - list of records có field 'fold_id' + 'champion' hoặc 'best_params'
      - dict keyed theo fold_id
    """
    obj = load_json(champ_json)
    # normalize thành list record
    if isinstance(obj, dict) and "results" in obj:
        obj = obj["results"]
    if isinstance(obj, dict):
        # có thể là { "347": {...}, ... }
        obj = [v | {"fold_id": int(k)} for k, v in obj.items() if isinstance(v, dict)]
    for r in obj:
        try:
            fid = int(r.get("fold_id"))
        except Exception:
            continue
        if fid != fold_id:
            continue
        bp = r.get("champion") or r.get("best_params") or r
        layers = int(bp.get("layers", bp.get("n_layers", 1)))
        batch  = int(bp.get("batch_size", bp.get("batch", 32)))
        drop   = float(bp.get("dropout", 0.2))
        return {"layers": layers, "batch_size": batch, "dropout": drop}
    return None

# ==================== Main ====================
def main():
    ap = argparse.ArgumentParser(description="Refit Tier-1 backbone for a single fold and log effective epochs & best val RMSE.")
    ap.add_argument("--manifest", required=True, help="Path to LSTM folds manifest JSON (contains global_fold_id + train/val CSVs).")
    ap.add_argument("--base-dir", default="", help="Base directory prefix to resolve relative CSV paths.")
    ap.add_argument("--fold-id", type=int, required=True, help="Target fold id to refit.")
    ap.add_argument("--champ-json", default=None, help="(Optional) Tier-1 results JSON to read champion hyperparams.")
    ap.add_argument("--layers", type=int, default=None, help="Override: number of LSTM layers (if no champ-json).")
    ap.add_argument("--batch-size", type=int, default=None, help="Override: batch size (if no champ-json).")
    ap.add_argument("--dropout", type=float, default=None, help="Override: dropout rate (if no champ-json).")
    args = ap.parse_args()

    manifest = load_json(args.manifest)
    # Chuẩn hoá manifest thành dict {fold_id: record}
    if isinstance(manifest, dict) and "results" in manifest:
        manifest = manifest["results"]
    if isinstance(manifest, dict):
        recs = []
        for k, v in manifest.items():
            if isinstance(v, dict):
                v = v.copy()
                v["global_fold_id"] = v.get("global_fold_id", k)
                recs.append(v)
        manifest = recs

    idx: Dict[int, Dict[str, Any]] = {}
    for d in manifest:
        if not isinstance(d, dict):
            continue
        try:
            fid = int(d.get("global_fold_id"))
        except Exception:
            continue
        idx[fid] = d

    if args.fold_id not in idx:
        raise SystemExit(f"Fold {args.fold_id} not found in manifest.")

    rec = idx[args.fold_id]
    # Tên key phổ biến trong manifest; tuỳ bạn đổi cho khớp real manifest
    train_key = rec.get("final_train_path") or rec.get("train_path") or rec.get("final_train_path")
    val_key   = rec.get("final_val_path")   or rec.get("val_path")

    if train_key is None or val_key is None:
        raise SystemExit("Manifest must contain train and val CSV paths (e.g., 'train_path_lstm' and 'val_path_lstm').")

    base = args.base_dir
    train_csv = os.path.join(base, train_key) if base else train_key
    val_csv   = os.path.join(base, val_key)   if base else val_key

    if not os.path.exists(train_csv):
        raise SystemExit(f"Train CSV not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise SystemExit(f"Val CSV not found: {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # Kiểm tra có cột 'target'
    if "target" not in train_df.columns or "target" not in val_df.columns:
        raise SystemExit("Both train/val CSVs must contain a 'target' column (one-step-ahead log return).")

    feature_cols = infer_feature_cols(train_df)
    if not feature_cols:
        raise SystemExit("No feature columns inferred. Check your CSV columns and NON_FEATURE_KEEP list.")

    # Lấy champion hyperparams
    champ = None
    if args.champ_json:
        champ = get_champion_from_json(args.champ_json, args.fold_id)
        if champ is None:
            print("[WARN] Champion not found in champ-json for this fold; falling back to manual flags.")

    if champ is None:
        if args.layers is None or args.batch_size is None or args.dropout is None:
            raise SystemExit("Provide --layers, --batch-size, --dropout when champ-json is not available.")
        champ = {"layers": args.layers, "batch_size": args.batch_size, "dropout": args.dropout}

    # TF determinism (best-effort)
    np.random.seed(42)
    tf.random.set_seed(42)

    stats = refit_fold(train_df, val_df, feature_cols, champ, label=f"Fold {args.fold_id}")

    # In kết quả gọn gàng (có thể copy vào Table)
    print("\n=== SUMMARY ===")
    print(f"Fold: {args.fold_id}")
    print(f"Layers / Batch / Dropout: {stats['layers']} / {stats['batch_size']} / {stats['dropout']}")
    print(f"Effective epochs: {stats['effective_epochs']} / {MAX_EPOCHS}")
    print(f"Best val RMSE: {stats['best_val_rmse']:.6f}")
    print(f"Last train RMSE: {stats['last_train_rmse']:.6f}")

if __name__ == "__main__":
    main()