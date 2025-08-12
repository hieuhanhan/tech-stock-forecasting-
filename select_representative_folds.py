#!/usr/bin/env python3
"""
Select representative folds for LSTM/ARIMA from meta-features.

Modes:
- --eval_only : in kết quả cho nhiều k (không lưu)
- --adaptive_k : chọn k tự động theo tiêu chí dừng
- (mặc định) fixed-k với --k

Optionally enrich selected items with original CSV paths if you pass
--folds_summary_path (the post-QC folds summary JSON).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

# ---------- IO roots ----------
OUTPUT_BASE_DIR = os.path.join("data", "processed_folds")
ARIMA_META_DIR = os.path.join(OUTPUT_BASE_DIR, "arima_meta")
LSTM_META_DIR  = os.path.join(OUTPUT_BASE_DIR, "lstm_meta")
os.makedirs(ARIMA_META_DIR, exist_ok=True)
os.makedirs(LSTM_META_DIR,  exist_ok=True)

# ---------- helpers ----------
def standardize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = out[c].astype(float)
        mu = float(x.mean())
        sd = float(x.std()) or 1.0
        out[c] = (x - mu) / sd
    return out

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Y = X / norms
    S = Y @ Y.T
    return np.clip(S, 0.0, 1.0)

def jaccard_from_date_lists(a_list: List[str], b_list: List[str]) -> float:
    try:
        set_a, set_b = set(a_list), set(b_list)
        if not set_a and not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def build_penalized_dissimilarity(meta_df: pd.DataFrame,
                                  base_sim: np.ndarray,
                                  lambda_overlap: float,
                                  gamma_same_ticker: float) -> np.ndarray:
    n = len(meta_df)
    dates = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]
    tickers = meta_df["ticker"].fillna("UNK").tolist() if "ticker" in meta_df.columns else ["UNK"] * n

    overlap = np.zeros((n, n), dtype=np.float32)
    same_tkr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        di = dates[i]
        for j in range(i + 1, n):
            dj = dates[j]
            o = jaccard_from_date_lists(di, dj)
            overlap[i, j] = overlap[j, i] = o
            if tickers[i] == tickers[j]:
                same_tkr[i, j] = same_tkr[j, i] = 1.0

    penalized_sim = base_sim - lambda_overlap * overlap - gamma_same_ticker * same_tkr
    penalized_sim = np.clip(penalized_sim, 0.0, 1.0)
    return 1.0 - penalized_sim

def fixed_k_facility_location(D: np.ndarray,
                              meta_df: pd.DataFrame,
                              k: int,
                              hard_overlap: bool,
                              max_overlap: float,
                              per_ticker_max: int) -> List[int]:
    n = D.shape[0]
    sim = 1.0 - D
    dates  = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]
    tickers = meta_df["ticker"].fillna("UNK").tolist() if "ticker" in meta_df.columns else ["UNK"] * n

    overlap_raw = np.zeros((n, n), dtype=np.float32)
    if hard_overlap:
        for i in range(n):
            di = dates[i]
            for j in range(i + 1, n):
                dj = dates[j]
                overlap_raw[i, j] = overlap_raw[j, i] = jaccard_from_date_lists(di, dj)

    selected: List[int] = []
    covered = np.zeros(n, dtype=np.float32)

    def violates(idx: int) -> bool:
        tkr = tickers[idx]
        if sum(1 for s in selected if tickers[s] == tkr) >= per_ticker_max:
            return True
        if hard_overlap:
            for s in selected:
                if overlap_raw[idx, s] > max_overlap:
                    return True
        return False

    while len(selected) < min(k, n):
        best_gain, best_idx = -1e18, None
        for i in range(n):
            if i in selected:
                continue
            if violates(i):
                continue
            gain = float(np.maximum(covered, sim[:, i]).sum() - covered.sum())
            if gain > best_gain:
                best_gain, best_idx = gain, i
        if best_idx is None:
            break
        selected.append(best_idx)
        covered = np.maximum(covered, sim[:, best_idx])
    return selected

def adaptive_facility_location(D: np.ndarray,
                               meta_df: pd.DataFrame,
                               k_min: int,
                               k_max: int,
                               hard_overlap: bool,
                               max_overlap: float,
                               per_ticker_max: int,
                               coverage_target: float,
                               min_gain: float,
                               elbow_tol: float) -> Dict:
    n = D.shape[0]
    sim = 1.0 - D
    dates  = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]
    tickers = meta_df["ticker"].fillna("UNK").tolist() if "ticker" in meta_df.columns else ["UNK"] * n

    overlap_raw = np.zeros((n, n), dtype=np.float32)
    if hard_overlap:
        for i in range(n):
            di = dates[i]
            for j in range(i + 1, n):
                dj = dates[j]
                overlap_raw[i, j] = overlap_raw[j, i] = jaccard_from_date_lists(di, dj)

    selected: List[int] = []
    covered = np.zeros(n, dtype=np.float32)
    coverage_curve: List[float] = []
    gain_curve: List[float] = []
    first_gain = None

    def violates(idx: int) -> bool:
        tkr = tickers[idx]
        if sum(1 for s in selected if tickers[s] == tkr) >= per_ticker_max:
            return True
        if hard_overlap:
            for s in selected:
                if overlap_raw[idx, s] > max_overlap:
                    return True
        return False

    for t in range(1, min(k_max, n) + 1):
        best_gain, best_idx = -1e18, None
        for i in range(n):
            if i in selected:
                continue
            if violates(i):
                continue
            gain = float(np.maximum(covered, sim[:, i]).sum() - covered.sum())
            if gain > best_gain:
                best_gain, best_idx = gain, i
        if best_idx is None:
            break

        selected.append(best_idx)
        covered = np.maximum(covered, sim[:, best_idx])
        mean_cov = float(covered.mean())
        coverage_curve.append(mean_cov)
        gain_curve.append(float(best_gain))
        if first_gain is None:
            first_gain = max(1e-12, float(best_gain))

        if t >= k_min:
            stop_by_cov   = coverage_target > 0 and mean_cov >= coverage_target
            stop_by_gain  = min_gain > 0 and best_gain <= min_gain
            stop_by_elbow = elbow_tol > 0 and (best_gain / first_gain) <= elbow_tol
            if stop_by_cov or stop_by_gain or stop_by_elbow:
                break

    return {"indices": selected, "coverage_curve": coverage_curve, "gain_curve": gain_curve}

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Select representative folds with optional enrichment of paths.")
    p.add_argument("--model_type", required=True, choices=["lstm", "arima"])
    p.add_argument("--meta_json_path", required=True, help="Meta-features JSON")

    # evaluation mode (print only)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--k_min", type=int, default=10)
    p.add_argument("--k_max", type=int, default=40)
    p.add_argument("--k_step", type=int, default=5)

    # selection modes
    p.add_argument("--adaptive_k", action="store_true")
    p.add_argument("--k", type=int, default=40)

    # penalties & constraints
    p.add_argument("--lambda_overlap", type=float, default=0.9)
    p.add_argument("--gamma_same_ticker", type=float, default=0.25)
    p.add_argument("--hard_overlap", action="store_true")
    p.add_argument("--max_overlap", type=float, default=0.7)
    p.add_argument("--per_ticker_max", type=int, default=2)

    # adaptive stopping
    p.add_argument("--coverage_target", type=float, default=0.0)
    p.add_argument("--min_gain", type=float, default=0.0)
    p.add_argument("--elbow_tol", type=float, default=0.0)

    # outputs
    p.add_argument("--out_selected_json", type=str, default="")
    p.add_argument("--out_report_json", type=str, default="")

    args = p.parse_args()

    # load meta
    with open(args.meta_json_path, "r") as f:
        meta_rows = json.load(f)
    meta_df = pd.DataFrame(meta_rows)

    # numeric cols
    exclude = {"global_fold_id", "ticker", "date_min", "date_max", "date_list"}
    num_cols = [c for c in meta_df.columns if c not in exclude and np.issubdtype(meta_df[c].dtype, np.number)]
    if not num_cols:
        raise RuntimeError("[ERROR] No numeric meta-features available.")

    X = standardize_columns(meta_df[num_cols], num_cols).values.astype(np.float32, copy=False)
    base_sim = cosine_similarity_matrix(X)
    D = build_penalized_dissimilarity(
        meta_df=meta_df,
        base_sim=base_sim,
        lambda_overlap=float(args.lambda_overlap),
        gamma_same_ticker=float(args.gamma_same_ticker),
    )

    # evaluation only
    if args.eval_only:
        print(f"[INFO] EVALUATE k in [{args.k_min}, {args.k_max}] step {args.k_step} | "
              f"hard_overlap={bool(args.hard_overlap)} max_overlap={args.max_overlap} per_ticker_max={args.per_ticker_max}")
        sim = 1.0 - D
        for k in range(int(args.k_min), int(args.k_max) + 1, int(args.k_step)):
            idx = fixed_k_facility_location(D, meta_df, k, bool(args.hard_overlap),
                                            float(args.max_overlap), int(args.per_ticker_max))
            coverage = float(sim[:, idx].max(axis=1).mean()) if idx else 0.0
            if len(idx) > 1:
                subD = D[np.ix_(idx, idx)]
                tri = subD[np.triu_indices(len(idx), 1)]
                diversity = float(tri.mean()) if tri.size else 0.0
            else:
                diversity = 0.0
            print(f"k={k:>3} | coverage_mean={coverage:.3f} | diversity_mean={diversity:.3f} | selected={len(idx)}")
        return

    # selection
    if args.adaptive_k:
        res = adaptive_facility_location(
            D, meta_df,
            int(args.k_min), int(args.k_max),
            bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
            float(args.coverage_target), float(args.min_gain), float(args.elbow_tol)
        )
        sel_idx = res["indices"]
        coverage_curve = res["coverage_curve"]
        gain_curve = res["gain_curve"]
    else:
        sel_idx = fixed_k_facility_location(
            D, meta_df, int(args.k),
            bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max)
        )
        # simple curves from fixed picks
        sim = 1.0 - D
        coverage_curve, gain_curve = [], []
        covered = np.zeros(D.shape[0], dtype=np.float32)
        for idx in sel_idx:
            gain = float(np.maximum(covered, sim[:, idx]).sum() - covered.sum())
            covered = np.maximum(covered, sim[:, idx])
            coverage_curve.append(float(covered.mean()))
            gain_curve.append(gain)

    sim = 1.0 - D
    coverage = float(sim[:, sel_idx].max(axis=1).mean()) if sel_idx else 0.0
    if len(sel_idx) > 1:
        subD = D[np.ix_(sel_idx, sel_idx)]
        tri = subD[np.triu_indices(len(sel_idx), 1)]
        diversity = float(tri.mean()) if tri.size else 0.0
    else:
        diversity = 0.0


    selected_simple = []
    for rank, i in enumerate(sel_idx, 1):
        row = meta_rows[i]
        selected_simple.append({
            "rank": rank,
            "global_fold_id": row.get("global_fold_id"),
            "ticker": row.get("ticker", "UNK"),
            "date_min": row.get("date_min", ""),
            "date_max": row.get("date_max", "")
        })


    result = {
        "mode": "adaptive" if args.adaptive_k else "fixed",
        "k": len(sel_idx),
        "metrics": {
            "coverage_mean": coverage,
            "diversity_mean": diversity
        },
        "curves": {
            "coverage": coverage_curve,
            "gain": gain_curve
        },
        "selected_folds": selected_simple,
        "params": {
            "lambda_overlap": float(args.lambda_overlap),
            "gamma_same_ticker": float(args.gamma_same_ticker),
            "hard_overlap": bool(args.hard_overlap),
            "max_overlap": float(args.max_overlap),
            "per_ticker_max": int(args.per_ticker_max),
            "k_fixed": int(args.k),
            "k_min": int(args.k_min),
            "k_max": int(args.k_max),
            "coverage_target": float(args.coverage_target),
            "min_gain": float(args.min_gain),
            "elbow_tol": float(args.elbow_tol)
        }
    }

    # output path
    if not args.out_selected_json:
        base = "selected_lstm.json" if args.model_type.lower() == "lstm" else "selected_arima.json"
        out_path = os.path.join(LSTM_META_DIR if args.model_type.lower()=="lstm" else ARIMA_META_DIR, base)
    else:
        out_path = args.out_selected_json

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved selected folds -> {out_path}")

    if args.out_report_json:
        with open(args.out_report_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved diagnostics -> {args.out_report_json}")

if __name__ == "__main__":
    main()