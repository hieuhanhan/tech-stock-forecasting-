#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from rep_utils import (
    OUTPUT_BASE_DIR, ARIMA_META_DIR, LSTM_META_DIR,
    standardize_columns, cosine_similarity_matrix, build_penalized_dissimilarity,
    run_fixed_k_multistart, adaptive_facility_location, one_swap_improve,
    coverage_stats
)

FINAL_DIR_ARIMA = os.path.join(OUTPUT_BASE_DIR, "final", "arima")
FINAL_DIR_LSTM  = os.path.join(OUTPUT_BASE_DIR, "final", "lstm")
os.makedirs(FINAL_DIR_ARIMA, exist_ok=True)
os.makedirs(FINAL_DIR_LSTM,  exist_ok=True)


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _enrich_selected_with_dates(meta_rows: List[dict],
                                sel_idx: List[int],
                                folds_summary_path: str) -> List[dict]:
    """Trả về selected_folds [{rank, global_fold_id, ticker, date_min, date_max}] như cũ,
    nhưng ưu tiên lấy date_min/date_max từ chính meta nếu có; nếu thiếu thì fallback
    vào CSV trong folds_summary."""
    selected_simple = []
    summary_map = {}
    if folds_summary_path and os.path.exists(folds_summary_path):
        try:
            folds = _load_json(folds_summary_path)
            for rec in folds:
                gid = rec.get("global_fold_id")
                if gid is not None:
                    summary_map[gid] = rec
        except Exception as e:
            print(f"[WARN] Failed to load folds summary for date fallback: {e}")

    def _infer_dates(row: dict) -> Tuple[str, str]:
        dm = row.get("date_min")
        dx = row.get("date_max")
        if dm and dx:
            return str(dm), str(dx)

        # Fallback: đọc từ CSV trong summary nếu có
        def _try_csv(rel_path: Optional[str]) -> Optional[Tuple[str, str]]:
            if not rel_path:
                return None
            csv_path = os.path.join(OUTPUT_BASE_DIR, rel_path)
            if not os.path.exists(csv_path):
                return None
            try:
                df = pd.read_csv(csv_path)
                for col in ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"]:
                    if col in df.columns:
                        s = pd.to_datetime(df[col], errors="coerce").dropna()
                        if not s.empty:
                            return str(s.min().date()), str(s.max().date())
            except Exception as e:
                print(f"[WARN] Read dates failed at {rel_path}: {e}")
            return None

        gid = row.get("global_fold_id")
        if gid in summary_map:
            entry = summary_map[gid]
            for key in [
                "val_meta_path_arima", "val_meta_path_lstm",
                "val_path_arima", "val_path_lstm",
                "train_path_arima", "train_path_lstm",
            ]:
                rel = entry.get(key)
                got = _try_csv(rel)
                if got is not None:
                    return got
        return "", ""

    for rank, i in enumerate(sel_idx, 1):
        row = meta_rows[i]
        dmin, dmax = _infer_dates(row)
        selected_simple.append({
            "rank": rank,
            "global_fold_id": row.get("global_fold_id"),
            "ticker": row.get("ticker", "UNK"),
            "date_min": dmin,
            "date_max": dmax
        })
    return selected_simple


def _build_tuning_folds(selected_simple: List[dict],
                        folds_summary_path: str) -> List[dict]:
    """Map selected_folds -> full records từ folds_summary_{model}.json để tạo tuning_folds."""
    if not folds_summary_path or not os.path.exists(folds_summary_path):
        print(f"[WARN] folds_summary_path missing or not found: {folds_summary_path}")
        return []

    folds_summary = _load_json(folds_summary_path)
    summary_map = {rec.get("global_fold_id"): rec for rec in folds_summary}
    tuning_folds = []
    missing = 0

    for item in selected_simple:
        gid = item.get("global_fold_id")
        rec = summary_map.get(gid)
        if rec is not None:
            tuning_folds.append(rec)
        else:
            missing += 1
            print(f"[WARN] global_fold_id {gid} not found in folds_summary.")
    if missing:
        print(f"[INFO] Tuning folds: {missing} ids were missing in folds_summary (skipped).")
    return tuning_folds


def main():
    p = argparse.ArgumentParser(description="Select representative folds and emit tuning_folds for ARIMA/LSTM.")
    p.add_argument("--model-type", required=True, choices=["lstm", "arima"])
    p.add_argument("--meta-json-path", required=True, help="Meta-features JSON (from build_meta_features)")
    p.add_argument("--folds-summary-path", type=str, required=True, help="Path to folds_summary_{model}.json for enriching & tuning_folds")

    # evaluation mode
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

    # quality/speed knobs
    p.add_argument("--multistart", type=int, default=1)
    p.add_argument("--tie_break", choices=["diversity", "random", "first"], default="diversity")
    p.add_argument("--local_search", action="store_true")
    p.add_argument("--swap_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)

    # outputs
    p.add_argument("--out_selected_json", type=str, default="")
    p.add_argument("--out_report_json", type=str, default="")
    # ✨ NEW: xuất luôn tuning_folds
    p.add_argument("--tuning_out_path", type=str, default="")
    args = p.parse_args()

    # load meta
    meta_rows = _load_json(args.meta_json_path)
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
            sel, stats = run_fixed_k_multistart(
                D, meta_df, k,
                bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
                tie_break=args.tie_break, multistart=max(1, int(args.multistart)), base_seed=args.seed,
                local_search=bool(args.local_search), swap_iters=int(args.swap_iters)
            )
            diversity = 0.0
            if len(sel) > 1:
                subD = D[np.ix_(sel, sel)]
                tri = subD[np.triu_indices(len(sel), 1)]
                diversity = float(tri.mean()) if tri.size else 0.0
            print(
                f"k={k:>3} | cov_mean={stats['mean']:.3f} min={stats['min']:.3f} p05={stats['p05']:.3f} p95={stats['p95']:.3f} | "
                f"diversity_mean={diversity:.3f} | selected={len(sel)}"
            )
        return

    # selection
    if args.adaptive_k:
        rng = np.random.default_rng(args.seed)
        res = adaptive_facility_location(
            D, meta_df,
            int(args.k_min), int(args.k_max),
            bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
            float(args.coverage_target), float(args.min_gain), float(args.elbow_tol),
            tie_break=args.tie_break, rng=rng
        )
        sel_idx = res["indices"]
        coverage_curve = res["coverage_curve"]
        gain_curve = res["gain_curve"]
        if args.local_search and sel_idx:
            sel_idx, _ = one_swap_improve(
                D, meta_df, sel_idx,
                int(args.per_ticker_max), bool(args.hard_overlap), float(args.max_overlap),
                int(args.swap_iters), rng
            )
    else:
        sel_idx, stats_best = run_fixed_k_multistart(
            D, meta_df, int(args.k),
            bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
            tie_break=args.tie_break, multistart=max(1, int(args.multistart)), base_seed=args.seed,
            local_search=bool(args.local_search), swap_iters=int(args.swap_iters)
        )
        # build curves from fixed picks
        sim = 1.0 - D
        coverage_curve, gain_curve = [], []
        covered = np.zeros(D.shape[0], dtype=np.float32)
        for idx in sel_idx:
            gain = float(np.maximum(covered, sim[:, idx]).sum() - covered.sum())
            covered = np.maximum(covered, sim[:, idx])
            coverage_curve.append(float(covered.mean()))
            gain_curve.append(gain)

    # metrics
    sim = 1.0 - D
    cov_stats = coverage_stats(sim, sel_idx)
    if len(sel_idx) > 1:
        subD = D[np.ix_(sel_idx, sel_idx)]
        tri = subD[np.triu_indices(len(sel_idx), 1)]
        diversity = float(tri.mean()) if tri.size else 0.0
    else:
        diversity = 0.0

    # selected list with date range (human-friendly)
    selected_simple = _enrich_selected_with_dates(meta_rows, sel_idx, args.folds_summary_path)
    mapped = sum(1 for r in selected_simple if r["date_min"] and r["date_max"])
    print(f"[INFO] Dates mapped for {mapped}/{len(selected_simple)} selected folds")

    result = {
        "mode": "adaptive" if args.adaptive_k else "fixed",
        "k": len(sel_idx),
        "metrics": {
            "coverage_mean": cov_stats["mean"],
            "coverage_min":  cov_stats["min"],
            "coverage_p05":  cov_stats["p05"],
            "coverage_p25":  cov_stats["p25"],
            "coverage_p50":  cov_stats["p50"],
            "coverage_p95":  cov_stats["p95"],
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
            "elbow_tol": float(args.elbow_tol),
            "multistart": int(args.multistart),
            "tie_break": args.tie_break,
            "local_search": bool(args.local_search),
            "swap_iters": int(args.swap_iters),
            "seed": args.seed,
        }
    }

    # paths
    is_lstm = args.model_type.lower() == "lstm"
    selected_default = os.path.join(LSTM_META_DIR if is_lstm else ARIMA_META_DIR,
                                    "selected_lstm.json" if is_lstm else "selected_arima.json")
    out_selected_json = args.out_selected_json or selected_default
    _save_json(result, out_selected_json)
    print(f"[INFO] Saved selected folds -> {out_selected_json}")

    if args.out_report_json:
        _save_json(result, args.out_report_json)
        print(f"[INFO] Saved diagnostics -> {args.out_report_json}")

    # ✨ NEW: emit tuning_folds_{model}.json (full records from folds_summary)
    tuning_default = (os.path.join(FINAL_DIR_LSTM, "lstm_tuning_folds.json")
                      if is_lstm else os.path.join(FINAL_DIR_ARIMA, "arima_tuning_folds.json"))
    tuning_out_path = args.tuning_out_path or tuning_default

    tuning_folds = _build_tuning_folds(result["selected_folds"], args.folds_summary_path)
    payload = tuning_folds
    # Nếu muốn bọc theo key, bật 1 trong 2 dòng sau:
    # payload = { "lstm": tuning_folds } if is_lstm else { "arima": tuning_folds }

    _save_json(payload, tuning_out_path)
    print(f"[INFO] Saved tuning folds -> {tuning_out_path} | rows={len(tuning_folds)}")


if __name__ == "__main__":
    main()