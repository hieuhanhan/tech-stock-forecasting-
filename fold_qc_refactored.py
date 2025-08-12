import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Config dataclass
@dataclass
class QCConfig:
    model_type: str  # 'lstm' | 'arima'
    min_pos: float = 0.10
    max_pos: float = 0.60
    min_vol: float = 1e-4
    min_folds_per_ticker: int = 2
    use_iqr_outliers: bool = True
    per_ticker_thresholds: bool = False
    figures_dir: str = "data/figures"

# I/O helpers
def load_fold_summary(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"fold_summary_path not found: {path}")
    with open(path, "r") as f:
        folds = json.load(f)
    if not isinstance(folds, list):
        raise ValueError("fold_summary JSON must be a list of fold dicts")
    return folds

def build_meta_cache(model_type: str) -> Dict[str, Tuple[float, float]]:
    """Return {global_fold_id: (positive_ratio, val_vol)} from meta CSVs if present."""
    meta_dir = os.path.join("data", "processed_folds", f"{model_type}_meta")
    cache: Dict[str, Tuple[float, float]] = {}
    if not os.path.isdir(meta_dir):
        return cache
    for fn in os.listdir(meta_dir):
        if not fn.startswith("val_meta_fold_") or not fn.endswith(".csv"):
            continue
        gid = fn.replace("val_meta_fold_", "").replace(".csv", "")
        fpath = os.path.join(meta_dir, fn)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        ratio = float(df["target"].mean()) if "target" in df.columns else np.nan
        vol = float(df["Log_Returns"].std()) if "Log_Returns" in df.columns else np.nan
        cache[gid] = (ratio, vol)
    return cache

def merge_folds_with_meta(folds: List[dict], meta_cache: Dict[str, Tuple[float, float]] ) -> Tuple[pd.DataFrame, List[dict]]:
    """Produce a DataFrame for QC; return (df, dropped_missing)."""
    rows = []
    dropped_missing: List[dict] = []
    for fold in folds:
        gid = fold.get("global_fold_id")
        ticker = fold.get("ticker")
        # Safe get: don't treat 0.0 as falsy
        ratio = fold.get("val_pos_ratio")
        vol = fold.get("val_vol")
        if ratio is None or vol is None:
            # Try cache
            if gid in meta_cache:
                cached_ratio, cached_vol = meta_cache[gid]
                if ratio is None:
                    ratio = cached_ratio
                if vol is None:
                    vol = cached_vol
        if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
            dropped_missing.append({**fold, "drop_reason": ["missing_target_or_meta"]})
            continue
        # val_vol may stay NaN; we handle later consistently
        rows.append({
            "fold_id": gid,
            "ticker": ticker,
            "positive_ratio": float(ratio),
            "val_vol": (float(vol) if (vol is not None and not pd.isna(vol)) else np.nan)
        })
    if not rows:
        return pd.DataFrame(columns=["fold_id","ticker","positive_ratio","val_vol"]), dropped_missing
    df = pd.DataFrame(rows)
    return df, dropped_missing

# Thresholds & rules
def compute_thresholds(df: pd.DataFrame, cfg: QCConfig) -> pd.DataFrame:
    """Return thresholds per ticker if per_ticker_thresholds else global constants.
    Output columns: ticker, min_pos, max_pos, min_vol
    """
    if not cfg.per_ticker_thresholds or df.empty:
        base = pd.DataFrame({
            "ticker": df["ticker"].unique() if not df.empty else [],
            "min_pos": cfg.min_pos,
            "max_pos": cfg.max_pos,
            "min_vol": cfg.min_vol,
        })
        return base

# Per-ticker dynamic bounds using percentiles; fall back to globals for tiny samples
    recs = []
    for tkr, grp in df.groupby("ticker"):
        n = len(grp)
        if n < max(20, cfg.min_folds_per_ticker * 3):
            recs.append({"ticker": tkr, "min_pos": cfg.min_pos, "max_pos": cfg.max_pos, "min_vol": cfg.min_vol})
            continue
        p10, p90 = grp["positive_ratio"].quantile([0.10, 0.90])
        vol_p10 = grp["val_vol"].quantile(0.10)
        # Slightly widen to avoid overfitting thresholds
        min_pos = float(max(cfg.min_pos, p10 - 0.02))
        max_pos = float(min(cfg.max_pos, p90 + 0.02))
        min_vol = float(max(cfg.min_vol, vol_p10 if not pd.isna(vol_p10) else cfg.min_vol))
        recs.append({"ticker": tkr, "min_pos": min_pos, "max_pos": max_pos, "min_vol": min_vol})
    return pd.DataFrame(recs)

def split_rule_ok_bad(df: pd.DataFrame, thr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = df.merge(thr, on="ticker", how="left")
    # Consistent NA handling
    vol_filled = merged["val_vol"].fillna(-np.inf)  # ensure NA fails vol rule
    ok_mask = (
        merged["positive_ratio"].between(merged["min_pos"], merged["max_pos"], inclusive="both") &
        (vol_filled >= merged["min_vol"])
    )
    return merged[ok_mask].copy(), merged[~ok_mask].copy()


def apply_iqr_outliers(df_ok: pd.DataFrame, per_ticker: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_ok.empty:
        return df_ok.copy(), df_ok.head(0).copy()

    def _iqr_mask(series: pd.Series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (series >= lo) & (series <= hi)

    if per_ticker:
        parts = []
        outliers = []
        for tkr, grp in df_ok.groupby("ticker"):
            mask = _iqr_mask(grp["positive_ratio"])  # ratio-focused IQR
            parts.append(grp[mask])
            outliers.append(grp[~mask])
        kept = pd.concat(parts, axis=0) if parts else df_ok.head(0)
        out = pd.concat(outliers, axis=0) if outliers else df_ok.head(0)
        return kept, out
    else:
        mask = _iqr_mask(df_ok["positive_ratio"])
        return df_ok[mask].copy(), df_ok[~mask].copy()

# Restore & weights
def restore_min_per_ticker(df_keep: pd.DataFrame, df_rule_ok: pd.DataFrame, cfg: QCConfig) -> Tuple[pd.DataFrame, List[Tuple[str, List[str]]]]:
    restored: List[Tuple[str, List[str]]] = []
    keep = df_keep.copy()

    if df_rule_ok.empty:
        return keep, restored

    med_ratio = 0.5
    med_vol = float(df_rule_ok["val_vol"].median()) if not df_rule_ok["val_vol"].isna().all() else 0.0

    for tkr, _ in df_rule_ok.groupby("ticker"):
        have = (keep["ticker"] == tkr).sum()
        need = max(0, cfg.min_folds_per_ticker - have)
        if need <= 0:
            continue
        pool = df_rule_ok[df_rule_ok["ticker"] == tkr].copy()
        if pool.empty:
            continue
        # Multi-criteria distance (ratio closeness to 0.5 + vol closeness to med)
        pool["dist"] = (
            0.7 * (pool["positive_ratio"] - med_ratio).abs() / 0.5 +
            0.3 * ( (pool["val_vol"] - med_vol).abs() / (abs(med_vol) + 1e-9) ).fillna(1.0)
        )
        add_back = pool.sort_values("dist").head(need)
        if not add_back.empty:
            keep = pd.concat([keep, add_back.drop(columns=["dist"])], axis=0)
            keep = keep.drop_duplicates(subset=["fold_id"], keep="first")
            restored.append((tkr, add_back["fold_id"].tolist()))
    return keep, restored


def assign_weights(df_keep: pd.DataFrame) -> pd.DataFrame:
    if df_keep.empty:
        df_keep["weight"] = []
        return df_keep

    # Softer penalty around 0.5 + volatility moderation
    w_ratio = 1.0 - (df_keep["positive_ratio"] - 0.5).abs() * 1.2
    w_ratio = w_ratio.clip(0.2, 1.0)

    med_vol = float(df_keep["val_vol"].median()) if not df_keep["val_vol"].isna().all() else 0.0
    w_vol = 1.0 - ( (df_keep["val_vol"] - med_vol).abs() / (abs(med_vol) + 1e-9) ).clip(0, 1)
    w = (0.7 * w_ratio + 0.3 * w_vol).clip(0.2, 1.0)

    out = df_keep.copy()
    out["weight"] = w.astype(float)
    return out

# Reasons & outputs
def reason_for(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    r, v, min_pos, max_pos, min_vol = row["positive_ratio"], row["val_vol"], row["min_pos"], row["max_pos"], row["min_vol"]
    if (r < min_pos) or (r > max_pos):
        reasons.append("ratio_out_of_range")
    if (v is None) or (pd.isna(v)) or (v < min_vol):
        reasons.append("low_vol_or_missing")
    return reasons or ["other_rule_fail"]


def build_drop_list(folds: List[dict], kept_ids: set, df_rule_bad: pd.DataFrame, df_out_iqr: pd.DataFrame, dropped_missing: List[dict]) -> List[dict]:
    bad_ids = set(df_rule_bad["fold_id"]) if not df_rule_bad.empty else set()
    iqr_ids = set(df_out_iqr["fold_id"]) if not df_out_iqr.empty else set()

    dropped: List[dict] = list(dropped_missing)  # start with earlier missing
    bad_map = {rid: row for rid, row in df_rule_bad.set_index("fold_id").iterrows()} if not df_rule_bad.empty else {}

    for fold in folds:
        gid = fold.get("global_fold_id")
        if gid in kept_ids:
            continue
        if gid in bad_ids:
            row = bad_map[gid]
            dropped.append({**fold, "drop_reason": reason_for(row)})
        elif gid in iqr_ids:
            dropped.append({**fold, "drop_reason": ["iqr_outlier"]})
        else:
            dropped.append({**fold, "drop_reason": ["unknown"]})
    return dropped


def save_outputs(base_json_path: str, kept_records: List[dict], dropped_records: List[dict], qc_df: pd.DataFrame) -> None:
    out_clean = base_json_path.replace(".json", "_cleaned.json")
    out_drop = base_json_path.replace(".json", "_dropped.json")
    qc_path = base_json_path.replace(".json", "_qc_summary.csv")

    with open(out_clean, "w") as f:
        json.dump(kept_records, f, indent=4)
    with open(out_drop, "w") as f:
        json.dump(dropped_records, f, indent=4)
    qc_df.to_csv(qc_path, index=False)

    print(f"[INFO] Saved cleaned: {out_clean}")
    print(f"[INFO] Saved dropped: {out_drop}")
    print(f"[INFO] Saved QC CSV: {qc_path}")

# Plots
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_qc_with_suffix(df_all: pd.DataFrame, cfg: QCConfig, suffix: str = "all") -> None:
    if df_all.empty:
        print("[PLOT] No data to plot.")
        return
    ensure_dir(cfg.figures_dir)

    # Histogram of positive_ratio
    plt.figure(figsize=(8, 5))
    plt.hist(df_all["positive_ratio"], bins=15, edgecolor="black")
    plt.title(f"{cfg.model_type.upper()} – Histogram of Positive Label Ratio ({suffix})")
    plt.xlabel("Positive Label Ratio")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    hist_path = os.path.join(cfg.figures_dir, f"{cfg.model_type}_positive_ratio_hist_{suffix}.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved histogram to: {hist_path}")

    # Boxplot by ticker (ordered by median)
    order = df_all.groupby("ticker")["positive_ratio"].median().sort_values().index.tolist()
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_all, x="ticker", y="positive_ratio", order=order)
    plt.xticks(rotation=90)
    plt.title(f"{cfg.model_type.upper()} – Boxplot of Positive Label Ratio per Ticker ({suffix})")
    plt.ylabel("Positive Label Ratio")
    plt.xlabel("Ticker")
    plt.grid(True)
    plt.tight_layout()
    boxplot_path = os.path.join(cfg.figures_dir, f"{cfg.model_type}_positive_ratio_boxplot_{suffix}.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved boxplot to: {boxplot_path}")

    # Scatter ratio vs vol (mark NA separately)
    plt.figure(figsize=(8, 6))
    has_vol = ~df_all["val_vol"].isna()
    plt.scatter(df_all.loc[has_vol, "positive_ratio"], df_all.loc[has_vol, "val_vol"], s=12, alpha=0.7, label="with vol")
    if (~has_vol).any():
        plt.scatter(df_all.loc[~has_vol, "positive_ratio"], [0] * (~has_vol).sum(), s=14, alpha=0.7, label="vol NA")
    plt.title(f"{cfg.model_type.upper()} – Ratio vs Volatility (val) ({suffix})")
    plt.xlabel("Positive Label Ratio")
    plt.ylabel("Val Volatility (std)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    sc_path = os.path.join(cfg.figures_dir, f"{cfg.model_type}_ratio_vs_vol_scatter_{suffix}.png")
    plt.savefig(sc_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved scatter to: {sc_path}")

# Main
def main():
    ap = argparse.ArgumentParser(description="QC & filter folds with optional per-ticker thresholds")
    ap.add_argument("--fold_summary_path", type=str, required=True)
    ap.add_argument("--model_type", type=str, required=True, choices=["lstm","arima"]) 
    ap.add_argument("--min_pos", type=float, default=0.10)
    ap.add_argument("--max_pos", type=float, default=0.60)
    ap.add_argument("--min_vol", type=float, default=1e-4)
    ap.add_argument("--min_folds_per_ticker", type=int, default=2)
    ap.add_argument("--use_iqr_outliers", action="store_true", help="apply IQR filtering on positive_ratio")
    ap.add_argument("--per_ticker_thresholds", action="store_true", help="use percentile-based per-ticker thresholds")
    ap.add_argument("--figures_dir", type=str, default="data/figures")

    args = ap.parse_args()
    cfg = QCConfig(
        model_type=args.model_type,
        min_pos=args.min_pos,
        max_pos=args.max_pos,
        min_vol=args.min_vol,
        min_folds_per_ticker=args.min_folds_per_ticker,
        use_iqr_outliers=args.use_iqr_outliers,
        per_ticker_thresholds=args.per_ticker_thresholds,
        figures_dir=args.figures_dir,
    )

    folds = load_fold_summary(args.fold_summary_path)
    meta_cache = build_meta_cache(cfg.model_type)
    df, dropped_missing = merge_folds_with_meta(folds, meta_cache)

    if df.empty:
        print("[WARN] No valid folds found to analyze.")
        return

    # Compute thresholds & rule split
    thr = compute_thresholds(df, cfg)
    df_rule_ok, df_rule_bad = split_rule_ok_bad(df, thr)

    # IQR-based outlier removal on OK subset
    if cfg.use_iqr_outliers:
        df_keep, df_out_iqr = apply_iqr_outliers(df_rule_ok, per_ticker=True)
    else:
        df_keep, df_out_iqr = df_rule_ok.copy(), df_rule_ok.head(0).copy()

    # Restore min per ticker
    df_keep_restored, restored = restore_min_per_ticker(df_keep, df_rule_ok, cfg)

    # Assign weights
    df_keep_final = assign_weights(df_keep_restored)

    kept_ids = set(df_keep_final["fold_id"].tolist())

    # Build dropped list with reasons
    dropped_records = build_drop_list(
        folds=folds,
        kept_ids=kept_ids,
        df_rule_bad=df_rule_bad,
        df_out_iqr=df_out_iqr,
        dropped_missing=dropped_missing,
    )

    # Print summary
    print(f"Total folds: {len(df)}")
    print(f"Rule-out: {len(df_rule_bad)}")
    if cfg.use_iqr_outliers:
        print(f"IQR-out among rule-OK: {len(df_out_iqr)}")
    if restored:
        print(f"Restored folds per ticker: {restored}")
    print(f"Remaining (kept): {len(kept_ids)}")

    # Build kept records aligned to original folds + weight
    kept_map = {rid: w for rid, w in df_keep_final.set_index("fold_id")["weight"].to_dict().items()}
    kept_records: List[dict] = []
    for fold in folds:
        gid = fold.get("global_fold_id")
        if gid in kept_map:
            rec = dict(fold)
            rec["weight"] = float(kept_map[gid])
            kept_records.append(rec)

    # QC CSV (mark kept)
    qc_df = df.copy()
    qc_df["kept"] = qc_df["fold_id"].isin(kept_ids).astype(int)

    # Save outputs
    save_outputs(args.fold_summary_path, kept_records, dropped_records, qc_df)

    # Plots: before & after
    plot_qc_with_suffix(df, cfg, suffix="before")         
    plot_qc_with_suffix(df_keep_final, cfg, suffix="after_kept")  


if __name__ == "__main__":
    main()