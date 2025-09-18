# plot_tier2.py
# One script for Tier-2 GA vs GA+BO: works for ARIMA, LSTM, ...
# Usage examples at bottom.

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== helpers ==========
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_df_or_empty(path: Path, required_cols=None):
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=required_cols or [])
    df = pd.read_csv(path)
    if required_cols:
        for c in required_cols:
            if c not in df.columns:
                df[c] = np.nan
    return df

def infer_label_from_path(p: Path) -> str:
    s = p.stem.lower()
    if "lstm" in s: return "LSTM"
    if "arima" in s: return "ARIMA"
    return "Model"

def knee_index(F: np.ndarray) -> int:
    """F: array of shape (n,2), objectives are min-min: [-Sharpe, MDD]."""
    if F.size == 0:
        return -1
    f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    d = np.sqrt((f**2).sum(axis=1))
    return int(np.argmin(d))

def plot_front_scatter(ax, df_ga, df_union, title):
    """df_* must have columns ['sharpe','mdd']."""
    if len(df_ga):
        ax.scatter(df_ga["mdd"], df_ga["sharpe"], s=24, alpha=0.85, label="GA front")
    if len(df_union):
        ax.scatter(df_union["mdd"], df_union["sharpe"], s=28, alpha=0.9, marker="^", label="GA+BO front")
        F_union = np.c_[ -df_union["sharpe"].to_numpy(float), df_union["mdd"].to_numpy(float) ]
        kidx = knee_index(F_union)
        if kidx >= 0:
            krow = df_union.iloc[kidx]
            ax.scatter([krow["mdd"]], [krow["sharpe"]], s=80, marker="*", label="Knee (union)")
    ax.set_xlabel("Maximum Drawdown (MDD)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

def plot_hv_curves(ax, hv_df_subset, title):
    """Plot HV per generation (GA), plus final GA/UNION points if available."""
    hv_ga = hv_df_subset[hv_df_subset["stage"].str.lower() == "ga"].sort_values("gen")
    if len(hv_ga):
        ax.plot(hv_ga["gen"], hv_ga["hv"], linewidth=1.5, label="HV (GA per-gen)")

    fin_ga = hv_df_subset[hv_df_subset["stage"].str.lower() == "final_ga"]
    fin_union = hv_df_subset[hv_df_subset["stage"].str.lower() == "final_union"]

    if len(fin_ga):
        ax.scatter(fin_ga["gen"].fillna(-1), fin_ga["hv"], marker="o", s=60, label="HV final (GA)")
    if len(fin_union):
        ax.scatter(fin_union["gen"].fillna(-1), fin_union["hv"], marker="^", s=60, label="HV final (GA+BO)")
        if len(fin_ga):
            delta = float(fin_union["hv"].iloc[0]) - float(fin_ga["hv"].iloc[0])
            ax.annotate(f"ΔHV={delta:.4g}",
                        xy=(fin_union["gen"].fillna(-1).iloc[0], fin_union["hv"].iloc[0]),
                        xytext=(8,10), textcoords="offset points")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume (↑)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

# ========== main ==========
def main():
    ap = argparse.ArgumentParser(description="Tier-2 GA vs GA+BO plots (ARIMA/LSTM).")
    ap.add_argument("--tier2-csv", required=True,
                    help="Main Tier-2 log CSV (generic; optional for plotting).")
    ap.add_argument("--front-csv", default=None,
                    help="Pareto fronts CSV with columns: fold_id,retrain_interval,front_type,sharpe,mdd.")
    ap.add_argument("--hv-csv", default=None,
                    help="Hypervolume CSV with columns: fold_id,retrain_interval,stage,gen,hv,ref0,ref1,n_front.")
    ap.add_argument("--knee-csv", default=None,
                    help="(Optional) knee summary CSV: fold_id,retrain_interval,knee_sharpe,knee_mdd,knee_turnover...")
    ap.add_argument("--label", default=None, help="Label for titles (e.g., ARIMA or LSTM).")
    ap.add_argument("--outdir", default="figures/tier2", help="Output directory.")
    ap.add_argument("--per_fold", action="store_true", help="Also plot per-fold×interval figures.")
    args = ap.parse_args()

    tier2_csv = Path(args.tier2_csv)
    label = args.label or infer_label_from_path(tier2_csv)
    outdir = Path(args.outdir); ensure_dir(outdir)

    # Defaults if user omits front/hv paths
    front_csv = Path(args.front_csv) if args.front_csv else tier2_csv.with_name(tier2_csv.stem.replace("_hv","").replace("_front","") + "_front.csv")
    hv_csv    = Path(args.hv_csv)    if args.hv_csv    else tier2_csv.with_name(tier2_csv.stem.replace("_front","").replace("_hv","") + "_hv.csv")
    knee_csv  = Path(args.knee_csv)  if args.knee_csv  else tier2_csv.with_name(tier2_csv.stem.replace("_front","").replace("_hv","") + "_knee.csv")

    # Load
    fr_df = load_df_or_empty(front_csv, required_cols=["fold_id","retrain_interval","front_type","sharpe","mdd"])
    hv_df = load_df_or_empty(hv_csv,    required_cols=["fold_id","retrain_interval","stage","gen","hv","ref0","ref1","n_front"])
    knee_df = load_df_or_empty(knee_csv)

    # -------- Global fronts --------
    if len(fr_df):
        fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=160)
        df_ga    = fr_df[fr_df["front_type"].str.upper() == "GA"]
        df_union = fr_df[fr_df["front_type"].str.upper().isin(["GA+BO", "UNION", "GA_BO"])]
        plot_front_scatter(ax, df_ga, df_union, f"{label} — Pareto Fronts (All Folds)")
        fig.tight_layout()
        fig.savefig(outdir / f"{label.lower()}_fronts_all.png", dpi=300)

    # -------- Global HV summary table --------
    if len(hv_df):
        recs = []
        for (fid, interval), g in hv_df.groupby(["fold_id","retrain_interval"]):
            row = {"fold_id": fid, "retrain_interval": interval,
                   "HV_final_GA": np.nan, "HV_final_UNION": np.nan}
            ga = g[g["stage"].str.lower()=="final_ga"]
            un = g[g["stage"].str.lower()=="final_union"]
            if len(ga): row["HV_final_GA"] = float(ga["hv"].iloc[0])
            if len(un): row["HV_final_UNION"] = float(un["hv"].iloc[0])
            recs.append(row)
        hv_sum = pd.DataFrame(recs)
        hv_sum["Delta_HV"] = hv_sum["HV_final_UNION"] - hv_sum["HV_final_GA"]
        hv_sum.sort_values(["Delta_HV"], ascending=False, inplace=True)
        hv_sum.to_csv(outdir / f"{label.lower()}_hv_final_summary.csv", index=False)

    # -------- Per fold × interval --------
    if args.per_fold and len(fr_df):
        for (fid, interval), g in fr_df.groupby(["fold_id","retrain_interval"]):
            fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
            df_ga    = g[g["front_type"].str.upper() == "GA"]
            df_union = g[g["front_type"].str.upper().isin(["GA+BO","UNION","GA_BO"])]
            plot_front_scatter(ax, df_ga, df_union,
                               f"{label} — Pareto Front (Fold {fid}, Interval {interval})")
            fig.tight_layout()
            fig.savefig(outdir / f"{label.lower()}_front_fold{fid}_int{interval}.png", dpi=300)

    if args.per_fold and len(hv_df):
        for (fid, interval), g in hv_df.groupby(["fold_id","retrain_interval"]):
            fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
            plot_hv_curves(ax, g, f"{label} — Hypervolume (Fold {fid}, Interval {interval})")
            fig.tight_layout()
            fig.savefig(outdir / f"{label.lower()}_hv_fold{fid}_int{interval}.png", dpi=300)

    # -------- Optional: knee summary clean table --------
    if len(knee_df):
        keep = [c for c in knee_df.columns
                if c.lower() in {"fold_id","retrain_interval","knee_sharpe","knee_mdd","knee_turnover"}]
        if keep:
            knee_tbl = (knee_df[keep]
                        .rename(columns={
                            "fold_id":"Fold","retrain_interval":"Interval",
                            "knee_sharpe":"Sharpe", "knee_mdd":"MDD", "knee_turnover":"Turnover"
                        })
                        .sort_values(["Fold","Interval"]))
            knee_tbl.to_csv(outdir / f"{label.lower()}_knee_summary.csv", index=False)

    print(f"Done. Outputs -> {outdir.resolve()}")

if __name__ == "__main__":
    main()