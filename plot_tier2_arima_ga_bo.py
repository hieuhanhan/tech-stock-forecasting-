import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Utils ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_df_or_empty(path: Path, required_cols=None):
    if not path.exists():
        return pd.DataFrame(columns=required_cols or [])
    df = pd.read_csv(path)
    if required_cols:
        for c in required_cols:
            if c not in df.columns:
                df[c] = np.nan
    return df

def knee_index(F: np.ndarray) -> int:
    """F: array of shape (n,2), objectives are min-min: [ -Sharpe, MDD ]"""
    if F.size == 0:
        return -1
    # normalize cols to [0,1]
    f = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    d = np.sqrt((f**2).sum(axis=1))
    return int(np.argmin(d))

def plot_front_scatter(ax, df_ga, df_union, title):
    """
    df_ga, df_union: must have columns ['sharpe','mdd']
    GA plotted in one color, GA+BO in another.
    """
    # GA-only
    if len(df_ga):
        ax.scatter(df_ga["mdd"], df_ga["sharpe"], s=22, alpha=0.85, label="GA front")
    # Union
    if len(df_union):
        ax.scatter(df_union["mdd"], df_union["sharpe"], s=22, alpha=0.85, marker="^", label="GA+BO front")

        # knee on union
        F_union = np.c_[ -df_union["sharpe"].to_numpy(float), df_union["mdd"].to_numpy(float) ]
        kidx = knee_index(F_union)
        if kidx >= 0:
            krow = df_union.iloc[kidx]
            ax.scatter([krow["mdd"]], [krow["sharpe"]], s=80, marker="*", label="knee (union)")

    ax.set_xlabel("MDD ↓")
    ax.set_ylabel("Sharpe ↑")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

def plot_hv_curves(ax, hv_df_subset, title):
    """Plot HV per generation for GA; and add two points for final_ga & final_union if present."""
    hv_ga = hv_df_subset[hv_df_subset["stage"] == "GA"].sort_values("gen")
    if len(hv_ga):
        ax.plot(hv_ga["gen"], hv_ga["hv"], linewidth=1.5, label="HV (GA per-gen)")

    # final points
    fin_ga = hv_df_subset[hv_df_subset["stage"] == "final_ga"]
    fin_union = hv_df_subset[hv_df_subset["stage"] == "final_union"]

    if len(fin_ga):
        ax.scatter([fin_ga["gen"].iloc[0] if "gen" in fin_ga else -1],
                   [fin_ga["hv"].iloc[0]], marker="o", s=60, label="HV final (GA)")
    if len(fin_union):
        ax.scatter([fin_union["gen"].iloc[0] if "gen" in fin_union else -1],
                   [fin_union["hv"].iloc[0]], marker="^", s=60, label="HV final (GA+BO)")

        # annotate improvement if both available
        if len(fin_ga) and len(fin_union):
            delta = float(fin_union["hv"].iloc[0]) - float(fin_ga["hv"].iloc[0])
            ax.annotate(f"ΔHV={delta:.4g}",
                        xy=(fin_union["gen"].iloc[0] if "gen" in fin_union else -1,
                            fin_union["hv"].iloc[0]),
                        xytext=(10, 10), textcoords="offset points")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume (↑)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot Tier-2 GA vs GA+BO fronts & hypervolume.")
    ap.add_argument("--tier2-csv", required=True,
                    help="Main GA/BO log CSV (e.g., data/tuning_results/csv/tier2_arima_cont_gabo.csv)")
    ap.add_argument("--hv-csv", default=None,
                    help="Hypervolume CSV (default: replace tier2-csv name with *_hv.csv)")
    ap.add_argument("--front-csv", default=None,
                    help="Front CSV (default: replace tier2-csv name with *_front.csv)")
    ap.add_argument("--outdir", default="data/figures/arima",
                    help="Output directory for figures")
    ap.add_argument("--per-fold", action="store_true",
                    help="Also produce per-fold×interval plots")
    args = ap.parse_args()

    tier2_csv = Path(args.tier2_csv)
    hv_csv = Path(args.hv_csv) if args.hv_csv else tier2_csv.with_name(tier2_csv.stem + "_hv.csv")
    front_csv = Path(args.front_csv) if args.front_csv else tier2_csv.with_name(tier2_csv.stem + "_front.csv")
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load data
    # tier2 main (stage="GA"/"BO" rows) – not strictly needed for fronts, but you can use it if muốn
    df_all = load_df_or_empty(tier2_csv)
    # Hypervolume history + finals
    hv_df = load_df_or_empty(hv_csv, required_cols=["fold_id","retrain_interval","stage","gen","hv","ref0","ref1","n_front"])
    # Fronts
    fr_df = load_df_or_empty(front_csv, required_cols=["fold_id","retrain_interval","front_type","p","q","threshold","sharpe","mdd"])

    # -------- Global plots (all folds pooled) --------
    # Scatter of fronts (global)
    fig, ax = plt.subplots(figsize=(6, 4))
    df_ga_front = fr_df[fr_df["front_type"] == "GA"]
    df_union_front = fr_df[fr_df["front_type"] == "GA+BO"]
    plot_front_scatter(ax, df_ga_front, df_union_front, "Pareto Fronts (All folds)")
    fig.tight_layout()
    fig.savefig(outdir / "arima_fronts_all_folds.png", dpi=160)

    # Hypervolume summary: chỉ có ý nghĩa theo fold×interval, nhưng ta cũng có thể vẽ phân phối
    # Ở đây: bảng so sánh cuối cùng (per fold×interval)
    if len(hv_df):
        piv = []
        for (fid, interval), g in hv_df.groupby(["fold_id","retrain_interval"]):
            rec = {"fold_id": fid, "retrain_interval": interval,
                   "HV_final_GA": np.nan, "HV_final_UNION": np.nan}
            g_ga = g[g["stage"]=="final_ga"]
            g_union = g[g["stage"]=="final_union"]
            if len(g_ga): rec["HV_final_GA"] = float(g_ga["hv"].iloc[0])
            if len(g_union): rec["HV_final_UNION"] = float(g_union["hv"].iloc[0])
            piv.append(rec)
        df_hv_fin = pd.DataFrame(piv)
        df_hv_fin["Delta"] = df_hv_fin["HV_final_UNION"] - df_hv_fin["HV_final_GA"]
        df_hv_fin.sort_values(["Delta"], ascending=False, inplace=True)
        df_hv_fin.to_csv(outdir / "arima_hv_final_summary.csv", index=False)

    # -------- Per-fold×interval plots --------
    if args.per_fold:
        for (fid, interval), g_front in fr_df.groupby(["fold_id", "retrain_interval"]):
            fig, ax = plt.subplots(figsize=(6, 4))
            df_ga = g_front[g_front["front_type"]=="GA"]
            df_union = g_front[g_front["front_type"]=="GA+BO"]
            plot_front_scatter(ax, df_ga, df_union,
                               f"Pareto Front – fold {fid}, interval {interval}")
            fig.tight_layout()
            fig.savefig(outdir / f"front_fold{fid}_int{interval}.png", dpi=160)

        for (fid, interval), g_hv in hv_df.groupby(["fold_id", "retrain_interval"]):
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_hv_curves(ax, g_hv, f"Hypervolume – fold {fid}, interval {interval}")
            fig.tight_layout()
            fig.savefig(outdir / f"hv_fold{fid}_int{interval}.png", dpi=160)

    print(f"Done. Figures saved to: {outdir.resolve()}")
    if len(hv_df):
        print(f"HV summary CSV: {(outdir / 'hv_final_summary.csv').resolve()}")

if __name__ == "__main__":
    main()