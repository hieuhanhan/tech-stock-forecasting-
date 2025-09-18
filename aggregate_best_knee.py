# best_vs_knee_all_folds.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths (adjust if needed)
# -----------------------------
ARIMA_FRONTS = "data/tuning_results/csv/tier2_arima_front.csv"
ARIMA_KNEE   = "data/tuning_results/csv/tier2_arima_knee.csv"
LSTM_FRONTS  = "data/tuning_results/csv/tier2_lstm_front.csv"
LSTM_KNEE    = "data/tuning_results/csv/tier2_lstm_knee.csv"
OUTDIR       = Path("figures")

OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Helper: sanitize/standardize columns
# ---------------------------------------
def _infer_source_from_pick(val: str) -> str:
    """Map 'pick' values to canonical sources."""
    if val is None:
        return "GA"
    s = str(val).strip().upper()
    # strip knee suffixes and normalize
    s = s.replace("_KNEE", "").replace("KNEE", "")
    s = s.replace("GA_BO", "GA+BO").replace("GA-BO", "GA+BO").replace("UNION", "GA+BO")
    # decide
    if "GA+BO" in s or "BO" in s:
        return "GA+BO"
    if "GA" in s:
        return "GA"
    return "GA"  # sensible default


def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_lc = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols_lc:
                return cols_lc[n]
        return None

    # Required cores
    c_fold   = pick("fold_id")
    c_int    = pick("retrain_interval","interval")
    c_sharpe = pick("sharpe","sr")
    c_mdd    = pick("mdd","max_drawdown","drawdown")

    if any(x is None for x in [c_fold, c_int, c_sharpe, c_mdd]):
        raise ValueError(f"Missing required columns. Seen: {list(df.columns)}")

    out = pd.DataFrame({
        "fold_id": df[c_fold].astype(int),
        "retrain_interval": df[c_int].astype(int),
        "sharpe": pd.to_numeric(df[c_sharpe], errors="coerce"),
        "mdd": pd.to_numeric(df[c_mdd], errors="coerce"),
    })

    # Prefer explicit source/front_type if present
    c_src  = pick("source","front_type")
    c_pick = pick("pick")

    if c_src is not None:
        src = df[c_src].astype(str)
    elif c_pick is not None:
        src = df[c_pick].map(_infer_source_from_pick)
    else:
        # fallback: everything labeled GA
        src = pd.Series(["GA"] * len(df), index=df.index)

    out["source"] = (
        src.astype(str)
           .str.upper()
           .str.replace("GA\\+BO", "GA+BO", regex=True)
           .str.replace("GA_BO", "GA+BO", regex=True)
           .str.replace("GA-BO", "GA+BO", regex=True)
    )

    out = out.dropna(subset=["sharpe","mdd"])
    return out


def load_best_and_knee(fronts_path: str, knee_path: str) -> pd.DataFrame:
    fronts = pd.read_csv(fronts_path)
    knee   = pd.read_csv(knee_path)

    fronts = standardize_cols(fronts)
    knee   = standardize_cols(knee)

    # BEST = max Sharpe per (fold, interval, source) from the fronts
    best = (
        fronts.sort_values(
            ["fold_id","retrain_interval","source","sharpe"],
            ascending=[True, True, True, False]
        )
        .groupby(["fold_id","retrain_interval","source"], as_index=False)
        .first()
        .assign(kind="Best")
    )

    knee = knee.assign(kind="Knee")

    use_cols = ["fold_id","retrain_interval","source","sharpe","mdd","kind"]
    both = pd.concat([best[use_cols], knee[use_cols]], ignore_index=True)

    # If your knee file only contains GA+BO (or only GA), that’s fine—the plot will reflect it.
    return both

# ---------------------------------------
# Plot function (one figure per model)
# ---------------------------------------
def plot_best_vs_knee(df_all: pd.DataFrame, title: str, outfile: Path):
    # Map intervals to markers for readability
    intervals = sorted(df_all["retrain_interval"].unique().tolist())
    markers = ["o","s","^","D","P","X"]
    mk_map = {iv: markers[i % len(markers)] for i,iv in enumerate(intervals)}

    # Colors for kind
    color_map = {"Best":"tab:orange", "Knee":"tab:blue"}

    # Jitter to reduce overlap slightly on MDD axis
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=0.0015, size=len(df_all))
    df_all = df_all.copy()
    df_all["mdd_jit"] = np.clip(df_all["mdd"] + jitter, 0, None)

    plt.figure(figsize=(9,6))
    for kind, subk in df_all.groupby("kind"):
        for iv, sub in subk.groupby("retrain_interval"):
            plt.scatter(sub["sharpe"], sub["mdd_jit"],
                        label=f"{kind} — {iv}d",
                        alpha=0.75,
                        edgecolor="white",
                        linewidths=0.6,
                        s=48,
                        marker=mk_map[iv],
                        c=color_map.get(kind, "gray"))

    plt.xlabel("Sharpe Ratio (higher is better)")
    plt.ylabel("Maximum Drawdown (lower is better)")
    plt.title(title)
    # Deduplicate legend entries while preserving order
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set(); h2=[]; l2=[]
    for h,l in zip(handles,labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    plt.legend(h2, l2, ncol=2, frameon=True, fontsize=9, loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

# ---------------------------------------
# Combined side-by-side (optional)
# ---------------------------------------
def plot_combined(df_arima, df_lstm, outfile: Path):
    fig, axes = plt.subplots(1,2, figsize=(13,5), sharex=False, sharey=True)
    for ax, (title, df_all) in zip(axes, [("ARIMA–GARCH: Best vs. Knee (all folds)", df_arima),
                                          ("LSTM: Best vs. Knee (all folds)", df_lstm)]):

        intervals = sorted(df_all["retrain_interval"].unique().tolist())
        markers = ["o","s","^","D","P","X"]
        mk_map = {iv: markers[i % len(markers)] for i,iv in enumerate(intervals)}
        color_map = {"Best":"tab:orange", "Knee":"tab:blue"}

        rng = np.random.default_rng(42)
        jitter = rng.normal(loc=0.0, scale=0.0015, size=len(df_all))
        df_all = df_all.copy()
        df_all["mdd_jit"] = np.clip(df_all["mdd"] + jitter, 0, None)

        for kind, subk in df_all.groupby("kind"):
            for iv, sub in subk.groupby("retrain_interval"):
                ax.scatter(sub["sharpe"], sub["mdd_jit"],
                           label=f"{kind} — {iv}d",
                           alpha=0.75, edgecolor="white", linewidths=0.6,
                           s=48, marker=mk_map[iv], c=color_map.get(kind, "gray"))

        ax.set_title(title)
        ax.set_xlabel("Sharpe Ratio")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Maximum Drawdown")
    # Shared legend (unique)
    handles, labels = axes[1].get_legend_handles_labels()
    seen=set(); h2=[]; l2=[]
    for h,l in zip(handles,labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, ncol=3, frameon=True, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(outfile, dpi=300)
    plt.close()

# -----------------------------
# Run for both models
# -----------------------------
arima_both = load_best_and_knee(ARIMA_FRONTS, ARIMA_KNEE)
lstm_both  = load_best_and_knee(LSTM_FRONTS, LSTM_KNEE)

plot_best_vs_knee(arima_both,
                  title="ARIMA–GARCH: Best vs. Knee across all folds and retraining intervals",
                  outfile=OUTDIR/"fig_best_vs_knee_arima.png")

plot_best_vs_knee(lstm_both,
                  title="LSTM: Best vs. Knee across all folds and retraining intervals",
                  outfile=OUTDIR/"fig_best_vs_knee_lstm.png")

# Optional combined two-panel figure
plot_combined(arima_both, lstm_both, OUTDIR/"fig_best_vs_knee_both.png")

print("Saved:",
      OUTDIR/"fig_best_vs_knee_arima.png",
      OUTDIR/"fig_best_vs_knee_lstm.png",
      OUTDIR/"fig_best_vs_knee_both.png", sep="\n - ")