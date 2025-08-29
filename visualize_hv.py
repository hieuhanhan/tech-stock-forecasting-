import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------- paths --------
HV_CSV  = Path("data/tuning_results/csv/tier2_arima_cont_gabo_hv.csv")
OUT_PNG = Path("results/chap5/Fig_5_3_HV_progression_arima_markers.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# -------- style --------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# -------- load --------
hv = pd.read_csv(HV_CSV)

# keep GA logs per generation
ga_gen = hv[(hv["stage"] == "GA") & (hv["gen"] >= 1)].copy()

# aggregate mean ± std
agg = (ga_gen.groupby(["retrain_interval","gen"], as_index=False)
              .agg(hv_mean=("hv","mean"), hv_std=("hv","std")))
agg["hv_std"] = agg["hv_std"].fillna(0.0)

intervals = sorted(agg["retrain_interval"].unique())
colors = plt.cm.tab10.colors

# -------- plot --------
fig, ax = plt.subplots(figsize=(7.6, 4.6))

for i, interval in enumerate(intervals):
    sub = agg[agg["retrain_interval"] == interval].sort_values("gen")
    c = colors[i % len(colors)]
    ax.plot(sub["gen"], sub["hv_mean"], color=c, lw=2,
            label=f"GA (interval={interval})", zorder=3)
    ax.fill_between(sub["gen"],
                    sub["hv_mean"] - sub["hv_std"],
                    sub["hv_mean"] + sub["hv_std"],
                    color=c, alpha=0.18, linewidth=0, zorder=2)

# y-padding
ymin = float((agg["hv_mean"] - agg["hv_std"]).min())
ymax = float((agg["hv_mean"] + agg["hv_std"]).max())
pad  = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_xlabel("Generation")
ax.set_ylabel("Hypervolume (HV)")
ax.set_title("Hypervolume (HV) progression across generations — Tier-2 ARIMA (mean ± std)",
             fontsize=12, pad=8)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

# -------- add final HV markers --------
df_final = hv[hv["stage"].isin(["final_ga", "final_union"])].copy()
df_final["stage_clean"] = df_final["stage"].map({"final_ga": "GA", "final_union": "GA+BO"})

# mean across folds at final gen
final_means = df_final.groupby(["retrain_interval","stage_clean"], as_index=False)["hv"].mean()

xpos = agg["gen"].max() + 1.5  # shift markers beyond last gen
y_offset = 0.02 * (ymax - ymin)  # push markers slightly upward

for i, interval in enumerate(intervals):
    sub_ga  = final_means[(final_means["retrain_interval"]==interval) & (final_means["stage_clean"]=="GA")]
    sub_gbo = final_means[(final_means["retrain_interval"]==interval) & (final_means["stage_clean"]=="GA+BO")]
    c = colors[i % len(colors)]

    if not sub_ga.empty:
        ax.scatter(xpos, sub_ga["hv"].values[0] + y_offset, marker="o", color=c, s=50,
                   edgecolor="black", zorder=5, label="_nolegend_")
    if not sub_gbo.empty:
        ax.scatter(xpos, sub_gbo["hv"].values[0] + y_offset, marker="s", color=c, s=50,
                   edgecolor="black", zorder=5, label="_nolegend_")

# vertical guide line
ax.axvline(x=agg["gen"].max(), color="gray", linestyle="--", linewidth=0.8)
ax.text(xpos+2.2, ymax, "final HV\n(GA ● / GA+BO ■)", fontsize=9, va="top")

# legend outside
ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.subplots_adjust(right=0.78, bottom=0.14)

fig.savefig(OUT_PNG, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved → {OUT_PNG}")