import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

OUT_PNG2 = Path("results/chap5/Fig_5_3b_HV_final_ga_vs_gabo.png")
OUT_PNG2.parent.mkdir(parents=True, exist_ok=True)

hv = pd.read_csv("data/tuning_results/csv/tier2_arima_cont_gabo_hv.csv")

# take only the final rows we care about
df_final = hv[hv["stage"].isin(["final_ga", "final_union"])].copy()
df_final = df_final[np.isfinite(df_final["hv"])]
df_final["stage_clean"] = df_final["stage"].map({"final_ga": "GA", "final_union": "GA+BO"})

# summary (mean ± std across folds), sorted by interval
sumtab = (df_final.groupby(["retrain_interval","stage_clean"], as_index=False)
          .agg(mean_hv=("hv","mean"), std_hv=("hv","std")))
sumtab["std_hv"] = sumtab["std_hv"].fillna(0.0)
sumtab = sumtab.sort_values(["retrain_interval","stage_clean"])

# pivot for plotting; ensure both columns exist
wide = sumtab.pivot(index="retrain_interval", columns="stage_clean", values="mean_hv")
wide_std = sumtab.pivot(index="retrain_interval", columns="stage_clean", values="std_hv")
for col in ["GA","GA+BO"]:
    if col not in wide.columns:
        wide[col] = np.nan
    if col not in wide_std.columns:
        wide_std[col] = np.nan
wide = wide.sort_index()
wide_std = wide_std.reindex(wide.index)

# ---- plot ----
plt.rcParams.update({
    "font.family": "Times New Roman",
    "figure.dpi": 150,
    "savefig.dpi": 300
})

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(wide.index))
w = 0.34

ga_means, ga_stds = wide["GA"].to_numpy(), wide_std["GA"].to_numpy()
gabo_means, gabo_stds = wide["GA+BO"].to_numpy(), wide_std["GA+BO"].to_numpy()

ax.bar(x - w/2, ga_means,    yerr=ga_stds,    width=w, label="GA",
       color="tab:blue",   alpha=0.85, capsize=4, edgecolor="none")
ax.bar(x + w/2, gabo_means,  yerr=gabo_stds,  width=w, label="GA+BO",
       color="tab:orange", alpha=0.85, capsize=4, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in wide.index])
ax.set_xlabel("Retraining interval")
ax.set_ylabel("Final Hypervolume (HV)")
ax.set_title("Final HV by interval — GA vs GA+BO (union)", fontsize=12, pad=8)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, loc="upper right")

# nice y padding; keep zero visible if values tiny
ymin = np.nanmin([ga_means - ga_stds, gabo_means - gabo_stds])
ymax = np.nanmax([ga_means + ga_stds, gabo_means + gabo_stds])
if not np.isfinite(ymin) or not np.isfinite(ymax):
    ymin, ymax = 0.0, 1.0
span = (ymax - ymin) if ymax > ymin else 1.0
pad  = 0.12 * span
ax.set_ylim(ymin - pad, ymax + pad)

plt.tight_layout()
fig.savefig(OUT_PNG2, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved → {OUT_PNG2}")

# (optional) also save the summary behind the bar chart
sum_out = OUT_PNG2.with_suffix(".csv")
sumtab.to_csv(sum_out, index=False)
print(f"[OK] Saved summary → {sum_out}")