"""
Generate fig_ablation_acc.pdf — ablation bar chart (Final Accuracy).
Two sub-panels: σ_n=0.3 (left) and σ_n=1.5 (right).
Color palette matches plot_all_jpdc_figures.py (DASH=#D62728, etc.)
Output: figures/fig_ablation_acc.pdf
"""

import json
import statistics
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Palette — consistent with plot_all_jpdc_figures.py ───────────────────────
COLOR_FULL        = "#D62728"   # DASH red (main figures)
COLOR_NO_STRAGGLER= "#1F77B4"   # blue  (Sync-Greedy slot)
COLOR_NO_WF       = "#2CA02C"   # green (FedBuff-FD slot)
COLOR_NO_TIMEOUT  = "#FF7F0E"   # orange — and highlighted in σ=1.5
COLOR_NO_QUALITY  = "#9467BD"   # purple (Full-Async slot)

VARIANT_ORDER  = ["full", "no_straggler", "no_wf", "no_timeout", "no_quality"]
VARIANT_COLORS = {
    "full":         COLOR_FULL,
    "no_straggler": COLOR_NO_STRAGGLER,
    "no_wf":        COLOR_NO_WF,
    "no_timeout":   COLOR_NO_TIMEOUT,
    "no_quality":   COLOR_NO_QUALITY,
}
VARIANT_LABELS = {
    "full":         "Full\nDASH",
    "no_straggler": "w/o\nStraggler\nSelection",
    "no_wf":        "w/o\nWater-\nFilling",
    "no_timeout":   "w/o\nAdaptive\nTimeout",
    "no_quality":   "w/o\nQuality\nWeights",
}

# ── Data ─────────────────────────────────────────────────────────────────────
def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    means, stds = {}, {}
    for variant, vdata in d.items():
        accs = [s["history"][-1]["accuracy"] for s in vdata["seeds"]]
        means[variant] = statistics.mean(accs)
        stds[variant]  = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return means, stds

means03, stds03 = load_stats("results/ablation_full.json")
means15, stds15 = load_stats("results/ablation_sigma15.json")

# ── Shared Y-axis range ───────────────────────────────────────────────────────
all_vals = [means03[v] for v in VARIANT_ORDER] + [means15[v] for v in VARIANT_ORDER]
all_errs = [stds03[v]  for v in VARIANT_ORDER] + [stds15[v]  for v in VARIANT_ORDER]
Y_MIN = min(v - e for v, e in zip(all_vals, all_errs)) - 0.008
Y_MAX = max(v + e for v, e in zip(all_vals, all_errs)) + 0.020

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
plt.subplots_adjust(wspace=0.10)

x     = np.arange(len(VARIANT_ORDER))
BAR_W = 0.55

PANELS = [
    (means03, stds03, 0.3, r"(a) $\sigma_n = 0.3$ — Low Heterogeneity"),
    (means15, stds15, 1.5, r"(b) $\sigma_n = 1.5$ — High Heterogeneity"),
]

for ax, (means, stds, sigma, title) in zip(axes, PANELS):
    vals = [means[v] for v in VARIANT_ORDER]
    errs = [stds[v]  for v in VARIANT_ORDER]
    cols = [VARIANT_COLORS[v] for v in VARIANT_ORDER]

    bars = ax.bar(x, vals, BAR_W, color=cols, zorder=3,
                  yerr=errs, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor="#444444", capthick=1.2))

    # value labels on top of each bar
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.0012,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7.5, color="#222222"
        )

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER],
                       fontsize=8.2, linespacing=1.25)
    ax.set_title(title, fontsize=10, pad=7)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # annotate the no_timeout drop only at σ=1.5
    if sigma == 1.5:
        nt_idx = VARIANT_ORDER.index("no_timeout")
        drop   = means["full"] - means["no_timeout"]
        y_tip  = means["no_timeout"] - stds["no_timeout"] - 0.003
        ax.annotate(
            f"−{drop*100:.2f}%",
            xy=(x[nt_idx], means["no_timeout"] - stds["no_timeout"]),
            xytext=(x[nt_idx] + 0.55, y_tip - 0.003),
            ha="left", va="top",
            fontsize=8.5, color=COLOR_NO_TIMEOUT, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_NO_TIMEOUT, lw=1.3),
        )

# only left panel needs y-label (sharey=True)
axes[0].set_ylabel("Final Accuracy", fontsize=9)

# ── Legend — placed at top centre inside the figure ───────────────────────────
legend_handles = [
    mpatches.Patch(color=COLOR_FULL,         label="Full DASH"),
    mpatches.Patch(color=COLOR_NO_STRAGGLER, label="w/o Straggler Selection"),
    mpatches.Patch(color=COLOR_NO_WF,        label="w/o Water-Filling"),
    mpatches.Patch(color=COLOR_NO_TIMEOUT,   label="w/o Adaptive Timeout"),
    mpatches.Patch(color=COLOR_NO_QUALITY,   label="w/o Quality Weights"),
]
fig.legend(handles=legend_handles,
           loc="upper center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 1.01),
           frameon=True, edgecolor="#CCCCCC",
           handlelength=1.2, handletextpad=0.5, columnspacing=1.0)

fig.suptitle("Ablation Study — Final Accuracy", fontsize=11, y=1.09)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
out_pdf = "figures/fig_ablation_acc.pdf"
out_png = "figures/fig_ablation_acc.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
