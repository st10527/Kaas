"""
Generate fig_ablation_acc.pdf — ablation bar chart (Final Accuracy).
Two sub-panels: σ_n=0.3 (left) and σ_n=1.5 (right).
Output: figures/fig_ablation_acc.pdf
"""

import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
VARIANT_ORDER = ["full", "no_straggler", "no_wf", "no_timeout", "no_quality"]
VARIANT_LABELS = {
    "full":         "Full DASH",
    "no_straggler": "w/o Straggler\nSelection",
    "no_wf":        "w/o\nWater-Filling",
    "no_timeout":   "w/o Adaptive\nTimeout",
    "no_quality":   "w/o Quality\nWeights",
}

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

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_FULL    = "#2166AC"   # strong blue — full DASH
COLOR_VARIANT = "#B2B2B2"   # neutral gray — ablated variants
COLOR_PUNCH   = "#D6604D"   # red-orange — highlights no_timeout at σ=1.5

def bar_colors(sigma):
    """Return color list; at σ=1.5 highlight no_timeout in red."""
    cols = []
    for v in VARIANT_ORDER:
        if v == "full":
            cols.append(COLOR_FULL)
        elif sigma == 1.5 and v == "no_timeout":
            cols.append(COLOR_PUNCH)
        else:
            cols.append(COLOR_VARIANT)
    return cols

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
plt.subplots_adjust(wspace=0.32)

x = np.arange(len(VARIANT_ORDER))
BAR_W = 0.55

for ax, (means, stds, sigma, title) in zip(
    axes,
    [
        (means03, stds03, 0.3, r"$\sigma_n = 0.3$  (Low Heterogeneity)"),
        (means15, stds15, 1.5, r"$\sigma_n = 1.5$  (High Heterogeneity)"),
    ],
):
    vals  = [means[v] for v in VARIANT_ORDER]
    errs  = [stds[v]  for v in VARIANT_ORDER]
    cols  = bar_colors(sigma)

    bars = ax.bar(x, vals, BAR_W, color=cols, zorder=3,
                  yerr=errs, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor="#555555", capthick=1.2))

    # value labels on top of each bar
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.0015,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7.5, color="#222222"
        )

    # --- axis formatting ---
    y_min = min(vals) - max(errs) - 0.010
    y_max = max(vals) + max(errs) + 0.018
    ax.set_ylim(y_min, y_max)

    # tighten y-tick range so differences are visible
    tick_bot = round(y_min * 100) / 100
    tick_top = round(y_max * 100) / 100
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER],
                       fontsize=8.5)
    ax.set_ylabel("Final Accuracy", fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # annotate the no_timeout drop at σ=1.5
    if sigma == 1.5:
        nt_idx  = VARIANT_ORDER.index("no_timeout")
        fl_idx  = VARIANT_ORDER.index("full")
        drop    = means["full"] - means["no_timeout"]
        y_annot = means["no_timeout"] - stds["no_timeout"] - 0.004
        ax.annotate(
            f"−{drop*100:.2f}%",
            xy=(x[nt_idx], means["no_timeout"]),
            xytext=(x[nt_idx], y_annot),
            ha="center", va="top",
            fontsize=8, color=COLOR_PUNCH, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLOR_PUNCH, lw=1.2),
        )

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=COLOR_FULL,    label="Full DASH"),
    mpatches.Patch(color=COLOR_VARIANT, label="Ablated variant"),
    mpatches.Patch(color=COLOR_PUNCH,   label="w/o Adaptive Timeout (σ=1.5)"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04),
           frameon=True, edgecolor="#CCCCCC")

fig.suptitle("Ablation Study — Final Accuracy", fontsize=11, y=1.01)

# ── Save ──────────────────────────────────────────────────────────────────────
import os
os.makedirs("figures", exist_ok=True)
out_pdf = "figures/fig_ablation_acc.pdf"
out_png = "figures/fig_ablation_acc.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
