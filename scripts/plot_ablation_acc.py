"""
Generate fig_ablation_low.pdf and fig_ablation_high.pdf
Two separate single-column bar charts (Final Accuracy).
  fig_ablation_low.pdf  — σ_n = 0.3 (Low Heterogeneity)
  fig_ablation_high.pdf — σ_n = 1.5 (High Heterogeneity)
Color palette matches plot_all_jpdc_figures.py (DASH=#D62728, etc.)
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
COLOR_FULL         = "#D62728"   # DASH red
COLOR_NO_STRAGGLER = "#1F77B4"   # blue
COLOR_NO_WF        = "#2CA02C"   # green
COLOR_NO_TIMEOUT   = "#FF7F0E"   # orange (also the punch-line colour at σ=1.5)
COLOR_NO_QUALITY   = "#9467BD"   # purple

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

LEGEND_HANDLES = [
    mpatches.Patch(color=COLOR_FULL,         label="Full DASH"),
    mpatches.Patch(color=COLOR_NO_STRAGGLER, label="w/o Straggler Selection"),
    mpatches.Patch(color=COLOR_NO_WF,        label="w/o Water-Filling"),
    mpatches.Patch(color=COLOR_NO_TIMEOUT,   label="w/o Adaptive Timeout"),
    mpatches.Patch(color=COLOR_NO_QUALITY,   label="w/o Quality Weights"),
]

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

# ── Shared Y-axis range (both figures use the same scale) ─────────────────────
all_vals = [means03[v] for v in VARIANT_ORDER] + [means15[v] for v in VARIANT_ORDER]
all_errs = [stds03[v]  for v in VARIANT_ORDER] + [stds15[v]  for v in VARIANT_ORDER]
Y_MIN = min(v - e for v, e in zip(all_vals, all_errs)) - 0.008
Y_MAX = max(v + e for v, e in zip(all_vals, all_errs)) + 0.022

# ── Helper: draw one single-column figure ─────────────────────────────────────
def make_figure(means, stds, sigma, out_stem):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    x     = np.arange(len(VARIANT_ORDER))
    BAR_W = 0.55
    vals  = [means[v] for v in VARIANT_ORDER]
    errs  = [stds[v]  for v in VARIANT_ORDER]
    cols  = [VARIANT_COLORS[v] for v in VARIANT_ORDER]

    bars = ax.bar(x, vals, BAR_W, color=cols, zorder=3,
                  yerr=errs, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor="#444444", capthick=1.2))

    # value labels above each bar
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.0012,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7.5, color="#222222"
        )

    # annotate the no_timeout accuracy drop at σ=1.5
    if sigma == 1.5:
        nt_idx = VARIANT_ORDER.index("no_timeout")
        drop   = means["full"] - means["no_timeout"]
        ax.annotate(
            f"−{drop*100:.2f}%",
            xy=(x[nt_idx], means["no_timeout"] - stds["no_timeout"]),
            xytext=(x[nt_idx] + 0.60, means["no_timeout"] - stds["no_timeout"] - 0.006),
            ha="left", va="top",
            fontsize=8.5, color=COLOR_NO_TIMEOUT, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_NO_TIMEOUT, lw=1.3),
        )

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER],
                       fontsize=8.5, linespacing=1.25)
    ax.set_ylabel("Final Accuracy", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # legend inside top of the axes (2 rows × 3 / 2 cols layout)
    ax.legend(handles=LEGEND_HANDLES,
              loc="upper right", ncol=1, fontsize=7.5,
              frameon=True, edgecolor="#CCCCCC",
              handlelength=1.1, handletextpad=0.5,
              borderpad=0.6, labelspacing=0.4)

    os.makedirs("figures", exist_ok=True)
    for ext, dpi in [("pdf", 300), ("png", 150)]:
        path = f"figures/{out_stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)

# ── Generate both figures ─────────────────────────────────────────────────────
make_figure(means03, stds03, 0.3, "fig_ablation_low")
make_figure(means15, stds15, 1.5, "fig_ablation_high")
