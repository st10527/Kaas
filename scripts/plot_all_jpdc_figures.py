#!/usr/bin/env python3
"""Generate ALL JPDC 2026 paper figures (Fig 2–10).

Output: figures/jpdc/fig{N}_{name}.pdf

Each figure is a single-panel PDF.  Figures from Fig 4 onwards have
NO embedded titles — titles are handled by LaTeX \\subfigure captions.

Style:
  - Hollow markers (white face) on all line / errorbar plots
  - Consistent colour per method across every figure
  - Legend carefully positioned to avoid overlapping data
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Global Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

RESULTS = Path("results/jpdc")
FIGDIR = Path("figures/jpdc")
FIGDIR.mkdir(parents=True, exist_ok=True)

SEEDS3 = [42, 123, 456]
SEED_KEYS = ["seed42", "seed123", "seed456"]

# ── Colour palette — same method ⟹ same colour everywhere ──
COLORS = {
    'DASH':          '#D62728',  # red
    'Sync-Greedy':   '#1F77B4',  # blue
    'FedBuff-FD':    '#2CA02C',  # green
    'Random-Async':  '#FF7F0E',  # orange
    'Full-Async':    '#9467BD',  # purple
    'Sync-Full':     '#8C564B',  # brown
}
MARKERS = {
    'DASH': 'o', 'Sync-Greedy': 's', 'FedBuff-FD': '^',
    'Random-Async': 'D', 'Full-Async': 'v', 'Sync-Full': 'P',
}

# Hollow-marker kwargs applied to every plot() / errorbar()
MK = dict(markerfacecolor='white', markeredgewidth=1.2)

# Privacy colours (Fig 10 — not methods, so separate palette is correct)
PRIV_COLORS = {
    'no_privacy': '#1F77B4', 'mild_privacy': '#2CA02C',
    'moderate_privacy': '#FF7F0E', 'strong_privacy': '#D62728',
}
PRIV_MARKERS = {
    'no_privacy': 'o', 'mild_privacy': 's',
    'moderate_privacy': '^', 'strong_privacy': 'D',
}
PRIV_LABELS = {
    'no_privacy':       r'None ($\tilde{\rho}\!\approx\!0.98$)',
    'mild_privacy':     r'Mild ($\tilde{\rho}\!\approx\!0.80$)',
    'moderate_privacy': r'Moderate ($\tilde{\rho}\!\approx\!0.55$)',
    'strong_privacy':   r'Strong ($\tilde{\rho}\!\approx\!0.10$)',
}


def load(fn):
    with open(RESULTS / fn) as f:
        return json.load(f)


def mean_std(vals):
    return np.mean(vals), np.std(vals)


# ====================================================================
# Fig 2: Accuracy vs Round  (Exp 1)
# ====================================================================
def fig2():
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD",
               "Random-Async", "Full-Async", "Sync-Full"]

    fig, ax = plt.subplots(figsize=(6, 4))
    rounds = np.arange(1, 51)

    for m in methods:
        acc_per_round = []
        for r in range(50):
            accs = [data[f"{m}_seed{s}"]["history"][r]["accuracy"] * 100
                    for s in SEEDS3 if r < len(data[f"{m}_seed{s}"]["history"])]
            acc_per_round.append(np.mean(accs))
        ax.plot(rounds, acc_per_round, color=COLORS[m], marker=MARKERS[m],
                markevery=5, label=m,
                zorder=3 if m == 'DASH' else 2, **MK)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(0, 51)
    ax.set_ylim(None, 55)
    ax.legend(loc='upper left', ncol=2,
              framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGDIR / "fig2_acc_vs_round.pdf")
    plt.close(fig)
    print("  ✅ Fig 2: acc_vs_round")


# ====================================================================
# Fig 3: Accuracy vs Wall-Clock  (Exp 1) — the "killer" figure
# ====================================================================
def fig3():
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD",
               "Random-Async", "Full-Async", "Sync-Full"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for m in methods:
        wc_per_round, acc_per_round = [], []
        for r in range(50):
            wcs, accs = [], []
            for s in SEEDS3:
                h = data[f"{m}_seed{s}"]["history"]
                if r < len(h):
                    wcs.append(h[r].get("wall_clock_time", 0))
                    accs.append(h[r]["accuracy"] * 100)
            wc_per_round.append(np.mean(wcs))
            acc_per_round.append(np.mean(accs))
        ax.plot(wc_per_round, acc_per_round, color=COLORS[m],
                marker=MARKERS[m], markevery=5, label=m,
                zorder=3 if m == 'DASH' else 2, **MK)

    ax.set_xlabel('Wall-Clock Time (s)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(0, 3500)
    ax.legend(loc='center right', ncol=1,
              framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGDIR / "fig3_acc_vs_wallclock.pdf")
    plt.close(fig)
    print("  ✅ Fig 3: acc_vs_wallclock")


# ====================================================================
# Fig 4: Straggler Sweep — Final Accuracy  (Exp 2)
# ====================================================================
def fig4():
    data = load("exp2_straggler_sweep.json")
    sigmas = [0.0, 0.3, 0.5, 1.0, 1.5]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, ax = plt.subplots(figsize=(5, 4))

    for m in methods:
        accs, acc_stds = [], []
        for sig in sigmas:
            sk = f"sigma={sig}"
            finals = [data[sk][s][m]["final_accuracy"] * 100 for s in SEED_KEYS]
            accs.append(np.mean(finals))
            acc_stds.append(np.std(finals))
        ax.errorbar(sigmas, accs, yerr=acc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3, **MK)

    ax.set_xlabel(r'Straggler Severity $\sigma_n$')
    ax.set_ylabel('Final Accuracy (%)')
    ax.legend(framealpha=0.95, edgecolor='gray', loc='right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_straggler_acc.pdf")
    plt.close(fig)
    print("  ✅ Fig 4: straggler_acc")


# ====================================================================
# Fig 5: Straggler Sweep — Wall-Clock  (Exp 2)
# ====================================================================
def fig5():
    data = load("exp2_straggler_sweep.json")
    sigmas = [0.0, 0.3, 0.5, 1.0, 1.5]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, ax = plt.subplots(figsize=(5, 4))

    for m in methods:
        wcs, wc_stds = [], []
        for sig in sigmas:
            sk = f"sigma={sig}"
            ws = [data[sk][s][m]["wall_clock_time"] for s in SEED_KEYS]
            wcs.append(np.mean(ws))
            wc_stds.append(np.std(ws))
        ax.errorbar(sigmas, wcs, yerr=wc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3, **MK)

    ax.set_xlabel(r'Straggler Severity $\sigma_n$')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.legend(framealpha=0.95, edgecolor='gray', loc='right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_straggler_wc.pdf")
    plt.close(fig)
    print("  ✅ Fig 5: straggler_wc")


# ====================================================================
# Fig 6a: Policy Comparison — Accuracy vs WC Scatter  (Exp 3)
# ====================================================================
def fig6a():
    d3  = load("exp3_policy_comparison.json")
    d3b = load("exp3b_policy_nofloor.json")
    policies = ["fixed(5.0)", "fixed(10.0)", "fixed(20.0)",
                "adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)"]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    pcols = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    pnames = [p.replace("(", " ").replace(")", "") for p in policies]

    for i, p in enumerate(policies):
        w_acc = np.mean([d3[p]["seeds"][s]["final_accuracy"]  * 100 for s in SEED_KEYS])
        w_wc  = np.mean([d3[p]["seeds"][s]["wall_clock_time"]       for s in SEED_KEYS])
        o_acc = np.mean([d3b[p]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS])
        o_wc  = np.mean([d3b[p]["seeds"][s]["wall_clock_time"]      for s in SEED_KEYS])

        ax.scatter(w_wc, w_acc, facecolors='white', edgecolors=pcols[i],
                   marker='o', s=80, zorder=3, linewidths=1.5,
                   label=pnames[i])
        ax.scatter(o_wc, o_acc, color=pcols[i], marker='x',
                   s=60, zorder=3, linewidths=1.5)
        ax.annotate('', xy=(w_wc, w_acc), xytext=(o_wc, o_acc),
                    arrowprops=dict(arrowstyle='->', color=pcols[i],
                                    alpha=0.4, lw=1))

    mk_handles = [
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='white',
               markeredgecolor='gray', markersize=7, linestyle='None',
               label=r'+$D_{\min}$'),
        Line2D([0], [0], marker='x', color='gray', markersize=7,
               linestyle='None', label=r'$-D_{\min}$'),
    ]
    leg_mk = ax.legend(handles=mk_handles, fontsize=7, loc='upper left',
                       framealpha=0.95, edgecolor='gray')
    ax.add_artist(leg_mk)
    ax.legend(fontsize=7, loc='lower right', ncol=2,
              framealpha=0.95, edgecolor='gray',
              title='Policy', title_fontsize=7)

    ax.set_xlabel('Wall-Clock Time (s)')
    ax.set_ylabel('Final Accuracy (%)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6a_policy_scatter.pdf")
    plt.close(fig)
    print("  ✅ Fig 6a: policy_scatter")


# ====================================================================
# Fig 6b: Deadline Evolution  (Exp 3)
# ====================================================================
def fig6b():
    d3  = load("exp3_policy_comparison.json")
    d3b = load("exp3b_policy_nofloor.json")

    fig, ax = plt.subplots(figsize=(5, 4))

    show = ["adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)"]
    dl_cols = ['#E41A1C', '#377EB8', '#4DAF4A']   # red / blue / green

    for i, p in enumerate(show):
        pname = p.replace("(", " ").replace(")", "")
        # +D_min (solid)
        h = d3[p]["seeds"]["seed42"].get("history", [])
        dl = [r.get("extra", {}).get("deadline", None) for r in h]
        dl = [d for d in dl if d is not None]
        if dl:
            ax.plot(range(1, len(dl) + 1), dl, color=dl_cols[i],
                    linewidth=1.8, label=f'{pname} +$D_{{\\min}}$')
        # -D_min (dashed)
        h2 = d3b[p]["seeds"]["seed42"].get("history", [])
        dl2 = [r.get("extra", {}).get("deadline", None) for r in h2]
        dl2 = [d for d in dl2 if d is not None]
        if dl2:
            ax.plot(range(1, len(dl2) + 1), dl2, color=dl_cols[i],
                    linewidth=1.2, linestyle='--',
                    label=f'{pname} $-D_{{\\min}}$')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Deadline D (s)')
    ax.legend(framealpha=0.95, fontsize=7, ncol=2, edgecolor='gray',
              loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6b_deadline_evolution.pdf")
    plt.close(fig)
    print("  ✅ Fig 6b: deadline_evolution")


# ====================================================================
# Fig 7: Scalability — Accuracy  (Exp 4)
# ====================================================================
def fig7():
    data = load("exp4_scalability.json")
    ms = [20, 50, 100, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, ax = plt.subplots(figsize=(5, 4))

    for m in methods:
        accs, acc_stds = [], []
        for M in ms:
            mk = f"M={M}"
            finals = [data[mk]["seeds"][s][m]["final_accuracy"] * 100
                      for s in SEED_KEYS]
            accs.append(np.mean(finals))
            acc_stds.append(np.std(finals))
        ax.errorbar(ms, accs, yerr=acc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3, **MK)

    ax.set_xlabel('Number of Devices (M)')
    ax.set_ylabel('Final Accuracy (%)')
    ax.set_xticks(ms)
    ax.legend(framealpha=0.95, edgecolor='gray', loc='center right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_scalability_acc.pdf")
    plt.close(fig)
    print("  ✅ Fig 7: scalability_acc")


# ====================================================================
# Fig 8: Scalability — Wall-Clock  (Exp 4)
# ====================================================================
def fig8():
    data = load("exp4_scalability.json")
    ms = [20, 50, 100, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, ax = plt.subplots(figsize=(5, 4))

    for m in methods:
        wcs, wc_stds = [], []
        for M in ms:
            mk = f"M={M}"
            ws = [data[mk]["seeds"][s][m]["wall_clock_time"]
                  for s in SEED_KEYS]
            wcs.append(np.mean(ws))
            wc_stds.append(np.std(ws))
        ax.errorbar(ms, wcs, yerr=wc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3, **MK)

    # Speedup annotations below each DASH point
    for M in ms:
        mk = f"M={M}"
        dash_wc = np.mean([data[mk]["seeds"][s]["DASH"]["wall_clock_time"]
                           for s in SEED_KEYS])
        sync_wc = np.mean([data[mk]["seeds"][s]["Sync-Greedy"]["wall_clock_time"]
                           for s in SEED_KEYS])
        spd = sync_wc / dash_wc
        ax.annotate(f'{spd:.1f}\u00d7', xy=(M, dash_wc), fontsize=8,
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', color=COLORS['DASH'], fontweight='bold')

    ax.set_xlabel('Number of Devices (M)')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_xticks(ms)
    ax.legend(framealpha=0.95, edgecolor='gray', loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_scalability_wc.pdf")
    plt.close(fig)
    print("  ✅ Fig 8: scalability_wc")


# ====================================================================
# Fig 9a: EMNIST — Best Accuracy  (Exp 5)
# ====================================================================
def fig9a():
    data = load("exp5_emnist.json")
    ms = [50, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(ms))
    width = 0.22

    for i, m in enumerate(methods):
        bests, best_stds = [], []
        for M in ms:
            mk = f"M={M}"
            b = [data[mk]["seeds"][s][m]["best_accuracy"] * 100
                 for s in SEED_KEYS]
            bests.append(np.mean(b))
            best_stds.append(np.std(b))
        ax.bar(x + i * width, bests, width, yerr=best_stds,
               color=COLORS[m], label=m, capsize=3,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Number of Devices (M)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'M={M}' for M in ms])
    ax.set_ylim(60, 90)
    ax.legend(framealpha=0.95, edgecolor='gray', loc='upper center',
              ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9a_emnist_acc.pdf")
    plt.close(fig)
    print("  ✅ Fig 9a: emnist_acc")


# ====================================================================
# Fig 9b: EMNIST — Wall-Clock  (Exp 5)
# ====================================================================
def fig9b():
    data = load("exp5_emnist.json")
    ms = [50, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(ms))
    width = 0.22

    for i, m in enumerate(methods):
        wcs_vals, wc_stds = [], []
        for M in ms:
            mk = f"M={M}"
            w = [data[mk]["seeds"][s][m]["wall_clock_time"]
                 for s in SEED_KEYS]
            wcs_vals.append(np.mean(w))
            wc_stds.append(np.std(w))
        ax.bar(x + i * width, wcs_vals, width, yerr=wc_stds,
               color=COLORS[m], label=m, capsize=3,
               edgecolor='black', linewidth=0.5)

    # Speedup annotations
    for j, M in enumerate(ms):
        mk = f"M={M}"
        dash_wc = np.mean([data[mk]["seeds"][s]["DASH"]["wall_clock_time"]
                           for s in SEED_KEYS])
        sync_wc = np.mean([data[mk]["seeds"][s]["Sync-Greedy"]["wall_clock_time"]
                           for s in SEED_KEYS])
        spd = sync_wc / dash_wc
        ax.annotate(f'{spd:.1f}\u00d7 speedup',
                    xy=(j + 0.5 * width, dash_wc), fontsize=9,
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', color=COLORS['DASH'], fontweight='bold')

    ax.set_xlabel('Number of Devices (M)')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'M={M}' for M in ms])
    ax_ymax = ax.get_ylim()[1]
    ax.set_ylim(None, ax_ymax * 1.2)
    ax.legend(framealpha=0.95, edgecolor='gray', loc='upper center',
              ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9b_emnist_wc.pdf")
    plt.close(fig)
    print("  ✅ Fig 9b: emnist_wc")


# ====================================================================
# Fig 10a: Privacy — Accuracy & Wall-Clock dual-axis  (Exp 6)
# ====================================================================
def fig10a():
    data = load("exp6_privacy_async.json")
    privs = ["no_privacy", "mild_privacy", "moderate_privacy", "strong_privacy"]

    fig, ax = plt.subplots(figsize=(5, 4))

    rho_actual, acc_means, acc_stds_v, wc_means = [], [], [], []
    for pk in privs:
        rhos  = [data[pk]["seeds"][s]["avg_rho"]             for s in SEED_KEYS]
        bests = [data[pk]["seeds"][s]["best_accuracy"] * 100 for s in SEED_KEYS]
        wcs   = [data[pk]["seeds"][s]["wall_clock_time"]     for s in SEED_KEYS]
        rho_actual.append(np.mean(rhos))
        acc_means.append(np.mean(bests))
        acc_stds_v.append(np.std(bests))
        wc_means.append(np.mean(wcs))

    x = np.arange(len(privs))
    ax.bar(x, acc_means, 0.5, yerr=acc_stds_v,
           color=[PRIV_COLORS[p] for p in privs],
           capsize=4, edgecolor='black', linewidth=0.5)

    ax_wc = ax.twinx()
    ax_wc.plot(x, wc_means, 'k--', marker='o', markersize=6,
               markerfacecolor='white', markeredgewidth=1.2,
               label='Wall-Clock', zorder=5)
    ax_wc.set_ylabel('Wall-Clock Time (s)', color='gray')
    ax_wc.set_ylim(800, 1100)
    ax_wc.tick_params(axis='y', labelcolor='gray')

    ax.set_xlabel(r'Privacy Level (avg $\tilde{\rho}$)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.2f}' for r in rho_actual])
    ax.set_ylim(30, 55)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10a_privacy_acc.pdf")
    plt.close(fig)
    print("  ✅ Fig 10a: privacy_acc")


# ====================================================================
# Fig 10b: Privacy — Convergence curves  (Exp 6)
# ====================================================================
def fig10b():
    data = load("exp6_privacy_async.json")
    privs = ["no_privacy", "mild_privacy", "moderate_privacy", "strong_privacy"]

    fig, ax = plt.subplots(figsize=(5, 4))
    rounds = np.arange(1, 51)

    for pk in privs:
        accs = []
        for r in range(50):
            vals = [data[pk]["seeds"][s]["history"][r]["accuracy"] * 100
                    for s in SEED_KEYS
                    if r < len(data[pk]["seeds"][s]["history"])]
            accs.append(np.mean(vals))
        ax.plot(rounds, accs, color=PRIV_COLORS[pk],
                marker=PRIV_MARKERS[pk], markevery=5,
                label=PRIV_LABELS[pk], linewidth=1.8, **MK)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend(fontsize=8, framealpha=0.95, edgecolor='gray',
              loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10b_privacy_conv.pdf")
    plt.close(fig)
    print("  ✅ Fig 10b: privacy_conv")


# ====================================================================
# BONUS: Table I  (LaTeX-ready)
# ====================================================================
def table1_latex():
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "Sync-Full",
               "FedBuff-FD", "Random-Async", "Full-Async"]
    sync_wc = np.mean([data[f"Sync-Greedy_seed{s}"]["wall_clock_time"]
                       for s in SEEDS3])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of six methods on CIFAR-100 "
        r"(M=50, $\sigma$=0.5, $\alpha$=0.3). Mean$\pm$std over 3 seeds.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Method & Final Acc (\%) & Best Acc (\%) & WC (s) & Speedup \\",
        r"\midrule",
    ]
    for m in methods:
        finals = [data[f"{m}_seed{s}"]["final_accuracy"]  * 100 for s in SEEDS3]
        bests  = [data[f"{m}_seed{s}"]["best_accuracy"]   * 100 for s in SEEDS3]
        wcs    = [data[f"{m}_seed{s}"]["wall_clock_time"]        for s in SEEDS3]
        fm, fs = mean_std(finals)
        bm, bs = mean_std(bests)
        wm, ws = mean_std(wcs)
        spd = sync_wc / wm
        if m == "DASH":
            lines.append(
                f"\\textbf{{{m}}} & \\textbf{{{fm:.2f}$\\pm${fs:.2f}}} & "
                f"\\textbf{{{bm:.2f}$\\pm${bs:.2f}}} & "
                f"\\textbf{{{wm:.0f}$\\pm${ws:.0f}}} & "
                f"\\textbf{{{spd:.2f}$\\times$}} \\\\")
        else:
            lines.append(
                f"{m} & {fm:.2f}$\\pm${fs:.2f} & {bm:.2f}$\\pm${bs:.2f} & "
                f"{wm:.0f}$\\pm${ws:.0f} & {spd:.2f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(FIGDIR / "table1_main.tex", "w") as f:
        f.write("\n".join(lines))
    print("  ✅ Table I: table1_main.tex")


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 60)
    print("  GENERATING ALL JPDC 2026 FIGURES (single-panel)")
    print("=" * 60)

    fig2()
    fig3()
    fig4()
    fig5()
    fig6a()
    fig6b()
    fig7()
    fig8()
    fig9a()
    fig9b()
    fig10a()
    fig10b()
    table1_latex()

    print("\n" + "=" * 60)
    figs = list(FIGDIR.glob("*.pdf"))
    texs = list(FIGDIR.glob("*.tex"))
    print(f"  DONE — {len(figs)} PDF + {len(texs)} TeX in {FIGDIR}/")
    for p in sorted(figs + texs):
        print(f"    {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
