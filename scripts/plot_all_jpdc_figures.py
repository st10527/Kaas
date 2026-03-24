#!/usr/bin/env python3
"""Generate ALL JPDC 2026 paper figures (Fig 2–10).

Output: figures/jpdc/fig{N}_{name}.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    'lines.markersize': 5,
})

RESULTS = Path("results/jpdc")
FIGDIR = Path("figures/jpdc")
FIGDIR.mkdir(parents=True, exist_ok=True)

SEEDS3 = [42, 123, 456]
SEED_KEYS = ["seed42", "seed123", "seed456"]

# Color palette — consistent across all figures
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
PRIV_COLORS = {
    'no_privacy': '#1F77B4', 'mild_privacy': '#2CA02C',
    'moderate_privacy': '#FF7F0E', 'strong_privacy': '#D62728',
}
PRIV_LABELS = {
    'no_privacy': r'None ($\tilde{\rho}$≈0.98)',
    'mild_privacy': r'Mild ($\tilde{\rho}$≈0.80)',
    'moderate_privacy': r'Moderate ($\tilde{\rho}$≈0.55)',
    'strong_privacy': r'Strong ($\tilde{\rho}$≈0.10)',
}


def load(fn):
    with open(RESULTS / fn) as f:
        return json.load(f)


def mean_std(vals):
    return np.mean(vals), np.std(vals)


# ============================================================================
# Fig 2: Accuracy vs Round (Exp 1)
# ============================================================================
def fig2():
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD", "Random-Async", "Full-Async", "Sync-Full"]

    fig, ax = plt.subplots(figsize=(6, 4))
    rounds = np.arange(1, 51)

    for m in methods:
        acc_per_round = []
        for r in range(50):
            accs = [data[f"{m}_seed{s}"]["history"][r]["accuracy"] * 100
                    for s in SEEDS3 if r < len(data[f"{m}_seed{s}"]["history"])]
            acc_per_round.append(np.mean(accs))
        ax.plot(rounds, acc_per_round, color=COLORS[m], marker=MARKERS[m],
                markevery=5, label=m, zorder=3 if m == 'DASH' else 2)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(1, 50)
    ax.legend(loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('(a) Accuracy vs. Round')
    fig.savefig(FIGDIR / "fig2_acc_vs_round.pdf")
    plt.close(fig)
    print("  ✅ Fig 2: acc_vs_round")


# ============================================================================
# Fig 3: Accuracy vs Wall-Clock (Exp 1) — the "killer" figure
# ============================================================================
def fig3():
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD", "Random-Async", "Full-Async", "Sync-Full"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for m in methods:
        wc_per_round = []
        acc_per_round = []
        for r in range(50):
            wcs = []
            accs = []
            for s in SEEDS3:
                h = data[f"{m}_seed{s}"]["history"]
                if r < len(h):
                    wcs.append(h[r].get("wall_clock_time", 0))
                    accs.append(h[r]["accuracy"] * 100)
            wc_per_round.append(np.mean(wcs))
            acc_per_round.append(np.mean(accs))
        ax.plot(wc_per_round, acc_per_round, color=COLORS[m], marker=MARKERS[m],
                markevery=5, label=m, zorder=3 if m == 'DASH' else 2)

    ax.set_xlabel('Wall-Clock Time (s)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(0, 3500)   # exclude Sync-Full tail (59K) for readability
    ax.legend(loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b) Accuracy vs. Wall-Clock Time')
    fig.savefig(FIGDIR / "fig3_acc_vs_wallclock.pdf")
    plt.close(fig)
    print("  ✅ Fig 3: acc_vs_wallclock")


# ============================================================================
# Fig 4 & 5: Straggler Sweep (Exp 2) — combined into one 2-panel figure
# ============================================================================
def fig4_5():
    data = load("exp2_straggler_sweep.json")
    sigmas = [0.0, 0.3, 0.5, 1.0, 1.5]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for m in methods:
        accs = []
        acc_stds = []
        wcs = []
        wc_stds = []
        for sig in sigmas:
            sk = f"sigma={sig}"
            finals = [data[sk][s][m]["final_accuracy"] * 100 for s in SEED_KEYS]
            ws = [data[sk][s][m]["wall_clock_time"] for s in SEED_KEYS]
            accs.append(np.mean(finals))
            acc_stds.append(np.std(finals))
            wcs.append(np.mean(ws))
            wc_stds.append(np.std(ws))

        ax1.errorbar(sigmas, accs, yerr=acc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3)
        ax2.errorbar(sigmas, wcs, yerr=wc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3)

    ax1.set_xlabel(r'Straggler Severity $\sigma$')
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title('(a) Accuracy vs. σ')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r'Straggler Severity $\sigma$')
    ax2.set_ylabel('Wall-Clock Time (s)')
    ax2.set_title('(b) Wall-Clock vs. σ')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_5_straggler_sweep.pdf")
    plt.close(fig)
    print("  ✅ Fig 4-5: straggler_sweep (2 panels)")


# ============================================================================
# Fig 6: Policy Comparison + D_min Ablation (Exp 3 + 3b) — scatter + timeline
# ============================================================================
def fig6():
    d3 = load("exp3_policy_comparison.json")
    d3b = load("exp3b_policy_nofloor.json")
    policies = ["fixed(5.0)", "fixed(10.0)", "fixed(20.0)",
                "adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Accuracy vs WC scatter
    policy_colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))

    for i, p in enumerate(policies):
        # +D_min
        with_acc = np.mean([d3[p]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS])
        with_wc = np.mean([d3[p]["seeds"][s]["wall_clock_time"] for s in SEED_KEYS])
        # -D_min
        wo_acc = np.mean([d3b[p]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS])
        wo_wc = np.mean([d3b[p]["seeds"][s]["wall_clock_time"] for s in SEED_KEYS])

        label_base = p.replace("(", " ").replace(")", "")
        ax1.scatter(with_wc, with_acc, color=policy_colors[i], marker='o',
                    s=80, zorder=3, edgecolors='black', linewidths=0.5,
                    label=f'{label_base} +$D_{{min}}$')
        ax1.scatter(wo_wc, wo_acc, color=policy_colors[i], marker='x',
                    s=60, zorder=3, linewidths=1.5)

        # Arrow from -D_min to +D_min
        ax1.annotate('', xy=(with_wc, with_acc), xytext=(wo_wc, wo_acc),
                     arrowprops=dict(arrowstyle='->', color=policy_colors[i],
                                     alpha=0.4, lw=1))

    ax1.set_xlabel('Wall-Clock Time (s)')
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title(r'(a) Policy Comparison (● +$D_{min}$, × −$D_{min}$)')
    ax1.legend(fontsize=7, loc='lower right', ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Deadline evolution — +D_min (solid) vs -D_min (dashed)
    d3b_data = load("exp3b_policy_nofloor.json")
    policies_to_show = ["adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)"]
    colors_dl = plt.cm.Set1(np.linspace(0, 0.5, len(policies_to_show)))

    for i, p in enumerate(policies_to_show):
        # +D_min
        h = d3[p]["seeds"]["seed42"].get("history", [])
        dl_with = [r.get("extra", {}).get("deadline", None) for r in h]
        dl_with = [d for d in dl_with if d is not None]
        if dl_with:
            ax2.plot(range(1, len(dl_with) + 1), dl_with,
                     color=colors_dl[i], linewidth=1.8,
                     label=f'{p} +$D_{{min}}$')

        # -D_min
        h2 = d3b_data[p]["seeds"]["seed42"].get("history", [])
        dl_wo = [r.get("extra", {}).get("deadline", None) for r in h2]
        dl_wo = [d for d in dl_wo if d is not None]
        if dl_wo:
            ax2.plot(range(1, len(dl_wo) + 1), dl_wo,
                     color=colors_dl[i], linewidth=1.2, linestyle='--',
                     label=f'{p} −$D_{{min}}$')

    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Deadline D (s)')
    ax2.set_title(r'(b) Deadline Evolution (solid: +$D_{min}$, dashed: −$D_{min}$)')
    ax2.legend(framealpha=0.9, fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_policy_dmin.pdf")
    plt.close(fig)
    print("  ✅ Fig 6: policy_dmin (2 panels)")


# ============================================================================
# Fig 7 & 8: Scalability (Exp 4) — combined into one 2-panel figure
# ============================================================================
def fig7_8():
    data = load("exp4_scalability.json")
    ms = [20, 50, 100, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for m in methods:
        accs, acc_stds = [], []
        wcs, wc_stds = [], []
        for M in ms:
            mk = f"M={M}"
            finals = [data[mk]["seeds"][s][m]["final_accuracy"] * 100 for s in SEED_KEYS]
            ws = [data[mk]["seeds"][s][m]["wall_clock_time"] for s in SEED_KEYS]
            accs.append(np.mean(finals))
            acc_stds.append(np.std(finals))
            wcs.append(np.mean(ws))
            wc_stds.append(np.std(ws))

        ax1.errorbar(ms, accs, yerr=acc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3)
        ax2.errorbar(ms, wcs, yerr=wc_stds, color=COLORS[m],
                     marker=MARKERS[m], label=m, capsize=3)

    # Add speedup annotation on ax2
    for M in ms:
        mk = f"M={M}"
        dash_wc = np.mean([data[mk]["seeds"][s]["DASH"]["wall_clock_time"] for s in SEED_KEYS])
        sync_wc = np.mean([data[mk]["seeds"][s]["Sync-Greedy"]["wall_clock_time"] for s in SEED_KEYS])
        spd = sync_wc / dash_wc
        ax2.annotate(f'{spd:.1f}×', xy=(M, dash_wc), fontsize=8,
                     xytext=(0, -15), textcoords='offset points',
                     ha='center', color=COLORS['DASH'], fontweight='bold')

    ax1.set_xlabel('Number of Devices (M)')
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title('(a) Accuracy vs. M')
    ax1.set_xticks(ms)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Number of Devices (M)')
    ax2.set_ylabel('Wall-Clock Time (s)')
    ax2.set_title('(b) Wall-Clock vs. M')
    ax2.set_xticks(ms)
    ax2.legend(framealpha=0.9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_8_scalability.pdf")
    plt.close(fig)
    print("  ✅ Fig 7-8: scalability (2 panels)")


# ============================================================================
# Fig 9: EMNIST Cross-Dataset (Exp 5) — 2-panel: acc bar + WC bar
# ============================================================================
def fig9():
    data = load("exp5_emnist.json")
    ms = [50, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(ms))
    width = 0.22

    for i, m in enumerate(methods):
        bests, best_stds = [], []
        wcs_vals, wc_stds = [], []
        for M in ms:
            mk = f"M={M}"
            b = [data[mk]["seeds"][s][m]["best_accuracy"] * 100 for s in SEED_KEYS]
            w = [data[mk]["seeds"][s][m]["wall_clock_time"] for s in SEED_KEYS]
            bests.append(np.mean(b))
            best_stds.append(np.std(b))
            wcs_vals.append(np.mean(w))
            wc_stds.append(np.std(w))

        bars1 = ax1.bar(x + i * width, bests, width, yerr=best_stds,
                        color=COLORS[m], label=m, capsize=3, edgecolor='black', linewidth=0.5)
        bars2 = ax2.bar(x + i * width, wcs_vals, width, yerr=wc_stds,
                        color=COLORS[m], label=m, capsize=3, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Number of Devices (M)')
    ax1.set_ylabel('Best Accuracy (%)')
    ax1.set_title('(a) EMNIST Best Accuracy')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'M={M}' for M in ms])
    ax1.set_ylim(60, 90)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.set_xlabel('Number of Devices (M)')
    ax2.set_ylabel('Wall-Clock Time (s)')
    ax2.set_title('(b) EMNIST Wall-Clock')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'M={M}' for M in ms])
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Speedup annotations
    for j, M in enumerate(ms):
        mk = f"M={M}"
        dash_wc = np.mean([data[mk]["seeds"][s]["DASH"]["wall_clock_time"] for s in SEED_KEYS])
        sync_wc = np.mean([data[mk]["seeds"][s]["Sync-Greedy"]["wall_clock_time"] for s in SEED_KEYS])
        spd = sync_wc / dash_wc
        ax2.annotate(f'{spd:.1f}× speedup', xy=(j + 0.5 * width, dash_wc),
                     fontsize=9, xytext=(0, 10), textcoords='offset points',
                     ha='center', color=COLORS['DASH'], fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9_emnist.pdf")
    plt.close(fig)
    print("  ✅ Fig 9: emnist (2 panels)")


# ============================================================================
# Fig 10: Privacy Sweep (Exp 6) — 2-panel: acc vs ρ̃ + convergence curves
# ============================================================================
def fig10():
    data = load("exp6_privacy_async.json")
    privs = ["no_privacy", "mild_privacy", "moderate_privacy", "strong_privacy"]
    rho_targets = [1.0, 0.8, 0.55, 0.1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Bar chart accuracy + WC line
    rho_actual = []
    acc_means = []
    acc_stds_v = []
    wc_means = []

    for pk in privs:
        rhos = [data[pk]["seeds"][s]["avg_rho"] for s in SEED_KEYS]
        bests = [data[pk]["seeds"][s]["best_accuracy"] * 100 for s in SEED_KEYS]
        wcs = [data[pk]["seeds"][s]["wall_clock_time"] for s in SEED_KEYS]
        rho_actual.append(np.mean(rhos))
        acc_means.append(np.mean(bests))
        acc_stds_v.append(np.std(bests))
        wc_means.append(np.mean(wcs))

    x = np.arange(len(privs))
    bars = ax1.bar(x, acc_means, 0.5, yerr=acc_stds_v, color=[PRIV_COLORS[p] for p in privs],
                   capsize=4, edgecolor='black', linewidth=0.5)

    # Add WC on secondary axis
    ax1_wc = ax1.twinx()
    ax1_wc.plot(x, wc_means, 'k--o', markersize=6, label='Wall-Clock', zorder=5)
    ax1_wc.set_ylabel('Wall-Clock Time (s)', color='gray')
    ax1_wc.set_ylim(800, 1100)
    ax1_wc.tick_params(axis='y', labelcolor='gray')

    ax1.set_xlabel(r'Privacy Level (avg $\tilde{\rho}$)')
    ax1.set_ylabel('Best Accuracy (%)')
    ax1.set_title(r'(a) Accuracy & WC vs. $\tilde{\rho}$')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r:.2f}' for r in rho_actual])
    ax1.set_ylim(30, 55)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel (b): Convergence curves
    rounds = np.arange(1, 51)
    for pk in privs:
        accs = []
        for r in range(50):
            vals = [data[pk]["seeds"][s]["history"][r]["accuracy"] * 100
                    for s in SEED_KEYS if r < len(data[pk]["seeds"][s]["history"])]
            accs.append(np.mean(vals))
        ax2.plot(rounds, accs, color=PRIV_COLORS[pk], label=PRIV_LABELS[pk], linewidth=1.8)

    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('(b) Convergence under Different Privacy')
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 50)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_privacy.pdf")
    plt.close(fig)
    print("  ✅ Fig 10: privacy (2 panels)")


# ============================================================================
# BONUS: Table I as a LaTeX-ready table
# ============================================================================
def table1_latex():
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "Sync-Full", "FedBuff-FD", "Random-Async", "Full-Async"]

    sync_wc = np.mean([data[f"Sync-Greedy_seed{s}"]["wall_clock_time"] for s in SEEDS3])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of six methods on CIFAR-100 (M=50, $\sigma$=0.5, $\alpha$=0.3). Mean$\pm$std over 3 seeds.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Method & Final Acc (\%) & Best Acc (\%) & WC (s) & Speedup \\",
        r"\midrule",
    ]

    for m in methods:
        finals = [data[f"{m}_seed{s}"]["final_accuracy"] * 100 for s in SEEDS3]
        bests = [data[f"{m}_seed{s}"]["best_accuracy"] * 100 for s in SEEDS3]
        wcs = [data[f"{m}_seed{s}"]["wall_clock_time"] for s in SEEDS3]
        fm, fs = mean_std(finals)
        bm, bs = mean_std(bests)
        wm, ws = mean_std(wcs)
        spd = sync_wc / wm

        m_fmt = r"\textbf{DASH}" if m == "DASH" else m.replace("-", r"\text{-}")
        if m == "DASH":
            lines.append(f"\\textbf{{{m}}} & \\textbf{{{fm:.2f}$\\pm${fs:.2f}}} & "
                         f"\\textbf{{{bm:.2f}$\\pm${bs:.2f}}} & "
                         f"\\textbf{{{wm:.0f}$\\pm${ws:.0f}}} & \\textbf{{{spd:.2f}$\\times$}} \\\\")
        else:
            lines.append(f"{m} & {fm:.2f}$\\pm${fs:.2f} & {bm:.2f}$\\pm${bs:.2f} & "
                         f"{wm:.0f}$\\pm${ws:.0f} & {spd:.2f}$\\times$ \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(FIGDIR / "table1_main.tex", "w") as f:
        f.write("\n".join(lines))
    print("  ✅ Table I: table1_main.tex")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  GENERATING ALL JPDC 2026 FIGURES")
    print("=" * 60)

    fig2()
    fig3()
    fig4_5()
    fig6()
    fig7_8()
    fig9()
    fig10()
    table1_latex()

    print("\n" + "=" * 60)
    figs = list(FIGDIR.glob("*.pdf"))
    texs = list(FIGDIR.glob("*.tex"))
    print(f"  DONE — {len(figs)} PDF figures + {len(texs)} TeX files in {FIGDIR}/")
    for p in sorted(figs + texs):
        print(f"    {p.name} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
