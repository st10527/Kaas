#!/usr/bin/env python3
"""
KaaS-Edge Paper Figure Generator v2 — IEEE EDGE 2026
======================================================
Updated for: v_i subsampling, comm_mb metric, multi-seed budget/scale/privacy.

Usage:
    python plot_edge_figures.py --data_dir results/edge/ --out_dir figures/
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def setup_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'figure.figsize': (3.5, 2.6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })


METHOD_CONFIG = {
    'KaaS-Edge': {'color': '#D62728', 'marker': 's', 'linestyle': '-',
                  'label': 'KaaS-Edge (Proposed)', 'zorder': 10},
    'FedMD': {'color': '#1F77B4', 'marker': 'o', 'linestyle': '--',
              'label': 'FedMD [Full]', 'zorder': 5},
    'FedSKD': {'color': '#2CA02C', 'marker': '^', 'linestyle': '-.',
               'label': 'FedSKD [Selective]', 'zorder': 5},
    'FedCS-FD': {'color': '#FF7F0E', 'marker': 'D', 'linestyle': ':',
                 'label': 'FedCS-FD [Equal Alloc.]', 'zorder': 5},
    'RandomSelection': {'color': '#9467BD', 'marker': 'v', 'linestyle': '--',
                        'label': 'Random Selection', 'zorder': 5},
}
METHOD_ORDER = ['KaaS-Edge', 'FedMD', 'FedSKD', 'FedCS-FD', 'RandomSelection']
SEEDS = [42, 123, 456]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_histories(data, method_name, seeds=SEEDS):
    """Extract per-round arrays across seeds from main_comparison format."""
    all_acc, all_cost, all_comm = [], [], []
    for s in seeds:
        key = f"{method_name}_seed{s}"
        if key not in data:
            continue
        hist = data[key]['history']
        all_acc.append([h['accuracy'] for h in hist])
        all_cost.append([sum(h['energy'].values()) for h in hist])
        all_comm.append([h['extra'].get('comm_mb', 0) for h in hist])

    if not all_acc:
        return None, None, None, None, None

    n = min(len(a) for a in all_acc)
    acc = np.array([a[:n] for a in all_acc])
    cost = np.array([c[:n] for c in all_cost])
    comm = np.array([c[:n] for c in all_comm])
    return acc, cost, comm, acc.mean(0), acc.std(0)


# ============================================================================
# Figure 3: Accuracy vs Round
# ============================================================================

def plot_accuracy_vs_round(data, out_dir, n_rounds=30):
    fig, ax = plt.subplots()
    rounds = np.arange(1, n_rounds + 1)

    for method in METHOD_ORDER:
        acc, _, _, mean_acc, std_acc = extract_histories(data, method)
        if mean_acc is None:
            continue
        cfg = METHOD_CONFIG[method]
        n = min(n_rounds, len(mean_acc))
        r = rounds[:n]; ma = mean_acc[:n] * 100; sa = std_acc[:n] * 100

        step = max(1, n // 8)
        ax.plot(r, ma, color=cfg['color'], linestyle=cfg['linestyle'],
                marker=cfg['marker'], markevery=list(range(0, n, step)),
                label=cfg['label'], zorder=cfg['zorder'])
        if acc.shape[0] > 1:
            ax.fill_between(r, ma - sa, ma + sa, alpha=0.15, color=cfg['color'])

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(1, n_rounds)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray',
              ncol=1, handlelength=2.0)

    fig.savefig(os.path.join(out_dir, 'fig3_accuracy_vs_round.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig3_accuracy_vs_round.png'))
    plt.close(fig)
    print("  ✓ fig3_accuracy_vs_round")


# ============================================================================
# Figure 4: Accuracy vs Cumulative Communication (MB) — THE key figure
# ============================================================================

def plot_accuracy_vs_comm(data, out_dir):
    """Accuracy vs cumulative comm MB (log-scale x). Pareto dominance figure."""
    fig, ax = plt.subplots()

    for method in METHOD_ORDER:
        acc, _, comm, mean_acc, _ = extract_histories(data, method)
        if mean_acc is None:
            continue
        cfg = METHOD_CONFIG[method]
        n = len(mean_acc)

        # Cumulative comm MB
        cum_comms = [np.cumsum(comm[s]) for s in range(comm.shape[0])]
        cum_mean = np.mean(cum_comms, axis=0)

        step = max(1, n // 8)
        ax.plot(cum_mean, mean_acc * 100, color=cfg['color'],
                linestyle=cfg['linestyle'], marker=cfg['marker'],
                markevery=list(range(0, n, step)),
                label=cfg['label'], zorder=cfg['zorder'])

    ax.set_xscale('log')
    ax.set_xlabel('Cumulative Communication (MB, log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray',
              ncol=1, handlelength=2.0)

    fig.savefig(os.path.join(out_dir, 'fig4_accuracy_vs_comm.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig4_accuracy_vs_comm.png'))
    plt.close(fig)
    print("  ✓ fig4_accuracy_vs_comm")


# ============================================================================
# Figure 5: Budget Sensitivity (multi-seed)
# ============================================================================

def plot_budget_sensitivity(data, out_dir):
    """Budget vs accuracy (mean±std) + n_selected."""
    budgets, accs, stds, n_sels = [], [], [], []

    for key in sorted(data.keys(), key=lambda x: float(x.split('=')[1])):
        B = float(key.split('=')[1])
        entry = data[key]
        budgets.append(B)
        accs.append(entry['final_accuracy'] * 100)
        stds.append(entry.get('final_accuracy_std', 0) * 100)

        # n_selected from first seed's history
        hist = entry.get('history', [])
        if hist:
            avg_nsel = np.mean([h['extra'].get('n_selected', 0) for h in hist])
        else:
            avg_nsel = 0
        n_sels.append(avg_nsel)

    budgets = np.array(budgets)
    accs = np.array(accs)
    stds = np.array(stds)
    n_sels = np.array(n_sels)

    fig, ax1 = plt.subplots()

    color1 = '#D62728'
    ax1.errorbar(budgets, accs, yerr=stds, fmt='s-', color=color1,
                 capsize=3, markersize=5, linewidth=1.5, label='Accuracy')
    ax1.set_xlabel('Budget $B$')
    ax1.set_ylabel('Test Accuracy (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#1F77B4'
    ax2.plot(budgets, n_sels, 'o--', color=color2, markersize=5,
             linewidth=1.2, label='Selected Devices')
    ax2.set_ylabel('Avg. Selected Devices', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
               framealpha=0.9, edgecolor='gray')

    fig.savefig(os.path.join(out_dir, 'fig5_budget_sensitivity.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig5_budget_sensitivity.png'))
    plt.close(fig)
    print("  ✓ fig5_budget_sensitivity")


# ============================================================================
# Figure 6: Device Scalability (multi-seed)
# ============================================================================

def plot_device_scalability(data, out_dir):
    ms, accs, stds, parts, n_sels = [], [], [], [], []

    for key in sorted(data.keys(), key=lambda x: int(x.split('=')[1])):
        M = int(key.split('=')[1])
        entry = data[key]
        ms.append(M)
        accs.append(entry['final_accuracy'] * 100)
        stds.append(entry.get('final_accuracy_std', 0) * 100)

        hist = entry.get('history', [])
        if hist:
            parts.append(np.mean([h['participation_rate'] for h in hist]) * 100)
            n_sels.append(np.mean([h['extra'].get('n_selected', 0) for h in hist]))
        else:
            parts.append(0); n_sels.append(0)

    ms = np.array(ms); accs = np.array(accs); stds = np.array(stds)
    parts = np.array(parts); n_sels = np.array(n_sels)

    fig, ax1 = plt.subplots()

    color1 = '#D62728'
    ax1.errorbar(ms, accs, yerr=stds, fmt='s-', color=color1,
                 capsize=3, markersize=5, linewidth=1.5, label='Accuracy')
    ax1.set_xlabel('Number of Devices $M$')
    ax1.set_ylabel('Test Accuracy (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(ms)

    ax2 = ax1.twinx()
    color2 = '#1F77B4'
    ax2.plot(ms, parts, 'o--', color=color2, markersize=5,
             linewidth=1.2, label='Participation Rate')
    ax2.bar(ms, n_sels, width=np.diff(np.append(ms, ms[-1]+10))*0.3,
            alpha=0.2, color=color2, label='Avg. Selected')
    ax2.set_ylabel('Part. Rate (%) / Selected', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               framealpha=0.9, edgecolor='gray')

    fig.savefig(os.path.join(out_dir, 'fig6_device_scalability.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig6_device_scalability.png'))
    plt.close(fig)
    print("  ✓ fig6_device_scalability")


# ============================================================================
# Figure 7: Privacy Impact (multi-seed)
# ============================================================================

def plot_privacy_impact(data, out_dir):
    fig, ax = plt.subplots()

    scenarios = {
        'no_privacy':     {'label': 'No Privacy ($\\rho_i=1.0$)',
                           'color': '#2CA02C', 'marker': 'o'},
        'mild_privacy':   {'label': 'Mild ($\\bar{\\rho}\\approx 0.8$)',
                           'color': '#1F77B4', 'marker': 's'},
        'mixed_privacy':  {'label': 'Mixed (Default)',
                           'color': '#D62728', 'marker': 'D'},
        'strong_privacy': {'label': 'Strong ($\\bar{\\rho}\\approx 0.1$)',
                           'color': '#FF7F0E', 'marker': '^'},
    }

    for key, cfg in scenarios.items():
        if key not in data:
            continue
        entry = data[key]

        # Multi-seed: average histories
        if 'seeds' in entry:
            seed_hists = []
            for sk, sv in entry['seeds'].items():
                seed_hists.append([h['accuracy'] for h in sv['history']])
            n = min(len(h) for h in seed_hists)
            acc_arr = np.array([h[:n] for h in seed_hists])
            mean_acc = acc_arr.mean(0) * 100
            std_acc = acc_arr.std(0) * 100
        else:
            hist = entry['history']
            n = len(hist)
            mean_acc = np.array([h['accuracy'] for h in hist]) * 100
            std_acc = np.zeros(n)

        rounds = np.arange(1, n + 1)
        step = max(1, n // 8)
        ax.plot(rounds, mean_acc, color=cfg['color'], marker=cfg['marker'],
                markevery=list(range(0, n, step)), label=cfg['label'], linewidth=1.2)
        if std_acc.max() > 0.5:
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.15, color=cfg['color'])

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(1, n)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

    fig.savefig(os.path.join(out_dir, 'fig7_privacy_impact.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig7_privacy_impact.png'))
    plt.close(fig)
    print("  ✓ fig7_privacy_impact")


# ============================================================================
# Table 1: Summary (LaTeX) — now with Comm MB
# ============================================================================

def generate_table(main_data, out_dir):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Performance Comparison (Mean $\pm$ Std, 3 Seeds, 50 Rounds)}",
        r"\label{tab:main_comparison}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Accuracy (\%) & Comm./Round (MB) & Total Comm. (MB) & Ratio \\",
        r"\midrule",
    ]

    kaas_total_comm = None
    for method in METHOD_ORDER:
        accs, comms_per_round, total_comms = [], [], []
        for s in SEEDS:
            key = f"{method}_seed{s}"
            if key not in main_data: continue
            entry = main_data[key]
            accs.append(entry['final_accuracy'] * 100)
            hist = entry['history']
            round_comm = [h['extra'].get('comm_mb', 0) for h in hist]
            comms_per_round.append(np.mean(round_comm))
            total_comms.append(np.sum(round_comm))

        if not accs: continue
        ma, sa = np.mean(accs), np.std(accs)
        mc = np.mean(comms_per_round)
        tc = np.mean(total_comms)

        if method == 'KaaS-Edge':
            kaas_total_comm = tc

        ratio = tc / kaas_total_comm if kaas_total_comm and kaas_total_comm > 0 else 1.0

        if method == 'KaaS-Edge':
            name = r"\textbf{KaaS-Edge (Proposed)}"
            acc_s = f"\\textbf{{{ma:.1f} $\\pm$ {sa:.1f}}}"
            mc_s = f"\\textbf{{{mc:.1f}}}"
            tc_s = f"\\textbf{{{tc:.0f}}}"
            r_s = r"\textbf{1$\times$}"
        else:
            name = METHOD_CONFIG[method]['label']
            acc_s = f"{ma:.1f} $\\pm$ {sa:.1f}"
            mc_s = f"{mc:.1f}" if mc < 100 else f"{mc:.0f}"
            tc_s = f"{tc:.0f}" if tc < 10000 else f"{tc/1000:.1f}K"
            r_s = f"{ratio:.0f}$\\times$"

        lines.append(f"{name} & {acc_s} & {mc_s} & {tc_s} & {r_s} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    with open(os.path.join(out_dir, "table1_summary.tex"), 'w') as f:
        f.write(tex)
    print("  ✓ table1_summary.tex")
    print(); print(tex)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='results/edge/')
    parser.add_argument('--out_dir', type=str, default='figures/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_ieee_style()

    print(f"\nKaaS-Edge Figure Generator v2")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.out_dir}\n")

    main_data = load_json(os.path.join(args.data_dir, 'main_comparison.json'))

    print("Generating figures...")
    plot_accuracy_vs_round(main_data, args.out_dir, n_rounds=30)
    plot_accuracy_vs_comm(main_data, args.out_dir)

    try:
        budget_data = load_json(os.path.join(args.data_dir, 'budget_sensitivity.json'))
        plot_budget_sensitivity(budget_data, args.out_dir)
    except Exception as e:
        print(f"  ⚠ budget: {e}")

    try:
        scale_data = load_json(os.path.join(args.data_dir, 'device_scalability.json'))
        plot_device_scalability(scale_data, args.out_dir)
    except Exception as e:
        print(f"  ⚠ scale: {e}")

    try:
        privacy_data = load_json(os.path.join(args.data_dir, 'privacy_impact.json'))
        plot_privacy_impact(privacy_data, args.out_dir)
    except Exception as e:
        print(f"  ⚠ privacy: {e}")

    print("\nGenerating table...")
    generate_table(main_data, args.out_dir)

    print(f"\nDone! All outputs in {args.out_dir}")


if __name__ == '__main__':
    main()
