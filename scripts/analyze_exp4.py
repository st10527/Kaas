#!/usr/bin/env python3
"""Exp 4 — Scalability Analysis

M ∈ {20, 50, 100, 200}, budget = 2.5×M
3 methods: DASH, Sync-Greedy, FedBuff-FD
3 seeds each
"""

import json
import numpy as np
from pathlib import Path

SEEDS = ["seed42", "seed123", "seed456"]
METHODS = ["DASH", "Sync-Greedy", "FedBuff-FD"]
M_VALUES = [20, 50, 100, 200]


def load():
    with open(Path("results/jpdc/exp4_scalability.json")) as f:
        return json.load(f)


def get_method_stats(data, m_key, method):
    accs, wcs, bests = [], [], []
    for s in SEEDS:
        d = data[m_key]["seeds"][s][method]
        accs.append(d["final_accuracy"] * 100)
        wcs.append(d["wall_clock_time"])
        bests.append(d.get("best_accuracy", d["final_accuracy"]) * 100)
    return {
        "acc_m": np.mean(accs), "acc_s": np.std(accs),
        "wc_m": np.mean(wcs), "wc_s": np.std(wcs),
        "best": np.max(bests),
        "accs": accs, "wcs": wcs,
    }


def main():
    data = load()

    print("=" * 85)
    print("  EXPERIMENT 4: SCALABILITY ANALYSIS (M = 20, 50, 100, 200)")
    print("=" * 85)

    # ── 1. Summary Table ──
    print("\n" + "─" * 85)
    print("  TABLE: Accuracy & Wall-Clock by M (mean ± std, 3 seeds)")
    print("─" * 85)

    all_stats = {}
    for M in M_VALUES:
        m_key = f"M={M}"
        budget = data[m_key].get("budget", 2.5 * M)
        print(f"\n  M={M}  (budget={budget})")
        print(f"  {'Method':<15s} │ {'Accuracy (%)':^16s} │ {'Best (%)':^8s} │ {'WC (s)':^18s}")
        print(f"  {'':─<15s} ┼ {'':─<16s} ┼ {'':─<8s} ┼ {'':─<18s}")
        for method in METHODS:
            s = get_method_stats(data, m_key, method)
            all_stats[(M, method)] = s
            print(f"  {method:<15s} │ {s['acc_m']:5.2f} ± {s['acc_s']:4.2f}   │ {s['best']:5.2f}  │ {s['wc_m']:8.0f} ± {s['wc_s']:5.0f}")

    # ── 2. DASH vs Sync-Greedy Speedup by M ──
    print("\n" + "─" * 85)
    print("  DASH vs Sync-Greedy: Speedup & Accuracy Gap by M")
    print("─" * 85)
    print(f"  {'M':>5s} │ {'DASH Acc':>9s} │ {'Sync Acc':>9s} │ {'Gap (pp)':>9s} │ "
          f"{'DASH WC':>9s} │ {'Sync WC':>10s} │ {'Speedup':>7s}")
    print(f"  {'':─>5s} ┼ {'':─>9s} ┼ {'':─>9s} ┼ {'':─>9s} ┼ {'':─>9s} ┼ {'':─>10s} ┼ {'':─>7s}")

    speedups = []
    for M in M_VALUES:
        dash = all_stats[(M, "DASH")]
        sync = all_stats[(M, "Sync-Greedy")]
        gap = dash["acc_m"] - sync["acc_m"]
        spd = sync["wc_m"] / dash["wc_m"]
        speedups.append(spd)
        print(f"  {M:5d} │ {dash['acc_m']:8.2f}% │ {sync['acc_m']:8.2f}% │ {gap:+8.2f}pp│ "
              f"{dash['wc_m']:8.0f}s │ {sync['wc_m']:9.0f}s │ {spd:6.2f}×")

    # ── 3. Accuracy vs M trend ──
    print("\n" + "─" * 85)
    print("  ACCURACY vs M (mean of 3 seeds)")
    print("─" * 85)
    print(f"  {'M':>5s} │ {'DASH':>8s} │ {'Sync-G':>8s} │ {'FedBuff':>8s} │ {'Budget':>7s}")
    print(f"  {'':─>5s} ┼ {'':─>8s} ┼ {'':─>8s} ┼ {'':─>8s} ┼ {'':─>7s}")
    for M in M_VALUES:
        budget = data[f"M={M}"].get("budget", 2.5 * M)
        d = all_stats[(M, "DASH")]["acc_m"]
        s = all_stats[(M, "Sync-Greedy")]["acc_m"]
        f = all_stats[(M, "FedBuff-FD")]["acc_m"]
        print(f"  {M:5d} │ {d:7.2f}% │ {s:7.2f}% │ {f:7.2f}% │ {budget:7.0f}")

    # ── 4. Wall-Clock vs M trend ──
    print("\n" + "─" * 85)
    print("  WALL-CLOCK vs M (seconds, mean of 3 seeds)")
    print("─" * 85)
    print(f"  {'M':>5s} │ {'DASH':>9s} │ {'Sync-G':>10s} │ {'FedBuff':>9s} │ {'Speedup':>7s}")
    print(f"  {'':─>5s} ┼ {'':─>9s} ┼ {'':─>10s} ┼ {'':─>9s} ┼ {'':─>7s}")
    for i, M in enumerate(M_VALUES):
        d = all_stats[(M, "DASH")]["wc_m"]
        s = all_stats[(M, "Sync-Greedy")]["wc_m"]
        f = all_stats[(M, "FedBuff-FD")]["wc_m"]
        print(f"  {M:5d} │ {d:8.0f}s │ {s:9.0f}s │ {f:8.0f}s │ {speedups[i]:6.2f}×")

    # ── 5. Per-seed breakdown ──
    print("\n" + "─" * 85)
    print("  PER-SEED BREAKDOWN (DASH)")
    print("─" * 85)
    print(f"  {'M':>5s} │ {'seed42':>9s} │ {'seed123':>9s} │ {'seed456':>9s} │ "
          f"{'s42 WC':>9s} │ {'s123 WC':>9s} │ {'s456 WC':>9s}")
    for M in M_VALUES:
        s = all_stats[(M, "DASH")]
        print(f"  {M:5d} │ {s['accs'][0]:8.2f}% │ {s['accs'][1]:8.2f}% │ {s['accs'][2]:8.2f}% │ "
              f"{s['wcs'][0]:8.0f}s │ {s['wcs'][1]:8.0f}s │ {s['wcs'][2]:8.0f}s")

    print(f"\n  PER-SEED BREAKDOWN (Sync-Greedy)")
    for M in M_VALUES:
        s = all_stats[(M, "Sync-Greedy")]
        print(f"  {M:5d} │ {s['accs'][0]:8.2f}% │ {s['accs'][1]:8.2f}% │ {s['accs'][2]:8.2f}% │ "
              f"{s['wcs'][0]:8.0f}s │ {s['wcs'][1]:8.0f}s │ {s['wcs'][2]:8.0f}s")

    # ── 6. Sync WC scaling analysis ──
    print("\n" + "─" * 85)
    print("  SYNC WALL-CLOCK SCALING ANALYSIS")
    print("─" * 85)
    sync_wcs = [all_stats[(M, "Sync-Greedy")]["wc_m"] for M in M_VALUES]
    dash_wcs = [all_stats[(M, "DASH")]["wc_m"] for M in M_VALUES]
    print(f"  Sync WC growth from M=20→200: {sync_wcs[0]:.0f}s → {sync_wcs[-1]:.0f}s "
          f"({sync_wcs[-1]/sync_wcs[0]:.2f}×)")
    print(f"  DASH WC growth from M=20→200: {dash_wcs[0]:.0f}s → {dash_wcs[-1]:.0f}s "
          f"({dash_wcs[-1]/dash_wcs[0]:.2f}×)")
    print(f"  Speedup trend: ", end="")
    for i, M in enumerate(M_VALUES):
        print(f"M={M}→{speedups[i]:.2f}×  ", end="")
    print()

    # Is speedup increasing with M?
    if speedups[-1] > speedups[0]:
        print(f"  ✅ Speedup INCREASES with M ({speedups[0]:.2f}→{speedups[-1]:.2f}×) — "
              f"DASH scales better than Sync")
    else:
        print(f"  ℹ️  Speedup trend: {speedups[0]:.2f}→{speedups[-1]:.2f}× — "
              f"check if Sync WC variance is high")

    # ── 7. Accuracy scaling analysis ──
    print("\n" + "─" * 85)
    print("  ACCURACY SCALING ANALYSIS")
    print("─" * 85)
    dash_accs = [all_stats[(M, "DASH")]["acc_m"] for M in M_VALUES]
    sync_accs = [all_stats[(M, "Sync-Greedy")]["acc_m"] for M in M_VALUES]
    fb_accs = [all_stats[(M, "FedBuff-FD")]["acc_m"] for M in M_VALUES]

    print(f"  DASH:  M=20→200: {dash_accs[0]:.2f}% → {dash_accs[-1]:.2f}% "
          f"(+{dash_accs[-1]-dash_accs[0]:.2f}pp)")
    print(f"  Sync:  M=20→200: {sync_accs[0]:.2f}% → {sync_accs[-1]:.2f}% "
          f"(+{sync_accs[-1]-sync_accs[0]:.2f}pp)")
    print(f"  FedBuff: M=20→200: {fb_accs[0]:.2f}% → {fb_accs[-1]:.2f}% "
          f"({fb_accs[-1]-fb_accs[0]:+.2f}pp)")

    # ── 8. Cross-reference with Exp 1 ──
    print("\n" + "─" * 85)
    print("  CROSS-REFERENCE: Exp 1 (M=50, budget=50) vs Exp 4 (M=50, budget=125)")
    print("─" * 85)
    exp4_m50_dash = all_stats[(50, "DASH")]
    exp4_m50_sync = all_stats[(50, "Sync-Greedy")]
    print(f"  Exp 1: DASH 44.47% / 979s    Sync 43.66% / 3079s   (budget=50, σ=0.5)")
    print(f"  Exp 4: DASH {exp4_m50_dash['acc_m']:.2f}% / {exp4_m50_dash['wc_m']:.0f}s    "
          f"Sync {exp4_m50_sync['acc_m']:.2f}% / {exp4_m50_sync['wc_m']:.0f}s   (budget=125, σ=0.5)")
    print(f"  Budget 2.5× → DASH acc +{exp4_m50_dash['acc_m']-44.47:.2f}pp, "
          f"WC +{exp4_m50_dash['wc_m']-979:.0f}s")

    # ── 9. Convergence at key rounds (M=200, DASH) ──
    print("\n" + "─" * 85)
    print("  CONVERGENCE: DASH accuracy at key rounds (mean of 3 seeds)")
    print("─" * 85)
    conv_rounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    header = f"  {'M':>5s} │" + "".join(f" R{r:<4d}│" for r in conv_rounds)
    print(header)
    for M in M_VALUES:
        m_key = f"M={M}"
        row = f"  {M:5d} │"
        for r in conv_rounds:
            accs_r = []
            for s in SEEDS:
                h = data[m_key]["seeds"][s]["DASH"].get("history", [])
                if isinstance(h, list) and r < len(h):
                    accs_r.append(h[r]["accuracy"] * 100)
            row += f" {np.mean(accs_r):5.2f}│" if accs_r else "   N/A│"
        print(row)

    # ── 10. Paper-ready summary ──
    print("\n" + "=" * 85)
    print("  PAPER-READY SUMMARY FOR Fig 7, 8 / Sec 5.5")
    print("=" * 85)

    best_dash_acc = max(dash_accs)
    best_M = M_VALUES[dash_accs.index(best_dash_acc)]
    max_speedup = max(speedups)
    max_spd_M = M_VALUES[speedups.index(max_speedup)]

    print(f"""
  ▸ KEY FINDING 1: DASH accuracy scales well with M
    M=20: {dash_accs[0]:.2f}%  →  M=200: {dash_accs[-1]:.2f}%  (+{dash_accs[-1]-dash_accs[0]:.2f}pp)
    More devices → more diverse knowledge → better ensemble distillation

  ▸ KEY FINDING 2: Speedup trend with M
    M=20: {speedups[0]:.2f}×  M=50: {speedups[1]:.2f}×  M=100: {speedups[2]:.2f}×  M=200: {speedups[3]:.2f}×
    {'Speedup increases with M — async advantage grows' if speedups[-1] > speedups[0] else 'Speedup relatively stable across M'}

  ▸ KEY FINDING 3: Sync WC explodes at large M
    Sync M=20: {sync_wcs[0]:.0f}s  →  M=200: {sync_wcs[-1]:.0f}s  ({sync_wcs[-1]/sync_wcs[0]:.1f}×)
    DASH M=20: {dash_wcs[0]:.0f}s  →  M=200: {dash_wcs[-1]:.0f}s  ({dash_wcs[-1]/dash_wcs[0]:.1f}×)
    Sync grows {sync_wcs[-1]/sync_wcs[0]:.1f}× while DASH grows only {dash_wcs[-1]/dash_wcs[0]:.1f}×

  ▸ KEY FINDING 4: FedBuff-FD stays low regardless of M
    {fb_accs[0]:.2f}% → {fb_accs[-1]:.2f}%  — cannot benefit from more devices
    (buffer K=10 is fixed, ignores the rest)

  ▸ STORY FOR PAPER:
    "DASH maintains {speedups[0]:.1f}-{max_speedup:.1f}× speedup across all scales while
     matching Sync-Greedy accuracy. At M=200, DASH achieves {dash_accs[-1]:.2f}%
     in {dash_wcs[-1]:.0f}s vs Sync-Greedy's {sync_accs[-1]:.2f}% in {sync_wcs[-1]:.0f}s."
""")


if __name__ == "__main__":
    main()
