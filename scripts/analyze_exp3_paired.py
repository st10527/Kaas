#!/usr/bin/env python3
"""Exp 3 Complete Analysis — With D_min (3a) vs Without D_min (3b)

Paired comparison of the same 6 policies, with and without D_min floor.
Shows the protective value of D_min and the optimal policy choice.
"""

import json
import numpy as np
from pathlib import Path

SEEDS = ["seed42", "seed123", "seed456"]

POLICIES_6 = [
    "fixed(5.0)", "fixed(10.0)", "fixed(20.0)",
    "adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)",
]

def load(name):
    with open(Path("results/jpdc") / name) as f:
        return json.load(f)


def get_stats(data, pol):
    d = data[pol]
    accs = [d["seeds"][s]["final_accuracy"] * 100 for s in SEEDS]
    wcs  = [d["seeds"][s]["wall_clock_time"] for s in SEEDS]
    bests = [d["seeds"][s].get("best_accuracy", d["seeds"][s]["final_accuracy"]) * 100 for s in SEEDS]
    return {
        "acc_m": np.mean(accs), "acc_s": np.std(accs),
        "wc_m": np.mean(wcs), "wc_s": np.std(wcs),
        "best": np.max(bests),
        "accs": accs, "wcs": wcs,
    }


def get_history_stats(data, pol, seed="seed42"):
    """Extract deadline & participation from per-round history."""
    h = data[pol]["seeds"][seed]["history"]
    deadlines, n_complete, n_partial, n_timeout, n_selected = [], [], [], [], []
    for r in h:
        ex = r.get("extra", {})
        if "deadline" in ex:
            deadlines.append(ex["deadline"])
        if "n_complete" in ex:
            n_complete.append(ex["n_complete"])
            n_partial.append(ex.get("n_partial", 0))
            n_timeout.append(ex.get("n_timeout", 0))
        if "n_selected" in ex:
            n_selected.append(ex["n_selected"])
    return {
        "deadlines": np.array(deadlines) if deadlines else None,
        "n_complete": np.array(n_complete) if n_complete else None,
        "n_partial": np.array(n_partial) if n_partial else None,
        "n_timeout": np.array(n_timeout) if n_timeout else None,
        "n_selected": np.array(n_selected) if n_selected else None,
    }


def main():
    d3a = load("exp3_policy_comparison.json")
    d3b = load("exp3b_policy_nofloor.json")

    print("=" * 85)
    print("  EXPERIMENT 3 — COMPLETE ANALYSIS: WITH vs WITHOUT D_min FLOOR")
    print("=" * 85)

    # ══════════════════════════════════════════════════════════════════
    # 1. PAIRED COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  TABLE: Paired Comparison — With D_min (3a) vs Without D_min (3b)")
    print("─" * 85)
    print(f"  {'Policy':<15s} │ {'With D_min':^22s} │ {'Without D_min':^22s} │ {'Δ Acc':>7s} │ {'Δ WC':>7s}")
    print(f"  {'':─<15s} ┼ {'':─<22s} ┼ {'':─<22s} ┼ {'':─>7s} ┼ {'':─>7s}")

    for pol in POLICIES_6:
        a = get_stats(d3a, pol)
        b = get_stats(d3b, pol)
        d_acc = a["acc_m"] - b["acc_m"]
        d_wc = a["wc_m"] - b["wc_m"]
        print(f"  {pol:<15s} │ {a['acc_m']:5.2f}%  {a['wc_m']:6.0f}s      │ "
              f"{b['acc_m']:5.2f}%  {b['wc_m']:6.0f}s      │ {d_acc:+5.2f}pp│ {d_wc:+6.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # 2. FULL TABLE WITH STD
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  WITH D_min (Exp 3a) — mean ± std, 3 seeds")
    print("─" * 85)
    print(f"  {'Policy':<15s} │ {'Accuracy (%)':^16s} │ {'Best (%)':^8s} │ {'WC (s)':^16s}")
    print(f"  {'':─<15s} ┼ {'':─<16s} ┼ {'':─<8s} ┼ {'':─<16s}")
    for pol in POLICIES_6 + (["partial(0.7)"] if "partial(0.7)" in d3a else []):
        s = get_stats(d3a, pol)
        print(f"  {pol:<15s} │ {s['acc_m']:5.2f} ± {s['acc_s']:4.2f}   │ {s['best']:5.2f}  │ {s['wc_m']:7.0f} ± {s['wc_s']:3.0f}")

    print(f"\n  WITHOUT D_min (Exp 3b) — mean ± std, 3 seeds")
    print("─" * 85)
    print(f"  {'Policy':<15s} │ {'Accuracy (%)':^16s} │ {'Best (%)':^8s} │ {'WC (s)':^16s}")
    print(f"  {'':─<15s} ┼ {'':─<16s} ┼ {'':─<8s} ┼ {'':─<16s}")
    for pol in POLICIES_6:
        s = get_stats(d3b, pol)
        print(f"  {pol:<15s} │ {s['acc_m']:5.2f} ± {s['acc_s']:4.2f}   │ {s['best']:5.2f}  │ {s['wc_m']:7.0f} ± {s['wc_s']:3.0f}")

    # ══════════════════════════════════════════════════════════════════
    # 3. DEADLINE EVOLUTION COMPARISON (seed42)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  DEADLINE EVOLUTION (seed42) — WITH D_min")
    print("─" * 85)
    checkpoints = [0, 3, 5, 10, 20, 30, 40, 49]
    header = f"  {'Policy':<15s} │" + "".join(f" R{r:<3d} │" for r in checkpoints)
    print(header)
    for pol in POLICIES_6:
        hs = get_history_stats(d3a, pol, "seed42")
        row = f"  {pol:<15s} │"
        if hs["deadlines"] is not None:
            for r in checkpoints:
                if r < len(hs["deadlines"]):
                    row += f" {hs['deadlines'][r]:5.1f}│"
                else:
                    row += "   N/A│"
        print(row)

    print(f"\n  DEADLINE EVOLUTION (seed42) — WITHOUT D_min")
    print("─" * 85)
    print(header)
    for pol in POLICIES_6:
        hs = get_history_stats(d3b, pol, "seed42")
        row = f"  {pol:<15s} │"
        if hs["deadlines"] is not None:
            for r in checkpoints:
                if r < len(hs["deadlines"]):
                    row += f" {hs['deadlines'][r]:5.1f}│"
                else:
                    row += "   N/A│"
        print(row)

    # ══════════════════════════════════════════════════════════════════
    # 4. PARTICIPATION & TIMEOUT COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  PARTICIPATION STATS (mean over 50 rounds, seed42)")
    print("─" * 85)
    print(f"  {'Policy':<15s} │ {'── With D_min ──':^30s} │ {'── Without D_min ──':^30s}")
    print(f"  {'':─<15s} │ {'comp':>5s} {'part':>5s} {'t.out':>5s} {'sel':>5s} │ "
          f"{'comp':>5s} {'part':>5s} {'t.out':>5s} {'sel':>5s}")

    for pol in POLICIES_6:
        ha = get_history_stats(d3a, pol, "seed42")
        hb = get_history_stats(d3b, pol, "seed42")
        row = f"  {pol:<15s} │"
        if ha["n_complete"] is not None:
            row += f" {ha['n_complete'].mean():5.1f} {ha['n_partial'].mean():5.1f} {ha['n_timeout'].mean():5.1f} {ha['n_selected'].mean():5.1f} │"
        else:
            row += "   N/A   N/A   N/A   N/A │"
        if hb["n_complete"] is not None:
            row += f" {hb['n_complete'].mean():5.1f} {hb['n_partial'].mean():5.1f} {hb['n_timeout'].mean():5.1f} {hb['n_selected'].mean():5.1f}"
        else:
            row += "   N/A   N/A   N/A   N/A"
        print(row)

    # ══════════════════════════════════════════════════════════════════
    # 5. ACCURACY CONVERGENCE (seed-averaged)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  ACCURACY CONVERGENCE (mean of 3 seeds) — WITHOUT D_min")
    print("─" * 85)
    conv_rounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    header = f"  {'Policy':<15s} │" + "".join(f" R{r:<4d}│" for r in conv_rounds)
    print(header)
    for pol in POLICIES_6:
        row = f"  {pol:<15s} │"
        for r in conv_rounds:
            accs_r = []
            for s in SEEDS:
                h = d3b[pol]["seeds"][s]["history"]
                if r < len(h):
                    accs_r.append(h[r]["accuracy"] * 100)
            row += f" {np.mean(accs_r):5.2f}│" if accs_r else "   N/A│"
        print(row)

    # ══════════════════════════════════════════════════════════════════
    # 6. D_min IMPACT ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  D_min IMPACT ANALYSIS — KEY METRICS")
    print("=" * 85)

    print("\n  Impact of D_min by policy (3a accuracy - 3b accuracy):")
    impacts = []
    for pol in POLICIES_6:
        a = get_stats(d3a, pol)
        b = get_stats(d3b, pol)
        d_acc = a["acc_m"] - b["acc_m"]
        d_wc = a["wc_m"] - b["wc_m"]
        impacts.append((pol, d_acc, d_wc, a["acc_m"], b["acc_m"], a["wc_m"], b["wc_m"]))
        marker = " ← D_min CRITICAL" if d_acc > 2.0 else (" ← D_min helps" if d_acc > 0.5 else " ← minimal impact")
        print(f"  {pol:<15s}  Δacc={d_acc:+6.2f}pp  ΔWC={d_wc:+7.0f}s  "
              f"({b['acc_m']:.2f}% → {a['acc_m']:.2f}%){marker}")

    # Sort by D_min impact
    impacts.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Largest D_min benefit: {impacts[0][0]} (+{impacts[0][1]:.2f}pp)")
    print(f"  Smallest D_min benefit: {impacts[-1][0]} ({impacts[-1][1]:+.2f}pp)")

    # ══════════════════════════════════════════════════════════════════
    # 7. OVERALL RANKING (all 13 configurations)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  OVERALL RANKING — ALL CONFIGURATIONS")
    print("=" * 85)

    all_results = {}
    for pol in POLICIES_6:
        s3a = get_stats(d3a, pol)
        s3b = get_stats(d3b, pol)
        all_results[f"{pol} +D_min"] = s3a
        all_results[f"{pol} -D_min"] = s3b
    if "partial(0.7)" in d3a:
        all_results["partial(0.7) +D_min"] = get_stats(d3a, "partial(0.7)")

    by_acc = sorted(all_results.items(), key=lambda x: x[1]["acc_m"], reverse=True)
    print("\n  By Accuracy:")
    for i, (name, r) in enumerate(by_acc, 1):
        print(f"  {i:2d}. {name:<28s}  {r['acc_m']:5.2f}% ± {r['acc_s']:.2f}  WC={r['wc_m']:.0f}s")

    by_wc = sorted(all_results.items(), key=lambda x: x[1]["wc_m"])
    print("\n  By Wall-Clock (fastest first):")
    for i, (name, r) in enumerate(by_wc, 1):
        print(f"  {i:2d}. {name:<28s}  {r['wc_m']:6.0f}s  acc={r['acc_m']:.2f}%")

    # ══════════════════════════════════════════════════════════════════
    # 8. CROSS-REFERENCE WITH EXP 1 DASH
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  CROSS-REFERENCE: Exp 1 DASH (adaptive p=0.85, +D_min)")
    print("=" * 85)
    print("  Exp 1 DASH:  acc=44.47%, WC=979s  (σ=0.5)")
    print("  Exp 3 uses σ=1.0 (harsher). So direct numbers differ, but trends valid.\n")

    for name, r in by_acc[:5]:
        print(f"  {name:<28s}  {r['acc_m']:5.2f}%  WC={r['wc_m']:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # 9. PAPER-READY SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  PAPER-READY SUMMARY FOR Fig 6 / Sec 5.4")
    print("=" * 85)

    # Fixed analysis
    f5_a = get_stats(d3a, "fixed(5.0)")
    f5_b = get_stats(d3b, "fixed(5.0)")
    f10_b = get_stats(d3b, "fixed(10.0)")
    f20_b = get_stats(d3b, "fixed(20.0)")
    a7_a = get_stats(d3a, "adaptive(0.7)")
    a7_b = get_stats(d3b, "adaptive(0.7)")
    a5_b = get_stats(d3b, "adaptive(0.5)")
    partial_a = get_stats(d3a, "partial(0.7)") if "partial(0.7)" in d3a else None

    print(f"""
  ▸ KEY FINDING 1: D_min prevents catastrophic failure
    fixed(5.0) without D_min: {f5_b['acc_m']:.2f}% ({f5_b['wc_m']:.0f}s)  ← COLLAPSED
    fixed(5.0) with D_min:    {f5_a['acc_m']:.2f}% ({f5_a['wc_m']:.0f}s)  ← rescued (+{f5_a['acc_m']-f5_b['acc_m']:.1f}pp)

  ▸ KEY FINDING 2: Fixed deadline = hard accuracy-speed dilemma
    D=5s:  {f5_b['acc_m']:.2f}% / {f5_b['wc_m']:.0f}s  (fast but terrible acc)
    D=10s: {f10_b['acc_m']:.2f}% / {f10_b['wc_m']:.0f}s
    D=20s: {f20_b['acc_m']:.2f}% / {f20_b['wc_m']:.0f}s  (good acc but slow)

  ▸ KEY FINDING 3: Adaptive beats fixed at every operating point
    adaptive(0.5): {a5_b['acc_m']:.2f}% / {a5_b['wc_m']:.0f}s  vs fixed(10): {f10_b['acc_m']:.2f}% / {f10_b['wc_m']:.0f}s
    adaptive(0.7): {a7_b['acc_m']:.2f}% / {a7_b['wc_m']:.0f}s  vs fixed(20): {f20_b['acc_m']:.2f}% / {f20_b['wc_m']:.0f}s

  ▸ KEY FINDING 4: D_min + adaptive = best of both worlds
    adaptive(0.7) +D_min: {a7_a['acc_m']:.2f}% / {a7_a['wc_m']:.0f}s  ← PARETO OPTIMAL
    adaptive(0.7) -D_min: {a7_b['acc_m']:.2f}% / {a7_b['wc_m']:.0f}s  (+{a7_a['acc_m']-a7_b['acc_m']:.1f}pp from D_min)
""")
    if partial_a:
        print(f"  ▸ KEY FINDING 5: Partial accept")
        print(f"    partial(0.7) +D_min: {partial_a['acc_m']:.2f}% / {partial_a['wc_m']:.0f}s")
        print(f"    vs adaptive(0.7) +D_min: {a7_a['acc_m']:.2f}% / {a7_a['wc_m']:.0f}s")
        print(f"    Partial adds {partial_a['acc_m']-a7_a['acc_m']:+.2f}pp (same WC)")

    print(f"""
  ▸ STORY FOR PAPER:
    1. Fixed deadlines fail without careful tuning (D=5→{f5_b['acc_m']:.1f}%, D=20→slow)
    2. Adaptive deadline automatically finds good operating points
    3. D_min floor provides safety net: +{f5_a['acc_m']-f5_b['acc_m']:.1f}pp for aggressive settings
    4. DASH = adaptive(p=0.85) + D_min → optimal acc-speed trade-off
""")


if __name__ == "__main__":
    main()
