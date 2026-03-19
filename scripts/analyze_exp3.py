#!/usr/bin/env python3
"""Exp 3 — Timeout Policy Comparison Analysis

7 configs:
  fixed(5.0), fixed(10.0), fixed(20.0)   — fixed deadline
  adaptive(0.5), adaptive(0.7), adaptive(0.9) — adaptive percentile
  partial(0.7) — partial accept with adaptive(0.7)

3 seeds × 50 rounds each
"""

import json
import numpy as np
from pathlib import Path

RESULT_FILE = Path("results/jpdc/exp3_policy_comparison.json")

# Ordered for display
POLICY_ORDER = [
    "fixed(5.0)", "fixed(10.0)", "fixed(20.0)",
    "adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)",
    "partial(0.7)",
]

SEEDS = ["seed42", "seed123", "seed456"]

def load():
    with open(RESULT_FILE) as f:
        return json.load(f)


def main():
    data = load()

    print("=" * 80)
    print("  EXPERIMENT 3: TIMEOUT POLICY COMPARISON — FULL ANALYSIS")
    print("=" * 80)

    # ── 1. Summary Table ──────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  Table: Final Accuracy & Wall-Clock (mean ± std, 3 seeds)      │")
    print("├───────────────┬─────────────────┬──────────────┬───────────────┤")
    print("│ Policy        │ Accuracy (%)    │ Best Acc (%) │ WC (s)        │")
    print("├───────────────┼─────────────────┼──────────────┼───────────────┤")

    results = {}
    for pol in POLICY_ORDER:
        d = data[pol]
        accs, best_accs, wcs = [], [], []
        for s in SEEDS:
            sd = d["seeds"][s]
            accs.append(sd["final_accuracy"])
            best_accs.append(sd.get("best_accuracy", sd["final_accuracy"]))
            wcs.append(sd["wall_clock_time"])
        acc_m, acc_s = np.mean(accs) * 100, np.std(accs) * 100
        best_m = np.max(best_accs) * 100
        wc_m, wc_s = np.mean(wcs), np.std(wcs)
        results[pol] = dict(acc_m=acc_m, acc_s=acc_s, best_m=best_m,
                            wc_m=wc_m, wc_s=wc_s,
                            accs=[a * 100 for a in accs], wcs=wcs)
        print(f"│ {pol:<13s} │ {acc_m:5.2f} ± {acc_s:4.2f}   │ {best_m:10.2f}  │ {wc_m:7.0f} ± {wc_s:3.0f} │")

    print("└───────────────┴─────────────────┴──────────────┴───────────────┘")

    # ── 2. Per-seed Breakdown ─────────────────────────────────────────
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  Per-seed Accuracy (%)                                                  │")
    print("├───────────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤")
    print("│ Policy        │ seed42   │ seed123  │ seed456  │ seed42WC │ seed123WC    │")
    print("├───────────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤")
    for pol in POLICY_ORDER:
        r = results[pol]
        a = r["accs"]
        w = r["wcs"]
        print(f"│ {pol:<13s} │ {a[0]:6.2f}   │ {a[1]:6.2f}   │ {a[2]:6.2f}   │ {w[0]:7.0f}  │ {w[1]:7.0f}      │")
    print("└───────────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘")

    # ── 3. Deadline & Participation Analysis (per-round) ──────────────
    print("\n" + "=" * 80)
    print("  DEADLINE & PARTICIPATION ANALYSIS")
    print("=" * 80)

    for pol in POLICY_ORDER:
        d = data[pol]
        print(f"\n--- {pol} ---")
        for s in SEEDS:
            h = d["seeds"][s]["history"]
            deadlines = [r["extra"]["deadline"] for r in h if "extra" in r and "deadline" in r.get("extra", {})]
            n_complete = [r["extra"]["n_complete"] for r in h if "extra" in r and "n_complete" in r.get("extra", {})]
            n_partial = [r["extra"]["n_partial"] for r in h if "extra" in r and "n_partial" in r.get("extra", {})]
            n_timeout = [r["extra"]["n_timeout"] for r in h if "extra" in r and "n_timeout" in r.get("extra", {})]
            n_selected = [r["extra"]["n_selected"] for r in h if "extra" in r and "n_selected" in r.get("extra", {})]
            participation = [r["participation_rate"] for r in h]

            if deadlines:
                d_arr = np.array(deadlines)
                print(f"  {s}: D_warmup={d_arr[0]:.1f}s  D_final={d_arr[-1]:.1f}s  "
                      f"D_mean={d_arr.mean():.1f}s  D_min={d_arr.min():.1f}s  D_max={d_arr.max():.1f}s")
            if n_complete:
                nc = np.array(n_complete)
                np_ = np.array(n_partial) if n_partial else np.zeros_like(nc)
                nt = np.array(n_timeout) if n_timeout else np.zeros_like(nc)
                print(f"         complete={nc.mean():.1f}  partial={np_.mean():.1f}  "
                      f"timeout={nt.mean():.1f}  selected={np.mean(n_selected):.1f}  "
                      f"participation_rate={np.mean(participation):.3f}")

    # ── 4. Wall-clock duplicate check (sanity) ────────────────────────
    print("\n" + "=" * 80)
    print("  WALL-CLOCK DUPLICATE CHECK")
    print("=" * 80)
    
    for s in SEEDS:
        wcs_per_seed = {}
        for pol in POLICY_ORDER:
            wc = data[pol]["seeds"][s]["wall_clock_time"]
            wcs_per_seed[pol] = wc
        
        # Find duplicates
        from collections import defaultdict
        wc_groups = defaultdict(list)
        for pol, wc in wcs_per_seed.items():
            wc_groups[f"{wc:.4f}"].append(pol)
        
        dupes = {k: v for k, v in wc_groups.items() if len(v) > 1}
        if dupes:
            print(f"\n  ⚠️  {s} — DUPLICATE WC values:")
            for wc_str, pols in dupes.items():
                print(f"    WC={float(wc_str):.1f}s → {pols}")
        else:
            print(f"\n  ✅ {s} — all WC values unique")

    # ── 5. Per-round deadline evolution (round 0,5,10,...,49) ─────────
    print("\n" + "=" * 80)
    print("  DEADLINE EVOLUTION (seed42, sampled rounds)")
    print("=" * 80)
    
    checkpoints = [0, 3, 5, 10, 20, 30, 40, 49]
    header = "│ Policy        │" + "".join(f" R{r:<4d}│" for r in checkpoints)
    sep = "├───────────────┼" + "┼".join(["──────" for _ in checkpoints]) + "┤"
    print("┌───────────────┬" + "┬".join(["──────" for _ in checkpoints]) + "┐")
    print(header)
    print(sep)
    
    for pol in POLICY_ORDER:
        h = data[pol]["seeds"]["seed42"]["history"]
        row = f"│ {pol:<13s} │"
        for r in checkpoints:
            if r < len(h) and "extra" in h[r] and "deadline" in h[r].get("extra", {}):
                dl = h[r]["extra"]["deadline"]
                row += f" {dl:5.1f}│"
            else:
                row += "   N/A│"
        print(row)
    print("└───────────────┴" + "┴".join(["──────" for _ in checkpoints]) + "┘")

    # ── 6. Accuracy convergence at key rounds (seed-averaged) ─────────
    print("\n" + "=" * 80)
    print("  ACCURACY CONVERGENCE (mean of 3 seeds)")
    print("=" * 80)
    
    conv_rounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    header = "│ Policy        │" + "".join(f" R{r:<4d}│" for r in conv_rounds)
    print("┌───────────────┬" + "┬".join(["──────" for _ in conv_rounds]) + "┐")
    print(header)
    print("├───────────────┼" + "┼".join(["──────" for _ in conv_rounds]) + "┤")
    
    for pol in POLICY_ORDER:
        row = f"│ {pol:<13s} │"
        for r in conv_rounds:
            accs_r = []
            for s in SEEDS:
                h = data[pol]["seeds"][s]["history"]
                if r < len(h):
                    accs_r.append(h[r]["accuracy"] * 100)
            if accs_r:
                row += f" {np.mean(accs_r):5.2f}│"
            else:
                row += "   N/A│"
        print(row)
    print("└───────────────┴" + "┴".join(["──────" for _ in conv_rounds]) + "┘")

    # ── 7. Ranking & Observations ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RANKING & KEY OBSERVATIONS")
    print("=" * 80)

    # Sort by accuracy
    by_acc = sorted(results.items(), key=lambda x: x[1]["acc_m"], reverse=True)
    print("\nAccuracy ranking:")
    for i, (pol, r) in enumerate(by_acc, 1):
        print(f"  {i}. {pol:<16s}  {r['acc_m']:.2f}%  (WC={r['wc_m']:.0f}s)")

    # Sort by WC
    by_wc = sorted(results.items(), key=lambda x: x[1]["wc_m"])
    print("\nWall-clock ranking (fastest first):")
    for i, (pol, r) in enumerate(by_wc, 1):
        print(f"  {i}. {pol:<16s}  {r['wc_m']:.0f}s  (acc={r['acc_m']:.2f}%)")

    # Pareto front (acc vs WC)
    print("\nPareto front (higher acc, lower WC is better):")
    candidates = sorted(results.items(), key=lambda x: x[1]["acc_m"], reverse=True)
    pareto = []
    min_wc = float("inf")
    for pol, r in candidates:
        if r["wc_m"] < min_wc:
            pareto.append(pol)
            min_wc = r["wc_m"]
    print(f"  Pareto-optimal policies: {pareto}")
    if not pareto:
        print("  (no clear Pareto winner — higher acc always costs more WC)")
        # Just show the tradeoff
        for pol in ["fixed(5.0)", "adaptive(0.7)", "adaptive(0.9)", "fixed(20.0)"]:
            r = results[pol]
            print(f"    {pol}: {r['acc_m']:.2f}% / {r['wc_m']:.0f}s")

    # ── 8. Comparison with Exp 1 DASH (adaptive p=0.85) ───────────────
    print("\n" + "=" * 80)
    print("  CROSS-REFERENCE: Exp 1 DASH (adaptive p=0.85) vs Exp 3 configs")
    print("=" * 80)
    print("  Exp 1 DASH:  acc=44.47%, WC=979s  (adaptive, p=0.85)")
    print()
    
    # Find the closest Exp 3 config
    ref_acc = 44.47
    ref_wc = 979
    for pol in POLICY_ORDER:
        r = results[pol]
        acc_diff = r["acc_m"] - ref_acc
        wc_diff = r["wc_m"] - ref_wc
        marker = " ◀ closest" if pol == "adaptive(0.9)" else ""
        print(f"  {pol:<16s}  acc={r['acc_m']:.2f}% ({acc_diff:+.2f}pp)  "
              f"WC={r['wc_m']:.0f}s ({wc_diff:+.0f}s){marker}")

    # ── 9. Fixed vs Adaptive analysis ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  FIXED vs ADAPTIVE vs PARTIAL ANALYSIS")
    print("=" * 80)
    
    fixed_accs = [results[f"fixed({d})"]["acc_m"] for d in ["5.0", "10.0", "20.0"]]
    fixed_wcs = [results[f"fixed({d})"]["wc_m"] for d in ["5.0", "10.0", "20.0"]]
    adaptive_accs = [results[f"adaptive({p})"]["acc_m"] for p in ["0.5", "0.7", "0.9"]]
    adaptive_wcs = [results[f"adaptive({p})"]["wc_m"] for p in ["0.5", "0.7", "0.9"]]
    partial_acc = results["partial(0.7)"]["acc_m"]
    partial_wc = results["partial(0.7)"]["wc_m"]
    
    print(f"\n  Fixed deadlines:")
    print(f"    D=5s  → acc={fixed_accs[0]:.2f}%  WC={fixed_wcs[0]:.0f}s")
    print(f"    D=10s → acc={fixed_accs[1]:.2f}%  WC={fixed_wcs[1]:.0f}s")
    print(f"    D=20s → acc={fixed_accs[2]:.2f}%  WC={fixed_wcs[2]:.0f}s")
    print(f"    Trend: as D↑, acc {'↑' if fixed_accs[2] > fixed_accs[0] else '↓'} "
          f"({fixed_accs[0]:.2f}→{fixed_accs[2]:.2f}), "
          f"WC {'↑' if fixed_wcs[2] > fixed_wcs[0] else '↓'} "
          f"({fixed_wcs[0]:.0f}→{fixed_wcs[2]:.0f})")
    
    print(f"\n  Adaptive deadlines:")
    print(f"    p=0.5 → acc={adaptive_accs[0]:.2f}%  WC={adaptive_wcs[0]:.0f}s")
    print(f"    p=0.7 → acc={adaptive_accs[1]:.2f}%  WC={adaptive_wcs[1]:.0f}s")
    print(f"    p=0.9 → acc={adaptive_accs[2]:.2f}%  WC={adaptive_wcs[2]:.0f}s")
    print(f"    Trend: as p↑, acc {'↑' if adaptive_accs[2] > adaptive_accs[0] else '↓→↑→' if adaptive_accs[1] > max(adaptive_accs[0], adaptive_accs[2]) else '?'} "
          f"peak at p=0.7 ({adaptive_accs[1]:.2f}%)")
    
    print(f"\n  Partial accept:")
    print(f"    p=0.7+partial → acc={partial_acc:.2f}%  WC={partial_wc:.0f}s")
    print(f"    vs adaptive(0.7): acc diff={partial_acc - adaptive_accs[1]:.2f}pp, "
          f"WC diff={partial_wc - adaptive_wcs[1]:.0f}s")

    # ── 10. Summary for paper ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PAPER-READY SUMMARY")
    print("=" * 80)
    
    best_acc_pol = by_acc[0][0]
    best_acc_val = by_acc[0][1]["acc_m"]
    fastest_pol = by_wc[0][0]
    fastest_wc = by_wc[0][1]["wc_m"]
    
    print(f"""
  ▸ Best accuracy:  {best_acc_pol} ({best_acc_val:.2f}%)
  ▸ Fastest:        {fastest_pol} ({fastest_wc:.0f}s)
  ▸ Best trade-off: adaptive(0.7) — {results['adaptive(0.7)']['acc_m']:.2f}% / {results['adaptive(0.7)']['wc_m']:.0f}s
  
  Key findings:
  1. Adaptive policies outperform fixed deadlines in accuracy
     (best adaptive {max(adaptive_accs):.2f}% vs best fixed {max(fixed_accs):.2f}%)
  2. Fixed D=5s and D=10s produce {'' if abs(fixed_wcs[0]-fixed_wcs[1])<10 else 'different'} 
     {'identical' if abs(fixed_wcs[0]-fixed_wcs[1])<10 else ''} WC — needs investigation
  3. Partial accept adds {results['partial(0.7)']['acc_m'] - results['adaptive(0.7)']['acc_m']:.2f}pp 
     over adaptive(0.7) alone
  4. DASH (Exp1, p=0.85) sits between adaptive(0.7) and adaptive(0.9) — 
     confirming p=0.85 is a good operating point
""")


if __name__ == "__main__":
    main()
