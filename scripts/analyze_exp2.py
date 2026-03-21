#!/usr/bin/env python3
"""Analyze Exp 2 Straggler Severity Sweep — 3 seeds aggregated."""
import json
import numpy as np
from pathlib import Path

data = json.load(open(Path(__file__).resolve().parent.parent / "results/jpdc/exp2_straggler_sweep.json"))

methods = ['DASH', 'Sync-Greedy', 'FedBuff-FD']
sigmas = ['sigma=0.0', 'sigma=0.3', 'sigma=0.5', 'sigma=1.0', 'sigma=1.5']
seeds = ['seed42', 'seed123', 'seed456']

# ============================================================
# 1. Main summary table
# ============================================================
print("=" * 90)
print("  EXP 2 — STRAGGLER SEVERITY SWEEP (3 seeds, mean ± std)")
print("=" * 90)

# Accuracy table
print("\n--- Final Accuracy (%) ---")
header = f"{'σ':>8s}"
for m in methods:
    header += f"  {m:>20s}"
print(header)
print("-" * 72)

acc_data = {}
for sigma in sigmas:
    line = f"{sigma.split('=')[1]:>8s}"
    for m in methods:
        vals = []
        for s in seeds:
            r = data[sigma][s].get(m)
            if r is not None:
                vals.append(r['final_accuracy'])
        if vals:
            mean, std = np.mean(vals)*100, np.std(vals)*100
            line += f"  {mean:6.2f} ± {std:4.2f}     "
            if m not in acc_data:
                acc_data[m] = []
            acc_data[m].append((sigma, mean, std))
        else:
            line += f"  {'N/A':>20s}"
    print(line)

# Wall-clock table
print("\n--- Wall-Clock Time (s) ---")
header = f"{'σ':>8s}"
for m in methods:
    header += f"  {m:>20s}"
print(header)
print("-" * 72)

wc_data = {}
for sigma in sigmas:
    line = f"{sigma.split('=')[1]:>8s}"
    for m in methods:
        vals = []
        for s in seeds:
            r = data[sigma][s].get(m)
            if r is not None:
                vals.append(r['wall_clock_time'])
        if vals:
            mean, std = np.mean(vals), np.std(vals)
            line += f"  {mean:7.0f} ± {std:5.0f}     "
            if m not in wc_data:
                wc_data[m] = []
            wc_data[m].append((sigma, mean, std))
        else:
            line += f"  {'N/A':>20s}"
    print(line)

# ============================================================
# 2. Speedup: DASH vs Sync-Greedy at each σ
# ============================================================
print("\n" + "=" * 90)
print("  DASH vs Sync-Greedy: SPEEDUP & ACC GAP per σ")
print("=" * 90)
print(f"{'σ':>8s}  {'DASH Acc':>10s}  {'Sync Acc':>10s}  {'Gap(pp)':>8s}  "
      f"{'DASH WC':>10s}  {'Sync WC':>10s}  {'Speedup':>8s}")
print("-" * 72)

for sigma in sigmas:
    s_label = sigma.split('=')[1]
    dash_accs = [data[sigma][s]['DASH']['final_accuracy'] for s in seeds]
    sync_accs = [data[sigma][s]['Sync-Greedy']['final_accuracy'] for s in seeds]
    dash_wcs = [data[sigma][s]['DASH']['wall_clock_time'] for s in seeds]
    sync_wcs = [data[sigma][s]['Sync-Greedy']['wall_clock_time'] for s in seeds]
    
    da, sa = np.mean(dash_accs)*100, np.mean(sync_accs)*100
    dw, sw = np.mean(dash_wcs), np.mean(sync_wcs)
    gap = da - sa
    speedup = sw / dw if dw > 0 else float('inf')
    
    print(f"{s_label:>8s}  {da:8.2f}%   {sa:8.2f}%   {gap:+6.2f}   "
          f"{dw:8.1f}s   {sw:8.1f}s   {speedup:6.2f}x")

# ============================================================
# 3. Key trend analysis
# ============================================================
print("\n" + "=" * 90)
print("  TREND ANALYSIS")
print("=" * 90)

# DASH accuracy degradation
print("\n[DASH Accuracy vs σ]")
for sigma in sigmas:
    s_label = sigma.split('=')[1]
    vals = [data[sigma][s]['DASH']['final_accuracy'] for s in seeds]
    print(f"  σ={s_label}: {np.mean(vals)*100:.2f}%")

# Sync-Greedy WC inflation  
print("\n[Sync-Greedy Wall-Clock vs σ (should increase with σ)]")
for sigma in sigmas:
    s_label = sigma.split('=')[1]
    vals = [data[sigma][s]['Sync-Greedy']['wall_clock_time'] for s in seeds]
    print(f"  σ={s_label}: {np.mean(vals):.0f}s")

# DASH WC stability
print("\n[DASH Wall-Clock vs σ (should be relatively stable)]")
for sigma in sigmas:
    s_label = sigma.split('=')[1]
    vals = [data[sigma][s]['DASH']['wall_clock_time'] for s in seeds]
    print(f"  σ={s_label}: {np.mean(vals):.0f}s")

# ============================================================
# 4. DASH deadline behavior per σ
# ============================================================
print("\n" + "=" * 90)
print("  DASH DEADLINE BEHAVIOR per σ (seed=42)")
print("=" * 90)
for sigma in sigmas:
    s_label = sigma.split('=')[1]
    hist = data[sigma]['seed42']['DASH']['history']
    wcs = [h['wall_clock_time'] for h in hist]
    diffs = [wcs[0]] + [wcs[i]-wcs[i-1] for i in range(1, len(wcs))]
    n_parts = [h['n_participants'] for h in hist]
    print(f"  σ={s_label}: D_warmup={diffs[0]:.1f}  D_final={diffs[-1]:.1f}  "
          f"D_min={min(diffs):.1f}  avg_participants={np.mean(n_parts):.1f}/50")

# ============================================================
# 5. σ=0 sanity check (no straggler → DASH ≈ Sync)
# ============================================================
print("\n" + "=" * 90)
print("  σ=0 SANITY CHECK (no straggler noise)")
print("=" * 90)
for s in seeds:
    da = data['sigma=0.0'][s]['DASH']['final_accuracy'] * 100
    sa = data['sigma=0.0'][s]['Sync-Greedy']['final_accuracy'] * 100
    dw = data['sigma=0.0'][s]['DASH']['wall_clock_time']
    sw = data['sigma=0.0'][s]['Sync-Greedy']['wall_clock_time']
    print(f"  {s}: DASH={da:.2f}%/{dw:.0f}s  Sync={sa:.2f}%/{sw:.0f}s  "
          f"gap={da-sa:+.2f}pp  speedup={sw/dw:.2f}x")
