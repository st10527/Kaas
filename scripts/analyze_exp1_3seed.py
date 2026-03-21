#!/usr/bin/env python3
"""Analyze Exp 1 Main Comparison — 3 seeds aggregated."""
import json
import numpy as np
from pathlib import Path

data = json.load(open(Path(__file__).resolve().parent.parent / "results/jpdc/exp1_main_comparison.json"))

methods = ['DASH', 'Sync-Greedy', 'FedBuff-FD', 'Random-Async', 'Full-Async', 'Sync-Full']
seeds = [42, 123, 456]

# ============================================================
# 1. Summary table: mean ± std across 3 seeds
# ============================================================
print("=" * 80)
print("  EXP 1 — MAIN COMPARISON (3 seeds: 42, 123, 456)")
print("=" * 80)
print(f"{'Method':20s} {'Acc (mean±std)':>18s} {'Best Acc':>10s} {'WC (mean±std)':>20s} {'GPU time':>10s}")
print("-" * 80)

summary = {}
for m in methods:
    accs, wcs, gpus, bests = [], [], [], []
    for s in seeds:
        key = f"{m}_seed{s}"
        r = data[key]
        accs.append(r['final_accuracy'])
        bests.append(r['best_accuracy'])
        wcs.append(r['wall_clock_time'])
        gpus.append(r['total_time'])
    
    acc_mean, acc_std = np.mean(accs), np.std(accs)
    wc_mean, wc_std = np.mean(wcs), np.std(wcs)
    gpu_mean = np.mean(gpus)
    best_mean = np.mean(bests)
    
    summary[m] = {
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'wc_mean': wc_mean, 'wc_std': wc_std,
        'best_mean': best_mean,
    }
    
    print(f"{m:20s} {acc_mean*100:6.2f}% ± {acc_std*100:4.2f}%  {best_mean*100:7.2f}%  "
          f"{wc_mean:8.1f}s ± {wc_std:6.1f}s  {gpu_mean/60:6.1f}min")

# ============================================================
# 2. Speedup relative to Sync-Greedy
# ============================================================
print("\n" + "=" * 80)
print("  SPEEDUP vs Sync-Greedy")
print("=" * 80)
sync_wc = summary['Sync-Greedy']['wc_mean']
sync_acc = summary['Sync-Greedy']['acc_mean']
for m in methods:
    s = summary[m]
    speedup = sync_wc / s['wc_mean'] if s['wc_mean'] > 0 else float('inf')
    acc_gap = (s['acc_mean'] - sync_acc) * 100
    print(f"  {m:20s}  speedup={speedup:5.2f}x  acc_gap={acc_gap:+5.2f}pp")

# ============================================================
# 3. Per-seed breakdown
# ============================================================
print("\n" + "=" * 80)
print("  PER-SEED BREAKDOWN")
print("=" * 80)
print(f"{'Method':20s} {'seed42 acc':>10s} {'seed123 acc':>12s} {'seed456 acc':>12s}  "
      f"{'seed42 WC':>10s} {'seed123 WC':>12s} {'seed456 WC':>12s}")
print("-" * 90)
for m in methods:
    accs = [data[f"{m}_seed{s}"]['final_accuracy'] for s in seeds]
    wcs = [data[f"{m}_seed{s}"]['wall_clock_time'] for s in seeds]
    print(f"{m:20s} {accs[0]*100:8.2f}%  {accs[1]*100:10.2f}%  {accs[2]*100:10.2f}%  "
          f"{wcs[0]:8.1f}s  {wcs[1]:10.1f}s  {wcs[2]:10.1f}s")

# ============================================================
# 4. Accuracy convergence (averaged across 3 seeds)
# ============================================================
print("\n" + "=" * 80)
print("  ACCURACY CONVERGENCE (mean of 3 seeds, every 5 rounds)")
print("=" * 80)
n_rounds = 50
header = f"{'Rnd':>4s}"
for m in methods:
    header += f"  {m:>14s}"
print(header)
print("-" * (4 + 16 * len(methods)))

for t in range(0, n_rounds, 5):
    line = f"{t:4d}"
    for m in methods:
        vals = []
        for s in seeds:
            hist = data[f"{m}_seed{s}"]['history']
            if t < len(hist):
                vals.append(hist[t]['accuracy'])
        if vals:
            line += f"  {np.mean(vals)*100:12.2f}%"
        else:
            line += f"  {'N/A':>13s}"
    print(line)

# last round
line = f"{n_rounds-1:4d}"
for m in methods:
    vals = []
    for s in seeds:
        hist = data[f"{m}_seed{s}"]['history']
        vals.append(hist[-1]['accuracy'])
    line += f"  {np.mean(vals)*100:12.2f}%"
print(line)

# ============================================================
# 5. DASH deadline behavior (check D_min floor is working)
# ============================================================
print("\n" + "=" * 80)
print("  DASH DEADLINE CHECK (per seed)")
print("=" * 80)
for s in seeds:
    hist = data[f"DASH_seed{s}"]['history']
    deadlines = []
    for h in hist:
        extra = h.get('extra', {})
        d = extra.get('deadline', None)
        if d is not None:
            deadlines.append(d)
    if deadlines:
        print(f"  seed={s}: D_warmup={deadlines[0]:.1f}  D_min={min(deadlines):.1f}  "
              f"D_final={deadlines[-1]:.1f}  D_mean={np.mean(deadlines):.1f}")
    else:
        # Try from round wall-clock diff
        wcs = [h['wall_clock'] for h in hist]
        diffs = [wcs[i]-wcs[i-1] for i in range(1, len(wcs))]
        print(f"  seed={s}: dWC range=[{min(diffs):.1f}, {max(diffs):.1f}]  "
              f"dWC_final={diffs[-1]:.1f}  dWC_mean={np.mean(diffs):.1f}")

# ============================================================
# 6. Sync-Greedy wall-clock variation check
# ============================================================
print("\n" + "=" * 80)
print("  SYNC-GREEDY WALL-CLOCK VARIATION CHECK")
print("=" * 80)
for s in seeds:
    hist = data[f"Sync-Greedy_seed{s}"]['history']
    wcs = [h['wall_clock_time'] for h in hist]
    diffs = [wcs[i]-wcs[i-1] for i in range(1, len(wcs))]
    print(f"  seed={s}: dWC range=[{min(diffs):.1f}, {max(diffs):.1f}]  "
          f"mean={np.mean(diffs):.1f}  std={np.std(diffs):.2f}")

# ============================================================
# 7. Full-Async warmup check
# ============================================================
print("\n" + "=" * 80)
print("  FULL-ASYNC WARMUP CHECK")
print("=" * 80)
for s in seeds:
    hist = data[f"Full-Async_seed{s}"]['history']
    wcs = [h['wall_clock_time'] for h in hist]
    d0 = wcs[0]
    d3 = wcs[2] - wcs[1] if len(wcs) > 2 else 0
    print(f"  seed={s}: warmup_D(r0)={d0:.1f}s  D(r3)={d3:.1f}s  total_WC={wcs[-1]:.1f}s")
