#!/usr/bin/env python3
"""Diagnose issues in Exp1 v2 results."""
import json
import numpy as np

with open('results/jpdc/exp1_main_comparison.json') as f:
    data = json.load(f)

# DASH deadline spiral
dash = data['DASH_seed42']['history']
deadlines = [r['extra']['deadline'] for r in dash]
comms = [r['extra']['comm_mb'] for r in dash]
n_comp = [r['extra']['n_complete'] for r in dash]

print("=== DASH Deadline Spiral ===")
print(f"  Warmup (R0-2):  D_avg={np.mean(deadlines[:3]):.1f}  comm={np.mean(comms[:3]):.2f}  complete={np.mean(n_comp[:3]):.0f}")
print(f"  Post   (R3-9):  D_avg={np.mean(deadlines[3:10]):.1f}  comm={np.mean(comms[3:10]):.2f}  complete={np.mean(n_comp[3:10]):.0f}")
print(f"  Mid    (R10-24): D_avg={np.mean(deadlines[10:25]):.1f}  comm={np.mean(comms[10:25]):.2f}  complete={np.mean(n_comp[10:25]):.0f}")
print(f"  Late   (R25-49): D_avg={np.mean(deadlines[25:]):.1f}  comm={np.mean(comms[25:]):.2f}  complete={np.mean(n_comp[25:]):.0f}")
print(f"  D: {max(deadlines):.1f} -> {min(deadlines):.1f} (shrinks {max(deadlines)/min(deadlines):.1f}x)")
print(f"  comm: {max(comms):.2f} -> {min(comms):.2f}")

# Sync-Greedy constant wall-clock
sg = data['Sync-Greedy_seed42']['history']
sg_deltas = []
for i in range(len(sg)):
    wc = sg[i]['wall_clock_time']
    prev = sg[i-1]['wall_clock_time'] if i > 0 else 0
    sg_deltas.append(wc - prev)
unique_deltas = set([round(d, 1) for d in sg_deltas])
print(f"\n=== Sync-Greedy Wall-Clock Bug ===")
print(f"  Unique per-round WC values: {unique_deltas}")
print(f"  All identical: {len(unique_deltas) == 1}  <-- BUG: seed not varying!")

# Full-Async warmup
fa = data['Full-Async_seed42']['history']
print(f"\n=== Full-Async Warmup ===")
for i in range(5):
    d = fa[i]['extra'].get('deadline', '?')
    wc = fa[i].get('wall_clock_time', 0)
    nc = fa[i]['extra'].get('n_complete', '?')
    print(f"  R{i}: D={d}  WC={wc:.1f}  complete={nc}")
print(f"  Warmup D={fa[0]['extra']['deadline']:.1f} vs DASH warmup D={deadlines[0]:.1f}")
print(f"  Ratio: {fa[0]['extra']['deadline']/deadlines[0]:.1f}x too large!")

# Sync-Full
sf = data['Sync-Full_seed42']['history']
sf_d0 = sf[0]['wall_clock_time']
print(f"\n=== Sync-Full ===")
print(f"  WC/round = {sf_d0:.1f}s (constant: {sf_d0 == sf[1]['wall_clock_time'] - sf[0]['wall_clock_time']})")
print(f"  comm_mb = {sf[0]['extra']['comm_mb']:.2f} (full logit)")
print(f"  Total = {sf[-1]['wall_clock_time']:.0f}s = {sf[-1]['wall_clock_time']/3600:.1f} hours")

# Big picture
print(f"\n{'='*60}")
print(f"SUMMARY: What's good vs what's buggy")
print(f"{'='*60}")
print(f"  DASH:        44.30%  WC= 896s   <-- Accuracy recovered! (was 35.2%)")
print(f"  Sync-Greedy: 45.20%  WC=2697s   <-- gap only 0.9pp now (was 9pp)")
print(f"  DASH speedup: {2696.5/896.3:.1f}x wall-clock")
print()
print(f"  BUG 1: Sync wall-clock constant seed (cosmetic)")
print(f"  BUG 2: Full-Async warmup D={fa[0]['extra']['deadline']:.0f}s (should be ~55s like DASH)")
print(f"  BUG 3: DASH deadline spiral: 55->9s (needs D_min floor)")
