#!/usr/bin/env python3
"""Diagnose DASH performance issue: trace through deadline → v_feasible → allocation chain."""
import sys
sys.path.insert(0, '/Users/nclab/Kaas')

import numpy as np
from src.async_module.straggler_model import StragglerModel

# ── 1. Generate devices (same as experiment, seed=42, n=50) ──
rates = StragglerModel.generate_device_rates(50, seed=42)
rate_sums = [r['comp_rate'] + r['comm_rate'] for r in rates]

print("=" * 70)
print("1. DEVICE RATE DISTRIBUTION (50 devices, seed=42)")
print("=" * 70)
print(f"   rate_sum: min={min(rate_sums):.4f}  median={np.median(rate_sums):.4f}  "
      f"p70={np.percentile(rate_sums, 70):.4f}  p90={np.percentile(rate_sums, 90):.4f}  "
      f"max={max(rate_sums):.4f}")

# ── 2. Warmup deadline ──
v_max = 10000  # n_public
p90_rate = np.percentile(rate_sums, 90)
estimated_v = v_max * 0.4  # 4000
D_warmup = p90_rate * estimated_v * 2.0

print(f"\n{'=' * 70}")
print("2. WARMUP DEADLINE CALCULATION")
print("=" * 70)
print(f"   v_max = {v_max}")
print(f"   p90(rate_sum) = {p90_rate:.4f}")
print(f"   estimated_v = v_max * 0.4 = {estimated_v:.0f}")
print(f"   D_warmup = {p90_rate:.4f} * {estimated_v:.0f} * 2.0 = {D_warmup:.1f}s")
print(f"   ← THIS IS ~750s per warmup round, 3 rounds = {D_warmup*3:.0f}s wall-clock burned!")

# ── 3. Simulated latencies during warmup ──
sm = StragglerModel(sigma_noise=0.5, seed=42)
# What v_star does each device actually get? For warmup, straggler_aware=False
# so it's the same as Sync-Greedy. From logs: comm_mb=8.20 → total logits:
warmup_total_v = 8.20 * 1024 * 1024 / (100 * 4)  # 100 classes × 4 bytes
print(f"\n   During warmup: comm_mb=8.20 → total v ≈ {warmup_total_v:.0f}")
per_device_v_warmup = warmup_total_v / 48  # ~48 selected
print(f"   per-device v ≈ {per_device_v_warmup:.0f}")

# Simulate warmup latencies
warmup_devs = [{'device_id': i, 'v_star': int(per_device_v_warmup),
                'comp_rate': rates[i]['comp_rate'],
                'comm_rate': rates[i]['comm_rate']} for i in range(50)]
lat = sm.simulate_round(warmup_devs, deadline=D_warmup)
taus = [l.tau_total for l in lat]
print(f"\n   Warmup latencies: min={min(taus):.1f}  median={np.median(taus):.1f}  "
      f"p70={np.percentile(taus, 70):.1f}  p90={np.percentile(taus, 90):.1f}  max={max(taus):.1f}")

# ── 4. Round 3: Adaptive deadline kicks in ──
D_adaptive = np.percentile(taus, 70)  # percentile=0.7
print(f"\n{'=' * 70}")
print("3. ROUND 3: ADAPTIVE DEADLINE TRANSITION")
print("=" * 70)
print(f"   Adaptive D = p70(warmup latencies) = {D_adaptive:.1f}s")
print(f"   CRASH from {D_warmup:.1f}s → {D_adaptive:.1f}s (={D_adaptive/D_warmup*100:.1f}%)")

# ── 5. Impact on v_feasible ──
print(f"\n{'=' * 70}")
print("4. v_FEASIBLE CAP WITH TIGHT DEADLINE")
print("=" * 70)
print(f"   Formula: v_feasible = D / rate_sum * 0.5  (margin)")
print(f"   {'Device':>6} {'rate_sum':>10} {'v_feasible':>12} {'vs warmup':>10}")
print(f"   {'-'*42}")
v_feasibles = []
for i in range(50):
    rs = rate_sums[i]
    vf = D_adaptive / rs * 0.5
    vf_capped = min(vf, v_max)
    v_feasibles.append(vf_capped)

# Show tiers
sorted_idx = np.argsort(rate_sums)
for tier_name, idxs in [("Fast (p10)", sorted_idx[:5]),
                         ("Med  (p50)", sorted_idx[22:27]),
                         ("Slow (p90)", sorted_idx[45:])]:
    for i in idxs:
        vf = v_feasibles[i]
        print(f"   {i:>6} {rate_sums[i]:>10.4f} {vf:>12.0f} {vf/per_device_v_warmup:>9.1f}x")
    print()

print(f"   median v_feasible = {np.median(v_feasibles):.0f}")
print(f"   min v_feasible = {min(v_feasibles):.0f}")
print(f"   vs warmup per-device v = {per_device_v_warmup:.0f}")

# ── 6. How does this compare to budget constraint? ──
print(f"\n{'=' * 70}")
print("5. BUDGET vs DEADLINE: WHICH BINDS?")
print("=" * 70)

# Load edge device params (b_i, theta_i)
from src.methods.kaas_edge import generate_edge_devices
edge_devs = generate_edge_devices(50, seed=42)
b_values = [d['b_i'] for d in edge_devs]
a_values = [d.get('a_i', 0) for d in edge_devs]
print(f"   b_i: min={min(b_values):.4f}  median={np.median(b_values):.4f}  max={max(b_values):.4f}")
print(f"   a_i: min={min(a_values):.4f}  median={np.median(a_values):.4f}  max={max(a_values):.4f}")
total_a = sum(a_values)
print(f"   sum(a_i) = {total_a:.2f}  (for all 50)")
print(f"   Budget = 50.0")
print(f"   Residual for variable cost = {50 - total_a:.2f}")

# With budget, what v_i would water-filling give WITHOUT deadline cap?
# v_i* = sqrt(rho*theta/(nu*b)) - theta, with sum(b_i*v_i) = B_res
# Rough estimate: assume all devices, equal rho=1
median_b = np.median(b_values)
median_theta = np.median([d['theta_i'] for d in edge_devs])
approx_v_per = (50 - total_a) / (50 * median_b)
print(f"\n   Rough v per device (no deadline cap): (B_res)/(N*median_b) ≈ {approx_v_per:.0f}")
comm_approx = approx_v_per * 100 * 4 / (1024*1024) * 50
print(f"   → total comm ≈ {comm_approx:.1f} MB (matches Sync-Greedy's 8.20 MB? {'YES' if abs(comm_approx-8.2) < 3 else 'NO'})")

print(f"\n   With deadline cap: median v_feasible = {np.median(v_feasibles):.0f}")
print(f"   Binding constraint: {'DEADLINE' if np.median(v_feasibles) < approx_v_per else 'BUDGET'}")

# ── 7. Post-warmup DASH per-round quality ──
print(f"\n{'=' * 70}")
print("6. ESTIMATED QUALITY LOSS")
print("=" * 70)
# Quality = sum rho_i * v_i / (v_i + theta_i)
# With v_i ≈ 60 (from comm_mb=0.99 / 40 devices) and theta_i ≈ 50
v_dash_post = 0.99 * 1024 * 1024 / (100 * 4) / 40  # ~65
v_sync = 8.20 * 1024 * 1024 / (100 * 4) / 48  # ~450

print(f"   DASH (post-warmup):  v_per_device ≈ {v_dash_post:.0f}, "
      f"q ≈ {1.0 * v_dash_post / (v_dash_post + median_theta):.3f}")
print(f"   Sync-Greedy:         v_per_device ≈ {v_sync:.0f}, "
      f"q ≈ {1.0 * v_sync / (v_sync + median_theta):.3f}")
print(f"   Quality ratio: {(v_dash_post/(v_dash_post+median_theta)) / (v_sync/(v_sync+median_theta)):.2f}x")

# ── 8. Wall-clock breakdown ──
print(f"\n{'=' * 70}")
print("7. WALL-CLOCK BREAKDOWN")
print("=" * 70)
print(f"   DASH total:   2492.0s")
print(f"   Warmup (R0-2): 3 × {D_warmup:.0f} = {3*D_warmup:.0f}s ({3*D_warmup/2492*100:.0f}%)")
print(f"   Post-warmup:   {2492-3*D_warmup:.0f}s over 47 rounds = {(2492-3*D_warmup)/47:.1f}s/round")
print(f"   Sync-Greedy:   2696.5s / 50 = {2696.5/50:.1f}s/round (constant)")

# ── 9. Proposed fixes ──
print(f"\n{'=' * 70}")
print("8. PROPOSED FIXES")
print("=" * 70)
# Fix A: Reduce warmup deadline
D_warmup_fix = p90_rate * estimated_v * 0.3  # reduce safety from 2.0 to 0.3
print(f"   A. Reduce warmup safety: 2.0 → 0.3")
print(f"      D_warmup: {D_warmup:.0f} → {D_warmup_fix:.0f}s")
print(f"      3-round cost: {3*D_warmup:.0f} → {3*D_warmup_fix:.0f}s")

# Fix B: Increase adaptive percentile
D_p85 = np.percentile(taus, 85)
D_p90 = np.percentile(taus, 90)
print(f"\n   B. Increase adaptive_percentile: 0.7 → 0.85")
print(f"      D_adaptive: {D_adaptive:.1f} → {D_p85:.1f}s (p85) or {D_p90:.1f}s (p90)")

# Fix C: Remove or relax v_feasible margin
print(f"\n   C. Relax v_feasible margin: 0.5 → 1.0 (or remove)")
vf_relaxed = [D_adaptive / rs * 1.0 for rs in rate_sums]
print(f"      median v_feasible: {np.median(v_feasibles):.0f} → {np.median(vf_relaxed):.0f}")

# Fix D: Set D_min floor
D_min_candidate = 20.0
print(f"\n   D. Set D_min floor = {D_min_candidate}s")
vf_with_floor = [max(D_min_candidate, D_adaptive) / rs * 0.5 for rs in rate_sums]
print(f"      Effective deadline = max({D_adaptive:.1f}, {D_min_candidate}) = {max(D_adaptive, D_min_candidate):.1f}")
