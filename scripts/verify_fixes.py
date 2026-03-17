#!/usr/bin/env python3
"""Verify the 3 fixes produce reasonable numbers."""
import sys
sys.path.insert(0, '/Users/nclab/Kaas')

import numpy as np
from src.async_module.straggler_model import StragglerModel
from src.methods.kaas_edge import generate_edge_devices

# Generate same devices
rates = StragglerModel.generate_device_rates(50, seed=42)
edge_devs = generate_edge_devices(50, seed=42)
for dev, rate in zip(edge_devs, rates):
    dev['comp_rate'] = rate['comp_rate']
    dev['comm_rate'] = rate['comm_rate']

rate_sums = [d['comp_rate'] + d['comm_rate'] for d in edge_devs]

# ── Fix A: New warmup deadline ──
M = 50
total_a = sum(d.get('a_i', 0.0) for d in edge_devs)
budget = 50.0
v_max = 10000
residual = max(budget - total_a, 1.0)
median_b = float(np.median([d.get('b_i', 0.001) for d in edge_devs]))
estimated_v = min(residual / (M * median_b), v_max)
p85_rate = float(np.percentile(rate_sums, 85))
D_warmup_new = max(p85_rate * estimated_v * 1.5, 10.0)

# Old for comparison
p90_rate = float(np.percentile(rate_sums, 90))
D_warmup_old = p90_rate * v_max * 0.4 * 2.0

print("=" * 60)
print("FIX A: Warmup Deadline")
print("=" * 60)
print(f"  estimated_v (budget-based): {estimated_v:.0f}")
print(f"  OLD: D_warmup = {D_warmup_old:.1f}s  (3 rounds = {3*D_warmup_old:.0f}s)")
print(f"  NEW: D_warmup = {D_warmup_new:.1f}s  (3 rounds = {3*D_warmup_new:.0f}s)")
print(f"  Reduction: {D_warmup_old/D_warmup_new:.1f}x")

# Simulate warmup latencies with new deadline
sm = StragglerModel(sigma_noise=0.5, seed=42)
warmup_v = int(estimated_v)
warmup_devs = [{'device_id': i, 'v_star': warmup_v,
                'comp_rate': rates[i]['comp_rate'],
                'comm_rate': rates[i]['comm_rate']} for i in range(50)]
lat = sm.simulate_round(warmup_devs, deadline=D_warmup_new)
taus = [l.tau_total for l in lat]
n_complete = sum(1 for l in lat if l.outcome == 'complete')
n_partial = sum(1 for l in lat if l.outcome == 'partial')
n_timeout = sum(1 for l in lat if l.outcome == 'timeout')
print(f"  With D_warmup={D_warmup_new:.1f}s: {n_complete} complete, {n_partial} partial, {n_timeout} timeout")
print(f"  Latencies: p50={np.median(taus):.1f}  p85={np.percentile(taus, 85):.1f}  max={max(taus):.1f}")

# ── Fix B: New adaptive deadline ──
print(f"\n{'=' * 60}")
print("FIX B: Adaptive Percentile")
print("=" * 60)
D_p70 = np.percentile(taus, 70)
D_p85 = np.percentile(taus, 85)
print(f"  OLD (p70): D_adaptive = {D_p70:.1f}s")
print(f"  NEW (p85): D_adaptive = {D_p85:.1f}s")
print(f"  Improvement: {D_p85/D_p70:.1f}x more generous")

# ── Fix C: New v_feasible ──
D_post = D_p85  # This is what round 3+ will use
print(f"\n{'=' * 60}")
print("FIX C: v_feasible Margin")
print("=" * 60)
vf_old = [D_post / rs * 0.5 for rs in rate_sums]
vf_new = [D_post / rs * 0.8 for rs in rate_sums]
print(f"  With D_adaptive = {D_post:.1f}s:")
print(f"  OLD (margin=0.5): median v_feasible = {np.median(vf_old):.0f}")
print(f"  NEW (margin=0.8): median v_feasible = {np.median(vf_new):.0f}")
print(f"  Budget v_per_device: {estimated_v:.0f}")
print(f"  Binding: {'DEADLINE' if np.median(vf_new) < estimated_v else 'BUDGET (good!)'}")

# ── Combined effect on quality ──
print(f"\n{'=' * 60}")
print("COMBINED EFFECT: Quality Estimate")
print("=" * 60)
median_theta = float(np.median([d['theta_i'] for d in edge_devs]))

# DASH post-warmup: v is min(budget-based, deadline-based)
v_dash = min(estimated_v, np.median(vf_new))
v_sync = estimated_v  # Sync-Greedy uses full budget allocation

q_dash = v_dash / (v_dash + median_theta)
q_sync = v_sync / (v_sync + median_theta)

print(f"  DASH v_per_device ≈ {v_dash:.0f}, quality ≈ {q_dash:.3f}")
print(f"  Sync v_per_device ≈ {v_sync:.0f}, quality ≈ {q_sync:.3f}")
print(f"  Quality ratio: {q_dash/q_sync:.2f}x {'✓ GOOD' if q_dash/q_sync > 0.85 else '⚠ Still low'}")

# ── Wall-clock projection ──
print(f"\n{'=' * 60}")
print("WALL-CLOCK PROJECTION (50 rounds)")
print("=" * 60)
# Post-warmup: simulate a round with D_post
sm2 = StragglerModel(sigma_noise=0.5, seed=99)
post_devs = [{'device_id': i, 'v_star': int(v_dash),
              'comp_rate': rates[i]['comp_rate'],
              'comm_rate': rates[i]['comm_rate']} for i in range(50)]
lat2 = sm2.simulate_round(post_devs, deadline=D_post)
n_c2 = sum(1 for l in lat2 if l.outcome == 'complete')
print(f"  Post-warmup: D={D_post:.1f}s, {n_c2}/50 complete")

dash_wc = 3 * D_warmup_new + 47 * D_post
sync_wc = 50 * max(taus)  # sync waits for slowest every round
print(f"  DASH total WC ≈ {3}×{D_warmup_new:.0f} + {47}×{D_post:.0f} = {dash_wc:.0f}s")
print(f"  Sync total WC ≈ {50}×{max(taus):.0f} = {sync_wc:.0f}s")
print(f"  DASH speedup: {sync_wc/dash_wc:.2f}x {'✓' if dash_wc < sync_wc else '✗'}")
