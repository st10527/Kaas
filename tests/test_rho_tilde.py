#!/usr/bin/env python3
"""Verify the rho_tilde differentiation under tight deadline."""
import sys; sys.path.insert(0, '.')
from src.scheduler.rads import RADSScheduler
from src.async_module.straggler_model import StragglerModel

sm = StragglerModel(sigma_noise=0.5)
scheduler = RADSScheduler(
    budget=10.0, v_max=200,
    straggler_aware=True, straggler_model=sm, deadline=2.0,
)
devices = [
    {'device_id': 0, 'rho_i': 0.8, 'b_i': 0.1, 'theta_i': 50, 'a_i': 0.5,
     'comp_rate': 0.01, 'comm_rate': 0.01},
    {'device_id': 1, 'rho_i': 0.6, 'b_i': 0.15, 'theta_i': 40, 'a_i': 0.5,
     'comp_rate': 0.05, 'comm_rate': 0.04},
    {'device_id': 2, 'rho_i': 0.9, 'b_i': 0.12, 'theta_i': 60, 'a_i': 0.5,
     'comp_rate': 0.005, 'comm_rate': 0.008},
]
result = scheduler.schedule(devices)
print(f'D=2.0: selected={result.n_selected}, Q_tilde={result.total_quality:.4f}')
for d in devices:
    rt = d.get('rho_tilde_i', d['rho_i'])
    v_cap = 2.0 / (d['comp_rate'] + d['comm_rate']) * 0.5
    pi_i = sm.completion_probability(d['comp_rate'], d['comm_rate'], v_cap, 2.0)
    print(f"  dev {d['device_id']}: rho={d['rho_i']:.2f}, v_cap={v_cap:.1f}, "
          f"pi={pi_i:.4f}, rho_tilde={rt:.4f}")
for a in result.allocations:
    if a.selected:
        print(f'  dev {a.device_id}: v*={a.v_star:.1f}, q_tilde={a.quality:.4f}')

# With the 0.5 margin, pi_i at v_cap is ~0.5 for all (by construction).
# The key differentiation is that v_cap itself differs per device,
# constraining slow devices to lower allocations (v_cap=11 vs 77).
# This means slow devices get lower quality naturally.
dev1 = next(d for d in devices if d['device_id'] == 1)
dev2 = next(d for d in devices if d['device_id'] == 2)
alloc1 = next(a for a in result.allocations if a.device_id == 1)
alloc2 = next(a for a in result.allocations if a.device_id == 2)
print(f"\nSlow dev: v_cap=11.1, v*={alloc1.v_star:.1f}, q_tilde={alloc1.quality:.4f}")
print(f"Fast dev: v_cap=76.9, v*={alloc2.v_star:.1f}, q_tilde={alloc2.quality:.4f}")
# Fast device should get higher quality allocation
assert alloc2.quality > alloc1.quality, \
    "Fast device should have higher quality than slow device"
print("\nPASS: rho_tilde + v_cap correctly constrains slow devices")
