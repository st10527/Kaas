#!/usr/bin/env python3
"""
KaaS-Edge v2 Diagnostic & Fix
Run BEFORE experiments to verify config is correct.

Usage:
    cd paid-fd-main
    python diagnostic_vmax.py
"""
import subprocess, os, sys, shutil

PROJECT = os.getcwd()  # Run from project root: cd paid-fd-main && python diagnostic_vmax.py

# ── Step 1: Clear all __pycache__ ──
print("=" * 60)
print("Step 1: Clearing __pycache__ (stale bytecode)")
print("=" * 60)
count = 0
for root, dirs, _ in os.walk(PROJECT):
    for d in dirs:
        if d == "__pycache__":
            p = os.path.join(root, d)
            shutil.rmtree(p)
            print(f"  Removed: {p}")
            count += 1
print(f"  Cleared {count} __pycache__ directories.\n")

# ── Step 2: Verify kaas_edge.py has v2 markers ──
print("=" * 60)
print("Step 2: Verify kaas_edge.py is v2")
print("=" * 60)
kaas_path = os.path.join(PROJECT, "src", "methods", "kaas_edge.py")
if not os.path.exists(kaas_path):
    print(f"  ✗ NOT FOUND: {kaas_path}")
    sys.exit(1)

with open(kaas_path) as f:
    content = f.read()

checks = {
    "v2 header":         "v_i subsampling" in content,
    "apply_ldp_noise":   "def apply_ldp_noise" in content,
    "compute_comm_mb":   "def compute_comm_mb" in content,
    "rng.choice (subsample)": "rng.choice(n_ref, size=v_i" in content,
    "masked aggregation": "agg_logits" in content or "wsum" in content,
    "log-normal costs":  "loc=-6.2" in content,
}
all_ok = True
for name, ok in checks.items():
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        all_ok = False

if not all_ok:
    print("\n  ⚠ kaas_edge.py is NOT the v2 version!")
    print("  Please replace it with the v2 file from edge_v2_code/kaas_edge.py")
    sys.exit(1)
print("  ✓ All v2 markers present.\n")

# ── Step 3: Verify v_max config (parsed from source, no import needed) ──
print("=" * 60)
print("Step 3: Verify v_max config propagation")
print("=" * 60)

import re
m = re.search(r'v_max:\s*float\s*=\s*([\d.]+)', content)
if m:
    default_vmax = float(m.group(1))
    print(f"  KaaSEdgeConfig default v_max = {default_vmax}")
    if default_vmax < 1000:
        print(f"  ✗ v_max={default_vmax} is too small! Should be 10000.")
        print(f"    This is why CommMB was ~0.85 instead of ~8.5")
    else:
        print(f"  ✓ v_max default is correct ({default_vmax}).")
else:
    print("  ⚠ Could not parse v_max from source.")

# Also check run_edge_experiments.py passes v_max correctly
exp_path = os.path.join(PROJECT, "scripts", "run_edge_experiments.py")
if os.path.exists(exp_path):
    with open(exp_path) as f:
        exp_content = f.read()
    if "v_max=n_public" in exp_content or "v_max=10000" in exp_content:
        print(f"  ✓ run_edge_experiments.py passes v_max=n_public (10000).")
    else:
        print(f"  ⚠ run_edge_experiments.py may not set v_max correctly.")
print()

# ── Step 4: Test RADS allocation ──
print("=" * 60)
print("Step 4: Test RADS allocation (should see ~8-9 MB, NOT ~0.85)")
print("=" * 60)

# Import only rads.py (no relative imports) and reproduce device generation inline
sys.path.insert(0, os.path.join(PROJECT, "src"))
from scheduler.rads import RADSScheduler

import numpy as np

# Reproduce generate_edge_devices inline (avoids relative import)
def generate_edge_devices(n_devices=20, seed=42):
    rng = np.random.RandomState(seed)
    log_b = rng.normal(loc=-6.2, scale=0.8, size=n_devices)
    b_vals = np.clip(np.exp(log_b), 0.0003, 0.1)
    theta_vals = rng.uniform(30.0, 100.0, size=n_devices)
    a_vals = rng.uniform(0.1, 0.5, size=n_devices)
    rho_pool = []
    for val, frac in [(1.0, 0.15), (0.8, 0.20), (0.5, 0.30), (0.2, 0.20), (0.05, 0.15)]:
        rho_pool.extend([val] * max(1, int(n_devices * frac)))
    while len(rho_pool) < n_devices: rho_pool.append(0.5)
    rho_pool = rho_pool[:n_devices]
    rng.shuffle(rho_pool)
    devices = []
    for i in range(n_devices):
        rho_i = float(np.clip(rho_pool[i] + rng.uniform(-0.05, 0.05), 0.01, 1.0))
        devices.append({'device_id': i, 'rho_i': rho_i, 'b_i': float(b_vals[i]),
                        'theta_i': float(theta_vals[i]), 'a_i': float(a_vals[i])})
    return devices

devices = generate_edge_devices(n_devices=20, seed=42)
sched = RADSScheduler(budget=50.0, v_max=10000)
result = sched.schedule(devices)

total_v = sum(int(min(a.v_star, 10000)) for a in result.allocations if a.selected)
comm_mb = total_v * 100 * 4 / (1024**2)

print(f"  Selected: {result.n_selected}/20 devices")
print(f"  Total vectors: {total_v}")
print(f"  CommMB: {comm_mb:.2f} MB")
print(f"  RADS v_max: {sched.v_max}")

if comm_mb < 2.0:
    print(f"\n  ✗ CommMB too low ({comm_mb:.2f})! v_max may be capped.")
    print(f"    sched.v_max = {sched.v_max}")
    for a in result.allocations:
        if a.selected:
            print(f"      Dev {a.device_id}: v*={a.v_star:.1f}")
            break
else:
    print(f"  ✓ Allocation looks correct.\n")

# ── Step 5: Quick fix — also set v_max default higher ──
print("=" * 60)
print("Step 5: Safety fix — change default v_max from 200 to 10000")
print("=" * 60)

if "v_max: float = 200.0" in content:
    new_content = content.replace("v_max: float = 200.0", "v_max: float = 10000.0")
    with open(kaas_path, 'w') as f:
        f.write(new_content)
    print("  ✓ Changed default v_max: 200.0 → 10000.0 in KaaSEdgeConfig")
    print("    (This is a safety measure; run_edge_experiments.py also sets it)")
elif "v_max: float = 10000.0" in content:
    print("  ✓ Default v_max is already 10000.0")
else:
    print("  ⚠ Could not find v_max default to modify")

print("\n" + "=" * 60)
print("Diagnostic complete. Now re-run:")
print("  python scripts/run_edge_experiments.py --experiment main --device cuda:0")
print("=" * 60)
