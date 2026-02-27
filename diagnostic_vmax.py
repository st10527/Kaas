#!/usr/bin/env python3
"""
KaaS-Edge v2 Diagnostic & Fix
Run BEFORE experiments to verify config is correct.

Usage:
    cd paid-fd-main
    python diagnostic_vmax.py
"""
import subprocess, os, sys, shutil

PROJECT = os.path.dirname(os.path.abspath(__file__))

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

# ── Step 3: Verify config propagation ──
print("=" * 60)
print("Step 3: Verify v_max config propagation")
print("=" * 60)

sys.path.insert(0, os.path.join(PROJECT, "src"))
try:
    from methods.kaas_edge import KaaSEdgeConfig
    cfg = KaaSEdgeConfig(budget=50.0, v_max=10000)
    print(f"  KaaSEdgeConfig(budget=50, v_max=10000):")
    print(f"    cfg.budget = {cfg.budget}")
    print(f"    cfg.v_max  = {cfg.v_max}")
    assert cfg.v_max == 10000, f"v_max is {cfg.v_max}, expected 10000!"
    print(f"  ✓ Config propagation correct.\n")
except Exception as e:
    print(f"  ✗ Error: {e}\n")
    sys.exit(1)

# ── Step 4: Test RADS allocation ──
print("=" * 60)
print("Step 4: Test RADS allocation (should see ~8-9 MB, NOT ~0.85)")
print("=" * 60)

from methods.kaas_edge import generate_edge_devices
from scheduler.rads import RADSScheduler

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
