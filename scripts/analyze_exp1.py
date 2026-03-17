#!/usr/bin/env python3
"""Detailed analysis of JPDC Exp 1: Main Comparison (quick run)."""
import json
import numpy as np

with open('results/jpdc/exp1_main_comparison.json') as f:
    data = json.load(f)

display_order = ['DASH', 'KaaS-Edge', 'Sync-Greedy', 'FedBuff-FD',
                 'Random-Async', 'Full-Async', 'Sync-Full']

print("=" * 80)
print("  JPDC Exp 1: Main Comparison -- Quick Run (seed=42, 10 rounds, 10 devices)")
print("=" * 80)

# Summary Table
print()
header = f"{'Method':<16} {'FinalAcc':>9} {'BestAcc':>9} {'WallClk(s)':>11} {'CommMB':>8} {'AvgSel':>7} {'CompRate':>9}"
print(header)
print("-" * len(header))

for name in display_order:
    key = name + "_seed42"
    if key not in data:
        continue
    r = data[key]
    h = r['history']
    fa = r['final_accuracy']
    ba = r['best_accuracy']
    wc = h[-1].get('wall_clock_time', 0)

    total_comm = sum(rd.get('extra', {}).get('comm_mb', 0) for rd in h)
    n_sel_list = [rd.get('extra', {}).get('n_selected', rd.get('n_participants', 0)) for rd in h]
    n_comp_list = [rd.get('extra', {}).get('n_complete', 0) for rd in h]
    avg_sel = np.mean(n_sel_list)

    total_sel = sum(n_sel_list)
    total_comp = sum(n_comp_list)
    comp_rate = total_comp / total_sel * 100 if total_sel > 0 else 0

    label = 'Sync-Greedy' if name == 'KaaS-Edge' else name
    print(f"{label:<16} {fa:>9.4f} {ba:>9.4f} {wc:>11.1f} {total_comm:>8.1f} {avg_sel:>7.1f} {comp_rate:>8.1f}%")

# DASH Round-by-Round
print()
print("=" * 80)
print("  DASH: Round-by-Round Detail")
print("=" * 80)
print(f"  {'Rnd':>3} {'Acc':>7} {'Loss':>7} {'WallClk':>8} {'CommMB':>7} {'nSel':>5} {'nComp':>6} {'nPart':>6} {'nTout':>6} {'Deadline':>9} {'Quality':>8}")
print("  " + "-" * 80)

dash = data['DASH_seed42']
for rd in dash['history']:
    ex = rd.get('extra', {})
    print(f"  {rd['round']:>3} {rd['accuracy']:>7.4f} {rd['loss']:>7.4f} "
          f"{rd['wall_clock_time']:>8.1f} {ex.get('comm_mb', 0):>7.1f} "
          f"{ex.get('n_selected', 0):>5} {ex.get('n_complete', 0):>6} "
          f"{ex.get('n_partial', 0):>6} {ex.get('n_timeout', 0):>6} "
          f"{ex.get('deadline', 0):>9.1f} {ex.get('total_quality', 0):>8.3f}")

# Sync-Greedy / KaaS-Edge Round-by-Round
sync_key = 'KaaS-Edge_seed42' if 'KaaS-Edge_seed42' in data else 'Sync-Greedy_seed42'
if sync_key in data:
    print()
    print("=" * 80)
    print("  Sync-Greedy: Round-by-Round Detail")
    print("=" * 80)
    print(f"  {'Rnd':>3} {'Acc':>7} {'Loss':>7} {'nPart':>6} {'CommMB':>7} {'Quality':>8} {'Cost':>8}")
    print("  " + "-" * 55)
    sync = data[sync_key]
    for rd in sync['history']:
        ex = rd.get('extra', {})
        print(f"  {rd['round']:>3} {rd['accuracy']:>7.4f} {rd['loss']:>7.4f} "
              f"{rd.get('n_participants', 0):>6} {ex.get('comm_mb', 0):>7.1f} "
              f"{ex.get('total_quality', 0):>8.3f} {ex.get('total_cost', 0):>8.3f}")

# Convergence: accuracy at wall-clock milestones
print()
print("=" * 80)
print("  Accuracy at Wall-Clock Milestones")
print("=" * 80)

milestones = [50, 100, 200, 400, 600]
header2 = f"  {'Method':<16}"
for ms in milestones:
    header2 += f" {'@' + str(ms) + 's':>8}"
print(header2)
print("  " + "-" * (16 + 9 * len(milestones)))

for name in display_order:
    key = name + "_seed42"
    if key not in data:
        continue
    r = data[key]
    h = r['history']
    label = 'Sync-Greedy' if name == 'KaaS-Edge' else name
    line = f"  {label:<16}"
    for ms in milestones:
        best_at = 0
        for rd in h:
            wc = rd.get('wall_clock_time', 0)
            if wc <= ms:
                best_at = max(best_at, rd['accuracy'])
        line += f" {best_at:>8.4f}"
    print(line)

# FedBuff-FD round-by-round
if 'FedBuff-FD_seed42' in data:
    print()
    print("=" * 80)
    print("  FedBuff-FD: Round-by-Round Detail")
    print("=" * 80)
    print(f"  {'Rnd':>3} {'Acc':>7} {'Loss':>7} {'WallClk':>8} {'nPart':>6} {'CommMB':>7} {'nComp':>6} {'nTout':>6}")
    print("  " + "-" * 60)
    fb = data['FedBuff-FD_seed42']
    for rd in fb['history']:
        ex = rd.get('extra', {})
        print(f"  {rd['round']:>3} {rd['accuracy']:>7.4f} {rd['loss']:>7.4f} "
              f"{rd.get('wall_clock_time', 0):>8.1f} {rd.get('n_participants', 0):>6} "
              f"{ex.get('comm_mb', 0):>7.1f} {ex.get('n_complete', 0):>6} "
              f"{ex.get('n_timeout', 0):>6}")

print()
