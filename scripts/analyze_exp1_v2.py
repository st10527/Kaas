#!/usr/bin/env python3
"""Analyze Exp 1 results (v2) — per-round breakdown."""
import json
import sys

with open('results/jpdc/exp1_main_comparison.json') as f:
    data = json.load(f)

# Summary table
print("=" * 70)
print("SUMMARY")
print("=" * 70)
fmt = "{:<20} {:>6} {:>10} {:>12} {:>10}"
print(fmt.format("Method", "Rounds", "Final Acc", "WC(s)", "WC/rnd"))
print("-" * 62)
for key, val in data.items():
    h = val['history']
    last = h[-1]
    acc = last['accuracy']
    wc = last.get('wall_clock_time', 0)
    n = len(h)
    wc_per = wc / n if n > 0 else 0
    print(fmt.format(val['method'], n, f"{acc:.4f}", f"{wc:.1f}", f"{wc_per:.1f}"))

print()

# Per-method round-by-round
for key, val in data.items():
    method = val['method']
    h = val['history']
    print("=" * 70)
    print(f"  {method}")
    print("=" * 70)
    hdr = "{:>3} {:>7} {:>8} {:>7} {:>5} {:>8}  {}"
    print(hdr.format("Rnd", "Acc", "WC(s)", "dWC", "nPrt", "comm_mb", "Extra"))
    print("-" * 70)
    prev_wc = 0
    for r in h:
        acc = r['accuracy']
        wc = r.get('wall_clock_time', 0)
        dwc = wc - prev_wc
        prev_wc = wc
        ex = r.get('extra', {})
        n_part = r.get('n_participants', '?')
        comm = ex.get('comm_mb', 0)
        parts = []
        if 'n_complete' in ex:
            parts.append("C:{} P:{} T:{}".format(
                ex['n_complete'], ex.get('n_partial', 0), ex.get('n_timeout', 0)))
        if 'deadline' in ex:
            parts.append("D={:.1f}".format(ex['deadline']))
        if 'n_selected' in ex:
            parts.append("sel={}".format(ex['n_selected']))
        if 'sync_simulated' in ex:
            parts.append("(sync-sim)")
        info = ' | '.join(parts)
        row = "{:>3} {:>7.4f} {:>8.1f} {:>7.1f} {:>5} {:>8.2f}  {}"
        print(row.format(r['round'], acc, wc, dwc, n_part, comm, info))
    print()

# Accuracy convergence comparison
print("=" * 70)
print("ACCURACY CONVERGENCE (every round)")
print("=" * 70)
methods = list(data.keys())
header = "{:>3}".format("Rnd")
for k in methods:
    header += "  {:>12}".format(data[k]['method'][:12])
print(header)
print("-" * (3 + 14 * len(methods)))
max_rounds = max(len(data[k]['history']) for k in methods)
for i in range(max_rounds):
    row = "{:>3}".format(i)
    for k in methods:
        h = data[k]['history']
        if i < len(h):
            row += "  {:>12.4f}".format(h[i]['accuracy'])
        else:
            row += "  {:>12}".format("-")
    print(row)

# Wall-clock milestones
print()
print("=" * 70)
print("WALL-CLOCK AT ACCURACY MILESTONES")
print("=" * 70)
thresholds = [0.08, 0.10, 0.12, 0.14, 0.16]
header = "{:<20}".format("Method")
for t in thresholds:
    header += "  {:>8}".format(f">{t:.0%}")
print(header)
print("-" * (20 + 10 * len(thresholds)))
for k, val in data.items():
    row = "{:<20}".format(val['method'])
    h = val['history']
    for t in thresholds:
        wc_at = None
        for r in h:
            if r['accuracy'] >= t:
                wc_at = r.get('wall_clock_time', 0)
                break
        if wc_at is not None:
            row += "  {:>8.1f}".format(wc_at)
        else:
            row += "  {:>8}".format("never")
    print(row)
