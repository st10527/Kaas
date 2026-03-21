#!/usr/bin/env python3
"""Extract all raw numbers from Exp 1 and Exp 2 for summary."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# === Exp 1 ===
d1 = json.load(open(ROOT / "results/jpdc/exp1_main_comparison.json"))
methods1 = ['DASH', 'Sync-Greedy', 'FedBuff-FD', 'Random-Async', 'Full-Async', 'Sync-Full']
seeds = [42, 123, 456]

print("=== EXP1 SUMMARY ===")
for m in methods1:
    accs = [d1[f"{m}_seed{s}"]["final_accuracy"] for s in seeds]
    bests = [d1[f"{m}_seed{s}"]["best_accuracy"] for s in seeds]
    wcs = [d1[f"{m}_seed{s}"]["wall_clock_time"] for s in seeds]
    gpus = [d1[f"{m}_seed{s}"]["total_time"] for s in seeds]
    print(f"  {m}: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}  "
          f"best={np.mean(bests)*100:.2f}  wc={np.mean(wcs):.0f}±{np.std(wcs):.0f}  "
          f"gpu={np.mean(gpus)/60:.1f}min  "
          f"per_seed=[{accs[0]*100:.2f}, {accs[1]*100:.2f}, {accs[2]*100:.2f}]")

print("\n=== EXP1 CONVERGENCE (mean of 3 seeds) ===")
for rnd in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
    parts = [f"{rnd:3d}"]
    for m in methods1:
        a = np.mean([d1[f"{m}_seed{s}"]["history"][rnd]["accuracy"] for s in seeds])
        parts.append(f"{a*100:6.2f}")
    print("  " + " | ".join(parts))

# Speedup
print("\n=== EXP1 SPEEDUP vs Sync-Greedy ===")
sync_wc = np.mean([d1[f"Sync-Greedy_seed{s}"]["wall_clock_time"] for s in seeds])
sync_acc = np.mean([d1[f"Sync-Greedy_seed{s}"]["final_accuracy"] for s in seeds])
for m in methods1:
    mwc = np.mean([d1[f"{m}_seed{s}"]["wall_clock_time"] for s in seeds])
    macc = np.mean([d1[f"{m}_seed{s}"]["final_accuracy"] for s in seeds])
    print(f"  {m}: speedup={sync_wc/mwc:.2f}x  acc_gap={((macc-sync_acc)*100):+.2f}pp")

# === Exp 2 ===
d2 = json.load(open(ROOT / "results/jpdc/exp2_straggler_sweep.json"))
sigmas = ['sigma=0.0', 'sigma=0.3', 'sigma=0.5', 'sigma=1.0', 'sigma=1.5']
methods2 = ['DASH', 'Sync-Greedy', 'FedBuff-FD']
seedkeys = ['seed42', 'seed123', 'seed456']

print("\n=== EXP2 SUMMARY ===")
for sigma in sigmas:
    s_label = sigma.split("=")[1]
    print(f"\n  σ={s_label}:")
    for m in methods2:
        accs = [d2[sigma][s][m]["final_accuracy"] for s in seedkeys]
        wcs = [d2[sigma][s][m]["wall_clock_time"] for s in seedkeys]
        nps = []
        for s in seedkeys:
            h = d2[sigma][s][m]["history"]
            nps.append(np.mean([r["n_participants"] for r in h]))
        print(f"    {m}: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}  "
              f"wc={np.mean(wcs):.0f}±{np.std(wcs):.0f}  "
              f"avg_part={np.mean(nps):.1f}")

print("\n=== EXP2 DASH vs Sync-Greedy per σ ===")
for sigma in sigmas:
    s_label = sigma.split("=")[1]
    da = np.mean([d2[sigma][s]["DASH"]["final_accuracy"] for s in seedkeys]) * 100
    sa = np.mean([d2[sigma][s]["Sync-Greedy"]["final_accuracy"] for s in seedkeys]) * 100
    dw = np.mean([d2[sigma][s]["DASH"]["wall_clock_time"] for s in seedkeys])
    sw = np.mean([d2[sigma][s]["Sync-Greedy"]["wall_clock_time"] for s in seedkeys])
    print(f"  σ={s_label}: DASH={da:.2f}% Sync={sa:.2f}% gap={da-sa:+.2f}pp  "
          f"DASH_WC={dw:.0f} Sync_WC={sw:.0f} speedup={sw/dw:.2f}x")
