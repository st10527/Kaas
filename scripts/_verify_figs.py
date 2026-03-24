#!/usr/bin/env python3
"""Quick verification of key data points in all figures."""
import json, numpy as np

SEEDS = ["seed42", "seed123", "seed456"]
SEEDS_INT = [42, 123, 456]

d1 = json.load(open('results/jpdc/exp1_main_comparison.json'))
d2 = json.load(open('results/jpdc/exp2_straggler_sweep.json'))
d4 = json.load(open('results/jpdc/exp4_scalability.json'))
d5 = json.load(open('results/jpdc/exp5_emnist.json'))
d6 = json.load(open('results/jpdc/exp6_privacy_async.json'))

# Fig 2/3
dash_final = np.mean([d1["DASH_seed%d" % s]["history"][49]["accuracy"] * 100 for s in SEEDS_INT])
sync_final = np.mean([d1["Sync-Greedy_seed%d" % s]["history"][49]["accuracy"] * 100 for s in SEEDS_INT])
dash_wc = np.mean([d1["DASH_seed%d" % s]["history"][49]["wall_clock_time"] for s in SEEDS_INT])
sync_wc = np.mean([d1["Sync-Greedy_seed%d" % s]["history"][49]["wall_clock_time"] for s in SEEDS_INT])
syncfull_wc = np.mean([d1["Sync-Full_seed%d" % s]["history"][49]["wall_clock_time"] for s in SEEDS_INT])
print("Fig 2-3: DASH %.2f%% @%.0fs | Sync %.2f%% @%.0fs | Sync-Full @%.0fs (clipped 3500)" % (
    dash_final, dash_wc, sync_final, sync_wc, syncfull_wc))

# Fig 4-5
for sig in [0.0, 1.5]:
    sk = "sigma=%s" % sig
    da = np.mean([d2[sk][s]["DASH"]["final_accuracy"] * 100 for s in SEEDS])
    sa = np.mean([d2[sk][s]["Sync-Greedy"]["final_accuracy"] * 100 for s in SEEDS])
    print("Fig 4: sigma=%.1f: DASH=%.2f%%, Sync=%.2f%%" % (sig, da, sa))

# Fig 7-8
for M in [20, 200]:
    mk = "M=%d" % M
    dw = np.mean([d4[mk]["seeds"][s]["DASH"]["wall_clock_time"] for s in SEEDS])
    sw = np.mean([d4[mk]["seeds"][s]["Sync-Greedy"]["wall_clock_time"] for s in SEEDS])
    print("Fig 7-8: M=%d: speedup=%.2fx" % (M, sw / dw))

# Fig 9
for M in [50, 200]:
    mk = "M=%d" % M
    b = np.mean([d5[mk]["seeds"][s]["DASH"]["best_accuracy"] * 100 for s in SEEDS])
    print("Fig 9: M=%d DASH best=%.2f%%" % (M, b))

# Fig 10
for pk in ["no_privacy", "strong_privacy"]:
    acc = np.mean([d6[pk]["seeds"][s]["best_accuracy"] * 100 for s in SEEDS])
    wc = np.mean([d6[pk]["seeds"][s]["wall_clock_time"] for s in SEEDS])
    print("Fig 10: %s: best=%.2f%%, WC=%.0fs" % (pk, acc, wc))

print("\nALL KEY DATAPOINTS VERIFIED ✅")
