#!/usr/bin/env python3
"""Export clean raw data tables for all JPDC experiments.

Output: results/jpdc/JPDC_raw_data_clean.md
This file is designed to be copy-pasted into Claude web for paper writing.
"""

import json
import numpy as np
from pathlib import Path

RESULTS = Path("results/jpdc")
SEEDS3 = [42, 123, 456]
SEED_KEYS = ["seed42", "seed123", "seed456"]
OUT = RESULTS / "JPDC_raw_data_clean.md"


def stats(vals):
    return np.mean(vals), np.std(vals)


def load(fn):
    with open(RESULTS / fn) as f:
        return json.load(f)


def export_exp1(f):
    data = load("exp1_main_comparison.json")
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD", "Random-Async", "Full-Async", "Sync-Full"]

    f.write("## Exp 1: Main Comparison (CIFAR-100, M=50, B=50, σ=0.5)\n\n")
    f.write("### Table I — Mean ± Std over 3 seeds\n\n")
    f.write("| Method | Final Acc (%) | Best Acc (%) | WC (s) | Speedup vs Sync |\n")
    f.write("|--------|--------------|-------------|--------|----------------|\n")

    sync_wc = None
    rows = {}
    for m in methods:
        finals = [data[f"{m}_seed{s}"]["final_accuracy"] * 100 for s in SEEDS3]
        bests = [data[f"{m}_seed{s}"]["best_accuracy"] * 100 for s in SEEDS3]
        wcs = [data[f"{m}_seed{s}"]["wall_clock_time"] for s in SEEDS3]
        rows[m] = {"final": finals, "best": bests, "wc": wcs}
        if m == "Sync-Greedy":
            sync_wc = np.mean(wcs)

    for m in methods:
        r = rows[m]
        fm, fs = stats(r["final"])
        bm, bs = stats(r["best"])
        wm, ws = stats(r["wc"])
        spd = f"{sync_wc / wm:.2f}×" if wm > 0 else "—"
        f.write(f"| {m} | {fm:.2f} ± {fs:.2f} | {bm:.2f} ± {bs:.2f} | {wm:.0f} ± {ws:.0f} | {spd} |\n")

    # Per-round convergence (for Fig 2, 3)
    f.write("\n### Per-Round Convergence (mean of 3 seeds)\n\n")
    f.write("| Round |")
    for m in methods:
        f.write(f" {m} Acc |")
    f.write(f" DASH WC | Sync WC |\n")
    f.write("|-------|" + "---------|" * len(methods) + "---------|----------|\n")

    for r in range(50):
        f.write(f"| {r+1} |")
        for m in methods:
            accs = []
            for s in SEEDS3:
                h = data[f"{m}_seed{s}"]["history"]
                if r < len(h):
                    accs.append(h[r]["accuracy"] * 100)
            f.write(f" {np.mean(accs):.2f} |")
        # WC cumulative for DASH and Sync
        dash_wcs = []
        sync_wcs = []
        for s in SEEDS3:
            dh = data[f"DASH_seed{s}"]["history"]
            sh = data[f"Sync-Greedy_seed{s}"]["history"]
            if r < len(dh):
                dash_wcs.append(dh[r].get("wall_clock_time", 0))
            if r < len(sh):
                sync_wcs.append(sh[r].get("wall_clock_time", 0))
        f.write(f" {np.mean(dash_wcs):.1f} | {np.mean(sync_wcs):.1f} |\n")

    f.write("\n---\n\n")


def export_exp2(f):
    data = load("exp2_straggler_sweep.json")
    sigmas = [0.0, 0.3, 0.5, 1.0, 1.5]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    f.write("## Exp 2: Straggler Severity Sweep (CIFAR-100, M=50, B=50)\n\n")
    f.write("### Accuracy vs σ\n\n")
    f.write("| σ |")
    for m in methods:
        f.write(f" {m} Final | {m} Best |")
    f.write("\n|---|" + "---------|---------|" * len(methods) + "\n")

    for sig in sigmas:
        sk = f"sigma={sig}"
        f.write(f"| {sig} |")
        for m in methods:
            finals = [data[sk][s][m]["final_accuracy"] * 100 for s in SEED_KEYS]
            bests = [data[sk][s][m]["best_accuracy"] * 100 for s in SEED_KEYS]
            f.write(f" {np.mean(finals):.2f}±{np.std(finals):.2f} | {np.mean(bests):.2f}±{np.std(bests):.2f} |")
        f.write("\n")

    f.write("\n### Wall-Clock & Speedup vs σ\n\n")
    f.write("| σ | DASH WC (s) | Sync WC (s) | FedBuff WC (s) | Speedup |\n")
    f.write("|---|------------|------------|---------------|--------|\n")
    for sig in sigmas:
        sk = f"sigma={sig}"
        dash_wc = np.mean([data[sk][s]["DASH"]["wall_clock_time"] for s in SEED_KEYS])
        sync_wc = np.mean([data[sk][s]["Sync-Greedy"]["wall_clock_time"] for s in SEED_KEYS])
        fb_wc = np.mean([data[sk][s]["FedBuff-FD"]["wall_clock_time"] for s in SEED_KEYS])
        spd = sync_wc / dash_wc if dash_wc > 0 else 0
        f.write(f"| {sig} | {dash_wc:.0f} | {sync_wc:.0f} | {fb_wc:.0f} | {spd:.2f}× |\n")

    f.write("\n---\n\n")


def export_exp3(f):
    d3 = load("exp3_policy_comparison.json")
    d3b = load("exp3b_policy_nofloor.json")
    policies = ["fixed(5.0)", "fixed(10.0)", "fixed(20.0)",
                "adaptive(0.5)", "adaptive(0.7)", "adaptive(0.9)"]

    f.write("## Exp 3: Policy Comparison & D_min Ablation (CIFAR-100, M=50, B=50)\n\n")
    f.write("### Paired Comparison: +D_min vs −D_min\n\n")
    f.write("| Policy | +D_min Final | +D_min Best | +D_min WC | −D_min Final | −D_min Best | −D_min WC | ΔAcc (pp) |\n")
    f.write("|--------|------------|-----------|---------|------------|-----------|---------|----------|\n")

    for p in policies:
        # +D_min
        with_finals = [d3[p]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS]
        with_bests = [d3[p]["seeds"][s]["best_accuracy"] * 100 for s in SEED_KEYS]
        with_wcs = [d3[p]["seeds"][s]["wall_clock_time"] for s in SEED_KEYS]
        # -D_min
        wo_finals = [d3b[p]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS]
        wo_bests = [d3b[p]["seeds"][s]["best_accuracy"] * 100 for s in SEED_KEYS]
        wo_wcs = [d3b[p]["seeds"][s]["wall_clock_time"] for s in SEED_KEYS]

        delta = np.mean(with_finals) - np.mean(wo_finals)
        f.write(f"| {p} | {np.mean(with_finals):.2f}±{np.std(with_finals):.2f} | "
                f"{np.mean(with_bests):.2f}±{np.std(with_bests):.2f} | "
                f"{np.mean(with_wcs):.0f} | "
                f"{np.mean(wo_finals):.2f}±{np.std(wo_finals):.2f} | "
                f"{np.mean(wo_bests):.2f}±{np.std(wo_bests):.2f} | "
                f"{np.mean(wo_wcs):.0f} | "
                f"{delta:+.2f} |\n")

    f.write("\n---\n\n")


def export_exp4(f):
    data = load("exp4_scalability.json")
    ms = [20, 50, 100, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    f.write("## Exp 4: Scalability (CIFAR-100, B=2.5×M)\n\n")
    f.write("### Accuracy & WC vs M\n\n")
    f.write("| M | Budget |")
    for m in methods:
        f.write(f" {m} Final | {m} Best | {m} WC |")
    f.write(" Speedup |\n")
    f.write("|---|--------|" + "---------|---------|--------|" * len(methods) + "---------|\n")

    for M in ms:
        mk = f"M={M}"
        budget = data[mk].get("budget", 2.5 * M)
        f.write(f"| {M} | {budget:.0f} |")
        wcs = {}
        for m in methods:
            finals = [data[mk]["seeds"][s][m]["final_accuracy"] * 100 for s in SEED_KEYS]
            bests = [data[mk]["seeds"][s][m]["best_accuracy"] * 100 for s in SEED_KEYS]
            ws = [data[mk]["seeds"][s][m]["wall_clock_time"] for s in SEED_KEYS]
            wcs[m] = np.mean(ws)
            f.write(f" {np.mean(finals):.2f}±{np.std(finals):.2f} | "
                    f"{np.mean(bests):.2f}±{np.std(bests):.2f} | {np.mean(ws):.0f} |")
        spd = wcs["Sync-Greedy"] / wcs["DASH"] if wcs["DASH"] > 0 else 0
        f.write(f" {spd:.2f}× |\n")

    f.write("\n---\n\n")


def export_exp5(f):
    data = load("exp5_emnist.json")
    ms = [50, 200]
    methods = ["DASH", "Sync-Greedy", "FedBuff-FD"]

    f.write("## Exp 5: Cross-Dataset (EMNIST-ByClass, 62 classes, 687K samples, B=2.5×M)\n\n")
    f.write("### Accuracy & WC\n\n")
    f.write("| M | Method | Final Acc (%) | Best Acc (%) | WC (s) |\n")
    f.write("|---|--------|--------------|-------------|--------|\n")

    for M in ms:
        mk = f"M={M}"
        for m in methods:
            finals = [data[mk]["seeds"][s][m]["final_accuracy"] * 100 for s in SEED_KEYS]
            bests = [data[mk]["seeds"][s][m]["best_accuracy"] * 100 for s in SEED_KEYS]
            wcs = [data[mk]["seeds"][s][m]["wall_clock_time"] for s in SEED_KEYS]
            f.write(f"| {M} | {m} | {np.mean(finals):.2f}±{np.std(finals):.2f} | "
                    f"{np.mean(bests):.2f}±{np.std(bests):.2f} | {np.mean(wcs):.0f}±{np.std(wcs):.0f} |\n")

    f.write("\n### Speedup\n\n")
    f.write("| M | DASH WC | Sync WC | Speedup |\n")
    f.write("|---|---------|---------|--------|\n")
    for M in ms:
        mk = f"M={M}"
        dash_wc = np.mean([data[mk]["seeds"][s]["DASH"]["wall_clock_time"] for s in SEED_KEYS])
        sync_wc = np.mean([data[mk]["seeds"][s]["Sync-Greedy"]["wall_clock_time"] for s in SEED_KEYS])
        f.write(f"| {M} | {dash_wc:.0f} | {sync_wc:.0f} | {sync_wc/dash_wc:.2f}× |\n")

    # Per-round convergence for plotting
    f.write("\n### Per-Round Best Accuracy (M=50, mean of 3 seeds)\n\n")
    f.write("| Round | DASH | Sync-Greedy | FedBuff-FD |\n")
    f.write("|-------|------|-------------|------------|\n")
    for r in range(50):
        f.write(f"| {r+1} |")
        for m in methods:
            accs = []
            for s in SEED_KEYS:
                h = data["M=50"]["seeds"][s][m].get("history", [])
                if r < len(h):
                    accs.append(h[r]["accuracy"] * 100)
            f.write(f" {np.mean(accs):.2f} |")
        f.write("\n")

    f.write("\n---\n\n")


def export_exp6(f):
    data = load("exp6_privacy_async.json")
    privs = [("no_privacy", "None", 1.0), ("mild_privacy", "Mild", 0.8),
             ("moderate_privacy", "Moderate", 0.55), ("strong_privacy", "Strong", 0.1)]

    f.write("## Exp 6: Privacy Sweep (CIFAR-100, M=50, B=50, DASH only)\n\n")
    f.write("### Summary\n\n")
    f.write("| Privacy | Target ρ̃ | Actual avg ρ̃ | Final Acc (%) | Best Acc (%) | WC (s) |\n")
    f.write("|---------|----------|-------------|--------------|-------------|--------|\n")

    for pk, label, target_rho in privs:
        finals = [data[pk]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS]
        bests = [data[pk]["seeds"][s]["best_accuracy"] * 100 for s in SEED_KEYS]
        wcs = [data[pk]["seeds"][s]["wall_clock_time"] for s in SEED_KEYS]
        rhos = [data[pk]["seeds"][s]["avg_rho"] for s in SEED_KEYS]
        f.write(f"| {label} | {target_rho} | {np.mean(rhos):.4f} | "
                f"{np.mean(finals):.2f}±{np.std(finals):.2f} | "
                f"{np.mean(bests):.2f}±{np.std(bests):.2f} | "
                f"{np.mean(wcs):.0f}±{np.std(wcs):.0f} |\n")

    # Convergence
    f.write("\n### Per-Round Convergence (mean of 3 seeds)\n\n")
    f.write("| Round | None | Mild | Moderate | Strong |\n")
    f.write("|-------|------|------|----------|--------|\n")
    for r in range(50):
        f.write(f"| {r+1} |")
        for pk, _, _ in privs:
            accs = [data[pk]["seeds"][s]["history"][r]["accuracy"] * 100
                    for s in SEED_KEYS if r < len(data[pk]["seeds"][s]["history"])]
            f.write(f" {np.mean(accs):.2f} |")
        f.write("\n")

    f.write("\n---\n\n")


def export_cross_reference(f):
    """Exp 1 vs Exp 6 cross-reference section."""
    exp1 = load("exp1_main_comparison.json")
    exp6 = load("exp6_privacy_async.json")

    f.write("## Cross-Reference: Exp 1 vs Exp 6 (Important Note)\n\n")
    f.write("### Parameter Difference\n\n")
    f.write("| | Exp 1 (Main Comparison) | Exp 6 (Privacy Sweep) |\n")
    f.write("|---|---|---|\n")
    f.write("| ρ distribution | **Heterogeneous** — pool [1.0×15%, 0.8×20%, 0.5×30%, 0.2×20%, 0.05×15%], avg≈0.51 | **Uniform** — N(ρ_target, 0.05), clipped [0.01,1] |\n")
    f.write("| Device generator | `generate_edge_devices()` (built-in ρ) | `generate_async_devices()` + ρ override |\n")
    f.write("| Methods tested | 6 methods | DASH only |\n")
    f.write("| Purpose | Compare methods under realistic mixed-privacy | Isolate ρ→accuracy relationship |\n")

    f.write("\n### Numerical Cross-Check\n\n")
    f.write("| Setting | avg ρ̃ | ρ type | DASH Final (%) | DASH Best (%) |\n")
    f.write("|---------|--------|--------|---------------|-------------|\n")

    # Exp 1
    e1_finals = [exp1[f"DASH_seed{s}"]["final_accuracy"] * 100 for s in SEEDS3]
    e1_bests = [exp1[f"DASH_seed{s}"]["best_accuracy"] * 100 for s in SEEDS3]
    f.write(f"| Exp 1 | ≈0.51 | Hetero | {np.mean(e1_finals):.2f}±{np.std(e1_finals):.2f} | "
            f"{np.mean(e1_bests):.2f}±{np.std(e1_bests):.2f} |\n")

    # Exp 6 levels
    for pk, label, rho_t in [("no_privacy", "Exp 6 None", 1.0),
                               ("mild_privacy", "Exp 6 Mild", 0.8),
                               ("moderate_privacy", "Exp 6 Mod", 0.55),
                               ("strong_privacy", "Exp 6 Strong", 0.1)]:
        finals = [exp6[pk]["seeds"][s]["final_accuracy"] * 100 for s in SEED_KEYS]
        bests = [exp6[pk]["seeds"][s]["best_accuracy"] * 100 for s in SEED_KEYS]
        rhos = [exp6[pk]["seeds"][s]["avg_rho"] for s in SEED_KEYS]
        f.write(f"| {label} | {np.mean(rhos):.2f} | Homo | "
                f"{np.mean(finals):.2f}±{np.std(finals):.2f} | "
                f"{np.mean(bests):.2f}±{np.std(bests):.2f} |\n")

    f.write("\n> **Interpretation**: Exp 1 (hetero ρ≈0.51) yields 44.47%, which is HIGHER than\n")
    f.write("> Exp 6 moderate (homo ρ≈0.55) at 42.96%. This is because heterogeneous ρ allows\n")
    f.write("> some devices with ρ≈1.0 to contribute clean logits, disproportionately improving\n")
    f.write("> distillation quality. This is NOT a contradiction — it's an additional insight.\n")
    f.write(">\n")
    f.write("> **Paper note for Sec 5.7**: \"In the main experiment (Table I), devices have\n")
    f.write("> heterogeneous privacy levels (avg ρ̃≈0.51). The privacy sweep (Fig 10) uses\n")
    f.write("> uniform ρ̃ to isolate the privacy–accuracy trade-off. The slightly higher accuracy\n")
    f.write("> under heterogeneous ρ̃ suggests that a few low-noise devices contribute\n")
    f.write("> disproportionately to distillation quality.\"\n")

    f.write("\n---\n\n")


def main():
    with open(OUT, "w") as f:
        f.write("# JPDC 2026 — Clean Raw Data for Paper Writing\n\n")
        f.write("> Auto-generated from experiment JSONs. All values: mean ± std over 3 seeds.\n")
        f.write("> Copy this file to Claude web along with JPDC_outline_v2.md and JPDC_changelog.md.\n\n")
        f.write("---\n\n")

        export_exp1(f)
        export_exp2(f)
        export_exp3(f)
        export_exp4(f)
        export_exp5(f)
        export_exp6(f)
        export_cross_reference(f)

    print(f"✅ Exported to {OUT}")
    print(f"   Size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
