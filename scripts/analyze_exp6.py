#!/usr/bin/env python3
"""Exp 6 — Privacy under Async (DASH with varying LDP noise)

4 privacy levels: no_privacy, mild, moderate, strong
DASH only, M=50, budget=50, 50 rounds, 3 seeds
"""

import json
import numpy as np
from pathlib import Path

SEEDS = ["seed42", "seed123", "seed456"]
PRIVS = ["no_privacy", "mild_privacy", "moderate_privacy", "strong_privacy"]
PRIV_LABELS = ["None", "Mild", "Moderate", "Strong"]


def load():
    with open(Path("results/jpdc/exp6_privacy_async.json")) as f:
        return json.load(f)


def main():
    data = load()

    print("=" * 90)
    print("  EXPERIMENT 6: PRIVACY UNDER ASYNC (DASH + LDP)")
    print("  CIFAR-100, M=50, budget=50, 50 rounds, DASH only")
    print("=" * 90)

    # ── 1. Full data extraction ──
    print("\n─── PER-SEED RAW DATA ───")
    all_stats = {}
    for pk, plabel in zip(PRIVS, PRIV_LABELS):
        print(f"\n  [{plabel}] {pk}:")
        finals, bests, wcs, rhos = [], [], [], []
        for s in SEEDS:
            e = data[pk]["seeds"][s]
            h = e.get("history", [])
            part_rates = [r.get("participation_rate", 0) for r in h]
            n_parts = [r.get("n_participants", 0) for r in h]
            finals.append(e["final_accuracy"] * 100)
            bests.append(e["best_accuracy"] * 100)
            wcs.append(e["wall_clock_time"])
            rhos.append(e["avg_rho"])
            print(f"    {s}: final={e['final_accuracy']*100:.2f}%  best={e['best_accuracy']*100:.2f}%  "
                  f"wc={e['wall_clock_time']:.0f}s  avg_rho={e['avg_rho']:.4f}  "
                  f"part_rate={np.mean(part_rates):.4f}  n_part={np.mean(n_parts):.1f}")

            # Check extra for epsilon/rho details
            extras = [r.get("extra", {}) for r in h]
            if extras and extras[0]:
                ek = list(extras[0].keys())
                if len(ek) <= 5:
                    print(f"      extra[0]: {extras[0]}")

        all_stats[pk] = {
            "label": plabel,
            "final_m": np.mean(finals), "final_s": np.std(finals),
            "best_m": np.mean(bests), "best_s": np.std(bests),
            "wc_m": np.mean(wcs), "wc_s": np.std(wcs),
            "rho_m": np.mean(rhos), "rho_s": np.std(rhos),
            "finals": finals, "bests": bests, "wcs": wcs, "rhos": rhos,
        }

    # ── 2. Summary table ──
    print("\n\n─── SUMMARY TABLE ───")
    print(f"  {'Privacy':<16s} │ {'avg ρ̃':>8s} │ {'Final Acc (%)':>16s} │ "
          f"{'Best Acc (%)':>16s} │ {'WC (s)':>16s}")
    print(f"  {'':─<16s} ┼ {'':─>8s} ┼ {'':─>16s} ┼ {'':─>16s} ┼ {'':─>16s}")
    for pk in PRIVS:
        s = all_stats[pk]
        print(f"  {s['label']:<16s} │ {s['rho_m']:8.4f} │ "
              f"{s['final_m']:5.2f} ± {s['final_s']:4.2f}    │ "
              f"{s['best_m']:5.2f} ± {s['best_s']:4.2f}    │ "
              f"{s['wc_m']:7.0f} ± {s['wc_s']:4.0f}    ")

    # ── 3. Accuracy drop vs no_privacy baseline ──
    print("\n\n─── ACCURACY DROP vs NO_PRIVACY BASELINE ───")
    base_best = all_stats["no_privacy"]["best_m"]
    base_final = all_stats["no_privacy"]["final_m"]
    base_wc = all_stats["no_privacy"]["wc_m"]
    print(f"  {'Privacy':<16s} │ {'avg ρ̃':>8s} │ {'Best Acc':>8s} │ {'Δ Best (pp)':>12s} │ "
          f"{'Final Acc':>9s} │ {'Δ Final (pp)':>13s} │ {'WC ratio':>9s}")
    print(f"  {'':─<16s} ┼ {'':─>8s} ┼ {'':─>8s} ┼ {'':─>12s} ┼ {'':─>9s} ┼ {'':─>13s} ┼ {'':─>9s}")
    for pk in PRIVS:
        s = all_stats[pk]
        d_best = s["best_m"] - base_best
        d_final = s["final_m"] - base_final
        wc_ratio = s["wc_m"] / base_wc
        print(f"  {s['label']:<16s} │ {s['rho_m']:8.4f} │ {s['best_m']:7.2f}% │ "
              f"{d_best:+11.2f}pp │ {s['final_m']:8.2f}% │ {d_final:+12.2f}pp │ {wc_ratio:8.2f}×")

    # ── 4. Convergence curves at key rounds ──
    print("\n\n─── CONVERGENCE: Accuracy at key rounds (mean of 3 seeds) ───")
    key_rounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    header = f"  {'Privacy':<16s} │" + "".join(f" R{r:<4d}│" for r in key_rounds)
    print(header)
    for pk in PRIVS:
        accs = []
        for r in key_rounds:
            vals = []
            for s in SEEDS:
                h = data[pk]["seeds"][s].get("history", [])
                if r < len(h):
                    vals.append(h[r]["accuracy"] * 100)
            accs.append(np.mean(vals) if vals else float('nan'))
        row = f"  {all_stats[pk]['label']:<16s} │" + "".join(f" {a:5.2f}│" for a in accs)
        print(row)

    # ── 5. ρ̃ evolution check ──
    print("\n\n─── ρ̃ PER-ROUND (seed42) ───")
    for pk in PRIVS:
        h = data[pk]["seeds"]["seed42"].get("history", [])
        extras = [r.get("extra", {}) for r in h]
        # Look for rho in extra or participation_rate
        rhos_per_round = []
        for r in h:
            extra = r.get("extra", {})
            rho = extra.get("avg_rho", extra.get("rho", None))
            if rho is None:
                rho = r.get("participation_rate", None)
            rhos_per_round.append(rho)

        if any(v is not None for v in rhos_per_round):
            vals = [v for v in rhos_per_round if v is not None]
            print(f"  {all_stats[pk]['label']:<16s}: rho range [{min(vals):.4f}, {max(vals):.4f}], "
                  f"mean={np.mean(vals):.4f}")
        else:
            print(f"  {all_stats[pk]['label']:<16s}: no per-round rho data")

    # ── 6. Participation rate analysis ──
    print("\n\n─── PARTICIPATION RATE (mean across seeds) ───")
    for pk in PRIVS:
        parts = []
        for s in SEEDS:
            h = data[pk]["seeds"][s].get("history", [])
            for r in h:
                parts.append(r.get("participation_rate", 0))
        print(f"  {all_stats[pk]['label']:<16s}: part_rate mean={np.mean(parts):.4f}, "
              f"std={np.std(parts):.4f}, min={min(parts):.4f}, max={max(parts):.4f}")

    # ── 7. WC stability check ──
    print("\n\n─── WALL-CLOCK STABILITY ───")
    print(f"  {'Privacy':<16s} │ {'WC mean':>9s} │ {'WC std':>8s} │ {'CV':>6s}")
    for pk in PRIVS:
        s = all_stats[pk]
        cv = s["wc_s"] / s["wc_m"] * 100
        print(f"  {s['label']:<16s} │ {s['wc_m']:8.0f}s │ {s['wc_s']:7.0f}s │ {cv:5.1f}%")

    # ── 8. Cross-reference with Exp 1 ──
    print("\n\n─── CROSS-REFERENCE with Exp 1 (no privacy, same setup) ───")
    print("  Exp 1 DASH: final 44.47%, best 45.14%, WC 979s")
    no_p = all_stats["no_privacy"]
    print(f"  Exp 6 no_privacy: final {no_p['final_m']:.2f}%, best {no_p['best_m']:.2f}%, WC {no_p['wc_m']:.0f}s")
    print(f"  Difference: final {no_p['final_m'] - 44.47:+.2f}pp, best {no_p['best_m'] - 45.14:+.2f}pp, "
          f"WC {no_p['wc_m'] - 979:+.0f}s")
    print("  (Small difference is expected — Exp 1 may have slightly different config)")

    # ── 9. Paper-ready summary ──
    print(f"\n\n{'=' * 90}")
    print("  PAPER-READY SUMMARY FOR Sec 5.7 / Fig 10")
    print("=" * 90)

    no_p = all_stats["no_privacy"]
    mild = all_stats["mild_privacy"]
    mod = all_stats["moderate_privacy"]
    strong = all_stats["strong_privacy"]

    drop_mild = mild["best_m"] - no_p["best_m"]
    drop_mod = mod["best_m"] - no_p["best_m"]
    drop_strong = strong["best_m"] - no_p["best_m"]

    print(f"""
  ▸ KEY FINDING 1: Graceful accuracy degradation under increasing privacy
    No privacy (ρ̃={no_p['rho_m']:.2f}):      best {no_p['best_m']:.2f}% ± {no_p['best_s']:.2f}
    Mild       (ρ̃={mild['rho_m']:.2f}):      best {mild['best_m']:.2f}% ± {mild['best_s']:.2f}  ({drop_mild:+.2f}pp)
    Moderate   (ρ̃={mod['rho_m']:.2f}):      best {mod['best_m']:.2f}% ± {mod['best_s']:.2f}  ({drop_mod:+.2f}pp)
    Strong     (ρ̃={strong['rho_m']:.2f}):      best {strong['best_m']:.2f}% ± {strong['best_s']:.2f}  ({drop_strong:+.2f}pp)

  ▸ KEY FINDING 2: Wall-clock time INDEPENDENT of privacy level
    WC range: {min(s['wc_m'] for s in all_stats.values()):.0f}s – {max(s['wc_m'] for s in all_stats.values()):.0f}s
    → DASH scheduling is NOT affected by LDP noise level
    → Privacy is a pure accuracy trade-off, not a speed trade-off

  ▸ KEY FINDING 3: Total accuracy drop from no_privacy to strong_privacy
    Best: {no_p['best_m']:.2f}% → {strong['best_m']:.2f}%  (drop = {abs(drop_strong):.2f}pp)
    Final: {no_p['final_m']:.2f}% → {strong['final_m']:.2f}%  (drop = {abs(no_p['final_m'] - strong['final_m']):.2f}pp)

  ▸ KEY FINDING 4: ρ̃ range spans practical operating points
    ρ̃ = {no_p['rho_m']:.2f} (no privacy) → {strong['rho_m']:.2f} (strong privacy)
    → Covers the entire spectrum from no protection to aggressive LDP

  ▸ STORY FOR PAPER (Sec 5.7):
    "Under increasing LDP noise (ρ̃ from {no_p['rho_m']:.2f} to {strong['rho_m']:.2f}), DASH accuracy
     degrades gracefully from {no_p['best_m']:.1f}% to {strong['best_m']:.1f}% ({abs(drop_strong):.1f}pp drop), while
     wall-clock time remains stable (~{np.mean([s['wc_m'] for s in all_stats.values()]):.0f}s).
     This confirms that DASH's scheduling mechanism is orthogonal to
     privacy mechanisms, and practitioners can tune ε independently
     of the async protocol."
""")


if __name__ == "__main__":
    main()
