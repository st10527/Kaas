#!/usr/bin/env python3
"""
Merge per-seed JSON results into combined files for plotting.
Handles: main_comparison, budget_sensitivity, device_scalability, privacy_impact.

Usage: python merge_seed_results.py [--data_dir results/edge/]
"""
import json, os, glob, argparse

def merge_main(data_dir, seeds=[42, 123, 456]):
    """Merge main_comparison_seed*.json into main_comparison.json"""
    merged = {}
    for seed in seeds:
        path = os.path.join(data_dir, f"main_comparison_seed{seed}.json")
        if not os.path.exists(path):
            print(f"  ⚠ Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        merged.update(data)  # Keys are already "Method_seedN"
        print(f"  ✓ Loaded {path} ({len(data)} entries)")

    out = os.path.join(data_dir, "main_comparison.json")
    with open(out, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"  → Saved: {out} ({len(merged)} entries)")


def merge_parameterized(data_dir, prefix, seeds=[42, 123, 456]):
    """Merge budget_sensitivity / device_scalability / privacy_impact per-seed files."""
    merged = {}
    for seed in seeds:
        path = os.path.join(data_dir, f"{prefix}_seed{seed}.json")
        if not os.path.exists(path):
            print(f"  ⚠ Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)

        for key, val in data.items():
            if key not in merged:
                merged[key] = val  # First seed: use as base
            else:
                # Average accuracies across seeds
                if 'final_accuracy' in val and 'final_accuracy' in merged[key]:
                    # Collect all seed accuracies for averaging later
                    if '_seed_accs' not in merged[key]:
                        merged[key]['_seed_accs'] = [merged[key]['final_accuracy']]
                    merged[key]['_seed_accs'].append(val['final_accuracy'])
        print(f"  ✓ Loaded {path}")

    # Compute mean/std from collected seed accuracies
    for key, val in merged.items():
        if '_seed_accs' in val:
            import numpy as np
            accs = val.pop('_seed_accs')
            val['final_accuracy'] = float(np.mean(accs))
            val['final_accuracy_std'] = float(np.std(accs))

    out = os.path.join(data_dir, f"{prefix}.json")
    with open(out, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"  → Saved: {out} ({len(merged)} entries)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='results/edge/')
    args = parser.parse_args()

    print("\n── Merging per-seed results ──\n")

    print("[main_comparison]")
    merge_main(args.data_dir)

    for prefix in ['budget_sensitivity', 'device_scalability', 'privacy_impact']:
        print(f"\n[{prefix}]")
        merge_parameterized(args.data_dir, prefix)

    print("\n✓ Merge complete.\n")


if __name__ == '__main__':
    main()
