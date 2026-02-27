#!/usr/bin/env python3
"""
KaaS-Edge Experiment Runner v2 — IEEE EDGE 2026
=================================================
Changes from v1:
  - alpha=0.3 (strong non-IID) for meaningful class coverage differentiation
  - Budget sensitivity: 3 seeds averaged
  - Cost model: log-normal b_i, results include comm_mb
  - Privacy: proper Laplace LDP with separated rho levels

Usage:
    python scripts/run_edge_experiments.py --all
    python scripts/run_edge_experiments.py --exp main --quick
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path}")


def load_data(n_public=10000, quick=False):
    from src.data.datasets import load_cifar100_safe_split
    if quick: n_public = 5000
    return load_cifar100_safe_split(root='./data', n_public=n_public, seed=42)


def partition_data(private_set, n_devices, alpha=0.3, seed=42):
    """Partition with alpha=0.3 (strong non-IID, each device ~10-15 dominant classes)."""
    from src.data.partition import DirichletPartitioner, create_client_loaders
    if hasattr(private_set, 'dataset') and hasattr(private_set.dataset, 'targets'):
        all_targets = np.array(private_set.dataset.targets)
        subset_targets = all_targets[np.array(private_set.indices)]
    else:
        subset_targets = np.array([private_set[i][1] for i in range(len(private_set))])
    partitioner = DirichletPartitioner(alpha=alpha, n_clients=n_devices, seed=seed)
    client_indices = partitioner.partition(private_set, targets=subset_targets)
    client_loaders = create_client_loaders(private_set, client_indices, batch_size=32)
    return client_loaders, client_indices


def create_model(n_classes=100):
    from src.models.utils import get_model
    return get_model('cnn', num_classes=n_classes)


def run_method(method, devices, client_loaders, public_loader, test_loader,
               n_rounds=50, method_name=""):
    print(f"\n{'='*60}")
    print(f"  Running: {method_name} ({n_rounds} rounds)")
    print(f"{'='*60}")
    history = []
    start_time = time.time()
    for t in range(n_rounds):
        result = method.run_round(t, devices, client_loaders, public_loader,
                                  test_loader=test_loader)
        history.append({
            'round': t,
            'accuracy': result.accuracy,
            'loss': result.loss,
            'participation_rate': result.participation_rate,
            'n_participants': result.n_participants,
            'energy': result.energy,
            'extra': result.extra,
        })
        if (t + 1) % 10 == 0 or t == 0:
            elapsed = time.time() - start_time
            comm = result.extra.get('comm_mb', 0)
            print(f"  Round {t+1:3d}/{n_rounds}: "
                  f"Acc={result.accuracy:.4f}  "
                  f"Part={result.participation_rate:.2f}  "
                  f"CommMB={comm:.2f}  "
                  f"[{elapsed:.1f}s]")
    final_acc = history[-1]['accuracy']
    best_acc = max(h['accuracy'] for h in history)
    total_time = time.time() - start_time
    print(f"\n  {method_name} Done: final={final_acc:.4f}, best={best_acc:.4f}, "
          f"time={total_time:.1f}s")
    return {
        'method': method_name, 'history': history,
        'final_accuracy': final_acc, 'best_accuracy': best_acc,
        'total_time': total_time,
    }


# ============================================================================
# Experiment 1: Main Comparison
# ============================================================================

def run_main_comparison(args):
    from src.methods.kaas_edge import (
        KaaSEdge, KaaSEdgeConfig,
        FullParticipationFD, RandomSelectionFD,
        FedCSFD, FedSKDFD,
        generate_edge_devices
    )
    from torch.utils.data import DataLoader

    print("\n" + "="*70)
    print("  EXPERIMENT: Main Comparison (v2 — strong non-IID, v_i subsampling)")
    print("="*70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 20
    n_public = 5000 if args.quick else 10000
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])
    budget = 50.0

    all_results = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        private_set, public_set, test_set = load_data(n_public=n_public, quick=args.quick)
        client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
        public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                   num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                 num_workers=2, pin_memory=True)
        devices = generate_edge_devices(n_devices=n_devices, seed=seed)

        config = KaaSEdgeConfig(
            budget=budget, v_max=n_public,
            local_epochs=2, distill_epochs=3, distill_lr=0.001,
            pretrain_epochs=10, n_ref_samples=n_public,
        )

        methods = {
            'KaaS-Edge': lambda: KaaSEdge(create_model(), config=config, device=args.device),
            'FedMD': lambda: FullParticipationFD(create_model(), config=config, device=args.device),
            'FedSKD': lambda: FedSKDFD(create_model(), config=config, device=args.device, select_ratio=0.5),
            'FedCS-FD': lambda: FedCSFD(create_model(), config=config, device=args.device, budget=budget),
            'RandomSelection': lambda: RandomSelectionFD(create_model(), config=config, device=args.device, select_fraction=0.5),
        }

        for name, make_method in methods.items():
            torch.manual_seed(seed); np.random.seed(seed)
            method = make_method()
            result = run_method(method, devices, client_loaders, public_loader,
                                test_loader, n_rounds=n_rounds, method_name=name)
            all_results[f"{name}_seed{seed}"] = result

    suffix = f"_seed{seeds[0]}" if len(seeds) == 1 else ""
    save_json(all_results, str(PROJECT_ROOT / "results" / "edge" / f"main_comparison{suffix}.json"))
    return all_results


# ============================================================================
# Experiment 2: Budget Sensitivity (3 seeds!)
# ============================================================================

def run_budget_sensitivity(args):
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig, generate_edge_devices
    from torch.utils.data import DataLoader

    print("\n" + "="*70)
    print("  EXPERIMENT: Budget Sensitivity (3 seeds)")
    print("="*70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 20
    budgets = [10, 20, 30, 40, 50, 60, 80] if not args.quick else [20, 40, 60]
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    results = {}

    for B in budgets:
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            private_set, public_set, test_set = load_data(quick=args.quick)
            client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)
            devices = generate_edge_devices(n_devices=n_devices, seed=seed)

            config = KaaSEdgeConfig(
                budget=B, v_max=len(public_set),
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10,
            )
            method = KaaSEdge(create_model(), config=config, device=args.device)
            result = run_method(
                method, devices, client_loaders, public_loader, test_loader,
                n_rounds=n_rounds, method_name=f"KaaS-Edge (B={B}, seed={seed})"
            )
            seed_results.append(result)

        # Store all seeds for this budget
        results[f"B={B}"] = {
            'seeds': {f"seed{s}": r for s, r in zip(seeds, seed_results)},
            'final_accuracy': float(np.mean([r['final_accuracy'] for r in seed_results])),
            'final_accuracy_std': float(np.std([r['final_accuracy'] for r in seed_results])),
            'best_accuracy': float(np.mean([r['best_accuracy'] for r in seed_results])),
            # Keep first seed's history for plotting (or average later)
            'history': seed_results[0]['history'],
        }

    suffix = f"_seed{seeds[0]}" if len(seeds) == 1 else ""
    save_json(results, str(PROJECT_ROOT / "results" / "edge" / f"budget_sensitivity{suffix}.json"))
    return results


# ============================================================================
# Experiment 3: Device Scalability (fixed per-device budget)
# ============================================================================

def run_device_scalability(args):
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig, generate_edge_devices
    from torch.utils.data import DataLoader

    print("\n" + "="*70)
    print("  EXPERIMENT: Device Scalability")
    print("="*70)

    n_rounds = 10 if args.quick else 50
    device_counts = [5, 10, 20] if args.quick else [5, 10, 20, 30, 50]
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    results = {}

    for M in device_counts:
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            n_public = 5000 if args.quick else 10000
            private_set, public_set, test_set = load_data(n_public=n_public, quick=args.quick)
            client_loaders, _ = partition_data(private_set, M, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)
            devices = generate_edge_devices(n_devices=M, seed=seed)

            # Per-device budget ~2.5: enough for meaningful v_i allocation
            # M=5→B=12.5 (select ~2-3), M=20→B=50 (select ~8-10), M=50→B=125
            budget = 2.5 * M

            config = KaaSEdgeConfig(
                budget=budget, v_max=len(public_set),
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10,
            )
            method = KaaSEdge(create_model(), config=config, device=args.device)
            result = run_method(
                method, devices, client_loaders, public_loader, test_loader,
                n_rounds=n_rounds,
                method_name=f"KaaS-Edge (M={M}, B={budget:.1f}, seed={seed})"
            )
            seed_results.append(result)

        results[f"M={M}"] = {
            'seeds': {f"seed{s}": r for s, r in zip(seeds, seed_results)},
            'final_accuracy': float(np.mean([r['final_accuracy'] for r in seed_results])),
            'final_accuracy_std': float(np.std([r['final_accuracy'] for r in seed_results])),
            'best_accuracy': float(np.mean([r['best_accuracy'] for r in seed_results])),
            'history': seed_results[0]['history'],
            'budget': 2.5 * M,
        }

    suffix = f"_seed{seeds[0]}" if len(seeds) == 1 else ""
    save_json(results, str(PROJECT_ROOT / "results" / "edge" / f"device_scalability{suffix}.json"))
    return results


# ============================================================================
# Experiment 4: Privacy Impact
# ============================================================================

def run_privacy_impact(args):
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig
    from torch.utils.data import DataLoader

    print("\n" + "="*70)
    print("  EXPERIMENT: Privacy Impact (proper Laplace LDP)")
    print("="*70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 20
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    # Privacy scenarios: control rho_i for ALL devices
    # With Laplace LDP: rho=1 → no noise, rho=0.05 → eps=0.053 → very heavy noise
    scenarios = {
        'no_privacy':     lambda n, rng: [1.0] * n,
        'mild_privacy':   lambda n, rng: [rng.uniform(0.7, 0.95) for _ in range(n)],
        'mixed_privacy':  lambda n, rng: [rng.choice([1.0, 0.8, 0.5, 0.2, 0.05],
                                          p=[0.15, 0.25, 0.30, 0.20, 0.10]) for _ in range(n)],
        'strong_privacy': lambda n, rng: [rng.uniform(0.02, 0.15) for _ in range(n)],
    }

    results = {}

    for scenario_name, rho_fn in scenarios.items():
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.RandomState(seed)
            rho_values = rho_fn(n_devices, rng)

            private_set, public_set, test_set = load_data(quick=args.quick)
            client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)

            # Generate devices with controlled rho values
            rng2 = np.random.RandomState(seed)
            log_b = rng2.normal(loc=-6.2, scale=0.8, size=n_devices)
            b_vals = np.clip(np.exp(log_b), 0.0003, 0.1)

            devices = []
            for i in range(n_devices):
                b_i = float(b_vals[i])
                theta_i = float(rng2.uniform(30, 100))
                a_i = float(rng2.uniform(0.1, 0.5))
                rho_i = float(np.clip(rho_values[i], 0.01, 1.0))
                devices.append({
                    'device_id': i, 'rho_i': rho_i, 'b_i': b_i,
                    'theta_i': theta_i, 'a_i': a_i,
                    'eta_i': float(rho_i / (b_i * theta_i)),
                })

            config = KaaSEdgeConfig(
                budget=50.0, v_max=len(public_set),
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10,
            )
            method = KaaSEdge(create_model(), config=config, device=args.device)
            result = run_method(
                method, devices, client_loaders, public_loader, test_loader,
                n_rounds=n_rounds,
                method_name=f"KaaS-Edge ({scenario_name}, seed={seed})"
            )
            result['rho_values'] = rho_values
            result['avg_rho'] = float(np.mean(rho_values))
            seed_results.append(result)

        results[scenario_name] = {
            'seeds': {f"seed{s}": r for s, r in zip(seeds, seed_results)},
            'final_accuracy': float(np.mean([r['final_accuracy'] for r in seed_results])),
            'final_accuracy_std': float(np.std([r['final_accuracy'] for r in seed_results])),
            'avg_rho': float(np.mean([r['avg_rho'] for r in seed_results])),
            'history': seed_results[0]['history'],
        }

    suffix = f"_seed{seeds[0]}" if len(seeds) == 1 else ""
    save_json(results, str(PROJECT_ROOT / "results" / "edge" / f"privacy_impact{suffix}.json"))
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KaaS-Edge Experiments v2")
    parser.add_argument('--exp', type=str, default=None,
                        choices=['main', 'budget', 'scale', 'privacy'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Run single seed (for parallel execution)')
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Single-seed mode: override the seeds list globally
    if args.seed is not None:
        args._seeds = [args.seed]
    else:
        args._seeds = None  # Use defaults

    print(f"\n{'#'*70}")
    print(f"  KaaS-Edge Experiments v2 — IEEE EDGE 2026")
    print(f"  Key changes: alpha=0.3, B=50, v_i subsampling, Laplace LDP, log-normal costs")
    print(f"  Device: {args.device}")
    print(f"  Quick: {args.quick}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    experiments = {
        'main': run_main_comparison,
        'budget': run_budget_sensitivity,
        'scale': run_device_scalability,
        'privacy': run_privacy_impact,
    }

    if args.all:
        for name, fn in experiments.items():
            try:
                fn(args)
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback; traceback.print_exc()
    elif args.exp:
        experiments[args.exp](args)
    else:
        print("Specify --exp <name> or --all. Use --quick for fast testing.")
        parser.print_help()


if __name__ == "__main__":
    main()
