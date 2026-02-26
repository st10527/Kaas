#!/usr/bin/env python3
"""
KaaS-Edge Experiment Runner for IEEE EDGE 2026
================================================

Runs all experiments for the KaaS-Edge paper:
  1. Main comparison: KaaS-Edge vs baselines (accuracy, cost, participation)
  2. Budget sensitivity: Vary B to show accuracy-cost tradeoff
  3. Device scalability: Vary M to show scheduling efficiency
  4. Privacy impact: Vary rho distribution to show degradation effects

Usage:
    # Run all experiments
    python scripts/run_edge_experiments.py --all

    # Run specific experiment
    python scripts/run_edge_experiments.py --exp main
    python scripts/run_edge_experiments.py --exp budget
    python scripts/run_edge_experiments.py --exp scale
    python scripts/run_edge_experiments.py --exp privacy

    # Quick test (fewer rounds, synthetic data)
    python scripts/run_edge_experiments.py --exp main --quick

    # Specify GPU
    python scripts/run_edge_experiments.py --all --device cuda:0
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

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch


# ============================================================================
# Utilities
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path}")


def load_data(n_public=10000, quick=False):
    """Load CIFAR-100 with safe split (private/public/test)."""
    from src.data.datasets import load_cifar100_safe_split
    
    if quick:
        n_public = 2000
    
    private_set, public_set, test_set = load_cifar100_safe_split(
        root='./data', n_public=n_public, seed=42
    )
    return private_set, public_set, test_set


def partition_data(private_set, n_devices, alpha=0.5, seed=42):
    """Partition private data among devices using Dirichlet."""
    from src.data.partition import DirichletPartitioner, create_client_loaders
    
    # Get targets from the subset
    if hasattr(private_set, 'dataset') and hasattr(private_set.dataset, 'targets'):
        all_targets = np.array(private_set.dataset.targets)
        subset_targets = all_targets[np.array(private_set.indices)]
    else:
        subset_targets = np.array([private_set[i][1] for i in range(len(private_set))])
    
    partitioner = DirichletPartitioner(
        alpha=alpha, n_clients=n_devices, seed=seed
    )
    client_indices = partitioner.partition(private_set, targets=subset_targets)
    client_loaders = create_client_loaders(private_set, client_indices, batch_size=32)
    
    return client_loaders, client_indices


def create_model(n_classes=100):
    """Create CNN model."""
    from src.models.utils import get_model
    return get_model('cnn', num_classes=n_classes)


def run_method(method, devices, client_loaders, public_loader, test_loader,
               n_rounds=50, method_name=""):
    """Run a federated method for n_rounds and return history."""
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
            print(f"  Round {t+1:3d}/{n_rounds}: "
                  f"Acc={result.accuracy:.4f}  "
                  f"Part={result.participation_rate:.2f}  "
                  f"Cost={sum(result.energy.values()):.2f}  "
                  f"[{elapsed:.1f}s]")
    
    final_acc = history[-1]['accuracy']
    best_acc = max(h['accuracy'] for h in history)
    total_time = time.time() - start_time
    
    print(f"\n  {method_name} Done: final={final_acc:.4f}, best={best_acc:.4f}, "
          f"time={total_time:.1f}s")
    
    return {
        'method': method_name,
        'history': history,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
        'total_time': total_time,
    }


# ============================================================================
# Experiment 1: Main Comparison
# ============================================================================

def run_main_comparison(args):
    """
    Main experiment: KaaS-Edge vs Full Participation vs Random Selection.
    
    Metrics: accuracy over rounds, cumulative cost, participation rate.
    Produces data for Fig. 3 (accuracy vs round) and Fig. 4 (accuracy vs cost).
    """
    from src.methods.kaas_edge import (
        KaaSEdge, KaaSEdgeConfig,
        FullParticipationFD, RandomSelectionFD,
        generate_edge_devices
    )
    from torch.utils.data import DataLoader
    
    print("\n" + "="*70)
    print("  EXPERIMENT: Main Comparison (KaaS-Edge vs Baselines)")
    print("="*70)
    
    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 20
    n_public = 2000 if args.quick else 10000
    seeds = [42] if args.quick else [42, 123, 456]
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Data
        private_set, public_set, test_set = load_data(n_public=n_public, quick=args.quick)
        client_loaders, _ = partition_data(private_set, n_devices, seed=seed)
        
        public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                   num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                 num_workers=2, pin_memory=True)
        
        # Devices
        devices = generate_edge_devices(n_devices=n_devices, seed=seed)
        
        # Shared config
        config = KaaSEdgeConfig(
            budget=8.0,
            v_max=200,
            local_epochs=2,
            distill_epochs=3,
            distill_lr=0.005,
            pretrain_epochs=5 if args.quick else 10,
            n_ref_samples=n_public,
        )
        
        methods = {
            'KaaS-Edge': lambda: KaaSEdge(
                create_model(), config=config, device=args.device
            ),
            'FullParticipation': lambda: FullParticipationFD(
                create_model(), config=config, device=args.device
            ),
            'RandomSelection': lambda: RandomSelectionFD(
                create_model(), config=config, device=args.device,
                select_fraction=0.5
            ),
        }
        
        for name, method_fn in methods.items():
            key = f"{name}_seed{seed}"
            method = method_fn()
            result = run_method(
                method, devices, client_loaders, public_loader, test_loader,
                n_rounds=n_rounds, method_name=f"{name} (seed={seed})"
            )
            all_results[key] = result
    
    save_json(all_results, str(PROJECT_ROOT / "results" / "edge" / "main_comparison.json"))
    return all_results


# ============================================================================
# Experiment 2: Budget Sensitivity
# ============================================================================

def run_budget_sensitivity(args):
    """
    Vary budget B to show accuracy-cost tradeoff.
    
    Budget values: [2, 4, 6, 8, 10, 15, 20]
    Produces data for Fig. 5 (accuracy vs budget) and Fig. 6 (participation vs budget).
    """
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig, generate_edge_devices
    from torch.utils.data import DataLoader
    
    print("\n" + "="*70)
    print("  EXPERIMENT: Budget Sensitivity")
    print("="*70)
    
    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 20
    budgets = [2, 5, 10] if args.quick else [2, 4, 6, 8, 10, 15, 20]
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    private_set, public_set, test_set = load_data(quick=args.quick)
    client_loaders, _ = partition_data(private_set, n_devices, seed=seed)
    public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)
    devices = generate_edge_devices(n_devices=n_devices, seed=seed)
    
    results = {}
    
    for B in budgets:
        config = KaaSEdgeConfig(
            budget=B,
            v_max=200,
            local_epochs=2,
            distill_epochs=3,
            distill_lr=0.005,
            pretrain_epochs=5 if args.quick else 10,
        )
        method = KaaSEdge(create_model(), config=config, device=args.device)
        result = run_method(
            method, devices, client_loaders, public_loader, test_loader,
            n_rounds=n_rounds, method_name=f"KaaS-Edge (B={B})"
        )
        results[f"B={B}"] = result
    
    save_json(results, str(PROJECT_ROOT / "results" / "edge" / "budget_sensitivity.json"))
    return results


# ============================================================================
# Experiment 3: Device Scalability
# ============================================================================

def run_device_scalability(args):
    """
    Vary number of devices M to show scheduling scalability.
    
    Device counts: [5, 10, 20, 30, 50]
    Produces data for Fig. 7 (accuracy vs M) and scheduling overhead analysis.
    """
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig, generate_edge_devices
    from torch.utils.data import DataLoader
    
    print("\n" + "="*70)
    print("  EXPERIMENT: Device Scalability")
    print("="*70)
    
    n_rounds = 10 if args.quick else 50
    device_counts = [5, 10, 20] if args.quick else [5, 10, 20, 30, 50]
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_public = 2000 if args.quick else 10000
    private_set, public_set, test_set = load_data(n_public=n_public, quick=args.quick)
    public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    results = {}
    
    for M in device_counts:
        client_loaders, _ = partition_data(private_set, M, seed=seed)
        devices = generate_edge_devices(n_devices=M, seed=seed)
        
        # Scale budget proportionally with M
        budget = 2.0 + 0.4 * M
        
        config = KaaSEdgeConfig(
            budget=budget,
            v_max=200,
            local_epochs=2,
            distill_epochs=3,
            distill_lr=0.005,
            pretrain_epochs=5 if args.quick else 10,
        )
        method = KaaSEdge(create_model(), config=config, device=args.device)
        result = run_method(
            method, devices, client_loaders, public_loader, test_loader,
            n_rounds=n_rounds, method_name=f"KaaS-Edge (M={M}, B={budget:.1f})"
        )
        results[f"M={M}"] = result
    
    save_json(results, str(PROJECT_ROOT / "results" / "edge" / "device_scalability.json"))
    return results


# ============================================================================
# Experiment 4: Privacy Impact
# ============================================================================

def run_privacy_impact(args):
    """
    Vary privacy distribution to show rho_i impact on accuracy.
    
    Scenarios: all_clear (rho=1), mixed (default), all_private (rho~0.2)
    Produces data for Fig. 8 (accuracy under different privacy regimes).
    """
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig
    from torch.utils.data import DataLoader
    
    print("\n" + "="*70)
    print("  EXPERIMENT: Privacy Impact")
    print("="*70)
    
    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 20
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    private_set, public_set, test_set = load_data(quick=args.quick)
    client_loaders, _ = partition_data(private_set, n_devices, seed=seed)
    public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    rng = np.random.RandomState(seed)
    
    # Define privacy scenarios
    scenarios = {
        'no_privacy': lambda: [1.0] * n_devices,
        'mild_privacy': lambda: [rng.uniform(0.7, 1.0) for _ in range(n_devices)],
        'mixed_privacy': lambda: [rng.choice([1.0, 0.8, 0.5, 0.2, 0.05],
                                              p=[0.15, 0.25, 0.30, 0.20, 0.10])
                                  for _ in range(n_devices)],
        'strong_privacy': lambda: [rng.uniform(0.05, 0.3) for _ in range(n_devices)],
    }
    
    results = {}
    
    for scenario_name, rho_fn in scenarios.items():
        rng = np.random.RandomState(seed)  # Reset for each scenario
        rho_values = rho_fn()
        
        # Generate devices with specific rho values
        devices = []
        for i in range(n_devices):
            b_i = rng.uniform(0.05, 0.3) * ([1.0, 2.0, 4.0][i % 3])
            devices.append({
                'device_id': i,
                'rho_i': float(rho_values[i]),
                'b_i': float(b_i),
                'theta_i': float(rng.uniform(20, 80)),
                'a_i': float(rng.uniform(0.05, 0.2)),
            })
        
        config = KaaSEdgeConfig(
            budget=8.0,
            v_max=200,
            local_epochs=2,
            distill_epochs=3,
            distill_lr=0.005,
            pretrain_epochs=5 if args.quick else 10,
        )
        method = KaaSEdge(create_model(), config=config, device=args.device)
        result = run_method(
            method, devices, client_loaders, public_loader, test_loader,
            n_rounds=n_rounds,
            method_name=f"KaaS-Edge ({scenario_name}, avg_rho={np.mean(rho_values):.2f})"
        )
        result['rho_values'] = rho_values
        result['avg_rho'] = float(np.mean(rho_values))
        results[scenario_name] = result
    
    save_json(results, str(PROJECT_ROOT / "results" / "edge" / "privacy_impact.json"))
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KaaS-Edge Experiments for IEEE EDGE 2026")
    parser.add_argument('--exp', type=str, default=None,
                        choices=['main', 'budget', 'scale', 'privacy'],
                        help='Specific experiment to run')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'#'*70}")
    print(f"  KaaS-Edge Experiments — IEEE EDGE 2026")
    print(f"  Device: {args.device}")
    print(f"  Quick mode: {args.quick}")
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
                import traceback
                traceback.print_exc()
    elif args.exp:
        experiments[args.exp](args)
    else:
        print("Specify --exp <name> or --all. Use --quick for fast testing.")
        parser.print_help()


if __name__ == "__main__":
    main()
