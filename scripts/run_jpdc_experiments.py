#!/usr/bin/env python3
"""
Async-RADS Experiment Runner — JPDC 2026
=========================================

Six experiments (see JPDC_simulation_spec_v2.md):
  1. Main Comparison: Sync vs Async  (Fig 2, 3, Table I)
  2. Straggler Severity Sweep         (Fig 4, 5)
  3. Timeout Policy Comparison         (Fig 6)
  4. Scalability                       (Fig 7, 8)
  5. Cross-Dataset: EMNIST             (Fig 9)
  6. Privacy under Async               (Fig 10)

Usage:
    python scripts/run_jpdc_experiments.py --all
    python scripts/run_jpdc_experiments.py --exp main --quick
    python scripts/run_jpdc_experiments.py --exp main --seed 42
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


# ============================================================================
# Utilities (shared with EDGE runner)
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path}")


RESULTS_DIR = PROJECT_ROOT / "results" / "jpdc"


def load_cifar_data(n_public=10000, quick=False):
    from src.data.datasets import load_cifar100_safe_split
    if quick:
        n_public = 5000
    return load_cifar100_safe_split(root='./data', n_public=n_public, seed=42)


def load_emnist_data(n_public=10000, quick=False):
    from src.data.datasets import load_emnist_safe_split
    if quick:
        n_public = 5000
    return load_emnist_safe_split(root='./data', n_public=n_public, seed=42)


def partition_data(private_set, n_devices, alpha=0.3, seed=42):
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


def generate_async_devices(n_devices, seed=42):
    """Generate devices with both EDGE cost model AND async rate model."""
    from src.methods.kaas_edge import generate_edge_devices
    from src.async_module.straggler_model import StragglerModel

    devices = generate_edge_devices(n_devices=n_devices, seed=seed)
    rates = StragglerModel.generate_device_rates(n_devices, seed=seed)

    for dev, rate in zip(devices, rates):
        dev['comp_rate'] = rate['comp_rate']
        dev['comm_rate'] = rate['comm_rate']
    return devices


# ============================================================================
# Run helper  (adds wall-clock tracking to output)
# ============================================================================

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
        entry = {
            'round': t,
            'accuracy': result.accuracy,
            'loss': result.loss,
            'participation_rate': result.participation_rate,
            'n_participants': result.n_participants,
            'energy': result.energy,
            'extra': result.extra,
        }
        # Capture wall-clock from async methods
        wc = result.extra.get('wall_clock_time', None)
        if wc is not None:
            entry['wall_clock_time'] = wc
        history.append(entry)

        if (t + 1) % 10 == 0 or t == 0:
            elapsed = time.time() - start_time
            comm = result.extra.get('comm_mb', 0)
            wc_str = f"  WC={wc:.1f}" if wc else ""
            nc = result.extra.get('n_complete', '?')
            nt = result.extra.get('n_timeout', '?')
            print(f"  Round {t+1:3d}/{n_rounds}: "
                  f"Acc={result.accuracy:.4f}  "
                  f"Part={result.participation_rate:.2f}  "
                  f"Complete={nc}  Timeout={nt}  "
                  f"CommMB={comm:.2f}{wc_str}  "
                  f"[{elapsed:.1f}s]")

    final_acc = history[-1]['accuracy']
    best_acc = max(h['accuracy'] for h in history)
    total_time = time.time() - start_time
    wall_clock = history[-1].get('wall_clock_time', total_time)

    print(f"\n  {method_name} Done: final={final_acc:.4f}, best={best_acc:.4f}, "
          f"wall_clock={wall_clock:.1f}, real_time={total_time:.1f}s")

    return {
        'method': method_name,
        'history': history,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
        'total_time': total_time,
        'wall_clock_time': wall_clock,
    }


# ============================================================================
# Experiment 1: Main Comparison — Sync vs Async
# ============================================================================

def run_main_comparison(args):
    from src.methods.kaas_edge import (
        KaaSEdge, KaaSEdgeConfig,
        FullParticipationFD, generate_edge_devices,
    )
    from src.methods.async_kaas_edge import (
        AsyncKaaSEdge, AsyncKaaSEdgeConfig,
        FullAsyncFD, RandomAsyncFD,
    )
    from src.methods.fedbuff_fd import FedBuffFD, FedBuffFDConfig
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("  JPDC EXP 1: Main Comparison — Sync vs Async")
    print("=" * 70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 50
    n_public = 5000 if args.quick else 10000
    budget = 50.0
    sigma = 0.5
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    all_results = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        private_set, public_set, test_set = load_cifar_data(n_public, args.quick)
        client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
        public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                   num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                 num_workers=2, pin_memory=True)
        devices = generate_async_devices(n_devices, seed=seed)

        sync_cfg = KaaSEdgeConfig(
            budget=budget, v_max=n_public,
            local_epochs=2, distill_epochs=3, distill_lr=0.001,
            pretrain_epochs=10, n_ref_samples=n_public,
        )
        async_cfg = AsyncKaaSEdgeConfig(
            budget=budget, v_max=n_public,
            local_epochs=2, distill_epochs=3, distill_lr=0.001,
            pretrain_epochs=10, n_ref_samples=n_public,
            sigma_noise=sigma, timeout_policy='adaptive',
            adaptive_percentile=0.7, straggler_aware=True,
        )
        fedbuff_cfg = FedBuffFDConfig(
            local_epochs=2, distill_epochs=3, distill_lr=0.001,
            pretrain_epochs=10, n_ref_samples=n_public,
            buffer_size=10, v_fixed=100, sigma_noise=sigma,
        )

        methods = {
            'Async-RADS': lambda: AsyncKaaSEdge(
                create_model(), config=copy.deepcopy(async_cfg), device=args.device),
            'Sync-RADS': lambda: KaaSEdge(
                create_model(), config=sync_cfg, device=args.device),
            'FedBuff-FD': lambda: FedBuffFD(
                create_model(), config=copy.deepcopy(fedbuff_cfg), device=args.device),
            'Random-Async': lambda: RandomAsyncFD(
                create_model(), config=copy.deepcopy(async_cfg), device=args.device),
            'Full-Async': lambda: FullAsyncFD(
                create_model(), config=copy.deepcopy(async_cfg), device=args.device),
            'Sync-Full': lambda: FullParticipationFD(
                create_model(), config=sync_cfg, device=args.device),
        }

        for name, make_method in methods.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            method = make_method()
            result = run_method(method, devices, client_loaders,
                                public_loader, test_loader,
                                n_rounds=n_rounds, method_name=name)
            all_results[f"{name}_seed{seed}"] = result

    save_json(all_results, str(RESULTS_DIR / "exp1_main_comparison.json"))
    return all_results


# ============================================================================
# Experiment 2: Straggler Severity Sweep
# ============================================================================

def run_straggler_sweep(args):
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig
    from src.methods.async_kaas_edge import AsyncKaaSEdge, AsyncKaaSEdgeConfig
    from src.methods.fedbuff_fd import FedBuffFD, FedBuffFDConfig
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("  JPDC EXP 2: Straggler Severity Sweep")
    print("=" * 70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 50
    budget = 50.0
    sigmas = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.3, 0.5, 1.0, 1.5]
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    results = {}

    for sigma in sigmas:
        sigma_results = {}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            private_set, public_set, test_set = load_cifar_data(quick=args.quick)
            client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)
            devices = generate_async_devices(n_devices, seed=seed)

            n_public = len(public_set)

            # Async-RADS
            async_cfg = AsyncKaaSEdgeConfig(
                budget=budget, v_max=n_public,
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, sigma_noise=sigma,
                timeout_policy='adaptive', straggler_aware=True,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_async = AsyncKaaSEdge(create_model(), config=copy.deepcopy(async_cfg),
                                    device=args.device)
            r_async = run_method(m_async, devices, client_loaders,
                                 public_loader, test_loader, n_rounds,
                                 f"Async-RADS(σ={sigma}, seed={seed})")

            # Sync-RADS (σ doesn't affect sync, but we run it at σ=0 only to save time)
            if sigma == 0.0 or not args.quick:
                sync_cfg = KaaSEdgeConfig(
                    budget=budget, v_max=n_public,
                    local_epochs=2, distill_epochs=3, distill_lr=0.001,
                    pretrain_epochs=10,
                )
                torch.manual_seed(seed); np.random.seed(seed)
                m_sync = KaaSEdge(create_model(), config=sync_cfg, device=args.device)
                r_sync = run_method(m_sync, devices, client_loaders,
                                    public_loader, test_loader, n_rounds,
                                    f"Sync-RADS(σ={sigma}, seed={seed})")
            else:
                r_sync = None

            # FedBuff-FD
            fb_cfg = FedBuffFDConfig(
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, buffer_size=10, v_fixed=100,
                sigma_noise=sigma,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_fb = FedBuffFD(create_model(), config=copy.deepcopy(fb_cfg),
                             device=args.device)
            r_fb = run_method(m_fb, devices, client_loaders,
                              public_loader, test_loader, n_rounds,
                              f"FedBuff-FD(σ={sigma}, seed={seed})")

            sigma_results[f"seed{seed}"] = {
                'Async-RADS': r_async,
                'Sync-RADS': r_sync,
                'FedBuff-FD': r_fb,
            }

        results[f"sigma={sigma}"] = sigma_results

    save_json(results, str(RESULTS_DIR / "exp2_straggler_sweep.json"))
    return results


# ============================================================================
# Experiment 3: Timeout Policy Comparison
# ============================================================================

def run_policy_comparison(args):
    from src.methods.async_kaas_edge import AsyncKaaSEdge, AsyncKaaSEdgeConfig
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("  JPDC EXP 3: Timeout Policy Comparison")
    print("=" * 70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 50
    budget = 50.0
    sigma = 1.0  # harsh conditions
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    policies = [
        ('fixed', {'D_0': 5.0}),
        ('fixed', {'D_0': 10.0}),
        ('fixed', {'D_0': 20.0}),
        ('adaptive', {'percentile': 0.5}),
        ('adaptive', {'percentile': 0.7}),
        ('adaptive', {'percentile': 0.9}),
        ('partial', {'percentile': 0.7}),
    ]
    if args.quick:
        policies = [
            ('fixed', {'D_0': 10.0}),
            ('adaptive', {'percentile': 0.7}),
            ('partial', {'percentile': 0.7}),
        ]

    results = {}

    for policy_name, policy_kwargs in policies:
        label = f"{policy_name}({list(policy_kwargs.values())[0]})"
        seed_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            private_set, public_set, test_set = load_cifar_data(quick=args.quick)
            client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)
            devices = generate_async_devices(n_devices, seed=seed)

            cfg = AsyncKaaSEdgeConfig(
                budget=budget, v_max=len(public_set),
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, sigma_noise=sigma,
                timeout_policy=policy_name,
                straggler_aware=True,
            )
            # Apply policy-specific kwargs
            if 'D_0' in policy_kwargs:
                cfg.fixed_deadline = policy_kwargs['D_0']
            if 'percentile' in policy_kwargs:
                cfg.adaptive_percentile = policy_kwargs['percentile']

            method = AsyncKaaSEdge(create_model(), config=cfg, device=args.device)
            result = run_method(method, devices, client_loaders,
                                public_loader, test_loader, n_rounds,
                                f"{label} (seed={seed})")
            seed_results.append(result)

        results[label] = {
            'seeds': {f"seed{s}": r for s, r in zip(seeds, seed_results)},
            'final_accuracy': float(np.mean([r['final_accuracy'] for r in seed_results])),
            'final_accuracy_std': float(np.std([r['final_accuracy'] for r in seed_results])),
            'wall_clock_time': float(np.mean([r['wall_clock_time'] for r in seed_results])),
        }

    save_json(results, str(RESULTS_DIR / "exp3_policy_comparison.json"))
    return results


# ============================================================================
# Experiment 4: Scalability
# ============================================================================

def run_scalability(args):
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig
    from src.methods.async_kaas_edge import AsyncKaaSEdge, AsyncKaaSEdgeConfig
    from src.methods.fedbuff_fd import FedBuffFD, FedBuffFDConfig
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("  JPDC EXP 4: Scalability")
    print("=" * 70)

    n_rounds = 10 if args.quick else 50
    sigma = 0.5
    device_counts = [20, 50] if args.quick else [20, 50, 100, 200]
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    results = {}

    for M in device_counts:
        seed_results = []
        budget = 2.5 * M  # per-device budget ~2.5

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            private_set, public_set, test_set = load_cifar_data(quick=args.quick)
            client_loaders, _ = partition_data(private_set, M, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)
            devices = generate_async_devices(M, seed=seed)
            n_public = len(public_set)

            # Async-RADS
            async_cfg = AsyncKaaSEdgeConfig(
                budget=budget, v_max=n_public,
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, sigma_noise=sigma,
                timeout_policy='adaptive', straggler_aware=True,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_async = AsyncKaaSEdge(create_model(), config=copy.deepcopy(async_cfg),
                                    device=args.device)
            r_async = run_method(m_async, devices, client_loaders,
                                 public_loader, test_loader, n_rounds,
                                 f"Async-RADS(M={M}, seed={seed})")

            # Sync-RADS
            sync_cfg = KaaSEdgeConfig(
                budget=budget, v_max=n_public,
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_sync = KaaSEdge(create_model(), config=sync_cfg, device=args.device)
            r_sync = run_method(m_sync, devices, client_loaders,
                                public_loader, test_loader, n_rounds,
                                f"Sync-RADS(M={M}, seed={seed})")

            # FedBuff-FD
            fb_cfg = FedBuffFDConfig(
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, buffer_size=min(10, M // 2),
                v_fixed=100, sigma_noise=sigma,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_fb = FedBuffFD(create_model(), config=copy.deepcopy(fb_cfg),
                             device=args.device)
            r_fb = run_method(m_fb, devices, client_loaders,
                              public_loader, test_loader, n_rounds,
                              f"FedBuff-FD(M={M}, seed={seed})")

            seed_results.append({
                'Async-RADS': r_async,
                'Sync-RADS': r_sync,
                'FedBuff-FD': r_fb,
            })

        results[f"M={M}"] = {
            'seeds': {f"seed{s}": sr for s, sr in zip(seeds, seed_results)},
            'budget': budget,
        }

    save_json(results, str(RESULTS_DIR / "exp4_scalability.json"))
    return results


# ============================================================================
# Experiment 5: Cross-Dataset — EMNIST
# ============================================================================

def run_emnist_validation(args):
    from src.methods.kaas_edge import KaaSEdge, KaaSEdgeConfig
    from src.methods.async_kaas_edge import AsyncKaaSEdge, AsyncKaaSEdgeConfig
    from src.methods.fedbuff_fd import FedBuffFD, FedBuffFDConfig
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("  JPDC EXP 5: Cross-Dataset Validation (EMNIST-ByClass)")
    print("=" * 70)

    n_rounds = 10 if args.quick else 50
    n_classes = 62
    sigma = 0.5
    device_counts = [50] if args.quick else [50, 200]
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    results = {}

    for M in device_counts:
        seed_results = []
        budget = 2.5 * M

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            private_set, public_set, test_set = load_emnist_data(quick=args.quick)
            client_loaders, _ = partition_data(
                private_set, M, alpha=0.1, seed=seed,  # extreme non-IID for EMNIST
            )
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)
            devices = generate_async_devices(M, seed=seed)
            n_public = len(public_set)

            # Async-RADS
            async_cfg = AsyncKaaSEdgeConfig(
                budget=budget, v_max=n_public,
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, sigma_noise=sigma,
                timeout_policy='adaptive', straggler_aware=True,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_async = AsyncKaaSEdge(
                create_model(n_classes), config=copy.deepcopy(async_cfg),
                n_classes=n_classes, device=args.device,
            )
            r_async = run_method(m_async, devices, client_loaders,
                                 public_loader, test_loader, n_rounds,
                                 f"Async-RADS(EMNIST, M={M}, seed={seed})")

            # Sync-RADS
            sync_cfg = KaaSEdgeConfig(
                budget=budget, v_max=n_public,
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_sync = KaaSEdge(
                create_model(n_classes), config=sync_cfg,
                n_classes=n_classes, device=args.device,
            )
            r_sync = run_method(m_sync, devices, client_loaders,
                                public_loader, test_loader, n_rounds,
                                f"Sync-RADS(EMNIST, M={M}, seed={seed})")

            # FedBuff-FD
            fb_cfg = FedBuffFDConfig(
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, buffer_size=min(10, M // 2),
                v_fixed=100, sigma_noise=sigma,
            )
            torch.manual_seed(seed); np.random.seed(seed)
            m_fb = FedBuffFD(
                create_model(n_classes), config=copy.deepcopy(fb_cfg),
                n_classes=n_classes, device=args.device,
            )
            r_fb = run_method(m_fb, devices, client_loaders,
                              public_loader, test_loader, n_rounds,
                              f"FedBuff-FD(EMNIST, M={M}, seed={seed})")

            seed_results.append({
                'Async-RADS': r_async,
                'Sync-RADS': r_sync,
                'FedBuff-FD': r_fb,
            })

        results[f"M={M}"] = {
            'seeds': {f"seed{s}": sr for s, sr in zip(seeds, seed_results)},
            'budget': budget,
        }

    save_json(results, str(RESULTS_DIR / "exp5_emnist.json"))
    return results


# ============================================================================
# Experiment 6: Privacy under Async
# ============================================================================

def run_privacy_async(args):
    from src.methods.async_kaas_edge import AsyncKaaSEdge, AsyncKaaSEdgeConfig
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("  JPDC EXP 6: Privacy under Async Conditions")
    print("=" * 70)

    n_rounds = 10 if args.quick else 50
    n_devices = 10 if args.quick else 50
    budget = 50.0
    sigma = 0.5
    seeds = args._seeds or ([42] if args.quick else [42, 123, 456])

    rho_scenarios = {
        'no_privacy':       1.0,
        'mild_privacy':     0.8,
        'moderate_privacy': 0.55,
        'strong_privacy':   0.1,
    }
    if args.quick:
        rho_scenarios = {'no_privacy': 1.0, 'strong_privacy': 0.1}

    results = {}

    for scenario, rho_mean in rho_scenarios.items():
        seed_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            private_set, public_set, test_set = load_cifar_data(quick=args.quick)
            client_loaders, _ = partition_data(private_set, n_devices, alpha=0.3, seed=seed)
            public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                                       num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                                     num_workers=2, pin_memory=True)

            # Generate devices with controlled rho
            devices = generate_async_devices(n_devices, seed=seed)
            rng_priv = np.random.RandomState(seed)
            for dev in devices:
                dev['rho_i'] = float(np.clip(
                    rng_priv.normal(rho_mean, 0.05), 0.01, 1.0
                ))
                dev['eta_i'] = dev['rho_i'] / (dev['b_i'] * dev['theta_i'])

            cfg = AsyncKaaSEdgeConfig(
                budget=budget, v_max=len(public_set),
                local_epochs=2, distill_epochs=3, distill_lr=0.001,
                pretrain_epochs=10, sigma_noise=sigma,
                timeout_policy='adaptive', straggler_aware=True,
            )
            method = AsyncKaaSEdge(create_model(), config=copy.deepcopy(cfg),
                                   device=args.device)
            result = run_method(method, devices, client_loaders,
                                public_loader, test_loader, n_rounds,
                                f"Async-RADS({scenario}, seed={seed})")
            result['avg_rho'] = float(np.mean([d['rho_i'] for d in devices]))
            seed_results.append(result)

        results[scenario] = {
            'seeds': {f"seed{s}": r for s, r in zip(seeds, seed_results)},
            'final_accuracy': float(np.mean([r['final_accuracy'] for r in seed_results])),
            'final_accuracy_std': float(np.std([r['final_accuracy'] for r in seed_results])),
            'avg_rho': rho_mean,
        }

    save_json(results, str(RESULTS_DIR / "exp6_privacy_async.json"))
    return results


# ============================================================================
# Main
# ============================================================================

EXPERIMENTS = {
    'main':     run_main_comparison,
    'straggler': run_straggler_sweep,
    'policy':   run_policy_comparison,
    'scale':    run_scalability,
    'emnist':   run_emnist_validation,
    'privacy':  run_privacy_async,
}


def main():
    parser = argparse.ArgumentParser(description="Async-RADS Experiments — JPDC 2026")
    parser.add_argument('--exp', type=str, default=None,
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced scale for fast validation')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Run single seed (for parallel execution)')
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args._seeds = [args.seed] if args.seed is not None else None

    print(f"\n{'#' * 70}")
    print(f"  Async-RADS Experiments — JPDC 2026")
    print(f"  Device: {args.device}")
    print(f"  Quick:  {args.quick}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    if args.all:
        for name, fn in EXPERIMENTS.items():
            try:
                fn(args)
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
    elif args.exp:
        EXPERIMENTS[args.exp](args)
    else:
        print("Specify --exp <name> or --all.  Use --quick for fast testing.")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        parser.print_help()


if __name__ == "__main__":
    main()
