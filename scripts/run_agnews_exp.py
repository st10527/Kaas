#!/usr/bin/env python3
"""
AG News experiment for DASH (R2 Q1 -- FGCS major revision).

Demonstrates DASH on text modality (AG News, 4-class news classification).
Design mirrors run_ablation_exp.py / run_edge_experiments.py exactly:
  - Same split structure  (public / private / test)
  - Same Dirichlet non-IID partition (alpha=0.3)
  - Same DASH config (budget, water-filling, straggler-aware, adaptive timeout)
  - Only differences: dataset=AG News, model=TextCNN, num_classes=4

Usage:
    python scripts/run_agnews_exp.py --quick          # smoke-test
    python scripts/run_agnews_exp.py --rounds 100 --seeds 3
"""

import argparse, json, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_data(n_public=5000, quick=False):
    from src.data.agnews import load_agnews_safe_split
    if quick:
        n_public = 2000
    return load_agnews_safe_split(root="./data", n_public=n_public, seed=42)


def partition_data(private_set, n_clients, alpha=0.3, seed=42):
    from src.data.partition import DirichletPartitioner, create_client_loaders
    # AG News labels are plain ints stored in dataset.labels
    if hasattr(private_set, "labels"):
        subset_targets = np.array(private_set.labels)
    elif hasattr(private_set, "dataset") and hasattr(private_set.dataset, "labels"):
        all_labels = np.array(private_set.dataset.labels)
        subset_targets = all_labels[np.array(private_set.indices)]
    else:
        subset_targets = np.array([private_set[i][1] for i in range(len(private_set))])
    partitioner = DirichletPartitioner(alpha=alpha, n_clients=n_clients, seed=seed)
    client_indices = partitioner.partition(private_set, targets=subset_targets)
    return create_client_loaders(private_set, client_indices, batch_size=64)


def create_model(num_classes=4):
    from src.models.utils import get_model
    return get_model("textcnn", num_classes=num_classes)


# ─────────────────────────────────────────────────────────────────
# Methods to compare
# ─────────────────────────────────────────────────────────────────

METHODS = ["DASH", "FedAvg", "RandomSelection"]


def run_one(method_name, args, seed):
    import torch.nn as nn
    from src.methods.dash import DASH, DASHConfig
    from src.methods.kaas_edge import generate_edge_devices
    from src.methods.fedavg import FedAvg, FedAvgConfig

    torch.manual_seed(seed)
    np.random.seed(seed)

    private_set, public_set, test_set = load_data(quick=args.quick)
    client_loaders = partition_data(private_set, args.n_clients, alpha=0.3, seed=seed)
    public_loader = DataLoader(public_set, batch_size=128, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    devices = generate_edge_devices(n_devices=args.n_clients, seed=seed)

    # ── Build method ──────────────────────────────────────────
    num_classes = 4  # AG News
    if method_name == "DASH":
        config = DASHConfig(
            budget=50.0,
            v_max=len(public_set),
            local_epochs=2,
            local_lr=0.001,        # TextCNN converges fast; 0.01 causes too much client drift
            distill_epochs=3,
            distill_lr=0.001,
            distill_alpha=0.2,     # 20% KL + 80% CE; 4-class soft-labels carry less info
            temperature=2.0,       # slightly sharper than 3.0 default for 4-class
            pretrain_epochs=5 if args.quick else 10,
            n_ref_samples=len(public_set),
            straggler_aware=True,
            timeout_policy="adaptive",
            fixed_deadline=5.0,
            sigma_noise=0.3,
        )
        method = DASH(create_model(num_classes), config=config, device=args.device,
                     n_classes=num_classes)

    elif method_name == "FedAvg":
        config = FedAvgConfig(
            local_epochs=2,
            local_lr=0.01,
            participation_rate=0.1,
        )
        method = FedAvg(create_model(num_classes), config=config, device=args.device)

    elif method_name == "RandomSelection":
        from src.methods.kaas_edge import RandomSelectionFD, KaaSEdgeConfig
        config = KaaSEdgeConfig(
            budget=50.0,
            v_max=len(public_set),
            local_epochs=2,
            local_lr=0.001,        # reduce client drift for TextCNN
            distill_epochs=3,
            distill_lr=0.001,
            distill_alpha=0.2,     # 20% KL + 80% CE
            temperature=2.0,
            pretrain_epochs=5 if args.quick else 10,
            n_ref_samples=len(public_set),
        )
        method = RandomSelectionFD(create_model(num_classes), config=config,
                                   device=args.device, select_fraction=0.1,
                                   n_classes=num_classes)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # ── Training loop ─────────────────────────────────────────
    history, t0 = [], time.time()
    for t in range(args.rounds):
        result = method.run_round(t, devices, client_loaders, public_loader,
                                  test_loader=test_loader)
        history.append({
            "round":              t,
            "accuracy":           result.accuracy,
            "loss":               result.loss,
            "participation_rate": result.participation_rate,
            "n_participants":     result.n_participants,
            "energy":             result.energy,
            "real_time_s":        result.extra.get("real_time_s", 0.0),
            "wall_clock":         result.extra.get("wall_clock_time", 0.0),
        })
        if (t + 1) % 10 == 0 or t == 0:
            elapsed = time.time() - t0
            print(
                f"    [{method_name}|s{seed}] "
                f"round {t+1:3d}/{args.rounds}  "
                f"acc={result.accuracy:.4f}  "
                f"part={result.participation_rate:.2f}  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )

    final_acc = history[-1]["accuracy"]
    best_acc  = max(h["accuracy"] for h in history)
    total_t   = time.time() - t0
    print(f"    Done: final={final_acc:.4f}  best={best_acc:.4f}  time={total_t:.1f}s")
    return {
        "method": method_name, "seed": seed, "history": history,
        "final_accuracy": final_acc, "best_accuracy": best_acc,
        "total_time": total_t,
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DASH AG News experiment (FGCS R2 Q1)")
    p.add_argument("--rounds",    type=int, default=100)
    p.add_argument("--seeds",     type=int, default=3)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",    type=str, default="results/agnews.json")
    p.add_argument("--quick",     action="store_true",
                   help="Smoke-test: 3 rounds, 10 clients, 1 seed")
    p.add_argument("--methods",   nargs="+", default=None,
                   help=f"Subset of methods (default: {METHODS})")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.rounds    = 3
        args.n_clients = 10
        args.seeds     = 1

    methods = args.methods or METHODS
    print(f"Device: {args.device}  Methods: {methods}")
    print(f"Seeds: {args.seeds}  Rounds: {args.rounds}  Clients: {args.n_clients}")

    all_results = {}
    for method_name in methods:
        print(f"\n{'='*60}\n  Method: {method_name}\n{'='*60}")
        seed_results = []
        for seed in range(args.seeds):
            try:
                seed_results.append(run_one(method_name, args, seed))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                seed_results.append({"error": str(exc), "seed": seed})

        valid = [r for r in seed_results if "error" not in r]
        agg = {}
        if valid:
            fa = [r["final_accuracy"] for r in valid]
            ba = [r["best_accuracy"]  for r in valid]
            agg["mean_final_accuracy"] = float(np.mean(fa))
            agg["std_final_accuracy"]  = float(np.std(fa))
            agg["mean_best_accuracy"]  = float(np.mean(ba))
            print(f"  >> {method_name}: {agg['mean_final_accuracy']:.4f} +/- {agg['std_final_accuracy']:.4f}")

        all_results[method_name] = {"seeds": seed_results, "aggregate": agg}

    save_json(all_results, args.output)
    print("\nAG News experiment complete.")
    print("\n" + "="*55)
    print(f"{'Method':<18} {'Final Acc':>10} {'sd':>6} {'Best Acc':>10}")
    print("-"*55)
    for name, res in all_results.items():
        agg = res.get("aggregate", {})
        if agg:
            print(f"  {name:<16} {agg['mean_final_accuracy']:>10.4f}"
                  f" {agg['std_final_accuracy']:>6.4f}"
                  f" {agg['mean_best_accuracy']:>10.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
