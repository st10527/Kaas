#!/usr/bin/env python3
"""DASH ablation study -- FGCS R2 Q2.

Five variants on CIFAR-100, M=100 clients, 3 seeds:
  full         -- all components enabled
  no_straggler -- ablation_use_straggler_selection=False
  no_wf        -- ablation_use_water_filling=False
  no_timeout   -- ablation_use_adaptive_timeout=False
  no_quality   -- ablation_use_quality_weights=False

Usage:
    python scripts/run_ablation_exp.py [--quick] [--rounds N] [--seeds N]
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


ABLATION_VARIANTS = {
    "full":         {},
    "no_straggler": {"ablation_use_straggler_selection": False},
    "no_wf":        {"ablation_use_water_filling": False},
    "no_timeout":   {"ablation_use_adaptive_timeout": False},
    "no_quality":   {"ablation_use_quality_weights": False},
}


def load_data(n_public=5000, quick=False):
    from src.data.datasets import load_cifar100_safe_split
    if quick:
        n_public = 2000
    return load_cifar100_safe_split(root="./data", n_public=n_public, seed=42)


def partition_data(private_set, n_clients, alpha=0.3, seed=42):
    from src.data.partition import DirichletPartitioner, create_client_loaders
    if hasattr(private_set, "dataset") and hasattr(private_set.dataset, "targets"):
        all_targets = np.array(private_set.dataset.targets)
        subset_targets = all_targets[np.array(private_set.indices)]
    else:
        subset_targets = np.array([private_set[i][1] for i in range(len(private_set))])
    partitioner = DirichletPartitioner(alpha=alpha, n_clients=n_clients, seed=seed)
    client_indices = partitioner.partition(private_set, targets=subset_targets)
    return create_client_loaders(private_set, client_indices, batch_size=32)


def create_model():
    from src.models.utils import get_model
    return get_model("cnn", num_classes=100)


def run_one(variant_name, flag_overrides, args, seed):
    from src.methods.dash import DASH, DASHConfig
    from src.methods.kaas_edge import generate_edge_devices
    torch.manual_seed(seed)
    np.random.seed(seed)
    private_set, public_set, test_set = load_data(quick=args.quick)
    client_loaders = partition_data(private_set, args.n_clients, alpha=0.3, seed=seed)
    public_loader = DataLoader(public_set, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)
    devices = generate_edge_devices(n_devices=args.n_clients, seed=seed)
    cfg = dict(
        budget=50.0, v_max=len(public_set), local_epochs=2,
        distill_epochs=3, distill_lr=0.001,
        pretrain_epochs=5 if args.quick else 10,
        n_ref_samples=len(public_set), straggler_aware=True,
        timeout_policy="adaptive", fixed_deadline=5.0, sigma_noise=0.3,
        ablation_use_straggler_selection=True,
        ablation_use_water_filling=True,
        ablation_use_adaptive_timeout=True,
        ablation_use_quality_weights=True,
    )
    cfg.update(flag_overrides)
    method = DASH(create_model(), config=DASHConfig(**cfg), device=args.device)
    history, t0 = [], time.time()
    for t in range(args.rounds):
        result = method.run_round(t, devices, client_loaders, public_loader,
                                  test_loader=test_loader)
        history.append({
            "round": t,
            "accuracy": result.accuracy,
            "loss": result.loss,
            "participation_rate": result.participation_rate,
            "n_participants": result.n_participants,
            "energy": result.energy,
            "real_time_s": result.extra.get("real_time_s", 0.0),
            "wall_clock": result.extra.get("wall_clock_time", 0.0),
        })
        if (t + 1) % 10 == 0 or t == 0:
            elapsed = time.time() - t0
            print(
                f"    [{variant_name}|s{seed}] "
                f"round {t+1:3d}/{args.rounds}  "
                f"acc={result.accuracy:.4f}  "
                f"part={result.participation_rate:.2f}  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
    final_acc = history[-1]["accuracy"]
    best_acc = max(h["accuracy"] for h in history)
    total_t = time.time() - t0
    print(f"    Done: final={final_acc:.4f}  best={best_acc:.4f}  time={total_t:.1f}s")
    return {
        "variant": variant_name, "seed": seed, "history": history,
        "final_accuracy": final_acc, "best_accuracy": best_acc,
        "total_time": total_t,
    }


def parse_args():
    p = argparse.ArgumentParser(description="DASH ablation study")
    p.add_argument("--rounds",    type=int, default=100)
    p.add_argument("--seeds",     type=int, default=3)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",    type=str, default="results/ablation.json")
    p.add_argument("--quick",     action="store_true")
    p.add_argument("--variants",  nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.rounds = 3
        args.n_clients = 10
        args.seeds = 1
    variants = args.variants or list(ABLATION_VARIANTS.keys())
    print(f"Device: {args.device}  Seeds: {args.seeds}  Rounds: {args.rounds}  Clients: {args.n_clients}")
    all_results = {}
    for variant_name in variants:
        flag_overrides = ABLATION_VARIANTS[variant_name]
        print(f"\n{'='*60}\n  Variant: {variant_name}  overrides={flag_overrides}\n{'='*60}")
        seed_results = []
        for seed in range(args.seeds):
            try:
                seed_results.append(run_one(variant_name, flag_overrides, args, seed))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                seed_results.append({"error": str(exc), "seed": seed})
        valid = [r for r in seed_results if "error" not in r]
        agg = {}
        if valid:
            fa_list = [r["final_accuracy"] for r in valid]
            ba_list = [r["best_accuracy"] for r in valid]
            agg["mean_final_accuracy"] = float(np.mean(fa_list))
            agg["std_final_accuracy"]  = float(np.std(fa_list))
            agg["mean_best_accuracy"]  = float(np.mean(ba_list))
            m = agg["mean_final_accuracy"]
            s = agg["std_final_accuracy"]
            print(f"  >> {variant_name}: {m:.4f} +/- {s:.4f}")
        all_results[variant_name] = {"seeds": seed_results, "aggregate": agg}
    save_json(all_results, args.output)
    print("\nAblation study complete.")
    print("\n" + "="*55)
    print(f"{'Variant':<16} {'Final Acc':>10} {'sd':>6} {'Best Acc':>10}")
    print("-"*55)
    for name, res in all_results.items():
        agg = res.get("aggregate", {})
        if agg:
            print(f"  {name:<14} {agg['mean_final_accuracy']:>10.4f}"
                  f" {agg['std_final_accuracy']:>6.4f}"
                  f" {agg['mean_best_accuracy']:>10.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
