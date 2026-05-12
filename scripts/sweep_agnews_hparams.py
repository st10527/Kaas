#!/usr/bin/env python3
"""
Hyperparameter sweep for DASH on AG News (FGCS R2 Q1).

Sweeps three distillation-sensitive parameters across a 3-D grid:
  - local_lr      : learning rate for per-round local training
  - distill_alpha : weight of KL loss  (1-alpha = CE loss weight)
  - temperature   : softmax temperature for teacher logits

Design
------
  • 30 rounds per config  (enough to see convergence direction)
  • 2 seeds per config    (balance between speed and variance estimate)
  • Only DASH is swept    (FedAvg / RandomSelection don't use these params)
  • Full per-round history saved → reproducible, paper-disclosable

Usage
-----
    # Full grid (~48 configs × 2 seeds × 30 rounds)
    python scripts/sweep_agnews_hparams.py --output results/agnews_sweep.json

    # Quick sanity check (2×2×2 sub-grid, 1 seed, 10 rounds)
    python scripts/sweep_agnews_hparams.py --quick

Output
------
    results/agnews_sweep.json   — full per-config records
    results/agnews_sweep_best.json — top-10 configs sorted by mean_final_acc
"""

import argparse, copy, itertools, json, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────
# JSON helper
# ─────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────
# Data / partition helpers  (same as run_agnews_exp.py)
# ─────────────────────────────────────────────────────────────────

def load_data(n_public=5000):
    from src.data.agnews import load_agnews_safe_split
    return load_agnews_safe_split(root="./data", n_public=n_public, seed=42)


def partition_data(private_set, n_clients, alpha=0.3, seed=42):
    from src.data.partition import DirichletPartitioner, create_client_loaders
    if hasattr(private_set, "labels"):
        targets = np.array(private_set.labels)
    elif hasattr(private_set, "dataset") and hasattr(private_set.dataset, "labels"):
        all_lbl = np.array(private_set.dataset.labels)
        targets  = all_lbl[np.array(private_set.indices)]
    else:
        targets = np.array([private_set[i][1] for i in range(len(private_set))])
    part = DirichletPartitioner(alpha=alpha, n_clients=n_clients, seed=seed)
    idx  = part.partition(private_set, targets=targets)
    return create_client_loaders(private_set, idx, batch_size=64)


def get_model():
    from src.models.utils import get_model as _gm
    return _gm("textcnn", num_classes=4)


# ─────────────────────────────────────────────────────────────────
# Single DASH run with explicit hparam overrides
# ─────────────────────────────────────────────────────────────────

def run_dash(hparams: dict, args, seed: int,
             shared_data=None) -> dict:
    """
    Run DASH for `args.rounds` rounds and return a result record.

    hparams keys:  local_lr, distill_alpha, temperature
    shared_data:   (private_set, public_set, test_set) pre-loaded tuple
                   to avoid reloading HuggingFace dataset per config.
    """
    from src.methods.dash import DASH, DASHConfig
    from src.methods.kaas_edge import generate_edge_devices

    torch.manual_seed(seed)
    np.random.seed(seed)

    private_set, public_set, test_set = shared_data
    client_loaders = partition_data(private_set, args.n_clients,
                                    alpha=0.3, seed=seed)
    public_loader  = DataLoader(public_set, batch_size=128, shuffle=False,
                                num_workers=2, pin_memory=True)
    test_loader    = DataLoader(test_set,   batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=True)
    devices = generate_edge_devices(n_devices=args.n_clients, seed=seed)

    config = DASHConfig(
        budget          = 50.0,
        v_max           = len(public_set),
        local_epochs    = 2,
        local_lr        = hparams["local_lr"],
        distill_epochs  = 3,
        distill_lr      = 0.001,
        distill_alpha   = hparams["distill_alpha"],
        temperature     = hparams["temperature"],
        pretrain_epochs = 10,
        n_ref_samples   = len(public_set),
        straggler_aware = True,
        timeout_policy  = "adaptive",
        fixed_deadline  = 5.0,
        sigma_noise     = 0.3,
    )
    method = DASH(get_model(), config=config, device=args.device, n_classes=4)

    history, t0 = [], time.time()
    for t in range(args.rounds):
        result = method.run_round(t, devices, client_loaders, public_loader,
                                  test_loader=test_loader)
        history.append({
            "round":    t,
            "accuracy": result.accuracy,
            "loss":     result.loss,
        })

    final_acc = history[-1]["accuracy"]
    best_acc  = max(h["accuracy"] for h in history)
    elapsed   = time.time() - t0

    label = (f"lr={hparams['local_lr']:.0e}  "
             f"α={hparams['distill_alpha']:.2f}  "
             f"T={hparams['temperature']:.1f}")
    print(f"    [{label}  seed={seed}]  "
          f"final={final_acc:.4f}  best={best_acc:.4f}  ({elapsed:.0f}s)",
          flush=True)

    return {
        "hparams":        hparams,
        "seed":           seed,
        "final_accuracy": final_acc,
        "best_accuracy":  best_acc,
        "history":        history,
        "elapsed_s":      elapsed,
    }


# ─────────────────────────────────────────────────────────────────
# Grid definition
# ─────────────────────────────────────────────────────────────────

FULL_GRID = {
    "local_lr":      [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "distill_alpha": [0.0,  0.1,  0.2,  0.5,  0.8],
    "temperature":   [1.0,  2.0,  3.0,  4.0],
}

QUICK_GRID = {
    "local_lr":      [1e-4, 1e-3, 1e-2],
    "distill_alpha": [0.0,  0.2,  0.5],
    "temperature":   [1.0,  2.0,  3.0],
}


def build_grid(grid_def):
    keys   = list(grid_def.keys())
    combos = list(itertools.product(*[grid_def[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep: DASH on AG News")
    p.add_argument("--quick",      action="store_true",
                   help="3×3×3 grid, 1 seed, 10 rounds (sanity check)")
    p.add_argument("--rounds",     type=int, default=30,
                   help="Training rounds per config (default 30)")
    p.add_argument("--seeds",      type=int, default=2,
                   help="Seeds per config (default 2)")
    p.add_argument("--n_clients",  type=int, default=100)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",     type=str,
                   default="results/agnews_sweep.json")
    return p.parse_args()


def summarise(all_runs):
    """Aggregate per-config records and return sorted summary list."""
    from collections import defaultdict
    groups = defaultdict(list)
    for run in all_runs:
        key = (run["hparams"]["local_lr"],
               run["hparams"]["distill_alpha"],
               run["hparams"]["temperature"])
        groups[key].append(run)

    summary = []
    for key, runs in groups.items():
        fa = [r["final_accuracy"] for r in runs]
        ba = [r["best_accuracy"]  for r in runs]
        hp = runs[0]["hparams"]
        summary.append({
            "local_lr":             hp["local_lr"],
            "distill_alpha":        hp["distill_alpha"],
            "temperature":          hp["temperature"],
            "mean_final_accuracy":  float(np.mean(fa)),
            "std_final_accuracy":   float(np.std(fa)),
            "mean_best_accuracy":   float(np.mean(ba)),
            "n_seeds":              len(runs),
        })
    summary.sort(key=lambda x: x["mean_final_accuracy"], reverse=True)
    return summary


def print_table(summary, top_n=15):
    print(f"\n{'Rank':>4}  {'local_lr':>9}  {'alpha':>5}  {'T':>4}  "
          f"{'final_acc':>10}  {'± std':>7}  {'best_acc':>10}")
    print("─" * 65)
    for i, r in enumerate(summary[:top_n]):
        print(f"  {i+1:2d}   {r['local_lr']:>9.0e}  {r['distill_alpha']:>5.2f}  "
              f"{r['temperature']:>4.1f}  "
              f"{r['mean_final_accuracy']:>10.4f}  "
              f"±{r['std_final_accuracy']:>6.4f}  "
              f"{r['mean_best_accuracy']:>10.4f}")


def main():
    args = parse_args()
    if args.quick:
        args.rounds = 10
        args.seeds  = 1
        grid = build_grid(QUICK_GRID)
    else:
        grid = build_grid(FULL_GRID)

    total = len(grid) * args.seeds
    print(f"Device   : {args.device}")
    print(f"Grid     : {len(grid)} configs  ×  {args.seeds} seeds  =  {total} runs")
    print(f"Rounds   : {args.rounds} per run")
    print(f"Output   : {args.output}")
    print()

    # Pre-load data once → shared across all configs / seeds
    print("Loading AG News dataset …")
    shared_data = load_data()
    print(f"  public={len(shared_data[1])}  test={len(shared_data[2])}\n")

    all_runs = []
    t_global = time.time()

    for cfg_idx, hparams in enumerate(grid):
        label = (f"lr={hparams['local_lr']:.0e}  "
                 f"α={hparams['distill_alpha']:.2f}  "
                 f"T={hparams['temperature']:.1f}")
        done  = cfg_idx * args.seeds
        pct   = 100 * done / total
        print(f"[{cfg_idx+1:3d}/{len(grid)}  {pct:4.0f}%]  {label}")

        for seed in range(args.seeds):
            try:
                rec = run_dash(hparams, args, seed, shared_data=shared_data)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                rec = {"hparams": hparams, "seed": seed,
                       "error": str(exc),
                       "final_accuracy": 0.0, "best_accuracy": 0.0}
            all_runs.append(rec)

        # Checkpoint after each config (safe against remote disconnect)
        save_json({"runs": all_runs, "args": vars(args)}, args.output)

    elapsed_total = time.time() - t_global
    print(f"\nSweep complete in {elapsed_total/60:.1f} min")

    summary = summarise(all_runs)
    print_table(summary)

    best_output = args.output.replace(".json", "_best.json")
    save_json({
        "summary":    summary,
        "top1_hparams": summary[0] if summary else {},
        "sweep_args": vars(args),
        "all_runs":   all_runs,
    }, best_output)

    print(f"\nBest config:")
    b = summary[0]
    print(f"  local_lr={b['local_lr']:.0e}  "
          f"distill_alpha={b['distill_alpha']:.2f}  "
          f"temperature={b['temperature']:.1f}")
    print(f"  mean_final={b['mean_final_accuracy']:.4f} ± {b['std_final_accuracy']:.4f}  "
          f"mean_best={b['mean_best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
