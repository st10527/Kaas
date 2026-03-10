# KaaS — KaaS-Edge: Knowledge-as-a-Service for Edge Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation framework for **Knowledge-as-a-Service at the Edge (KaaS-Edge)**, extending the original [PAID-FD](https://github.com/st10527/paid-fd) codebase with a new resource-aware distillation scheduling algorithm targeting IEEE EDGE 2026.

> **Base paper (PAID-FD)**: Privacy-Aware Incentive-Driven Federated Distillation — Targeting IEEE Transactions on Mobile Computing (TMC)
>
> **This extension (KaaS-Edge)**: Knowledge-as-a-Service for Edge Intelligence — Targeting IEEE EDGE 2026

---

## 🌟 What's New in This Fork

Compared to the original PAID-FD, this repository adds:

- **KaaS-Edge method** (`src/methods/kaas_edge.py`): federated distillation where each device contributes a variable number of reference-data logits, capped by a per-round budget.
- **RADS scheduler** (`src/scheduler/rads.py`): Resource-Aware Distillation Scheduling using water-filling allocation (Proposition 1) and greedy submodular device selection (Theorem 1), achieving a `(1-1/e)/2 ≈ 0.316 × OPT` approximation guarantee.
- **Edge experiment runner** (`scripts/run_edge_experiments.py`): four experiment suites (main comparison, budget sensitivity, device scalability, privacy impact).
- **Parallel execution script** (`parallel_run.sh`): runs multiple seeds in parallel on a single GPU and merges results.
- **Edge configuration** (`config/edge_default.yaml`): dedicated YAML for KaaS-Edge hyper-parameters.

The full PAID-FD code (Stackelberg game, TMC experiments, FedAvg / FedMD / CSRA / FedGMKD baselines) is still present and fully operational.

---

## 📁 Project Structure

```
Kaas/
├── config/
│   ├── default.yaml            # PAID-FD default settings (TMC)
│   ├── edge_default.yaml       # KaaS-Edge default settings (EDGE 2026)
│   ├── devices/
│   │   └── heterogeneity.yaml  # 3-D device heterogeneity parameters
│   └── experiments/            # Per-experiment YAML overrides
├── src/
│   ├── data/
│   │   ├── datasets.py         # CIFAR-100, STL-10, safe-split helper
│   │   └── partition.py        # Dirichlet / IID partitioning
│   ├── devices/
│   │   ├── heterogeneity.py    # 3-D device profile generator
│   │   └── energy.py           # Energy consumption model
│   ├── game/
│   │   ├── stackelberg.py      # PAID-FD: Algorithms 1 & 2
│   │   └── utility.py          # Quality / utility functions
│   ├── privacy/
│   │   └── ldp.py              # Laplace / Gaussian LDP
│   ├── scheduler/
│   │   └── rads.py             # KaaS-Edge: RADS water-filling + greedy
│   ├── models/
│   │   ├── resnet.py           # ResNet-18/34
│   │   ├── cnn.py              # Lightweight CNN (~1.2 M params)
│   │   └── utils.py            # get_model() factory
│   ├── methods/
│   │   ├── base.py             # FederatedMethod base class
│   │   ├── kaas_edge.py        # KaaS-Edge (this fork's main method)
│   │   ├── paid_fd.py          # PAID-FD (original TMC method)
│   │   ├── fixed_eps.py        # Fixed-ε ablation
│   │   ├── fedmd.py            # FedMD baseline
│   │   ├── fedavg.py           # FedAvg baseline
│   │   ├── csra.py             # CSRA baseline
│   │   └── fedgmkd.py          # FedGMKD baseline
│   └── utils/                  # Logging, seeding, results helpers
├── experiments/
│   └── run_experiment.py       # PAID-FD unified runner (TMC)
├── scripts/
│   ├── run_edge_experiments.py # KaaS-Edge experiment runner (EDGE 2026)
│   ├── run_experiments.py      # PAID-FD multi-phase runner (TMC)
│   ├── run_all_experiments.py  # Full PAID-FD phase runner
│   └── run_parallel.sh         # PAID-FD parallel runner (3 seeds)
├── parallel_run.sh             # KaaS-Edge parallel runner (3 seeds)
├── merge_seed_results.py       # Merge per-seed JSONs for KaaS-Edge
├── plot_edge_figures.py        # Generate figures for KaaS-Edge paper
├── results/                    # Output directory
│   └── edge/                   # KaaS-Edge JSON results
└── tests/                      # Unit tests
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/st10527/Kaas.git
cd Kaas

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run KaaS-Edge Experiments (this fork's main contribution)

```bash
# Single experiment — main method comparison (3 seeds × 50 rounds)
python scripts/run_edge_experiments.py --exp main

# Quick smoke-test (fewer samples)
python scripts/run_edge_experiments.py --exp main --quick

# Budget sensitivity analysis
python scripts/run_edge_experiments.py --exp budget

# Device scalability (vary number of devices)
python scripts/run_edge_experiments.py --exp scale

# Privacy impact (vary rho distribution)
python scripts/run_edge_experiments.py --exp privacy

# Run all four experiment suites
python scripts/run_edge_experiments.py --all

# Specify GPU
python scripts/run_edge_experiments.py --all --device cuda:0
```

**Parallel execution across seeds** (recommended on a single GPU with ≥ 8 GB VRAM):

```bash
# Runs seeds 42 & 123 in parallel (Wave 1), then seed 456 alone (Wave 2)
bash parallel_run.sh main cuda:0

# Run all experiments (main, budget, scale, privacy) the same way
bash parallel_run.sh all cuda:0
```

Results are saved to `results/edge/` as JSON files. After parallel runs, merge them:

```bash
python merge_seed_results.py
```

### 3. Run Original PAID-FD Experiments (TMC paper)

```bash
# Quick test with synthetic data (no download, CPU-friendly)
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method PAID-FD \
    --synthetic \
    --rounds 10

# Full run with real data
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method PAID-FD

# All methods
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method all \
    --device cuda:0

# Force re-run (ignore cached results)
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method PAID-FD \
    --force
```

---

## ⚙️ Configuration

### KaaS-Edge (`config/edge_default.yaml`)

Key parameters:

```yaml
scheduler:
  budget: 8.0       # Per-round total cost budget B
  v_max: 200        # Maximum upload volume per device
  delta: 1e-6       # Bisection tolerance for water-filling

devices:
  n_devices: 20
  cost_range: [0.05, 0.3]
  theta_range: [20.0, 80.0]   # Half-saturation constants

training:
  n_rounds: 50
  local_epochs: 2
  distill_epochs: 3
  pretrain_epochs: 10
  clip_bound: 2.0   # Laplace LDP clipping bound

data:
  dataset: cifar100
  n_public: 10000   # Reference dataset size |D_ref|
  dirichlet_alpha: 0.5
```

### PAID-FD (`config/default.yaml`)

Key parameters:

```yaml
system:
  n_devices: 50
  seed: 42

paid_fd:
  gamma: 50.0       # Server valuation coefficient
  clip_bound: 5.0   # LDP clipping bound

training:
  n_rounds: 200
  local_epochs: 1
  distill_epochs: 5
```

---

## 🔬 KaaS-Edge: Algorithm Overview

### Quality Model

Each device `i` uploads `v_i` reference-data logit vectors. Quality follows a saturation function:

```
q_i(v_i) = ρ_i · v_i / (v_i + θ_i)
```

where `ρ_i ∈ (0,1]` is the privacy degradation factor and `θ_i` is the half-saturation constant.

### RADS: Two-Stage Scheduling

**Stage 1 — Water-filling allocation** (Proposition 1):  
Given a fixed device set `S` and residual budget `B_res`, solve for the optimal upload volumes via KKT conditions:

```
v_i* = min{ [√(ρ_i·θ_i / (ν·b_i)) − θ_i]⁺ , v_max }
```

where `ν` (water level) is found by bisection such that `Σ b_i·v_i* = B_res`.

**Stage 2 — Greedy device selection** (Theorem 1):  
Iteratively add the device with the highest marginal quality gain subject to the budget constraint. Devices are pre-sorted by efficiency index `η_i = ρ_i / (b_i · θ_i)`.

**Approximation guarantee**: `Q(S^G) ≥ (1 − 1/e)/2 · Q(S*) ≈ 0.316 · OPT`

### Privacy

Local Differential Privacy is applied to uploaded logits using Laplace noise:

```
scale = 2C / ε,   ε = ρ / (1 − ρ)
```

where `C` is the clipping bound and `ρ` is the device's privacy degradation factor.

---

## 📊 Experiment Suites

### KaaS-Edge (EDGE 2026)

| Suite | Flag | Description |
|-------|------|-------------|
| Main comparison | `--exp main` | KaaS-Edge vs FedMD, FedAvg, PAID-FD over 50 rounds |
| Budget sensitivity | `--exp budget` | Sweep `B ∈ {2, 4, 6, 8, 10, 15, 20}` |
| Device scalability | `--exp scale` | Sweep `M ∈ {5, 10, 20, 30, 50}` |
| Privacy impact | `--exp privacy` | No-privacy / mild / mixed / strong `ρ` distributions |

### PAID-FD (TMC paper)

| Phase | Config | Description |
|-------|--------|-------------|
| Phase 1 | `phase1_gamma.yaml` / `phase1_lambda.yaml` | Parameter sensitivity |
| Phase 2 | `phase2_convergence.yaml` | Convergence & accuracy |
| Phase 3 | `phase3_privacy.yaml` | Privacy-accuracy tradeoff |
| Phase 4 | `phase4_incentive.yaml` | Incentive analysis |
| Phase 5 | `phase5_heterogeneity.yaml` | Heterogeneity impact |
| Phase 6 | `phase6_scalability.yaml` | Scalability |
| Phase 7 | `phase7_ablation.yaml` | Ablation study |

---

## 📈 Viewing Results

### KaaS-Edge

Results are saved to `results/edge/` as JSON files (one per seed per experiment suite).

```bash
# After running all experiments with 3 seeds, merge them
python merge_seed_results.py

# Generate all paper figures
python plot_edge_figures.py
```

### PAID-FD

```python
from src.utils.results import ResultManager

manager = ResultManager()

# List available results
files = manager.list_results("exp2_convergence")

# Compare methods on a metric
comparison = manager.compare_results("exp2_convergence", metric="final_accuracy")
print(comparison)
```

---

## 🔧 Methods

| Method | Description | Paper |
|--------|-------------|-------|
| **KaaS-Edge** | RADS water-filling + greedy selection + Laplace LDP | EDGE 2026 (this work) |
| **PAID-FD** | Stackelberg game + adaptive ε | TMC (base work) |
| **Fixed-ε** | Fixed privacy budget ablation | — |
| **FedMD** | FD baseline, no privacy | NeurIPS 2019 |
| **FedAvg** | Parameter averaging | AISTATS 2017 |
| **CSRA** | Reverse auction DP-FL | TIFS 2024 |
| **FedGMKD** | GMM prototype KD + DAT | 2024 |

---

## 💻 Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Quick smoke-test (`--quick`) | CPU | CPU |
| Single experiment (50 rounds) | GTX 1080 (4 GB) | RTX 3090 |
| Full experiments (3 seeds) | RTX 3090 | RTX 5070 Ti / A100 |

**Memory**: ~1 GB VRAM per concurrent CIFAR-100 experiment with the lightweight CNN.  
`parallel_run.sh` runs Wave 1 (seeds 42 & 123) concurrently (~2 GB total) then Wave 2 (seed 456) alone, so ≥ 4 GB VRAM is sufficient; the 16 GB RTX 5070 Ti can run additional experiments in parallel if desired.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CIFAR-100
- PyTorch team
- Federated learning community
- Original PAID-FD authors
