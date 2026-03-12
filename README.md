# KaaS-Edge: Knowledge-as-a-Service for Edge Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation framework for **Knowledge-as-a-Service at the Edge (KaaS-Edge)**, implementing a resource-aware distillation scheduling algorithm for federated knowledge distillation under heterogeneous edge environments.

---

## 📁 Project Structure

```
KaaS-Edge/
├── config/
│   ├── edge_default.yaml       # Default hyper-parameters
│   └── devices/
│       └── heterogeneity.yaml  # 3-D device heterogeneity profiles
├── src/
│   ├── data/
│   │   ├── datasets.py         # CIFAR-100 dataset loading & safe-split
│   │   └── partition.py        # Dirichlet / IID partitioning
│   ├── devices/
│   │   ├── heterogeneity.py    # 3-D device profile generator
│   │   └── energy.py           # Energy consumption model
│   ├── privacy/
│   │   └── ldp.py              # Laplace / Gaussian LDP mechanisms
│   ├── scheduler/
│   │   └── rads.py             # RADS: water-filling + greedy selection
│   ├── models/
│   │   ├── resnet.py           # ResNet-18/34
│   │   ├── cnn.py              # Lightweight CNN (~1.2 M params)
│   │   └── utils.py            # get_model() factory
│   ├── methods/
│   │   ├── base.py             # FederatedMethod base class
│   │   ├── kaas_edge.py        # KaaS-Edge (proposed method)
│   │   ├── fedmd.py            # FedMD baseline
│   │   ├── fedavg.py           # FedAvg baseline
│   │   ├── csra.py             # CSRA baseline
│   │   └── fedgmkd.py          # FedGMKD baseline
│   └── utils/                  # Logging, seeding, results helpers
├── scripts/
│   └── run_edge_experiments.py # Experiment runner (4 suites)
├── parallel_run.sh             # Parallel execution across seeds
├── merge_seed_results.py       # Merge per-seed JSON results
├── plot_edge_figures.py        # Generate paper figures
├── diagnostic_vmax.py          # Pre-experiment config verification
├── results/
│   └── edge/                   # JSON experiment outputs
└── figures/                    # Generated tables & figures
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <REPO_URL>
cd KaaS-Edge

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Single experiment — main method comparison (3 seeds × 50 rounds)
python scripts/run_edge_experiments.py --exp main

# Quick smoke-test (fewer samples, CPU-friendly)
python scripts/run_edge_experiments.py --exp main --quick

# Budget sensitivity analysis
python scripts/run_edge_experiments.py --exp budget

# Device scalability (vary number of devices)
python scripts/run_edge_experiments.py --exp scale

# Privacy impact (vary ρ distribution)
python scripts/run_edge_experiments.py --exp privacy

# Run all four experiment suites
python scripts/run_edge_experiments.py --all

# Specify GPU
python scripts/run_edge_experiments.py --all --device cuda:0
```

**Parallel execution across seeds** (recommended for GPU with ≥ 8 GB VRAM):

```bash
# Runs seeds 42 & 123 in parallel (Wave 1), then seed 456 alone (Wave 2)
bash parallel_run.sh main cuda:0

# Run all experiments
bash parallel_run.sh all cuda:0
```

### 3. Process Results & Generate Figures

```bash
# Merge per-seed results
python merge_seed_results.py

# Generate paper figures
python plot_edge_figures.py
```

Results are saved to `results/edge/` as JSON files.

---

## ⚙️ Configuration

Key parameters in `config/edge_default.yaml`:

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

---

## 🔬 Algorithm Overview

### Quality Model

Each device *i* uploads *v_i* reference-data logit vectors. Quality follows a saturation function:

$$q_i(v_i) = \rho_i \cdot \frac{v_i}{v_i + \theta_i}$$

where ρ_i ∈ (0,1] is the privacy degradation factor and θ_i is the half-saturation constant.

### RADS: Two-Stage Scheduling

**Stage 1 — Water-filling allocation (Proposition 1):**

Given a fixed device set *S* and residual budget *B_res*, the optimal upload volumes are:

$$v_i^* = \min\left(\left[\sqrt{\frac{\rho_i \cdot \theta_i}{\nu \cdot b_i}} - \theta_i\right]^+,\; v_{\max}\right)$$

where ν (water level) is found by bisection such that Σ b_i · v_i* = B_res.

**Stage 2 — Greedy device selection (Theorem 1):**

Iteratively add the device with the highest marginal quality gain subject to the budget constraint. Devices are pre-sorted by efficiency index η_i = ρ_i / (b_i · θ_i).

**Approximation guarantee:** Q(S^G) ≥ (1 − 1/e)/2 · Q(S*) ≈ 0.316 · OPT

### Privacy

Local Differential Privacy is applied to uploaded logits using Laplace noise:

$$\text{scale} = \frac{2C}{\varepsilon}, \quad \varepsilon = \frac{\rho}{1 - \rho}$$

---

## 📊 Experiment Suites

| Suite | Flag | Description |
|-------|------|-------------|
| Main comparison | `--exp main` | KaaS-Edge vs FedMD, FedSKD, FedCS-FD, Random over 50 rounds |
| Budget sensitivity | `--exp budget` | Sweep B ∈ {10, 20, 30, 40, 50, 60, 70, 80} |
| Device scalability | `--exp scale` | Sweep M ∈ {5, 10, 20, 30, 50} |
| Privacy impact | `--exp privacy` | No-privacy / mild / mixed / strong ρ distributions |

---

## 🔧 Compared Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| **KaaS-Edge** | RADS water-filling + greedy selection + Laplace LDP | This work |
| **FedMD [Full]** | All devices upload complete logits every round | Li & Wang, NeurIPS 2019 |
| **FedSKD [Selective]** | Each device uploads top-50% logits | Gad et al., ICC 2024 |
| **FedCS-FD [Equal]** | Budget split equally across selected devices | — |
| **Random Selection** | Random 50% device selection per round | — |

---

## 💻 Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Quick smoke-test (`--quick`) | CPU only | CPU only |
| Single experiment (50 rounds) | GPU with 4 GB VRAM | RTX 3090 |
| Full experiments (3 seeds) | GPU with 8 GB VRAM | RTX 5070 Ti / A100 |

**Memory**: ~1 GB VRAM per concurrent CIFAR-100 experiment with the lightweight CNN.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
