# KaaS-Edge: Knowledge-as-a-Service for Edge Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation framework for **Knowledge-as-a-Service at the Edge (KaaS-Edge)** — a resource-aware federated distillation system where edge devices contribute logits under per-round communication budgets and Local Differential Privacy constraints.

> Note: This repository also contains the original PAID-FD codebase (Privacy-Aware Incentive-Driven Federated Distillation). Those files remain intact but are not the focus of this documentation.

---

## 🔗 連結 (Links)

| Resource | URL |
|----------|-----|
| Repository | https://github.com/st10527/Kaas |
| CIFAR-100 Dataset | https://www.cs.toronto.edu/~kriz/cifar.html |
| PyTorch | https://pytorch.org/ |
| KaaS-Edge Method | `src/methods/kaas_edge.py` |
| RADS Scheduler | `src/scheduler/rads.py` |
| Edge Configuration | `config/edge_default.yaml` |

---

## 📖 使用者指南 (User Guide)

### Installation

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

### Running KaaS-Edge Experiments

```bash
# Main method comparison (KaaS-Edge vs baselines, 3 seeds × 50 rounds)
python scripts/run_edge_experiments.py --exp main

# Quick smoke-test (reduced dataset, CPU-friendly)
python scripts/run_edge_experiments.py --exp main --quick

# Budget sensitivity analysis (sweep B)
python scripts/run_edge_experiments.py --exp budget

# Privacy impact analysis (vary ρ distribution)
python scripts/run_edge_experiments.py --exp privacy

# Run all four experiment suites
python scripts/run_edge_experiments.py --all

# Specify GPU device
python scripts/run_edge_experiments.py --all --device cuda:0
```

**Parallel execution across seeds** (recommended for single-GPU setups):

```bash
# Runs seeds 42 & 123 in parallel (Wave 1), then seed 456 alone (Wave 2)
bash parallel_run.sh main cuda:0

# Run all experiments in parallel mode
bash parallel_run.sh all cuda:0
```

### Viewing and Plotting Results

Results are saved to `results/edge/` as JSON files (one per seed per suite). After all runs complete:

```bash
# Merge per-seed JSON files into a single summary
python merge_seed_results.py

# Generate all paper figures from merged results
python plot_edge_figures.py
```

### Key Configuration (`config/edge_default.yaml`)

```yaml
scheduler:
  budget: 50          # Per-round total cost budget B
  v_max: 10000        # Maximum upload volume per device (= |D_ref|)
  delta: 1e-6         # Bisection tolerance for water-filling

devices:
  n_devices: 20
  theta_range: [20.0, 80.0]   # Half-saturation constants θ_i
  activation_cost_range: [0.05, 0.2]

training:
  n_rounds: 50
  local_epochs: 2
  lr: 0.01
  distill_temperature: 3.0    # Knowledge distillation temperature τ
  kd_loss_weight: 0.7         # α_KD

data:
  dataset: cifar100
  n_public: 10000             # Reference dataset size |D_ref|
  dirichlet_alpha: 0.5        # Non-IID heterogeneity parameter
```

---

## 🌐 環境 (Environment)

### Software Dependencies

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0.0 |
| torchvision | ≥ 0.15.0 |
| NumPy | ≥ 1.24.0 |
| SciPy | ≥ 1.10.0 |
| scikit-learn | ≥ 1.3.0 |
| PyYAML | ≥ 6.0 |
| matplotlib | ≥ 3.7.0 |
| tqdm | ≥ 4.65.0 |

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Project Structure (KaaS-Edge relevant files)

```
Kaas/
├── config/
│   ├── edge_default.yaml       # KaaS-Edge hyper-parameters
│   └── devices/
│       └── heterogeneity.yaml  # 3-tier device heterogeneity profile
├── src/
│   ├── data/
│   │   ├── datasets.py         # CIFAR-100 loader and safe-split helper
│   │   └── partition.py        # Dirichlet non-IID partitioning
│   ├── devices/
│   │   └── heterogeneity.py    # 3-tier device profile generator
│   ├── privacy/
│   │   └── ldp.py              # Laplace LDP noise mechanism
│   ├── scheduler/
│   │   └── rads.py             # RADS: water-filling + greedy selection
│   ├── models/
│   │   └── cnn.py              # Lightweight CNN (~1.2 M params)
│   └── methods/
│       └── kaas_edge.py        # KaaS-Edge main federated method
├── scripts/
│   └── run_edge_experiments.py # KaaS-Edge experiment runner
├── parallel_run.sh             # Parallel seed runner (3 seeds)
├── merge_seed_results.py       # Merge per-seed JSON results
├── plot_edge_figures.py        # Generate paper figures
├── results/
│   └── edge/                   # KaaS-Edge JSON output files
└── tests/                      # Unit tests
```

---

## 🔬 資料與方法 (Data and Methods)

### Dataset

KaaS-Edge is evaluated on **CIFAR-100** (100 classes, 32×32 colour images).

- **Private data**: Each of the M = 20 devices holds a private partition generated by **Dirichlet allocation** with α = 0.5, simulating non-IID statistical heterogeneity across devices.
- **Reference dataset D_ref**: 10,000 samples drawn from a non-overlapping partition of the CIFAR-100 training set, ensuring **zero data leakage** between private and public sources. D_ref is hosted on the Edge Server (ES) and used for distillation.

### Model Architecture

All devices use the same **lightweight CNN**:

- 3 convolutional blocks with batch normalisation
- 2 fully connected layers

This homogeneous architecture isolates the scheduling algorithm's contribution from confounding effects of varying model capacities.

### KaaS-Edge Method

Each device `i` uploads `v_i` logit vectors computed on D_ref. Upload volume is governed by a **saturation quality model**:

```
q_i(v_i) = ρ_i · v_i / (v_i + θ_i)
```

where `ρ_i ∈ (0,1]` is the privacy degradation factor and `θ_i ∈ [20, 80]` is the half-saturation constant.

**RADS — Resource-Aware Distillation Scheduling** operates in two stages each round:

1. **Water-filling allocation** (Proposition 1): For a fixed device set S and residual budget B, compute optimal upload volumes via KKT conditions:

   ```
   v_i* = min{ [√(ρ_i·θ_i / (ν·b_i)) − θ_i]⁺ , v_max }
   ```

   where `ν` (water level) is found by bisection such that `Σ b_i·v_i* = B`.

2. **Greedy device selection** (Theorem 1): Iteratively add the device with the highest marginal quality gain subject to the budget. Devices are pre-sorted by efficiency index `η_i = ρ_i / (b_i · θ_i)`.

   **Approximation guarantee**: `Q(S^G) ≥ (1 − 1/e)/2 · Q(S*) ≈ 0.316 · OPT`

**Local Differential Privacy**: Laplace noise is added to uploaded logits before transmission:

```
noise scale = 2C / ε,   ε = ρ / (1 − ρ)
```

where `C` is the clipping bound and `ρ` is the privacy degradation factor.

### Communication and Computation Cost Model

Per-round communication costs `ω_i` (the cost of uploading one logit vector over the wireless channel) are drawn from a **Rayleigh fading channel** model:

| Parameter | Value |
|-----------|-------|
| Path loss exponent α_PL | 3.5 |
| Device-to-server distance | Uniform [20, 300] m |
| Bandwidth | 1 MHz |
| Transmit power | 100 mW |

The per-logit activation cost `b_i` used in the water-filling formula represents the marginal resource cost (combining communication and computation) of uploading one additional logit vector; it is derived from `ω_i` and the device's computational load.

Computation costs `μ_i` follow a log-normal distribution reflecting diverse hardware capabilities. Device-tier multipliers of 1×, 2×, and 4× yield a **three-tier heterogeneity profile**. Activation costs `a_i` are drawn uniformly from [0.05, 0.2].

### Training Hyper-parameters

| Parameter | Value |
|-----------|-------|
| Rounds T | 50 |
| Local epochs per round | 2 |
| Optimiser | SGD (lr = 0.01) |
| Distillation temperature τ | 3.0 |
| KD loss weight α_KD | 0.7 |
| Budget B | 50 |
| Max upload v_max | 10,000 (= \|D_ref\|) |
| Privacy setting | "mixed" (ρ̄ ≈ 0.55): half the devices have high privacy (small ρ) and the other half have low privacy (large ρ), yielding an average ρ̄ ≈ 0.55 |
| Seeds | 3 (42, 123, 456) |

---

## 📊 評估方式 (Evaluation Methods)

### Metrics

All results report **mean ± standard deviation over 3 random seeds**.  
Primary metric: **Top-1 test accuracy on CIFAR-100** across T = 50 communication rounds.

Secondary metrics tracked per experiment suite:

| Suite | Flag | Primary Variable | Metric |
|-------|------|-----------------|--------|
| Main comparison | `--exp main` | Method | Convergence curve + final accuracy |
| Budget sensitivity | `--exp budget` | Budget B ∈ {10, 20, 30, 50, 80} | Final accuracy vs B |
| Device scalability | `--exp scale` | Devices M ∈ {5, 10, 20, 30, 50} | Final accuracy vs M |
| Privacy impact | `--exp privacy` | Privacy setting (none / mild / mixed / strong) | Accuracy vs privacy level |

### Baselines

KaaS-Edge is compared against four baselines:

| Baseline | Description | Reference |
|----------|-------------|-----------|
| **FedMD** | All devices upload complete logits every round (no budget constraint) | Li et al., NeurIPS 2019 |
| **FedSKD** | Each device uploads 50% of its logits | Sattler et al., 2021 |
| **FedCS-FD** | Budget split equally across all devices | — |
| **Random Selection** | Half the devices selected at random each round to upload complete logits | — |

---

## 💻 系統需求 (System Requirements)

| Task | Minimum | Recommended |
|------|---------|-------------|
| Quick smoke-test (`--quick`) | CPU | CPU |
| Single experiment (50 rounds) | GPU with 4 GB VRAM | RTX 3090 (24 GB) |
| Full experiments (3 seeds) | RTX 3090 | RTX 5070 Ti / A100 |

**GPU Memory**: approximately 1 GB VRAM per concurrent CIFAR-100 experiment with the lightweight CNN.

`parallel_run.sh` runs Wave 1 (seeds 42 & 123) concurrently (~2 GB total), then Wave 2 (seed 456) alone, so **≥ 4 GB VRAM** is sufficient for parallel execution. A 16 GB GPU can run additional experiment suites in parallel.

**OS**: Linux or macOS (Windows supported via `venv\Scripts\activate`).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
