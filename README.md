# KaaS: Knowledge-as-a-Service for Edge Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation framework for **resource-aware federated knowledge distillation** on heterogeneous edge devices. This codebase supports two papers:

| Paper | Method | Venue | Key Feature |
|-------|--------|-------|-------------|
| **DASH** | Deadline-Aware Straggler-Tolerant Scheduling | JPDC 2026 | Asynchronous FD with straggler handling |
| **KaaS-Edge** | Resource-Aware Distillation Scheduling | IEEE EDGE 2026 | Synchronous FD with quality-aware scheduling |

---

## 📁 Project Structure

```
KaaS/
├── src/
│   ├── data/                    # Dataset loading & Dirichlet partitioning
│   ├── devices/                 # 3-tier device profiles & energy model
│   ├── privacy/                 # Laplace/Gaussian LDP mechanisms
│   ├── scheduler/
│   │   └── rads.py              # RADS: water-filling + greedy selection
│   ├── async_module/
│   │   ├── straggler_model.py   # LogNormal straggler latency simulation
│   │   └── timeout_policy.py    # Fixed / Adaptive / Partial-Accept policies
│   ├── models/
│   │   ├── cnn.py               # CNN (4.7M params)
│   │   └── resnet.py            # ResNet-18/34
│   └── methods/
│       ├── dash.py              # DASH — async straggler-aware FD (JPDC)
│       ├── kaas_edge.py         # KaaS-Edge — sync FD (EDGE)
│       ├── fedbuff_fd.py        # FedBuff-FD baseline
│       ├── fedmd.py             # FedMD baseline
│       └── ...
├── scripts/
│   ├── run_jpdc_experiments.py  # JPDC: 6 experiment suites
│   ├── run_edge_experiments.py  # EDGE: 4 experiment suites
│   ├── analyze_exp*.py          # Result analysis & summary generation
│   └── edge/                    # EDGE-specific utilities
├── config/
│   ├── jpdc_default.yaml        # JPDC default parameters
│   └── edge_default.yaml        # EDGE default parameters
├── results/
│   ├── jpdc/                    # JPDC experiment JSON outputs
│   └── edge/                    # EDGE experiment JSON outputs
├── tests/                       # Unit tests for async modules
└── figures/                     # Generated tables & figures (EDGE)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <REPO_URL>
cd KaaS
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run JPDC Experiments (DASH)

```bash
# Exp 1: Main comparison (6 methods × 3 seeds)
python scripts/run_jpdc_experiments.py --exp main --device cuda:0

# Exp 2: Straggler severity sweep (σ = 0, 0.3, 0.5, 1.0, 1.5)
python scripts/run_jpdc_experiments.py --exp straggler --device cuda:0

# Exp 3: Timeout policy comparison
python scripts/run_jpdc_experiments.py --exp policy --device cuda:0

# Exp 4: Scalability (M = 20, 50, 100, 200)
python scripts/run_jpdc_experiments.py --exp scale --device cuda:0

# Exp 5: Cross-dataset validation (EMNIST-ByClass)
python scripts/run_jpdc_experiments.py --exp emnist --device cuda:0

# Exp 6: Privacy robustness under async
python scripts/run_jpdc_experiments.py --exp privacy --device cuda:0
```

### 3. Analyze Results

```bash
python scripts/analyze_exp1_3seed.py    # Exp 1 analysis
python scripts/analyze_exp2.py          # Exp 2 analysis
python scripts/analyze_exp3_paired.py   # Exp 3 analysis (with/without D_min)
python scripts/analyze_exp4.py          # Exp 4 scalability analysis
```

Results are saved to `results/jpdc/` as JSON files.

### 4. Run EDGE Experiments (KaaS-Edge)

```bash
python scripts/run_edge_experiments.py --exp main --device cuda:0
python scripts/run_edge_experiments.py --all --device cuda:0
```

---

## ⚙️ DASH Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Adaptive percentile | $p$ | 0.85 | Deadline set at p-th percentile of recent latencies |
| EMA smoothing | $\alpha$ | 0.3 | Exponential moving average for deadline adaptation |
| Warmup rounds | $W$ | 3 | Fixed deadline rounds before switching to adaptive |
| Min deadline ratio | $D_{\min}/D^{(0)}$ | 0.3 | Floor to prevent deadline spiral |
| v_feasible margin | — | 0.8 | Feasibility margin for volume allocation |
| Warmup safety | — | 1.5 | Safety multiplier for warmup deadline estimation |
| Budget | $B$ | 50.0 (M=50) | Per-round total communication volume budget |
| Straggler noise | $\sigma$ | 0.5 | LogNormal noise scale for latency simulation |
| Dirichlet | $\alpha_{Dir}$ | 0.3 | Non-IID data partition concentration |

---

## 🔬 Methods

### JPDC — 6 Compared Methods

| Method | Type | Selection | Deadline |
|--------|------|-----------|----------|
| **DASH (ours)** | Async FD | Straggler-aware greedy ($\pi_i$-weighted) | Adaptive + $D_{\min}$ floor |
| Sync-Greedy | Sync FD | Greedy ($\pi_i=1$) | Wait for all |
| FedBuff-FD | Async FD | First-come buffer K=10 | None |
| Random-Async | Async FD | Random 50% | Fixed D=10s |
| Full-Async | Async FD | All devices | Adaptive |
| Sync-Full | Sync FD | All devices | Wait for all |

### Core Algorithm: RADS Two-Stage Scheduling

**Stage 1 — Water-filling:** Given device set $S$ and budget $B$, solve for optimal volumes:
$$v_i^* = \min\left(\left[\sqrt{\frac{\tilde{\rho}_i \cdot \theta_i}{\nu \cdot b_i}} - \theta_i\right]^+, v_{\max}\right)$$

where $\tilde{\rho}_i = \pi_i(D) \cdot \rho_i$ integrates straggler completion probability.

**Stage 2 — Greedy selection:** Add devices by marginal quality gain, approximation ratio $(1-1/e)/2$.

---

## 📊 JPDC Experiment Overview

| # | Experiment | Variables | Key Finding |
|---|-----------|-----------|-------------|
| 1 | Main Comparison | 6 methods, M=50 | DASH 44.47%/979s vs Sync 43.66%/3,079s (3.14× speedup) |
| 2 | Straggler Sweep | σ ∈ {0–1.5} | DASH accuracy drops only 1.2pp; speedup 3.15→3.20× |
| 3 | Policy Comparison | 13 configs ± D_min | adaptive(0.7)+D_min Pareto optimal; D_min saves +7.58pp |
| 4 | Scalability | M ∈ {20–200} | Speedup increases 2.79→3.49× with M |
| 5 | EMNIST | Cross-dataset | Validates conclusions on naturally non-IID data |
| 6 | Privacy | ρ sweep | Privacy degradation pattern consistent under async |

---

## 💻 Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Single experiment (50 rounds) | GPU with 4 GB VRAM | RTX 3090 |
| Full JPDC experiments (6 suites × 3 seeds) | GPU with 8 GB VRAM | RTX 5070 Ti / A100 |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
