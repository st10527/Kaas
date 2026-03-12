# Artifact Description: KaaS-Edge

## Knowledge-as-a-Service for Edge Intelligence: Resource-Aware Distillation Scheduling

---

## 1. Access & Links

The complete source code, configuration files, and experiment scripts are provided in this artifact package. The repository is also available at:

- **Repository:** [Anonymous GitHub link to be provided]

---

## 2. User Guide

### 2.1 Installation

```bash
# Clone the repository
git clone <REPO_URL>
cd KaaS-Edge

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS

# Install all dependencies
pip install -r requirements.txt
```

### 2.2 Reproducing Main Results (Table I & Figures 2–3)

Run the main comparison experiment across three random seeds:

```bash
python scripts/run_edge_experiments.py --exp main --device cuda:0
```

Or use parallel execution for faster runs:

```bash
bash parallel_run.sh main cuda:0
python merge_seed_results.py
```

### 2.3 Reproducing Sensitivity Analysis (Figures 4–5)

**Budget sensitivity (Figure 4):**

```bash
python scripts/run_edge_experiments.py --exp budget --device cuda:0
```

**Device scalability:**

```bash
python scripts/run_edge_experiments.py --exp scale --device cuda:0
```

**Privacy impact (Figure 5):**

```bash
python scripts/run_edge_experiments.py --exp privacy --device cuda:0
```

### 2.4 Running All Experiments at Once

```bash
python scripts/run_edge_experiments.py --all --device cuda:0
```

### 2.5 Generating Figures

After experiments complete:

```bash
python plot_edge_figures.py
```

Figures are saved to the `figures/` directory.

### 2.6 Quick Verification (CPU-Only)

For reviewers without GPU access, a quick smoke-test validates basic executability:

```bash
python scripts/run_edge_experiments.py --exp main --quick
```

This runs with reduced sample sizes on CPU and completes in approximately 5–10 minutes.

---

## 3. Environment

### 3.1 Software Dependencies

| Package        | Version  |
|----------------|----------|
| Python         | ≥ 3.8   |
| PyTorch        | ≥ 2.0.0 |
| torchvision    | ≥ 0.15.0|
| numpy          | ≥ 1.24.0|
| scipy          | ≥ 1.10.0|
| scikit-learn   | ≥ 1.3.0 |
| matplotlib     | ≥ 3.7.0 |
| pyyaml         | ≥ 6.0   |
| tqdm           | ≥ 4.65.0|

A complete list is provided in `requirements.txt`.

### 3.2 Hardware Used for Paper Results

| Component | Specification |
|-----------|---------------|
| GPU       | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| CPU       | Apple M-series / Intel i7 equivalent |
| RAM       | ≥ 16 GB |
| Storage   | ≥ 5 GB free (for datasets + results) |

### 3.3 Minimum Requirements

| Task | Minimum Hardware |
|------|-----------------|
| Quick test (`--quick`) | CPU only, 8 GB RAM |
| Single experiment | GPU with ≥ 4 GB VRAM |
| Full reproduction | GPU with ≥ 8 GB VRAM |

---

## 4. Data & Methods

### 4.1 Dataset

- **CIFAR-100** (Krizhevsky, 2009): Automatically downloaded by `torchvision` on first run.
  - 50,000 training images (100 classes, 32×32 pixels)
  - 10,000 test images
  - The training set is split into private data (40k) and public reference data (10k) using a fixed random seed for reproducibility.

### 4.2 Data Partitioning

Non-IID data distribution across devices is created using Dirichlet partitioning with concentration parameter α = 0.5 (configurable in `config/edge_default.yaml`).

### 4.3 Device Simulation

Heterogeneous edge devices are simulated with log-normal cost distributions. Device profiles include:
- **Cost (b_i):** marginal per-sample upload cost
- **Privacy (ρ_i):** privacy degradation factor ∈ (0, 1]
- **Capacity (θ_i):** half-saturation constant

Device generation is deterministic given a random seed, ensuring full reproducibility.

---

## 5. Assessment: Interpreting Results

### 5.1 Output Format

Each experiment produces a JSON file in `results/edge/` containing per-round metrics:

```json
{
  "method": "KaaS-Edge",
  "accuracy_history": [0.12, 0.25, ...],
  "final_accuracy": 0.487,
  "total_cost": 7.82,
  "n_rounds": 50
}
```

### 5.2 Key Metrics

| Metric | Description | Expected Range |
|--------|-------------|----------------|
| `final_accuracy` | Test accuracy after 50 rounds | 45–55% (CIFAR-100) |
| `total_cost` | Cumulative resource expenditure | ≤ budget × n_rounds |
| `n_participants` | Devices selected per round | 5–15 (with B=8.0) |
| `total_quality` | Sum of quality contributions | Proportional to accuracy |

### 5.3 Expected Outcomes

The experiments should demonstrate:

1. **Main comparison (Table I, Figure 2):** KaaS-Edge achieves higher accuracy than baseline methods under the same budget constraint, with significantly lower communication cost than full-participation methods.

2. **Budget sensitivity (Figure 4):** Accuracy increases with budget B following a concave curve (diminishing returns).

3. **Device scalability:** Quality improves with more devices due to the submodular greedy selection exploiting device diversity.

4. **Privacy impact (Figure 5):** Higher privacy (lower ρ) degrades accuracy, but KaaS-Edge's quality-weighted aggregation mitigates the impact compared to uniform baselines.

### 5.4 Variance

All experiments use 3 random seeds (42, 123, 456). Results are reported as mean ± standard deviation. Minor numerical differences across hardware are expected but should not affect relative rankings.

---

## 6. System Requirements

### 6.1 Full Reproduction

- **GPU:** NVIDIA GPU with ≥ 8 GB VRAM and CUDA support
- **Time:** ~2–3 hours for all experiments on a modern GPU
- **Disk:** ~2 GB for CIFAR-100 download + results

### 6.2 Simplified Verification

For reviewers with limited resources:

```bash
# CPU-only quick test (~5–10 minutes)
python scripts/run_edge_experiments.py --exp main --quick
```

This validates that:
- All dependencies install correctly
- The RADS scheduler produces valid allocations
- Training loop completes without errors
- Results are saved in the expected format

---

## 7. File Inventory

| File/Directory | Description |
|----------------|-------------|
| `src/scheduler/rads.py` | Core RADS algorithm (Proposition 1 + Theorem 1) |
| `src/methods/kaas_edge.py` | KaaS-Edge method implementation |
| `src/methods/fed*.py` | Baseline method implementations |
| `src/privacy/ldp.py` | Local differential privacy mechanisms |
| `scripts/run_edge_experiments.py` | Main experiment runner |
| `config/edge_default.yaml` | Default configuration |
| `parallel_run.sh` | Multi-seed parallel execution |
| `merge_seed_results.py` | Result aggregation across seeds |
| `plot_edge_figures.py` | Figure generation script |
| `requirements.txt` | Python package dependencies |
