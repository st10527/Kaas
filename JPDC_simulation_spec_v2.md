# Async-RADS Simulation Spec v2 — JPDC 2026
## Purpose: Copilot implementation spec for codebase modification
## Base: src/methods/kaas_edge.py + scripts/run_edge_experiments.py

---

## 1. Architecture Changes Overview

```
existing code (EDGE version, DO NOT DELETE)
├── src/methods/kaas_edge.py       → KaaSEdge class (synchronous)
├── src/scheduler/rads.py          → RADSScheduler
├── scripts/run_edge_experiments.py → 4 EDGE experiments
└── src/data/datasets.py           → CIFAR-100 + STL-10

new code (JPDC version, ADD alongside existing)
├── src/async/straggler_model.py   → latency simulation       ← NEW
├── src/async/timeout_policy.py    → 3 timeout policies        ← NEW
├── src/async/__init__.py          → module init               ← NEW
├── src/methods/async_kaas_edge.py → AsyncKaaSEdge class       ← NEW
├── src/methods/fedbuff_fd.py      → FedBuff-FD baseline       ← NEW
├── src/scheduler/rads.py          → add straggler-aware flag  ← MODIFY
├── src/data/datasets.py           → add EMNIST-ByClass loader ← MODIFY
├── scripts/run_jpdc_experiments.py → 6 JPDC experiments       ← NEW
└── config/jpdc_default.yaml       → JPDC experiment config    ← NEW
```

---

## 2. Module Spec: src/async/straggler_model.py

```python
"""
Straggler Latency Model for Async-RADS.

Each device i has a per-round latency:
  tau_i = tau_comp_i + tau_comm_i + tau_noise_i

Where:
  tau_comp_i = c_i * |D_i| * local_epochs       (deterministic, device-dependent)
  tau_comm_i = v_i * payload_per_logit / r_i     (deterministic, allocation-dependent)
  tau_noise_i ~ LogNormal(mu_noise, sigma_noise)  (stochastic)

Outcome given deadline D:
  - COMPLETE:  tau_i <= D              → v_recv = v_star
  - PARTIAL:   tau_comp_i <= D < tau_i → v_recv = floor(v_star * (D - tau_comp) / tau_comm)
  - TIMEOUT:   tau_comp_i > D          → v_recv = 0
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class DeviceLatency:
    device_id: int
    tau_comp: float       # computation time (seconds)
    tau_comm: float       # communication time (seconds)
    tau_noise: float      # random perturbation (seconds)
    tau_total: float      # = tau_comp + tau_comm + tau_noise
    outcome: str          # 'complete' | 'partial' | 'timeout'
    v_star: int           # allocated logit vectors
    v_received: int       # actual logit vectors received (0 to v_star)


class StragglerModel:
    """
    Simulates per-device latency and determines outcome.

    Parameters:
        mu_noise: float = 0.0          # log-normal mean (log scale)
        sigma_noise: float = 0.5       # log-normal std (log scale)
            - sigma=0.0: no straggler (deterministic)
            - sigma=0.3: ~10% straggler at typical deadline
            - sigma=0.5: ~20% straggler
            - sigma=1.0: ~35% straggler
            - sigma=1.5: ~50% straggler
        seed: int = 42
    """

    def __init__(self, mu_noise=0.0, sigma_noise=0.5, seed=42):
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        self.rng = np.random.RandomState(seed)

    def simulate_round(
        self,
        devices: List[Dict],    # device profiles with 'v_star', 'comp_rate', 'comm_rate'
        deadline: float          # D^(t) in seconds
    ) -> List[DeviceLatency]:
        """
        For each selected device, compute latency and determine outcome.

        Device dict expected keys:
            - device_id: int
            - v_star: int       (logit volume from RADS)
            - comp_rate: float  (seconds per logit computation)
            - comm_rate: float  (seconds per logit upload)

        Returns list of DeviceLatency objects.
        """
        results = []
        for dev in devices:
            tau_comp = dev['comp_rate'] * dev['v_star']
            tau_comm = dev['comm_rate'] * dev['v_star']
            tau_noise = self.rng.lognormal(self.mu_noise, self.sigma_noise)
            tau_total = tau_comp + tau_comm + tau_noise

            if tau_total <= deadline:
                outcome = 'complete'
                v_recv = dev['v_star']
            elif tau_comp <= deadline:
                outcome = 'partial'
                v_recv = int(dev['v_star'] * (deadline - tau_comp) / tau_comm)
                v_recv = max(0, min(v_recv, dev['v_star']))
            else:
                outcome = 'timeout'
                v_recv = 0

            results.append(DeviceLatency(
                device_id=dev['device_id'],
                tau_comp=tau_comp,
                tau_comm=tau_comm,
                tau_noise=tau_noise,
                tau_total=tau_total,
                outcome=outcome,
                v_star=dev['v_star'],
                v_received=v_recv,
            ))
        return results

    def compute_completion_probability(
        self,
        device: Dict,
        deadline: float
    ) -> float:
        """
        Compute Pr[tau_i <= D] for straggler-aware scheduling.

        pi_i(D) = Pr[tau_noise <= D - tau_comp - tau_comm]
                = CDF_LogNormal(D - tau_comp - tau_comm; mu, sigma)

        Used by RADSScheduler when straggler_aware=True.
        """
        from scipy.stats import lognorm
        tau_det = device['comp_rate'] * device['v_star'] + \
                  device['comm_rate'] * device['v_star']
        slack = deadline - tau_det
        if slack <= 0:
            return 0.0
        # scipy lognorm: s=sigma, scale=exp(mu)
        return float(lognorm.cdf(slack, s=self.sigma_noise,
                                 scale=np.exp(self.mu_noise)))
```

### Design decisions:
- comp_rate per device drawn from 3-tier distribution (matching EDGE's cost tiers)
- comm_rate per device drawn from Rayleigh fading model
- sigma_noise is the PRIMARY knob for straggler severity
- `compute_completion_probability()` is the interface for straggler-aware RADS (Sec 4.2)
- All randomness seeded for reproducibility

---

## 3. Module Spec: src/async/timeout_policy.py

```python
"""
Three Timeout Policies for Async-RADS.

All policies output a deadline D^(t) for round t.
"""
from typing import List, Optional
import numpy as np


class TimeoutPolicy:
    """Base class for timeout policies."""
    def get_deadline(self, round_idx: int, history: List[List[float]]) -> float:
        raise NotImplementedError

    @property
    def accepts_partial(self) -> bool:
        """Whether this policy accepts partial logit uploads."""
        return False


class FixedDeadlinePolicy(TimeoutPolicy):
    """
    Policy A: D^(t) = D_0 for all t.

    Simple and predictable. Good baseline.
    """
    def __init__(self, D_0: float = 10.0):
        self.D_0 = D_0

    def get_deadline(self, round_idx: int, history: List[List[float]]) -> float:
        return self.D_0


class AdaptiveDeadlinePolicy(TimeoutPolicy):
    """
    Policy B: D^(t) = percentile_p of round (t-1)'s latency distribution.

    Parameters:
        percentile: float = 0.7     # target completion rate (~70%)
        warmup_rounds: int = 3      # use fixed deadline for first rounds
        D_default: float = 10.0     # default deadline during warmup
        ema_alpha: float = 0.3      # exponential moving average smoothing
    """
    def __init__(self, percentile=0.7, warmup_rounds=3,
                 D_default=10.0, ema_alpha=0.3):
        self.percentile = percentile
        self.warmup_rounds = warmup_rounds
        self.D_default = D_default
        self.ema_alpha = ema_alpha
        self._ema_deadline = None

    def get_deadline(self, round_idx: int, history: List[List[float]]) -> float:
        if round_idx < self.warmup_rounds or len(history) == 0:
            return self.D_default

        prev_latencies = history[-1]
        raw = np.percentile(prev_latencies, self.percentile * 100)

        # EMA smoothing to avoid oscillation
        if self._ema_deadline is None:
            self._ema_deadline = raw
        else:
            self._ema_deadline = (self.ema_alpha * raw +
                                  (1 - self.ema_alpha) * self._ema_deadline)
        return self._ema_deadline


class PartialAcceptPolicy(AdaptiveDeadlinePolicy):
    """
    Policy C: Same adaptive deadline as B, but accepts partial logits.

    Key difference from A & B:
    - A & B: devices with outcome='partial' are treated as timeout (discarded)
    - C: partial devices contribute their v_received logits at reduced quality

    The accept_partial flag is checked by AsyncKaaSEdge during aggregation.
    """
    def __init__(self, percentile=0.7, **kwargs):
        super().__init__(percentile=percentile, **kwargs)

    @property
    def accepts_partial(self) -> bool:
        return True


def create_timeout_policy(name: str, **kwargs) -> TimeoutPolicy:
    """Factory function for creating timeout policies from config."""
    policies = {
        'fixed': FixedDeadlinePolicy,
        'adaptive': AdaptiveDeadlinePolicy,
        'partial': PartialAcceptPolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Choose from {list(policies.keys())}")
    return policies[name](**kwargs)
```

---

## 4. Module Spec: src/methods/fedbuff_fd.py

> **原版 spec 缺少此 baseline。Experiment 1 列了 FedBuff-FD 但沒有實作規格。**

```python
"""
FedBuff-FD: Buffered Asynchronous Federated Distillation.

Adapted from FedBuff (Nguyen et al., AISTATS 2022) for FD setting.

Original FedBuff: server collects K async parameter updates into a buffer,
then performs one aggregation step. This decouples aggregation frequency
from individual device speed.

FedBuff-FD adaptation:
- Instead of parameter updates, devices upload logit vectors asynchronously
- Server buffers K logit uploads (regardless of which devices)
- When buffer is full, aggregate via distillation on public set
- Devices use the LATEST server model when they start (may be stale)

Key parameters:
    buffer_size: int = 10      # K: number of logit uploads to buffer
    max_wall_time: float       # total wall-clock time budget
"""

class FedBuffFD(FederatedMethod):
    """
    Protocol:
    1. All M devices begin local training asynchronously
    2. Each device computes logits at its own pace (affected by straggler model)
    3. As each device finishes, it uploads logits to the server buffer
    4. When buffer has K entries:
       a. Aggregate logits (weighted average on public set)
       b. Distill to server model
       c. Clear buffer
       d. Broadcast new server model to all devices
    5. Repeat until wall_clock_time >= max_wall_time

    Device selection: NONE — all devices participate, straggler-tolerant by design.
    Volume allocation: UNIFORM — every device uploads v_fixed logits.

    The "unfairness" of FedBuff-FD is that fast devices contribute more
    frequently than slow ones, biasing the model toward fast-device data.
    This is the main disadvantage vs. RADS-based selection.
    """

    def __init__(self, config):
        super().__init__(config)
        self.buffer_size = config.get('buffer_size', 10)
        self.buffer = []   # List of (device_id, logits, quality_weight)

    def run_round(self, ...):
        """
        Note: "round" for FedBuff = one buffer-fill-and-aggregate cycle,
        not one synchronous communication round.

        Simulated via StragglerModel:
        - Sort devices by tau_total (ascending = fastest first)
        - First K devices to finish fill the buffer
        - Remaining devices: their computation is "wasted" for this cycle
        """
        pass
```

### Design notes:
- FedBuff-FD 的 wall-clock 比較公平，因為 fast devices 不用等 slow devices
- 但 FedBuff-FD 沒有 quality-aware scheduling → data-biased toward fast devices
- 這是 Async-RADS 的主要優勢：straggler-aware 但仍 quality-aware

---

## 5. Modification Spec: src/scheduler/rads.py — Add Straggler-Aware Flag

> **原版 spec 說 RADS is UNCHANGED。修改版：RADS 的 water-filling (Stage 1) 不動，greedy selection (Stage 2) 加入 completion probability。**

```python
# In RADSScheduler, modify the greedy selection step:

class RADSScheduler:
    def __init__(self, ..., straggler_aware=False, straggler_model=None,
                 deadline=None):
        """
        New parameters (only used when straggler_aware=True):
            straggler_aware: bool   — whether to use expected quality
            straggler_model: StragglerModel  — for computing pi_i(D)
            deadline: float         — current round deadline D^(t)
        """
        self.straggler_aware = straggler_aware
        self.straggler_model = straggler_model
        self.deadline = deadline

    def _greedy_select(self, candidates, budget):
        """
        EDGE version (straggler_aware=False):
            marginal_gain(i) = rho_i * q_i(v_i*)

        JPDC version (straggler_aware=True):
            marginal_gain(i) = rho_i * q_i(v_i*) * pi_i(D)

        where pi_i(D) = Pr[tau_i <= D] from straggler_model.

        Submodularity is preserved because pi_i(D) is a per-device
        constant (given D and v_i*), so the marginal gain structure
        is unchanged. Same (1-1/e)/2 approximation applies.
        """
        selected = []
        remaining_budget = budget

        while remaining_budget > 0 and candidates:
            best_gain = -1
            best_idx = -1

            for idx, cand in enumerate(candidates):
                gain = cand['rho'] * cand['quality']

                if self.straggler_aware and self.straggler_model:
                    pi_i = self.straggler_model.compute_completion_probability(
                        device=cand, deadline=self.deadline
                    )
                    gain *= pi_i    # expected quality

                cost = cand['cost']
                if cost <= remaining_budget and gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx == -1:
                break

            chosen = candidates.pop(best_idx)
            selected.append(chosen)
            remaining_budget -= chosen['cost']

        return selected
```

### Key point:
- Stage 1 (water-filling) 完全不動
- Stage 2 (greedy) 只加一行 `gain *= pi_i`
- 改動量極小但效果顯著：自然偏好 reliable device，但 submodular diversity 會平衡

---

## 6. Module Spec: src/methods/async_kaas_edge.py

```python
"""
AsyncKaaSEdge: Asynchronous variant of KaaSEdge.

Differences from KaaSEdge (synchronous):
1. After RADS scheduling, applies StragglerModel to simulate latencies
2. Uses TimeoutPolicy to determine deadline D^(t)
3. Only collects logits from complete devices (+partial if policy allows)
4. Uses v_received (not v_star) for quality weighting
5. Tracks wall-clock time (cumulative deadline sum)
6. Stores latency history for adaptive policy

The RADS scheduling (water-filling + greedy) is reused.
Local training, logit computation, and KL distillation are identical.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AsyncKaaSEdgeConfig:
    # === Inherited from KaaSEdge ===
    budget_per_round: float = 50.0
    n_rounds: int = 50
    local_epochs: int = 5
    public_ratio: float = 0.2
    # === Async-specific ===
    timeout_policy: str = 'adaptive'       # 'fixed' | 'adaptive' | 'partial'
    fixed_deadline: float = 10.0           # for fixed policy
    adaptive_percentile: float = 0.7       # for adaptive/partial policy
    warmup_rounds: int = 3                 # for adaptive/partial policy
    sigma_noise: float = 0.5              # straggler severity
    straggler_aware: bool = True           # use pi_i in greedy selection


class AsyncKaaSEdge(FederatedMethod):
    """
    Round protocol:
    1. Get deadline: D^(t) = timeout_policy.get_deadline(t, history)
    2. RADS scheduling with straggler_aware flag:
       - Stage 1: water-filling → {v_i*}  (unchanged)
       - Stage 2: greedy with gain *= pi_i(D)  (straggler-aware)
       → selected set S^(t), allocations {v_i*}
    3. Simulate latencies: StragglerModel.simulate_round(S, D)
    4. Filter by outcome:
       - complete → use v_star logits
       - partial  → use v_received logits (only if policy.accepts_partial)
       - timeout  → discard
    5. Collect logits from surviving devices on public set
    6. Quality-weighted aggregation:
       w_i = rho_i * q_i(v_i^recv) / sum(...)
    7. Distill to server model (KL divergence, identical to sync)
    8. Update tracking:
       - wall_clock_time += D^(t)
       - latency_history.append([tau_i for all i in S])
       - Record: n_complete, n_partial, n_timeout per round
    """

    def __init__(self, config: AsyncKaaSEdgeConfig, ...):
        super().__init__(config)

        # Straggler model
        self.straggler_model = StragglerModel(
            sigma_noise=config.sigma_noise,
            seed=config.seed,
        )

        # Timeout policy
        self.timeout_policy = create_timeout_policy(
            name=config.timeout_policy,
            D_0=config.fixed_deadline,
            percentile=config.adaptive_percentile,
            warmup_rounds=config.warmup_rounds,
        )

        # RADS scheduler (reuse existing, add straggler flag)
        self.scheduler = RADSScheduler(
            budget=config.budget_per_round,
            straggler_aware=config.straggler_aware,
            straggler_model=self.straggler_model,
        )

        # Tracking
        self.wall_clock_time = 0.0
        self.latency_history = []
        self.round_stats = []    # per-round: {n_complete, n_partial, n_timeout, deadline}

    def run_round(self, round_idx, devices, client_loaders,
                  public_loader, test_loader=None):
        # Step 1: Deadline
        deadline = self.timeout_policy.get_deadline(round_idx, self.latency_history)
        self.scheduler.deadline = deadline

        # Step 2: RADS scheduling (with straggler-aware if enabled)
        sched_result = self.scheduler.schedule(devices)

        # Step 3: Simulate latencies
        latencies = self.straggler_model.simulate_round(
            devices=sched_result.allocations,
            deadline=deadline,
        )

        # Step 4: Filter by outcome
        surviving = []
        for lat in latencies:
            if lat.outcome == 'complete':
                surviving.append((lat.device_id, lat.v_star))
            elif lat.outcome == 'partial' and self.timeout_policy.accepts_partial:
                surviving.append((lat.device_id, lat.v_received))
            # else: timeout → skip

        # Step 5-7: Collect logits + aggregate + distill
        # (reuse KaaSEdge's _collect_and_distill, but pass v_received instead of v_star)
        if len(surviving) > 0:
            self._collect_and_distill(surviving, client_loaders, public_loader)

        # Step 8: Update tracking
        self.wall_clock_time += deadline
        self.latency_history.append([lat.tau_total for lat in latencies])
        self.round_stats.append({
            'round': round_idx,
            'deadline': deadline,
            'n_selected': len(latencies),
            'n_complete': sum(1 for l in latencies if l.outcome == 'complete'),
            'n_partial': sum(1 for l in latencies if l.outcome == 'partial'),
            'n_timeout': sum(1 for l in latencies if l.outcome == 'timeout'),
            'wall_clock_time': self.wall_clock_time,
        })
```

---

## 7. Data Spec: EMNIST-ByClass Loader

> **原版用 FEMNIST (LEAF benchmark)。改用 torchvision 內建的 EMNIST-ByClass。**
>
> **原因**：LEAF 需要另外下載 + 解析 JSON per user，reviewers 很難 reproduce。
> EMNIST-ByClass 一行 torchvision 就能載入，且包含 writer ID 可做 natural non-IID partition。

```python
# Add to src/data/datasets.py

def load_emnist_byclass(root='./data', n_devices=200, n_public=5000,
                         seed=42, img_size=32):
    """
    Load EMNIST-ByClass and partition by writer for natural non-IID.

    EMNIST-ByClass:
    - 62 classes (0-9, A-Z, a-z)
    - 814,255 total images
    - 28x28 grayscale → resize to img_size×img_size

    Partition strategy:
    - Use writer_id metadata to group samples by writer
    - Select top n_devices writers with most samples → private sets
    - Hold out n_public samples from remaining writers → public reference set
    - Standard test split → test set

    Returns:
        private_sets: Dict[int, Dataset]  # device_id → dataset
        public_set: Dataset
        test_set: Dataset
    """
    from torchvision.datasets import EMNIST
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale → 3ch for CNN compatibility
    ])

    train_ds = EMNIST(root, split='byclass', train=True,
                      download=True, transform=transform)
    test_ds  = EMNIST(root, split='byclass', train=False,
                      download=True, transform=transform)

    # Group by writer (EMNIST doesn't expose writer_id directly;
    # use Dirichlet partition on class labels as proxy for natural non-IID,
    # OR download the official mapping from NIST if writer-level split is needed)
    #
    # Pragmatic approach: Dirichlet with very low alpha (0.1) to mimic
    # writer-level heterogeneity. This is acceptable since the point of
    # EMNIST is to test on a different domain, not to prove writer-level split.

    # ... partition logic (reuse existing partition.py with alpha=0.1) ...

    return private_sets, public_set, test_set
```

### Note on writer-level partition:
- torchvision EMNIST 不直接提供 writer ID metadata
- 兩個選擇：
  1. 從 NIST 官網下載 writer mapping（複雜但 ground-truth）
  2. 用 Dirichlet α=0.1 模擬 extreme non-IID（簡單且 sufficient for our purpose）
- 建議用選擇 2，在 paper 中說明 "we use Dirichlet α=0.1 to simulate naturally heterogeneous distributions"

---

## 8. Experiment Plan: scripts/run_jpdc_experiments.py

```
Experiment 1: Main Comparison — Sync vs Async (Fig. 2, 3, Table I)
  Dataset:    CIFAR-100, M=50
  Straggler:  sigma=0.5 (~20% straggler rate)
  Methods:    6 methods (see outline Sec 5.1)
    - Async-RADS (adaptive, straggler-aware)
    - Sync-RADS (EDGE version, wait all)
    - FedBuff-FD (buffer_size=10, no selection)
    - Random-Async (random selection + fixed timeout)
    - Full-Async (all devices + adaptive timeout)
    - Sync-Full (all devices, synchronous)
  Metrics:    accuracy vs round, accuracy vs wall-clock time
  Seeds:      3 (42, 123, 456)
  → Produces: Fig 2 (acc vs round), Fig 3 (acc vs wall-clock), Table I

Experiment 2: Straggler Severity Sweep (Fig. 4, 5)
  Dataset:    CIFAR-100, M=50
  Methods:    Async-RADS vs Sync-RADS vs FedBuff-FD
  Straggler:  sigma ∈ {0.0, 0.3, 0.5, 1.0, 1.5}
              (≈ 0%, 10%, 20%, 35%, 50% straggler rate)
  Metrics:    final accuracy, total wall-clock time
  → Produces: Fig 4 (accuracy vs straggler ratio), Fig 5 (wall-clock vs straggler)

Experiment 3: Timeout Policy Comparison (Fig. 6)
  Dataset:    CIFAR-100, M=50
  Straggler:  sigma=1.0 (harsh)
  Policies:   Fixed(D=5,10,20), Adaptive(p=0.5,0.7,0.9), Partial(p=0.7)
  → Produces: Fig 6 (accuracy vs wall-clock Pareto front per policy)

Experiment 4: Scalability (Fig. 7, 8)
  Dataset:    CIFAR-100
  Methods:    Async-RADS vs Sync-RADS vs FedBuff-FD
  Scale:      M ∈ {20, 50, 100, 200}
  Straggler:  sigma=0.5
  → Produces: Fig 7 (accuracy vs M at fixed wall-clock), Fig 8 (round time vs M)

Experiment 5: Cross-Dataset — EMNIST (Fig. 9)
  Dataset:    EMNIST-ByClass, M ∈ {50, 200}
  Methods:    Async-RADS, Sync-RADS, FedBuff-FD
  Straggler:  sigma=0.5
  → Produces: Fig 9 (EMNIST accuracy vs wall-clock)

Experiment 6: Privacy under Async (Fig. 10)
  Dataset:    CIFAR-100, M=50
  Methods:    Async-RADS (adaptive)
  Privacy:    rho ∈ {1.0, 0.8, 0.55, 0.1}
  Straggler:  sigma=0.5
  → Produces: Fig 10 (privacy ρ vs accuracy under async)
```

### Figures & Tables summary:
| # | Type | Content |
|---|---|---|
| Fig 1 | Diagram | System architecture |
| Fig 2 | Line plot | Accuracy vs round (6 methods) |
| Fig 3 | Line plot | Accuracy vs wall-clock time (6 methods) ← 殺手圖 |
| Fig 4 | Line plot | Accuracy vs straggler ratio |
| Fig 5 | Line plot | Wall-clock time vs straggler ratio |
| Fig 6 | Scatter/Pareto | Policy comparison: accuracy-time trade-off |
| Fig 7 | Bar chart | Accuracy vs M at fixed wall-clock budget |
| Fig 8 | Line plot | Round time vs M |
| Fig 9 | Line plot | EMNIST accuracy vs wall-clock |
| Fig 10 | Line plot | Privacy sweep under async |
| Table I | Summary | 6 methods × {accuracy, wall-clock, comm cost} |

### Expected runtime: ~3-5 hours on RTX 5070 Ti
- Exp 1: ~45 min (6 methods × 50 rounds × 3 seeds)
- Exp 2: ~40 min (3 methods × 5 sigma × 3 seeds)
- Exp 3: ~30 min (7 configs × 50 rounds × 3 seeds)
- Exp 4: ~60 min (3 methods × 4 scales × 3 seeds, M=200 is slow)
- Exp 5: ~40 min (3 methods × 2 scales × 3 seeds)
- Exp 6: ~15 min (1 method × 4 rho × 3 seeds)

---

## 9. Config Spec: config/jpdc_default.yaml

```yaml
# JPDC Async-RADS default configuration
experiment:
  n_rounds: 50
  seeds: [42, 123, 456]
  device: "cuda:0"

dataset:
  name: "cifar100"        # or "emnist_byclass"
  n_classes: 100           # 62 for EMNIST
  public_ratio: 0.2
  dirichlet_alpha: 0.5     # 0.1 for EMNIST

devices:
  n_devices: 50
  cost_distribution: "lognormal"
  cost_mean: 1.0
  cost_std: 0.5

model:
  name: "lightweight_cnn"
  n_channels: 3
  img_size: 32

scheduler:
  budget_per_round: 50.0
  straggler_aware: true    # JPDC: enable completion probability in greedy

async:
  timeout_policy: "adaptive"    # 'fixed' | 'adaptive' | 'partial'
  fixed_deadline: 10.0
  adaptive_percentile: 0.7
  warmup_rounds: 3
  sigma_noise: 0.5              # straggler severity

training:
  local_epochs: 5
  learning_rate: 0.01
  batch_size: 64
  temperature: 3.0              # KD temperature

privacy:
  rho_range: [0.1, 1.0]

fedbuff:
  buffer_size: 10
  v_fixed: 100                  # uniform volume for all devices
```

---

## 10. Implementation Priority Order

```
Week 1: Core async infrastructure
  Day 1:   src/async/straggler_model.py + unit tests
  Day 2:   src/async/timeout_policy.py + unit tests
  Day 3:   Modify src/scheduler/rads.py (add straggler_aware flag)
  Day 4-5: src/methods/async_kaas_edge.py (wrap existing KaaSEdge)

Week 2: Baselines + Data
  Day 1-2: src/methods/fedbuff_fd.py + unit tests
  Day 3:   EMNIST-ByClass loader in src/data/datasets.py
  Day 4-5: scripts/run_jpdc_experiments.py (wire everything up)

Week 3: Run experiments + debug
  Day 1-2: Exp 1-3 (main comparison, straggler sweep, policy)
  Day 3-4: Exp 4-6 (scalability, EMNIST, privacy)
  Day 5:   Plot generation, sanity checks

Week 4: Paper writing
  (parallel with experiment re-runs if needed)
```

---

## 11. Testing Strategy

```python
# tests/test_straggler.py
def test_deterministic_no_straggler():
    """sigma=0 → all devices complete on time."""

def test_all_timeout():
    """deadline=0 → all devices timeout."""

def test_partial_logit_calculation():
    """Verify v_received formula for partial devices."""

def test_completion_probability():
    """Verify pi_i(D) matches scipy lognorm CDF."""

# tests/test_timeout_policy.py
def test_fixed_deadline_constant():
    """Fixed policy returns same D for all rounds."""

def test_adaptive_warmup():
    """Adaptive returns D_default during warmup."""

def test_adaptive_tracks_history():
    """After warmup, uses percentile of previous round."""

def test_partial_accept_flag():
    """PartialAcceptPolicy.accepts_partial == True."""

# tests/test_async_kaas.py
def test_sync_equivalent():
    """sigma=0, fixed policy → results identical to sync KaaSEdge."""

def test_straggler_reduces_participants():
    """sigma=1.0 → fewer devices contribute than selected."""

def test_wall_clock_accumulates():
    """wall_clock_time = sum of deadlines over rounds."""

# tests/test_fedbuff.py
def test_buffer_fills():
    """After simulating round, buffer has <= K entries."""

def test_fast_devices_dominate():
    """With straggler, fast devices appear more often in buffer."""
```
