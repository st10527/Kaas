# Async-RADS Simulation Spec — JPDC 2026
## Purpose: Feed this to GitHub Copilot for codebase modification
## Base: paid-fd-main/src/methods/kaas_edge.py + run_edge_experiments.py

---

## 1. Architecture Changes Overview

```
existing code (EDGE version)
├── kaas_edge.py          → KaaSEdge class (synchronous)
├── run_edge_experiments.py → 4 experiments
├── scheduler/rads.py      → RADSScheduler
└── data/datasets.py       → CIFAR-100 + STL-10

new code (JPDC version, ADD don't delete)
├── async_kaas_edge.py     → AsyncKaaSEdge class ← NEW
├── straggler_model.py     → latency simulation  ← NEW
├── timeout_policy.py      → 3 timeout policies  ← NEW
├── run_jpdc_experiments.py → 6 experiments       ← NEW
└── data/datasets.py       → add FEMNIST loader   ← MODIFY
```

---

## 2. Module Spec: straggler_model.py

```python
"""
Straggler Latency Model for Async-RADS.

Each device i has a per-round latency:
  tau_i = tau_comp_i + tau_comm_i + tau_noise_i

Where:
  tau_comp_i = c_i * |D_i| * local_epochs  (deterministic)
  tau_comm_i = v_i * payload_per_logit / channel_rate_i  (deterministic)
  tau_noise_i ~ LogNormal(mu_noise, sigma_noise)  (stochastic)

A device's outcome depends on deadline D:
  - COMPLETE:  tau_i <= D               → receives full v_i* logits
  - PARTIAL:   tau_comp_i <= D < tau_i  → receives v_i_recv logits
  - TIMEOUT:   tau_comp_i > D           → receives nothing
"""

@dataclass
class DeviceLatency:
    device_id: int
    tau_comp: float      # computation time (seconds)
    tau_comm: float      # communication time (seconds)
    tau_noise: float     # random perturbation (seconds)
    tau_total: float     # tau_comp + tau_comm + tau_noise
    outcome: str         # 'complete' | 'partial' | 'timeout'
    v_received: int      # actual logit vectors received (0 to v_i*)

class StragglerModel:
    """
    Parameters:
        mu_noise: float = 0.0        # log-normal mean (log scale)
        sigma_noise: float = 0.5     # log-normal std (log scale)
            - sigma=0.5: ~20% straggler at tight deadline
            - sigma=1.0: ~35% straggler
            - sigma=1.5: ~50% straggler
        comp_rate: dict              # device_id -> seconds per logit computation
        comm_rate: dict              # device_id -> seconds per logit upload
        seed: int = 42
    """

    def simulate_round(
        self,
        devices: List[Dict],         # device profiles with v_star allocations
        deadline: float              # D^(t) in seconds
    ) -> List[DeviceLatency]:
        """
        For each selected device, compute latency and determine outcome.

        For PARTIAL devices:
          v_received = int(v_star * (deadline - tau_comp) / tau_comm)
          (proportional to how much comm time they had before deadline)
        """
        pass
```

### Key design decisions:
- comp_rate per device drawn from 3-tier distribution (matching EDGE's cost tiers)
- comm_rate per device drawn from Rayleigh fading (matching EDGE's channel model)
- sigma_noise is the PRIMARY knob to control straggler ratio
- All randomness seeded for reproducibility

---

## 3. Module Spec: timeout_policy.py

```python
"""
Three Timeout Policies for Async-RADS.

All policies output a deadline D^(t) for round t.
"""

class FixedDeadlinePolicy:
    """Policy A: D^(t) = D_0 for all t."""
    def __init__(self, D_0: float):
        self.D_0 = D_0

    def get_deadline(self, round_idx: int, history: List) -> float:
        return self.D_0


class AdaptiveDeadlinePolicy:
    """
    Policy B: D^(t) = percentile_p of round (t-1)'s latency distribution.

    Parameters:
        percentile: float = 0.7   # 70th percentile → ~30% straggler
        warmup_rounds: int = 3    # use fixed deadline for first 3 rounds
        D_default: float = 10.0   # default deadline during warmup
    """
    def get_deadline(self, round_idx: int, history: List[List[float]]) -> float:
        if round_idx < self.warmup_rounds:
            return self.D_default
        prev_latencies = history[round_idx - 1]
        return np.percentile(prev_latencies, self.percentile * 100)


class PartialAcceptPolicy(AdaptiveDeadlinePolicy):
    """
    Policy C: Same adaptive deadline as B, but also accepts partial logits.

    The key difference is in how AsyncKaaSEdge handles the PARTIAL outcome:
    - Policy A & B: discard partial (treat as timeout)
    - Policy C: keep partial logits with reduced quality weight
    """
    def __init__(self, percentile=0.7, accept_partial=True, **kwargs):
        super().__init__(percentile=percentile, **kwargs)
        self.accept_partial = accept_partial
```

---

## 4. Module Spec: async_kaas_edge.py

```python
"""
AsyncKaaSEdge: Asynchronous variant of KaaSEdge.

Differences from KaaSEdge (synchronous):
1. After RADS scheduling, applies straggler_model to simulate latencies
2. Uses timeout_policy to determine deadline
3. Collects only complete + partial logits
4. Applies staleness-aware quality weighting
5. Tracks wall-clock time (cumulative deadline sum)

The RADS scheduling itself is UNCHANGED — we reuse RADSScheduler.
The only difference is what happens AFTER scheduling.
"""

@dataclass
class AsyncKaaSEdgeConfig(KaaSEdgeConfig):
    # Inherit all KaaSEdge parameters, add:
    timeout_policy: str = 'adaptive'    # 'fixed' | 'adaptive' | 'partial'
    fixed_deadline: float = 10.0        # for fixed policy
    adaptive_percentile: float = 0.7    # for adaptive/partial policy
    staleness_lambda: float = 0.1       # staleness discount factor
    sigma_noise: float = 0.5            # straggler severity

class AsyncKaaSEdge(FederatedMethod):
    """
    Protocol per round:
    1. RADS scheduling (identical to sync version) → S, {v_i*}
    2. Simulate device latencies via StragglerModel
    3. Apply TimeoutPolicy → deadline D^(t)
    4. Determine outcomes: complete / partial / timeout
    5. Collect logits from complete + partial devices
    6. Staleness-aware aggregation:
       weight_i = rho_i * q_tilde_i
       where q_tilde_i = q_i(v_i_recv) / (1 + lambda * staleness_i)
    7. Distill to server model
    8. Record wall_clock_time += D^(t)
    """

    def run_round(self, round_idx, devices, client_loaders,
                  public_loader, test_loader=None):
        # Step 1: RADS (reuse existing scheduler)
        sched_result = self.scheduler.schedule(device_dicts)

        # Step 2: Simulate latencies
        latencies = self.straggler_model.simulate_round(
            devices=sched_result.allocations,
            deadline=self.timeout_policy.get_deadline(round_idx, self.latency_history)
        )

        # Step 3-5: Filter by outcome
        for lat in latencies:
            if lat.outcome == 'complete':
                # compute full logits, use v_star
                pass
            elif lat.outcome == 'partial' and self.config.timeout_policy == 'partial':
                # compute logits but truncate to v_received
                pass
            else:  # timeout
                # skip this device entirely
                pass

        # Step 6: Staleness-aware aggregation
        # staleness_i = 0 for all devices in sync case (no model aging here)
        # For JPDC: staleness_i = 0 (same round), but partial logits get
        # quality discount: q_tilde = q(v_recv) which is already < q(v_star)
        # The staleness parameter is reserved for future multi-round extension

        # Step 7: Distill (identical to sync)
        # Step 8: Track wall-clock
        self.wall_clock_time += deadline
```

### Critical note on staleness_i:
In this JPDC version, staleness = 0 for all devices (everyone uses the
same round's model). The λ·staleness discount is architecturally present
but only activates for partial logits through reduced v_received → reduced
q_i(v_received). This is honest and clean — we don't fake multi-round
staleness that doesn't exist in our protocol.

For the paper, frame it as: "the staleness parameter provides an
extensible hook for multi-round model aging, which we leave to future
work." This explicitly reserves the territory without implementing it.

---

## 5. Module Spec: FEMNIST data loader

```python
"""
Add to data/datasets.py:

FEMNIST: Federated Extended MNIST
- 62 classes (digits + upper/lower case letters)
- 3,500+ writers → naturally non-IID
- 28x28 grayscale images
- Download from LEAF benchmark: https://leaf.cmu.edu/

For JPDC experiments:
- Use 200 writers as 200 devices (natural non-IID partition)
- No artificial Dirichlet needed — the data IS non-IID
- Public reference set: 5,000 samples from held-out writers
"""

def load_femnist(root='./data', n_devices=200, n_public=5000, seed=42):
    """
    Returns: private_sets (dict: device_id -> dataset),
             public_set, test_set
    """
    pass
```

---

## 6. Experiment Plan: run_jpdc_experiments.py

```
Experiment 1: Main Comparison (Sync vs Async)
  - Dataset: CIFAR-100, M=20
  - Methods: Sync-RADS, Async-RADS(fixed), Async-RADS(adaptive),
             Async-RADS(partial), FedBuff-FD, Random-Async
  - Straggler: sigma=0.5 (~20% straggler rate)
  - Metrics: accuracy vs round, accuracy vs wall-clock time
  - Seeds: 3
  → Produces: Fig 2 (acc vs round), Fig 3 (acc vs wall-clock)

Experiment 2: Straggler Severity Sweep
  - Dataset: CIFAR-100, M=50
  - Method: Async-RADS(adaptive) only
  - Straggler: sigma ∈ {0.0, 0.3, 0.5, 1.0, 1.5}
              (corresponds to ~0%, 10%, 20%, 35%, 50% straggler rate)
  - Metrics: final accuracy, wall-clock time, straggler rate
  → Produces: Fig 4 (accuracy vs straggler ratio), Fig 5 (time vs straggler)

Experiment 3: Timeout Policy Comparison
  - Dataset: CIFAR-100, M=50
  - Methods: Fixed(D=5), Fixed(D=10), Fixed(D=20),
             Adaptive(p=0.5), Adaptive(p=0.7), Adaptive(p=0.9),
             Partial(p=0.7)
  - Straggler: sigma=1.0 (harsh conditions)
  → Produces: Fig 6 (accuracy-time Pareto front for each policy)

Experiment 4: Scalability
  - Dataset: CIFAR-100
  - Method: Async-RADS(adaptive) vs Sync-RADS
  - M ∈ {20, 50, 100, 200}
  - Straggler: sigma=0.5
  → Produces: Fig 7 (accuracy vs M), Fig 8 (wall-clock per round vs M)

Experiment 5: FEMNIST Validation
  - Dataset: FEMNIST, M ∈ {50, 200}
  - Methods: Sync-RADS, Async-RADS(adaptive), Async-RADS(partial)
  - Straggler: sigma=0.5
  → Produces: Fig 9 (FEMNIST accuracy vs wall-clock)

Experiment 6: Privacy under Async (retain from EDGE)
  - Dataset: CIFAR-100, M=50
  - Method: Async-RADS(adaptive)
  - rho sweep: {1.0, 0.8, 0.55, 0.1}
  - Straggler: sigma=0.5
  → Produces: Fig 10 (privacy vs accuracy under async conditions)
```

### Expected figures total: ~10 figures + 2-3 tables
### Expected runtime: ~2-4 hours on RTX 5070 Ti (mostly Exp 4 at M=200)

---

## 7. Implementation Priority Order

```
Week 1: Core async modules
  Day 1-2: straggler_model.py + unit tests
  Day 3-4: timeout_policy.py + unit tests
  Day 5:   async_kaas_edge.py (wrap existing KaaSEdge)

Week 2: Data + Experiments
  Day 1:   FEMNIST loader + partition
  Day 2-3: run_jpdc_experiments.py (Exp 1-3)
  Day 4-5: run Exp 4-6, collect results

Week 3: Paper writing
  (parallel with experiment debugging)
```

---

## 8. Copilot Prompt Templates

### For straggler_model.py:
```
Create a Python module straggler_model.py that simulates device
latency in a federated learning system. Each device has:
- computation time = comp_rate * v_star (deterministic)
- communication time = comm_rate * v_star (deterministic)
- noise = sample from LogNormal(mu, sigma) distribution

Given a deadline D, classify each device as:
- 'complete' if total_latency <= D
- 'partial' if comp_time <= D < total_latency
  (v_received = v_star * (D - comp_time) / comm_time)
- 'timeout' if comp_time > D (v_received = 0)

Return a list of DeviceLatency dataclass objects.
Include seeded RNG for reproducibility.
```

### For async_kaas_edge.py:
```
Create AsyncKaaSEdge class that extends the existing KaaSEdge class.
The only difference is in run_round():
1. After RADS scheduling, call StragglerModel.simulate_round()
2. Use TimeoutPolicy.get_deadline() to get the round deadline
3. For devices with outcome='timeout', skip them entirely
4. For devices with outcome='partial', use v_received instead of v_star
5. Compute quality weight: w_i = rho_i * q_i(v_received) / (1 + lambda * 0)
   (staleness=0 in this version)
6. Track self.wall_clock_time += deadline
7. Store latency history for adaptive policy

Keep the RADS scheduling, local training, logit computation, and
distillation code IDENTICAL to the sync version.
```

### For FEMNIST:
```
Add a load_femnist() function to data/datasets.py.
Download FEMNIST from the LEAF benchmark.
Partition naturally by writer (each writer = one device).
Select n_devices writers with the most samples.
Hold out n_public samples from excluded writers as public reference set.
Return private_sets (dict), public_set, test_set.
Images are 28x28 grayscale, resize to 32x32 to match CNN architecture.
```
