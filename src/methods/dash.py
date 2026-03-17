"""
DASH: Deadline-Aware Straggler-Handling for Asynchronous Edge Intelligence.
============================================================================

JPDC 2026 — Extends the synchronous KaaSEdge (IEEE EDGE 2026) with:

1. DASH-Select scheduling: rho_tilde_i = pi_i(D)*rho_i substitution
   converts deterministic scheduling into straggler-aware expected-
   quality optimization Q_tilde(S) (Algorithm 2 in the paper).
2. TimeoutPolicy: fixed / adaptive / partial-accept deadline D^(t).
3. Three-outcome device model (Complete / Partial / Timeout).
4. Quality-weighted aggregation using v_received-based quality (Eq. 26).
5. Wall-clock time tracking (cumulative deadline sum).
6. Latency history for adaptive deadline estimation.

The water-filling allocation (Proposition 1), KL distillation, local
training, logit computation, and LDP noise are identical to the
synchronous version --- only the scheduling objective changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .base import FederatedMethod, RoundResult
from .kaas_edge import (
    KaaSEdgeConfig,
    apply_ldp_noise,
    compute_comm_mb,
    _do_pretrain,
    _do_distill,
    _collect_ref,
    _local_train,
    _compute_logits,
)
from ..scheduler.rads import RADSScheduler
from ..async_module.straggler_model import StragglerModel, DeviceLatency
from ..async_module.timeout_policy import TimeoutPolicy, create_timeout_policy
from ..models.utils import copy_model


@dataclass
class DASHConfig(KaaSEdgeConfig):
    """Config for DASH (async variant).  Inherits all sync parameters."""
    # ── Straggler model ──
    sigma_noise: float = 0.5           # LogNormal σ  (straggler severity)
    mu_noise: float = 0.0              # LogNormal μ

    # ── Timeout policy ──
    timeout_policy: str = 'adaptive'   # 'fixed' | 'adaptive' | 'partial'
    fixed_deadline: float = 10.0       # D_0 for fixed / warmup
    adaptive_percentile: float = 0.85  # p for adaptive / partial
    warmup_rounds: int = 3

    # ── Deadline floor ──
    min_deadline_ratio: float = 0.3    # D_min = ratio * D_warmup (prevents spiral)

    # ── DASH-Select: straggler-aware scheduling ──
    straggler_aware: bool = True       # rho_tilde_i = pi_i(D) * rho_i


# =========================================================================
# DASH method + async baselines
# =========================================================================

class DASH(FederatedMethod):
    """
    DASH: Deadline-Aware Straggler-Handling method.

    Round protocol (Algorithm 1 in JPDC paper)
    -------------------------------------------
    1. Get deadline  D^(t) from timeout policy.
    2. DASH-Select scheduling (Algorithm 2):
       - Compute rho_tilde_i = pi_i(D) * rho_i for each device
       - Stage 1: water-filling with rho_tilde_i  ->  {v_i*}
       - Stage 2: greedy on Q_tilde(S) using rho_tilde_i-based quality
    3. Simulate device latencies via StragglerModel.
    4. Filter by outcome:
       - complete  ->  use v_star logits
       - partial   ->  use v_received (if policy.accepts_partial)
       - timeout   ->  discard
    5. Collect logits on public ref set.
    6. Quality-weighted aggregation (Eq. 26, uses original rho_i).
    7. Distill to server model (KL divergence, Eq. 28).
    8. Update tracking:
       wall_clock += D^(t);  latency_history += [tau_i, ...];
       record n_complete / n_partial / n_timeout.
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: Optional[DASHConfig] = None,
        n_classes: int = 100,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or DASHConfig()

        # ── Straggler model ──
        self.straggler_model = StragglerModel(
            mu_noise=self.config.mu_noise,
            sigma_noise=self.config.sigma_noise,
            seed=42,
        )

        # ── Timeout policy ──
        self.timeout_policy: TimeoutPolicy = create_timeout_policy(
            name=self.config.timeout_policy,
            D_0=self.config.fixed_deadline,
            percentile=self.config.adaptive_percentile,
            warmup_rounds=self.config.warmup_rounds,
            D_default=self.config.fixed_deadline,
        )

        # ── DASH scheduler (with straggler-aware flag) ──
        self.scheduler = RADSScheduler(
            budget=self.config.budget,
            v_max=self.config.v_max,
            straggler_aware=self.config.straggler_aware,
            straggler_model=self.straggler_model,
            deadline=self.config.fixed_deadline,
        )

        # ── Distillation optimiser ──
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr,
        )

        # ── Deadline floor (prevents adaptive spiral) ──
        self._D_min: Optional[float] = None  # set on first run_round

        # ── Tracking ──
        self._pretrained = False
        self.wall_clock_time: float = 0.0
        self.latency_history: List[List[float]] = []
        self.round_stats: List[Dict[str, Any]] = []
        self.scheduling_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_warmup_deadline(self, devices: List[Dict]) -> float:
        """
        Estimate a reasonable initial deadline from device rates and budget.

        During warmup we have no latency history, so the Adaptive policy
        falls back to D_default.  If D_default is too tight relative to
        the actual workload (comp_rate + comm_rate) * v*, nothing
        completes → no history → permanent stall.  This helper computes
        a workload-proportional deadline to break the cycle.

        We estimate v_per_device from the budget constraint (not v_max),
        then compute:
            D_warmup = p85(rate_sum) * v_est * safety
        where safety = 1.5 accommodates LogNormal noise.
        """
        M = len(devices)
        if M == 0:
            return 10.0

        rate_sums = [
            d.get('comp_rate', 0.01) + d.get('comm_rate', 0.01)
            for d in devices
        ]
        # Use budget to estimate per-device v, not v_max
        total_a = sum(d.get('a_i', 0.0) for d in devices)
        residual = max(self.config.budget - total_a, 1.0)
        median_b = float(np.median([d.get('b_i', 0.001) for d in devices]))
        estimated_v = residual / (M * median_b)  # budget-feasible v per device
        # Use scheduler.v_max if it differs from config (e.g. FullAsyncFD
        # sets scheduler.v_max = v_fixed before calling super().run_round).
        effective_v_max = getattr(self.scheduler, 'v_max', self.config.v_max)
        estimated_v = min(estimated_v, effective_v_max)

        p85_rate = float(np.percentile(rate_sums, 85))
        D_warmup = p85_rate * estimated_v * 1.5  # safety=1.5
        return max(D_warmup, 10.0)  # floor of 10 seconds

    # ------------------------------------------------------------------
    # Main round
    # ------------------------------------------------------------------

    def run_round(
        self,
        round_idx: int,
        devices: List[Dict],
        client_loaders: Dict[int, Any],
        public_loader: Any,
        test_loader: Any = None,
    ) -> RoundResult:
        self.current_round = round_idx
        rng = np.random.RandomState(round_idx * 1000 + 7)

        # Pre-train once
        if not self._pretrained:
            _do_pretrain(
                self.server_model, public_loader,
                self.config.pretrain_epochs, self.config.pretrain_lr,
                self.device,
            )
            self._pretrained = True

        # ── Step 1: Deadline ──
        # During warmup the adaptive policy returns D_default, which may
        # be too tight for the actual workload.  Estimate a reasonable
        # deadline from device rates to avoid the "0-select death spiral".
        in_warmup = (
            round_idx < self.config.warmup_rounds
            or len(self.latency_history) == 0
        )
        if in_warmup:
            deadline = self._estimate_warmup_deadline(devices)
            # Also seed D_default so adaptive policy starts from a
            # sensible value once it kicks in.
            if hasattr(self.timeout_policy, 'D_default'):
                self.timeout_policy.D_default = deadline
            # Compute D_min floor (once, from warmup estimate)
            if self._D_min is None and self.config.min_deadline_ratio > 0:
                self._D_min = deadline * self.config.min_deadline_ratio
        else:
            deadline = self.timeout_policy.get_deadline(
                round_idx, self.latency_history,
            )
            # Enforce D_min floor to prevent deadline spiral
            if self._D_min is not None and deadline < self._D_min:
                deadline = self._D_min

        # ── Step 2: DASH-Select scheduling ──
        # During warmup: disable straggler-aware to guarantee selection,
        # so we collect latency history for the adaptive policy.
        if in_warmup:
            self.scheduler.straggler_aware = False
        else:
            self.scheduler.straggler_aware = self.config.straggler_aware
        self.scheduler.deadline = deadline
        sched = self.scheduler.schedule(devices)
        self.scheduling_history.append({
            'round': round_idx,
            'n_selected': sched.n_selected,
            'selected_ids': sched.selected_ids,
            'total_quality': sched.total_quality,
            'total_cost': sched.total_cost,
            'deadline': deadline,
        })

        # ── Step 3: Simulate latencies ──
        selected_devices_for_sim = []
        for alloc in sched.allocations:
            if not alloc.selected or alloc.v_star <= 0:
                continue
            # Find the device dict to get comp_rate / comm_rate
            dev_dict = next(
                (d for d in devices if d['device_id'] == alloc.device_id),
                None,
            )
            if dev_dict is None:
                continue
            selected_devices_for_sim.append({
                'device_id': alloc.device_id,
                'v_star': int(alloc.v_star),
                'comp_rate': dev_dict.get('comp_rate', 0.01),
                'comm_rate': dev_dict.get('comm_rate', 0.01),
            })

        latencies = self.straggler_model.simulate_round(
            selected_devices_for_sim, deadline,
        )

        # ── Step 4: Filter by outcome ──
        surviving = []  # list of (device_id, v_to_use, quality_weight)
        for lat in latencies:
            if lat.outcome == 'complete':
                surviving.append((lat.device_id, lat.v_star))
            elif lat.outcome == 'partial' and self.timeout_policy.accepts_partial:
                if lat.v_received > 0:
                    surviving.append((lat.device_id, lat.v_received))
            # timeout → discard

        # ── Steps 5-7: Collect logits + aggregate + distill ──
        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs)
        C = self.config.clip_bound

        uploads = []       # (indices, logits, weight)
        participants = []
        total_comm_mb = 0.0
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}

        # Build a quick lookup from device_id → alloc for quality info
        alloc_map = {
            a.device_id: a for a in sched.allocations if a.selected
        }

        for dev_id, v_use in surviving:
            if dev_id not in client_loaders:
                continue
            participants.append(dev_id)

            # Local training (identical to sync)
            local_m = _local_train(
                self.server_model, client_loaders[dev_id],
                self.config, self.device,
            )

            # Logit computation — use v_use (may be < v_star for partial)
            v_i = int(min(v_use, n_ref))
            if v_i >= n_ref:
                idx = np.arange(n_ref)
            else:
                idx = rng.choice(n_ref, size=v_i, replace=False)
                idx.sort()

            logits = _compute_logits(local_m, ref_imgs, idx, C, self.device)

            # LDP noise
            alloc = alloc_map.get(dev_id)
            rho_i = alloc.rho_i if alloc else 1.0
            logits = apply_ldp_noise(logits, rho_i, C)

            # Quality weight — Eq.(26): w_i = rho_i * q_i(v_recv)
            # where q_i(v) = rho_i * v / (v + theta_i)
            theta_i = alloc.theta_i if alloc else 50.0
            q_recv = rho_i * v_i / (v_i + theta_i) if v_i > 0 else 0.0
            w_i = rho_i * q_recv  # extra rho_i: trust higher-fidelity logits
            uploads.append((idx, logits, w_i))

            # Bookkeeping
            cost_i = (alloc.cost if alloc else 0.0)
            total_energy["training"] += cost_i * 0.4
            total_energy["inference"] += cost_i * 0.3
            total_energy["communication"] += cost_i * 0.3
            total_comm_mb += compute_comm_mb(v_i, self.n_classes)
            del local_m

        # Masked quality-weighted aggregation (identical logic to sync)
        n_valid = 0
        if uploads:
            agg = torch.zeros(n_ref, self.n_classes)
            wsum = torch.zeros(n_ref)
            for idx, logits, w in uploads:
                agg[idx] += w * logits
                wsum[idx] += w
            mask = wsum > 0
            agg[mask] /= wsum[mask].unsqueeze(1)
            valid = torch.where(mask)[0]
            n_valid = len(valid)
            if n_valid > 0:
                T = self.config.temperature
                teacher = F.softmax(agg[valid] / T, dim=1)
                _do_distill(
                    self.server_model, self.distill_optimizer,
                    teacher, ref_imgs[valid], ref_lbls[valid],
                    self.config.distill_epochs, T,
                    self.config.distill_alpha, self.device,
                )

        # ── Step 8: Update tracking ──
        self.wall_clock_time += deadline
        self.latency_history.append(
            [lat.tau_total for lat in latencies]
        )

        n_complete = sum(1 for l in latencies if l.outcome == 'complete')
        n_partial = sum(1 for l in latencies if l.outcome == 'partial')
        n_timeout = sum(1 for l in latencies if l.outcome == 'timeout')

        self.round_stats.append({
            'round': round_idx,
            'deadline': deadline,
            'n_selected': len(latencies),
            'n_complete': n_complete,
            'n_partial': n_partial,
            'n_timeout': n_timeout,
            'n_contributing': len(participants),
            'wall_clock_time': self.wall_clock_time,
        })

        # ── Evaluate ──
        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader)
            acc, loss = ev["accuracy"], ev["loss"]

        result = RoundResult(
            round_idx=round_idx,
            accuracy=acc,
            loss=loss,
            participation_rate=(
                len(participants) / len(devices) if devices else 0
            ),
            n_participants=len(participants),
            energy=total_energy,
            extra={
                'n_selected': sched.n_selected,
                'total_quality': sched.total_quality,
                'total_cost': sched.total_cost,
                'water_level': sched.water_level,
                'budget': self.config.budget,
                'n_valid_samples': n_valid,
                'comm_mb': total_comm_mb,
                'selected_ids': sched.selected_ids,
                'deadline': deadline,
                'wall_clock_time': self.wall_clock_time,
                'n_complete': n_complete,
                'n_partial': n_partial,
                'n_timeout': n_timeout,
            },
        )
        self.round_history.append(result)
        return result

    def aggregate(self, updates, weights):
        pass  # aggregation happens inline in run_round

    def get_statistics(self):
        return {
            "n_rounds": len(self.round_history),
            "best_accuracy": self.get_best_accuracy(),
            "final_accuracy": self.get_final_accuracy(),
            "wall_clock_time": self.wall_clock_time,
            "round_stats": self.round_stats,
        }


# =========================================================================
# Baseline: Full-Async  (all devices + adaptive timeout, no budget)
# =========================================================================

class FullAsyncFD(DASH):
    """
    All devices participate, adaptive timeout, no budget constraint.

    This is the async analogue of FullParticipationFD: no device
    selection, no volume optimisation.  Every device uploads the same
    fixed number of logits (v_fixed).  Straggler handling still applies.
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: Optional[DASHConfig] = None,
        n_classes: int = 100,
        device: Optional[str] = None,
        v_fixed: int = 100,
    ):
        cfg = config or DASHConfig()
        # Override budget to be very large so all devices are selected
        cfg.budget = 1e6
        cfg.straggler_aware = False
        super().__init__(server_model, cfg, n_classes, device)
        self.v_fixed = v_fixed

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        # Override v_max so DASH gives everyone ~v_fixed
        self.scheduler.v_max = self.v_fixed
        return super().run_round(
            round_idx, devices, client_loaders, public_loader, test_loader,
        )


# =========================================================================
# Baseline: Random-Async  (random selection + fixed timeout)
# =========================================================================

class RandomAsyncFD(FederatedMethod):
    """
    Random 50 % device selection + fixed timeout.

    Combines the simplest selection strategy with async straggler
    handling.  Serves as a lower-bound baseline.
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: Optional[DASHConfig] = None,
        n_classes: int = 100,
        device: Optional[str] = None,
        select_fraction: float = 0.5,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or DASHConfig()
        self.select_fraction = select_fraction

        self.straggler_model = StragglerModel(
            mu_noise=self.config.mu_noise,
            sigma_noise=self.config.sigma_noise,
            seed=42,
        )
        self.timeout_policy = create_timeout_policy(
            'fixed', D_0=self.config.fixed_deadline,
        )
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr,
        )
        self._pretrained = False
        self.wall_clock_time = 0.0
        self.latency_history: List[List[float]] = []
        self.rng = np.random.RandomState(42)

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        rng = np.random.RandomState(round_idx * 1000 + 13)

        if not self._pretrained:
            _do_pretrain(
                self.server_model, public_loader,
                self.config.pretrain_epochs, self.config.pretrain_lr,
                self.device,
            )
            self._pretrained = True

        deadline = self.timeout_policy.get_deadline(
            round_idx, self.latency_history,
        )

        # Random selection
        n_select = max(1, int(len(devices) * self.select_fraction))
        all_ids = [d['device_id'] for d in devices]
        selected_ids = self.rng.choice(
            all_ids, size=n_select, replace=False,
        ).tolist()

        # Uniform volume
        v_uniform = 100

        # Simulate latencies
        sim_devices = []
        for did in selected_ids:
            dev = next((d for d in devices if d['device_id'] == did), None)
            if dev is None:
                continue
            sim_devices.append({
                'device_id': did,
                'v_star': v_uniform,
                'comp_rate': dev.get('comp_rate', 0.01),
                'comm_rate': dev.get('comm_rate', 0.01),
            })
        latencies = self.straggler_model.simulate_round(sim_devices, deadline)

        # Filter surviving
        surviving = []
        for lat in latencies:
            if lat.outcome == 'complete':
                surviving.append((lat.device_id, lat.v_star))

        # Collect + distill
        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs)
        C = self.config.clip_bound
        all_logits, participants = [], []
        total_comm_mb = 0.0

        for dev_id, v_use in surviving:
            if dev_id not in client_loaders:
                continue
            participants.append(dev_id)
            lm = _local_train(
                self.server_model, client_loaders[dev_id],
                self.config, self.device,
            )
            v_i = int(min(v_use, n_ref))
            idx = np.arange(n_ref) if v_i >= n_ref else np.sort(
                rng.choice(n_ref, size=v_i, replace=False)
            )
            logits = _compute_logits(lm, ref_imgs, idx, C, self.device)
            dev = next((d for d in devices if d['device_id'] == dev_id), {})
            rho = dev.get('rho_i', 1.0)
            logits = apply_ldp_noise(logits, rho, C)
            all_logits.append((idx, logits))
            total_comm_mb += compute_comm_mb(v_i, self.n_classes)
            del lm

        if all_logits:
            agg = torch.zeros(n_ref, self.n_classes)
            cnt = torch.zeros(n_ref)
            for idx, logits in all_logits:
                agg[idx] += logits
                cnt[idx] += 1.0
            mask = cnt > 0
            agg[mask] /= cnt[mask].unsqueeze(1)
            valid = torch.where(mask)[0]
            if len(valid) > 0:
                T = self.config.temperature
                teacher = F.softmax(agg[valid] / T, dim=1)
                _do_distill(
                    self.server_model, self.distill_optimizer,
                    teacher, ref_imgs[valid], ref_lbls[valid],
                    self.config.distill_epochs, T,
                    self.config.distill_alpha, self.device,
                )

        self.wall_clock_time += deadline
        self.latency_history.append(
            [lat.tau_total for lat in latencies]
        )

        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader)
            acc, loss = ev["accuracy"], ev["loss"]

        n_complete = sum(1 for l in latencies if l.outcome == 'complete')
        n_timeout = sum(1 for l in latencies if l.outcome != 'complete')

        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants) / len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": 0, "inference": 0, "communication": 0},
            extra={
                'method': 'RandomAsync',
                'comm_mb': total_comm_mb,
                'wall_clock_time': self.wall_clock_time,
                'deadline': deadline,
                'n_complete': n_complete,
                'n_timeout': n_timeout,
            },
        )
        self.round_history.append(result)
        return result

    def aggregate(self, u, w):
        pass


# =========================================================================
# Backward-compatible aliases (for existing scripts / imports)
# =========================================================================
AsyncKaaSEdge = DASH
AsyncKaaSEdgeConfig = DASHConfig
