"""
FedBuff-FD: Buffered Asynchronous Federated Distillation (JPDC 2026).
=====================================================================

Adapted from FedBuff (Nguyen et al., AISTATS 2022) for the FD setting.

Original FedBuff:
  - Server collects K async *parameter updates* into a buffer,
    then performs one aggregation step.

FedBuff-FD adaptation:
  - Devices upload *logit vectors* asynchronously.
  - Server buffers K logit uploads (first K to finish, regardless
    of which devices).
  - When buffer is full → aggregate via distillation on public set.
  - No device selection or volume optimisation — all devices compete.

Key parameters:
    buffer_size K : int = 10
    v_fixed       : int = 100   (uniform volume for all devices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
from ..async_module.straggler_model import StragglerModel


@dataclass
class FedBuffFDConfig(KaaSEdgeConfig):
    """Configuration for FedBuff-FD."""
    buffer_size: int = 10         # K: how many logit uploads trigger one aggregation
    v_fixed: int = 100            # uniform per-device upload volume
    sigma_noise: float = 0.5     # straggler severity
    mu_noise: float = 0.0


class FedBuffFD(FederatedMethod):
    """
    Buffered Asynchronous Federated Distillation.

    Round protocol (one "aggregation step"):
    1. All M devices start local training + logit computation.
    2. StragglerModel simulates each device's latency.
    3. Sort devices by total latency (ascending).
    4. The first K devices to finish fill the buffer.
    5. Aggregate buffered logits via equal-weight averaging.
    6. Distill to server model.
    7. Wall-clock time for this step = latency of the K-th device.

    Notes
    -----
    - "round" for FedBuff = one buffer-fill-and-aggregate cycle.
    - Fast devices naturally contribute more often, biasing the model
      toward fast-device data.  This is the main disadvantage vs RADS.
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: Optional[FedBuffFDConfig] = None,
        n_classes: int = 100,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or FedBuffFDConfig()

        self.straggler_model = StragglerModel(
            mu_noise=self.config.mu_noise,
            sigma_noise=self.config.sigma_noise,
            seed=42,
        )

        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr,
        )
        self._pretrained = False
        self.wall_clock_time: float = 0.0
        self.round_stats: List[Dict] = []

    def run_round(
        self,
        round_idx: int,
        devices: List[Dict],
        client_loaders: Dict[int, Any],
        public_loader: Any,
        test_loader: Any = None,
    ) -> RoundResult:
        self.current_round = round_idx
        rng = np.random.RandomState(round_idx * 1000 + 17)

        # Pre-train once
        if not self._pretrained:
            _do_pretrain(
                self.server_model, public_loader,
                self.config.pretrain_epochs, self.config.pretrain_lr,
                self.device,
            )
            self._pretrained = True

        K = self.config.buffer_size
        v_fixed = self.config.v_fixed

        # ── Simulate latencies for ALL devices ──
        sim_devices = []
        for dev in devices:
            did = dev['device_id']
            if did not in client_loaders:
                continue
            sim_devices.append({
                'device_id': did,
                'v_star': v_fixed,
                'comp_rate': dev.get('comp_rate', 0.01),
                'comm_rate': dev.get('comm_rate', 0.01),
            })

        # Use a very large deadline so everyone "eventually" finishes
        large_deadline = 1e6
        latencies = self.straggler_model.simulate_round(
            sim_devices, large_deadline,
        )

        # Sort by total latency — first K are the buffer
        latencies.sort(key=lambda l: l.tau_total)
        buffered = latencies[:K]

        # Wall-clock for this step = latency of K-th device
        step_time = buffered[-1].tau_total if buffered else 0.0
        self.wall_clock_time += step_time

        # ── Collect logits from buffered devices ──
        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs)
        C = self.config.clip_bound

        all_logits = []
        participants = []
        total_comm_mb = 0.0

        for lat in buffered:
            did = lat.device_id
            if did not in client_loaders:
                continue
            participants.append(did)

            local_m = _local_train(
                self.server_model, client_loaders[did],
                self.config, self.device,
            )

            v_i = int(min(v_fixed, n_ref))
            if v_i >= n_ref:
                idx = np.arange(n_ref)
            else:
                idx = rng.choice(n_ref, size=v_i, replace=False)
                idx.sort()

            logits = _compute_logits(local_m, ref_imgs, idx, C, self.device)
            dev = next((d for d in devices if d['device_id'] == did), {})
            rho = dev.get('rho_i', 1.0)
            logits = apply_ldp_noise(logits, rho, C)
            all_logits.append((idx, logits))
            total_comm_mb += compute_comm_mb(v_i, self.n_classes)
            del local_m

        # ── Equal-weight masked aggregation ──
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

        self.round_stats.append({
            'round': round_idx,
            'step_time': step_time,
            'wall_clock_time': self.wall_clock_time,
            'n_buffered': len(participants),
            'buffer_device_ids': [l.device_id for l in buffered],
        })

        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader)
            acc, loss = ev["accuracy"], ev["loss"]

        result = RoundResult(
            round_idx=round_idx,
            accuracy=acc,
            loss=loss,
            participation_rate=len(participants) / len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": 0, "inference": 0, "communication": 0},
            extra={
                'method': 'FedBuff-FD',
                'comm_mb': total_comm_mb,
                'wall_clock_time': self.wall_clock_time,
                'step_time': step_time,
                'buffer_size': K,
            },
        )
        self.round_history.append(result)
        return result

    def aggregate(self, u, w):
        pass
