"""
KaaS-Edge: Knowledge-as-a-Service for Edge Intelligence
=========================================================
v2: Fixed v_i subsampling, proper Laplace LDP, log-normal costs.

Key fix: v_i* now ACTUALLY limits the number of reference samples
uploaded by each device, creating meaningful accuracy-cost tradeoffs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import copy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .base import FederatedMethod, RoundResult
from ..scheduler.rads import RADSScheduler, SchedulingResult, DeviceAllocation
from ..models.utils import copy_model


@dataclass
class KaaSEdgeConfig:
    budget: float = 10.0
    v_max: float = 200.0
    local_epochs: int = 2
    local_lr: float = 0.01
    local_momentum: float = 0.9
    distill_epochs: int = 3
    distill_lr: float = 0.001
    distill_alpha: float = 0.5
    temperature: float = 3.0
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1
    n_ref_samples: int = 10000
    clip_bound: float = 5.0


# ============================================================================
# Shared helpers
# ============================================================================

def apply_ldp_noise(logits, rho, clip_bound):
    """Laplace LDP: eps = rho/(1-rho), scale = 2C/eps."""
    if rho >= 0.999:
        return logits
    rho_c = max(rho, 0.01)
    eps = rho_c / (1.0 - rho_c)
    scale = 2.0 * clip_bound / eps
    u = torch.zeros_like(logits).uniform_(-0.5, 0.5)
    noise = -scale * u.sign() * torch.log1p(-2.0 * u.abs())
    return logits + noise


def compute_comm_mb(n_vectors, n_classes=100):
    return n_vectors * n_classes * 4 / (1024 * 1024)


def _do_pretrain(model, loader, epochs, lr, device):
    print(f"  [Pre-train] {epochs} epochs ...")
    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip()
    ])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for d, t in loader:
            d = aug(d).to(device); t = t.to(device)
            opt.zero_grad(); crit(model(d), t).backward(); opt.step()
        sched.step()
    print(f"  [Pre-train] Done.")


def _do_distill(model, optimizer, teacher_probs, ref_imgs, ref_lbls,
                epochs, T, alpha, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip()
    ])
    n = min(len(teacher_probs), len(ref_imgs))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 256):
            e = min(s + 256, n); idx = perm[s:e]
            d = aug(ref_imgs[idx]).to(device)
            tp = teacher_probs[idx].to(device)
            sl = model(d)
            loss_kl = F.kl_div(
                F.log_softmax(sl / T, dim=1), tp, reduction='batchmean'
            ) * (T * T)
            if ref_lbls is not None and alpha < 1.0:
                loss = alpha * loss_kl + (1 - alpha) * ce(sl, ref_lbls[idx].to(device))
            else:
                loss = loss_kl
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def _collect_ref(public_loader):
    imgs, lbls = [], []
    for d, l in public_loader:
        imgs.append(d); lbls.append(l)
    return torch.cat(imgs), torch.cat(lbls)


def _local_train(server_model, client_loader, config, device):
    m = copy_model(server_model, device=device)
    opt = torch.optim.SGD(m.parameters(), lr=config.local_lr, momentum=0.9, weight_decay=5e-4)
    m.train(); crit = nn.CrossEntropyLoss()
    for _ in range(config.local_epochs):
        for d, t in client_loader:
            d, t = d.to(device), t.to(device)
            opt.zero_grad(); crit(m(d), t).backward(); opt.step()
    return m


def _compute_logits(model, images, indices, clip_bound, device, bs=512):
    model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(indices), bs):
            e = min(s + bs, len(indices))
            batch = images[indices[s:e]].to(device)
            chunks.append(torch.clamp(model(batch), -clip_bound, clip_bound).cpu())
    return torch.cat(chunks)


# ============================================================================
# KaaS-Edge (Proposed)
# ============================================================================

class KaaSEdge(FederatedMethod):
    def __init__(self, server_model, config=None, n_classes=100, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.scheduler = RADSScheduler(budget=self.config.budget, v_max=self.config.v_max)
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr
        )
        self._pretrained = False
        self.scheduling_history = []
        self.participation_history = []

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        rng = np.random.RandomState(round_idx * 1000 + 7)

        if not self._pretrained:
            _do_pretrain(self.server_model, public_loader,
                         self.config.pretrain_epochs, self.config.pretrain_lr, self.device)
            self._pretrained = True

        sched = self.scheduler.schedule(devices)
        self.scheduling_history.append({
            'round': round_idx, 'n_selected': sched.n_selected,
            'selected_ids': sched.selected_ids,
            'total_quality': sched.total_quality, 'total_cost': sched.total_cost,
        })

        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs)
        C = self.config.clip_bound

        # ── Subsampled logit collection ──
        uploads = []  # (indices, logits, weight)
        participants = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        total_comm_mb = 0.0

        for alloc in sched.allocations:
            if not alloc.selected or alloc.v_star <= 0:
                continue
            if alloc.device_id not in client_loaders:
                continue
            participants.append(alloc.device_id)

            local_m = _local_train(self.server_model, client_loaders[alloc.device_id],
                                   self.config, self.device)

            # v_i* determines upload count
            v_i = int(min(alloc.v_star, n_ref))
            if v_i >= n_ref:
                idx = np.arange(n_ref)
            else:
                idx = rng.choice(n_ref, size=v_i, replace=False)
                idx.sort()

            logits = _compute_logits(local_m, ref_imgs, idx, C, self.device)
            logits = apply_ldp_noise(logits, alloc.rho_i, C)
            uploads.append((idx, logits, alloc.quality))

            total_energy["training"] += alloc.cost * 0.4
            total_energy["inference"] += alloc.cost * 0.3
            total_energy["communication"] += alloc.cost * 0.3
            total_comm_mb += compute_comm_mb(v_i, self.n_classes)
            del local_m

        # ── Masked quality-weighted aggregation ──
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
                _do_distill(self.server_model, self.distill_optimizer,
                            teacher, ref_imgs[valid], ref_lbls[valid],
                            self.config.distill_epochs, T,
                            self.config.distill_alpha, self.device)

        self.participation_history.append(len(participants))
        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]

        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants) / len(devices) if devices else 0,
            n_participants=len(participants),
            energy=total_energy,
            extra={'n_selected': sched.n_selected, 'total_quality': sched.total_quality,
                   'total_cost': sched.total_cost, 'water_level': sched.water_level,
                   'budget': self.config.budget, 'n_valid_samples': n_valid,
                   'comm_mb': total_comm_mb, 'selected_ids': sched.selected_ids}
        )
        self.round_history.append(result)
        return result

    def aggregate(self, updates, weights): pass
    def get_statistics(self):
        return {"n_rounds": len(self.round_history),
                "best_accuracy": self.get_best_accuracy(),
                "final_accuracy": self.get_final_accuracy()}


# ============================================================================
# Baseline: FedMD (Full Participation)
# ============================================================================

class FullParticipationFD(FederatedMethod):
    """All devices, full logits, equal-weight aggregation."""
    def __init__(self, server_model, config=None, n_classes=100, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr)
        self._pretrained = False

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        if not self._pretrained:
            _do_pretrain(self.server_model, public_loader,
                         self.config.pretrain_epochs, self.config.pretrain_lr, self.device)
            self._pretrained = True

        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs); C = self.config.clip_bound
        all_logits, participants = [], []
        total_cost = 0.0; total_comm_mb = 0.0
        full_idx = np.arange(n_ref)

        for dev in devices:
            did = dev['device_id'] if isinstance(dev, dict) else getattr(dev, 'device_id', 0)
            if did not in client_loaders: continue
            participants.append(did)
            lm = _local_train(self.server_model, client_loaders[did], self.config, self.device)
            logits = _compute_logits(lm, ref_imgs, full_idx, C, self.device)
            rho = dev.get('rho_i', 1.0) if isinstance(dev, dict) else 1.0
            logits = apply_ldp_noise(logits, rho, C)
            all_logits.append(logits)
            b_i = dev.get('b_i', 0.002) if isinstance(dev, dict) else 0.002
            a_i = dev.get('a_i', 0.2) if isinstance(dev, dict) else 0.2
            total_cost += a_i + b_i * n_ref
            total_comm_mb += compute_comm_mb(n_ref, self.n_classes)
            del lm

        if all_logits:
            agg = sum(all_logits) / len(all_logits)
            T = self.config.temperature
            teacher = F.softmax(agg / T, dim=1)
            _do_distill(self.server_model, self.distill_optimizer,
                        teacher, ref_imgs, ref_lbls,
                        self.config.distill_epochs, T, self.config.distill_alpha, self.device)

        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]
        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants)/len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": total_cost*0.4, "inference": total_cost*0.3, "communication": total_cost*0.3},
            extra={"total_cost": total_cost, "method": "FullParticipation", "comm_mb": total_comm_mb}
        )
        self.round_history.append(result)
        return result
    def aggregate(self, u, w): pass


# ============================================================================
# Baseline: Random Selection
# ============================================================================

class RandomSelectionFD(FederatedMethod):
    """Random 50% devices, full logits."""
    def __init__(self, server_model, config=None, n_classes=100, device=None,
                 select_fraction=0.5):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.select_fraction = select_fraction
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr)
        self._pretrained = False
        self.rng = np.random.RandomState(42)

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        if not self._pretrained:
            _do_pretrain(self.server_model, public_loader,
                         self.config.pretrain_epochs, self.config.pretrain_lr, self.device)
            self._pretrained = True

        n_select = max(1, int(len(devices) * self.select_fraction))
        all_ids = [d['device_id'] if isinstance(d, dict) else i for i, d in enumerate(devices)]
        selected = self.rng.choice(all_ids, size=n_select, replace=False).tolist()

        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs); C = self.config.clip_bound
        all_logits, participants = [], []
        total_cost = 0.0; total_comm_mb = 0.0
        full_idx = np.arange(n_ref)

        for did in selected:
            if did not in client_loaders: continue
            participants.append(did)
            lm = _local_train(self.server_model, client_loaders[did], self.config, self.device)
            logits = _compute_logits(lm, ref_imgs, full_idx, C, self.device)
            dev = devices[did] if did < len(devices) else devices[0]
            rho = dev.get('rho_i', 1.0) if isinstance(dev, dict) else 1.0
            logits = apply_ldp_noise(logits, rho, C)
            all_logits.append(logits)
            b_i = dev.get('b_i', 0.002) if isinstance(dev, dict) else 0.002
            a_i = dev.get('a_i', 0.2) if isinstance(dev, dict) else 0.2
            total_cost += a_i + b_i * n_ref
            total_comm_mb += compute_comm_mb(n_ref, self.n_classes)
            del lm

        if all_logits:
            agg = sum(all_logits) / len(all_logits)
            T = self.config.temperature
            teacher = F.softmax(agg / T, dim=1)
            _do_distill(self.server_model, self.distill_optimizer,
                        teacher, ref_imgs, ref_lbls,
                        self.config.distill_epochs, T, self.config.distill_alpha, self.device)

        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]
        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants)/len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": total_cost*0.4, "inference": total_cost*0.3, "communication": total_cost*0.3},
            extra={"total_cost": total_cost, "method": "RandomSelection",
                   "select_fraction": self.select_fraction, "comm_mb": total_comm_mb}
        )
        self.round_history.append(result)
        return result
    def aggregate(self, u, w): pass


# ============================================================================
# Baseline: FedCS-FD (Equal Allocation under same budget)
# ============================================================================

class FedCSFD(FederatedMethod):
    """Greedy selection + EQUAL volume allocation. Same budget as KaaS-Edge."""
    def __init__(self, server_model, config=None, n_classes=100, device=None,
                 budget=8.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.budget = budget
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr)
        self._pretrained = False

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        rng = np.random.RandomState(round_idx * 1000 + 13)
        if not self._pretrained:
            _do_pretrain(self.server_model, public_loader,
                         self.config.pretrain_epochs, self.config.pretrain_lr, self.device)
            self._pretrained = True

        # Greedy selection sorted by b_i (cheapest first)
        dev_info = []
        for i, dev in enumerate(devices):
            did = dev['device_id'] if isinstance(dev, dict) else i
            b_i = dev.get('b_i', 0.002) if isinstance(dev, dict) else 0.002
            a_i = dev.get('a_i', 0.2) if isinstance(dev, dict) else 0.2
            rho = dev.get('rho_i', 0.8) if isinstance(dev, dict) else 0.8
            dev_info.append({'dev_id': did, 'b_i': b_i, 'a_i': a_i, 'rho_i': rho})
        dev_info.sort(key=lambda x: x['b_i'])

        selected = []
        remaining = self.budget
        for d in dev_info:
            if d['a_i'] + d['b_i'] <= remaining:
                selected.append(d)
                remaining -= d['a_i']

        # EQUAL allocation
        if selected:
            per_dev = remaining / len(selected)
            for d in selected:
                d['v_alloc'] = int(min(per_dev / d['b_i'], self.config.v_max))

        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs); C = self.config.clip_bound
        uploads = []
        participants = []
        total_cost = 0.0; total_comm_mb = 0.0

        for d in selected:
            if d['dev_id'] not in client_loaders: continue
            participants.append(d['dev_id'])
            lm = _local_train(self.server_model, client_loaders[d['dev_id']],
                              self.config, self.device)
            v_i = int(min(d.get('v_alloc', n_ref), n_ref))
            if v_i >= n_ref:
                idx = np.arange(n_ref)
            else:
                idx = rng.choice(n_ref, size=v_i, replace=False); idx.sort()
            logits = _compute_logits(lm, ref_imgs, idx, C, self.device)
            logits = apply_ldp_noise(logits, d['rho_i'], C)
            uploads.append((idx, logits))
            total_cost += d['a_i'] + d['b_i'] * v_i
            total_comm_mb += compute_comm_mb(v_i, self.n_classes)
            del lm

        # Equal-weight masked aggregation
        if uploads:
            agg = torch.zeros(n_ref, self.n_classes)
            cnt = torch.zeros(n_ref)
            for idx, logits in uploads:
                agg[idx] += logits; cnt[idx] += 1.0
            mask = cnt > 0
            agg[mask] /= cnt[mask].unsqueeze(1)
            valid = torch.where(mask)[0]
            if len(valid) > 0:
                T = self.config.temperature
                teacher = F.softmax(agg[valid] / T, dim=1)
                _do_distill(self.server_model, self.distill_optimizer,
                            teacher, ref_imgs[valid], ref_lbls[valid],
                            self.config.distill_epochs, T, self.config.distill_alpha, self.device)

        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]
        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants)/len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": total_cost*0.4, "inference": total_cost*0.3, "communication": total_cost*0.3},
            extra={"total_cost": total_cost, "method": "FedCS-FD",
                   "n_selected": len(selected), "budget": self.budget, "comm_mb": total_comm_mb}
        )
        self.round_history.append(result)
        return result
    def aggregate(self, u, w): pass


# ============================================================================
# Baseline: FedSKD (Selective Knowledge — top-K classes)
# ============================================================================

class FedSKDFD(FederatedMethod):
    """All devices, full samples, but only top-K% classes per logit vector."""
    def __init__(self, server_model, config=None, n_classes=100, device=None,
                 select_ratio=0.5):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.select_ratio = select_ratio
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr)
        self._pretrained = False

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        if not self._pretrained:
            _do_pretrain(self.server_model, public_loader,
                         self.config.pretrain_epochs, self.config.pretrain_lr, self.device)
            self._pretrained = True

        ref_imgs, ref_lbls = _collect_ref(public_loader)
        n_ref = len(ref_imgs); C = self.config.clip_bound
        K_keep = max(1, int(self.n_classes * self.select_ratio))
        all_logits, participants = [], []
        total_cost = 0.0; total_comm_mb = 0.0

        for dev in devices:
            did = dev['device_id'] if isinstance(dev, dict) else 0
            if did not in client_loaders: continue
            participants.append(did)
            lm = _local_train(self.server_model, client_loaders[did], self.config, self.device)
            lm.eval()
            chunks = []
            with torch.no_grad():
                for s in range(0, n_ref, 512):
                    e = min(s + 512, n_ref)
                    logits = torch.clamp(lm(ref_imgs[s:e].to(self.device)), -C, C)
                    _, top_idx = logits.topk(K_keep, dim=1)
                    mask = torch.zeros_like(logits)
                    mask.scatter_(1, top_idx, 1.0)
                    chunks.append((logits * mask).cpu())
            device_logits = torch.cat(chunks)
            rho = dev.get('rho_i', 1.0) if isinstance(dev, dict) else 1.0
            device_logits = apply_ldp_noise(device_logits, rho, C)
            all_logits.append(device_logits)
            b_i = dev.get('b_i', 0.002) if isinstance(dev, dict) else 0.002
            a_i = dev.get('a_i', 0.2) if isinstance(dev, dict) else 0.2
            total_cost += a_i + b_i * n_ref * self.select_ratio
            total_comm_mb += compute_comm_mb(n_ref, K_keep)
            del lm

        if all_logits:
            agg = sum(all_logits) / len(all_logits)
            T = self.config.temperature
            teacher = F.softmax(agg / T, dim=1)
            _do_distill(self.server_model, self.distill_optimizer,
                        teacher, ref_imgs, ref_lbls,
                        self.config.distill_epochs, T, self.config.distill_alpha, self.device)

        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]
        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants)/len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": total_cost*0.4, "inference": total_cost*0.3, "communication": total_cost*0.3},
            extra={"total_cost": total_cost, "method": "FedSKD",
                   "select_ratio": self.select_ratio, "comm_mb": total_comm_mb}
        )
        self.round_history.append(result)
        return result
    def aggregate(self, u, w): pass


# ============================================================================
# Device Generation (Log-normal costs)
# ============================================================================

def generate_edge_devices(n_devices=20, seed=42):
    """Log-normal b_i distribution for realistic heterogeneity."""
    rng = np.random.RandomState(seed)
    # b_i ~ LogNormal: median ~0.002, long right tail to ~0.05
    log_b = rng.normal(loc=-6.2, scale=0.8, size=n_devices)
    b_vals = np.clip(np.exp(log_b), 0.0003, 0.1)
    theta_vals = rng.uniform(30.0, 100.0, size=n_devices)
    a_vals = rng.uniform(0.1, 0.5, size=n_devices)

    rho_pool = []
    for val, frac in [(1.0, 0.15), (0.8, 0.20), (0.5, 0.30), (0.2, 0.20), (0.05, 0.15)]:
        rho_pool.extend([val] * max(1, int(n_devices * frac)))
    while len(rho_pool) < n_devices:
        rho_pool.append(0.5)
    rho_pool = rho_pool[:n_devices]
    rng.shuffle(rho_pool)

    devices = []
    for i in range(n_devices):
        rho_i = float(np.clip(rho_pool[i] + rng.uniform(-0.05, 0.05), 0.01, 1.0))
        b_i = float(b_vals[i])
        devices.append({
            'device_id': i, 'rho_i': rho_i, 'b_i': b_i,
            'theta_i': float(theta_vals[i]), 'a_i': float(a_vals[i]),
            'eta_i': float(rho_i / (b_i * theta_vals[i])),
        })
    return devices
