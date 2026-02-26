"""
KaaS-Edge: Knowledge-as-a-Service for Edge Intelligence

Centralized resource-aware federated distillation where the edge server
runs RADS (Resource-Aware Distillation Scheduling) to determine:
  1. Which devices participate (greedy submodular selection)
  2. How much each device uploads (water-filling allocation)

Key differences from PAID-FD (TMC):
  - Centralized ES scheduling vs decentralized device self-selection
  - Saturation quality q_i = rho_i * v_i / (v_i + theta_i) vs log(1 + s*g(eps))
  - Privacy as fixed attribute rho_i vs decision variable eps_i
  - Quality-weighted aggregation (w_i ∝ rho_i * v_i) vs BLUE (eps^2 weights)
  - Budget on total cost vs Stackelberg pricing
  - No LDP noise injection (privacy modeled as degradation factor)

Protocol per round (Algorithm 3 in EDGE paper):
  1. ES collects channel feedback, runs RADS → (S, {v_i*})
  2. ES broadcasts (w^(t-1), {v_i*}) to selected devices
  3. Each selected device i:
     a. Trains local model on private data
     b. Computes logits on reference dataset D_ref
     c. Applies privacy perturbation (attenuated by rho_i)
     d. Uploads v_i* logit vectors
  4. ES aggregates with quality weights: w_i = rho_i * q_i / sum(rho_j * q_j)
  5. ES updates server model via KL-distillation on (D_ref, z_agg)
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
    """
    Configuration for KaaS-Edge.
    
    Uses completely different notation from TMC's PAIDFDConfig
    to maintain academic separation.
    """
    # ── RADS scheduling parameters ──
    budget: float = 10.0             # Per-round total cost budget B
    v_max: float = 200.0             # Maximum upload volume per device
    
    # ── Device heterogeneity ──
    # rho_i, theta_i, b_i, a_i are generated per device (see generate_edge_devices)
    
    # ── Local training ──
    local_epochs: int = 2            # Local epochs per round (fewer than TMC's 5)
    local_lr: float = 0.01           # SGD learning rate
    local_momentum: float = 0.9
    
    # ── Distillation ──
    distill_epochs: int = 3          # Server distillation epochs per round
    distill_lr: float = 0.001        # Adam learning rate (conservative to avoid destruction)
    distill_alpha: float = 0.5       # α×KL(teacher) + (1-α)×CE(true): CE anchors model
    temperature: float = 3.0         # Soft-label temperature
    
    # ── Pre-training ──
    pretrain_epochs: int = 10        # Pre-train on public/reference data
    pretrain_lr: float = 0.1
    
    # ── Reference dataset ──
    n_ref_samples: int = 10000       # |D_ref| — reference dataset size
    
    # ── Privacy simulation ──
    # rho_i ∈ (0, 1]: 1.0 = no privacy, lower = more privacy degradation
    # Privacy is pre-determined, NOT a decision variable
    clip_bound: float = 2.0          # Logit clipping for privacy perturbation


class KaaSEdge(FederatedMethod):
    """
    KaaS-Edge: Knowledge-as-a-Service with RADS Scheduling.
    
    Usage:
        config = KaaSEdgeConfig(budget=10.0)
        method = KaaSEdge(server_model, config)
        
        for t in range(T):
            result = method.run_round(t, devices, client_loaders, public_loader)
    """
    
    def __init__(
        self,
        server_model: nn.Module,
        config: KaaSEdgeConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        super().__init__(server_model, n_classes, device)
        
        self.config = config or KaaSEdgeConfig()
        
        # RADS scheduler (replaces Stackelberg solver)
        self.scheduler = RADSScheduler(
            budget=self.config.budget,
            v_max=self.config.v_max,
        )
        
        # Persistent distillation optimizer
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(),
            lr=self.config.distill_lr
        )
        
        # Pre-training state
        self._pretrained = False
        
        # Tracking
        self.scheduling_history = []
        self.participation_history = []
    
    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any,
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """
        Execute one round of KaaS-Edge.
        
        Args:
            round_idx: Current round index
            devices: List of device dicts/objects with rho_i, b_i, theta_i, a_i
            client_loaders: Dict mapping device_id -> DataLoader
            public_loader: DataLoader for reference dataset D_ref
            test_loader: Optional test DataLoader
            
        Returns:
            RoundResult with metrics
        """
        self.current_round = round_idx
        
        # ── Pre-train on reference data (once) ──
        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True
        
        # ── Stage 1: RADS scheduling (centralized) ──
        device_dicts = self._prepare_device_dicts(devices)
        sched_result = self.scheduler.schedule(device_dicts)
        
        self.scheduling_history.append({
            'round': round_idx,
            'n_selected': sched_result.n_selected,
            'total_quality': sched_result.total_quality,
            'total_cost': sched_result.total_cost,
            'water_level': sched_result.water_level,
        })
        
        # ── Collect reference data into tensors ──
        ref_images_list = []
        ref_labels_list = []
        for data, labels in public_loader:
            ref_images_list.append(data)
            ref_labels_list.append(labels)
        ref_images = torch.cat(ref_images_list, dim=0)
        ref_labels = torch.cat(ref_labels_list, dim=0)
        n_ref = len(ref_images)
        
        # ── Stage 2: Selected devices compute logits ──
        all_logits = []
        all_weights = []
        participants = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        C = self.config.clip_bound
        
        for alloc in sched_result.allocations:
            if not alloc.selected or alloc.v_star <= 0:
                continue
            
            dev_id = alloc.device_id
            if dev_id not in client_loaders:
                continue
            
            participants.append(dev_id)
            local_loader = client_loaders[dev_id]
            
            # ── Local training: fresh copy from server each round ──
            local_model = copy_model(self.server_model, device=self.device)
            local_optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.local_lr,
                momentum=self.config.local_momentum,
                weight_decay=5e-4
            )
            
            local_model.train()
            criterion = nn.CrossEntropyLoss()
            for _ in range(self.config.local_epochs):
                for data, target in local_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    local_optimizer.zero_grad()
                    loss = criterion(local_model(data), target)
                    loss.backward()
                    local_optimizer.step()
            
            # ── Compute logits on reference dataset ──
            local_model.eval()
            n_logits = min(int(alloc.v_star), n_ref)
            logit_chunks = []
            bs = 512
            with torch.no_grad():
                for start in range(0, n_logits, bs):
                    end = min(start + bs, n_logits)
                    batch = ref_images[start:end].to(self.device)
                    logits = local_model(batch)
                    logits = torch.clamp(logits, -C, C)
                    logit_chunks.append(logits.cpu())
            
            if logit_chunks:
                device_logits = torch.cat(logit_chunks, dim=0)
                
                # ── Privacy perturbation (modeled by rho_i) ──
                # rho_i < 1 means some information is lost to privacy
                # We simulate this by adding noise proportional to (1 - rho_i)
                rho = alloc.rho_i
                if rho < 1.0:
                    noise_scale = C * (1.0 - rho) / max(rho, 0.01)
                    noise = torch.randn_like(device_logits) * noise_scale
                    device_logits = device_logits + noise
                
                all_logits.append(device_logits)
                # Quality-weighted aggregation: w_i = rho_i * q_i
                all_weights.append(rho * alloc.quality)
            
            # Energy accounting (simplified for EDGE: abstract costs)
            total_energy["training"] += alloc.cost * 0.4   # ~40% training
            total_energy["inference"] += alloc.cost * 0.3   # ~30% inference
            total_energy["communication"] += alloc.cost * 0.3  # ~30% comm
            
            # Free local model to save memory
            del local_model, local_optimizer
        
        # ── Stage 3: Quality-weighted aggregation + distillation ──
        if all_logits:
            min_len = min(len(l) for l in all_logits)
            total_w = sum(all_weights)
            
            if total_w > 0:
                norm_w = [w / total_w for w in all_weights]
            else:
                norm_w = [1.0 / len(all_logits)] * len(all_logits)
            
            aggregated = sum(
                w * l[:min_len] for w, l in zip(norm_w, all_logits)
            )
            
            # Distill to server model (with CE anchor)
            T = self.config.temperature
            teacher_probs = F.softmax(aggregated / T, dim=1)
            self._distill_to_server(
                teacher_probs, ref_images[:min_len], ref_labels[:min_len]
            )
        
        # ── Evaluate ──
        accuracy = 0.0
        loss = 0.0
        if test_loader is not None:
            eval_result = self.evaluate(test_loader)
            accuracy = eval_result["accuracy"]
            loss = eval_result["loss"]
        
        participation_rate = len(participants) / len(devices) if devices else 0
        self.participation_history.append(participation_rate)
        
        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=len(participants),
            energy=total_energy,
            extra={
                "n_selected": sched_result.n_selected,
                "total_quality": sched_result.total_quality,
                "total_cost": sched_result.total_cost,
                "water_level": sched_result.water_level,
                "budget": sched_result.budget,
            }
        )
        
        self.round_history.append(result)
        return result
    
    def _prepare_device_dicts(self, devices: List[Any]) -> List[Dict]:
        """Convert device objects/dicts to RADS-compatible dicts."""
        device_dicts = []
        for d in devices:
            if isinstance(d, dict):
                device_dicts.append(d)
            else:
                # Convert DeviceProfile-like objects
                device_dicts.append({
                    'device_id': getattr(d, 'device_id', 0),
                    'rho_i': getattr(d, 'rho_i', 1.0),
                    'b_i': getattr(d, 'b_i', getattr(d, 'c_total', 0.15)),
                    'theta_i': getattr(d, 'theta_i', 50.0),
                    'a_i': getattr(d, 'a_i', 0.1),
                })
        return device_dicts
    
    def _pretrain_on_public(self, public_loader):
        """Pre-train server model on reference data."""
        print(f"  [KaaS-Edge Pre-training] {self.config.pretrain_epochs} epochs ...")
        
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=self.config.pretrain_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_epochs
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.pretrain_epochs):
            self.server_model.train()
            epoch_loss = 0.0
            n_batches = 0
            for data, target in public_loader:
                data = augment(data).to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.server_model(data), target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{self.config.pretrain_epochs}: "
                      f"loss={epoch_loss/n_batches:.4f}")
        
        print(f"  [KaaS-Edge Pre-training] Done.")
    
    def _distill_to_server(
        self,
        teacher_probs: torch.Tensor,
        ref_images: torch.Tensor,
        ref_labels: torch.Tensor = None
    ):
        """
        Mixed-loss distillation: α×KL(teacher) + (1-α)×CE(true).
        
        KL transfers ensemble knowledge from aggregated logits.
        CE anchors to ground truth, preventing catastrophic forgetting
        when teachers are noisy (key lesson from TMC development).
        """
        self.server_model.train()
        optimizer = self.distill_optimizer
        T = self.config.temperature
        alpha = self.config.distill_alpha
        ce_criterion = nn.CrossEntropyLoss()
        
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        
        n_target = min(len(teacher_probs), len(ref_images))
        batch_size = 256
        
        for epoch in range(self.config.distill_epochs):
            perm = torch.randperm(n_target)
            
            for start in range(0, n_target, batch_size):
                end = min(start + batch_size, n_target)
                idx = perm[start:end]
                
                data = augment(ref_images[idx]).to(self.device)
                target_probs = teacher_probs[idx].to(self.device)
                
                student_logits = self.server_model(data)
                
                # KL distillation from teacher
                loss_kl = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)
                
                # CE anchor to ground truth (prevents destruction by noisy teachers)
                if ref_labels is not None and alpha < 1.0:
                    true_labels = ref_labels[idx].to(self.device)
                    loss_ce = ce_criterion(student_logits, true_labels)
                    loss = alpha * loss_kl + (1 - alpha) * loss_ce
                else:
                    loss = loss_kl
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()
    
    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """Interface compatibility — aggregation happens in run_round."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "n_rounds": len(self.round_history),
            "scheduling_history": self.scheduling_history,
            "participation_history": self.participation_history,
            "best_accuracy": self.get_best_accuracy(),
            "final_accuracy": self.get_final_accuracy(),
            "avg_participation": np.mean(self.participation_history) if self.participation_history else 0,
        }


# ============================================================================
# Baseline Methods for EDGE Paper Comparison
# ============================================================================

class FullParticipationFD(FederatedMethod):
    """
    Baseline: All devices upload max logits every round.
    No scheduling, no cost awareness.
    """
    
    def __init__(self, server_model, config=None, n_classes=100, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr
        )
        self._pretrained = False
    
    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        if not self._pretrained:
            self._pretrain(public_loader)
            self._pretrained = True
        
        # Collect reference data
        ref_imgs, ref_lbls = [], []
        for d, l in public_loader:
            ref_imgs.append(d); ref_lbls.append(l)
        ref_imgs = torch.cat(ref_imgs); ref_lbls = torch.cat(ref_lbls)
        n_ref = len(ref_imgs)
        
        all_logits, participants = [], []
        C = self.config.clip_bound
        total_cost = 0.0
        
        for dev in devices:
            dev_id = dev['device_id'] if isinstance(dev, dict) else getattr(dev, 'device_id', 0)
            if dev_id not in client_loaders:
                continue
            participants.append(dev_id)
            
            local_model = copy_model(self.server_model, device=self.device)
            opt = torch.optim.SGD(local_model.parameters(), lr=self.config.local_lr,
                                  momentum=0.9, weight_decay=5e-4)
            local_model.train()
            crit = nn.CrossEntropyLoss()
            for _ in range(self.config.local_epochs):
                for data, target in client_loaders[dev_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    opt.zero_grad(); crit(local_model(data), target).backward(); opt.step()
            
            local_model.eval()
            chunks = []
            with torch.no_grad():
                for s in range(0, n_ref, 512):
                    e = min(s+512, n_ref)
                    chunks.append(torch.clamp(local_model(ref_imgs[s:e].to(self.device)), -C, C).cpu())
            all_logits.append(torch.cat(chunks))
            
            b_i = dev.get('b_i', 0.15) if isinstance(dev, dict) else getattr(dev, 'b_i', 0.15)
            a_i = dev.get('a_i', 0.1) if isinstance(dev, dict) else getattr(dev, 'a_i', 0.1)
            total_cost += a_i + b_i * n_ref
            del local_model, opt
        
        if all_logits:
            min_len = min(len(l) for l in all_logits)
            agg = sum(l[:min_len] for l in all_logits) / len(all_logits)
            T = self.config.temperature
            teacher = F.softmax(agg / T, dim=1)
            self._distill(teacher, ref_imgs[:min_len], ref_lbls[:min_len])
        
        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]
        
        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants)/len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": total_cost*0.4, "inference": total_cost*0.3, "communication": total_cost*0.3},
            extra={"total_cost": total_cost, "method": "FullParticipation"}
        )
        self.round_history.append(result)
        return result
    
    def _pretrain(self, loader):
        print(f"  [FullFD Pre-train] {self.config.pretrain_epochs} epochs ...")
        aug = transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip()])
        opt = torch.optim.SGD(self.server_model.parameters(), lr=self.config.pretrain_lr,
                              momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.config.pretrain_epochs)
        crit = nn.CrossEntropyLoss()
        for ep in range(self.config.pretrain_epochs):
            self.server_model.train()
            for d, t in loader:
                d=aug(d).to(self.device); t=t.to(self.device)
                opt.zero_grad(); crit(self.server_model(d),t).backward(); opt.step()
            sched.step()
        print(f"  [FullFD Pre-train] Done.")
    
    def _distill(self, teacher_probs, ref_imgs, ref_lbls=None):
        self.server_model.train()
        T = self.config.temperature
        alpha = getattr(self.config, 'distill_alpha', 0.5)
        ce_crit = nn.CrossEntropyLoss()
        aug = transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip()])
        n = min(len(teacher_probs), len(ref_imgs))
        for _ in range(self.config.distill_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 256):
                e = min(s+256, n); idx = perm[s:e]
                d = aug(ref_imgs[idx]).to(self.device)
                tp = teacher_probs[idx].to(self.device)
                sl = self.server_model(d)
                loss_kl = F.kl_div(F.log_softmax(sl/T,dim=1), tp, reduction='batchmean')*(T*T)
                if ref_lbls is not None and alpha < 1.0:
                    tl = ref_lbls[idx].to(self.device)
                    loss = alpha * loss_kl + (1-alpha) * ce_crit(sl, tl)
                else:
                    loss = loss_kl
                self.distill_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                self.distill_optimizer.step()
    
    def aggregate(self, updates, weights): pass


class RandomSelectionFD(FederatedMethod):
    """
    Baseline: Randomly select fraction of devices each round.
    Budget-unaware random scheduling.
    """
    
    def __init__(self, server_model, config=None, n_classes=100, device=None,
                 select_fraction=0.5):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or KaaSEdgeConfig()
        self.select_fraction = select_fraction
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr
        )
        self._pretrained = False
        self.rng = np.random.RandomState(42)
    
    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        self.current_round = round_idx
        if not self._pretrained:
            # Reuse FullParticipationFD's pretrain logic
            self._pretrain(public_loader)
            self._pretrained = True
        
        # Random device selection
        n_select = max(1, int(len(devices) * self.select_fraction))
        all_ids = [d['device_id'] if isinstance(d, dict) else getattr(d, 'device_id', i)
                   for i, d in enumerate(devices)]
        selected = self.rng.choice(all_ids, size=n_select, replace=False).tolist()
        
        ref_imgs, ref_lbls = [], []
        for d, l in public_loader:
            ref_imgs.append(d); ref_lbls.append(l)
        ref_imgs = torch.cat(ref_imgs); ref_lbls = torch.cat(ref_lbls)
        n_ref = len(ref_imgs)
        
        all_logits, participants = [], []
        C = self.config.clip_bound
        total_cost = 0.0
        
        for dev_id in selected:
            if dev_id not in client_loaders:
                continue
            participants.append(dev_id)
            
            local_model = copy_model(self.server_model, device=self.device)
            opt = torch.optim.SGD(local_model.parameters(), lr=self.config.local_lr,
                                  momentum=0.9, weight_decay=5e-4)
            local_model.train()
            crit = nn.CrossEntropyLoss()
            for _ in range(self.config.local_epochs):
                for data, target in client_loaders[dev_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    opt.zero_grad(); crit(local_model(data), target).backward(); opt.step()
            
            local_model.eval()
            chunks = []
            with torch.no_grad():
                for s in range(0, n_ref, 512):
                    e = min(s+512, n_ref)
                    chunks.append(torch.clamp(local_model(ref_imgs[s:e].to(self.device)), -C, C).cpu())
            all_logits.append(torch.cat(chunks))
            
            dev = devices[dev_id] if dev_id < len(devices) else devices[0]
            b_i = dev.get('b_i', 0.15) if isinstance(dev, dict) else getattr(dev, 'b_i', 0.15)
            a_i = dev.get('a_i', 0.1) if isinstance(dev, dict) else getattr(dev, 'a_i', 0.1)
            total_cost += a_i + b_i * n_ref
            del local_model, opt
        
        if all_logits:
            min_len = min(len(l) for l in all_logits)
            agg = sum(l[:min_len] for l in all_logits) / len(all_logits)
            T = self.config.temperature
            teacher = F.softmax(agg / T, dim=1)
            self._distill(teacher, ref_imgs[:min_len], ref_lbls[:min_len])
        
        acc, loss = 0.0, 0.0
        if test_loader:
            ev = self.evaluate(test_loader); acc = ev["accuracy"]; loss = ev["loss"]
        
        result = RoundResult(
            round_idx=round_idx, accuracy=acc, loss=loss,
            participation_rate=len(participants)/len(devices) if devices else 0,
            n_participants=len(participants),
            energy={"training": total_cost*0.4, "inference": total_cost*0.3, "communication": total_cost*0.3},
            extra={"total_cost": total_cost, "method": "RandomSelection",
                   "select_fraction": self.select_fraction}
        )
        self.round_history.append(result)
        return result
    
    def _pretrain(self, loader):
        print(f"  [RandomFD Pre-train] {self.config.pretrain_epochs} epochs ...")
        aug = transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip()])
        opt = torch.optim.SGD(self.server_model.parameters(), lr=self.config.pretrain_lr,
                              momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.config.pretrain_epochs)
        crit = nn.CrossEntropyLoss()
        for ep in range(self.config.pretrain_epochs):
            self.server_model.train()
            for d, t in loader:
                d=aug(d).to(self.device); t=t.to(self.device)
                opt.zero_grad(); crit(self.server_model(d),t).backward(); opt.step()
            sched.step()
        print(f"  [RandomFD Pre-train] Done.")
    
    def _distill(self, teacher_probs, ref_imgs, ref_lbls=None):
        self.server_model.train()
        T = self.config.temperature
        alpha = getattr(self.config, 'distill_alpha', 0.5)
        ce_crit = nn.CrossEntropyLoss()
        aug = transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip()])
        n = min(len(teacher_probs), len(ref_imgs))
        for _ in range(self.config.distill_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 256):
                e = min(s+256, n); idx = perm[s:e]
                d = aug(ref_imgs[idx]).to(self.device)
                tp = teacher_probs[idx].to(self.device)
                sl = self.server_model(d)
                loss_kl = F.kl_div(F.log_softmax(sl/T,dim=1), tp, reduction='batchmean')*(T*T)
                if ref_lbls is not None and alpha < 1.0:
                    tl = ref_lbls[idx].to(self.device)
                    loss = alpha * loss_kl + (1-alpha) * ce_crit(sl, tl)
                else:
                    loss = loss_kl
                self.distill_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                self.distill_optimizer.step()
    
    def aggregate(self, updates, weights): pass


# ============================================================================
# Device Generator for EDGE Paper
# ============================================================================

def generate_edge_devices(
    n_devices: int = 20,
    seed: int = 42,
    cost_range: Tuple[float, float] = (0.05, 0.3),
    theta_range: Tuple[float, float] = (20.0, 80.0),
    activation_cost_range: Tuple[float, float] = (0.05, 0.2),
) -> List[Dict]:
    """
    Generate device profiles for KaaS-Edge experiments.
    
    Uses abstract cost interface (no Rayleigh fading, no DVFS formulas).
    Three tiers with cost ratios ~1:2:4 (matching paper).
    
    Args:
        n_devices: Number of devices M
        seed: Random seed
        cost_range: Range for base marginal cost b_i
        theta_range: Range for half-saturation constant theta_i
        activation_cost_range: Range for fixed activation cost a_i
        
    Returns:
        List of device dicts ready for RADS
    """
    rng = np.random.RandomState(seed)
    devices = []
    
    # Device tiers: high(30%), medium(40%), low(30%)
    n_high = int(n_devices * 0.3)
    n_med = int(n_devices * 0.4)
    n_low = n_devices - n_high - n_med
    
    tier_labels = ['high'] * n_high + ['medium'] * n_med + ['low'] * n_low
    rng.shuffle(tier_labels)
    
    # Cost multipliers by tier (abstract, no hardware names)
    cost_mult = {'high': 1.0, 'medium': 2.0, 'low': 4.0}
    
    # Privacy rho_i distribution: most devices have moderate privacy
    rho_levels = {
        'none': (1.0, 0.15),      # No privacy degradation, 15%
        'low': (0.8, 0.25),       # Slight degradation
        'medium': (0.5, 0.30),    # Moderate 
        'high': (0.2, 0.20),      # Strong privacy
        'extreme': (0.05, 0.10),  # Very strong
    }
    
    rho_assignments = []
    for level, (val, ratio) in rho_levels.items():
        count = int(n_devices * ratio)
        rho_assignments.extend([(val, level)] * count)
    while len(rho_assignments) < n_devices:
        rho_assignments.append((0.5, 'medium'))
    rng.shuffle(rho_assignments)
    
    for i in range(n_devices):
        tier = tier_labels[i]
        base_b = rng.uniform(*cost_range)
        b_i = base_b * cost_mult[tier]
        
        # theta_i: half-saturation (higher = harder to learn from)
        theta_i = rng.uniform(*theta_range)
        
        # Activation cost
        a_i = rng.uniform(*activation_cost_range)
        
        # Privacy
        rho_base, rho_level = rho_assignments[i]
        rho_i = np.clip(rho_base + rng.uniform(-0.05, 0.05), 0.01, 1.0)
        
        devices.append({
            'device_id': i,
            'rho_i': float(rho_i),
            'b_i': float(b_i),
            'theta_i': float(theta_i),
            'a_i': float(a_i),
            'tier': tier,
            'rho_level': rho_level,
            'eta_i': float(rho_i / (b_i * theta_i)),  # Pre-compute efficiency
        })
    
    return devices
