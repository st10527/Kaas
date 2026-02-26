"""
RADS: Resource-Aware Distillation Scheduling

Implements the two-stage scheduling algorithm for KaaS-Edge:
  Stage 1 (Proposition 1): Water-filling allocation for fixed device set
  Stage 2 (Theorem 1):     Greedy submodular device selection with (1-1/e)/2 guarantee

Notation (matches EDGE paper, NOT TMC):
  v_i   : upload volume (continuous, [0, v_max])
  rho_i : privacy degradation factor ∈ (0, 1]
  b_i   : marginal cost (comm + comp, abstract)
  theta_i: half-saturation constant (device learning difficulty)
  eta_i : efficiency index = rho_i / (b_i * theta_i)
  B     : total per-round budget
  nu    : water level (KKT dual variable)

Quality model (saturation):
  q_i(v_i) = rho_i * v_i / (v_i + theta_i)

This is fundamentally different from TMC's log(1 + s*g(eps)).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class DeviceAllocation:
    """Result of scheduling for a single device."""
    device_id: int
    selected: bool
    v_star: float         # Allocated upload volume
    quality: float        # q_i(v_i) = rho_i * v_i / (v_i + theta_i)
    cost: float           # a_i + b_i * v_i
    rho_i: float          # Privacy degradation factor
    b_i: float            # Marginal cost
    theta_i: float        # Half-saturation constant
    eta_i: float          # Efficiency index


@dataclass
class SchedulingResult:
    """Result of RADS scheduling for one round."""
    allocations: List[DeviceAllocation]
    selected_ids: List[int]
    total_quality: float
    total_cost: float
    budget: float
    water_level: float
    n_selected: int


class RADSScheduler:
    """
    Resource-Aware Distillation Scheduling (RADS).
    
    Two-stage algorithm:
      1. Water-filling: Given a device set S, optimally allocate budget
         via KKT conditions → v_i* = [sqrt(rho_i*theta_i / (nu*b_i)) - theta_i]^+
      2. Greedy selection: Build S by iteratively adding the device with
         highest marginal quality gain, subject to budget constraint.
    
    Approximation guarantee: Q(S^G) >= (1-1/e)/2 * Q(S*) ≈ 0.316 * OPT
    (Khuller et al. budgeted submodular maximization)
    
    Complexity: O(M^2 * log(1/delta)) per round
    """
    
    def __init__(
        self,
        budget: float = 10.0,
        v_max: float = 200.0,
        delta: float = 1e-6,
        min_devices: int = 1
    ):
        """
        Args:
            budget: Per-round total cost budget B
            v_max: Maximum upload volume per device
            delta: Bisection tolerance for water-level search
            min_devices: Minimum devices to select (if budget allows)
        """
        self.budget = budget
        self.v_max = v_max
        self.delta = delta
        self.min_devices = min_devices
    
    def schedule(
        self,
        devices: List[Dict],
    ) -> SchedulingResult:
        """
        Run RADS scheduling.
        
        Args:
            devices: List of dicts with keys:
                - device_id (int)
                - rho_i (float): privacy degradation ∈ (0,1]
                - b_i (float): marginal cost > 0
                - theta_i (float): half-saturation > 0
                - a_i (float): fixed activation cost >= 0
                
        Returns:
            SchedulingResult with device allocations
        """
        M = len(devices)
        
        # Compute efficiency index for each device
        for d in devices:
            d['eta_i'] = d['rho_i'] / (d['b_i'] * d['theta_i'])
        
        # Sort by efficiency (descending) for greedy phase
        sorted_devices = sorted(devices, key=lambda d: d['eta_i'], reverse=True)
        
        # ── Stage 2: Greedy device selection ──
        # Iteratively add the device with highest marginal gain
        selected_set = []
        selected_ids = set()
        current_cost = 0.0
        
        for candidate in sorted_devices:
            dev_id = candidate['device_id']
            a_i = candidate.get('a_i', 0.0)
            
            # Check if adding this device is budget-feasible
            # Minimum cost to add: activation cost + minimal variable cost
            min_add_cost = a_i + candidate['b_i'] * 1.0  # At least 1 unit
            if current_cost + min_add_cost > self.budget:
                continue
            
            # Tentatively add device and compute water-filling allocation
            trial_set = selected_set + [candidate]
            residual_budget = self.budget - sum(d.get('a_i', 0.0) for d in trial_set)
            
            if residual_budget <= 0:
                continue
            
            # Water-filling for trial set
            trial_allocs = self._water_filling(trial_set, residual_budget)
            
            # Compute marginal quality gain
            if trial_allocs is not None:
                trial_quality = sum(a['quality'] for a in trial_allocs)
                current_quality = sum(
                    a['quality'] for a in self._water_filling(
                        selected_set,
                        self.budget - sum(d.get('a_i', 0.0) for d in selected_set)
                    )
                ) if selected_set else 0.0
                
                marginal_gain = trial_quality - current_quality
                
                if marginal_gain > 0:
                    selected_set.append(candidate)
                    selected_ids.add(dev_id)
                    current_cost = sum(d.get('a_i', 0.0) for d in selected_set)
        
        # ── Final allocation for selected set ──
        if selected_set:
            residual_budget = self.budget - sum(d.get('a_i', 0.0) for d in selected_set)
            final_allocs = self._water_filling(selected_set, residual_budget)
        else:
            final_allocs = []
        
        # Build full allocation list (including non-selected devices)
        alloc_map = {a['device_id']: a for a in final_allocs}
        all_allocations = []
        
        for d in devices:
            dev_id = d['device_id']
            if dev_id in alloc_map:
                fa = alloc_map[dev_id]
                total_cost = d.get('a_i', 0.0) + d['b_i'] * fa['v_star']
                all_allocations.append(DeviceAllocation(
                    device_id=dev_id,
                    selected=True,
                    v_star=fa['v_star'],
                    quality=fa['quality'],
                    cost=total_cost,
                    rho_i=d['rho_i'],
                    b_i=d['b_i'],
                    theta_i=d['theta_i'],
                    eta_i=d['eta_i']
                ))
            else:
                all_allocations.append(DeviceAllocation(
                    device_id=dev_id,
                    selected=False,
                    v_star=0.0,
                    quality=0.0,
                    cost=0.0,
                    rho_i=d['rho_i'],
                    b_i=d['b_i'],
                    theta_i=d['theta_i'],
                    eta_i=d['eta_i']
                ))
        
        total_quality = sum(a.quality for a in all_allocations if a.selected)
        total_cost = sum(a.cost for a in all_allocations if a.selected)
        water_level = final_allocs[0]['nu'] if final_allocs else 0.0
        
        return SchedulingResult(
            allocations=all_allocations,
            selected_ids=sorted(selected_ids),
            total_quality=total_quality,
            total_cost=total_cost,
            budget=self.budget,
            water_level=water_level,
            n_selected=len(selected_ids)
        )
    
    def _water_filling(
        self,
        device_set: List[Dict],
        residual_budget: float
    ) -> Optional[List[Dict]]:
        """
        Stage 1: Water-filling allocation (Proposition 1).
        
        For fixed device set S with residual budget B_res:
          v_i* = min{ [sqrt(rho_i * theta_i / (nu * b_i)) - theta_i]^+, v_max }
        
        where nu > 0 is the unique water level satisfying:
          sum_i b_i * v_i*(nu) = B_res
        
        Solved via bisection on nu.
        
        Args:
            device_set: List of device dicts (must have rho_i, b_i, theta_i)
            residual_budget: B - sum(a_i) for selected devices
            
        Returns:
            List of allocation dicts, or None if infeasible
        """
        if not device_set or residual_budget <= 0:
            return []
        
        def compute_allocations(nu: float) -> Tuple[List[Dict], float]:
            """Given water level nu, compute v_i* for all devices and total cost."""
            allocs = []
            total_var_cost = 0.0
            
            for d in device_set:
                rho = d['rho_i']
                b = d['b_i']
                theta = d['theta_i']
                
                # KKT: v_i* = sqrt(rho * theta / (nu * b)) - theta
                inner = rho * theta / (nu * b)
                v_star = max(np.sqrt(inner) - theta, 0.0)
                v_star = min(v_star, self.v_max)
                
                # Quality: q_i = rho * v / (v + theta)
                quality = rho * v_star / (v_star + theta) if v_star > 0 else 0.0
                
                allocs.append({
                    'device_id': d['device_id'],
                    'v_star': v_star,
                    'quality': quality,
                    'nu': nu
                })
                total_var_cost += b * v_star
            
            return allocs, total_var_cost
        
        # Bisection on nu to match budget
        # When nu is small → large allocations → high cost
        # When nu is large → small allocations → low cost
        nu_low = 1e-12
        nu_high = 1e6
        
        # Check feasibility: even with minimum nu, can we use the budget?
        _, cost_at_low = compute_allocations(nu_low)
        if cost_at_low < residual_budget:
            # Budget is not binding; use nu_low (everyone gets v_max or unconstrained)
            allocs, _ = compute_allocations(nu_low)
            return allocs
        
        # Check upper bound
        _, cost_at_high = compute_allocations(nu_high)
        if cost_at_high > residual_budget:
            # Even with maximum nu, cost exceeds budget — shouldn't happen
            # Fall back to proportional allocation
            nu_high = 1e12
        
        # Bisection
        for _ in range(200):  # ~60 digits of precision
            if nu_high - nu_low < self.delta * nu_low:
                break
            
            nu_mid = (nu_low + nu_high) / 2.0
            _, cost_mid = compute_allocations(nu_mid)
            
            if cost_mid > residual_budget:
                nu_low = nu_mid  # Increase nu to reduce allocations
            else:
                nu_high = nu_mid  # Decrease nu to increase allocations
        
        nu_star = (nu_low + nu_high) / 2.0
        allocs, _ = compute_allocations(nu_star)
        return allocs
    
    def compute_quality(self, rho: float, v: float, theta: float) -> float:
        """Saturation quality function: q = rho * v / (v + theta)."""
        if v <= 0:
            return 0.0
        return rho * v / (v + theta)
