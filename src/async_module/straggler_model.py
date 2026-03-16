"""
Straggler Latency Model for Async-RADS (JPDC 2026).

Each device i has a per-round latency:
    tau_i = tau_comp_i + tau_comm_i + tau_noise_i

Where:
    tau_comp_i  = comp_rate_i * v_i          (deterministic, device-dependent)
    tau_comm_i  = comm_rate_i * v_i          (deterministic, allocation-dependent)
    tau_noise_i ~ LogNormal(mu, sigma)       (stochastic jitter)

Given a deadline D, each device's outcome is:
    COMPLETE:   tau_i   <= D   →  v_recv = v_star
    PARTIAL:    tau_comp <= D < tau_i   →  v_recv = floor(v_star * (D - tau_comp) / tau_comm)
    TIMEOUT:    tau_comp  > D   →  v_recv = 0

The sigma parameter controls straggler severity:
    sigma = 0.0  →  deterministic (no straggler)
    sigma = 0.3  →  ~10% straggler at typical deadline
    sigma = 0.5  →  ~20% straggler
    sigma = 1.0  →  ~35% straggler
    sigma = 1.5  →  ~50% straggler
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class DeviceLatency:
    """Latency simulation result for a single device in one round."""
    device_id: int
    tau_comp: float       # computation time (seconds)
    tau_comm: float       # communication time (seconds)
    tau_noise: float      # stochastic perturbation (seconds)
    tau_total: float      # = tau_comp + tau_comm + tau_noise
    outcome: str          # 'complete' | 'partial' | 'timeout'
    v_star: int           # allocated logit vectors (from RADS)
    v_received: int       # actual logit vectors received (0 to v_star)
    deadline: float       # the deadline that was applied


class StragglerModel:
    """
    Simulates per-device latency and determines round outcome.

    Each device has a comp_rate and comm_rate drawn from its hardware
    profile.  On top of the deterministic component, a LogNormal noise
    models random jitter from OS scheduling, channel fading, etc.

    Parameters
    ----------
    mu_noise : float
        Log-normal location parameter (log-scale).  Default 0.0.
    sigma_noise : float
        Log-normal scale parameter (log-scale).  Primary knob for
        straggler ratio.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mu_noise: float = 0.0,
        sigma_noise: float = 0.5,
        seed: int = 42,
    ):
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def simulate_round(
        self,
        devices: List[Dict],
        deadline: float,
    ) -> List[DeviceLatency]:
        """
        Simulate latencies for all selected devices in one round.

        Parameters
        ----------
        devices : list of dict
            Each dict must contain:
                device_id   : int
                v_star      : int   — allocated volume from RADS
                comp_rate   : float — seconds per logit (computation)
                comm_rate   : float — seconds per logit (upload)
        deadline : float
            The round deadline D^(t) in seconds.

        Returns
        -------
        list of DeviceLatency
        """
        results: List[DeviceLatency] = []

        for dev in devices:
            v_star = int(dev['v_star'])
            if v_star <= 0:
                results.append(DeviceLatency(
                    device_id=dev['device_id'],
                    tau_comp=0.0, tau_comm=0.0, tau_noise=0.0,
                    tau_total=0.0, outcome='timeout',
                    v_star=0, v_received=0, deadline=deadline,
                ))
                continue

            tau_comp = dev['comp_rate'] * v_star
            tau_comm = dev['comm_rate'] * v_star

            # Stochastic jitter (LogNormal)
            if self.sigma_noise > 0:
                tau_noise = float(
                    self.rng.lognormal(self.mu_noise, self.sigma_noise)
                )
            else:
                tau_noise = 1.0  # unit scale, no variance

            tau_total = tau_comp + tau_comm + tau_noise

            # Determine outcome
            if tau_total <= deadline:
                outcome = 'complete'
                v_recv = v_star
            elif tau_comp <= deadline:
                # Device finished computation but could not upload everything
                usable_comm_time = deadline - tau_comp
                frac = usable_comm_time / tau_comm if tau_comm > 0 else 0.0
                v_recv = int(v_star * min(frac, 1.0))
                v_recv = max(0, min(v_recv, v_star))
                outcome = 'partial' if v_recv > 0 else 'timeout'
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
                v_star=v_star,
                v_received=v_recv,
                deadline=deadline,
            ))

        return results

    # ------------------------------------------------------------------
    # Straggler-aware scheduling interface
    # ------------------------------------------------------------------

    def completion_probability(
        self,
        comp_rate: float,
        comm_rate: float,
        v_star: float,
        deadline: float,
    ) -> float:
        """
        Compute Pr[tau_i <= D] for a single device, used by RADS
        Stage 2 when straggler_aware=True.

            pi_i(D) = Pr[tau_noise <= D - tau_comp - tau_comm]
                     = CDF_{LogNormal}(slack; mu, sigma)

        Parameters
        ----------
        comp_rate, comm_rate : float
            Per-logit rates for this device.
        v_star : float
            Volume allocation (continuous, from water-filling).
        deadline : float
            Current round deadline.

        Returns
        -------
        float in [0, 1]
        """
        tau_det = comp_rate * v_star + comm_rate * v_star
        slack = deadline - tau_det
        if slack <= 0:
            return 0.0
        if self.sigma_noise <= 0:
            # Deterministic: complete iff slack > 1 (unit noise)
            return 1.0 if slack >= 1.0 else 0.0

        # LogNormal CDF via the relationship to Normal CDF:
        #   CDF_{LN}(x; mu, sigma) = Phi((ln(x) - mu) / sigma)
        from scipy.stats import norm
        z = (np.log(slack) - self.mu_noise) / self.sigma_noise
        return float(norm.cdf(z))

    # ------------------------------------------------------------------
    # Device rate generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_device_rates(
        n_devices: int,
        seed: int = 42,
    ) -> List[Dict[str, float]]:
        """
        Generate realistic comp_rate / comm_rate per device.

        Three tiers (matching EDGE paper's cost model):
            Tier A (30%): fast   — comp_rate ~ 0.005, comm_rate ~ 0.008
            Tier B (40%): medium — comp_rate ~ 0.015, comm_rate ~ 0.020
            Tier C (30%): slow   — comp_rate ~ 0.040, comm_rate ~ 0.050

        Values are drawn from log-normal around the tier means.

        Returns list of {'comp_rate': float, 'comm_rate': float}.
        """
        rng = np.random.RandomState(seed)
        tiers = {
            'A': {'frac': 0.30, 'comp_mean': -5.3, 'comm_mean': -4.8},  # ~0.005, ~0.008
            'B': {'frac': 0.40, 'comp_mean': -4.2, 'comm_mean': -3.9},  # ~0.015, ~0.020
            'C': {'frac': 0.30, 'comp_mean': -3.2, 'comm_mean': -3.0},  # ~0.040, ~0.050
        }

        rates = []
        for _ in range(n_devices):
            # Pick tier
            r = rng.random()
            if r < 0.30:
                tier = tiers['A']
            elif r < 0.70:
                tier = tiers['B']
            else:
                tier = tiers['C']

            comp_rate = float(np.clip(
                rng.lognormal(tier['comp_mean'], 0.3), 0.001, 0.2
            ))
            comm_rate = float(np.clip(
                rng.lognormal(tier['comm_mean'], 0.3), 0.001, 0.2
            ))
            rates.append({'comp_rate': comp_rate, 'comm_rate': comm_rate})

        return rates

    def reset_rng(self, seed: int):
        """Reset the internal RNG (useful between experiments)."""
        self.rng = np.random.RandomState(seed)
