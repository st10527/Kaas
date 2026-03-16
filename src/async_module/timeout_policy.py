"""
Timeout Policies for Async-RADS (JPDC 2026).

Three policies that determine the per-round deadline D^(t):

    Policy A — Fixed:    D^(t) = D_0  for all t.
    Policy B — Adaptive: D^(t) = percentile_p of previous round's latencies,
                         with EMA smoothing.
    Policy C — Partial-Accept:  Same adaptive deadline as B, but also accepts
                                partial logit uploads (v_received < v_star).

All policies implement a common `get_deadline(round_idx, history)` interface.
"""

import numpy as np
from typing import List, Optional


class TimeoutPolicy:
    """Base class for timeout policies."""

    def get_deadline(
        self,
        round_idx: int,
        history: List[List[float]],
    ) -> float:
        """
        Return the deadline D^(t) for this round.

        Parameters
        ----------
        round_idx : int
            Current round index (0-based).
        history : list of list of float
            history[r] = list of per-device tau_total values from round r.

        Returns
        -------
        float  — deadline in seconds.
        """
        raise NotImplementedError

    @property
    def accepts_partial(self) -> bool:
        """Whether this policy keeps partial logit uploads."""
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__


# =========================================================================
# Policy A: Fixed Deadline
# =========================================================================

class FixedDeadlinePolicy(TimeoutPolicy):
    """
    D^(t) = D_0  for all t.

    Simple, predictable, zero overhead.  Good as a baseline.
    """

    def __init__(self, D_0: float = 10.0):
        self.D_0 = D_0

    def get_deadline(self, round_idx: int, history: List[List[float]]) -> float:
        return self.D_0

    @property
    def name(self) -> str:
        return f"Fixed(D={self.D_0})"


# =========================================================================
# Policy B: Adaptive Deadline
# =========================================================================

class AdaptiveDeadlinePolicy(TimeoutPolicy):
    """
    D^(t) = EMA-smoothed p-th percentile of round (t-1)'s latencies.

    Parameters
    ----------
    percentile : float
        Target completion fraction, e.g. 0.7 means the deadline is set
        so that ~70 % of last round's devices would have completed.
    warmup_rounds : int
        Number of initial rounds that use D_default instead of adaptive.
    D_default : float
        Default deadline during warmup.
    ema_alpha : float
        Smoothing factor for exponential moving average of the deadline.
        Higher → more responsive to recent latency; lower → more stable.
    """

    def __init__(
        self,
        percentile: float = 0.7,
        warmup_rounds: int = 3,
        D_default: float = 10.0,
        ema_alpha: float = 0.3,
    ):
        if not 0 < percentile <= 1.0:
            raise ValueError(f"percentile must be in (0, 1], got {percentile}")
        self.percentile = percentile
        self.warmup_rounds = warmup_rounds
        self.D_default = D_default
        self.ema_alpha = ema_alpha
        self._ema_deadline: Optional[float] = None

    def get_deadline(self, round_idx: int, history: List[List[float]]) -> float:
        # During warmup or if no history, use the default
        if round_idx < self.warmup_rounds or len(history) == 0:
            return self.D_default

        prev_latencies = history[-1]
        if len(prev_latencies) == 0:
            return self.D_default

        raw = float(np.percentile(prev_latencies, self.percentile * 100))

        # EMA smoothing to prevent oscillation
        if self._ema_deadline is None:
            self._ema_deadline = raw
        else:
            self._ema_deadline = (
                self.ema_alpha * raw
                + (1.0 - self.ema_alpha) * self._ema_deadline
            )
        return self._ema_deadline

    def reset(self):
        """Reset EMA state (call between experiments)."""
        self._ema_deadline = None

    @property
    def name(self) -> str:
        return f"Adaptive(p={self.percentile})"


# =========================================================================
# Policy C: Partial-Accept
# =========================================================================

class PartialAcceptPolicy(AdaptiveDeadlinePolicy):
    """
    Same adaptive deadline as Policy B, but instructs the aggregation
    engine to *keep* partial logit uploads rather than discarding them.

    The key difference is not in the deadline calculation, but in the
    `accepts_partial` flag that AsyncKaaSEdge checks during aggregation.
    """

    def __init__(self, percentile: float = 0.7, **kwargs):
        super().__init__(percentile=percentile, **kwargs)

    @property
    def accepts_partial(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"Partial(p={self.percentile})"


# =========================================================================
# Factory
# =========================================================================

_POLICY_REGISTRY = {
    'fixed': FixedDeadlinePolicy,
    'adaptive': AdaptiveDeadlinePolicy,
    'partial': PartialAcceptPolicy,
}


def create_timeout_policy(name: str, **kwargs) -> TimeoutPolicy:
    """
    Create a timeout policy from its short name.

    Parameters
    ----------
    name : str
        One of 'fixed', 'adaptive', 'partial'.
    **kwargs
        Forwarded to the policy constructor.  Extra kwargs that do not
        match the constructor signature are silently ignored, so callers
        can safely pass a superset of all policy parameters.

    Returns
    -------
    TimeoutPolicy
    """
    import inspect

    key = name.lower().strip()
    if key not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown timeout policy '{name}'. "
            f"Choose from {list(_POLICY_REGISTRY.keys())}"
        )
    cls = _POLICY_REGISTRY[key]
    # Only pass kwargs that the constructor actually accepts
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
