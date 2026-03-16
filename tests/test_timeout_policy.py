"""
Unit tests for timeout policies (src/async_module/timeout_policy.py).

Tests:
  - FixedDeadlinePolicy always returns D_0
  - AdaptiveDeadlinePolicy uses warmup, then adapts from history
  - PartialAcceptPolicy has accepts_partial=True
  - Factory function
"""

import numpy as np
import pytest
from src.async_module.timeout_policy import (
    FixedDeadlinePolicy,
    AdaptiveDeadlinePolicy,
    PartialAcceptPolicy,
    TimeoutPolicy,
    create_timeout_policy,
)


class TestFixedDeadlinePolicy:

    def test_constant_value(self):
        p = FixedDeadlinePolicy(D_0=15.0)
        for t in range(10):
            assert p.get_deadline(t, []) == 15.0

    def test_ignores_history(self):
        p = FixedDeadlinePolicy(D_0=5.0)
        history = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
        assert p.get_deadline(5, history) == 5.0

    def test_not_partial(self):
        p = FixedDeadlinePolicy()
        assert p.accepts_partial is False

    def test_name(self):
        p = FixedDeadlinePolicy(D_0=7.5)
        assert 'Fixed' in p.name
        assert '7.5' in p.name


class TestAdaptiveDeadlinePolicy:

    def test_warmup_returns_default(self):
        p = AdaptiveDeadlinePolicy(percentile=0.7, warmup_rounds=3, D_default=10.0)
        history = [[1.0, 2.0]]
        assert p.get_deadline(0, history) == 10.0
        assert p.get_deadline(1, history) == 10.0
        assert p.get_deadline(2, history) == 10.0

    def test_after_warmup_uses_percentile(self):
        p = AdaptiveDeadlinePolicy(
            percentile=0.5, warmup_rounds=0, D_default=10.0, ema_alpha=1.0,
        )
        history = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        d = p.get_deadline(1, history)
        # 50th percentile of [1,2,3,4,5] = 3.0
        assert d == pytest.approx(3.0, abs=0.01)

    def test_ema_smoothing(self):
        p = AdaptiveDeadlinePolicy(
            percentile=0.7, warmup_rounds=0, D_default=10.0, ema_alpha=0.5,
        )
        # Round 1: all latencies = 10
        d1 = p.get_deadline(1, [[10.0] * 20])
        # Round 2: all latencies = 2
        d2 = p.get_deadline(2, [[10.0] * 20, [2.0] * 20])
        # EMA should have smoothed: not jumping to 2.0 immediately
        assert d2 > 2.0
        assert d2 < 10.0

    def test_empty_history_returns_default(self):
        p = AdaptiveDeadlinePolicy(warmup_rounds=0, D_default=8.0)
        assert p.get_deadline(5, []) == 8.0

    def test_not_partial(self):
        p = AdaptiveDeadlinePolicy()
        assert p.accepts_partial is False

    def test_invalid_percentile(self):
        with pytest.raises(ValueError):
            AdaptiveDeadlinePolicy(percentile=0.0)
        with pytest.raises(ValueError):
            AdaptiveDeadlinePolicy(percentile=1.5)

    def test_reset(self):
        p = AdaptiveDeadlinePolicy(warmup_rounds=0, ema_alpha=1.0)
        p.get_deadline(1, [[5.0]])
        assert p._ema_deadline is not None
        p.reset()
        assert p._ema_deadline is None


class TestPartialAcceptPolicy:

    def test_accepts_partial_true(self):
        p = PartialAcceptPolicy(percentile=0.7)
        assert p.accepts_partial is True

    def test_inherits_adaptive_behavior(self):
        p = PartialAcceptPolicy(percentile=0.5, warmup_rounds=0, ema_alpha=1.0)
        history = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        d = p.get_deadline(1, history)
        assert d == pytest.approx(3.0, abs=0.01)

    def test_name_contains_partial(self):
        p = PartialAcceptPolicy(percentile=0.9)
        assert 'Partial' in p.name


class TestFactory:

    def test_fixed(self):
        p = create_timeout_policy('fixed', D_0=7.0)
        assert isinstance(p, FixedDeadlinePolicy)
        assert p.get_deadline(0, []) == 7.0

    def test_adaptive(self):
        p = create_timeout_policy('adaptive', percentile=0.8)
        assert isinstance(p, AdaptiveDeadlinePolicy)
        assert p.percentile == 0.8

    def test_partial(self):
        p = create_timeout_policy('partial', percentile=0.6)
        assert isinstance(p, PartialAcceptPolicy)
        assert p.accepts_partial is True

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_timeout_policy('unknown_policy')

    def test_all_are_timeout_policy(self):
        for name in ['fixed', 'adaptive', 'partial']:
            p = create_timeout_policy(name)
            assert isinstance(p, TimeoutPolicy)
