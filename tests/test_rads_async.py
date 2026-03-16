"""
Unit tests for straggler-aware RADS extension (src/scheduler/rads.py).

Tests:
  - straggler_aware=False → identical to baseline RADS
  - straggler_aware=True → pi_i modulates selection
  - greedy still respects budget
"""

import numpy as np
import pytest
from src.scheduler.rads import RADSScheduler
from src.async_module.straggler_model import StragglerModel


def _make_devices(n=10, seed=42):
    """Devices with both RADS cost-model and async rates."""
    from src.methods.kaas_edge import generate_edge_devices
    devices = generate_edge_devices(n, seed=seed)
    rates = StragglerModel.generate_device_rates(n, seed=seed)
    for d, r in zip(devices, rates):
        d['comp_rate'] = r['comp_rate']
        d['comm_rate'] = r['comm_rate']
    return devices


class TestBaselineUnchanged:
    """straggler_aware=False → same as original RADS."""

    def test_no_flag_gives_same_result(self):
        devices = _make_devices(20, seed=42)
        s1 = RADSScheduler(budget=50.0, v_max=1000, straggler_aware=False)
        s2 = RADSScheduler(budget=50.0, v_max=1000)
        r1 = s1.schedule(devices)
        r2 = s2.schedule(devices)
        assert r1.selected_ids == r2.selected_ids
        assert r1.total_quality == pytest.approx(r2.total_quality, abs=1e-6)


class TestStragglerAware:
    """straggler_aware=True → pi_i modulates greedy gain."""

    def test_tight_deadline_reduces_selection(self):
        devices = _make_devices(20, seed=42)
        sm = StragglerModel(sigma_noise=0.5, seed=42)

        s_sync = RADSScheduler(budget=50.0, v_max=1000, straggler_aware=False)
        s_async = RADSScheduler(
            budget=50.0, v_max=1000,
            straggler_aware=True, straggler_model=sm, deadline=2.0,
        )

        r_sync = s_sync.schedule(devices)
        r_async = s_async.schedule(devices)

        # With a tight deadline, some devices have low pi_i → may not be selected
        # or total expected quality should be lower
        assert r_async.total_quality <= r_sync.total_quality + 1e-6

    def test_large_deadline_approaches_sync(self):
        devices = _make_devices(20, seed=42)
        sm = StragglerModel(sigma_noise=0.5, seed=42)

        s_sync = RADSScheduler(budget=50.0, v_max=1000, straggler_aware=False)
        s_async = RADSScheduler(
            budget=50.0, v_max=1000,
            straggler_aware=True, straggler_model=sm, deadline=1000.0,
        )

        r_sync = s_sync.schedule(devices)
        r_async = s_async.schedule(devices)

        # With huge deadline, pi_i ≈ 1 for all → results ~same
        # Allow some tolerance because pi_i might be 0.999... not 1.0
        assert len(r_async.selected_ids) >= len(r_sync.selected_ids) - 2
        assert r_async.total_quality >= r_sync.total_quality * 0.95


class TestBudgetConstraint:
    """Greedy always respects budget, even with straggler modifications."""

    def test_cost_within_budget(self):
        devices = _make_devices(30, seed=42)
        sm = StragglerModel(sigma_noise=1.0, seed=42)
        budget = 30.0

        for straggler_aware in [False, True]:
            kwargs = {'budget': budget, 'v_max': 500}
            if straggler_aware:
                kwargs.update(
                    straggler_aware=True, straggler_model=sm, deadline=5.0,
                )
            s = RADSScheduler(**kwargs)
            r = s.schedule(devices)
            assert r.total_cost <= budget + 1e-6, \
                f"Cost {r.total_cost} exceeds budget {budget} " \
                f"(straggler_aware={straggler_aware})"
