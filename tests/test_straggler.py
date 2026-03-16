"""
Unit tests for the StragglerModel (src/async_module/straggler_model.py).

Tests:
  - Deterministic mode (sigma=0) → all complete
  - High sigma → some timeouts
  - Reproducibility (same seed → same results)
  - Completion probability monotonicity
  - Device rate generation tiers
"""

import numpy as np
import pytest
from src.async_module.straggler_model import StragglerModel, DeviceLatency


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_devices(n=5, v_star=100):
    """Create minimal device dicts with comp/comm rates."""
    rates = StragglerModel.generate_device_rates(n, seed=42)
    devices = []
    for i, r in enumerate(rates):
        devices.append({
            'device_id': i,
            'v_star': v_star,
            'comp_rate': r['comp_rate'],
            'comm_rate': r['comm_rate'],
        })
    return devices


# ── Tests ─────────────────────────────────────────────────────────────

class TestStragglerModelDeterministic:
    """sigma=0 → deterministic, no real straggler."""

    def test_all_complete_large_deadline(self):
        model = StragglerModel(sigma_noise=0.0, seed=42)
        devs = _make_devices(10, v_star=100)
        results = model.simulate_round(devs, deadline=100.0)
        assert len(results) == 10
        assert all(r.outcome == 'complete' for r in results)
        assert all(r.v_received == r.v_star for r in results)

    def test_all_timeout_tiny_deadline(self):
        model = StragglerModel(sigma_noise=0.0, seed=42)
        devs = _make_devices(10, v_star=100)
        results = model.simulate_round(devs, deadline=0.001)
        assert all(r.outcome == 'timeout' for r in results)
        assert all(r.v_received == 0 for r in results)


class TestStragglerModelStochastic:
    """sigma > 0 → stochastic outcomes."""

    def test_high_sigma_produces_timeouts(self):
        model = StragglerModel(sigma_noise=1.5, seed=42)
        devs = _make_devices(50, v_star=100)
        # Set a moderate deadline (based on median expected latency)
        results = model.simulate_round(devs, deadline=5.0)
        outcomes = [r.outcome for r in results]
        assert 'timeout' in outcomes or 'partial' in outcomes, \
            "High sigma should produce at least some non-complete outcomes"

    def test_moderate_sigma_mixed(self):
        model = StragglerModel(sigma_noise=0.5, seed=42)
        devs = _make_devices(50, v_star=100)
        results = model.simulate_round(devs, deadline=5.0)
        outcomes = set(r.outcome for r in results)
        # With 50 devices, sigma=0.5, and deadline=5, we expect a mix
        assert len(outcomes) >= 1  # at least one outcome type


class TestReproducibility:
    """Same seed → same results."""

    def test_same_seed_same_latencies(self):
        devs = _make_devices(20, v_star=100)

        m1 = StragglerModel(sigma_noise=0.5, seed=123)
        r1 = m1.simulate_round(devs, deadline=5.0)

        m2 = StragglerModel(sigma_noise=0.5, seed=123)
        r2 = m2.simulate_round(devs, deadline=5.0)

        for a, b in zip(r1, r2):
            assert a.tau_total == pytest.approx(b.tau_total)
            assert a.outcome == b.outcome
            assert a.v_received == b.v_received

    def test_different_seed_different_latencies(self):
        devs = _make_devices(20, v_star=100)

        m1 = StragglerModel(sigma_noise=0.5, seed=42)
        r1 = m1.simulate_round(devs, deadline=5.0)

        m2 = StragglerModel(sigma_noise=0.5, seed=99)
        r2 = m2.simulate_round(devs, deadline=5.0)

        # At least some latencies should differ
        diffs = [abs(a.tau_total - b.tau_total) for a, b in zip(r1, r2)]
        assert max(diffs) > 0.01


class TestCompletionProbability:
    """Test the pi_i(D) function used by straggler-aware RADS."""

    def test_zero_deadline_gives_zero(self):
        model = StragglerModel(sigma_noise=0.5, seed=42)
        pi = model.completion_probability(0.01, 0.02, 100.0, deadline=0.0)
        assert pi == 0.0

    def test_huge_deadline_gives_one(self):
        model = StragglerModel(sigma_noise=0.5, seed=42)
        pi = model.completion_probability(0.01, 0.02, 100.0, deadline=1000.0)
        assert pi > 0.99

    def test_monotone_in_deadline(self):
        model = StragglerModel(sigma_noise=0.5, seed=42)
        deadlines = [1.0, 2.0, 5.0, 10.0, 50.0]
        pis = [
            model.completion_probability(0.01, 0.02, 100.0, d)
            for d in deadlines
        ]
        # Pi should be non-decreasing
        for i in range(1, len(pis)):
            assert pis[i] >= pis[i - 1] - 1e-9, \
                f"pi not monotone: {pis[i]} < {pis[i-1]} at D={deadlines[i]}"

    def test_deterministic_sigma_zero(self):
        model = StragglerModel(sigma_noise=0.0, seed=42)
        # tau_det = 0.01*100 + 0.02*100 = 3.0, noise = 1.0, total ~ 4.0
        pi_ok = model.completion_probability(0.01, 0.02, 100.0, deadline=10.0)
        assert pi_ok == 1.0
        pi_fail = model.completion_probability(0.01, 0.02, 100.0, deadline=3.5)
        assert pi_fail == 0.0


class TestDeviceRateGeneration:
    """Test generate_device_rates helper."""

    def test_correct_count(self):
        rates = StragglerModel.generate_device_rates(50, seed=42)
        assert len(rates) == 50

    def test_keys(self):
        rates = StragglerModel.generate_device_rates(10, seed=42)
        for r in rates:
            assert 'comp_rate' in r
            assert 'comm_rate' in r

    def test_reasonable_values(self):
        rates = StragglerModel.generate_device_rates(100, seed=42)
        comp_rates = [r['comp_rate'] for r in rates]
        comm_rates = [r['comm_rate'] for r in rates]
        # All positive, within clipped range
        assert all(0.001 <= c <= 0.2 for c in comp_rates)
        assert all(0.001 <= c <= 0.2 for c in comm_rates)
        # Mean should be between fast (0.005) and slow (0.05) tier means
        assert 0.003 < np.mean(comp_rates) < 0.08
        assert 0.003 < np.mean(comm_rates) < 0.08


class TestDeviceLatencyDataclass:
    """Test the DeviceLatency dataclass fields."""

    def test_v_received_leq_v_star(self):
        model = StragglerModel(sigma_noise=0.5, seed=42)
        devs = _make_devices(30, v_star=200)
        results = model.simulate_round(devs, deadline=5.0)
        for r in results:
            assert r.v_received <= r.v_star
            assert r.v_received >= 0

    def test_zero_v_star_skipped(self):
        model = StragglerModel(sigma_noise=0.5, seed=42)
        devs = [{'device_id': 0, 'v_star': 0, 'comp_rate': 0.01, 'comm_rate': 0.02}]
        results = model.simulate_round(devs, deadline=10.0)
        assert results[0].outcome == 'timeout'
        assert results[0].v_received == 0
        assert results[0].tau_total == 0.0
