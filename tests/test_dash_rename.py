#!/usr/bin/env python3
"""Quick test to verify the DASH rename + rho_tilde substitution."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.scheduler.rads import RADSScheduler, DASHScheduler
from src.async_module.straggler_model import StragglerModel


def test_sync_scheduling():
    """Test: Without straggler_aware, rho_tilde_i == rho_i."""
    scheduler = RADSScheduler(budget=10.0, v_max=200, straggler_aware=False)
    devices = [
        {'device_id': 0, 'rho_i': 0.8, 'b_i': 0.1, 'theta_i': 50, 'a_i': 0.5},
        {'device_id': 1, 'rho_i': 0.6, 'b_i': 0.15, 'theta_i': 40, 'a_i': 0.5},
        {'device_id': 2, 'rho_i': 0.9, 'b_i': 0.12, 'theta_i': 60, 'a_i': 0.5},
    ]
    result = scheduler.schedule(devices)
    print(f'Sync: selected={result.n_selected}, Q={result.total_quality:.4f}')
    for a in result.allocations:
        if a.selected:
            print(f'  dev {a.device_id}: v*={a.v_star:.1f}, q={a.quality:.4f}')
    assert result.n_selected > 0, 'No devices selected in sync mode'
    assert result.total_quality > 0, 'Zero quality in sync mode'
    # Verify rho_tilde == rho in non-straggler mode
    for d in devices:
        assert abs(d['rho_tilde_i'] - d['rho_i']) < 1e-10, \
            f"rho_tilde_i should equal rho_i when straggler_aware=False"
    print('  PASS: sync scheduling')


def test_async_scheduling():
    """Test: With straggler_aware=True, rho_tilde_i = pi_i * rho_i."""
    sm = StragglerModel(sigma_noise=0.5)
    scheduler = RADSScheduler(
        budget=10.0, v_max=200,
        straggler_aware=True, straggler_model=sm, deadline=50.0,
    )
    devices = [
        {'device_id': 0, 'rho_i': 0.8, 'b_i': 0.1, 'theta_i': 50, 'a_i': 0.5,
         'comp_rate': 0.01, 'comm_rate': 0.01},
        {'device_id': 1, 'rho_i': 0.6, 'b_i': 0.15, 'theta_i': 40, 'a_i': 0.5,
         'comp_rate': 0.05, 'comm_rate': 0.04},   # slow device
        {'device_id': 2, 'rho_i': 0.9, 'b_i': 0.12, 'theta_i': 60, 'a_i': 0.5,
         'comp_rate': 0.005, 'comm_rate': 0.008},  # fast device
    ]
    result = scheduler.schedule(devices)
    print(f'\nAsync: selected={result.n_selected}, Q_tilde={result.total_quality:.4f}')
    for d in devices:
        rt = d.get('rho_tilde_i', d['rho_i'])
        print(f'  dev {d["device_id"]}: rho={d["rho_i"]:.2f}, rho_tilde={rt:.4f}')
    for a in result.allocations:
        if a.selected:
            print(f'  dev {a.device_id}: v*={a.v_star:.1f}, q_tilde={a.quality:.4f}')
    assert result.n_selected > 0, 'No devices selected in async mode'
    # The slow device (id=1) should have lower rho_tilde than rho
    dev1 = next(d for d in devices if d['device_id'] == 1)
    assert dev1['rho_tilde_i'] < dev1['rho_i'], \
        'Slow device should have rho_tilde < rho'
    # The fast device (id=2) should have rho_tilde close to rho
    dev2 = next(d for d in devices if d['device_id'] == 2)
    assert dev2['rho_tilde_i'] <= dev2['rho_i'], \
        'Fast device rho_tilde should be <= rho'
    print('  PASS: async scheduling with rho_tilde substitution')


def test_aliases():
    """Test backward-compat aliases."""
    assert RADSScheduler is DASHScheduler
    from src.methods.dash import DASH, DASHConfig
    from src.methods.async_kaas_edge import AsyncKaaSEdge, AsyncKaaSEdgeConfig
    assert AsyncKaaSEdge is DASH
    assert AsyncKaaSEdgeConfig is DASHConfig
    from src.methods import DASH as D2, DASHConfig as DC2
    assert D2 is DASH
    assert DC2 is DASHConfig
    print('\n  PASS: all aliases correct')


if __name__ == '__main__':
    test_sync_scheduling()
    test_async_scheduling()
    test_aliases()
    print('\n=== All DASH rename tests passed ===')
