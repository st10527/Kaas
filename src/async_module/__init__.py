"""
Async-RADS: Asynchronous extensions for KaaS-Edge.

Modules:
    straggler_model: Device latency simulation with LogNormal noise
    timeout_policy:  Three deadline policies (fixed / adaptive / partial-accept)
"""

from .straggler_model import StragglerModel, DeviceLatency
from .timeout_policy import (
    TimeoutPolicy,
    FixedDeadlinePolicy,
    AdaptiveDeadlinePolicy,
    PartialAcceptPolicy,
    create_timeout_policy,
)

__all__ = [
    'StragglerModel', 'DeviceLatency',
    'TimeoutPolicy', 'FixedDeadlinePolicy',
    'AdaptiveDeadlinePolicy', 'PartialAcceptPolicy',
    'create_timeout_policy',
]
