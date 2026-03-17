"""
Backward-compatibility shim.

All classes have moved to src.methods.dash.  This module re-exports
them so that existing scripts with ``from src.methods.async_kaas_edge
import AsyncKaaSEdge`` continue to work.
"""

from .dash import (                     # noqa: F401
    DASH,
    DASH as AsyncKaaSEdge,
    DASHConfig,
    DASHConfig as AsyncKaaSEdgeConfig,
    FullAsyncFD,
    RandomAsyncFD,
)

__all__ = [
    'DASH', 'DASHConfig',
    'AsyncKaaSEdge', 'AsyncKaaSEdgeConfig',  # backward compat
    'FullAsyncFD', 'RandomAsyncFD',
]
