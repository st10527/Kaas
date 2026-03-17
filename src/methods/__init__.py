"""
Federated learning methods.

Available methods:
  - KaaSEdge:  Synchronous scheduler (IEEE EDGE 2026)
  - DASH:      Straggler-tolerant async scheduler (JPDC 2026)
  - FedMD:     Classic FD baseline (no privacy, no game)
  - FedAvg:    Parameter averaging baseline (no distillation)
  - CSRA:      Reverse auction DPFL baseline (Yang et al., TIFS 2024)
  - FedGMKD:   Prototype-based FL with GMM + DAT (Zhang et al., 2024)
"""

from .base import FederatedMethod, RoundResult
from .kaas_edge import KaaSEdge, KaaSEdgeConfig
from .dash import DASH, DASHConfig, FullAsyncFD, RandomAsyncFD
from .fedmd import FedMD, FedMDConfig
from .fedavg import FedAvg, FedAvgConfig
from .csra import CSRA, CSRAConfig
from .fedgmkd import FedGMKD, FedGMKDConfig

__all__ = [
    "FederatedMethod",
    "RoundResult",
    "KaaSEdge",
    "KaaSEdgeConfig",
    "DASH",
    "DASHConfig",
    "FullAsyncFD",
    "RandomAsyncFD",
    "FedMD",
    "FedMDConfig",
    "FedAvg",
    "FedAvgConfig",
    "CSRA",
    "CSRAConfig",
    "FedGMKD",
    "FedGMKDConfig",
]
