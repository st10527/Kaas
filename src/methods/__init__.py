"""
Federated learning methods for KaaS-Edge.

Available methods:
  - KaaSEdge:  Our method (RADS water-filling + greedy selection + LDP)
  - FedMD:     Classic FD baseline (no privacy, no game)
  - FedAvg:    Parameter averaging baseline (no distillation)
  - CSRA:      Reverse auction DPFL baseline (Yang et al., TIFS 2024)
  - FedGMKD:   Prototype-based FL with GMM + DAT (Zhang et al., 2024)
"""

from .base import FederatedMethod, RoundResult
from .kaas_edge import KaaSEdge, KaaSEdgeConfig
from .fedmd import FedMD, FedMDConfig
from .fedavg import FedAvg, FedAvgConfig
from .csra import CSRA, CSRAConfig
from .fedgmkd import FedGMKD, FedGMKDConfig

__all__ = [
    "FederatedMethod",
    "RoundResult",
    "KaaSEdge",
    "KaaSEdgeConfig",
    "FedMD",
    "FedMDConfig",
    "FedAvg",
    "FedAvgConfig",
    "CSRA",
    "CSRAConfig",
    "FedGMKD",
    "FedGMKDConfig",
]
