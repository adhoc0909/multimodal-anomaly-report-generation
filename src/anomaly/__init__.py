"""Anomaly detection module.

Provides unified interface for anomaly detection models:
- shared base interfaces
"""
from .base import (
    AnomalyResult,
    BaseAnomalyModel,
    PerClassAnomalyModel,
    UnifiedAnomalyModel,
)

__all__ = [
    # Base classes
    "AnomalyResult",
    "BaseAnomalyModel",
    "PerClassAnomalyModel",
    "UnifiedAnomalyModel",
]
