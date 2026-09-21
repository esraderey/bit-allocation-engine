"""Bit Allocation Engine.

A pre-processing quantization engine that determines bit-width precision
(INT2 / INT4 / INT8 / INT16) based on cheap statistical block profiling.
"""

from .analysis import (
    estimate_compression_ratio,
    find_critical_blocks,
    summarize_allocations,
)
from .engine import BitAllocationEngine
from .models import (
    Allocation,
    BlockProfile,
    EngineConfig,
    Precision,
    Thresholds,
)

__all__ = [
    # Core
    "BitAllocationEngine",
    # Models
    "Allocation",
    "BlockProfile",
    "EngineConfig",
    "Precision",
    "Thresholds",
    # Analysis
    "estimate_compression_ratio",
    "find_critical_blocks",
    "summarize_allocations",
]
