"""Bit Allocation Engine — Data models.

Dataclasses that define the core structures used throughout the engine:
block profiles, allocation results, thresholds, and configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Precision(IntEnum):
    """Supported quantization bit-widths."""

    INT2 = 2
    INT4 = 4
    INT8 = 8
    INT16 = 16


@dataclass(frozen=True, slots=True)
class BlockProfile:
    """Statistical profile for a single block.

    Attributes:
        mean:   μ_i — arithmetic mean of the block values.
        std:    σ_i — standard deviation.
        min_val: min_i — minimum value in the block.
        max_val: max_i — maximum value in the block.
        outlier_score: o_i = max|B_i - μ_i| / (σ_i + ε) — outlier measure.
    """

    mean: float
    std: float
    min_val: float
    max_val: float
    outlier_score: float


@dataclass(frozen=True, slots=True)
class Allocation:
    """Result of the bit-allocation decision for one block.

    Attributes:
        block_id:  Identifier or index of the block.
        num_elements: Number of elements in the block.
        profile:   The computed statistical profile.
        score:     The composite score S_i = αR_i + βO_i.
        precision: The selected quantization bit-width.
    """

    block_id: int
    num_elements: int
    profile: BlockProfile
    score: float
    precision: Precision

    def __post_init__(self) -> None:
        if not isinstance(self.num_elements, int) or self.num_elements <= 0:
            raise ValueError(
                f"'num_elements' must be a positive integer, got {self.num_elements!r}"
            )


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Score thresholds that determine precision boundaries.

    The mapping is:
        S < int4  → INT2
        int4 ≤ S < int8  → INT4
        int8 ≤ S < int16 → INT8
        S ≥ int16         → INT16
    """

    int4: float = 1.0
    int8: float = 3.0
    int16: float = 6.0

    def __post_init__(self) -> None:
        for name, val in (("int4", self.int4), ("int8", self.int8), ("int16", self.int16)):
            if not math.isfinite(val):
                raise ValueError(f"Threshold '{name}' must be finite, got {val}")
        if not (self.int4 < self.int8 < self.int16):
            raise ValueError(
                f"Thresholds must be strictly increasing: "
                f"int4={self.int4} < int8={self.int8} < int16={self.int16}"
            )


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Full configuration for the BitAllocationEngine.

    Attributes:
        alpha:   Weight for the relative-variability component R_i.
        beta:    Weight for the outlier component O_i.
        epsilon: Small constant for numerical stability.
        thresholds: Score → precision mapping boundaries.
    """

    alpha: float = 1.0
    beta: float = 0.5
    epsilon: float = 1e-8
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        for name, val in (("alpha", self.alpha), ("beta", self.beta), ("epsilon", self.epsilon)):
            if not math.isfinite(val):
                raise ValueError(f"'{name}' must be finite, got {val}")
            if val < 0:
                raise ValueError(f"'{name}' must be non-negative, got {val}")
        if self.epsilon <= 0:
            raise ValueError(
                f"'epsilon' must be strictly positive, got {self.epsilon}"
            )
