"""Bit Allocation Engine — Core engine.

Implements the full pipeline:
    block → profile → score → precision
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .models import Allocation, BlockProfile, EngineConfig, Precision, Thresholds


class BitAllocationEngine:
    """Assigns quantization bit-widths to blocks based on cheap statistical profiles.

    The engine computes, for each block B_i:

        Profile   P_i = (μ, σ, min, max, o)
        Score     S_i = α·R_i + β·O_i
        Precision p_i = f(S_i) ∈ {2, 4, 8, 16}

    where:
        R_i = σ_i / (max(|μ_i|, σ_i) + ε) — coefficient of variation
        O_i = max|B_i − μ_i| / (σ_i + ε)  — outlier measure

    Parameters
    ----------
    config : EngineConfig, optional
        Full engine configuration.  Individual keyword arguments (``alpha``,
        ``beta``, ``epsilon``, ``thresholds``) override the corresponding
        config fields when both are provided.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        epsilon: float | None = None,
        thresholds: Thresholds | None = None,
    ) -> None:
        cfg = config or EngineConfig()
        self._alpha = alpha if alpha is not None else cfg.alpha
        self._beta = beta if beta is not None else cfg.beta
        self._epsilon = epsilon if epsilon is not None else cfg.epsilon
        self._thresholds = thresholds if thresholds is not None else cfg.thresholds

    # -- public properties ---------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    # -- core pipeline -------------------------------------------------------

    def compute_profile(self, block: ArrayLike) -> BlockProfile:
        """Compute the statistical profile P_i for a single block.

        Parameters
        ----------
        block : array-like
            1-D or N-D array of numerical values representing the block.

        Returns
        -------
        BlockProfile
            The computed (μ, σ, min, max, o) profile.

        Raises
        ------
        ValueError
            If the block is empty.
        """
        arr = np.asarray(block, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("Block must contain at least one element.")

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        # o_i = max|B_i - μ_i| / (σ_i + ε)
        max_deviation = float(np.max(np.abs(arr - mean)))
        outlier_score = max_deviation / (std + self._epsilon)

        return BlockProfile(
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            outlier_score=outlier_score,
        )

    def compute_score(self, profile: BlockProfile) -> float:
        """Compute the composite score S_i from a block profile.

        S_i = α · R_i + β · O_i

        where:
            R_i = σ_i / (max(|μ_i|, σ_i) + ε)
            O_i = profile.outlier_score

        Parameters
        ----------
        profile : BlockProfile
            Pre-computed block statistics.

        Returns
        -------
        float
            The composite score.
        """
        r_i = profile.std / (max(abs(profile.mean), profile.std) + self._epsilon)
        o_i = profile.outlier_score
        return self._alpha * r_i + self._beta * o_i

    def select_precision(self, score: float) -> Precision:
        """Map a composite score to a quantization precision.

        Parameters
        ----------
        score : float
            The composite score S_i.

        Returns
        -------
        Precision
            One of INT2, INT4, INT8, INT16.
        """
        if score < self._thresholds.int4:
            return Precision.INT2
        if score < self._thresholds.int8:
            return Precision.INT4
        if score < self._thresholds.int16:
            return Precision.INT8
        return Precision.INT16

    def allocate_single(self, block: ArrayLike, block_id: int = 0) -> Allocation:
        """Run the full pipeline on a single block.

        Parameters
        ----------
        block : array-like
            Numerical data for one block.
        block_id : int, optional
            Identifier for the block (default 0).

        Returns
        -------
        Allocation
            The allocation result including profile, score, and precision.
        """
        profile = self.compute_profile(block)
        score = self.compute_score(profile)
        precision = self.select_precision(score)
        return Allocation(
            block_id=block_id,
            profile=profile,
            score=score,
            precision=precision,
        )

    def allocate(self, blocks: Sequence[ArrayLike]) -> list[Allocation]:
        """Run the full pipeline on a sequence of blocks.

        Parameters
        ----------
        blocks : sequence of array-like
            Each element is one block to profile and allocate.

        Returns
        -------
        list[Allocation]
            One allocation per block, in the same order.
        """
        return [
            self.allocate_single(block, block_id=i)
            for i, block in enumerate(blocks)
        ]
