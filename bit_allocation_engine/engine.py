"""Bit Allocation Engine — Core engine.

Implements the full pipeline:
    block → profile → score → precision
"""

from __future__ import annotations

import math
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

    .. warning:: **Heuristic thresholds — not calibrated guarantees.**

       The default score thresholds (INT4 ≥ 1.0, INT8 ≥ 3.0, INT16 ≥ 6.0) are
       hand-chosen heuristics that work well on synthetic benchmarks but have
       **not** been validated against real model weights.  They do not define a
       quantization scheme and they do not bound reconstruction error (MSE,
       max error, etc.).

       For production use you should:

       1. Profile representative data with :pymethod:`compute_profile` /
          :pymethod:`compute_score`.
       2. Quantize + dequantize at each candidate precision and measure the
          actual reconstruction error.
       3. Set thresholds so that the assigned precision keeps error within an
          explicit budget.

       The :pymethod:`calibrate_thresholds` method defines the intended
       signature for this workflow but is **not implemented** — you must
       supply the body (or subclass) with your domain-specific error
       function.

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

        # Validate resolved values (kwargs may bypass EngineConfig validation)
        for name, val in (("alpha", self._alpha), ("beta", self._beta), ("epsilon", self._epsilon)):
            if not math.isfinite(val):
                raise ValueError(f"'{name}' must be finite, got {val}")
            if val < 0:
                raise ValueError(f"'{name}' must be non-negative, got {val}")
        if self._epsilon <= 0:
            raise ValueError(
                f"'epsilon' must be strictly positive, got {self._epsilon}"
            )

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
            If the block is empty, contains non-finite values
            (NaN or ±inf), or if intermediate statistics overflow
            (e.g. extreme-magnitude data near float64 limits).

        Notes
        -----
        The outlier score uses the maximum deviation from the mean.
        For a given distribution, larger blocks tend to produce larger
        maximum deviations (an extreme-value effect), so block size
        may influence the assigned precision when sizes vary widely.

        Numerical stability: when the input values are very large in
        magnitude, intermediate computations (variance, deviations)
        can overflow float64.  The implementation shifts data by the
        mean before squaring to reduce overflow risk and raises a
        clear ``ValueError`` if overflow still occurs.
        """
        arr = np.asarray(block, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("Block must contain at least one element.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "Block contains non-finite values (NaN or ±inf)."
            )

        mean = float(np.mean(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        # Shift by mean before computing std and deviations to reduce
        # the magnitude of squared terms and avoid overflow.
        centered = arr - mean
        std = float(np.sqrt(np.mean(centered * centered)))

        # o_i = max|B_i - μ_i| / (σ_i + ε)
        max_deviation = float(np.max(np.abs(centered)))
        outlier_score = max_deviation / (std + self._epsilon)

        # Guard: if any statistic overflowed despite shifting, reject the
        # block rather than silently propagating inf/nan downstream.
        computed = {"mean": mean, "std": std, "min_val": min_val,
                    "max_val": max_val, "outlier_score": outlier_score}
        non_finite = [k for k, v in computed.items() if not math.isfinite(v)]
        if non_finite:
            raise ValueError(
                f"Numerical overflow while profiling block: the following "
                f"statistics are non-finite: {', '.join(non_finite)}. "
                f"This typically happens with extreme-magnitude data near "
                f"float64 limits (~1.7e308)."
            )

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

        Raises
        ------
        ValueError
            If *score* is not finite (NaN or ±inf).
        """
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score: {score}")
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

        # Recover element count from block (profile doesn't store it).
        num_elements = np.asarray(block).size

        return Allocation(
            block_id=block_id,
            num_elements=num_elements,
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

    # -- calibration ---------------------------------------------------------

    @classmethod
    def calibrate_thresholds(
        cls,
        blocks: Sequence[ArrayLike],
        error_fn: object,
        *,
        error_budget: float = 0.01,
        alpha: float = 1.0,
        beta: float = 0.5,
        epsilon: float = 1e-8,
    ) -> Thresholds:
        """Extension point: derive score thresholds from representative data.

        This method is **not implemented**.  It defines the intended
        signature for data-driven calibration so that subclasses or
        call-sites can follow a consistent contract.  A typical
        implementation would:

        1. Profile every block and compute its score with the given α, β, ε.
        2. For each block, call ``error_fn(block, precision)`` to measure
           reconstruction error at each candidate precision (INT2 … INT16).
        3. For each precision boundary, find the score value that separates
           blocks whose error is within ``error_budget`` at the lower
           precision from those that are not.

        Parameters
        ----------
        blocks : sequence of array-like
            Representative weight / activation blocks.
        error_fn : callable
            ``error_fn(block, precision) -> float`` — returns the
            reconstruction error when *block* is quantized at *precision*.
        error_budget : float
            Maximum acceptable reconstruction error per block.
        alpha, beta, epsilon : float
            Engine hyper-parameters used during profiling.

        Returns
        -------
        Thresholds
            Data-derived thresholds ready to pass to ``EngineConfig``.

        Raises
        ------
        NotImplementedError
            Always — override in a subclass or monkey-patch with your
            domain-specific logic.
        """
        raise NotImplementedError(
            "calibrate_thresholds() is not implemented.  Override this "
            "method with your domain-specific error function "
            "(e.g. quantize → dequantize → MSE) and an explicit "
            "error_budget.  See the docstring for the expected contract."
        )
