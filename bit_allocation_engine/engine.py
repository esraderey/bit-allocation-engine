"""Bit Allocation Engine — Core engine.

Implements the full pipeline:
    block → profile → score → precision
"""

from __future__ import annotations

import math
import numbers
import warnings
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

    .. note:: **The score is bounded by the block size.**

       Because ``R_i < 1`` and ``O_i <= sqrt(n - 1)`` for a block of ``n``
       elements, the composite score can never exceed
       ``alpha + beta * sqrt(n - 1)`` (see :pymethod:`max_score`).  With the
       default configuration a block of fewer than 18 elements can never be
       assigned INT8 and a block of fewer than 102 elements can never be
       assigned INT16, whatever its contents.  Conversely, for a fixed
       distribution the maximum deviation grows with ``n`` (an extreme-value
       effect), so larger blocks drift towards higher precision.  Scores are
       therefore only comparable between blocks of equal size:
       :pymethod:`allocate` enforces a uniform block size and warns when a
       precision is unreachable for that size.

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
        if not isinstance(self._thresholds, Thresholds):
            raise TypeError(
                f"'thresholds' must be a Thresholds instance, "
                f"got {type(self._thresholds).__name__}"
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
        TypeError
            If the block holds complex or boolean values.  Only real
            numbers are profiled; a complex array would otherwise be
            silently truncated to its real part.
        ValueError
            If the block is a scalar (0-d), is empty, contains non-finite
            values (NaN or ±inf), or if intermediate statistics overflow
            (e.g. extreme-magnitude data near float64 limits).

        Notes
        -----
        The outlier score uses the maximum deviation from the mean, so it
        is bounded by ``sqrt(n - 1)`` and, for a given distribution, grows
        with the block size.  See the class docstring and
        :pymethod:`max_score`.

        Numerical stability: when the input values are very large in
        magnitude, intermediate computations (variance, deviations)
        can overflow float64.  The implementation shifts data by the
        mean before squaring to reduce overflow risk and raises a
        clear ``ValueError`` if overflow still occurs.  At the other
        extreme, ``epsilon`` is an absolute constant: for data whose
        standard deviation is below roughly ``1e-5`` it stops being
        negligible and the score is no longer scale-invariant.
        """
        raw = np.asarray(block)
        if raw.dtype == np.bool_ or np.iscomplexobj(raw):
            raise TypeError(
                f"Block must contain real numbers, got dtype {raw.dtype}."
            )
        if raw.ndim == 0:
            raise ValueError(
                "Block must be an array of values, got a scalar. If you "
                "passed a flat array to allocate(), wrap it in a list or "
                "split it into blocks first."
            )
        arr = np.asarray(raw, dtype=np.float64).ravel()
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

    def max_score(self, num_elements: int) -> float:
        """Upper bound of the composite score for a block of a given size.

        ``R_i < 1`` always, and ``O_i = max|B_i - μ| / (σ + ε)`` is bounded
        by ``sqrt(n - 1)`` for the population standard deviation, so

            S_i < α + β · sqrt(n − 1)

        A precision whose threshold is at or above this bound can never be
        assigned to blocks of ``num_elements`` elements.

        Parameters
        ----------
        num_elements : int
            Number of elements per block (must be a positive integer).

        Returns
        -------
        float
            The (strict) upper bound of the score.
        """
        if (
            isinstance(num_elements, bool)
            or not isinstance(num_elements, numbers.Integral)
            or num_elements < 1
        ):
            raise ValueError(
                f"'num_elements' must be a positive integer, got {num_elements!r}"
            )
        return self._alpha + self._beta * math.sqrt(num_elements - 1)

    def _warn_if_unreachable(self, num_elements: int) -> None:
        """Warn when the current thresholds make a precision unreachable."""
        bound = self.max_score(num_elements)
        unreachable = [
            precision.name
            for precision, threshold in (
                (Precision.INT4, self._thresholds.int4),
                (Precision.INT8, self._thresholds.int8),
                (Precision.INT16, self._thresholds.int16),
            )
            if bound <= threshold
        ]
        if unreachable:
            warnings.warn(
                f"With blocks of {num_elements} elements the composite score "
                f"is bounded by {bound:.3f}, so {', '.join(unreachable)} can "
                f"never be assigned under the current thresholds. Use larger "
                f"blocks or lower thresholds (see BitAllocationEngine.max_score).",
                UserWarning,
                stacklevel=3,
            )

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
            Each element is one block to profile and allocate.  All blocks
            must have the same number of elements, because the score is
            bounded by (and drifts with) the block size and is only
            comparable between blocks of equal size.

        Returns
        -------
        list[Allocation]
            One allocation per block, in the same order.

        Raises
        ------
        ValueError
            If the blocks do not all have the same number of elements.

        Warns
        -----
        UserWarning
            If the block size makes some precision unreachable under the
            current thresholds (see :pymethod:`max_score`).
        """
        allocations: list[Allocation] = []
        expected: int | None = None
        for i, block in enumerate(blocks):
            allocation = self.allocate_single(block, block_id=i)
            if expected is None:
                expected = allocation.num_elements
            elif allocation.num_elements != expected:
                raise ValueError(
                    f"All blocks must have the same number of elements: block 0 "
                    f"has {expected}, block {i} has {allocation.num_elements}. "
                    f"Scores are only comparable between blocks of equal size."
                )
            allocations.append(allocation)
        if expected is not None:
            self._warn_if_unreachable(expected)
        return allocations

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
