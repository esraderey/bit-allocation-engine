"""Tests for the Bit Allocation Engine."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from bit_allocation_engine import (
    Allocation,
    BitAllocationEngine,
    BlockProfile,
    EngineConfig,
    Precision,
    Thresholds,
    estimate_compression_ratio,
    find_critical_blocks,
    summarize_allocations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> BitAllocationEngine:
    """Default engine with standard config."""
    return BitAllocationEngine()


@pytest.fixture
def custom_engine() -> BitAllocationEngine:
    """Engine with custom thresholds for deterministic testing."""
    return BitAllocationEngine(
        alpha=1.0,
        beta=1.0,
        epsilon=1e-10,
        thresholds=Thresholds(int4=2.0, int8=5.0, int16=10.0),
    )


# ---------------------------------------------------------------------------
# BlockProfile computation
# ---------------------------------------------------------------------------


class TestComputeProfile:
    """Tests for BitAllocationEngine.compute_profile."""

    def test_known_values(self, engine: BitAllocationEngine) -> None:
        """Profile of a simple array should match hand-computed values."""
        block = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        profile = engine.compute_profile(block)

        assert profile.mean == pytest.approx(3.0)
        assert profile.std == pytest.approx(np.std(block))
        assert profile.min_val == 1.0
        assert profile.max_val == 5.0

        # max|B_i - μ| = max(|1-3|, |2-3|, |3-3|, |4-3|, |5-3|) = 2.0
        expected_outlier = 2.0 / (np.std(block) + engine.epsilon)
        assert profile.outlier_score == pytest.approx(expected_outlier)

    def test_constant_block(self, engine: BitAllocationEngine) -> None:
        """A constant block should have σ=0, outlier ≈ 0."""
        block = np.full(100, 42.0)
        profile = engine.compute_profile(block)

        assert profile.mean == pytest.approx(42.0)
        assert profile.std == pytest.approx(0.0)
        assert profile.min_val == 42.0
        assert profile.max_val == 42.0
        # max|B-μ| = 0, so outlier = 0 / (0 + ε) ≈ 0
        assert profile.outlier_score == pytest.approx(0.0)

    def test_single_element(self, engine: BitAllocationEngine) -> None:
        """A single-element block is a valid edge case."""
        profile = engine.compute_profile(np.array([7.5]))
        assert profile.mean == pytest.approx(7.5)
        assert profile.std == pytest.approx(0.0)

    def test_empty_block_raises(self, engine: BitAllocationEngine) -> None:
        """An empty block should raise ValueError."""
        with pytest.raises(ValueError, match="at least one element"):
            engine.compute_profile(np.array([]))

    def test_multidimensional_block(self, engine: BitAllocationEngine) -> None:
        """N-D arrays should be flattened automatically."""
        block_2d = np.array([[1, 2], [3, 4]])
        block_flat = np.array([1, 2, 3, 4])
        p2d = engine.compute_profile(block_2d)
        pflat = engine.compute_profile(block_flat)
        assert p2d.mean == pytest.approx(pflat.mean)
        assert p2d.std == pytest.approx(pflat.std)

    def test_negative_values(self, engine: BitAllocationEngine) -> None:
        """Blocks with negative values should work correctly."""
        block = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        profile = engine.compute_profile(block)
        assert profile.mean == pytest.approx(-1.0)
        assert profile.min_val == -5.0
        assert profile.max_val == 3.0


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


class TestComputeScore:
    """Tests for BitAllocationEngine.compute_score."""

    def test_low_variability_low_outlier(self, engine: BitAllocationEngine) -> None:
        """Score should be low when σ is small relative to μ and outliers are low."""
        profile = BlockProfile(mean=10.0, std=0.1, min_val=9.8, max_val=10.2, outlier_score=1.0)
        score = engine.compute_score(profile)
        # R = 0.1/max(10,0.1) = 0.1/10 = 0.01, O = 1.0 → S = 1.0*0.01 + 0.5*1.0 = 0.51
        assert score == pytest.approx(0.51, abs=0.01)

    def test_high_variability(self, engine: BitAllocationEngine) -> None:
        """Score should increase with relative variability."""
        low_var = BlockProfile(mean=10.0, std=0.1, min_val=9.8, max_val=10.2, outlier_score=1.0)
        high_var = BlockProfile(mean=10.0, std=5.0, min_val=0.0, max_val=20.0, outlier_score=1.0)

        s_low = engine.compute_score(low_var)
        s_high = engine.compute_score(high_var)
        assert s_high > s_low

    def test_high_outlier(self, engine: BitAllocationEngine) -> None:
        """Score should increase with outlier severity."""
        low_out = BlockProfile(mean=5.0, std=1.0, min_val=3.0, max_val=7.0, outlier_score=2.0)
        high_out = BlockProfile(mean=5.0, std=1.0, min_val=3.0, max_val=7.0, outlier_score=20.0)

        s_low = engine.compute_score(low_out)
        s_high = engine.compute_score(high_out)
        assert s_high > s_low

    def test_zero_mean_uses_epsilon(self, engine: BitAllocationEngine) -> None:
        """When μ ≈ 0, R_i should not blow up — ε stabilises the denominator."""
        profile = BlockProfile(mean=0.0, std=1.0, min_val=-2.0, max_val=2.0, outlier_score=2.0)
        score = engine.compute_score(profile)
        # R = 1.0 / max(0, 1.0) = 1.0 — ε-stabilised and bounded
        assert score == pytest.approx(1.0 * 1.0 + 0.5 * 2.0, abs=0.01)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Precision selection
# ---------------------------------------------------------------------------


class TestSelectPrecision:
    """Tests for BitAllocationEngine.select_precision."""

    def test_int2_range(self, engine: BitAllocationEngine) -> None:
        assert engine.select_precision(0.0) == Precision.INT2
        assert engine.select_precision(0.5) == Precision.INT2
        assert engine.select_precision(0.99) == Precision.INT2

    def test_int4_range(self, engine: BitAllocationEngine) -> None:
        assert engine.select_precision(1.0) == Precision.INT4
        assert engine.select_precision(2.0) == Precision.INT4
        assert engine.select_precision(2.99) == Precision.INT4

    def test_int8_range(self, engine: BitAllocationEngine) -> None:
        assert engine.select_precision(3.0) == Precision.INT8
        assert engine.select_precision(4.5) == Precision.INT8
        assert engine.select_precision(5.99) == Precision.INT8

    def test_int16_range(self, engine: BitAllocationEngine) -> None:
        assert engine.select_precision(6.0) == Precision.INT16
        assert engine.select_precision(100.0) == Precision.INT16

    def test_negative_score(self, engine: BitAllocationEngine) -> None:
        """Negative scores (theoretically impossible but defensively handled) → INT2."""
        assert engine.select_precision(-1.0) == Precision.INT2

    def test_custom_thresholds(self, custom_engine: BitAllocationEngine) -> None:
        assert custom_engine.select_precision(1.5) == Precision.INT2
        assert custom_engine.select_precision(2.0) == Precision.INT4
        assert custom_engine.select_precision(5.0) == Precision.INT8
        assert custom_engine.select_precision(10.0) == Precision.INT16


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestAllocate:
    """Tests for allocate / allocate_single."""

    def test_allocate_single(self, engine: BitAllocationEngine) -> None:
        """allocate_single should return a valid Allocation."""
        block = np.random.default_rng(42).normal(loc=5.0, scale=0.1, size=256)
        alloc = engine.allocate_single(block, block_id=7)

        assert isinstance(alloc, Allocation)
        assert alloc.block_id == 7
        assert alloc.precision in Precision
        assert np.isfinite(alloc.score)

    def test_allocate_multiple(self, engine: BitAllocationEngine) -> None:
        """allocate should handle a list of blocks and preserve order."""
        rng = np.random.default_rng(123)
        blocks = [rng.normal(0, 1, 128) for _ in range(10)]
        allocations = engine.allocate(blocks)

        assert len(allocations) == 10
        for i, alloc in enumerate(allocations):
            assert alloc.block_id == i

    def test_uniform_block_gets_low_precision(self, engine: BitAllocationEngine) -> None:
        """A nearly-constant block should be allocated INT2."""
        block = np.full(256, 100.0) + np.random.default_rng(0).uniform(-0.001, 0.001, 256)
        alloc = engine.allocate_single(block)
        assert alloc.precision == Precision.INT2

    def test_heavy_tail_gets_high_precision(self, engine: BitAllocationEngine) -> None:
        """A block with extreme outliers should get INT8 or INT16."""
        rng = np.random.default_rng(99)
        block = rng.normal(0, 1, 256)
        # Inject severe outliers
        block[0] = 100.0
        block[1] = -100.0
        alloc = engine.allocate_single(block)
        assert alloc.precision.value >= Precision.INT8.value


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


class TestAnalysis:
    """Tests for analysis utilities."""

    def _make_allocations(self) -> list[Allocation]:
        """Create a small set of known allocations."""
        dummy_profile = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)
        return [
            Allocation(block_id=0, num_elements=256, profile=dummy_profile, score=0.5, precision=Precision.INT2),
            Allocation(block_id=1, num_elements=256, profile=dummy_profile, score=1.5, precision=Precision.INT4),
            Allocation(block_id=2, num_elements=256, profile=dummy_profile, score=4.0, precision=Precision.INT8),
            Allocation(block_id=3, num_elements=256, profile=dummy_profile, score=8.0, precision=Precision.INT16),
        ]

    def test_summarize(self) -> None:
        allocations = self._make_allocations()
        summary = summarize_allocations(allocations)

        assert summary["total_blocks"] == 4
        assert summary["distribution"]["INT2"] == 1
        assert summary["distribution"]["INT4"] == 1
        assert summary["distribution"]["INT8"] == 1
        assert summary["distribution"]["INT16"] == 1
        # avg = (2+4+8+16) / 4 = 7.5
        assert summary["avg_bits"] == pytest.approx(7.5)

    def test_summarize_empty(self) -> None:
        summary = summarize_allocations([])
        assert summary["total_blocks"] == 0

    def test_compression_ratio(self) -> None:
        allocations = self._make_allocations()
        ratio = estimate_compression_ratio(allocations, original_bits=32)
        # avg = 7.5, ratio = 32/7.5 ≈ 4.27
        assert ratio == pytest.approx(32 / 7.5, abs=0.1)

    def test_compression_ratio_empty(self) -> None:
        assert estimate_compression_ratio([]) == 1.0

    def test_find_critical(self) -> None:
        allocations = self._make_allocations()
        critical = find_critical_blocks(allocations, min_precision=Precision.INT8)
        assert len(critical) == 2
        # Sorted by score descending
        assert critical[0].precision == Precision.INT16
        assert critical[1].precision == Precision.INT8


# ---------------------------------------------------------------------------
# Weighted avg_bits / compression (P2-compression)
# ---------------------------------------------------------------------------


class TestWeightedCompression:
    """Verify that avg_bits and compression ratio weight by element count."""

    def test_avg_bits_weighted_by_elements(self) -> None:
        """A large INT2 block should dominate a tiny INT16 block."""
        dummy = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)
        allocations = [
            Allocation(block_id=0, num_elements=1_000_000, profile=dummy, score=0.5, precision=Precision.INT2),
            Allocation(block_id=1, num_elements=1, profile=dummy, score=8.0, precision=Precision.INT16),
        ]
        summary = summarize_allocations(allocations)
        # Weighted: (2*1_000_000 + 16*1) / 1_000_001 ≈ 2.000014
        assert summary["avg_bits"] == pytest.approx(2.0, abs=0.01)

    def test_avg_bits_unweighted_would_be_wrong(self) -> None:
        """Without weighting, avg would be (2+16)/2 = 9.0, which is wrong."""
        dummy = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)
        allocations = [
            Allocation(block_id=0, num_elements=1_000_000, profile=dummy, score=0.5, precision=Precision.INT2),
            Allocation(block_id=1, num_elements=1, profile=dummy, score=8.0, precision=Precision.INT16),
        ]
        summary = summarize_allocations(allocations)
        # The old (broken) average would have been 9.0
        assert summary["avg_bits"] != pytest.approx(9.0, abs=0.1)

    def test_compression_ratio_weighted(self) -> None:
        """Compression ratio should reflect element-weighted avg_bits."""
        dummy = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)
        allocations = [
            Allocation(block_id=0, num_elements=1_000_000, profile=dummy, score=0.5, precision=Precision.INT2),
            Allocation(block_id=1, num_elements=1, profile=dummy, score=8.0, precision=Precision.INT16),
        ]
        ratio = estimate_compression_ratio(allocations, original_bits=32)
        # 32 / ~2.0 ≈ 16.0
        assert ratio == pytest.approx(16.0, abs=0.1)

    def test_equal_size_blocks_match_simple_average(self) -> None:
        """When all blocks have the same size, weighted == simple average."""
        dummy = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)
        allocations = [
            Allocation(block_id=0, num_elements=256, profile=dummy, score=0.5, precision=Precision.INT2),
            Allocation(block_id=1, num_elements=256, profile=dummy, score=1.5, precision=Precision.INT4),
            Allocation(block_id=2, num_elements=256, profile=dummy, score=4.0, precision=Precision.INT8),
            Allocation(block_id=3, num_elements=256, profile=dummy, score=8.0, precision=Precision.INT16),
        ]
        summary = summarize_allocations(allocations)
        # (2+4+8+16)/4 = 7.5 — same as simple average when sizes are equal
        assert summary["avg_bits"] == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Non-finite data handling (P2-NaN)
# ---------------------------------------------------------------------------


class TestNonFiniteHandling:
    """Tests for NaN and ±inf rejection."""

    def test_nan_in_block_raises(self, engine: BitAllocationEngine) -> None:
        """compute_profile should reject blocks with NaN."""
        block = np.array([1.0, 2.0, float("nan"), 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            engine.compute_profile(block)

    def test_inf_in_block_raises(self, engine: BitAllocationEngine) -> None:
        """compute_profile should reject blocks with ±inf."""
        block = np.array([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            engine.compute_profile(block)

    def test_neg_inf_in_block_raises(self, engine: BitAllocationEngine) -> None:
        block = np.array([1.0, float("-inf"), 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            engine.compute_profile(block)

    def test_select_precision_nan_raises(self, engine: BitAllocationEngine) -> None:
        """select_precision should reject NaN scores."""
        with pytest.raises(ValueError, match="Non-finite score"):
            engine.select_precision(float("nan"))

    def test_select_precision_inf_raises(self, engine: BitAllocationEngine) -> None:
        with pytest.raises(ValueError, match="Non-finite score"):
            engine.select_precision(float("inf"))


# ---------------------------------------------------------------------------
# Config validation (P2-config)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Tests for EngineConfig and Thresholds validation."""

    def test_epsilon_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon.*strictly positive"):
            EngineConfig(epsilon=0)

    def test_epsilon_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EngineConfig(epsilon=-1e-8)

    def test_alpha_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            EngineConfig(alpha=float("nan"))

    def test_beta_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            EngineConfig(beta=float("inf"))

    def test_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EngineConfig(alpha=-0.5)

    def test_thresholds_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Thresholds(int4=float("nan"), int8=3.0, int16=6.0)

    def test_thresholds_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Thresholds(int4=1.0, int8=float("inf"), int16=6.0)

    def test_engine_kwarg_epsilon_zero_raises(self) -> None:
        """Kwargs bypass EngineConfig — engine constructor must also validate."""
        with pytest.raises(ValueError, match="epsilon.*strictly positive"):
            BitAllocationEngine(epsilon=0)

    def test_engine_kwarg_alpha_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            BitAllocationEngine(alpha=float("nan"))


# ---------------------------------------------------------------------------
# Allocation includes num_elements
# ---------------------------------------------------------------------------


class TestAllocationNumElements:
    """Verify that allocate_single populates num_elements correctly."""

    def test_num_elements_populated(self, engine: BitAllocationEngine) -> None:
        block = np.random.default_rng(42).normal(0, 1, 128)
        alloc = engine.allocate_single(block)
        assert alloc.num_elements == 128

    def test_num_elements_2d(self, engine: BitAllocationEngine) -> None:
        block = np.ones((4, 8))
        alloc = engine.allocate_single(block)
        assert alloc.num_elements == 32


# ---------------------------------------------------------------------------
# Models validation (original tests preserved)
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for model dataclasses."""

    def test_thresholds_validation(self) -> None:
        """Thresholds must be strictly increasing."""
        with pytest.raises(ValueError, match="strictly increasing"):
            Thresholds(int4=5.0, int8=3.0, int16=10.0)

    def test_thresholds_equal_values(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            Thresholds(int4=3.0, int8=3.0, int16=6.0)

    def test_engine_config_defaults(self) -> None:
        cfg = EngineConfig()
        assert cfg.alpha == 1.0
        assert cfg.beta == 0.5
        assert cfg.epsilon == 1e-8

    def test_precision_enum_values(self) -> None:
        assert Precision.INT2 == 2
        assert Precision.INT4 == 4
        assert Precision.INT8 == 8
        assert Precision.INT16 == 16

    def test_profile_immutable(self) -> None:
        profile = BlockProfile(mean=1.0, std=2.0, min_val=0.0, max_val=3.0, outlier_score=1.5)
        with pytest.raises(AttributeError):
            profile.mean = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Allocation num_elements validation (P2-zero-elements)
# ---------------------------------------------------------------------------


class TestAllocationNumElementsValidation:
    """num_elements must be a positive integer."""

    _dummy_profile = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)

    def test_zero_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Allocation(
                block_id=0, num_elements=0,
                profile=self._dummy_profile, score=1.0, precision=Precision.INT4,
            )

    def test_negative_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Allocation(
                block_id=0, num_elements=-5,
                profile=self._dummy_profile, score=1.0, precision=Precision.INT4,
            )

    def test_float_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Allocation(
                block_id=0, num_elements=3.5,  # type: ignore[arg-type]
                profile=self._dummy_profile, score=1.0, precision=Precision.INT4,
            )

    def test_positive_elements_ok(self) -> None:
        alloc = Allocation(
            block_id=0, num_elements=128,
            profile=self._dummy_profile, score=1.0, precision=Precision.INT4,
        )
        assert alloc.num_elements == 128


# ---------------------------------------------------------------------------
# Numerical overflow in profiling (P2a)
# ---------------------------------------------------------------------------


class TestNumericalOverflow:
    """compute_profile should raise on extreme-magnitude data that overflows."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extreme_magnitude_block_raises(self, engine: BitAllocationEngine) -> None:
        """Values near float64 max should trigger overflow detection."""
        huge = np.array([1e308, -1e308, 1e307])
        with pytest.raises(ValueError, match="Numerical overflow"):
            engine.compute_profile(huge)

    def test_large_but_safe_block_ok(self, engine: BitAllocationEngine) -> None:
        """Large values that don't overflow should still work."""
        block = np.array([1e100, 2e100, 3e100])
        profile = engine.compute_profile(block)
        assert np.isfinite(profile.mean)
        assert np.isfinite(profile.std)

    def test_large_clustered_values_ok(self, engine: BitAllocationEngine) -> None:
        """Profiling works on large values whose spread is representable in float64.

        At magnitude 1e150 the ULP is ~2.2e134, so a spread of 1e140 produces
        three genuinely distinct values.  The profile should be finite with
        non-zero std — unlike the previous test that used 1e200 ± 1.0 where
        the ±1 was swallowed by float64 rounding.
        """
        base = 1e150
        spread = 1e140  # >> ULP at 1e150, so representable
        block = np.array([base - spread, base, base + spread])
        profile = engine.compute_profile(block)
        assert profile.mean == pytest.approx(base, rel=1e-10)
        assert np.isfinite(profile.std)
        assert profile.std > 0, "spread should be representable, std must not be zero"
        assert profile.std == pytest.approx(float(np.std(block)), rel=1e-10)


# ---------------------------------------------------------------------------
# estimate_compression_ratio validation (P2b)
# ---------------------------------------------------------------------------


class TestCompressionRatioValidation:
    """original_bits must be a positive integer."""

    _dummy_profile = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)

    def _allocs(self) -> list[Allocation]:
        return [
            Allocation(
                block_id=0, num_elements=256,
                profile=self._dummy_profile, score=1.0, precision=Precision.INT4,
            )
        ]

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            estimate_compression_ratio(self._allocs(), original_bits=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            estimate_compression_ratio(self._allocs(), original_bits=-8)

    def test_float_raises(self) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            estimate_compression_ratio(self._allocs(), original_bits=32.5)  # type: ignore[arg-type]

    def test_valid_bits_ok(self) -> None:
        ratio = estimate_compression_ratio(self._allocs(), original_bits=16)
        assert ratio == pytest.approx(16 / 4, abs=0.01)

    def test_one_bit_ok(self) -> None:
        """Edge case: original_bits=1 is valid (unusual but not illegal)."""
        ratio = estimate_compression_ratio(self._allocs(), original_bits=1)
        assert ratio == pytest.approx(1 / 4, abs=0.01)


# ---------------------------------------------------------------------------
# Calibrate thresholds scaffold (P1)
# ---------------------------------------------------------------------------


class TestCalibrateThresholds:
    """calibrate_thresholds should raise NotImplementedError (not implemented)."""

    def test_raises_not_implemented(self) -> None:
        blocks = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(NotImplementedError, match="not implemented"):
            BitAllocationEngine.calibrate_thresholds(
                blocks, error_fn=lambda b, p: 0.0
            )


# ---------------------------------------------------------------------------
# Block-size bound on the score (audit PER-COR-001)
# ---------------------------------------------------------------------------


class TestBlockSizeBound:
    """S_i < alpha + beta * sqrt(n - 1): the bound is exposed and enforced."""

    def test_max_score_formula(self, engine: BitAllocationEngine) -> None:
        assert engine.max_score(1) == pytest.approx(1.0)
        assert engine.max_score(101) == pytest.approx(1.0 + 0.5 * 10.0)

    def test_max_score_rejects_bad_sizes(self, engine: BitAllocationEngine) -> None:
        for bad in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="positive integer"):
                engine.max_score(bad)

    @pytest.mark.parametrize("n", [2, 16, 32, 64, 101])
    def test_single_huge_outlier_never_exceeds_bound(
        self, engine: BitAllocationEngine, n: int
    ) -> None:
        """One extreme outlier is the worst case; it still respects the bound."""
        block = np.zeros(n)
        block[0] = 1e6
        alloc = engine.allocate_single(block)
        assert alloc.score <= engine.max_score(n)
        assert alloc.precision != Precision.INT16

    def test_int16_reachable_above_bound(self, engine: BitAllocationEngine) -> None:
        block = np.zeros(128)
        block[0] = 1e6
        assert engine.allocate_single(block).precision == Precision.INT16

    def test_allocate_warns_when_precision_unreachable(
        self, engine: BitAllocationEngine
    ) -> None:
        rng = np.random.default_rng(0)
        blocks = [rng.normal(0, 1, 32) for _ in range(3)]
        with pytest.warns(UserWarning, match="INT16"):
            engine.allocate(blocks)

    def test_allocate_silent_when_all_reachable(
        self, engine: BitAllocationEngine
    ) -> None:
        rng = np.random.default_rng(0)
        blocks = [rng.normal(0, 1, 256) for _ in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            engine.allocate(blocks)

    def test_allocate_rejects_mixed_sizes(self, engine: BitAllocationEngine) -> None:
        with pytest.raises(ValueError, match="same number of elements"):
            engine.allocate([np.ones(128), np.ones(64)])

    def test_allocate_empty_sequence(self, engine: BitAllocationEngine) -> None:
        assert engine.allocate([]) == []


# ---------------------------------------------------------------------------
# Input dtype / shape rejection (audit PER-ROB-002, PER-ROB-003)
# ---------------------------------------------------------------------------


class TestInputTypeRejection:
    """compute_profile only accepts real-valued, at least 1-D input."""

    def test_complex_ndarray_raises(self, engine: BitAllocationEngine) -> None:
        with pytest.raises(TypeError, match="real numbers"):
            engine.compute_profile(np.array([1 + 100j, 2 + 0j, 3 - 50j]))

    def test_complex_list_raises(self, engine: BitAllocationEngine) -> None:
        with pytest.raises(TypeError, match="real numbers"):
            engine.compute_profile([1 + 100j, 2, 3])

    def test_bool_ndarray_raises(self, engine: BitAllocationEngine) -> None:
        with pytest.raises(TypeError, match="real numbers"):
            engine.compute_profile(np.array([True, False, True]))

    def test_integer_block_still_accepted(self, engine: BitAllocationEngine) -> None:
        profile = engine.compute_profile(np.array([1, 2, 3, 4]))
        assert profile.mean == pytest.approx(2.5)

    def test_scalar_block_raises(self, engine: BitAllocationEngine) -> None:
        with pytest.raises(ValueError, match="scalar"):
            engine.compute_profile(3.5)

    def test_flat_array_to_allocate_raises(self, engine: BitAllocationEngine) -> None:
        """A flat array is not a sequence of blocks; fail loudly."""
        flat = np.random.default_rng(0).normal(0, 1, 100)
        with pytest.raises(ValueError, match="scalar"):
            engine.allocate(flat)

    def test_flat_array_wrapped_is_one_block(self, engine: BitAllocationEngine) -> None:
        flat = np.random.default_rng(0).normal(0, 1, 128)
        allocations = engine.allocate([flat])
        assert len(allocations) == 1
        assert allocations[0].num_elements == 128


# ---------------------------------------------------------------------------
# API type validation (audit PER-ROB-005)
# ---------------------------------------------------------------------------


class TestApiTypeValidation:
    """Type errors are caught at construction with a clear message."""

    _profile = BlockProfile(mean=0, std=1, min_val=-1, max_val=1, outlier_score=1)

    def test_numpy_integer_num_elements_accepted(self) -> None:
        alloc = Allocation(0, np.int64(5), self._profile, 1.0, Precision.INT4)
        assert alloc.num_elements == 5

    def test_bool_num_elements_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Allocation(0, True, self._profile, 1.0, Precision.INT4)

    def test_engine_thresholds_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Thresholds"):
            BitAllocationEngine(thresholds=(1.0, 3.0, 6.0))

    def test_config_thresholds_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Thresholds"):
            EngineConfig(thresholds=(1.0, 3.0, 6.0))

    def test_find_critical_accepts_plain_int(self) -> None:
        allocations = [
            Allocation(i, 4, self._profile, float(p), p)
            for i, p in enumerate(Precision)
        ]
        critical = find_critical_blocks(allocations, min_precision=8)
        assert [a.precision for a in critical] == [Precision.INT16, Precision.INT8]

    def test_find_critical_rejects_unknown_width(self) -> None:
        with pytest.raises(ValueError):
            find_critical_blocks([], min_precision=5)

