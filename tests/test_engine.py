"""Tests for the Bit Allocation Engine."""

from __future__ import annotations

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
        # R = 0.1/10 = 0.01, O = 1.0 → S = 1.0*0.01 + 0.5*1.0 = 0.51
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
        # R = 1.0 / ε ≈ 1e8, which is large but finite
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
            Allocation(block_id=0, profile=dummy_profile, score=0.5, precision=Precision.INT2),
            Allocation(block_id=1, profile=dummy_profile, score=1.5, precision=Precision.INT4),
            Allocation(block_id=2, profile=dummy_profile, score=4.0, precision=Precision.INT8),
            Allocation(block_id=3, profile=dummy_profile, score=8.0, precision=Precision.INT16),
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
# Models validation
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
