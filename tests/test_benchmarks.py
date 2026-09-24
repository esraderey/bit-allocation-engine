"""Performance benchmarks for the Bit Allocation Engine.

Requires ``pytest-benchmark``.  Install via::

    pip install -e ".[bench]"   # or pip install -e ".[dev]"

If ``pytest-benchmark`` is not installed these tests are skipped
automatically rather than failing with an import error.
"""

from __future__ import annotations

import numpy as np
import pytest

# Graceful skip when pytest-benchmark is not installed.
pytest.importorskip("pytest_benchmark")

from bit_allocation_engine import BitAllocationEngine


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

ENGINE = BitAllocationEngine()
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Benchmark 1 — Single-block allocation (full pipeline)
# ---------------------------------------------------------------------------


def test_bench_allocate_single(benchmark) -> None:
    """Benchmark the full allocate_single pipeline on a 1024-element block."""
    block = RNG.normal(loc=0.0, scale=1.0, size=1024)
    benchmark(ENGINE.allocate_single, block, block_id=0)


# ---------------------------------------------------------------------------
# Benchmark 2 — Batch allocation (100 blocks)
# ---------------------------------------------------------------------------


def test_bench_allocate_batch(benchmark) -> None:
    """Benchmark allocate() over 100 blocks of 256 elements each."""
    blocks = [RNG.normal(loc=0.0, scale=1.0, size=256) for _ in range(100)]
    benchmark(ENGINE.allocate, blocks)
