"""Bit Allocation Engine — Demo script.

Generates synthetic blocks with different distributions and runs the
full allocation pipeline, printing a summary report.
"""

from __future__ import annotations

import numpy as np

from bit_allocation_engine import (
    BitAllocationEngine,
    Precision,
    estimate_compression_ratio,
    find_critical_blocks,
    summarize_allocations,
)


def make_demo_blocks(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Create synthetic blocks with varying statistical properties."""
    blocks: list[tuple[str, np.ndarray]] = []

    # 1. Nearly constant (should → INT2)
    blocks.append(("constant_like", np.full(256, 3.14) + rng.uniform(-1e-4, 1e-4, 256)))

    # 2. Tight Gaussian around a large mean (should → INT2)
    blocks.append(("tight_gaussian", rng.normal(loc=100.0, scale=0.05, size=256)))

    # 3. Moderate Gaussian (should → INT4)
    blocks.append(("moderate_gaussian", rng.normal(loc=5.0, scale=2.0, size=256)))

    # 4. Wide Gaussian (should → INT4/INT8)
    blocks.append(("wide_gaussian", rng.normal(loc=0.0, scale=10.0, size=256)))

    # 5. Uniform wide (should → INT4/INT8)
    blocks.append(("uniform_wide", rng.uniform(-10, 10, 256)))

    # 6. Laplace (heavier tails → INT8)
    blocks.append(("laplace", rng.laplace(loc=0.0, scale=2.0, size=256)))

    # 7. Gaussian with mild outliers (should → INT8)
    mild = rng.normal(0, 1, 256)
    mild[0] = 15.0
    mild[1] = -15.0
    blocks.append(("mild_outliers", mild))

    # 8. Gaussian with severe outliers (should → INT16)
    severe = rng.normal(0, 1, 256)
    severe[0] = 200.0
    severe[1] = -200.0
    blocks.append(("severe_outliers", severe))

    # 9. Bimodal (two clusters → higher variability)
    bimodal = np.concatenate([rng.normal(-5, 0.5, 128), rng.normal(5, 0.5, 128)])
    blocks.append(("bimodal", bimodal))

    # 10. Zero-centred sparse (many zeros + some values → tests ε stability)
    sparse = np.zeros(256)
    sparse[:25] = rng.normal(0, 0.5, 25)
    blocks.append(("zero_sparse", sparse))

    return blocks


def format_precision(p: Precision) -> str:
    """Color-code precision for terminal output."""
    colors = {
        Precision.INT2: "\033[92m",   # green
        Precision.INT4: "\033[93m",   # yellow
        Precision.INT8: "\033[33m",   # orange
        Precision.INT16: "\033[91m",  # red
    }
    reset = "\033[0m"
    return f"{colors.get(p, '')}{p.name}{reset}"


def main() -> None:
    rng = np.random.default_rng(2024)
    engine = BitAllocationEngine()

    print("=" * 72)
    print("  BIT ALLOCATION ENGINE - Demo")
    print("=" * 72)
    print()
    print(f"  Config: alpha={engine.alpha}, beta={engine.beta}, eps={engine.epsilon}")
    print(f"  Thresholds: INT4>={engine.thresholds.int4}, "
          f"INT8>={engine.thresholds.int8}, INT16>={engine.thresholds.int16}")
    print()

    named_blocks = make_demo_blocks(rng)
    names = [name for name, _ in named_blocks]
    blocks = [block for _, block in named_blocks]

    allocations = engine.allocate(blocks)

    # --- Per-block results ---
    print(f"{'Block':<22} {'mean':>10} {'std':>10} {'o_i':>10} "
          f"{'R_i':>10} {'Score':>10}  {'Precision'}")
    print("-" * 92)

    for name, alloc in zip(names, allocations):
        p = alloc.profile
        r_i = p.std / (abs(p.mean) + engine.epsilon)
        print(
            f"  {name:<20} {p.mean:>10.4f} {p.std:>10.4f} "
            f"{p.outlier_score:>10.4f} {r_i:>10.4f} {alloc.score:>10.4f}  "
            f"{format_precision(alloc.precision)}"
        )

    # --- Summary ---
    summary = summarize_allocations(allocations)
    ratio = estimate_compression_ratio(allocations)
    critical = find_critical_blocks(allocations)

    print()
    print("-" * 72)
    print("  SUMMARY")
    print("-" * 72)
    print(f"  Total blocks: {summary['total_blocks']}")
    print(f"  Distribution: {summary['distribution']}")
    print(f"  Percentages:  {summary['distribution_pct']}")
    print(f"  Avg bits:     {summary['avg_bits']}")
    print(f"  Compression:  {ratio}x vs FP32")
    print(f"  Critical blocks (>=INT8): {len(critical)}")
    print()


if __name__ == "__main__":
    main()
