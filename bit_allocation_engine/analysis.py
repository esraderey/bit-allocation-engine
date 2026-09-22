"""Bit Allocation Engine — Analysis utilities.

Post-allocation helpers for summarising, estimating compression,
and identifying critical blocks.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .models import Allocation, Precision


def summarize_allocations(allocations: Sequence[Allocation]) -> dict:
    """Produce a summary of bit-allocation results.

    Parameters
    ----------
    allocations : sequence of Allocation
        Results from ``BitAllocationEngine.allocate``.

    Returns
    -------
    dict
        Keys:
        - ``total_blocks``: int
        - ``distribution``: dict mapping precision label → count
        - ``distribution_pct``: dict mapping precision label → percentage
        - ``avg_bits``: float — weighted average bits per element
    """
    if not allocations:
        return {
            "total_blocks": 0,
            "distribution": {},
            "distribution_pct": {},
            "avg_bits": 0.0,
        }

    counts: Counter[Precision] = Counter()
    for alloc in allocations:
        counts[alloc.precision] += 1

    total = len(allocations)
    distribution = {
        f"INT{p.value}": counts.get(p, 0)
        for p in sorted(Precision, key=lambda x: x.value)
    }
    distribution_pct = {
        k: round(v / total * 100, 2) for k, v in distribution.items()
    }
    total_elements = sum(a.num_elements for a in allocations)
    avg_bits = (
        sum(a.precision.value * a.num_elements for a in allocations)
        / total_elements
    )

    return {
        "total_blocks": total,
        "distribution": distribution,
        "distribution_pct": distribution_pct,
        "avg_bits": round(avg_bits, 2),
    }


def estimate_compression_ratio(
    allocations: Sequence[Allocation],
    original_bits: int = 32,
) -> float:
    """Estimate the compression ratio vs. the original bit-width.

    Parameters
    ----------
    allocations : sequence of Allocation
        Results from ``BitAllocationEngine.allocate``.
    original_bits : int, optional
        Bit-width of the original representation (default FP32 = 32).

    Returns
    -------
    float
        Compression ratio (e.g., 4.0 means 4× smaller).
    """
    if not allocations:
        return 1.0

    total_elements = sum(a.num_elements for a in allocations)
    avg_bits = (
        sum(a.precision.value * a.num_elements for a in allocations)
        / total_elements
    )
    if avg_bits == 0:
        return float("inf")
    return round(original_bits / avg_bits, 2)


def find_critical_blocks(
    allocations: Sequence[Allocation],
    min_precision: Precision = Precision.INT8,
) -> list[Allocation]:
    """Return blocks that were assigned at least ``min_precision`` bits.

    These are the "hard" blocks that the engine considers difficult to
    quantize aggressively.

    Parameters
    ----------
    allocations : sequence of Allocation
        Results from ``BitAllocationEngine.allocate``.
    min_precision : Precision, optional
        Minimum precision to qualify as critical (default INT8).

    Returns
    -------
    list[Allocation]
        Allocations whose precision ≥ ``min_precision``, sorted by score
        descending.
    """
    critical = [a for a in allocations if a.precision.value >= min_precision.value]
    return sorted(critical, key=lambda a: a.score, reverse=True)
