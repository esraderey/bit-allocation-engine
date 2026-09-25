"""Evaluate the default allocation heuristic on Pythia-410M weights.

The evaluation profiles two-dimensional weight tensors in fixed-size blocks.
For context, it also measures a simple symmetric, per-block quantize/dequantize
MSE.  That MSE is a weight-reconstruction proxy, not a perplexity evaluation.

Each tensor's trailing remainder (fewer than ``block_size`` values) is profiled
as its own, smaller block.  Because the composite score is bounded by the block
size (see ``BitAllocationEngine.max_score``), remainder blocks are not directly
comparable with full blocks in ``highest_scoring_blocks``.

Run from the repository root:

    .\\.venv\\Scripts\\python scripts/evaluate_pythia_410m.py
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bit_allocation_engine import BitAllocationEngine, Precision

MODEL_ID = "EleutherAI/pythia-410m"
PRECISIONS = tuple(sorted(Precision, key=lambda precision: precision.value))


def symmetric_quantization_sse(blocks: np.ndarray, bits: int) -> float:
    """Return SSE after symmetric, per-row uniform quantize/dequantize."""
    values = np.atleast_2d(blocks)
    qmax = (1 << (bits - 1)) - 1
    max_abs = np.max(np.abs(values), axis=1, keepdims=True)
    # A zero-valued row is reconstructed exactly; a scale of one avoids 0 / 0.
    scales = np.where(max_abs == 0.0, 1.0, max_abs / qmax)
    quantized = np.clip(np.rint(values / scales), -qmax, qmax)
    residual = values - quantized * scales
    return float(np.einsum("ij,ij->", residual, residual))


def _iter_tensors(weight_files: list[Path]):
    """Yield ``(name, array)`` for every tensor across all safetensors shards."""
    for weight_file in weight_files:
        with safe_open(weight_file, framework="np") as weights:
            for tensor_name in weights.keys():
                yield tensor_name, weights.get_tensor(tensor_name)


def evaluate(model_id: str, block_size: int, cache_dir: Path) -> dict:
    """Profile linear weight matrices and calculate allocation statistics."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    started = perf_counter()
    snapshot_dir = Path(snapshot_download(
        model_id,
        cache_dir=str(cache_dir),
        allow_patterns=["*.safetensors"],
    ))
    weight_files = sorted(snapshot_dir.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError(f"No safetensors files found in {snapshot_dir}")

    engine = BitAllocationEngine()
    allocation_counts: Counter[Precision] = Counter()
    allocation_elements: Counter[Precision] = Counter()
    sse_by_scheme = {"allocated": 0.0}
    sse_by_scheme.update({f"INT{p.value}": 0.0 for p in PRECISIONS})
    sum_squares = 0.0
    considered_parameters = 0
    skipped_parameters = 0
    considered_tensors = 0
    skipped_tensors = 0
    source_storage_bits = 0
    source_dtypes: Counter[str] = Counter()
    block_id = 0
    largest_scores: list[tuple[float, str, int, int]] = []

    for tensor_name, tensor in _iter_tensors(weight_files):
        if tensor.ndim < 2 or not np.issubdtype(tensor.dtype, np.floating):
            skipped_tensors += 1
            skipped_parameters += tensor.size
            continue

        considered_tensors += 1
        considered_parameters += tensor.size
        source_storage_bits += tensor.size * tensor.dtype.itemsize * 8
        source_dtypes[str(tensor.dtype)] += tensor.size
        values = tensor.astype(np.float64, copy=False).ravel()

        full_size = values.size - values.size % block_size
        full_blocks = values[:full_size].reshape(-1, block_size)
        assigned_bits = np.empty(full_blocks.shape[0], dtype=np.int8)

        for index, block in enumerate(full_blocks):
            allocation = engine.allocate_single(block, block_id=block_id)
            precision = allocation.precision
            allocation_counts[precision] += 1
            allocation_elements[precision] += block.size
            assigned_bits[index] = precision.value

            entry = (allocation.score, tensor_name, index * block_size, precision.value)
            if len(largest_scores) < 10:
                heapq.heappush(largest_scores, entry)
            else:
                heapq.heappushpop(largest_scores, entry)
            block_id += 1

        sum_squares += float(np.einsum("ij,ij->", full_blocks, full_blocks))
        for candidate in PRECISIONS:
            sse_by_scheme[f"INT{candidate.value}"] += symmetric_quantization_sse(
                full_blocks, candidate.value
            )
            mask = assigned_bits == candidate.value
            if np.any(mask):
                sse_by_scheme["allocated"] += symmetric_quantization_sse(
                    full_blocks[mask], candidate.value
                )

        if full_size == values.size:
            continue

        block = values[full_size:]
        allocation = engine.allocate_single(block, block_id=block_id)
        precision = allocation.precision
        allocation_counts[precision] += 1
        allocation_elements[precision] += block.size
        sum_squares += float(np.dot(block, block))
        sse_by_scheme["allocated"] += symmetric_quantization_sse(block, precision.value)
        for candidate in PRECISIONS:
            sse_by_scheme[f"INT{candidate.value}"] += symmetric_quantization_sse(
                block, candidate.value
            )

        entry = (allocation.score, tensor_name, full_size, precision.value)
        if len(largest_scores) < 10:
            heapq.heappush(largest_scores, entry)
        else:
            heapq.heappushpop(largest_scores, entry)
        block_id += 1

    if considered_parameters == 0:
        raise RuntimeError(
            "No floating-point tensors with two or more dimensions were found; "
            "nothing to evaluate."
        )

    weighted_bits = sum(
        precision.value * allocation_elements[precision] for precision in PRECISIONS
    ) / considered_parameters
    source_bits_per_value = source_storage_bits / considered_parameters

    return {
        "model_id": model_id,
        "methodology": {
            "tensor_filter": "Only floating-point tensors with two or more dimensions",
            "block_size": block_size,
            "quantization_reference": "symmetric uniform quantization, per-block absmax scale",
            "metric_caveat": "Weight reconstruction MSE, not activation error or perplexity",
        },
        "weights": {
            "considered_tensors": considered_tensors,
            "considered_parameters": considered_parameters,
            "skipped_tensors": skipped_tensors,
            "skipped_parameters": skipped_parameters,
            "source_dtypes": dict(source_dtypes),
        },
        "allocation": {
            "total_blocks": block_id,
            "blocks": {
                f"INT{precision.value}": allocation_counts[precision]
                for precision in PRECISIONS
            },
            "elements": {
                f"INT{precision.value}": allocation_elements[precision]
                for precision in PRECISIONS
            },
            "element_percentages": {
                f"INT{precision.value}": round(
                    allocation_elements[precision] / considered_parameters * 100, 4
                )
                for precision in PRECISIONS
            },
            "average_bits": weighted_bits,
            "ideal_payload_compression_vs_fp32": 32 / weighted_bits,
            "ideal_payload_compression_vs_source": source_bits_per_value / weighted_bits,
        },
        "quantization_reference": {
            name: {
                "mse": sse / considered_parameters,
                "relative_mse": sse / sum_squares,
            }
            for name, sse in sse_by_scheme.items()
        },
        "highest_scoring_blocks": [
            {
                "score": score,
                "tensor": tensor_name,
                "start_offset": start,
                "assigned_bits": bits,
            }
            for score, tensor_name, start, bits in sorted(largest_scores, reverse=True)
        ],
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate(args.model_id, args.block_size, args.cache_dir)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
