# Bit Allocation Engine

A pre-processing quantization engine that determines bit-width precision (**INT2 / INT4 / INT8 / INT16**) based on cheap statistical block profiling — **before** running any expensive quantization pass.

## Motivation

Traditional quantization approaches (like CDP) determine precision *after* processing. This engine inverts that: it computes a lightweight statistical profile per block and assigns bit-widths upfront, enabling faster and more predictable quantization pipelines.

## Installation

```bash
# Clone or copy the project, then:
pip install -e .

# For development (includes pytest):
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24

## Quick Start

```python
import numpy as np
from bit_allocation_engine import BitAllocationEngine, summarize_allocations

engine = BitAllocationEngine()

# Your model weights / activations split into blocks
blocks = [np.random.normal(0, 1, 256) for _ in range(100)]

# Run the allocation pipeline
allocations = engine.allocate(blocks)

# Summarise results
print(summarize_allocations(allocations))
```

## How It Works

For each block $B_i$, the engine runs a three-stage pipeline:

### 1. Statistical Profile

$$P_i = (\mu_i, \sigma_i, \min_i, \max_i, o_i)$$

where the **outlier score** is:

$$o_i = \frac{\max |B_i - \mu_i|}{\sigma_i + \epsilon}$$

### 2. Composite Score

$$S_i = \alpha \cdot R_i + \beta \cdot O_i$$

where:
- $R_i = \frac{\sigma_i}{\max(|\mu_i|, \sigma_i) + \epsilon}$ — coefficient of variation (relative variability)
- $O_i = o_i$ — outlier measure

> **Note on numerical stability:** The denominator uses $\max(|\mu_i|, \sigma_i)$ instead of just $|\mu_i|$ to prevent $R_i$ from exploding when the mean is near zero — a common scenario with ML weight distributions. This keeps $R_i \leq 1$ for zero-centred data while preserving the original behaviour when $|\mu| \gg \sigma$.

### 3. Precision Selection

| Score Range | Precision | Interpretation |
|-------------|-----------|----------------|
| $S < 1.0$ | INT2 | Concentrated distribution, safe to quantize aggressively |
| $1.0 \leq S < 3.0$ | INT4 | Moderate variability |
| $3.0 \leq S < 6.0$ | INT8 | High variability or moderate outliers |
| $S \geq 6.0$ | INT16 | Severe outliers, needs maximum precision |

## Configuration

```python
from bit_allocation_engine import BitAllocationEngine, Thresholds, EngineConfig

# Option 1: Individual parameters
engine = BitAllocationEngine(alpha=1.5, beta=0.3)

# Option 2: Full config object
config = EngineConfig(
    alpha=1.0,
    beta=0.5,
    epsilon=1e-8,
    thresholds=Thresholds(int4=1.0, int8=3.0, int16=6.0),
)
engine = BitAllocationEngine(config=config)

# Option 3: Mix — config base + keyword overrides
engine = BitAllocationEngine(config=config, alpha=2.0)
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Weight for relative variability $R_i$. Must be finite and non-negative. |
| `beta` | `0.5` | Weight for outlier score $O_i$. Must be finite and non-negative. |
| `epsilon` | `1e-8` | Numerical stability constant. Must be strictly positive and finite. |
| Threshold INT4 | `1.0` | Score above which INT4 is assigned |
| Threshold INT8 | `3.0` | Score above which INT8 is assigned |
| Threshold INT16 | `6.0` | Score above which INT16 is assigned |

> **Validation:** `EngineConfig` and `Thresholds` reject non-finite values (NaN, ±inf) at construction time. Thresholds must be strictly increasing and finite. `epsilon=0` is rejected because it can cause division-by-zero on constant blocks.

## API Reference

### Core

| Class / Function | Description |
|------------------|-------------|
| `BitAllocationEngine` | Main engine — profile, score, and allocate blocks |
| `BitAllocationEngine.compute_profile(block)` | Compute statistical profile for a single block |
| `BitAllocationEngine.compute_score(profile)` | Compute composite score from a profile |
| `BitAllocationEngine.select_precision(score)` | Map a score to a `Precision` enum value |
| `BitAllocationEngine.allocate_single(block)` | Full pipeline on one block |
| `BitAllocationEngine.allocate(blocks)` | Full pipeline on a sequence of blocks |

### Models

| Class | Description |
|-------|-------------|
| `Precision` | Enum: `INT2`, `INT4`, `INT8`, `INT16` |
| `BlockProfile` | Frozen dataclass: `mean`, `std`, `min_val`, `max_val`, `outlier_score` |
| `Allocation` | Frozen dataclass: `block_id`, `num_elements`, `profile`, `score`, `precision` |
| `Thresholds` | Frozen dataclass: `int4`, `int8`, `int16` (must be strictly increasing and finite) |
| `EngineConfig` | Frozen dataclass: `alpha`, `beta`, `epsilon`, `thresholds` (validated at construction) |

### Analysis Utilities

```python
from bit_allocation_engine import (
    summarize_allocations,
    estimate_compression_ratio,
    find_critical_blocks,
    Precision,
)

allocations = engine.allocate(blocks)

# Distribution summary (avg_bits is weighted by element count per block)
summary = summarize_allocations(allocations)
# → {'total_blocks': 100, 'distribution': {'INT2': 20, ...}, 'avg_bits': 5.4}

# Compression estimate vs FP32 (also element-weighted)
ratio = estimate_compression_ratio(allocations, original_bits=32)
# → 5.16

# Find blocks needing high precision
critical = find_critical_blocks(allocations, min_precision=Precision.INT8)
```

> **Note:** `avg_bits` and `estimate_compression_ratio` weight each block's precision by its number of elements. This ensures a 1-element block doesn't carry the same weight as a million-element block in the final estimate.

> **Input validation:** `compute_profile` rejects blocks containing NaN or ±inf with a clear `ValueError`, rather than silently propagating non-finite scores through the pipeline.

## Demo

```bash
python demo.py
```

Generates 10 synthetic blocks with varying distributions and runs the full pipeline.

**Sample output:**

```
Block                  mean        std        o_i        R_i      Score  Precision
--------------------------------------------------------------------------------------------
  constant_like       3.1400     0.0001     1.8334     0.0000     0.9167  INT2
  tight_gaussian    100.0009     0.0508     3.0554     0.0005     1.5282  INT4
  moderate_gaussian   4.9380     2.0733     3.2011     0.4199     2.0204  INT4
  wide_gaussian       0.4288    10.3284     2.7250     1.0000     2.3625  INT4
  uniform_wide        0.4276     5.8540     1.7694     1.0000     1.8847  INT4
  laplace            -0.0206     2.7419     5.1516     1.0000     3.5758  INT8
  mild_outliers      -0.0471     1.6503     9.1177     1.0000     5.5589  INT8
  severe_outliers    -0.0968    17.7058    11.3012     1.0000     6.6506  INT16
  bimodal            -0.0322     5.0288     1.2197     1.0000     1.6098  INT4
  zero_sparse        -0.0031     0.1422     5.7354     1.0000     3.8677  INT8

  Distribution: {INT2: 1, INT4: 5, INT8: 3, INT16: 1}
  Avg bits: 6.2 | Compression: 5.16x vs FP32
```

## Running Tests

```bash
pytest tests/ -v
```

50 tests covering profiles, scores, precision selection, full pipeline, analysis utilities, model validation, element-weighted compression, NaN/inf rejection, and configuration validation.

## License

MIT
