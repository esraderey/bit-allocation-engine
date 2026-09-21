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

For each block $B_i$, the engine computes:

### 1. Statistical Profile

$$P_i = (\mu_i, \sigma_i, \min_i, \max_i, o_i)$$

where the **outlier score** is:

$$o_i = \frac{\max |B_i - \mu_i|}{\sigma_i + \epsilon}$$

### 2. Composite Score

$$S_i = \alpha \cdot R_i + \beta \cdot O_i$$

where:
- $R_i = \frac{\sigma_i}{|\mu_i| + \epsilon}$ — coefficient of variation (relative variability)
- $O_i = o_i$ — outlier measure

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
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Weight for relative variability $R_i$ |
| `beta` | `0.5` | Weight for outlier score $O_i$ |
| `epsilon` | `1e-8` | Numerical stability constant |
| Threshold INT4 | `1.0` | Score above which INT4 is assigned |
| Threshold INT8 | `3.0` | Score above which INT8 is assigned |
| Threshold INT16 | `6.0` | Score above which INT16 is assigned |

## Analysis Utilities

```python
from bit_allocation_engine import (
    summarize_allocations,
    estimate_compression_ratio,
    find_critical_blocks,
    Precision,
)

allocations = engine.allocate(blocks)

# Distribution summary
summary = summarize_allocations(allocations)
# → {'total_blocks': 100, 'distribution': {'INT2': 20, ...}, 'avg_bits': 5.4}

# Compression estimate vs FP32
ratio = estimate_compression_ratio(allocations, original_bits=32)
# → 5.93

# Find blocks needing high precision
critical = find_critical_blocks(allocations, min_precision=Precision.INT8)
```

## Demo

```bash
python demo.py
```

Generates 10 synthetic blocks with varying distributions and runs the full pipeline.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
