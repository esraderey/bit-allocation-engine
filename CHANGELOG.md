# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Sin publicar]

### Corregido (auditoría 2026-09-25)

- **Cota del score por tamaño de bloque hecha explícita** (PER-COR-001):
  `S_i < α + β·√(n−1)`, así que con los umbrales por defecto INT8 es
  inalcanzable con menos de 18 elementos e INT16 con menos de 102, y a la
  inversa bloques grandes derivan hacia INT8. Nuevo método
  `BitAllocationEngine.max_score(n)`; `allocate()` exige bloques del mismo
  tamaño (`ValueError`) y emite `UserWarning` cuando una precisión es
  inalcanzable para ese tamaño. Documentado en docstring y README.
  ([engine.py](bit_allocation_engine/engine.py))

- **Rechazo de dtypes y formas inválidas en `compute_profile`**
  (PER-ROB-002, PER-ROB-003): arrays complejos y booleanos lanzan
  `TypeError` en lugar de truncarse en silencio; escalares 0-d lanzan
  `ValueError`, de modo que un array plano pasado a `allocate()` falla en
  vez de tratarse como miles de bloques de un elemento.

- **Validación de tipos coherente** (PER-ROB-005): `Allocation.num_elements`
  acepta enteros de NumPy y rechaza `bool`; `EngineConfig` y
  `BitAllocationEngine` verifican que `thresholds` sea `Thresholds`;
  `find_critical_blocks` acepta un ancho de bits entero.

- **`scripts/evaluate_pythia_410m.py`** (PER-ROB-006): recorre todos los
  shards `*.safetensors` en lugar de exigir uno y falla con mensaje claro
  cuando no hay tensores elegibles (antes `ZeroDivisionError`).

- **Licencia** (PER-SUP-007): archivo `LICENSE` (MIT) añadido; `pyproject`
  usa `license = "MIT"` (SPDX) con `license-files`, y declara
  `[tool.setuptools] packages` explícitamente.

- **Documentación** (PER-MAN-008): umbrales descritos como inclusivos,
  conteo de tests corregido, resultados de la evaluación de Pythia
  versionados en `results/`, rango de magnitud de `epsilon` documentado.

- Lint: import sin usar en `models.py` y `zip()` sin `strict=` en `demo.py`.

- **25 tests ancla nuevos** en `test_engine.py` y 4 tests offline del script
  de evaluación en `test_evaluate_script.py` (checkpoints sintéticos, sin
  red; se omiten sin el extra `eval`).

### Añadido

- **Detección de desbordamiento numérico** en `compute_profile`: guarda
  post-cómputo que lanza `ValueError` descriptivo cuando estadísticas
  intermedias (`mean`, `std`, `outlier_score`, etc.) desbordan `float64`,
  en lugar de propagar `inf`/`nan` silenciosamente.
  ([engine.py](bit_allocation_engine/engine.py))

- **Validación de `original_bits`** en `estimate_compression_ratio`:
  rechaza valores no enteros (`TypeError`) y menores a 1 (`ValueError`).
  ([analysis.py](bit_allocation_engine/analysis.py))

- **`calibrate_thresholds()` (extension point)**: método de clase no
  implementado que define la firma y el contrato esperados para calibración
  de umbrales con datos representativos y una función de error
  domain-specific. ([engine.py](bit_allocation_engine/engine.py))

- **`.gitignore`**: excluye `__pycache__/`, `*.egg-info/`, `.pytest_cache/`,
  `.benchmarks/`, entornos virtuales, archivos IDE y artefactos de OS.

- **Dependencia `pytest-benchmark`**: añadida a los grupos opcionales `dev`
  y `bench` en `pyproject.toml`.
  ([pyproject.toml](pyproject.toml))

- **Skip graceful en benchmarks**: `test_benchmarks.py` usa
  `pytest.importorskip` para no fallar cuando `pytest-benchmark` no está
  instalado. ([test_benchmarks.py](tests/test_benchmarks.py))

- **15 tests nuevos** (63 unitarios + 2 benchmarks): desbordamiento numérico, validación de
  `original_bits`, `calibrate_thresholds`, y valores grandes con spread
  representable en `float64`.
  ([test_engine.py](tests/test_engine.py))

### Cambiado

- **Cómputo de `std` centrado**: `compute_profile` ahora calcula la
  desviación estándar sobre datos centrados (`arr − mean`) para reducir la
  magnitud de los términos cuadráticos y mitigar desbordamiento en bloques
  de gran magnitud. ([engine.py](bit_allocation_engine/engine.py))

- **Documentación de umbrales heurísticos** (P1): la docstring de
  `BitAllocationEngine` y el README incluyen un aviso prominente de que los
  umbrales por defecto son heurísticas no calibradas, no garantizan error de
  reconstrucción acotado, y deben calibrarse contra datos reales para
  producción. ([engine.py](bit_allocation_engine/engine.py),
  [README.md](README.md))

- **`calibrate_thresholds` descrito honestamente**: la documentación ya no
  lo presenta como helper operativo que "automatiza" la calibración, sino
  como extension point no implementado que el usuario debe completar.

- **README actualizado**: sección de instalación documenta grupo `bench`,
  conteo de tests actualizado a 65.

### Eliminado

- **Artefactos de build del índice git**: 12 archivos cacheados
  (`__pycache__/*.pyc`, `*.egg-info/*`) eliminados del seguimiento con
  `git rm --cached`.

---

## [0.1.0] — 2026-09-22

### Añadido

- Motor de asignación de bits (`BitAllocationEngine`) con pipeline
  completo: block → profile → score → precision.
- Fórmulas: perfil estadístico $P_i$, score compuesto $S_i = αR_i + βO_i$,
  selección de precisión por umbrales (INT2/INT4/INT8/INT16).
- Estabilidad numérica: denominador $\max(|μ_i|, σ_i) + ε$ en $R_i$
  para evitar explosión con media cercana a cero.
- Modelos de datos inmutables: `BlockProfile`, `Allocation`, `Thresholds`,
  `EngineConfig`, `Precision` (IntEnum).
- Validación estricta: `EngineConfig` y `Thresholds` rechazan valores
  no finitos; `epsilon=0` rechazado; umbrales deben ser estrictamente
  crecientes; `num_elements` debe ser entero positivo.
- Rechazo de NaN/±inf en `compute_profile` y `select_precision`.
- Utilidades de análisis: `summarize_allocations`,
  `estimate_compression_ratio`, `find_critical_blocks`.
- Pesos por número de elementos en `avg_bits` y ratio de compresión.
- Demo con 10 bloques sintéticos (`demo.py`).
- 50 tests unitarios con pytest.
- Documentación completa en `README.md`.
