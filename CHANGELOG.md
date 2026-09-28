# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Sin publicar]

### Corregido (auditoría 2026-09-28)

- **Cota del score frente al redondeo** (PER-LOG-001): con σ ≫ ε, `R_i` y
  `O_i` redondean hacia arriba y el score podía alcanzar o superar
  `max_score(n)` (p. ej. n=101 con un pico de 1e10 recibía INT16 mientras
  `allocate()` avisaba de que era inalcanzable). `allocate_single` acota el
  score a `nextafter(max_score(n), 0)` y `_warn_if_unreachable` ya no avisa
  cuando `max_score` es 0 y coincide con el umbral. Docstrings actualizados.
  ([engine.py](bit_allocation_engine/engine.py))

- **Bloques constantes** (PER-LOG-002): `np.mean` inexacto daba `std` y
  `outlier_score` espurios (p. ej. `np.full(7, 1e12+0.3)`); ahora un bloque
  constante devuelve `std=0` y `outlier_score=0` sin calcular la media.
  Los bloques casi constantes (p. ej. un valor distinto en 1 ulp) también
  tenían una `std` inflada por la misma causa; la media se refina con una
  segunda pasada (`mean += mean(arr - mean)`). Esto puede cambiar la
  precisión asignada a bloques cuya media es enorme frente a su `std`
  (antes la `std` espuria dominaba el score).

- **Dtypes no numéricos** (PER-ROB-003): `compute_profile` solo acepta
  dtypes `i`, `u`, `f` y `O`; cadenas, bytes, `datetime64` y `timedelta64`
  lanzan `TypeError` en lugar de perfilarse. En arrays `object` se valida
  cada elemento: cadenas, `bool`/`np.bool_` y cualquier valor que no sea
  `numbers.Real` lanzan `TypeError`. `Decimal` deja de aceptarse (no es
  `numbers.Real`); `Fraction` y los `int`/`float` de Python o NumPy dentro
  de un array `object` siguen aceptándose.

- **Enteros mayores que `2**53`** (PER-LOG-004): `compute_profile` lanza
  `ValueError` en vez de perder precisión al convertir a `float64`. Cubre
  también los enteros de Python dentro de arrays `object` (una lista como
  `[2**70, 2**70+1]` acaba en dtype `object`); en dtypes `i`/`u` la
  comparación se hace con `int()` para no depender de la promoción a
  `float64` de NumPy 1.x.

- **`original_bits` en `estimate_compression_ratio`** (PER-ROB-005): acepta
  enteros de NumPy y rechaza `bool`.
  ([analysis.py](bit_allocation_engine/analysis.py))

- **Referencia INT2 del evaluador** (PER-LOG-006): el rango simétrico
  restringido hace ternaria la referencia de 2 bits; se documenta en el
  docstring, en `methodology.quantization_reference` y en el README (los
  resultados de `results/` siguen siendo válidos, el algoritmo no cambia).

- **Checkpoints BF16/FP8 en el evaluador** (PER-ROB-007): error `RuntimeError`
  claro (tensor, archivo, dtype y remedio) para los dtypes de safetensors que
  el backend NumPy no carga (`BF16`, `F8_E4M3`, `F8_E5M2`, `F8_E8M0`,
  `F6_E2M3`, `F6_E3M2`, `F4`) en lugar de `TypeError: data type 'bfloat16'
  not understood` o `AttributeError`.

- **Pesos todo cero en el evaluador** (PER-ROB-008): `relative_mse` es
  `null` en vez de lanzar `ZeroDivisionError`.
  ([evaluate_pythia_410m.py](scripts/evaluate_pythia_410m.py))

- **34 tests ancla nuevos**: 30 en `test_engine.py` y 4 en
  `test_evaluate_script.py`.

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
