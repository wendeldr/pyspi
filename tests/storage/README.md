# Storage and analysis verification

These tests exercise `pyspi.storage` and the separate `pyspi.analysis.time_mean`
helper using synthetic matrices and temporary directories. They import the
package, with no comparison-directory path manipulation or experimental codec
registry. Existing patient stores are never opened or changed.

## Run locally

From the repository root:

```bash
uv sync --locked --extra testing --extra storage
uv run --no-sync python -m pytest -c tests/storage/pytest.ini tests/storage
```

The explicit configuration avoids the calculation suite and its legacy benchmark.
Storage uses the same root lock as calculations; no standalone project, lock,
or virtual environment is maintained. Tests intentionally fail rather than skip
when required storage dependencies are missing.

```bash
uv run --no-sync python -m pytest -c tests/storage/pytest.ini tests/storage \
  --cov=pyspi.storage --cov=pyspi.analysis --cov-branch --cov-report=term-missing
```

Installed-package verification also builds and installs a noneditable wheel,
changes out of the checkout, asserts that `pyspi` loads from site-packages, and
runs these tests against that installed package. The fresh local installation
and outside-checkout invocation passed, as recorded in
[PRODUCTION_RESULTS.md](PRODUCTION_RESULTS.md). This checkout uses local
verification; its unused GitHub Actions workflows were removed.

## Coverage and cases

The final production suite has **419 passing cases**, with no skips or expected
failures. The selected exact implementation has approximately 98% combined
statement/branch coverage. Exact packing, Calculator adaptation, and the analysis
helper have full statement and branch coverage.

Cases include all five exact representations, float32/float64 and arbitrary IEEE
bits, one channel/window, empty windows, noncontiguous/read-only inputs, supplied
metric/channel/window ordering, metadata, codec levels, overwrite refusal,
integer selections, target-size boundaries, and oversize-window warnings.
Corruption cases include missing metric directories, primary/residual shards,
missing inner chunks, payload/index bit flips, truncation, invalid and inconsistent
JSON, partial writes, flush/publication failures, and exact surviving entries with
availability masks. Stored NaNs and infinities are explicitly distinguished from
unavailable data. Independent expected arrays are constructed by the tests.

The adapter tests exercise table orientation, process-label alignment, metric
selection/order, channel-name metadata, dtype/label consistency, and unchanged
source tables. Real computed single and windowed matrices are also checked by
[the calculation integration test](../compatibility/test_calculation_storage.py).
`time_mean` has separate numerical assertions for float64 accumulation, nonfinite
series, partial-read rejection, and the antisymmetric subnormal signed-zero case.

The prototype's tolerance experiment and its two known-failure tests were
archived, rather than repaired or silently accepted. The production packer only
supports exact representations, with a test rejecting the lossy structure option.
Float32 conversion/range-policy tests were replaced by native-dtype preservation
and unsupported-dtype checks; precision conversion is deliberately absent.
No original calculation regression functions, goldens, or tolerances changed.

## Bounded benchmarks and historical evidence

```bash
uv run --no-sync python tests/storage/benchmark_production.py \
  --repetitions 3 --output tests/storage/production_benchmark.json
```

The benchmark includes packing, adaptive compression selection, file/directory
flushes, full/selected reads, verification, and a tiny 185-metric batch. It uses
bounded synthetic inputs, not EEG calculations or patient conversion. Component
timers overlap and must not be summed. Timings and their limits are documented in
[PRODUCTION_RESULTS.md](PRODUCTION_RESULTS.md).

The original comparison, scientific findings, and source archive are in
[docs/storage_comparison](../../docs/storage_comparison/README.md).
[INTEGRITY_PLAN.md](INTEGRITY_PLAN.md) and [INTEGRITY_RESULTS.md](INTEGRITY_RESULTS.md)
retain the agreed contract and historical integrity measurements.
