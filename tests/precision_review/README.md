# Output precision and bounded-write evidence

- `evaluate.py` assesses current reviewed/amplitude outputs on cached real EEG
  windows and reuses completed candidate results. `results.json` records errors,
  input validity counts, supported coefficient bounds and later mean differences.
- `benchmark_storage.py` measures conversion, actual durable batch writes,
  diagnostics (including original overflow values), shared reads, disk bytes and
  process RSS. Each child checks exact converted-bit recovery. It uses fresh
  temporary output, never a patient store.
- `validate_eeg.py` runs two real windows for six representative metrics into a
  disposable dataset with copied metadata. It verifies raw per-window results
  against explicitly converted calculator outputs and metadata against original
  hashes. `eeg_integration.json` contains the bounded receipt.
- `example_preflight.json` verifies current working references against built
  metadata without writing connectivity.

Use the repository `.venv` and one numerical thread. On this machine Zarr operations
require execution outside the sandbox. The local input cache is
`/tmp/pyspi-candidate-pilot`; generated raw results are retained only under the
ignored `.candidate-scratch/precision` directory so completed estimator runs need
not be repeated. They are not portable committed test data.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tests/precision_review/evaluate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tests/precision_review/benchmark_storage.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tests/precision_review/validate_eeg.py
```

Do not reuse cached numerical outputs after estimator/input changes without
checking their provenance. The evaluator checks expected identities and recomputes
missing metrics; it does not automatically detect every numerical implementation
change. The tracked receipt fingerprints inputs and calculation source files.
Raw output is never replaced by means; time means in precision reports are only
validation of later analysis. Existing nonfinite results are counted separately
from output conversion. Neither precision testing nor storage clips values.

Storage timings use three sampled windows, not the full600-window workflow.
Filesystem cache and host contention are uncontrolled; read/write timing differences
are bounded observations, not guaranteed speedups or cohort projections.
