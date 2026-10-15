# Serial workflow verification

The serial runner reuses production result stores in bounded window batches.
It adds a run index/reader and diagnostic journal; the storage format,
estimators, dependency locks, and scientific regression goldens are unchanged.
The machine-readable receipt is [execution_results.json](execution_results.json).

Final focused verification runs against both the checkout and a noneditable
wheel outside the checkout. It covers preparation at native/mixed rates,
explicit configuration, every supported reference (including an actual MNE
REST forward model), channel/metric/window order, exact values, selective reads,
portable metadata links, successful NaNs versus failures, bounded output buffers,
interruption/publication failure, explicit retry, incompatible resume, cohort
resume, and a session terminated before its final report was saved.

The full-size scientific/calculation tier passed **924 cases with 10 known
expected failures**, excluding the original standalone legacy benchmark. That
run included 174 workflow cases; three subsequent ordering/recovery cases are
included in the final **177-case workflow run**. The unchanged production
storage suite passed **439 cases**. These are behavior/integration checks, not
independent validation of every estimator's mathematical definition.

## Bounded real data

The opt-in [validate_real.py](validate_real.py) reads existing electrode order
from legacy Zarrs, maps every exported contact to the actual EDF header, then
calculates only four retained contacts and three windows. Continuous preparation
uses the entire recording, with explicit 2048 Hz, 0.5–500 Hz, CAR, native harmonic
notches and polyphase conversion. The five metrics are covariance, precision,
Spearman correlation, coherence magnitude and signed PLI using explicitly
configured existing estimators. One patient uses overlapping windows.

| Patient | Native rate | Full exported contact count | Calculated contacts/windows |
|---|---:|---:|---:|
| 004 | 1024 Hz | 40 | 4 / 3 |
| 037 | 512 Hz | 94 | 4 / 3 |
| 009 | 2048 Hz | 71 | 4 / 3 |

All stored arrays match a freshly constructed Calculator for each window
bit-for-bit, checking representative SPI reuse against fresh construction. Resume preserves
them bit-for-bit. EDF and accessed Zarr metadata file sizes/mtimes remain
unchanged; legacy connectivity arrays are never read. Final receipt timings,
software versions, configuration, file counts and peak process RSS are retained
in the JSON. These small trials are not a full-patient or 185-metric benchmark.

## Output batch tradeoff

[benchmark_output.py](benchmark_output.py) measures the existing durable writer,
including packing/compression choice, at 50, 128 and 300 channels, twelve
synthetic directed/symmetric/antisymmetric metrics and eight windows. The
selective read reconstructs one metric exactly. One repeat was measured:

| Channels | Windows/batch | Write seconds | One metric read seconds | Files |
|---:|---:|---:|---:|---:|
| 50 | 1 / 4 / 8 | 2.416 / 0.604 / 0.347 | 0.033 / 0.008 / 0.005 | 304 / 76 / 38 |
| 128 | 1 / 4 / 8 | 2.450 / 0.827 / 0.530 | 0.039 / 0.013 / 0.010 | 304 / 76 / 38 |
| 300 | 1 / 4 / 8 | 3.396 / 1.540 / 0.938 | 0.040 / 0.027 / 0.016 | 304 / 76 / 38 |

Larger batches reduce file count and fixed write/read cost here. Compressed size
is not monotonically smaller: at 300 channels the three layouts are all about
40.0–40.2 MB. These measurements exclude calculation and run-journal costs;
the real-data receipt includes those in end-to-end wall time.

The default 64 MiB matrix-buffer target fits about 18 windows at 50 channels,
three at 118, and two at 128 for 185 metrics. At 300 channels one window already
needs about 127 MiB, so the runner warns and uses one-window batches. Packing,
continuous EEG and estimator caches need additional RAM. Fine-grained recovery
therefore has substantial file-count overhead at large channel/metric counts.
No global optimal batch size or automatic memory/concurrency policy is claimed.

## Limits and deferred work

Resume retains committed batches; successful computations in an unfinished
batch may be repeated. Durable failures are skipped unless retry is explicit.
Stochastic recomputation can differ, and no new seed scheme is introduced.
Hard termination cannot supply a final wall time; this is recorded as unknown.
Resume checks source statistics/header hash and output declarations, not full
EDF checksums or a complete scan of all previously written payloads.

Preparation holds continuous retained EEG in RAM. Bounded output is not a total
RAM guarantee. Many small batches create more files; the present design avoids
a production-storage rewrite. NumPy can calculate a time mean from the composite
reader; the existing packed mean helper still targets individual production
stores. Mathematical corrections, parallelization, algorithm optimization,
custom references, naming cleanup and data-lake/cache changes remain separate.

Zarr async I/O stalled inside the execution sandbox; integration/storage tests
and read-only real-data checks were run outside it. All generated data stayed
under temporary directories. This is an execution-environment limitation, not a
reason to change pyspi's I/O or filtering algorithms.
