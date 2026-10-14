# Exact connectivity storage

`pyspi.storage` stores connectivity matrices in a local Zarr v3 directory, one
group per metric. It preserves native **float32 and float64 bits**, including
NaN payloads, infinities, subnormals, and signed zeros. It never changes metric
definitions, takes absolute values, drops reciprocal differences, or converts
precision. Other input dtypes are rejected; explicitly convert them upstream
if that is intended.

## Installation

From this checkout, on the tested Linux/Python 3.12 platform:

```bash
uv sync --locked --extra storage
```

For pip, explicitly select the CPU PyTorch build because pip ignores uv's index
configuration:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu' '.[storage]'
```

The storage extra adds Zarr, numcodecs, pcodec, and google-crc32c. Zstd is provided
by numcodecs; no experimental compressor registry is imported. Calculations
that use JIDT also require a JVM; Java 21 is the tested major version. Storage
itself does not start a JVM or import Calculator.

## Write and read matrices

```python
import numpy as np
from pyspi.storage import write_results, open_results, verify

results = {
    "covariance": np.zeros((4, 4, 600), dtype=np.float64),
    "another_metric": np.ones((4, 4, 600), dtype=np.float32),
}
write_results(
    "patient.pspi", results,
    recording={
        "patient": "example",
        "channels": ["B1", "B2", "A1", "A2"],
        "sampling_rate": 2048,
        "window_onsets": list(range(0, 600 * 1024, 1024)),
    },
    labels={"covariance": ["undirected", "signed"]},
    attrs={"covariance": {"parameters": {"estimator": "empirical"}}},
)
reader = open_results("patient.pspi")
assert reader.metrics == ["covariance", "another_metric"]
matrix = reader.load("covariance", windows=0)          # (4, 4)
subset = reader.load("covariance", windows=[4, 0, 4])  # (4, 4, 3)
assert verify("patient.pspi") == {}
```

`write_metric(path, name, values, *, labels=(), attrs=None, codec="pcodec",
level=None, overwrite=False, target_chunk_values=262144,
max_shard_values=134217728)` writes one metric and returns its declared metadata.
`values` has shape `(channels, channels)` or `(channels, channels, windows)`.
There is no channel-count bound. Zero windows are supported; zero channels and
nonsquare matrices are rejected.
Metric names must be nonempty single path components; `.`, `..`, `zarr.json`,
`pspi-manifest.json`, and the diagnostic key `__store__` are reserved.

`write_results(path, mapping, *, labels=None, attrs=None, recording=None, ...)`
uses the same codec, overwrite, and size options for a batch. Each matrix retains
its own dtype and shape. `labels` and `attrs` are keyed by metric name. All batch
inputs/options are validated and all metric names are declared before the first
payload write. A failed batch is not transactional: completed metrics remain
readable and unfinished metrics remain declared. No failed write claims success.

Metric inventory follows mapping insertion order. Adding metrics appends them;
overwriting a metric retains its inventory position. Numeric channel and window
axes are never sorted. User metadata lives under `info(metric)["attrs"]` and
`reader.recording`; it cannot overwrite decoding fields. Metadata uses JSON
semantics: string object keys, finite numbers, strings, booleans, null, lists,
and nested objects. Tuples become JSON lists; NumPy objects require explicit
conversion. `set_recording_info(path, mapping)` merges shared metadata by top-level
key. Core storage records supplied channel/onset lists without interpreting them.

`Results(path)` and `open_results(path)` are equivalent. `reader.metrics` includes
incomplete declared metrics. `reader.info(metric)` returns layout, dtype, labels,
user attributes, and `state` (`pending`, `writing`, or `complete`). Metadata reads
do not scan numeric data. No persistent reader handle needs closing.

`reader.load(metric, windows=None, return_availability=False)` reconstructs the
selected matrix values. A metric written as a single 2D matrix loads as 2D;
a windowed metric loads as 3D, including a one-window or empty result. An integer
selection always returns 2D; a one-dimensional sequence always returns 3D.
Negative indices, duplicates, supplied order, and empty selections are supported.
Booleans and nonintegers raise `TypeError`; out-of-range and multidimensional
selections raise `IndexError`. Selection is positional, not a time-onset lookup.

## Explicit Calculator adapter

```python
from pyspi.storage import write_calculator

# calc.compute() has already run.
write_calculator("one_window.pspi", calc,
                 channel_names=["B1", "B2", "A1", "A2"])

# Each item is an already computed Calculator for one window.
write_calculator("patient.pspi", calculated_windows,
                 metrics=["pec", "spearmanr-sq"],
                 channel_names=["B1", "B2", "A1", "A2"],
                 recording={"sampling_rate": 2048,
                            "window_onsets": [0, 1024]})
```

`write_calculator` accepts one Calculator or an ordered sequence. A sequence of
length one still produces windowed matrices. By default metric order comes from
the first table; `metrics` can select and reorder it. Row order comes from the
first table, and columns and subsequent tables are aligned by process label.
Duplicate or inconsistent process labels, dtype changes, and differing metric
labels across windows are rejected before writing. `channel_names` supplies
recording names in that row order, which is useful with Calculator's existing
`Data.procnames` issue; it does not reorder numeric axes. An explicitly supplied
`recording["channels"]` must agree with these names. SPI labels are retained as
metadata and never used to infer matrix symmetry.

The adapter does not calculate, preprocess, window, resample, or track calculation
success. Calculator currently does not expose whether its NaN-filled table has
been computed. Call `compute()` first; failed-calculation NaNs are saved as actual
values, rather than inferred to be unavailable. Holding a sequence of calculated
windows in memory is an adapter convenience; streaming calculation orchestration
and append/resume APIs are future work.

## Integrity, partial recovery, and durability

An atomic, fsynced `pspi-manifest.json` records metric inventory, write state,
and expected layouts. Numeric chunks and shard indexes have CRC32C checksums.
Missing chunks are checked explicitly so Zarr fill zeros cannot silently become
calculated output. Files and directories are flushed before publishing completion.
Writes propagate I/O errors and never treat a partial write as complete.

Missing, unreadable, or uncommitted requested data emits `PartialReadWarning`.
Reads retain recoverable entries at their original positions and fill unavailable
entries with NaN. With `return_availability=True`, the return value is
`(values, boolean_mask)` of matching shape. Saved NaNs/infinities remain available
and produce no missing-data warning. If only a residual is unavailable, the upper
triangle and diagonal survive; the affected strict lower triangle is unavailable.
Warnings also identify inconsistent metric/root headers even when payload values
remain recoverable. A warning can be promoted to an error with Python's standard
warning filters. No automatic copy repair or recalculation runs.

`verify(path)` reads all declared numeric chunks, including unfinished or deleted
metrics, and returns `{metric_name: diagnostic}`; `{}` means clean. A store-wide
manifest/root problem uses the key `__store__`. If the manifest is unreadable or
inconsistent enough that shape/layout cannot be trusted, reads raise
`StorageFormatError` or `MetadataError` rather than guessing. Other public
exceptions/warnings are `StorageIntegrityError` and `ShardSizeWarning`.

```bash
uv run --no-sync python -m pyspi.storage verify patient.pspi
```

The command exits 0 when clean, 1 on verification failure. Checksums detect
accidental numeric corruption, not malicious modification. Metadata is duplicated
and compared where needed for decoding, not cryptographically authenticated.
Copy tools still own transfer verification. Local POSIX filesystems with one
writer per recording path are supported; concurrent writing, object stores,
network-filesystem durability, and reader snapshots during writes are not promised.
`overwrite=True` replaces the selected metric without a rollback guarantee.

## Packing, chunks, and shards

Packing is inferred from the actual bits: exact symmetric/sign-flipped mirrors
use diagonal plus upper triangle; other arrays use every entry or a triangle
plus an exact XOR residual. Antisymmetric packing is a reconstruction convention,
not a scientific directed/undirected label. Reciprocal roundoff and substantive
differences are both preserved. No tolerance or lossy packing API exists.

The default codec is pcodec at level 8; `codec="zstd"` defaults to level 3.
The residual-versus-full choice compares whole-payload encodings with the selected
codec/level. It is a size heuristic; actual chunking, indexes, and metadata can
change final file size. Packing and this comparison consume CPU and temporary
memory before writing. They are included in the production benchmark's total.

`target_chunk_values` targets independently decoded windows; `max_shard_values`
targets uncompressed stored values per array shard. Both are positive integers.
Shards contain at least one whole stored window. If one window exceeds the shard
target it is written intact with `ShardSizeWarning`. The target is not a hard
memory cap or a promise about compressed bytes. The writer packs one supplied
metric in memory; batching several metrics does not provide streaming windows.

Production schema is `pspi-store/1.0`. Prototype/legacy stores are rejected
explicitly; no automatic conversion or rewriting of existing patient Zarr data
is performed. File extensions are optional; `.pspi` is a directory convention.

## Optional time averaging

Ordinary time averaging needs no special storage operation:

```python
mean_matrix = np.nanmean(reader.load("covariance"), axis=2, dtype=np.float64)
```

The relocated `pyspi.analysis.time_mean(reader, metric)` uses packed payloads to
avoid reconstructing every mirrored entry. It loads those payloads into memory,
reduces in column blocks, and explicitly accumulates float64, including for
float32 inputs. It requires complete readable input, warning and raising on
unavailable data. Empty/all-NaN series return NaN with reduction warnings
suppressed. The antisymmetric subnormal signed-zero defect from the prototype is
fixed by reconstructing lower series before reducing. Storage has no mean method.

## Validation and bounded timings

See [the storage test guide](../tests/storage/README.md) and
[production validation](../tests/storage/PRODUCTION_RESULTS.md). Benchmarks use
synthetic matrices and temporary directories; existing patient stores are untouched.
