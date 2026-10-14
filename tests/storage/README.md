# Storage regression tests

These main-repository tests validate the candidate in `storage_comparison/` using deterministic arrays and temporary stores. No patient output is modified and no EEG calculations run in this suite. The calculation and storage dependency environment is now unified; production storage integration remains later work.

After the agreed selection and shard-sizing corrections on 2026-10-14, the candidate passes **381 cases**, with **two strict expected failures** confined to experimental tolerance packing, on Python 3.12.3. All nine original R1 integrity regressions pass. The original prototype script also passed its 115 synthetic round trips. See [INTEGRITY_RESULTS.md](INTEGRITY_RESULTS.md) for behavior, measured overhead, and limits.

## Run from the repository root

Install and run the unified locked environment:

```bash
uv sync --locked --extra testing --group storage-test
uv run --no-sync python -m pytest \
  -c tests/storage/pytest.ini tests/storage
```

The explicit configuration avoids collecting the legacy SPI benchmark, which calculates metrics during import. The [unified CI workflow](../../.github/workflows/run_unit_tests.yaml) installs the root lock and runs calculation and storage suites on Python 3.12, saving JUnit and coverage reports. No remote CI run is claimed.

The older standalone comparison environment remains available with `uv sync --project tests/storage --locked` and `uv run --project tests/storage --locked python -m pytest -c tests/storage/pytest.ini tests/storage`. Zarr 3.4.1 provides the per-array strict-read setting used by the integrity implementation. `fpzip`, `zfpy`, and Blosc2 remain development dependencies because the prototype's compressor registry imports them. Production dependency cleanup remains deferred.

## Coverage

```bash
mkdir -p tests/storage/.artifacts
COVERAGE_FILE=tests/storage/.artifacts/.coverage \
uv run --no-sync python -m pytest \
  -c tests/storage/pytest.ini tests/storage \
  --cov=packing --cov=pspi_store --cov=store_integrity --cov-branch \
  --cov-report=term-missing \
  --cov-report=html:tests/storage/.artifacts/html \
  --cov-report=json:tests/storage/.artifacts/coverage.json \
  --junitxml=tests/storage/.artifacts/results.xml
```

Open `.artifacts/html/index.html` for annotated coverage. Reports, temporary artifacts, caches, and the environment are excluded from Git.

| Module | Statements exercised | Branches exercised |
| --- | ---: | ---: |
| `packing.py` | 104/104, 100% | 24/24, 100% |
| `pspi_store.py` | 345/350, 99% | 129/132, 98% |
| `store_integrity.py` | 248/254, 98% | 88/94, 94% |

The combined statement/branch figure is approximately 98%. Coverage includes the temporary mean helper; no code is excluded from the denominator. Remaining gaps include uncommon metadata/error paths and a second failure during completion-state rollback. Executing every line would not prove numerical correctness or simulate power loss.

## What the suite checks

- Every packing mode, both real codecs, float32/float64 bits, signed zeros, subnormals, infinities, NaN payloads, and arbitrary bit patterns.
- Channel/window orientation, one channel/window, 2D matrices, noncontiguous inputs, empty/duplicate/reordered selections, and reads across block/shard boundaries.
- Integer-only window selection, including rejection of mixed boolean/integer lists, floats, and timestamps. Large signed/unsigned indices are checked before conversion and cannot wrap into a valid window.
- Shard-size targets that limit chunk/shard geometry when a window fits; identifying `ShardSizeWarning` messages and one-window shards when it does not. Both codecs, dtypes, and every packing mode retain exact values. Empty payloads and values exactly at the target do not warn.
- Positive integer size targets, validation before changing single/batch output, and an incomplete write state if the caller promotes a size warning to an exception.
- Missing outer shards and inner chunks, including inner chunks removed with a valid rechecksummed index. Out-of-shape padding is allowed.
- Payload/index corruption, truncation, invalid offsets, deleted metrics, damaged metadata, unsupported formats, and read-only verification.
- Exact availability masks, original window positions, surviving entries, NaN placeholders, one identifying warning per affected call, healthy selections within damaged stores, and stored NaNs that remain available without warnings.
- Registered batch inventory and write interruptions before layout, during data, during flushing, and while publishing completion, including explicit overwrites.
- Preservation of readable upper/diagonal entries when a residual is unavailable. The mean helper rejects partial input instead of reducing substituted zeros.
- Invalid input, overwrite refusal, root/metric metadata consistency, unsafe path names, source immutability, and verification command exit status. Programming errors propagate.

Fixtures isolate the prototype import location in `conftest.py`, so production integration can change that adapter without rewriting assertions. Fault injection edits disposable output bytes or uses temporary monkeypatches; it does not alter codec implementations on disk.

## Remaining known defects

The initial suite had 144 passes and 16 expected failures. R1 and R6 are now ordinary passing tests. The original R5 strict-cap test was replaced by tests of the agreed size-target/warning behavior. Two expected failures remain visible:

| Review item | Expected failing cases | Unresolved behavior |
| --- | ---: | --- |
| R4 | 2 | Tolerance packing can replace a finite entry with infinity |

Each `known_defect` case uses `xfail(strict=True, raises=AssertionError)`. Unrelated exceptions fail normally, and an unexpected pass fails the suite so the marker must be removed after a reviewed fix. To expose the defects as ordinary failures:

```bash
uv run --project tests/storage --locked python -m pytest \
  -c tests/storage/pytest.ini tests/storage \
  -m known_defect --runxfail --tb=short
```

This diagnostic command should fail; it is not the normal CI command. R4 belongs to the unselected lossy experiment; the candidate writer uses exact packing. R3, the mean helper's signed-zero edge case, remains documented in [the review](../../storage_comparison/REVIEW.md). Exact residual retention is agreed. Mathematical definitions, broader SPI regression gates, production integration, mean-helper relocation, migration, and comparison-folder cleanup remain outside this piece.

## Reader selection and shard sizing

`Results.load(metric, windows=...)` selects already saved result windows. It accepts an integer or a one-dimensional integer sequence/array, with negative indices, duplicates, order, and empty selections preserved. None loads all windows. Floats, booleans, and other noninteger values raise `TypeError`; out-of-range and multidimensional selections raise `IndexError`. This argument does not set EEG window duration or overlap.

The writer's existing `max_shard_values` argument is an uncompressed stored-value target, with a one-window minimum. If an array's single window exceeds it, the writer emits `ShardSizeWarning`, identifying the metric/array, required values/bytes, and target, then writes one window per shard. Each affected array warns once per write, subject to normal Python warning filters. Otherwise, it reduces inner chunk length as needed to respect the target. Compressed file bytes include codec and index overhead, so this is not an absolute file-size limit. Both size settings must be positive integers. No channel-count restriction or splitting within matrices was added.
