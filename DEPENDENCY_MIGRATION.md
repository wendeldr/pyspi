# Dependency migration

This checkout targets Python **3.12**, NumPy **2.5**, and a unified calculation
and storage-test environment. The supported reference installation is the root
`uv.lock`, using CPU PyTorch wheels. Local verification uses Linux x86-64 and
Python 3.12.3. The original migration artifacts did not record the JVM version;
the post-review check records OpenJDK 21.0.12.1 and its JPype library path in
[review_verification.json](tests/compatibility/review_verification.json).
Verification is local; the unused GitHub Actions workflows were removed at the
user's request. Other operating systems are not certified by these
results. The package declares `>=3.12,<3.13`; older interpreters and NumPy 1
are outside this checkout's supported range.

The migration is complete and locally verified. Golden values, tolerance
policy, and existing patient outputs are unchanged. Production storage is now
integrated; scientific review remains a separate piece.

## Baseline

The starting commit was `be859fe`. Before edits or installations, all tracked
files were copied to `/tmp/pyspi-dependency-migration-20261014/baseline` and
hashed. [baseline.json](tests/compatibility/baseline.json) records the original
Python 3.11.16 environment, its installed distributions, source/test/golden
hashes, known expected failures, and test counts.

| Original environment run | Passed | Skipped | Expected failures |
| --- | ---: | ---: | ---: |
| Default | 553 | 15 | 10 |
| Full and slow, excluding legacy benchmark | 566 | 0 | 11 |
| Legacy benchmark | 285 | 0 | 0 |

There were no unexpected failures. The original `.venv` was preserved at
`/tmp/pyspi-dependency-migration-20261014/baseline311-venv` after its runs ended.
Temporary paths are local evidence, not required installation inputs. Commit
`be859fe` and its original dependency lock retain the reproducible baseline.

## Selected versions and upgrade decisions

| Component | Selected | Reason |
| --- | --- | --- |
| Python | 3.12 | Matches the retained pyEDM binary installation and modern storage requirements. |
| NumPy | 2.5.4 | Current release at investigation time; supported minor range `>=2.5.4,<2.6`. |
| SciPy | 1.17.1 | Version 1.18.1 changed ANM outputs beyond existing tolerances. Restoring only SciPy restored all five ANM comparisons. This does not establish that 1.18 is mathematically wrong; its optimizer drift needs separate evaluation. |
| MNE / MNE-Connectivity | 1.13.2 / 0.9.0 | Upgrade replaces obsolete `mne.connectivity` imports with the standalone API. |
| CDT | 0.6.0 | Upgrade covers the four pairwise methods used by pyspi. |
| pyEDM | 1.15.2.0 | Retained estimator. The earlier 2.x probe changed CCM values and needs a numerical review. |
| spectral-connectivity | 1.1.2 | Retained frequency conventions and metric definitions. Version 2.x changes the Nyquist grid and frequency-dependent summaries. |
| setuptools | 81.0.0 | Newest allowed by the retained pyEDM resource loader; `pkg_resources` was removed in 82. |
| pytest | 9.1.1 | Updated test tooling. |
| Zarr / numcodecs / pcodec | 3.4.1 / 0.17.0 / 1.0.4 | Chosen storage implementation and strict-read support. |

Other dependencies were freshly resolved within the declared ranges; their
exact versions and artifact hashes are in `uv.lock`. The old NumPy constraint
was originally for tslearn; current tslearn 0.9.0 works in the covered paths.
The [earlier investigation](DEPENDENCY_COMPATIBILITY_REVIEW.md) explains the
original restrictions and the numerical differences found in candidate upgrades.

## Compatibility changes

* `pyspi._spectral_compat` gives only spectral-connectivity 1.1.2's connectivity
  module a NumPy namespace with `in1d` implemented using `isin`. It preserves
  flattened `in1d` output and delegates other attributes to NumPy. NumPy itself
  and installed dependency files are unchanged. Both obsolete call sites
  (group-delay frequency selection and canonical coherence) are exercised.
  The exact dependency pin bounds this adapter's assumptions; it can be removed
  after a newer spectral implementation is validated.
* Wavelet code catches `numpy.exceptions.AxisError`. A targeted test exercises
  the three-dimensional cached-result fallback absent from the old default run.
* CCM extracts its maximum embedding dimension as a NumPy scalar before `int`
  conversion. NumPy 2.5 rejects the old length-one-array conversion. This keeps
  the existing embedding choice, including the known automatic-embedding bug;
  it does not change the estimator or choose a different embedding criterion.
* New MNE results are converted to dense arrays. Envelope correlation explicitly
  combines epochs with the old default mean. Wavelet connectivity uses
  `spectral_connectivity_epochs`; PSI extracts `freqs_computed`, matching the old
  API's frequency vector rather than the new object's band centres. Existing
  frequency selection and aggregation remain as recorded in the goldens.

The PSI wavelet wrapper's existing selection indexes the last result axis using
positions from that frequency vector. This is retained here and belongs in the
scientific review alongside the previously recorded PSI/band issues. Preserving
goldens does not establish mathematical correctness of that behavior.

## Packaging and installation

`pyproject.toml` is the dependency and package-data source of truth. `setup.py`
delegates to that metadata. Runtime requirements and the testing requirements
file install the local project instead of maintaining duplicate dependency lists.
The exported `tests/requirements-test-lock.txt` contains portable pinned versions
without the previous machine-local editable path.

From the repository root, with uv 0.8.17 and a Java runtime available:

```bash
uv sync --locked --extra testing --extra storage
uv pip check
uv run --no-sync python -m pytest
uv run --no-sync python -m pytest -c tests/storage/pytest.ini tests/storage
uv run --no-sync python -m pytest --pyspi-size=full --runslow --timeout=1800 --ignore=tests/test_SPIs.py
uv run --no-sync python -m pytest tests/test_SPIs.py --runslow --timeout=1800
```

For pip, create a Python 3.12 environment and run:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tests/requirements-test-lock.txt
python -m pip install --no-deps -e '.[testing]'
```

The migration initially used a `storage-test` group containing experimental
comparison compressors. Production integration has replaced it with the
`storage` extra (Zarr, numcodecs, pcodec, and google-crc32c), removed the registry
dependency, and archived the prototype. The root lock and exported test
requirements now reflect that smaller dependency set. See
[storage usage](docs/storage.md) and [production verification](tests/storage/PRODUCTION_RESULTS.md).
Existing patient stores remain unchanged.

Use the local commands above for calculation/compatibility, storage, full/slow,
and legacy benchmark checks. GitHub Actions test, dataset-generation, and
publishing workflows have been removed because this project does not use them.

## Validation and limits

Local JUnit, installation, build, and diagnostic evidence is under
`tests/storage/.artifacts/dependency_migration/` (ignored). The tests preserve
golden files and tolerance policy; failing comparisons were investigated rather
than regenerated or relaxed. The new canonical-coherence reference was captured
from the unmodified original environment before any upgrade.

| Final unified environment run | Passed | Skipped | Expected failures |
| --- | ---: | ---: | ---: |
| Default | 563 | 15 | 10 |
| Small goldens, all slow tests | 576 | 0 | 11 |
| Full goldens, all slow tests | 576 | 0 | 11 |
| Legacy benchmark | 285 | 0 | 0 |
| Storage with branch coverage | 381 | 0 | 2 |
| Reinstalled wheel compatibility and calculation/storage checks | 10 | 0 | 0 |

There are no unexpected failures or errors. Default skips match the original
baseline exactly. The full and small/slow tiers ran the Java tests and retain
the original 11 expected-failure identities and reasons. Both slow tiers used
two pytest workers. Storage coverage remains
about 98%; its two expected failures are confined to experimental tolerance
packing. The legacy benchmark's reported 23-entry deviation table is identical
to the original baseline's table.

The original calculation test files, golden files, and tolerance policy retain
their hashes. Ten new compatibility checks cover the removed NumPy APIs,
canonical-coherence outputs captured before migration, and a four-metric
calculation over two 1024-sample windows with exact pcodec storage/reconstruction
(including diagonal NaNs, availability, and repeated/out-of-order selection).
The migration results and source/lock hashes for commit `a2c3300` are in
[results.json](tests/compatibility/results.json). Subsequent review changes to
test configuration are recorded separately in
[review_verification.json](tests/compatibility/review_verification.json).

Both a source distribution and wheel built successfully. Required configurations,
bundled datasets, JIDT jar, and calculation modules were checked inside the wheel.
The exported pip lock installed into another clean environment; both pyspi and
spectral-connectivity were then force-reinstalled. All ten compatibility checks
passed with pyspi imported from that environment's site-packages, outside the
checkout. The dependency source matched its wheel RECORD checksum and retained
both original `np.in1d` calls, proving the packaged adapter works without source
edits. Dependency checks passed for both installations. A separate runtime-only
installation calculated all four fabfour metrics with finite off-diagonal values
without pytest or Zarr installed.

These tests are change detectors, not a comprehensive mathematical validation.
Known expected failures remain documented in the baseline and test guide. Some
goldens intentionally represent unsuccessful/all-NaN calculations. The legacy
benchmark reports numerical deviations and does not fail merely because values
differ, so full per-SPI golden comparisons are the stronger preservation check.
Test durations are not controlled performance benchmarks. No patient recordings
or existing patient output stores were opened or modified for this migration.

## Review follow-up

The independent review found no calculation differences. Its four setup and
documentation findings were resolved and checked:

* The test guide describes locked local verification and troubleshooting.
* Local verification uses Java 21. A new
  test-run receipt records OpenJDK 21.0.12.1, vendor, runtime version, and JPype
  library path. The original migration artifacts did not record their JVM;
  their provenance has not been retroactively rewritten.
* The pip runtime command explicitly selects the CPU index and torch CPU build.
  That command installed into a fresh Python 3.12 environment. From outside the
  checkout, the installed package calculated all four fabfour metrics on a
  synthetic 4-channel, 1024-sample window with finite off-diagonal results.
  Torch reported `2.14.1+cpu`, no CUDA runtime, and no NVIDIA packages were
  installed. No pytest or Zarr was needed for that runtime calculation.
* The duplicate `tests/storage` project, lock, and derived virtual environment
  were removed. Storage tests now use only the root lock; all 381 passed with
  the same two expected failures. The active comparison implementation remains
  required by these tests at that point; production integration has since
  replaced those imports and archived the comparison source.

An initial diagnostic driver started the JVM before pytest and crashed when
pytest installed its fault handler. The identical compatibility checks passed
with that plugin disabled. Following [JPype's signal-handler guidance](https://jpype.readthedocs.io/en/stable/userguide.html#errors-reported-by-python-fault-handler),
the root pytest configuration now disables `faulthandler` before JVM startup.
The current default suite then passed **563 tests, with the original 15 skips
and 10 expected failures unchanged**. The crash log and successful control are
retained with the other local verification artifacts.

No calculation implementations, original test functions, goldens, dependency
lock, or patient stores changed during this cleanup. Local commands were
verified. Java 17 was not backtested because the user chose the current Java 21
setup. The unused GitHub Actions workflows were subsequently removed at the
user's request; earlier receipts retain their historical workflow hashes.

The original migration and review receipts describe their respective earlier
states. Current storage integration, dependency, source, and wheel evidence is
recorded in [production_verification.json](tests/storage/production_verification.json).
