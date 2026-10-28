# pyspi test suite

The suite is a **change detector**. The golden values under `tests/regression/goldens/` record what the code produced when they were generated; a passing comparison means agreement within its tolerance and status checks, not mathematical correctness. Known bugs are locked in as they are (see [Known bugs](#known-bugs-locked-not-fixed)).

The [adversarial test review](TEST_REVIEW.md) records weaknesses found and fixed,
independent references, mutation checks, coverage, and remaining limitations.

For an isolated calculation change, run its focused mathematical tests and
affected regression cases. Do not routinely rerun already-passing unrelated
suites as more metric tests are added. Broader runs need a reason, such as a
shared cache/base-class/configuration change or evidence of wider impact.

## Layout

| path | what it covers |
|---|---|
| `regression/test_spi_goldens.py` | one golden comparison per SPI, preprocessing variants, `Calculator` end-to-end passes (fast subset by default, full config with `--runslow`), catalogue checks |
| `regression/test_extra_goldens.py`, `regression/extra_config.yaml` | snapshot goldens for seven wavelet classes that `config.yaml` leaves out (see [Extra goldens](#extra-goldens)) |
| `regression/harness.py`, `regression/spi_policy.py` | shared build/seed/compute/compare code and the per-SPI policy (tolerances, slow, stochastic) |
| `regression/generate_goldens.py` | writes and checks the goldens |
| `compatibility/` | targeted NumPy 2 spectral and wavelet paths, frozen baseline inventory |
| `unit/` | `Data`, base-class dispatch, `Calculator` config loading and failure handling, frames, utils, `phi_native` golden, known bugs |
| `unit/test_calculation_reporting.py` | completion/failure versus NaN values, callback ordering/ownership/error propagation, timing boundaries, warning filters/visibility, interruption, and report lifecycle |
| `unit/test_regression_guards.py`, `regression/test_golden_contract.py` | tests of comparison/error handling, baseline completeness, generator checks, and required-JVM verification |
| `unit/test_numeric_boundaries.py`, `unit/test_ids_reference.py` | nondegenerate spectral/wavelet aggregation, independent covariance/precision and IDS references, preprocessing axes, dataset reload, and seeded shared-cache calculations |
| `unit/test_spectral.py` | spectral estimator agreement, shared transforms/factorizations, bounded temporary arrays/caches, sampling-rate isolation, mutable inputs, empty bands, and phase branch boundaries |
| `unit/test_psi.py` | independent raw PSI definition checks, inclusive adjacent-bin selection, direction, frequency units, and unavailable-band reporting |
| `unit/test_spectral_bands.py` | shared DC/Nyquist grid, scalar endpoint formulas, transform-derived group-delay fits, unit scaling, Granger DC, and unavailable-band reporting |
| `unit/test_group_delay.py` | independent beta-null/BH/OLS reference, real/complex null calibration, pair/window isolation, contiguous-run handling, diagnostics and shared fits |
| `unit/test_wilson.py`, `unit/test_directed_formulas.py`, `unit/test_directed_integration.py` | exact VAR/reference formulas, channel-order independent fits, native-grid dDTF, source-first direction, failure isolation and shared enabled requests |
| `unit/test_envelope.py` | six PEC definitions, projection/absolute order, numerical degeneracy, scale invariance and shared cache ownership |
| `unit/test_robust.py` | the Cholesky FAST-MCD (`pyspi._robust`) against seeded scikit-learn `MinCovDet`: identical supports, dense-formula distances/log-determinant, singular fallback |
| `benchmark_run_hotspots.py`, `run_hotspots_results.json`, `run_hotspots/` | per-metric timing of the whole reviewed preset for one 118x1024 window, with the before/after receipts of the run-hotspot series |
| `benchmark_review_selection.py`, `review_selection_results.json`, `benchmark_envelope.py`, `envelope_refactor_results.json` | measured complete/incremental candidate costs and PEC before/after family timing |
| `benchmark_spectral.py`, `spectral*_refactor_results.json` | paired spectral timings and historical/current receipts, including traced allocation peaks and explicit definition/frequency-selection differences |
| `test_calc.py`, `test_utils.py`, `test_phi_native.py` | original tests, unchanged |
| `storage/` | storage-format suite, run separately (below) |
| `workflow/` | strict configuration/header-only preflight, EEG preparation, serial/spawned execution, resource limits/planning, bounded output, ordering, interruption/retry/resume and terminal execution |

## Environment

Calculator reporting can be checked separately with:

```bash
uv run --no-sync python -m pytest tests/unit/test_calculation_reporting.py tests/unit/test_calculator.py --require-java
```

The main and extra-wavelet shared-cache Calculator golden passes now run with
and without completion callbacks. Their numerical comparison policies are
unchanged. These checks supplement controlled failure/NaN cases, exact snapshot
bits and ownership, consumer exceptions, monotonic timing, warning visibility
and suppression, and report reset/interruption tests. Reporting does not fix
existing calculation limitations, including covariance's one-channel error.
See [the reporting API](../docs/calculation_reporting.md).
The [reporting verification receipt](calculation_reporting_results.json) records
the default/full tiers, installed-wheel checks, coverage scope, direct
pre-change comparisons, and the test-namespace/sandbox findings.

The supported checkout environment is **Python 3.12 with NumPy 2.5**. Calculation
and storage tests use one root `uv.lock`; install both into `.venv` with:

```bash
uv sync --locked --extra testing --extra storage
uv pip check
uv run --no-sync python -m pytest
uv run --no-sync python -m pytest -c tests/storage/pytest.ini tests/storage
```

`pyproject.toml` is the dependency source of truth. `tests/requirements-test-lock.txt`
is an exported, portable pip equivalent of the unified lock, without an editable
machine-local path. In a clean Python 3.12 environment:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tests/requirements-test-lock.txt
python -m pip install --no-deps -e '.[testing]'
```

The lock selects CPU PyTorch wheels. `pyEDM==1.15.2.0` and
`spectral-connectivity==1.1.2` preserve the selected estimators and frequency
conventions; the packaged spectral adapter supports NumPy 2 without editing
site-packages. The former requires setuptools below 82 because it imports
`pkg_resources`. See [the migration report](../DEPENDENCY_MIGRATION.md) for
baseline hashes, test results, remaining restrictions, and upgrade decisions.

A Java runtime is needed for the JIDT SPIs. Local verification uses OpenJDK
21.0.12.1. Without a JVM,
the Java-dependent tests explicitly skip. A complete verification run must
confirm that Java is available and those tests ran. Add `--require-java` to make
an unavailable JVM fail rather than silently reducing coverage. Storage tests remain an
explicit second invocation because they have their own pytest configuration;
they run in the same locked environment as calculations. This checkout uses
local verification; its unused GitHub Actions workflows were removed.

`meta.json` in each golden set records the Python, Java and package versions and the `OMP_NUM_THREADS` setting it was generated with.

## Running

Always run from the repository root.

| command | what runs |
|---|---|
| `.venv/bin/python -m pytest` | default tier: small goldens (5 processes x 64 observations), no `slow` tests, no legacy benchmark; test-review checkpoint: 694 passed, 15 skipped, 9 xfailed (see [test review](TEST_REVIEW.md)); the newer workflow cases are verified separately below |
| `.venv/bin/python -m pytest --runslow` | adds slow SPI tests, the full-config `Calculator` end-to-end test and slow known-bug tests (ccm) |
| `.venv/bin/python -m pytest --pyspi-size=full --runslow --timeout=1800` | full tier: 7 x 100 goldens, everything |
| `.venv/bin/python -m pytest --runslow -m slow` | only slow tests |
| `.venv/bin/python -m pytest tests/regression -k "cov_ or dcorr"` | select SPIs by identifier |
| `.venv/bin/python -m pytest -m "not java"` | deselects tests explicitly marked `java`; use `--collect-only` to inspect the selected tests |

Options:

- `--pyspi-size small|full` picks the golden set (default `small`; environment fallback `PYSPI_TEST_SIZE`). Missing committed goldens fail; restore them rather than regenerating references to make a run pass.
- `--runslow` runs tests marked `slow`; otherwise they are reported as skipped (`slow: needs --runslow`).
- `--require-java` fails if the JIDT JVM cannot start. Use it for complete local verification, together with `--runslow` for the slow calculations.
- The archived CML7 warning-only test was removed after its old identifiers blocked collection. Current full-size golden and Calculator tests use that same CML7 dataset with strict numerical assertions; no calculations now run during collection for the removed test. The old pickle/generator remain historical artifacts, not current references.
- `-n 4` runs in parallel with pytest-xdist. Every test is independent; each worker starts its own JVM.
- `--timeout=N` (pytest-timeout) overrides the default 300 s per test. It uses `signal` and cannot interrupt code inside Java or numba frames, so it is a safety net only.

`addopts` already passes `--ignore=tests/storage`, `--strict-markers`, `--strict-config` and `-ra`; `xfail_strict = true`. It also disables pytest's `faulthandler` plugin: JPype's JVM uses its own native signal handlers, which that plugin can replace and cause a process crash. This follows [JPype's guidance](https://jpype.readthedocs.io/en/stable/userguide.html#errors-reported-by-python-fault-handler). That `--ignore` path only resolves from the repository root, so `tests/conftest.py` also sets `collect_ignore = ["storage"]`; a run started from inside `tests/` skips the storage suite too.

### Markers

| marker | meaning |
|---|---|
| `slow` | measured above `SLOW_THRESHOLD_S` (2.0 s) at the selected size (`time_s` in `meta.json`); `FORCE_SLOW` applies only when no time is recorded. Skipped unless `--runslow` |
| `java` | needs the JIDT JVM (from the SPI's `deps`) |
| `regression` | golden-value comparison |
| `stochastic` | draws from the global numpy RNG; reproducible only with `seed_all()` immediately before compute |
| `nan_golden` | golden is all-NaN (status `all_nan` or `error`); this locks missing-value or failure behavior, without finite-value coverage for that case |

`slow`, `java`, `stochastic` and `nan_golden` are attached by `tests/conftest.py`:

- per-SPI tests (`identifier`): `slow`, `nan_golden` and `java` from `goldens/<size>/meta.json` `spis[identifier]` (`time_s`, `status`, `deps`);
- preprocessing variants (`preproc_key`): `slow` and `nan_golden` from `goldens/preproc/meta.json` `variants[key]` (that variant's own `time_s` and `status`), `java` from `spis[identifier].deps` in the same file;
- `stochastic` from `spi_policy.STOCHASTIC` in both cases.

`test_extra_goldens.py` attaches the same markers itself from `goldens/extra/small/meta.json` (none applies today). The `Calculator` and catalogue tests carry `java` (and the full end-to-end test `slow`) explicitly.

`FORCE_SLOW` (used only for an identifier or preprocessing variant without a recorded `time_s`; every current golden records one): `ccm_*`, `*kraskov*theilerauto*`, `hhg`, `mgc*`, `*sgddtw*`, `softdtw*`, `anm`, `gpfit_*`, `*mincovdet*`, `*ellipticenvelope*`, `*graphicallassocv*`, `sgc_parametric*`, `cce_*`, `di_*`.

## How the SPI regression works

- `test_spi_matches_golden[<identifier>]`: builds the SPI from its `meta.json` entry (module, class, params; `params: null` records default construction, otherwise recorded params are passed without fallback), makes a fresh `Data`, seeds the RNGs, runs `multivariate`, and sets the diagonal to NaN like `Calculator.compute`. Output must be a real square process matrix. The NaN mask and success/error status must match; values use the policy tolerance. A recorded calculation error must reproduce its exact exception representation (type and message). Another exception, or a successful all-NaN return replacing an error, fails. Constructor and seeding failures always propagate.
- `test_preprocessing_variant_matches_golden[<identifier>-n?d?]` (golden key `<identifier>::n?d?`): 24 representative SPIs (`spi_policy.PREPROC_SPIS`: three to five per statistics module, except wavelet, which has one because its six SPIs are all `psi_wavelet` band variants; `anm` is included because its output changes with `normalise`, while `igci` and `reci` differ between `n1d0` and `n0d0` by at most 3e-15) under all four `(normalise, detrend)` combinations, 96 keys in all, on the small window of cml7 with a fixed per-channel affine distortion (`x * scale + offset + slope * t`, recorded in `goldens/preproc/meta.json`) so that each flag measurably changes the input. Suffix `n1d0` is the Calculator default; `n1d1` is the EEG workflow (detrend, then z-score). 92 variants have status `ok`; the four cross-correlation variants now return partial NaNs for pairs without significant lags, without calculation errors. No variant is `slow`; the slowest, `te_kraskov_neighbours04_targethistory01_targetdelay01_sourcehistory01_sourcedelay01::n1d0`, measured 1.94 s, just under the 2.0 s threshold. The 24 resolved identifiers (from `goldens/preproc/meta.json` `spis`): `anm`, `bary_euclidean_mean`, `cohmag_mean_band0-0p5cyc`, `coint_johansen_lineartrend_maxeig_lagdiffs01`, `cov_empirical`, `dcorr`, `dtw`, `gwtau`, `ids_pnormmax_terms06_bandwidth0p5`, `igci`, `kendalltau`, `lmfit_ridge_penaltysum_alpha1`, `mi_gaussian`, `pdist_euclidean`, `pec`, `phi_star_delay01`, `plv_mean_band0-0p5cyc`, `prec_empirical`, `psi_wavelet_mean_band0-0p5cyc`, `reci`, `sgc_nonparametric_mean_band0-0p5cyc`, `spearmanr`, `te_kraskov_neighbours04_targethistory01_targetdelay01_sourcehistory01_sourcedelay01`, `xcorr_holm_max_maxlag0p125_alpha0p05`.
- Two fixture checks: `spearmanr`, `kendalltau` and `dcorr` give the same golden for `n1d0` and `n0d0` (rank-based or per-variable scale-invariant; measured max difference 0, 0 and 1.9e-15), and `cov_empirical` differs between them.
- `test_calculator_table_matches_goldens_fast` (default tier, `java`): writes a temporary config holding every identifier of the package's `config.yaml` that has a golden and is not `slow` at the selected size (module and class order kept, so the shared caches fill in the same order as a full pass), checks that the `Calculator` loads exactly those identifiers, runs one seeded `Calculator` pass with shared caches and compares each table block with its per-SPI golden. At small size that is 278 SPIs (289 minus the 11 slow ones) measured here at 32 s when run alone (plus 4 s setup for the JVM and imports) and 17 s inside a full default run, where earlier tests have already paid the first-call costs. It is the default-tier test that notices a cache-keying change (two SPIs reading each other's cached spectra), which the per-SPI tests cannot see because each runs on a fresh `Data`.
- `test_calculator_table_matches_goldens` (`slow`, `java`): the same over the full `config.yaml`.
- Both end-to-end tests skip `spi_policy.CALC_E2E_SKIP` (`bary*_sgddtw_*`; 4 identifiers): a single seeded pass draws from the global RNG in a different order than per-SPI seeding, and these then differ from their goldens. The eight robust covariance SPIs (`*mincovdet`, `*ellipticenvelope`) were skipped or stochastic until pyspi fixed their FAST-MCD seed (`basic.ROBUST_RANDOM_STATE`, equal to the harness seed) and served `EllipticEnvelope` from the `MinCovDet` fit; they now reproduce their per-SPI goldens in every pass and are compared. All mismatches are reported together.
- A separate shared-cache test includes all stochastic identifiers, reseeding immediately before each SPI while retaining the shared `Data` caches. This provides numerical comparisons for the exclusions under controlled RNG conditions; the normal Calculator tests retain their existing seeding behavior.
- `test_catalogue_matches_golden_meta` (`java`): loads the imported package's `config.yaml` through both `harness.catalogue` and `Calculator`, and asserts that the identifier sets equal the recorded ones, that the harness order and classes match what the `Calculator` loaded, and that for every identifier `module`, `cls`, `params` and `deps` equal `meta.json` (after a JSON round trip) and `labels` match as sorted sets. The per-SPI tests rebuild each SPI from the recorded params, so this is what reports a `config.yaml` parameter edit that keeps the identifier (for example `nan_threshold` of the `sgc_*` SPIs, which the identifier does not record).
- `test_spi_catalogue[all|fast|sonnet|fabfour]`: subset sizes 301/233/14/4; for `all`, identifier order equals `calculator_order` and each SPI's labels equal the recorded ones.

### Tolerance policy (`regression/spi_policy.py`)

First matching `fnmatch` pattern wins.

| pattern | rtol | atol | reason |
|---|---|---|---|
| `bary*_sgddtw_*` | 1e-6 | 1e-8 | tslearn subgradient DBA shuffles with the global numpy RNG (marker `stochastic`; regenerate on a tslearn upgrade) |
| everything else | 1e-6 | 1e-8 | default; absorbs BLAS drift between machines; `atol` covers pure rounding noise (e.g. `bary_euclidean_mean` ~1e-16) |

The stochastic pattern matches 4 identifiers (the `_sq` variants included); its row keeps the default values and is listed to document that they match only because `seed_all()` runs right before them. The robust covariance SPIs (`cov*`/`prec*` `MinCovDet` and `EllipticEnvelope`) left this list when pyspi fixed their FAST-MCD seed; they are deterministic and keep the default tolerances.

The iterative optimisers (`*graphicallassocv*`, `phi_*`, `gpfit_*`, `anm`) have no row: seeded run to run and against the stored goldens they were measured bit-for-bit reproducible on the generating machine (the 11 matching identifiers, each recomputed twice at both sizes: maximum absolute difference from the stored goldens 0), so the default applies. If another machine shows drift for one of them, re-add a row for that pattern with the measured drift written in its reason, not a guessed tolerance. Overrides are documented exceptions; do not loosen a tolerance to make a failure go away.

## Basic rank and cache refactor

`pyspi/_rank.py` computes Kendall's existing tau-b coefficient with channel
sort orders and tie counts prepared once. Each unordered pair uses SciPy
1.17.1's compiled `_kendall_dis` counter; joint ties are counted separately.
The SciPy wrapper's repeated preprocessing and discarded p-values are omitted.
Ordinary and squared Kendall/Spearman variants share their signed results.
The private SciPy kernel is covered by exhaustive short tied sequences,
independent concordance counts, and larger comparisons with `kendalltau`.

`pyspi/_cache.py` supplies one input snapshot for the reviewed basic and
spectral calculations. It detects setters, channel insertion/removal, dtype
changes, and edits through writable NumPy views or the caller's input array.
Covariance/precision return independent arrays and leave their fitted model
intact; fitting warnings are visible. A single channel now returns the proper
1-by-1 NaN matrix instead of failing sklearn's two-dimensional input check.
Covariance, precision, cross-correlation, and both rank correlations explicitly
require a single realisation.

At that first basic-refactor checkpoint, cross-correlation shared channel
standard deviations and unique-pair work while preserving the previous lag,
normalization and significance definitions. The cross-correlation refactor
below now supersedes that implementation. The generic `Undirected` loop is
still unchanged: cointegration's reverse-pair cache and stochastic information
estimators require separate orientation/RNG checks before a global switch.

`unit/test_basic_refactor.py` covers numerical definitions, permutations,
constants, large integers, shared work, data mutation, output ownership, fitting
warnings. Cross-correlation preservation tests were subsequently replaced by
the independent reference tests described below. At the basic-refactor
checkpoint, numerical golden arrays
were not regenerated; only Kendall's signed/unsigned label metadata changed.
The former Kendall label expected failure is replaced by passing tests.

Run `.venv/bin/python -B tests/benchmark_basic.py` for the rank-correlation workloads
in `basic_refactor_results.json`. Its historical cross-correlation timings are
superseded by `benchmark_xcorr.py` because the definitions have changed.
The basic benchmark compares against commit `c28512b` (which
already includes the earlier Spearman optimization) with three paired repeats,
fresh Data per repeat, and one numerical-library thread. A unique-pair SciPy
alternative isolates the additional benefit of the coefficient-only Kendall
implementation. Benchmark results describe synthetic 118-by-1024 inputs,
not total workflow throughput.

## Cross-correlation refactor

Definitions and operation order are in [calculation definitions](../docs/calculation_definitions.md).
`unit/test_xcorr.py` checks FFT results against independent overlap sums,
Bartlett variances against explicit covariance-matrix products, Holm selection
against statsmodels, channel-order invariance, delayed peaks, missing reasons,
cache invalidation/eviction, and output ownership. Workflow tests check target
sampling-rate propagation and conflicting parameters.

Only cross-correlation golden arrays were refreshed: 279 unrelated arrays in
each size and 92 preprocessing arrays were verified unchanged. Partial
regeneration metadata records this scope and preserves earlier provenance.

Run `.venv/bin/python -B tests/validate_xcorr_inference.py` for the simulation
audit saved in `xcorr_inference_results.json`: 1000 independent pairs for each
of nine null settings plus one delayed-coupling setting. It is an empirical
check of approximate inference, not a guarantee for arbitrary recordings.

Run `.venv/bin/python -B tests/benchmark_xcorr.py` for fresh 118-by-1024 windows
at 2048 Hz, with three alternating repeats and one numerical thread. In
`xcorr_refactor_results.json`, four all-lag summaries took 0.178 s (8.8x faster),
four significant-lag summaries 0.506 s (3.5x), and all eight 0.576 s (3.5x).
Twelve including min/peak took 0.546 s; the small inversion is timing noise,
since those summaries are already produced with the shared computation.
Comparisons are against `3d993e0`; numerical definitions intentionally differ.
These timings exclude preprocessing and I/O and describe synthetic data.

## Spectral backend refactor

`pyspi/_spectral.py` shares one multitaper transform per input and spectral
settings. It accumulates cross-spectra directly and bounds taper-level phase
products in blocks. Derived-array caches have byte limits; at most two sampling
rate workspaces and sixteen ordered pair models are retained per workspace.
The core Fourier coefficients, powers, and coherency are shared separately.
Input snapshots invalidate spectral work after `Data` mutations or writes
through `to_numpy()`. Parametric Granger also keys its cache by sampling rate
and uses that rate for its frequency axis.

Frequencies use the units of `fs`: Hz for a sampling rate in samples/second,
or cycles/sample with the default `fs=1`. Invalid bands are rejected. Ordinary
mean/max summaries of empty bands return NaN. Multitaper requires a single
realisation and more than six observations with the current bandwidth setting.
All Fourier estimators use closed `[fmin, fmax]` bands on a one-sided grid
including DC and Nyquist (the last positive bin for odd FFT lengths). Directed
factorization still uses the full two-sided spectrum. dDTF's full-frequency
normalization includes Nyquist too. Granger no longer silently moves a zero
lower bound to `1e-5`; its default band is `[0, fs/2]`. Configurations that
previously inherited `fmin=1e-5` consequently have `fmin-0` identifiers.

Group delay selects regression points at
`ceil(multitaper.frequency_resolution / FFT_grid_spacing)` bins, where the
transform's nominal resolution is `2*NW/window_duration`. The spacing is
independent of the requested band and does not change the transform settings.
The dependency's coherence significance test, largest significant cluster,
minimum of three retained points, phase unwrapping, and fit definition are
preserved. A band too narrow to hold three spaced points raises an identified
`ValueError`; Calculator records an unavailable NaN matrix and the reason.
Bandwidth-based spacing approximates independence; it does not establish exact
statistical independence. The coherence significance procedure itself remains
unchanged. The subsequent sign review corrects `rvalue` so reversing the pair
negates it, as required by its signed Pearson-correlation definition. The
previous symmetric fill could change approximately +1 to -1 merely by
reordering channels. `GroupDelay()` now defaults to `statistic="delay"` and
rejects unsupported summaries during construction.

`unit/test_spectral_direction.py` checks explicit signed/directed traits for
PSI, group delay and coherence phase, independent OLS/correlation formulas,
known-delay signs, channel permutations and the existing row-target/column-source
convention of factorization-based directed metrics. Only labels were updated
in the small/full golden metadata; numerical golden arrays were not changed.
The sign-review checkpoint passed 1,283 unit, regression and workflow tests
with 15 skips and 7 existing expected failures. The subsequent PLI/wPLI changes
and their validation are described below, followed by the circular coherence-phase review.

The initial performance stage preserved metric definitions. The subsequent
approved multitaper PSI correction is described below. Coherence phase now uses
the circular summary described below. PLI/wPLI now use the definitions below.
Tests compare unchanged formulas against the dependency on the same bins;
`unit/test_spectral_bands.py` adds scalar endpoint formulas, even/odd frequency
and transfer-function checks, analytical group-delay fits, unit scaling,
zero-bound Granger checks, and narrow-band failure reporting. Directed pair
models retain channel order because the current factorization produces
measurably different results when that order is reversed. Direct accumulation
avoids BLAS-induced imaginary roundoff that can move exact antiphase channels
across the angle branch boundary. The performance-stage measurements in
`spectral_refactor_results.json` describe the implementation before the PSI
definition change; their source hashes and numerical comparisons are historical.

Multitaper PSI now implements the raw (unnormalized) formula in
[Nolte et al. (2008), equation 3](https://doc.ml.tu-berlin.de/causality/psi.pdf):
`sum(imag(conj(C(f)) * C(f + df)))`. It uses every existing FFT bin in the
inclusive `[fmin, fmax]` band, including DC/Nyquist when requested. Both members
of each adjacent pair must be in the band. There is no band-width-dependent
`/ 50` thinning, interpolation, or pair-count normalization. The authors'
[reference implementation](https://doc.ml.tu-berlin.de/causality/data2psi.m)
likewise includes the final frequency in the supplied list. Positive `[i, j]`
means channel `i` leads `j`; the raw matrix is antisymmetric.

Fewer than two selected bins raises an identified `ValueError`; `Calculator`
records a failed metric with the reason and an unavailable NaN matrix. Undefined
coherency invalidates affected pairs rather than bridging missing bins. Tests
in `unit/test_psi.py` check independent scalar calculations, MNE's adjacent
formula on identical coherency, known delays, endpoint inclusion, odd/even FFTs,
channel permutations, normalized/Hz units, cache ownership, bounded temporaries,
and failure reporting. The PSI correction initially regenerated only its three
multitaper snapshots. The subsequent shared frequency-selection change refreshes
Fourier spectral snapshots in the small/full and preprocessing sets; unrelated
values and records are checked unchanged. Each set records scoped regeneration
provenance instead of relabeling its older snapshots as newly generated.

This correction does not establish the suitability of short windows for narrow
bands. At 2048 Hz with 0.5 s windows, the FFT grid spacing is 2 Hz while the
current `NW=3` multitaper full smoothing bandwidth is 12 Hz. Window length and
smoothing remain a separate scientific decision. Wavelet PSI and the remaining
metric classifications have not been changed by this correction.

```bash
.venv/bin/python -B -m pytest tests/unit/test_spectral.py
.venv/bin/python -B -m pytest tests/unit/test_psi.py tests/unit/test_spectral_bands.py
.venv/bin/python -B tests/benchmark_spectral.py
.venv/bin/python -B tests/benchmark_spectral.py --workload psi_three_bands_118x1024
```

The benchmark loads the specified baseline revision into an isolated module;
there is no old calculation path in production. It records medians of three
paired runs, numerical agreement, source hashes, and peak traced allocations
(not total process RSS), with numerical libraries restricted to one thread.
Its current output defaults to `spectral_band_refactor_results.json`. The old
performance and PSI-only reports retain their original hashes and measurements.
The current report identifies the intentional formula/frequency-selection changes
per measure and records the observed differences; it is a timing comparison,
not a claim of numerical equivalence. Independent correctness checks live in
the spectral unit tests.

## Circular coherence phase

`unit/test_coherence_phase.py` checks equal-weight circular means using angle
geometry and scalar FFT references: wrap boundaries, cancellation, zero/missing
coherency, tiny defined coherency, channel permutations, blocked reductions and
cache ownership. The implementation normalizes the shared complex coherency
before summing, avoiding a full extra angle tensor. Undefined directions are
NaN. Ordinary phase maxima are removed from the default/fast configurations
and rejected at construction. Counts became 298 and 230; the retained selection at that checkpoint stayed
185 because it already contained only phase means. Wavelet phase is unchanged.

Three phase-mean arrays were regenerated and three phase-max arrays removed
from each small/full golden set. The other 295 arrays in each set, all
preprocessing arrays and all extra-wavelet arrays are unchanged. Metadata
records the scoped regeneration and source hashes.

Phase-checkpoint validation: 1,390 unit/regression/workflow checks passed, with 15 skips
and 7 existing expected failures. The separate larger multitaper numerical
regression run (`--pyspi-size=full`) passed all 99 checks. A catalogue-count
assertion was also verified in that targeted run, for 100 passed checks total.

`coherence_phase_results.json` compares against `923b9f6` on synthetic
118 x 1,024 input, seven alternating runs on one CPU and one numerical-library
thread. Run `benchmark_phase_sync.py --baseline 923b9f6 --coherence-phase
--repeats 7 --cpu 2 --output tests/coherence_phase_results.json` (choose an
available CPU). These are different approved mathematical definitions, not
equivalent-output benchmarks.

| Workload | Arithmetic baseline | Circular mean | Change |
| --- | ---: | ---: | ---: |
| Full band | 0.938 s | 0.739 s | 21.2% faster |
| Three bands | 0.864 s | 0.747 s | 13.5% faster |

Peak traced allocation fell from 244.7 to 230.9 MiB (not process RSS).
Timing varies with machine load; raw runs are retained in the receipt.
Directed-model diagnosis, approved decisions and completed refactoring are recorded
in [spectral_directed_review.md](../docs/spectral_directed_review.md).

## Completed directed/PEC refactor

`unit/test_wilson.py` validates the discrete plus identity, exact VAR transfer
functions/correlated innovations, channel permutation, numerical rank and
failure isolation. `unit/test_directed_formulas.py` checks DC-family normalization,
dDTF against direct inverse-spectrum formulas, native-grid endpoints and stable
Granger power decomposition independently of the factorizer.
`unit/test_directed_integration.py` and affected spectral/direction tests cover
source-first output, pair-local diagnostics, bounded shared fits, disabled
families, native-grid reconstruction and Data mutation. The obsolete spectral
Granger whole-matrix ValueError fallback is removed: unsupported inputs raise,
while numerical fit failures remain local to their pair.

All **49 PEC checks** passed against independent scalar definitions and
adversarial constant/duplicate/near-proportional, analytic and extreme-unit
inputs. All focused Wilson, formula and integration checks passed as well.
After the independent checks, the affected regression run passed **73**
small/preprocessing/catalogue checks and **60** larger-fixture checks. Two
remaining affected cache checks and three config/reporting checks also passed.
These were scoped runs; earlier unrelated suites and the completed GD/smoothing
experiments were not repeated.

The affected regressions can be reproduced with:

```bash
.venv/bin/python -B -m pytest tests/regression/test_spi_goldens.py --runslow --require-java -q -k '(test_spi_matches_golden and (dcoh or pdcoh or dtf or sgc or pec)) or (test_preprocessing_variant_matches_golden and (sgc or pec)) or test_catalogue_matches_golden_meta or test_spi_catalogue'
.venv/bin/python -B -m pytest tests/regression/test_spi_goldens.py --pyspi-size=full --runslow --require-java -q -k 'test_spi_matches_golden and (dcoh or pdcoh or dtf or sgc or pec)'
```

`audit/checks/probes/refresh_spectral_pec_goldens.py` refreshes only the affected
snapshots in a temporary copy and verifies unrelated arrays before writing.
Sixty arrays per main set and eight preprocessing arrays were refreshed;
238 arrays per main set and 88 preprocessing arrays were byte-identical.
Unrelated entries and previous partial-regeneration provenance were preserved.
The preprocessing catalogue-order list additionally drops the three previously
removed phase-max identifiers, without removing any numerical variant.
The first refresh stopped on that stale metadata before writing; the corrected
refresh used one generation pass rather than repeating determinism checks that
had already passed. JSON formatting was retained to keep the diff reviewable.

Runtime scripts and raw receipts are listed above. Six shared PEC variants
improve from 0.833 to 0.775 seconds (~1.08x) on 118x1024 synthetic data.
dDTF/all six directed families require new pair fits even when the original
selection has populated shared transforms: measured increments are tens of
seconds, so they remain excluded. Timing ranges, x600/x600x69 estimates,
config changes and limitations are in
[spectral_directed_review.md](../docs/spectral_directed_review.md#measured-runtime-and-selection-gate).

## Squared directed coherence: earlier checkpoint

`unit/test_directed_coherence.py` checks the approved source-variance-weighted
squared definition against independent scalar formulas and exact VAR transfer
functions. It covers unequal innovation variances, normalization over sources
including self, zero inflow, channel-unit scaling through 1e-120 to 1e120,
formula permutation equivariance, model-array ownership and squaring before
band mean/max. The known example changes from 2.9373 to 0.9878, matching the
standard formula within 2.3e-16. Existing pairwise model/orientation tests also
pass. Factorization order sensitivity was still unresolved at that checkpoint;
the completed refactor below repairs it.

Only six `dcoh` numerical arrays per small/full golden set changed in this
stage. All 292 other arrays per set, including the prior phase correction,
and the complete preprocessing/extra-wavelet arrays were preserved.
The focused spectral run passed 162 checks. The already-started broader run
finished with 1,402 passed, 15 skipped and 7 existing expected failures.
Subsequent validation was restricted to the six changed larger DC regressions:
all six passed (`--pyspi-size=full -k 'test_spi_matches_golden and
dcoh_mean and not pdcoh'`).

`benchmark_directed_coherence.py --baseline 923b9f6 --repeats 7 --cpu 2`
records the corrected implementation in `directed_coherence_results.json`.
On synthetic 4 x 1,024 data, fresh Data per run and alternating implementations,
the full-band median was 0.270 -> 0.270 seconds, six summaries 0.260 -> 0.261
seconds, and five shared directed estimators 0.269 -> 0.269 seconds. These
differences are smaller than observed run-to-run variation: no material runtime
change was measured. Existing fits are reused. Traced allocation peaks were
unchanged at about 4.4 MiB for DC alone and 5.7 MiB for the five-estimator run.
These are measured small-channel workloads, not a full-dataset projection.

## PLI/wPLI definitions and shared phase work

`unit/test_phase_lag.py` independently checks magnitude before frequency-band
aggregation, explicit signed variants, weighted averages, debiasing via distinct
observation-pair products, negative estimates, shared moment computation, bounded
taper products and cache eviction. Ordinary `pli`/`wpli` use nonnegative
per-frequency definitions; `pli-signed`/`wpli-signed` retain phase-lag signs.
At that checkpoint, debiased squared PLI/wPLI and PPC retained their numerical
definitions and negative estimates; their signed/undirected metadata was corrected.

The weighted moments requested together now share each cross-product block;
previously cached moments are reused. All Fourier band summaries use a slice of
the sorted frequency grid, avoiding a redundant tensor copy. PLV no longer takes
a second absolute value of an already nonnegative result. The new signed
variants are present in the all/fast catalogues (301/233 total SPIs), while the
retained185 selection stayed at 185 at that checkpoint; the subsequent
reviewed preset incorporates the approved additions/removals.

Only the 12 existing ordinary PLI/wPLI numerical snapshots per small/full set
are replaced; 12 signed snapshots are added. The other 277 numerical snapshots
per set and all 96 preprocessing values are preserved exactly. Debiased/PPC
metadata changes cover 18 entries per small/full set. Scoped provenance records
identify the updates. These goldens are change detectors; the independent
formula tests establish the intentional definition changes.

Validation: **1,324 passed, 15 skipped, 7 existing expected failures** across
unit, regression and workflow tests with Java required. The larger spectral
regression fixtures add **102 passing checks**.

`benchmark_phase_sync.py` compares a committed baseline with the current code
on 118 channels x 1024 observations, including single estimators, all six original
estimators (three bands, mean/max), and the two additional signed variants.
`phase_sync_refactor_results.json` records raw timings, allocation peaks, source
hashes and the workload definitions. These are synthetic measurements, not an
EEG validation. The paired baseline/current medians were 0.63/0.51 s for wPLI, 1.02/0.76 s
for debiased squared wPLI, and 4.47/3.58 s for the six-estimator workload. A
separate alternating five-repeat comparison of current code with/without the
12 signed band summaries measured about 0.094 s (3.8%) extra, using median
within-repeat differences; absolute timings varied between benchmark sessions.

The measurements above describe the signed/unsigned split at commit `8b66a4a`.

### Zero-sign and weighted-denominator corrections

Debiased PLI now counts actual nonzero signs in its distinct-observation
pair-product formula. Five zero signs return zero instead of -0.25; legitimate
negative estimates remain. The compact count is reduced from the existing
sign array, avoiding explicit observation-pair enumeration and extra taper
products. Known phase symmetries allow upper-triangle channel tiles to be
reused in reverse order across the phase family.

Signed/unsigned wPLI no longer replace small denominators with 1. Zero weight
returns NaN. Debiased wPLI computes its equivalent distinct-pair numerator and
denominator through scaled prefix sums, avoiding overflow/underflow from raw
squaring and cancellation when one weight dominates. Zero usable pair weight
also returns NaN. Band summaries skip undefined bins and leave all-undefined
bands as NaN, with their existing warnings.

Independent tests cover zero/mixed signs, valid negatives, amplitude factors
from 1e-100 through 1e100, tiny second weights down to 1e-250, all-undefined
bands, compact count overflow, bounded temporaries, and channel permutations
across tiles with multiple windows/trials/tapers. The small/full golden refresh
is scoped to 18 wPLI/signed-wPLI/dsPLI entries per set; the other 283 arrays per
set and all preprocessing arrays remain unchanged. PLV, PPC and debiased wPLI
retain their earlier numerical snapshots.

Validation: **1,373 passed, 15 skipped, 7 existing expected failures** across
unit, regression and workflow tests with Java required; **102** additional
larger spectral regression checks passed. Of the 18 refreshed entries, 12
arrays actually changed per small/full set; all 96 preprocessing arrays remain
unchanged.

`phase_zero_policy_results.json` records timings against `8b66a4a`, including
standalone dsPLI and PLI+dsPLI together. Reproduce using
`benchmark_phase_sync.py --baseline 8b66a4a --zero-policy --repeats 5 --cpu 2
--output tests/phase_zero_policy_results.json` (choose an available CPU or omit
the affinity option). This refactor's receipt is separate from earlier timings.

On synthetic 118x1024 inputs (five alternating repeats), median dsPLI time
changed from 0.297 to 0.262 s; PLI+dsPLI together from 0.329 to 0.295 s.
wPLI changed from 0.357 to 0.288 s. The stable debiased-wPLI evaluation costs
more alone (0.500 to 0.631 s), while its traced allocation peak drops from
447 to 243 MiB. All six estimators over three bands with mean/max summaries
changed from 2.522 to 2.087 s, with a traced peak of 504 to 450 MiB. These
are per-workload comparisons from this receipt, not cumulative speedups across
earlier benchmark sessions.

Taper independence remains a scientific review item; see the
[calculation definitions](../docs/calculation_definitions.md).

## Extra goldens

`regression/extra_config.yaml` lists seven wavelet classes from `pyspi/statistics/wavelet.py` that run today but are absent from `pyspi/config.yaml`: `cohmag` (`CoherenceMagnitude`), `phase` (`CoherencePhase`), `icoh` (`ImaginaryCoherence`), `plv` (`PhaseLockingValue`), `pli` (`PhaseLagIndex`), `wspli` (`weighted_PhaseLagIndex`) and `dwspli` (`debiased_weighted_squared_PhaseLagIndex`). `ppc` (`PairwisePhaseConsistency`) and `dspli` (`DebiasedSquaredWeightedPhaseLagIndex`) were removed: they are never finite on single-epoch input, and `Calculator` refuses them. Each listed class has the six band/statistic configs that `PhaseSlopeIndex` uses in `config.yaml`, giving 42 identifiers named `<id>_wavelet_{mean|max}_band{0-0p5|0-0p25|0p25-0p5}cyc`, for example `cohmag_wavelet_mean_band0-0p5cyc`. They all go through the per-`Data` `data.mne[(measure, fs)]` cache, which no SPI in the default config exercises.

The goldens are in `regression/goldens/extra/small/{values.npz,meta.json}`, generated on the small window (5 processes x 64 observations, `normalise=True`, `detrend=False`, seed 42) with mne 0.23.0. All 42 have status `ok`, and none is `slow`, `java` or `stochastic`. These are snapshots of current output, and many are degenerate on the single-epoch input: off the diagonal, `cohmag` and `plv` are 1, `dwspli` is 0 and `icoh` is rounding noise (below 2e-14, inside the 1e-8 `atol`); `plv` and `dwspli` give the same matrix for all six bands. Only `phase`, `pli` and `wspli` carry non-trivial values. They are locked as they are.

`regression/test_extra_goldens.py` has 44 tests: one check that the set's identifiers, order, constructor parameters, dependencies, and labels match `extra_config.yaml`, 42 per-SPI tests, and one `Calculator` end-to-end test checking the shared cache. They need no Java, use their own stored dataset spec independently of `--pyspi-size`, and fail if the required golden files are absent. Nondegenerate frequency/time tensors in `unit/test_numeric_boundaries.py` supplement these snapshots, checking aggregation, endpoint selection, angle/magnitude transforms, orientation, and cache reuse without claiming to validate MNE's estimator formulas.

## Regenerating the goldens

**Regenerate only for an intentional change to an output, and say why in the commit message.** A failing golden test after an optimisation is the suite doing its job: first find out why the value moved.

Checks also compare dataset/seed and failure identity. Partial regeneration
refuses to mix a different dataset or seed into the existing set. Detected
non-reproducible repeats fail `--check` and refuse to write that unstable set.
Phi checking compares its defining parameter metadata as well as its values.

```bash
# check the current code against the goldens; writes nothing, exit 1 on drift
.venv/bin/python tests/regression/generate_goldens.py --size all --preproc --phi --check

# the extra goldens have their own check (same command as their regeneration below, plus --check)

# regenerate (two repeats check determinism; disagreements refuse to write)
.venv/bin/python tests/regression/generate_goldens.py --size small --repeat 2
.venv/bin/python tests/regression/generate_goldens.py --size full --repeat 2
.venv/bin/python tests/regression/generate_goldens.py --preproc --repeat 2
.venv/bin/python tests/regression/generate_goldens.py --phi
.venv/bin/python tests/regression/generate_goldens.py --size small --repeat 2 \
    --configfile tests/regression/extra_config.yaml --out tests/regression/goldens/extra   # extra goldens

# regenerate only some SPIs (merged into the existing set)
.venv/bin/python tests/regression/generate_goldens.py --size all --spis 'cov_*'
```

`--size` takes `small`, `full` or `all` (both). `--spis GLOB` is an fnmatch filter on identifiers; when writing, the result is merged into the existing set and stored SPIs that are no longer in the catalogue are dropped (and printed). `--out DIR` writes elsewhere (default `tests/regression/goldens`). `--configfile PATH` uses another SPI config (default `pyspi/config.yaml`). `--check` compares values with the `spi_policy` tolerances and also compares each identifier's catalogue fields (`module`, `cls`, `params`, `deps`, and `labels` as sets) with the stored meta, printing a `CATALOGUE CHANGE` line for each mismatch. After regenerating, run `--check` once more on the same environment; it must pass. After upgrading sklearn or tslearn, investigate stochastic-output changes before approving any regeneration; an upgrade alone is not a reason to accept changed values. Refresh the exported `tests/requirements-test-lock.txt` from the root lock when the environment intentionally changes.

A parameter change in `config.yaml` that keeps the identifier fails `test_catalogue_matches_golden_meta` (and `--check`) until the goldens are regenerated. That is expected for an intentional change; regenerate and say why in the commit message.

## Run hotspots: robust covariance and phase-moment sharing

`benchmark_run_hotspots.py` times every metric of the reviewed preset on one
synthetic 118x1024 window through `Calculator.compute` (one numerical-library
thread; loading, preprocessing and writing excluded) and writes a receipt with
the per-metric medians. `run_hotspots_results.json` summarises the series:
the window went from 20.40 s to 11.75 s. Run
`.venv/bin/python tests/benchmark_run_hotspots.py --label after --repeats 3 --out tests/run_hotspots/after.json`.
Those receipts are historical full-rank synthetic measurements with their
recorded 200-to-198 selection, not a performance claim for the current 195-output
CAR run. Current combined measurements are linked below.

- `EllipticEnvelope` serves from the cached `MinCovDet` fit (its `fit` is
  `MinCovDet.fit` plus an outlier offset the covariance never uses), and both
  use the fixed seed `basic.ROBUST_RANDOM_STATE = 42`, which equals the
  harness seed. The eight robust goldens are unchanged (max_abs 0), the
  identifiers left `STOCHASTIC`/`CALC_E2E_SKIP`, and the two
  `prec*_EllipticEnvelope` requests left the reviewed preset as exact
  duplicates (198 requests).
- `pyspi._robust.FastMinCovDet` copies scikit-learn 1.9.1's FAST-MCD with
  Cholesky Mahalanobis C-steps (3.9 s -> 1.2 s at 118x1024). The same seed
  selects the same supports (`unit/test_robust.py`, five synthetic sets, and
  the benchmark window). Integration found a material support/covariance change
  on singular CAR input. Such inputs now use the complete reference MinCovDet
  fit, verified against scikit-learn supports/covariance/precision for three
  CAR controls. The full-rank shortcut remains.
- Phase-synchrony classes register their taper moments before compute
  (`PHASE_MOMENTS`). The original prototype formed every moment in one pass
  and expanded its cache, reducing 4.46 s to 2.67 s at 118x1024 while adding
  about 520 MB peak RSS. The integrated implementation instead fuses moments
  only when they fit the configured cache. Small/large-cache equality and
  bounded retention are covered in `unit/test_phase_lag.py`.
- `GraphicalLassoCV` (4.2 s) is unchanged by decision: its cost is 80
  scikit-learn glasso fits, `n_jobs` would only move the time onto more cores.

### Integrated workflow/resource validation

The full `--runslow --pyspi-size=full --require-java` run passed **1,777 tests**
with **seven known expected failures**; the separate storage suite passed
**439 tests**. No numerical goldens changed. The superseded collection-time
`test_SPIs.py` was removed after its old cross-correlation identifiers failed;
strict current-catalogue tests use the same full `cml7.npy` data and assert
numerical differences. Subsequent queue/dependency changes received focused
checks rather than another full run.

[`workflow/resource_results.json`](workflow/resource_results.json) records
verification and packaging evidence. The
[CAR resource profiles](workflow/resource_profiles/README.md) compare cache
sizes, worker counts and numerical threads on four 118x1024 windows, including
preparation/output and process-tree memory. All 195 outputs matched for the
cache/worker comparisons and final JVM-free workflow. Two numerical threads
changed seven convergence-sensitive graphical-lasso outputs; retain one.
[`workflow/benchmark_resources.py`](workflow/benchmark_resources.py) reproduces
the comparison with fresh processes and checkpoints completed cases.

## Reproducing failures across environments

Install the root `uv.lock` with `uv sync --locked --extra testing --extra storage`, then use the calculation and storage commands above. Select the same default/full tier and slow-test options as the run being investigated.

Compare the checkout, lock, selected test tier, installed packages (`uv pip check` and `uv pip freeze`), Python and JVM versions, and thread settings between environments. Local verification uses Java 21; vendor, patch, platform, BLAS, and runtime differences can still matter with identical Python package versions. Package drift can follow an intentional lock update or a local install outside the lock; a failing test alone does not identify its cause. The versions in golden `meta.json` describe the original generation environment, not necessarily the current lock. `generate_goldens.py --check` reports per-SPI differences; investigate those before regenerating values or changing tolerances.

## Known bugs (locked, not fixed)

The confirmed implementation issues below are locked in `unit/test_known_bugs.py`, generally with a current-behavior test and a strict xfail describing a proposed change. These are change alarms; they do not independently establish the mathematical correctness of the proposed behavior. Item 5 records the agreed standard PLI definition, backed by independent scalar tests.

1. **Fixed:** `TransferEntropy(estimator='symbolic')` computes ordinal-pattern TE with explicit pattern length, spacing and prediction shift. Independent categorical-count tests replace the former Gaussian-dispatch bug lock.
2. `ConvergentCrossMapping(embedding_dimension=None)` picks max E (10), not best-rho E (`ccm_E-None == ccm_E-10`). Both tests are `slow`.
3. `SoftDynamicTimeWarping` ignores `global_constraint` (three identical configs).
4. **Fixed:** `KendallTau` and `SpearmanR` own their instance labels. Their signed/unsigned label metadata in the small/full/preprocessing golden sets was corrected without regenerating numerical values.
5. Standard PLI now takes absolute value at each frequency before the band summary and returns a symmetric nonnegative matrix. Explicit `pli-signed`/`wpli-signed` variants retain phase-lag signs. Independent scalar and pair-product tests in `unit/test_phase_lag.py` establish the definitions, including valid negative debiased estimates; this test only guards standard PLI symmetry.
6. `Calculator` duplicate-identifier check is dead code (dict keys are unique; collisions overwrite silently).
7. `Data(procnames=...)` is validated but never stored.
8. `Data.convert_to_numpy` for `.csv` passes `','` as dtype.
9. `CorrelationFrame(calc_frame)` calls the nonexistent `getstatlabels()`.
10. `utils.convert_mdf_to_ddf` raises under pandas>=3 (`stack(dropna=...)` is rejected); found while writing the unit tests, locked without a paired xfail.

Bug 9 needs the JVM (`java`). **Some SPIs are never exercised by a default run.** The bug-2 tests are `slow`, all nine `ccm_*` goldens and the two auto-embedding `te_kraskov_*_autoembed_*` goldens are `slow` at both sizes, and the six `ddtf_multitaper_*` and `mgcx_maxlag10` goldens are `slow` at full size, so none of these runs without `--runslow`. After touching ccm, TE kraskov auto-embedding, ddtf or mgcx, run `.venv/bin/python -m pytest --runslow -k "ccm or te_kraskov" --ignore=tests/test_SPIs.py` (20 tests at small size) and the full tier before trusting a green default run.

### Flagged for investigation (locked as-is, not confirmed bugs)

- **Degenerate outputs in the extra wavelet set** (`goldens/extra/small`, the seven `*_wavelet_*` classes absent from `config.yaml`). Off the diagonal, `cohmag` and `plv` are exactly 1.0 everywhere, `dwspli` is 0 everywhere, `icoh` is rounding noise (about 1e-16), and those four are identical for all six band/statistic configs. `ppc` and `dspli` were `-inf` everywhere and are now refused. Only `phase`, `pli` and `wspli` vary with the input and the band. The wrapper passes a single epoch (`1 x processes x samples`) to `mne_connectivity.spectral_connectivity_epochs`, replacing the original `mne.connectivity.spectral_connectivity` path during migration while preserving its outputs. The likely cause of the degeneracy is aggregation across a single epoch: phase consistency is trivially 1 and pairwise phase consistency divides by `n(n-1) = 0`. The six `psi_wavelet_*` SPIs in `config.yaml` use a separate cache (`data.mne_psi`); their nondegenerate snapshots do not validate their axis selection or formula. Estimator usability remains a scientific review item; the goldens preserve the observed values so changes are detected.

### Dependency constraints that affect calculations

- **`spectral-connectivity` is pinned to 1.1.2 in `pyproject.toml`.** Version 2 requires a 3-D `(time, trials, signals)` input and changes the frequency grid. The existing wrapper passes 2-D; an unadapted upgrade turns its multitaper outputs into all-NaN blocks through `Calculator.compute`. An upgrade needs API and numerical validation. The packaged adapter supports the retained version on NumPy 2 without editing dependency files.
- **`torch` is installed but no calculation runs on it.** `pyspi/statistics/causal.py` imports ANM, CDS, IGCI and RECI from `cdt.causality.pairwise`; those four are numpy, scikit-learn and scipy code. `cdt` imports torch at package import time for its neural-network algorithms (SAM, CGNN, NCC) and does not declare it as a dependency, so pyspi lists torch itself; blocking the torch import makes `import cdt` fail. The CPU wheel is sufficient (see Environment). Dropping torch would mean vendoring the four pairwise functions out of cdt or stubbing the import before `cdt` loads; both are package changes and have not been made. The `anm`, `cds`, `igci` and `reci` goldens would show any numeric difference from such a change.

Also worth knowing (not bugs): `utils.normalise` default `nan_policy='propogate'` always raises; the spectral cache's first insert is keyed by `self.measure`, causing one redundant recompute; `dtf_* == pdcoh_*`, and auto-embedding TE kraskov equals fixed k=1 on cml7.

## Storage suite

`tests/storage/` has its own `pytest.ini` and runs explicitly in the same root-locked environment as calculations. The default root invocation ignores it. Run it as described in [storage/README.md](storage/README.md); there is no separate storage-test project or lock.

## Workflow preflight suite

Run `.venv/bin/python -m pytest tests/workflow --require-java` for the configuration
and header-only preflight goal. These tests prohibit signal Data construction and
Calculator computation, and include Java-dependent metric constructor checks.
The suite is also collected by the normal root invocation. See the
[workflow guide](../docs/workflow.md) and [verification receipt](workflow/verification.json)
for its independently recorded results; earlier full calculation-suite receipts
remain historical evidence.

## Regression and Euclidean-barycenter refactor

`tests/unit/test_regression_refactor.py` independently checks the exact
single-predictor Ridge/Lasso/ElasticNet objectives, the retained BayesianRidge
updates and cubic in-sample RECI, including raw units, constants, invalid rows,
near-exact fits, cache ownership, mutation and shared cross-products. Its 53
cases passed. `tests/unit/test_euclidean_barycenter.py` adds eight direct-series
and normalized-window redundancy checks, including square-before-summary.

The approved ElasticNet solver correction and exact per-sample Ridge replacing
SGD have a scoped golden refresh recorded in small/full metadata; unrelated
arrays are preserved. Preprocessing metadata changes only catalogue order.
Affected golden checks passed (18 small/preprocessing/catalogue, 9 full).
Timings and definition decisions are in
[`docs/regression_gaussian_review.md`](../docs/regression_gaussian_review.md).
These counts cover the regression chunk; the completed Gaussian/history checks follow.

## SPI coverage

The first native Gaussian group has independent coverage in
`tests/unit/test_gaussian_native.py`: 21 cases across covariance formulas,
raw/extreme units, source/target orientation, permutation, near-perfect fits,
local singular/nonfinite handling, cache ownership/invalidation, normalization
redundancy, serialization and JVM-free construction/execution. Numerical
goldens for JE/CE/MI/TLMI are retained; metadata updates only remove their Java
requirements and correct JE/CE signed labels. Nine affected small/preprocessing/catalogue
checks and four full Gaussian checks passed without changing numerical arrays.

History-based native Gaussian work now has 39 passing independent checks in
`unit/test_gaussian_history.py`: OLS/determinant references, fixed/automatic
histories, known directed models, DI without target-only inflation, exact XME
lag counts, units, degeneracy, mutation and bounded-cache behavior. A filtered
design checks the reciprocal-condition guard that the EEG pilot showed was
necessary; Cholesky diagonal thresholds alone were insufficient.
`unit/test_information_history.py` has 20 passing checks against independently
constructed Java embeddings and entropy calls, plus sharing, mutation and
serialization. `unit/test_symbolic_transfer.py` has nine independent ordinal
count/direction/edge checks. Family benchmarks and the EEG/runtime gates are now complete. Only the eight
intentionally changed numerical outputs were refreshed at each size, with
provenance. The final affected checks passed: 52 small/preprocessing/catalogue,
39 full-size and 10 integration/configuration checks. That goal ended with
200 outputs; its selection and all runtime/memory projections are recorded in
`docs/regression_gaussian_review.md`.

The subsequent kernel port has independent box-count references in
`unit/test_kernel_native.py` for all nine classes, including temporal exclusion,
TE bandwidth alignment, units, known models, permutations, cache invalidation,
bounded streaming and diagnostics. Empty eligible comparison sets return NaN
with a workflow-recorded warning. A fresh-process test confirms construction,
dependency resolution and execution without starting Java. Small and full
kernel numerical goldens are unchanged; only Java dependency metadata changes.
Focused validation, Java comparisons, and the 118-channel EEG family timing
are recorded in [`kernel_refactor_results.json`](kernel_refactor_results.json).
The main/amplitude run selections remain 188/12.

The later systematic amplitude review covered all 200 outputs and selected 12
for the unnormalized alternative. Five normalized redundancies were removed
using independent timings, leaving 195 in `workflow/reviewed.yaml`. The affected
inventory test passed; completed numerical regressions were not repeated.
`review_amplitude_selection.py` records scale/unit checks, and
`benchmark_amplitude_selection.py` records both whole-family and standalone
costs (`--standalone-only` appends the latter without repeating the former).
`benchmark_redundant_representatives.py` compares equivalent outputs with fresh
Data/cache per calculation. Decisions and receipt links are in
`docs/amplitude_selection_review.md` and `audit/amplitude_review.csv`.

Historical 289-identifier inventory before the signed phase-lag additions, in `Calculator` order, from `goldens/small/meta.json` (5 x 64) and `goldens/full/meta.json` (7 x 100). `time_s` is the generator's fastest measured time for one `multivariate` call on the generating machine (a 64-core machine with `OMP_NUM_THREADS=1`; versions and the thread setting are in `meta.json`); `slow` is the marker at that size.

| size | ok | partial_nan | all_nan / error | slow (skipped by default) | per-SPI tests run by the default tier |
|---|---:|---:|---:|---:|---:|
| small (5 x 64) | 274 | 14 | 1 | 11 | 278 |
| full (7 x 100) | 280 | 9 | 0 | 12 | 277 |

The default tier's fast `Calculator` end-to-end test computes the same non-slow SPIs (278 at small size) in one pass and compares all of them except the 8 in `CALC_E2E_SKIP`. Missing-value markers and partial-NaN statuses follow the current metadata below. Cross-correlation missing values record empty significant-lag selections; the previous preprocessing IndexError is fixed.

Information-family rows below reflect current golden metadata. Generation times
come from their individual snapshots and are not production benchmarks; use the
118-channel family receipts above for performance comparisons.

<!-- Generated from goldens/{small,full}/meta.json with a short inline Python snippet (slow = spi_policy.is_slow(identifier, time_s), stochastic = spi_policy.is_stochastic, java = "java" in deps); regenerate rather than hand-edit. -->
| identifier | module | status (small) | time_s (small) | slow (small) | status (full) | time_s (full) | slow (full) | stochastic | java |
|---|---|---|---:|:-:|---|---:|:-:|:-:|:-:|
| `cov_empirical` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_ellipticenvelope` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov_graphicallasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `cov_graphicallassocv` | .statistics.basic | ok | 0.227 |  | ok | 0.328 |  |  |  |
| `cov_ledoitwolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_mincovdet` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov_oas` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_shrunk` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_sq_empirical` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_sq_ellipticenvelope` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov_sq_graphicallasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `cov_sq_graphicallassocv` | .statistics.basic | ok | 0.233 |  | ok | 0.332 |  |  |  |
| `cov_sq_ledoitwolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_sq_mincovdet` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov_sq_oas` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_sq_shrunk` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_empirical` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_ellipticenvelope` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `prec_graphicallasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `prec_graphicallassocv` | .statistics.basic | ok | 0.226 |  | ok | 0.333 |  |  |  |
| `prec_ledoitwolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_mincovdet` | .statistics.basic | ok | 0.021 |  | ok | 0.023 |  | yes |  |
| `prec_oas` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_shrunk` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_sq_empirical` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_sq_ellipticenvelope` | .statistics.basic | ok | 0.021 |  | ok | 0.023 |  | yes |  |
| `prec_sq_graphicallasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `prec_sq_graphicallassocv` | .statistics.basic | ok | 0.225 |  | ok | 0.331 |  |  |  |
| `prec_sq_ledoitwolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_sq_mincovdet` | .statistics.basic | ok | 0.021 |  | ok | 0.023 |  | yes |  |
| `prec_sq_oas` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_sq_shrunk` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `spearmanr_sq` | .statistics.basic | ok | 0.005 |  | ok | 0.011 |  |  |  |
| `spearmanr` | .statistics.basic | ok | 0.005 |  | ok | 0.011 |  |  |  |
| `kendalltau_sq` | .statistics.basic | ok | 0.005 |  | ok | 0.010 |  |  |  |
| `kendalltau` | .statistics.basic | ok | 0.005 |  | ok | 0.010 |  |  |  |
| `xcorr_holm_max_maxlag0p125_alpha0p05` | .statistics.basic | partial_nan | 0.001 |  | partial_nan | 0.001 |  |  |  |
| `xcorr_sq_holm_max_maxlag0p125_alpha0p05` | .statistics.basic | partial_nan | 0.001 |  | partial_nan | 0.001 |  |  |  |
| `xcorr_holm_mean_maxlag0p125_alpha0p05` | .statistics.basic | partial_nan | 0.001 |  | partial_nan | 0.001 |  |  |  |
| `xcorr_sq_holm_mean_maxlag0p125_alpha0p05` | .statistics.basic | partial_nan | 0.001 |  | partial_nan | 0.001 |  |  |  |
| `xcorr_mean_maxlag0p125` | .statistics.basic | ok | 0.001 |  | ok | 0.001 |  |  |  |
| `xcorr_sq_mean_maxlag0p125` | .statistics.basic | ok | 0.000 |  | ok | 0.001 |  |  |  |
| `xcorr_min_maxlag0p125` | .statistics.basic | ok | 0.001 |  | ok | 0.001 |  |  |  |
| `xcorr_holm_min_maxlag0p125_alpha0p05` | .statistics.basic | partial_nan | 0.001 |  | partial_nan | 0.001 |  |  |  |
| `xcorr_peak_maxlag0p125` | .statistics.basic | ok | 0.000 |  | ok | 0.001 |  |  |  |
| `xcorr_holm_peak_maxlag0p125_alpha0p05` | .statistics.basic | partial_nan | 0.001 |  | partial_nan | 0.001 |  |  |  |
| `pdist_euclidean` | .statistics.distance | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pdist_cityblock` | .statistics.distance | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pdist_cosine` | .statistics.distance | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pdist_chebyshev` | .statistics.distance | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pdist_canberra` | .statistics.distance | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pdist_braycurtis` | .statistics.distance | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `dcorr` | .statistics.distance | ok | 0.009 |  | ok | 0.023 |  |  |  |
| `dcorr_biased` | .statistics.distance | ok | 0.009 |  | ok | 0.023 |  |  |  |
| `mgc` | .statistics.distance | ok | 0.054 |  | ok | 0.178 |  |  |  |
| `hsic` | .statistics.distance | ok | 0.018 |  | ok | 0.047 |  |  |  |
| `hsic_biased` | .statistics.distance | ok | 0.018 |  | ok | 0.047 |  |  |  |
| `hhg` | .statistics.distance | ok | 0.043 |  | ok | 0.295 |  |  |  |
| `mgcx_maxlag01` | .statistics.distance | ok | 0.239 |  | ok | 0.768 |  |  |  |
| `mgcx_maxlag10` | .statistics.distance | ok | 0.687 |  | ok | 2.278 | yes |  |  |
| `dcorrx_maxlag01` | .statistics.distance | ok | 0.025 |  | ok | 0.066 |  |  |  |
| `dcorrx_maxlag10` | .statistics.distance | ok | 0.103 |  | ok | 0.285 |  |  |  |
| `dtw` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `dtw_itakura` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `dtw_sakoechiba` | .statistics.distance | ok | 0.005 |  | ok | 0.015 |  |  |  |
| `softdtw` | .statistics.distance | ok | 0.013 |  | ok | 0.042 |  |  |  |
| `softdtw_itakura` | .statistics.distance | ok | 0.013 |  | ok | 0.042 |  |  |  |
| `softdtw_sakoechiba` | .statistics.distance | ok | 0.013 |  | ok | 0.041 |  |  |  |
| `lcss` | .statistics.distance | ok | 0.005 |  | ok | 0.017 |  |  |  |
| `lcss_itakura` | .statistics.distance | ok | 0.005 |  | ok | 0.017 |  |  |  |
| `lcss_sakoechiba` | .statistics.distance | ok | 0.005 |  | ok | 0.017 |  |  |  |
| `bary_euclidean_mean` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `bary_euclidean_max` | .statistics.distance | ok | 0.005 |  | ok | 0.015 |  |  |  |
| `bary_dtw_mean` | .statistics.distance | ok | 0.006 |  | ok | 0.023 |  |  |  |
| `bary_dtw_max` | .statistics.distance | ok | 0.005 |  | ok | 0.022 |  |  |  |
| `bary_sgddtw_mean` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary_sgddtw_max` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary_softdtw_mean` | .statistics.distance | ok | 0.319 |  | ok | 1.124 |  |  |  |
| `bary_softdtw_max` | .statistics.distance | ok | 0.324 |  | ok | 1.143 |  |  |  |
| `bary_sq_euclidean_mean` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `bary_sq_euclidean_max` | .statistics.distance | ok | 0.005 |  | ok | 0.015 |  |  |  |
| `bary_sq_dtw_mean` | .statistics.distance | ok | 0.005 |  | ok | 0.024 |  |  |  |
| `bary_sq_dtw_max` | .statistics.distance | ok | 0.005 |  | ok | 0.023 |  |  |  |
| `bary_sq_sgddtw_mean` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary_sq_sgddtw_max` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary_sq_softdtw_mean` | .statistics.distance | ok | 0.322 |  | ok | 1.115 |  |  |  |
| `bary_sq_softdtw_max` | .statistics.distance | ok | 0.320 |  | ok | 1.114 |  |  |  |
| `gwtau` | .statistics.distance | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `anm` | .statistics.causal | ok | 0.246 |  | ok | 0.827 |  |  |  |
| `cds` | .statistics.causal | ok | 0.061 |  | ok | 0.213 |  |  |  |
| `reci` | .statistics.causal | ok | 0.036 |  | ok | 0.072 |  |  |  |
| `igci` | .statistics.causal | ok | 0.018 |  | ok | 0.043 |  |  |  |
| `ccm_autodim_mean` | .statistics.causal | ok | 3.373 | yes | ok | 14.601 | yes |  |  |
| `ccm_autodim_max` | .statistics.causal | ok | 3.348 | yes | ok | 14.736 | yes |  |  |
| `ccm_autodim_diff` | .statistics.causal | ok | 3.365 | yes | ok | 15.000 | yes |  |  |
| `ccm_mean_dim01` | .statistics.causal | ok | 2.899 | yes | ok | 12.590 | yes |  |  |
| `ccm_max_dim01` | .statistics.causal | ok | 2.912 | yes | ok | 12.658 | yes |  |  |
| `ccm_diff_dim01` | .statistics.causal | ok | 2.903 | yes | ok | 12.313 | yes |  |  |
| `ccm_mean_dim10` | .statistics.causal | ok | 3.322 | yes | ok | 14.419 | yes |  |  |
| `ccm_max_dim10` | .statistics.causal | ok | 3.338 | yes | ok | 14.442 | yes |  |  |
| `ccm_diff_dim10` | .statistics.causal | ok | 3.323 | yes | ok | 14.558 | yes |  |  |
| `je_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.005 |  |  |  |
| `je_kozachenko` | .statistics.infotheory | ok | 0.002 |  | ok | 0.014 |  |  | yes |
| `je_kernel_width0p5` | .statistics.infotheory | ok | 0.002 |  | ok | 0.009 |  |  |  |
| `ce_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.004 |  |  |  |
| `ce_kozachenko` | .statistics.infotheory | ok | 0.002 |  | ok | 0.008 |  |  | yes |
| `ce_kernel_width0p5` | .statistics.infotheory | ok | 0.002 |  | ok | 0.005 |  |  |  |
| `cce_gaussian_horizon05` | .statistics.infotheory | ok | 0.037 |  | ok | 0.113 |  |  |  |
| `cce_kozachenko_horizon05` | .statistics.infotheory | ok | 0.052 |  | ok | 0.186 |  |  | yes |
| `cce_kernel_width0p5_horizon05` | .statistics.infotheory | ok | 1.406 |  | ok | 0.955 |  |  |  |
| `xme_gaussian_history01` | .statistics.infotheory | ok | 0.003 |  | ok | 0.010 |  |  |  |
| `xme_kozachenko_history01` | .statistics.infotheory | ok | 0.004 |  | ok | 0.020 |  |  | yes |
| `xme_kernel_width0p5_history01` | .statistics.infotheory | ok | 0.004 |  | ok | 0.015 |  |  |  |
| `xme_gaussian_history10` | .statistics.infotheory | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `xme_kozachenko_history10` | .statistics.infotheory | ok | 0.006 |  | ok | 0.013 |  |  | yes |
| `xme_kernel_width0p5_history10` | .statistics.infotheory | ok | 0.002 |  | ok | 0.006 |  |  |  |
| `di_gaussian_horizon05` | .statistics.infotheory | ok | 0.006 |  | ok | 0.006 |  |  |  |
| `di_kozachenko_horizon05` | .statistics.infotheory | ok | 0.043 |  | ok | 0.113 |  |  | yes |
| `di_kernel_width0p5_horizon05` | .statistics.infotheory | ok | 1.780 |  | ok | 0.770 |  |  |  |
| `si_gaussian_delay01` | .statistics.infotheory | ok | 0.009 |  | ok | 0.027 |  |  |  |
| `si_kozachenko_delay01` | .statistics.infotheory | ok | 0.013 |  | ok | 0.046 |  |  | yes |
| `si_kernel_width0p5_delay01` | .statistics.infotheory | ok | 0.014 |  | ok | 0.047 |  |  |  |
| `mi_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `mi_kraskov_neighbours04` | .statistics.infotheory | ok | 0.426 |  | ok | 0.228 |  |  | yes |
| `mi_kraskov_theilerauto_neighbours04` | .statistics.infotheory | ok | 0.283 |  | ok | 0.226 |  |  | yes |
| `mi_kernel_width0p25` | .statistics.infotheory | ok | 0.002 |  | ok | 0.007 |  |  |  |
| `tlmi_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `tlmi_kraskov_neighbours04` | .statistics.infotheory | ok | 0.115 |  | ok | 0.220 |  |  | yes |
| `tlmi_kraskov_theilerauto_neighbours04` | .statistics.infotheory | ok | 0.105 |  | ok | 0.224 |  |  | yes |
| `tlmi_kernel_width0p25` | .statistics.infotheory | ok | 0.002 |  | ok | 0.006 |  |  |  |
| `te_kraskov_autoembed_maxcorrais_neighbours04_maxtargethistory10_maxtargetdelay04` | .statistics.infotheory | ok | 12.967 | yes | ok | 16.209 | yes |  | yes |
| `te_kraskov_theilerauto_autoembed_maxcorrais_neighbours04_maxtargethistory10_maxtargetdelay04` | .statistics.infotheory | ok | 13.883 | yes | ok | 16.197 | yes |  | yes |
| `te_kraskov_theilerauto_neighbours04_targethistory02_targetdelay01_sourcehistory01_sourcedelay01` | .statistics.infotheory | ok | 0.515 |  | ok | 0.234 |  |  | yes |
| `te_kraskov_theilerauto_neighbours04_targethistory01_targetdelay01_sourcehistory01_sourcedelay01` | .statistics.infotheory | ok | 0.323 |  | ok | 0.232 |  |  | yes |
| `te_kraskov_neighbours04_targethistory01_targetdelay01_sourcehistory01_sourcedelay01` | .statistics.infotheory | ok | 0.527 |  | ok | 0.226 |  |  | yes |
| `te_kernel_width0p25_targethistory01` | .statistics.infotheory | ok | 0.004 |  | ok | 0.005 |  |  |  |
| `gc_gaussian_autoembed_maxcorrais_maxtargethistory10_maxtargetdelay02` | .statistics.infotheory | ok | 0.045 |  | ok | 0.075 |  |  |  |
| `gc_gaussian_targethistory01_targetdelay01_sourcehistory01_sourcedelay01` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `te_symbolic_patternlength03_spacing04_shift04` | .statistics.infotheory | ok | 0.001 |  | ok | 0.001 |  |  |  |
| `te_symbolic_patternlength03_spacing16_shift16` | .statistics.infotheory | ok | 0.001 |  | ok | 0.001 |  |  |  |
| `phi_star_delay01` | .statistics.infotheory | ok | 0.036 |  | ok | 0.077 |  |  |  |
| `phi_star_normalized_delay01` | .statistics.infotheory | ok | 0.037 |  | ok | 0.082 |  |  |  |
| `phi_geo_delay01` | .statistics.infotheory | ok | 0.097 |  | ok | 0.190 |  |  |  |
| `phi_geo_normalized_delay01` | .statistics.infotheory | ok | 0.097 |  | ok | 0.191 |  |  |  |
| `phase_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `psi_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.007 |  |  |  |
| `psi_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.003 |  | ok | 0.005 |  |  |  |
| `psi_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.003 |  | ok | 0.005 |  |  |  |
| `plv_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.003 |  | ok | 0.004 |  |  |  |
| `plv_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.003 |  | ok | 0.004 |  |  |  |
| `plv_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `plv_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `plv_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `plv_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_max_band0-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_max_band0-0p25cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dtf_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.403 |  | ok | 1.260 |  |  |  |
| `dtf_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.422 |  | ok | 1.271 |  |  |  |
| `dtf_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.415 |  | ok | 1.269 |  |  |  |
| `dtf_max_band0-0p5cyc` | .statistics.spectral | ok | 0.416 |  | ok | 1.283 |  |  |  |
| `dtf_max_band0-0p25cyc` | .statistics.spectral | ok | 0.412 |  | ok | 1.287 |  |  |  |
| `dtf_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.406 |  | ok | 1.285 |  |  |  |
| `dcoh_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.616 |  | ok | 1.886 |  |  |  |
| `dcoh_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.607 |  | ok | 1.888 |  |  |  |
| `dcoh_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.595 |  | ok | 1.879 |  |  |  |
| `dcoh_max_band0-0p5cyc` | .statistics.spectral | ok | 0.625 |  | ok | 1.890 |  |  |  |
| `dcoh_max_band0-0p25cyc` | .statistics.spectral | ok | 0.628 |  | ok | 1.893 |  |  |  |
| `dcoh_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.589 |  | ok | 1.912 |  |  |  |
| `pdcoh_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.418 |  | ok | 1.314 |  |  |  |
| `pdcoh_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.419 |  | ok | 1.299 |  |  |  |
| `pdcoh_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.407 |  | ok | 1.293 |  |  |  |
| `pdcoh_max_band0-0p5cyc` | .statistics.spectral | ok | 0.399 |  | ok | 1.281 |  |  |  |
| `pdcoh_max_band0-0p25cyc` | .statistics.spectral | ok | 0.400 |  | ok | 1.300 |  |  |  |
| `pdcoh_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.414 |  | ok | 1.347 |  |  |  |
| `gpdcoh_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.612 |  | ok | 1.958 |  |  |  |
| `gpdcoh_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.599 |  | ok | 1.956 |  |  |  |
| `gpdcoh_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.594 |  | ok | 1.950 |  |  |  |
| `gpdcoh_max_band0-0p5cyc` | .statistics.spectral | ok | 0.607 |  | ok | 1.948 |  |  |  |
| `gpdcoh_max_band0-0p25cyc` | .statistics.spectral | ok | 0.611 |  | ok | 1.958 |  |  |  |
| `gpdcoh_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.604 |  | ok | 1.949 |  |  |  |
| `ddtf_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.799 |  | ok | 2.563 | yes |  |  |
| `ddtf_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.807 |  | ok | 2.502 | yes |  |  |
| `ddtf_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.808 |  | ok | 2.490 | yes |  |  |
| `ddtf_max_band0-0p5cyc` | .statistics.spectral | ok | 0.797 |  | ok | 2.490 | yes |  |  |
| `ddtf_max_band0-0p25cyc` | .statistics.spectral | ok | 0.801 |  | ok | 2.491 | yes |  |  |
| `ddtf_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.795 |  | ok | 2.491 | yes |  |  |
| `gd_delay_band0-0p5cyc` | .statistics.spectral | partial_nan | 0.005 |  | partial_nan | 0.012 |  |  |  |
| `gd_delay_band0-0p25cyc` | .statistics.spectral | partial_nan | 0.005 |  | partial_nan | 0.010 |  |  |  |
| `gd_delay_band0p25-0p5cyc` | .statistics.spectral | partial_nan | 0.005 |  | partial_nan | 0.011 |  |  |  |
| `sgc_nonparametric_mean_band0-0p5cyc` | .statistics.spectral | ok | 0.103 |  | ok | 0.326 |  |  |  |
| `sgc_nonparametric_mean_band0-0p25cyc` | .statistics.spectral | ok | 0.102 |  | ok | 0.350 |  |  |  |
| `sgc_nonparametric_mean_band0p25-0p5cyc` | .statistics.spectral | ok | 0.101 |  | ok | 0.342 |  |  |  |
| `sgc_nonparametric_max_band0-0p5cyc` | .statistics.spectral | ok | 0.104 |  | ok | 0.347 |  |  |  |
| `sgc_nonparametric_max_band0-0p25cyc` | .statistics.spectral | ok | 0.103 |  | ok | 0.337 |  |  |  |
| `sgc_nonparametric_max_band0p25-0p5cyc` | .statistics.spectral | ok | 0.102 |  | ok | 0.344 |  |  |  |
| `sgc_parametric_autoorder_mean_band0-0p5cyc_maxorder50` | .statistics.spectral | ok | 0.006 |  | ok | 0.014 |  |  |  |
| `sgc_parametric_autoorder_mean_band0-0p25cyc_maxorder50` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_autoorder_mean_band0p25-0p5cyc_maxorder50` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_mean_band0-0p25cyc_order01` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_mean_band0p25-0p5cyc_order01` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_mean_band0-0p25cyc_order20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_mean_band0p25-0p5cyc_order20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_autoorder_max_band0-0p25cyc_maxorder50` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_autoorder_max_band0p25-0p5cyc_maxorder50` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_max_band0-0p25cyc_order01` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_max_band0p25-0p5cyc_order01` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_max_band0-0p25cyc_order20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_max_band0p25-0p5cyc_order20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `psi_wavelet_mean_band0-0p5cyc` | .statistics.wavelet | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_mean_band0-0p25cyc` | .statistics.wavelet | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_mean_band0p25-0p5cyc` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_max_band0-0p5cyc` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_max_band0-0p25cyc` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_max_band0p25-0p5cyc` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `lmfit_ridge_penaltysum_alpha1` | .statistics.misc | ok | 0.016 |  | ok | 0.035 |  |  |  |
| `lmfit_lasso_alpha1` | .statistics.misc | ok | 0.018 |  | ok | 0.037 |  |  |  |
| `lmfit_SGDRegressor` | .statistics.misc | ok | 0.017 |  | ok | 0.038 |  |  |  |
| `lmfit_elasticnet_alpha1_lassoratio0p5` | .statistics.misc | ok | 0.018 |  | ok | 0.038 |  |  |  |
| `lmfit_bayesianridge` | .statistics.misc | ok | 0.019 |  | ok | 0.039 |  |  |  |
| `gpfit_dotproduct` | .statistics.misc | ok | 0.204 |  | ok | 0.610 |  |  |  |
| `gpfit_rbf` | .statistics.misc | ok | 0.320 |  | ok | 0.977 |  |  |  |
| `coint_johansen_constant_maxeig_lagdiffs10` | .statistics.misc | ok | 0.008 |  | ok | 0.019 |  |  |  |
| `coint_johansen_constant_trace_lagdiffs10` | .statistics.misc | ok | 0.008 |  | ok | 0.020 |  |  |  |
| `coint_johansen_constant_maxeig_lagdiffs01` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_johansen_constant_trace_lagdiffs01` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_johansen_lineartrend_maxeig_lagdiffs10` | .statistics.misc | ok | 0.009 |  | ok | 0.020 |  |  |  |
| `coint_johansen_lineartrend_trace_lagdiffs10` | .statistics.misc | ok | 0.009 |  | ok | 0.020 |  |  |  |
| `coint_johansen_lineartrend_maxeig_lagdiffs01` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_johansen_lineartrend_trace_lagdiffs01` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_aeg_constant_autolagaic_tstat_maxlag10` | .statistics.misc | ok | 0.025 |  | ok | 0.056 |  |  |  |
| `coint_aeg_lineartrend_autolagaic_tstat_maxlag10` | .statistics.misc | ok | 0.025 |  | ok | 0.054 |  |  |  |
| `coint_aeg_lineartrend_autolagbic_tstat_maxlag10` | .statistics.misc | ok | 0.025 |  | ok | 0.053 |  |  |  |
| `pec` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_log` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth_log` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth_abs` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth_log_abs` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `ids_pnormmax_terms06_bandwidth0p5` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
# Additional pairwise candidates

`tests/unit/test_pairwise_candidates.py` independently checks histogram KL and
both distance-centering conventions, invalid inputs, extreme scales, mutation,
channel order and owned outputs. It also checks IDS parameter identities and
that importing the implementation does not reseed NumPy. Existing
`test_ids_reference.py` checks the transformation/correlation definition.

`tests/unit/test_banded_dtw.py` compares exact banded DTW against a separate
full-grid dynamic program and tslearn, including boundaries, ties/constants,
radius zero and wider bands, permutations and extreme units. It does not expand
the optimization scope to unconstrained or soft DTW.

`tests/benchmark_pairwise_candidates.py` measures formula costs from an explicit
prepared EEG window. `tests/pairwise_candidate_math_results.json` records the
first patient-001 comparison; it is not a multi-patient or workflow benchmark.
Progress and remaining gates are in `docs/additional_pairwise_review.md`.
## Additional pairwise band candidates

`tests/unit/test_band_candidates.py` checks the prototype PAC formula and a
known phase/amplitude model, source-row orientation, band PEC preprocessing,
owned/copyable inputs, invalidation and numerical degeneracy. Workflow checks
in `tests/workflow/execution/test_band_candidates.py` cover continuous filtering,
shared file-backed views, reference masks, serial/spawn equivalence and calibration.
Run only affected cases after subsequent changes; these are separate from the
existing broadband PEC goldens.

`tests/benchmark_band_candidates.py` measures selected real EEG windows using
an existing workflow's source/selection/preprocessing. It calculates only the
two broadband PEC comparisons and three band candidates, independent of that
workflow's metric selection. Supply `--scratch` on a disk filesystem. The first
receipt is `tests/band_candidate_001_results.json`; it is not a cohort or
parallel/output throughput benchmark. The separate
`pyspi/workflow/additional_candidate_pilot.yaml` selects eleven candidates for
bounded follow-up evaluation, not the established production run.
