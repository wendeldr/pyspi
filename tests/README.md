# pyspi test suite

The suite is a **change detector**. The golden values under `tests/regression/goldens/` record what the code produced when they were generated; a passing run means outputs have not changed, not that they are mathematically correct. Known bugs are locked in as they are (see [Known bugs](#known-bugs-locked-not-fixed)).

## Layout

| path | what it covers |
|---|---|
| `regression/test_spi_goldens.py` | one golden comparison per SPI, preprocessing variants, `Calculator` end-to-end passes (fast subset by default, full config with `--runslow`), catalogue checks |
| `regression/test_extra_goldens.py`, `regression/extra_config.yaml` | snapshot goldens for nine wavelet classes that `config.yaml` leaves out (see [Extra goldens](#extra-goldens)) |
| `regression/harness.py`, `regression/spi_policy.py` | shared build/seed/compute/compare code and the per-SPI policy (tolerances, slow, stochastic) |
| `regression/generate_goldens.py` | writes and checks the goldens |
| `unit/` | `Data`, base-class dispatch, `Calculator` config loading and failure handling, frames, utils, `phi_native` golden, known bugs |
| `test_calc.py`, `test_utils.py`, `test_phi_native.py` | original tests, unchanged |
| `test_SPIs.py` | original CML7 z-score benchmark (`legacy`, `slow`) |
| `storage/` | storage-format suite, run separately (below) |

## Environment

Python 3.11 in a uv environment at the repository root. Install from `requirements.txt` so the upstream pins apply (`pyEDM==1.15.2.0`, `setuptools<81`, `numpy<2`, `mne==0.23.0`, `cdt==0.5.23`).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel; cdt needs torch
uv pip install --python .venv -r requirements.txt -r tests/requirements-test.txt -e .
uv pip freeze --python .venv > tests/requirements-test-lock.txt
```

`spectral-connectivity` must stay below 2 (2.x breaks pyspi); the lock already has 1.1.2. To reproduce the golden environment exactly, install from the lock instead: `uv pip install --python .venv -f https://download.pytorch.org/whl/cpu/torch/ -r tests/requirements-test-lock.txt -e .` (the lock pins the CPU-only torch wheel, `torch==2.14.1+cpu`, which only the PyTorch index serves; with pip use `--extra-index-url https://download.pytorch.org/whl/cpu` instead, as the CI lock jobs do; pyspi reaches torch only through `cdt`, and the SPIs it uses do not run on it). A JDK is needed for the 39 JIDT (`java`) SPIs. Without a JVM, `pyspi.statistics.infotheory` still imports (its Java packages resolve lazily), but its classes cannot be instantiated, so the test harness leaves that module out. Every `java`-marked test then skips: the 39 JIDT per-SPI tests, the 8 `mi_gaussian`/`te_kraskov` preprocessing variants, `test_catalogue_matches_golden_meta`, both `Calculator` end-to-end tests over `config.yaml`, and the Java-dependent unit tests (`test_subset_counts`, the two tests in `unit/test_frames.py`, and the `te_symbolic` and `CorrelationFrame` known-bug pairs). `test_spi_catalogue` checks for the JVM itself and skips too. Everything else still runs, including `test_extra_goldens.py`, which needs no Java.

`meta.json` in each golden set records the Python, Java and package versions and the `OMP_NUM_THREADS` setting it was generated with.

## Running

Always run from the repository root.

| command | what runs |
|---|---|
| `.venv/bin/python -m pytest` | default tier: small goldens (5 processes x 64 observations), no `slow` tests, no legacy benchmark; 553 passed, 15 skipped, 10 xfailed in 89-98 s serial on the generating machine |
| `.venv/bin/python -m pytest --runslow` | adds the slow SPI tests, the full-config `Calculator` end-to-end test, the slow known-bug tests (ccm) and the legacy benchmark |
| `.venv/bin/python -m pytest --pyspi-size=full --runslow --timeout=1800` | full tier: 7 x 100 goldens, everything |
| `.venv/bin/python -m pytest --runslow -m slow --ignore=tests/test_SPIs.py` | only the slow tests (14 at small size), without the legacy benchmark's recompute (see below) |
| `.venv/bin/python -m pytest tests/regression -k "cov_ or dcorr"` | select SPIs by identifier |
| `.venv/bin/python -m pytest -m "not java"` | deselects every `java`-marked test: 57 of the 578 collected at small size (39 JIDT per-SPI tests, 8 preprocessing variants, the catalogue test, 2 `Calculator` end-to-end tests, 7 unit tests) |

Options:

- `--pyspi-size small|full` picks the golden set (default `small`; environment fallback `PYSPI_TEST_SIZE`). A missing golden set skips its tests with a message naming the generator command.
- `--runslow` runs tests marked `slow`; otherwise they are reported as skipped (`slow: needs --runslow`).
- Any `--runslow` invocation makes `tests/test_SPIs.py` recompute all 285 SPIs on the CML7 benchmark while it is being collected (about 2.5 min on the generating machine, measured with `--collect-only`), whatever `-m` or `-k` selects; with `-n N` every worker repeats it. Add `--ignore=tests/test_SPIs.py` (or give explicit test paths) unless you want the legacy benchmark.
- `-n 4` runs in parallel with pytest-xdist. Every test is independent; each worker starts its own JVM.
- `--timeout=N` (pytest-timeout) overrides the default 300 s per test. It uses `signal` and cannot interrupt code inside Java or numba frames, so it is a safety net only.

`addopts` already passes `--ignore=tests/storage`, `--strict-markers`, `--strict-config` and `-ra`; `xfail_strict = true`. That `--ignore` path only resolves from the repository root, so `tests/conftest.py` also sets `collect_ignore = ["storage"]`; a run started from inside `tests/` skips the storage suite too.

### Markers

| marker | meaning |
|---|---|
| `slow` | measured above `SLOW_THRESHOLD_S` (2.0 s) at the selected size (`time_s` in `meta.json`); `FORCE_SLOW` applies only when no time is recorded. Skipped unless `--runslow` |
| `java` | needs the JIDT JVM (from the SPI's `deps`) |
| `regression` | golden-value comparison |
| `stochastic` | draws from the global numpy RNG; reproducible only with `seed_all()` immediately before compute |
| `nan_golden` | golden is all-NaN (status `all_nan` or `error`) for this test; a per-SPI test gets value coverage only at `--pyspi-size=full`, a preprocessing variant none |
| `legacy` | original CML7 benchmark in `test_SPIs.py` |

`slow`, `java`, `stochastic` and `nan_golden` are attached by `tests/conftest.py`:

- per-SPI tests (`identifier`): `slow`, `nan_golden` and `java` from `goldens/<size>/meta.json` `spis[identifier]` (`time_s`, `status`, `deps`);
- preprocessing variants (`preproc_key`): `slow` and `nan_golden` from `goldens/preproc/meta.json` `variants[key]` (that variant's own `time_s` and `status`), `java` from `spis[identifier].deps` in the same file. So `test_preprocessing_variant_matches_golden[xcorr_max_sig-True-n0d0]` is `nan_golden` (its variant status is `error`);
- `stochastic` from `spi_policy.STOCHASTIC` in both cases.

`test_extra_goldens.py` attaches the same markers itself from `goldens/extra/small/meta.json` (none applies today). The `Calculator` and catalogue tests carry `java` (and the full end-to-end test `slow`) explicitly.

`FORCE_SLOW` (used only for an identifier or preprocessing variant without a recorded `time_s`; every current golden records one): `ccm_*`, `*kraskov*DCE*`, `hhg`, `mgc*`, `*sgddtw*`, `softdtw*`, `anm`, `gpfit_*`, `*MinCovDet*`, `*EllipticEnvelope*`, `*GraphicalLassoCV*`, `sgc_parametric*`, `cce_*`, `di_*`.

## How the SPI regression works

- `test_spi_matches_golden[<identifier>]`: builds the SPI from its `meta.json` entry (module, class, params; `params: null` records that `Calculator._load_yaml` default-constructed the class, otherwise the recorded params are passed as-is with no fallback, so a renamed or removed constructor argument fails loudly), makes a fresh `Data` from the recorded dataset spec, calls `seed_all(42)`, runs `multivariate`, sets the diagonal to NaN like `Calculator.compute`, and compares with the golden. The NaN mask must match exactly; then `assert_allclose` with the policy tolerance. An SPI whose golden status is `error` is compared as all-NaN: an exception is mapped to all-NaN exactly as `Calculator.compute` does, so either raising or returning all-NaN keeps the test green, while any real values fail the NaN-mask check.
- `test_preprocessing_variant_matches_golden[<identifier>-n?d?]` (golden key `<identifier>::n?d?`): 24 representative SPIs (`spi_policy.PREPROC_SPIS`: three to five per statistics module, except wavelet, which has one because its six SPIs are all `psi_wavelet` band variants; `anm` is included because its output changes with `normalise`, while `igci` and `reci` differ between `n1d0` and `n0d0` by at most 3e-15) under all four `(normalise, detrend)` combinations, 96 keys in all, on the small window of cml7 with a fixed per-channel affine distortion (`x * scale + offset + slope * t`, recorded in `goldens/preproc/meta.json`) so that each flag measurably changes the input. Suffix `n1d0` is the Calculator default; `n1d1` is the EEG workflow (detrend, then z-score). 95 variants have status `ok`; `xcorr_max_sig-True::n0d0` raises `IndexError` today and is locked as an error (marker `nan_golden`). No variant is `slow`; the slowest, `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1::n1d0`, measured 1.94 s, just under the 2.0 s threshold. The 24 resolved identifiers (from `goldens/preproc/meta.json` `spis`): `anm`, `bary_euclidean_mean`, `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5`, `coint_johansen_max_eig_stat_order-1_ardiff-1`, `cov_EmpiricalCovariance`, `dcorr`, `dtw`, `gwtau`, `ids`, `igci`, `kendalltau`, `lmfit_Ridge`, `mi_gaussian`, `pdist_euclidean`, `pec`, `phi_star_t-1_norm-0`, `plv_multitaper_mean_fs-1_fmin-0_fmax-0-5`, `prec_EmpiricalCovariance`, `psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean`, `reci`, `sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-5`, `spearmanr`, `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1`, `xcorr_max_sig-True`.
- Two fixture checks: `spearmanr`, `kendalltau` and `dcorr` give the same golden for `n1d0` and `n0d0` (rank-based or per-variable scale-invariant; measured max difference 0, 0 and 1.9e-15), and `cov_EmpiricalCovariance` differs between them.
- `test_calculator_table_matches_goldens_fast` (default tier, `java`): writes a temporary config holding every identifier of the package's `config.yaml` that has a golden and is not `slow` at the selected size (module and class order kept, so the shared caches fill in the same order as a full pass), checks that the `Calculator` loads exactly those identifiers, runs one seeded `Calculator` pass with shared caches and compares each table block with its per-SPI golden. At small size that is 274 SPIs (285 minus the 11 slow ones) measured here at 32 s when run alone (plus 4 s setup for the JVM and imports) and 17 s inside a full default run, where earlier tests have already paid the first-call costs. It is the default-tier test that notices a cache-keying change (two SPIs reading each other's cached spectra), which the per-SPI tests cannot see because each runs on a fresh `Data`.
- `test_calculator_table_matches_goldens` (`slow`, `java`): the same over the full `config.yaml`.
- Both end-to-end tests skip `spi_policy.CALC_E2E_SKIP` (`cov*MinCovDet`, `prec*MinCovDet`, `bary*_sgddtw_*`; 8 identifiers): a single seeded pass draws from the global RNG in a different order than per-SPI seeding, and these then differ from their goldens. The four `*EllipticEnvelope` SPIs are `stochastic` too but reproduce their per-SPI goldens in that pass, so they are compared. All mismatches are reported together.
- `test_catalogue_matches_golden_meta` (`java`): loads the imported package's `config.yaml` through both `harness.catalogue` and `Calculator`, and asserts that the identifier sets equal the recorded ones, that the harness order and classes match what the `Calculator` loaded, and that for every identifier `module`, `cls`, `params` and `deps` equal `meta.json` (after a JSON round trip) and `labels` match as sorted sets. The per-SPI tests rebuild each SPI from the recorded params, so this is what reports a `config.yaml` parameter edit that keeps the identifier (for example `terms` of `ids`).
- `test_spi_catalogue[all|fast|sonnet|fabfour]`: subset sizes 285/217/14/4; for `all`, identifier order equals `calculator_order` and each SPI's labels equal the recorded ones.

### Tolerance policy (`regression/spi_policy.py`)

First matching `fnmatch` pattern wins.

| pattern | rtol | atol | reason |
|---|---|---|---|
| `cov*EllipticEnvelope`, `prec*EllipticEnvelope`, `cov*MinCovDet`, `prec*MinCovDet` | 1e-6 | 1e-8 | sklearn robust covariance draws from the global numpy RNG; reproducible only via `seed_all()` (marker `stochastic`; regenerate on an sklearn upgrade) |
| `bary*_sgddtw_*` | 1e-6 | 1e-8 | tslearn subgradient DBA shuffles with the global numpy RNG (marker `stochastic`; regenerate on a tslearn upgrade) |
| everything else | 1e-6 | 1e-8 | default; absorbs BLAS drift between machines; `atol` covers pure rounding noise (e.g. `bary_euclidean_mean` ~1e-16) |

The stochastic patterns match 12 identifiers (the `-sq` variants included); their rows keep the default values and are listed to document that they match only because `seed_all()` runs right before them.

The iterative optimisers (`*GraphicalLassoCV*`, `phi_*`, `gpfit_*`, `anm`) have no row: seeded run to run and against the stored goldens they were measured bit-for-bit reproducible on the generating machine (the 11 matching identifiers, each recomputed twice at both sizes: maximum absolute difference from the stored goldens 0), so the default applies. If another machine shows drift for one of them, re-add a row for that pattern with the measured drift written in its reason, not a guessed tolerance. Overrides are documented exceptions; do not loosen a tolerance to make a failure go away.

## Extra goldens

`regression/extra_config.yaml` lists nine wavelet classes from `pyspi/statistics/wavelet.py` that run today but are absent from `pyspi/config.yaml`: `cohmag` (`CoherenceMagnitude`), `phase` (`CoherencePhase`), `icoh` (`ImaginaryCoherence`), `plv` (`PhaseLockingValue`), `ppc` (`PairwisePhaseConsistency`), `pli` (`PhaseLagIndex`), `dspli` (`DebiasedSquaredWeightedPhaseLagIndex`), `wspli` (`weighted_PhaseLagIndex`) and `dwspli` (`debiased_weighted_squared_PhaseLagIndex`). Each has the six band/statistic configs that `PhaseSlopeIndex` uses in `config.yaml`, giving 54 identifiers named `<id>_wavelet_{mean|max}_fs-1_fmin-{0|0-25}_fmax-{0-25|0-5}` (the band pairs are `0`-`0-5`, `0`-`0-25` and `0-25`-`0-5`), for example `cohmag_wavelet_mean_fs-1_fmin-0_fmax-0-5`. They all go through the per-`Data` `data.mne[measure]` cache, which no SPI in the default config exercises.

The goldens are in `regression/goldens/extra/small/{values.npz,meta.json}`, generated on the small window (5 processes x 64 observations, `normalise=True`, `detrend=False`, seed 42) with mne 0.23.0. All 54 have status `ok`, and none is `slow`, `java` or `stochastic`. These are snapshots of current output, and many are degenerate on the single-epoch input: off the diagonal, `cohmag` and `plv` are 1, `ppc` and `dspli` are `-inf` (status `ok` counts only NaN), `dwspli` is 0 and `icoh` is rounding noise (below 2e-14, inside the 1e-8 `atol`); `plv`, `ppc`, `dspli` and `dwspli` give the same matrix for all six bands. Only `phase`, `pli` and `wspli` carry non-trivial values. They are locked as they are.

`regression/test_extra_goldens.py` has 56 tests: one check that the set's identifiers and order match `extra_config.yaml` and its dataset is the small window, 54 per-SPI tests (`test_extra_spi_matches_golden[<identifier>]`) and one `Calculator` end-to-end test over `extra_config.yaml`, which also checks that the shared `data.mne` cache holds exactly the 9 measures. They run in the default tier in a few seconds, need no Java, use their own data from the set's `meta.json` (so `--pyspi-size` does not affect them) and skip cleanly if `goldens/extra/small` is absent.

## Regenerating the goldens

**Regenerate only for an intentional change to an output, and say why in the commit message.** A failing golden test after an optimisation is the suite doing its job: first find out why the value moved.

```bash
# check the current code against the goldens; writes nothing, exit 1 on drift
.venv/bin/python tests/regression/generate_goldens.py --size all --preproc --phi --check

# the extra goldens have their own check (same command as their regeneration below, plus --check)

# regenerate (add --repeat 2 to audit determinism: min time kept, disagreeing repeats reported)
.venv/bin/python tests/regression/generate_goldens.py --size small --repeat 2
.venv/bin/python tests/regression/generate_goldens.py --size full --repeat 2
.venv/bin/python tests/regression/generate_goldens.py --preproc --repeat 2
.venv/bin/python tests/regression/generate_goldens.py --phi
.venv/bin/python tests/regression/generate_goldens.py --size small --repeat 2 \
    --configfile tests/regression/extra_config.yaml --out tests/regression/goldens/extra   # extra goldens

# regenerate only some SPIs (merged into the existing set)
.venv/bin/python tests/regression/generate_goldens.py --size all --spis 'cov_*'
```

`--size` takes `small`, `full` or `all` (both). `--spis GLOB` is an fnmatch filter on identifiers; when writing, the result is merged into the existing set and stored SPIs that are no longer in the catalogue are dropped (and printed). `--out DIR` writes elsewhere (default `tests/regression/goldens`). `--configfile PATH` uses another SPI config (default `pyspi/config.yaml`). `--check` compares values with the `spi_policy` tolerances and also compares each identifier's catalogue fields (`module`, `cls`, `params`, `deps`, and `labels` as sets) with the stored meta, printing a `CATALOGUE CHANGE` line for each mismatch. After regenerating, run `--check` once more on the same environment; it must pass. Regenerate after upgrading sklearn or tslearn (stochastic SPIs) and refresh `tests/requirements-test-lock.txt` when the environment changes.

A parameter change in `config.yaml` that keeps the identifier fails `test_catalogue_matches_golden_meta` (and `--check`) until the goldens are regenerated. That is expected for an intentional change; regenerate and say why in the commit message.

## Red in CI, green locally

The push CI matrix installs from `requirements.txt`, which is mostly unpinned, so a library upgrade (scikit-learn, tslearn, JIDT, scipy) can move an output past its tolerance without any change to pyspi. The stochastic SPIs (see the tolerance table) and the iterative optimisers (`*GraphicalLassoCV*`, `phi_*`, `gpfit_*`, `anm`, which use the default tolerance) are the usual suspects. The `test-pinned-env` CI job runs the same default tier on the lock file the goldens were generated in; if it is green while the matrix is red, compare the failing job's package versions with `tests/requirements-test-lock.txt` and the versions recorded in the golden `meta.json` before looking for a code change. Locally, `generate_goldens.py --check` prints the per-SPI differences.

## Known bugs (locked, not fixed)

Each is locked in `unit/test_known_bugs.py`: bugs 1-9 have a test of the current behaviour paired with a strict xfail of the fixed behaviour, so a fix shows up as XPASS(strict) and must be acknowledged.

1. `TransferEntropy(estimator='symbolic')` falls through to the Gaussian calculator (`te_symbolic_* == gc_gaussian_*`).
2. `ConvergentCrossMapping(embedding_dimension=None)` picks max E (10), not best-rho E (`ccm_E-None == ccm_E-10`). Both tests are `slow`.
3. `SoftDynamicTimeWarping` ignores `global_constraint` (three identical configs).
4. `SpearmanR`/`KendallTau` mutate the class-level `labels` list (instances carry both `signed` and `unsigned`).
5. "Undirected" spectral SPIs (`phase`, `psi`, `pli`, `wpli`, ...) are not symmetrised (MRO bypasses `Undirected.multivariate`).
6. `Calculator` duplicate-identifier check is dead code (dict keys are unique; collisions overwrite silently).
7. `Data(procnames=...)` is validated but never stored.
8. `Data.convert_to_numpy` for `.csv` passes `','` as dtype.
9. `CorrelationFrame(calc_frame)` calls the nonexistent `getstatlabels()`.
10. `utils.convert_mdf_to_ddf` raises under pandas>=3 (`stack(dropna=...)` is rejected); found while writing the unit tests, locked without a paired xfail.

Bugs 1 and 9 need the JVM (`java`). **Some SPIs are never exercised by a default run.** The bug-2 tests are `slow`, all nine `ccm_*` goldens and the two auto-embedding `te_kraskov_*_k-max-10_tau-max-4` goldens are `slow` at both sizes, and the six `ddtf_multitaper_*` and `mgcx_maxlag-10` goldens are `slow` at full size, so none of these runs without `--runslow`. After touching ccm, TE kraskov auto-embedding, ddtf or mgcx, run `.venv/bin/python -m pytest --runslow -k "ccm or te_kraskov" --ignore=tests/test_SPIs.py` (20 tests at small size) and the full tier before trusting a green default run.

### Flagged for investigation (locked as-is, not confirmed bugs)

- **Degenerate outputs in the extra wavelet set** (`goldens/extra/small`, the nine `*_wavelet_*` classes absent from `config.yaml`). Off the diagonal, `cohmag` and `plv` are exactly 1.0 everywhere, `ppc` and `dspli` are `-inf` everywhere, `dwspli` is 0 everywhere, `icoh` is rounding noise (about 1e-16), and those six are identical for all six band/statistic configs. Only `phase`, `pli` and `wspli` vary with the input and the band. The likely cause is that `mne._get_cache` hands `mne.connectivity.spectral_connectivity` a single epoch (the array is `1 x processes x samples`), and these measures are defined as averages of phase consistency across epochs: with one epoch a consistency measure is trivially 1 and pairwise phase consistency divides by `n(n-1) = 0`. The six `psi_wavelet_*` SPIs in `config.yaml` use a separate cache (`data.mne_psi`) and are not degenerate. These nine classes need checking (epoching, or a different estimator) before they are used for anything; the goldens lock the degenerate values so a change shows up as a failure, not as silent agreement.

### Dependency constraints that affect calculations

- **`spectral-connectivity` must stay below 2** (pinned in `requirements.txt` and `setup.py`). Version 2 requires a 3-D `(time, trials, signals)` input; pyspi passes 2-D, every multitaper call raises, and `Calculator.compute` turns the exception into an all-NaN block for all 114 spectral SPIs without failing. An unpinned install therefore loses all spectral output silently.
- **`torch` is installed but no calculation runs on it.** `pyspi/statistics/causal.py` imports ANM, CDS, IGCI and RECI from `cdt.causality.pairwise`; those four are numpy, scikit-learn and scipy code. `cdt` imports torch at package import time for its neural-network algorithms (SAM, CGNN, NCC) and does not declare it as a dependency, so pyspi lists torch itself; blocking the torch import makes `import cdt` fail. The CPU wheel is sufficient (see Environment). Dropping torch would mean vendoring the four pairwise functions out of cdt or stubbing the import before `cdt` loads; both are package changes and have not been made. The `anm`, `cds`, `igci` and `reci` goldens would show any numeric difference from such a change.

Also worth knowing (not bugs): `utils.normalise` default `nan_policy='propogate'` always raises; the spectral cache's first insert is keyed by `self.measure`, causing one redundant recompute; `dtf_* == pdcoh_*`, and auto-embedding TE kraskov equals fixed k=1 on cml7.

## Storage suite

`tests/storage/` is a separate suite with its own locked environment (Python 3.12, NumPy 2) and `pytest.ini`. The root run ignores it. Run it as described in [storage/README.md](storage/README.md).

## SPI coverage

All 285 identifiers in `config.yaml`, in `Calculator` order, from `goldens/small/meta.json` (5 x 64) and `goldens/full/meta.json` (7 x 100). `time_s` is the generator's fastest measured time for one `multivariate` call on the generating machine (a 64-core machine with `OMP_NUM_THREADS=1`; versions and the thread setting are in `meta.json`); `slow` is the marker at that size.

| size | ok | partial_nan | all_nan / error | slow (skipped by default) | per-SPI tests run by the default tier |
|---|---:|---:|---:|---:|---:|
| small (5 x 64) | 276 | 9 | 0 | 11 | 274 |
| full (7 x 100) | 282 | 3 | 0 | 18 | 267 |

The default tier's fast `Calculator` end-to-end test computes the same non-slow SPIs (274 at small size) in one pass and compares all of them except the 8 in `CALC_E2E_SKIP`. No per-SPI test carries `nan_golden` at either size; the only `nan_golden` test is the preprocessing variant `xcorr_max_sig-True-n0d0`. The `partial_nan` SPIs are the three `gd_multitaper_delay_*` at both sizes plus, at small size only, the six `sgc_parametric_*_order-20`.

<!-- Generated from goldens/{small,full}/meta.json with a short inline Python snippet (slow = spi_policy.is_slow(identifier, time_s), stochastic = spi_policy.is_stochastic, java = "java" in deps); regenerate rather than hand-edit. -->
| identifier | module | status (small) | time_s (small) | slow (small) | status (full) | time_s (full) | slow (full) | stochastic | java |
|---|---|---|---:|:-:|---|---:|:-:|:-:|:-:|
| `cov_EmpiricalCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_EllipticEnvelope` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov_GraphicalLasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `cov_GraphicalLassoCV` | .statistics.basic | ok | 0.227 |  | ok | 0.328 |  |  |  |
| `cov_LedoitWolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_MinCovDet` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov_OAS` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov_ShrunkCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov-sq_EmpiricalCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov-sq_EllipticEnvelope` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov-sq_GraphicalLasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `cov-sq_GraphicalLassoCV` | .statistics.basic | ok | 0.233 |  | ok | 0.332 |  |  |  |
| `cov-sq_LedoitWolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov-sq_MinCovDet` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `cov-sq_OAS` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `cov-sq_ShrunkCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_EmpiricalCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_EllipticEnvelope` | .statistics.basic | ok | 0.022 |  | ok | 0.023 |  | yes |  |
| `prec_GraphicalLasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `prec_GraphicalLassoCV` | .statistics.basic | ok | 0.226 |  | ok | 0.333 |  |  |  |
| `prec_LedoitWolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_MinCovDet` | .statistics.basic | ok | 0.021 |  | ok | 0.023 |  | yes |  |
| `prec_OAS` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec_ShrunkCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec-sq_EmpiricalCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec-sq_EllipticEnvelope` | .statistics.basic | ok | 0.021 |  | ok | 0.023 |  | yes |  |
| `prec-sq_GraphicalLasso` | .statistics.basic | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `prec-sq_GraphicalLassoCV` | .statistics.basic | ok | 0.225 |  | ok | 0.331 |  |  |  |
| `prec-sq_LedoitWolf` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec-sq_MinCovDet` | .statistics.basic | ok | 0.021 |  | ok | 0.023 |  | yes |  |
| `prec-sq_OAS` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `prec-sq_ShrunkCovariance` | .statistics.basic | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `spearmanr-sq` | .statistics.basic | ok | 0.005 |  | ok | 0.011 |  |  |  |
| `spearmanr` | .statistics.basic | ok | 0.005 |  | ok | 0.011 |  |  |  |
| `kendalltau-sq` | .statistics.basic | ok | 0.005 |  | ok | 0.010 |  |  |  |
| `kendalltau` | .statistics.basic | ok | 0.005 |  | ok | 0.010 |  |  |  |
| `xcorr_max_sig-True` | .statistics.basic | ok | 0.002 |  | ok | 0.004 |  |  |  |
| `xcorr-sq_max_sig-True` | .statistics.basic | ok | 0.002 |  | ok | 0.004 |  |  |  |
| `xcorr_mean_sig-True` | .statistics.basic | ok | 0.002 |  | ok | 0.004 |  |  |  |
| `xcorr-sq_mean_sig-True` | .statistics.basic | ok | 0.002 |  | ok | 0.004 |  |  |  |
| `xcorr_mean_sig-False` | .statistics.basic | ok | 0.002 |  | ok | 0.003 |  |  |  |
| `xcorr-sq_mean_sig-False` | .statistics.basic | ok | 0.002 |  | ok | 0.003 |  |  |  |
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
| `mgcx_maxlag-1` | .statistics.distance | ok | 0.239 |  | ok | 0.768 |  |  |  |
| `mgcx_maxlag-10` | .statistics.distance | ok | 0.687 |  | ok | 2.278 | yes |  |  |
| `dcorrx_maxlag-1` | .statistics.distance | ok | 0.025 |  | ok | 0.066 |  |  |  |
| `dcorrx_maxlag-10` | .statistics.distance | ok | 0.103 |  | ok | 0.285 |  |  |  |
| `dtw` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `dtw_constraint-itakura` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `dtw_constraint-sakoe-chiba` | .statistics.distance | ok | 0.005 |  | ok | 0.015 |  |  |  |
| `softdtw` | .statistics.distance | ok | 0.013 |  | ok | 0.042 |  |  |  |
| `softdtw_constraint-itakura` | .statistics.distance | ok | 0.013 |  | ok | 0.042 |  |  |  |
| `softdtw_constraint-sakoe-chiba` | .statistics.distance | ok | 0.013 |  | ok | 0.041 |  |  |  |
| `lcss` | .statistics.distance | ok | 0.005 |  | ok | 0.017 |  |  |  |
| `lcss_constraint-itakura` | .statistics.distance | ok | 0.005 |  | ok | 0.017 |  |  |  |
| `lcss_constraint-sakoe-chiba` | .statistics.distance | ok | 0.005 |  | ok | 0.017 |  |  |  |
| `bary_euclidean_mean` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `bary_euclidean_max` | .statistics.distance | ok | 0.005 |  | ok | 0.015 |  |  |  |
| `bary_dtw_mean` | .statistics.distance | ok | 0.006 |  | ok | 0.023 |  |  |  |
| `bary_dtw_max` | .statistics.distance | ok | 0.005 |  | ok | 0.022 |  |  |  |
| `bary_sgddtw_mean` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary_sgddtw_max` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary_softdtw_mean` | .statistics.distance | ok | 0.319 |  | ok | 1.124 |  |  |  |
| `bary_softdtw_max` | .statistics.distance | ok | 0.324 |  | ok | 1.143 |  |  |  |
| `bary-sq_euclidean_mean` | .statistics.distance | ok | 0.005 |  | ok | 0.016 |  |  |  |
| `bary-sq_euclidean_max` | .statistics.distance | ok | 0.005 |  | ok | 0.015 |  |  |  |
| `bary-sq_dtw_mean` | .statistics.distance | ok | 0.005 |  | ok | 0.024 |  |  |  |
| `bary-sq_dtw_max` | .statistics.distance | ok | 0.005 |  | ok | 0.023 |  |  |  |
| `bary-sq_sgddtw_mean` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary-sq_sgddtw_max` | .statistics.distance | ok | 0.009 |  | ok | 0.035 |  | yes |  |
| `bary-sq_softdtw_mean` | .statistics.distance | ok | 0.322 |  | ok | 1.115 |  |  |  |
| `bary-sq_softdtw_max` | .statistics.distance | ok | 0.320 |  | ok | 1.114 |  |  |  |
| `gwtau` | .statistics.distance | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `anm` | .statistics.causal | ok | 0.246 |  | ok | 0.827 |  |  |  |
| `cds` | .statistics.causal | ok | 0.061 |  | ok | 0.213 |  |  |  |
| `reci` | .statistics.causal | ok | 0.036 |  | ok | 0.072 |  |  |  |
| `igci` | .statistics.causal | ok | 0.018 |  | ok | 0.043 |  |  |  |
| `ccm_E-None_mean` | .statistics.causal | ok | 3.373 | yes | ok | 14.601 | yes |  |  |
| `ccm_E-None_max` | .statistics.causal | ok | 3.348 | yes | ok | 14.736 | yes |  |  |
| `ccm_E-None_diff` | .statistics.causal | ok | 3.365 | yes | ok | 15.000 | yes |  |  |
| `ccm_E-1_mean` | .statistics.causal | ok | 2.899 | yes | ok | 12.590 | yes |  |  |
| `ccm_E-1_max` | .statistics.causal | ok | 2.912 | yes | ok | 12.658 | yes |  |  |
| `ccm_E-1_diff` | .statistics.causal | ok | 2.903 | yes | ok | 12.313 | yes |  |  |
| `ccm_E-10_mean` | .statistics.causal | ok | 3.322 | yes | ok | 14.419 | yes |  |  |
| `ccm_E-10_max` | .statistics.causal | ok | 3.338 | yes | ok | 14.442 | yes |  |  |
| `ccm_E-10_diff` | .statistics.causal | ok | 3.323 | yes | ok | 14.558 | yes |  |  |
| `je_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.005 |  |  | yes |
| `je_kozachenko` | .statistics.infotheory | ok | 0.002 |  | ok | 0.014 |  |  | yes |
| `je_kernel_W-0.5` | .statistics.infotheory | ok | 0.002 |  | ok | 0.009 |  |  | yes |
| `ce_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.004 |  |  | yes |
| `ce_kozachenko` | .statistics.infotheory | ok | 0.002 |  | ok | 0.008 |  |  | yes |
| `ce_kernel_W-0.5` | .statistics.infotheory | ok | 0.002 |  | ok | 0.005 |  |  | yes |
| `cce_gaussian` | .statistics.infotheory | ok | 0.037 |  | ok | 0.113 |  |  | yes |
| `cce_kozachenko` | .statistics.infotheory | ok | 0.052 |  | ok | 0.186 |  |  | yes |
| `cce_kernel_W-0.5` | .statistics.infotheory | ok | 1.406 |  | ok | 0.955 |  |  | yes |
| `xme_gaussian_k1` | .statistics.infotheory | ok | 0.003 |  | ok | 0.010 |  |  | yes |
| `xme_kozachenko_k1` | .statistics.infotheory | ok | 0.004 |  | ok | 0.020 |  |  | yes |
| `xme_kernel_W-0.5_k1` | .statistics.infotheory | ok | 0.004 |  | ok | 0.015 |  |  | yes |
| `xme_gaussian_k10` | .statistics.infotheory | ok | 0.004 |  | ok | 0.012 |  |  | yes |
| `xme_kozachenko_k10` | .statistics.infotheory | ok | 0.006 |  | ok | 0.029 |  |  | yes |
| `xme_kernel_W-0.5_k10` | .statistics.infotheory | ok | 0.005 |  | ok | 0.018 |  |  | yes |
| `di_gaussian` | .statistics.infotheory | ok | 0.037 |  | ok | 0.119 |  |  | yes |
| `di_kozachenko` | .statistics.infotheory | ok | 0.051 |  | ok | 0.214 |  |  | yes |
| `di_kernel_W-0.5` | .statistics.infotheory | ok | 0.848 |  | ok | 0.810 |  |  | yes |
| `si_gaussian_k-1` | .statistics.infotheory | ok | 0.009 |  | ok | 0.027 |  |  | yes |
| `si_kozachenko_k-1` | .statistics.infotheory | ok | 0.013 |  | ok | 0.046 |  |  | yes |
| `si_kernel_W-0.5_k-1` | .statistics.infotheory | ok | 0.014 |  | ok | 0.047 |  |  | yes |
| `mi_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  | yes |
| `mi_kraskov_NN-4` | .statistics.infotheory | ok | 0.426 |  | ok | 0.228 |  |  | yes |
| `mi_kraskov_NN-4_DCE` | .statistics.infotheory | ok | 0.283 |  | ok | 0.226 |  |  | yes |
| `mi_kernel_W-0.25` | .statistics.infotheory | ok | 0.002 |  | ok | 0.007 |  |  | yes |
| `tlmi_gaussian` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  | yes |
| `tlmi_kraskov_NN-4` | .statistics.infotheory | ok | 0.115 |  | ok | 0.220 |  |  | yes |
| `tlmi_kraskov_NN-4_DCE` | .statistics.infotheory | ok | 0.105 |  | ok | 0.224 |  |  | yes |
| `tlmi_kernel_W-0.25` | .statistics.infotheory | ok | 0.002 |  | ok | 0.006 |  |  | yes |
| `te_kraskov_NN-4_k-max-10_tau-max-4` | .statistics.infotheory | ok | 12.967 | yes | ok | 16.209 | yes |  | yes |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` | .statistics.infotheory | ok | 13.883 | yes | ok | 16.197 | yes |  | yes |
| `te_kraskov_NN-4_DCE_k-2_kt-1_l-1_lt-1` | .statistics.infotheory | ok | 0.515 |  | ok | 0.234 |  |  | yes |
| `te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1` | .statistics.infotheory | ok | 0.323 |  | ok | 0.232 |  |  | yes |
| `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1` | .statistics.infotheory | ok | 0.527 |  | ok | 0.226 |  |  | yes |
| `te_kernel_W-0.25_k-1` | .statistics.infotheory | ok | 0.004 |  | ok | 0.005 |  |  | yes |
| `gc_gaussian_k-max-10_tau-max-2` | .statistics.infotheory | ok | 0.045 |  | ok | 0.075 |  |  | yes |
| `gc_gaussian_k-1_kt-1_l-1_lt-1` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  | yes |
| `te_symbolic_k-1_kt-1_l-1_lt-1` | .statistics.infotheory | ok | 0.001 |  | ok | 0.002 |  |  | yes |
| `te_symbolic_k-10_kt-1_l-1_lt-1` | .statistics.infotheory | ok | 0.002 |  | ok | 0.004 |  |  | yes |
| `phi_star_t-1_norm-0` | .statistics.infotheory | ok | 0.036 |  | ok | 0.077 |  |  |  |
| `phi_star_t-1_norm-1` | .statistics.infotheory | ok | 0.037 |  | ok | 0.082 |  |  |  |
| `phi_Geo_t-1_norm-0` | .statistics.infotheory | ok | 0.097 |  | ok | 0.190 |  |  |  |
| `phi_Geo_t-1_norm-1` | .statistics.infotheory | ok | 0.097 |  | ok | 0.191 |  |  |  |
| `phase_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `phase_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `cohmag_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `icoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `psi_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.007 |  |  |  |
| `psi_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.003 |  | ok | 0.005 |  |  |  |
| `psi_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.003 |  | ok | 0.005 |  |  |  |
| `plv_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.003 |  | ok | 0.004 |  |  |  |
| `plv_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.003 |  | ok | 0.004 |  |  |  |
| `plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.003 |  | ok | 0.002 |  |  |  |
| `plv_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `plv_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `plv_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `pli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `wpli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dspli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dswpli_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `ppc_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `dtf_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.403 |  | ok | 1.260 |  |  |  |
| `dtf_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.422 |  | ok | 1.271 |  |  |  |
| `dtf_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.415 |  | ok | 1.269 |  |  |  |
| `dtf_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.416 |  | ok | 1.283 |  |  |  |
| `dtf_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.412 |  | ok | 1.287 |  |  |  |
| `dtf_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.406 |  | ok | 1.285 |  |  |  |
| `dcoh_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.616 |  | ok | 1.886 |  |  |  |
| `dcoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.607 |  | ok | 1.888 |  |  |  |
| `dcoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.595 |  | ok | 1.879 |  |  |  |
| `dcoh_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.625 |  | ok | 1.890 |  |  |  |
| `dcoh_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.628 |  | ok | 1.893 |  |  |  |
| `dcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.589 |  | ok | 1.912 |  |  |  |
| `pdcoh_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.418 |  | ok | 1.314 |  |  |  |
| `pdcoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.419 |  | ok | 1.299 |  |  |  |
| `pdcoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.407 |  | ok | 1.293 |  |  |  |
| `pdcoh_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.399 |  | ok | 1.281 |  |  |  |
| `pdcoh_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.400 |  | ok | 1.300 |  |  |  |
| `pdcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.414 |  | ok | 1.347 |  |  |  |
| `gpdcoh_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.612 |  | ok | 1.958 |  |  |  |
| `gpdcoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.599 |  | ok | 1.956 |  |  |  |
| `gpdcoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.594 |  | ok | 1.950 |  |  |  |
| `gpdcoh_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.607 |  | ok | 1.948 |  |  |  |
| `gpdcoh_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.611 |  | ok | 1.958 |  |  |  |
| `gpdcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.604 |  | ok | 1.949 |  |  |  |
| `ddtf_multitaper_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.799 |  | ok | 2.563 | yes |  |  |
| `ddtf_multitaper_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.807 |  | ok | 2.502 | yes |  |  |
| `ddtf_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.808 |  | ok | 2.490 | yes |  |  |
| `ddtf_multitaper_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.797 |  | ok | 2.490 | yes |  |  |
| `ddtf_multitaper_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.801 |  | ok | 2.491 | yes |  |  |
| `ddtf_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.795 |  | ok | 2.491 | yes |  |  |
| `gd_multitaper_delay_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | partial_nan | 0.005 |  | partial_nan | 0.012 |  |  |  |
| `gd_multitaper_delay_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | partial_nan | 0.005 |  | partial_nan | 0.010 |  |  |  |
| `gd_multitaper_delay_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | partial_nan | 0.005 |  | partial_nan | 0.011 |  |  |  |
| `sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.103 |  | ok | 0.326 |  |  |  |
| `sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.102 |  | ok | 0.350 |  |  |  |
| `sgc_nonparametric_mean_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.101 |  | ok | 0.342 |  |  |  |
| `sgc_nonparametric_max_fs-1_fmin-0_fmax-0-5` | .statistics.spectral | ok | 0.104 |  | ok | 0.347 |  |  |  |
| `sgc_nonparametric_max_fs-1_fmin-0_fmax-0-25` | .statistics.spectral | ok | 0.103 |  | ok | 0.337 |  |  |  |
| `sgc_nonparametric_max_fs-1_fmin-0-25_fmax-0-5` | .statistics.spectral | ok | 0.102 |  | ok | 0.344 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-5_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.014 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0_fmax-0-25_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-None` | .statistics.spectral | ok | 0.006 |  | ok | 0.013 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-1` | .statistics.spectral | ok | 0.004 |  | ok | 0.009 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-0_fmax-0-25_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-20` | .statistics.spectral | partial_nan | 0.022 |  | ok | 0.046 |  |  |  |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean` | .statistics.wavelet | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-25_mean` | .statistics.wavelet | ok | 0.002 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_mean_fs-1_fmin-0-25_fmax-0-5_mean` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_max_fs-1_fmin-0_fmax-0-5_max` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_max_fs-1_fmin-0_fmax-0-25_max` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `psi_wavelet_max_fs-1_fmin-0-25_fmax-0-5_max` | .statistics.wavelet | ok | 0.001 |  | ok | 0.002 |  |  |  |
| `lmfit_Ridge` | .statistics.misc | ok | 0.016 |  | ok | 0.035 |  |  |  |
| `lmfit_Lasso` | .statistics.misc | ok | 0.018 |  | ok | 0.037 |  |  |  |
| `lmfit_SGDRegressor` | .statistics.misc | ok | 0.017 |  | ok | 0.038 |  |  |  |
| `lmfit_ElasticNet` | .statistics.misc | ok | 0.018 |  | ok | 0.038 |  |  |  |
| `lmfit_BayesianRidge` | .statistics.misc | ok | 0.019 |  | ok | 0.039 |  |  |  |
| `gpfit_DotProduct` | .statistics.misc | ok | 0.204 |  | ok | 0.610 |  |  |  |
| `gpfit_RBF` | .statistics.misc | ok | 0.320 |  | ok | 0.977 |  |  |  |
| `coint_johansen_max_eig_stat_order-0_ardiff-10` | .statistics.misc | ok | 0.008 |  | ok | 0.019 |  |  |  |
| `coint_johansen_trace_stat_order-0_ardiff-10` | .statistics.misc | ok | 0.008 |  | ok | 0.020 |  |  |  |
| `coint_johansen_max_eig_stat_order-0_ardiff-1` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_johansen_trace_stat_order-0_ardiff-1` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_johansen_max_eig_stat_order-1_ardiff-10` | .statistics.misc | ok | 0.009 |  | ok | 0.020 |  |  |  |
| `coint_johansen_trace_stat_order-1_ardiff-10` | .statistics.misc | ok | 0.009 |  | ok | 0.020 |  |  |  |
| `coint_johansen_max_eig_stat_order-1_ardiff-1` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_johansen_trace_stat_order-1_ardiff-1` | .statistics.misc | ok | 0.007 |  | ok | 0.016 |  |  |  |
| `coint_aeg_tstat_trend-c_autolag-aic_maxlag-10` | .statistics.misc | ok | 0.025 |  | ok | 0.056 |  |  |  |
| `coint_aeg_tstat_trend-ct_autolag-aic_maxlag-10` | .statistics.misc | ok | 0.025 |  | ok | 0.054 |  |  |  |
| `coint_aeg_tstat_trend-ct_autolag-bic_maxlag-10` | .statistics.misc | ok | 0.025 |  | ok | 0.053 |  |  |  |
| `pec` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_log` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth_log` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth_abs` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `pec_orth_log_abs` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
| `ids` | .statistics.misc | ok | 0.000 |  | ok | 0.000 |  |  |  |
