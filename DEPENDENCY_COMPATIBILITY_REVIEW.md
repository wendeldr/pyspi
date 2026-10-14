# Dependency compatibility review

This records the investigation before implementation. The subsequent migration
and current installation instructions are in [DEPENDENCY_MIGRATION.md](DEPENDENCY_MIGRATION.md).

Investigated 2026-10-14. The question was whether older calculation dependencies are necessary, or newer versions can support both pyspi calculations and the chosen storage format.

**Finding: Python 3.12 and NumPy 2 can run calculation and storage tests together.** A temporary environment with Python 3.12.3, NumPy 2.5.4, Zarr 3.4.1, and numcodecs 0.17.0 passed the frozen default calculation suite and the storage suite. It retained the older metric-library versions and required two `np.in1d` → `np.isin` substitutions inside a temporary copy of spectral-connectivity 1.1.2. This is evidence for a small compatibility migration, not approval to change the project's dependencies or a proof that every calculation is mathematically correct.

No production calculation code, dependency declarations, existing environments, patient outputs, or golden values were changed. All dependency installs and diagnostic adapters were under `/tmp`. A frozen source/test snapshot prevented other agents' ongoing test work from changing the comparison during the runs. Its files and goldens remained byte-for-byte unchanged.

## Controlled comparisons

The snapshot's default tier collected 574 cases. It skips slow tests and the legacy benchmark. Known-bug expected failures remain intentional locks; passing these tests means agreement with existing behavior within the recorded tolerances and NaN masks, not universal bit equality or scientific correctness.

Cases whose recorded result is an error/all-NaN can pass while remaining unsuccessful calculations; the harness can map an exception to that saved all-NaN result. Those cases do not establish that the dependency upgrade made the method work or preserved its precise failure cause. Slow/full-tier and representative EEG runs remain unverified.

| Environment | Calculation result | Notes |
| --- | --- | --- |
| Python 3.11.16, NumPy 1.26.4, existing metric dependencies | 549 passed, 15 skipped, 10 expected failures | Baseline, 120.5 s |
| Same, NumPy 2.4.6 only | 545 passed, 4 failed, 15 skipped, 10 expected failures | All new failures trace to missing `np.in1d` in old spectral-connectivity; 130.9 s |
| Same NumPy 2.4.6 environment, two temporary compatibility substitutions | 549 passed, 15 skipped, 10 expected failures | 78.5 s |
| Python 3.12.3, NumPy 2.5.4, original metric dependency versions, modern storage dependencies, same temporary substitutions | 549 passed, 15 skipped, 10 expected failures | 129.5 s; independently installed wheels, not reuse of Python 3.11 binaries |

The final unified environment also passed **381 storage tests with two expected failures** in 18.7 s, and `uv pip check` reported all 98 installed distributions compatible. The two storage expected failures belong to the unselected lossy tolerance experiment.

These durations are recorded test-run times, not controlled speed benchmarks. They include cache, JIT, interpreter, and scheduling differences and do not establish a NumPy performance improvement.

The unified environment was constructed from the calculation lock plus storage dependencies, replacing the NumPy version and omitting the editable pyspi distribution. Tests imported the frozen pyspi source. Therefore the run deliberately bypassed the project's existing `numpy<2` metadata constraint; an unchanged `pip install pyspi` is not claimed to accept this configuration. Formal dependency declarations and distribution testing remain migration work.

## Why the restrictions exist

| Restriction | Evidence for its origin/current effect | Upgrade assessment |
| --- | --- | --- |
| `numpy<2` | Commit `db9af188` (2024-08-24) explicitly cites tslearn incompatibility. Current tslearn 0.9.0 imported and passed the default covered calculations under NumPy 2. | The original reason was not reproduced. A different removed-API problem exists in spectral-connectivity 1.1.2; small substitutions restore the tested behavior. |
| `mne==0.23.0` | Equality pin dates to commit `e66057f9` (2022-02-10). pyspi imports `mne.connectivity` in `misc.py` and `wavelet.py`; newer MNE moved connectivity to `mne-connectivity`. | New MNE imports, but unadapted pyspi modules fail. Small input/output adapters look promising in the selected probes below. |
| `spectral-connectivity<2` | Added by the ongoing test work. Version 2.0.1 rejects the 2D input passed by current wrappers and requires explicit `(time, trials, signals)` input. Its frequency grid also changed. | Not a drop-in upgrade. The extra trial axis is straightforward; frequency conventions and affected summaries need separate decisions. |
| `pyEDM==1.15.2.0` | Commit `eb62cafe` (2025-04-12) reintroduced the pin after an unpinned interval, without documenting a precise failure. Version 2.x replaces cppEDM with NumPy/SciPy. | Version 2.5.7 runs a fixed-embedding CCM probe, but its numerical results change. Do not treat it as a transparent packaging fix. |
| `setuptools<81` | Commit `6f3f8f11` (2026-09-30) preserves `pkg_resources`, imported by pyEDM 1.x to load bundled example CSV files. | Actual removal was in setuptools 82.0.0. The 81.0.0 wheel still includes it; 84.0.0 does not. The existing bound is conservative. Latest setuptools requires addressing the resource-loading dependency or upgrading pyEDM. |
| `cdt==0.5.23` | Commit `2d346285` (2024-05-18) records a downgrade but not its specific failure. | Version 0.6.0 imported, and the four pairwise methods used here matched the selected comparison. Broader regression testing is still needed before adopting it. |

The recent SciPy, scikit-learn, pandas, numba, and tslearn versions already in the calculation environment advertise NumPy ranges admitting NumPy 2. The isolated calculation runs provide stronger evidence than metadata alone for the paths they exercised. NumPy's [migration guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html) explains that API, ABI, and type-promotion changes still require testing.

MNE documents that connectivity [moved out of MNE as of 0.24](https://mne.tools/stable/api/connectivity.html). The current [pyEDM API documentation](https://sugiharalab.github.io/EDM_Documentation/edm_functions/) distinguishes its old C++ backend from the 2.x NumPy/SciPy implementation. The [setuptools history](https://setuptools.pypa.io/en/latest/history.html#v82-0-0) records the `pkg_resources` removal in 82.0.0; the repository comment saying removal begins at 81 is inaccurate and was left unchanged during this investigation.

## Actual NumPy blocker and compatibility substitution

NumPy 2.4.6 lacks `np.in1d`. Spectral-connectivity 1.1.2 calls it when finding independent significant frequencies for group delay. Three group-delay golden comparisons fail directly, and the fast shared-cache Calculator comparison fails because those results become NaN.

The temporary dependency copy has two uses: one tests a one-dimensional group-label vector, and one tests a one-dimensional frequency-index vector. Replacing those calls with `np.isin` preserves their intended output shape for those inputs. No blanket NumPy monkeypatch was installed. Both complete default-suite comparisons passed after the substitution, including the group-delay cases and shared-cache Calculator check.

This dependency edit is not a durable installation strategy: reinstalling the wheel would replace it. A migration must choose a maintainable patch/adapter, an upstream-supported version with equivalent behavior, or another explicit approach, with regression coverage. The current source also contains `except np.AxisError` in `pyspi/statistics/wavelet.py`; NumPy 2 moved that exception to `np.exceptions`. The covered successful wavelet paths did not hit that handler, so it remains an additional source-level compatibility item rather than a demonstrated default-suite failure.

NumPy 2.5.4 requires Python ≥3.12; the Python 3.11 comparison consequently used 2.4.6. PyEDM 1.15.2.0 does provide a Linux Python 3.12 wheel, and that wheel worked in the unified environment. Upgrading the interpreter does not by itself require adopting pyEDM 2.x or newer MNE.

## Newer metric-library probes

A second isolated environment used NumPy 2.4.6, MNE 1.13.2, MNE-Connectivity 0.9.0, spectral-connectivity 2.0.1, pyEDM 2.5.7, and CDT 0.6.0. It reused unchanged compatible Python 3.11 dependencies from the baseline environment read-only. These were small comparisons on one identical seeded 3-channel × 100-sample input, with detrending followed by z-scoring. They are not a full regression run of the newer libraries.

Diagnostic adapters were confined to the probe script: add a single trial axis for spectral-connectivity 2; use MNE-Connectivity's new function names and extract dense NumPy data; run new CCM serially for the small test. No tolerances or golden values were rewritten.

### MNE and MNE-Connectivity

Unmodified `pyspi.statistics.misc` and `.wavelet` fail to import with new MNE because `mne.connectivity` is absent. With the standalone package and object-to-array extraction:

- Three envelope-correlation cases (ordinary, log, and pairwise orthogonalization) matched within a maximum absolute difference of approximately `1e-15`.
- Four wavelet methods (`coh`, `plv`, `pli`, `wpli`) matched exactly in the probe.

This supports an API migration candidate, not a universal equivalence claim. Other parameters, dimensions, and estimator changes need coverage before selecting it.

### Spectral-connectivity 2.0.1

Adding the trial axis makes the calls execute, but the returned frequency grid includes the Nyquist bin for even-length inputs. For this 100-sample, fs=1 example:

- Version 1.1.2: 50 bins, ending at 0.49.
- Version 2.0.1: 51 bins, ending at 0.5.
- Coherence magnitude and phase agree exactly on the 50 shared bins.

Consequently, full-band averages change when the new final bin is included. Excluding it restores exact coherence-magnitude and phase summaries in this probe. Other measures can depend internally on the whole frequency grid: direct directed transfer function still differed by approximately `7.20e-4` after changing only the wrapper's aggregation boundary.

The PSI wrapper also exposed a separate issue: a configured band is not forwarded by its current generic cache path when the dependency method accepts default arguments. A spy on `PhaseSlopeIndex(fmin=0, fmax=0.25)` observed an empty keyword-argument dictionary. Changing only the wrapper constructor's fmax therefore did not change its full-band PSI calculation. Calling the new dependency directly with the old frequency range restored exact agreement in this probe. This is a targeted finding for the scientific review backlog, not a correction applied here. Both inspected versions retain the all-frequency-pairs PSI implementation previously documented in the storage review.

Thus newer spectral-connectivity is usable, but adopting it requires explicit decisions about input shape, frequency-band forwarding, Nyquist inclusion, and frequency-dependent summaries. It must not silently change historical metric identities.

### pyEDM and CDT

New pyEDM 2.5.7 ran the fixed-E=2 CCM call with the current argument names, but the maximum absolute matrix difference was approximately `0.00291745`, beyond the probe's `rtol=1e-6, atol=1e-8` comparison. Its implementation and sampling behavior differ from 1.x. Automatic embedding and all configurations were not tested. Upgrading it to eliminate `pkg_resources` is therefore a scientific/numerical migration, whereas replacing the old loader's resource access could be a separate packaging change.

CDT 0.6.0 matched ANM, CDS, RECI, and IGCI in the selected probe; maximum observed absolute difference was approximately `5.6e-17`. Both comparisons explicitly used CPU settings. This does not certify all CDT features or its automatic device behavior.

## Recommended next discussion

The earlier suggestion to adapt storage down to the old NumPy stack is no longer the preferred starting point. A temporary modern Python/NumPy environment already runs both suites while retaining metric-library versions.

Discuss a narrow compatibility migration to Python 3.12/NumPy 2, with a maintainable solution for the old spectral dependency's removed calls and the wavelet exception namespace. Preserve the selected metric implementations initially. Keep MNE/CDT upgrades as separate candidates for API adaptation and broader regression checks; evaluate pyEDM and newer spectral-connectivity individually where outputs or frequency conventions change.

Before promoting any migration, run the slow/full calculation tier, add targeted checks for changed compatibility paths and relevant EEG sizes, validate packaging installation, and measure performance if speed is claimed. The default-suite result establishes feasibility, not completion of that migration. No upgrade has been selected or installed into the working environments.

## Saved evidence

Local reports, source/golden hashes, requirements, small probe scripts, arrays, PyPI metadata, and comparison JSON are saved under [tests/storage/.artifacts/dependency_audit_20261014](tests/storage/.artifacts/dependency_audit_20261014). The frozen source snapshot and isolated environments remain under `/tmp/pyspi-dependency-audit-20261014` for further inspection. Generated artifacts and environments are not tracked in Git.

The local test summary is [test_summary.json](tests/storage/.artifacts/dependency_audit_20261014/test_summary.json); native comparisons are in [probe_comparison.json](tests/storage/.artifacts/dependency_audit_20261014/probe_comparison.json), with frequency diagnostics in [frequency_comparison.json](tests/storage/.artifacts/dependency_audit_20261014/frequency_comparison.json) and [psi_band_probe.json](tests/storage/.artifacts/dependency_audit_20261014/psi_band_probe.json).
