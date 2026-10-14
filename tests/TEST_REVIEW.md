# Adversarial review of the test suite

Reviewed against commit `36ac45a`, 2026-10-14. Scope: whether the calculation
change detectors, their helpers, and production storage tests make meaningful
assertions and expose regressions. This is a test review, not a comprehensive
mathematical audit of the estimators.

## Findings and changes

| Finding | Resolution |
| --- | --- |
| A recorded calculation error accepted any exception, or a successful all-NaN result. Both false passes were reproduced. | Require the recorded exception representation (type and message); success replacing an error fails. Constructor/seeding failures propagate. Preprocessing variants now supply their recorded error, not just their status. |
| The numerical comparator and compute helper cast complex values to float64, potentially hiding an imaginary component. | Reject complex and nonnumeric results before conversion; check square process-matrix shape. Explicit adversarial cases protect these guards. |
| Missing committed golden files could silently skip the regression set. | Required small/full/preprocessing/extra/phi files now fail when absent. Added payload/metadata key, shape, dtype, diagonal, status, catalogue, and preprocessing-combination checks. |
| Generator checks ignored dataset/seed and some parameter/failure metadata; partial regeneration could mix different inputs. Repeat disagreements could still end with a successful check or saved set. | Check dataset/seed, error identity, and phi parameter metadata; refuse partial merges across different datasets/seeds and refuse saving a non-reproducible set. Different exception causes across repeats are detected. Tests exercise these failure cases. |
| Extra wavelet catalogue checks compared identities/order without all constructor parameters, dependencies, or labels. | Compare all recorded catalogue fields too. |
| Many extra wavelet snapshots are constant, zero, or infinite; high execution coverage did not make them sensitive to all aggregation changes. | Added nondegenerate synthetic backend tensors that independently check band endpoints, row/column orientation, mean/max, phase-angle and PLV-magnitude placement, time/frequency axes, and cache reuse. Existing snapshots are retained unchanged. |
| Eight stochastic SPIs were omitted from normal Calculator numerical comparisons because one global seed gives a different RNG history. | Added a separate shared-cache Calculator test covering all twelve stochastic SPIs, reseeding each SPI to its existing golden's conditions. The normal Calculator comparisons retain their original RNG policy. |
| The proposed PLI “fixed” test demanded symmetry, although the current signed variant is antisymmetric and the scientific decision is still pending. | Removed that unsupported expected-failure assertion. Strengthened the current-behavior test to require finite, nonzero signed antisymmetry. Definitions and labels are unchanged. This accounts for one fewer expected failure. |
| Storage round trips alone could miss a packer and decoder sharing an ordering mistake. Mean tests did not exercise multiple column blocks. | Added independently specified payload/decoder oracles for all five modes and both precisions; added mean tests spanning several blocks and a partial final block. |
| An unavailable JVM could reduce a complete run to partial coverage through optional skips. | Added and tested `--require-java`, which fails when the JVM is unavailable. Ordinary optional-Java runs remain available. |
| Difference diagnostics could report zero relative error against zero, or misleading results for unequal nonfinite values. | Report infinite relative error/invalid-output differences correctly, with guard tests. |

Additional independent references cover empirical covariance and precision on
2, 5, and 7 channels, including 1024 observations; all preprocessing flag
combinations on multiple replications; replacing a Calculator's dataset; and
IDS feature correlations, cross-inputs, three reductions, two bandwidths, and
explicit permutation p-values. These references supplement the broad snapshots;
they do not certify every estimator or parameter configuration.

## Adversarial mutation checks

Ten isolated pytest runs temporarily changed runtime behavior without editing
production source. Every mutation produced an assertion failure in its intended
test, with no collection or setup error:

1. Replace covariance output with zeros.
2. Transpose directed spectral edges.
3. Skip detrending.
4. Ignore the selected wavelet band.
5. Remove the wavelet phase-angle transform.
6. Take PLV magnitude after averaging instead of before.
7. Reverse packed upper-triangle order in both packer and decoder.
8. Reuse the first mean-reduction block for subsequent blocks.
9. Restore the permissive recorded-error handling.
10. Discard imaginary components during comparison.

The paired packing mutation would preserve a self-consistent round trip; the
fixed payload oracle catches it. The block mutation leaves one-block inputs
unaffected; the new multi-block test catches it. Ten mutations are evidence for
these particular guards, not an exhaustive mutation score.

Reproduce the optional audit with `uv run --no-sync python tests/review_mutations.py`.
It first runs unmodified controls, then each mutation in a separate process,
and saves results locally. It is excluded from ordinary test collection.

## Verification

Final run counts and weighted coverage are recorded in
[test_review_results.json](test_review_results.json). Raw local logs, JUnit,
coverage data, and mutation traces are under
`tests/storage/.artifacts/test_review/` (ignored). The original full/slow baseline
passed **576 tests with 11 expected failures**, before strengthening the suite.

| Final check | Passed | Skipped | Expected failures |
| --- | ---: | ---: | ---: |
| Default, Java required | 694 | 15 | 9 |
| Full/slow, Java required, legacy diagnostic excluded | 707 | 0 | 10 |
| Storage and analysis | 439 | 0 | 0 |
| Storage against installed wheel, outside checkout | 439 | 0 | 0 |
| Final comparison guards and corpus policies | 51 | 0 | 0 |

The default skip identities are unchanged; they concern slow tests and the
legacy placeholder. The sole removed expected failure is the unsupported PLI
symmetry assertion. There are 131 additional calculation-suite cases and 20
additional storage cases.

| Measured scope | Statements covered | Branches covered |
| --- | ---: | ---: |
| Entire pyspi package, calculation and storage runs combined | 88.57% | 82.62% |
| Calculation code, excluding storage/analysis | 85.58% | 75.91% |
| Statistics wrappers | 94.58% | 87.88% |
| Storage and analysis | 98.57% | 96.55% |
| Regression harness, policy, and generator helpers | 71.90% | 62.04% |

The six critical helper functions for computation, recorded-error handling,
real-value validation, numerical comparison, difference diagnostics, and required
golden files have no uncovered statements or branches. Other helper coverage
includes unexercised CLI/reporting/provenance and optional paths; aggregate
percentages are not presented as proof of assertion quality.

The real generator CLI also passed a bounded two-repeat small/full/preprocessing
and phi check, including the recorded cross-correlation error. Real partial
regeneration and rechecking passed in a temporary baseline copy. Existing
committed goldens were not regenerated.

Commands for complete local checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m pytest --require-java
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m pytest \
  --require-java --pyspi-size=full --runslow --ignore=tests/test_SPIs.py -n 2 --timeout=1800
uv run --no-sync python -m pytest -c tests/storage/pytest.ini tests/storage
```

All production calculation/storage source, dependency locks, configuration, and
existing golden values remain byte-identical to `36ac45a`. Original patient
recordings and output stores were not read or modified. The resolved untracked
`upgrade_migration.md` report was removed at the user's request; its resolutions
and verification remain in the committed migration and production reports.

## Remaining limits

* Golden comparisons cover the configured 285 identifiers at two input sizes,
  96 preprocessing cases, 54 extra wavelet variants, and phi-native snapshots.
  They preserve current behavior, including documented defects. New parameter
  combinations, sampling rates, and workload sizes still need targeted cases
  when changed.
* Default tolerances remain `rtol=1e-6`, `atol=1e-8`. Changes smaller than those
  bounds can pass; tightening them across all estimators without measured
  justification is not part of this review.
* The synthetic spectral/wavelet cases validate wrapper assembly and backend
  boundaries. They do not independently establish the mathematical correctness
  of MNE, spectral-connectivity, JIDT, or the other external estimators. Degenerate
  single-epoch wavelet outputs remain a scientific usability question.
* PSI definitions/bands/axes, signed PLI/wPLI, phase aggregation, labels, CCM
  embedding selection, and dependency-induced numerical changes still need
  their individual mathematical reviews. Existing expected failures are
  behavior alarms rather than complete correctness oracles.
* The legacy z-score benchmark reports deviations rather than failing on them.
  It was inspected, not rerun: the existing per-SPI golden tests are the gating
  comparisons. Repeating that expensive diagnostic adds little to this scope.
* Coverage of auxiliary Calculator/CorrelationFrame analysis APIs, optional
  branches, and some native-phi fallback/alternative methods remains incomplete.
  These are recorded rather than hidden by excluding their code from coverage.
  More targeted tests should accompany work on those paths.
* Storage checks do not simulate physical power loss, unsupported concurrent
  writers, or object stores. No patient conversion or recalculation occurred.
