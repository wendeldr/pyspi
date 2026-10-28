# Regression and Gaussian calculation review

Active goal: optimize runtime across compatible regression and Gaussian
information calculations, correct implementation errors, and agree on estimator
changes before applying them. Starting commit: `31993ac`. Preserve the completed
spectral/PEC review, 0.5-second nonoverlapping windows, and existing preprocessing.

## Decisions

- RECI: retain the documented min–max-scaled cubic plus intercept, with
  **in-sample** MSE. Approved 2026-10-27. Do not substitute held-out fitting or a
  different polynomial family.
- Lasso: retain a lower-penalty version if its measured total cost is negligible
  (well below one or two hours for the cohort) and it produces varying results.
  Test alpha=0.1 first on representative EEG; if degenerate try 0.05 then 0.01.
  This supersedes the initial removal decision. The test checks nondegeneracy,
  not clinical optimality. Approved 2026-10-27.
- ElasticNet: preserve alpha=1 and L1 ratio=0.5, expose settings, and use the
  exact penalized solution instead of reproducing the solver's premature-zero
  artifact. Approved 2026-10-27.
- SGDRegressor: replace the approximate fit with the exact same squared-loss
  plus L2 objective, named as Ridge with per-sample penalty 0.0001. Remove the
  obsolete stochastic optimization path. Approved 2026-10-27.
- Gaussian Granger / TE: retain **half** the log prediction-variance ratio, in
  nats, and explicitly document the convention. Conventional Geweke F is
  twice this value. Approved 2026-10-27.
- Directed information: use the standard conditional-information sum, including
  present source observations; retain five terms and expose/document the length.
  Remove the erroneous target-only autocorrelation contribution across existing
  Gaussian, kernel and Kozachenko implementations. Approved 2026-10-27.
- Crossmap entropy: explicitly define it as source-history conditional entropy
  H(B now | A's previous k samples), with no B-history conditioning and no
  connection to CCM. Fix k=10 using nine lags across the existing Gaussian,
  kernel and Kozachenko variants. Approved 2026-10-27.
- Additional Gaussian Granger candidates: orders 4, 16 and 32, alongside order 1;
  both histories contain consecutive preceding samples and prediction remains
  one sample ahead. Benchmark each order before enabling. Approved 2026-10-27.
- Symbolic TE: implement genuine ordinal-pattern TE with length 3 in both
  channels. Two candidates: pattern spacing / prediction shift 4/4 and 16/16
  samples at 2048 Hz. These predict overlapping pattern transitions. Validate
  known coupling, independent autocorrelated signals, EEG pattern occupancy,
  and incremental runtime before run inclusion. They replace the mislabeled
  Gaussian variants, not reproduce their values. Approved 2026-10-27.
- Run redundancy: remove both Euclidean mean summaries (ordinary and squared),
  original Ridge, and the exact SGD replacement from this run. At fixed window
  length and normalization, these are constant or affine transformations of
  selected empirical covariance / squared covariance, even after averaging
  windows. Retain their optimized library implementations for other inputs.
  Approved 2026-10-27.
- Gaussian run redundancy: retain Gaussian MI and remove Gaussian joint and
  conditional entropy. Under per-window normalization, their marginal Gaussian
  entropies are constant, so JE/CE are affine functions of MI even after
  averaging. Approved 2026-10-27. Non-Gaussian variants are unchanged.
- All redundancy removals are specific to the working normalized, fixed-length
  selection. Unnormalized channel variances/offsets generally break them.
  Ordinary Euclidean barycenter mean is still zero if detrending/centering
  removes means. Library implementations remain available for other runs.
- BayesianRidge
  will retain its documented iterative updates and convergence rule unless a
  change is agreed.

## Work and verification ledger

| Scope | State / required evidence |
| --- | --- |
| Shared computation | In progress: mutation-checked bounded moment cache, transformation and observation-interval keys. Gaussian consumers pending. |
| Ridge and RECI | Complete initial chunk: 19 focused cases (18 initial passes plus corrected near-cubic fixture), 10 small/preprocessing and 2 full goldens pass; no golden changes. Three-repeat timings below. |
| BayesianRidge, Lasso, ElasticNet, SGD | Implemented and independently checked: 53 regression tests; exact penalties and separately stopped Bayesian updates. Full-family and retained-selection timings below. |
| Euclidean barycenters | Shared moment means and bounded unique-pair extrema; 8 independent cases pass. Other barycenter modes are unchanged. |
| Gaussian MI, joint/conditional entropy and lagged MI | Native shared moments implemented; 21 independent checks pass, including voltage/extreme units, singular-pair isolation, JVM-free construction/execution and redundancy under averaging. Family timing pending with remaining Gaussian work. |
| Fixed/automatic Gaussian TE, stochastic interaction, crossmap/causal entropy, DI | Pending shared aligned histories, formula checks and independent references. |
| Cross-metric corrections | Current Data.add_process already preprocesses consistently; do not repeat the old audit's now-fixed input bug. Generic undirected loop still duplicates pairs; change only proven symmetric affected metrics. |
| Config / redundancy | Alpha=0.1 Lasso varies on all 36 EEG pilot windows and meets runtime gate. Six approved redundant outputs removed: current selection 199, before pending Granger/symbolic candidates. Starting selection: 205. Separate nine-output amplitude/reference template uses detrend=true, zscore=false. |
| Validation / goldens | Focused tests and affected regression checks; update only changed outputs with provenance. |
| Timings / memory | Comparable 118 × 1,024 workload; full families and shared incremental stages; per-window, ×600 and ×41,400 projections. |
| Documentation / CSV | Keep this ledger current; CSV only brief verified corrections and measured speedups. |
| Workflow opportunities | Record window parallelism/thread and memory limits for a later goal; no production multiprocessing here. |

The goal authorizes commits in reasonable chunks. Pushing is not requested.
DTW, CCM, distance-dependence refactors, and non-Gaussian Java replacement remain
outside this goal except affected shared DI formulas and symbolic TE decisions.

## Initial Ridge / RECI measurements

Receipt: `tests/regression_refactor_results.json`; reproduce with
`.venv/bin/python tests/benchmark_regression.py`. Three alternating baseline/new
runs on a synthetic 118-channel × 1,024-sample window, one BLAS thread, fresh
Data/cache for each workload. Baseline is committed code at `31993ac`; timings
exclude preprocessing, imports and I/O. These are serial projections on this
computer, not measured cohort throughput.

| Workload | Old window | New window | Speedup | New ×600 | New ×41,400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge alone | 23.23 s | 3.34 ms | 6,956× | 2.00 s | 2.30 min |
| RECI alone | 51.17 s | 8.93 ms | 5,732× | 5.36 s | 6.16 min |
| Both, shared Data | 73.65 s | 8.36 ms | 8,814× | 5.01 s | 5.77 min |

Combined-stage medians: Ridge 1.81 ms and RECI an additional 6.42 ms. Different
workload medians fluctuate with machine load; RECI's nonlinear transform is
different from Ridge's and does not reuse its Gram matrix. The comparison does
not imply that running both intrinsically costs less than RECI alone.

Maximum absolute differences: Ridge 1.23e-15, RECI 3.13e-17. Combined derived
cache 5.04 MiB plus a 0.92 MiB input snapshot; separately measured tracemalloc
peak 6.45 MiB. Tracemalloc is not a whole-process/native-library RSS measure.

Validation commands for this chunk:

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/unit/test_regression_refactor.py -q
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/pyspi-mpl _MNE_FAKE_HOME_DIR=/tmp/pyspi-mne .venv/bin/python -m pytest tests/regression/test_spi_goldens.py -q -k 'lmfit_Ridge or reci' --runslow
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/pyspi-mpl _MNE_FAKE_HOME_DIR=/tmp/pyspi-mne .venv/bin/python -m pytest tests/regression/test_spi_goldens.py -q -k 'test_spi_matches_golden and (lmfit_Ridge or reci)' --runslow --pyspi-size=full
```

The initial regression launch failed during MNE import before any calculation
because MNE opens its home config read/write; the temporary MNE config root
above resolved that environment error. The initial near-cubic unit fixture
incorrectly assumed cubing before min–max scaling preserved an exact cubic;
fixing its endpoints made the intended test pass. Only that failed case was
repeated, not the 18 passing cases.

## Remaining regression family and Euclidean summaries

`tests/regression_family_results.json` compares all six models in their original
order at 118 × 1,024, using the same three-repeat baseline protocol. Median
complete-family time is **193.21 s → 19.38 ms** (about 9,971×). New serial
projections: **11.63 s for 600 windows**, **13.37 min for 41,400**. These are
synthetic measurements, excluding imports, preprocessing, I/O and parallelism.
The Lasso penalty and SGD solver changed by approval, so this compares the old
family with its approved replacement, not six numerically identical outputs.

| Stage after preceding family calculations | Added time/window |
| --- | ---: |
| Ridge, including shared moments | 2.87 ms |
| Lasso, alpha=0.1 | 1.22 ms |
| ElasticNet, original penalties | 0.77 ms |
| BayesianRidge | 5.84 ms |
| Exact per-sample Ridge replacing SGD | 0.97 ms |
| RECI, including its distinct transforms | 8.65 ms |

The derived cache was 5.70 MiB, plus a 0.92 MiB input snapshot; separately
measured tracemalloc peak was 7.55 MiB. Maximum differences for unchanged
Ridge/BayesianRidge/RECI definitions were at most 1.23e-15. This random benchmark
does not expose ElasticNet's threshold artifact; a focused r=0.505 test does.

The retained subset (Lasso, ElasticNet, BayesianRidge, RECI) was also measured
from a cold cache without the removed Ridge outputs warming it:
`tests/regression_selection_results.json`, nine native-only repetitions.
Median **28.31 ms/window**, **16.99 s ×600**, **19.54 min ×41,400**. The higher
time despite fewer outputs reflects differing machine load, including concurrent
focused regression checks, not a claim that removing Ridge makes it slower.
No historical baseline fits were repeated for this subset. Derived cache was
5.41 MiB; tracemalloc peak 7.04 MiB, excluding whole-process/native RSS.

The alpha=0.1 Lasso increment projects to **0.73 s ×600 / 50.33 s ×41,400**.
The separate EEG probe (`audit/briefing/lasso_penalty_review.json`) tested 12
selected contacts and 12 windows in each of patients 001, 041 and 096. All 36
windows had varying pair errors; 61–92% of pair slopes survived alpha=0.1.
This establishes nondegeneracy and acceptable runtime, not an optimal clinical
penalty. No smaller penalties were needed for run inclusion.

Focused verification: 53 regression cases, 8 barycenter cases, 18 small/
preprocessing/catalogue checks and 9 full affected checks passed. The latter
selectors also included the unchanged Euclidean pairwise-distance golden.
Only ElasticNet and the renamed exact SGD replacement were regenerated at
small/full sizes. All unrelated arrays and records were preserved; preprocessing
arrays were not regenerated. Provenance is in each affected `meta.json`.
The alternative workflow schema and all nine constructors were resolved after
substituting only the user-specific input/metadata placeholders; no EDF samples
were read and no cohort run was launched.

Reproduce the new retained-selection timing without repeating old fits:

```
.venv/bin/python tests/benchmark_regression.py --models Lasso ElasticNet BayesianRidge RECI --family-only --reviewed --native-only --repeats 9 --out tests/regression_selection_results.json
```

## Native Gaussian foundation

JE, CE, MI and lag-one MI now reuse the mutation-checked moment workspace.
Ordinary MI/JE/CE share the same pair statistics and the regression Gram matrix;
lagged MI uses separately centered, correctly aligned source/target intervals.
The native constructors and these calculations do not require a JVM. Both the
workflow resolver and Calculator now resolve Java requirements per estimator
within mixed-estimator classes, so their non-Gaussian variants still require it.

The covariance convention remains N−1, entropies/information remain in nats,
and CE[i,j] means H(channel j | channel i). No physical-unit variance floor or
artificial regularization is introduced. Extreme units are scaled internally,
with log variances restoring differential-entropy units. Near-perfect fits use
explicit residual energy instead of subtracting rounded squared correlations.
Exact/numerically unresolved linear dependence has infinite MI and undefined
JE/CE; constant-variable MI is zero, while conditioning on a constant leaves
the other variable's entropy unchanged. Nonfinite input and insufficient samples
are distinguished in `diagnostics`; bad pairs do not blank the whole matrix.
JE/CE now correctly have signed labels, including their existing non-Gaussian
differential estimators. No non-Gaussian estimator formula changed in this chunk.

The 21 independent cases passed (20 initially; one fixture needed to insert NaN
through Data's writable view because Data construction correctly rejects it).
The corrected case and 12 affected dependency/configuration checks then passed.
Gaussian numerical goldens are retained; only the affected dependency/signed
metadata was updated with provenance. Benchmarking is deferred until the
remaining Gaussian family can be measured with actual shared costs.
The affected checks passed: 9 small/preprocessing/catalogue cases and 4 full
Gaussian cases. Their numerical arrays were not regenerated.

Still pending: fixed/automatic Granger, stochastic interaction, corrected XME
and DI, shared causal entropy, genuine symbolic TE, higher-history runtime gates,
final config inclusion/count and the full Gaussian benchmark. This foundation
does not finish the active goal.
