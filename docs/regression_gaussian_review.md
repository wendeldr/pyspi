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
- Additional Gaussian Granger candidates: orders 4, 16 and 32, alongside order 1;
  both histories contain consecutive preceding samples and prediction remains
  one sample ahead. Benchmark each order before enabling. Approved 2026-10-27.
- Symbolic TE: implement genuine ordinal-pattern TE with length 3 in both
  channels. Two candidates: pattern spacing / prediction shift 4/4 and 16/16
  samples at 2048 Hz. These predict overlapping pattern transitions. Validate
  known coupling, independent autocorrelated signals, EEG pattern occupancy,
  and incremental runtime before run inclusion. They replace the mislabeled
  Gaussian variants, not reproduce their values. Approved 2026-10-27.
- Still to discuss: redundant barycenter
  outputs. BayesianRidge
  will retain its documented iterative updates and convergence rule unless a
  change is agreed.

## Work and verification ledger

| Scope | State / required evidence |
| --- | --- |
| Shared computation | In progress: mutation-checked bounded moment cache, transformation and observation-interval keys. Gaussian consumers pending. |
| Ridge and RECI | Complete initial chunk: 19 focused cases (18 initial passes plus corrected near-cubic fixture), 10 small/preprocessing and 2 full goldens pass; no golden changes. Three-repeat timings below. |
| BayesianRidge, Lasso, ElasticNet, SGD | Pending implementation and definition decisions above. |
| Gaussian MI, joint/conditional entropy and lagged MI | Pending native formulas and JVM-independent construction. |
| Fixed/automatic Gaussian TE, stochastic interaction, crossmap/causal entropy, DI | Pending shared aligned histories, formula checks and independent references. |
| Cross-metric corrections | Current Data.add_process already preprocesses consistently; do not repeat the old audit's now-fixed input bug. Generic undirected loop still duplicates pairs; change only proven symmetric affected metrics. |
| Config / redundancy | Lower-penalty Lasso conditional on cheap varying output; replace SGD with exactly solved Ridge. Starting run selection: 205 outputs. Other additions/removals require decisions. |
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
