# Regression and Gaussian calculation review

Completed goal: optimize runtime across compatible regression and Gaussian
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
| Shared computation | Complete: mutation-checked bounded moment/history cache; transformation, sample-alignment and lag keys; shared target fits and independent auto-embedding searches. Full-family and actual shared-increment benchmarks recorded below. |
| Ridge and RECI | Complete initial chunk: 19 focused cases (18 initial passes plus corrected near-cubic fixture), 10 small/preprocessing and 2 full goldens pass; no golden changes. Three-repeat timings below. |
| BayesianRidge, Lasso, ElasticNet, SGD | Implemented and independently checked: 53 regression tests; exact penalties and separately stopped Bayesian updates. Full-family and retained-selection timings below. |
| Euclidean barycenters | Shared moment means and bounded unique-pair extrema; 8 independent cases pass. Other barycenter modes are unchanged. |
| Gaussian MI, joint/conditional entropy and lagged MI | Complete: shared native moments; 21 independent checks, affected regressions and full/shared family benchmarks. JVM-free integration includes the history and symbolic calculations too. |
| Fixed/automatic Gaussian TE, stochastic interaction, crossmap/causal entropy, DI | Complete: 39 independent Gaussian-history checks, including a filtered ill-conditioned design; 20 retained-estimator checks; affected small/full goldens and benchmarks. |
| Genuine symbolic TE | Implemented with independent categorical counts, direction, permutation, ties, sparse/dense and mutation checks (9 passed). Three-patient EEG occupancy and autocorrelated-null checks completed. Both approved m=3 replacements enabled. |
| Cross-metric corrections | Shared bounded raw histories, target conditional entropies and CCE/DI sums; mutation-aware automatic Theiler windows with one ACF per channel; Java TE serialization retains configured histories. Gaussian SI evaluates unique pairs. Retained non-Gaussian reverse evaluations remain until numerical permutation behavior is established. |
| Config / redundancy | Alpha=0.1 Lasso varies on all 36 EEG pilot windows and meets runtime gate. Six approved redundant outputs removed; both symbolic variants replaced; Gaussian order 4 added: selection 200, starting from 205. Orders 16/32 disabled by EEG runtime gate. Separate amplitude template uses detrend=true, zscore=false; the subsequent systematic review of all 200 outputs selected 12 amplitude-sensitive outputs (see amplitude selection review). |
| Validation / goldens | 52 affected small/preprocessing/catalogue and 39 full checks passed; 10 integration/configuration checks passed. Eight intentionally changed numerical outputs refreshed at each size; unrelated arrays preserved, with provenance. |
| Timings / memory | Complete: comparable 118 × 1,024 full families, actual shared increments, native memory and per-window/×600/×41,400 projections; real EEG conditioning gate. |
| Documentation / CSV | Definitions, benchmark limitations and final selection recorded; brief verified CSV notes preserve unrelated records and user flags. |
| Workflow opportunities | Window parallelism, numerical-thread/JVM and whole-process memory measurements documented for a later goal. No production multiprocessing added. |

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
The original alternative workflow schema and all nine constructors were resolved after
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

The following history work and final verification complete this initial
foundation's remaining scope.

## Shared histories and information corrections

Fixed and MAX_CORR_AIS Gaussian TE, SI, CCE, DI and XME now use native aligned
moments/observations. The source-history count and DI conditional-information
formula are corrected across retained kernel and Kozachenko estimators too.
Gaussian outputs preserve nats and the half-log prediction-error convention;
retained kernel entropy outputs preserve bits. Definitions and singular-case
behavior are in `docs/calculation_definitions.md`.

The independent checks now include 39 Gaussian-history cases and 20 retained
Java-estimator cases. They cover direct OLS/determinants, independently selected
automatic embeddings, known lagged/instantaneous coupling, independent-source
DI, exact XME lag counts, sample alignment, voltage units, constants, singular
conditioning, permutation/direction, serialization and mutation/cache ownership.
One new Theiler test initially needed `str()` to convert a Java string before
`int()`; only that failed fixture was repeated after the correction.

The first EEG pilot (`audit/briefing/information_eeg_review.json`) revealed that
long-history covariance fits needed a stronger numerical guard: small Cholesky
diagonal entries alone missed ill-conditioned restricted models. The reciprocal
condition check now uses the existing factorization and routes unreliable fits
to direct least squares. On the failing 12-channel EEG window, order-32 NaNs
disappeared and inspected restricted/full residuals matched independent direct
fits. The initial long-history timings and values predate this fix and must not
serve as the final runtime/validity gate. Updated representative timings are
recorded below. No regularization or alternate scientific estimator was introduced.

Symbolic TE uses m=3, spacing/shift 4/4 and 16/16, with shared symbolization and
bounded categorical counts. All 36 twelve-contact EEG windows had finite,
varying outputs, all six marginal patterns, and no tied patterns. The full
118-contact pilot window also passed those checks. This proves feasibility of
the requested state space, not adequate joint sampling, absence of null bias,
optimal settings or EZ discrimination.

Final verification passed: 52 small/preprocessing/catalogue checks, 39 full
affected checks and 10 integration/configuration checks, including JVM-free
history/symbolic execution and the exact 200-output working inventory.
Only DI, XME k=10 and the two genuine symbolic replacements were numerically
regenerated at small/full sizes (eight outputs per size), after independent
formula tests. Other affected dependency/label fields changed with provenance;
all unrelated arrays and records were preserved. The original alternative nine-metric
template also executed on unequal-amplitude synthetic signals without a JVM:
all outputs were finite and JE/CE ceased to be affine functions of MI alone.
Only template paths were substituted for schema validation; no cohort run or
output directory was created.

## Information-family benchmark and symbolic controls

`tests/information_family_results.json` compares the committed Java wrappers at
`092f6e7` with the native implementation on the same seeded 118 × 1024 window,
one numerical thread, fresh Data per family, and the listed family order.
One complete historical pass took **719.07 s**; the replacement took **2.107 s**
(about **341× faster**). New serial projections are **21.07 minutes ×600** and
**24.23 hours ×41,400**. This is the entire reviewed information library family,
including automatic embedding and measures absent from the working run.

| Stage after preceding family calculations | Old increment | New increment |
| --- | ---: | ---: |
| Joint entropy, including shared moments | 6.722 s | 2.688 ms |
| Conditional entropy | 33.07 ms | 0.246 ms |
| Causal entropy | 363.80 s | 396.18 ms |
| Source-history entropy, k=1 | 25.79 s | 8.109 ms |
| Source-history entropy, k=10 | 33.27 s | 174.38 ms |
| Directed information after causal entropy | 10.57 s | 0.275 ms |
| Stochastic interaction | 76.17 s | 12.02 ms |
| MI after the shared entropy calculations | 3.627 s | 0.186 ms |
| Lag-one MI | 3.191 s | 1.295 ms |
| Automatic Gaussian embedding / TE | 167.81 s | 1322.74 ms |
| Gaussian TE, order 1 | 6.240 s | 11.83 ms |
| Symbolic spacing/shift 4/4 replacement | 6.319 s | 90.51 ms |
| Symbolic spacing/shift 16/16 replacement | 15.53 s | 86.14 ms |

These are actual stage increments, not independently cold timings. In
particular, causal entropy already computes the native DI terms. Unchanged
definitions differ by at most **6.22e-15**, including automatic embedding.
DI, k=10 source-history entropy and both symbolic outputs intentionally changed;
their correctness comes from independent formulas/counts and models, not old
output agreement. No old full-family pass was repeated after completion.
The benchmark now checkpoints completed historical stages and shared Data so
an interrupted pass can resume without repeating finished fits.

Separately measured native retained cache: **255.65 MiB**, plus **0.92 MiB**
input snapshot; tracemalloc peak **265.22 MiB**. This is Python/NumPy allocation
accounting, not whole-process RSS or Java/native-library heap usage. Imports
took 0.904 s and JVM startup 0.176 s separately; preprocessing and I/O are
excluded. A single old pass has no estimated timing dispersion. Load and
patient-specific conditioning limit serial cohort projections.

`audit/briefing/symbolic_null_review.json` uses eight seeded AR(1), coefficient
0.9, controls per setting and sample length. At 1024 samples, mean independent
TE is **0.0321 / 0.0585 nats** for spacing 4/16; mean deliberately coupled TE is
**0.784 / 0.772**, with reverse estimates **0.0326 / 0.0492**. At 8192 samples,
independent means fall to **0.00415 / 0.00644** while coupled means remain
**0.785 / 0.780**. Longer synthetic records characterize sampling bias only;
the run retains 1024-sample windows. This supports the approved candidates'
nondegeneracy and directional sensitivity, not patient-specific significance or
EZ discrimination. Both genuine symbolic variants replace their two misleading
Gaussian-dispatch entries in the working config without changing its count.

## Selected information family and additional-history gate

`tests/information_selection_results.json` measures the original selected
Gaussian/incorrect-symbolic family together, versus the approved replacement
including order 4. The original family took **42.17 s/window**; the replacement
took **0.292 s/window**, about **144× faster**. New serial projections are
**2.92 minutes for 600 windows** and **3.36 hours for 600×69**. These are this
information subset's costs, not the total runtime of all 200 selected metrics.
The benchmark excludes imports, preprocessing and I/O and uses one numerical
thread on the same synthetic 118 × 1024 data. The original JE/CE were included
in the old workload and removed from the new workload by the approved
redundancy decision. Corrected symbolic definitions also intentionally differ.

| New selected stage | Cold-family increment | Increment after retained regressions |
| --- | ---: | ---: |
| Gaussian MI | 2.67 ms | 1.32 ms |
| Lag-one Gaussian MI | 3.66 ms | 3.13 ms |
| Gaussian TE, order 1 | 15.03 ms | 14.12 ms |
| Gaussian TE, order 4 | 105.11 ms | 103.66 ms |
| Symbolic spacing/shift 4/4 | 84.68 ms | 86.83 ms |
| Symbolic spacing/shift 16/16 | 81.00 ms | 82.39 ms |

The second column is the actual original/new comparison pass; the third is
the median of five native-only passes in
`tests/information_regression_increment_results.json`, after Lasso alpha=0.1,
ElasticNet, BayesianRidge and RECI have populated shared moments. No old
regression or information fits were repeated for that incremental experiment.
The information increment's median is **0.292 s** (2.92 min / 3.36 h at the
two projections). The combined retained regression plus information workload
has median **0.301 s** (**3.01 min ×600 / 3.47 h ×41,400**). The gain from
cross-family sharing is modest compared with eliminating pairwise estimators;
do not multiply individual speedups. The selected information cache retains
**25.23 MiB** plus the 0.92 MiB snapshot, with cold tracemalloc peak **33.95 MiB**.
The actual combined regression/information derived cache is **29.61 MiB**.

The post-fix EEG check is
`audit/briefing/information_eeg_conditioning_review.json`. All Gaussian candidate
outputs were finite in all 36 twelve-contact windows, across patients 001,
041 and 096, and in the full 118-contact window. The latter took:

| Added/history setting | One full EEG window | ×600 | ×41,400 | Run decision |
| --- | ---: | ---: | ---: | --- |
| Order 1 (already selected) | 0.0187 s | 11.21 s | 12.89 min | Keep |
| Order 4 | 0.1097 s | 65.80 s | 1.26 h | Add |
| Order 16 | 11.65 s | 1.94 h | 134.0 h | Disabled |
| Order 32 | 31.87 s | 5.31 h | 366.5 h | Disabled |

The three-repeat synthetic candidate receipt is
`tests/information_candidate_results.json`: order-4/16/32 increments were
0.100/0.590/2.026 s. Stable direct fitting on smooth real EEG is materially
more expensive at longer histories. These projections assume the single full
pilot window's cost throughout; they do not estimate cohort heterogeneity or
parallel throughput. Runtime alone rules out 16/32 under the approved gate;
their library implementations remain available and correctly defined.

Working selection at this goal's completion: **205 → 200 outputs**. Remove six redundant outputs
(ordinary/squared Euclidean barycenter means, original Ridge, exact SGD/Ridge
replacement, Gaussian JE and CE). Replace default Lasso with alpha=0.1 and
both mislabeled symbolic entries with genuine m=3 variants (no count change).
Add one Gaussian order-4 output. Earlier basic/phase changes remain selected;
group delay and the expensive directed spectral additions remain absent.
At goal completion, the separate amplitude/reference template had nine metrics.
The subsequent systematic review of all 200 main-run outputs selected 12 for
the amplitude run; see [the complete review](amplitude_selection_review.md).
That later review also removed five normalized redundancies, choosing
representatives by independent runtime, leaving **195** in the main selection.
It retains `detrend: true`, `zscore: false`, CAR and 0.5-second nonoverlapping
windows. The earlier nine-output validation receipt remains a historical record.

## Subsequent workflow goal

Measure independent-window parallelism only after the serial calculation work
is settled. Use one BLAS/OpenMP thread per worker initially and test worker
counts against throughput, rather than multiplying serial speedups by core
count. Each window owns its mutation-checked input snapshot and derived caches;
the 256 MiB moment/history limit is per window, not a whole-process memory cap.
Spectral and cross-correlation caches, temporary arrays, recording buffers,
imports and any retained JVM heap add to it. Measure peak process RSS under
the complete selected family before choosing the number of workers.

Native Gaussian/symbolic runs no longer need Java startup. If another selected
metric needs Java, initialize it safely inside a worker; do not fork an active
JVM. Keep output writes coordinated and record the actual preprocessing/window
order. These are follow-up measurements, not production multiprocessing changes
in this goal.
