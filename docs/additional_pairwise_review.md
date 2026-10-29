# Additional pairwise candidate review

The approved implementation is tracked in [final_workflow_review.md](final_workflow_review.md).
The section below records the earlier source review and prototype evidence; its
"not implemented" statuses describe that review date, not the current worktree.

## Subsequent top_features and workflow-sharing review

After the adaptive-local reference was approved, the source review was extended
to `top_features/further_feature_exploration`, not just its initial recommendations.
Its 33 candidates comprise reference combinations and temporal summaries, not 33
different estimators. The complete newer report is
`further_feature_exploration/reports/batches_1_2/report.md` in that source tree.

Current histogram relative entropy matches its core: ten independent-range bins,
1e-10 pseudocount, natural logarithms and the maximum of directional KL values.
Comparison with the exact extracted prototype function on three cached real
118-by-1024 CAR windows differed by at most 2.89e-15 off diagonal. Production
keeps diagonals and undefined constant/nonfinite channels missing. It is enabled
in both candidate selections. This does not reproduce the source's full pipeline:
the newer source uses Butterworth 0.5–300 Hz filtering, different local-reference
rules, boundary trimming and z-score-before-detrend preprocessing. Current
workflow definitions remain those separately approved here.

| Candidate from the source | Current state / next treatment |
|---|---|
| Relative entropy | Implemented and selected for CAR/custom local. |
| Ordinary and absolute orthogonalized envelope correlation | Implemented. Main CAR selection includes broadband variants; local candidate selection currently uses 30–70 Hz variants. Broadband local/shaft requests would be additional reference/band combinations. |
| Beta/high-gamma coupling | Implemented and selected for CAR/custom local. Shaft reference is an additional combination. |
| Pearson / absolute Pearson | Derive from stored per-window cosine distance when centered: `r=1-distance`, then `abs(r)`. Different references still require their own calculation. Apply absolute value before temporal aggregation. |
| Consecutive graph changes | Downstream derivation from raw per-window graphs; require genuinely adjacent windows and two valid values. No new raw EEG estimator needed. |
| Fixed 20-sample lag correlation | Not implemented as a dedicated output; not equivalent to the selected max/mean/min cross-correlation summaries. Source uses destination-first orientation; any production addition must use documented source-first orientation. |
| Threshold-event excess | Not implemented. Covariance of absolute standardized threshold exceedances above 3; not a clinical spike detector. |
| Pairwise log scale ratio | Not implemented. Needs detrended, unnormalized voltage. Marginal amplitude contrast, not temporal coupling. |
| Shrinkage partial correlation | Suggested separately; source's final 33 omit it. Current precision is not partial correlation. Normalize before discarding the fitted diagonal; cannot reconstruct from mean saved off-diagonal precision alone. |
| Jensen–Shannon divergence | Not implemented. Very cheap shared histogram addition; source reports 0.9967 summary rank correlation with CAR relative entropy, an empirical near-redundancy rather than an algebraic identity. |
| Shaft-average combinations | Existing explicit-layout reference, not an additional estimator. Automatic metadata resolution and a small selected profile would be needed for convenient cohort use. |

The [bounded probe receipt](../tests/final_workflow_review/top_feature_probe.json)
and [reproducer](../tests/final_workflow_review/probe_top_features.py) compare
three existing real EEG windows. One numerical thread; medians below exclude
preparation, imports, JIT warmup, output and production diagnostics. Cohort
columns repeat the representative 118-channel cost for 600×69 windows and are
not measured cohort elapsed times.

| Prototype work | Window | ×600 | ×600×69 |
|---|---:|---:|---:|
| Fixed lag 20 | 1.52 ms | 0.91 s | 1.05 min |
| Event excess | 0.72 ms | 0.43 s | 0.50 min |
| Log scale ratio after detrending | 0.41 ms | 0.24 s | 0.28 min |
| LedoitWolf partial correlation, including fit | 6.27 ms | 3.76 s | 4.33 min |
| Partial correlation after shared fit | 0.061 ms | 0.037 s | 0.042 min |
| Jensen–Shannon after shared histograms | 1.15 ms | 0.69 s | 0.79 min |

The source's compiled amplitude-only orthogonal-envelope kernel took 23.9 ms
versus 742.5 ms for the current workspace path. The current path also computes
log variants and implements numerical-degeneracy safeguards absent from the
prototype. Last-window finite outputs differed by 2.66e-15, but that does not
validate edge cases or justify dropping the safeguards. This is a concrete
remaining optimization investigation, not a delivered 31× family speedup.

The source's combined compiled RE/JS kernel took 2.41 ms; current RE including
Data construction took 10.93 ms. This comparison also differs in included work;
histogram bin boundaries, extreme scales and undefined-input behavior need
preservation before replacing production code.

For workflow sharing, prefer one experiment description with global defaults
and named groups of metrics sharing reference/preprocessing. A one-metric group
serves as an override; multiple groups request multiple references. Resolve
these into ordinary homogeneous Calculator/worker runs, preserving the existing
output/resume contracts. Share compatible continuous reading/filtering/resampling
before reference branches and share compatible band preparation. Avoid teaching
every estimator about references or reusing result caches across differently
preprocessed input. Separate existing configs should remain usable. This is the
proposed next scope, not an implemented configuration schema.

New additions should be assessed by runtime and meaningful redundancy; weak or
uncertain predictive evidence alone is not an exclusion criterion. No additional
estimators or reference combinations were enabled during this source review.

This implements the approved ten-part additional-candidate goal. Established
188-output normalized and 12-output amplitude selections remain separate.
The target is roughly one day per measure for 72 people with 600 half-second
windows each, not one day for the complete panel. Projection assumptions and
reference costs must be reported explicitly.

The completed bounded EEG assessment, timings and candidate selections are in
[the pilot report](additional_pairwise_pilot.md). Kernel MI is enabled for CAR
by user choice: ten CAR outputs and nine Laplacian outputs. The eleven-output
pilot stays available; no definition or runtime decisions remain pending.

## Completion ledger

| Requirement | State / evidence |
| --- | --- |
| Histogram relative entropy | Implemented; explicit KL, invariance, invalid-input and cache checks pass. |
| Continuous 30–70 Hz PEC views | Implemented with shared bounded filtering, file-backed views, exact PEC preprocessing, and serial/parallel checks. |
| Kernel MI, IDS, Gaussian DI/SI assessment | IDS cleanup validated; all three assessed on three patients/two references with controls and timing receipts. |
| Cross-channel 16–30 / 70–150 Hz coupling | Prototype integrated and validated; matched surrogates show a substantial positive null baseline; retained as exploratory. |
| Exact bounded distance correlation | Implemented and independently validated; three-patient standalone/shared costs, fast-library comparison and comparative tracked allocations recorded. |
| Exact radius-one DTW | Implemented and validated; pilot rankings closely track cosine distance, so absent from the selected CAR/Laplacian configs. |
| Matched CAR / contact-centered Laplacian | Implemented; three-patient matched comparison and shared-reference controls complete. Historical coordinate validation flagged patient 090. The later [custom local reference](adaptive_local_reference.md) uses contact numbering instead. |
| Representative EEG runtime / memory / workflow costs | Three sizes, both references, actual nine/eleven-output family costs, 12-window worker profiles, PSS and cohort projections recorded. |
| Novelty, coverage and surrogate assessment | Descriptive baseline and surrogate checks complete; no new predictive-value claim or cohort model search. |
| Candidate configs, docs, CSV, regressions and commits | Ten CAR / nine Laplacian outputs prepared and checked; kernel MI enabled for CAR by user choice. Main 188/12 selections unchanged. |

## Definitions being implemented

Histogram relative entropy uses ten equal-width bins per channel's own range,
adds 1e-10 to each count, normalizes to probabilities, and returns the larger of
the two directional KL divergences in nats. It compares marginal distribution
shapes, not temporal coupling or absolute amplitude. Positive affine scaling
and independent time permutations leave it invariant apart from floating-point
bin-boundary effects. Constant/nonfinite channels and diagonal entries are
missing, with reasons. The pseudocount is regularization, not bias correction.

Distance correlation retains the existing two conventions: the bias-corrected
U-centered normalized coefficient can be negative and is on the squared scale;
the biased double-centered version takes the square root and is nonnegative.
Both reuse raw sample-distance products and channel row sums. Sample-distance
blocks are bounded; no full sample-distance matrix is retained. Constants
retain the existing zero-coefficient convention; insufficient observations or
nonfinite input remain missing. Rare cancellation is resolved using explicitly
centered blocks. Returned arrays are owned copies and caches detect mutation.

IDS keeps six terms, max reduction and bandwidth 0.5 by default, without an
extra internal normalization. Nondefault settings receive distinct identifiers.
Invalid settings fail explicitly; nonfinite/constant channels are missing.
Importing the IDS implementation no longer reseeds NumPy's global RNG. The
finite transformation family does not detect all possible dependence, and its
positive, autocorrelation-sensitive null is not a significance threshold.

Sakoe–Chiba DTW retains radius one by default, uses a rolling band, and evaluates
each unordered pair once. The library accepts an explicit nonnegative radius
and includes nondefault radii in identifiers; no wider radius has been selected
for the run. One sample is 0.48828125 ms at 2048 Hz. Unconstrained DTW, Itakura,
SoftDTW and other distance algorithms are unchanged. The exact compiled kernel
has a process startup/cache-loading cost outside warm calculation measurements.

## First production formula benchmark

[Receipt](../tests/pairwise_candidate_math_results.json), reproduced with
`tests/benchmark_pairwise_candidates.py --input <prepared-window.npy> --out <receipt.json>`.
One numerical thread; patient 001's already prepared CAR/detrended/z-scored
118-by-1024 window. Three native passes with fresh Data/cache per standalone
metric. Imports, JIT warmup, continuous preprocessing and output are excluded.
The matching historical algorithm comparisons use eight channels and one
baseline pass, rather than repeating expensive full-size baseline calculations.

| Standalone output | Seconds/window | 600 windows | 600 x 72 windows |
| --- | ---: | ---: | ---: |
| Histogram relative entropy | 0.0121 | 7.25 s | 8.71 min |
| Bias-corrected distance correlation | 1.023 | 10.23 min | 12.27 h |
| Biased distance correlation | 1.003 | 10.03 min | 12.03 h |
| Radius-one DTW | 0.1272 | 1.27 min | 1.53 h |
| IDS | 0.0331 | 19.89 s | 23.86 min |

Both distance-correlation variants together took 0.984 seconds in a separate
sharing measurement, including the first result and a 0.000253-second actual
second increment. Their retained shared arrays occupy 1.13 MiB, apart from the
common input snapshot. Raw distance blocks are 16 MiB; several temporaries can
coexist. The whole benchmark process peaked at 542 MiB RSS, including imported
libraries, JIT/reference infrastructure and earlier trials; this is not a
per-metric memory attribution or a complete workflow worker-tree measurement.

At eight EEG channels, distance correlation changed from 3.923 to 0.0620 seconds
(63.2x); maximum absolute difference was 5.54e-14. Constrained DTW changed from
0.0917 to 0.000595 seconds (154x), with exact agreement on that comparison.
These gains are measured at eight channels and must not be presented as
measured 118-channel speedups. Cohort costs in this first receipt are serial
projections from one window; the subsequent pilot report measures patient/channel
variation and workflow costs.

## Focused validation so far

- 38 initial candidate/IDS-reference checks passed, including independent
  U/V-centering, tied/constant inputs, extreme units, parameter validation,
  channel permutation, output ownership, mutation and process import RNG state.
- 12 DTW/small-regression checks passed: eight new DTW checks plus the four
  affected existing output goldens.
- Four affected full-size output goldens passed.
- Eight affected preprocessing goldens plus the catalogue metadata check passed.
- Seven histogram checks passed after adding logarithmic pseudocount arithmetic,
  including finite positive pseudocounts at subnormal and very large scales.

Numerical golden arrays remain unchanged. Only the bias-corrected distance
correlation's signed label is updated in golden metadata, with source provenance.
No old passing unrelated suite has been rerun.

## Reference workflow validation

Strict Laplacian layouts retain missing physical neighbors in the ordered shaft
list. Endpoints/gaps are explicitly unavailable, without fallback; output axes
retain their original contact names. CAR and shaft references record linear
coefficients, shared sources, proportional rows, and the equal-variance independent
source null correlation. These geometry diagnostics are not empirical evidence
of connectivity. Shaft-average support is available for exploratory comparison;
no third production reference is selected after the exploratory PAC comparison.

Ten geometry/config checks passed. Nine preparation/execution checks passed,
covering the direct continuous Laplacian formula, serial and spawned masks with
normalization disabled, usable-channel calibration versus full-size output,
three unchanged CAR pipeline comparisons, and real/synthetic calibration
regressions. No numerical goldens changed.
Two additional focused calibration CLI/validity checks passed.
Local references use direct neighbor arithmetic and one shared mean per shaft,
avoiding dense channel-by-channel mixing of an entire recording. Two independent
linear-operator comparisons and the affected continuous-preparation check passed
after this optimization; no runtime gain is claimed before measurement.

## Band candidate definitions and first EEG timing

`pec_band-30-70Hz` and `pec_orth_abs_band-30-70Hz` use the corrected PEC
definitions on continuously filtered 30–70 Hz signals. Window detrending and
z-scoring follow the workflow flags, then the existing shared analytic/envelope
workspace computes the coefficients and diagnostics. Broadband PEC remains
separate. A band with an unusable channel does not remove that channel from
other metrics or poison healthy band pairs.

`pac_phase-16-30Hz_amp-70-150Hz` starts from the top_features prototype. If
`u_i(t) = exp(i*phase_i(t))` and `a_j(t)` is the target amplitude minus its
window mean, the output is `abs(sum(u_i*a_j))/(sqrt(T)*norm(a_j))`.
Rows are phase sources and columns amplitude targets, transposing the old
prototype's orientation. This is a bounded association, not a causal estimator
or a Tort modulation index. Zero/unresolved phase and constant/unresolved
amplitude return NaN with reasons instead of zero. Diagonals remain missing.
The analytic transforms use the same next-fast-length FFT convention as PEC;
at the approved 1,024 samples this is exactly the prototype's FFT length.

PAC applies no extra per-window detrending/z-scoring to the filtered band
samples, preserving the prototype definition. PEC does apply those flags.
Consequently transforms are shared only for matching band and preprocessing
choices. Continuous bands use the existing workflow's MNE FIR settings,
including its prior notches and broad filter; the historical exploration used
Butterworth filters. Historical predictive evidence is therefore motivation,
not validation of these current outputs. Half-second beta windows contain only
8–15 cycles; magnitude estimates can have autocorrelation-sensitive null bias.
Shared references, nonsinusoidal waveforms and transients can also produce
cross-frequency association. Completed surrogate checks and their limits are in the pilot report.

Band inputs are owned window copies, with explicit missing-band errors and
invalidation when the main Data changes. Continuous views are stored in temporary
files, filtered in channel blocks of about 16 MiB (at least one full channel),
and mapped by spawned workers without per-worker recording copies. Time is
never blocked for filtering. Scratch should use a disk filesystem, not tmpfs.

[First band receipt](../tests/band_candidate_001_results.json): patient 001,
118 channels, 1,024 samples, windows 0/304/608, one numerical thread. Three
distinct EEG windows per standalone output; values below are medians, not
repeated same-window microbenchmarks. Data construction, band window copies and
lazy per-metric preprocessing are included; continuous preparation and output
writing are excluded from the projected calculation times.

| Output | Seconds/window | 600 windows | 600 × 72 windows |
| --- | ---: | ---: | ---: |
| Broadband ordinary PEC (comparison) | 0.00926 | 5.55 s | 6.67 min |
| Broadband absolute orthogonalized PEC (comparison) | 0.692 | 6.92 min | 8.30 h |
| 30–70 Hz ordinary PEC | 0.0131 | 7.85 s | 9.42 min |
| 30–70 Hz absolute orthogonalized PEC | 0.715 | 7.15 min | 8.58 h |
| Beta/high-gamma PAC | 0.0163 | 9.77 s | 11.73 min |

All five together took a median 1.471 seconds/window, including common input
setup. The receipt records actual increments in that order; it does not infer
sharing gains by subtracting unrelated standalone timings. Every off-diagonal
pair was finite in these three windows. This does not establish predictive value.

Continuous preparation took 26.22 seconds for the complete recording; the three
extra band filters and their file writes accounted for 14.80 seconds. Band files
occupied 1.65 GiB; process peak RSS, including preparation/imports/all candidates,
was 1.32 GiB. It is a single-process high-water mark, not a worker-tree memory
measurement. The subsequent multi-patient, reference-specific, output and parallel throughput
measurements are recorded in the pilot report.

The separate [11-output pilot selection](../pyspi/workflow/additional_candidate_pilot.yaml)
contains all candidates for bounded evaluation, including secondary radius-one
DTW. It is not a full-cohort inclusion decision. Established selections remain
188 normalized outputs and 12 amplitude outputs.

Focused validation covers 25 band unit cases (independent PAC formula/known
modulation, all PEC preprocessing flags, ownership, permutation, extreme units,
invalid inputs and matching-transform sharing), continuous-filter comparisons,
serial/spawned masked outputs, calibration, and file-backed block equivalence.
The six affected broadband PEC goldens and relevant cache/window regressions
passed; no numerical goldens were changed.

## Prior evidence

The source exploration is
`/run/media/dan/DansFlash/ram_needs_rma_git/top_features`. Its full 33-candidate
report includes 40 primary and 29 secondary patients, historical 185-feature
comparisons, and a partly different preprocessing pipeline. Results motivate
candidates; they do not establish added predictive value over the corrected
current selection. The initial 118-channel EEG feasibility screen is recorded
in `/tmp/pyspi_additional_pairwise_screen.json`; production benchmarks and
validation in the pilot report supersede those prototype-only runtime receipts.
