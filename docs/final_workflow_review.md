# Final workflow and lightweight pairwise review

This implements the approved final-workflow goal. This document tracks completed
work and outstanding gates; it is not a declaration that the goal is complete.

The revised goal additionally requires **3–5 complete patients through the final
workflow with float32 outputs** after implementation/validation. This is a required
completion gate, not satisfied by the bounded profiling windows.

## Requirements and evidence

| Requirement | Evidence / remaining work |
|---|---|
| Five inexpensive additions | Implemented in `statistics/basic.py` and `statistics/distance.py`; independent tests in `test_lightweight_candidates.py`. Run selections still to be updated. |
| Histogram and envelope optimization | Implemented and measured; receipt and limits below. |
| JS redundancy across patients/references and temporal summaries | Completed three-patient/three-reference pilot below; omit default selection under the approved empirical criterion. |
| Downstream Pearson / absolute Pearson / consecutive graph changes | Bounded `workflow.derived` iterator and ten focused tests; formulas, identity and temporal-summary guidance below. |
| All 33 source outputs plus partial correlation accounted for | Source inventory exists in `additional_pairwise_review.md`; final coverage table/config selections pending. |
| Broadband local and limited shaft-average combinations | Automatic shaft membership implemented/tested; final selections pending. |
| Group configuration / bounded shared continuous preparation | Pending implementation and comparison against separate jobs. |
| Output/reference/preprocessing contracts | Existing contracts retained; new metrics have focused serial/spawn/resume coverage. Group isolation still to be validated. |
| Final workflow timings, larger recording, complete process memory and tuning | Pending. Kernel receipts below are insufficient for this requirement. |
| Final full-patient runs | Pending 3–5 complete patients, all selected final groups, float32 outputs and completion/availability audit. |
| Final configs/counts, brief CSV notes, review and chunked commits | In progress. No push authorized. |

## New calculation definitions

All outputs have a missing diagonal, owned return arrays and source rows / target
columns. Caches use the existing mutation-checked Data snapshot. No new estimator
internally detrends or rereferences its input.

- `FixedLagCorrelation(lag_samples=20, fs=2048)`: Pearson correlation between
  source samples `[:-20]` and target samples `[20:]`. Each overlapping segment
  is centered and normalized separately. Positive lag means the source precedes
  the target. Twenty samples are exactly 9.765625 ms at 2048 Hz, not exactly
  10 ms. The workflow binds the sampling rate and rejects conflicting `fs`.
  At least two overlapping observations are required. A constant/nonfinite
  segment produces a diagnosed NaN. Correlation does not establish causality.
- `ThresholdEventExcess(threshold=3)`: for each channel, sample-standardize the
  supplied window (`ddof=1`), set `E(t)=1` when `abs(z(t))>3`, and return
  `mean(E_i E_j)-mean(E_i)mean(E_j)`. This is signed indicator covariance,
  dimensionless, with zero valid when there are no threshold events. It is an
  event proxy, not a clinical spike detector. Constant or nonfinite input is
  undefined. Re-standardization gives the same definition for either workflow
  normalization setting; it does not internally detrend.
- `LogScaleRatio`: `log(std(target)/std(source))`, using natural logarithms.
  Its selected input must be detrended, unnormalized voltage; z-scoring removes
  the intended information. Sample/population standard deviations give the
  same ratio for equal window lengths. Positive means the target has greater
  scale; the matrix is antisymmetric, not causal. Log-variance arithmetic avoids
  squaring extreme units or an arbitrary amplitude floor. Zero variance and
  nonfinite input produce diagnosed NaNs rather than artificial finite ratios.
- `PartialCorrelation`: `-P[i,j]/sqrt(P[i,i]*P[j,j])` from a LedoitWolf fit to
  all supplied channels. It shares that exact fit with covariance/precision,
  retaining its diagonal for normalization. This is a regularized conditional
  association given the other supplied channels, unlike a separate two-channel
  fit. The conditioning set is part of its interpretation. A nonfinite channel
  makes the whole estimate unavailable; it is not silently dropped. Constant
  channel pairs and numerical failures have explicit reasons. Units are
  dimensionless and results lie in [-1,1] up to endpoint roundoff.
- `HistogramJensenShannon(bins=10, pseudocount=1e-10)`: equal-weight
  `0.5*KL(p_i || m)+0.5*KL(p_j || m)`, `m=(p_i+p_j)/2`, natural logs, with no
  square root. Bounds are [0, log(2)]. It shares independent-range channel
  histogram probabilities with existing relative entropy. This compares marginal
  distribution shapes, not temporal coupling. Constants/nonfinite channels are
  undefined as in relative entropy.

The top_features source uses destination-first lag output, different preprocessing
order and an arbitrary tiny scale floor. Production uses the approved source-first
orientation, detrend-before-zscore workflow and existing undefined-input policy.
The source's different filters, trimming and local reference are not adopted.

## Production family measurements

[Reproducer](../tests/final_workflow_review/benchmark_families.py) and
[receipt](../tests/final_workflow_review/production_family_timings.json) compare
against commit `99ed77d`. Three real 118-channel x 1024-sample CAR windows from
patient 001, one numerical thread; each run includes Data construction, cache
checks and diagnostics. Preparation, imports/JIT loading and storage are excluded.
The cohort columns repeat one representative workload sequentially; they are not
measured elapsed cohort times or full-process memory estimates.

| Workload | Before/window | After/window | After x600 | After x600x69 |
|---|---:|---:|---:|---:|
| Orthogonalized amplitude envelope, signed or absolute | 738 ms | 125 ms | 75 s | 1.43 h |
| Complete six-envelope family | 750 ms | 365 ms | 219 s | 4.20 h |
| Relative entropy standalone | 11.0 ms | 3.37 ms | 2.02 s | 2.32 min |
| Relative entropy + JS | — | 5.69 ms | 3.41 s | 3.92 min |
| Fixed-lag correlation | — | 3.99 ms | 2.39 s | 2.75 min |
| Event excess | — | 4.49 ms | 2.70 s | 3.10 min |
| Partial correlation including LedoitWolf fit | — | 6.87 ms | 4.12 s | 4.74 min |
| Log scale ratio, unnormalized | — | 2.07 ms | 1.24 s | 1.43 min |
| Lag + event + partial correlation + RE + JS | — | 21.1 ms | 12.6 s | 14.5 min |
| Actual partial + JS increment after precision + RE | — | 4.78 ms | 2.87 s | 3.30 min |

Envelope speedups are about 5.9x for amplitude-only orthogonalization and 2.1x for
the complete family. The compiled kernel retains the propagated floating-point
resolution checks and reason codes, uses O(samples) scratch arrays, and computes
only the requested amplitude/log transform. Both signs and absolute summaries
reuse directional results; Hilbert transforms remain shared across the family.
Projection arithmetic is repeated if both amplitude and log variants are requested;
this is included in the complete-family timing, not omitted from a speedup claim.
There is no unsafe fast-math mode or sample/pair approximation.

Histogram counts now use a compiled search with the same half-open/final-closed
bin edges and existing extreme-scale minmax transformation. Log pseudocount
arithmetic is retained, including subnormal/very large positive pseudocounts.
Counts/probabilities/logs share the existing bounded moment cache. The reported
increment includes production diagnostics and cache checks; it should not be
compared directly to the earlier bare-kernel prototype.

Independent scalar formulas and numerical guards are covered by focused tests.
The benchmark also checks original/new values and diagnostics for ordinary,
proportional, near-proportional, constant, zero and nonfinite envelope inputs.
No historical golden was changed to make these checks pass.

Validation for this chunk: 20 new formula/cache/alignment tests, 26 existing
candidate regressions, 50 envelope tests, 12 banded-PEC preprocessing cases,
and one serial/spawn/float32/resume integration test passed. The overlap-only
invalid-input diagnostic and partial-correlation minimum-observation preflight
guard were refined after the timing receipt; finite-data formulas are unchanged.
Final workflow timings remain outstanding.

## Downstream derived features

`pyspi.workflow.derived.iter_derived_windows` reads saved per-window outputs in
bounded batches. It does not read EEG, write over raw outputs, or average over
time. Each batch carries its run path, recording/reference/preprocessing identity,
input metric, original window indices and pairwise validity mask.

```python
from pyspi.workflow.derived import iter_derived_windows

for batch in iter_derived_windows(
    run_path, "pdist_cosine", transform="absolute_pearson",
    absolute_change=True, batch_windows=16,
):
    # batch.values: abs(abs(r[t]) - abs(r[t-1])), float64
    # batch.valid: both values available, finite, successful, and adjacent
    # batch.window_indices: original positions, never compressed across gaps
    pass  # downstream analysis/persistence, not part of the raw EEG run
```

`transform="pearson"` computes `1-cosine_distance`; `absolute_pearson` applies
absolute value per window. These automatic derivations require recorded z-scoring,
which guarantees centered, nonzero-variance accepted channels. Detrending-only
input can also satisfy the algebraic identity, but cosine outputs alone cannot
prove a nonzero centered norm; the helper rejects that unverified case.
`transform="identity"` supports changes of any saved metric, including RE and PEC.
No change is defined at the first selected window, a missing predecessor, a
failed/unfinished window, an unavailable contact or a nonfinite numerical value.
Batch boundaries preserve the previous window; selection gaps never bridge it.

For later temporal summaries, accumulate `where(batch.valid, batch.values, 0)`
in float64 and divide by per-edge valid counts, leaving zero-count entries NaN.
Average signed correlations, absolute correlations and their absolute changes
separately: `mean(abs(r))` is generally not `abs(mean(r))`. Do not combine references,
preprocessing identities or different recordings into a single adjacent-window
sequence. Ten focused tests cover transformations, gap/failure handling, metadata
checks and bounded reads; workflow groups will receive integration coverage later.

## Redundancy and source coverage

[The histogram redundancy receipt](../tests/final_workflow_review/histogram_redundancy.json)
uses three windows each from current included patients 001 (118 contacts), 006
(80), and 111 (249), under CAR, adaptive local, and shaft-average references.
Per-window upper-triangle RE/JS rank correlations were 0.9922–0.9969; correlations
of per-edge temporal means were 0.9902–0.9942. Temporal-mean Pearson correlations
were 0.9844–0.9922. Residuals after a fitted linear relationship were still
12–18% of JS's standard deviation: these are not algebraically identical metrics.
Following the approved approximate-redundancy criterion, JS is omitted from the
**default selection**, while its implementation remains available. The pilot does
not establish redundancy for every patient, unusual distribution, or summary.
Patient 004 from the historical cache is excluded in the current data-lake
metadata (no labeled data), so it was replaced by 006, not silently reintroduced.

[The coverage inventory](../tests/final_workflow_review/top_features_coverage.csv)
accounts for all 33 new exploration outputs: 7 CAR, 13 local, 13 shaft. It also
records the three stored baseline controls and the separately suggested partial
correlation. Local/shaft Pearson, absolute Pearson and graph changes are derived
from per-window saved graphs. The source's 20-sample “10 ms” label is corrected
to its exact duration; the intended lag stays 20 samples. Broadband local/shaft
envelope and amplitude log-ratio groups will be selected explicitly without
multiplying the entire main selection over references. Configuration integration
is still pending; the inventory marks that status.

`reference: {method: shaft_average, shafts: metadata}` resolves the authoritative
patient `electrodegroup` membership automatically. It preserves contact axes,
subtracts each shaft's retained-contact mean, and records the resolved membership
and coefficients. It has the existing shaft-mean definition: a singleton shaft
is unavailable; two-contact shafts yield opposite signals. It does not inherit
adaptive-local CAR fallbacks. Explicit shaft layouts retain their behavior.
Three new tests and the 13 affected adaptive-reference tests passed.

Source candidate ledger SHA256: `b9e15f442ffa8e054103c899a5424dedb096242230c95b2fcea1c709f845579e`.
