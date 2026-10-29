# Final workflow and lightweight pairwise review

This implements the approved final-workflow goal. This document tracks completed
work and outstanding gates; it is not a declaration that the goal is complete.

## Requirements and evidence

| Requirement | Evidence / remaining work |
|---|---|
| Five inexpensive additions | Implemented in `statistics/basic.py` and `statistics/distance.py`; independent tests in `test_lightweight_candidates.py`. Run selections still to be updated. |
| Histogram and envelope optimization | Implemented and measured; receipt and limits below. |
| JS redundancy across patients/references and temporal summaries | Pending representative comparison; retain provisionally. |
| Downstream Pearson / absolute Pearson / consecutive graph changes | Pending API/documentation and tests. |
| All 33 source outputs plus partial correlation accounted for | Source inventory exists in `additional_pairwise_review.md`; final coverage table/config selections pending. |
| Broadband local and limited shaft-average combinations | Pending selection and metadata-resolved shaft-average support. |
| Group configuration / bounded shared continuous preparation | Pending implementation and comparison against separate jobs. |
| Output/reference/preprocessing contracts | Existing contracts retained; new metrics have focused serial/spawn/resume coverage. Group isolation still to be validated. |
| Final workflow timings, larger recording, complete process memory and tuning | Pending. Kernel receipts below are insufficient for this requirement. |
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
