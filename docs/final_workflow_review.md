# Final workflow and lightweight pairwise review

This implements the approved final-workflow goal. This document tracks completed
work and outstanding gates; it is not a declaration that the goal is complete.

The revised goal additionally requires **3–5 complete patients through the final
workflow with float32 outputs** after implementation/validation. This is a required
completion gate, not satisfied by the bounded profiling windows.

## Requirements and evidence

| Requirement | Evidence / remaining work |
|---|---|
| Five inexpensive additions | Implemented in `statistics/basic.py` and `statistics/distance.py`; independent tests in `test_lightweight_candidates.py`. Selected in the final groups below (JS omitted under the redundancy decision). |
| Histogram and envelope optimization | Implemented and measured; receipt and limits below. |
| JS redundancy across patients/references and temporal summaries | Completed three-patient/three-reference pilot below; omit default selection under the approved empirical criterion. |
| Downstream Pearson / absolute Pearson / consecutive graph changes | Bounded `workflow.derived` iterator and ten focused tests; formulas, identity and temporal-summary guidance below. |
| All 33 source outputs plus partial correlation accounted for | Complete source coverage inventory and final six-group selection below. |
| Broadband local and limited shaft-average combinations | Automatic shaft membership and limited final selection implemented/tested. |
| Group configuration / bounded shared continuous preparation | Implemented; synthetic and real separate/shared equivalence passes, including all 239 stored outputs. |
| Output/reference/preprocessing contracts | Existing contracts retained; new metrics have focused serial/spawn/resume coverage. Group isolation and serial/spawn/resume tests pass; real full-patient audit pending. |
| Final workflow timings, larger recording, complete process memory and tuning | Bounded 118/249-contact profiles and tuning completed below; complete-patient throughput remains pending. |
| Final full-patient runs | Patients 001 and 006 complete and audited across all six groups (609/605 windows), float32. Patient 111 running. Three complete audited patients remain the completion gate. |
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
checks and bounded reads. The
[saved-output integration receipt](../tests/final_workflow_review/derived_stored_output_validation.json)
adds 36 cases across CAR, adaptive-local and shaft-average outputs: Pearson,
absolute Pearson, their changes, and changes of RE/orthogonalized envelope
correlation agree exactly with independent saved-array formulas. Two-window
batches and gapped selections verify boundary behavior; raw outputs and reference
provenance remain unchanged. These checks use the bounded four-window trial,
not a substitute for the full-patient payload audit.

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
multiplying the entire main selection over references. Configuration integration is complete; the inventory maps each selected source output to its final group.

`reference: {method: shaft_average, shafts: metadata}` resolves the authoritative
patient `electrodegroup` membership automatically. It preserves contact axes,
subtracts each shaft's retained-contact mean, and records the resolved membership
and coefficients. It has the existing shaft-mean definition: a singleton shaft
is unavailable; two-contact shafts yield opposite signals. It does not inherit
adaptive-local CAR fallbacks. Explicit shaft layouts retain their behavior.
Three new tests and the 13 affected adaptive-reference tests passed.

Source candidate ledger SHA256: `b9e15f442ffa8e054103c899a5424dedb096242230c95b2fcea1c709f845579e`.

## Combined experiment and final selection

[The experiment](../examples/workflow/final_experiment.yaml) requests six homogeneous
jobs per recording. Global reference and preprocessing defaults apply unless a
named group overrides them. A one-metric group supplies a metric-specific choice;
several groups can request that metric under distinct references. Group names
are appended to the run name, and the existing output contracts retain reference,
preprocessing, metric, contact order and window identity independently.

| Group | Outputs | Reference | Window preprocessing |
|---|---:|---|---|
| car-normalized | 201 | CAR | detrend, sample z-score |
| car-amplitude | 13 | CAR | detrend only |
| local-normalized | 15 | adaptive local | detrend, sample z-score |
| local-amplitude | 1 | adaptive local | detrend only |
| shaft-normalized | 8 | shaft average | detrend, sample z-score |
| shaft-amplitude | 1 | shaft average | detrend only |
| Total | 239 | | |

CAR's previous 188-output main selection now includes the ten previously separate
CAR candidates plus fixed-lag correlation, event excess and partial correlation.
This keeps compatible per-window fits/transforms in one Calculator. The separate
CAR candidate file remains usable alone (13 outputs), but must not be run in
addition to the complete main selection for the same input. The local selection
adds those three metrics, cosine for downstream Pearson, and two broadband
envelope outputs to its prior nine. Shaft-average uses only the eight normalized
candidates needed for the source coverage and partial correlation. Each reference
also has its detrended, unnormalized log-scale ratio; CAR adds it to the existing
12 amplitude outputs. Jensen–Shannon is the sole omission among the new estimator
candidates, under the measured redundancy decision above. Expensive directed
families, group delay and GraphicalLasso remain excluded.

```sh
python -m pyspi.workflow preflight-experiment examples/workflow/final_experiment.yaml
python -m pyspi.workflow run-experiment examples/workflow/final_experiment.yaml
# Resume complete/unfinished groups and create later groups not reached before interruption:
python -m pyspi.workflow run-experiment examples/workflow/final_experiment.yaml --resume
```

`preflight-experiment` validates every group before calculation. Existing standalone
`preflight`/`run` configurations continue to use the same metric, preparation and
output code. Group overrides support metrics, reference, preprocessing, output
precision/overrides and execution settings; EDF selection, filters and window grid
are global in this schema. No reference decisions enter individual estimators.

The invocation-local LRU caches disk-backed prepared recordings, bounded by
`sharing.preparation_cache_mib` (8192 MiB initially; zero disables sharing for
comparison). Keys include EDF identity, selected contact/order mapping, patient
and recording, sampling rate, notch and broad filtering; reference keys additionally
include the fully resolved weights/rules and continuous-band union. Window
normalization is excluded from the preparation key because it still happens
separately after slicing each group's windows. Cache lifetime ends at each
recording boundary. Worker processes reopen existing mapped files instead of
copying a mapped recording into a second scratch file.

For none/CAR/explicit shaft/adaptive-local references, continuous band filtering
is shared before the fixed instantaneous reference. These linear operations commute;
independent reference-first tests compare the resulting arrays and metadata.
Other reference methods retain the ordinary preparation path. Reference application
uses bounded time blocks (no time filtering is blocked), preserving exact channel
weights and authoritative output axes. The cache budget bounds retained files,
not total process RAM or temporary working files: preparing one new entry may
coexist with retained entries until insertion/eviction. Scratch belongs on a disk
filesystem, not a small RAM-backed `/tmp`. The full process-tree memory profiles
remain the authority for worker settings.

Validation: eight new grouped execution tests cover defaults, serial/spawn
agreement with separate jobs, custom-reference band commutation, cache bounds,
read-only maps, settings isolation, complete/partial-group resume, invalid
configuration rejection and process-tree profiling. Forty affected standalone
config tests and the updated full-main inventory check passed. No mathematical
goldens were changed.

## Initial real grouped execution comparison (before further I/O tuning)

Patient 001, all 239 outputs, four full windows, float32, one worker/thread,
256 MiB spectral cache and 32 MiB output batches:

| Mode | End-to-end | Summed preparation | Peak process-tree PSS |
|---|---:|---:|---:|
| Shared preparation | 157.8 s | 85.2 s | 1648 MiB |
| Separate preparation | 163.1 s | 89.7 s | 2411 MiB |

Receipts: [shared](../tests/final_workflow_review/001-shared-w1.json),
[separate](../tests/final_workflow_review/001-separate-w1.json).
Shared preparation loaded/filtered the base once, reused it for two further
references, and reused each referenced input for its amplitude group. Amplitude
preparation dropped to 3–29 ms, but writing the three referenced cache entries
consumed most of the saved wall time. The initial total gain is only about 3%,
although process memory fell about 32%; cache I/O is a demonstrated next hotspot.
System-memory availability also includes file-cache state and other programs;
these process peaks are not a promise of identical system pressure.

[All 239 stored outputs](../tests/final_workflow_review/001-shared-separate-equivalence.json)
were compared across every trial window: **bit-identical float32 values**, identical
finite masks, availability, statuses, channels and reference/preprocessing
identity. This includes the independently prepared bands and robust covariance
outputs. Final tuning, the larger recording and complete-patient runs remain
outstanding. These four-window elapsed times amortize a full recording's
preparation over too few windows to predict cohort wall time directly.

## Final execution tuning

The [reference-I/O comparison](../tests/final_workflow_review/reference_io_timings.json)
measured three alternating repetitions on the complete patient-001 prepared
recording. Flushing a completed private scratch array once, rather than after
every time block, reduced median CAR-array construction from 2.94 to 2.46 seconds
(16%). All arrays were bit-identical; measured process PSS remained about
1010–1025 MiB. Output durability/checkpoint behavior is unchanged. Dirty filesystem
pages also consume system resources; the scratch cache budget is not a RAM cap.

The [two-worker trial](../tests/final_workflow_review/001-tuned-w2.json) used
128 MiB output batches and a 512 MiB spectral cache. All six groups/four windows
took 146.8 seconds, versus the initial shared trial's 157.8 seconds. Process-tree
PSS rose from 1648 to 3389 MiB. Main-CAR calculation wall time fell from 47.0 to
34.2 seconds, despite worker startup; small groups spent more time starting workers
than calculating. Summed writing time fell from 12.2 to 7.3 seconds. These changes
were combined in this trial, so their individual speedups cannot be inferred.
[Every stored output](../tests/final_workflow_review/001-tuned-equivalence.json)
remained bit-identical, including availability and statuses. No JVM was started.

The [249-contact trial](../tests/final_workflow_review/111-tuned-w1.json) used one
worker/thread, 256 MiB spectral cache, 128 MiB output batches and 16 GiB disk
preparation cache. All groups/four windows took 518.5 seconds, including 183.6
seconds preparation, 307.2 seconds calculation and 18.9 seconds writing. Peak
process-tree PSS was 3034 MiB; minimum system-available memory was 1425 MiB.
The disk cache reused the base for both additional references and each referenced
input for its amplitude group; peak retained files were about 13.7 GiB.

That larger trial exposed a specific PLI cache problem: the requested sign matrix
fits 256 MiB, but adding an optional count matrix evicted it and caused repeated
calculation across bands. The fix caches the optional count only when both fit.
Debiased PLI still explicitly obtains the count whenever it needs it. Formula,
precision and undefined-input handling are unchanged. All 69 focused phase-lag
tests passed, including a cache-boundary regression and independent scalar tests.
The [complete 132-output spectral benchmark](../tests/final_workflow_review/spectral_cache_timings.json)
compares the old policy, fixed 256 MiB and fixed 1024 MiB using three real windows
at each channel count; it excludes preparation/storage and is not a full-run
throughput measurement.

| Complete selected spectral family | Median/window | x600 | x600x69 |
|---|---:|---:|---:|
| 118 contacts, old 256 MiB policy | 2.93 s | 29.3 min | 33.7 h |
| 118 contacts, fixed 256 MiB | 2.95 s | 29.5 min | 34.0 h |
| 118 contacts, fixed 1024 MiB | 2.67 s | 26.7 min | 30.7 h |
| 249 contacts, old 256 MiB policy | 30.60 s | 5.10 h | 351.9 h |
| 249 contacts, fixed 256 MiB | 19.56 s | 3.26 h | 224.9 h |
| 249 contacts, fixed 1024 MiB | 21.27 s | 3.55 h | 244.6 h |

Every result was bit-identical across all three policies on all six windows.
At 249 contacts the fixed 256 MiB policy saves 36% of this family's time, or
about 1.84 hours per 600 windows at that contact count. The 118-contact difference
between old/fixed 256 MiB is timing noise, not a claimed regression or improvement.
Sampled process PSS for fixed 256/1024 MiB peaked at 815/1306 MiB for 118 contacts
and 2108/2867 MiB for 249; these isolated-family values include comparison arrays
and are not whole-workflow memory limits. The cohort columns simply repeat each
same-sized workload sequentially, excluding preparation/output/startup; real
patients have different contact counts and may behave differently.

Final defaults use 128 MiB output batches, 16 GiB disk preparation cache, one
numerical thread and 256 MiB spectral cache. The general experiment keeps one
worker for conservative larger-recording memory use. The patient-specific full-run
configs use two workers only for main CAR at 80/118 contacts, and serial execution
for inexpensive groups; all 249-contact groups remain serial. Two workers at 80
contacts are an extrapolation from the measured 118-contact case, to be verified
by the complete-patient profile. Larger cache settings are not selected: the
118-contact savings are small and the 249-contact trial did not benefit.

New telemetry records worker readiness and result-delivery delay. Delivery includes
serialization, transport and queue waiting; these overlap other worker activity
and must not be added as independent wall-time components. The two affected
serial/parallel shared-execution tests passed after this instrumentation change.
Likewise, `calculation_wall_seconds` includes result callbacks/journal handling;
in parallel mode it also includes worker startup, preprocessing and transport.
`output_seconds` includes callback handling as well as batch persistence, so
calculation and output columns overlap. Use measured end-to-end elapsed time for
total throughput; do not obtain it by adding these diagnostic stage counters.

Full-patient execution and the bounded-memory payload audit remain required.
`tests/final_workflow_review/run_full_patient.py` profiles every planned window;
`audit_outputs.py` reads every saved payload and checks the full contact/reference,
preprocessing/window, availability, float32, status and overflow contracts.
Numerical NaNs and calculation failures are counted separately for investigation.

The [001](../tests/final_workflow_review/001-bounded-audit.json) and
[111](../tests/final_workflow_review/111-bounded-audit.json) bounded audits read
all four windows of every one of the 239 outputs. All attempts succeeded; no
payload was unavailable, no reference contact was unavailable, and no precision
overflow occurred. Off-diagonal NaNs occurred only in the six significance-filtered
cross-correlation summaries, with recorded `no_significant_lags` diagnostics.
The 249-contact trial retained only 696 of 247008 directed pair/windows for each
such output (0.282%); missing significance is not a calculation failure or zero
connectivity. Existing robust-covariance rank/determinant warnings remain in the
logs; the approved safeguards and estimator settings were not weakened to remove
them. These bounded results do not establish the behavior of every later window.

## Complete-patient results (in progress)

Patient 001 has completed all six groups with 118 contacts and 609 full windows.
The [whole-run profile](../tests/final_workflow_review/001-full-profile.json) records
6526.45 seconds (1 h 49 min) including preparation, startup, calculations and
writing. Two workers were used only for main CAR; other groups used one. Peak
process-tree PSS was 3809 MiB (RSS 4587 MiB); minimum system-available memory was
1656 MiB. No JVM was started. Shared preparation had one base miss/two hits,
three reference misses/three hits, no evictions, and 8.78 GiB peak retained disk
files within the 16 GiB budget.

| Patient 001 workload | Elapsed per recording window | x600 | x600x69 |
|---|---:|---:|---:|
| Main CAR, 201 outputs | 8.060 s | 80.60 min | 92.69 h |
| CAR amplitude, 13 outputs | 0.119 s | 1.19 min | 1.37 h |
| Custom-local normalized, 15 outputs | 2.292 s | 22.92 min | 26.36 h |
| Custom-local amplitude, 1 output | 0.0088 s | 5.26 s | 0.10 h |
| Shaft normalized, 8 outputs | 0.220 s | 2.20 min | 2.53 h |
| Shaft amplitude, 1 output | 0.0089 s | 5.35 s | 0.10 h |
| All groups including orchestration | 10.717 s | 107.17 min | 123.24 h |

These are measured full-group elapsed times divided by 609, not sums of overlapping
stage counters. The projections repeat this same 118-contact workload and include
its amortized preparation/startup/writing. They are not predictions for patients
with different contact counts or conditioning, and exclude the subsequent audit.
Individual metric timings within a shared family must not be interpreted as
standalone costs; the independent family/incremental benchmarks remain above.

The [full payload audit](../tests/final_workflow_review/001-full-audit.json) read
every saved matrix, checking complete window coverage, authoritative contacts and
order, reference/preprocessing identity, float32 dtype, storage availability,
diagonal/reference masks, attempt statuses and recoverable overflow records.
All **145551 attempts succeeded**. There were no missing payloads, unavailable
reference contacts, infinite outputs or precision overflows. Saved run files
total 2.78 GiB, including metadata and diagnostics. Clinical input files and
historical runs were not changed.

Off-diagonal NaNs occurred only in the six significance-filtered cross-correlation
outputs. Each retained 11232 of 8407854 directed pair/windows (0.134%); the rest
had no significant lag under the approved test. These are successful missing
estimates, not failed calculations or zero connectivity. The full audit also
records the existing robust-covariance rank/determinant warnings. Coherence
magnitude's mean emits an empty-slice warning for its deliberately missing
self-connection diagonal; its off-diagonal results were all finite. No warnings
were suppressed to obtain a clean-looking run.

Patient 006 has also completed all six groups: 80 contacts, 605 windows and
**144595 successful attempts**. Its
[profile](../tests/final_workflow_review/006-full-profile.json) records 3097.77
seconds (51 min 38 s), with the same two-worker main-CAR/serial-other-groups
settings. Peak process-tree PSS was 3697 MiB (RSS 4848 MiB), minimum available
system memory 2237 MiB, and peak retained disk preparation 5.91 GiB with no
evictions. Its [full payload audit](../tests/final_workflow_review/006-full-audit.json)
again found no failed attempts, missing payloads, unavailable reference contacts,
infinite outputs or precision overflows. Stored run files total 1.30 GiB.
The same six significance-filtered cross-correlations are the only outputs with
off-diagonal NaNs: each retained 6212/3823600 directed pair/windows (0.162%).
Covariance and missing-diagonal warning categories match patient 001.

| Patient 006 workload | Elapsed per recording window | x600 | x600x69 |
|---|---:|---:|---:|
| Main CAR, 201 outputs | 3.769 s | 37.69 min | 43.34 h |
| CAR amplitude, 13 outputs | 0.0653 s | 39.19 s | 0.75 h |
| Custom-local normalized, 15 outputs | 1.141 s | 11.41 min | 13.12 h |
| Custom-local amplitude, 1 output | 0.0065 s | 3.89 s | 0.075 h |
| Shaft normalized, 8 outputs | 0.123 s | 1.23 min | 1.41 h |
| Shaft amplitude, 1 output | 0.0065 s | 3.90 s | 0.075 h |
| All groups including orchestration | 5.120 s | 51.20 min | 58.88 h |

The same projection limits apply: these repeat an 80-contact workload, not a
mixed cohort, and exclude the subsequent audit. The substantially different
118/80-contact times demonstrate why one channel count should not be used as
a blanket cohort prediction.

Patient 111 is running through the same driver with one worker, a full profile
and automatic payload audit. The goal is not complete until that larger patient
and the final cross-patient review finish.
