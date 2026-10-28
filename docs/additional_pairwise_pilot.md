# Additional pairwise EEG pilot: evidence and run selection

The implementation and bounded evaluation cover all eleven candidates. The
separate review selection currently enables **nine outputs per reference**:
histogram relative entropy, both distance-correlation definitions, two 30–70 Hz
PEC definitions, beta/high-gamma PAC, IDS, Gaussian DI (five terms), and Gaussian
SI (one lag). These are candidates for evaluation, not demonstrated improvements
in EZ prediction. Kernel MI inclusion awaits the runtime choice; radius-one DTW
remains secondary in the eleven-output pilot. No estimator was changed to meet
the runtime budget.

Use [the candidate metric selection](../pyspi/workflow/additional_candidates.yaml)
and matched patient-001 workflow examples for
[CAR](../examples/workflow/additional_candidates_car.yaml) and
[strict Laplacian](../examples/workflow/additional_candidates_laplacian.yaml).
Each uses two workers, one numerical thread, 2048 Hz output, 0.5-second
nonoverlapping windows, detrending and z-scoring. Band PEC follows these window
flags; PAC retains its documented prototype preprocessing. The established
188-output normalized and 12-output amplitude selections are unchanged. This
adds nine metric definitions, or eighteen outputs if both references are run;
it does not add eighteen different estimators to the established selection.

## Measured scope and provenance

Receipts and reproduction scripts are in [tests/candidate_pilot](../tests/candidate_pilot/README.md).
Three windows spanning each recording were measured for patients 004, 001 and
111: respectively 40, 118 and 249 retained CAR contacts, and 28, 92 and 205
usable strict-Laplacian contacts. Endpoints and missing physical neighbors remain
on the output axes as NaN. All eleven metrics had 100% finite usable off-diagonal
coverage in all eighteen patient/reference/window cases; this does not imply
that every retained contact is usable or that all cohort windows will succeed.
None of these 198 metric/window/reference results was constant across usable
pairs. No outcome labels, EZ labels or feature values from the historical cache were
used. The metadata-only cohort scan used its patient/contact names.

The common output sampling rate is 2048 Hz. Patients 004 and 001 were acquired
at 1024 Hz and resampled by the existing workflow; 111 was acquired at 2048 Hz.
Upsampling adds no acquired bandwidth. Continuous filtering precedes windows;
band filtering was shared before fixed linear rereferencing for the matched
pilot. Two direct workflow checks verify that these operations commute. Full
workflow profiles below use the normal production preparation order.

The historical cache includes 72 people with 40–249 retained contacts. Coordinates
support complete, forward-ordered shaft layouts for 71. Patient **090** has a
coordinate-order reversal on E′: its local layout must be reviewed before use.
No physical order was silently changed. This check establishes metadata
consistency, not clinical localization accuracy. The three concrete pilot
layouts retain excluded contacts in the shaft lists so gaps cannot be bridged.

## Standalone runtime

One numerical thread; fresh per-metric Data/cache. These times include window
preprocessing and required band-window copies, but exclude imports/JIT,
continuous preparation, output writing and worker startup. Medians use three
different EEG windows, not repeated measurements of one window.

The first table makes the requested multiplication explicit using the
representative **118-channel CAR** patient. Its ×72 column assumes every person
has that same size; the next table models the actual contact-count distribution.

| Output | Seconds/window | ×600 (minutes) | ×600×72 (hours) |
| --- | ---: | ---: | ---: |
| Histogram relative entropy | 0.0204 | 0.204 | 0.244 |
| Distance correlation, corrected | 0.9263 | 9.263 | 11.116 |
| Distance correlation, biased | 0.9008 | 9.008 | 10.810 |
| DTW, radius one (pilot only) | 0.1155 | 1.155 | 1.385 |
| 30–70 Hz PEC | 0.0129 | 0.129 | 0.155 |
| 30–70 Hz orthogonalized absolute PEC | 0.7461 | 7.461 | 8.953 |
| Beta/high-gamma PAC | 0.0161 | 0.161 | 0.194 |
| IDS | 0.0399 | 0.399 | 0.479 |
| Kernel MI (decision pending) | 1.4968 | 14.968 | 17.961 |
| Gaussian DI, five terms | 0.3421 | 3.421 | 4.105 |
| Gaussian SI, one lag | 0.0202 | 0.202 | 0.242 |

The distribution-adjusted planning model interpolates log time versus log
contact count between the three measured sizes. CAR includes all 72 historical
selections; local includes 71 verified layouts. It is a projection, not measured
cohort throughput or a confidence interval. The JSON also records scenarios
using sampled window minima/maxima and an upper-contact-count scenario for 090.
Only per-measure costs determine feasibility; the whole panel may take longer
than one day.

| Output | CAR, 72 people (h) | Local, 71 verified (h) | Both known selections (h) |
| --- | ---: | ---: | ---: |
| Histogram relative entropy | 0.26 | 0.16 | 0.42 |
| Distance correlation, corrected | 13.75 | 10.24 | 23.99 |
| Distance correlation, biased | 13.44 | 10.11 | 23.55 |
| DTW, radius one (pilot only) | 2.05 | 1.34 | 3.39 |
| 30–70 Hz PEC | 0.18 | 0.14 | 0.32 |
| 30–70 Hz orthogonalized absolute PEC | 13.02 | 8.10 | 21.13 |
| Beta/high-gamma PAC | 0.24 | 0.17 | 0.40 |
| IDS | 0.62 | 0.37 | 1.00 |
| Kernel MI (decision pending) | 24.92 | 16.55 | 41.47 |
| Gaussian DI, five terms | 6.04 | 4.02 | 10.06 |
| Gaussian SI, one lag | 0.33 | 0.23 | 0.56 |

Distance correlation is within budget for either reference, but **both references
serially are about 24 hours per definition before overhead**. Two-worker family
execution is the recommended planning setting; the observed family speedup is
not a measured standalone distance-correlation speedup. When both definitions
are requested together, their shared calculation avoids almost all second-output
cost. Running either definition alone still costs the standalone amount above.

Kernel MI is the remaining inclusion decision: CAR alone is approximately
24.9 hours serial or 15.9 hours using the measured kernel-only two-worker
speedup. Both known references would be about 26.4 hours with that speedup,
before preparation/output and the unresolved local patient. These parallel
projections extrapolate one patient's measured speedup across the cohort.

## Actual sharing and complete-family cost

The all-eleven family order is exactly the table order below. Each increment
was timed while evaluating that metric on the same window/cache; it was not
inferred by subtracting separate benchmark medians. A different selection/order
may share differently. These are 118-channel CAR and 92-usable-contact local
medians, excluding the small common window setup.

| Output | CAR increment (s) | Local increment (s) |
| --- | ---: | ---: |
| Histogram relative entropy | 0.011932 | 0.008143 |
| Distance correlation, corrected | 0.921007 | 0.683899 |
| Distance correlation, biased | 0.000445 | 0.000325 |
| DTW, radius one (pilot only) | 0.110850 | 0.066862 |
| 30–70 Hz PEC | 0.011126 | 0.007151 |
| 30–70 Hz orthogonalized absolute PEC | 0.721438 | 0.412499 |
| Beta/high-gamma PAC | 0.012680 | 0.009544 |
| IDS | 0.029933 | 0.018574 |
| Kernel MI (decision pending) | 1.417975 | 0.923861 |
| Gaussian DI, five terms | 0.337557 | 0.209523 |
| Gaussian SI, one lag | 0.011746 | 0.007261 |

The complete all-eleven family, including common window setup, took:

| Patient | CAR s/window | CAR ×600 min | CAR ×43200 h | Local s/window | Local ×600 min | Local ×43200 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 004 | 0.598 | 5.98 | 7.17 | 0.367 | 3.67 | 4.41 |
| 001 | 3.592 | 35.92 | 43.10 | 2.359 | 23.59 | 28.31 |
| 111 | 13.028 | 130.28 | 156.34 | 9.247 | 92.47 | 110.96 |

Distribution-adjusted all-eleven family projections are 59.64 hours CAR and
41.06 hours for verified local layouts, serial calculation only. This is not
the final nine-output selection's measured runtime. Removing kernel/DTW avoids
their increments, but the reduced panel has not been separately profiled.
The second distance-correlation definition costs about 0.00045 seconds/window
in the representative CAR family (roughly 26 seconds over the modeled cohort),
versus about 0.90 seconds standalone. The two definitions remain distinct after
averaging over windows.

At sixteen EEG contacts, the independent Hyppo **fast one-dimensional**
coefficient path, already evaluating unique pairs without p-values, took
0.742/0.763 seconds for corrected/biased distance correlation. The shared native
implementation took 0.122/0.112 seconds: **6.1×/6.8× faster** with maximum
absolute differences 2.1e-14/8.7e-14. Initial reference compilation took about
20 seconds and is excluded. Earlier eight-contact comparisons against the old
wrapper measured 63× distance-correlation and 154× constrained-DTW gains.
These are measured small-panel gains, not measured full-cohort speedups.

## Workflow, storage and memory

Actual patient-001 workflow runs processed twelve windows and wrote Zarr output.
The full eleven-output family used 24.0 seconds continuous preparation, 43.39
seconds calculation wall time and 0.50 seconds output with one worker; total
invocation time was 71.34 seconds. Two workers used 23.8 seconds preparation,
32.06 seconds calculation wall time and 0.91 seconds output; total was 60.84
seconds. Calculation-stage speedup was **1.35×**, including dispatch/startup in
that stage. The corresponding conservative same-patient 600-window stage
projections are 37.1 and 27.9 minutes. These are extrapolations, not full runs.

Kernel-only calculation-stage times were 16.89 versus 10.75 seconds over twelve
windows (**1.57×**), with total invocation times 26.33/20.41 seconds. Its
same-patient 600-window stage projections are 14.4/9.2 minutes. Projecting the
short stage also charges its residual startup repeatedly, so it should not be
interpreted as steady-state scaling. One numerical thread per worker avoids
nested numerical thread oversubscription. Do not assume linear scaling to more
workers or identical speedup for every metric and patient.

Continuous full-band preparation took 7.52, 28.73 and 59.99 seconds for the three
sizes. This happens once per recording/reference, not once per window. Even
72 copies of the largest measured preparation cost would be about 1.2 hours
per reference; that is a conservative scenario, not a measured cohort overhead.
Output costs depend on metric count, full retained axes, filesystem and batching;
the twelve-window measurements do not establish full-cohort storage throughput.

Sampled complete-process PSS peaks for the three matched benchmark invocations
were 790, 1543 and 2851 MiB. The twelve-window family profile peaked at 1572 MiB
PSS serial and 1560 MiB across the parallel process tree; preparation can dominate
the overall peak. Aggregate sampled RSS was 1574/1602 MiB. RSS double-counts
shared pages; worker high-water marks must not simply be added. Sampling every
50 ms can miss brief peaks, and the parallel family profile recorded three
counter-read failures as processes exited. The largest patient has not been
profiled with two workers, so this is not a universal memory bound.

Continuous band files and parallel input mappings use disk-backed scratch and
read-only worker views. Three band arrays alone require roughly 0.56/1.65/3.43
GiB at these recording sizes; the parallel base-signal mapping and filesystem
headroom are additional. Avoid a tmpfs scratch location. Distance products use
bounded sample blocks; their retained caches contain channel-scale arrays,
not a stored sample-by-sample matrix for every channel pair.

Thirteen stored-result comparisons cover all eleven family outputs across
worker counts, kernel alone across worker counts, and kernel alone versus the
family. Every numerical array matched **exactly**, with equal missing masks
and success statuses. No JVM was started. These checks validate execution
consistency, not scientific usefulness.

## Novelty and surrogate findings

Novelty comparisons first average the three windows, then compare pair rankings
against a selected existing panel (rank correlations, covariance/cosine, RECI,
Gaussian MI/Granger, broadband PEC, cross-correlation and band PLV). This follows
the user's current averaging practice but does not reproduce a 600-window
average. It is not an exhaustive test against all 188 outputs.

- Radius-one DTW tracks cosine distance very closely: Spearman rho 0.996–0.998
  in five cases, 0.953 for patient 111 local. It is not algebraically redundant,
  but the evidence favors leaving it secondary. Its unchanged allowance is
  only 0.49 ms; a wider allowance is a separate scientific choice.
- Distance correlation's strongest existing rank/covariance similarity is about
  0.91–0.97 for the corrected definition and 0.89–0.95 for the biased version.
  It offers a nonlinear definition, not proven extra EZ information. Nonlinear
  transforms before window averaging must not be treated as exact redundancy.
- Histogram divergence, band PEC and PAC are less similar to this baseline
  panel. Low similarity can reflect noise or reference sensitivity; it is not
  evidence of predictive value. Histogram divergence compares distribution
  shapes, not timing or causal influence.
- IDS and Gaussian DI can be fairly close to existing dependence measures
  (strongest absolute rho about 0.65–0.83 and 0.75–0.93 respectively); Gaussian
  SI's strongest similarity is about 0.52–0.78. Kernel MI's is about 0.36–0.63.
  The full receipts retain the top three comparisons for each case.

Surrogates used common valid contact probes, all three windows, eight replicates
and three controls per reference: independent shifts before reference,
independent Fourier phases before reference, and independent shifts after
reference. The first two retain the reference mixing operation; the third
breaks its synchronous shared-source structure. Band views use the same source
shifts/phases. Fourier controls preserve periodograms, not marginal shape;
shifts preserve input samples/autocorrelation before detrending. Their circular
boundary and the short, autocorrelated windows limit interpretation. No p-values
or significance decisions are claimed.

PAC's mean across probe pairs was 0.106–0.121 across the six cases; control means
were generally 0.104–0.126. Absolute orthogonalized PEC means were 0.123–0.142,
with comparable controls (about 0.121–0.139). This pilot therefore **does not
show a general mean coupling excess** for either output. Magnitude, short-window
sampling, autocorrelation and reference mixing contribute positive baselines;
circular shifts can also retain stationary narrowband PAC. A localized subset
of informative pairs is neither established nor ruled out by this aggregate
check. Keep both clearly exploratory rather than infer coupling from a positive
number alone.

Ordinary band PEC CAR means (0.028–0.036) exceeded the control means in these
windows; Gaussian DI/SI CAR means also exceeded their controls. Local results
were less consistent. This is descriptive evidence of sensitivity, not a
calibrated test, causal result or predictive validation. Kernel MI, IDS and
distance correlation also retained substantial positive control values;
nominally bias-corrected distance correlation does not correct temporal
sample dependence.

References matter: matched CAR/local pair-ranking correlations were generally
low (roughly −0.06 to 0.29), even after restricting to the same available output
contacts. The reference changes the signal mixture, rather than just scaling
it. A synthetic independent-source check gives neighboring strict-Laplacian
outputs correlation near −2/3 solely from shared neighbors; shifting after
reference removes it. Proportional/overlapping reference-source metadata must
accompany interpretation, particularly for orthogonalization.

Exploratory shaft-average PAC produced different rankings too (rho with CAR
0.10–0.30 on the common probes), but comparable positive means. No third
production reference run is selected: this pilot does not establish its added
value. Original-reference and bipolar runs remain secondary. Raw difference
metrics invariant to common subtraction are not added as duplicate reference
runs; the normalized candidates here need not retain that invariance.

Historical top_features evidence gives histogram divergence the strongest
motivation: the combined panel's AP was 0.408 versus 0.386 for the historical
185-feature baseline, but the improvement interval included zero
[−0.009, 0.060]. Histogram CAR was selected in 40/40 primary outer folds;
Jensen–Shannon was almost identical (rho 0.997) and remains unadded. Historical
PAC increments were small with intervals including zero. Those analyses used
different filters and local-reference endpoint handling; they cannot validate
this corrected selection. No new predictive model or cohort run was launched.

## Validation and remaining decision

The implementation/formula tests and earlier affected regressions are recorded
in [the implementation ledger](additional_pairwise_review.md). This stage added
two filter/reference-commutation checks, three surrogate-invariant/model checks
and thirteen real stored-output comparisons. All passed. Existing unrelated
passing tests and goldens were not rerun or refreshed.
Both nine-output workflow examples pass preflight without starting Java;
the established 188/12 counts and CSV identity/flag columns are preserved.

Kernel MI is deliberately absent from the nine-output selection pending the
runtime decision. Patient 090 local geometry remains an explicit input-data
limitation, not an inferred layout. Full-cohort production, predictive
validation, wider DTW and additional reference runs remain outside this goal.
