# Remaining Java optimization investigation

2026-10-28, baseline commit `7578137`. The initial investigation below used
bounded prototypes. The subsequently approved kernel refactor is now complete;
its production results are recorded at the end of this document. KSG and
Kozachenko production changes remain deferred. The main 188-output and amplitude
12-output configs remain JVM-free and unchanged; no outputs were added.

## What is already done

Gaussian information measures and genuine symbolic TE are native. The selected
information-family comparison recorded 42.17 -> 0.292 seconds/window, including
the approved definition/selection changes; see the
[completed review](regression_gaussian_review.md#information-family-benchmark-and-symbolic-controls).
Native-only Calculators no longer initialize Java. Integrated information was
already ported from Octave before the audit; its remaining bottlenecks are in
native optimizers and are outside this Java investigation.

Several non-Gaussian audit suggestions were already cross-applied: one ACF per
channel for automatic exclusion windows, buffered `JArray.of` entropy inputs,
shared marginal/source-history/target-history entropy terms, and shared CCE/DI
results. Their old proposed gains must not be counted again.

## New measured probes

Synthetic AR(1) signals with one lagged cross-channel link, 1,024 samples,
standardized before timing. Each Java/candidate comparison uses identical data
and settings. Kernel and Kozachenko timings are medians of three passes; KSG
cache experiments have one timed pass per path. Small independent processes
separate the experiments. Imports/JVM startup and signal generation are excluded;
Java wrappers include Data construction. Full inputs, versions, source hashes,
timing samples and differences are in the
[receipt](../audit/briefing/remaining_java_opportunities.json), with the
[reproduction script](../audit/profiles/micro/explore_remaining_java.py).

| Tested subset | Channels | Existing seconds | Prototype seconds | Gain | Largest difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kernel JE + CE (width .5) + MI (width .25) | 8 | 1.104 | 0.0550 | 20.1x | 8.9e-15 bits |
| Kozachenko JE + CE | 8 | 0.580 | 0.0311 | 18.7x | 1.4e-14 nats |
| KSG automatic TE, no exclusion | 4 | 24.534 | 4.252 | 5.77x | zero |
| KSG automatic TE, automatic exclusion | 4 | 23.658 | 4.121 | 5.74x | zero |

All finite masks matched; the KSG chosen embedding histories also matched
exactly. These comparisons establish initial equivalence, not scientific
validity for all inputs. The existing independent audit remains the starting
point for production validation.

The two native prototypes were also measured directly at 118 channels:

| Tested subset, neighborhood/tree setup included | One window | x600 | x600x69 |
| --- | ---: | ---: | ---: |
| Kernel JE + CE + MI | 1.488 s | 14.88 min | 17.11 h |
| Kozachenko JE + CE | 6.586 s | 65.86 min | 75.73 h |

These are complete costs for the listed subsets with internal sharing, not
per-output costs and not the whole kernel/Kozachenko family. Large Java baselines
were not repeated; no measured 118-channel speedup is claimed. Projections are
serial calculation work, excluding production workflow, preprocessing and I/O.
No full-size KSG run or extrapolated KSG cohort time is claimed. EEG conditioning,
channel count, histories and machine load can materially change costs.

## Initial recommendations and constraints

1. **Shared native kernel counts.** Pack each channel's closed-box neighborhood
   comparisons once, then intersect/count them across pairs. Reuse joint counts
   for compatible entropy and information outputs. The current JE/CE and MI
   widths differ, so they need separate preprocessing. The prototype builds those
   widths sequentially; one width retains 15.67 MiB of packed comparisons and
   marginal counts at 118 x 1,024, excluding temporary arrays/process overhead.
   Future TE/SI/XME/CCE/DI sharing must key on sample alignment, width and lag;
   a different trimmed interval also changes the standard deviation used for
   the width. Use the existing bounded, mutation-aware cache. Keep bit units,
   self-counting, closed boundaries and current bandwidth settings. The prototype
   does not yet implement arbitrary window lengths, temporal exclusion,
   degenerate inputs or production diagnostics.

2. **Cache KSG embedding searches and control Java threads.** Current auto TE
   repeats each channel's search for every partner. The prototype reduces 24
   searches to four at four channels, then uses the existing Java fixed-history
   estimator. This follows JIDT's separate source/destination AIS searches
   ([source](https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/continuous/TransferEntropyCalculatorViaCondMutualInfo.java)).
   With exclusion, cache by `(channel, exclusion length, estimator/search settings)`;
   a channel-only cache is incorrect. The first exclusion fixture happened to
   use one length. A separate heterogeneous-autocorrelation fixture at 4 x 256
   produced lengths 2, 4, 5 and 10: 24 searches -> 10, 2.973 -> 1.234 seconds
   (2.41x), again exact values and embeddings. This confirms the smaller,
   data-dependent benefit when exclusions differ.

   Workflow BLAS/OpenMP limits currently do **not** set JIDT's `NUM_THREADS`.
   Existing audit measurements establish excess CPU use from Java thread fan-out,
   but not a universal single-thread latency gain: automatic TE can be slower
   with one Java thread. Expose/control Java threads before enabling these
   estimators in parallel windows, then benchmark complete throughput. Both new
   cache paths used one Java thread, so their gains isolate sharing. Retaining
   Java's KSG estimator initially avoids an unnecessary full estimator rewrite.

3. **Native Kozachenko nearest-neighbor entropy.** Replace quadratic distance
   searches with an exact Euclidean tree query, using the existing shared entropy
   composition layer. This can benefit JE/CE, SI, XME, CCE and DI together.
   [SciPy's tree implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
   supports the required nearest-neighbor lookup. A separate scalar-entropy
   dimensionality probe found about 14-15x at two/four dimensions, falling to
   3.56x at ten dimensions. Therefore the 18.7x JE/CE result must not be applied
   to long-history families. The 6.59-second full-size subset is still expensive
   for optional run inclusion.

## Correctness and definition decisions before adoption

- Correct the [audit's proven Kozachenko search defect](../audit/notes/it_kozachenko.md)
  above 2,000 rows rather
  than preserving its wrong distances. It does not affect current 1,024-sample
  windows. Existing exact-distance references already diagnose this; no repeat
  investigation is needed.
- The prototypes preserve the seeded 1e-8 jitter convention. Changing jitter or
  tie treatment requires discussion, especially for quantized, identical or
  constant channels. Initial Kozachenko comparison exposed column-order-dependent
  noise assignment in the existing cache; matching its first-fill order restored
  equivalence. Production design must explicitly address permutation behavior.
- Preserve distinct kernel/KSG temporal-exclusion rules during optimization;
  they have different boundary conventions. Do not silently substitute one
  estimator for another or alter units, widths, neighbor count or histories.
- Review nonlinear/sign labels and failure diagnostics across the affected
  classes. The audit's Kozachenko `linear` classification issue remains in the
  current constructor. Symmetric pair reuse must account for jitter and actual
  estimator symmetry rather than inheriting the base class's label.

The user approved the kernel work first. KSG search reuse/thread control and
Kozachenko remain possible subsequent tasks, with run inclusion assessed
separately. None of these opportunities further accelerate the current 188/12
selections until additional estimators are deliberately enabled.

## Completed native kernel refactor

All nine existing kernel classes now use native packed box comparisons. The
ten catalogue outputs include XME at both one and ten source lags. JE/CE and
CCE/DI share results; marginal/history counts and compatible target-only terms
are reused across pairs. JE, MI and SI compute each symmetric pair once.
The cache retains at most 256 MiB of derived arrays, with mutation detection,
safe returned copies, and streamed comparisons for larger windows.

Histories, bandwidths, sample-standard-deviation normalization, bit units,
closed boundaries and self-counting are preserved. One approved edge correction:
if temporal exclusion leaves any observation with no eligible comparison,
return NaN with a reason and a warning recorded by the workflow. Zero joint
matches with eligible comparisons retain the old local-zero convention. See
the [definitions](calculation_definitions.md#native-box-kernel-information-measures).
Previously ignored unsupported kernel TE settings now raise an explicit error.
All kernel Java construction and execution paths have been removed.

Measurements use the same ten outputs and order on both paths. The Java
baseline loads the pre-refactor wrapper from `7578137`. Each path has one timed
pass, with Java warmed separately. Native work includes actual shared setup
and increments. Data construction, imports, JVM startup, preprocessing and
output writing are outside the timed stages. Source hashes and input hashes
are recorded in [the results receipt](../tests/kernel_refactor_results.json);
the [benchmark script](../tests/benchmark_kernel.py) checkpoints completed stages.

| Complete ten-output family, 1,024 samples | Java | Native | Measured gain | Maximum difference |
| --- | ---: | ---: | ---: | ---: |
| Synthetic, 8 channels | 12.563 s | 1.071 s | 11.7x | 6.94e-13 bits |
| Actual EEG, 8 channels | 11.962 s | 0.867 s | 13.8x | 4.71e-14 bits |

All finite masks matched. EEG uses the first 0.5-second window of patient 001,
CAR, 2,048 Hz, detrending and sample-SD normalization. The synthetic comparison
predates the history-first CCE loop; independent tests and the subsequent EEG
comparison validate that change. An initial full-size synthetic run exposed
cache thrashing in the pair-first CCE loop and was stopped; it is not a complete
timing or a speedup baseline.

The final native implementation was measured in a fresh process on all 118
channels of that EEG window, with no JVM started:

| Output in execution order | Actual increment | x600 windows | x600x69 |
| --- | ---: | ---: | ---: |
| JE, width .5 | 1.950 s | 19.50 min | 22.42 h |
| CE, width .5, after JE | 0.016 s | 0.16 min | 0.19 h |
| MI, width .25 | 1.807 s | 18.07 min | 20.78 h |
| Lagged MI, width .25 | 3.349 s | 33.49 min | 38.52 h |
| TE, width .25, k=1 | 3.957 s | 39.57 min | 45.50 h |
| SI, width .5, delay=1 | 2.689 s | 26.89 min | 30.93 h |
| XME, width .5, k=1 | 2.087 s | 20.87 min | 24.00 h |
| XME, width .5, k=10 | 6.112 s | 61.12 min | 70.28 h |
| CCE, width .5, five terms | 17.626 s | 176.26 min | 202.70 h |
| DI, width .5, after CCE | 1.471 s | 14.71 min | 16.92 h |
| **Complete family** | **41.065 s** | **6.844 h** | **472.248 h / 19.68 days** |

These are measured increments in this family order, not independent per-output
costs. Changing the selection or order changes cache reuse and eviction.
Peak process RSS was 402.5 MiB, including Python and imported libraries;
retained derived arrays were 256.0 MiB. The small Java comparison processes
peaked at 1.56–2.96 GiB, but include JVM history (and EEG preparation in one
case), so they are not controlled native-versus-Java memory comparisons.
Full-record EEG preparation took 12.66 seconds separately; loading a prepared
window was excluded from calculation timing.

Every off-diagonal result was finite in this EEG window. The average fraction
of observations with self-only joint neighborhoods was below 0.15% for each
output; CCE/DI use the worst term per pair. This is one window from one patient,
not a general reliability claim. There is no new matching 118-channel Java
measurement, so the small-channel speedup is not extrapolated to 118 channels.
Projected times are serial calculation work and exclude workflow overhead,
I/O and between-window/patient variability. The complete family is still
expensive; the working run selection remains unchanged.

Validation combines independent box-count formulas, nonlinear and delayed
known models, exclusions, exact boundaries, arbitrary packed lengths,
streaming, cache limits, mutation, permutations, serialization, extreme
amplitude units, and a fresh-process check that all kernel classes and workflow
dependency resolution work without starting Java. Only affected kernel
goldens and dependency contracts are checked; numerical golden arrays are
unchanged. Metadata records removal of Java dependencies with source hashes.
