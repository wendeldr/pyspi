# Workflow integration and resource review

Goal started 2026-10-28: integrate the committed hotspot work without disturbing
the active `run-hotspots` worktree; assess CAR correctness and cache memory;
resolve reported failures on their merits; run the complete suite once; measure
window parallelism, thread limits, whole-process memory, preparation and output.

## Implementation and evidence ledger

- Integrated `e7fef66` as `d0ff9ab`. Retained the newer **195 normalized / 12
  amplitude** selections. The other worktree and its uncommitted audit work
  remain separate. Historical hotspot receipts still describe their original
  200-to-198 comparison; they are not timings for the combined 195-output run.
- The Cholesky robust-fit shortcut changed a CAR control's covariance by about
  3% (seed 1, 256 samples, eight channels, standardized after CAR). This is not
  acceptable roundoff. Singular/ill-conditioned centered input now uses the
  reference MinCovDet fit with the same fixed seed; full-rank inputs retain the
  shortcut. Independent CAR support/covariance/precision checks passed.
- The 11 reported failures were stale tests, established by reviewing actual
  definitions and execution: ordinary PEC is signed; directed aggregation now
  streams unique pair fits rather than calling `_get_cache`. Updated tests
  retain explicit label assertions and exercise the real inclusive-band,
  source-first aggregation in both directions. All 11 focused cases passed.
- Spectral caches have an explicit limit, default 256 MiB per workspace.
  Registered phase moments fuse only when their planned retained arrays fit;
  registration never silently expands the limit. Small/large-cache output
  equality and bounded retention are checked. This is not a process RSS cap:
  transforms, other caches, temporaries and results are additional allocations.
- Workflow execution exposes worker count, numerical threads per worker and
  spectral-cache MiB. Spawned workers share read-only mapped prepared signals;
  the parent owns journals and output publication. Outstanding jobs/results are
  bounded by worker count. Resource settings are recorded per session and may
  change on resume without changing the scientific contract.
- Focused checks passed for serial/spawned numerical agreement, stored failure
  reports/retry isolation, and restoring caller thread limits. A sandboxed
  storage test stalled and was interrupted; the unsandboxed checks ran normally.
  Test-fixture naming/placement errors were corrected without changing outputs.

The complete `--runslow --pyspi-size=full --require-java` suite passed:
**1,777 passed, seven documented expected failures**, 461 seconds. The separate
storage suite passed **439 tests**, 13.5 seconds. No numerical goldens changed.
The expected failures concern existing unrelated API/data conversion/estimator
issues; this does not claim every calculation in the repository is correct.
After a small queue-reference cleanup, the affected serial/parallel and
interruption checks passed again (two tests); the full suite was not repeated.

Profiling exposed unnecessary eager JVM startup in Calculator. Optional runtime
checks are now lazy per required dependency, preserving explicit unavailable
dependency overrides. Native-only workflows and their spawned workers do not
start Java; a later Java request still does. Constructor/dependency checks
passed (57 initially; the new kernel test needed its actual `W-0.5` identifier).
The corrected transition test and affected worker/profiler checks then passed
(three tests). This change leaves the numerical estimators untouched.

The final profile and comparisons are complete. The
[verification receipt](../tests/workflow/resource_results.json) records test
results, packaging checks and provenance; the
[profile directory](../tests/workflow/resource_profiles/README.md) contains the
measured timings, source hashes and per-output comparisons.

The first complete-suite invocation stopped during collection: the old
`test_SPIs.py` indexed an archived pickle with renamed cross-correlation
identifiers. Source review also confirmed that it only logged numerical
differences and asserted shapes. It recomputed the entire catalogue during
collection. Removed this superseded test rather than mapping corrected
estimators onto old definitions. The retained strict per-SPI and Calculator
golden tests use the same `cml7.npy` data at full size, check the current
catalogue and numerical outputs, and retain independent formula/model tests.
No numerical goldens were changed to resolve this collection error.

## Resource controls

```yaml
execution:
  workers: 1
  threads_per_worker: 1
  spectral_cache_mib: 256
```

The larger-cache candidate is 1024 MiB. Both settings use identical estimators,
frequency bands, 0.5-second nonoverlapping windows and preprocessing. No worker
count is hardcoded for a 32-core/256-GB machine. First measure the actual config,
channel count and cache policy with a new output location:

```sh
python -m pyspi.workflow profile workflow.yaml --max-windows 4 --report trial.json
python -m pyspi.workflow estimate-resources --memory-mib 16384 --worker-peak-mib 1800 --parent-peak-mib 1200 --cpus 8
```

The second command's numbers are illustrative inputs, not measured results.
It reserves 20% of the supplied memory by default and reports the smaller of
the CPU and memory worker limits. Summed RSS conservatively double-counts
shared pages. The profiler also samples process-tree PSS on Linux; that shares
the cost of mapped/library pages across processes. Short peaks can escape the
sampler, so the runner separately records process high-water RSS.

A bounded profile includes full-record preparation, worker startup, data
mapping and durable output. Multiplying its elapsed time per window by 600 or
41,400 charges the short trial's fixed setup repeatedly. Inspect calculation
stages separately before extrapolating; neither projection is measured cohort
throughput. Numerical conditioning and other jobs can affect elapsed time.

## CAR resource comparison

Four consecutive 0.5-second windows, 118 retained contacts from patient 001,
2048-Hz prepared data, current 195-output selection, CAR, detrend and z-score.
All cases use the same input and 128-MiB output-buffer target. Each case starts
a fresh process and prepares the entire 304.75-second recording. These are
single four-window trials, not repeated timing distributions. They precede
the final lazy-JVM cleanup; its final profile is recorded separately.

| Workers | Threads/worker | Spectral cache MiB | Total seconds | Calculation wall seconds | Peak tree RSS MiB | Peak tree PSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 256 | 114.61 | 90.92 | 1908.7 | 1901.2 |
| 1 | 1 | 1024 | 108.35 | 84.01 | 2390.4 | 2382.8 |
| 2 | 1 | 256 | 75.87 | 50.83 | 3013.2 | 2772.2 |
| 1 | 2 | 256 | 117.09 | 92.21 | 1909.0 | 1901.6 |

Preparation took 14.5–15.8 seconds and output about 5.1–5.5 seconds per trial.
Parallel calculation wall time includes worker initialization, window
preprocessing and result transfer; summed metric time measures work, not elapsed
parallel time. Preparation has its own peak, and summing separate process high
water marks can overestimate simultaneous memory. The sampled tree counters
above measure concurrent resident memory; PSS avoids counting shared pages
multiple times. One sample lost a process during shutdown and was excluded.

Changing cache size or worker count preserved every output to rtol=1e-10,
atol=1e-12, including NaN/infinity masks and failure statuses. Two numerical
threads changed seven GraphicalLasso/GraphicalLassoCV outputs; the largest
absolute difference was 0.6543 in squared precision. Both thread settings
recorded 791 graphical-lasso convergence warnings across the four windows.
This is not harmless last-bit roundoff. Retain **one numerical thread per
worker** for this reviewed run. Penalties, iteration limits and solver
definitions remain unchanged; graphical-lasso convergence needs a separate
scientific/numerical review before changing those choices.

The larger cache saved about 6.3 seconds for roughly 482 MiB additional RSS in
this pilot. Two smaller-cache workers saved about 38.7 seconds versus one,
with roughly 1.1 GiB additional tree RSS. Use the 256-MiB default on a constrained
machine and increase workers only after accounting for measured memory. A
larger-memory machine can benchmark larger caches and more workers; these
results do not justify assuming linear speedups up to 32 cores.

Across the eight phase families, measured mean calculation time per window was
5.94 seconds with the 256-MiB cache and 3.96 seconds with 1024 MiB. These are
family totals across the selected bands, not a saving for each output. Other
families' timings varied between trials, so their difference need not equal
the complete-run difference. The brief CSV notes now distinguish this CAR
measurement from the older full-rank robust-fit benchmark.

## Final native-only profile

After removing eager JVM startup, the two-worker, one-thread, 256-MiB-cache run
completed the same four windows in **70.03 seconds**, with **2895.6 MiB peak tree
RSS / 2691.9 MiB peak tree PSS**. Preparation took 13.56 seconds, calculation
wall time 46.92 seconds, and output 4.76 seconds; stage counters overlap and
should not be summed. Parent and both workers remained JVM-free. All 195 stored
outputs, NaN/infinity masks and statuses matched the one-worker reference;
the recorded maximum absolute difference was zero for every output.

| Timing basis | Per-window throughput equivalent | ×600 windows | ×600×69 windows |
| --- | ---: | ---: | ---: |
| Complete bounded trial | 17.51 s | 2.92 h | 201.35 h |
| Calculation wall time | 11.73 s | 1.96 h | 134.91 h |

These are projections from four windows, not measured patient/cohort durations
or individual-window latency. Full-record preparation is amortized over only
four windows here; longer runs amortize that cost differently. Host load and
patient conditioning also vary. The change from 75.87 to 70.03 seconds cannot
be attributed entirely to removing Java from one trial each.

Final defaults remain **one worker, one numerical thread and 256 MiB spectral
cache**. Two workers are validated for this pilot; choose higher concurrency
using measured memory and increasing-worker trials. Keep one numerical thread
for the reviewed estimator set, including on resume. A larger cache is an
explicit speed/memory option. The normalized and amplitude selections remain
195 and 12 outputs, respectively; no calculations were added or removed here.

The resource validation above describes hotspot integration `d0ff9ab` and the
workflow changes committed as `15b50b4`. Remaining follow-up topics are
graphical-lasso convergence and larger-machine throughput scaling; neither
blocks using the conservative settings validated here.

## Subsequent worktree integration

The later `run-hotspots` commit `4051229` was integrated after the resource
profile. It corrects the public source-row/target-column convention for
`dcorrx`, `mgcx` and CCM, corrects sign/direction labels, and adds the other
agent's orientation and old/new comparison audit tables. Those tables retain
their original observations and open issues; they are not blanket scientific
validation of every listed estimator.

The earlier CAR safeguards, bounded cache policy, workflow implementation and
brief calculation-review notes were preserved. The 195/12 selections and their
parameters are unchanged. The numerical golden changes are exactly 13 matrix
transposes at each of the small/full sizes; every other stored value is
unchanged. The three orientation-adjusted families are absent from the reviewed
run. Focused integration checks and provenance are recorded in
[`run_hotspots_merge_results.json`](../tests/run_hotspots_merge_results.json).
The earlier profile receipts remain records of their exact measured code/config
hashes. The temporary worktree is removed after the merged changes are committed.
