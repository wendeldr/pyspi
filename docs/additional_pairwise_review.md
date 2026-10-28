# Additional pairwise candidate review

This implements the approved ten-part additional-candidate goal. Established
188-output normalized and 12-output amplitude selections remain separate.
The target is roughly one day per measure for 72 people with 600 half-second
windows each, not one day for the complete panel. Projection assumptions and
reference costs must be reported explicitly.

## Completion ledger

| Requirement | State / evidence |
| --- | --- |
| Histogram relative entropy | Implemented; explicit KL, invariance, invalid-input and cache checks pass. |
| Continuous 30–70 Hz PEC views | Pending. |
| Kernel MI, IDS, Gaussian DI/SI assessment | IDS input/cache/RNG cleanup passes focused checks; matched EEG assessment pending. |
| Cross-channel 16–30 / 70–150 Hz coupling | Pending. |
| Exact bounded distance correlation | Implemented; both conventions share products; independent centering and golden checks pass. Multi-patient gate pending. |
| Exact radius-one DTW | Implemented; full-grid oracle and tslearn comparisons pass. Run inclusion remains separate. |
| Matched CAR / contact-centered Laplacian | Explicit topology, masks, geometry metadata and serial/parallel execution implemented; real-data matched comparison and sensitivity checks pending. |
| Representative EEG runtime / memory / workflow costs | Pending; prior single-window feasibility receipts are not completion evidence. |
| Novelty, coverage and surrogate assessment | Pending; reuse historical top_features evidence with its limitations. |
| Candidate configs, docs, CSV, regressions and commits | Pending. |

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
measured 118-channel speedups. Cohort costs are serial projections from one
window; patient/channel variation and workflow costs remain to be measured.

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
candidate run inclusion is still pending.

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

## Prior evidence

The source exploration is
`/run/media/dan/DansFlash/ram_needs_rma_git/top_features`. Its full 33-candidate
report includes 40 primary and 29 secondary patients, historical 185-feature
comparisons, and a partly different preprocessing pipeline. Results motivate
candidates; they do not establish added predictive value over the corrected
current selection. The initial 118-channel EEG feasibility screen is recorded
in `/tmp/pyspi_additional_pairwise_screen.json`; production benchmarks and
validation will supersede those prototype-only receipts.
