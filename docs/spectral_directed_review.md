# Directed spectral follow-up, 2026-10-27

The approved directed and PEC refactor is implemented. Public directed matrices
use row=source, column=target. A consistent Wilson fit is shared by both
directions, requested bands and applicable metrics; dDTF uses the approved
squared formula and native-grid normalization. PEC now shares centered envelope
correlations and orthogonal projections, with explicit numerical diagnostics.
Pairwise fitting, 0.5-second nonoverlapping windows and NW=3/five tapers remain.

The six factorized directed families remain disabled in the working analysis
selection because of their measured cost. Implementation, runtime and validation
details are recorded at the end of this document and in
[calculation_definitions.md](calculation_definitions.md). The findings below
preserve the investigation and scientific decisions that led to the changes.

## Historical investigation before this refactor

Reproduce with `.venv/bin/python -B
audit/checks/probes/spectral_directed_followup.py --repeats 7 --cpu 2`.
Choose an available CPU or omit affinity. Results and source hashes are in
`audit/briefing/spectral_directed_followup.json`. Installed dependencies are
spectral-connectivity 1.1.2, NumPy 2.5.4 and SciPy 1.17.1.
The initial investigation is preserved as a historical checkpoint. After the
DC correction, only `--formula-only` was run; its result is recorded under
`directed_coherence_correction`. Factorization experiments and timings were
not repeated. Current DC pipeline timing is in
`tests/directed_coherence_results.json`.

## Formula checks independent of spectral estimation

Inject the exact transfer function and innovation covariance of a known stable
VAR(1) model. DTF, squared PDC and squared generalized PDC agree with direct
matrix formulas within 7e-16. The installed dependency's directed coherence
differs by up to 1.9494: its largest off-diagonal value is 2.9373, versus
0.9878 for the corrected squared DC. The installed formula mixes powers and broadcasts innovation variance along
the target axis, where it cancels. Clipping values to [0, 1] would not fix it.

For target i and source j, define `w_ij = Sigma_jj * abs(H_ij)**2`.
Standard squared DC is `w_ij / sum_k(w_ik)`, including the target's own
innovation among the sources. The magnitude form is its square root. They
are distinct valid conventions; the dependency's formula is neither. Squared DC
matches the existing DTF/PDC/gPDC convention and is now implemented. Apply it
at each frequency before the band's mean/max; squaring a band mean differs
from averaging squared values. Independent tests cover unequal innovation
variances, zero inflow, exact VAR transfer functions, channel-unit changes,
formula permutation equivariance and cache ownership. The expensive fitted
models are reused. Only the six `dcoh` snapshots per main golden set are
refreshed in this stage; no other numerical goldens change.

Innovation variance means unexplained model-input variance, not sensor-noise
variance. If innovations are correlated, the denominator omits cross terms:
these normalized strengths are not generally an additive allocation of the
observed target's total power. They do not establish physical causation.

The [current upstream DC implementation](https://spectral-connectivity.readthedocs.io/en/latest/_modules/spectral_connectivity/connectivity.html#Connectivity.directed_coherence)
uses the standard squared formula. Upstream also now transposes directed
outputs to source-first order. This separate change was approved:
public spectral outputs use row=source,
column=target, so entry [A1,C1] denotes A1 -> C1. Internal transfer matrices
retain their mathematical target/source indexing. Each wrapper's orientation
was verified independently, including both spectral Granger methods;
signed phase/lead measures keep their documented interpretation. The current
implementation at this checkpoint used row=target, column=source for factorized outputs.

Current upstream dDTF uses full-frequency squared DTF times squared partial
coherence. The installed method uses a product of magnitudes with PDC instead.
The squared dDTF definition was approved after discussion: form the product at
each frequency, then take the band mean/max. Implementation, independent
formula tests and focused timing are recorded below. Subsequent decisions
retain pairwise fitting and specify the native frequency grid for normalization.

## Channel order and factorization

Before the DC formula correction, fitting A1/B1 and B1/A1 on the seeded one-way coupling fixture and then
realigning the axes changes DTF by up to 0.01982 per bin and DC by 0.02114.
This is not the same as permuting a returned full SPI matrix:

- The five transfer-based wrappers calculate every ordered pair separately.
  Their matrices are exactly permutation-equivariant in this test, but each
  direction comes from a different fitted model, and reverse fits cost work.
- Nonparametric spectral Granger uses one fit for both directions. Reordering
  all three channels changes a band-mean entry by 0.01792 after realignment.

The exact VAR spectrum recovers the known transfer function within 9.5e-14,
so the finite estimated spectrum needs separate numerical treatment. On a
1,024-sample pair, raising the iteration cap from 60 to 600, or tightening
tolerance from 1e-8 to 1e-12, leaves a DTF order gap near 0.02398. More
iterations alone are not a remedy.

| FFT length | Reconstruction error, two pair orders | Maximum DTF order gap |
| --- | --- | ---: |
| 1,024 | 2.16%, 2.39% | 0.02398 |
| 2,048 | 0.083%, 0.036% | 0.01511 |
| 4,096 | 0.0196%, 0.0152% | 0.00453 |

Reconstruction error is `max(abs(factor @ factor.H - S)) / max(abs(S))`.
The gaps remain at the original FFT bins, not just newly interpolated bins.
Padding improves reconstruction but does not fully resolve order dependence.
No padding default is recommended from this single fixture.

Seven-run medians for a fresh two-channel transform, one shared factorization,
and five directed estimators were 0.0430, 0.0651 and 0.1087 seconds respectively
(1x, 1.51x and 2.53x cost). These are diagnostic pair workloads, not measured
118-channel window costs. Reusing both directions could save factorizations
only after consistency is established. A production timing estimate must
include the actual cache lifetime and all pairs.

The [upstream reconstruction diagnostic](https://spectral-connectivity.readthedocs.io/en/latest/_modules/spectral_connectivity/minimum_phase_decomposition.html#minimum_phase_reconstruction_error)
likewise distinguishes small iteration changes from faithful reconstruction.
A factorization-quality policy, singular-pair behavior, and acceptable
approximation require further investigation; no silent ridge or averaging of
inconsistent pair orders is introduced.

The user approved pair-local NaN results with a recorded reason for undefined
quantities or demonstrably unusable fits, conditional on mathematical and
numerical justification; usable pairs should still be returned. Acceptance
tolerances must be supported by reconstruction and known-model checks. Weak
coupling, statistical uncertainty, or negligible rounding differences alone
do not constitute a failed fit. No numerical rejection threshold has been
selected or implemented. Investigate a consistent fit before sharing it across
both directions, then measure the complete directed-family workload.

## Pairwise versus conditional calculations

Decision after discussion: retain explicitly pairwise fitting for this
refactor, including the corrected dDTF. Each pair's directed relationship may
reflect direct influence, an indirect path through other channels, or a shared
driver. These outputs do not condition on the rest of the recorded system and
must not be described as isolating direct connections after accounting for it.
No automatic shaft/distance-based conditioning set, average-of-other-channels
covariate, or full-channel conditional estimator is selected. Conditional
model development is outside this completion scope. This is a choice of the
question being measured, not a reason to preserve formula or numerical bugs.

On an exact A1 -> B1 -> C1 model with no A1 -> C1 coefficient, three-channel
squared PDC for A1 -> C1 is zero (below 7e-32). Fitting only A1/C1 gives a
band mean of 0.1872. The omitted middle channel changes the scientific question.
[Dhamala et al. (2008)](https://math.iisc.ac.in/~rangaraj/wp-content/uploads/2020/07/neuroimage_granger_causality.pdf)
explain the corresponding distinction between pairwise and conditional Granger
causality. Pairwise fits do not condition on other recorded channels.

A direct switch to a 118-channel nonparametric fit is not currently justified:
one trial with five tapers supplies only five Fourier observations per
frequency. Their 118x118 cross-spectrum has rank at most five. Conditioning
on all channels therefore needs more independent observations, a justified
regularization/model, or smaller prespecified groups. These choices alter the
estimator and would need a separate proposal if conditional analysis is pursued;
replacing an inverse with a pseudoinverse is not an automatically equivalent fix.

## Window and smoothing preparation

The workflow default is 0.5 seconds; this does not prove every user run uses
that setting. At 2,048 Hz it is 1,024 samples. The shared Fourier workspace
sets time-halfbandwidth product NW=3, giving five tapers, FFT spacing 1/T and
nominal full smoothing bandwidth 2*NW/T. Physical Hz are obtained equivalently
by converting the retained preset's normalized frequencies using 2,048 Hz.

| Duration | Samples | FFT spacing | Nominal full smoothing bandwidth |
| --- | ---: | ---: | ---: |
| 0.5 s | 1,024 | 2 Hz | 12 Hz |
| 1 s | 2,048 | 1 Hz | 6 Hz |
| 2 s | 4,096 | 0.5 Hz | 3 Hz |
| 4 s | 8,192 | 0.25 Hz | 1.5 Hz |

Padding changes grid density, not the taper's smoothing bandwidth or the
observed duration. At NW=3, longer windows still use five tapers. This is why
padding alone cannot validate narrow low-frequency bands or remove the
limited-observation issue. The next discussion should establish desired
temporal resolution and frequency bands if these settings are revisited.
No window/smoothing settings or claims of mathematical review were added to
the calculation-review CSV.

The user prefers keeping the common 0.5-second, non-overlapping intervals used
by the other calculations and is open to reduced spectral smoothing. Mixed
window durations/overlap have not been approved. A focused comparison is
proposed before choosing settings: retain 0.5 seconds while reducing NW, and
compare against a small number of longer-window alternatives. Assess frequency
leakage, null bias/variability, brief changes, undefined results and runtime;
greater smoothness alone is not evidence of greater accuracy. The current
workflow uses one window length/step for all metrics. At a 0.5-second step,
1/2/4-second windows overlap by 50/75/87.5 percent, with 599/597/593 complete
windows in 300 seconds, versus 600 half-second windows. For a fixed taper count,
they process about 2/4/8 times as many samples across windows; these are workload
counts, not measured runtime multipliers. Overlap adds evaluations of shifted
windows; it does not change the estimate for an identical individual segment.

The user subsequently clarified that the final analysis averages each metric
over all windows. The approved focused experiment therefore prioritizes changes
in recording-mean connections, within-patient connection rankings, and runtime,
with temporal localization secondary. Noise reduction by averaging must be
distinguished from estimator bias, which can persist in the mean. The supplied
`/home/dan/data/IEEG/connectivity` stores contain electrode metadata/EZ targets
for 69 patients; matching raw EDFs are in the adjacent `baseline_edfs` directory.
The exploratory pilot uses a small contact sample from five full recordings
selected by acquisition rate/contact count, without using EZ labels. It cannot
establish EZ discrimination or generalization to held-out patients. The user's
edge-to-electrode feature reduction and historical preprocessing are not yet
confirmed. The experiment is in `audit/checks/probes/spectral_window_smoothing.py`;
it is now complete and documented in
[spectral_window_smoothing_review.md](spectral_window_smoothing_review.md).
Decision after the pilot: keep the existing 0.5-second non-overlapping windows
and NW=3/five tapers. Interpretation must account for their frequency smoothing
and estimation limitations. No overlap, reduced smoothing, further parameter
search or EZ evaluation is required for this refactor. A new smoothing-parameter
API is not required by this decision. No production window/smoothing changes
were made. The pilot does not establish EZ predictive value.

One separate aggregation question remains: if coherence-phase angles are
averaged across windows arithmetically in the downstream analysis, that would
need circular aggregation as well. The downstream implementation has not been
inspected, so this is a question to verify, not a confirmed pipeline defect.

## Group-delay significance follow-up

The user accepts frequency-wise correction separately for each pair/window,
conditional on reasonable runtime. An isolated 118-channel benchmark found the
per-pair BH prototype faster than pooled BH. Historical output inspection
confirmed extensive missingness, but does not establish each saved NaN's cause.
A focused null simulation also confirmed a separate calibration defect in the
installed Fisher significance test; current upstream documents its replacement.
The user approved that correction; zero-null significance and per-pair BH are
now implemented with batched selection/regression and per-pair diagnostics.
The focused EEG sample still has substantial missingness from significance
and frequency-span requirements, without numerical failures. Findings, measured
NaN rates and timing limits are in [group_delay_review.md](group_delay_review.md).

## Approved completion scope

After reviewing the remaining recommendations, the user approved all four
choices below. Implementation results follow this decision record.

### Envelope correlation (PEC)

- Keep envelope correlation of the supplied signal, without an internal band
  filter. Broadband input therefore gives broadband envelope correlation;
  externally filtered input determines its own frequency scope. No additional
  band-specific variants or filtering defaults are part of this completion.
- Preserve the six existing definitions: amplitude envelopes for ordinary PEC,
  log-power envelopes for the log variants, and pairwise orthogonalization for
  the orth variants. Orthogonalized results average the two directional
  correlations. The absolute variants take each directional coefficient's
  absolute value before averaging. Correct signed/unsigned labels and document
  the amplitude-versus-power distinction and operation order explicitly.
- Correctly center correlation inputs and handle numerical degeneracy. Constant
  envelopes have undefined correlation. An exactly zero orthogonal component
  also gives undefined correlation, not evidence of zero coupling; rounding
  residue must not become apparent coupling. Return NaN with an unavailable
  reason for undefined results. Use numerically justified, scale-aware handling
  that preserves valid small signals; do not introduce a weak-signal cutoff or
  an arbitrary floor for logarithms.
- Share the directional calculations between signed and absolute variants,
  reuse appropriate intermediate work, and remove redundant averaging over a
  single epoch. Measure the complete family workload. The historical audit's
  approximately 0.58 s/window for six variants is a baseline observation, not
  a promised new runtime or measured improvement.

The [PEC audit](../audit/notes/pec.md) describes the original implementation.
Its references to a definitional zero for duplicate channels concern the zero
orthogonal component; the approved correlation result is NaN when its variance
is zero. The [MNE API](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.envelope_correlation.html)
documents log-power and absolute-before-symmetrization options. Its
[frequency-specific example](https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_envelope_correlation.html)
filters the supplied signal before calculating envelopes.

### dDTF normalization grid

Use the full nonnegative native, unpadded frequency grid for the supplied window
and sampling frequency. Normalize by the discrete sum across that grid and
sources before selecting a reported band; every band uses the same denominator.
Include the native DC and, for even-length transforms, Nyquist endpoints once.
If numerical factorization requires internal padding, evaluate and normalize
the final transfer-based result on the original grid. Padding must not rescale
the statistic merely by adding normalization bins. This is the approved standard
squared full-frequency DTF times squared partial coherence, followed by the
requested band mean/max; no frequency-mean or continuous-density replacement
is introduced under the same identifier. Different native window lengths can
still change the discrete normalization scale. The user's fixed 0.5-second
windows avoid that particular cross-window-length comparison.

### Implementation and completion checks

Complete the already approved directed-family formula and source-first output
changes, investigate and repair pair-order-sensitive factorization, then reuse
one consistent fit for both directions and applicable family members. Apply
mathematically justified pair-local failure handling with recorded reasons;
weak coupling and insignificant rounding differences are not failures. Preserve
pairwise interpretation and the chosen window/taper settings. A materially
different estimator or substantial runtime tradeoff requires further discussion.

Verify changed formulas against independent references and known models; cover
channel permutation, direction, degenerate inputs, cache ownership and sharing
where affected. Run focused tests and affected regressions only. Re-run passing
checks only when subsequent changes or evidence justify it, and update only
affected numerical goldens with provenance. Measure representative complete
family workloads and report practical timing limits. Update calculation-review
CSV notes only for completed work, using brief timing/TLDR notes and preserving
the user's other fields. Existing group-delay and window-pilot results remain
completed evidence, not tasks to repeat. Commit or push only when requested.

### Working config membership and runtime constraint

The user also requires a runnable analysis config that reflects the approved
additions and removals from this review, including earlier basic/phase variant
changes. The starting selection was `pyspi/workflow/retained185.yaml` at cb60808;
the historical count of 185 is not a requirement for the revised selection.
Reconcile the final identifiers and parameters, applicable packaged catalogs,
workflow selection, documentation and affected config checks so the chosen
calculations are actually requested and removed definitions are not requested.
Keep unrelated selections intact. Report the final additions, removals and
count. That preset contained ten group-delay band configurations,
contrary to an earlier review statement; remove those from the working analysis
selection under the user's accepted short-window GD recommendation. Corrected
GD remains available as a calculation for suitable explicit configurations.

Enabling dDTF or any other candidate addition depends on practical runtime.
Compare candidates with the calculations in the original retained selection on
the same representative workload. Measure both complete family cost and the
actual incremental cost of adding candidates to the shared workflow, including
any new factorization work. Do not count cached fits as free unless the enabled
selection already computes and retains them. Use measured comparable timings
where available; do not extrapolate a full-channel cost from a two-channel fit
or promise speedups from operation counts alone.

Report seconds per representative 118-channel, 1024-sample window, then simple
sequential estimates for 600 windows and 600 x 69 windows. State limitations
from differing channel counts, data-dependent convergence, parallelism and I/O.
Candidates that remain clearly substantially more expensive than the relevant
originally selected calculations should stay out of the working run config;
being repaired or available in the library does not require enabling them.
The user has not selected a numerical cutoff. Borderline cost/benefit cases
should be presented with measured timings before inclusion, while other work
continues. Do not compensate for expensive additions by silently weakening
their definitions or changing the settled window/taper settings. Final config
membership follows these measurements, as recorded below.

## Implemented refactor and numerical policy

The old finite-grid projection discarded the even-length midpoint lag and used
a triangular zero-lag convention. The replacement splits both self-mirrored
lags equally, retains positive lags and starts from a symmetric positive square
root. It satisfies the discrete Hermitian split identity without imposing a
channel order. This is a Wilson factorization correction, not a new conditioning
model. See the [Wilson–Burg description](https://sepwww.stanford.edu/data/media/public/docs/sep107/paper_html/node38.html)
and the [matrix plus-operator derivation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10217183/).

Both reconstruction residual and relative iterate change must be at most 1e-10,
with at most 100 iterations. Checks operate after diagonal channel-power
equilibration; numerical rank uses `channels * eps * largest_eigenvalue`.
These are numerical-resolution criteria, not minimum coupling or significance
thresholds. Nonfinite, invalid, unresolved singular and nonconvergent fits have
separate reasons. No padding, ridge or fallback estimator is introduced.
Accurate finite-grid reconstruction still does not establish statistical
accuracy or remove impulse-response aliasing from short observations.

Independent exact VAR tests recover transfer functions and correlated innovation
covariances, including odd/even grids and extreme channel-unit changes.
Estimated-spectrum tests verify channel permutation; one-way coupling verifies
all six public nonparametric directions and parametric Granger independently.
The representative 118-channel benchmark fit all 6,903 unique pairs successfully.
Parametric Granger retains its existing model/order selection but now isolates
failed pairs and uses the same public source-first convention.

Calculator registers requested directed bands/families before execution, then
streams bounded pair batches through those summaries. Both directions use the
same fit. PDC/DTF and generalized PDC/DC reuse their equal off-diagonal results
under pairwise fitting. Full pair spectra are not retained across the entire
network. Standalone calls share band summaries and a bounded pair cache.
dDTF's normalization uses all native nonnegative bins exactly once, independent
of the reported band; FFT-speed padding is removed for directed calculations
by reusing the already tapered samples. Nonparametric Granger uses a stable
correlated-innovation power decomposition and preserves genuine zeros.

All six PEC definitions share the analytic transform, pair products and
applicable directional correlations. Both correlation inputs are centered.
Log power is evaluated without squaring raw amplitudes, and scale-aware
arithmetic bounds distinguish unresolved cancellation from small valid
variation. Undefined correlations return NaN with reasons; signed versus
absolute operation order and metadata are explicit. The former MNE delegate
and redundant single-epoch averaging path are removed. Detailed definitions
and diagnostic APIs are in [calculation_definitions.md](calculation_definitions.md).

## Measured runtime and selection gate

Receipts: [review selection](../tests/review_selection_results.json) and
[PEC before/after](../tests/envelope_refactor_results.json). Reproduce with
`tests/benchmark_review_selection.py` and `tests/benchmark_envelope.py`.
These use synthetic 118-channel, 1,024-sample windows, fresh Data caches,
one numerical-library thread and three alternating repeats. The original
comparison selection is frozen at cb60808. Its **120 related spectral, PEC and
cross-correlation outputs**, excluding GD, are evaluated with the corrected
current code. This is not a benchmark of all 185 historical calculations.

| Workload | Seconds/window | x600 windows | x600x69 windows |
| --- | ---: | ---: | ---: |
| Existing related selection, in dDTF comparison | 6.42 | 1.07 h | 73.8 h |
| Add dDTF, all 11 bands, after existing shared work | 66.85 | 11.14 h | 768.7 h (32.0 d) |
| Existing selection plus dDTF | 73.27 | 12.21 h | 842.6 h (35.1 d) |
| Add complete directed family, 66 requests, shared fits | 57.91 | 9.65 h | 665.9 h (27.7 d) |
| Existing selection plus complete directed family | 65.09 | 10.85 h | 748.6 h (31.2 d) |
| All six PEC variants before refactor | 0.833 | 8.33 min | 9.58 h |
| All six PEC variants after refactor | 0.775 | 7.75 min | 8.91 h |

The family rows include every requested directed output and the new pair fits;
the preceding existing calculations supply their shared transform, not cached
factorizations. The combined rows include that existing work as well. No
factorization cost is being treated as free. Adding all families does not
multiply fit cost by six. Nonetheless, these additions remain substantially
more expensive and are **disabled** in the working config.

Machine-load variation was substantial: dDTF took 52.6–79.1 s and the complete
family 53.2–81.8 s. Their overlapping ranges do not establish that all families
are faster than dDTF alone. Combined table rows sum the measured stage medians.
PEC's approximately 1.08x improvement is modest; its main gains are correctness
and shared intermediate work, not a large promised speedup.

The final signed/unsigned phase grouping was checked separately because
appending signed variants after all original metrics can evict shared arrays.
The 22 signed-phase outputs plus 8 cross-correlation additions took 0.59–1.44 s
in added calls, about 5.9–14.4 min for 600 windows or 6.8–16.6 h for 600x69.
Summing the two stage medians gives 1.07 s, 10.7 min and 12.3 h respectively.
Baseline-only passes varied from 4.99 to 16.89 s; subtracting total medians is
therefore unreliable (the receipt preserves those observations separately).
These figures describe timed extra calls after shared work, not a precise
prediction of the net whole-run difference. After reviewing this runtime range,
the user approved keeping all 30 additions in the working selection.

All extrapolations are sequential arithmetic, not 69-patient runtime forecasts.
I/O, workflow scheduling, preprocessing and unrelated calculations are excluded.
Patient channel counts, data-dependent convergence, cache pressure, parallelism
and machine contention can materially change wall time. These benchmarks do
not measure EZ discrimination or physiological validity. The completed EEG GD
and window/smoothing investigations were not repeated.

## Working configuration and validation

`pyspi/workflow/reviewed.yaml` replaces the former count-named preset, without a
legacy alias. The final selection has **205 metrics: 185 - 10 + 22 + 8**.
It removes all ten GD bands; adds signed PLI and signed wPLI on the same eleven
bands each; and adds unrestricted signed cross-correlation minimum/peak plus
six significant-only summaries (max, max of squares, mean, mean of squares,
minimum and signed strongest peak). The original four unrestricted summaries
remain. All 30 additions were retained after the measured-runtime discussion.

All six PEC variants and the previously reviewed basic/phase calculations remain
available with their corrected definitions. Ordinary coherence-phase maxima
were already absent from this run selection and remain removed from all/fast
catalogues, which still contain 298/230 metrics. Factorized directed metrics,
including dDTF, are repaired but disabled. No new conditional or band-filtered
PEC estimator is added. Unrelated selected identifiers/parameters are preserved.

Validation includes independent scalar/matrix definitions, exact VAR models,
permutation and source-first checks, pair-local failures, native-grid dDTF,
cache invalidation/ownership and enabled-only Calculator sharing. PEC tests
cover six variants, signed/absolute order, constant/duplicate/near-proportional
channels, analytic input, tiny resolved variation and extreme amplitude units.
All focused checks passed; the final affected regressions passed **73 checks**
for the small/preprocessing/catalogue set and **60** for the larger fixture.
Two additional affected cache checks and three config/report checks passed.
Already passing unrelated suites were not rerun.

Only 60 directed/PEC arrays per main golden set and 8 preprocessing arrays were
regenerated after the independent checks. The other 238 arrays in each main set
and 88 preprocessing arrays were verified byte-identical. Prior GD and phase
provenance is retained. The preprocessing catalogue-order metadata also drops
three obsolete phase-max names; no preprocessing numerical array used them.
The scoped refresh script and each metadata file record sources, environment,
arguments and affected identifiers. CSV notes contain brief verified corrections
and measured timings; other cells and unrelated records are preserved.
