# Directed spectral follow-up, 2026-10-27

Directed coherence now uses the approved standard squared definition after
discussion and verification against current upstream. The remaining work here
is diagnostic: matrix orientation, other directed formulas, factorization,
padding, window length and taper settings remain unchanged.
These six directed variants are absent from the current `retained185` preset;
the findings apply when they are selected through other presets or custom
configurations. Preset membership does not limit which justified fixes are made.

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
outputs to source-first order; that is a separate change and is not adopted
here. The local convention remains row=target, column=source.

Current upstream dDTF uses full-frequency squared DTF times squared partial
coherence. The installed method uses a product of magnitudes with PDC instead.
This additional definition issue remains to be discussed and tested in detail.

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

## Pairwise versus conditional calculations

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
estimator and need discussion; replacing an inverse with a pseudoinverse is
not an automatically equivalent fix.

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
temporal resolution and frequency bands before selecting duration or NW.
No window/smoothing settings or claims of mathematical review were added to
the calculation-review CSV.
