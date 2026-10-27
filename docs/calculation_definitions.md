# Reviewed calculation definitions

This page records operation order and the meaning of variants that have been
reviewed. It is not a claim that every pySPI calculation has been reviewed.
Performance notes belong in `audit/calculation_review.csv`; implementation
history belongs in Git.

## Cross-correlation

For two length-T channels, subtract each channel's full-window mean. For lag h,
sum the overlapping products and divide by the full-window energies:

```
r(h) = sum_t x_centered[t] * y_centered[t-h]
       / sqrt(sum_t x_centered[t]**2 * sum_t y_centered[t]**2)
```

The numerator includes only indices present in both channels. The denominator
does not shrink with overlap. Zero lag equals Pearson correlation. This is not
Pearson correlation recomputed on each overlapping segment, and there is no
T/(T-1) multiplier. Any configured `Data` preprocessing runs before this step.

The searched lags are **-L through +L, inclusive**, including zero, where
`L = floor(fs * max_lag_seconds)`. The default time limit is **0.125 seconds**.
At **2048 Hz**, that means **-256 through +256 samples**, or 513 lag values.
The limit is independent of window length. A requested L >= T is rejected.
An exact integer product is preserved despite floating-point rounding.

The workflow supplies its **target** sampling rate, including after resampling,
and rejects a conflicting `fs` in the metric configuration. Direct API and
standalone Calculator callers must set `fs` to their actual sampling rate:

```python
CrossCorrelation(statistic="min", sigonly=False,
                 fs=2048, max_lag_seconds=0.125)
```

Outside the workflow, the default `fs=1` means one sample per time unit;
with the default 0.125-unit limit only zero lag is available. It does not
infer an EEG rate from an array. Use a custom Calculator YAML configuration
with `fs: 2048` for a 2048-Hz input. Identifiers include sampling rate and lag
limit, plus the inference method and alpha when applicable.

### Summaries and squares

First compute the signed curve. Then select lags. Then square each selected
value if requested. Finally aggregate. For selected values **[-0.9, 0.2]**:

| Configuration | Meaning | Result |
| --- | --- | ---: |
| `statistic="max"` | Largest signed value | 0.2 |
| `statistic="min"` | Smallest signed value | -0.9 |
| `statistic="peak"` | Value with greatest magnitude, retaining its sign | -0.9 |
| `statistic="mean"` | Mean signed value | -0.35 |
| `statistic="max", squared=True` | max(r²) | 0.81 |
| `statistic="min", squared=True` | min(r²) | 0.04 |
| `statistic="mean", squared=True` | mean(r²) | 0.425 |

Thus max(r²) differs from max(r)² = 0.04, and mean(r²) differs from
mean(r)² = 0.1225. Squared minimum describes the **weakest magnitude**;
it does not describe the strongest negative relationship. It is supported by
the API but is not added to the default metric catalogue. Squared `peak` is
rejected because it duplicates `max` with `squared=True`.

Minimum is a numerical minimum: when all values are positive it returns the
smallest positive value. Maximum can be negative when all values are negative.
A negative sign describes opposite fluctuations, not which channel leads.
These summaries are undirected: swapping channels reverses the lag curve and
leaves the summary unchanged. They do not return the peak's lag or infer
causality.

If the largest positive and negative magnitudes tie, `peak` returns NaN with
`ambiguous_peak`. Ties include roundoff within 32 machine epsilons relative to
the magnitudes. This avoids choosing a sign based on channel or lag ordering.
An all-zero selected curve has an unambiguous peak of zero.

### Significant-lag selection

`sigonly=False` selects every searched lag. `sigonly=True` uses a two-sided,
autocorrelation-adjusted **approximate** test, then Holm correction over all
searched lags **within each channel pair**. It retains every qualifying lag,
including separated peaks away from zero. The same signed-lag selection is
used by all summaries and squared variants. Default `alpha=0.05` is configurable.
There is no correction across channel pairs or across windows.

The null is independence of covariance-stationary channels. Let a_x(k) and
a_y(k) be their full-window, energy-normalized sample autocorrelations, and
M = T - abs(h). The plug-in null variance is

```
v(h) = [M + 2 * sum_{k=1}^{M-1} (M-k) * a_x(k) * a_y(k)] / T**2
p(h) = 2 * NormalCDF(-abs(r(h)) / sqrt(v(h)))
```

The overlap weights count covariance terms in the sum of M products; all
available overlapping autocorrelation terms are included. This is a
finite-overlap adaptation of Bartlett's variance, followed by Gaussian tails;
it is **not** an exact finite-sample test or an implementation of the modified
Student t test in [Cliff et al., Appendix D](https://arxiv.org/html/2003.03887v3#A4).
The [Holm procedure](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
is checked against statsmodels. Its nominal error control still depends on
the approximate input p-values being suitable for the data.

Strong autocorrelation, narrow bands, short records and nonstationarity can
make the approximation inaccurate or conservative. Missing significant-lag
results are not proof of no relationship. The reproducible simulation audit
in `tests/validate_xcorr_inference.py` checks independent white noise, several
AR processes, two filtered-noise bands, and delayed coupling. Its results are
in `tests/xcorr_inference_results.json`; this is not calibration on the user's
recordings. No surrogate resampling is performed.

### Missing values and diagnostics

Each call exposes `spi.last_diagnostics`, including per-pair `status` and
`selected_lag_count`. Returned arrays and diagnostic arrays are caller-owned.

| Status | Meaning |
| --- | --- |
| `ok` | Summary available |
| `no_significant_lags` | Calculation succeeded, but no searched lag passed selection |
| `undefined_input` | Constant or nonfinite channel |
| `undefined_inference` | Fewer than three observations or unusable estimated variance |
| `ambiguous_peak` | Opposite-sign strongest magnitudes tie |
| `diagonal` | Self-pairs are intentionally not summarized |

Missing summaries are NaN. One warning per metric call reports the number of
unique pairs for each reason; Calculator and workflow metric reports retain
these warnings. Actual exceptions remain failed calculations. Pair-specific
diagnostic matrices are available on the SPI instance; workflow journals save
the warning counts, not these matrices.

One realisation and at least two observations are required. No qualifying
lags are replaced by zero, no central-lobe fallback is used, and no previous
calculation path is retained.

### Shared work and configuration

Channels are centered and transformed once per input. Batched inverse FFTs
evaluate each unique channel pair once, with a bounded curve cache. Summaries
share both curves and lag selection. Data mutation invalidates shared work.
The all/fast catalogues include signed `min` and `peak` with both selection
modes. `retained185.yaml` keeps its four existing max/mean, ordinary/squared,
`sigonly=False` requests; adding variants there is an explicit analysis choice.

## Spearman, Kendall, covariance and precision

Spearman is Pearson correlation of per-channel average ranks. Kendall is
tie-corrected tau-b. Their `squared=True` variants square the **final pair
coefficient**: rho² or tau². They have no lag aggregation.

Covariance and precision squared variants square each entry of the fitted
matrix. They do not square the input, perform a matrix product, or invert a
squared covariance matrix. Fitted estimators are shared, with returned arrays
copied so callers cannot alter subsequent results.

## Spectral signs and ordered pairs

Multitaper PSI, group delay, and coherence phase have **signed, directed**
metadata and return `issigned() == True`. Here directed means the ordering of
the channel pair matters; it does not by itself establish a causal effect.
The common transform uses `C[i,j] = S[i,j]/sqrt(S[i,i]*S[j,j])`, with
`S[i,j] = mean(X[i]*conj(X[j]))`.

For a pure delayed copy, positive PSI and positive group delay at `[A1,B1]`
mean **A1 leads B1**. Reversing the pair reverses the sign. Group delay is in
seconds if `fs` is in Hz, or samples if `fs=1`. Coherence phase is the phase
of `C[i,j]` in radians. Its sign describes a wrapped phase difference, not
an unambiguous broadband time delay.

Group delay offers three explicitly named results:

| Statistic | Definition | Reverse pair |
| --- | --- | --- |
| `delay` (default) | Fitted phase slope divided by 2π | Negated |
| `slope` | OLS slope of unwrapped phase against frequency | Negated |
| `rvalue` | Pearson correlation between frequency and unwrapped phase | Negated |

The sign of `rvalue` was previously copied unchanged into both matrix
triangles, causing a channel-order dependence. It now follows the signed
regression definition. `rvalue` is **r**, not r² or absolute r; r² describes
the fraction of phase variance explained by this fitted line. This matches
the [SciPy regression result definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html).
The coherence significance rule and frequency-point selection have not been
changed in this sign review.

Coherence phase (`phase_multitaper_mean`) uses an **equal-weight circular
mean** of the coherency phases across the inclusive band:
`angle(sum(C(f) / abs(C(f))))`. This follows the
[circular-mean definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).
Each defined frequency has equal weight, including very small nonzero
coherency. There is no coherence-strength weighting or significance cutoff.
For example, +179° and -179° average to 180°, and phases 0° and 90° average
to 45° even if their coherency magnitudes are 0.9 and 0.1.

Zero or nonfinite coherency has no usable phase and is omitted. An empty band,
all-undefined band, or cancelling unit vectors returns NaN. Cancellation is
checked against a float64 summation/normalization roundoff bound, not a
statistical threshold. Results remain radians in [-pi, pi]; +pi and -pi are
the same direction, so reversal is antisymmetric modulo 2*pi.

Ordinary phase `max` is removed and rejected at construction: it depends on
where the angle wraps rather than a meaningful largest phase. All three mean
bands remain in the default and fast configurations; the retained selection
already contained only phase means. Shared coherency and bounded temporary
arrays avoid a full extra tensor of angles. Wavelet phase has a separate
orientation/estimation path and is not changed by this Fourier-phase review.

The factorization-based directed metrics currently use **row = target,
column = source**, the transfer-matrix convention. This differs from the
positive-lead interpretation above; no matrix transposition was made in this
review. Synthetic one-way-coupling checks confirm that convention. Reversing
the input order also changes fitted values beyond rounding error; that
numerical factorization issue remains under review. The parametric Granger
path and wavelet sign conventions have not been reviewed here.

## Directed coherence

`dcoh` returns **squared directed coherence at each frequency**, followed by
the requested band mean or maximum. For target `i` and source `j`:

`DC[i,j](f) = variance[j] * abs(H[i,j](f))**2 / sum_k(variance[k] * abs(H[i,k](f))**2)`.

`H` maps the model's innovations into the observed channels; `variance[j]`
is source j's innovation variance. The denominator includes the target's own
innovation. Defined values lie in [0, 1] and sum to 1 across sources within
each fitted model, before pyspi masks the self-entry. The wrapper fits each
ordered pair separately, so the assembled off-diagonal multichannel matrix
does **not** have a row-sum-one guarantee.

This matches the [current upstream squared DC formula](https://spectral-connectivity.readthedocs.io/en/latest/_modules/spectral_connectivity/connectivity.html#Connectivity.directed_coherence).
It replaces the installed dependency's mixed powers and incorrectly oriented
variance weights. A common scale is removed from weighted transfer amplitudes
before squaring to avoid overflow/underflow from channel-unit changes. Zero
total inflow gives NaN, without an absolute epsilon floor or clipping. Existing
band mean/max omit undefined bins; an all-undefined band remains NaN.

Squaring happens before the frequency summary. For per-frequency magnitudes
`[0.2, 0.8]`, the squared band's mean is `mean([0.04, 0.64]) = 0.34`, whereas
the square of the magnitude band's mean would be `0.25`.

This diagonal-variance normalization assumes uncorrelated innovations when
interpreted as shares of the target's observed power. With correlated
innovations it omits cross terms; interpret it as normalized weighted transfer
strength. Matrix orientation, pairwise fitting and factorization policy are
unchanged by this formula correction. Both band summaries and every frequency
band use the corrected definition, independently of preset membership.

## Phase-lag, phase-locking and debiased variants

For each frequency and channel pair, let `y[k] = Im(X_i[k] * conj(X_j[k]))`
for the `N` taper/trial observations. These observations are averaged before
the frequency-band summary; `N` is not the number of time samples or bins.
Write `s[k] = sign(y[k])`, including `sign(0) = 0`.

| Identifier | Per-frequency result | Interpretation |
| --- | --- | --- |
| `pli` | `abs(mean(sign(y)))` | Nonnegative phase-lag consistency |
| `pli-signed` | `mean(sign(y))` | Consistency retaining phase-lag sign |
| `wpli` | `abs(sum(y)) / sum(abs(y))` | Nonnegative, magnitude-weighted lag consistency |
| `wpli-signed` | `sum(y) / sum(abs(y))` | Weighted consistency retaining phase-lag sign |
| `dspli` | `(sum(s)**2 - sum(s**2)) / (N * (N - 1))` | Bias-corrected estimate of squared PLI |
| `dswpli` | `(sum(y)**2 - sum(y**2)) / (sum(abs(y))**2 - sum(y**2))` | Debiased estimate of squared wPLI |
| `plv` | `abs(mean(z))`, where `z = cross / abs(cross)` | Phase locking at any consistent phase offset |
| `ppc` | `(abs(sum(z))**2 - N) / (N * (N - 1))` | Bias-corrected estimate of squared PLV |

The existing `pli`/`wpli` identifiers now use the standard nonnegative definitions.
Their magnitudes are taken **at each frequency before** `mean` or `max` across
the inclusive band. Signed frequency values `[0.6, -0.6]` therefore give unsigned
mean `0.6` and signed mean `0`, rather than applying absolute value after averaging.
See [the original PLI/wPLI debiasing paper](https://www.psychologie.hhu.de/fileadmin/redaktion/Fakultaeten/Mathematisch-Naturwissenschaftliche_Fakultaet/Psychologie/CompPsy/Papers/pennartz_2011.pdf)
and [the PPC/PLV relationship](https://pmc.ncbi.nlm.nih.gov/articles/PMC3674231/).

Signed variants are separate classes (`SignedPhaseLagIndex`,
`SignedWeightedPhaseLagIndex`). Reversing the channel pair negates each frequency
value and its band mean. Their `max` is the ordinary numerical maximum: for
`[-0.9, 0.2]` it returns `0.2`, whereas unsigned `max` returns `0.9`. A signed
maximum need not be antisymmetric. These signs describe phase-lag orientation;
they do not establish causality or an unambiguous broadband time delay.

The debiased estimators and PPC can legitimately be negative. For example,
`N=5` and `PLI=0.2` give debiased squared PLI `-0.2`. That is a finite-sample
bias correction, not negative coupling or phase-lead direction. Their values
are retained without absolute value or clipping, and their metadata now says
`signed` and `undirected`. They do not need separate phase-direction variants.
The [dependency's estimator range documentation](https://spectral-connectivity.readthedocs.io/en/latest/CONNECTIVITY_METRIC_RANGES.html)
also distinguishes these negative estimates from directional signs.

Shared work: PLI and its squared debiased estimate reuse the sign average and
a compact count of nonzero signs. The count is reduced from the already formed
sign array, without enumerating observation pairs or repeating taper products.
PLV and PPC reuse the normalized phase vectors. Weighted estimators request
their missing reductions together, forming each bounded taper cross-product
once per request. Already cached reductions are reused. Signed and
unsigned outputs share the signed spectrum; the unsigned band summary creates
only its requested magnitude slice. Band selection uses contiguous views of
the shared sorted frequency grid, avoiding a redundant tensor copy. No separate
full-frequency unsigned spectrum is cached by the SPI wrappers.
Phase reductions use upper-triangle channel tiles and fill the reverse through
the appropriate conjugate, sign reversal or symmetry, avoiding repeated work.

The `all` and `fast` catalogues include both signed variants with mean/max and
three bands; their sizes are 301 and 233. The selected 185-metric EEG workflow
keeps its metric selection; its existing PLI/wPLI names now use the corrected
unsigned definitions. Added signed variants can be selected explicitly.

### Zero signs and undefined weighted ratios

The wPLI denominator is no longer compared with an absolute machine-epsilon
cutoff. Changing amplitude units must scale its numerator and denominator
together, preserving their ratio. A genuinely zero denominator produces NaN
for both signed and unsigned wPLI.

Debiased PLI subtracts the actual number of nonzero signs, `sum(s**2)`, rather
than assuming all `N` signs have magnitude one. This is the distinct-observation
pair-product formula, evaluated in linear time through reductions. Five zero
signs therefore produce **0**, rather than the old **-0.25**. For signs
`[1, 1, -1, 0, 0]`, the result is **-0.1**, rather than **-0.2**. Negative
estimates from mixed positive/negative signs remain valid and are not clipped.

Debiased wPLI uses the algebraically equivalent ratio of distinct-observation
products. Within each bounded block, imaginary values are scaled by their
largest magnitude per frequency/pair; this common positive scale cancels from
the ratio. Prefix sums then accumulate products of distinct observations in
linear time. This avoids overflow/underflow from directly squaring raw
amplitudes and cancellation in `sum(abs(y))**2 - sum(y**2)`. For example,
`y = [1, 1e-20, 0, 0, 0]` has a positive pair-weight denominator even though
the squared-sum subtraction rounds to zero. Its debiased wPLI is **1**.

Zero pair weight produces NaN: debiased wPLI needs at least two observations
with nonzero imaginary values. Ordinary wPLI can still be defined when only
one observation has a nonzero weight. Undefined bins are skipped by existing
`nanmean`/`nanmax` band summaries; an entirely undefined band remains NaN and
emits the existing NumPy warning. Frequency selection stays inclusive, including
DC/Nyquist; these bins are undefined for weighted lag ratios when their
cross-spectra are purely real. PLI and corrected debiased PLI instead give zero
for those zero imaginary observations.

The statistical assumptions behind taper-based debiasing remain a separate
review item. The formulas and sign conventions of the wavelet implementations
were not changed here.
