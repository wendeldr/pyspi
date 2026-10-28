# Reviewed calculation definitions

This page records operation order and the meaning of variants that have been
reviewed. It is not a claim that every pySPI calculation has been reviewed.
Performance notes belong in `audit/calculation_review.csv`; implementation
history belongs in Git.

## SPI identifiers

Identifiers follow one grammar (`pyspi/_naming.py`), so they can be parsed and
used unchanged as file names or table columns:

```
<measure>[_<variant>...][_<parameter>...]
cohmag_mean_band0030-0070hz
xcorr_sq_holm_mean_maxlag0125ms_alpha0p05
gc_gaussian_targethistory01_targetdelay01_sourcehistory01_sourcedelay01
```

- Only `a-z`, `0-9`, `_` and `-` occur. `_` separates fields, `-` only joins
  the two ends of a range, and `p` is the decimal point (`0p05` = 0.05).
- The first field is the measure. Variant fields contain no digits; they are
  transforms (`sq`, `abs`, `log`, `orth`), the method or estimator, the
  significance test (`holm`) and the statistic, in that order. A boolean option
  appears only when it is on (`biased`, `normalized`).
- Parameter fields are `<name><number or range><unit>`. A number without a unit
  counts samples. Numbers are written exactly and zero-padded to a minimum
  width, so identifiers in one family line up and sort numerically; a wider
  value is written in full.
- Frequency bands are in Hz (`band0030-0070hz`) when `fs` is the sampling rate.
  With the default `fs=1` the rate is unknown and the band is in cycles per
  sample (`band0p25-0p5cyc`). Both describe the same bins: `fs=2048, fmin=30,
  fmax=70` and `fs=1, fmin=30/2048, fmax=70/2048` give the same values to
  within rounding (about 1e-15).
- Values chosen automatically become variant words (`autoorder`, `autodim`,
  `autoembed`, `theilerauto`), so parameter fields always hold numbers.

`docs/metric_names.csv` maps every identifier produced by the shipped
configurations before this grammar to its current form; the `configs` column
names the files that produce it. The workflow preset `reviewed.yaml` now gives
its bands in Hz at `fs: 2048` (the workflow rejects any other
`sampling_rate_hz` for them), so its old fractional names map to Hz names,
while the same old names in the library configs map to cycles per sample.

## Ridge regression error

`lmfit_ridge_penaltysum_alpha1` reports the in-sample mean squared error of predicting channel j
from channel i, including an unpenalized intercept. It minimizes the **sum** of
squared residuals plus the squared slope (Ridge penalty alpha=1):

```
b = sum(x_centered*y_centered) / (sum(x_centered**2) + 1)
MSE[i,j] = mean((y_centered - b*x_centered)**2)
```

Lower values mean greater linear predictability. This is neither an
independence test nor evidence of causal direction. With the same sample count
and ddof=1 normalization for each channel, the result is a symmetric, affine
decreasing function of squared Pearson correlation. With unnormalized inputs,
the target's units and variance matter and the two directions can differ.
All channel means and cross-products are shared, replacing separate model fits.

`alpha` is configurable. `penalty_scale="sum"` minimizes `SSE + alpha*b**2`;
`penalty_scale="mean"` minimizes `MSE + alpha*b**2`. Thus mean scaling uses
`alpha*T` in the coefficient's sum-of-squares denominator. The former
`lmfit_SGDRegressor` is replaced by
`lmfit_ridge_penaltymean_alpha0p0001`, the exact solution to its squared-loss
plus L2 objective. The stochastic optimizer, its seed effects and its early
stopping error are removed. No SGD compatibility alias is provided.

Both Ridge variants remain available in the library but are excluded from the
working normalized fixed-length run because selected squared empirical
covariance already determines them, including after window averaging.

## Lasso, ElasticNet and BayesianRidge errors

All report the in-sample MSE of predicting the column channel from the row
channel, with an unpenalized intercept; they share centered observations and
cross-products with Ridge. Penalties are applied to the coefficient before
computing the unpenalized prediction error reported by the metric.

Lasso minimizes `MSE/2 + alpha*abs(b)`. ElasticNet minimizes
`MSE/2 + alpha*l1_ratio*abs(b) + alpha*(1-l1_ratio)*b**2/2`.
They now use exact one-predictor minimizers. The library defaults remain
alpha=1, with ElasticNet L1 ratio=0.5; explicit nondefault parameters appear
in identifiers. In normalized windows, default Lasso is constant. The reviewed
run uses alpha=0.1 after the approved EEG variability/runtime checks. ElasticNet
keeps its existing penalties but no longer reproduces the iterative solver's
premature-zero threshold. Neither penalty cutoff is a significance threshold.

BayesianRidge retains its MacKay updates, four 1e-6 hyperpriors, noise precision
initialization `1/(variance(y)+eps)`, coefficient tolerance 1e-3, maximum 300
iterations, and final posterior-mean update. Iterations stop independently for
each pair. Its default convergence approximation is preserved; an independently
optimized evidence maximum is a different choice. `diagnostics(data)` reports
iterations and reasons, including iteration-limit and numerical failures.

## Euclidean barycenter summaries

The Euclidean barycenter of two channels is `(x+y)/2` at each time sample.
`squared=True` squares this series **before** mean/max/argmax. For a series
`[-0.9, 0.2]`, ordinary maximum is 0.2 and squared maximum is 0.81; it is not
the square of the ordinary maximum (0.04). `max_time` returns a sample index.

Means use shared moments, and extrema use bounded blocks of unique pairs.
The full time series is not retained for every pair. On normalized fixed-length
windows, ordinary mean is zero and mean of the square is an affine function of
empirical covariance. Both are removed only from the working run selection;
the library retains them for other preprocessing. With normalization disabled,
variances and offsets can make these summaries distinct. Ordinary mean is still
zero if detrending/centering removes each channel's mean.

## Regression error causal inference (RECI)

`reci` independently scales each supplied channel to [0,1], then fits
`y = a + b*x**3`. It reports **in-sample** MSE at row i, column j for predicting
j from i. This preserves pyspi's documented fixed-cubic variant. It does not
select a polynomial degree, use held-out samples, or return the difference
between the two directions. Comparing directions requires comparing their
separate errors; a smaller error alone is not proof of causality.

Min–max scaling, cubing and centering happen once per channel; cross-products
provide all pair fits. Positive affine rescaling of a channel leaves the
mathematical result unchanged. A constant channel (allowed only when Data
normalization is disabled) scales to zero. Near exact fits use explicit
residuals to avoid cancellation in the moment formula. Both regressions use
NaN diagonals, reject multiple realizations, and return copies of cached output.

## Gaussian entropy and mutual information

Gaussian joint entropy, conditional entropy, MI and lag-one MI use sample
covariance (N−1) and report **nats**. Entropies are differential entropies:
they depend on amplitude units and can be negative. JE[i,j] = H(i,j),
CE[i,j] = H(j | i), and MI[i,j] = I(i;j). Lag-one MI[i,j] = I(i[t];j[t+1]);
it does not condition on the target's own past. These four Gaussian calculations
use shared native numerical work and require no Java runtime.

For normalized channels, marginal Gaussian entropy is the same constant,
so JE and CE are affine transformations of MI. The equivalence survives
averaging windows; the reviewed normalized selection omits JE/CE. With
normalization disabled, differing channel variances break that equivalence.
Keep amplitude units consistent in the alternative run.

Gaussian MI is zero if either variable is constant. A constant conditioning
source leaves the target's conditional entropy unchanged. Singular differential
entropies are undefined (NaN); exact or numerically unresolved nonconstant
linear dependence gives infinite MI. Near-perfect but resolvable relationships
use direct residuals to avoid artificial infinity from rounded correlations.
There is no variance cutoff in volts, added jitter or ridge penalty. Diagnostics
distinguish constants, singular covariance, nonfinite inputs and insufficient
observations; failure in one pair does not invalidate unrelated pairs.

Formula/covariance provenance is recorded in
[`audit/checks/refs/_kernels_infotheory.py`](../audit/checks/refs/_kernels_infotheory.py)
against the bundled JIDT source; see also the upstream
[Gaussian MI implementation](https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/continuous/gaussian/MutualInfoCalculatorMultiVariateGaussian.java).
The history-based Gaussian measures below also use native numerical work.

## Gaussian Granger / transfer entropy and aligned histories

`gc_gaussian_*[i,j]` is **half** the natural log of the prediction-error
variance ratio: target-only history versus target plus source history.
It is Gaussian transfer entropy in nats; conventional Geweke F is twice this
number. It remains a pairwise calculation and does not account for other
channels. Rows are sources and columns are targets.

For fixed histories, target lags are `1 + k_tau * arange(k_history)` and source
lags are `1 + l_tau * arange(l_history)`. Both models use the same aligned
observations and predict the next sample. At 2048 Hz, consecutive history
lengths 1, 4, 16 and 32 reach back 0.488, 1.953, 7.813 and 15.625 ms; they do
not change the prediction horizon to those values. Outer analysis windows
remain 0.5 seconds without overlap.

Automatic Gaussian embedding retains JIDT's `MAX_CORR_AIS` rule: maximize each
channel's own Gaussian active information storage minus `k/(2*N)`, where N is
the candidate's aligned sample count. Search once per channel, retaining the
first strict maximum; a one-sample history has only one distinct spacing.
For each source/target choice, refit the restricted model on the same
observations as the full model. This is the existing selection rule, not
cross-validated prediction or a jointly selected multichannel VAR order.

Centered observations, aligned histories, target-only fits and cross-products
are shared only when their sample intervals and lag sets agree. Covariance
solves use a reciprocal-condition estimate; unreliable solves and tiny
residuals fall back to least squares on the observations. Numerically redundant
conditioning columns are rank-reduced, without ridge, jitter or a threshold
in physical amplitude units. Saturated fits, nonfinite input and other
failures have separate diagnostics. The derived moment/history cache is
bounded at 256 MiB; large requests use blocks instead of an unbounded global
Gram matrix. Returned values and diagnostics do not expose mutable cache arrays.

## Source-history conditional entropy, causal entropy and directed information

`xme_*_k{k}[i,j]` means `H(j[t] | i[t-1], ..., i[t-k])`: exactly k source
lags, no target-history conditioning. The former k=10 implementation used nine
lags. The identifier remains `xme`, but its descriptive name is **source-history
conditional entropy**; it is unrelated to convergent cross mapping (CCM).

For term m, define `Ypast = (j[t-1], ..., j[t-m+1])` and
`Xpresentpast = (i[t], ..., i[t-m+1])`. Causally conditioned entropy sums
`H(j[t] | Ypast, Xpresentpast)` over m=1..n. Directed information sums
`I(j[t]; Xpresentpast | Ypast)` over the same terms. Each term uses its own
aligned `T-m+1` observations. The default n remains five; a nondefault n is
included in the identifier.

DI includes the present source, so instantaneous relationships contribute.
It differs from strictly past-to-future TE. At n=1 it reduces to ordinary MI.
This follows the conditional-information sum in
[Massey (1990), equation 7](https://www.isiweb.ee.ethz.ch/archive/massey_pub/pdf/BI532.pdf).
The old target-only autocorrelation contribution has been removed across
Gaussian, kernel and Kozachenko variants. CCE and XME are differential
entropies and can be negative. Finite-sample non-Gaussian DI estimates can also
be negative; they are not silently clipped.

Gaussian and Kozachenko outputs use nats. Retained JIDT kernel entropy outputs
keep their existing **bits** convention. Aligned raw histories, target-only
conditional entropies and causal sums are shared by the retained Java
estimators where their parameters and sample intervals match. Their
data-dependent caches, including automatic Theiler windows, invalidate on
data mutation. The existing Theiler formula and integer truncation are
unchanged; each channel's autocorrelation is now calculated once.

Stochastic interaction at delay d is
`H(i[t]|i[t-d]) + H(j[t]|j[t-d]) - H(i[t],j[t]|i[t-d],j[t-d])`.
The Gaussian calculation evaluates each unordered pair once, shares target
prediction errors with compatible Granger calculations, and diagnoses singular
joint residuals rather than inventing a finite value. Retained Java stochastic
interaction shares its target-only terms with DI. Reverse-pair evaluations
remain for the non-Gaussian joint estimators: a seeded, finite-sample estimator
must not be presumed numerically permutation-invariant solely from its formula.

Independent OLS, determinant and conditional-information references are in
[`audit/checks/refs/_kernels_infotheory.py`](../audit/checks/refs/_kernels_infotheory.py);
the retained estimator checks also construct JIDT embeddings independently.

## Symbolic transfer entropy

`te_symbolic_m-{m}_tau-{tau}_h-{h}` computes empirical conditional mutual
information `I(source_pattern[t]; target_pattern[t+h] | target_pattern[t])`,
in nats. A pattern orders m observations spaced tau samples apart, from oldest
to newest. Equal values retain their temporal order. The implementation uses
ordinal symbols and categorical counts; it no longer dispatches to Gaussian TE.
Pattern length one is rejected because its only symbol carries no information.
The ordinal-pattern approach comes from
[Staniek and Lehnertz (2008)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.158101);
the explicit settings below define this implementation's sampled transitions.

The reviewed candidates use m=3, with spacing/prediction shift 4/4 and 16/16
samples. At 2048 Hz their pattern spans are 3.906 and 15.625 ms; the shifts are
1.953 and 7.813 ms. A 1024-sample window supplies 1012 and 976 aligned pattern
transitions, respectively, with six possible patterns per channel. Patterns
and successive transitions overlap inside a window; outer windows still do not.

These are plug-in estimates, not bias-corrected effects or significance tests.
Independent autocorrelated signals can have positive finite-window estimates;
overlapping symbols also reduce the effective independent sample count.
Diagnostics report available transitions, observed patterns, tie fractions,
target transition entropy and joint-state occupancy. All six marginal symbols
being observed does not prove adequate sampling of every joint state or
clinical usefulness. Longer patterns can make the state space impractical;
the sparse counting path avoids allocating factorial-sized dense arrays.

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
modes. `workflow/reviewed.yaml` includes the original four unrestricted max/mean
requests, unrestricted signed minimum/peak, and the six corresponding
significant-only summaries. The workflow resolves their sampling frequency
before converting the 125 ms lag limit to samples.

## Spearman, Kendall, covariance and precision

Spearman is Pearson correlation of per-channel average ranks. Kendall is
tie-corrected tau-b. Their `squared=True` variants square the **final pair
coefficient**: rho² or tau². They have no lag aggregation.

Covariance and precision squared variants square each entry of the fitted
matrix. They do not square the input, perform a matrix product, or invert a
squared covariance matrix. Fitted estimators are shared, with returned arrays
copied so callers cannot alter subsequent results.

Precision outputs are the fitted estimator's `precision_` entries. They are
not partial correlations: the latter require `-Theta[i,j] / sqrt(Theta[i,i] *
Theta[j,j])`, which this class does not apply. Empirical and robust fits use a
Moore-Penrose pseudoinverse on singular covariance matrices, including CAR data.
Shrunk covariance is invertible when its shrinkage supplies a positive diagonal.
The [amplitude selection review](amplitude_selection_review.md) explains why
removing z-scoring has different effects on these cases.

MinCovDet and EllipticEnvelope share a fixed-seed robust fit; EllipticEnvelope's
unused outlier threshold does not change covariance or precision. Cholesky
C-steps accelerate sufficiently well-conditioned full-rank inputs. Singular or
ill-conditioned centered inputs, including CAR, use the reference MinCovDet
fit throughout: tiny rounded positive pivots in a singular covariance must not
change the selected support. This preserves the estimator instead of adding a
ridge or changing the reference. It does not fix the existing voltage-scale
limitations documented in the amplitude review.

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

Factorization-based metrics and both spectral Granger methods now return
**row = source, column = target**. Internal transfer matrices retain their
mathematical target/source convention. Known-model and channel-permutation
checks cover the public orientation and consistent pair fitting described below.
Wavelet sign conventions remain outside this Fourier review.

## Cross-distance correlation and convergent cross mapping orientation

`dcorrx`, `mgcx` and `ccm` use the same public convention as the other
directed measures: entry `[i, j]` is evidence that row i drives column j.
hyppo's `DcorrX(x, y)`/`MGCX(x, y)` relate `x[t+lag]` to `y[t]` (y -> x), so the
target is passed first. pyEDM's `"i:j"` column is i's manifold estimating j,
which is evidence of j -> i, so it is stored at `[j, i]`. These are pure
transposes of the earlier outputs; values are unchanged. `ccm_*_diff[i, j]`
is `ccm[i, j] - ccm[j, i]`, positive when i -> j is the stronger direction.

## Directed coherence

`dcoh` returns **squared directed coherence at each frequency**, followed by
the requested band mean or maximum. For target `i` and source `j`:

`DC[i,j](f) = variance[j] * abs(H[i,j](f))**2 / sum_k(variance[k] * abs(H[i,k](f))**2)`.

`H` maps the model's innovations into the observed channels; `variance[j]`
is source j's innovation variance. The denominator includes the target's own
innovation. Defined values lie in [0, 1] and sum to 1 across sources within
each fitted model, before pyspi masks the self-entry. One model is fitted for
each unique pair and shared by both directions. Public matrices use row=source,
column=target, whereas the formula above uses the internal target/source
indexing of H. The assembled off-diagonal multichannel matrix does **not** have
a row- or column-sum-one guarantee: its entries come from different pair models.

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
strength. Both band summaries and every frequency band use the corrected
definition, independently of preset membership. The shared directed-family
fit and public orientation are described below.

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
three bands; after removing ordinary coherence-phase maxima their sizes are
298 and 230. Existing PLI/wPLI names use the corrected unsigned definitions.
The reviewed EEG selection includes signed PLI/wPLI on the same eleven bands
as their unsigned counterparts, sharing the underlying reductions.

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

## Shared directed spectral calculations

DC, DTF, PDC, generalized PDC, dDTF and nonparametric spectral Granger use one
two-channel fit per unique pair. Public entry `[A1, C1]` means A1 -> C1;
parametric spectral Granger uses the same public convention. These pairwise
models do not condition on other recorded channels. A relationship can therefore
reflect omitted common drivers or indirect paths; the name "direct dDTF" does
not change that limitation.

DTF and PDC use squared magnitudes at each frequency. Generalized PDC also
weights the autoregressive coefficients by innovation variances. In a two-channel
model, off-diagonal PDC equals DTF, and off-diagonal generalized PDC equals DC.
The implementation shares those results. These are not independent features
under pairwise fitting; the equalities do not generally hold for larger models.

Squared dDTF is `ffDTF_squared * partial_coherence_squared` at each frequency.
For internal target i, source j,
`ffDTF_squared[i,j,f] = abs(H[i,j,f])**2 / sum_(g,k)(abs(H[i,k,g])**2)`.
Partial coherence is obtained from the inverse model spectrum, not PDC. In a
pair it equals ordinary coherence magnitude squared. Multiply before taking
the band's mean or maximum. The denominator uses all native nonnegative FFT
bins, including DC and even-length Nyquist once, irrespective of the reported
band. The grid is unpadded, so FFT-speed padding cannot change its normalization.
Other Fourier metrics retain their previously reviewed grids; tapered samples
are reused if a native-length directed transform is needed. Different native
window lengths can still change the discrete dDTF scale.

The Wilson iteration uses a symmetric positive square-root start and a discrete
plus operator that splits zero and even-grid midpoint lags equally. It retains
positive lags and removes negative lags, satisfying `G_plus + G_plus.H = G`
on a Hermitian frequency grid without a triangular channel-order choice.
Both relative iterate change and reconstructed-spectrum residual must be at
most 1e-10; the iteration cap is 100. Validation uses channel-power equilibration
so units do not hide errors. Singular, invalid, nonfinite or nonconvergent pairs
produce NaN while healthy pairs remain usable. No ridge, substitute model or
averaging of opposite channel-order fits is used. Finite-grid reconstruction
does not validate statistical assumptions or eliminate impulse-response aliasing.

Nonparametric Granger uses the correlated-innovation power decomposition and a
stable logarithmic ratio. A genuine zero remains zero; it is no longer discarded
as NaN. Zero intrinsic power has no finite estimate and is left unavailable,
without an epsilon replacement. Parametric Granger retains nitime's model and
order-selection definition, fits each unique pair once, and isolates failures
to their pair rather than discarding the full matrix.
Unsupported inputs raise an error instead of being converted into an apparently
successful all-NaN calculation by the former whole-matrix exception fallback.

`spi.diagnostics(data)` exposes pair reasons and unavailable-bin fractions.
Nonparametric directed diagnostics also provide reconstruction residual and
iteration count. They are detached copies; these detailed matrices are not
automatically persisted by the workflow's generic metric-report schema.
Calculator registers enabled directed requests before evaluation, then streams
the necessary bands and families through bounded pair batches. Standalone
requests share the four distinct spectral families for the requested band.
Cached band matrices replace retaining every fitted pair spectrum. Modifying
Data invalidates the shared results.

## Envelope correlation (PEC)

These metrics use envelopes of the supplied signal and apply no internal band
filter. Broadband input therefore gives broadband envelope correlation.
`pec` is Pearson correlation of Hilbert **amplitudes**, despite the historical
family name "power envelope correlation". `pec_log` correlates **log power**,
computed as `2*log(amplitude)` to avoid squaring overflow/underflow.

For the orthogonalized variants, correlate the envelope of X with the envelope
of `Im(Y * conj(X) / abs(X))`, repeat with X and Y reversed, and average the
two coefficients. Log variants transform the envelopes to log power before
correlation. Ordinary orthogonalized outputs average the signed coefficients;
`_abs` outputs average their absolute values. Thus coefficients -0.9 and +0.2
give -0.35 signed and +0.55 absolute. This is not the absolute value of -0.35.
Projection direction does not indicate causal direction; the final matrix is
symmetric. Only absolute variants have unsigned labels.

Constant envelopes have undefined Pearson correlation. A zero orthogonal
component also has undefined correlation, not a measured zero association.
The implementation centers both inputs, handles exact duplicate/proportional
signals without converting cancellation residue into coupling, and uses
scale-aware propagated rounding bounds to identify unresolved envelopes.
Small but resolved variation remains valid. Exact zero log envelopes and zero
orthogonalization reference amplitudes produce NaN without arbitrary floors
or dropping samples. If either directional coefficient is unavailable, their
symmetrized value is unavailable. `spi.diagnostics(data)` gives the reasons;
orthogonalized variants also expose each projection direction's reason.

All six variants share one analytic transform per Data. The signed and absolute
orthogonalized variants share directional coefficients, and amplitude/log-power
variants share the pair products. Redundant single-epoch container averaging is
removed. Results and diagnostic arrays are detached from the cache, which is
invalidated by Data mutation. The numerical guards are not statistical
significance tests, and orthogonalization does not establish causation.
