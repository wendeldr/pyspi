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
