# xcorr family: define, trace and check

Units: `basic.xcorr.max_sig-True`, `basic.xcorr.mean_sig-True`, `basic.xcorr.mean_sig-False`, `basic.xcorr.max_sig-False`.
Each unit covers the plain and the `-sq` identifier (`squared=True` squares the lag-wise correlations before the summary, not the output).
Class: `CrossCorrelation(Undirected, Signed)` in `pyspi/statistics/basic.py:75-133`. Delegate: `scipy.signal.correlate` (scipy 1.17.1, numpy 2.5.4).
The probe scripts are in the session scratchpad under `xcorr/probe1.py` to `probe5.py`, plus `xcorr_fix/p1.py` for the data-dependent magnitudes of `mean_sig-False` (cointegrated, trending, random-walk and short-memory datasets). The regression goldens are snapshots of current output, and nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Documented definition (shared by all four units)

Supplement §1.1.5, p.5 (heading at supplement.txt line 192), with the same text on the GitBook page `basic-statistics.md:55-57`:

> "The cross-correlation function is defined as the Pearson correlation between two time series for all lags [8], giving values in [−1, 1] for each lag ... the correlation from the MLE of the cross-covariance at a given lag, normalised by the autocovariance ... we do by only using the first T/4 lags ... a correlation below 1.96/√T is considered statistically insignificant [8], thus we optionally cut-off the lags at this level (sig-True ... only ... significant values, sig-False ... all values) ... the maximum over the considered lags (max), and the average over the considered lags (mean)."

The supplement lists `xcorr_max_sig-True`, `xcorr_mean_sig-True` and `xcorr_mean_sig-False`. The GitBook adds the `-sq` versions. `xcorr_max_sig-False` and its `-sq` version appear in neither, and in neither `pyspi/config.yaml` nor `fast_config.yaml`. They come only from the retained EEG list (`audit/env/retained_metrics.txt:186,188`). The EEG selection keeps only the sig-False units.

## Canonical definition

The canonical definition is in Brockwell & Davis, *Time Series: Theory and Methods* (reference [8] in the supplement, supplement.txt:1987-1988). I recalled the locator as §11.2 from memory and did not check it against a copy, so treat the section number as unverified. The sample cross-covariance is γ̂_xy(h) = n⁻¹ Σ_{t=1}^{n−h} (x_{t+h} − x̄)(y_t − ȳ) for h ≥ 0, with γ̂_xy(h) = γ̂_yx(−h) for h < 0. The sample cross-correlation is ρ̂_xy(h) = γ̂_xy(h)/√(γ̂_xx(0)γ̂_yy(0)). It is mean-centred, uses divisor n at every lag, and satisfies |ρ̂| ≤ 1.

The ±1.96/√n significance bound (Bartlett-type asymptotics; I recalled it as Theorem 11.2.2, which is unverified) holds when the two series are independent and at least one is white noise. Otherwise the bound is anti-conservative and prewhitening or an effective-degrees-of-freedom correction is needed. Supplement refs [9] (Afyouni, Smith & Nichols 2019, NeuroImage 199:609) and [10] (Cliff, Novelli, Fulcher, Shine & Lizier 2021, Phys Rev Res 3:013145), at supplement.txt:1990-1995, deal with significance under autocorrelation.

Lag truncation: the supplement (§1.1.5, p.5, supplement.txt:197-198) cites [9, 10] for the practice that "it is common to truncate the function at a given level". The specific T/4 cut is pyspi's choice. I consulted no canonical source for T/4, and none is attributed here (an earlier draft wrongly credited Box & Jenkins "via [9,10]"; neither reference is Box & Jenkins). No canonical source defines a "max" or "mean of the CCF over a lag window" as a dependence measure. That summary is pyspi's own construction.

## What is implemented (trace)

1. **Preprocessing.** `pyspi/data.py:184-195`: optional `scipy.signal.detrend`, then `zscore(axis=1, ddof=1)` (default `normalise=True`).
2. **Pair order.** `pyspi/calculator.py:327-339` (HEAD a1f20c7; the trace cited the pre-a1f20c7 lines 292-303) calls `multivariate` (:329); any exception (except branch :336-339) turns the whole matrix into NaN. `pyspi/base.py:132-138` `Undirected.multivariate` calls `Directed.multivariate` (`base.py:85-94`; outer loop over j, inner over i), then mirrors the upper triangle onto the lower one. For a pair a<b the first call is `bivariate(i=b, j=a)`.
3. **Normalisation.** `basic.py:101-104`: `x, y = data.to_numpy()[[i, j]]` (each (T,1)), then `r = squeeze(signal.correlate(x, y, 'full')) / x.std() / y.std() / (T-1)`. `std` uses ddof=0 and the SPI removes no mean.
4. **Lag window.** `basic.py:107`: `r[T - T//4 : T + T//4]` keeps lags −(T//4−1)…+T//4. The probe gives lags −15…16 at T=64 and −63…64 at T=256, with lag 0 at position T//4−1.
5. **Cache.** `basic.py:109-113`: the same unreversed array is cached for (i,j) and (j,i). The probe confirmed that `data.xcorr[(0,1)]` equals the slice computed from `correlate(x1, x0)`. Every matrix entry therefore comes from the (higher index, lower index) orientation.
6. **sigonly window.** `basic.py:116-120`: `N = len//2`, threshold `1.96/sqrt(N)`. `fzf` is the first insignificant index in `r[N:]`, `fzr` is the last insignificant index in `r[:N]`, and the kept window is `r[N-fzr : N+fzf]`.
7. **Summary.** `basic.py:122-131`: `np.max` or `np.mean` of `r` or `r**2`.
8. **scipy dispatch.** `_signaltools.py:104` `correlate` → `:258-259` method 'auto' → `convolve(in1, _reverse_and_conj(in2))` (`:1165-1182`) → `:1490-1494` `choose_conv_method` (`:1236-1374`; the probe returns 'fft' for (T,1) inputs at every T tested) → `fftconvolve` (`:591`, `:706-708`). Docstring `:168-170`: z[k] = Σ_l x_l y_{l−k}, k = −(M−1)…(N−1).

**Direction convention (from scipy's docstring and a probe).** A peak at positive lag k in `correlate(x, y)` means x(t) ≈ y(t−k), so y leads. In pyspi's stored r (x = higher-index channel), positive lags therefore mean the lower-index channel leads. The output matrix is symmetric.

## Deviations common to all units

- **Scale (pack hypothesis confirmed).** For ddof=1 z-scored data, std₀(x)std₀(y)(T−1) = (T−1)²/T, so r = T/(T−1) · ρ̂ at every lag. The probe at T=64 gives a constant ratio of 1.0158730 = 64/63 against both the Pearson r and the B&D CCF. Identical series give 1.0159 at T=64, which is outside the documented [−1, 1].
- **Window.** The documentation says "first T/4 lags". The code uses both sides, 2·(T//4) lags, one more positive lag than negative. The value therefore depends slightly on channel order: in the probe on iid_gaussian, `mean_sig-False` differs by 1.05e-3 (T=256) and 1.09e-4 (T=1024) when the channel order is reversed.
- **normalise=False.** No centring happens inside the SPI, so the result is not a correlation. On scaled_offset at T=1024, `max_sig-False` for pair (2,3) is 0.597 on raw data against 0.798 on z-scored data.
- **Signed max.** The plain `max` ignores negative peaks. The `-sq` version takes the peak magnitude.

## basic.xcorr.max_sig-True and basic.xcorr.mean_sig-True

Additional deviations, all candidates:

- **Threshold (pack hypothesis confirmed).** The code uses 1.96/√(T//4) ≈ 3.92/√T instead of the documented 1.96/√T. At T=256 that is 0.245 against 0.1225.
- **Centre off by one (pack hypothesis confirmed).** `N = len//2` is the element at lag +1, not lag 0.
- **Mirrored negative cut (new).** The code starts the window at `N - fzr`, but `fzr` counts from the far-left end, so a contiguous cut would start at `fzr+1`. For a null pair, lag 0 is insignificant, which gives `fzr = N-1` and a window covering lags −(T//4−2)…0 (63 insignificant lags at T=256). The documented rule would give an empty window. A crafted r that is significant only at |lag| ≤ 2 gives the window −11…2 and a mean of 0.321 instead of 0.9.
- **Channel-order dependence (new, large).** On delayed_noise at T=256 (x0 leads x3 by 5 samples), `xcorr_max_sig-True` A[0,3] is 0.2035 in the given order and 0.9700 with the channel order reversed. For `-sq` the values are 0.0414 and 0.9409. Lag +1 is insignificant, so the positive side is cut to nothing and the true peak at +5 is never seen. It can happen for any pair where the lower-index channel leads by two or more samples and lag +1 is insignificant (a hypothesis beyond the probed cases). On iid_gaussian at T=256 the maximum difference between orders is 0.081 for max and 0.019 for mean.
- **Errors.** The documented case is in tests/README.md: `xcorr_max_sig-True::n0d0` raises IndexError and is locked as an error. The probe shows the same IndexError also happens under the default preprocessing whenever one half of the window never falls below the threshold:
  - narrowband_f0-0.25 at every T
  - logistic_maps and logistic_maps_bidir at every T
  - cointegrated at T ≥ 256 (T ≥ 1024 with detrend)
  - shared_envelope at T ≥ 1024
  - narrowband_f0-0.1 and f0-0.4 at T=4096
  - trending without detrend

  Inside the Calculator a single failing pair makes the whole matrix NaN. Separately, an empty window (`fzr = fzf = 0`, shown with a crafted cache) makes `max` raise ValueError and makes `mean` return NaN with a RuntimeWarning.

Proposed checks:

- a scipy-free reimplementation of the literal algorithm (tolerance 1e-12)
- the B&D CCF plus the documented significance rule, to quantify the deviation
- the delayed_noise direction check
- a channel-permutation check (expected to fail)
- the iid null exceedance rates: about 5% at the documented threshold and about 0.009% at the code threshold
- the error-path sweep across datasets, with and without detrend
- shared_envelope (x2 identical to x0), where the lag-0 value is T/(T−1) to 1e-12 (not bit-exact on the FFT path: 1 ulp off at T=256)
- a raw-input (normalise=False) comparison on scaled_offset

## basic.xcorr.mean_sig-False

This unit is in the EEG selection. Only the common deviations apply, and there is no error path.

One further candidate, which the probe confirmed: z-scored series have zero mean, so the sum of r over all 2T−1 lags is 0 up to round-off. The window mean therefore equals −(tail sum)/(2·(T//4)), where L = T//4.

- **Lag 0 has little weight.** The lag-0 correlation enters with weight 1/(2L). For bivariate_gaussian at T=1024 (pair (2,3), lag-0 r = 0.798), its contribution is 0.798/512 = 0.00156. The other window lags offset it, and the window mean and −tail/(2L) both come out at −0.000208.
- **The value tracks the tail, not lag 0.** It depends on how much CCF mass lies outside the window, so its size depends on the data:
  - Short-memory data stays near 0. Default pipeline, max|A| at T=256 / 1024: var2_bivariate 0.0078 / 0.0009, gauss_ar1_pair 0.0110 / 0.0083, bivariate_gaussian 0.0024 / 0.0013.
  - Long-memory or trending data is large. cointegrated 0.458 / 0.468, trending 0.612 / 0.632, and a random-walk pair at T=1024 gives 0.618 (probe `xcorr_fix/p1.py`; the identity check there uses pair (0,1) of each dataset).
- The −tail/(2L) identity held on every one of these to the printed precision.

Whether this makes it a weak or misleading dependence measure is left for the check stage.

Proposed checks: the literal reimplementation, the B&D comparison, direction, permutation (expected differences of about 1e-3/1e-4), the zero-sum identity (tolerance, on short- and long-memory datasets), sig-False finiteness on the datasets where sig-True raises, shared_envelope, and raw input.

## basic.xcorr.max_sig-False

This unit is retained from the EEG list and is not listed in the supplement, GitBook or config. The common deviations apply. Order dependence arises only when the maximum lies at the end lag ±T//4 (the probe difference on iid is about 1e-16). The scale inflation gives A[0,2] = T/(T−1) on shared_envelope. Already documented: only docs/storage_comparison/REVIEW.md:200, a metadata note about missing labels.

Proposed checks: as for mean_sig-False, without the zero-sum identity.

## Open questions

1. The Brockwell & Davis section and theorem numbers (§11.2, Theorem 11.2.2) were cited from memory and remain unverified; no copy is available locally. No canonical source for the T/4 truncation was consulted. The supplement cites [9, 10] only for truncation in general.
2. Was the intended sigonly rule "contiguous significant lags around lag 0" (the code comment says "Truncate at first statistically significant zero")? And did the authors intend the window threshold to use √(T//4) (the window half-length) on purpose? This is a hypothesis: the comment says 1.96/sqrt(T), which suggests the √N is unintended.
3. Should an undirected SPI built from an asymmetric window be order-invariant? The documentation implies yes. Reversing channel order changes sig-True values by up to 0.8 in the probe.
4. Is the historical EEG data (`pyspi_version=1.1.1.1`) produced by this same code path, for example the same window slice and ddof? Not traced here.
5. The 1.96/√T bound assumes at least one white series. For autocorrelated EEG it is anti-conservative (refs [9, 10]). This is a caveat of the documented definition rather than a code deviation.

## Checks

Files: manifest `audit/checks/manifest/xcorr.json` (574 runs), references `audit/checks/refs/xcorr.py` (comparison venv; no pyspi; `scipy.signal.correlate` is not used: the xcorr kernels `xcorr_full` and `pyspi_xcorr_replica` use `np.dot` / `np.convolve` only, although the shared kernel module `_kernels_basic_misc.py` imports `scipy.signal.hilbert` at module level, line 25), probes `audit/checks/probes/xcorr_probes.py` (project venv), results `audit/checks/out/results/xcorr.json`: 450 pass, 124 fail, 0 without reference. All 124 failures are deliberate deviation measurements:
- 84 `|canon` runs compare pyspi with the canonical or documented value, and each is decomposed below;
- 40 `|perm-selfcons` runs are same-lineage channel-order checks on the order-dependent units.

Every pyspi run took at most 0.011 s. The goldens are not used anywhere.

**Paths.** The trace's proposed checks 0–2 (replica, Brockwell & Davis, delayed_noise direction) and, for the sig-True units, 4 (iid null) named path `kernel`. They ran on path `pipeline` (Data default, z-score ddof=1) instead, and the reference z-scores the raw dataset itself (`refs/xcorr.py:zscore1`/`prep`). The comparison therefore still covers the SPI on known input and also matches the EEG default. Path `kernel` (no z-score, and no centring inside the SPI) was used only for scaled_offset: the sig-False replica and canon runs, where the raw-input effect is the point, and the sig-True error sweep (the documented n0d0 case). The sig-True units have no separate raw scaled_offset value comparison (their proposed check 7); each record's `check_cost` states this substitution.

Run tags (the last `|` field; `-sq` marks the squared identifier):
- `replica`: line-by-line replica `_kernels_basic_misc.pyspi_xcorr_replica` (direct `np.convolve` sum) in pyspi's orientation (x = higher-index channel), on series preprocessed by the reference itself (own z-score ddof=1, own least-squares linear detrend). Exact, 1e-10/1e-12.
- `canon`: Brockwell & Davis ρ̂(h) via `xcorr_full(ddof=1)`, which equals the divisor-T B&D estimator because (T−1)s₁ₓs₁ᵧ = T s₀ₓs₀ᵧ, over the documented window |h| ≤ T//4. For sig-True the documented filter applies: keep lags with |ρ̂| > 1.96/√T, and an empty set gives 0. The extra keys `gap_d0`…`gap_d4` give max|pyspi − step| after each successive change:
  - d1: × T/(T−1)
  - d2: + pyspi window −(T//4−1)..T//4
  - d3: + threshold 1.96/√(T//4)
  - d3c: contiguous run about lag 0
  - d3c1: contiguous run about lag +1
  - d4: pyspi's mirrored cut

  `gap_c_doc` is the contiguous-about-lag-0 reading with the documented threshold.
- `cached-r`: pyspi's own cached r (probe) against T/(T−1) × `xcorr_full` on lags −(T//4−1)..T//4.
- `lag-cached`, `lag-fresh`: peak lags on delayed_noise.
- `selfcopy`: shared_envelope processes [0,2], where x2 is identical to x0.
- `pop`, `pop01`, `pop23`: population peak CCF from the truth arrays, with se from Bartlett's formula at the peak lag.
- `null`: exceedance counts on iid_gaussian against the analytic expectation with Var ρ̂(h) = (T−|h|)/T².
- `errsweep`: status codes from a probe that catches the exception, compared with codes the reference predicts from the replica.
- `crafted`: hand-made cached r injected into `data.xcorr`.
- `zerosum`: the −tail/(2L) identity.
- `perm-replica`: reversed channel order against the replica.
- `perm-selfcons`: reversed against forward pyspi output. Same-lineage, so it caps the score at 6.
- `endlag` (max_sig-False only): a constructed pair, y = x0 delayed by T//4 (circular roll of iid_gaussian x0), evaluated in both channel orders by the probe `endlag_pair` and compared exactly with the replica.

**The T/(T−1) fact (final check).** The script produces this on z-scored data at T=1024, pair (hi=1, lo=0) (`…|1024|pipeline|replica`, keys `ref_lag0_xcorr_full_ddof1`, `ref_lag0_replica`, `ref_lag0_ratio`, `ref_lag0_ratio_minus_factor`). The ddof=1 normalised cross-correlation at lag 0 and pyspi's lag-0 value are:

| dataset | xcorr_full (ddof=1) | pyspi replica |
|---|---|---|
| bivariate_gaussian | 0.28097090550314563 | 0.2812455593697176 |
| var2_bivariate | 0.19246574879655134 | 0.1926538873584248 |
| delayed_noise | −0.023644977198118994 | −0.023668090567814125 |

The ratio is 1.0009775171065494 = T/(T−1), and ratio − T/(T−1) = 0.0 on all three. On pyspi's own cache (`basic.xcorr.max_sig-False|*|cached-r`, 4/4 pass, max_abs ≤ 3.3e-16), the ratio holds at every lag with |ρ̂| > 1e-3 to within 2.95e-14 (`ref_max_ratio_dev_all_lags`). At T=64 it is 1.0159 (`ref_lag0_ratio_minus_factor` −2.2e-16).

**Window and centre.** The replica passes exactly only with the window lags −(T//4−1)..+T//4 (−255..256 at T=1024, −63..64 at T=256; `ref_window_lag_min/max`), with the sigonly centre at lag +1 (`ref_centre_lag` 1). Aligning pyspi's cache to −T//4..T//4−1 instead mismatches by 0.78–1.02 (`ref_shifted_window_max_abs`).

**Threshold.** `ref_thr_code` 1.96/√(T//4) = 0.1225 against `ref_thr_doc` 1.96/√T = 0.06125 at T=1024 (0.245 against 0.1225 at T=256). On iid_gaussian (`…|iid_gaussian|1024|pipeline|null`, `…|4096|pipeline|null`, both pass at max_z 0.84):

| T | lags | above code threshold (expected) | above documented threshold (expected) | nominal |
|---|---|---|---|---|
| 1024 | 5120 | 0 (0.18) | 3.79% (3.65%) | 5% |
| 4096 | 20480 | 0 (0.70) | 3.72% (3.64%) | 5% |

The expectation is below the nominal 5% because Var ρ̂(h) shrinks as (T−|h|)/T².

**Sign and lead convention (delayed_noise T=1024).** `|lag-cached`: the cached r that multivariate uses for both (i,j) and (j,i) is correlate(x_hi, x_lo). It peaks at +(d_hi − d_lo) for every pair (max_abs 0, direction agreement 1.0), so a positive lag means the lower-index channel leads. `|lag-fresh`: bivariate(i,j) evaluated first peaks at d_i − d_j (max_abs 0), so a positive lag means x_j leads x_i. Over all 434 SPI-matrix runs, max|A − A.T| = 0.0. The matrix carries no direction.

### basic.xcorr.max_sig-False: variant, 8

- **Replica:** 52/52 exact (max_abs 6.7e-16). This includes the 8 datasets where sig-True raises; all are finite (`ref_all_finite` 1).
- **Canonical:** 0/22. On every pipeline run `gap_d0` = 1.15e-4 to 3.9e-3, max_rel 9.78e-4 at T=1024 and 3.92e-3 at T=256, which is 1/(T−1). `gap_d1` ≤ 6.7e-16, so the value is exactly T/(T−1) × the canonical maximum of ρ̂ over |h| ≤ T//4. Raw input (`scaled_offset|1024|kernel|canon`) gives a gap of 0.217: the non-centred value 0.597 against the centred 0.797.
- **Population peak CCF:** 12/12 within 2 se (max_z 1.08). bivariate_gaussian (2,3) gives 0.7983 against 0.8 (T=1024). var2_bivariate gives 0.5080 against 0.5151 (T=4096). gauss_ar1_pair gives 0.5258 against 0.5419 (T=4096).
- **selfcopy:** A = T/(T−1) and (T/(T−1))² to ≤ 4.4e-16 at T=64–4096, i.e. above 1.
- **Channel reversal:** 16/16 pass (≤ 2.2e-16). No maximum fell on ±T//4 on any dataset.
- **End lag (constructed, `…|iid_gaussian|{256,1024}|pipeline|endlag`, ±`-sq`):** 4/4 exact against the replica (max_abs 0). With the peak at lag +T//4 (forward order, inside the window) pyspi gives 0.7596 at T=1024. With the peak at −T//4 (reversed order, outside the window) it gives 0.1054 (`ref_order_gap` 0.654). At T=256 the values are 0.7961 and 0.1389 (gap 0.657); the −sq gaps are 0.566 and 0.614. The canonical maximum over |h| ≤ T//4 is the same in both orders (0.7589 at T=1024, `ref_canon_order_gap` 0).

Anchor 8: a constant factor T/(T−1). The end-lag order dependence needs a peak exactly at ±T//4, so it stays a caveat rather than lowering the anchor. Candidates: all 3 confirmed. The second candidate (order dependence only through the end lag) is confirmed by the constructed `|endlag` runs; the dataset runs never put a maximum on the end lag.

### basic.xcorr.mean_sig-False: variant, 6

- **Replica:** 52/52 exact (max_abs 4.1e-16), finite on every dataset.
- **Canonical:** 0/22. `gap_d0` = 3.4e-5 to 8.5e-4 absolute, but relative up to 0.54. `gap_d1` (factor only) = 3.3e-5 to 8.1e-4. `gap_d2` ≤ 6.1e-18. So the deviation is exactly the factor plus the one-lag-asymmetric window.
- **zerosum:** 18/18 pass at atol 1e-12 (max_abs 4.1e-16), so the value is −(T/(T−1)) Σ_outside ρ̂/(2 T//4). The magnitude follows the out-of-window mass (`ref_max_abs_value`, T=256 / 1024):

  | dataset | max abs value |
  |---|---|
  | iid | 0.0102 / 0.0021 |
  | var2 | 0.0078 / 0.00086 |
  | gauss_ar1 | 0.011 / 0.0083 |
  | cointegrated | 0.458 / 0.468 |
  | trending | 0.612 / 0.632 |

  On bivariate_gaussian T=1024 the ρ=0.8 pair scores −0.0002 and the ρ=0.3 pair 0.0013. The lag-0 weight r(0)/(2L) is 0.00156.
- **Channel reversal (same-lineage):** 0/16. Up to 1.05e-3 on iid at T=256 (where |A| ≤ 0.0102), 1.64e-3 on var2 at T=256, and 1.09e-4 on iid at T=1024. `perm-replica` reproduces both orders exactly.

Anchor 6: the library formula deviates from both the supplement ("average over the considered lags", "first T/4 lags") and the canonical CCF in an interpretable way. It applies the constant factor T/(T−1) and the window shifted by one lag, −(T//4−1)..T//4 (`gap_d2` ≤ 6.1e-18 once both are applied). The window shift changes the value (relative `gap_d0` up to 0.54), so anchor 8 (constant factor only) does not apply. The zero-sum behaviour above is a property of the documented quantity. It is recorded in `what_changed` and does not lower the score. (This was scored 5 before review; the review pointed out that the same "no literature definition" argument was not applied to max_sig-False, which has the same factor and window.) Candidates: 4/4 confirmed.

### basic.xcorr.max_sig-True: related-quantity, 4

- **Replica, crafted, errsweep, selfcopy:** all pass (26/26, 2/2, 58/58, 4/4).
- **Documented filter reading (canon), `gap_d0`:**

  | dataset | T=1024 | T=256 |
  |---|---|---|
  | bivariate_gaussian | 0.0172 | 0.27 |
  | iid | 0.040 | 0.248 |
  | delayed_noise | 0.911 | 0.793 |

  On var2 and gauss_ar1 only the factor remains (`gap_d1` ≤ 2.2e-16). All gaps close only at d4, the literal mirrored cut (≤ 6.7e-16).
- **Population peaks:** 12/12, where the peak lies at lag 0 or ±1.
- **delayed_noise at T=1024:** the kept window for pair (0,3) is lags −254..0 (`ref_pair03_fzf` 0, lag +1 r = −0.0019), so the lag-5 correlation of 0.993 is excluded and A[0,3] = 0.0808. Reversing channel order gives 0.993 (`perm-selfcons` 0.912 at T=1024, 0.797 at T=256 where 0.2035 becomes 0.9700; −sq 0.953/0.980). The iid gap is 0.081/0.040.
- **Null kept windows:** 255 lags with 0 significant.

Anchor 4 rather than the anchor-6 example ("sigonly threshold/origin"): the checks add the mirrored cut, order dependence of up to 0.98, and IndexError on structured data.

### basic.xcorr.mean_sig-True: wrong, 3

- **Replica, crafted, errsweep, selfcopy:** all pass.
- **crafted, case A:** pyspi averages lags −11..2 and gets 0.3214, where the documented rule gives 0.9.
- **Kept windows at T=1024:** 254–256 lags, at most 0.39% of them significant (`ref_kept_frac_significant_max` 0.0039).
- **Values do not rank dependence** (bivariate_gaussian T=1024): ρ=0.8 pair 0.0010, ρ=0.3 pair 0.0012, independent pairs up to 0.0029. The contiguous reading gives 0.80 (`gap_c_doc` 0.796).
- **Sign:** on delayed_noise pair (0,3) at T=1024 pyspi gives −0.0028 against +0.048 for the documented summary under the filter reading (0 under the contiguous reading; `refs/xcorr.py:canon_steps` d0 / c_doc); at T=256 −0.0025 against +0.323 / 0. It flips sign under reversal at both lengths (T=1024 −0.0028 → 0.0021, T=256 −0.0025 → 0.0062).
- **Channel reversal:** 0/16 pass, max 0.0226.

Anchor 3: the label "only the statistically significant values" does not describe the computation; this holds under both readings. Anchor 2 would also apply on the sign, but only under the filter reading (under the contiguous reading the documented value is 0).

### IndexError sweep (both sig-True units, `|errsweep`, 58/58 predictions equal to the probe codes)

Under the default pipeline the whole multivariate call raises IndexError, so the Calculator matrix is all NaN, on:
- narrowband_f0-0.25 at every T (2/3 pairs);
- logistic_maps and logistic_maps_bidir at every T and on both paths;
- cointegrated: T=256 3/6 pairs, 1024 6/6, 4096 5/6, including the independent random-walk pair (2,3);
- shared_envelope: T=1024 1/6, 4096 3/6;
- narrowband_f0-0.1 and f0-0.4 at T=4096 only;
- trending: T=256 2/6, T ≥ 1024 3/6.

Detrending (pipeline-detrend) removes the error only for trending and for cointegrated at T=256. Cointegrated at T ≥ 1024 still has 2/6 pairs. The raw scaled_offset case, which is the documented n0d0 case, fails on 1/6 pairs.

Cause: on one half of the window no |r| falls to 1.96/√(T//4) or below. This happens with periodic CCFs (narrowband, the period-4 logistic fixture), slowly decaying CCFs (trend, random walk, envelope) and the 0.99 copies; `ref_n_fail_pos_side` and `ref_n_fail_neg_side` record which half. sig-False is finite on all of these (48/48 replica runs). Crafted case B (fzr = fzf = 0) gives ValueError for max and NaN for mean, as the trace predicted.

### Trace corrections

- The calculator citation was updated to HEAD a1f20c7 (`pyspi/calculator.py:327-339`, multivariate at :329, except branch :336-339), in all four records and in the trace section above.
- The channel-permutation proposed checks are relabelled `reference_kind: same-lineage`.
- The zero-sum proposed check uses `path: pipeline`, because the identity needs mean-zero series.
- The `max_sig-False` `units_and_range` says channel order matters only when the maximum sits on ±T//4. That was not seen on the datasets (perm gap ≤ 2.2e-16), but it was shown on the constructed `|endlag` pair (0.7596 against 0.1054 at T=1024).
- The sig-True records' `already_documented` cited tests/README.md lines 111 and 220. At HEAD a1f20c7 the n0d0 statements are at lines 120, 130 and 249, and the records now cite those.
- The two remaining trace issues (the "2nd ed." edition in `canonical_ref` and the unnamed `xcorr_fix/p1.py` probe) were already fixed in the records and the note before this stage. I re-checked both and needed no edit.
- The trace's random-walk figure 0.6176 was not re-run. The cointegrated dataset's independent random-walk pair (2,3) gives 0.329 at T=1024.

### Contract note

`compare.py` marks any pyspi-side exception as a failure before it reads the reference. The expected IndexErrors were therefore caught inside the probe `sig_status`, which returns status codes; the reference predicts the codes independently and sets `pass_flag`. Plain SPI runs that raise were not kept in the manifest.

## Open questions: answered

1. **Brockwell & Davis locators.** Still unverified: no copy was consulted, and the checks cannot settle a citation. The reference implements the B&D estimator from its definition. Bartlett's variance formula, used for the population checks, gave z ≤ 1.08 in 12 runs. That supports the formula but not the section numbers. Settling it needs the book (§11.2, Thm 11.2.2 in the 2nd edition is the recollection). No source for T/4 was found. It remains pyspi's choice.
2. **Intended sigonly rule.** Intent cannot be read from the outputs. The evidence:
   - The code comment says |r| ≤ 1.96/√T and the supplement says "only use the statistically significant values". The code uses 1.96/√(T//4), centres at lag +1, and cuts the left side at a mirrored index.
   - Under either documented reading (filter or contiguous about lag 0), pyspi's value is reached only by applying all three code choices (`gap_d4` ≤ 6.7e-16; every earlier step leaves gaps up to 0.99).
   - The rule has been unchanged since at least tag v1.0.3 (`git show v1.0.3:pyspi/statistics/basic.py`, lines 116-120).

   Both readings are recorded (`definition_under_review: true` for the sig-True units). Settling the intent needs the authors or a design note.
3. **Order invariance.** Answered empirically. sig-True values change with channel order by up to 0.98 (delayed_noise −sq), 0.912 (plain), and 0.081 on iid. mean_sig-False changes by 1e-4 to 1.6e-3. max_sig-False shows no change on the datasets (≤ 2.2e-16), but on a constructed pair whose CCF peak lies on lag T//4 it changes from 0.7596 to 0.1054 (T=1024, `|endlag`). The independent replica reproduces both orders exactly, so this is the algorithm as written, not numerical noise. A symmetric window and rule, or evaluating both orientations, would remove it.
4. **Historical EEG data (pyspi 1.1.1.1).** At tag v1.1.1 the CrossCorrelation lines 103-120 are identical to HEAD: same `correlate`, same `/x.std()/y.std()/(T-1)`, same slice and same sigonly rule. `pyspi/data.py` at v1.1.1 z-scores (line 181) before detrending (line 183). Linear detrending keeps the mean at 0 and the SPI renormalises by the series' own std, so the factor T/(T−1) relative to B&D carries over. I did not run this; it follows from the identity. Whether "1.1.1.1" is exactly tag v1.1.1 was not established.
5. **Anti-conservative 1.96/√T bound.** It holds for white independent pairs: the iid null gives 3.7% exceedances, below 5% because of the (T−|h|) shrink. The checks also show the failure for autocorrelated independent series. The independent random walks in cointegrated (pair (2,3)) never fall below even the doubled code threshold on one half at T=1024 (`basic.xcorr.max_sig-True|cointegrated|1024|pipeline|errsweep`: 6/6 pairs IndexError). Their mean_sig-False value is 0.329. This is a limit of the documented definition rather than a code deviation. A null of two independent AR(1) series would quantify it more finely; that was not run.
6. **Is mean_sig-False a meaningful dependence measure?** Not for small-lag dependence. The zero-sum identity holds to 4.1e-16 on all 18 runs, so the value is −(out-of-window CCF mass)/(2 T//4). It is about 0 for short-memory data whatever the lag-0 correlation (the ρ=0.8 pair scores −0.0002) and 0.33–0.63 for trends, random walks and cointegration, including independent random walks. It measures shared low-frequency or trend content, not correlation. Apart from the factor and the one-lag window shift, the code computes what the supplement describes, so this is recorded as a property of the definition, not as an implementation error, and it does not enter the score (6, for the factor and window deviations).

## Optimisation

Workflow 3, group basic. Walls are from `audit/profiles/summary.json` (quiet machine, all shapes measured, `measure_mode: measured` at 118x1024). Percentages are shares of the profiled `compute_spi` call. Nothing in pyspi was changed. Micro-benchmark: `audit/profiles/micro/opp_xcorr-1.py` / `.json`.

| unit | fresh 8x1024 | fresh 32x1024 | fresh 118x1024 | family-warm 118x1024 (both identifiers) |
|---|---|---|---|---|
| basic.xcorr.max_sig-True | 5.5 ms | 0.096 s | 1.37 s | 1.53 s (runs first: pays the pair cache) |
| basic.xcorr.mean_sig-True | 5.7 ms | 0.099 s | 1.42 s | 0.35 s |
| basic.xcorr.mean_sig-False | 5.1 ms | 0.088 s | 1.25 s | 0.13 s |
| basic.xcorr.max_sig-False | 5.0 ms | 0.086 s | 1.24 s | 0.086 s |

**Complexity.** `CrossCorrelation.bivariate` runs `scipy.signal.correlate(x, y, 'full')` once per unordered pair. The `data.xcorr` cache serves the second ordered call and every later identifier. With method='auto', scipy picks `fftconvolve` at T=1024: all 28 correlate calls in the 8x1024 profiles went through fftconvolve. The cost is O(P^2 T log T) plus O(P^2 T) for the std, normalisation and window slice, and the sigonly variants add a per-call truncation. Measured fresh P_exponent 2.05-2.06 and T_exponent 0.09-0.10. At 118x1024 the pair term dominates: 6903 correlate calls at about 0.18 ms.

**Hotspots (8x1024 pstats).** `scipy/signal/_signaltools.py:correlate` (fftconvolve) takes 70-72% of the call for sig-False and 65-66% for sig-True. Numpy `_std` (two calls per pair) takes 13-14%. `bivariate` self time is 7.6-7.9% for sig-False and 13.7-14.1% for sig-True, where the extra is the sigonly `np.where` truncation. `opaque_share` is 0.29-0.31, mostly the pocketfft C calls.

**Avenues.**
- `xcorr-1` (all four units, vectorise-batch, effort M): one rfft per channel, then for each a one batched irfft of F[b]·conj(F[a]) over b>a, keeping pyspi's window and its (higher index, lower index) orientation. The (P, P, 2·(T//4)) array would be cached for all eight identifiers. Measured in `opp_xcorr-1.json` for all four sig-False identifiers, signed and -sq (OMP_NUM_THREADS=1, min of 2-3 repeats, Data construction timed on both sides; re-run by the fix pass after the -sq identifiers were added, so the signed timings differ slightly from the first run's 15.6x / 17.4x):

  | shape | xcorr_mean | xcorr_max | xcorr-sq_mean | xcorr-sq_max |
  |---|---|---|---|---|
  | 118x1024 | 1.18 s -> 0.076 s (15.5x) | 1.15 s -> 0.068 s (16.9x) | 1.19 s -> 0.069 s (17.3x) | 1.18 s -> 0.069 s (17.1x) |
  | 32x1024 | 12.9x | 12.7x | 13.0x | 12.8x |
  | 8x1024 | 3.7x | 3.7x | 3.8x | 3.7x |

  max_abs_diff at 118x1024 is 1.8e-16 (mean, max_rel 3.3e-11 on near-zero means), 7.8e-16 (max), 5.0e-17 (-sq mean) and 6.4e-16 (-sq max); the NaN masks are equal at every shape. The 7x100 golden snapshot matches to 7.8e-18 for xcorr_mean_sig-False and 1.4e-17 for xcorr-sq_mean_sig-False; the golden set has no entry for the two max identifiers. The candidate squares r elementwise before the mean / max, as `bivariate` does. The output is within-tolerance, not bit-identical. For the sig-True units the expected gain is about 5x, because the per-pair truncation loop remains (0.16-0.18 s per sig-True identifier at 118 even with a warm cache). No sig-True identifier was run, so their records give a profile basis with the sig-False figure carried over, not a microbench one. Their risk is unverified-within-tolerance: the cut compares |r| with 1.96/sqrt(N), so a lag within ~1e-16 of the threshold could move the cut.
- `xcorr-2` (sig-True units, vectorise-batch, ~3x after xcorr-1, ~15x combined, effort M): vectorise the sigonly truncation over the batched array (first crossing per side via argmax of a boolean mask). It must keep the IndexError when a side has no crossing, which pyspi turns into an all-NaN matrix. Expected unverified-bit-identical when applied to the same r. Not measured.

**Seed check.** "Batched rfft cross-correlation plus vectorised sigonly truncation, within-tolerance": confirmed and measured for all four sig-False identifiers (15.5-17.3x at 118x1024, max_abs <= 7.8e-16). The sigonly part is `xcorr-2` and was not measured. scipy's method='auto' choice at T=1024 is fft (fftconvolve in every profiled call), so the current values already come from an FFT path.

**Shared versus incremental cost and EEG scaling.** The first xcorr identifier on a Data object pays the correlate for all 6903 pairs: 1.37 s for xcorr_max_sig-True in the family-warm order. Every later identifier reads the pair cache: 0.04-0.07 s for sig-False and 0.16-0.18 s for sig-True at 118x1024, for a family total of 2.10 s. The EEG selection keeps four sig-False identifiers: xcorr_mean_sig-False, xcorr-sq_mean_sig-False, xcorr_max_sig-False and xcorr-sq_max_sig-False. (The group brief mentions six; `audit/env/retained_metrics.txt` lists four.) The per-window EEG cost at 118 channels is about 1.25 s for the first of them plus 3 x 0.04-0.07 s, about 1.4 s per window or about 840 s per 600-window patient. There are no band variants. With `xcorr-1` it would be about 0.1 s per window.
