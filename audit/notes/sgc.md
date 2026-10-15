# sgc family: define, trace and check

Units: `spectral.sgc.nonparametric`, `spectral.sgc.parametric_order-None`, `spectral.sgc.parametric_order-1`, `spectral.sgc.parametric_order-20`.
Each unit covers six identifiers: mean or max over three bands (full, low half, high half). Within a unit the band/statistic variants read one cached per-frequency array (`data.spectral_gc[key]`).
Class: `SpectralGrangerCausality(NonparametricSpectralMultivariate, Directed, Unsigned)` in `pyspi/statistics/spectral.py:320-440`.
Delegates: spectral_connectivity 1.1.2 for the nonparametric path, and nitime 0.12.1 (with `scipy.signal.freqz`, scipy 1.17.1) for the parametric path. numpy is 2.5.4.
The probe scripts are in the session scratchpad under `sgc/p1_instantiate.py` to `p14_gauss_true.py`. The regression goldens are snapshots of current output. Nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Documented definition (shared by all four units)

Supplement §1.5.12, p.17 (supplement.txt lines 809-818: heading 809, quoted text 811-818). The same text is on GitBook `spectral.md:350-358`:

> "Spectral Granger causality [40] is the frequency domain equivalent to Granger causality, computed via the spectral transfer matrix and noise covariance that are estimated using either a parametric (VAR model) approach [41] or nonparametric (spectral factorization) approach [57, 58]. We implement the nonparametric form from the Spectral Connectivity Toolbox [56] and the parametric implementation from v0.9 of NiTime [74]. The VAR model is estimated via least squares using the same order for both processes, where the order is either inferred (order-None) or fixed. We infer a VAR model order from the Bayesian information criterion (with a maximum lag of 50, implemented in the NiTime toolkit), as well as choosing fixed orders of one (order-1) and 20 (order-20)."

The keywords are "directed, linear, unsigned, bivariate, frequency-dependent".

The band rule is in supplement §1.5, p.12 (supplement.txt heading line 570, quoted text lines 589-599):

> "125 uniformly sampled bins across the entire frequency range, f ∈ [f0, fs/2] ... f0 = 4/T ... fmin-0_fmax-0-5: f ∈ [0, fs/2]; fmin-0_fmax-0-25: f ∈ [0, fs/4]; fmin-0-25_fmax-0-5: f ∈ [fs/4, fs/2]"

The summary is the mean or max over the band. The matrix convention for directed SPIs is in GitBook `glossary-of-terms.md:27`: "The rows in this table reflect the sources while the columns reflect the targets."

## Canonical definition

The canonical form is Geweke (1982), J. Am. Stat. Assoc. 77:304. Take a bivariate process with transfer function H(f), innovation covariance Σ and spectrum S = HΣH^H. Then

f_{y→x}(f) = ln( S_xx(f) / [S_xx(f) − (Σ_yy − Σ_xy²/Σ_xx)|H_xy(f)|²] ) ≥ 0.

Under Geweke's regularity condition, the integral (1/2π)∫f_{y→x} equals the time-domain F_{y→x} = ln(Σ_xx^restricted/Σ_xx).

- **Parametric (canonical):** H and Σ come from a fitted VAR(p). The VAR is fitted by OLS or by Yule-Walker/Whittle (the LWR recursion of Morf, Vieira & Kailath 1978). The order p is fixed, or chosen by the Schwarz criterion: Lütkepohl 2005 §4.3 gives SC(m) = ln|Σ̂_m| + m K² ln T/T, minimised globally over 0..p_max.
- **Nonparametric (canonical):** Dhamala, Rangarajan & Ding (2008, PRL 100:018701 and NeuroImage 41:354) estimate S with Fourier or multitaper transforms, factorise it as S = ψψ^H with Wilson's (1972) algorithm, and set Σ = ψ₀ψ₀ᵀ and H = ψψ₀⁻¹.

The band mean or max is a pyspi construction. No literature summary corresponds to it. The equation numbers above are given from memory and were not re-checked against the papers.

## What is implemented (trace)

### Shared pyspi layer (`spectral.py`)
- `__init__` (:325-373) forces any fmin ≤ 0 to 1e-5 with a warning (:343-345). The identifier is still built from the original argument (:363, :369), so every "fmin-0" identifier runs with fmin = 1e-5. Probe: `sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-5` gives `_fmin 1e-05`. The full-band configs without fmin (order-1 and order-20 mean and max, and order-None max) get identifiers with `fmin-1e-05`. The supplement lists them as `fmin-0`. The computation is the same.
- `multivariate` (:417-440):
  - It keeps the bins with `freq >= fmin` and `freq <= fmax`. Both ends are inclusive.
  - It reduces each cell with `np.nanmean` or `np.nanmax`.
  - It sets a cell to NaN when more than 50% of its bins are NaN.
  - It catches `ValueError`, including `numpy.linalg.LinAlgError`, which subclasses ValueError (:438). The handler is meant to return an all-NaN P×P matrix (:440), but under the message-regex warnings filters numpy installs on import it never gets there:
    - `warnings.warn(err)` at :439 passes the exception instance as the message.
    - numpy registers message-regex `ignore` filters on import (`numpy/__init__.py:696-698`). There are four such filters after `import pyspi.calculator`.
    - Matching a regex against a non-string raises `TypeError: expected string or bytes-like object, got 'ValueError'` (or `'LinAlgError'`) inside the except block.
    - With Python's own default filters alone (no message-regex filter), `warnings.warn(ValueError('x'))` succeeds and emits a UserWarning. The TypeError needs the filters numpy installs on import, which every pyspi process has (probe in the project venv: 0 regex filters before `import numpy`, 3 after, and TypeError only after).
  - Probe `sgc_fix/r1.py`, fresh interpreter, filters as installed by numpy on import: narrowband_f0-0.1 at T = 256 (order-None) and shared_envelope at T = 256 (order-1 and order-20) all raise TypeError at `pyspi/statistics/spectral.py:439`.
  - What happens next depends on the caller:
    - A direct `multivariate()` call raises TypeError. That includes `tests/regression/harness.py:264-272 compute_spi` and `audit/checks/pyspi_outputs.py:148-162 run_one`.
    - `Calculator.compute` catches it (`calculator.py:336-337` at HEAD a42ebc1, `except Exception`) and sets the whole table to NaN. Its warning names TypeError, not the original message.
    - The all-NaN return at :440 runs only after `warnings.resetwarnings()` or inside `catch_warnings(record=True)` plus `simplefilter("always")`. The earlier probe p7 used that setup, which is why it saw 100% NaN.
  - Other exceptions reach `calculator.py:336-337` directly, which sets the whole table to NaN.
- In both paths A[a, b] is computed directly from `cache[0, f, a, b]`.

### Nonparametric (`spectral_connectivity`)
1. `NonparametricSpectralMultivariate._get_cache` (:64-83) builds `Multitaper(z(T×P), sampling_frequency=1)` with the library defaults (`transforms.py:74-102`): NW = 3, 5 tapers (:155-164, low-bias filter :668-673), constant detrend, one window spanning the whole series. It uses n_fft = next_fast_len(T) = T, so there is no zero padding (:181-190). It then builds `Connectivity.from_multitaper`.
2. `pairwise_spectral_granger_prediction` (`connectivity.py:697-748`) runs once per unordered pair:
   - It Wilson-factorises the 2×2 taper-averaged CSM (`minimum_phase_decomposition.py:161-222`, tolerance 1e-8, at most 60 iterations, non-convergence only logged).
   - It computes Σ = ψ₀ψ₀ᵀ (:1116-1142) and H = ψψ₀⁻¹ (:1145-1170).
   - It computes rotated[k,l] = Σ_ll − Σ_kl²/Σ_kk (:1193-1211).
   - It computes pred[k,l] = ln S_kk − ln(S_kk − rotated[k,l]|H_kl|²) (:1173-1180), with S_kk the raw multitaper auto-spectrum. Values ≤ 0, and logs of negative intrinsic power, become NaN.
   - A `LinAlgError` in a pair sets that pair to NaN.
3. Frequencies are the non-negative slice 0 … (n+1)//2 − 1 (:73-89, :174-180). The even-n Nyquist bin is dropped. With the fmin override the full band is k/T for k = 1 … T/2 − 1: 31 bins at T = 64 and 2047 at T = 4096.
4. `pyspi/_spectral_compat.py` only maps `np.in1d` for the connectivity module. It does not touch `minimum_phase_decomposition.py`. That file's Cholesky fallback handler (:48) names `xp.linalg.linalg.LinAlgError`, and numpy 2.5.4 has no `numpy.linalg.linalg` (probe: AttributeError).

### Parametric (`nitime`)
1. pyspi (:391-409) runs `GrangerAnalyzer(TimeSeries(z(P×T), sampling_interval=1), order, max_order=50)`.
   - Pairs are (i, j) with i < j (`granger.py:106-112`). Each pair gets its own bivariate VAR (:116-131).
   - pyspi sets F[i,j] = causality_xy[i,j] (GC i → j) and F[j,i] = causality_yx[i,j] (GC j → i).
   - It then transposes with axes [3,2,1,0], so cache[0,f,a,b] = F[b,a,f].
2. `fit_model` (`granger.py:17-71`) fits the VAR:
   - Fixed order: lag = order + 1, then `autocov_vector` and `lwr_recursion` give VAR(order) (:41-44).
   - order = None: for lag in range(1, 50) it fits VAR(lag − 1) and computes BIC = 2 ln det Σ + 2p²m ln(N_total)/N_total with N_total = 2T (`utils.py:2318-2371`, `granger.py:38`). It **breaks at the first increase** (:54-56). If it never breaks it raises ValueError (:66-69). Orders 0 … 48 are fitted, but only 0 … 47 can be returned (check stage, maxorder runs): a BIC still falling at order 48 ends the loop without a break and raises ValueError. Order 0 gives H = I and GC ≡ 0.
3. `crosscov_vector` (`utils.py:2127-2177`) averages x(t)y(t−k) over the N−k overlapping samples. It does **not** demean. `lwr_recursion` (`autoregressive.py:168-249`) solves the block Yule-Walker system.
4. `granger_causality_xy` (`autoregressive.py:455-525`) applies the standard Geweke rotated-noise formula (:485-509). It evaluates H through `freq_response` → `scipy.signal.freqz(worN=513, whole=False)` (`nitime/algorithms/spectral.py:737-764`, `scipy/signal/_filter_design.py:516-526`). The evaluation grid is ω_k = πk/513 (f = k/1026). nitime labels it with `get_freqs` = linspace(0, 0.5, 513) = k/1024 (`utils.py:1207-1211`). Probe: the label 0.25 is the point 0.24951 and the label 0.5 is the point 0.49903.

## Deviations common to all units (candidates)

- **Direction is the transpose of the documented convention.** In every sgc unit, A[i, j] is GC from j to i: target in the row, source in the column. The glossary documents rows = sources, and the TE and GC SPIs in `infotheory.py` use `src = data[i]`.
  - The trace gives this for the nonparametric path: pred[k,l] is f_{l→k}, and pyspi does not transpose.
  - The trace gives the same for the parametric path: nitime's gc_xy[i,j] is i → j, and pyspi transposes it.
  - Probes on var2_bivariate (truth 0 → 1 only) at T = 4096:

    | Unit | A[0,1] | A[1,0] |
    |---|---|---|
    | nonparametric | 0.123 | 0.402 |
    | order-None | 0.0008 | 0.281 |
    | order-1 | 0.014 | 0.250 |
    | order-20 | 0.005 | 0.286 |

  - On delayed_noise at T = 1024 (x0 leads) the large values sit in column 0.
  - On gauss_ar1_pair the analytic GC is 0.0997 for 1→0 and 0.0470 for 0→1, and pyspi has A[0,1] > A[1,0].
  - This is not in KNOWN_ISSUES.md.
- **Frequency grid and labels.** The supplement documents 125 bins on [4/T, fs/2]. Neither path uses that grid:
  - Nonparametric: k/T, with DC and Nyquist excluded.
  - Parametric: 513 freqz points, mislabelled by a factor 1026/1024.
  - In both paths fmin-0 means fmin = 1e-5, and both half-bands include the 0.25 point.
- **All-or-nothing failure.** A single bad pair aborts the whole SPI:
  - Parametric: any ValueError. The pyspi handler then raises TypeError at `pyspi/statistics/spectral.py:439` (see the shared layer). A direct call raises, and the Calculator table is all NaN.
  - Nonparametric: the AttributeError escape described in its section.

## spectral.sgc.nonparametric

- **Kernel.** The kernel matches the Geweke/Dhamala formula. On an exact two-sided VAR(2) spectrum, Wilson reconstructs S to 3e-9 and recovers Σ exactly to 6 decimals (probe p11).
- **Estimator.** The single-window, 5-taper spectral estimate gives about 10 degrees of freedom per bin whatever T. The candidate is that the GC estimate is biased upwards by about 0.1–0.15 nats and that the bias **does not decrease with T**.

  Probe p3/p4, iid_gaussian full-band mean (two directions):

  | T | NW = 3 (5 tapers) | NW = 10 | NW = 40 |
  |---|---|---|---|
  | 64 | 0.356 / 0.153 | | |
  | 256 | 0.146 / 0.106 | | |
  | 1024 | 0.114 / 0.121 | | |
  | 4096 | 0.127 / 0.121 | 0.025 | 0.0044 |

  On var2_bivariate the true direction reads 0.40–0.44 against an analytic full-band mean of 0.289. The null direction stays at 0.10–0.13.
- **Wilson factor on estimated spectra.** On the rough single-window estimates the factor does not reproduce the CSM. The relative reconstruction error runs from 2.8e-3 (iid, T = 4096) to 0.38 (logistic_maps). "Maximum iterations reached" was logged for logistic_maps at T ≥ 1024 and for trending at T = 256 and 1024. For logistic_maps the error is 0.323 at T = 1024 and 0.38 at T = 4096.

  NaN bins appear among the kept full-band bins (probe `sgc_fix/r2.py`):

  | Dataset | Cell | NaN bins |
  |---|---|---|
  | logistic_maps, T = 1024 | [0,1] (GC 1 → 0) | 9.0% |
  | logistic_maps, T = 1024 | [1,0] (GC 0 → 1) | 40.9% |
  | logistic_maps, T = 1024 | mean of the two directions | 25% |
  | trending, T = 256 | mean of the 12 off-diagonal cells | 6.4% |
  | trending, T = 256 | [1,3] (GC 3 → 1), the largest | 66.9% |

  The trending [1,3] cell is above the 50% rule, so pyspi outputs NaN for it. Hypothesis: the cause is the lack of smoothing and zero padding, not the iteration.
- **Error path.** When the Cholesky initialisation fails (identical channels, as in shared_envelope where x2 == x0), the handler at `minimum_phase_decomposition.py:48` raises AttributeError under numpy 2.5.4. The Calculator then NaNs the whole sgc_nonparametric table (probe p7). Under numpy < 2.4 as checked (2.2.6 and 2.3.3 still resolve `xp.linalg.linalg.LinAlgError` with a DeprecationWarning; 2.4.6 and 2.5.4 raise AttributeError), or under a library release whose handler catches `xp.linalg.LinAlgError` (v2.0.1 does), the fallback would use an unseeded random start.
- **NaN rule.** The pack hypothesis that values ≤ 0 become NaN is confirmed (`connectivity.py:1179`). With an exact factor the rotated variance is ≥ 0, so a NaN bin marks a factorisation failure, not zero causality. nanmean then silently averages the remaining bins.
- **Bin conventions (candidate, low impact).** These are not documented elsewhere: fmin-0 runs with fmin = 1e-5, both half-bands include the 0.25 bin, and the documented 125-bin grid is not used.
- **Known issues cited.** These are cited, not new: the 1.1.2 pin (tests/README dependency constraints), the cache first-insert keying (tests/README "worth knowing"), and the even-n Nyquist drop (DEPENDENCY_COMPATIBILITY_REVIEW Nyquist handling, REVIEW M4). The Nyquist drop is kept out of the bin-convention candidate.

**Proposed checks:**
1. Kernel against the analytic Geweke formula. Feed the exact spectrum of var2_bivariate and gauss_ar1_pair as two pseudo-tapers √2·H·L·e_m. Tolerance 1e-6.
2. Pipeline against a from-scratch implementation: scipy dpss, numpy FFT, numpy Wilson, Geweke formula, pyspi band rule. Tolerance 1e-6.
3. Direction check on var2_bivariate, gauss_ar1_pair and delayed_noise.
4. Statistical bias against the analytic band means.
5. iid null, with the taper count varied.
6. Relation of the band mean to the time-domain Geweke F from OLS (statistical).
7. Qualitative error path on shared_envelope.

## spectral.sgc.parametric_order-None

- **Order search (candidate, high impact).** The search stops at the first local minimum and admits order 0. Order 0 makes GC exactly 0.
  - On delayed_noise at T ≥ 256, every pair with relative delay ≥ 2 gets order 0 and A = 0. At T = 1024 only x0 → x1 is detected (A[1,0] = 4.71).
  - For pair (0,3) at T = 256, BIC is −0.0257 at order 0 and 0.044 at order 1, so the loop breaks there. Order 5 would reach −6.78.
  - iid_gaussian gets order 0 at every T, so its GC is exactly 0.
- **Failure mode (candidate).** The noise-covariance determinant can be ≤ 0 because of the N−k divisor. BIC is then NaN, and NaN never triggers the break, so the loop runs out and nitime raises ValueError. Probes show this for narrowband f0 = 0.1 and 0.25 (all T), shared_envelope (all T), logistic_maps (T ≥ 1024) and delayed_noise (T = 64). Under the message-regex warnings filters numpy installs on import the pyspi handler then raises TypeError at `pyspi/statistics/spectral.py:439`. A direct call raises, and the Calculator table is all NaN.
- **Penalty.** N_total = 2T roughly halves the per-observation Schwarz penalty. Hypothesis at the trace stage; the check stage quantifies it (`ntot` run, see Checks).
  - nitime contradicts its own definition here. The `bayesian_information_criterion` docstring (`nitime/utils.py:2348-2358`) cites Ding et al. (2006) eq. (51) and defines N_total as "the number of time-points", which is T. `granger.py:38` passes `n_process * x1.shape[-1]` = 2T.
- **"Maximum lag of 50".** In practice the search fits orders 0 … 48 and can return 0 … 47.
- **Where it works.** On well-specified data it behaves: var2_bivariate gets order 2 and A[1,0] = 0.281 at T = 4096 (analytic 0.289), and gauss_ar1_pair gets order 1.

**Proposed checks:**
1. Kernel: from-scratch Geweke on nitime's fitted (coef, ecov). Tolerance 1e-10.
2. Kernel: from-scratch block Yule-Walker with divisor N−k, compared with divisor N and with OLS.
3. Exact replica of the order rule, compared with the standard global-minimum BIC.
4. Direction check.
5. Statistical convergence to the analytic GC. Also run statsmodels OLS coefficients through the from-scratch formula.
6. Null check on iid_gaussian and bivariate_gaussian.
7. Error-path check on narrowband, shared_envelope, logistic_maps and delayed_noise at T = 64. Expect nitime's ValueError. On the pyspi side, expect a TypeError from `compute_spi` and an all-NaN Calculator table. Do not expect an all-NaN matrix from `multivariate`.
8. Offset sensitivity with normalise=False.

## spectral.sgc.parametric_order-1

Fixed VAR(1) fitted by LWR. This is a misspecification for var2_bivariate, which is VAR(2): at T = 4096 the result is 0.250 against the analytic 0.289. It is exact in model class for gauss_ar1_pair.

On delayed_noise at T = 64 there are small negative bins (min −6.2e-5) and one NaN cell from the 50% rule.

Offset sensitivity (candidate): nitime does not demean. With normalise=False, x·[2, 0.5] + [3, −5] gives 0.043 instead of 0.289. Scaling alone gives 0.288. The nonparametric path is unaffected.

On shared_envelope, the identical channels make `lwr_recursion` raise a singular-matrix LinAlgError. Under the message-regex warnings filters numpy installs on import the pyspi handler raises TypeError ("got 'LinAlgError'") at `pyspi/statistics/spectral.py:439`. A direct call raises, and the Calculator table is all NaN. Only under `catch_warnings(record=True)` does `multivariate` return 100% NaN.

**Proposed checks:** as for order-None, without the order-search checks. Add a check that confirms misspecification bias on var2_bivariate.

## spectral.sgc.parametric_order-20

At T = 64, a VAR(20) has 80 coefficients to fit from 64 samples. With the N−k divisor this gives indefinite noise covariances: σ < 0 or det < 0. The results are negative GC values and NaN bins. Per-bin NaN fractions are 20% for iid, 55% for var2 and 65% for gauss_ar1.

pyspi output at T = 64:

| Dataset | Result |
|---|---|
| iid_gaussian | min −1.126, 4 negative cells, 20% NaN cells |
| bivariate_gaussian | min −0.581 |
| var2_bivariate | all NaN |
| gauss_ar1_pair | all NaN |

This explains the partial_nan status that tests/README.md:249 records for the six order-20 goldens at small size (5×64) without giving a cause. At T = 256 the iid null shows no negatives. At T = 4096 var2_bivariate gives 0.286 against the analytic 0.289.

**Proposed checks:** as for order-1. Add a degeneracy check at T = 64: NaN fraction, negative cells and det(ecov) sign.

## Open questions

1. Should the target-by-source direction be presented as a defect or as a documented convention to fix in the docs? The glossary and the other directed SPIs disagree with sgc. Is there an upstream pyspi issue?
2. Resolved: the 185-metric EEG store is not affected. `audit/env/retained_metrics.txt` contains no sgc identifier (grep count 0).
3. Hypothesis: the nonparametric Wilson reconstruction error on single-window estimates comes from n_fft = T with no smoothing. Would zero padding or larger NW remove it? Not tested beyond the bias-versus-NW probe.
4. Ding et al. (2006) eq. (51): does "total number of data points" mean T or n_channels·T? This is now secondary. nitime's own docstring (`nitime/utils.py:2348-2358`) defines N_total as the number of time-points (T), so N_total = 2T departs from nitime's stated definition however the paper reads.
5. Geweke's integral identity holds under a condition on the normalised transfer function. Equation numbers and the exact condition were cited from memory and should be checked against Geweke (1982) before the identity is used as a check criterion.
6. Does any EEG-like input trigger the nonparametric AttributeError path, for example flat or duplicated channels? It would turn every nonparametric table NaN.
7. Should the `warnings.warn(err)` defect at `pyspi/statistics/spectral.py:439` be reported upstream? It makes the ValueError handler dead code in any process that has imported numpy.

## Checks

Manifest `audit/checks/manifest/sgc.json` (339 runs), references `audit/checks/refs/sgc.py` (comparison venv, no pyspi, no nitime, no spectral_connectivity), probes `audit/checks/probes/sgc_probes.py`. Results are in `audit/checks/out/results/sgc.json`: 303 pass, 36 fail, 0 without reference. Run ids below drop the `spectral.sgc.` prefix. Every number is quoted from results.json (`ref_*` keys included).

How the references work:

- **Nonparametric.** The from-scratch pipeline uses `_kernels_spectral`: `multitaper_fft` (NW = 3, 5 tapers, constant detrend, n_fft = T), `cross_spectral_matrix`, `wilson_factorise` (tolerance 1e-8, 60 iterations, the library's plus-operator) and `granger_nonparametric` (values ≤ 0 set to NaN). It then applies the pyspi band rule.
- **Parametric.** The from-scratch pipeline computes non-demeaned autocovariances with divisor N−k. It solves the block Yule-Walker system directly with `numpy.linalg.solve`. This is the same system nitime solves with the LWR recursion, by a different algorithm. Order None uses a replica of nitime's BIC loop. The Geweke ratio is evaluated at w = πk/513, and bins are selected by the labels k/1024.
- **Statistical runs.** Each uses 200 fresh draws of the generating model through the same from-scratch estimator. The band is |pyspi − value| ≤ z_(1−0.05/2m) · SD, familywise over the m judged cells.
- **`conv` runs.** These judge consistency. They pass when the Monte Carlo bias at T = 4096 is at most a quarter of the bias at T = 256.
- **`dir` runs.** These pass when A[target, source] > A[source, target] on every true link (row = target). Each run also reports how many links the documented rows = sources reading gets right.

### spectral.sgc.nonparametric — variant, 7

- **Implementation.**
  - All 44 `impl` runs pass: six identifiers on seven dataset/path combinations, plus var2 at T = 256 and T = 4096. Max_abs is 3.9e-11 and max_rel 8.4e-12, with equal NaN masks.
  - That includes logistic_maps (the Wilson iteration does not converge, 25% NaN bins) and trending at T = 256 under the pipeline path (cell [1,3] is NaN by the 50% rule).
  - The per-bin cache matches to 8.4e-11 (`bins`). The library CSM matches the from-scratch CSM to 7.8e-12 (`libcsm`). Wilson plus Geweke applied to the library's own CSM matches pyspi per bin to 4.6e-12 (`wilsonlib`).
- **Kernel against the analytic value.** The library was fed an exact VAR spectrum through P pseudo-tapers (probe `exact_lib`). It matches the analytic Geweke values from the true (A_k, Σ) to 1.9e-15 on var2 and 4.2e-10 on gauss_ar1_pair (`exact`). On var2 every bin of the zero-causality direction 1 → 0 is NaN in both, because ln 1 = 0 falls under the ≤ 0 rule.
- **Estimator.**
  - var2_bivariate mean-full, A[1,0]: 0.412 / 0.436 / 0.402 at T = 256 / 1024 / 4096, against the population value 0.289. The non-causal A[0,1] is 0.103 / 0.133 / 0.123 against 0.
  - All 7 `truth` runs fail. Max_z is 3.4 / 9.7 / 17.1 for mean-full, and 3.8 for max-full, where the max over bins is 2.25 against 0.49.
  - Both `mcexp` runs pass (max_z 0.59 and 0.60), so pyspi is exactly the 5-taper estimator.
  - The bias does not shrink with T. `conv` on var2: 0.128 / 0.126 / 0.125. `conv` on iid: 0.125 / 0.127 / 0.126. Both fail.
  - The iid null mean is 0.113 / 0.122 / 0.126 at T = 256 / 1024 / 4096, and all three `null` runs fail. On the same data the from-scratch estimator gives 0.023 with NW = 10 (19 tapers) and 0.0051 with NW = 40.
  - gauss_ar1_pair: A[0,1] = 0.283 against 0.0995, and A[1,0] = 0.138 against 0.047.
  - The integral identity holds for the true models. On var2, the Kolmogorov F_{0→1} = 0.289056 equals ∫f = 0.289056. On gauss_ar1_pair, 0.099626 and 0.046992 match in both directions (`ref_analytic_F_kolmogorov_*`, `ref_integral_f_*`).
- **Direction.** All 5 `dir` runs pass under row = target, 9 of 9 links. The documented rows = sources reading gets 0 of 9.
- **Error path.**
  - `shared_envelope|256|pipeline|err` passes. The library and the direct call both raise AttributeError, and a real pyspi Calculator with a one-SPI config gives an all-NaN table.
  - The same check at T = 1024 **fails**: no exception is raised. In `shared_envelope|1024|pipeline|dup` the Cholesky start succeeds on a numerically singular lag-0 matrix (eigenvalues 0.0 and 1.96). pyspi then returns finite A[0,2] = 0.2457 and A[2,0] = 1.1e-12 for a channel and its exact copy.
  - Sweeps: at T = 256, shared_envelope, eeg:duplicate_channel and eeg:near_duplicate_1e-9 raise. At T = 1024 none raise. A flat channel is rejected by `Data()` before the SPI runs.
- **Affine.** `affine` and `scale` fail at 1e-6 with max_abs 1.1e-4. The constant detrend removes the offset. The residual comes from Wilson non-equivariance under channel scaling: the from-scratch value converges to the same scale-dependent fixed point at tolerance 1e-14. pyspi z-scores by default, so this does not arise in normal use.
- **Factorisation and NaN bins** (`bins` extra keys). Reconstruction errors: logistic_maps 0.323 (not converged), trending T = 256 not converged, var2 0.0206, iid 0.0132. Zero padding to 4T cuts these to 0.185 / 0.026 / 3e-5 / 1.5e-4, and cuts the NaN bins on logistic_maps from 25% to 0.12% and on trending from 6.4% to 0.03%. NW = 20 removes the NaN bins entirely. Padding leaves the null bias unchanged (iid 0.1216). Only more tapers reduce it.
- **Verdict.** variant, score 7, rubric anchor 7 ("expectation over 5 tapers"). The new findings are the direction transpose, the T-independent bias, and the duplicated-channel behaviour that depends on round-off.

### spectral.sgc.parametric_order-None — variant, 6

- **Implementation.**
  - 38/38 `impl` runs pass (max_abs 7.7e-14).
  - `yw`: nitime's LWR (coef, ecov) equal the direct Yule-Walker solve to 6.7e-16.
  - `geweke`: the Geweke step on nitime's own fits matches pyspi's cache to 1.1e-13, with cache[f, a, b] = GC b → a.
  - `grid` is exact: labels linspace(0, 0.5, 513), evaluation points k/1026, maximum label shift 9.7e-4. Its effect on the population band statistics is at most 6e-4.
- **Order rule.**
  - `orders`: 11 of 11 datasets give exactly the replica's order per pair.
  - On delayed_noise (relative delays 1..5) nitime picks (1, 0, 0, 0, 0, 0). The standard Schwarz criterion (statsmodels, OLS, global minimum) picks (1, 3, 5, 2, 4, 2), which are the true delays.
  - With Ntot = T instead of 2T, nitime's rule picks the same order for every pair on all 11 suite datasets (`ref_ntotT*`). None of them has a borderline order.
  - `parametric_order-None|var2_bivariate|256|pipeline|ntot` (exact pass, max_abs 7.1e-15) adds 200 bivariate VAR(2) draws with a weak second lag: A1 = [[0.4, 0], [0.3, 0.3]], A2 = a2·[[1, 0], [−1, 1]], a2 ~ U(0.02, 0.25), T = 256. The probe generates them and records nitime's chosen order and pyspi's matrix.
    - The replica with Ntot = 2T reproduces nitime's order on 200/200 draws and pyspi's A[1,0], A[0,1] to 6.7e-16 (`ref_replica_vs_pyspi_max_abs`).
    - With Ntot = T the same rule picks a different order on 42/200 draws (`ref_n_order_differs_ntotT`), always lower (`ref_n_ntotT_lower` 42, `ref_n_ntotT_higher` 0): 40 draws 2 → 1, one 3 → 2 and one 3 → 1, the split the order counts below force. Order counts: {1: 90, 2: 108, 3: 2} with 2T, {1: 131, 2: 69} with T, {1: 133, 2: 67} for the standard Schwarz criterion.
    - The T-rule agrees with the standard Schwarz order (statsmodels, OLS, global minimum) on 190/200 draws, the shipped 2T-rule on 155/200.
    - Where the order changes, the full-band mean A[1,0] changes by a median 17% (`ref_median_rel_A10_change_at_ntotT` 0.175) and at most 0.060.
    - This run was added after review. The reviewer's probe (`verify_spectral_directed_sgc/t6_ntot.py`, another seed, nitime `fit_model` with Ntotal // 2) found 48/200.
  - `maxorder` (exact): fit_model with max_order m = 2, 3, 4 raises on var2 at T = 4096, and m ≥ 5 returns 2. So orders 0..m−2 are fitted and only 0..m−3 can be returned. With the default 50, that is 0..48 fitted and 0..47 returnable.
- **Convergence.**
  - var2 A[1,0] is 0.313 / 0.315 / 0.281 at T = 256 / 1024 / 4096, against 0.289. All 7 `truth` runs pass (max_z ≤ 1.69).
  - `conv` passes: bias 0.0074 / 0.0044 / 0.00065.
  - The nulls are exactly 0, because order 0 is chosen.
  - LWR against OLS at T = 1024 (`ols`, statsmodels): 0.0066 on var2 (sampling SD 0.030) and 8.9e-5 on gauss_ar1_pair. Divisor-N Yule-Walker differs from pyspi by 9.7e-4.
- **Error path.** All five `err` runs pass:
  - narrowband 0.1 and 0.25 at T = 256, logistic_maps at T = 1024, delayed_noise at T = 64, shared_envelope at T = 256.
  - The replica's BIC turns NaN from order 3 (narrowband, logistic) or order 1 (delayed_noise T = 64, shared_envelope), so the loop never breaks.
  - nitime raises ValueError, the direct call raises TypeError at `pyspi/statistics/spectral.py:439`, and the real Calculator returns all-NaN.
  - The sweep failures are the ones the replica predicts: narrowband ×3, plus logistic_maps and logistic_maps_bidir at T = 1024.
- **Direction and offset.**
  - `dir` passes on var2 and gauss_ar1_pair. It fails on delayed_noise because 5 of 6 links are tied at exactly 0, not because of orientation.
  - `affine` fails: A[0,1] / A[1,0] = 0.0322 / 0.3619 against 0.0016 / 0.3142. `scale` passes (2.8e-16).
- **Verdict.** variant, score 6. The method deviates from both the supplement and the canonical form in interpretable ways: first-local-minimum BIC with order 0 allowed, NaN-BIC non-termination, a BIC penalty with Ntot = 2T that selects higher orders on borderline models, and LWR instead of OLS.

### spectral.sgc.parametric_order-1 — variant, 7

- **Implementation.** 38/38 `impl` runs (7.7e-14), `yw` 4.4e-16, `geweke` 1.1e-13.
- **Convergence.**
  - gauss_ar1_pair, a true VAR(1), passes its `truth` run (z 1.62).
  - var2_bivariate is a VAR(2), so order 1 is misspecified. The estimate converges to the population VAR(1) projection: 0.2607 for 0 → 1, and a spurious 0.0159 for the absent 1 → 0. All four `proj` runs pass (max_z ≤ 0.82).
  - Against the VAR(2) truth (0.2893 / 0), 6 of 7 truth runs fail (max_z 3.61 at T = 4096).
  - `conv` on var2 fails: the bias against the truth is 0.026 / 0.028 / 0.028, while the bias against the projection is 0.0046 / 0.0004 / 0.0009. `conv` on iid passes (ratio 15.2).
  - That is the recorded consequence of order 1: a different estimand, about 10% low on the true link, plus a spurious reverse link.
- **Other results.**
  - LWR against OLS: 0.0030 (var2) and 8.9e-5 (gauss_ar1_pair).
  - `dir` fails on delayed_noise because delays above 1 are outside a VAR(1): those links are about 1e-3 in both directions.
  - `null` on bivariate_gaussian fails narrowly (max_z 2.06).
  - `affine` fails (0.245); `scale` passes.
- **Short series, delayed_noise T = 64** (`deg`, `indef`, `yw`, `impl`, all passing). Pair (0,1) has an indefinite LWR noise covariance (Σ_11 = −0.021). Its 1 → 0 bins are negative at all 513 points (A[0,1] = −6.1e-5). Its 0 → 1 bins are NaN, so the true link A[1,0] is NaN.
- **Verdict.** variant, score 7, rubric anchor 7 ("LWR instead of OLS").

### spectral.sgc.parametric_order-20 — variant, 7

- **Implementation.** 41/41 `impl` runs (3.4e-11), including T = 64. `yw` 9.6e-12, `geweke` 9.6e-12.
- **Convergence.** Order 20 at T = 256 against T = 1024:
  - Causal A[1,0]: 0.354 / 0.328 / 0.286 at T = 256 / 1024 / 4096, against 0.289.
  - Non-causal A[0,1]: 0.081 / 0.017 / 0.0047.
  - `conv` passes on var2 (bias 0.087 / 0.020 / 0.0047, ratio 18.5) and on iid (ratio 19.3).
  - The `truth` and `null` failures are the positive small-sample bias of the non-causal cells against a zero target. Examples: iid null mean 0.087 at T = 256 and 0.016 at T = 1024, against Monte Carlo 0.090 and 0.020. `mcexp` passes.
  - LWR against OLS: 0.0080 (var2, SD 0.032).
- **T = 64** (`deg` and `indef`, all passing).
  - NaN bins: 19.8% (iid), 55.2% (var2), 65.1% (gauss_ar1_pair). Negative bins: 20.2%, 44.8%, 34.9%.
  - The var2 and gauss_ar1_pair matrices are all NaN. The iid minimum is −1.126.
  - Every anomaly sits in a pair whose LWR noise covariance is indefinite (4 of 10 pairs on iid).
  - This explains the partial_nan status of the small order-20 goldens. The goldens are snapshots and were not used as a reference.
- **Verdict.** variant, score 7, rubric anchor 7.

### Candidates

Confirmed:

- **Direction (all four units).** This is a new finding. It is not in KNOWN_ISSUES.md.
- **Nonparametric.** Confirmed: the T-independent bias, the AttributeError path (refined below), factorisation accuracy and NaN bins, the bin conventions, and NaN-bin dropping.
  - The AttributeError path is round-off dependent. At some lengths the duplicated channel does not raise and gives finite values instead.
  - NaN handling is corrected: an exactly zero GC is also turned into NaN.
- **Parametric.** Confirmed: offset sensitivity, grid labels, and the all-or-nothing failure through the TypeError-raising handler.
- **Order-None.** Confirmed: first-local-minimum order 0, NaN-BIC non-termination, and the BIC penalty with Ntot = 2T (42/200 borderline VAR(2) draws change order under Ntot = T; none of the 11 suite datasets does). The maximum order is refined to 0..48 fitted and 0..47 returnable.
- **Order-1 and order-20.** Confirmed: negative values from indefinite covariances. Order-20 also confirmed: short-series degeneracy.

Refuted: none. The BIC-penalty candidate was first marked refuted on the 11 suite datasets alone. Review showed that those datasets have no borderline order, and the `ntot` run confirms it.

Inconclusive:

- **Order-None negative values.** No order-None run produced a negative value. Every indefinite fit ended in ValueError instead.

### Trace-stage fields corrected

- `calculator.py:301-303` became `calculator.py:336-337`, and `292-303` became `325-339`, at HEAD a42ebc1.
- "orders 0..48 reachable" became "0..48 fitted, 0..47 returnable" (order-None record and this note).
- The NaN-handling candidate is qualified as described above.
- "under the default (warnings) filters" became "under the message-regex warnings filters numpy installs on import" in all four records and this note. Python's own default filters alone let `warnings.warn(err)` through as a UserWarning.
- The two remaining trace issues passed to this stage were already resolved in the files:
  - The open-question list runs 1 to 7 in order.
  - The records cite the supplement.txt lines 809-818 and 570/589-599 and no longer contain the stale "pack lines 826-867" range.

### Kernel note

The shared `_kernels_spectral.wilson_factorise` and `granger_nonparametric` were used unchanged.

For the parametric path, `refs/sgc.py` has its own `geweke_ratio`. It uses nitime's ratio inside the log, `ln(S/intr)`, not `ln S − ln intr`. The two forms differ when the noise covariance is indefinite: when both S and intr are negative, the ratio form gives a finite value and the difference form gives NaN. `granger_nonparametric` uses the difference form, so it would not reproduce the T = 64 order-20 values.

## Open questions: answered

1. **Direction: defect or documentation fix?**
   - Established empirically: every sgc unit has row = target (the `dir` runs above). The glossary and the TE/GC SPIs use row = source.
   - Because the values match the references once transposed and only the orientation differs, transposing the output in `SpectralGrangerCausality._get_cache` or documenting the convention are both one-line remedies.
   - Whether an upstream pyspi issue exists was not checked. It needs a search of the pyspi issue tracker.
2. **Resolved at the trace stage.** The EEG store is not affected.
3. **Wilson error and NaN bins: zero padding or more tapers?** Both reduce them, so the hypothesis is confirmed (`bins` runs).
   - Zero padding to 4T cuts the reconstruction error by 2 to 4 orders of magnitude on var2, iid and trending (logistic_maps only from 0.323 to 0.185), and the NaN bins to ≤ 0.12%.
   - NW = 20 removes the NaN bins.
   - Padding to 2T is not monotone: on logistic_maps the reconstruction error rises to 3.36 and does not converge.
   - Padding does not reduce the 0.12-nat null bias. Only more tapers do (NW = 10: 0.023).
4. **Ding, Chen & Bressler 2006 eq. (51).** Checked against arXiv q-bio/0608035v1, pp. 15-17. Eq. (51) is BIC(m) = 2 log det Σ + 2p²m log N_total / N_total, with N_total "the total number of data points from all the trials" (stated under eq. 50).
   - Read with the trial structure of that section, the count is time points summed over trials, which is T for one trial. That matches nitime's own docstring. nitime's 2T counts channels as well.
   - On the 11 suite datasets the choice between T and 2T never changed the selected order (`orders` runs, `ref_ntotT*`), because none of them has a borderline order.
   - On borderline models it does: `parametric_order-None|var2_bivariate|256|pipeline|ntot` shows a lower order with Ntot = T on 42/200 weak-second-lag VAR(2) draws at T = 256, and a median 17% change in A[1,0] where the order changes. The T-rule matches the standard Schwarz order more often (190/200 against 155/200).
   - The same section (eq. 48) uses the divisor N − n for multi-trial data. It notes that "for a single short trial of data one uses the divisor N ... to reduce inconsistency". nitime applies N − n to a single trial, which is the root of the indefinite covariances at T = 64.
5. **Geweke's integral identity.** Ding et al. 2006 eq. (28) states the time/frequency relation "under general conditions" without giving them. Geweke (1982) itself was not retrieved, so the exact condition is still unchecked.
   - Numerically it holds for both test models: var2 F_{0→1} = 0.289056 = ∫f, and gauss_ar1_pair 0.099626 / 0.046992 in both directions, with instantaneous correlation 0.2 (`truth` run extra keys).
   - The check stage therefore used the per-frequency population values directly and did not rely on the identity.
6. **Can EEG-like inputs trigger the nonparametric AttributeError?** Yes, but only for exact or near-exact duplicate channels, and depending on round-off.
   - At T = 256 an exact duplicate and a duplicate plus 1e-9 noise raise, and a scaled copy does not. At T = 1024 none raise.
   - When nothing is raised the copy pair gets finite, meaningless values (`dup`: 0.2457 / 1.1e-12).
   - A flat channel is rejected earlier by `Data()` (NaN after z-scoring). A copy with 1e-6 noise is fine.
   - Real EEG with a bridged or duplicated electrode could therefore NaN the whole SPI or silently return values for the copy pair.
7. **Report the `pyspi/statistics/spectral.py:439` defect upstream?** Yes, it is reproducible. Every parametric `err` run shows TypeError from `warnings.warn(err)` under the message-regex warnings filters numpy installs on import (5 message-regex filters in the probe process), and a real pyspi Calculator turns it into an all-NaN table. The fix is `warnings.warn(str(err))`.
   - That fix only restores the intended all-NaN matrix. One bad pair would still NaN all pairs.
   - Whether an upstream report already exists was not checked.

## Optimisation

Sources: walls from `audit/profiles/summary.json` (fresh / family-warm; quiet machine; 118x1024 walls are single repeats), frame shares from cProfile. The official profiles at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`) are dominated by the P-independent DPSS cost for every multivariate unit, so for ranking at 118 channels this group added cProfile runs at 118x1024 (32x1024 for sgc nonparametric): `audit/profiles/micro/prof_<unit>__<shape>.pstats`, written by `audit/profiles/micro/prof_spectral.py`, indexed with walls, peak RSS and thread settings in `audit/profiles/micro/prof_spectral__index.json` (OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1; no numba or JIDT code on these paths). Those runs overlapped other Workflow 3 groups; the memory-bound multitaper units varied 2-7x in wall between runs (cohmag: 6.4 s on a first run, 0.93 s on the kept run, 2.42 s in summary.json), so only shares are quoted from them. Percentages are of the profiled `harness.compute_spi` frame. Micro-benchmarks (the group's two): `audit/profiles/micro/opp_spectral_csm-1.{py,json}` and `audit/profiles/micro/opp_spectral_csm-shared-csm.{py,json}`. A candidate that reproduces the current output "matches the baseline"; that says nothing about whether either is mathematically right. Nothing in `pyspi/` was changed.

| unit | 8x1024 fresh / fw (s) | 32x1024 fresh (s) | 118x1024 fresh (s), mode | peak RSS 118 fresh (MB) |
|---|---|---|---|---|
| spectral.sgc.nonparametric | 0.5937 / 0.5962 | not run (infeasible list) | none in summary.json; ~146 s by unordered-pair scaling of the 8x1024 wall (0.594 s x 6903/28) | n/a |
| spectral.sgc.parametric_order-None | 0.0488 / 0.0501 | 0.8743 | 9.940, measured | 843 |
| spectral.sgc.parametric_order-1 | 0.0118 / 0.0131 | 0.2037 | 2.987, measured | 838 |
| spectral.sgc.parametric_order-20 | 0.0617 / 0.0623 | 1.0759 | 15.249, measured | 844 |

Family-warm records above 8x1024 are marked infeasible for every sgc unit because the family-warm job contains the nonparametric unit from the infeasible list (`reason: --infeasible: ['spectral.sgc.nonparametric']`). summary.json projects the nonparametric unit like the other infeasible-list units, `extrapolated-pairs-from-8`, using the unordered pair count of the delegate's pair loop (`PAIR_LOOP_BASIS` in `profile_units.py`, since the class itself is multivariate): 0.594 s for 28 pairs at 8x1024 gives 146.4 s per window fresh and 147.0 s family-warm at 118x1024 (until Workflow 4c it had no projection and its avenue ranked with no basis). Band identifiers share one cache per (method, order, max_order): increments after the first are 0.0002-0.0003 s. Not in the EEG selection.

**Hotspots.**
- nonparametric (official 8x1024 profile; the extra 32x1024 profile `audit/profiles/micro/prof_spectral.sgc.nonparametric__32x1024.pstats`, 496 pairs, keeps the shares): `minimum_phase_decomposition` 95.9% (97.6% at 32x1024), `numpy linalg solve` on 2x2 stacks 65.5% (65.5%), factorisation self 14.7%, `_get_causal_signal` 10.9%; the multivariate Multitaper and CSM are 2.4%.
- parametric order 20 (`prof_spectral.sgc.parametric_order-20__118x1024.pstats`): `nitime lwr_recursion` 65.5% (self 47.1%, `scipy.linalg.inv` inside it 15.8%), `granger_causality_xy` 24.1% (freqz 18.8%), `crosscov_vector` 8.7%.
- parametric order None (BIC search, `..._order-None__118x1024.pstats`): `lwr_recursion` 42.9% over 41,626 calls (~6 per pair), `granger_causality_xy` 32.5% (freqz 21.8%), `autocov_vector` 14.7%.
- parametric order 1 (`..._order-1__118x1024.pstats`): `granger_causality_xy` 83.8%, of which `scipy.signal.freqz` 64.4% (pocketfft r2c 22.1%, the rest per-call overhead over 27,612 calls); `lwr_recursion` 7.3%.
opaque_share (8x1024 JSON): 0.083 (nonparametric), 0.239, 0.262, 0.166 (parametric None, 1, 20); lower bounds.

**Complexity.** All four are pair-dominated: nonparametric O(K n_fft P^2) CSM + P(P-1)/2 Wilson factorisations; parametric O(P^2 (T p + p^2 + n_freqs log n_freqs)) with p = 2, 21 or the BIC order (measured P exponents 2.03-2.08; T exponents 0.00-0.31, the autocovariance being the only T-dependent part).

**Opportunities.**
- nonparametric: `spectral_directed-2` 3-10x (batched closed-form 2x2 Wilson over all unordered pairs replaces the library pair loop at `spectral_connectivity/connectivity.py:722-743`; Wilson share 95.9%; unverified-within-tolerance). The batch has to keep two failure paths. Per pair: the loop catches `np.linalg.LinAlgError` (e.g. a singular solve) and writes NaN for that pair (`:742-743`; a closed-form inverse does not raise, so singular pairs must be detected and set to NaN explicitly). Whole SPI: a failed Cholesky initial condition raises AttributeError at `minimum_phase_decomposition.py:48` under numpy 2.5.4 (Error path above). That is neither the LinAlgError the loop catches nor the ValueError the pyspi wrapper catches (`pyspi/statistics/spectral.py:438`), so the Calculator NaNs the whole table and no RNG is drawn; the batch has to keep that or record the change. The random start from the global numpy RNG (`minimum_phase_decomposition.py:54-63`), which in a batched call would apply to every pair, is reachable only under a numpy that still has the deprecated `numpy.linalg.linalg` shim (numpy < 2.4 as checked: 2.2.6 and 2.3.3 resolve `xp.linalg.linalg.LinAlgError` with a DeprecationWarning, 2.4.6 and 2.5.4 raise AttributeError) or a library release whose handler catches `xp.linalg.LinAlgError`; that rng-stream risk is conditional on such a numpy or library version and does not apply in this environment (`pyproject.toml` pins numpy>=2.5.4,<2.6).
- order 20: `sgc-1` 2-3x (vectorise the LWR recursion over pairs or compile it; bound 2.9x; unverified-within-tolerance), `sgc-2` ~1.3x (batched rfft and 2x2 GC algebra; unverified-bit-identical, same pocketfft transform per row).
- order None: `sgc-3` ~1.5x (incremental BIC search: one recursion grown order by order and the autocovariance once; unverified-bit-identical), `sgc-1` ~1.7x, `sgc-2` ~1.4x.
- order 1: `sgc-2` 3-6x (bound 6.2x).
All parametric avenues are complexity/profile-share estimates; no micro-benchmark was written for this family.

**Refuted seeds.** Shared CSM for the nonparametric unit (multivariate Multitaper + CSM are 2.4% of the call; the pair loop dominates). Autocovariance once for order 20 (fixed order: computed once per pair already, 8.7%) and for order 1 (`autocov_vector` 3.9% at 118x1024, 4.7% at 8x1024: under the 10% threshold); it survives only inside `sgc-3` for the BIC search. numba LWR for order 1 (lwr_recursion 7.3%).
