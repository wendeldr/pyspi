# spectral_directed family: define, trace and check

Units: `spectral.dtf`, `spectral.dcoh`, `spectral.pdcoh`, `spectral.gpdcoh`, `spectral.ddtf` (six identifiers each: `{mean|max}` × bands `fmin-0_fmax-0-5`, `fmin-0_fmax-0-25`, `fmin-0-25_fmax-0-5`), and the infrastructure unit `spectral.wilson_factorisation`.

Classes: `pyspi/statistics/spectral.py:238-285`. All five inherit `NonparametricSpectralBivariate` (`spectral.py:105-134`) and `Directed` (`pyspi/base.py:70-94`). The config entries are at `pyspi/config.yaml:908-1067`. The delegate is spectral_connectivity 1.1.2 (`connectivity.py`, `transforms.py`, `minimum_phase_decomposition.py`), running on numpy 2.5.4 and scipy 1.17.1.

The probe scripts are in the session scratchpad under `spectral_directed/` (`p1_classes.py` to `p10_dc_gpdc.py`). Review probes are under `fable_spectral_review/` (`pD_kernel_directed.py`, `pF_wilson.py`, `pG_sigma_eeg.py`), and the probes added for the review fixes are under `spectral_directed_fix/` (`pR_lagshare.py`, `pS_ddtf_factor.py`, `pT_ddtf_exact.py`). The regression goldens are snapshots of current output. Nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`, for example `spectral_directed/p1_classes.py` or `fable_spectral_review/pD_kernel_directed.py`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Supplement line numbers are the ones in this checkout of `supplement.txt`. They are 12 to 16 lines lower than the pack's numbers: §1.5 is at line 570, §1.5.9 at 752, §1.5.10 at 774 and §1.5.11 at 789. Page numbers come from the page-number lines in the file.

## Shared pipeline (all five units)

1. **Preprocessing.** `pyspi/data.py:184-195`: optional detrend, then `zscore(ddof=1)`. The default is normalise=True and detrend=False.
2. **Pair loop.** `pyspi/base.py:84-94`: `A[i, j] = bivariate(data, i=i, j=j)` for all P(P−1) ordered pairs. `pyspi/calculator.py:325-339` (HEAD a42ebc1): `S = self._spis[spi].multivariate(self.dataset)` at :329; if any exception occurs, `except Exception` at :336-337 sets the whole SPI matrix to NaN.
3. **Per pair, per measure.** `spectral.py:106-126` builds the 2-channel array `[x_i, x_j]`, `sc.Multitaper(z, sampling_frequency=1)` and `Connectivity.from_multitaper`, then calls `getattr(conn, measure)()`. The result is cached under the key `(measure, i, j)`. The six band/statistic identifiers of a unit therefore share one computation per ordered pair. Different units share nothing.
4. **Multitaper settings** (`transforms.py:74-100, 155-185, 238-260`): NW = 3, 5 DPSS tapers, constant detrend, one window covering the whole series, and n_fft = next_fast_len(T), which equals T for the manifest sizes. The cross-spectral matrix is S[a,b] = E[X_a X_b*], averaged over the tapers (`connectivity.py:195-220`).
5. **Wilson factorisation** (`minimum_phase_decomposition.py:161-222`) gives S = Ψ Ψᴴ. Then H = Ψ A0⁻¹ and Σ = A0 A0ᵀ, with A0 = ifft(Ψ)[lag 0] (`connectivity.py:1116-1170`). H is restricted to the first (n_fft+1)//2 bins (`connectivity.py:73-89, 263-266`).
6. **Band summary** (`spectral.py:128-134`): `statfn(L[0, freq_id, 0, 1])`, where `freq_id` selects fmin < f < fmax with strict inequalities. The multivariate spectral path at `spectral.py:88` uses inclusive bounds instead.

**Direction convention.** H[a,b] maps innovation b into signal a, so the library's [a,b] entry is the flow b → a. This holds for every measure. pyspi reads [0,1] on `[x_i, x_j]`, so **A[i,j] is the estimated flow x_j → x_i: rows are targets and columns are sources.** That is the opposite of the glossary (`glossary-of-terms.md:27`: "rows ... reflect the sources") and of the supplement's "from x → y" reading.

The evidence:
- Kernel probe `p2_kernel.py` on the exact var2_bivariate spectrum: x0 → x1 appears only at L[1,0]. The review rerun on an exact toy VAR(1) with x0 → x1 gives L[1,0] = 0.1608 and L[0,1] = 0.
- Pipeline probe `p3_pipeline.py` at T=4096: var2_bivariate (x0 → x1) gives dtf `[[nan, 0.1281], [0.2617, nan]]`.
- logistic_maps (y = process 1 drives x = process 0, `audit/checks/make_data.py:207` (beta_xy = 0.1, beta_yx = 0) and `:309-310` (update equations)) gives dtf `[[nan, 0.9996], [0.0001, nan]]`, which agrees in sign. But the Wilson factorisation does not converge on that dataset at T ≥ 1024 (60 iterations, residual 0.32 to 0.38, `p7_wilson.py`), so it does not count as confirming evidence.

The direction convention is still a candidate until the check stage confirms it.

**Band rule (all units).** The documentation describes 125 bins from f0 = 4/T and closed bands (supplement §1.5, p.12-13). The code uses the k/T grid with open bands. DC is always dropped, and the f = 0.25 bin falls in neither half-band. At T=64 the full, low and high bands get 31, 15 and 15 bins (`p6_misc.py`).

**The 2×2 identities explain several equalities.** In 2×2, H = adj(Ā)/det Ā. This gives:
- Off-diagonal DTF = off-diagonal |PDC|². This is already documented in tests/README.md ("dtf_* == pdcoh_*"). `p10_dc_gpdc.py` reproduces it to 2.2e-16 and the pipeline to 1.1e-16.
- Off-diagonal canonical squared DC = off-diagonal squared gPDC (2.2e-16).

## spectral.dtf

**Documented.** Supplement §1.5.9, p.16: DTF "quantifies the inflow from x → y (according to the transfer matrix) normalized by the total inflow from all other signals into y (the row-wise sum of the spectral transfer matrix)". The GitBook text is the same (`spectral.md:107`).

**Canonical.** Kamiński & Blinowska 1991, Biol Cybern 65:203: γ²_ij(f) = |H_ij|² / Σ_m |H_im|², the flow j → i, including indirect flow.

**Implemented.** `connectivity.py:758-778` computes `|H / sqrt(Σ_cols |H|²)|²`. On exact VAR spectra this equals the canonical squared DTF to 1e-15 (`p2_kernel.py`).

**Deviations.**
- The direction is reversed relative to the documentation.
- Each pair is fitted as its own 2-channel model.
- The output is identical to pdcoh.
- There is a T-independent positive bias in the null direction, about 0.11 for T = 64 to 16384 (`p5_bias_T.py`). This is a candidate attributed to the fixed 5 tapers.
- DTF does not normalise by innovation variance. For a delayed copy with 100:1 innovation variances, the estimate gives 0.83 for the null direction and 0.17 for the true one, against exact values of 0 and 0.5 (`p4_delay.py`). That inverts the estimated direction. On delayed_noise the pyspi matrix therefore looks like "row = source" by accident.

**Proposed checks.**
- Exact-spectrum kernel test against the analytic DTF.
- A from-scratch numpy multitaper and Wilson pipeline as an independent reference.
- A direction check on var2_bivariate, logistic_maps and var2_chain against the true model and a from-scratch OLS VAR. logistic_maps is decisive only where the Wilson factorisation converges.
- A statistical bias study over T.
- The delayed_noise failure-mode check.
- An equality check of dtf against pdcoh.
- An exact count of band bins.

## spectral.dcoh

**Documented.** Supplement §1.5.10, p.16: "obtained from the inflow of x → y using the spectral transfer matrix ..., normalized by their noise covariance". The GitBook text is the same (`spectral.md:82`).

**Canonical.** Baccalá et al. 1998, Applied Signal Processing 5:40: γ_ij = σ_j H_ij / sqrt(Σ_k σ_k² |H_ik|²). The squared form lies in [0,1] and each row sums to 1.

**Implemented.** `connectivity.py:800-805` computes `sqrt(nv)·|H|² / sqrt(Σ_k nv·|H_ik|²)`. `_get_noise_variance` (`:1236-1238`) broadcasts diag(Σ) along the row (target) axis, so σ_i cancels. The library value is **|H_ij|² / sqrt(Σ_k |H_ik|²) = |H_ij|·sqrt(DTF_ij)**. That matches `p2_kernel.py` to 1.8e-15 and differs from the canonical DC by up to 0.61.

**Deviations (candidates).**
- The formula is wrong. It is neither the squared nor the modulus DC, and it does no noise normalisation.
- It is unbounded. Observed values reach 1.606 on the exact var2 spectrum, 94.46 on logistic_maps and about 2.9 on delayed_noise.
- It is not scale-invariant at the kernel level.
- Because the canonical bivariate DC equals gPDC, the quantity the documentation describes is in practice what gpdcoh returns.
- I could not check upstream: fetching later spectral_connectivity source failed, so it is unknown whether newer releases changed this formula.

**Proposed checks.**
- Kernel test against both the canonical formula and the hypothesised library formula.
- From-scratch estimator.
- Direction check.
- Range check on logistic_maps, delayed_noise and iid_gaussian.
- Comparison with gpdcoh.

## spectral.pdcoh

**Documented.** Supplement §1.5.11, p.17: "the inflow ... normalized by the total outflow from all other signals into y (the column-wise sum ...)".

**Canonical.** Baccalá & Sameshima 2001, Biol Cybern 84:463: π_ij = Ā_ij / sqrt(Σ_k |Ā_kj|²), a modulus, computed from the full multivariate VAR. It is zero exactly when there is no direct coupling.

**Implemented.** `connectivity.py:807-843` computes `|Ā / sqrt(Σ_rows |Ā|²)|²`, with Ā = inv(H). This matches the canonical |π|² to 9e-16.

**Deviations.**
- The output is squared, whereas the canonical PDC is a modulus.
- The defining property, partialisation, is lost because each pair is a 2-channel model. On var2_chain (no direct 0→2 link), pyspi gives A[2,0] = 0.1907. The 3-channel PDC is exactly 0 (`p2_kernel.py`).
- The output is identical to dtf (already documented).
- The direction is reversed relative to the documentation.
- The same estimator bias and inversion occur as for dtf.

**Proposed checks.**
- Exact kernel test.
- Multivariate OLS-VAR reference on var2_chain.
- Direction check.
- From-scratch estimator.
- Null bias over T.
- Equality with dtf.

## spectral.gpdcoh

**Documented.** Supplement §1.5.11, p.17: gPDC "scales the relative strength of inflow from x → y by their noise covariance".

**Canonical.** Baccalá, Sameshima & Takahashi 2007, DSP 2007, pp. 163-166: π^g_ij = (1/σ_i) Ā_ij / sqrt(Σ_k |Ā_kj|²/σ_k²).

**Implemented.** `connectivity.py:845-874` computes the squared form. It matches the canonical formula to 9e-16.

**Deviations.**
- The output is squared rather than a modulus.
- Pairs are fitted bivariately, so the measure is not partial: var2_chain gives A[2,0] = 0.1847.
- There is a bias floor of about 0.11.
- The direction is reversed relative to the documentation.

gpdcoh is the only unit in the family that keeps the designed direction under unequal innovation variances (`p4_delay.py`: 0.112 vs 0.881) and is scale-invariant (`p6_misc.py`).

**Proposed checks.**
- Exact kernel test.
- Direction check on var2_bivariate, logistic_maps and delayed_noise.
- From-scratch estimator.
- Scale invariance on scaled_offset vs bivariate_gaussian.
- Bias study.

## spectral.ddtf

**Documented.** Supplement §1.5.9, p.16: ddtf "extends DTF by conditioning out the influence of other signals (using the partialized cross-spectrum)".

**Canonical.** Korzeniewska et al. 2003, J Neurosci Methods 125:195, defines the dDTF as χ²_ij = F²_ij · C²_ij. Here F² is the full-frequency DTF and C the partial coherence. I have not re-checked the paper's equation numbers.

**Implemented.** `connectivity.py:876-900` computes `|H / sqrt(Σ_f Σ_k |H|²)| · sqrt(|PDC|²)`, which is |F|·|PDC|. That matches `p2_kernel.py` to 6e-17 and differs from F²C² by up to 0.048.

**Deviations (candidates).**
- PDC is used instead of partial coherence, despite the library docstring.
- The value is a product of moduli instead of a product of squares.
- With bivariate fitting nothing is conditioned out, since |PDC| = sqrt(DTF) in 2×2.
- The frequency normalisation covers the whole grid, not the band. Values therefore scale like T^(-1/2): 0.084, 0.044, 0.022, 0.011 and 0.005 for T = 64 to 16384. DEPENDENCY_COMPATIBILITY_REVIEW.md already notes that ddtf depends on the whole frequency grid.

**Proposed checks.**
- Exact kernel test against both forms.
- Multivariate reference on var2_chain.
- Direction check.
- Check the √(T/2) scaling law. On the exact var2_bivariate spectrum the band-mean slope is −0.50 within 0.01 (`pT_ddtf_exact.py`). In the pipeline the slope is only approximately −0.5, because the estimated |PDC| and H also move with T and with the realisation: var2_bivariate gives −0.62 from T=256 to 1024 and −0.52 from 1024 to 4096. The pipeline tolerance has to allow for that.
- From-scratch estimator.

## spectral.wilson_factorisation (infrastructure)

**Documented.** Supplement §1.5, p.12: "All directed measures ... are nonparametrically estimated through spectral matrix factorization [57, 58]".

**Canonical.** Wilson 1972, SIAM J Appl Math 23:420, and Dhamala et al. 2008, NeuroImage 41:354.

**Implemented.** The iteration Ψ ← Ψ[Ψ⁻¹SΨ⁻ᴴ + I]₊ uses Dhamala's plus operator: the zero lag is halved and made upper-triangular, and lags from n/2 upward are zeroed. It starts from a Cholesky initial value and stops when the sup-norm change in Ψ falls below 1e-8, or after 60 iterations. On exact VAR spectra it recovers H to about 4e-15. Σ is recovered only to 3.5e-9 to 4.1e-9 at the shipped tolerance of 1e-8, and to 4.3e-13 at a tolerance of 1e-12. The Σ accuracy is therefore set by the stopping tolerance. An earlier version of this note said Σ was recovered to 4e-15; that was read off 5-decimal printing and is withdrawn.

**Candidates.**
1. **NumPy 2.4+ breaks the Cholesky fallback (as checked).** `minimum_phase_decomposition.py:48` catches `xp.linalg.linalg.LinAlgError`, and `np.linalg.linalg` does not exist in numpy 2.5.4 (nor in 2.4.6; 2.2.6 and 2.3.3 still resolve it with a DeprecationWarning). A singular pair therefore raises AttributeError. One example is shared_envelope, where x2 is bit-identical to x0. Calculator then turns the whole SPI matrix to NaN (`p9_degenerate.py`). The `_spectral_compat` shim patches only `connectivity.py`.
2. **The factorisation is recomputed on every property access.** Per call it runs 2, 3, 2, 3 and 4 times for dtf, dcoh, pdcoh, gpdcoh and ddtf. It is not shared between units or between the orientations (i,j) and (j,i). This affects speed only, not values (pack hypothesis 3 confirmed).
3. **No zero padding.** Lags from n/2 upward cannot be represented. The relative residual is 2e-2 at n_fft=T and 9e-5 at n_fft=2T. The lag-domain residual peaks at lag n/2 but is spread around it: on var2_bivariate at T=1024, lag 512 alone carries 5% of it and lags within ±128 of 512 carry 66% (`pR_lagshare.py`). The effect on DTF band means is about 1e-3.
4. **Non-convergence passes silently.** On logistic_maps at T ≥ 1024, 60 iterations are not enough, and only a log warning is emitted. On narrowband data the step criterion reports convergence with a residual of 0.6.
5. **Fixed 5 tapers.** The null-direction bias floor of about 0.11 does not shrink as T grows.
6. **Channel-order dependence.** Results change by up to 5e-3 per bin when the channel order is swapped.

**Proposed checks.**
- Exact-spectrum recovery: H within 1e-12, Σ within 1e-8 at the shipped tolerance.
- From-scratch Wilson with a residual comparison at n_fft = T and 2T.
- Convergence and residual flags on logistic_maps, narrowband and var2.
- A singular-pair behaviour check on shared_envelope.
- An exact count of Wilson calls.

## Open questions

- The direction convention (row = target) contradicts the glossary for all five units. Should pyspi transpose these outputs, or document the convention? The check stage should first confirm it on var2_bivariate and var2_chain. logistic_maps counts only if the Wilson factorisation converges on it.
- Did later spectral_connectivity releases fix `directed_coherence` (the cancelling row noise variance and the mixed powers)? Fetching the upstream source failed, so this is unverified.
- Korzeniewska et al. 2003 (dDTF as the product of squares) and Baccalá et al. 1998 (DC normalisation) are cited from memory. I have not re-checked their equation numbers against the papers.
- Is the ~0.11 null bias best fixed by more tapers (NW scaled with T) or by subtracting a null (surrogate) baseline? Either way the values would change, which needs a decision on new identifiers.
- For P > 2, should the units be computed multivariately (conditioned on all channels)? That is what makes PDC, gPDC and dDTF meaningful. The code comment at `spectral.py:232-235` says they were switched to bivariate because of solver segfaults on large VAR processes. That reason has not been re-tested.
- The random-start fallback (`minimum_phase_decomposition.py:54-63`) uses the global numpy RNG. It is reachable only under a numpy that still has the deprecated `numpy.linalg.linalg` shim (numpy < 2.4 as checked: 2.2.6 and 2.3.3 resolve `xp.linalg.linalg.LinAlgError` with a DeprecationWarning, 2.4.6 and 2.5.4 raise AttributeError), or under a library release whose handler catches `xp.linalg.LinAlgError` (v2.0.1 does). If such an environment is ever used, results on singular pairs would not be reproducible.
- On narrowband or line-spectrum inputs (narrowband_f0-*), the factorisation residual of about 0.6 suggests that the directed measures are not meaningful there. Should pyspi flag such inputs?

## Checks

Files: `audit/checks/manifest/spectral_directed.json` (415 runs), `audit/checks/refs/spectral_directed.py` (comparison venv, no spectral_connectivity), `audit/checks/probes/spectral_directed_probes.py` (project venv). Results: `audit/checks/out/results/spectral_directed.json`, 352 pass, 63 fail, 0 without reference. Every failure is explained below; none is a script error. The regression goldens were not used.

**References.**
- *Independent chain.* `_kernels_spectral.multitaper_fft` + `cross_spectral_matrix` + `wilson_factorise` + the measure formulas, with z-scoring and detrending re-implemented. The library formulas are transcribed literally (`lib_measure`), and the canonical forms are computed beside them (`canon_measure`).
- *Analytic.* The VAR coefficients from `make_data` give A(f) = I − Σ A_k e^{−2πifk}, H = A⁻¹ and S = H Σ Hᴴ. For the pipeline path these are scaled to unit marginal variance. Population values of pyspi's *bivariate* quantity come from Wilson factorisation (tolerance 1e-13) of the exact marginal 2-channel spectrum.
- *Statistical.* A Monte Carlo of the independent estimator on fresh draws of the generating model: R = 400/200/100 at T = 256/1024/4096, cached in `out/exchange/spectral_directed/mc_*.npz`. The tolerance is se = MC SD × z_Bonferroni(m)/2 over the m off-diagonal entries, so each run is a familywise 5% test.

### Shared estimator: pyspi = the method, exactly
- **Replication.** For all five units the independent chain reproduces every config identifier: all six band/statistic variants on var2_bivariate and var2_chain at T1024, plus iid (kernel), T64, pipeline-detrend, gauss_ar1_pair, delayed_noise, logistic_maps_chaotic and scaled_offset (kernel). The rep runs pass 21/21 per unit, max_abs ≤ 9.8e-14. Per-bin values pass `indep` 11/11 per unit, ≤ 9.7e-12 (dcoh gauss_ar1_pair T4096; ≤ 2.7e-12 for the other units). The runs cover six ordered pairs at T1024/T256 plus var2_bivariate T256 and T4096, gauss_ar1_pair T4096 and the iid pair (kernel) at T1024 and T4096, e.g. `spectral.dtf|var2_bivariate|4096|pipeline|indep~p01` 1.6e-12 and `spectral.dtf|var2_bivariate|256|pipeline|indep~p01` 8.4e-15. The T256/T4096 runs were added after review, because the dtf proposed check named T in {256, 1024, 4096} and the first pass ran per-bin independent checks only at T1024 (and iid at T256).
- **Wilson factors.** On the library's exported CSM the library factors equal the independent Wilson factors: H to 4.3e-13 and Σ to 3.3e-16 (`spectral.wilson_factorisation|*|H~*`, `|Sigma~*`, 7/7 each). The CSM equals the independent multitaper to 2.1e-10 (15/15). Iteration counts, step convergence and residuals are identical (conv 14/14).
- **Exact spectra.** Library H matches the analytic H to 3.8e-15. Library Σ matches to 3.5e-9 at the shipped tolerance 1e-8 and to 4.3e-13 at 1e-12 (`exactH`, `exactSigma`, `exactSigma-tol12`, 6/6). This confirms the trace correction that the Σ accuracy is set by the stopping tolerance.
- **Band rule.** The strict rule fmin < f < fmax reproduces pyspi. The inclusive rule differs by up to 5.1e-3 (dtf) and 3.2e-2 (dcoh) (rep extras `max_abs_pyspi_vs_inclusive_rule`). Bin counts are 31/32 for the full band and 15/17 for the low band at T64, and 511/512 and 255/257 at T1024. This agrees with 2b-A `spectral.band_selection|gauss_ar1_pair|1024|pipeline|bvagg` and `|delayed_noise|256|pipeline|fftcount`.
- **Direction convention (confirmed).** A[i,j] is the flow x_j → x_i, so rows are targets.
  - On var2_bivariate (x0 → x1), A[1,0] − A[0,1] is +0.126/+0.134 (dtf, pdcoh), +0.198/+0.196 (dcoh), +0.199/+0.189 (gpdcoh) and +0.006/+0.003 (ddtf) at T1024/T4096.
  - var2_chain and gauss_ar1_pair follow the population sign for every unit (dir runs pass).
  - On the exact spectrum the library's L[0,1] is 0 at every bin (exactbin).
  - This is opposite to the glossary's "rows reflect the sources".
- **Inconsistent estimator (confirmed).** The truth runs compare pyspi with the population value at T = 256/1024/4096, using the multitaper SD as tolerance. All 40 truth runs fail, and |z| grows with T (dtf var2_bivariate: 4.4, 12.2, 22.2). The Monte Carlo shows why:
  - The estimator's mean does not move with T: DTF[1,0] is 0.263/0.262/0.263 against a population value of 0.207, and the null DTF[0,1] is 0.131/0.132/0.132 against 0.
  - The SD falls from 0.028 to 0.007.
  - On iid data every unit has a null floor near 0.11 (ddtf 0.0090/0.0046/0.0023, which follows its own T^−1/2 scale).
  - pyspi itself is a typical draw: mc runs pass 34/35. The exception is ddtf on var2_chain: raw |z| 2.82 against the Bonferroni bound 2.64, yet 1% of the 200 draws are at least as extreme, which is above 0.05/6. The rep run shows that pyspi equals the estimator on that draw to 3.6e-15, so this is a tail exceedance, not a discrepancy.
  - With 5 tapers and one window the per-bin CSM has fixed degrees of freedom. The bias of the nonlinear functionals of the Wilson factor therefore never vanishes.
- **Direction inversion on delayed_noise (confirmed, estimator).** delayed_noise has innovation variances of 1 against 0.01.
  - Population DTF(0 → k) is 0.498 and DTF(k → 0) is 0.
  - The estimator's mean is 0.17-0.20 against 0.79-0.82, so the direction is inverted (`spectral.dtf|delayed_noise|1024|pipeline|mc:mean-full`).
  - pyspi follows the estimator: dir fails for dtf, pdcoh, dcoh and ddtf.
  - Only gpdcoh keeps the population direction (estimator 0.88 vs 0.11; dir passes).
  - logistic_maps_chaotic (x → y) is inverted in the same four units. The factorisation there has a residual of 0.40/0.37 (conv), so those runs are reported but not used as evidence.

### spectral.dtf (variant, 7)
- **Formula.** The library computes the canonical squared DTF: 1.1e-15 on the exact spectra and 7.8e-16 on its own factors.
- **Deviation.** The deviation is in the setting. Each pair is fitted bivariately: on var2_chain the full-model DTF(0 → 2) is 0.0682, the bivariate marginal population value is 0.0887, and pyspi gives 0.2117 (chain run). On top of that come the T-independent bias, the inversion under unequal innovation variances, row = target and open band edges.
- **Score.** Rubric anchor 7: value-changing bivariate normalisation and an expectation over 5 tapers.
- **Candidates.** All six are confirmed. Scale: the band mean changes by 0.871 under channel scaling (population 0.750), which is what canonical DTF does.

### spectral.dcoh (wrong, 2)
- **What the library computes.** The library value is |H_ij|²/sqrt(Σ_k|H_ik|²) = |H_ij|·sqrt(DTF_ij), reproduced on the library's own factors to ≤ 8.9e-16.
- **Canonical form, checked against the paper.** Baccalá et al. 1998 (scanned asp.pdf linked from the GitBook, p.43) define γ_ij(f) = σ_jj H_ij(f)/sqrt(S_i(f)) in Eq. (21), with S_i(f) = Σ_j σ_jj²|H_ij(f)|² in Eq. (22). Eqs. (23) and (24) give 0 ≤ |γ_ij|² ≤ 1 and Σ_j|γ_ij|² = 1. Remark 6 says DTF is the case σ_jj = 1. This is what `_kernels_spectral.directed_coherence` implements.
- **Size of the difference.**
  - Per bin the library differs from the squared DC by 0.64 (var2_bivariate), 1.61 (gauss_ar1_pair) and 6.96 (delayed_noise), and by 0.177 on the exact var2 spectrum.
  - Off-diagonal band maxima reach 1.75 (var2_bivariate T4096), 9.80 (delayed_noise), 10.9 (logistic_maps_chaotic) and 2833 (logistic_maps); range passes 1/5.
  - Under channel scaling the value changes by 16.9 (population 5.18), whereas canonical DC is invariant (scale fails).
  - The formula alone can reverse the direction relative to canonical DC (added in Workflow 4b-2). On the exact spectrum of a bidirectional VAR(1) with A1 = [[0.2, b], [a, 0.2]] and unequal innovation SDs, the as-coded band-mean A[1,0] − A[0,1] is −0.040 against +0.937 for the canonical squared DC (a = 0.4, b = 0.45, SDs 1 and 0.1), and −0.083 against +0.786 (a = 0.4, b = 0.5, SDs 1 and 0.2). Scaling x1 by 10 moves the as-coded [1,0] from 0.165 to 4.13 while the canonical value stays 0.9392 (scratchpad `wf4b_low-spectral/probe_directed_population.py`, numpy only). These two cases are on the raw channel scale, which pyspi sees only with normalise=False: on unit-variance channels, as pyspi's default z-scoring gives, the same two systems have as-coded A[1,0] - A[0,1] = +0.612 and +0.531, the sign of the canonical DC. Reversals also occur after z-scoring: 158 of 4000 random stable 2-channel VAR(1) systems, e.g. A1 = [[-0.522, -0.708], [-0.545, 0.031]], SDs (0.119, 0.922): canonical -0.961 against z-scored as-coded +0.043 (scratchpad `wf4b2_low-spectral_recheck/zscored_dcoh.py` and `zscored_search.py`, numpy only). The z-scored reversals are what support rubric anchor 2 for pyspi's default input; the raw-scale cases above show the same formula effect with normalise=False.
  - On delayed_noise the pyspi asymmetry is also opposite to canonical DC, but the population of the as-coded quantity matches the designed direction there (`mc:mean-full` ref_truth [1,0] = 0.70, [0,1] ≈ 0). That reversal is the estimator failure shared with dtf, pdcoh and ddtf, not evidence about the formula.
- **Which one the library computes.** spectral_connectivity 1.1.2 computes the as-coded quantity, not the canonical DC. The latest release, v2.0.1 (2026-05-12, `connectivity.py:1298-1303`), is unchanged. Upstream master (commit 138f3f7f0f, 2026-10-06, unreleased 3.0) rewrote it to the canonical squared DC nv_j|H_ij|²/Σ_k nv_k|H_ik|² and made the output source-first.
- **Relation to gpdcoh.** For two channels the canonical squared DC equals the off-diagonal squared gPDC to ≤ 3.9e-16, so gpdcoh is the quantity dcoh is documented to be.
- **Bivariate keyword.** The supplement's dcoh keywords include "bivariate" (supplement.txt:775), so the pairwise 2-channel fit agrees with the documented keywords; the record now quotes them. The formula error decides the score, not the bivariate fit.
- **Trace correction.** The 1.606 quoted at the trace stage for the exact spectrum is a diagonal entry. The off-diagonal maximum there is 0.564. units_and_range and canonical_ref are corrected in the record.

### spectral.pdcoh (related-quantity, 3)
- **Formula.** The library computes the canonical squared PDC (1.1e-15).
- **The 2×2 identity.** In 2×2, H = adj(A)/det(A); on the library factors this holds to ≤ 1.1e-14. As a result the off-diagonal |PDC|² equals DTF to ≤ 4.4e-16, and pyspi's pdcoh equals its dtf for all six identifiers to 4.4e-16 (ident 7/7, same-lineage). The equality is already documented in tests/README.md; this check adds the mechanism.
- **Consequence.** The partialisation that defines PDC is gone. On var2_chain the conditioned PDC(0 → 2) is 0 (2.5e-32 per bin on the exact 3-channel spectrum). The bivariate population value is 0.0887 and pyspi gives 0.2117.
- **Squared form.** The squared form is monotone: the band mean is 0.2689 squared against 0.4846 for the modulus.
- **Score.** Rubric anchor 3, the label does not describe the computation. This is the same case as the rubric's own example "dDTF bivariate": a measure defined by conditioning, computed on a 2-channel model so that the conditioning is gone. pdcoh returns exactly the dtf matrix for every P and every identifier (ident 7/7, 4.4e-16), so nothing partial is left. The score was 4 (anchor 4, related but different quantity) before review; anchor 3 is preferred because it names this case, not because of any criterion outside the rubric. Anchor 7 (supplement variant with bivariate normalisation) was considered and rejected: the pdcoh keywords do say "bivariate" (supplement.txt:790), but the pairwise fit removes the partialisation that defines PDC rather than renormalising it, exactly as for the rubric's "dDTF bivariate" example, whose supplement section (supplement.txt:753) carries the same keyword. The record now quotes the keywords. new_finding rests on the loss of partialisation for P > 2 (the var2_chain chain run), not on the dtf equality, which is already documented. For P = 2 inputs the value is the canonical squared PDC of the 2-channel model, which in 2×2 is itself DTF. gpdcoh stays at 4 on the rubric's own terms: anchor 3 fits pdcoh because its label does not describe the computation (its output is exactly the dtf matrix for every P), while gpdcoh's output is the canonical squared gPDC of each 2-channel model, the quantity its label names for P = 2 and a related but different quantity from the conditioned gPDC for P > 2 (var2_chain 0.2031 against 0.0), which is anchor 4.

### spectral.gpdcoh (related-quantity, 4)
- **Formula.** The library computes the canonical squared gPDC (1.0e-15), and the 2×2 gPDC² equals the canonical DC² (3.9e-16).
- **Not partial.** The bivariate fit removes the partialisation: on the chain the conditioned value is 0.0 against 0.0845 bivariate, and pyspi gives 0.2031.
- **Score.** Rubric anchor 4 (related but different quantity): for P > 2 the pairwise fit computes the 2-channel gPDC instead of the conditioned one; for P = 2 the value is the canonical squared gPDC. Anchor 3 does not apply, because the output is the quantity the label names on the 2-channel model and is not another SPI's matrix, as pdcoh's is (see the pdcoh Score bullet).
- **What it does right.** gpdcoh is the only unit that keeps the population direction on delayed_noise. It is scale invariant up to the Wilson non-equivariance: the band mean changes by 3.1e-3 and the population value by 0.0.
- **Scale criterion revised.** The scale criterion was first set at 1e-6 and then revised to 1e-2. Reason: the CSM is exactly equivariant (4.7e-16), but the step-converged factor of a CSM that is not exactly factorisable is not. In a scratch probe the per-bin H changed by up to O(1) at isolated bins under rescaling. That is the same mechanism as the channel-order dependence. dtf changes by 0.871 and dcoh by 16.9, so 1e-2 still separates invariant formulas from non-invariant ones.

### spectral.ddtf (wrong, 3)
- **What the library computes.** The library returns |ffDTF|·|PDC|, with ffDTF normalised over all n_fft/2 non-negative bins. This matches to 4.2e-17 on its own factors and 6.9e-17 on the exact spectra.
- **Against Korzeniewska.** Korzeniewska et al. 2003 define ffDTF²·(partial coherence)². That form differs from the library by 0.042 per bin (var2_bivariate) and 0.0196 on the exact spectrum. Upstream master now returns `full_frequency_dtf * squared_partial_coherence`; v2.0.1 does not.
- **2×2 degeneracy.** In 2×2 the library value is |ffDTF_ij|·sqrt(DTF_ij) (1.4e-17), and the partial coherence equals the ordinary coherence (≤ 1.1e-15). On the chain the indirect 0 → 2 gets 0.0042 (population) and 0.0091 (pyspi), where the conditioned value is 0.
- **Length dependence.** On the exact spectrum, band mean × sqrt(n/2) is 0.21680, 0.21515, 0.21475 and 0.21466: a spread of 1.0% over n = 64-4096 and 0.23% over 256-4096, with slopes −0.506, −0.501 and −0.500 (ddtfscale).
  - Pipeline slopes (ddtfslope, values reproduced to 3e-16 by the independent chain):
    - var2_bivariate [1,0]: −0.44, −0.62, −0.52.
    - var2_bivariate off-diagonal mean: −0.50, −0.52, −0.53.
    - Report-only: null [0,1] −0.68, −0.24, −0.55, and iid off-diagonal mean −0.78, −0.43, −0.49.
  - This settles the remaining trace issue on proposed_checks[3]. The record's expected outcome already stated the 1.5% band, the [1,0] / off-diagonal-mean scope and iid as report-only, and the runs fall inside it.
  - **Which part is library-specific (revised after review).** The length dependence itself is not a library deviation. Korzeniewska's ffDTF is normalised over the whole analysed frequency range, so on an n_fft = T grid the canonical ffDTF² × (partial coherence)² also depends on T. On the same exact var2_bivariate spectra its band mean scales as about T^−1: slopes −1.007, −1.002 and −1.000, and mean × n/2 = 0.0733, 0.0726, 0.0725, 0.0724 (ddtfscale extras `canonical_K2003_*`). Renormalising over the low half-band only gives slopes −1.043, −1.011 and −1.003. Only two parts are specific to the library. The exponent −1/2 follows from using moduli instead of squares. The half-band identifiers are normalised by full-band energy, where a band-restricted analysis would normalise over the analysed band. The record's candidate resolution, deviation_from_canonical (d) and what_changed now separate the two.
- **Score.** Rubric anchor 3 (dDTF bivariate).
  - The supplement's dtf/ddtf keywords include "bivariate" (supplement.txt:753), so the pairwise fit itself is documented; the ddtf sentence still promises conditioning on other signals, which a 2-channel fit cannot give. The record now quotes the keywords.
  - Anchor 2 checked and not supported (added in Workflow 4b-2): in 3961 random stable 2-channel VAR(1) systems with unequal innovation variances, the band-mean sign of A − A.T from the library formula never differed from the canonical 2-channel dDTF (ffDTF² × coherence²) on the exact spectrum (scratchpad `wf4b_low-spectral/probe_ddtf_sign.py`, flips 0/3961).
  - Canonical form basis: the Korzeniewska et al. 2003 PDF was not opened. The form ffDTF² × (partial coherence)² was checked against the docstring and code of `direct_directed_transfer_function` in upstream master 138f3f7f0f, which cites the paper; `canonical_ref` says so, so the caveat reaches the CSV.

### spectral.wilson_factorisation (variant, 7)
- **Algorithm.** The algorithm matches the independent implementation to 4e-13 and the analytic factors to 4e-15.
- **Residuals without zero padding.** The residual max|ΨΨᴴ − S|/max|S| is 0.021/0.009 on var2_bivariate (T1024/T4096), 0.028 on delayed_noise and 0.016 on iid. Zero padding to 2T brings it to 9.1e-5, and DTF[1,0] moves from 0.2689 to 0.2682 (nfft2).
- **Unreliable inputs pass silently.** On logistic_maps the iteration stops at 60 without converging (residual 0.32/0.38). logistic_maps_chaotic (residual 0.40/0.37) and narrowband_f0-0.25 (0.61/0.59) converge by the step criterion. Only a log warning, and only in the logistic_maps case, marks any of these.
- **Order dependence.** Swapping the channels changes per-bin values by 5.3e-3 (dtf), 8.6e-3 (dcoh), 4.5e-3 (gpdcoh) and 3.0e-4 (ddtf) on var2_bivariate, and by up to 3.4e-2 (dcoh) on gauss_ar1_pair (kernel).
- **Call counts.** Wilson runs 2, 3, 2, 3 and 4 times per pair for dtf, dcoh, pdcoh, gpdcoh and ddtf: [12, 18, 12, 18, 24] at P = 3. The six dtf identifiers on one Data share the cache (12 runs), and one shared factorisation per pair gives bit-identical values.
- **NumPy-2 fallback (confirmed, with a refinement).** On shared_envelope (x2 = x0) the behaviour depends on rounding:
  - Where Cholesky fails (T256 pipeline, T1024 kernel; the independent kernel raises LinAlgError), mpd:48 `xp.linalg.linalg.LinAlgError` raises AttributeError and the whole SPI matrix is NaN.
  - Where it succeeds (T1024 pipeline, T4096 pipeline-detrend), the duplicated pair silently gets 0.528 (dtf), 0.507 (gpdcoh) or 0.522. The kernel gives the same values.
  - Neither branch gives the safe behaviour of NaN for the degenerate pair only.
  - spectral_connectivity v2.0.1 already catches `xp.linalg.LinAlgError`.

### Trace-stage fields corrected after review
- `calculator.py:292-303` (dtf, dcoh, pdcoh, gpdcoh, ddtf implementation_trace and the shared-pipeline step 2 above) and `calculator.py:301-303` (wilson_factorisation candidate 1) were stale; at those lines HEAD a42ebc1 has the `compute()` docstring. They became `calculator.py:325-339`, with the `multivariate` call at :329 and `except Exception` → `self._table[spi] = np.nan` at :336-337 (the same lines as at a1f20c7).
- spectral.ddtf deviation_from_canonical (d), implemented_as and new_finding_candidates[2]: the T dependence is now attributed to the canonical definition on a T-dependent grid. Only the T^−1/2 exponent and the full-band normalisation of the half-band identifiers are library-specific (see the ddtf section).
- spectral.pdcoh score: 4 → 3 (anchor 3); verdict stays related-quantity (see the pdcoh section).

## Open questions: answered

1. **Should pyspi transpose these outputs or document the convention?** The convention is now established: row = target, column = source. The evidence is the var2_bivariate, var2_chain and gauss_ar1_pair dir runs, which pass for every unit, plus exactbin. This contradicts the glossary. Both remedies are cheap: transpose in `NonparametricSpectralBivariate.bivariate` by reading [1,0], or document the convention. Note that spectral_connectivity 3.0 (master) changes its own output to source-first. Any future upgrade of the pinned dependency therefore flips the convention unless pyspi adapts. logistic_maps_chaotic cannot settle the question, because its factorisation residual is 0.40.
2. **Did later spectral_connectivity releases fix directed_coherence?** v2.0.1, the latest release, did not: its formula is identical. Master 138f3f7f0f (unreleased) did, for both directed_coherence (canonical squared DC) and ddtf (ffDTF² × squared partial coherence). Both were fetched with `gh api` into the scratchpad `spectral_directed_check/upstream/`.
3. **Equation numbers of the papers.**
   - Baccalá et al. 1998: checked. Eq. (21)-(24) and Remark 6 on p.43 of the scanned paper match `_kernels_spectral.directed_coherence`, which uses source variance σ_jj² weighting.
   - Korzeniewska et al. 2003: not retrieved (paywalled). The F²C² form rests on secondary sources, now including the upstream master docstring and code. Retrieving the paper would settle it.
4. **How to address the ~0.11 null bias.** It is confirmed as an inconsistency of the 5-taper single-window estimator, not finite-sample noise: the bias is constant over T = 256-4096 while the SD halves at each step. Two remedies would work. One is NW, and with it the number of tapers, scaled with T. The other is a surrogate baseline. Zero padding does not help: nfft2 changes the dtf band mean by only 7e-4. Either remedy changes values and needs new identifiers.
5. **Multivariate PDC/gPDC/dDTF for P > 2.** The chain runs quantify what is lost:
   - Conditioned PDC/gPDC/dDTF(0 → 2) is 0, while the bivariate population values are 0.0887, 0.0845 and 0.0042.
   - On the full 3-channel spectrum the library itself reproduces the conditioned reference values (exactbin var2_chain, ≤ 4e-32 at [2,0]).
   - The segfault claim at `spectral.py:232-235` was not re-tested: no P ≥ 10 multivariate run was attempted. A multivariate Wilson run on a P = 10-20 dataset would settle it.
6. **Flag non-converged or high-residual inputs?** The diagnostics are cheap and reproducible: residual 0.32-0.61 on logistic and narrowband inputs, against ≤ 0.03 on VAR data (conv runs). A threshold near 0.1 on max|ΨΨᴴ − S|/max|S| would separate them in this suite. Whether to return NaN or a flag is a design decision.
7. **Supplement line numbers.** The records cite this checkout's lines. No change was needed at the check stage.

The trace-stage global-RNG question (random-start fallback) applies under numpy < 2.4 as checked, or under a library release whose handler catches `xp.linalg.LinAlgError` (the trace stage wrote "NumPy < 2"; see Open questions above). It is moot for this environment (numpy 2.5.4) and was not tested.

## Optimisation

Sources: walls from `audit/profiles/summary.json` (fresh / family-warm; quiet machine; 118x1024 walls are single repeats), frame shares from cProfile. The official profiles at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`) are dominated by the P-independent DPSS cost for every multivariate unit, so for ranking at 118 channels this group added cProfile runs at 118x1024 (32x1024 for sgc nonparametric): `audit/profiles/micro/prof_<unit>__<shape>.pstats`, written by `audit/profiles/micro/prof_spectral.py`, indexed with walls, peak RSS and thread settings in `audit/profiles/micro/prof_spectral__index.json` (OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1; no numba or JIDT code on these paths). Those runs overlapped other Workflow 3 groups; the memory-bound multitaper units varied 2-7x in wall between runs (cohmag: 6.4 s on a first run, 0.93 s on the kept run, 2.42 s in summary.json), so only shares are quoted from them. Percentages are of the profiled `harness.compute_spi` frame. Micro-benchmarks (the group's two): `audit/profiles/micro/opp_spectral_csm-1.{py,json}` and `audit/profiles/micro/opp_spectral_csm-shared-csm.{py,json}`. A candidate that reproduces the current output "matches the baseline"; that says nothing about whether either is mathematically right. Nothing in `pyspi/` was changed.

| unit | 8x1024 fresh / fw (s) | per ordered pair (ms) | 118x1024 projected (s), mode | patient (600 windows, s) |
|---|---|---|---|---|
| spectral.dtf | 3.217 / 3.224 | 57.4 | 793.0 / 794.7, extrapolated-pairs-from-8 | 475,788 |
| spectral.dcoh | 4.386 / 4.374 | 78.3 | 1081.4 / 1078.3, extrapolated-pairs-from-8 | 648,820 |
| spectral.pdcoh | 3.242 / 3.240 | 57.9 | 799.2 / 798.8, extrapolated-pairs-from-8 | 479,515 |
| spectral.gpdcoh | 4.397 / 4.401 | 78.5 | 1084.0 / 1084.9, extrapolated-pairs-from-8 | 650,376 |
| spectral.ddtf | 5.568 / 5.557 | 99.4 | 1372.7 / 1370.1, extrapolated-pairs-from-8 | 823,629 |

All five are in `infeasible_above_8x1024.txt`; the 118 values scale the 8x1024 wall by ordered-pair count (56 -> 13,806) with no fixed cost, which matches the code (one independent 2-channel computation per ordered pair, nothing shared). Peak RSS is 337-392 MB at 8x1024. Not in the EEG selection; together the five would cost about 5,100 s per window at 118 channels. Within a unit the six band identifiers share the per-pair cache, so family-warm increments after the first are 0.0007-0.0012 s.

**Hotspots at 8x1024** (official profiles; the per-pair loop makes these shares hold at any P): `minimum_phase_decomposition.py:minimum_phase_decomposition` is 73.0% (dtf, 112 calls = 2 per ordered pair), 80.7% (dcoh, 3 per pair), 72.6% (pdcoh, 2), 80.1% (gpdcoh, 3) and 83.5% (ddtf, 4) of the profiled call; inside it `numpy/linalg/_linalg.py:solve` on 2x2 stacks is 48.8%, 55.3%, 49.1%, 53.6% and 56.5%, `_get_causal_signal` (ifft/fft) 8.7-10.2%, and its own self time (matmul, copies) 10.6% for dtf. The second hotspot is the per-pair Multitaper: `connectivity.py:from_multitaper` 25.4%, 17.8%, 25.2%, 18.0%, 14.5%, almost all of it `transforms.py:dpss_windows` (25.1%, 17.6%, 24.7%, 17.7%, 14.3%). About 28 Wilson iterations per factorisation (3100 `_get_linear_predictor` calls / 112 factorisations for dtf). opaque_share 0.078-0.084 (lower bound).

**spectral.wilson_factorisation** (infrastructure, from the dtf profile): `solve` 48.8%, factorisation self time 10.6%, `_get_causal_signal` 9.2% of the dtf call. 14 factorisations per ordered pair across the five units.

**Opportunities.**
- `spectral_directed-1` (cache-sharing, S): memoise (H, Sigma) per ordered pair and serve every property access and all five units. Unit level: dtf ~1.6x, pdcoh ~1.6x (1 of 2 removable: 36.5% / 36.3%), dcoh ~2x, gpdcoh ~2x (2 of 3: 53.8% / 53.4%), ddtf ~2.7x (3 of 4: 62.6%); for the five units together 14 factorisations per ordered pair become 1 (784 -> 56 in the five 8x1024 profiles): 14x on the factorisation work itself, but ~3.7x on the five units' combined time by Amdahl (Wilson 30.2 s of 38.2 s summed over the five profiled calls; the per-pair Multitaper and the measures remain), which is the figure recorded on spectral.wilson_factorisation (~4x; an earlier 2-14x mixed the two quantities). unverified-bit-identical (deterministic factorisation reused).
- `spectral_directed-2` (vectorise-batch, L): one batched Wilson over all pairs (pairs on the leading axis, so convergence stays per pair) with a closed-form 2x2 inverse instead of `np.linalg.solve` per 1024-matrix stack. Unit level 2-3x (dtf, pdcoh) to 2-6x (ddtf), bounded by the Wilson share; on the factorisation itself 3-10x. unverified-within-tolerance: the closed form rounds differently and the 1e-8 sup-norm stop rule can shift by an iteration (Sigma is recovered only to ~4e-9 at that tolerance, wilson_factorisation section above). Failure paths: pyspi's directed pair loop (`pyspi/base.py:85-94`) has no exception handler, so today a pair that raises fails the whole SPI in the Calculator (`pyspi/calculator.py:336-337`). A failed Cholesky initial condition is one such raise: the handler at `minimum_phase_decomposition.py:48` names `xp.linalg.linalg.LinAlgError`, which numpy 2.5.4 does not have, so the failure raises AttributeError, the whole SPI is NaN and no RNG is drawn (Checks above, "NumPy-2 fallback"). A batch has to keep that whole-SPI behaviour or record the change. The random start from the global numpy RNG (`:54-63`), which in a batched call would hit every pair, is reachable only under a numpy that still has the deprecated `numpy.linalg.linalg` shim (numpy < 2.4 as checked: 2.2.6 and 2.3.3 resolve `xp.linalg.linalg.LinAlgError` with a DeprecationWarning, 2.4.6 and 2.5.4 raise AttributeError) or a library release whose handler catches `xp.linalg.LinAlgError` (v2.0.1 does); that rng-stream risk is conditional on such a numpy or library version and does not apply in this environment (`pyproject.toml` pins numpy>=2.5.4,<2.6). The Open questions item above, which the trace stage worded "Under NumPy < 2", now uses this same condition.
- `spectral_csm-shared-csm`: 2x2 blocks sliced from the shared P-channel FFT/CSM instead of one Multitaper per ordered pair: ~1.2-1.3x (removes `from_multitaper`, 14.5-25.4%); unverified-bit-identical (multivariate and bivariate CSM entries are bit-identical, spectral_csm.md). The DPSS cache alone (`spectral_csm-dpss-cache`) gets almost the same share (14.3-25.1%).
Combined (shared FFT + one factorisation per pair + batched closed form) the per-pair cost falls by roughly an order of magnitude; the remaining ordered-pair factor of 2 is the next item.

**Refuted seed: one factorisation for both orientations.** The bivariate factorisations of (i, j) and (j, i) are not permutations of each other within tolerance: the check stage measured per-bin changes of 5.3e-3 (dtf), 8.6e-3 (dcoh), 4.5e-3 (gpdcoh) and 3.0e-4 (ddtf) on var2_bivariate when the channel order is swapped, up to 3.4e-2 (dcoh) on gauss_ar1_pair (Checks above, "Order dependence"). Deriving A[j, i] from the (i, j) factorisation would halve the pair count (2x) but is a semantic change, so it is recorded here and not listed as an opportunity.
