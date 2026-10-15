# spectral_phase_sync family: define, trace and check

Units: `spectral.plv`, `spectral.pli`, `spectral.wpli`, `spectral.dspli`, `spectral.dswpli`, `spectral.ppc`.
Each unit covers six identifiers: `{mean, max}` × the bands `fmin-0_fmax-0-5`, `fmin-0_fmax-0-25` and `fmin-0-25_fmax-0-5`. The EEG selection retains only the `mean` variants, adding ten EEG-band identifiers per unit (`fmin-0-000488_fmax-0-00195` and so on).
Classes are in `pyspi/statistics/spectral.py:167-229`. All six delegate to spectral-connectivity 1.1.2 (`Multitaper` + `Connectivity`) on numpy 2.5.4.
The probe scripts are in the session scratchpad under `spectral_phase_sync/probe1.py` to `probe7.py`; the post-review probes (`sat.py`, `eeg.py`, `grid_tie.py`, `tie2.py`, `tie3.py`) are under `spectral_phase_sync_fix/`. The regression goldens are snapshots of current output, and nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared pipeline (all six units)

**Documented.** Supplement §1.5, pp.12-13 (heading at supplement.txt:570, grid text at :582-599), with the same text on GitBook `spectral.md:16-20`:

> "Each measure is based on the cross- and power-spectral densities ... estimated via the multitaper technique ... We use 125 uniformly sampled bins across the entire frequency range, f ∈ [f0, fs/2], where fs/2 is the Nyquist frequency and f0 = 4/T ... fmin-0_fmax-0-5: f ∈ [0, fs/2]; fmin-0_fmax-0-25: f ∈ [0, fs/4]; fmin-0-25_fmax-0-5: f ∈ [fs/4, fs/2]."

The pack's line ranges for §1.5.6 to §1.5.8 are offset. The actual headings are at supplement.txt:686 (PLV), :701 (PPC) and :715 (PLI family), all on p.15.

**Trace.**
1. **Preprocessing.** `pyspi/data.py:184-195` applies optional detrend, then z-scoring with ddof=1.
2. **Calculator.** `pyspi/calculator.py:295-355` (`Calculator.compute`, cited at HEAD a1f20c7) calls `multivariate` at :329 and sets the diagonal to NaN at :332; any exception stores an all-NaN table at :336-337.
3. **No symmetrisation.** The class MRO (probe1) is `[cls, NonparametricSpectralMultivariate, NonparametricSpectral, Unsigned, Undirected, Directed, object]`. `Undirected.multivariate` is therefore never reached. This is already documented as REVIEW M1 (tests/README.md bug 5 before b2f6d7e).
4. **Cache.** `spectral.py:64-83` (`_get_cache`) calls `sc.Multitaper(z, sampling_frequency=fs)` with every other argument left at its default, then `Connectivity.from_multitaper(m)` and then `getattr(conn, measure)()`. The cache key is `(measure,)`, so all band and statistic variants share one spectrum.
5. **Band summary.** `spectral.py:85-102` keeps bins with `freq >= fmin & freq <= fmax` (inclusive at both ends) and applies `np.nanmean` or `np.nanmax` over them. It then fills the diagonal with NaN.
6. **spectral-connectivity front end.**
   - `transforms.py:74-100`: NW=3, `detrend_type='constant'`, a single window covering the whole series, and `is_low_bias=True`.
   - `spectral_connectivity/transforms.py:155-164`: K = 5 DPSS tapers. All five pass the eigenvalue > 0.9 filter at every probed T >= 10 (10, 16, 30, 64, 100, 250, 256, 512, 1000, 1024, 4096). At T = 7 and 8 (n_fft = 8) only 4 are kept and `n_observations = 4`, so every K = 5 statement below assumes T >= 10.
   - `transforms.py:181-190`: `n_fft = next_fast_len(T)`.
   - `spectral_connectivity/transforms.py:238-260, 335-354`: taper, then FFT.
   - `connectivity.py:45-53`: `EXPECTATION['trials_tapers']` averages over 1 trial × 5 tapers, so `n_observations = 5` (`:280-287`).
   - `connectivity.py:195-208, 1187-1190`: S_ij,k = X_i,k · conj(X_j,k).
   - `connectivity.py:73-89`: keeps bins 0 … n_fft/2 − 1, so DC is kept and Nyquist is dropped.

**Facts established by probe.**
- **Grid.** The grid has 32, 128, 512 and 2048 bins at T = 64, 256, 1024 and 4096, and starts at f = 0. The f = 0.25 bin falls in both the low and the high band whenever 0.25 is a grid point, that is when n_fft = `scipy.fftpack.next_fast_len(T)` is a multiple of 4. That holds for every manifest size and for the 512- and 1024-sample EEG windows, but not in general: at T = 30 and 250 there is no 0.25 bin, and the inclusive low/high counts are 8/7 and 63/62 (`grid_tie.py`).
- **Pairwise separability.** Every measure is pairwise-separable: P-channel and 2-channel results are identical.
- **Scale invariance.** Every measure is invariant to `normalise` (scaled_offset, ≤ 1.1e-16).
- **Trends.** Every measure changes with `detrend` on trending data.
- **Exact reproduction.** A from-scratch reproduction of each measure from the dependency's own Fourier coefficients matches to ≤ 2.2e-15. A fully independent multitaper (`scipy.signal.windows.dpss` + `numpy.fft`) matches PLV and wPLI per bin to ≤ 1e-11 at T ≤ 1024 (probe6). The check-stage reference is feasible.
- **Sign ties in a reference.** The library forms S with a matmul (`connectivity.py:1187-1190`) and gets Im S exactly 0 on the diagonal and for bit-identical channels. An elementwise reference (`X[:, None] * conj(X[None, :])`) leaves rounding noise only at those two places: measured in the check stage, 1.1e-16 to 3.5e-15 on the diagonal and 2.6e-16 to 1.3e-15 for the bit-identical pair shared_envelope (0,2) at T = 64/256/1024. At DC it leaves none: the elementwise Im S is exactly 0.0 on every off-diagonal entry of every checked dataset, because the DC coefficient of real input is real (keys `tie_dc_offdiag_max_abs_im`, `tie_diag_max_abs_im`, `tie_identical_pair02_max_abs_im_elementwise` of the `perbin` runs). Without a tie rule the sign-based and ratio-based measures then differ by O(1) on the identical pair: PLI by 1.0, dsPLI by 1.25, wPLI by up to 0.993 (T = 256) or 1.0 (T = 1024), and dswPLI is finite where the library gives NaN (keys `no_tie_rule_*`). The check-stage reference sets |Im S_ij,k| <= 1e-12 · max_k |S_ij,k| to 0 before applying each formula and masks the diagonal; the rule zeroed exactly the identical-pair entries (5 tapers × 2 orientations × (n_bins − 1) non-DC bins, 1270 at T = 256) and nothing else.
- **DC bin.** The imaginary part of S at DC is exactly 0 for real input.

**Shared deviations.**
- **Grid.** The documented grid (125 bins, f0 = 4/T) is not what runs (candidate). DC inclusion and Nyquist exclusion are already documented (REVIEW M4, DEPENDENCY_COMPATIBILITY_REVIEW.md); the 125-bin and f0 = 4/T mismatch is not.
- **Expectation domain.** The expectation is over the **5 tapers of a single window**, not trials or time. Every per-bin estimate therefore rests on 5 observations at any T, which is the pack hypothesis, confirmed. Null levels do not shrink with T.
- **max variants.** The `max` variants take a maximum over n_fft/2 of these small-sample estimates, and under independence they approach or reach 1 (candidate). For PLI, wPLI, dsPLI and dswPLI the value 1 is reached in every pair of the audit iid_gaussian dataset at T ≥ 256, but this is a probability, not a certainty, at T = 256. Over 30 fresh iid seeds (P = 5, 20 off-diagonal entries each, `sat.py`) the fraction of entries equal to 1 is 0.50 / 0.96 / 1.0 / 1.0 for pli and wpli and 0.78 / 0.997 / 1.0 / 1.0 for dspli and dswpli at T = 64 / 256 / 1024 / 4096. The review probe (upper triangle, other seeds) gives 0.973 for pli and wpli and 1.0 for dspli and dswpli at T = 256.
- **EEG bands (hypothesis).** The EEG band identifiers fix the edges in cycles per sample at k/2048 (k = 1, 4, 8, 12, 30, 70, 150, 250). The sampling rates and window lengths are in the repo: `docs/storage_comparison/REVIEW.md:164` gives 1024 Hz for patient 004 and 2048 Hz for 105 and 111, with 512 versus 1024 samples per half-second window, and `REVIEW.md:16` reads the same 185 metrics for all three patients. `EEG_OPTIMIZATION_NOTES.md:414-415` at HEAD a42ebc1 (:246-247 at a1f20c7) adds two EDFs at 1024 Hz and three at 512 Hz. The same identifier is therefore 1-4 Hz at 2048 Hz but 0.5-2 Hz at 1024 Hz when each recording is computed at its native rate, as in the historical stored runs (patient 004: 1024 Hz, 512-sample windows, `REVIEW.md:164`). New runs at HEAD a42ebc1 bring every recording to the configured `sampling_rate_hz` before windowing, resampling native rates that differ (`pyspi/workflow/_prepare.py:104-106`), so with the example 2048 Hz (`examples/workflow/preflight.yaml:8`) the identifier is 1-4 Hz and windows hold 1024 samples whatever the native rate; nothing in the configuration requires `sampling_rate_hz` to be 2048 (`pyspi/workflow/_config.py:243-248`). With the inclusive rule (`eeg.py`, exact edges / edges rounded to the 3 significant figures printed in the identifier), the 1-4 band holds 2/1 bins at T = 1024 and 1/0 bins at T = 512 (the historical 1024 Hz windows). With rounded edges the 1-4 identifier would be all-NaN for 512-sample windows. The taper half-bandwidth 3/T is 6 Hz in both cases, wider than these bands. (Corrected in the check stage: the exact edges are in the repo. `pyspi/workflow/retained185.yaml:338-517` lists fmin/fmax = 0.00048828125 … 0.1220703125, i.e. exactly k/2048, for all six classes; see "Open questions: answered" 1.)

## spectral.plv

- **Documented.** Supplement §1.5.6, p.15: "computed using the same formula as coherence, however applied to the amplitude-normalized Fourier transformed signals (i.e., normalized by individual tapers)". GitBook spectral.md:297-316.
- **Canonical.** Lachaux et al. 1999, Hum Brain Mapp 8:194-208: PLV = |N⁻¹ Σₙ exp(iΔφₙ)| over N trials. The finite-N bias is ≈ √(π/4N).
- **Implemented.** `connectivity.py:510-539` computes PLV(f) = |mean_k S_ij,k/|S_ij,k||. pyspi additionally wraps `np.absolute` (`spectral.py:176-177`). That wrap is redundant and idempotent; the pack hypothesis is confirmed.
- **Deviations.**
  - **Null floor.** The taper expectation gives an independence-null level of 0.4016 (Monte Carlo E|R₅|; √(π/20) = 0.396) at every T. The probe on iid_gaussian gives 0.418, 0.398, 0.395 and 0.402.
  - **max under the null.** `plv_max` under the null rises with T: 0.835, 0.934, 0.951 and 0.978.
  - **DC bin.** The DC bin has a real cross-spectrum, so its PLV is one of {0.2, 0.6, 1.0}.
  - **Grid.** The grid deviation above applies.
  - **Symmetry.** The measure is symmetric, so the REVIEW M1 bypass has no effect here.
- **Checks.**
  - an independent multitaper reference (tolerance 1e-9)
  - an analytic null of 0.4016 (statistical)
  - narrowband PLV(f0) ≈ 1
  - the var2_bivariate population PLV from S_true (Monte Carlo)
  - exact symmetry
  - shared_envelope A[0,2] = 1
  - invariance, separability and detrend checks

## spectral.pli

- **Documented.** Supplement §1.5.8, p.15: "averaging the sign of the (per taper) phase difference". The labels say unsigned and undirected.
- **Canonical.** Stam et al. 2007, Hum Brain Mapp 28:1178-1193: PLI = |⟨sign Δφ(t_k)⟩|. In cross-spectral form it is |E sign(Im X)| over trials (Vinck et al. 2011).
- **Implemented.** `connectivity.py:541-574` computes E_k sign(Im S_ij,k). This is signed; the dependency's docstring says so, and the point is already documented in REVIEW M3. Per-bin values fall in {±0.2, ±0.6, ±1}, and the DC bin is always 0.
- **Direction convention.** The probe on delayed_noise at T = 1024 confirms it: A[i,j] > 0 means channel i phase-leads channel j by an angle in (0, π). The per-bin sign matches sin(2πfd) on 98.4-99.0 % of non-DC bins. The mean variants are exactly antisymmetric.
- **Deviations.**
  - **Signed, antisymmetric.** Already documented (M3, bug 5, EEG notes §review).
  - **Cancellation across frequency (already documented, REVIEW M3).** The band mean of the signed value cancels across frequency. For delayed_noise with a 5-sample delay the band mean is 0.180, against a mean |PLI(f)| of 0.937. This puts a number on M3's abs(mean) versus mean(abs) point; it is recorded under `already_documented`, not as a candidate.
  - **max variants (new).** The max variants are neither symmetric nor antisymmetric. On shared_envelope at T = 256, A[0,1] = 0 and A[1,0] = 1.
  - **Saturation (new).** Under independence `pli_max` reaches 1.0 in every pair of the audit dataset at T ≥ 256, in 0.96 of entries over 30 fresh seeds at T = 256 and in every entry at T ≥ 1024.
- **Checks.**
  - an independent reference, signed and canonical-unsigned (tolerance)
  - the delayed_noise direction check
  - the narrowband phase-shift signs: (0,1) = −1, (0,2) = +1, (1,2) = +1
  - null and saturation statistics
  - exact antisymmetry of the mean variants, and A[j,i] = −min_f PLI_ij for the max variants

## spectral.wpli

- **Documented.** Supplement §1.5.8: "weighting components based on the imaginary coherence".
- **Canonical.** Vinck et al. 2011, NeuroImage 55:1548-1565: Φ = |E Im X| / E|Im X|.
- **Implemented.** `spectral_connectivity/connectivity.py:576-601` computes E[Im S]/E|Im S| (signed). Any denominator below the absolute float eps is replaced by 1.
- **Deviations.**
  - **Signed.** Already documented (M3).
  - **Weighting wording.** The weighting uses the per-taper |Im S|, not the imaginary coherence; this is a wording mismatch.
  - **eps floor (new).** The floor makes the value scale-dependent on raw input. With `normalise=False` on delayed_noise, A[0,1] is 0.9536 at scale 1 and 1e-6 but 0.000000 at scale ≤ 1e-8. Under the default `normalise=True` the floor only bites at DC and for identical channels, which give 0 where the canonical value is 0/0.
  - **max variants.** They are asymmetric. Under the null they reach 1 in 0.96 of entries at T = 256 and in every entry at T ≥ 1024 (the same event as `pli_max` = 1).
  - **Cancellation across frequency (already documented, REVIEW M3).** Signed band means cancel for delays: 0.957, 0.319 and 0.199 for delays of 1, 3 and 5 samples. These numbers are recorded under `already_documented`.
- **Direction convention.** The same as PLI.
- **Checks.**
  - an independent reference, signed and canonical
  - direction and narrowband signs
  - a scale sweep with `normalise=False`
  - a Monte Carlo null
  - the common pipeline checks

## spectral.dspli

- **Documented.** Supplement §1.5.8 lists dspli among the "weighted variations"; the code applies no weighting.
- **Canonical.** Vinck et al. 2011, the unbiased estimator of PLI²: (1/(N(N−1))) Σ_{j≠k} sign(Im X_j)·sign(Im X_k).
- **Implemented.** `spectral_connectivity/connectivity.py:603-624` computes (K·PLI² − 1)/(K − 1) with K = 5. This confirms the pack hypothesis.
- **Deviations.**
  - **Closed form versus pairwise form (new).** The closed form equals the pairwise form only when no sign is 0. At DC the imaginary part of S is exactly 0, so the code gives −0.25 where the pairwise form gives 0. Every fmin = 0 band is therefore offset by −0.25/n_bins: −0.0078 for the full band and −0.0147 for the low band at T = 64.
  - **Identical channels (new).** Identical channels give −0.25, the minimum value and below the null level of 0 (shared_envelope x2 == x0 at T = 256 and 1024).
  - **Negative values.** These are valid and already documented (M7).
  - **max under the null (new).** `dspli_max` = 1 under the null for every pair of the audit dataset at T ≥ 256, for 0.997 of entries over 30 fresh seeds at T = 256 and for every entry at T ≥ 1024.
- **Checks.**
  - an independent reference in both the closed and the pairwise form, with the expected DC and identical-channel differences
  - a null distribution of {1 w.p. 1/16, 0.2 w.p. 10/32, −0.2 w.p. 20/32}, whose mean is 0
  - the var2_bivariate population PLI² (Monte Carlo)
  - exact symmetry
  - shared_envelope = −0.25

## spectral.dswpli

- **Documented.** Supplement §1.5.8.
- **Canonical.** Vinck et al. 2011: Σ_{j≠k} Im X_j·Im X_k / Σ_{j≠k} |Im X_j||Im X_k|.
- **Implemented.** `spectral_connectivity/connectivity.py:626-670` uses the same expression, written in sum form; it is an exact algebraic identity of the canonical one. A zero denominator gives NaN.
- **Deviations.**
  - **Expectation domain.** Only the taper expectation domain differs from the canonical definition.
  - **DC bin.** The DC bin is NaN, so it is silently excluded from the band.
  - **Identical channels.** Identical channels give NaN for every identifier. That is defensible, since the canonical value is 0/0, but it does put NaN into the matrix.
  - **max under the null.** `dswpli_max` = 1 under the null for every pair of the audit dataset at T ≥ 256, for 0.997 of entries over 30 fresh seeds at T = 256 and for every entry at T ≥ 1024.
  - **Negative values.** These are valid (M7).
- **Checks.**
  - an independent reference via an explicit double loop, including the NaN pattern
  - a Monte Carlo null
  - the population wPLI² (Monte Carlo)
  - symmetry
  - shared_envelope = NaN

## spectral.ppc

- **Documented.** Supplement §1.5.7, p.15: "quantifying the distribution of the (per taper) phase differences, which means that, unlike PLV, it is not biased by the sample size".
- **Canonical.** Vinck et al. 2010, NeuroImage 51:112-122: PPC = (2/(N(N−1))) Σ_{j<k} cos(θ_j − θ_k) = (N·PLV² − 1)/(N − 1).
- **Implemented.** `spectral_connectivity/connectivity.py:672-695` computes (|5·PLV_c|² − 5)/20. This confirms the pack hypothesis.
- **Deviations.**
  - **Sample size.** The "sample size" is always K = 5 tapers, so the per-bin variance (1/(K(K−1)) = 0.05 under the null) does not fall with T.
  - **Scale.** PPC targets PLV², so it is on a different scale from `plv`.
  - **DC bin.** The DC bin gives a value in {−0.2, 0.2, 1}.
  - **max under the null.** `ppc_max` grows with T under the null, from 0.635 to 0.946.
  - **Negative values.** These are valid (M7).
- **Null mean.** The null band mean is ≈ 0 at every T, which is consistent with unbiasedness over the tapers.
- **Checks.**
  - an independent pairwise-cosine reference
  - an analytic null with mean 0 and variance 0.05
  - the population PLV²
  - narrowband PPC(f0) ≈ 1
  - symmetry
  - shared_envelope = 1

## Open questions

1. **EEG band edges.** The window lengths and sampling rates are recorded (`REVIEW.md:164`: 512 samples at 1024 Hz for 004, 1024 samples at 2048 Hz for 105 and 111). What is missing is the exact fmin/fmax passed to pyspi; the identifier only prints 3 significant figures. Exact k/2048 edges and rounded edges include different edge bins, and with rounded edges the 1-4 identifier is empty (all-NaN) for 512-sample windows. Reading that identifier for patient 004 from the saved store would settle which was used. The fixed cycles-per-sample edges also mean the same identifier covers different Hz bands at 1024 and 2048 Hz, and every band is narrower than the multitaper resolution 2W = 6/T. (Added at synthesis: the Hz difference applies to recordings computed at their native rate, as in the historical stored runs; at HEAD a42ebc1 the workflow resamples to the configured rate before windowing, see "Open questions: answered" 1.)
2. **Taper versus trial estimators.** Is the taper-as-observation estimator intended? Any per-window phase-synchrony estimate with K = 5 has a null floor (PLV ≈ 0.40) and large variance at every T. The unbiasedness claims of PPC, dsPLI and dswPLI assume independent observations; the population checks on var2_bivariate are designed to measure the residual bias.
3. **Signed versus unsigned PLI and wPLI, and band aggregation.** This is already an open decision (REVIEW M3, EEG notes §review). The max variants add a further asymmetry that the decision should cover.
4. **max variants.** Should they be kept? Under independence PLI, wPLI, dsPLI and dswPLI reach 1 in most entries at T = 256 (0.96 to 0.997 over 30 seeds) and in every probed entry at T ≥ 1024.
5. **Literature citations.** The equation numbers in Lachaux 1999, Stam 2007, Vinck 2010 and Vinck 2011 were not re-checked against the papers. The definitions are cited from the dependency docstrings and from memory of the papers.
6. **Population formula.** For the population check, the closed form of E[exp(iΔφ)] for a circular complex Gaussian pair (a hypergeometric expression in |ρ|) was not verified. The proposed checks therefore use Monte Carlo targets.
7. **Cache key (minor, not probed to a failure).** The spectral cache key omits `fs`, and the stored `freq` grid comes from the first insert. Mixing different `fs` values on one Data object could pair cached spectra with a stale grid. Every configured identifier uses fs = 1.

## Checks

Files: `audit/checks/manifest/spectral_phase_sync.json` (1362 runs), `audit/checks/refs/spectral_phase_sync.py` (comparison venv; imports neither pyspi nor spectral_connectivity), `audit/checks/probes/spectral_phase_sync_probes.py` (project venv: pyspi's per-bin array from `_get_cache`, its frequency grid, the null summary, 30-seed saturation, the `normalise=False` scale sweep, the EEG band edges). Results: `audit/checks/out/results/spectral_phase_sync.json`, 1347 pass and 15 fail, 0 without a reference. All 15 failures are deliberate comparisons against a canonical form that the implementation does not follow: the PLI and wPLI band means against mne's unsigned forms (12 runs), the dsPLI closed form against the pairwise estimator (2 runs), and wPLI scale invariance (1 run). No failure is a disagreement with the implemented formula. The regression goldens were not used.

**References.**
- **Independent multitaper.** `_kernels_spectral.multitaper_fft` computes the coefficients: scipy DPSS with NW = 3 and K = 5 low-bias tapers, numpy FFT, channel-mean removal and the 5-smooth `next_fast_len`. The reference keeps the first n_fft/2 bins, forms S_ij,k elementwise with the sign-tie rule above, writes out each paper's formula over the tapers, and applies pyspi's inclusive band mask with nanmean or nanmax. Per-bin runs check the estimator; band runs (`rep`) check all six identifiers in both orientations on the kernel, pipeline and pipeline-detrend paths.
- **External package.** mne_connectivity 0.9.0 (`spectral_connectivity_epochs`) runs on an `EpochsSpectrumArray` whose "epochs" are the 5 taper coefficient sets (mode 'fourier', unit weights). mne's expectation over epochs is then the library's expectation over tapers. mne returns unsigned PLI/wPLI, so its per-bin values are compared with |pyspi per-bin| (probe `perbin_abs`). mne maps a zero dswPLI denominator to 0 (`epochs_bivariate.py:300-306`) where the library gives NaN, so the DC row is set to NaN on the reference side.
- **Analytic references.**
  - Null PLV: E|mean of K iid unit phasors| = (1/K)∫₀^∞ (1 − J0(t)^K)/t² dt = 0.401632 for K = 5. A 4·10⁶-draw Monte Carlo gives 0.40164 ± 0.0001; √(π/4K) = 0.396333 is only the large-K approximation.
  - Other nulls: E = 0 for PLI, wPLI, dsPLI, dswPLI and PPC under independence.
  - Phase signs on delayed_noise and on the narrowband phase shifts.
    - Population model on var2_bivariate: the expectation of each K = 5 estimator when the taper pairs are iid CN(0, S_true(f)). The se comes from 200 re-simulated VAR(2) series.
- **Same-lineage (pyspi self-consistency).** The symmetry (`sym`), normalisation-invariance (`inv`), separability (`sep02`, `sep13`) and `detrend-effect` runs compare pyspi with its own output (its transpose, its kernel-path run, its P-channel sub-block, its detrend=True run). They test properties, not the formula, and are labelled same-lineage in results.json. The max-variant `sym` runs of PLI and wPLI are the exception: their pass flag tests A[j,i] = −min_f X_ij(f) against the independent per-bin reference (independent-impl). No verdict or score rests on the same-lineage runs.

**Shared facts measured.**
- **Grid** (`spectral.plv|iid_gaussian|{64,256,1024,4096}|kernel|grid`). 32/128/512/2048 bins starting at f = 0, last bin 0.484/0.496/0.499/0.4998. The f = 0.25 bin is in both bands (low/high counts 17/16, 65/64, 257/256, 1025/1024). The documented grid is 125 bins from 4/T. Candidate confirmed; the 125-bin and f0 = 4/T part is new, DC/Nyquist is REVIEW M4.
- **DC bin** (`perbin` runs). The library values are PLV ∈ {0.2, 0.6, 1.0}, PPC ∈ {−0.2, 0.2, 1}, PLI = wPLI = 0, dsPLI = −0.25 and dswPLI = NaN, on every dataset.
- **Invariance and separability** (`inv`, `sep*`). Normalisation invariance holds to ≤ 7.8e-16 (6/6 per unit). Pairwise separability is exact (0.0, 4/4 per unit).
- **Detrending** (`detrend-effect`). The dependency removes only the mean. On trending T=256, pipeline vs pipeline-detrend changes the full-band means by 0.109 (PLV), 0.056 (PLI), 0.057 (wPLI), 0.044 (dsPLI), 0.082 (dswPLI) and 0.173 (PPC). Both paths are replicated by the reference to 1e-10.
- **On-grid line frequency (check-stage candidate, confirmed).**
  - When a sinusoid lies exactly on the grid, as at narrowband_f0-0.25, the odd DPSS tapers have zero gain at the exact line bin (relative DC gain 1.9e-14 at T=256). Two of the five "observations" therefore carry only noise.
  - At that bin, PLV is 0.42 to 0.78, PPC is −0.03 to 0.52 and dsPLI is −0.2 to 1. PLI magnitudes fall to as low as 0.2, but every sign still matches the designed lag direction. The neighbouring bins stay ≥ 0.9991 (`*|narrowband_f0-0.25|{256,1024}|kernel|nb-lock`, `nb-sign`).
  - The independent per-bin reference reproduces this, so it is a property of the taper-as-observation estimator, not a coding error.
  - The trace-stage expectation "PLV(f0) ≥ 0.99 at the f0 bin" was re-derived and corrected in the plv and ppc records. 50 Hz line noise at 2048 Hz with 1024-sample windows is on-grid.
- **Saturation** (`*|iid_gaussian|*|pipeline|sat`, 30 fresh seeds, full band, max). PLV and PPC max reach exactly 1.0 in 0.067 to 0.09 of null entries. Only the DC bin can produce an exact 1.0 for these measures (all five real DC cross-spectra of one sign, probability 1/16).

**spectral.plv.** Exact replication passes (150/150 band runs, max abs 1.5e-11, max rel 1.7e-11; 15/15 per-bin runs, max abs 8.5e-11), and mne passes per bin (8/8, 8.2e-12) and per band (6/6, 2.9e-14).
- **Null** (statistic = mean over upper-triangle pairs and interior bins). 0.4143 / 0.3976 / 0.3952 / 0.4019 at T = 64 / 256 / 1024 / 4096 against 0.4016, with z = 0.65 / −0.50 / −1.73 / 0.15 (`spectral.plv|iid_gaussian|T|pipeline|null`). The level does not fall with T, which confirms the taper expectation domain.
- **Max under the null.** The max variant averages 0.833 / 0.914 / 0.959 / 0.980.
- **Population** (var2_bivariate T=4096 full band). pyspi 0.5288 against the K = 5 independent-taper target 0.5309 (se 0.0074, z 0.28). The re-simulated mean exceeds the target by 0.0008 (z 1.6), so no significant taper-dependence bias at this resolution. The K → ∞ population PLV, (π/4)|ρ|₂F₁(½,½;2;|ρ|²), has band mean 0.399, so the K = 5 bias here is +0.13.
- **Other checks.** Scale invariance holds with `normalise=False` (1e-14). shared_envelope A[0,2] = 1.0. Symmetric: max |A − Aᵀ| = 0.0.
- **Verdict.** Variant, score 7 (rubric: matches the supplement's taper-normalised PLV; differs from canonical Lachaux in a value-changing way through the expectation over 5 tapers).
- **Candidates.** All seven confirmed: taper domain, max saturation, grid, DC, idempotent abs, the EEG band mechanism (`spectral.plv|iid_gaussian|1024|pipeline|eeg-bands`), and the on-grid line. The bin counts at L = 512 are exact [1, 17, 62, 2, 2, 5, 10, 20, 45, 25] against rounded [0, 17, 62, 2, 1, 4, 10, 20, 45, 25], and pyspi is NaN exactly in the empty 1-4 band.

**spectral.pli.** The signed estimator replicates exactly (150/150 band and 15/15 per-bin runs, max abs 0.0).
- **Against mne.** mne's unsigned per-bin PLI equals |pyspi per-bin| exactly (8/8).
- **Canonical band form (REVIEW M3; fails by design).** The Stam/Vinck magnitude form mean_f |PLI(f)|, via mne faverage, differs from pyspi's signed band mean. On delayed_noise T=1024 (full band, pair 0-3, d = 5), pyspi gives 0.1801 against 0.9371 (max abs 1.87 over both orientations). On iid_gaussian the off-diagonal mean is 0 against 0.369. |pyspi| equals |mean_f signed PLI| exactly, so the band value is the abs of a signed mean, not a mean of magnitudes.
- **Direction.** Band means on delayed_noise have the analytic sign on every checked pair: A[0,1..3] = 0.927 / 0.302 / 0.180. Per bin the sign agrees on every bin with |sin 2πfd| > 0.3, and on 0.984 of all non-DC bins. On the narrowband phase shifts (+π/4, −π/2) the signs at f0 = 0.1, 0.25 and 0.4 are (0,1) = −1, (0,2) = +1 and (1,2) = +1, with the transposes negated (6/6). So A[i,j] > 0 means i phase-leads j.
- **Null.** The null mean is consistent with 0 (|z| ≤ 1.70).
- **Max saturation.** The fraction of pli_max equal to 1 is 0.53 / 0.96 / 1.0 / 1.0 at T = 64 / 256 / 1024 / 4096, against an independent-bin prediction of 0.64 / 0.98 / 1 / 1.
- **Max saturation on coupled data** (Workflow 4b challenge probe, scratchpad `wf4b_mid-eeg/pli_max_sat.py`; not a check run). Saturation is not a null-only property. At T = 1024 and 4096 every off-diagonal entry is exactly 1.0, in both orientations and all three bands, on var2_bivariate, var2_chain, bivariate_gaussian and narrowband_f0-0.1. Below 1 stay some lagged delayed_noise pairs, shared_envelope's constant-phase-lag pair (x0 and x2 against x1, entries A[0,1] and A[2,1]: 0.0 in the full and low bands and −1.0 in the 0.25–0.5 band at T = 1024 and 4096; Workflow 4b-2 recheck probe scratchpad `wf4b2_recheck_mid-eeg/se.py`) and identical channels (0.0). The max identifiers (not in the EEG selection) are therefore close to constant at T ≥ 1024; the unit score rests on the mean identifiers.
- **EEG bands** (challenge probe `pli_eeg_bands.py`, P = 20 iid, T = 1024; no check run covers PLI values at the EEG bands, the `spectral.plv` eeg-bands run covers bin counts only). With inclusive edges the 1–4, 4–8 and 8–12 Hz bands (0.5–2, 2–4, 4–6 /1024) hold 2, 3 and 3 bins, and adjacent bands share their edge bins (bin 2, bin 4). Per-bin values are odd multiples of 0.2 with 5 tapers, so these band means take at most 11 or 16 levels; on the 380 off-diagonal entries they take 11, 12 and 16 distinct values with null SD 0.355, 0.370 and 0.371, against 0.037 for the full band. The 11 retained PLI identifiers are mean variants; the three narrow ones are coarsely quantised and noisy.
- **Symmetry.** The mean variants are antisymmetric (max |A + Aᵀ| = 0.0). For the max variants the identity A[j,i] = −min_f PLI_ij holds on all 9 max sym runs (asymmetric; saturated null matrices are all ones and hence symmetric).
- **Verdict.** Variant, score 6 (rubric anchor: signed PLI). Definition under review.
- **Candidates.** Four confirmed. The max-variant asymmetry and saturation are not in KNOWN_ISSUES (M3 and bug 5 cover the mean variants).

**spectral.wpli.** The signed estimator replicates with the eps floor reproduced (rep 150/150 at 1.9e-13; per bin 15/15 at 3.7e-12), and mne's unsigned per-bin wPLI matches |pyspi| (2.6e-12).
- **Canonical band form (M3; fails by design).** delayed_noise d = 5 gives 0.1992 against 0.9781; iid gives 0 against 0.488.
- **Direction.** Agreement is 1.0; band A[0,1..3] = 0.957 / 0.319 / 0.199, and per bin 0.991 of non-DC bins agree.
- **eps floor (confirmed, new).** With `normalise=False`, A[0,1] on delayed_noise T=256 is 0.9536 at scales 1 and 1e-6, but 6.2e-17 at 1e-8 and 6.2e-21 at 1e-10. The floor sets the denominator to 1, so the value becomes the raw mean Im S. The canonical-invariance run fails by 0.954, and the floor-reproducing reference matches to 5.4e-16. The trace's "0.000000" is corrected to these values in the record.
- **Other checks.** Identical channels and DC give 0.0. The null mean is consistent with 0 (|z| ≤ 1.52). Saturation is identical to PLI. Population: pyspi 0.4628 against 0.4521 (z 0.48).
- **Verdict.** Variant, score 6 (signed wPLI). Definition under review.

**spectral.dspli.** The closed form (K·PLI² − 1)/(K − 1) replicates exactly (150/150 and 15/15, 0.0). mne's `pli2_unbiased` uses the same closed form (Vinck 2011 eq. 30 per mne's comment) and agrees exactly, including −0.25 at DC (8/8 per bin, 6/6 per band).
- **Pairwise form (fails by design).** The canonical pairwise estimator, computed by an explicit double loop, agrees on non-DC bins to 5.6e-17. It differs by exactly 0.25 at DC and on the identical pair shared_envelope (0,2), where pyspi gives −0.25 against pairwise 0 (`spectral.dspli|{delayed_noise,shared_envelope}|256|kernel|canon-pairwise`). Every fmin = 0 band therefore carries an offset of −0.25/n_bins (−0.0078 at T=64 for the full band, −0.00049 at T=1024).
- **Negative values** are valid finite-sample outcomes (REVIEW M7, `docs/storage_comparison/REVIEW.md:189-193`): the band minimum is −0.10 on scaled_offset and the per-bin minimum −0.25.
- **Null and saturation.** The null mean is consistent with 0 (|z| ≤ 1.61). dspli_max equals 1 in 0.78 / 1 / 1 / 1 of null entries.
- **Population.** pyspi 0.1383 against 0.1318 (z 0.56).
- **Verdict.** Variant, score 7 (matches the supplement's debiased PLI² over tapers; the closed form deviates from the pairwise definition only through sign(0) at DC and on identical channels, by at most 0.25/n_bins on a band of distinct channels). All four candidates confirmed.

**spectral.dswpli.** Exact replication passes with equal NaN masks (rep 150/150 at 1.9e-13; per bin at 1.9e-11). The explicit pairwise double loop agrees to 2.2e-13. mne's `wpli2_debiased` agrees to 8.6e-12 per bin and 9.2e-14 per band once its zero-denominator convention (0 instead of NaN at DC) is aligned.
- **NaN entries.** DC is NaN on every dataset and identical channels are NaN in every identifier.
- **Null and saturation.** The null mean is consistent with 0 (|z| ≤ 1.24). The max variant saturates to 1 like dsPLI.
- **Population.** pyspi 0.2447 against 0.2439 (z 0.04). The re-simulated mean exceeds the independent-taper target by 0.0026 (z 2.03), a borderline indication of a small positive bias from taper dependence. 200 re-simulations cannot settle it.
- **Negative values.** These are valid (M7).
- **Verdict.** Variant, score 7. All three candidates confirmed.

**spectral.ppc.** The closed form replicates (rep 3.3e-11, max rel 7.1e-10; per bin 6.5e-11). The pairwise-cosine double loop agrees to 2.0e-12, and mne agrees to 1.4e-11 per bin and 3.4e-14 per band.
- **Null.** The mean is consistent with 0 at every T (|z| ≤ 1.92), so PPC is unbiased over tapers. The max variant still grows under the null: 0.630 / 0.798 / 0.900 / 0.950.
- **Population.** pyspi 0.1587 against 0.1641 (z 0.54). The target equals the K → ∞ population PLV² band mean of 0.1641, which is the unbiasedness Vinck 2010 claims, here with tapers as observations.
- **Other checks.** shared_envelope A[0,2] = 1.0. Symmetric.
- **Verdict.** Variant, score 7. All three candidates confirmed.

**Trace-stage fields corrected.**
- The tie-rule sentence in the pli, wpli, dspli and dswpli records and in this note now states the measured facts: no rounding noise at DC, and noise only on the diagonal and for the bit-identical channels.
- The wpli scale-sweep value at s ≤ 1e-8 is 6.2e-17 and 6.2e-21, not 0.000000.
- The plv and ppc narrowband expectations now exclude the exact on-grid line bin.
- The EEG notes citation gives the committed line only (`EEG_OPTIMIZATION_NOTES.md:246-247` at a1f20c7); the working-copy line moves as the uncommitted file changes. [Synthesis annotation: the file is now committed at HEAD a42ebc1, where this text is at `EEG_OPTIMIZATION_NOTES.md:414-415`.]
- The Calculator citation is updated to HEAD a1f20c7 (`pyspi/calculator.py:295-355`) in all six records and in this note.
- The trace-stage statement that the exact EEG fmin/fmax are absent from the repo is wrong: `pyspi/workflow/retained185.yaml` lists them (corrected in the plv candidate, its resolution, and "Open questions: answered" 1).
- A check-stage candidate (on-grid line) was appended to `spectral.plv`.

## Open questions: answered

1. **EEG band edges.** The mechanism is now measured (`eeg-bands` run).
   - With the inclusive rule, exact k/2048 edges and the .3g-rounded edges select different bins. At 512 samples the 1-4 band holds 1 bin (exact) or 0 bins (rounded), and pyspi returns NaN for the empty band. At 1024 samples it holds 2 or 1 bins.
   - The edges are in the repo. The `retained185` metrics preset (resolved by `pyspi/workflow/_config.py:230-235` at HEAD a42ebc1, :143-148 at a1f20c7 and used by `examples/workflow/preflight.yaml` with 2048 Hz and 0.5 s windows) is `pyspi/workflow/retained185.yaml`. It lists fs = 1 and fmin/fmax = 0.00048828125, 0.001953125, 0.00390625, 0.005859375, 0.0146484375, 0.0341796875, 0.0732421875, 0.1220703125 for PhaseLockingValue (:338-367) and the other five classes (:368-517). These are exactly k/2048 for k = 1, 4, 8, 12, 30, 70, 150, 250, the probe's `EEG_K` row, not the .3g-rounded identifier values.
   - With these edges the 1-4 band holds 1 bin at 512 samples (`ref_counts_L512_exact[0] = 1` in `spectral.plv|iid_gaussian|1024|pipeline|eeg-bands`) and 2 bins at 1024 samples, so the configured 1-4 identifier is not all-NaN by construction for either window length.
   - Caveat: the file header (`retained185.yaml:1-5`) calls it a reconstruction from the saved-store inventory. Only the historical store can confirm which edges the historical run used.
   - Which runs have 512-sample windows (added at synthesis): those computed at a native 1024 Hz, such as the historical stored run for patient 004 (`REVIEW.md:164`), where the 1-4 identifier covers 0.5-2 Hz. At HEAD a42ebc1 the workflow resamples every recording whose native rate differs from the configured `sampling_rate_hz` (`pyspi/workflow/_prepare.py:104-106`, polyphase) before windowing, so new runs with the example 2048 Hz (`examples/workflow/preflight.yaml:8`) have 1024-sample windows and 1-4 Hz for 1024 Hz and 512 Hz recordings as well. That rests on `sampling_rate_hz` being 2048, which the configuration does not enforce (`pyspi/workflow/_config.py:243-248`).
   - The bands remain narrower than the multitaper resolution 2W = 6/T.
2. **Taper versus trial estimators.** The check confirms what the code does: the null level is 0.4016 at every T, and the per-bin variances are fixed by K = 5.
   - The independent-taper assumption behind the debiased estimators holds closely on var2_bivariate. The re-simulated means differ from the independent-taper expectation by 0.0002 to 0.0026 (|z| ≤ 2.03), and every pyspi value is within 0.6 se of its target.
   - Whether this estimator is intended remains a design decision, not a computation error.
   - The on-grid line effect is a further consequence of using tapers as observations.
3. **Signed vs unsigned PLI/wPLI and band aggregation (M3).** Still a decision, now with numbers. The canonical mean_f |·| differs from pyspi's signed mean by up to 1.87 (PLI) and 1.95 (wPLI) on delayed_noise. The max variants obey A[j,i] = −min_f X_ij exactly.
4. **Should the max variants be kept?** Measured, not decided.
   - Under independence PLI, wPLI, dsPLI and dswPLI max equal 1 in 0.96 to 1.0 of entries at T = 256 and in all entries at T ≥ 1024 (30 seeds).
   - PLV and PPC max reach 0.98 and 0.95 at T = 4096, and are exactly 1 in about 1/16 of entries through the DC bin alone.
   - These values carry little information about coupling at the EEG window lengths.
5. **Literature equation numbers.** Not re-checked against the papers in this stage.
   - mne_connectivity's own comment cites Vinck 2011 eq. (30) for the closed-form debiased PLI², the form the library also uses.
   - The definitions themselves were checked numerically: closed forms agree with explicit pairwise loops to ≤ 2.2e-13 except where sign(0) intervenes (dsPLI).
6. **Population closed form.** Checked numerically in this stage. For a circular complex Gaussian pair with complex coherency ρ = |ρ|e^{iθ}, E[e^{iΔφ}] = (π/4)|ρ|₂F₁(½, ½; 2; |ρ|²)·e^{iθ}. A 2·10⁶-draw Monte Carlo at |ρ| = 0.1, 0.5 and 0.9 and θ = 0 or 1 agrees to about 1e-3, within Monte Carlo error. The proposed population checks still used Monte Carlo targets, because the K = 5 estimator's expectation (not the K → ∞ population value) is the right target.
7. **Cache key omits fs.** Not probed. Every configured and EEG identifier uses fs = 1, and the check runs build a fresh Data per SPI, so nothing here exercises it.
8. **Supplement line ranges (pack offset).** Confirmed by the trace: §1.5.6 to §1.5.8 start at supplement.txt:686, :701 and :715. No check depends on it.

## Optimisation

Sources: walls from `audit/profiles/summary.json` (fresh / family-warm; quiet machine; 118x1024 walls are single repeats), frame shares from cProfile. The official profiles at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`) are dominated by the P-independent DPSS cost for every multivariate unit, so for ranking at 118 channels this group added cProfile runs at 118x1024 (32x1024 for sgc nonparametric): `audit/profiles/micro/prof_<unit>__<shape>.pstats`, written by `audit/profiles/micro/prof_spectral.py`, indexed with walls, peak RSS and thread settings in `audit/profiles/micro/prof_spectral__index.json` (OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1; no numba or JIDT code on these paths). Those runs overlapped other Workflow 3 groups; the memory-bound multitaper units varied 2-7x in wall between runs (cohmag: 6.4 s on a first run, 0.93 s on the kept run, 2.42 s in summary.json), so only shares are quoted from them. Percentages are of the profiled `harness.compute_spi` frame. Micro-benchmarks (the group's two): `audit/profiles/micro/opp_spectral_csm-1.{py,json}` and `audit/profiles/micro/opp_spectral_csm-shared-csm.{py,json}`. A candidate that reproduces the current output "matches the baseline"; that says nothing about whether either is mathematically right. Nothing in `pyspi/` was changed.

| unit | 8x1024 fresh / fw (s) | 32x1024 fresh / fw (s) | 118x1024 fresh / fw (s), mode | peak RSS 32 / 118 fresh (MB) |
|---|---|---|---|---|
| spectral.plv | 0.0241 / 0.0484 | 0.1003 / 0.1991 | 7.970 / 15.925, measured | 557 / 3069 |
| spectral.pli | 0.0182 / 0.0205 | 0.0636 / 0.0641 | 2.945 / 3.406, measured | 527 / 1981 |
| spectral.wpli | 0.0191 / 0.0199 | 0.0976 / 0.0966 | 4.263 / 5.437, measured | 527 / 1982 |
| spectral.dspli | 0.0182 / 0.0193 | 0.0642 / 0.0643 | 2.806 / 3.670, measured | 527 / 1982 |
| spectral.dswpli | 0.0207 / 0.0220 | 0.1503 / 0.1486 | 6.797 / 10.224, measured | 527 / 2205 |
| spectral.ppc | 0.0247 / 0.0226 | 0.0990 / 0.1033 | 8.436 / 9.173, measured | 613 / 3067 |

plv is first in the family order and pays the build twice (7.80 s and 8.04 s at 118) through the first-insert string key; other bands cost 0.007-0.028 s.

**Hotspots at 118x1024** (`audit/profiles/micro/prof_spectral.<unit>__118x1024.pstats`); every unit applies an elementwise fcn to the (1,1,5,1024,118,118) per-taper tensor before the taper mean:
- plv: `connectivity.py:_phase_locking_value.fcn` (x/|x|) 78.5%, `_complex_inner_product` 11.0%, reduce 4.0%, `dpss_windows` 0.5%.
- ppc: the same fcn 79.5%, `_complex_inner_product` 9.6%, reduce 3.7%, `dpss_windows` 0.4%.
- pli: `_expectation_cross_spectral_matrix` 88.7% = `_complex_inner_product` 51.8% + fcn sign(Im) 23.5% + reduce 11.0%; `dpss_windows` 2.3%.
- dspli: `_expectation_cross_spectral_matrix` 84.5% = `_complex_inner_product` 50.6% + fcn 22.2% + reduce 9.7%; `dpss_windows` 2.4%.
- wpli: `_complex_inner_product` 60.3% over 2 builds, reduce 15.3%, fcn |Im| 12.0%, `dpss_windows` 1.4%.
- dswpli: `_complex_inner_product` 54.5% over 3 builds, reduce 13.5%, `fcn_imag_sq` 8.6%, `fcn_abs_imag` 7.9%, `dpss_windows` 0.8%.
At 8x1024 `dpss_windows` is 54.8-75.3% of the official profiles. opaque_share (8x1024 JSON): 0.032-0.068, a lower bound: the divide/abs/sign work sits in the self time of the library `fcn` frames.

**Complexity.** O(K P T log T) FFT + O(K n_fft P^2) per-taper CSM + O(K n_fft P^2) elementwise fcn + taper mean; the per-taper terms dominate at 118 and their temporaries are GB-sized.

**Opportunities.**
- `spectral_csm-1`: plv 1.8-2.5x (min of 3 repeats: 1.81x at 32x1024 and 2.45x at 118x1024, ratio of medians 2.88x, bit-identical; a Workflow 3 recheck re-run gave 1.89x min of 3 at 118x1024, ratio of medians 1.86x, and a Workflow 4c challenge re-run 1.83-2.46x per pair (`scratchpad/wf4c_b1spec/paired.py`, run as `paired.py 118x1024 3`), neither merged into the JSON; the 9.64x and 4.14x of earlier single-repeat runs came from slow memory-bound baselines and are not used, see spectral_csm.md). ppc 1.8-2.1x, wpli 1.8-3x and dswpli 1.76-3x are now micro-benchmark based: Workflow 4c added them to `opp_spectral_csm-1.json` (`--add` mode, paired repeats; candidate = the library's undecorated method on Fourier coefficients sliced to the non-negative bins), bit-identical with equal NaN masks at 32x1024, 118x1024 and on the 7x100 golden set. Paired figures: 32x1024 ppc 1.73-1.84x, wpli 1.90-1.93x, dswpli 1.95x; 118x1024 ppc 3.6-3.7x, wpli 9.9-10.7x, dswpli 8.9-10.4x in that run against 2.1x, 2.6-3.0x and 2.7-3.0x in the challenge re-run (`scratchpad/wf4c_b1spec/paired.py`; its output `paired_118x1024.json` is not merged into the micro-benchmark JSON and the probe archive keeps scripts only, so the records quote the per-repeat figures). The 118x1024 figures follow the memory state of each run (the two-sided baseline allocates the 1.14 GB per-taper tensor once, twice or three times), so on all four units that time this candidate the upper end is the challenge re-run's highest paired figure (plv 2.5x, ppc 2.1x, wpli and dswpli 3x), from the one run that timed the four in one process and memory state, and the saved `--add` run is reported but not used for the range (ppc's 3.6x there is above its own element-count figure 1.91x); each lower bound stays that unit's own Amdahl figure for halving the taper-level frames on the ranking basis, rounded down (ppc 1.89x, wpli 1.805x, dswpli 1.766x on the configured sums). pli 1.76-2.5x and dspli 1.7-2.5x keep the complexity basis (not run, so their records do not tag it microbench; lower bounds 1.765x and 1.707x on the configured sums, rounded down; the 2.5x upper end is plv's 2.45x min of 3 at 118x1024; unverified-bit-identical). The 118x1024 walls behind every ranking basis here are single family-warm repeats, and plv's build ran 1.5-1.8 s, 7.80 s and 9.0-10.1 s in three runs, so the patient-second savings of these rows and their order against compute-bound rows are uncertain by up to about 6x.
- `spectral_phase_sync-1` (cache-sharing, S; seed (d)): evaluate the per-taper CSM once per call and share the taper averages within measure groups (plv/ppc: x/|x|; pli/dspli: sign(Im); wpli/dswpli: Im, |Im|, Im^2). Within one call it removes one of wpli's two builds (30% of the call: 1.43x) and two of dswpli's three (36%: 1.57x); across a group the second consumer drops up to its whole expectation share (plv 96.4%, ppc 95.2%, pli 88.7%, dspli 84.5%), hence the records' ranges: plv and ppc 1-20x, pli 1-8x, dspli 1-6x, wpli 1.4-8x and dswpli 1.5-8x (the 1x lower bounds are the first consumer of a group; wpli's and dswpli's are their within-call figures, rounded down). Complexity basis only; unverified-bit-identical (the same per-taper values feed each fcn). In the ranking (README 'Optimisation columns', rule 4) `spectral_csm-1` is stacked first on all six units, and this avenue `depends_on` it on all six, ranking on its residual. On plv, ppc, pli and dspli its `residual_speedup` lower bound is 1x (the unit that pays the group's shared build; ranges 1-14x, 1-9.3x, 1-4.2x and 1-3.4x). On wpli and dswpli it is 1.36x and 1.46x: the repeated builds are taper-level frames that `spectral_csm-1` halves while the rest of the call stays, so they are 27.5% and 32% of the residual against 30.2% and 36.3% of today's cost (records' speedup_basis; an in-memory build_csv run of 2026-10-15: wpli about 1,812 -> 1,332 s per patient, dswpli about 3,486 -> 2,387 s).
- `spectral_csm-shared-csm` is refuted for these units: the shared Fourier coefficients save the Multitaper/DPSS/FFT (0.4-3.5% at 118) but not the per-taper products that dominate.

**EEG per-window cost at 118 channels** (11 EEG identifiers each, one build per unit): plv 7.80 + 10 x 0.021 = 8.0 s, pli 3.33 + 10 x 0.015 = 3.5 s, wpli 5.36 + 10 x 0.015 = 5.5 s, dspli 3.60 + 10 x 0.015 = 3.7 s, dswpli 10.15 + 10 x 0.015 = 10.3 s, ppc 9.09 + 10 x 0.016 = 9.3 s; 40.3 s for the six. With `spectral_phase_sync-1` the three groups cost about one build each (plv 8.0, pli 3.5, dswpli-type ~6 s: about 18 s); adding `spectral_csm-1` (1.8-2.5x measured on plv, min of 3; 1.76-3.6x paired on ppc, wpli and dswpli; element-count ranges 1.7-2.5x and 1.76-2.5x on dspli and pli) brings that to roughly 7-10 s (plv 8.0 / 1.8-2.5 = 3.2-4.4 s, pli group 3.5 / 1.79-2.5 = 1.4-2.0 s, dswpli-type group ~6 / 1.77-2.5 = 2.4-3.4 s). (An earlier version said 5-6 s from the plv 9.6x that did not reproduce.) The ranking takes plv on its retained build alone (`optimisation.eeg_basis: retained-increments`, 7.80 s per window, 4,681 s per patient, `basis_scope` retained-identifiers-approx; the 8.04 s first-insert duplicate is paid on spectral.phase in an EEG run); the five others rank on their configured sums (configured-identifiers-approx).

**Memory.** The two-sided taper-level tensor is 5 x 1024 x P^2 x 16 B = 1.14 GB at P = 118 and 5.08 GB at P = 249. Peak RSS (fresh records): 449-613 MB at 32x1024, 1.66-3.07 GB at 118x1024 (1.32-2.73 GB above the ~340 MB process baseline; family-warm records: 1.89 GB for the coherence family, 3.41 GB for the phase-sync family). Scaled by P^2 the increment is 5.9-12.2 GB per process at 249 channels. On this 31 GB machine that caps window-level concurrency at roughly 9 processes for the phase-sync units at 118 channels and 2-4 at 249; `spectral_csm-1` halves the tensor (one-sided) and the gemm expectation removes it for the linear measures (114 MB at 118, 508 MB at 249).

**Refuted seeds.** DPSS cache (0.4-2.4% at 118); shared expectation CSM (these units do not use it).
