# spectral_coherence family: define, trace and check

Units: `spectral.cohmag`, `spectral.phase`, `spectral.icoh`. Each unit covers six identifiers: three bands (`fmin-0_fmax-0-5`, `fmin-0_fmax-0-25`, `fmin-0-25_fmax-0-5`) times two statistics (`mean`, `max`), plus the 11 retained EEG `mean` identifiers with narrow bands.
Classes: `CoherenceMagnitude`, `CoherencePhase`, `ImaginaryCoherence` in `pyspi/statistics/spectral.py:137-164`.
Delegate: spectral-connectivity 1.1.2 (`Multitaper`, `Connectivity`), using scipy.fftpack FFTs, numpy 2.5.4.
Probe scripts are in the session scratchpad under `spectral_coherence/` (`probe1.py` to `probe7.py`, plus a from-scratch reference in `ref.py`). The reference takes n_fft from `scipy.fftpack.next_fast_len`, the rule the library imports (`transforms.py:29`); `scipy.fft.next_fast_len` gives a different grid for T that is not 5-smooth (63, 67 and 77 give 63, 70 and 77 instead of 64, 72 and 80). Probe 7 shows the fftpack rule matches the library to ≤ 1.4e-14 at T ∈ {63, 67, 77}, while the scipy.fft rule misses it by up to 0.24 for cohmag at T=67. At the manifest lengths (powers of two) the two rules agree. The regression goldens are snapshots of current output, and nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared documented definition

Supplement §1.5, p.12 (heading at supplement.txt:570; GitBook `spectral.md:14-20`):

> "Each measure is based on the cross- and power-spectral densities ... estimated via the multitaper technique ... we assume a sampling frequency fs = 1 ... We use 125 uniformly sampled bins across the entire frequency range, f ∈ [f0, fs/2], where ... f0 = 4/T ... we take the mean (mean) and the maximum (max) ... fmin-0_fmax-0-5: f ∈ [0, fs/2]; fmin-0_fmax-0-25: f ∈ [0, fs/4]; fmin-0-25_fmax-0-5: f ∈ [fs/4, fs/2]."

The briefing pack's supplement line ranges are 12 to 14 lines later than the headings in this copy of `supplement.txt`. The actual heading lines are §1.5 at 570 (p.12), §1.5.1 at 601 (p.13), §1.5.2 at 616 (p.13) and §1.5.5 at 670 (p.14).

## Shared implementation (trace)

1. **Preprocessing.** `pyspi/data.py:184-195` applies an optional linear detrend, then a z-score with ddof=1.
2. **Dispatch.** `pyspi/calculator.py:327-339` (HEAD a1f20c7) calls `S = self._spis[spi].multivariate(self.dataset)` (:329). Each class has the MRO `[cls, NonparametricSpectralMultivariate, NonparametricSpectral, Unsigned, Undirected, Directed]` (probe), so `spectral.py:85-102` replaces `Undirected.multivariate` (`base.py:121-138`) and nothing is mirrored. This is already documented in REVIEW M1. It was the earlier wording of tests/README bug 5; the current bug 5 (tests/README.md:206) is PLI signed antisymmetry (see `briefing/KNOWN_ISSUES.md` item 5), so the records cite M1.
3. **Cache.** `spectral.py:64-83`:
   - `z = data.T` (T × P).
   - `sc.Multitaper(z, sampling_frequency=1)` runs with every other argument at its default.
   - `Connectivity.from_multitaper(m)` builds the connectivity object, and `getattr(conn, measure)()` computes the measure.
   - The result is cached under the key `(measure,)` (`spectral.py:45-50`), so all six band/statistic identifiers of a unit share one spectrum.
4. **Multitaper** (`transforms.py`):
   - Defaults (74-102): NW = 3, n_tapers = floor(2NW−1) = 5 (154-164), low-bias filter on eigenvalues > 0.9 (668-673). The probe kept all 5 tapers for every T ≥ 8.
   - The whole series is one window (166-178). Each window has its mean removed (253-254, 747-748).
   - n_fft = `next_fast_len(T)` (180-185). This zero-pads when T is not 5-smooth: T=67 gives n_fft=72, and T=63 gives 64.
   - FFT of taper·x divided by fs (335-354). The tapers are DPSS × √fs (357-387). They equal `scipy.signal.windows.dpss(T,3,Kmax=5)` to within 3.7e-15.
5. **Connectivity** (`connectivity.py`):
   - Expectation `trials_tapers` is an unweighted mean over the tapers (45-53).
   - S_ij = mean_k X_ik conj(X_jk) (195-208, 1187-1190, 1485-1487).
   - Coherency C = S_ij/√(S_ii S_jj), with zero norms set to NaN and a NaN diagonal (295-313).
   - Only the first (n_fft+1)//2 bins are kept (73-89). DC is kept, and the Nyquist bin is dropped for even n_fft.
6. **Band summary.** `spectral.py:88` selects bins with `fmin <= f <= fmax`, a band closed at both ends. Line 91 applies `np.nanmean` or `np.nanmax` across those bins, and line 101 sets the diagonal to NaN.

**Probe 2** builds a from-scratch reference: numpy FFT, scipy DPSS and no spectral_connectivity import. It reproduces all 18 identifiers to ≤ 2e-12 on iid_gaussian, delayed_noise, var2_bivariate, narrowband_f0-0.1 and shared_envelope at T ∈ {64, 256, 1024}. Probe 5 shows that the Calculator output is identical to direct `multivariate`.

**Grid deviation (all units, candidate).** The documented "125 uniformly sampled bins on [4/T, fs/2]" is not what runs. The code uses the FFT grid k/n_fft: T=64 gives 32 bins and T=256 gives 128 bins (0 … 0.49609). DC is included and fs/2 never is. The low and high bands both contain the f = 0.25 bin. REVIEW M4 and DEPENDENCY_COMPATIBILITY_REVIEW.md already cover the DC/Nyquist part, but not the 125-bin/f0 statement.

**Empty bands (candidate).** At T=64 with the EEG band [1/2048, 4/2048], `mean` returns an all-NaN matrix and `max` raises `ValueError: zero-size array to reduction operation fmax`.

**Fixed NW (candidate).** NW = 3 and K = 5 for every T, so the estimator's degrees of freedom do not grow with T and the bandwidth 2W = 6/T shrinks instead. Bias and null levels therefore stay constant in T (numbers below).

## spectral.cohmag

- **Documented** (§1.5.1, p.13; GitBook `spectral.md:24-45`): "The coherence (also known as coherence magnitude and ordinary coherence [60] or coherence coefficient [59]) ... the frequency domain equivalent of the squared time-domain cross correlation [59]."
- **Canonical:** the magnitude-squared coherence γ² = |S_xy|²/(S_xx S_yy) (Carter 1987, Proc. IEEE 75:236), estimated with multitapers (Thomson 1982; Percival & Walden 1993 ch. 7). Bastos & Schoffelen 2016 [59] define the "coherence coefficient" as the unsquared |C| ("a normalized quantity bounded by 0 and 1", numerator "the length of the vector average" of the cross-spectra) and write "the coherence is the frequency domain equivalent to the time domain cross-correlation function", without "squared". The phrase "coherence magnitude" does not occur in that paper; the supplement attributes it to [60] (Schlögl & Supp 2006, not opened). Thomson's locations (section III pp.1060-1061, section XIV p.1087) and Percival & Walden ch. 7 (pp.331-377, from the authors' table of contents) are the ones checked in the spectral_csm notes; Carter's section numbers and bias formula are from memory.
- **Implemented:** `connectivity.py:343`, `_squared_magnitude(coherency())`, with `:1184` `abs(x)**2`. This is |C|², averaged or maximised over the band.
- **Deviations:**
  - *Squaring (pack hypothesis confirmed and refined).* The quantity is squared. The supplement sentence ("equivalent of the squared time-domain cross correlation [59]") describes the squared quantity, but its cited source [59] does not say "squared" and defines the coherence coefficient as |C|. So the code matches that supplement sentence and departs from both [59] and the plain reading of "magnitude". On bivariate_gaussian at T=4096, pyspi gives 0.2647 and 0.6715, equal to mean|C|²; mean|C| is 0.4766 and 0.8109.
  - *Bias.* The null level is about 1/K = 0.2 at every T: iid_gaussian gives 0.2295, 0.1983, 0.1964 and 0.1994 for T = 64 … 4096.
  - *Max statistic.* The `max` statistic grows with T under the null: 0.666, 0.765, 0.791, 0.878.
  - *Non-convergence.* On var2_bivariate at T=4096, pyspi gives 0.3648 against a population value of 0.2465. Carter's bias (1−γ²)²/K = 0.114 accounts for the difference.
  - *Symmetry.* The output is exactly symmetric, so the missing mirror does not matter here.
- **Proposed checks:**
  - Kernel independent re-implementation (tolerance 1e-10).
  - shared_envelope x2 ≡ x0: the result is exactly 1.
  - bivariate_gaussian against ρ² plus the Carter bias, which tells |C|² apart from |C|.
  - Beta(1, K−1) null on iid_gaussian.
  - var2 against the population S_true.
  - Exact invariance on scaled_offset.
  - pipeline-detrend on trending: without detrend, pair (1,3) at T=1024 is inflated to 0.5055, against 0.2148.
  - Exact symmetry and permutation equivariance.
  - Band-bin enumeration.
  - Optional same-lineage nitime check.

## spectral.phase

- **Documented** (§1.5.2, p.13; GitBook `spectral.md:49-70`): "By omitting the magnitude operator ... the complex-valued coherency, where the phase-difference angle (which we refer to as the coherence phase) has been used to infer a time-delayed dependence". It is labelled "undirected, unsigned".
- **Canonical:** the cross-spectral phase φ_ij = arg S_ij (Priestley 1981 ch. 9; Carter 1987). A band average of angles should be a circular mean (Mardia & Jupp 2000 §2.2).
- **Implemented:** `connectivity.py:324`, `np.angle(coherency())`, a principal value. The arithmetic `nanmean` or `nanmax` is then taken over the band. DC is exactly real, so its angle is exactly 0 or π.
- **Direction convention (probe):** A[i,j] > 0 means i leads j.
  - delayed_noise at T=1024 (x1 = x0 delayed by 1), low band: A[0,1] = +0.7768 and A[1,0] = −0.7768, against an analytic −0.7854.
  - narrowband f0 = 0.1: A[1,0] = +0.8044 (x1 is shifted by +π/4) and A[0,2] = +1.5609.
- **Deviations:**
  - *Already documented (M4).* The band summary is an arithmetic mean of principal angles, DC is included, the Nyquist bin is excluded, the antisymmetry offset is 2π/n_bins and the unsigned label is wrong.
  - *Exact offset identity (new).* A[i,j] + A[j,i] = π·n_π/n_bins. On iid_gaussian at T=64 in the low band this gives 0.369599 = 2π/17 for the two pairs that have a negative-real DC bin.
  - *Wrap error size (new evidence).*
    - var2_bivariate, high band: −0.026, 1.434 and 1.177 at T = 256, 1024 and 4096, against a population value of about 2.46.
    - delayed_noise pair (3,0): arithmetic mean −0.320 against a circular mean of −1.561.
  - *Max saturation (candidate).* `phase_*_max` saturates near π. On iid_gaussian at T=256 the entries lie in [2.960, 3.142], and on var2 in [3.005, 3.117]. The statistic carries little pair-specific information.
- **Proposed checks:**
  - Kernel independent re-implementation.
  - Direction check on delayed_noise and narrowband.
  - Exact 0 on shared_envelope.
  - Exact DC/π accounting identity.
  - Circular against arithmetic mean (qualitative).
  - Max-saturation statistical check.
  - Exact invariance on scaled_offset.
  - Band enumeration.

## spectral.icoh

- **Documented** (§1.5.5, p.14; GitBook `spectral.md:152-173`): "The imaginary part of the complex-valued coherency (referred to as imaginary coherence [59]) ... the coherence exclusively caused by a time delay ... [64]". It is labelled "unsigned".
- **Canonical:** Nolte et al. 2004 (Clin. Neurophysiol. 115:2292) define ImCoh = Im C_ij, which is signed and antisymmetric. Equation numbers are from memory.
- **Implemented:** `connectivity.py:370-373`, `abs(E[S].imag / sqrt(P_i P_j))`. The absolute value is taken per bin, before the band mean or max. There is no norm==0 guard.
- **Deviations:**
  - *Absolute value (pack hypothesis confirmed).* On delayed_noise at T=1024, A[0,1] = A[1,0] = 0.6293, while the signed reference gives +0.6283 and −0.6283.
  - *Order of abs and mean.* Because the absolute value comes first, mean|Im| ≠ |mean Im|. For pair (0,3) (delay 5) over the full band, pyspi gives 0.6315 while |mean Im| is 0.1262.
  - *Null level.* The null level is about 0.25 at every T: 0.262, 0.252, 0.255 and 0.258.
  - *Bias.* On var2 at T=4096, pyspi gives 0.3626 against a population value of 0.2871.
  - *DC bin.* DC is a structural zero, so bands starting at 0 are scaled by (n−1)/n. On var2 at T=256 in the low band, 0.389088 = 0.395168 × 64/65.
  - *Contrast with PLI/wPLI.* REVIEW M3 records that PLI/wPLI are signed while labelled unsigned. icoh is the opposite case: unsigned, and labelled unsigned.
- **Proposed checks:**
  - Kernel independent re-implementation.
  - Exact 0 on shared_envelope.
  - narrowband against |sin Δφ|: the probe gives 0.716, 0.995 and 0.697.
  - Signed-reference direction check.
  - iid null by Monte Carlo.
  - var2 population check.
  - Exact DC identity.
  - Invariance on scaled_offset.
  - Exact symmetry.

## Open questions

1. Are the EEG identifiers' fmin/fmax values Hz/2048 (0.000488 = 1/2048, …, 0.122 = 250/2048)? This is a hypothesis drawn from the `.3g` strings. If it holds, the narrow bands contain 1 to 3 FFT bins, which is narrower than the multitaper half-bandwidth W = 3/n_fft. For patient 004, recorded at 1024 Hz (REVIEW M4), the same normalised bands would map to different Hz ranges. (Added at synthesis: this holds for the historical stored run, computed at the native rate; at HEAD a42ebc1 the workflow resamples to the configured rate before windowing, see "Open questions: answered" 1.)
2. Should `cohmag` be renamed or redocumented as MSC, and should `icoh` keep |Im C| or move to signed Nolte ImCoh? These are definition decisions, related to the M3 decision on PLI/wPLI.
3. Should NW scale with T (fixed bandwidth) or stay at 3? The current choice keeps bias and null levels constant in T and keeps the `max` statistics growing with T.
4. The supplement says DC is excluded by f0 = 4/T and that the grid has 125 bins. Was the published analysis run on a different code path, such as an older spectral_connectivity with explicit frequency arguments? This is unconfirmed.
5. What should a zero-power channel produce? The library's icoh has no norm guard, unlike coherency, so it would give 0/0. The Data z-score already produces NaN for constant channels, so this was not probed.
6. Section and equation numbers for Carter 1987, Nolte 2004, Priestley 1981 and Mardia & Jupp 2000 are from memory and need checking against the papers.

## Checks

Files: `audit/checks/manifest/spectral_coherence.json` (582 runs: 190 cohmag, 201 phase, 191 icoh), `audit/checks/refs/spectral_coherence.py` (comparison venv, no spectral_connectivity import), `audit/checks/probes/spectral_coherence_bands.py` (project venv: custom bands, the 11 EEG identifiers, empty band with `max`, truncated lengths, pyspi's own frequency vector, constant channel). Results: `audit/checks/out/results/spectral_coherence.json`, 549 pass, 33 fail, 0 without reference. Every pyspi run took at most 0.7 s (32 s for all 582); the references, including 400-draw Monte Carlo runs, take about 70 s.

The failures fall into two groups.

- **Thirty planned failures.** These runs compare pyspi with the canonical form so that both forms appear in results.json: 24 `pop`, 3 `circular`, 2 `signed` and 1 `unsquared`.
- **Three chance-level statistical misses.** Each is explained under its unit below.

All check runs use the shared synthetic suite. Nothing here uses the regression goldens, which are snapshots of current output.

How the checks are built (the same for all three units):

- **impl.** An independent re-implementation, built from `_kernels_spectral.multitaper_fft`, `cross_spectral_matrix` and `coherency` with numpy FFT and scipy DPSS. It removes the constant, uses five NW = 3 tapers with concentration > 0.9, sets n_fft with the 5-smooth `next_fast_len` rule, averages over tapers without weights and keeps the first (n_fft+1)//2 bins. Bands are closed and reduced with nanmean or nanmax. For the pipeline paths the reference does its own z-score (ddof=1) and linear detrend. It runs on all six identifiers × 7 datasets at T=1024, on 3 datasets at T = 64, 256 and 4096, and on the pipeline, pipeline-detrend and trending paths. Each run also stores the size of the alternative rules in results.json: a strict band, DC excluded, and the unsquared, circular or signed forms.
- **Analytic checks.**
  - `identical`: shared_envelope x2 ≡ x0.
  - `f0bin`: the single FFT bin at f = 0.25 on narrowband_f0-0.25, selected with a probe band fmin = fmax = 0.25. The analytic value is C_ij(f0) = exp(i(φ_i − φ_j)). The tolerance is 4 × the Monte Carlo RMS deviation of the estimator.
  - `pop`: the population value. For var2 this is S = HΣH^H at the FFT grid. For bivariate_gaussian it is C = ρ.
- **Same-lineage self-consistency checks.** These compare pyspi with pyspi, so results.json labels them `same-lineage`. No verdict or score rests on them; the impl, identical, f0bin and population runs carry the verdicts.
  - `invariance`: pyspi on scaled_offset against pyspi on bivariate_gaussian (same draw, kernel path).
  - `perm`: pyspi on a permuted channel order against the permuted pyspi matrix.
  - `dcpi` (phase only): −A.T of the same pyspi run plus the analytic DC term 2π·[Re S_ij(0) < 0]/n_bins.
- **Monte Carlo checks.** `null` and `mcexp` use 400 fresh draws of the make_data construction, run through the independent estimator. A run passes if |pyspi − MC mean| ≤ 2·se, with se = MC sd × z(1−0.05/(2m))/2. This makes each run a familywise 5 % test over its m judged entries.
- **mne.** An external-package form check with mne-connectivity 0.9.0 (multitaper, 6/T bandwidth, adaptive off). mne weights the tapers by their eigenvalues, so it agrees with pyspi only to about 1e-4. The check is therefore qualitative: pyspi must sit at least 4× closer to the library form than to the canonical form.
- **Not run: nitime.** The record proposed a same-lineage nitime check, but nitime is not installed in the comparison venv. The mne-connectivity check replaces it.

### spectral.cohmag: verdict matches, score 9 (rubric 9: canonical up to the squared-vs-unsquared convention)

- **Score basis.** The score rests on the 120 independent re-implementation runs, so `check_method` is `independent-impl`. The identical-pair and f0-bin analytic runs equal 1 under both |C|² and |C| and cannot separate the forms; the population runs fail by design (fixed-K bias). The squaring is the anchor-9 convention: the supplement says "squared time-domain cross correlation" (supplement.txt:605).
- **Band-level items.** The grid (DC included, 125-bin / f0 = 4/T grid not used), the shared 0.25 bin and the empty-band behaviour below are band-selection deviations, not conventions: excluding DC moves band means by up to 0.036 (band maxima by up to 0.092), and a strict band moves band means by up to 0.031 in the impl runs and 0.16 on the narrow EEG bands (band maxima by up to 0.273). They are owned and scored by `spectral.band_selection` (variant 6; see `notes/spectral_csm.md`). They are confirmed here with this unit's numbers but are not scored again in this unit or counted as its new findings.
- **Exact reproduction.** The impl runs pass 120/120.
  - max_abs is 7.6e-13 on the kernel path, 8.9e-14 on the pipeline path and 1.3e-13 on the detrend path.
  - pyspi's preprocessed input equals the reference z-score and detrend exactly (0.0).
  - The EEG bands agree to 1.0e-13.
  - At T=67 and 97 (n_fft 72 and 100) pyspi agrees to ≤ 1.3e-15. A scipy.fft-rule grid would miss by up to 0.022.
- **Analytic checks.**
  - Identical pair: mean 1.0 exactly, max 1 + 8.9e-16.
  - The f0 bin is within 1.1e-4 of 1 at T=1024 and within 9.2e-6 at T=4096.
- **Self-consistency (same-lineage).** Scale and offset invariance hold to 1.8e-15, and permutation equivariance holds exactly.
- **Squaring, both forms recorded (candidate confirmed, new).**
  - `unsquared:mean-full` on bivariate_gaussian at T=4096 has max_abs 0.212. Pair (0,1) gives pyspi 0.2647 against mean|C| 0.4766, and pair (2,3) gives 0.6715 against 0.8109.
  - mne-connectivity's `coh`² band mean agrees to 7.9e-5, while `coh` itself differs by 0.204 (var2).
  - Across the impl runs, the unsquared form differs by up to 0.250.
- **Fixed-K bias and convergence (candidate confirmed, new; this is finite-sample bias and is not counted as a formula error).**
  - *Null level.* For T = 64, 256, 1024 and 4096, pyspi's pair mean is 0.2295, 0.1983, 0.1964 and 0.1994. The MC values are 0.2040, 0.2011, 0.2005 and 0.2001, and the per-bin theory is 1/K = 0.2.
  - *Null max.* The null max grows with T: 0.6658, 0.7652, 0.7905 and 0.8782, against MC 0.6055, 0.7200, 0.8025 and 0.8610.
  - *var2 full band.* pyspi gives 0.3591, 0.3850 and 0.3648 at T = 256, 1024 and 4096. The population MSC is 0.2465, giving `pop` max_z of 2.83, 7.28 and 12.53. Carter's bias formula predicts 0.3600. `mcexp` passes (max_z 0.60 at T=4096).
  - *bivariate_gaussian at T=4096.* pyspi gives 0.2647 against ρ² = 0.09 (max_z 25.56).
- **Grid (candidate confirmed; the 125-bin and f0 = 4/T part is new to this audit and counted on `spectral.band_selection`, and the DC/Nyquist part is REVIEW M4 and DEPENDENCY_COMPATIBILITY_REVIEW).**
  - pyspi's own frequency vector equals nonneg(fftfreq(n_fft)) exactly: 512 bins at T=1024 (top bin 0.49902), 36 bins at T=67 and 50 at T=97. The documented grid is 125 bins on [4/T, fs/2].
  - Excluding DC would change band means by up to 0.036 (`iid_gaussian|64|impl:mean-low`) and band maxima by up to 0.092 (`impl:max-low`).
- **Shared 0.25 bin (confirmed).**
  - At T=1024 the low band holds 257 bins and the high band 256, and both contain the bin at 0.25.
  - A strict band changes every mean run, by 1.6e-5 to 0.031 over the impl mean runs and up to 0.16 on the narrow EEG bands (`eeg:mean-eeg11`); the 0.273 is a max run (`iid_gaussian|256|impl:max-high`). The strict alternative (`band_mask` with `inclusive=False`, `checks/refs/_kernels_spectral.py:73`) also drops DC on the fmin = 0 bands, so on the low and full bands it mixes the DC effect with the shared-bin effect; only the high band (mean runs up to 0.020) isolates the closed edge.
- **Empty band (confirmed).** `emptymax` with `max` raises ValueError "zero-size array". With `mean`, the 1–4 Hz EEG band at T=256 is all-NaN on both sides.
- **Zero-power channel.** A constant channel, either 0.0 or 3.7, gives an all-NaN row and column (`const0` and `const3.7`).
- **Symmetry.** max|A − A.T| = 0.0 in all 120 impl runs. The cohmag matrix carries no direction information.

### spectral.phase: verdict related-quantity, score 4 (rubric 4: arithmetic mean of wrapped phases)

- **Exact reproduction.** The impl runs reproduce the arithmetic nanmean and nanmax of principal angles, DC included, in 120/120 runs. max_abs is 1.3e-11 on the kernel path and 1.8e-12 on the pipeline path, and the EEG bands agree to 1.5e-12. No bin flipped between +π and −π.
- **Sign convention (confirmed by analytic runs).**
  - The f0 bin gives A[1,0] = +π/4, A[0,2] = +π/2 and A[1,2] = +3π/4, all within 1.7e-3.
  - delayed_noise at T=1024: A[0,1] = +0.7768 in the low band against the analytic arithmetic mean +0.7854, and +1.5169 in the full band against +1.5677 (`delay`, max_z 1.37 and 1.38).
  - `direction` agrees 1.0 on the 6 non-wrapping entries in the low band and the 2 in the full band.
  - So A[i,j] > 0 means channel i leads channel j.
- **DC accounting, exact (candidate confirmed; elaborates M4, not new).**
  - In all 5 `dcpi` runs (same-lineage: the value is built from pyspi's own A.T), A[i,j] + A[j,i] = 2π·[Re S_ij(0) < 0]/n_bins to within 1.1e-16. Every one of the 60 mean impl runs gives |A + A.T| as an exact multiple of 2π/n_bins.
  - The offset comes from numpy.angle returning +π in both orientations, because the DC cross-spectrum has a +0 imaginary part.
- **Wrap error, against the canonical circular mean (candidate confirmed; mechanism M4, size new).**
  - On delayed_noise in the full band (`circular`), A[3,0] is −0.320 against a circular mean of −1.565, and A[2,0] is −0.518 against −1.558 (max_abs 2.879). In the low band max_abs is 2.088.
  - On var2 at T=4096 in the high band, pyspi gives 1.1765, the circular mean of the same estimate is 2.3255 and the population value is 2.4658 (circular 2.4684).
  - The `pop` runs fail in the bands that wrap. Full band: max_z 4.39, 3.93 and 10.41 at T = 256, 1024 and 4096. High band: max_z 4.77, 4.08 and 10.88. The error does not shrink with T.
  - The low band, which does not wrap, matches the population at every T: 0.5984 against 0.642 at T=4096 (max_z 0.91).
  - mne-connectivity agrees with the arithmetic mean to 7.8e-5 on delayed_noise and 0.025 on var2, where eigenvalue weighting moves bins across the branch cut. It sits 2.888 and 0.459 from the circular mean.
- **Max saturation (candidate confirmed, new).** The null max pair mean is 2.9035, 3.1188, 3.1340 and 3.1413 for T = 64 … 4096, against MC 2.9715 … 3.1400. On iid_gaussian at T=256 the entries lie in [2.960, 3.142], on var2 in [3.005, 3.117], and |A + A.T| reaches 6.283. The max statistic converges to π for any broadband pair.
- **Three chance-level statistical misses.**
  - *Null mean-full, iid T=1024.* max_z is 2.02, which is a per-entry z of 2.83, while the pair-average z is −1.26.
  - *mcexp, var2 T=256, full and high bands.* max_z is 2.09 and 2.36. The Monte Carlo puts this one draw at the 0.75 % and 0.5 % quantiles (`ref_mc_quantile_of_pyspi_offdiag` in results.json; a 2000-draw scratch run gave 1.2 % and 0.55 %).
  - *Why these are chance.* pyspi reproduces the same draws exactly in the impl runs. With 57 null and mcexp runs in the family, about three such misses are expected at the 5 % familywise level per run. They describe the data draws and say nothing about the computation.
- **Symmetry.** For the mean statistic the matrix is antisymmetric apart from the DC term above. For the max statistic it is asymmetric.
- **new_finding basis.** new_finding is true because of the max-saturation candidate, which is not in KNOWN_ISSUES. The 125-bin / f0 = 4/T part of the grid candidate is also new to this audit, but, like every band-level item, it is counted on `spectral.band_selection` (grid bullet above) and not as a new finding of this unit. The DC-accounting and wrap-size candidates elaborate REVIEW M4 and are not counted.
- **definition_under_review: true** (set in Workflow 4b-2; it was false). REVIEW M4 ends with "Decision needed: preserve the historical arithmetic band summary, define a branch/endpoint convention, or introduce a circular/complex phase summary with a new identifier", and EEG_OPTIMIZATION_NOTES.md:169 lists phase aggregation/endpoints as outstanding. The checks record both forms: the arithmetic mean in the impl runs and the circular mean in the circular runs. This is the same kind of pending maintainer decision for which cohmag and icoh carry true (open question 2 below).

### spectral.icoh: verdict variant, score 7 (rubric 7: matches the supplement's unsigned variant but differs from canonical in a value-changing way)

The score was 8 before the review. Anchor 8 needs a monotone relationship to the canonical quantity. Taking the absolute value per bin is monotone in magnitude, but the SPI is the band mean or max taken after the absolute value. That band quantity has no monotone relation to the canonical band mean of signed Im C: pair (0,3) gives 0.6315 against |mean Im C| 0.1262, and zero-lag mixing gives 0.24 to 0.26 where the population value is 0. The supplement keywords for icoh are "undirected, unsigned", so the library form matches the supplement variant. That is anchor 7.


- **Exact reproduction.** The impl runs reproduce |Im C| per bin, followed by the band statistic, in 120/120 runs. max_abs is 7.4e-13, and the EEG bands agree to 7.3e-14. |Im C(DC)| is exactly 0.0 in all 120 runs. The f0 bin is within 1.1e-3 of |sin Δφ| (0.7071, 1.0, 0.7071). The identical pair gives 0.0.
- **Absolute value, both forms recorded (candidate confirmed, new).**
  - `signed:mean-full` has max_abs 1.2575. pyspi gives A[0,1] = A[1,0] = 0.6293, while Nolte's signed Im C gives +0.6283 and −0.6283.
  - mne-connectivity's |imcoh| agrees to 7.1e-5 on var2 and 2.7e-5 on delayed_noise, while the signed `imcoh` differs by 0.700 and 1.258.
- **Absolute value before the band mean (confirmed, new).** For pair (0,3) over the full band, pyspi gives 0.6315 while the signed band mean is 0.1262.
- **Null level and instantaneous mixing (confirmed, new).**
  - On iid data, pyspi gives 0.2623, 0.2517, 0.2554 and 0.2577 for T = 64 … 4096, against MC 0.2456 … 0.2586 and a population value of 0.
  - On bivariate_gaussian, a zero-lag mixture where Nolte's population Im C = 0, pyspi gives 0.2437 at T=4096 (max_z 27.49).
  - On var2 in the full band at T=4096, pyspi gives 0.3626 against a population value of 0.2871 (max_z 8.82). `mcexp` passes (max_z 0.02).
- **DC structural zero (confirmed).** Bands that start at 0 equal (n−1)/n times the DC-free mean. The DC-free alternative differs by up to 0.039.
- **Grid, shared bin and empty band.** These behave as for cohmag.
- **Zero-power channel.** It gives a NaN row and column, so the library's missing norm guard produces 0/0 = NaN.
- **Symmetry.** max|A − A.T| = 0.0. The lead/lag sign is discarded.

### Trace-stage fields corrected or completed in this stage

- **Calculator citation.** The `implementation_trace` entry for `pyspi/calculator.py` now cites HEAD a1f20c7: `compute` is at lines 295-345, `multivariate` at 329, the diagonal is set to NaN at 332, and the exception path to NaN is at 336-339. The old citation was 292-303.
- **Phase references.** The `canonical_ref` disclaimer for phase is narrowed to the Priestley chapter, the Carter 1987 sections and the Mardia & Jupp section number. The Bastos & Schoffelen quotations were checked in the cohmag record. The cohmag disclaimer was already limited to Carter 1987, and the icoh disclaimer to the Nolte 2004 equation numbers.
- **icoh `units_and_range`.** It now records the zero-power behaviour, which has been probed.
- **Planned-check labels.** The planned invariance and self-consistency checks in all three records, and the phase DC-accounting check, were labelled `analytic`. They compare pyspi with pyspi, so they are now labelled `same-lineage`, matching results.json.
- **YAML anchors.** The `&id001` / `*id001` anchors named as a remaining trace issue were already absent when this stage began. The three records are now written with an alias-free dumper.

## Open questions: answered

1. **EEG band edges.** Yes. `pyspi/workflow/retained185.yaml:227-247` (CoherencePhase; CoherenceMagnitude :257-277, ImaginaryCoherence :287-307) gives exact Hz/2048 values, from 0.00048828125 = 1/2048 to 0.1220703125 = 250/2048, so fmin is 1 Hz and not 0.
   - The EEG workflow uses 0.5 s windows (EEG_OPTIMIZATION_NOTES.md:23), so T = 1024 at 2048 Hz, the bin spacing is 2 Hz and the multitaper half-bandwidth is W = 6 Hz.
   - Run `spectral.*|delayed_noise|1024|pipeline-detrend|eeg:mean-eeg11` shows that the closed bands hold 2 bins for 1-4 Hz (2 and 4 Hz), 3 for 4-8 Hz and 3 for 8-12 Hz.
   - Neighbouring bands share their edge bins (4, 8, 12, 30, 70 and 150 Hz). A strict rule would leave 1, 1 and 1 bins.
   - Because W is wider than these bands, the delta, theta and alpha values are not band-limited.
   - At T = 256 the 1–4 Hz band is empty, giving all-NaN with `mean`.
   - For a 1024 Hz recording, the metric bands are not rewritten from the EDF rate (retained185.yaml:4). At HEAD a42ebc1 the workflow resamples every native-rate group whose rate differs from the configured `sampling_rate_hz` to that rate (`mne.filter.resample(..., method="polyphase")`, `pyspi/workflow/_prepare.py:104-106`), after the native-rate notch and before the bandpass, rereferencing and windowing (EEG_OPTIMIZATION_NOTES.md:208-209, 419-420). New runs with the example configuration's 2048 Hz (`examples/workflow/preflight.yaml:8`) therefore get 1024-sample windows and the Hz mapping above for 1024 Hz and 512 Hz recordings too. Two points remain open. The mapping assumes `sampling_rate_hz` is 2048, and nothing in the workflow ties it to 2048 (`pyspi/workflow/_config.py:243-248` requires a positive rate above twice the bandpass upper edge), so a run configured at another rate shifts every band in Hz. And a recording computed at its native 1024 Hz, as in the historical stored run for patient 004 (512-sample half-second windows, `docs/storage_comparison/REVIEW.md:164`), has every band at half the Hz values.
2. **Relabel cohmag as MSC, or move icoh to signed ImCoh?** This is still a definition decision for the maintainers, linked to REVIEW M3. The checks record both forms for each unit. For cohmag, the unsquared |C| differs by up to 0.25. For icoh, the signed form differs by up to 1.35, and the absolute value removes all direction information and gives a null level of 0.26. Both records set `definition_under_review: true`.
3. **NW fixed at 3?** The checks confirm the consequence. The null levels are constant in T: cohmag 0.20 and icoh 0.26 at every T from 64 to 4096. The max statistics grow with T: cohmag 0.67 → 0.88. The var2 bias does not shrink: cohmag 0.365 against 0.2465 at T=4096. Whether to scale NW with T is a design decision. A fixed-bandwidth NW ∝ T, or a debiased MSC, would make the estimates consistent.
4. **125 bins and f0 = 4/T in the supplement?** Still unconfirmed. At HEAD with spectral-connectivity 1.1.2, the grid runs show the FFT grid, with (n_fft+1)//2 bins and DC included, and no T in the suite yields 125 bins. Settling this needs the code or library version behind the published analysis.
5. **Zero-power channel?** Answered. On the kernel path, which skips the z-score that would turn the channel into NaN anyway, a constant channel gives an all-NaN row and column for all three units, and the other pairs are unchanged (runs `const0` and `const3.7`). icoh gets 0/0 = NaN because the library has no guard. cohmag and phase get it from the coherency guard.
6. **Carter 1987, Nolte 2004, Priestley 1981 and Mardia & Jupp 2000 section and equation numbers?** Not re-checked in this stage, because the papers were not opened. The records keep the narrowed "from memory" disclaimers. The equations that were checked numerically are coherency, MSC, Im C and the circular mean. Checking them against the papers would settle this.
7. **Briefing line ranges.** Confirmed. The headings in supplement.txt are at §1.5 line 570, §1.5.1 line 601, §1.5.2 line 616 and §1.5.5 line 670, which the records cite.
8. **Adversarial Fable review of the trace records.** Not run in this stage either. It belongs at the workflow level.

## Optimisation

Sources: walls from `audit/profiles/summary.json` (fresh / family-warm; quiet machine; 118x1024 walls are single repeats), frame shares from cProfile. The official profiles at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`) are dominated by the P-independent DPSS cost for every multivariate unit, so for ranking at 118 channels this group added cProfile runs at 118x1024 (32x1024 for sgc nonparametric): `audit/profiles/micro/prof_<unit>__<shape>.pstats`, written by `audit/profiles/micro/prof_spectral.py`, indexed with walls, peak RSS and thread settings in `audit/profiles/micro/prof_spectral__index.json` (OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1; no numba or JIDT code on these paths). Those runs overlapped other Workflow 3 groups; the memory-bound multitaper units varied 2-7x in wall between runs (cohmag: 6.4 s on a first run, 0.93 s on the kept run, 2.42 s in summary.json), so only shares are quoted from them. Percentages are of the profiled `harness.compute_spi` frame. Micro-benchmarks (the group's two): `audit/profiles/micro/opp_spectral_csm-1.{py,json}` and `audit/profiles/micro/opp_spectral_csm-shared-csm.{py,json}`. A candidate that reproduces the current output "matches the baseline"; that says nothing about whether either is mathematically right. Nothing in `pyspi/` was changed.

| unit | 8x1024 fresh / fw (s) | 32x1024 fresh / fw (s) | 118x1024 fresh / fw (s), mode | peak RSS 32 / 118 fresh (MB) |
|---|---|---|---|---|
| spectral.cohmag | 0.0193 / 0.0204 | 0.0772 / 0.0755 | 2.420 / 2.739, measured | 504 / 1772 |
| spectral.phase | 0.0201 / 0.0408 | 0.0866 / 0.1687 | 2.966 / 4.970, measured | 449 / 1769 |
| spectral.icoh | 0.0191 / 0.0199 | 0.0771 / 0.0734 | 1.734 / 2.491, measured | 488 / 1658 |

Family-warm walls sum the unit's six identifiers; phase is first in the family order and pays the cache build twice (2.29 s and 2.63 s at 118) because of the first-insert string key (`spectral.py:81`); every other band of a unit costs 0.007-0.024 s.

**Hotspots at 118x1024** (`audit/profiles/micro/prof_spectral.<unit>__118x1024.pstats`):
- cohmag: `connectivity.py:coherency` 85.8% (self 24.8%), `_expectation_cross_spectral_matrix` 59.2%, `_complex_inner_product` 38.5%, taper-mean reduce 17.2%, `dpss_windows` 1.9%, `nanmean` 3.2%.
- phase: coherency 74.1%, `_complex_inner_product` 33.4%, `numpy angle` 17.0%, reduce 15.8%, `dpss_windows` 1.6%.
- icoh: `_expectation_cross_spectral_matrix` 66.1% (`_complex_inner_product` 43.8%, reduce 18.3%), `imaginary_coherence` self 23.0%, `dpss_windows` 2.1%.
At 8x1024 `transforms.py:dpss_windows` is 74.0% (cohmag), 71.3% (phase) and 77.8% (icoh) of the official profiles. opaque_share from the 8x1024 profile JSONs: 0.111, 0.098, 0.107 (lower bounds: numpy ufunc work such as matmul, divide and abs is hidden in the calling Python frame's self time; the '~' builtin share at 118 is 0.20-0.23).

**Complexity.** O(K P T log T) FFT + O(K n_fft P^2) taper-level CSM + O(n_fft P^2) normalisation (K = 5, n_fft = T) plus the P-independent DPSS. The P^2 CSM term dominates at 118.

**Opportunities.**
- `spectral_coherence-1` (cache-sharing, S; added in Workflow 4c): store the first cache insert under `self.key` instead of the measure string (`spectral.py:81`, the redundant recompute in tests/README 'worth knowing'), so the second identifier of the first spectral measure on a Data reads the cache. On phase, the first spectral class of an EEG run, 2-2.4x: 4.970 -> 2.341 s on the family-warm increments (2.12x, the 2x lower bound), and 1.88-2.46x same run at 118x1024 (1.93-2.01x at 8 and 32x1024) in `opp_spectral_csm-shared-csm.json` key `paired`, bit-identical. This one-line fix is the whole gain of `spectral_csm-shared-csm` on phase in the EEG order, so the phase record ranks shared-csm after it (`depends_on`) with `residual_speedup` 1x.
- `spectral_csm-shared-csm`: cohmag 6-7.2x, phase 2-2.2x (first consumer; 1.92-2.19x same run in the `paired` run), icoh 2.7-3x (the records' ranges: the profiled-call Amdahl figures below taken against the configured sum, the ranking basis, which also holds 5 band reads a later consumer still pays; cohmag 2.739/(2.739 - 0.858 x 2.663) = 6.03x and 7.2x with the shared Fourier transform, icoh 2.79x and 3.03x). The unit that builds the shared coherency keeps its cost (bar the first-insert recompute, which alone halves phase's family-warm cost: 2.29 + 2.63 s -> 2.29 s, ~2.1x), a later consumer drops the coherency (85.8% / 74.1%) or expectation (66.1%) share (Amdahl on the profiled call 7.0x / 3.9x / 2.9x; cohmag 8.8x and icoh 3.2x when the shared Fourier transform is counted too). In the EEG order CoherencePhase runs first (`pyspi/workflow/retained185.yaml:218`), so phase is the first consumer and its record's lower bound is the 2x end, while cohmag and icoh are later consumers and their records' lower bounds are the later-consumer figures, the ones the ranking uses (README 'Optimisation columns', rules 1 and 4); a run in which cohmag or icoh builds the shared arrays itself gains about 1x. The two phase cases are exclusive; the later-consumer 3.9x cannot occur in the EEG order, so phase's range is the first-consumer 2-2.2x (an earlier 2-4x and 2-8x combined the cases), and on phase the ranking credits the duplicate removal to `spectral_coherence-1`. Basis: `opp_spectral_csm-shared-csm.json` (cohmag and phase included, bit-identical); icoh is not in the micro-benchmark, so its record gives a profile basis, not a microbench one (unverified-bit-identical: same expectation and power arrays). icoh reads the taper-averaged CSM and power directly and never calls coherency(), so it gains only from the CSM and power memo; the micro-benchmark's candidate memoises coherency() alone, so beyond the shared Multitaper build (from_multitaper 3.0% of icoh's 118x1024 profile) it would save icoh nothing.
- `spectral_csm-1`: cohmag 2-4x (re-run, min of 3 at 118x1024: 1.95x sliced, 3.83x gemm; first single-repeat run 2.27x / 4.31x; within-tolerance, max_abs 1.4-1.7e-16), phase 1.7-2.7x (1.71x sliced / 2.71x gemm on today's two-build family-warm cost, by Amdahl on phase's coherency share 74.1%) and icoh 2-4x by the same path, as standalone figures on today's cost. The ranking stacks it after `spectral_csm-shared-csm` (`depends_on`, README ranking rule 4) with `residual_speedup` 1.6-2.6x on phase (its residual is the configured sum over `spectral_coherence-1`'s 2x and shared-csm's 1x) and 1x on cohmag and icoh: CoherencePhase runs first (`pyspi/workflow/retained185.yaml:218`), so cohmag and icoh are later consumers of the one shared build and this avenue's saving on that build is credited on phase alone (about 559 s per patient in an in-memory build_csv run of 2026-10-15: 1,491 -> 932 s). phase and icoh were not run separately, so their records give a profile basis with the cohmag figure carried over (not tagged microbench) and their risk class is unverified-within-tolerance; for phase a 1-ulp change of a coherency with Im near 0 and Re < 0 can flip angle() by 2*pi before the band mean, which a diff would have to rule out.

**EEG per-window cost at 118 channels** (11 EEG identifiers each, one cache build per unit, bands are reductions of the cached array): phase 2.29 + 10 x 0.014 = 2.4 s (5.0 s when phase is the first spectral SPI on the Data), cohmag 2.66 + 10 x 0.015 = 2.8 s, icoh 2.42 + 10 x 0.015 = 2.6 s; about 7.8 s for the three. With `spectral_csm-shared-csm` the three share one coherency (0.77 s in `opp_spectral_csm-shared-csm.json` candidate_segments_s, 2.4-2.7 s at summary.json scale) plus angle/abs/imag and 33 band reductions: about 3 s; with `spectral_csm-1` on top, about 1 s (a rough estimate at the gemm end; the ranking credits `spectral_csm-1` only once, on phase, at its 1.7x lower bound).

**Refuted seeds.** DPSS cache (0.4-2.4% at 118); the coherency's repeated `_power` passes (1.6-1.9%).

**Scaling.** summary.json fits P exponents of 0.9-1.4 between 8 and 32 channels (the fixed ~20 ms DPSS dominates there) but the local exponent between the measured 32x1024 and 118x1024 walls is 2.4-3.4 for every multivariate multitaper unit (cohmag 2.64, icoh 2.39, phase 2.71, pli 2.94, dspli 2.89, wpli 2.89, dswpli 2.92, plv 3.35, ppc 3.41, psi 2.61; gd 2.26). Above 2 this reflects memory-bound GB-sized temporaries, not extra arithmetic; the 118 walls are measurements, but extrapolation to 249 channels from the fitted exponents would understate the cost.
