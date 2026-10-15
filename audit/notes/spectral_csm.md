# spectral_csm family: define, trace and check

Units: `spectral.multitaper_csm` and `spectral.band_selection`. Both are infrastructure units with no identifiers of their own. Every multitaper spectral SPI in `pyspi/statistics/spectral.py` passes through them: cohmag, phase, icoh, plv, pli, wpli, dspli, dswpli, ppc, psi, gd, dcoh, pdcoh, gpdcoh, dtf, ddtf and the nonparametric sgc.

- pyspi code: `pyspi/statistics/spectral.py` (classes `NonparametricSpectral`, `NonparametricSpectralMultivariate`, `NonparametricSpectralBivariate`) and `pyspi/_spectral_compat.py`.
- Delegate: `spectral_connectivity==1.1.2` (`.venv/lib/python3.12/site-packages/spectral_connectivity/`; all third-party line numbers below are from this checkout), running on scipy 1.17.1 and numpy 2.5.4.
- Probe scripts: `spectral_csm/p1_multitaper.py` to `p9_quarter.py` in the session scratchpad.
- Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.
- The regression goldens are snapshots of current output. Nothing here treats them as a reference.
- The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.

## Documented definition (shared)

Supplement §1.5, p.12-13 (pack lines 582-613; the heading is at supplement.txt line 570). The GitBook page `spectral.md:14-20` has the same text.

> "Unless otherwise stated, the frequency-domain (i.e., Fourier-based) measures are computed using the Spectral Connectivity Toolbox [56]. Each measure is based on the cross- and power-spectral densities at a given frequency and sampling rate, which is estimated via the multitaper technique." ... "we assume a sampling frequency fs = 1 throughout ... We use 125 uniformly sampled bins across the entire frequency range, f ∈ [f0, fs/2], where fs/2 is the Nyquist frequency and f0 = 4/T is chosen as the minimal frequency for computational reasons. In order to obtain SPIs, we take the mean (denoted by modifier mean) and the maximum (max) values of each measure over three ranges ... fmin-0_fmax-0-5: f ∈ [0, fs/2]; fmin-0_fmax-0-25: f ∈ [0, fs/4]; fmin-0-25_fmax-0-5: f ∈ [fs/4, fs/2]."

Reference [56] is spectral_connectivity v0.2.5.dev0 (supplement.txt:2135-2136). The version actually executed is 1.1.2. The documentation does not state NW, the number of tapers, the windowing, the detrending or the FFT length.

## Unit `spectral.multitaper_csm`

### Canonical definition

The canonical form is Thomson's (1982, Proc. IEEE 70:1055-1096) multitaper estimator in its simple, equal-weight form: section III (p.1060-1061) forms the estimate (3.7) with unit weights on the eigenspectra S_k(f) = |y_k(f)|² (eq. 3.8), and section V (p.1065) adds the adaptive weights that the library does not use. Percival & Walden (1993) treat it in ch. 7, Multitaper Spectral Estimation (pp.331-377), and Walden (2000, Biometrika 87:767-788) writes multivariate cross-spectral matrix estimators as weighted averages of orthogonally tapered cross-periodograms. Each taper k gives a tapered Fourier transform

J_{k,i}(f) = √Δt Σ_t h_k(t)(x_i(t) − x̄_i) e^{−i2πftΔt},

where h_k is the k-th unit-energy DPSS (Slepian 1978) with half-bandwidth W = NW/T. The spectral matrix is the taper average

S_ij(f) = K⁻¹ Σ_k J_{k,i}(f) J*_{k,j}(f),

which is two-sided and Hermitian. Thomson (p.1061) states that the estimate "has about 2K degrees of freedom", so Ŝ_ii/S_ii is approximately χ²_{2K}/(2K). Mitra & Pesaran (1999) describe the same estimator as standard neuroscience practice: section 4.1.5, eq. 6-7, gives the equal-weight taper average and says to "take the leading 2NW − 1 DPSS as data tapers"; section 4.3, eq. 25-26, builds coherence from taper sums over a single pair of series.

Sources checked: Thomson against a scanned copy with an OCR text layer (equation images are not readable in it); Percival & Walden chapter and section titles and pages against the authors' table of contents only (section contents not checked); Walden 2000 citation and abstract only (full text not obtained, so no section or equation is cited); Mitra & Pesaran against the arXiv preprint q-bio/0309028v1, not the journal copy; Slepian 1978 citation details only.

### What is implemented (trace)

1. **Preprocessing.** `pyspi/data.py:184-195`: optional linear detrend, then z-scoring with ddof=1. `to_numpy(squeeze=True)` (`data.py:108-118`) gives a (P, T) array, which is transposed to (T, P).
2. **pyspi entry points.**
   - Multivariate path: `spectral.py:64-83` passes all channels to `sc.Multitaper(z, sampling_frequency=fs)` and then `sc.Connectivity.from_multitaper(m)`, and calls the measure. If that call raises TypeError, `_get_statistic` is used instead (`:72-75`).
   - Bivariate path for the directed SPIs: `spectral.py:106-126` builds the same objects on the two columns [i, j] only.
   - Caching: results are cached on the `Data` object by key (`:79`). The first insert is keyed by the plain measure string (`:81`), which the tests README already documents.
3. **Multitaper defaults** (`spectral_connectivity/transforms.py:74-102`):
   - NW = 3, and `n_tapers = floor(2NW − 1) = 5` (`:154-164`).
   - `is_low_bias=True` keeps only tapers with concentration > 0.9 (`:668-673`). The probe found the 5th taper at 0.946 for every T, so all 5 are kept.
   - The whole series is one window (`:166-178`, `:192-205`).
   - `detrend_type="constant"` removes the window mean (`:701-748`), which does nothing after z-scoring.
   - `n_fft = scipy.fftpack.next_fast_len(T)` (`:180-185`, imported at `:29`). This equals T for the manifest sizes 64, 256, 1024 and 4096. Other lengths are zero-padded to a 5-smooth length: T=1009 becomes 1024 and T=63 becomes 64.
4. **Tapers.** `dpss_windows` (`:469-543`) builds the Percival-Walden tridiagonal matrix, finds its top eigenvalues with `scipy.linalg.eigvals_banded`, and gets the eigenvectors by inverse iteration (`:575-629`). Signs are then fixed (`:632-657`). The tapers have unit energy and are multiplied by √fs (`:357-387`).
5. **FFT.** `_multitaper_fft` (`:335-354`) computes `fft(x·h, n=n_fft)/fs`, so J = √Δt·FFT(h x). This is two-sided PSD scaling; the probe confirmed that power scales as 1/fs. The result has shape (1 window, 1 trial, 5 tapers, n_fft, P).
6. **CSM.** `connectivity.py:195-208` and `:1187-1190` compute, per taper, `matmul(X, conj(X)ᵀ)`, so S[i,j] = X_i·conj(X_j). The expectation type "trials_tapers" (`:45-53`) takes an equal-weight mean over the trial and taper axes. With one trial, that is a mean over the 5 tapers, so `n_observations` = 5 (`:276-287`).
7. **Frequencies.** `_non_negative_frequencies` (`:73-89`) keeps indices 0 … (n_fft+1)//2 − 1, i.e. f_k = k/n_fft. DC is included, and the Nyquist bin (fftfreq −0.5) is dropped for even n_fft.

### Agreement with the canonical estimator (probes, not the check stage)

- **CSM.** The two-sided CSM from the library matches a from-scratch version (scipy dpss + numpy FFT) to 2.6e-14 at T=256.
- **Tapers.** The library tapers match the top eigenvectors of the dense Slepian sinc kernel (numpy `eigh`) to 1-2e-12 at T = 64, 256 and 1024.
- **Parseval.** The mean over all two-sided bins equals Σ_t K⁻¹Σ_k h_k(t)² x_c(t)², to 1.2e-15.
- **Hermitian symmetry.** The residual is 0.0.
- **DC.** Im S(0) is exactly 0.
- **VAR(2) with known spectral matrix** (`var2_bivariate`): the mean ratio Ŝ_ii/S_ii is 0.97-0.99 and the coverage of the χ²₁₀/10 central 95% interval is 0.95-0.96 at T=1024 and 4096. The mean coherence is positively biased (0.38 against a true 0.25). That is the expected small-K bias of MSC, not a defect.

### Direction convention

The library's convention is S[i,j] = E_k[X_i conj X_j]. On `delayed_noise` at T=1024 I fitted the slope of the unwrapped phase of S[0,j] and divided it by 2π. For delays of 1, 3 and 5 the results are +1.019, +2.986 and +5.034. So a positive phase in S[i,j] means the row process i leads the column process j.

The multivariate A[i,j] is built from S[i,j] (`spectral.py:91`). The bivariate path reads entry [0,1] of the [i,j] pair (`spectral.py:134`), which has the same orientation. Multivariate and bivariate CSM entries are bit-identical.

### Deviations

- **From the documentation.**
  - The grid is k/n_fft from DC to 0.5 − 1/n_fft: 32, 128, 512 and 2048 bins at the manifest T. It does not have "125 bins from f0 = 4/T". *Hypothesis:* the 125 bins describe the wavelet path (`wavelet.py:54`, `np.linspace(0.2, 0.5, 125)`).
  - The Nyquist bin is never available, although the documented ranges are closed at fs/2 (REVIEW M4, already documented).
  - The supplement cites toolbox v0.2.5.dev0, but 1.1.2 is executed.
- **From the canonical estimator.** These are variants, not errors:
  - There is no adaptive weighting, and the 0.946-concentration taper is included.
  - There is a single window, so every downstream "expectation" is an average over 5 tapers. PLV, PPC, PLI, wPLI and the debiased versions are canonically averages over trials or epochs, and the debiasing in `spectral_connectivity/connectivity.py:621-624` and `:690-695` treats the tapers as independent observations. This is a candidate for the spectral_phase_sync family.
  - The PSD scaling is two-sided, which cancels in every normalised measure.
  - S(0) is real and is estimated after mean removal. Together with the inclusion of DC (M4), this adds an exact-zero bin to icoh, pli and wpli in every fmin=0 band. That last part is a candidate.
  - If T is not 5-smooth, zero-padding changes the grid.
- **Efficiency** (the pack hypothesis is confirmed; this is optimisation input, not a correctness finding).
  - Every distinct measure rebuilds the Multitaper, the FFT and the full two-sided CSM. On one `Data` object a spy counted:
    - cohmag: 1 + 1 calls (the documented first-insert recompute);
    - icoh: 1;
    - each PSI band: 1;
    - GD: 1;
    - directed dcoh: 20, one per ordered pair at P=5.
  - PSI's failing default-argument call computes `self.coherency()` before it raises.

### Proposed checks

The record lists the full designs.

1. **From-scratch CSM** (independent-impl, tolerance ≤ 1e-12). Tapers from a dense eigendecomposition of the sinc kernel, with scipy dpss as a secondary source. scipy's dpss is a separate code path, but its derivation is related to the library's, as noted in the record.
2. **Parseval, Hermitian and real-DC identities** (analytic, exact).
3. **VAR S_true** (statistical): χ²₁₀/10 coverage and phase error.
4. **delayed_noise phase slope** (direction). This establishes what A[i,j] means for every spectral SPI.
5. **Pipeline cohmag full band against the from-scratch CSM on z-scored data**, plus multivariate-vs-bivariate identity.
6. **Grid and n_fft rule** for T ∈ {63, 64, 255, 256, 1000, 1009, 1024} (exact), plus taper concentrations.
7. **trending/scaled_offset, pipeline vs pipeline-detrend** (qualitative).

## Unit `spectral.band_selection`

### Canonical definition

No canonical source fixes whether band edges are inclusive. The documentation uses closed intervals. A minimal consistency requirement is that one (fmin, fmax) label means one bin set across all SPIs. For PSI, the canonical rule is Nolte et al. 2008 (PRL 100:234101) eq. 3: adjacent frequency pairs at the native resolution inside the band. The equation number and form were checked against arXiv:0712.2352v1 p.2, not the PRL journal copy: Ψ̃_ij = Im Σ_{f∈F} C*_ij(f) C_ij(f + δf), where δf "is the frequency resolution".

### What is implemented (trace)

| Path | Rule | Location |
|---|---|---|
| Multivariate (cohmag, phase, icoh, plv, pli, wpli, dspli, dswpli, ppc) | `freq >= fmin & freq <= fmax` (inclusive), then `np.nanmean`/`np.nanmax` over frequency | `spectral.py:88-91` |
| Nonparametric SGC | inclusive, with `fmin <= 0` overridden to 1e-5 (DC excluded) | `spectral.py:343-345, 421-423` |
| Bivariate directed (dcoh, pdcoh, gpdcoh, dtf, ddtf) | `freq > fmin & freq < fmax` (strict) | `spectral.py:132-134` |
| PSI | library `_bandpass` (strict, `connectivity.py:1329-1331`), then every `ceil(n_fft·(fmax−fmin)/50)`-th bin, then all-pairs sum (M5) | `pyspi/statistics/spectral.py:297-301`; `connectivity.py:1060-1113` |
| GD | strict `_bandpass`, then the largest significant cluster sub-sampled by the same step, with at least 3 points or the result is NaN | `spectral.py:313-317`; `connectivity.py:902-987, 1408-1482` |

For PSI and GD, pyspi's own `freq_id` is computed but never used. PSI reaches the result through the IndexError fallback and GD through the TypeError fallback (`spectral.py:92-100`). The band reaches the library only because the default-argument call raises TypeError: at `connectivity.py:1329` for PSI and at `:1357` for GD. A spy showed `kwargs {}` first, then `{frequencies_of_interest: [fmin, fmax], frequency_resolution: (fmax−fmin)/50}`. With the pinned 1.1.2 the band is therefore forwarded; `DEPENDENCY_COMPATIBILITY_REVIEW.md:84` reports the failure for the newer release.

Bin counts (inclusive / strict / PSI bins used):

| T | full [0, 0.5] | low [0, 0.25] | high [0.25, 0.5] |
|---|---|---|---|
| 64 | 32 / 31 / 31 | 17 / 15 / 15 | 16 / 15 / 15 |
| 256 | 128 / 127 / 43 | 65 / 63 / 32 | 64 / 63 / 32 |
| 1024 | 512 / 511 / 47 | 257 / 255 / 43 | 256 / 255 / 43 |
| 4096 | 2048 / 2047 / 50 | 1025 / 1023 / 49 | 1024 / 1023 / 49 |

### Deviations and candidates

- **The edge rule depends on the path** (pack hypothesis confirmed). The same identifier suffix averages different bin sets for multivariate and bivariate/PSI/GD SPIs.
- **The f = 0.25 bin is double-counted** by the inclusive half bands and excluded from both strict half bands. Probe on `narrowband_f0-0.25`, T=1024: 257·mean_low + 256·mean_high − 512·mean_full = 0.99990014, which is exactly MSC(0.25).
- **Narrow EEG bands.** The retained EEG identifiers use band edges such as 0.000488-0.00195, but the exact float values are not recorded. Under the *hypothesis* that the edges are exact multiples of 1/1024 (0.5, 2, 4, 6, 15, 35, 75, 125 divided by 1024):
  - at T=1024 the bands (0.5-2), (2-4) and (4-6) get 2/3/3 inclusive bins and 1/1/1 strict bins (pack hypothesis confirmed);
  - at T=512 they get 1/2/2 inclusive bins and 0/0/0 strict bins;
  - with the rounded identifier values, T=1024 gives 1/3/2 bins under both rules.
- **What happens with so few bins** (probe on 3 iid channels, exact edges):
  - PSI raises IndexError at T=1024, because one strict bin leaves no frequency pairs (`connectivity.py:1110-1111`).
  - GD is all NaN at T=1024 (below `min_group_size` 3), and at T=512 it raises ValueError.
  - dcoh is NaN at T=512.
  - cohmag returns single-bin values.
  - `Calculator` turns every exception into an all-NaN table (`calculator.py:336-339` at HEAD a1f20c7, `self._table[spi] = np.nan` at :337). This may explain the all-NaN GD cases in saved data (REVIEW M6), but that is a *hypothesis*.
- **PSI sub-sampling.** PSI uses at most about 50 bins, spaced by band width / 50 rather than by the taper bandwidth 2W = 6/T. This is undocumented.
- **NaN-ignoring aggregation.** NaN bins are dropped silently. An empty selection gives NaN for the mean but raises ValueError for the max, so the max turns the whole table to NaN.
- **Already documented:** M4 (DC/Nyquist), M5 (all-pairs PSI), M6 (GD NaN), and DEPENDENCY_COMPATIBILITY_REVIEW.md:74-84.

### Proposed checks

1. **Analytic bin-index sets** for every config band and EEG band (under both edge hypotheses) at T ∈ {64, 256, 512, 1024, 4096}, compared exactly with the indices pyspi uses and with the PSI pair count.
2. **From-scratch cohmag/icoh** over inclusive vs strict sets, compared with pyspi. Expected: tolerance 1e-12 against the inclusive set and a clear difference from the strict set.
3. **Same-lineage, bivariate path:** per-frequency dcoh/dtf from the library, aggregated over strict vs inclusive sets. Expected: pyspi equals the strict set.
4. **narrowband_f0-0.25 double-count identity.**
5. **Narrow EEG bands** at T = 1024 and 512: the qualitative outcomes listed above, plus which EEG identifiers degenerate.
6. **Spy on PSI/GD kwargs** to confirm band forwarding.

## Open questions

- What are the exact float band edges of the retained EEG identifiers, and what epoch length does the EEG workflow use? Bin counts, and whether PSI/GD/dcoh degenerate, depend on both. The identifiers only carry 3 significant digits.
- Does the supplement's "125 bins, f0 = 4/T" text describe the wavelet path (125 frequencies in `wavelet.py:54`)? Should the multitaper documentation be corrected, or the code changed?
- Should the bivariate path be made inclusive (or the multivariate path strict) so that one band label denotes one bin set? Either change would alter stored values.
- Is averaging over 5 tapers of one window an acceptable stand-in for the trial expectation in PLV/PPC/PLI/wPLI and their debiased forms? This belongs to the spectral_phase_sync family, but its root is the CSM expectation domain here.
- Literature locations were checked after review (see the canonical-definition sections). Still open: the Percival & Walden section contents, the full text of Walden 2000, the full text of Slepian 1978, and the journal copies of Nolte 2008 and Mitra & Pesaran 1999 (only preprints were read).

## Checks

Check stage, branch `optimizations` at HEAD a1f20c7. Files: `audit/checks/manifest/spectral_csm.json` (172 runs), `audit/checks/refs/spectral_csm.py` (comparison venv: numpy 2.5.4, scipy 1.18.1, mne_connectivity 0.9.0), `audit/checks/probes/spectral_csm_lib.py` and `audit/checks/probes/spectral_csm_bands.py` (project venv, spectral_connectivity 1.1.2). Results: `audit/checks/out/results/spectral_csm.json`, **172 pass, 0 fail, 0 without reference**. The regression goldens were not used.

### `spectral.multitaper_csm`

The probes build exactly the objects pyspi builds (`sc.Multitaper(z, sampling_frequency=1)`, `sc.Connectivity.from_multitaper`) and expose `_expectation_cross_spectral_matrix()`, `coherency()`, `frequencies`, the tapers and the taper concentrations. The probe applies the run's path (`kernel` = raw array, `pipeline` = `Data(x)`, `pipeline-detrend` = `Data(x, detrend=True)`); the reference re-implements the z-score (ddof=1) and the linear detrend itself.

| What | Compared with | Runs | Result |
|---|---|---|---|
| Two-sided CSM, both orientations, kernel and pipeline paths, T = 64-4096, plus odd n_fft (75, 125, 243) and padded lengths (63 -> 64, 1009 -> 1024) | `_kernels_spectral` (scipy dpss NW=3 K=5, numpy FFT, n_fft = next_fast_len_235) | 31 `...|csm*` | max abs diff / max abs S <= 1.08e-12 (delayed_noise T=4096); 1.6e-13 at T=1024 |
| Coherency, non-negative grid, NaN diagonal | kernel `coherency` | 18 `...|coh*` | max_abs <= 4.7e-12 |
| Coherency | mne_connectivity `spectral_connectivity_epochs(method='cohy', mode='multitaper', mt_bandwidth=6/T, mt_adaptive=False)` with its taper call (`_compute_mt_params`) patched to the reference's own scipy symmetric DPSS (NW=3, K=5) and unit weights, so the tapers are supplied by the reference and only mne's rfft, mean removal, CSD assembly and coherency normalisation are external; every ordered pair computed by mne | 13 `...|cohmne` | max_abs <= 4.7e-12 (labelled external-pkg; tapers from the reference) |
| Coherency | mne_connectivity defaults (periodic DPSS, sqrt(eigenvalue) weights) | 3 `...|1024|kernel|cohmnedef` | pyspi differs by up to 0.0305 (complex) and 0.0200 (MSC) on narrowband_f0-0.25; a scipy rebuild of mne's estimator matches mne to 6.0e-16 and pyspi matches the equal-weight kernel to 5.1e-13, so the difference is entirely the weighting/taper-symmetry choice |
| Grid, n_fft, number of tapers, n_observations | analytic rule | 20 `grid-n*`, `gridsum-n*` | exact. T = 63, 64, 75, 125, 243, 255, 256, 1000, 1009, 1024 give n_fft = 64, 64, 75, 125, 243, 256, 256, 1000, 1024, 1024; (n_fft+1)//2 bins kept (odd n_fft 243: 122 bins, top 121/243; even: Nyquist dropped); 5 tapers and n_observations = 5 at every T |
| Tapers; concentrations | dense eigendecomposition (scipy.linalg.eigh) of the Slepian commuting tridiagonal matrix; Rayleigh quotients of the dense sinc kernel | 4 `tapers`, 4 `eig` | max_abs 1.3e-13 and 2.3e-15; smallest concentration 0.94659 (T=64) to 0.94614 (T=4096), so the > 0.9 low-bias filter keeps all 5 |
| Hermitian, real DC, Parseval | analytic | 4 `ident` | max abs (S - S^H) = 0.0, max abs Im S(0) = 0.0, Parseval rel 9.2e-15 |
| Band averages of S_ii, Re S_ij, Im S_ij over 5 bands | analytic S = H Sigma H^H (equals the stored S_true), se from 300 Monte-Carlo draws with Bonferroni scaling | 4 `bandavg` | max_z 1.30 / 1.39 (var2_bivariate 1024 / 4096), 1.52 / 1.27 (var2_chain); pass threshold 2; estimator bias in the Monte Carlo <= 0.23 sd |
| Phase slope of S[i,j] over 0 < f < 0.1 on delayed_noise | analytic d_j - d_i, Monte-Carlo se | 2 `phslope`, 3 `phdir` | max_z 0.53 (T=1024), 0.99 (T=4096); direction agreement 1.0 |
| cohmag mean, full band, pipeline path | kernel on the re-implemented z-score | 3 `spi=`, 1 `invariance=` | max_abs <= 3.8e-15; scaled_offset equals the bivariate_gaussian draw to 2.9e-15 |
| Low-frequency power fraction (f < 6/T) on trending | kernel; qualitative judge | 3 `lfq`, 1 `lfqjudge` | max_abs 2.8e-15; without detrend the sloped channels put 0.956-0.982 of their power below 6/T (untrended draw 0.006-0.013); with pipeline-detrend 0.006-0.013 |

Verdict **matches**, score **9** (rubric: the independent reference agrees to round-off, and mne_connectivity agrees to round-off when given the reference's tapers and unit weights; its unpatched defaults differ by up to 0.0305, explained by the taper/weighting choice; not 10 because the documented "125 uniformly sampled bins ... f0 = 4/T" grid does not describe this path). Symmetry: Hermitian to 0.0. Direction: positive phase slope of S[i,j] means row i leads column j (row 0 of the slope matrix [1.016, 2.981, 5.030] at T=1024, [0.994, 2.985, 5.025] at T=4096).

Candidates: (1) the documented 125-bin grid from 4/T is not the multitaper grid: **confirmed** (grid runs). The 125-bin count matches the wavelet path (`pyspi/statistics/wavelet.py:54`), whose minimum frequency is 5/T (`pyspi/statistics/wavelet.py:62`), so f0 = 4/T matches neither path and the provenance stays a hypothesis. This is new (not in KNOWN_ISSUES.md). (2) expectation over the 5 tapers of one window: **confirmed** as a fact (n_observations = 5); its acceptability is for spectral_phase_sync. (3) every measure rebuilds the transform: **confirmed** (fftcount, exact [1, 1, 0, 1, 1, 1, 1, 12]); the second cohmag transform is the first-insert recompute already documented in tests/README.md "worth knowing". (4) zero padding for non-5-smooth T: **confirmed**; no manifest T and not the EEG workflow's T=1024. (5) exact-zero DC bin in icoh: **confirmed** (icoh DC bin 0.0; pyspi mean = (n-1)/n times the mean without DC to 7.9e-15); this elaborates REVIEW M4 (DC included), already documented.

### `spectral.band_selection`

pyspi's own reducers were instrumented rather than re-implemented: the probe seeds the per-Data spectral cache with an array whose value at bin k is k and replaces the instance's `_statfn` with a recorder, so `multivariate()` (`spectral.py:88`) and `bivariate()` (`spectral.py:132`) select bins on the real `Connectivity.frequencies` grid and the recorder sees which. PSI/GD were spied at `Connectivity.phase_slope_index`, `group_delay` and `connectivity._inner_combination`. EEG bands use the exact floats of `pyspi/workflow/retained185.yaml`.

| What | Reference | Runs | Result |
|---|---|---|---|
| Bin masks for the 3 config + 10 EEG bands | analytic on k/n_fft (inclusive for multivariate, strict for bivariate) | 5 `masks*` | exact. Full/low/high inclusive vs strict: 512/511, 257/255, 256/255 at T=1024 (others in the record). EEG 0.5-2, 2-4, 4-6 (/1024): 2/1, 3/1, 3/1 at T=1024; 1/0, 2/0, 2/0 at T=512; 0/0, 1/0, 1/0 at T=256 |
| cohmag and icoh, 6 identifiers each | independent kernel over the inclusive set | 36 `spi=` | max_abs <= 1.5e-13; the strict set would differ by up to 0.515 (icoh max, low band, iid T=64) |
| dcoh and DTF, 6 identifiers each (bivariate) | library per-bin values aggregated over the strict set (same-lineage) | 4 `bvagg` | max_abs 1.1e-15; the inclusive set would differ by 0.0017 (gauss_ar1_pair 1024), 0.0022 (var2_bivariate 1024), 0.129 (var2_chain 256) |
| f = 0.25 bin | analytic identities | 3 `id025`, `bvagg` on narrowband | inclusive: n_low m_low + n_high m_high - n_full m_full - MSC(0.25) <= 4.8e-13 (counted twice); strict (dcoh): n_full m_full - n_low m_low - n_high m_high - dcoh(0.25) = 1.4e-13 (counted in neither half) |
| PSI/GD arguments and bins | analytic | 4 `spy*` | exact: first call without kwargs (TypeError), second with [fmin, fmax] and (fmax - fmin)/50; PSI sums over 47/43/43 bins at T=1024, 50/49/49 at T=4096 |
| EEG band outcomes | predicted from bin counts | 5 `eeg*` | all predictions observed: T=1024 PSI IndexError and GD all-NaN for 0.5-2, 2-4, 4-6 (/1024); T=512 also GD ValueError and dcoh all-NaN; T=256 cohmag max ValueError and cohmag mean all-NaN for 0.5-2 |

Verdict **variant**, score **6** (rubric: deviates from the documented closed bands and from one-label-one-bin-set in an interpretable way). All six candidates **confirmed** (record lists evidence per candidate). Already documented and cited as such: REVIEW M4 (DC/Nyquist), M5 (all-pairs PSI), M6 (GD all-NaN; the narrow-band mechanism extends it), DEPENDENCY_COMPATIBILITY_REVIEW.md (band forwarding; the TypeError route refines it). New: the path-dependent edge rule and its f = 0.25 consequence, the PSI band-width-based sub-sampling, the PSI IndexError on the retained narrow EEG bands at T=1024, and nanmax raising on empty selections.

Cross-family observation, not judged here: on narrowband_f0-0.25 at T=1024 the library's per-bin directed coherence reaches 103 near f = 0.25 and pyspi's `dcoh_multitaper_max_*` reaches 86 (bvagg probe output). That belongs to spectral_directed.

### Trace-stage fields corrected

- `spectral.band_selection`: `implementation_trace` and the third candidate now cite `pyspi/calculator.py:327-339` at HEAD a1f20c7 (exception -> NaN table and a MetricReport with status failed), replacing the pre-a1f20c7 lines 292-303 / 301-303. `deviation_from_canonical` no longer calls the EEG band edges a hypothesis: they are stored exactly in `pyspi/workflow/retained185.yaml`.
- `spectral.multitaper_csm`: `deviation_from_documented` adds that the wavelet path's minimum frequency is 5/T, so f0 = 4/T matches neither path.
- `status` traced -> checked in both records.

## Open questions: answered

1. **Exact EEG band edges and epoch length.** Answered. `pyspi/workflow/retained185.yaml` stores exact floats, e.g. 0.00048828125 = 0.5/1024 up to 0.1220703125 = 125/1024, so the trace's k/1024 hypothesis holds. The workflow (docs/workflow.md, examples/workflow/preflight.yaml) uses 0.5 s windows (the default) at `sampling_rate_hz: 2048`, so T = 1024. At 2048 Hz the edges are 1, 4, 8, 12, 30, 70, 150, 250 Hz and the grid spacing is 2 Hz. At T = 1024 the 1-4, 4-8 and 8-12 Hz bands have 2, 3, 3 inclusive bins and 1 strict bin: PSI fails (all-NaN table and a failed MetricReport, per calculator.py:327-339) and GD is all-NaN for those three bands in every window (eeg runs). The rate is configurable and the bands are not rescaled by it ("EDF sampling_rate_hz does not rewrite them"); at a rate that makes T = 512 GD also raises and dcoh is all-NaN. The multitaper full bandwidth 2W = 6/T is 12 Hz at T = 1024, wider than these bands, so their values are smoothed across neighbouring bands even where they compute.
2. **125 bins from f0 = 4/T.** Partly answered. The multitaper path does not use it (grid runs). The count 125 matches the wavelet path (`pyspi/statistics/wavelet.py:54`), but that path starts at 0.2 with fmin = 5/T (`pyspi/statistics/wavelet.py:62`), so the sentence matches neither path exactly. Correcting the documentation keeps stored values; changing the code would change every multitaper SPI. The choice is a maintainer decision.
3. **Unify the band rule?** Evidence for the decision: inclusive vs strict changes cohmag/icoh means by up to 0.035 (T=64) and 0.0097 (T=256), and maxima by up to 0.515 (T=64), and dcoh/DTF by 0.0017-0.129 (spi and bvagg runs). Either unification changes stored values for one group of identifiers, so it needs new identifiers or an explicit versioned decision. Not settled by the checks.
4. **Taper average as the trial expectation.** The fact is confirmed (n_observations = 5, single window); the judgement belongs to spectral_phase_sync.
5. **Literature locations.** Not re-checked in this stage; no new copies were obtained. Still open: Percival & Walden section contents, full texts of Walden 2000 and Slepian 1978, journal copies of Nolte 2008 and Mitra & Pesaran 1999.
6. **EEG_OPTIMIZATION_NOTES.md shown as modified.** Not edited in this stage either.

## Optimisation

Sources: walls from `audit/profiles/summary.json` (fresh / family-warm; quiet machine; 118x1024 walls are single repeats), frame shares from cProfile. The official profiles at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`) are dominated by the P-independent DPSS cost for every multivariate unit, so for ranking at 118 channels this group added cProfile runs at 118x1024 (32x1024 for sgc nonparametric): `audit/profiles/micro/prof_<unit>__<shape>.pstats`, written by `audit/profiles/micro/prof_spectral.py`, indexed with walls, peak RSS and thread settings in `audit/profiles/micro/prof_spectral__index.json` (OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1; no numba or JIDT code on these paths). Those runs overlapped other Workflow 3 groups; the memory-bound multitaper units varied 2-7x in wall between runs (cohmag: 6.4 s on a first run, 0.93 s on the kept run, 2.42 s in summary.json), so only shares are quoted from them. Percentages are of the profiled `harness.compute_spi` frame. Micro-benchmarks (the group's two): `audit/profiles/micro/opp_spectral_csm-1.{py,json}` and `audit/profiles/micro/opp_spectral_csm-shared-csm.{py,json}`. A candidate that reproduces the current output "matches the baseline"; that says nothing about whether either is mathematically right. Nothing in `pyspi/` was changed.

Both units are infrastructure and have no timing of their own; their blocks are filled from the consumers (the CSM frames inside spectral.cohmag's pstats, plus the 118x1024 cohmag, psi and dtf profiles).

### spectral.multitaper_csm
- Cost per build, read from the consumers at 118x1024: cohmag fresh 2.42 s (measured), of which coherency 85.8% of the profiled call: `_complex_inner_product` (per-taper outer products) 38.5%, taper mean (ufunc reduce) 17.2%, normalisation/division (coherency self) 24.8%, DPSS 1.9%. At 8x1024 the same call is 0.0193 s and `transforms.py:tridisolve` (DPSS, pure Python) is 65.6% of it, `_complex_inner_product` 9.5% (`audit/profiles/spectral.cohmag__8x1024.pstats`).
- Builds per window in the EEG selection today: about 45 (phase-sync: plv, ppc, pli, dspli once each, wpli twice, dswpli three times; phase, cohmag, icoh once each plus one first-insert recompute for whichever spectral SPI is first on the Data; psi 11 bands x 2; gd 10 bands). Directed units build one 2-channel Multitaper per ordered pair (13,806 at 118 channels).
- Opportunities: `spectral_csm-shared-csm` (~2x on this unit's aggregate cost; bit-identical in the micro-benchmark). Only the coherency-type builds collapse (phase, cohmag, icoh, the first-insert recompute, 22 psi and 10 gd coherency evaluations become one); the 9 phase-sync per-taper builds are measure-specific (`connectivity.py:512-516, 571, 596, 648-654`) and stay, so about 45 builds become about 10 (a build-count ceiling below 4.5x) and the CSM-bound time per EEG window falls from about 96 s (coherence family 7.8 s, psi coherency 0.836 x 49.7 s, gd coherency 0.187 x 36.9 s, phase-sync 40.3 s) to about 43 s (one coherency ~2.5 s plus the unchanged 40.3 s), ~2.2x, in line with the measured 2.14x. An earlier version of this note claimed 10-20x by counting every build as collapsing; that contradicted the phase-sync refutation and is withdrawn. `spectral_csm-1` (2-4x per build, within-tolerance for the linear path at <= 2.2e-16, bit-identical for plv), `spectral_csm-dpss-cache` (1-1.3x: refuted at 118 for the multivariate path, 1.2-1.3x for the bivariate path).

### spectral.band_selection
- Own cost: per identifier after the first of a unit, 0.007-0.024 s at 118x1024 (family-warm increments), i.e. ~1.5 s per window for the 110 EEG band reductions of the 10 undirected classes; `np.nanmean` (including the `_replace_nan` copy) is 3.2% of the profiled cohmag call at 118.
- Indirect cost: psi and gd put the band in the cache key (`spectral.py:46-48`), so each band rebuilds the coherency (83.6% of a psi band call, 18.7% of a gd band call at 118).
- Opportunities: `spectral_csm-2` only. `spectral_csm-shared-csm` (dropping the band from the psi/gd cache key) is no longer listed on this unit: its gain is the per-band coherency rebuild, which is already credited on spectral.psi (6.7-8x) and spectral.gd (2-3.6x; records/spectral.gd.yaml, re-derived in Workflow 4c from un-profiled timings); listing it here too would count it twice in the ranking, and this unit's own cost (the band reductions) does not change. `spectral_csm-2` (~2x on the reduction itself: plain mean over a contiguous band slice when the slice has no off-diagonal NaN, nanmean otherwise; complexity argument only, unverified-bit-identical because nanmean is the same sum over NaN-replaced values divided by the same count).

**Shared avenues (ids owned by this group).**
- `spectral_csm-shared-csm` (cache-sharing, M): compute the multitaper Fourier coefficients once per Data and memoise the taper-averaged CSM, power and coherency under a key without measure or band (`pyspi/statistics/spectral.py:46-50, 64-83`; this also removes the first-insert string key at `:81`). Measured in `opp_spectral_csm-shared-csm.json` (re-run after adding segment timing): cohmag + phase (full band) + psi and gd in the 3 config bands on one Data, 1.179 s -> 0.514 s at 32x1024 (2.30x, min of 3) and 14.55 s -> 6.80 s at 118x1024 (2.14x, single pass; first run 2.31x and 2.14x, a reviewer re-run 2.27x and 2.10x); all 8 identifiers bit-identical to the baseline with equal NaN masks. The candidate's segment walls are now archived in the JSON (`candidate_segments_s`, fastest candidate repeat): at 118x1024 Multitaper + Connectivity + the one coherency 0.774 s, then 0.165-0.203 s per psi band and 1.553-2.036 s per gd band (32x1024: 0.078 s, 0.014-0.015 s, 0.114-0.142 s). An earlier unarchived scratch timing gave 0.72-0.75 s, 0.16-0.21 s and 1.55-2.02 s; a reviewer's scratch re-check gave 0.81-1.06 s, 0.16-0.28 s and 1.54-2.08 s.
- `spectral_csm-dpss-cache` (cache-sharing, S): cache the DPSS tapers by (T, NW, K, low-bias). `dpss_windows` is 41.7-77.8% of every multivariate call at 8x1024 but 0.4-2.4% at 118x1024, so it is refuted as a 118-channel avenue for the multivariate units and kept for the bivariate directed path (14.3-25.1% per ordered pair) and for small-P use.
- `spectral_csm-1` (algorithmic, M): build the taper-level products only on the (n_fft+1)//2 non-negative bins, in frequency blocks; for linear measures replace the per-taper tensor by one batched matmul over tapers. Ranking (README 'Optimisation columns', rule 4): on the coherency consumers it `depends_on` `spectral_csm-shared-csm` and is stacked after it, with `residual_speedup` 1.6-2.6x on spectral.phase (where it is stacked after `spectral_coherence-1` and `spectral_csm-shared-csm`) and 1x on spectral.cohmag, spectral.icoh and spectral.psi, which are later consumers of the one shared coherency that CoherencePhase builds first (`pyspi/workflow/retained185.yaml:218`), so its saving on that build is credited on phase alone; `spectral_phase_sync-1` `depends_on` it on all six phase-sync units: on plv, ppc, pli and dspli with a `residual_speedup` lower bound of 1x (the unit that pays the group's shared build), and on wpli and dswpli with `residual_speedup` 1.36x and 1.46x (the repeated builds that avenue removes are taper-level frames this one halves; see spectral_phase_sync.md). Measured in `opp_spectral_csm-1.json` (baseline = pyspi through `compute_spi`, candidates include the Multitaper/DPSS/FFT; re-run with min of 3 repeats at every shape and the per-repeat walls kept, because the first run timed the 118x1024 baselines once and the plv figure did not reproduce):

| shape | candidate | baseline s | candidate s | speedup | max_abs_diff | bit-identical | traced numpy peak MB (base -> cand) |
|---|---|---|---|---|---|---|---|
| 32x1024 | cohmag sliced | 0.0838 | 0.0499 | 1.68x | 1.1e-16 | no | 106.8 -> 53.5 |
| 32x1024 | cohmag gemm | 0.0838 | 0.0383 | 2.19x | 2.2e-16 | no | 106.8 -> 36.8 |
| 32x1024 | plv sliced | 0.1003 | 0.0553 | 1.81x | 0.0 | yes | 202.9 -> 103.9 |
| 118x1024 | cohmag sliced | 0.8684 | 0.4465 | 1.95x | 1.4e-16 | no | 1424.4 -> 712.6 |
| 118x1024 | cohmag gemm | 0.8684 | 0.2267 | 3.83x | 1.7e-16 | no | 1424.4 -> 462.7 |
| 118x1024 | plv sliced | 1.5063 | 0.6140 | 2.45x | 0.0 | yes | 2729.9 -> 1373.7 |

Walls are minima of 3 repeats. At 118x1024 the baseline repeats were 0.868/0.889/0.885 s (cohmag) and 1.777/1.776/1.506 s (plv); the plv ratio of medians is 2.88x. The first run of this micro-benchmark timed the 118x1024 baselines once and reported cohmag 2.27x / 4.31x and plv 9.64x (baseline 5.64 s); a reviewer re-run gave cohmag 1.72x / 3.70x and plv 4.14x, and three paired plv repeats gave 9.75x / 3.27x / 2.83x with baselines 9.02 / 3.12 / 2.85 s against candidates 0.93-1.01 s. The candidate walls are stable; the spread is in the memory-bound baseline. A Workflow 3 recheck re-run of the same script gave plv 1.89x min of 3 at 118x1024 (baselines 6.32/1.34/1.22 s, candidates 0.65-0.74 s, ratio of medians 1.86x; not saved to the JSON). The plv figure used in the records is therefore 1.8-2.5x, the min-of-3 figures; the 4x and 9.6x single-repeat figures are not used.

NaN masks are equal everywhere; on the 7x100 golden set the sliced cohmag differs by 2.2e-16 and the sliced plv matches the baseline bit for bit. The slow plv baselines point at the memory-bound one-shot `fcn`: a scratch probe (`<scratch>/wf3_spectral/divcheck.py`, not archived as a micro-benchmark) timed x/|x| on the real 118-channel per-taper tensor in 128-bin blocks at 8 x 0.042 s = 0.34 s, against 2.83 s for the one-shot `fcn` frame in the plv profile, which points at the GB-sized temporaries rather than the arithmetic (hence "in frequency blocks"). The library's own `blocks=` option (`connectivity.py:223-256`) is not a drop-in: it fills the lower triangle with the un-conjugated upper block (`:253-254`), which changes Im-based and phase measures.

Cross-cutting ids (owned by the cross-cutting group, `audit/profiles/crosscutting.yaml`): `xc-undirected-upper` (the library computes the full Hermitian P x P per bin; only i < j is needed by the undirected measures) and `xc-window-parallel` (memory below bounds how many windows run at once) apply here but are not listed as unit opportunities.

**Memory.** The two-sided taper-level tensor is 5 x 1024 x P^2 x 16 B = 1.14 GB at P = 118 and 5.08 GB at P = 249. Peak RSS (fresh records): 449-613 MB at 32x1024, 1.66-3.07 GB at 118x1024 (1.32-2.73 GB above the ~340 MB process baseline; family-warm records: 1.89 GB for the coherence family, 3.41 GB for the phase-sync family). Scaled by P^2 the increment is 5.9-12.2 GB per process at 249 channels. On this 31 GB machine that caps window-level concurrency at roughly 9 processes for the phase-sync units at 118 channels and 2-4 at 249; `spectral_csm-1` halves the tensor (one-sided) and the gemm expectation removes it for the linear measures (114 MB at 118, 508 MB at 249).

**Scaling.** summary.json fits P exponents of 0.9-1.4 between 8 and 32 channels (the fixed ~20 ms DPSS dominates there) but the local exponent between the measured 32x1024 and 118x1024 walls is 2.4-3.4 for every multivariate multitaper unit (cohmag 2.64, icoh 2.39, phase 2.71, pli 2.94, dspli 2.89, wpli 2.89, dswpli 2.92, plv 3.35, ppc 3.41, psi 2.61; gd 2.26). Above 2 this reflects memory-bound GB-sized temporaries, not extra arithmetic; the 118 walls are measurements, but extrapolation to 249 channels from the fitted exponents would understate the cost.

**Refuted or downgraded seeds.** (b) DPSS cache: refuted for the multivariate units at 118 channels (0.4-2.4% of each call), kept for the bivariate directed path. (c) one-sided CSM: confirmed and measured (`spectral_csm-1`); bit-identical for plv, 1-ulp differences for the coherency path. (d) repeated property re-evaluation: confirmed for wpli (2 builds), dswpli (3) and psi (2 coherency evaluations per band); coherency's two `_power` passes are 1.6-1.9% at 118 and not worth an avenue of their own.

**Shared versus incremental cost.** In a family-warm run at 118x1024 the first identifier of each unit pays the whole build (2.29-10.15 s) and every further band costs 0.007-0.024 s; the first spectral SPI on a Data pays twice (phase 2.29 + 2.63 s, plv 7.80 + 8.04 s in their family runs) because of the first-insert string key.
