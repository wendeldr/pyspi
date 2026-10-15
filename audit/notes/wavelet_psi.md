# wavelet_psi family: define, trace and check

Unit: `wavelet.psi` (class `PhaseSlopeIndex(mne, Undirected)`, `pyspi/statistics/wavelet.py:193-237`), six identifiers:
`psi_wavelet_{mean|max}_fs-1_fmin-{0|0-25}_fmax-{0-5|0-25}_{mean|max}` (bands full `[0, 0.5]`, low `[0, 0.25]`, high `[0.25, 0.5]`).
Delegate: mne-connectivity 0.9.0 `phase_slope_index(mode='cwt_morlet')`, with Morlet wavelets and the CWT from mne 1.13.2.
The probe scripts are in the session scratchpad under `wavelet_psi/probe1.py` to `probe5.py`. The regression goldens are snapshots of current output. Nothing here treats them as verified.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Documented definition

Supplement §1.5.4, p.14 (heading at supplement.txt file line 644; the briefing pack numbered it 658) says: "The phase slope index (PSI) [62] is a directed measure of information flow computed using the complex-valued coherency ... evaluates the consistency of the changes in phase differences across a pre-specified frequency range, weighted by the coherence." It continues: "The time-frequency decomposition is given by a Morlet wavelet, with spectral densities indexed by a time-frequency tuple ... we take the average (mean) or the maximum (max) PSI across all time points. We use the same pre-specified frequency ranges ... (i.e., low, high or full)". The keywords are "directed, linear/nonlinear, unsigned, bivariate, frequency-dependent and time-frequency dependent".

The bands come from §1.5, p.12-13 (heading at file line 570, band bullets at lines 595-599; the briefing pack gives §1.5 as lines 582-613): `[0, fs/2]`, `[0, fs/4]` and `[fs/4, fs/2]` with fs = 1; the GitBook overview repeats the bands at `spectral.md:14-20` (ranges at 18-20), and the GitBook Phase Slope Index section `spectral.md:322-346` repeats the §1.5.4 text quoted above. Reference [62] is Nolte et al. 2008.

## Canonical definition

Nolte, Ziehe, Nikulin, Schlögl, Krämer, Brismar & Müller (2008), *Phys. Rev. Lett.* 100, 234101 (arXiv:0712.2352; the equation numbers below were checked against the arXiv text), eq. 3:
Ψ̃_ij = Im Σ_{f∈F} C_ij*(f) C_ij(f+δf). This sums over adjacent frequencies of the analysis grid. C_ij = S_ij / sqrt(S_ii S_jj) is the coherency (eq. 4). Eq. 5 rewrites eq. 3 as Σ α_ij(f) α_ij(f+δf) sin(Φ(f+δf) − Φ(f)) with weights α_ij = |C_ij|; this is the "weighted by the coherence" form. S is an expectation, estimated by averaging over segments or epochs. The optional normalisation Ψ = Ψ̃ / std(Ψ̃) uses a jackknife estimate of the standard deviation (eq. 6). Direction: with S_12 = ⟨y1 y2*⟩ (eq. 2), a positive slope means y1 → y2.

PSI is antisymmetric. With S_ij = E[x_i x_j*], a positive Ψ_ij means i leads j. There is no canonical single-trial, time-resolved PSI. Time-resolved coherency needs trials or smoothing, because unsmoothed single-realisation wavelet coherency is identically 1 in modulus (Torrence & Webster 1999). The citation details are from memory.

## What is implemented (trace)

1. **Preprocessing.** `pyspi/data.py:184-195` z-scores each channel with ddof = 1. Detrending is optional and runs first.
2. **Single epoch.** `wavelet.py:207` passes the whole recording as a single epoch, shape `(1, N, T)`. `wavelet.py:209-217` calls `mnec.phase_slope_index(z, mode='cwt_morlet', sfreq=1, mt_adaptive=True, cwt_freqs=linspace(0.2, 0.5, 10))`. It does not pass `fmin`, `fmax` or `cwt_n_cycles`, so `n_cycles = 7`. `mt_adaptive` has no effect in cwt mode (`epochs.py:335-341`).
3. **Wavelets and CWT.** mne `tfr.py:151-178` builds 10 zero-mean Morlet wavelets with sigma_t = 7/(2πf) and support of ±5 sigma_t. Their lengths run from 55 samples (f = 0.2) to 23 samples (f = 0.5). `tfr.py:322-400` computes the CWT as a zero-padded FFT linear convolution in `same` mode. With sfreq = 1, the f = 0.5 wavelet `exp(iπt)` is purely real (probe: max |Im W| = 1.2e-16).
4. **Coherency.** `epochs.py:270-277` uses `tril_indices(n, -1)`, so the seeds are the rows i and the targets are the columns j, with i > j. `epochs.py:652` forms `csd = W_seed · conj(W_target)` at each time and frequency. `epochs_bivariate.py:84-94` divides by sqrt(psd_x psd_y) with n_epochs = 1, so **|C| = 1** (probe range: 1 ± 4e-16).
5. **PSI.** `effective.py:216-224` builds a single band (-inf, inf). `effective.py:495-528` (`_compute_psi`) sums `conj(C(f_k)) C(f_{k+1})` over **adjacent** pairs (lines 518-523) and takes the imaginary part (line 526). The result has shape `(N, N, 1, T)`: the upper triangle is 0 (`mne_connectivity/spectral/epochs.py:1555-1571`), and `mne_connectivity/base.py:812` reshapes it to dense form.
   Because |C| = 1, PSI(t) = Σ_k sin(Δφ_{k+1}(t) − Δφ_k(t)), where Δφ = φ_seed − φ_target. Probe: this matches MNE to 2.2e-15.
6. **Selection.** `wavelet.py:221` reads `freq = freqs_computed[0]`, which holds the 10 wavelet frequencies. Line 225 sets `freq_id = where(fmin ≤ f ≤ fmax)`. Line 232 then evaluates `statfn(real(psi[..., freq_id]), axis=(2, 3))`, and `[..., freq_id]` indexes the **last axis, which is time**. The selected samples are:
   - full band: t = 0..9
   - low band: t = 0, 1
   - high band: t = 2..9

   The probe on delayed_noise at T = 256 gives pyspi A[1,0] = −1.5647, equal to mean(psi[1,0,0,:10]). The mean over all time samples is −1.2219.
7. **Mirror and diagonal.** `wavelet.py:234-236` copies the lower triangle to the upper triangle and sets the diagonal to NaN. `calculator.py:329-335` stores the result.

## Deviation analysis

### From the documentation

- **Time window instead of frequency band (already documented).** The listed frequency range never restricts the frequencies. Every identifier uses the same 10-frequency PSI(t) and averages or maximises it over 2-10 samples at the start of the signal, not over all time points. Every selected sample lies inside the cone of influence, because the wavelet half-lengths (11-27 samples) all exceed 9. The issue is recorded in EEG_OPTIMIZATION_NOTES §review and DEPENDENCY_MIGRATION.md:84-85.
- **The documented bands cannot be reached.** The wavelet grid is 0.2-0.5. Even indexing the frequency axis, the low band would contain only 0.2 and 0.233, and 0.5 is the Nyquist frequency.
- **Directed versus symmetric.** The documentation says "directed". `config.yaml:1238` labels the unit undirected, and the code mirrors the matrix. The result is symmetric, and the sign encodes the direction of the higher-index channel relative to the lower-index one.
- **"Unsigned" versus signed output.** The output is signed. On delayed_noise at T = 256 the mean full-band variant spans −1.565 to +2.752 and the max full-band variant spans −0.414 to +5.640. All six variants have negative entries.
- **No coherence weighting.** The documentation says the slope is "weighted by the coherence", but with one epoch |C| = 1, so no weighting happens.
- **Library version.** The supplement cites MNE 0.23.0. The code that runs is mne-connectivity 0.9.0.

### From the canonical definition

- **No expectation.** The coherency is a single-epoch value, not an expected coherency, so PSI(t) is a sum of 9 unweighted sines of phase-difference increments.
- **No jackknife normalisation.** The raw Ψ̃ is returned. MNE does the same, and this is common practice.
- **Kernel.** The kernel matches the adjacent-frequency eq. 3. REVIEW M5 (all frequency pairs) applies only to the multitaper spectral PSI, not to this unit. The grid includes the Nyquist frequency, where Δφ is 0 or π (probe), so the 9th increment is not a phase slope. On delayed_noise at T = 1024 the mean |term| of that increment is 0.35.
- **Aggregation.** Averaging over edge samples is not a canonical summary. Neither is the signed maximum. The output is symmetrised, although canonical PSI is antisymmetric.

### Consequence (candidate)

Direction is not preserved. On delayed_noise, x0 leads x1, x2 and x3. The central-window raw PSI is negative for every seed > target pair at every T. pyspi nevertheless gives A[3,0] > 0 at all four lengths and A[2,0] = +1.26 at T = 4096.

On var2_bivariate (x0 → x1), pyspi A[1,0] = +1.84, −1.59, +0.46, −0.70 for T = 64 to 4096. The central-window values are −0.39, −0.39, −0.24 and −0.18. The analytic Nolte PSI_01 from S_true is +0.489 for the band 0.2-0.5, meaning x0 leads.

Reversing the channel order flips every mean-variant entry exactly (2.2e-16). For the max variant, A[1,0] is −0.414 in the original order and +2.014 after reversal.

### Direction convention

MNE's docstring (`effective.py:52-53`) says a positive PSI(seed, target) means the seed leads. The raw value psi[i, j] (i > j) is therefore PSI(i → j). pyspi stores A[i,j] = A[j,i] = PSI(max(i,j) → min(i,j)). The probe sign agrees with this convention on delayed_noise and var2_bivariate in the central window. The check stage still needs to confirm it.

## Proposed checks

Full specifications are in the record.

1. **Kernel, tolerance 1e-10.** A from-scratch numpy implementation without mne or mne_connectivity: closed-form Morlet wavelets, direct `np.convolve`, single-epoch coherency, adjacent-pair PSI and the pyspi selection and mirror. Datasets: iid_gaussian, delayed_noise, var2_bivariate, narrowband_f0-0.25 and shared_envelope.
2. **Analytic, exact.** Check that |C| = 1, that PSI(t) equals the sum of sines, and that the Nyquist Δφ is 0 or π.
3. **Analytic, exact.** Compare pyspi with the mean over t = freq_id (expected equal), with the all-time mean and with the mean outside the cone of influence. This quantifies the axis-selection deviation.
4. **Canonical direction.** Use Nolte eq. 3 from S_true for var2_bivariate and var2_chain, and from the pure-delay phase slope plus a numpy Welch coherency for delayed_noise. Report the fraction of pairs where the pyspi sign agrees.
5. **Empirical direction convention, pipeline.** Compare the raw psi[i,j] sign against the known leader, and confirm that A[i,j] == A[j,i].
6. **Channel-permutation self-consistency, pipeline.** Expect A_rev = −A for the mean variants.
7. **Null behaviour, statistical, iid_gaussian.** Expect the spread not to shrink with T, because at most 10 samples are ever averaged.
8. **Preprocessing sensitivity, pipeline-detrend.** Use trending and scaled_offset.

## Open questions

- What should wavelet PSI mean in a single-epoch setting? Possible definitions are (a) the time average of the time-resolved coherency product, which is closer to a smoothed or epoch-averaged coherency, (b) the band-limited PSI per time point averaged over the whole recording, or (c) segmenting the recording into epochs. Each would be a new identifier, not a fix.
- Should the Nyquist frequency (0.5) be excluded from the wavelet grid? At sfreq = 1 its wavelet is real-valued.
- Hypothesis: the direction errors come from the zero-padded edge samples. The check stage should confirm this by comparing with the mean outside the cone of influence.
- Should the output be stored antisymmetrically (directed) with its sign convention documented, given that the current symmetric mirror ties the sign to channel order?
- Are the cache semantics safe? `data.mne_psi` is keyed by nothing: a later SPI with a different `fs` would reuse the cached fs = 1 result. This cannot happen with the current config, which fixes fs = 1.

## Checks

Files: `audit/checks/manifest/wavelet_psi.json` (297 runs), `audit/checks/refs/wavelet_psi.py` (comparison venv; numpy/scipy plus `_kernels_spectral`, no mne import except the one same-lineage `sl` run) and `audit/checks/probes/wavelet_psi_probes.py`. Results are in `audit/checks/out/results/wavelet_psi.json`: 173 pass, 124 fail, 0 without reference. All 124 failures are expected deviations (documented quantity 36, intended quantity 12, direction 75 = 63 `dir` + 12 `welch`, null 1); none is a script error. The family took 4 min 18 s wall-clock (pyspi side 12 s; the rest is the reference-side Monte Carlo). T ≤ 1024 throughout, except the 12 Welch direction runs at T = 4096 (about 0.1 s per pyspi call). The run_ids below drop the `wavelet.psi|` prefix.

An interrupted earlier attempt left a draft manifest, refs, probes and results. Its design was reused after review. I added the input-window perturbation runs (`winperm-*`), the intended-quantity runs (`int:*`), the pyspi-side null Monte Carlo (`nullmc`) and the central-window control, then re-ran everything with `--force`.

After the adversarial review I added five more run groups and re-ran with `--force`:
- tail perturbations through the preprocessing (`tailpert`);
- the Welch direction check that proposed check 4 specified (`welch`);
- the band-robustness check of the var2 truth sign (`popband`);
- channel-permutation runs on iid_gaussian and var2_bivariate, which proposed check 6 specified;
- a probe of the `data.mne_psi` cache key (`fscache`).

With these, all eight proposed checks have been run. The intended-quantity runs (`int:*`) are an addition. They do not replace the Welch reference.

### Unit wavelet.psi

**What pyspi computes is confirmed to round-off.**
- An independent replica reproduces all six identifiers on five datasets at T = 64/256/1024: 106/106 `rep:*` runs pass with max_abs 1.0e-12. The replica uses closed-form Morlet wavelets, direct `np.convolve` with mne's `same` centring, single-epoch coherency, `_kernels_spectral.psi_adjacent`, pyspi's time-index selection and the mirror.
- The library's dense PSI(t) matches the replica to 2.3e-13 (`delayed_noise|256|kernel|rawpsi` and two more).
- The coherency has |C| − 1 ≤ 4.4e-16 (`delayed_noise|256|kernel|cohmod`), and PSI(t) equals the sum of sines of the adjacent phase-difference increments to 3.6e-15.
- At f = 0.5, Im C/|C| ≤ 1.2e-12 (`delayed_noise|1024|kernel|nyq`), and Δφ(0.5) mod π ≤ 5.9e-13.
- The dense shape is (N, N, 1, T), with the frequencies as traced (`libinfo`, 2/2).
- Shown with the same-lineage reference only: mne_connectivity 0.9.0 in the comparison venv returns exactly pyspi's value (`delayed_noise|256|kernel|sl:mean-full`, 0.0).

**The axis selection (already documented: KNOWN_ISSUES, EEG notes §review) is quantified as follows.**
- *Cache level* (`delayed_noise|256|{kernel,pipeline}|tsel`). Adding 100 to the cached PSI at a single time index changes the stored value exactly for t ∈ {0..9} (full), {0, 1} (low) and {2..9} (high). The maximum change for any other t is 0.0.
- *Input level, pipeline path* (`{delayed_noise|1024, var2_bivariate|1024, iid_gaussian|256}|pipeline|winperm-*`). Each window of every channel is reversed in time, which leaves the channel mean and SD seen by the z-score unchanged.
  - Reversing x[:, 37:] changes all six stored values by exactly 0.0 (3/3 pass). For the same perturbation, the documented all-time quantity would move by up to 2.87.
  - Reversing x[:, 0:10] changes them by 0.56–5.81.
  - Reversing x[:, 10:37] changes them by 0.12–4.13, because the selected samples t ≤ 9 see input up to t + 27 through the f = 0.2 wavelet.
- *Input horizon, kernel path* (`delayed_noise|256|kernel|xsel`, `var2_bivariate|1024|kernel|xsel`). Replacing x[:, t0:] puts the last influential t0 at 36 (full/high) and 28 (low), matching the bound max(t_sel) + 27. The change beyond that horizon is ≤ 9.9e-14.
- *Tail samples through the preprocessing* (`{delayed_noise|1024, var2_bivariate|1024, iid_gaussian|256}|pipeline|tailpert`, plus `delayed_noise|1024|{kernel,pipeline-detrend}|tailpert`; 5/5 pass against the replica, max_abs ≤ 3.9e-13). The horizon above holds for the temporal content of the tail, not for its summary statistics. Each run applies three perturbations to x[:, 37:]:
  - adding 0.5 to every channel changes the stored values by up to 3.13, 0.89 and 0.49 on the pipeline path (delayed_noise, var2_bivariate, iid_gaussian);
  - multiplying channel 0 by 3 changes them by 0.094, 0.211 and 0.012;
  - rescaling the tail by 3 about its own mean changes them by ≤ 2.9e-13. This keeps the channel mean fixed and raises the SD by up to 2.95×.

  On the kernel path, all three perturbations change the stored values by ≤ 1.2e-13. With detrending (delayed_noise) they change them by 1.60, 0.131 and 0.087.
- So on the default pipeline path, tail samples reach the stored value only through the channel mean that the z-score subtracts. The SD is a positive per-channel rescaling, and the single-epoch coherency phase ignores it. With detrending, the fitted line carries the tail as well.
  - A mean shift moves the stored value because the zero-mean Morlet transform of a constant is not zero inside the zero-padded edge window that pyspi reads.
  - Net effect: on a T = 1024 recording, on the kernel path and on the default pipeline path without detrending, the temporal content of 987 of the 1024 samples cannot affect any of the six identifiers. Those samples act only through one scalar per channel.
  - With detrending this does not hold. The fitted slope is a time-weighted sum of the tail, so the time order of the tail enters the stored value: `delayed_noise|1024|pipeline-detrend|tailpert` changes it by 0.087 when the tail is rescaled about its own mean, and the review probe `wavelet_psi_check/review_detrend_rev.py` (session scratchpad) shows the same for a time reversal of x[:, 37:].
  - The earlier statement that they cannot affect the output at all was true only for the kernel path. It was also true for perturbations that keep the channel mean, such as the time reversals in `winperm`. The adversarial review caught this, and I corrected it.

**The documented quantity is not what is stored.** The documented quantity is the per-time PSI on the band's wavelet frequencies, averaged (or maximised) over all time points. All 36 `doc:*` runs fail (expected), with max_abs 0.07–6.06; for example, `delayed_noise|1024|kernel|doc:mean-full` misses by 1.76.

**The intended quantity is not what is stored.** The intended quantity is Nolte eq. 3 (`_kernels_spectral.psi_adjacent`) on the time-averaged coherency of the own Morlet CWT (`_kernels_spectral.cwt_morlet`, 7 cycles). The sign convention is Ψ(i → j) > 0 when i leads, stored at A[i,j] = A[j,i] for i > j.
- All 12 `int:*` runs fail on value (expected), with max_abs 0.51–5.55.
- The intended quantity itself has the true sign on 12/12 runs. On delayed_noise at T = 1024 its lower-triangle entries are −1.77, −3.33, −2.90, −2.44, −3.02, −2.92 (mean-full), all negative as the lag structure requires.
- pyspi gets −1.38, −0.99, −1.31, +1.06, +0.85, −0.10 on the same pairs (sign agreement 0.67). On var2_bivariate at T = 1024 it gets +0.456 against the intended −0.473 (agreement 0.0).

**Direction** (`*|pipeline|dir:*`, 72 runs on delayed_noise, var2_bivariate, var2_chain and logistic_maps_chaotic at T = 64/256/1024; 9 pass).
- Mean sign agreement with the true lead/lag over the 12 runs per identifier: mean-full 0.40, mean-high 0.50, mean-low 0.28, max-full 0.10, max-high 0.10, max-low 0.28.
- Controls on the same runs (extra keys):
  - the same statistic over an equally long window centred at T/2: 0.69 (mean-full);
  - the documented all-time mean: 0.89;
  - the intended quantity: 1.0 (`ref_intended_whole_agree` is 1.0 for every full and high variant, but 0.86 for mean-low and max-low: 0.0 on logistic_maps_chaotic T = 64 and 0.33 on var2_chain T = 64).
- The truth sign for var2 comes from the population Nolte PSI on S_true. The band choice cannot explain the misses, because that sign does not depend on the band definition (`var2_bivariate|1024|pipeline|popband:mean-full` and `var2_chain|…|popband:mean-full`, 2/2 pass). On var2_bivariate, PSI(1 → 0) is:
  - −0.833, −0.521, −0.312, −0.465 and −0.152 over [0, 0.5], [0, 0.25], [0.25, 0.5], [0.2, 0.5] and [0.2, 0.25] on the grid k/64;
  - −0.483 and −0.113 over the wavelet grid in [0.2, 0.5] and [0.2, 0.25].

  All three var2_chain pairs are negative in all seven definitions.
- *Welch reference for delayed_noise* (`delayed_noise|{1024,4096}|pipeline|welch:*`, 0/12 pass, as expected). This uses Nolte eq. 3 on the analytic pure-delay coherency on the grid k/64, excluding 0 and 0.5.
  - A numpy Welch estimate (periodic Hann window, L = 64, 50% overlap) has the analytic sign on 12/12 runs. At T = 1024, mean-full gives −2.89, −8.33, −5.62, −12.88, −10.66, −5.68 against the analytic −2.91, −8.62, −5.74, −14.00, −11.25, −5.74.
  - pyspi agrees with that sign on 0.67 / 0.33 (mean-full), 0.50 / 0.33 (mean-low), 0.67 / 0.33 (mean-high), 0.0 / 0.0 (max-full), 0.50 / 0.33 (max-low) and 0.0 / 0.17 (max-high) of entries at T = 1024 / 4096. Agreement does not improve at the longer record. For logistic_maps_chaotic the truth is the coupling direction x0 → x1. That is an expectation, not an analytic PSI sign, but the intended quantity agrees with it.
- The signed max variants have a positive null bias (null mean 3.27), so they carry almost no direction.

**Null and convergence** (`iid_gaussian|{64,256,1024}|pipeline|nullmc`, pyspi on 300 fresh iid pairs per T; 3/3 pass, max_z 1.40).
- The SD of pyspi's mean-full value is 1.286, 1.267 and 1.280 at T = 64, 256 and 1024. The own Monte Carlo gives 1.292, 1.257 and 1.307.
- The SD of the all-time mean falls as 0.317, 0.129 and 0.060, so the ratio grows as 4.1, 9.8 and 21.4. pyspi's statistic does not converge with T.
- Over 10 central samples the SD is 1.08–1.11 against 1.26–1.31 for the first 10 samples (`mc_sd_central_window_mean_full`).
- The per-dataset null runs (`null:*`, Bonferroni over pairs within one run) pass 17/18. The one miss, `iid_gaussian|64|pipeline|null:max-high` with p_emp 0.002, is within the multiplicity of 18 dependent runs (180 tests). The `nullmc` design, with family-wise control declared before running, is the primary null evidence.

**Symmetry and orientation.**
- max|A − A.T| = 0.0 on all 261 plain runs.
- Channel reversal flips every mean-variant entry: max|A_rev + A| = 2.2e-16, 4.4e-16 and 0.0 on delayed_noise, iid_gaussian and var2_bivariate at T = 256 (`perm:mean-full`, 3/3 pass against the replica).
- For the max variant, A[1,0] goes from −0.414 to +2.014, from 0.374 to 5.517 and from −0.358 to 3.942 (`perm:max-full`, 3/3 pass).
- The library's dense output has a zero upper triangle and zero diagonal.

**Preprocessing.**
- scaled_offset equals its base draw to 3.1e-15 (`base:*`).
- On trending, the detrended path equals the base draw to 7.8e-16. Without detrending the value moves by 0.97 (mean-full) and 2.09 (max-full), because the trend enters the edge-dominated window (`trending|256|*|rep:*`).

**T < 25 edge case** (`delayed_noise|64|kernel|short`, x[:, :20]). freqs_computed is 0.267..0.5 (8 values). mean-low is NaN, and max-low raises `ValueError` (zero-size reduction), which leaves the matrix all-NaN. The replica with the same frequency drop agrees to 5.4e-15. This case is not reachable with the audit datasets.

**Verdict: degenerate, score 0.** The rubric anchor is "wrong axis indexed", and the axis issue itself is already documented. The computation is reproduced exactly, but the stored number is a 2–10-sample edge window of an unweighted single-epoch PSI. Without detrending, the temporal content of all but the first 37 samples cannot reach it; the rest of the record enters only through the channel mean used by the z-score (with detrending, the fitted line also carries the time order of the tail). It does not converge with T, and carries the true direction on 10–50% of entries, against 100% for the intended quantity on the full and high variants and 86% on the low variants. `definition_under_review` is true because there is no canonical single-epoch time-resolved PSI (see the open questions).

**Candidates:**
1. Mirror of an antisymmetric quantity: confirmed, new.
2. Direction not carried: confirmed. It is a consequence of the documented axis selection plus the single-epoch coherency.
3. Unit-modulus coherency, so no coherence weighting: confirmed, new. KNOWN_ISSUES covers only the nine other wavelet classes.
4. Real-valued Nyquist wavelet: confirmed, new. The Nyquist item in DEPENDENCY_COMPATIBILITY_REVIEW.md concerns the multitaper grid.
5. T < 25 frequency drop: confirmed, new, minor.
6. Label contradictions: confirmed, new, minor. An earlier version of this note treated it as covered by the general labels topic (EEG notes §review 'parameter-specific/shared labels', REVIEW M9). Neither names a metric, REVIEW M5's remark that signed/directed metadata suits PSI concerns the multitaper spectral PSI among the stored identifiers (none of the retained 185 is a wavelet metric), and KNOWN_ISSUES does not list it, so it is counted as new, as the same item is on wavelet.icoh (harmonised in Workflow 4b-2).

`new_finding` is therefore true.

**Trace-field corrections.** The final sentence of `direction_convention` ("To be confirmed in the check stage") is replaced with the confirmation. The remaining trace issue (the "582-602" citation of the briefing pack) was already corrected before this stage: the record and line 12 of this note both say "the briefing pack gives §1.5 as lines 582-613", which matches `briefing/wavelet_psi.json`. No further change was needed.

## Open questions: answered

- **What should wavelet PSI mean for single-epoch input?** Not settled by the evidence; it is a definition decision, so `definition_under_review` is true. The checks rank the options:
  - Nolte eq. 3 on the time-averaged (smoothed) wavelet coherency (`intended`) recovers the true sign on 12/12 runs and has a coherence weighting.
  - The all-time mean of the per-time PSI on the band frequencies (`documented`) gets 0.89 sign agreement for mean-full. It stays unweighted, and its null SD shrinks like 1/sqrt(T): 0.32 → 0.06.
  - Epoching was not tested.

  Any of these would be a new identifier, not a fix of the existing values.
- **Should the Nyquist frequency 0.5 be removed from the wavelet grid?** The checks confirm that its wavelet is real (Im W = 1.2e-16) and that its phase difference is 0 or π (nyq, rawpsi). The last of the 9 increments therefore carries no phase-slope information. Removing it would change every stored value, so it is a decision for the definition discussion. The evidence supports excluding it.
- **Is the direction error caused by the zero-padded edge samples?** Partly. The window length matters more:
  - the same statistic over 10 central samples agrees with the truth on 0.69 of entries (mean-full), against 0.40 for the first 10 samples, 0.89 for all time points and 1.0 for the smoothed coherency;
  - the null SD of a 10-sample mean is 1.08–1.11 centrally and 1.26–1.31 at the edge.

  So averaging 2–10 samples of a unit-modulus PSI is the main cause, and the cone-of-influence edge adds to it.
- **Should the output be stored antisymmetrically?** The checks show that the symmetric mirror discards nothing: the reverse orientation is exactly −PSI (2.2e-16). They also show that the stored sign flips under channel reversal (perm runs). Storing it antisymmetrically, with A[i,j] > 0 meaning j leads i or the opposite, documented either way, would make the sign interpretable. This is a definition decision.
- **Is the unkeyed `data.mne_psi` cache acceptable?** Not for fs ≠ 1. The config fixes fs = 1, so the current identifiers are not affected. All six read the same cache entry, which the `tsel` runs exploit, and the replica agreement shows that sharing is correct for fs = 1.
  - The probe `iid_gaussian|256|pipeline|fscache` (1/1 pass) measured the fs ≠ 1 case. A `PhaseSlopeIndex(fs=2, fmin=0, fmax=0.5)` on a Data that already holds the fs = 1 result returns the fs = 1 value bitwise.
  - On a fresh Data, the same SPI differs by 7.23 and equals the fs = 2 replica to 1.8e-15. The replica uses wavelets at f/fs cycles per sample and default fmin 5·fs/T.

  So a future fs ≠ 1 identifier computed after an fs = 1 one would silently store the fs = 1 value. Keying the cache by fs (and by the frequency grid) would prevent this. The sister family showed the same behaviour for `data.mne` (`wavelet.morlet_cwt|iid_gaussian|256|pipeline|fscache`).
- **Citation details given from memory.** The Nolte 2008 equation numbers were checked at the trace stage. Torrence & Webster 1999 was not re-checked here. The unit-modulus property it is cited for is checked numerically instead (cohmod, |C| − 1 ≤ 4.4e-16), so the claim does not rest on the citation.
- **EEG_OPTIMIZATION_NOTES.md modified in the working tree.** This is outside the scope of the check stage. This stage did not modify that file, and it read only the lines cited above.

## Optimisation

No opportunities are proposed. The user does not use the wavelet family, and its definition is under review: the psi_wavelet axis issue (aggregation over time indices 0-9 instead of a frequency band), and the one-sample Morlet wavelets of the sister family. The optimisation block holds hotspot fields only.

**Walls** (`audit/profiles/summary.json`):

| Shape | Fresh | Family-warm | Measure mode |
|---|---|---|---|
| 8x1024 | 0.024 s | 0.025 s | measured |
| 32x1024 | 0.298 s | 0.297 s | measured |
| 118x1024 | 35.05 s | 34.10 s | measured (repeats 1; fresh started 13:31 UTC, while Workflow 3 groups were running) |

**The 118x1024 wall does not reproduce.** Isolated re-times of the representative identifier at 118x1024 (one process, warm-up at 8x1024, threads 1; `audit/profiles/micro/retime_wavelet.py`, `audit/profiles/micro/prof_wavelet__index.json`) gave 8.81 s in this stage, and 6.22 s and 6.37 s in the reviewer's re-time. That is 4-5.6x below summary.json. The 32x1024 wall reproduces (0.299-0.307 s in the reviewer's re-time). The cause was not established.
- Re-timed 32 → 118 exponent: 2.3-2.6, close to the code's P². Connections grow 13.9x (496 → 6903) and the wall grows 21-30x. The 3.65 implied by summary.json comes from the non-reproducing 118 wall.
- The summary.json `projected_window_s_118` (35.05 s fresh, 34.10 s family-warm) and `projected_patient_s` are therefore about 4-5x high. A per-window cost of about 6-9 s for the cache build is the better estimate, though it rests on three isolated runs on a loaded machine.
- P_exponent from 8 to 32 is 1.80 fresh / 1.78 family-warm, which is overhead-heavy at P=8. T_exponent is 1.07 / 0.96.
- Peak RSS at 118x1024 is 3846 MB (summary) and 4178 MB (re-time process).

**Hotspots at 118x1024** (`audit/profiles/micro/prof_wavelet.psi__118x1024.pstats`, profiled call 8.0 s, opaque share 0.043; percentages of the `compute_spi` frame):
- `mne_connectivity/spectral/epochs.py:spectral_connectivity_epochs` own time: 43.9% self (75.4% cumulative).
- `mne_connectivity/effective.py:_compute_psi`: 22.5% self.
- `epochs_bivariate.py:_CohyEst.compute_con`: 15.5%.
- `epochs.py:_epoch_spectral_connectivity`: 8.9% self.
- `_CohEstBase.accumulate`: 3.1%.
- `mne/time_frequency/tfr.py:cwt`: 0.45%.
- `mne_connectivity/base.py:_prepare_xarray`: 0.16%.

**Hotspots at 8x1024** (`audit/profiles/wavelet.psi__8x1024.pstats`, 46 ms profiled call, opaque share 0.189): `_CohEstBase.accumulate` 19.9%, `_CohyEst.compute_con` 16.0% (both inside `_epoch_spectral_connectivity` 43.6%), `tfr.py:cwt` 10.6%, `_prepare_xarray` 9.1%, `_compute_psi` 7.1%. The fixed container and CWT costs that weigh on the 8x1024 call almost vanish at 118x1024. The record therefore quotes the 118x1024 profile.

**Shared versus incremental.**
- The `data.mne_psi` cache is built by the first identifier: 34.09 s at 118x1024 in the summary.json family-warm run, which does not reproduce (isolated fresh builds 6.2-8.8 s, above).
- The other five identifiers re-aggregate the cached (P, P, 1, T) array in about 1 ms each (family-warm increments).
- Nothing is shared with the wavelet_measures units, which use their own `data.mne[measure]` cache.

**EEG scaling.** The unit has no EEG identifiers (`in_eeg_selection: false`).
