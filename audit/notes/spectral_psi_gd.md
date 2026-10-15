# spectral_psi_gd family: define, trace and check

Units: `spectral.psi` (identifiers `psi_multitaper_mean_fs-1_fmin-0_fmax-0-5`, `..._fmin-0_fmax-0-25`, `..._fmin-0-25_fmax-0-5`; ten further EEG band identifiers, 11 EEG identifiers in all with the config identifier `psi_multitaper_mean_fs-1_fmin-0_fmax-0-5`, which the EEG selection also keeps) and `spectral.gd` (`gd_multitaper_delay_fs-1_*` for the same three bands; ten EEG band identifiers).
Classes: `PhaseSlopeIndex(NonparametricSpectralMultivariate, Undirected)` at `pyspi/statistics/spectral.py:288-301` and `GroupDelay(NonparametricSpectralMultivariate, Directed)` at `spectral.py:304-317`.
Delegate: spectral_connectivity 1.1.2 (`Connectivity.phase_slope_index`, `Connectivity.group_delay`), used through the NumPy-2 shim `pyspi/_spectral_compat.py`. The group-delay regression runs in scipy 1.17.1 `scipy.stats.mstats.linregress`.
The probe scripts are in the session scratchpad under `spectral_psi_gd/p1_instantiate.py` to `p8_misc.py`. The regression goldens are snapshots of current output, and nothing here treats them as a reference. The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Statements marked CANDIDATE or hypothesis in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared machinery (both units)

**Documented frame.** Supplement §1.5, p.12-13 (heading at supplement.txt line 570), with the same text on GitBook `spectral.md:14-20`:

> "We use 125 uniformly sampled bins across the entire frequency range, f ∈ [f0, fs/2] ... f0 = 4/T ... fmin-0_fmax-0-5: the full-frequency range, f ∈ [0, fs/2]; fmin-0_fmax-0-25: ... [0, fs/4]; fmin-0-25_fmax-0-5: ... [fs/4, fs/2]."

**Implemented frame (trace).**
1. `pyspi/data.py:184-195`: optional linear detrend, then z-score with ddof=1. Coherency does not depend on scale. In the probe, switching normalisation off changed PSI by at most 2.5e-14 and GD by at most 4e-17. Detrending does change the values: on `trending`, PSI changed by up to 10.7 and GD by up to 1.07.
2. `spectral.py:64-83` (`_get_cache`): `sc.Multitaper(z, sampling_frequency=1)` and `Connectivity.from_multitaper`. `transforms.py:74-102` sets the library defaults: NW=3, five DPSS tapers (`:155-164`; the eigenvalue > 0.9 filter at `:668-673` kept all five in the probe), constant detrend per window, one window covering the whole series. `n_fft = next_fast_len(T)` (`:181-185`). The non-negative frequencies are bins 0 … n_fft/2−1 (`connectivity.py:174-180`, `:73-89`), so the Nyquist bin never appears. The cross-spectrum is `S_ij = X_i·conj(X_j)` averaged over tapers (`connectivity.py:195-208`, `:1187-1190`, `:45-53`). The coherency is `S_ij/√(S_ii S_jj)` with a NaN diagonal (`:295-313`).
3. **Band forwarding.** `spectral.py:73` first calls the library method with no arguments. Under 1.1.2 that call raises TypeError: `_bandpass` subscripts `None` at `connectivity.py:1329`, and `_get_independent_frequency_step` divides `None` at `:1357`. `spectral.py:75` then calls `_get_statistic`, which passes `frequencies_of_interest=[fmin, fmax]` and `frequency_resolution=(fmax−fmin)/50`. A spy probe recorded both calls: first `{}`, then `{'frequencies_of_interest': [0, 0.25], 'frequency_resolution': 0.005}`. The pinned version therefore does forward the band. The failure in DEPENDENCY_COMPATIBILITY_REVIEW.md:84 sits under the heading 'Spectral-connectivity 2.0.1' (`:74`) and describes a method that "accepts default arguments". That it applies to 2.0.1 is my reading of that wording. 2.0.1 is not installed in this checkout, so I have not re-tested it. This is a refinement of the known band-forwarding item, not a new defect.
4. **Band rule.** `_bandpass` keeps only `fmin < f < fmax`, with strict inequalities at both ends (`connectivity.py:1313-1335`). DC is excluded. The bin f = 0.25 falls in neither half band.
5. **Decimation step.** `s = ceil(((fmax−fmin)/50)·n_fft)` (`connectivity.py:1338-1357`). This step depends on the band width. It is not the multitaper resolution 2W = 6/T.
6. `spectral.py:85-102` (`multivariate`). For PSI, the four-index slice of the three-dimensional result raises IndexError, and `adj_freq[0]` is returned as is. For GD, `group_delay` returns a 3-tuple, so evaluating the argument `adj_freq[0, freq_id, :, :]` raises "TypeError: tuple indices must be integers or slices, not tuple" before the None `_statfn` is ever called. The TypeError is caught and `delay[0]` is returned. `np.fill_diagonal` then writes into the cached array, because the returned matrix shares memory with `data.spectral_mv`.
7. `pyspi/calculator.py:295-355` (lines at HEAD a1f20c7; :329 the call, :336-339 the handler): any exception turns the whole matrix into NaN and records a failed MetricReport.

Deviations shared by both units, from the documented frame: the grid is n_fft/2 bins starting at 1/n_fft (in band), not 125 bins starting at 4/T; the band intervals are open, not closed; and the effective grid is decimated to about 50 bins. For the full band at T = 64/256/1024/4096 the number of kept bins K is 31/43/47/50 (probe p2).

## spectral.psi

**Documented.** Supplement §1.5.4, p.14 (heading at supplement.txt line 644), with the same text on GitBook `spectral.md:322-338`: "a directed measure of information flow computed using the complex-valued coherency ... evaluates the consistency of the changes in phase differences across a pre-specified frequency range, weighted by the coherence." The listed keywords are "directed, linear/nonlinear, unsigned, bivariate". `config.yaml:722-728` also lists "directed", but the class labels are `['unsigned','spectral','undirected']`, and `Calculator._load_yaml` reads the class labels, not the config list.

**Canonical.** Nolte et al. 2008, PRL 100, 234101. Equation numbers are from arXiv:0712.2352v1, p.2 (the copy REVIEW M5 cites), not from the PRL journal copy:
- Eq. 3: Ψ̃_ij = Im Σ_{f∈F} C*_ij(f) C_ij(f+δf), summed over adjacent frequencies at resolution δf.
- Eq. 4: the coherency, C_ij(f) = S_ij(f)/√(S_ii(f) S_jj(f)).
- Eq. 5: eq. 3 rewritten as a coherence-weighted sum of phase differences.
- Eq. 6: the normalised Ψ = Ψ̃/std(Ψ̃), "with std(Ψ̃) being estimated by the Jackknife method".

Sign: eq. 2 gives S_12(f) = ⟨y1(f) y2*(f)⟩ ∼ exp(i2πfτ) for y2(t) = a·y1(t−τ), and the next sentence says "the causal direction is estimated to go from y1 to y2 (y2 to y1) if it is positive (negative)". So Ψ_ij > 0 means i leads j. This matches the library's S_ij = X_i·conj(X_j) and the `delayed_noise` probe.

**Implemented.** `connectivity.py:1060-1104` bandpasses the coherency, decimates it by s, and calls `_inner_combination` (`:1107-1113`). That function sums `conj(C(f_a))·C(f_b)` over **all** K(K−1)/2 unordered pairs of kept bins and returns the imaginary part. pyspi takes time window 0 and applies no mean, even though the identifier says `mean`.

A hand recomputation of the all-pairs sum from the library coherency matched pyspi exactly (difference 0.0). An independent numpy/scipy implementation (scipy dpss, no spectral_connectivity) agreed to within 9e-12 on 6 dataset/T combinations × 3 bands. Each pair uses only its own two channels, so the measure is bivariate in effect: adding five iid channels left a pair's value unchanged. The output is exactly antisymmetric. The MRO bypasses `Undirected.multivariate`, which is REVIEW M1 (tests/README.md bug 5 before b2f6d7e).

**Deviation analysis.**
- All-pairs instead of adjacent pairs. This is already documented (REVIEW M5). The probes add the following detail. It is an elaboration of M5, recorded in `deviation_from_canonical` only and not as a candidate:
  - The magnitude scales roughly with K², so it depends on T and on the band. For pair (0,1) of `delayed_noise` in the full band, pyspi gives 296 / 580 / 694 / 787 at T = 64…4096. The adjacent-pair formula gives 2.75 / 3.06 / 3.03 / 3.01 in the trace-stage probe; the check-stage `nolte:full` runs give 3.061 / 3.094 / 3.081 at T = 256 / 1024 / 4096, and no check run covers T = 64 (see ## Checks).
  - For a pure delay with |C| = 1, the all-pairs value *falls* as the delay grows. At T=256 in the full band, delays τ = 1/5/30 give 588/116/10.6, while Nolte's PSI gives 3.09/15.1/33.7. Ranking pairs by pyspi's PSI therefore inverts the ranking by delay magnitude. On `delayed_noise` (0-0.25 band, T=256), A[0,1] = 238 (τ=1), A[0,2] = 255 (τ=3) and A[0,3] = 105 (τ=5).
- CANDIDATE: the `mean` in the identifier is not applied; the value is a raw sum (`spectral.py:91-93`).
- CANDIDATE: in narrow bands with 0 or 1 bins, `_inner_combination` raises IndexError and the whole matrix becomes NaN. Probe: the EEG bands 0.5-2, 2-4 and 4-6 (/1024) fail at T=512 and T=1024. A band with exactly 2 kept bins returns the single Nolte term. This confirms pack hypothesis 3.
- Labels: the output is signed and antisymmetric, but the class is labelled "undirected, unsigned". Already documented (REVIEW M1 and M5; tests/README.md bug 5 before b2f6d7e).

**Direction.** A[i,j] > 0 means row i leads column j. Probe on `delayed_noise` (0-0.25 band, T=256): A[0,1] = +238 and A[1,0] = −238. If the REVIEW M1 bypass were "fixed" by mirroring, the sign information would be lost.

**Proposed checks** (see the record for full detail):
1. Independent literal re-implementation: kernel, tolerance 1e-10 relative.
2. Nolte eq. 3 (and the eq. 6 normalisation) from the same independent coherency: qualitative. Record the ratio and sign agreement.
3. Direction on `delayed_noise`.
4. Analytic pure-delay curve.
5. Pair-alone vs embedded, exact: should be equal.
6. Bin-count failure for the EEG bands.
7. normalise/detrend paths on `trending` and `scaled_offset`.

## spectral.gd

**Documented.** Supplement §1.5.3, p.13-14 (heading at supplement.txt line 630), with the same text on GitBook `spectral.md:130-148`: "infers a directed, average time-delay between two signals by measuring the slope of the phase differences ... as a function of the frequency (obtained through linear regression). The slope is only computed for statistically significant coherence values and the time delay is obtained by a simple rescaling of the slope by 2π." The listed keywords are "directed, linear, unsigned, bivariate".

**Canonical.** Gotman 1983 (EEG Clin Neurophysiol 56:501-514): fit the coherency phase against frequency by linear regression over the frequencies with significant coherence, and take delay = slope/2π. The details of Gotman's selection and weighting rules are from memory (see Open questions).

**Implemented** (`connectivity.py:902-987`):
- The bandpassed coherency is computed for the upper-triangle pairs.
- Significance (`_find_significant_frequencies`, `:1432-1482`; `statistics.py`):
  - A Fisher z is computed as `(atanh|C| − b)/√b`, with `b = coherence_bias(5) = 1/8`.
  - p-values are one-sided.
  - Benjamini-Hochberg at 0.05 is applied to the **flattened (frequency × pair) array**.
  - Only the largest contiguous run is kept (`scipy.ndimage.label`).
  - Every s-th bin of that run is taken, and the pair is discarded if fewer than 3 points remain.
- Regression: the phase is `np.unwrap(angle C)` along the full band grid, then masked. An OLS fit with a free intercept follows (`mstats.linregress`). `delay = slope/2π`, and the reverse entry is −delay. If every point is masked, the slope is None and the entry becomes NaN.

An independent re-implementation reproduced pyspi's NaN pattern exactly and its finite values to within 4e-13 (probe p6).

**Deviation analysis.**
- CANDIDATE (new): the measure is **not bivariate**. Because the BH threshold is pooled across all pairs, adding unrelated channels changes a pair's estimate. Probe, comparing the pair alone with the pair plus 5 iid channels:
  - `var2_bivariate` T=256, full band: −0.4725 → NaN.
  - `narrowband_f0-0.1` T=256, full band: −0.0277 → −0.9866.
  - `gauss_ar1_pair` T=1024, full band: −1.793 → NaN.
- CANDIDATE: the "independent frequency" step depends on the band width ((fmax−fmin)/50), not on 2W = 6/T.
  - At T=64 (s=1), correlated neighbouring bins count as independent. `iid_gaussian` then produces finite spurious delays: 4/20 entries in the full band (A[0,2] = 0.716) and 6/20 in the high band.
  - At large T, narrowband coupling never yields 3 decimated points. `narrowband_f0-0.25` is all-NaN at T ≥ 1024.
- CANDIDATE: the strict band endpoints drop f = 0.25 from both half bands.
- CANDIDATE (pack hypothesis 3): in narrow EEG bands, 0 bins give a ValueError from `mstats.linregress` (`_mstats_basic.py:1167`, `np.amax` of an empty array), and 1 bin gives an all-NaN matrix.
- Hypothesis, not demonstrated: unwrapping across non-significant bins before masking could insert 2π jumps inside the fitted run.
- Already documented: signed output vs the "unsigned" label, the antisymmetric fill, and frequent all-NaN results (REVIEW M6; tests/README.md:249 partial_nan snapshots); the np.in1d shim (DEPENDENCY_COMPATIBILITY_REVIEW.md:51).
- NaN fractions by dataset (probe p5). `delayed_noise` is never NaN. `logistic_maps` is always NaN. `iid_gaussian` is all-NaN for T ≥ 256. `var2_bivariate` is NaN in the high band for T ≥ 256.

**Direction.** A[i,j] = +τ when row i leads column j by τ samples. Probe on `delayed_noise` (0-0.25 band, T=256): 1.033, 3.008, 5.081 for pairs (0,1), (0,2), (0,3), and A[1,0] = −1.033. For VAR coupling that is not a pure delay, the sign need not follow causality: `var2_bivariate` (0 drives 1) gives A[0,1] = −0.47.

**Proposed checks** (see the record):
1. Independent literal re-implementation: kernel, tolerance.
2. Direction on `delayed_noise`, with |error| < 0.2 samples.
3. Canonical OLS without FDR, cluster or decimation rules on `delayed_noise`.
4. Pair-alone vs embedded, exact: expected to fail.
5. False-positive rate on `iid_gaussian`, statistical.
6. Behaviour of the EEG band bin counts.
7. Narrowband NaN behaviour.
8. normalise/detrend paths.

## Open questions

1. Closed. Nolte 2008 equation numbers and sign were checked against arXiv:0712.2352v1, p.2: raw PSI is eq. 3, coherency eq. 4, weighted form eq. 5, jackknife normalisation eq. 6, and the sign rule follows eq. 2. The PRL journal copy was not consulted, so its numbering could in principle differ.
2. Gotman 1983: was the coherence threshold fixed or a test? Was the regression weighted? Was the intercept free or forced through zero? I have not re-checked any of these; they decide whether the library's FDR/cluster/decimation rules count as a variant or as an extension.
3. The EEG band identifiers (fmin 0.000488 etc.) look like Hz/1024. The exact fmin/fmax floats and the EEG window lengths (512/1024 at 1024 or 2048 Hz) should be confirmed from the EEG configuration, to settle which EEG identifiers are all-NaN by construction.
4. Should GD's dependence on the channel set (BH pooling) be treated as a defect, or as an intended multivariate FDR control? The library documents BH, but pyspi documents the SPI as bivariate.
5. Does `np.unwrap` before masking ever change the fitted slope on realistic data? This needs a crafted coherency test.
6. Wavelet PSI (`psi_wavelet_*`, MNE) belongs to another family. Its relation to this multitaper PSI (adjacent vs all pairs) is not traced here.

## Checks

All numbers below are from `audit/checks/out/results/spectral_psi_gd.json`. The run_ids are listed there in full. Files:
- manifest `audit/checks/manifest/spectral_psi_gd.json` (446 runs);
- reference `audit/checks/refs/spectral_psi_gd.py`, which runs in the comparison venv and imports neither pyspi nor spectral_connectivity;
- probes `audit/checks/probes/spectral_psi_gd_probe.py`.

Run with `.venv/bin/python audit/checks/run_checks.py --family spectral_psi_gd`. Result: 413 pass, 33 fail, 0 without reference, no script errors. Every failure is a deliberate comparison against the canonical definition or an invariance property; each one is explained below. Cost: about 40 s on the pyspi side and about 10 s for the references, with T up to 4096.

Shared multitaper coherency. The reference builds it from `_kernels_spectral`: scipy DPSS with NW=3 and K=5, constant detrend, numpy FFT, `S_ij = X_i conj(X_j)`, and the non-negative grid without the Nyquist bin. The data paths reproduce `pyspi/data.py`: `kernel` uses raw data, `pipeline` uses zscore with ddof=1, and `pipeline-detrend` applies a linear detrend before the zscore.

### spectral.psi

**What pyspi computes (independent replica, `psi_pyspi`).** The replica uses a strict band, a decimation step `ceil(((fmax-fmin)/50) n_fft)`, and Im Σ over all unordered pairs of the kept bins.
- It agrees with pyspi in 90/90 `rep` runs, 11/11 `alone`/`base` runs and 31/31 EEG tolerance runs. These cover 6 datasets × T 64/256/1024/4096 × 3 bands on the kernel path, plus `trending` and `scaled_offset` on all three paths.
- Largest absolute difference: 5.87e-10, in `spectral.psi|shared_envelope|4096|kernel|rep:low`, where |A| is up to about 800. results.json reports max_rel = 1.0 in the 12 `spectral.psi|shared_envelope|*|kernel|rep:*` runs. There x2 is bit-identical to x0, so pair (0,2) has pyspi 0.0 against a reference of at most 2.3e-15 in absolute value. Over the other 120 replica runs the largest max_rel is 3.86e-11 (`spectral.psi|iid_gaussian|4096|kernel|rep:low`). The NaN masks are identical in every run.
- Recorded keys on `trending`/`scaled_offset`:
  - Normalisation changes the value by at most 8.9e-14, as expected from scale invariance.
  - Detrending changes it by up to 23.7 (`trending`, low band).
  - The replica follows pyspi on all three paths.
- Closed-band counterfactual: the same replica with `fmin <= f <= fmax` differs from pyspi in all 101 replica runs (smallest max_abs 0.18). So the strict band is the one that runs.

**Canonical comparison.**

*Nolte 2008 eq. 3* (`psi_nolte`): adjacent bins at 1/n_fft, computed from the same coherency (`nolte:*` runs, criterion = sign agreement on every pair). 17 of 30 runs pass.
- `delayed_noise`, 5/5 runs:
  - Signs agree.
  - Per pair, pyspi is 7.59 to 275.3 times Nolte's value (`ref_ratio_pyspi_over_nolte_triu`). The minimum is pair (0,3) (delay 5) in `spectral.psi|delayed_noise|256|pipeline|nolte:full`; the maximum is pair (0,1) in `spectral.psi|delayed_noise|1024|pipeline|nolte:low`. The minimum is 8.97 at T=1024 and 10.1 at T=4096. Pair (0,1) in the full band: pyspi 580 / 694 / 787 against Nolte 3.061 / 3.094 / 3.081 at T = 256 / 1024 / 4096. The median ratio is 34 at T=256, 40 at T=1024 and 46 at T=4096 for the full band, and 143 for the half bands.
  - The Spearman rank correlation across pairs is −1.0 at T=256 and −0.943 at T=1024 and 4096 (full band), and −0.6 for the half bands. Example at T=1024, full band, pairs (0,1), (0,2), (0,3) (delays 1, 3, 5): pyspi gives 694, 230, 137; Nolte gives 3.09, 9.29, 15.27. pyspi's PSI therefore ranks delays in reverse.
- `gauss_ar1_pair`: 5/5 runs pass.
- `var2_chain`: 4/5 runs pass.
- `var2_bivariate`: 2/5 runs pass. The failures come from noise in the sample Nolte value, not from pyspi:
  - At T=1024, full band, Nolte = −0.48 with a jackknife z (eq. 6, leave one taper out) of −0.08.
  - The population Nolte value from `S_true` is +0.85.
  - pyspi's sign matches the population sign in every var2 run (`ref_sign_agree_pyspi_population` 1.0).
- `iid_gaussian` (0/5) and `narrowband_f0-0.1` (1/5): no broadband phase slope, so these sign disagreements carry no information.

*External package.* `mne_connectivity==0.9.0` `phase_slope_index` (multitaper, `mt_bandwidth = 6/T`, adjacent bins) agrees with our eq.-3 reference to 2.5e-3 absolute on `delayed_noise` and 3.6% relative on `gauss_ar1_pair`. The residual comes from MNE's taper-eigenvalue weighting. Its sign agrees with pyspi in 7/9 runs, and the two `var2_bivariate` failures are the same noise case as above.

*Direction.* On `delayed_noise` (`dir:*`), sign(A[i,j]) = sign(d_j − d_i) in 9/9 runs. The analytic pure-delay curve is ρ²·Σ_{a<b} sin(2πτ(f_b − f_a)) over pyspi's kept bins (`delaycurve:*`). It matches pyspi within max_rel 0.0234 in 6/6 runs.

*REVIEW M5 sign-difference example, reproduced* (`spectral.psi|delayed_noise|64|kernel|m5`). The probe runs the library's own `_inner_combination` on the four unit coherencies of REVIEW.md:174. It returns +1.11679427, equal to the all-pairs formula (+1.1167942658). The adjacent formula gives −1.3070153772. **Documented case: adjacent and all-pairs differ in sign on the REVIEW M5 phases [0.6925794593, −0.3522172326, 2.7390760913, 2.2241494312].**

*Sign reversal under long delays* (`spectral.psi|delayed_noise|1024|pipeline|longdelay:full`, fail, direction_agreement 0.7). The probe builds x0 from `delayed_noise` T=4096 and makes delayed copies with d = (0, 20, 45, 60, 100), T = 1024, full band.
- Pairs with leads of 55, 60 and 80 samples give A = −5.2, −9.6 and −25.1.
- Nolte at native resolution has the designed sign in 10/10 cases.
- Nolte on the decimated bins fails on the same pairs. The cause is therefore the decimation step s = 11 (spacing 11/1024), not all-pairs summation: the reported direction reverses whenever n_fft/(2s) < τ < n_fft/s (mod n_fft/s). That is 46.5 to 93 samples here; for the EEG full band (s = 3 at T = 1024) it is 171 to 341 samples. The consequence falls under REVIEW M5's "sign can differ", but M5 and the other already-documented items name only the all-pairs sum. The decimation mechanism is therefore recorded as a separate confirmed candidate (added in Workflow 4b-2; earlier versions of this note called it an elaboration of M5). Upstream spectral_connectivity master 138f3f7f0f replaces the all-pairs sum by the adjacent sum but keeps the decimation, so with pyspi's `(fmax-fmin)/50` argument the reversal would survive that rewrite.

*Bivariate in effect.* Adding 5 iid channels leaves the pair's value unchanged (`embed:*` 8/8, max_abs 0.0), and permuting the channels is exact (`perm:*` 3/3).

*Band forwarding.* A spy probe (`spy:low`) records 2 library calls: the first without kwargs, the last with `[0, 0.25]` and resolution 0.005.

*EEG bands* (probe constructs `PhaseSlopeIndex(fmin, fmax)` from `pyspi/workflow/retained185.yaml`; `delayed_noise` at T = 256/1024/4096 and `iid_gaussian` at T = 1024, pipeline-detrend). 40/40 runs pass.

| T | bands | in-band bins | outcome |
|---|---|---|---|
| 256 | 0.5–2, 2–4, 4–6 /1024 | 0 | IndexError, all-NaN |
| 1024 (the EEG window: 0.5 s at 2048 Hz) | 0.5–2, 2–4, 4–6 /1024 | 1 | IndexError, all-NaN |
| 256 | 6–15 /1024 | 2 | finite; the single adjacent term |
| 4096 | all 10 bands | ≥ 5 | finite |

All finite cases equal the replica. So 3 of the 10 EEG PSI identifiers are all-NaN by construction at the EEG window length: `psi_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195`, `psi_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391` and `psi_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586` (1 bin at T = 1024; 0 bins at T = 512 by the bin count k/512, not run). Rubric anchor 0 (all-NaN) applies to these three identifiers; the unit score below describes the identifiers with at least 2 kept bins. The record's `check_cost` names them, because the unit-level score cannot.

**Verdict.** `related-quantity`, score **2**. Rubric anchor 2 names this case directly: "substantially non-standard formula whose sign can differ from canonical (all-pairs PSI)". `check_method` is independent-impl (MNE is an external-package cross-check of the canonical side). `definition_under_review` is true. Verdict word: dcoh and ddtf (spectral_directed) are `wrong` because the delegate formula differs from the paper it cites; PSI is in the same position (upstream master 138f3f7f0f also rewrites `phase_slope_index` to Nolte's adjacent sum). `related-quantity` is kept here because REVIEW M5 treats the all-pairs form as a pending definition decision (keep it under an explicit identity, or adopt standard PSI); the score is anchor 2 under either word.

**Candidates.**
- **Confirmed:** band forwarding under 1.1.2. This refines a known item and is not new.
- **Confirmed, new:** `mean` is never applied.
- **Confirmed, new:** narrow bands raise IndexError and give all-NaN.
- **Confirmed:** strict endpoints. Related to the endpoint item already listed (EEG_OPTIMIZATION_NOTES §review, REVIEW M4).
- **Confirmed, new, minor:** the result is a view of the cache. The `shares:full` probe shows: shares_memory = 1. A mutation of the second result reappears on the third call. A mutation of the first result does not, because the first insert is keyed by `self.measure` (already documented as a "worth knowing" item).
- **Confirmed, new (added in Workflow 4b-2):** the band-relative decimation step reverses the reported direction for delays between n_fft/(2s) and n_fft/s samples (`longdelay:full`: Nolte on the decimated bins −17.49 / −24.61 / −17.52 on the same three pairs as pyspi, Nolte at native resolution the designed sign in 10/10).
- Already documented, not new: the labels (REVIEW M1 and M5; tests/README.md bug 5 before b2f6d7e) and all pairs (M5, including its "sign can differ" consequence).

### spectral.gd

**What pyspi computes (independent replica, `gd_pyspi`).** The replica implements:
- a Fisher z with bias 1/8 (5 tapers), one-sided normal p;
- hand-written Benjamini-Hochberg at 0.05 over the flattened frequency × pair array;
- the largest contiguous run (first on ties), keeping every s-th bin, with at least 3 points;
- `numpy.unwrap` over the in-band grid, then OLS, then slope/2π, with A[j,i] = −A[i,j].

It agrees with pyspi in 114/114 `rep` runs (8 datasets × 4 T × 3 bands, plus `trending`/`scaled_offset` on 3 paths), 11/11 `alone`/`base` runs and 37/37 EEG tolerance runs. Largest absolute difference: 4.2e-11 (`spectral.gd|delayed_noise|4096|pipeline-detrend|eeg:0.00195312-0.00390625`). results.json reports max_rel = 1.0 in the 12 `spectral.gd|shared_envelope|*|kernel|rep:*` runs. There pair (0,2) is a degenerate identical-channel pair: pyspi gives 0.0 and the reference at most 8.3e-18 in absolute value, and max_abs in these runs is at most 1.3e-12. Over the other 150 replica runs the largest max_rel is 3.7e-11 (the same EEG run). The NaN masks are identical in every run, including the all-NaN ones. Normalisation changes the delay by at most 4.4e-15; detrending changes it by up to 5.17, and changes the NaN pattern on `trending`.

**Canonical comparison.**
- *Pure delays.* `delayed_noise` direction 9/9. |A − (d_j − d_i)| is at most 0.081 samples (T=256, low band) and 0.045 at T=1024 (`delay:*` 9/9). The plain Gotman-style OLS over every in-band bin (`gotman:*`) agrees within 0.033 (9/9). On clean data the extra rules do not bias the estimate. The long-delay probe recovers delays up to 80 samples (60.03, 77.94); the pair with a 100-sample delay is NaN.
- *Null* (`null:*`). All-NaN for T ≥ 256 (9/9 runs) and in the T=64 low band. At T=64, 4/20 entries are finite in the full band and 6/20 in the high band, with |delay| up to 0.96. With the step set to 2W·n_fft = 6 the replica gives 0/20. In the EEG bands 0.5–35 and 15–35 /1024 on `iid_gaussian` at T=1024, 4/20 entries are finite (step 1) and 0 with the 2W step.
- *Population VAR delay* (`popgd:*`, 3/12 pass). The true spectrum gives group delays of +1.13 (var2_bivariate) and +1.13 / +2.21 / +1.09 (var2_chain pairs), so X leads. pyspi gives −0.47 and −0.56 for var2_bivariate at T=256 (wrong sign) and NaN in the full band at T ≥ 1024. The population |C| is 0.39 to 0.62. A per-bin one-sided 5% test with 5 tapers needs |C| > 0.61 even before BH, and its power does not grow with T. The plain OLS over all bins also degrades at large T (unwrap errors on noisy phase), so it is not a usable canonical reference there either.
- *Channel-set dependence* (`embed:*`, 6/8 fail). Pair alone versus pair plus 5 iid channels:

  | dataset | T | band | alone | with 5 iid channels |
  |---|---|---|---|---|
  | var2_bivariate | 256 | full | −0.4725 | NaN |
  | var2_bivariate | 256 | low | −0.5550 | −0.0427 |
  | narrowband_f0-0.1 | 256 | full | −0.0277 | −0.9866 |
  | narrowband_f0-0.1 | 256 | low | −0.646 | −0.668 |
  | gauss_ar1_pair | 1024 | full | −1.7927 | NaN |
  | gauss_ar1_pair | 1024 | low | 0.0607 | NaN |

  `delayed_noise` is unchanged. The replica reproduces every embedded value, so the mechanism is the pooled BH threshold. Permutation is exact (3/3).
- *Narrowband / strict endpoints* (`excl25:*`, 6/6). For `narrowband_f0-0.25` at T ≥ 1024, the half bands do not contain f = 0.25 and are all-NaN. The full band does contain 0.25 and is also all-NaN, because its significant run (19 bins at T=1024, 29 at T=4096) yields only 2 or 1 decimated points.
- *NaN fractions per band and T* (from `ref_n_finite` of the `rep:*` runs; there is no dedicated run family for `narrowband_f0-0.1` or `shared_envelope`). Counts are finite off-diagonal entries out of P(P−1); the replica NaN masks equal pyspi's in all of them.

  | dataset (P) | band | T=64 | T=256 | T=1024 | T=4096 |
  |---|---|---|---|---|---|
  | narrowband_f0-0.1 (3) | full | 6/6 | 6/6 | 6/6 | 0/6 |
  | narrowband_f0-0.1 (3) | low | 6/6 | 6/6 | 6/6 | 6/6 |
  | narrowband_f0-0.1 (3) | high | 0/6 | 0/6 | 0/6 | 0/6 |
  | shared_envelope (4) | full | 12/12 | 6/12 | 6/12 | 6/12 |
  | shared_envelope (4) | low | 12/12 | 8/12 | 6/12 | 6/12 |
  | shared_envelope (4) | high | 6/12 | 6/12 | 6/12 | 6/12 |
  | narrowband_f0-0.25 (3) | each band | 6/6 | 6/6 | 0/6 | 0/6 |

  For `shared_envelope`, the NaN entries are the pairs with x3, which has its own random-walk phase. This matches the trace-stage expectation for `narrowband_f0-0.1`: NaN in the 0.25–0.5 band at every T and in the full band at T=4096.
- *EEG bands* (40/40 pass).

  | T | bands | in-band bins | outcome |
  |---|---|---|---|
  | 256 | 0.5–2, 2–4, 4–6 /1024 | 0 | ValueError from `mstats.linregress`, all-NaN |
  | 256 | 6–15 /1024 | 2 | all-NaN |
  | 1024 | the same three narrowest bands | 1 | all-NaN |
  | 4096 | all bands | | finite on `delayed_noise` |

**Verdict.** `variant`, score **6**. Rubric anchor 6 names "group-delay significance machinery". The OLS core is exact on pure delays, but bin selection by FDR pooled across pairs, the largest run and the band-relative step deviates from both the supplement ("bivariate", "statistically significant coherence values") and the plain canonical estimator, in ways that are interpretable. `check_method` is independent-impl. `definition_under_review` is true.

**Candidates.**
- **Confirmed, new:** not bivariate (pooled BH).
- **Confirmed:** pipeline replica.
- **Confirmed:** narrow bands fail. This diagnoses part of REVIEW M6's all-NaN cases.
- **Confirmed, new:** band-relative step.
- **Confirmed:** f = 0.25 exclusion, though it is not the only cause of NaN.
- **Refuted:** the unwrap hypothesis. The difference between unwrapping over the grid and over the run only is at most 2.8e-14 across the 125 replica runs (as expected: within one contiguous run the two unwraps differ by a constant 2πk, which the free intercept absorbs).
- **Confirmed, new, minor:** the result is a view of the cache.

**Trace-stage fields corrected in the record.**
- `direction_convention`: the var2 sign statement now says that the population group delay is positive and that the negative sample value is an artefact of the selection. A Workflow 4b challenge probe (scratchpad `wf4b_mid-eeg/gd_localslope.py`) supports this: on the 7-bin run pyspi kept at T = 256 (f = 0.176–0.199) the true coherency phase has a local slope of +1.53 samples, against −0.4725 (full band) and −0.5550 (low band) from the replica, so the negative sign is not a local property of the true phase.
- `deviation_from_canonical`: the unwrap hypothesis is marked as refuted.
- `already_documented`: DEPENDENCY_COMPATIBILITY_REVIEW.md:48 corrected to :51, and tests/README.md:220 corrected to :249 (the `partial_nan` sentence; table rows :481-483).
- `symmetry_observed`: the run count is now 234 square GD matrices in out/pyspi (158 with at least one finite off-diagonal entry). The earlier count of 222 could not be reproduced. The quoted 155.9 and 0.0 hold on all 234.
- spectral.psi `deviation_from_canonical` (a): the adjacent-pair values now quote the check runs (3.061 / 3.094 / 3.081 at T = 256 / 1024 / 4096). The T = 64 value of 2.75 is marked as a trace-stage probe value with no check run behind it.
- Both records and this note now cite `pyspi/calculator.py` at HEAD a1f20c7 (:295-355, :230-258).

The other remaining trace issues (DEPENDENCY_COMPATIBILITY_REVIEW heading line, the "pack lines" range) already pointed at the cited lines in the records: :74 and :84, and supplement.txt 644-668.

### Kernel notes

- I found no defects in `_kernels_spectral`.
- `group_delay_fit` unwraps after masking. That differs from the library whenever masked points are more than π apart in phase, so it is used here only without a mask (every in-band bin) and on population spectra.
- `psi_all_pairs` raises ValueError when fewer than 2 bins remain. The reference uses the bin count to choose the qualitative branch instead.

## Open questions: answered

1. **Nolte 2008 equation numbers and sign.** Already closed against arXiv:0712.2352v1. The check adds an external confirmation of the eq. 3 implementation and its sign: MNE's adjacent-bin PSI agrees with our eq.-3 reference to 2.5e-3 (`mne:*`), and positive values mean the row leads on `delayed_noise`. The PRL journal copy is still not consulted.
2. **Gotman 1983 procedure** (threshold, weighting, intercept). Not settled: the paper was not available in this stage. The checks make the stakes concrete:
   - On pure delays the library rules do not matter (pyspi vs plain OLS ≤ 0.033).
   - On a moderately coherent VAR they decide between a wrong sign and NaN (`popgd:*`).

   The score is anchored on the rubric's "group-delay significance machinery" either way. Reading Gotman's methods section would settle whether the FDR/cluster/decimation rules are a variant or an extension.
3. **EEG band floats and window lengths.** Confirmed:
   - `pyspi/workflow/retained185.yaml` gives fmin/fmax = 0.00048828125 … 0.1220703125, which are 0.5/1024 … 125/1024 cycles/sample; its line-4 comment says the bands are not rewritten from the EDF rate.
   - `examples/workflow/preflight.yaml` uses 2048 Hz with 0.5 s windows, so T = 1024.
   - At T = 1024 the bands 0.5–2, 2–4 and 4–6 /1024 hold one bin each: PSI raises IndexError and GD is all-NaN, i.e. 3/10 EEG identifiers per unit are all-NaN by construction (`eeg:*` runs). The 6–15 band holds 8 bins. GD gives false-positive delays on iid noise (4/20 entries) in the step-1 bands 0.5–35 and 15–35.
   - At T = 512 (not a dataset length here) the bin count k/512 gives 0 bins for the same three bands, i.e. an exception.
4. **Is GD's channel-set dependence intended?** The dependence is confirmed and reproduced by the replica. The library applies BH over everything it is given, so the behaviour is "intended" at library level. pyspi documents the SPI as bivariate, so at pyspi level it is a deviation from the documented definition. Whether to restrict FDR to each pair is a maintainer decision; the evidence is the `embed:*` runs.
5. **Unwrap before masking.** Answered: refuted, as stated under spectral.gd candidates above.
6. **Wavelet PSI.** Not covered here; it belongs to the wavelet family. For comparison, the canonical adjacent-bin multitaper PSI (Nolte eq. 3, as in MNE) is now available as `refs/spectral_psi_gd.py:psi_nolte`.
7. **`git status` shows EEG_OPTIMIZATION_NOTES.md as modified.** Not touched by the check stage either. Files written in this stage:
   - the manifest, reference and probe listed above;
   - the two records;
   - this note;
   - outputs under `audit/checks/out/` (pyspi, ref, exchange/spectral_psi_gd, results).

## Optimisation

Sources: walls from `audit/profiles/summary.json` (fresh / family-warm; quiet machine; 118x1024 walls are single repeats), frame shares from cProfile. The official profiles at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`) are dominated by the P-independent DPSS cost for every multivariate unit, so for ranking at 118 channels this group added cProfile runs at 118x1024 (32x1024 for sgc nonparametric): `audit/profiles/micro/prof_<unit>__<shape>.pstats`, written by `audit/profiles/micro/prof_spectral.py`, indexed with walls, peak RSS and thread settings in `audit/profiles/micro/prof_spectral__index.json` (OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1; no numba or JIDT code on these paths). Those runs overlapped other Workflow 3 groups; the memory-bound multitaper units varied 2-7x in wall between runs (cohmag: 6.4 s on a first run, 0.93 s on the kept run, 2.42 s in summary.json), so only shares are quoted from them. Percentages are of the profiled `harness.compute_spi` frame. Micro-benchmarks (the group's two): `audit/profiles/micro/opp_spectral_csm-1.{py,json}` and `audit/profiles/micro/opp_spectral_csm-shared-csm.{py,json}`. A candidate that reproduces the current output "matches the baseline"; that says nothing about whether either is mathematically right. Nothing in `pyspi/` was changed.

Both units carry `definition_under_review: true` (narrow EEG bands at T = 1024: delta spans 2 inclusive / 1 strict bins, theta and alpha 3 / 1; PSI raises IndexError and GD is all-NaN there, see spectral_csm.md). The avenues below change no band rule and no estimator; they are recorded with that review open and are not tuned to the bins the review may change.

| unit | 8x1024 fresh / fw (s) | 32x1024 fresh / fw (s) | 118x1024 fresh / fw (s), mode | peak RSS 32 / 118 fresh (MB) |
|---|---|---|---|---|
| spectral.psi | 0.0216 / 0.0654 | 0.1474 / 0.4188 | 4.440 / 13.543, measured | 457 / 1883 |
| spectral.gd | 0.0288 / 0.0844 | 0.2116 / 0.5868 | 4.047 / 11.060, measured | 504 / 1770 |

Family-warm increments at 118x1024: psi 4.51, 4.62, 4.41 s for its three bands; gd 3.97, 3.50, 3.59 s. Unlike the other multitaper units every band pays in full because the band is in the cache key (`spectral.py:46-48`).

**Hotspots at 118x1024.**
- psi (`audit/profiles/micro/prof_spectral.psi__118x1024.pstats`): `phase_slope_index` 98.4% over 2 calls; `coherency` 83.6% over 2 evaluations (the failing default-argument call evaluates coherency before `_bandpass` raises, `connectivity.py:1091-1093`), `_complex_inner_product` 39.0% (2 builds), reduce 16.1%, `_inner_combination` (all frequency pairs) 10.3%, `dpss_windows` 1.0%. At 8x1024 `dpss_windows` 65.2% and `_inner_combination` 1.9%.
- gd (`audit/profiles/micro/prof_spectral.gd__118x1024.pstats`, 4.12 s profiled, close to the 4.05 s fresh wall): `numpy/ma/extras.py:apply_along_axis` -> `scipy.stats.mstats.linregress` 41.3% (6903 calls), `_find_significant_frequencies` 33.4% of which `np.apply_along_axis(_find_largest_independent_group)` 25.7% (scipy.ndimage.label per pair), `coherency` 18.7%, `_complex_inner_product` 8.6%. Masked-array bookkeeping (`numpy/ma/core.py:__new__` 18.8%, `__array_finalize__` 13.4%) sits inside those callbacks. At 8x1024 `dpss_windows` 41.7%. cProfile inflates the per-pair Python callbacks (2.76 s profiled against 1.32-1.55 s un-profiled per band in the Workflow 4c paired run), so these shares overstate them and understate the coherency; the gd opportunities below use the un-profiled walls.
opaque_share (8x1024 JSON): psi 0.127, gd 0.204 (lower bounds).

**Complexity.** psi per band: FFT + 2 x O(K n_fft P^2) CSM + O(m^2/2 P^2) frequency-pair combination over m <= ~50 bins; gd per band: FFT + O(K n_fft P^2) CSM + P(P-1)/2 Python callbacks (cluster search + masked regression over the band bins). Measured T exponents 0.90 (psi) and 0.45 (gd: the per-pair Python cost does not grow with T).

**Opportunities.**
- psi: `spectral_csm-shared-csm` 6.7-8x (bit-identical for all three config bands in `opp_spectral_csm-shared-csm.json`). In an EEG run psi is a later consumer: CoherencePhase runs first (`pyspi/workflow/retained185.yaml:218`, PhaseSlopeIndex at `:308`) and builds the shared coherency, so both coherency evaluations of every band read it and psi keeps its per-band remainder; the record's 6.7x lower bound is that EEG-order figure on the ranking basis (README 'Optimisation columns', rules 1 and 4). (a) By the shares of the profiled band (`prof_spectral.psi__118x1024.pstats`, 1.74 s: one coherency plus the Fourier transform 43.3%, the per-band remainder 14.9%) at summary.json scale, the scale the ranking uses: the remainder is 0.149 x 4.51 = 0.67 s per band, so 3 config bands 13.54 -> 2.0 s and 11 EEG bands 49.7 -> 7.4 s (6.7x each); if psi built the shared coherency itself it would keep one build, 3 config bands 13.54 -> 4.0 s (3.4x) and 11 EEG bands 49.7 -> 9.4 s (5.3x), as in the per-window paragraph below. (b) Same run (Workflow 4c, `opp_spectral_csm-shared-csm.json` key `paired`, 118x1024, 3 repeats on a machine shared with other agents): psi's 3 config bands through pyspi against the 3 psi segments reading one shared coherency gave 6.5x, 17.5x and 40x (baseline 32.7, 35.2 and 22.8 s; psi segments 5.06, 2.01 and 0.57 s, following the memory state of each repeat), and a challenge re-run of the same comparison (`scratchpad/wf4c_b1spec/paired.py`, run as `paired.py 118x1024 3`; its output `paired_118x1024.json` is not merged into the micro-benchmark JSON and the probe archive keeps scripts only, so the record quotes the per-repeat figures) 7.1-7.7x; the median of the six, about 7.7x, is the 8x upper end, and one repeat (6.5x) fell just below the 6.7x lower bound. With the build counted (psi as first consumer) the same run gave 3.3-6.4x (3.5-3.7x at 32x1024). The earlier 8.6-10.5x set the 'shapes' run's segments against the profiled band of another run and is not used for the range. The micro-benchmark's 'shapes' run gives 2.14x (14.55 -> 6.80 s, single pass) for its whole 8-identifier set (cohmag, phase, psi x3, gd x3), whose candidate time is mostly the gd bands. For scale only: psi's configured basis sums 3 band builds where an EEG run pays 11 (the band is in the cache key); on an 11-band basis (49.7 s per window) the 6.7x would save about 25,300 patient-s instead of about 6,900, more than any row ranked above it (record `basis_scope_note`). `spectral_psi_gd-2` ~1.7x on its own (call the band statistic directly so the discarded first call no longer evaluates coherency: 41.8% of a band; S effort; unverified-bit-identical). It is contained in `spectral_csm-shared-csm`: once the coherency is memoised per Data the discarded call costs almost nothing, and the shared-csm candidate already calls `phase_slope_index` with its arguments (`opp_spectral_csm-shared-csm.py:63-65`), so its figures include this change. The two savings do not add; the record lists shared-csm under `subsumed_by`, so build_csv ranks psi_gd-2 with no saving on top of it. `spectral_csm-1` ~2x on the remaining coherency (a profile basis with the cohmag figure carried over, not tagged microbench; unverified-within-tolerance: psi was not run separately; the cohmag measurement on the same coherency path is within 1.4e-16). The ranking stacks it after `spectral_csm-shared-csm` (`depends_on`) with `residual_speedup` 1x, on the 3 configured bands and the 11 EEG bands alike: CoherencePhase runs before PhaseSlopeIndex in an EEG run (`pyspi/workflow/retained185.yaml:218` and `:308`), so phase builds the shared coherency, psi's residual holds no coherency build for this avenue, and its saving on that one build is credited on spectral.phase.
- gd: `spectral_csm-shared-csm` 2-3.6x and `spectral_psi_gd-1` 1.3-1.6x, both re-derived in Workflow 4c from un-profiled timings (`opp_spectral_csm-shared-csm.json` key `paired`, 118x1024, 3 repeats). The cProfile run above overstates the per-pair callbacks: they took 2.76 s of the 4.12 s profiled band, while the un-profiled callback drivers (`np.ma.apply_along_axis` -> `linregress` and `np.apply_along_axis` -> `_find_largest_independent_group`, timed by wrapping the two numpy entry points) took 1.32-1.55 s per band in every repeat, 22-29% of the pyspi gd bands of that run (4.6-6.6 s per band). gd is a later consumer of the shared coherency in an EEG run (GroupDelay at `retained185.yaml:518`); its 3 bands reading one shared coherency cost 5.16-5.25 s against 14.4-19.2 s through pyspi (2.8-3.7x same run, bit-identical, NaN masks equal, also at 8x1024 where the output is not all-NaN), and 11.06/5.25 = 2.1x on the ranking basis (family-warm increments 3.97, 3.50, 3.59 s), the 2x lower bound. `spectral_psi_gd-1` (vectorised masked least squares and a vectorised largest-cluster / independent-step / min-group search; unverified-within-tolerance, the NaN pattern of `min_group_size` and empty masks has to be reproduced) removes 4.22 s of the 11.06 s ranking basis (38%, ceiling 1.62x if the vectorised code cost nothing; 1.28-1.41x in the same run, where the memory-bound coherency was slower); the 1.3x lower bound leaves about 0.55 s per band for the vectorised code. Its share depends on the memory state (the callbacks are compute-bound, the coherency is not): a challenge cProfile re-run under load gave coherency 60.6% and callbacks 32.5%. The synthetic 118x1024 data give an all-NaN gd output in every band and the 32x1024 data in one of the three config bands (`fmin-0_fmax-0-25`; `paired.<shape>.diffs.*.all_nan` in the JSON; empty significance masks); at 8x1024 a pair's callbacks cost 0.27 ms against 0.20-0.21 ms at the larger shapes, so EEG data with significant clusters may raise the callback share by about that factor, and a diff of a `spectral_psi_gd-1` candidate should use 8x1024 or the 7x100 set. The earlier figures (2-3x for `spectral_psi_gd-1`, ~1.2x for shared-csm) came from the profile shares and are superseded. gd's hotspot owner is recorded as the call site (spectral_connectivity, where the replacement lands); the time itself is in numpy.ma and scipy.stats frames. Both records list the other avenue as `independent_of` (disjoint frames); with the larger lower bound shared-csm is stacked first, and `spectral_psi_gd-1`'s `residual_speedup` 1.8-4x gives its figure on the residual: the 2.55 s per window that its 1.3x lower bound removes from today's 11.06 s, taken off the 5.53 s residual, is 1.86x (README 'Optimisation columns', rule 4); its ceiling there is about 5x (callbacks 4.22 of 5.16-5.25 s).

**EEG per-window cost at 118 channels.** psi: 11 EEG identifiers x 4.51 s = 49.7 s (three of them, 0.5-2, 2-4 and 4-6 /1024, raise IndexError after paying the coherency). gd: 10 EEG identifiers x 3.69 s = 36.9 s. With `spectral_csm-shared-csm` (one coherency per Data for all bands): psi about 0.77 + 11 x 0.165-0.203 = 2.6-3.0 s at micro-benchmark scale (against 11 x 1.74 = 19.2 s for the profiled band, 6.4-7.4x), or 1.95 + 11 x 0.67 = 9.4 s at summary.json scale by the profiled shares (43.3% once per Data, 14.9% per band; ~5.3x), if psi builds the shared coherency itself; in an EEG run CoherencePhase builds it first, so psi pays only its remainder, 11 x 0.67 = 7.4 s at summary.json scale (6.7x, the figure the ranking uses) or 11 x 0.165-0.203 = 1.8-2.2 s at micro-benchmark scale; gd about 10 x 1.75 = 17.5 s as a later consumer at the ranking scale (2.1x; the gd segments reading the shared coherency cost 1.55-2.08 s per band in both micro-benchmark runs). Adding `spectral_psi_gd-1`: gd about 10 x 0.35 = 3.5 s (~10x) if the vectorised code cost nothing, by the un-profiled callback walls (1.32-1.55 s of the 1.55-2.08 s band segments). The historical per-window figures (gd about 47 s for 10 bands, psi about 14 s, channel count unknown) are of the same order as the gd figure here; the psi figure here is larger, consistent with more channels or fewer bands in the historical runs (order-of-magnitude comparison only).

**Refuted seeds.** Vectorised frequency-pair combination for psi: `_inner_combination` is 10.3% of a band at 118x1024 (1.9% at 8x1024), so it only becomes the main remainder after the coherency is shared (0.16-0.21 s per band) and is not listed. DPSS cache: 1.0% (psi) and below 1% (gd) at 118. The historical target "gd about 47 s -> 1-2 s per window" is not supported by the Amdahl bound at 118 channels: shared coherency plus vectorisation leave Fisher z, Benjamini-Hochberg, unwrap/angle and bandpass (about 14% of a band) per band, about 6-8 s for 10 bands by the cProfile shares and about 3.5 s by the un-profiled walls of the Workflow 4c paired run, unless those are batched across bands too.
