# wavelet_measures family: define, trace and check

Units: `wavelet.cohmag`, `wavelet.phase`, `wavelet.icoh`, `wavelet.plv`, `wavelet.ppc`, `wavelet.pli`, `wavelet.dswpli` (identifier `dspli`), `wavelet.wspli`, `wavelet.dwspli`, plus the infrastructure unit `wavelet.morlet_cwt`.
Each SPI unit covers six identifiers: `{mean|max}` x bands `fmin-0_fmax-0-5`, `fmin-0_fmax-0-25` and `fmin-0-25_fmax-0-5`.
Code: `pyspi/statistics/wavelet.py:15-190` (classes `mne`, `CoherenceMagnitude` … `debiased_weighted_squared_PhaseLagIndex`).
Delegates: mne-connectivity 0.9.0 `spectral_connectivity_epochs(mode='cwt_morlet')`, and mne 1.13.2 `morlet`/`cwt` (scipy 1.17.1, numpy 2.5.4).
Probe scripts are in the session scratchpad under `wavelet_measures/p1_morlet.py` … `p6_order.py`. The adversarial review's probes are `fable_wl/wav1.py` … `wav4.py`; the follow-up probe is `wl_fix/fx1.py`. The record generator is `gen_records.py`.
The regression goldens in `tests/regression/goldens/extra/` are snapshots of current output. Nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Status of the documentation

None of the nine classes is documented as an SPI:

- They are absent from `pyspi/config.yaml`. `tests/README.md:155` says they "run today but are absent from `pyspi/config.yaml`", and they are exercised only by `tests/regression/extra_config.yaml`.
- They are absent from the supplement's SPI list and from the GitBook page.
- Supplement §1.5.4, p.14 (supplement.txt heading line 644, text lines 649-651), and GitBook `spectral.md:330` say the opposite: PSI "is the only spectral measure that is computed in the time-frequency domain (denoted by modifier wavelet)".

So the "documented definition" for each unit below is the text for the multitaper SPI of the same name, plus the shared §1.5 band rule. Supplement §1.5 preamble, p.13 (supplement.txt heading line 570, text lines 589-599): "125 uniformly sampled bins across the entire frequency range, f ∈ [f0, fs/2] … f0 = 4/T … the mean … and the maximum … over three ranges". The three ranges are [0, fs/2], [0, fs/4] and [fs/4, fs/2].

## Shared implementation (unit `wavelet.morlet_cwt`)

1. **Preprocessing.** `pyspi/data.py:184-195` z-scores each channel (ddof=1), after an optional detrend.
2. **Single epoch.** `wavelet.py:52`: `z = np.moveaxis(data.to_numpy(), 2, 0)` turns (P, T, 1) into (1, P, T). An ordinary `Data` object is therefore **one epoch**. With `dim_order='psr'` the realisations become epochs.
3. **Wavelet settings.** `wavelet.py:54-55`: `cwt_freqs = linspace(0.2, 0.5, 125)` and `cwt_n_cycles = cwt_freqs / 7`, which is 0.029–0.071 cycles.
4. **Call.** `wavelet.py:56-67` calls `spectral_connectivity_epochs(..., mode='cwt_morlet', sfreq=fs, mt_adaptive=True, fmin=5/T, fmax=fs/2, ...)`.
   - `epochs.py:216` → `:97-113`: `fmin` only warns.
   - `:219` → `:124-135`: `freqs = cwt_freqs`.
   - `:222` → `:141-153`: inclusive mask. All 125 frequencies are kept when T ≥ 25. Below that the low band can become empty: through pyspi, `CoherenceMagnitude(fmin=0, fmax=0.25)` keeps 78 frequencies at T = 16 and returns NaN with "Mean of empty slice"; at T = 24 it keeps 121 and returns 1.0 (`fable_wl/wav4.py`, rerun).
   - `:345-365`: `morlet(sfreq, freqs, n_cycles, zero_mean=True)`. `mt_adaptive` is used only in the multitaper branch (`:337-341`).
5. **The Morlet wavelet collapses to one sample.**
   - `tfr.py:159`: σ_t = n_cycles/(2πf) = 1/(14π) = 0.0227 samples for every f.
   - `:165-166`: `t = arange(0, 5σ_t, 1) = [0]`.
   - `:167-172`: the zero-mean offset leaves 1 − e^{−2(πfσ_t)²} ≈ 4e-4 … 2.5e-3.
   - `:178`: normalisation makes **W = √2**.
   - Probe: "n wavelets 125 lengths [1]", "W values unique (real,imag): [1.41421356] [0.]". With n_cycles = 7 the wavelets would be 23–55 samples long.
6. **The CWT is a scaled copy of the input.** `mne_connectivity/spectral/epochs.py:443-457` → `tfr.py:766-794` `cwt` → `:797-806` `_cwt_array` → `:322-401` `_cwt_gen` (FFT, `ifft(fft_x * fft_W)` at `:377`, `_centered` at `:400`).
   - X(f,t) = √2·x(t) at all 125 frequencies, up to round-off.
   - Probe: max |Re X − √2x| = 2.7e-15 and max |Im X| = 7.2e-16. About 2–5 % of the imaginary parts are exactly 0.
7. **Cross-spectrum.** `epochs.py:276` computes only lower-triangular connections (`np.tril_indices(P, −1)`). `:652` computes csd = X_hi · conj(X_lo) = 2 x_hi(t) x_lo(t) + i·(round-off). `:457` computes psd = 2x². `:1472` sets n_epochs = 1.
8. **Assembly.** `mne_connectivity/spectral/epochs.py:1555-1567` assembles a dense (P, P, 125, T) tensor with the upper triangle set to 0.
9. **Summary.** `wavelet.py:76` keeps f in [fmin, fmax] on the 0.2–0.5 grid: 21 bins for the "low" band (0.2–0.248), 104 for the high band and 125 for the full band. `wavelet.py:84` applies nanmean or nanmax over **frequency and time**.
10. **Symmetrisation.** `wavelet.py:87-89` mirrors lower → upper and sets the diagonal to NaN. Every A[i,j] = A[j,i] therefore comes from conn[max, min]. These classes are symmetric; the mirror in `mne.multivariate` precedes `Undirected.multivariate` in the MRO, so REVIEW M1 (tests/README.md bug 5 before b2f6d7e) does not apply here.
11. **Cache.** Results are cached per measure in `data.mne[measure]`. The key ignores fs.

The pack's hypothesis is confirmed: the time-frequency decomposition does not exist. Every downstream quantity is a function of the instantaneous product x_i(t)x_j(t) and of the round-off in the imaginary part.

The degeneracy has **two** independent causes:

- **Single epoch.** This is the cause documented in `tests/README.md:217`. With n_cycles = 7 and one epoch, coh, plv, pli and wpli are still 1, ppc and dspli are NaN or inf, and dwpli is 0 (probe p4).
- **One-sample wavelet.** This is not documented. Even with 8 realisations as epochs, the frequency axis carries nothing (spread 4.4e-16) and PLI/wPLI become averages of round-off signs (probe p4).

Hypothesis about where the setting came from: mne-connectivity's example `examples/cwt_sensor_connectivity.py` contains exactly `cwt_freqs = np.arange(7, 30, 2); cwt_n_cycles = cwt_freqs / 7.0`, written for frequencies in Hz (fetched from GitHub). `mt_adaptive=True` has no effect in this mode: outputs were bitwise identical with `False`.

## Per unit

### cohmag (`CoherenceMagnitude`, measure `coh`)

**Documented.** Supplement §1.5.1, p.13: "the frequency domain equivalent of the squared time-domain cross correlation".

**Canonical.** C = |⟨S_xy⟩|/√(⟨S_xx⟩⟨S_yy⟩), with ⟨·⟩ averaged over trials or over a time/scale window before normalising (Nolte et al. 2004; Bastos & Schoffelen 2016). For a single trial, see Lachaux et al. 2002 (Neurophysiol Clin 32:157) and Torrence & Webster 1999.

**Implemented.** `epochs_bivariate.py:71-81` computes |2x_ix_j| / (2|x_i||x_j|) = 1 at every (f,t).

**Output.** Exactly 1.0 for every pair, band and statistic, on all 19 datasets at T = 64 and T = 1024.

**Deviation.** Degenerate, as already documented. If the per-point normalisation came after time averaging, the result would be |Pearson r| at lag 0 in every band (an unprobed hypothesis).

**Checks.** An exact equality-to-1 check. A from-scratch numpy pipeline using direct convolution. A canonical numpy wavelet coherence (n_cycles = 7, time-averaged) on bivariate_gaussian, narrowband, delayed_noise, iid_gaussian and var2_bivariate, which is expected to disagree. Band invariance. Channel-order invariance.

### phase (`CoherencePhase`, measure `cohy`, `np.angle` applied before the statistic)

**Documented.** Supplement §1.5.2, p.13: the coherence phase "has been used to infer a time-delayed dependence".

**Canonical.** arg⟨S_xy(f)⟩ per frequency. It is antisymmetric and circular.

**Implemented.** The coherency is sign(x_hi x_lo) + i·ε, so each angle lies within 4.6e-14 of 0, +π or −π, but most angles are not exactly equal to one of them (iid_gaussian T = 256 pair (1,0): 416 exactly 0, 6169 exactly +π, 4672 exactly −π, 20743 none of the three, of 32000). The choice between +π and −π is set by the sign of the FFT round-off in the imaginary part.
- Exact-zero imaginary parts always give +π. The mne accumulator is `np.zeros` and the csd is added to it (`epochs_bivariate.py:64`, `:68`); (+0.0) + (−0.0) = +0.0, so a −0.0 imaginary part becomes +0.0. On narrowband_f0-0.25 at T = 256 the raw csd has 265 negative zeros among 515 exact zeros; the mne coherency has none.
- The mean equals π·(n₊π − n₋π)/N up to round-off, counting angles near ±π. On iid_gaussian at T = 256 it is 0.23895739121367354 against 0.23895739121367365 from the formula (difference 1.1e-16).
- The max is π or 1 ulp below π for every non-identical pair: exactly π for 83 of 85 pairs at T = 64 and 84 of 85 at T = 1024; anm at T = 64 gives π − 1 ulp. The identical pair (0,2) in shared_envelope gives about 1e-31 (1.0e-31 at T = 64, 2.0e-31 at T = 1024), not exactly 0.

**Candidates.**
- The value depends on round-off, which contradicts the README's "non-trivial values".
- It is not antisymmetric under channel reversal. On narrowband_f0-0.25, A[0,1] is 0.237 in the given order and −0.185 after reversing; on delayed_noise it is 0.137 versus −0.074. The +0.0 accumulator explains this. Points with a nonzero imaginary part negate under reversal, while exact-zero points with a negative real part stay at +π in both orders. So A_rev = −A + 2π·n(Im = 0, Re < 0)/N: 0.2372 − 0.1852 = 0.05203 = 2π·265/32000, and 0.1367 − 0.0736 = 0.06303 = 2π·321/32000 (agreement within 3e-16).
- The value is not stable to input changes at round-off level. Re-z-scoring an already z-scored input changes samples by at most 4.4e-16, yet it moves A[0,1] from 0.2372 to 0.2682 on narrowband_f0-0.25 (reversed order: −0.1852 to −0.2109) and from 0.1367 to −0.0383 on delayed_noise, both at T = 256.
- No delay information survives.

**Already documented.** The aggregation as an arithmetic mean of angles is REVIEW M4 (multitaper). As for spectral.phase, the record sets `definition_under_review` true: M4 leaves arithmetic against circular aggregation as a decision, and both forms are recorded (impl and canon runs). The degenerate output adds a fix-or-remove decision on top of that.

**Checks.**
- A tolerance check (1e-12) that every angle lies near one of {0, ±π}, that the mean equals π·(frac(near +π) − frac(near −π)), and that the max is near π (near 0 for identical pairs).
- A from-scratch pipeline, where Im = +0 gives +π. Expected to disagree, for example 0.859 (= π·frac(z0·z1 < 0)) against 0.237.
- A round-off stability check: the same pipeline on x and on a re-z-scored x, expected to disagree beyond round-off.
- A canonical numpy phase, expected to give ±π/4 and ∓π/2 for narrowband and −2πfd for delayed_noise.
- A channel-reversal direction check, which also tests A_rev = −A + 2π·n(Im = 0, Re < 0)/N.

### icoh (`ImaginaryCoherence`, measure `imcoh`)

**Documented.** Supplement §1.5.5, p.14: "coherence exclusively caused by a time delay".

**Canonical.** Im C_xy (Nolte et al. 2004). It is antisymmetric.

**Implemented.** ε/(2|x_ix_j|), which is round-off. The per-point maximum is 4.6e-14, and pair means are about 1e-16. The sign flips exactly under channel reversal.

**Already documented.** README:217 says "rounding noise", and README:157 says the values are "below 2e-14, inside the 1e-8 `atol`", which already implies that the golden test cannot detect any change in this unit.

**Candidate.** The cause is the real one-sample wavelet. With n_cycles = 7, the per-point values span −1 to 1.

**Checks.**
- max |A| < 1e-12.
- The from-scratch pipeline gives exactly 0.
- The canonical ImC (large for delayed_noise and narrowband) is expected to disagree.
- An orientation and sign check under channel reversal.

### plv (`PhaseLockingValue`, measure `plv`)

**Documented.** Supplement §1.5.6, p.15: the coherence formula "applied to the amplitude-normalized Fourier transformed signals".

**Canonical.** |E_n e^{iθ_n}| over trials (Lachaux et al. 1999), or over a time window for a single trial (Lachaux et al. 2000).

**Implemented.** `epochs_bivariate.py:110-131` gives |z/|z|| = 1, so the output is exactly 1.

**Already documented.** Yes.

**Checks.** Exactness. From-scratch pipeline. Canonical single-trial PLV, expected to disagree. Order invariance.

### ppc (`PairwisePhaseConsistency`, measure `ppc`)

**Documented.** Supplement §1.5.7, p.15: "not biased by the sample size".

**Canonical.** (|Σe^{iθ}|² − N)/(N(N−1)) (Vinck et al. 2010). It is undefined for N = 1.

**Implemented.** `epochs_bivariate.py:343-347` computes (±ε or 0)/0. On iid_gaussian at T = 256 the per-point values are 4756 −inf and 27244 NaN; nanmean/nanmax return −inf.

**Already documented.** README:217. README:157 ("status `ok` counts only NaN") already covers the consequence that a −inf table passes the Calculator status check. REVIEW M7 is unrelated: it covers legitimate negative values.

**Candidate.** The −inf comes from round-off. numpy divides the complex csd by the real |csd|, so Re acc = Re csd · fl(1/|csd|); where |csd| equals |Re csd| bitwise this is c·fl(1/c), which in round-to-nearest is 1 or 1 − 2⁻⁵³, never above 1. That bounds only Re(acc)². The residue |acc|² − 1 = fl(fl(Re(acc)²) + fl(Im(acc)²)) − 1 turns positive once Im(acc)² reaches about 2⁻⁵³, that is once |Im csd| exceeds about 1.05e-8·|Re csd| (about 1.5e-8 when Re acc = 1 − 2⁻⁵³), and |csd| still equals |Re csd| bitwise there, so bitwise equality does not exclude +inf. A synthetic probe (`cleanup/wavelet_measures/pinf.py`) gives bitwise equality and a positive residue at every ratio from 1.06e-8 to 1.6e-8 for Re csd = 1.0, 0.37 and 2.9e-7, and the pipeline expression at Re = 1, Im = 1.2e-8 returns +inf. On the synthetic setup of `wl_fix2/p1.py`, |Im| = 1e-9·|Re| never gave a positive residue, while |Im| about 1e-8·|Re| gave one at 216,800 of 10⁶ points, most of them (193,029 in the rerun) at points where |csd| equals |Re csd| bitwise. The audit data stay far below the threshold: over all lower pairs of the 19 datasets at T = 64, 256 and 1024 (14,280,000 points), the residue was never positive (1,984,491 negative, 12,295,509 zero), and the largest |Im csd|/|Re csd| is 2.41e-12 (bivariate_gaussian, T = 1024, pair (2, 1); no point above 1e-10; `cleanup/wavelet_measures/ratio.py`). So +inf is excluded only for the inputs probed; it needs a lag-0 product close to zero relative to the FFT round-off.

**Checks.** Non-finite everywhere. A canonical epoched PPC (K = 8 epochs, n_cycles = 7), expected to disagree.

### pli (`PhaseLagIndex`, measure `pli`)

**Documented.** Supplement §1.5.8, p.15: "averaging the sign of the (per taper) phase difference".

**Canonical.** |⟨sign Im S⟩|, with the magnitude taken after averaging over time samples (Stam et al. 2007) or trials.

**Implemented.** `epochs_bivariate.py:177,183-184` computes |sign(ε)| per (f,t): 1 where the round-off imaginary part is non-zero, 0 where it is exactly zero. The output is the fraction of non-zero round-off residues.
- Probe: "pli[1,0] mean over (f,t): 0.9791875  1-frac(Im==0): 0.9791875".
- Ceilings are 0.875 at T = 64 and 0.9922 at T = 1024.
- delayed_noise gives the same values as iid_gaussian.
- The value is not stable to input changes at round-off level: re-z-scoring an already z-scored input moves narrowband_f0-0.25 T = 256 A[0,1] from 0.983906 to 0.982594 and delayed_noise from 0.977781 to 0.973969. wspli is bitwise equal to pli and changes identically.

**Already documented.** The README (:217) treats this unit as non-degenerate, and the candidate contradicts that. REVIEW M3 (signed PLI) does not apply, because mne-connectivity 0.9.0 takes the magnitude (`:184`).

**Checks.**
- An exact identity with 1 − frac(Im == 0).
- The from-scratch direct convolution gives 0 and is expected to disagree.
- A canonical time-averaged PLI, expected to disagree.
- An order-invariance check.
- Band variants differing only through round-off.

### dswpli (`DebiasedSquaredWeightedPhaseLagIndex`, identifier `dspli`, measure `pli2_unbiased`)

**Documented.** Supplement §1.5.8 lists dspli as "the debiased squared phase lag index".

**Canonical.** (N·PLI² − 1)/(N − 1) (Vinck et al. 2011; the library comment cites Eq. 30). It is undefined for N = 1.

**Implemented.** `epochs_bivariate.py:200` gives 0/0 = NaN where Im ≠ 0 and −1/0 = −inf where Im = 0 (probe: 666 −inf, 31334 NaN). The output is −inf; it would be NaN if no exact zero occurred in the band (hypothesis).

**Naming.** The class name says "Weighted", but the class computes the unweighted estimator. This matches the identifier `dspli`.

**Already documented.** README:217.

**Checks.** Non-finite everywhere. Canonical epoched dsPLI, expected to disagree.

### wspli (`weighted_PhaseLagIndex`, measure `wpli`)

**Documented.** Supplement §1.5.8 lists "the weighted phase lag index (wpli)".

**Canonical.** |E Im X| / E|Im X| (Vinck et al. 2011).

**Implemented.** `epochs_bivariate.py:244-265` gives |ε|/|ε| = 1, or 0 when the denominator is 0. The output is bitwise identical to pli, so it is the same round-off artefact.

**Naming.** The identifier `wspli` and the display name "Weighted squared phase lag index" both say "squared", but the quantity is the unsquared wPLI.

**Already documented.** The README lists this unit as non-degenerate, which the candidate contradicts. M3 does not apply.

**Checks.** An exact equality with pli. From-scratch (gives 0). Canonical wPLI, expected to disagree.

### dwspli (`debiased_weighted_squared_PhaseLagIndex`, measure `wpli2_debiased`)

**Documented.** Supplement §1.5.8 lists dswpli.

**Canonical.** Σ_{j≠k} Im X_j Im X_k / Σ_{j≠k} |Im X_j||Im X_k| (Vinck et al. 2011).

**Implemented.** `epochs_bivariate.py:300-309`: the denominator is 0, and the zero-denominator guard sets the output to 0. The output is exactly 0, and it stays 0 for any wavelet as long as there is one epoch.

**Already documented.** README:217.

**Checks.** Exactly 0. From-scratch (gives 0). Canonical epoched dwPLI², expected to disagree.

## Direction convention

All nine units are labelled undirected and produce symmetric matrices. A[i,j] = A[j,i] is the band/time statistic of conn[max(i,j), min(i,j)], where csd = X_hi conj(X_lo):

- `mne_connectivity/spectral/epochs.py:276`: lower-triangular connections only.
- `mne_connectivity/spectral/epochs.py:652`: the csd product.
- `wavelet.py:87-88`: the lower → upper mirror.

Canonical phase and ImC are antisymmetric. pyspi forces them to be symmetric, and with the one-sample wavelet they encode no lead or lag. Each record proposes a channel-reversal and orientation check. The infrastructure record adds a check comparing the cached conn[hi, lo] with a from-scratch X_hi conj(X_lo).

## Open questions

1. Should these nine classes be fixed (proper n_cycles, for example 7 or a T-dependent value, epoching or time-window smoothing as in Lachaux 2002) or removed? They are not in config.yaml or the documentation. A fix is a definition decision, not a numerical correction.
2. `tests/README.md:157` says the extra goldens were generated "with mne 0.23.0", but the installed delegate is mne-connectivity 0.9.0 / mne 1.13.2. DEPENDENCY_MIGRATION reports that the goldens were unchanged after the migration. Did mne 0.23.0's `spectral_connectivity` take |·| for PLI/wPLI as mne-connectivity 0.9.0 does? This was not re-checked against the 0.23.0 source.
3. Is the exact-zero pattern of the FFT round-off (which sets pli, wspli and the sign of the phase's π) stable across scipy.fft versions, CPU architectures or SIMD paths? Cross-platform behaviour was not tested. On one machine the pattern already changes under input changes at round-off level: re-z-scoring an already z-scored input (sample changes ≤ 4.4e-16) moves the phase on narrowband_f0-0.25 T = 256 from 0.2372 to 0.2682 and on delayed_noise from 0.1367 to −0.0383, and pli from 0.983906 to 0.982594 and 0.977781 to 0.973969. The outputs are not functions of the data to round-off precision, so platform dependence is likely.
4. Can ppc or dspli produce +inf or NaN instead of −inf? For ppc, +inf needs a point with |Im csd| above about 1.05e-8·|Re csd| (Im(acc)² ≥ 2⁻⁵³), which bitwise equality of |csd| and |Re csd| does not rule out; the largest ratio in the audit data is 2.41e-12 (see the ppc section), so the case is a lag-0 product close to zero relative to the round-off, which no dataset produced. For dspli the numerator N·PLI² − 1 is 0 or −1 (`epochs_bivariate.py:177`, `:200`), so +inf cannot occur. Either output would be NaN only if no −inf point fell in the band, which no dataset has shown.
5. Equation numbers for Vinck et al. 2010/2011 and Nolte et al. 2004 are cited from memory or from the library comment and were not checked against the papers.
6. The documented band rule (125 bins on [4/T, fs/2]) also differs from the psi_wavelet grid (linspace(0.2, 0.5, 10), `wavelet.py:209`). That unit is outside this family, but the same fixed-grid question applies.

## Checks

Check stage, 2026-10-15. Manifest `audit/checks/manifest/wavelet_measures.json` has 355 runs. References are in `audit/checks/refs/wavelet_measures.py` (comparison venv; no mne or mne_connectivity import) and probes in `audit/checks/probes/wavelet_measures_probes.py` (project venv). Results are in `audit/checks/out/results/wavelet_measures.json`: 244 pass, 111 fail, 0 without reference. Every failure is accounted for below:

- 107 are expected disagreements (89 canonical illustrations: 61 `canon`, 14 `canonep`, 14 `canonwin`; 9 from-scratch; 9 re-z-scoring): the canonical illustrations, the from-scratch pipeline for phase/pli/wspli, and the re-z-scoring runs. Two canonical illustrations pass at atol 0.05 because the canonical value is itself near 0: icoh bivariate_gaussian (zero-lag coupling) and dwspli narrowband_f0-0.1 (0.0135; the 0.1 oscillation lies below the 0.2–0.5 grid).
- 4 are icoh `six` runs at T = 1024 that exceed the trace's proposed 1e-12 bound.

All pyspi runs took ≤ 1.5 s (sum of the recorded pyspi time_s over the 355 runs: 37 s; the whole forced re-run took a few minutes of wall clock). T ≤ 1024 throughout and there are no convergence runs, because these are extra-config classes. The regression goldens were not used as a reference.

Run tags:

- `six`: the six identifiers on one Data object.
- `ident`: an identity evaluated on the library CWT. The identity slice is computed by the project-venv probe from mne's own CWT and csd and the reference only copies it (for wspli the reference is pyspi's pli output), so these 48 runs are labelled **same-lineage** (pyspi/delegate self-consistency). For phase, pli and wspli the `rep` port does not import mne, but it replays mne's arithmetic (`tfr._cwt_gen` with scipy.fft, the mne-connectivity accumulators and pyspi's aggregation), so those 9 rep runs are labelled same-lineage too (reference_kind `same-lineage` in results/wavelet_measures.json); it is not an independent implementation. The 18 rep runs of cohmag, dswpli, dwspli, icoh, plv and ppc compare against the one-epoch, one-sample-wavelet constant and are labelled `analytic`.
- `rep`: the plain identifier, on the pipeline, pipeline-detrend and kernel paths.
- `scratch`: a from-scratch pipeline using the closed-form Morlet with n_cycles = f/7, DIRECT `np.convolve` (no FFT), textbook single-epoch formulas and pyspi's band, statistic and mirror.
- `canon`: Morlet with n_cycles = 7 (`_kernels_spectral.cwt_morlet`). cohmag, phase, icoh, plv, pli and wspli use the time-averaged single-trial form; ppc, dspli and dwspli use K = 8 non-overlapping epochs. These runs are a qualitative illustration, judged at atol 0.05. They run on all seven proposed datasets (bivariate_gaussian, delayed_noise, iid_gaussian, narrowband_f0-0.1, narrowband_f0-0.25, narrowband_f0-0.4, var2_bivariate) at T = 1024 for every unit.
- `canonep`: pli and wspli only, the proposed epoched variant: K = 8 non-overlapping epochs, PLI = |mean_k sign(Im S_k)| and wPLI = |Σ_k Im S_k| / Σ_k |Im S_k| per (f, t), then pyspi's band statistic (`canonical_epoched_lag`). Illustration at atol 0.05.
- `canonwin`: cohmag and plv only, the proposed sliding-window variant (Lachaux 2002): a centred window of n_co = 6 cycles (round(6/f) samples) over the cross- and auto-spectra or the unit phasors, NaN where the window does not fit (`canonical_window`). Illustration at atol 0.05.
- `perm`: channel reversal, mapped back to the original order. For icoh, phase, pli and wspli the reference is pyspi's own forward output (and, for phase, an offset computed by the probe from mne's csd), so these runs are labelled same-lineage; for the constant units the reference is the analytic constant.
- `roundoff`: the pipeline on z and on zscore(z) (pyspi vs pyspi, labelled same-lineage).

**One-sample wavelet (`wavelet.morlet_cwt`)**

- `wavelet.morlet_cwt|iid_gaussian|64|kernel|wavelets`: all 125 mne wavelets are 1 sample long with σ_t = 1/(14π). W equals √2 to within 2 ulp; there are 3 distinct values, and 98 of 125 are not bitwise √2. So "W = √2" in the trace is exact only up to the last bits. The pipeline wavelet's frequency response is flat (all-pass). The n_cycles = 7 wavelet is 23–55 samples long and has a half-power band of 0.208–0.292 at f = 0.25.
- `cwtx` (FFT path, as the pipeline): 3/3 pass. max|Re X − √2x| is 2.7e-15, 2.2e-15 and 1.1e-15 (iid, delayed_noise, narrowband), and max|Im X| is 7.0e-16. 4.6–5.7 % of the imaginary parts are exactly 0. The spread across the 125 frequencies is ≤ 2.7e-15.
- `cwtdirect` (mne direct convolution): **bitwise** equal to x·W0(f) on all three datasets (rtol = atol = 0). The CWT is the signal times a constant, exactly.
- `orient7`: with n_cycles = 7, the library cohy matches the independent CWT (max_abs 1.7e-15 and 4.2e-15). This fixes conn[hi, lo] = X_hi conj X_lo:
  - delayed_noise pair (1,0): −1.2541 at f = 0.2008, against −2πf = −1.2566 (x1 lags).
  - narrowband_f0-0.25 pair (1,0): +0.7837 against +π/4.
  - narrowband_f0-0.25 pair (2,0): −1.5761 against −π/2.
  - The upper triangle is 0.
- `orient1`: pyspi's cached cohy has Re = sign(z_hi z_lo) and |conn| = 1, with the upper triangle and diagonal 0 (max_abs 0.0).
- `freqs`: the grid is exactly linspace(0.2, 0.5, 125) at T = 64, 256 and 1024. It has 21 bins in [0, 0.25]; the documented [4/T, 0.5] grid would have 54, 61 and 62.
- `wavelet.cohmag|narrowband_f0-0.1|1024|pipeline|canon:mean-full`: the canonical coherence of the 0.1 oscillation peaks at 0.9998 (f = 0.096) on the documented grid but reaches only 0.1488 on pyspi's grid.
- `short:n16`: 78 frequencies are kept. The low band is empty, so the mean gives NaN and the max raises `ValueError` (stored as NaN). `short:n24`: 121 are kept and all values are 1.0.
- `fscache`: an fs = 10 icoh on the same Data returns the fs = 1 result bitwise (−1.5e-18). A fresh fs = 10 Data gives 0.0224, because at fs = 10 the wavelets are 3 samples long.
- `mtadapt`: 0 differing entries over the nine measures.

Verdict **degenerate, 0** (rubric anchor 0, constant: X(f,t) = √2·x(t) at every one of the 125 frequencies, so the transform is constant along its frequency axis); new_finding true. The one-sample wavelet, the fixed grid, the fs-blind cache and the short-T NaN are not in KNOWN_ISSUES.md.

**Constant units**

All are degenerate (score 0). The constant outputs themselves are already documented (tests/README.md:217, KNOWN_ISSUES "tests/README.md flagged: wavelet degenerate").

- **cohmag.** `six` passes 16/16. The mean identifiers are exactly 1.0, but the max identifiers are 1 + 2⁻⁵², which corrects the trace's "exactly 1.0 for every statistic". `rep` and `scratch` pass 3/3 each (max_abs 0.0). Canonical values for pair (1,0): narrowband_f0-0.25 0.6576, delayed_noise 0.9691, bivariate_gaussian 0.2975, iid_gaussian 0.0763, narrowband_f0-0.4 0.7176, narrowband_f0-0.1 0.0867, var2_bivariate 0.502 (`canon`, 0/7 pass). Sliding window (`canonwin`, 0/7 pass): 0.7924, 0.9687, 0.5897, 0.5634, 0.8253, 0.5382, 0.6576 in the same order; the short windows (12–30 samples) bias independent pairs up to about 0.55. new_finding false: the causes are counted on morlet_cwt.
- **plv.** Exactly 1.0 (16/16, max_abs 0.0). The from-scratch pipeline gives 1.0. Canonical, pair (1,0), in the order narrowband_f0-0.25, delayed_noise, bivariate_gaussian, iid_gaussian, narrowband_f0-0.4, narrowband_f0-0.1, var2_bivariate: 0.6623, 0.9309, 0.2445, 0.0568, 0.6924, 0.0585, 0.4149 (`canon`, 0/7 pass). Sliding window (`canonwin`, 0/7 pass): 0.7964, 0.9423, 0.570, 0.543, 0.8179, 0.5184, 0.6206.
- **dwspli.** Exactly 0.0 (16/16). With N = 1 the zero-denominator guard fires for any Im, so the from-scratch pipeline also gives 0. Canonical, epoched K = 8 (same dataset order as plv): 0.6539, 0.9501, 0.0008, 0.0034, 0.7333, 0.0135, 0.3815 (1/7 pass: narrowband_f0-0.1, where the canonical value is itself near 0).
- **icoh.** `six` passes 12/16. The four failures are the T = 1024 max identifiers: iid 2.01e-12, bivariate 2.41e-12, delayed 1.83e-12 and var2 2.40e-12, against the trace's 1e-12 bound. These equal the per-point max |Im csd|/|Re csd| found by the ppc `ratio` probe. That is still round-off, four orders below the golden atol of 1e-8. The trace's "per-point maximum 4.6e-14" is a T = 256 figure. `scratch` passes 3/3: the direct convolution gives exactly 0. `perm` passes 3/3: A_rev = −A exactly. Canonical ImC, pair (1,0) (same dataset order as plv): 0.4696, −0.6730, −0.0367 (passes at atol 0.05, as expected for zero-lag coupling), 0.0027, 0.4650, 0.0380, −0.3807 (1/7 pass). new_finding true for the confirmed `unsigned` label on a signed quantity (wavelet.py:125). Rubric anchor 0, constant: the output is 0 up to round-off (|A| ≤ 2.41e-12; the direct-convolution pipeline gives exactly 0). The label item is not already documented: REVIEW M3 (PLI/wPLI), M4 (coherence phase) and M8 (pec/pec_log, Gaussian entropies, spearman/kendall, tlmi) describe the same kind of mismatch for other metrics, and M9 and EEG_OPTIMIZATION_NOTES.md §review ('parameter-specific/shared labels') discuss label structure without naming imaginary coherence; REVIEW covered the stored metrics, and none of the retained 185 is a wavelet metric. Beyond round-off the sign is real: with fs = 10 (3-sample complex wavelets, fresh Data) ImaginaryCoherence mean gives off-diagonal values −0.011979 to +0.010275 on a lagged-copy fixture, and channel reversal maps A[1,0] = +0.0016114 to −0.0016114 (probe `wf4b_low-wavelet/p3_icoh_fs.py`, Workflow 4b). `definition_under_review` stays false: Nolte's signed ImC is not contested, and the unit's open question is whether to fix or remove the degenerate class (open question 1 below), which applies to all nine classes alike; spectral.icoh sets it true because its note treats the signed versus unsigned ImC choice of the multitaper unit as an open definition decision.
- **ppc.** −inf everywhere (16/16).
  - `scratch`: qualitative 3/3. The from-scratch pipeline, whose Im is exactly 0, *also* gives −inf: 5290 −inf and 26710 NaN points on iid T256 pair (1,0). numpy's complex/real division csd/|csd| returns c·fl(1/c) = 1 − 2⁻⁵³ at some points, so the −inf does not need FFT round-off.
  - `pts`: iid T256 has 4756 −inf / 27244 NaN / 0 +inf; T1024 has 18654 / 109346 / 0.
  - `ratio`: max |Im csd|/|Re csd| ≤ 2.41e-12 on the six datasets at T1024.
  - `pinf`: mne-connectivity's own `_PPCEst` returns +inf from ratio 1.1e-8 for Re = 1.0, 0.37 and 2.9e-7. At 15 of those +inf points per Re, |csd| == |Re csd| bitwise. The result matches the sign rule exactly.
  - Canonical, K = 8, pair (1,0) (same dataset order as plv): 0.6341, 0.8511, 0.0513, −0.0076, 0.6554, 0.0169, 0.1777 (0/7 pass).
  - The candidate is confirmed as the mechanism of an already-documented output; new_finding false.
- **dswpli (dspli).** −inf everywhere (16/16). `pts`: iid T256 has 666 −inf (at the Im == 0 points) and 31334 NaN; T1024 has 1000 and 127000. `dsplisyn`: the library estimator gives [NaN, −inf] without and with an exact zero. The from-scratch pipeline gives −inf exactly. Canonical, K = 8 (same dataset order as plv): 0.6331, 0.8185, −0.0016, 0.0042, 0.7066, 0.0090, 0.1935 (0/7 pass). new_finding false. The −inf/NaN mechanism explains an output that is already documented (tests/README.md:217), as for ppc. The Python class name says Weighted but the class computes the unweighted `pli2_unbiased` (wavelet.py:163-169). This is a code-naming remark only: the user-facing name "Debiased squared phase lag index (wavelet)" (wavelet.py:164) and the identifier `dspli` describe the computation, and tests/README.md:155 already lists dspli as `DebiasedSquaredWeightedPhaseLagIndex`. The check stage first set new_finding true on this remark; that was corrected after review.

**Round-off units** (phase, pli, wspli; tests/README.md:157 calls these "non-trivial")

- **phase.**
  - `ident` 16/16: pyspi equals π(n₊π − n₋π)/N within 3.6e-15. Every angle lies within 2.41e-12 of {0, ±π} (1.7e-13 at T = 64), and 22–73 % of the angles are not exactly in that set. The max identifiers are exactly π, except on the identical shared_envelope pair (0,2): 1.0e-31 at T64, 2.0e-31 at T1024. The coherency has no negative zeros, while the raw csd has 0.13–4.7 %.
  - `rep` 3/3 bitwise: a scipy.fft replay of mne's arithmetic on pyspi's stored input (scipy 1.18.1 in the comparison venv; pyspi runs on scipy 1.17.1) reproduces pyspi (max_abs 0.0). The same replay with numpy.fft differs by up to 0.3175, and the exact-zero Im pattern differs at 4.7–5.5 % of points.
  - `scratch` 0/3, an expected disagreement: with Im exactly 0 the mean is π·frac(negative products). iid gives 1.534 against pyspi 0.1342; narrowband gives 0.8468 against 0.1815 (kernel path, raw input).
  - `perm` 3/3: A_rev = −A + 2π·n(Im=0, Re<0)/N holds within 6.1e-16. The offsets are 0.0520 (narrowband), 0.0630 (delayed_noise) and 0.0491 (iid). max|A_rev + A| is 0.104–0.163, so the output is not antisymmetric. REVIEW M4 (`docs/storage_comparison/REVIEW.md:155-157`) documents the same branch-cut mechanism for the multitaper coherence phase (+π represents both directions, which produces an apparent antisymmetry offset) and says an unsigned label is inappropriate for this signed angle.
  - `roundoff` 0/3, an expected disagreement. With input changes ≤ 4.4e-16:
    - narrowband: 0.2372 → 0.2682
    - delayed_noise: 0.1367 → −0.0383
    - iid: 0.2390 → 0.1665
  - Canonical: narrowband pair (1,0) gives 0.7844 at f0 (+π/4) and a band mean of 0.5047; delayed_noise gives −1.567 at f = 0.25 (−π/2). Band means for pair (1,0) on the other datasets: bivariate_gaussian −0.1189, delayed_noise −2.199, iid_gaussian −0.5667, narrowband_f0-0.4 0.5429, narrowband_f0-0.1 0.4014, var2_bivariate −2.186 (0/7 pass).
  - Degenerate (0), rubric anchor 0, constant: the max identifiers are the constant π on every non-identical pair, and the mean identifiers are round-off noise with no reproducible dependence on the input. A Workflow 4b probe (`wf4b_low-wavelet/p1_morlet_phase.py`) shows the mean does not track the only data feature it could encode: over 30 draws per correlation at T = 256 the mean phase of pair (1,0) is −0.007, +0.064, +0.025, +0.004, −0.004 (SD 0.085–0.147) at ρ = −0.9, −0.5, 0, +0.5, +0.9, while π·frac(negative products) moves from 2.717 to 0.456.
  - new_finding true, carried by two candidates: the values are round-off (this contradicts tests/README.md:157 and :217, which call phase non-trivial; the cause, the one-sample wavelet, is counted on morlet_cwt, but the observable fact is not documented anywhere) and the max identifiers are exactly π on every non-identical pair. The reversal offset is the REVIEW M4 branch-cut mechanism, reached through mne's +0.0 accumulator rather than the DC bin, and the `unsigned` label is M4; both are documented. `check_method` independent-impl names the from-scratch pipeline, the strongest independent reference run; the degeneracy itself rests mainly on the same-lineage ident, perm and roundoff runs, and the cap is moot at 0.
- **pli.**
  - `ident` 16/16 exact (same-lineage): pyspi = 1 − frac(Im csd == 0), and the max is 1. On iid_gaussian, bivariate_gaussian, delayed_noise, narrowband_f0-0.25, var2_bivariate and trending the exact-zero fraction is 0.78–1.2 % at T = 1024 and 12.6–15 % at T = 64, and the outputs are 0.9878–0.9922 and 0.8505–0.875. The value carries no lead/lag information, but it is not the same on every dataset: identical or tied channels produce many more exact zeros.
    - logistic_maps (the tied period-4 fixture): 0.9778 at T = 1024 (fraction 2.2 %) and 0.8524 at T = 64 (`wavelet.pli|logistic_maps|1024|pipeline|ident`, ref_pyspi_mean_full_min = max = 0.9778).
    - shared_envelope, whose pair (0,2) is identical: down to 0.465 at T = 64 (fraction up to 53.5 %) and 0.8731 at T = 1024 (fraction up to 12.7 %).
    - An earlier version of this note said "0.85–0.875 at T = 64 and 0.988–0.992 at T = 1024 regardless of coupling". That range does not hold for these two datasets; it was corrected after review.
  - `rep`: the scipy.fft replay (scipy 1.18.1, against pyspi's 1.17.1) is bitwise. The numpy.fft replay differs by up to 0.027.
  - `scratch`: 0.0 against 0.97–0.98.
  - `perm`: order-invariant (exact).
  - `roundoff`: the value moves by up to 0.0099.
  - Canonical time-averaged, pair (1,0) (same dataset order as plv): 0.6636, 0.9076, 0.0811, 0.0573, 0.7134, 0.0353, 0.4227 (0/7 pass). Epoched K = 8 (`canonep`, 0/7 pass): 0.7383, 0.8992, 0.2713, 0.2747, 0.7954, 0.2720, 0.4524. With K = 8 the independent and zero-lag pairs sit near 0.27, the small-sample bias of |mean of 8 signs|; this bias is what the debiased dsPLI removes.
  - mne 0.23.0 also took |mean sign|; see open question 2. REVIEW M3 does not apply.
  - Degenerate (0); new_finding true.
- **wspli.** Bitwise equal to pli for all six identifiers, every dataset and both T (16/16). The identifier says "squared" but the computation is the unsquared wPLI. Canonical time-averaged, pair (1,0) (same dataset order as plv): 0.7020, 0.9854, 0.1036, 0.1197, 0.7557, 0.0969, 0.6635 (0/7 pass). Epoched K = 8 (`canonep`, 0/7 pass): 0.7935, 0.9790, 0.3792, 0.3895, 0.8463, 0.3946, 0.6601. Degenerate (0); new_finding true.

**Symmetry and direction.** max|A − A.T| = 0.0 on every (P, P) run of every unit. A[i,j] = A[j,i] = the statistic of conn[max, min]. No unit carries lead/lag information.

**Trace issues fixed or re-checked in this stage.**

- (a) The ppc record and the note already carry the Im(acc)² / about-1e-8-ratio condition instead of the false bitwise-equality condition. The check-stage `pinf` and `ratio` runs reproduce the dispute ruling's numbers: +inf from ratio 1.1e-8, and a maximum audit ratio of 2.41e-12.
- (b) All tests/README.md citations in the ten records and in this note already point at the current :155, :157 and :217. No ':138' or ':198' remains, and the "flagged: wavelet degenerate" text appears only as the KNOWN_ISSUES citation key.
- Corrections made in this stage:
  - The cohmag max statistic is 1 + 2⁻⁵², not "exactly 1.0".
  - The phase and icoh round-off bounds of 4.6e-14 hold at T = 256 only; at T = 1024 they reach 2.41e-12.
  - The phase and icoh `unsigned` labels are at wavelet.py:112 and :125.
  - W equals √2 to within 2 ulp, not bitwise.
- The comparison venv runs scipy 1.18.1 and numpy 2.5.4; the project venv (pyspi, mne) runs scipy 1.17.1 and numpy 2.5.4. An earlier version of this note said the comparison venv had scipy 1.17.1. That was wrong, and it was corrected after review: the `rep` runs record `scipy.fft==1.18.1` in their reference_detail.
- Corrected after review:
  - The ppc record's implemented_as no longer argues from Re(acc) alone. It now states that bitwise |csd| == |Re csd| bounds only Re(acc)², and that +inf needs Im(acc)² ≥ 2⁻⁵³, which the audit data (ratio ≤ 2.41e-12) do not reach.
  - The ident runs (and the perm runs of icoh, phase, pli, wspli, and the roundoff runs) are relabelled same-lineage. None of the verdicts or scores depends on this label (all 0).
  - The canonical illustrations now cover all seven proposed datasets for every unit, plus the proposed epoched K = 8 variant for pli/wspli and the sliding-window variant for cohmag/plv. Before this, they covered four datasets (six for cohmag) and neither variant.
  - The phase record's `unsigned` label citation is wavelet.py:112.

## Open questions: answered

1. **Fix or remove?** This is still a definition decision for the maintainers. The evidence:
   - All nine classes are degenerate (score 0).
   - A fix needs both a real wavelet (n_cycles of order 7) and averaging over epochs or time. With one epoch and n_cycles = 7, coh and plv would still be 1 and dwspli 0, by the trace's p4 probe and the dwspli algebra.
   - The `canon` runs show what time-averaged or epoched estimators would return on the audit data. For example, narrowband_f0-0.25 coherence is 0.999 at f0, and delayed_noise PLI is 0.91.
   - Recommendation: remove, or rewrite with an explicit averaging window. Changing only n_cycles is not enough.
2. **mne 0.23.0 and the PLI magnitude.** The fetched source of mne-python v0.23.0 `mne/connectivity/spectral.py` (raw.githubusercontent.com, fetched in this stage) has `_PLIEst.compute_con`: `self.con_scores[con_idx] = np.abs(pli_mean)`, and `_WPLIEst` uses `num = np.abs(self._acc[0, con_idx])`. Both versions take the magnitude, so the extra goldens are consistent with the migration in this respect. The PPC accumulate code and the cwt csd line are identical too.
3. **Stability of the round-off pattern.**
   - Across scipy versions: pyspi runs on scipy 1.17.1 (project venv). The `rep` replay runs in a separate process on scipy 1.18.1 / numpy 2.5.4 (comparison venv), with its own closed-form Morlet and no mne import. It reproduces pyspi bitwise on this machine: max_abs 0.0 on the three `rep` runs each of phase, pli and wspli, and the reference_detail records `scipy.fft==1.18.1`. So no difference was observed between scipy.fft 1.17.1 and 1.18.1 here.
   - numpy.fft on the same inputs gives a different exact-zero pattern (4.7–5.5 % of points) and different outputs: phase by up to 0.32, pli and wspli by up to 0.027.
   - Re-z-scoring (≤ 4.4e-16 input changes) moves phase by up to 0.175 and pli by up to 0.0099.
   - So the values depend on the FFT implementation and on last-bit input changes. They did not change between scipy 1.17.1 and 1.18.1 on this machine. Cross-CPU and SIMD-path behaviour (and scipy versions outside 1.17.1–1.18.1) remain untested, because no second machine was available. Given the input sensitivity, platform dependence is plausible but not demonstrated. Running the `rep` replay or the regression snapshot on a second CPU architecture would settle it. An earlier version of this answer said that no second scipy version was available. That was wrong and was corrected after review.
4. **+inf or NaN for ppc/dspli.**
   - ppc: +inf is reachable through the library estimator once |Im csd| ≥ 1.1e-8·|Re csd| (`pinf`), but the audit data peak at 2.41e-12 (`ratio`). No +inf occurred in 18 runs (16 `six`, 2 `pts`).
   - dspli: the output is NaN when a band contains no exact-zero Im (`dsplisyn`). No audit band lacked one; all 16 `six` runs gave −inf.
   - The from-scratch pipeline shows that −inf arises even without FFT round-off, from numpy's complex/real division.
5. **Equation numbers for Vinck 2010/2011 and Nolte 2004.** Not re-checked against the papers in this stage; the papers were not fetched. The check references implement the formulas as stated in the records and in `_kernels_spectral` (PPC = (|Σu|² − K)/(K(K−1)), debiased PLI² = (K·PLI² − 1)/(K − 1), Vinck's pairwise dwPLI²). Settling this needs the PDFs.
6. **psi_wavelet grid.** The psi_wavelet unit belongs to the wavelet_psi family, which is already checked; see its record and notes/wavelet_psi.md. This family confirmed the fixed-grid rule for the nine measures only (`freqs` runs).

## Optimisation

No opportunities are proposed for this family. The user does not use it, and its definition is under review: `cwt_n_cycles = freqs/7` makes every Morlet wavelet one sample long (`wavelet.morlet_cwt`), and the sister psi_wavelet unit has an axis issue. The optimisation blocks therefore hold hotspot fields only.

**The summary.json 32x1024 walls do not reproduce.** The summary.json walls below were measured with all nine fresh units and the family-warm pass in one subprocess per (family, shape) at P ≤ 32 (`profile_units.py:136`, `1060-1063`). In isolated processes (one identifier per process, warm-up at 8x1024, threads 1; `audit/profiles/micro/retime_wavelet.py`, results in `audit/profiles/micro/prof_wavelet__index.json`) the representative identifiers ran 1.6-7x faster at 32x1024. An independent reviewer re-time agrees within about 40%. The 8x1024 walls the reviewer re-timed (cohmag, pli) roughly reproduce. The shared subprocess is a probable cause but was not confirmed. The re-times ran while other Workflow 3 groups were computing (load average 1.7-6.2 on 64 CPUs), so they are indicative, but every one of them is at least 1.6x below summary.json. Consequences:
- The summary.json P exponents (3.5-4.0) and the 118x1024 projections (2,100-10,700 s) are inflated and should not be used for ranking.
- With the re-times the 8 → 32 exponents are 2.2-3.4. The PLI/PLV-type units sit close to the code's P², and the coherence-type units (cohmag, icoh, phase) sit around 3.
- Scaling the re-timed 32x1024 wall to 118 by (118/32)² up to (118/32)^E gives tens to a few thousand seconds per identifier build. That is order of magnitude only and was not measured.

| Unit | 8x1024 fresh (s, summary) | 32x1024 fresh / family-warm (s, summary) | 32x1024 isolated re-time (s) | P_exponent 8 → 32: summary fresh / fw → re-timed | summary.json projected 118 fresh / fw (s, extrapolated-power-from-32; 118 skipped-budget) | Re-timed 32 scaled to 118 (s, order of magnitude) | Family-warm 8x1024 wall_cv (spread of passes; not in wall_s) | 32x1024 first identifier / median later identifier, min_s (s) |
|---|---|---|---|---|---|---|---|---|
| wavelet.cohmag | 0.209 | 42.0 / 39.0 | 14.3; reviewer 15.3, 15.3 | 3.83 / 3.57 → 3.05-3.10 | 6184 / 4110 | ~200-900 | 0.010 | 34.5 / 0.21 |
| wavelet.phase | 0.322 | 56.3 / 85.8 | 32.2; reviewer 35.1 | 3.72 / 3.47 → 3.32-3.38 | 7265 / 7988 | ~440-2900 | 0.018 (fresh 0.31) | 53.0 / 8.01 |
| wavelet.icoh | 0.201 | 35.7 / 41.4 | 13.9 | 3.74 / 3.63 → 3.06 | 4686 / 4707 | ~190-760 | 0.009 | 36.6 / 0.21 |
| wavelet.plv | 0.195 | 29.6 / 43.3 | 5.25 | 3.62 / 3.67 → 2.38 | 3350 / 5224 | ~70-120 | 0.025 | 32.2 / 1.64 |
| wavelet.ppc | 0.219 | 30.3 / 59.4 | 9.57 | 3.56 / 3.82 → 2.72 | 3135 / 8647 | ~130-340 | 0.26 | 40.0 / 3.76 |
| wavelet.pli | 0.163 | 22.0 / 58.1 | 4.22, 4.35; reviewer 3.74-4.58 | 3.53 / 3.97 → 2.26-2.40 | 2213 / 10344 | ~50-105 | 0.40 | 38.9 / 3.62 |
| wavelet.dswpli | 0.178 | 22.4 / 58.9 | 11.0 (one run, load ~6) | 3.49 / 3.93 → 2.98 | 2123 / 9887 | ~150-540 | 0.75 | 39.5 / 3.74 |
| wavelet.wspli | 0.174 | 25.5 / 60.4 | 3.62 | 3.60 / 3.97 → 2.19 | 2781 / 10703 | ~50-65 | 0.97 | 41.8 / 3.58 |
| wavelet.dwspli | 0.271 | 38.2 / 69.9 | 14.3, 14.6; reviewer 10.3 | 3.57 / 3.68 → 2.63-2.87 | 4021 / 8459 | ~140-620 | 0.61 | 51.5 / 3.76 |

Reading the table:
- **Family-warm vs fresh.** The family-warm figure sums the unit's six identifiers: the first builds the `data.mne[measure]` cache, and the other five re-aggregate it. The fresh figure times the representative only. Family-warm is therefore expected to exceed fresh, for example plv 5224 vs 3350 s projected.
- **Family-warm/fresh gap (ppc, pli, dswpli, wspli, dwspli).** The family-warm projections of these five units are 2-5x their fresh projections. The gap comes from their family-warm 32x1024 walls. Each sums a first-identifier cache build from the same non-reproducing run (38.9-51.5 s, last column, against 3.6-14.6 s for the isolated re-times of the representative identifier) and five re-aggregations of about 3.6-3.8 s each (median `min_s`). That makes the family-warm 32x1024 wall 1.8-2.7x the fresh one, while the family-warm 8x1024 wall is only 1.4-1.6x the fresh one, so the family-warm exponent rises to 3.7-4.0.
- **8x1024 wall_cv.** The 8x1024 wall_cv plays no part in the exponents or projections. Where it is high (0.26-0.97 family-warm for the five units above, 0.31 fresh for phase) it comes from one or two slow first passes (for example wspli family-warm 1.42 s, then 0.25-0.27 s; dwspli also has spread later passes, 0.43-0.71 s). A unit's `wall_s` is the minimum over passes (`profile_units.py:69-71`), so those passes are excluded. A larger 8x1024 wall would in any case lower the 8 → 32 exponent, not raise it.
- **Last column.** The first identifier's increment comes from the same non-reproducing 32x1024 family-warm run (single timing). The later identifiers are quoted as the median of `min_s`, because `profile_units.py:72-75` documents that `median_s` is inflated when a first identifier above 30 s was not re-timed.

**Scaling and memory.** The code is quadratic in P: per-connection `accumulate`/`compute_con` over 125 frequencies × T, then a dense (P, P, 125, T) output and pyspi's nan-statistic over (freq, time).
- The excess over P² in the coherence-type units is in mne's own per-connection estimator, not in pyspi's dense output. From the 8x1024 to the 32x1024 cohmag profile, the self time of `_CohEst.compute_con` grows 148x (0.053 s → 7.85 s) for 17.7x more connections, and pyspi's `nanmean` is only 4.7% at 32x1024.
- For pli, `compute_con` grows 13x (0.037 s → 0.48 s), i.e. about linearly in connections.
- Attributing the cohmag excess to cache/memory effects of the (n_cons, 125, T) temporaries (`psd_xx * psd_yy`, `abs`, divide) is an inference.
- Dense tensor sizes: 0.07 GB real at P = 8; 1.05 GB real / 2.1 GB complex at P = 32; 14.3 GB real / 28.5 GB complex (`phase`, measure `cohy`) at P = 118. The isolated 32x1024 processes peaked at 3.7-6.0 GB.
- A 118x1024 run would probably not fit in the 31 GB machine (the dense tensor plus mne's per-connection complex accumulators of about 14 GB). This is inferred from array sizes; no 118 run was attempted.

**Hotspots.** Percentages are of the profiled SPI call (the `compute_spi` frame), for the representative identifier on a fresh Data, so the cache build is included.
- At 8x1024 (`audit/profiles/<unit>__8x1024.pstats`), `pyspi/statistics/wavelet.py:mne._get_cache` → `mne_connectivity.spectral_connectivity_epochs` takes 57.7-85.5% in every unit.
- Three units were also profiled at 32x1024 in this stage (`audit/profiles/micro/prof_wavelet.<unit>__32x1024.pstats`, script `prof_wavelet.py`). There the weight moves to mne's per-connection frames, and the CWT and pyspi's statistic shrink:
  - **cohmag:** `_CohEst.compute_con` 70.5% self; `ndarray.conjugate` (called from `_epoch_spectral_connectivity`) 10.5%; `nanmean` 4.7%; cwt 1.0%. Opaque share 0.145.
  - **phase:** `_CohyEst.compute_con` 46.5% self; pyspi's `modify_stats` 39.6% (`np.angle` 19.3%, `nanmean` 19.6% of which `_replace_nan` 17.9%); cwt 0.4%. Opaque share 0.189.
  - **pli:** `ndarray.conjugate` 30.8%; `_PLIEst.accumulate` 16.8%; `_PLIEst.compute_con` 9.9%; `nanmean` 15.4%; cwt 2.2%. Opaque share 0.390.
- The other six units keep their 8x1024 hotspots in the records, with a pointer to the sister 32x1024 profile (cohmag for icoh; pli for the PLV/PPC/PLI-type units).

| Unit | hotspot_1 | Other frames | Opaque share |
|---|---|---|---|
| cohmag (32x1024) | `_CohEst.compute_con` 70.5% self (22.6% at 8x1024) | `conjugate` 10.5%; `nanmean` 4.7%; cwt 1.0% (13.4% at 8x1024) | 0.145 (0.230 at 8x1024) |
| phase (32x1024) | `_CohyEst.compute_con` 46.5% self (18.3% at 8x1024) | `angle` 19.3% (25.2% at 8x1024); `nanmean` 19.6%; cwt 0.4% | 0.189 (0.147 at 8x1024) |
| icoh (8x1024) | `_ImCohEst.compute_con` 22.4% (in `_epoch_spectral_connectivity` 42.3%) | `_compute_spectra` 15.6% (cwt 13.1%); `nanmean` 11.7% | 0.202 |
| plv (8x1024) | `_epoch_spectral_connectivity` 58.4% (`compute_con` 17.7%, `accumulate` 14.9%) | `_compute_spectra` 13.7%; `nanmean` 11.4% | 0.247 |
| ppc (8x1024) | `_PPCEst.compute_con` 23.0% + `accumulate` 18.3% | `_compute_spectra` 13.8%; `nanmean` 13.5% | 0.175 |
| pli (32x1024) | `ndarray.conjugate` in `_epoch_spectral_connectivity` 30.8% self (wrapper 54.6% at 8x1024) | `accumulate` 16.8%; `compute_con` 9.9%; `nanmean` 15.4%; cwt 2.2% (17.4% at 8x1024) | 0.390 (0.257 at 8x1024) |
| dswpli (8x1024) | `_epoch_spectral_connectivity` 50.5% (`_PLIUnbiasedEst.compute_con` 13.4%) | `_compute_spectra` 18.6%; `nanmean` 18.2% | 0.266 |
| wspli (8x1024) | `_epoch_spectral_connectivity` 65.1% (`accumulate` 14.0%, `compute_con` 12.2%) | `_compute_spectra` 18.2%; `nanmean` 10.9% | 0.232 |
| dwspli (8x1024) | `_WPLIDebiasedEst.compute_con` 38.1% | `accumulate` 11.3%; `nanmean` 7.0%; cwt 5.5% | 0.158 |
| morlet_cwt | `mne/time_frequency/tfr.py:cwt` 13.4% of cohmag at 8x1024 (5.5-17.4% over the nine profiles), but 1.0% (cohmag), 0.4% (phase), 2.2% (pli) at 32x1024 and 0.45% of wavelet.psi at 118x1024 | `_compute_spectra` 16.3% of cohmag at 8x1024, 1.2% at 32x1024 | 0.230 (cohmag 8x1024) |

For plv, pli, dswpli and wspli the 8x1024 hotspot_1 names the `_epoch_spectral_connectivity` wrapper. That frame's own self time (19-24%) and the leaf estimator frames are quoted alongside it, and the 32x1024 pli profile resolves the wrapper's self time mainly to a single `ndarray.conjugate` call.

**Shared versus incremental.**
- Each unit caches its measure in `data.mne[measure]`. The first identifier pays the whole build. In the summary.json 32x1024 family-warm run that was 32-53 s, but that run does not reproduce: the isolated re-times of the same representative builds are 3.6-35 s.
- The other five identifiers re-aggregate the cached tensor. The median `min_s` is 0.21 s for cohmag and icoh, 1.6 s for plv, 3.6-3.8 s for ppc/pli/dswpli/wspli/dwspli, and 8.0 s for phase, whose `np.angle` + `nanmean` runs over a 2 GB complex array.
- No cache is shared between measures, so all nine units repeat the same 125-frequency CWT. That is 5.5-17.4% of each 8x1024 call, but only 0.4-2.2% at 32x1024 and 0.45% of wavelet.psi at 118x1024. A shared CWT cache is therefore below the ~10% threshold at realistic channel counts, and it is refuted as an avenue even if the family were in use.

**EEG scaling.** No unit of this family has EEG identifiers (`in_eeg_selection: false`).
