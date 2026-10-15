# Family it_gaussian: define, trace and check

Scope: the 13 units in `audit/briefing/it_gaussian.json`. That is 12 configured SPIs plus the shared JIDT Gaussian estimator layer (`infotheory.est.gaussian`). Every per-unit record is in `audit/records/infotheory.*.yaml` (status `checked`). The sections before `## Checks` are the trace stage; the probe results quoted there are small read-only confirmations of what the code does, from scripts under the session scratchpad: `it_gaussian/probes/probe1-4.py`, plus `itg_fix/mi_calls.py` (JIDT call counts, P=5: 20 for mi, 10 for je) and `itg_fix/ce_probe.py` (ce symmetry, max |A − Aᵀ| = 2.2e-16 z-scored, and the normalise=False offset 2.3159931 vs ln 10). The regression goldens were not used and are not a reference. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`. Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Supplement citations give the section and page, plus the line in `supplement.txt` found with `grep -n`. The briefing pack's line numbers are 9 to 12 lines higher. They look like they count form-feed page breaks as lines.

## Shared machinery (`infotheory.est.gaussian`)

**Documented.** Supplement §1.4, p.9: "The Gaussian-distribution model ... assumes a linear-Gaussian multivariate, where the measure is derived from the cross-covariance matrix". The supplement names JIDT v1.5. GitBook names v1.6.1.

**Canonical.** For a multivariate normal, h = ½(d(1+ln 2π) + ln det Σ). MI and CMI follow as ratios of determinants (Cover & Thomas 2006, Thm 8.4.1 and Ex. 8.5.1). For Gaussian variables, transfer entropy equals half of Geweke's Granger causality, F/2 (Barnett, Barrett & Seth 2009).

**Implemented.** The chain runs as follows:

1. `Data.set_data` z-scores each channel with ddof=1 (`data.py:193-195`).
2. `JIDTBase._getcalc` (`infotheory.py:103-134`) sends `gaussian`, and any other estimator it does not recognise, to the else branches. These are `EntropyCalculatorMultiVariateGaussian`, `MutualInfoCalculatorMultiVariateGaussian` and `TransferEntropyCalculatorGaussian`.
3. `_setup` (`:84-93`) sets `BIAS_CORRECTION=false` and `NOISE_SEED=42` on every calculator.

Inside JIDT (JIDT line numbers in this note are at tag v1.6.1 unless marked a9b6b3e. The bundled `pyspi/lib/jidt/infodynamics.jar` is labelled 1.6.1 by `version-1.6.1.txt`, `readme.txt` and `build.xml`, but it is a later build: all 337 jar entries are dated 2024-04-21 and it contains a class added upstream on 2024-04-20, which places it near commit a9b6b3e. The exact commit is not established; see the bundled-jar finding below.):

- Covariances use divisor n-1 (`MatrixUtils.covarianceTwoColumns:3370-3377`, v1.6.1 line numbers).
- Determinants come from a Cholesky factorisation with tolerance 1e-9 (`:4496-4560`, v1.6.1 line numbers). A non-positive-definite matrix throws `NonPositiveDefiniteMatrixException`.
- Entropy = ½(d(1+ln 2π)+ln det).
- MI and CMI use determinant ratios. Redundant components are pruned: a fully redundant marginal gives 0, and joint redundancy gives +∞.

All results are in nats.

**Probe findings**

- **Constant univariate entropy.** pyspi and JIDT both use ddof=1, so after z-scoring every univariate Gaussian entropy is exactly ½(1+ln 2π) = 1.4189385332 (`data.entropy` = [1.41893853, 1.41893853]).
- **NORMALISE settings differ but do not matter.** The entropy calculator reports `NORMALISE=true` (`EntropyCalculatorMultiVariateCommon:66` at a9b6b3e; the class does not exist at v1.6.1), but the Gaussian path never normalises: scaling a series by 10 raised H by exactly ln 10. The MI and TE calculators report `NORMALISE=false`. This confirms the hypothesis, with no effect on values.
- **Bundled jar is not the v1.6.1 tag build (candidate).** The jar contains `EntropyCalculatorMultiVariateCommon`, which is absent from the v1.6.1 source tree. Upstream added that class in commit 3d1f3fe on 2024-04-20, and the jar entries are dated 2024-04-21. JIDT citations for files that changed after v1.6.1 therefore point at commit a9b6b3e. Comparing v1.6.1 with a9b6b3e: the Gaussian MI, CMI, TE and AIS leaf classes, `MatrixUtils`, `MathsUtils` and `ChiSquareMeasurementDistribution` are byte-identical, and their `*MultiVariateCommon` / `*Via*` parents differ only in observation-ID and noise-seed bookkeeping. `EntropyCalculatorMultiVariateGaussian` changed more (128 diff lines): it was refactored onto the new `EntropyCalculatorMultiVariateCommon`, losing its own `setObservations`, `initialise()`, `setProperty`, `getProperty` and `setDebug`, gaining `finaliseAddObservations` and `isComputed` caching. Its value path (covariance → Cholesky → ½(d(1+ln 2π)+ln det)) is unchanged. Every probe matched from-scratch numpy formulas to 6e-14 or better.
- **One degenerate pair blanks whole matrices.** On two bit-identical channels (`shared_envelope` rows 0 and 2):
  - je, ce, cce, di and si raise `NonPositiveDefiniteMatrixException`.
  - mi returns +∞, gc_k1 returns 0, and tlmi and xme stay finite.
  - `Calculator.compute` (`calculator.py:336-339`, the `except` branch at HEAD a1f20c7: `self._table[spi] = np.nan`) then sets the whole P×P matrix to NaN for each raising SPI.

## je_gaussian (`infotheory.je.gaussian`)

- **Documented.** Supplement §1.4.1, p.10: "quantifies the uncertainty over the paired observations".
- **Canonical.** H(X,Y) = ½ ln((2πe)² det Σ).
- **Implemented.** The JIDT Gaussian entropy of the stacked pair [x_i, x_j] (`infotheory.py:159-177`). `Undirected.multivariate` (`base.py:133-137`) visits every ordered pair through `Directed.multivariate` (`base.py:85-93`) and then overwrites the lower triangle from the upper; the second order of each pair is a cache hit (`infotheory.py:168-175` stores both [i,j] and [j,i]), so JIDT runs P(P−1)/2 times (probe, P=5: 10 calls). After z-scoring this is je = 2·1.4189385 + ½ ln(1−r²) = 2·H1 − mi_gaussian. Probe on var2_bivariate T=1024: 2.81900377 = 2·1.41893853 − 0.01887329.
- **Deviations.** None in the formula. The value is an affine function of r², so it is rank-equivalent to mi_gaussian (candidate). Negative values are covered by REVIEW M8.
- **Proposed checks:**
  - Numpy determinant formula (exact).
  - Analytic ρ values on bivariate_gaussian.
  - The je = 2H1 − mi identity.
  - Scale shift ln(s_i s_j) on scaled_offset with normalise=False.

## ce_gaussian (`infotheory.ce.gaussian`)

- **Documented.** Supplement §1.4.2, p.10: "the uncertainty over the observations in y in the context of simultaneously observing x". The keywords say "undirected".
- **Canonical.** H(Y|X) = H(X,Y) − H(X).
- **Implemented.** A[i,j] = H(x_i,x_j) − H(x_i) = H(x_j | x_i) (`infotheory.py:246-250`). The class is Directed. REVIEW M8 already records the conditioning order.
- **Deviation (candidate): exact symmetry under default preprocessing.** z-scoring makes H(x_i) the same constant for every channel, so ce = H1 − mi exactly and A = Aᵀ.
  - var2_bivariate: A[0,1] = A[1,0] = 1.40006524.
  - scaled_offset z-scored: max |A − Aᵀ| = 2.2e-16. The record therefore sets `symmetry_expected: sym` for the default pipeline.
  - With normalise=False, ce becomes asymmetric (scaled_offset: 2.0787 vs −0.2373). The difference A[i,j] − A[j,i] = ln(sd_j/sd_i) uses the sample SDs of the scaled rows: 2.3159931 for (0,1), not ln 10 = 2.3025851, because the scaled rows are built from different base rows (`make_data.py:246-249`).
- **Direction.** Row i is the conditioning variable.
- **Proposed checks:**
  - Numpy reference.
  - Conditioning order on raw scaled_offset (A[i,j]−A[j,i] = ln(sd_j/sd_i) with sample SDs, to 1e-12).
  - Symmetry and affine identity under z-scoring.
  - Analytic ρ values.

## mi_gaussian (`infotheory.mi.gaussian`)

- **Documented.** Supplement §1.4.3, p.10. The estimator is defined at §1.4 (supplement.txt:414-415): "The Gaussian-distribution model (denoted by modifier gaussian) assumes a linear-Gaussian multivariate, where the measure is derived from the cross-covariance matrix". That is the text the value matches.
- **Canonical.** −½ ln(1−ρ²).
- **Implemented.** JIDT Gaussian MI on (x_i, x_j) with no lag, no bias correction and a bare `except` returning NaN (`infotheory.py:267-283`). Probe: 0.01887329 = −½ ln(1−r²). `MutualInfo.bivariate` has no cache, so `Undirected.multivariate` runs the JIDT MI for both orders of every pair and discards the lower triangle: P(P−1) calls (probe, P=5: 20 calls). This double computation is already documented at `EEG_OPTIMIZATION_NOTES.md:132` (the record's `already_documented` cites it).
- **Deviations.** None. Candidates: no small-sample bias correction (expected value 1/(2N) under independence), and rank-equivalence with je and ce.
- **Proposed checks:**
  - Numpy formula (exact).
  - `true_mi_nats` on bivariate_gaussian (statistical).
  - χ²(1)/(2N) null on iid_gaussian.
  - Qualitative near-zero result on nonlinear_dependence.

## tlmi_gaussian (`infotheory.tlmi.gaussian`)

- **Documented.** Supplement §1.4.4, p.10: "lag-one mutual information". The supplement calls it "undirected" while GitBook calls it "directed".
- **Canonical.** I(X_t; Y_{t+1}) (Fraser & Swinney 1986).
- **Implemented.** Gaussian MI of (x_i[:-1], x_j[1:]) (`infotheory.py:301-317`). A[i,j] = I(x_i(t); x_j(t+1)), so the row process leads. Probe on var2_bivariate: A[0,1] = 0.1666 and A[1,0] = 0.0028, each matching numpy exactly.
- **Deviations.** None in the value. Asymmetry is already noted in REVIEW M8. The documentation inconsistency between supplement and GitBook is a candidate.
- **Proposed checks:**
  - Numpy reference.
  - delayed_noise analytic ½ ln 101 = 2.30756 for 0→1 and ≈ 0 for 1→0 (direction).
  - gauss_ar1_pair population lagged correlation.

## gc_gaussian_k-1_kt-1_l-1_lt-1 (`infotheory.te.gaussian_k-1_kt-1_l-1_lt-1`)

- **Documented.** Supplement §1.4.6, p.11: "the log-ratio of residual variance for the two models ... popularized by Geweke ... later found to be equivalent to transfer entropy with a Gaussian estimator".
- **Canonical.** Geweke 1982: F = ln(RSS_r/RSS_f). Barnett et al. 2009: TE = F/2.
- **Implemented.** JIDT `TransferEntropyCalculatorGaussian` → Gaussian CMI with k=l=1 and delay 1 on T−1 samples: TE = I(x_i(t); x_j(t+1) | x_j(t)). Properties confirmed with `getProperty`.
- **Deviation from documented (candidate): factor ½.** The returned value is F/2, not the documented log-ratio F. Probe: pyspi 0.14832534 = numpy 0.5·ln(RSS_r/RSS_f) to 1e-15. Rankings are unaffected.
- **Direction.** A[i,j] = TE(i→j). var2_bivariate: 0.1483 vs 0.0127.
- **Proposed checks:**
  - Numpy OLS F/2 (exact, which also documents the ½).
  - VAR(2) population TE.
  - Direction on var2_bivariate and delayed_noise.
  - χ² null on iid_gaussian.

## gc_gaussian_k-max-10_tau-max-2 (`infotheory.te.gaussian_auto-MAX_CORR_AIS_kmax-10_taumax-2`)

- **Documented.** Supplement §1.4.5–1.4.6, p.11: embeddings are chosen to maximise active information storage (MAX_CORR_AIS), with k ≤ 10 and τ ≤ 2.
- **Implemented.** Inside each `bivariate` call, JIDT runs the MAX_CORR_AIS search twice, first on the target and then on the source:
  - It scans k = 1..10 and τ = 1..2 (τ is skipped when k = 1).
  - Each candidate scores the Gaussian AIS minus the χ² null mean k/(2N), where N = T − (k−1)τ − 1 is that candidate's own sample count (`ActiveInfoStorageCalculatorViaMutualInfo:440-451` at a9b6b3e), so candidates with different k are penalised on different N. A re-implementation must use this per-candidate N to reproduce JIDT's selections (`ActiveInfoStorageCalculatorViaMutualInfo:579-615` at a9b6b3e; `ActiveInfoStorageCalculatorGaussian:152-170` and `ChiSquareMeasurementDistribution:100-102` at v1.6.1).
  - It keeps the argmax, then computes Gaussian TE with those embeddings and delay 1.
- **Probe.** A from-scratch numpy re-implementation reproduced the selected (k, kt, l, lt) for every ordered pair, and the values to 6e-14 or better (var2_bivariate T=64 and T=1024, delayed_noise T=256).
- **Deviations:**
  - Same factor ½ as the fixed-lag unit.
  - Not stated in the docs: the source order comes from the source's own AIS, and the search is repeated for every ordered pair.
- **Candidates:**
  - **Unstable selection.** The penalised AIS curves are nearly flat, which makes the choice unstable. The AR(2) source x0 was given k=9 (penalised AIS 0.1165–0.1180 for k = 2..10), and white-noise channels in delayed_noise were given k = 7–10.
  - **Redundant searches (efficiency).** The searches are repeated 2·P·(P−1) times, although P searches would give the same result.
- **Proposed checks:**
  - Numpy re-implementation: exact parameter match and values to 1e-10.
  - Direction.
  - Selected-order stability across T against the true AR order.

## te_symbolic_* (`infotheory.te.symbolic`)

- **Documented.** Supplement §1.4.5, p.11: symbolic TE after Staniek & Lehnertz 2008 (ordinal patterns).
- **Implemented.** With `estimator='symbolic'`, `JIDTBase.__init__` still appends `_symbolic` to the identifier (`infotheory.py:38`) and adds the 'symbolic' label (`:50`). Its branch (`:45-52`) then sets `_dyn_corr_excl = None` and returns early, which skips only the trailing DCE suffix block (`:57-58`); the 'linear' label of the else branch (`:53-55`) is never added. `_getcalc` has no symbolic branch, so it builds `TransferEntropyCalculatorGaussian`. This is known bug 1 in `tests/README.md`.
- **Probe (elaborates known bug 1, not a new candidate):**
  - te_symbolic_k-1 is bitwise equal to gc_gaussian_k-1.
  - te_symbolic_k-10 = 0.13893927, which is Gaussian TE with 10 target lags and 1 source lag (numpy reference 0.13893927237).
- **Additional metadata candidate.** The instance labels include 'symbolic' and omit 'linear', so label-based grouping misclassifies these linear SPIs. Both identifiers are in the EEG selection.
- **Proposed checks:**
  - Numpy Gaussian TE (exact).
  - Same-lineage identity with gc_k1.
  - From-scratch STE on logistic_maps, anm and nonlinear_dependence, expected to disagree, including a monotone-transform invariance test.

## xme_gaussian_k1 and xme_gaussian_k10 (`infotheory.xme.gaussian_k1`, `infotheory.xme.gaussian_k10`)

- **Documented.** No definition anywhere. Only GitBook `table-of-spis.md:73` lists "Cross-Map Entropy" (candidate documentation gap).
- **Canonical.** None. The nearest standard quantity is the conditional entropy H(Y_t | X_{t−1..t−m}). The code has nothing to do with convergent cross mapping.
- **Implemented** (`infotheory.py:405-426`). A[i,j] = H(x_j(t) | lagged x_i) with Gaussian entropies.
  - The source lags are 1 and `range(2,k)`, so history_length=1 uses lag 1 and history_length=10 uses lags 1..9.
  - The target's own past is not conditioned on.
  - Probes matched numpy with lag sets {1} and {1..9} exactly: 1.25161306 and 1.2375756.
- **Off-by-one (candidate, hypothesis confirmed).** k=10 means 9 lags. k=1 and k=2 share lag set {1} and differ only by one dropped sample: 1.2516131 vs 1.2520804.
- **Sign (already documented, REVIEW M8; not a candidate).** Values are negative for well-predicted targets. The analytic value for delayed_noise 0→1 is −0.889. This follows the REVIEW M8 mechanism.
- **Proposed checks:**
  - Numpy reference.
  - delayed_noise analytic values and direction.
  - A constructed shift test from an iid_gaussian row (d = 9 vs d = 10) that documents the 1..9 lag set.

## cce_gaussian (`infotheory.cce.gaussian`)

- **Documented.** Supplement §1.4.7, p.11: a sum of conditional entropies of y "given both the past of x and the past of y", with the history restricted to 10.
- **Canonical.** Kramer 1998: H(Y^N‖X^N) = Σ_n H(Y_n | Y^{n−1}, X^n). X^n includes the present.
- **Implemented** (`infotheory.py:439-474`). Σ_{m=1}^{5} H(x_j(t) | x_j(t−1..t−m+1), x_i(t..t−m+1)), estimated from stationary sliding windows.
  - `makeDelayEmbeddingVector(src, m)` includes x_i(t). For k = 0 the call returns zero-width rows, which is what the code expects.
  - Probe: numpy reproduces 6.12472743 and 6.63088572 exactly.
- **Deviations:**
  - n = 5, not the documented 10 (`infotheory.py:435`; the config passes no n). Hypothesis confirmed.
  - Each term conditions on present and past x, which matches Kramer but not the supplement's "past of x".
  - The value is a 5-term block sum (≈ 7.09 nats for independent z-scored data), not a per-sample entropy.
- **Direction.** A[i,j] = H(x_j ‖ x_i). var2_bivariate: 6.125 (0→1) < 6.631.
- **Proposed checks:**
  - Numpy reference.
  - Analytic white-pair values 5·(H1 + ½ ln(1−ρ²)).
  - Direction.
  - Population value for gauss_ar1_pair from Γ(h) = A^h Σ_X.

## di_gaussian (`infotheory.di.gaussian`)

- **Documented.** Supplement §1.4.8, p.11–12: "a difference between the conditional entropy of y given its own past, and causally conditioned entropy", with the history restricted to 10.
- **Canonical.** Massey 1990 and Kramer 1998: I(X^N→Y^N) = Σ_n I(X^n; Y_n | Y^{n−1}) = H(Y^N) − H(Y^N‖X^N).
- **Implemented** (`infotheory.py:487-510`). Σ_{m=1}^{5} H(Y^m)/m − cce, where H(Y^m) is the Gaussian entropy of the m-dimensional delay vector of the target.

**Formula deviation (candidate, hypothesis confirmed).** The first term is a sum of block entropies per symbol, not H(Y^5) = Σ_m H(Y_m | Y^{m−1}). Because H(Y^m)/m ≥ H(Y_m | Y^{m−1}), the code value equals Massey's DI plus a non-negative excess that depends only on the target. For an AR(1) target with coefficient φ, the excess is ½(4 − (5 − H₅))·(−ln(1−φ²)) = 0.6417·(−ln(1−φ²)), where H₅ = 1 + ½ + ⅓ + ¼ + ⅕.

- **Probe, var2_bivariate.** pyspi 0.77460 / 0.19589 equals the numpy code form. The canonical DI₅ is 0.64825 / 0.03690.
- **Probe, independent source.** With an independent white source and an AR(1) target at T=4096:

| φ | di(source → target) | Predicted excess |
|---|---|---|
| 0.5 | 0.1803 | 0.1846 |
| 0.9 | 1.0553 | 1.0656 |

   The reverse direction is ≈ 0.

This produces spurious directed information for any autocorrelated target. It is not in the already-documented list. n = 5 versus the documented 10 also applies here.

- **Direction.** A[i,j] = DI(i→j).
- **Proposed checks:**
  - Numpy code form and canonical form (exact identities).
  - bivariate_gaussian, where both forms equal 5·MI.
  - Independent pair built from var2_bivariate row 0 (the AR(2) target) and an iid_gaussian row, compared against the analytic excess from the AR(2) autocovariance.
  - Direction.

## si_gaussian_k-1 (`infotheory.si.gaussian`)

- **Documented.** Supplement §1.4.9, p.12: individual versus joint entropies, conditioned on one step of history. The prose gives the order as joint minus individual, while the code computes individual minus joint.
- **Canonical.** Ay 2015: SI = Σ_v H(X_{v,t+1} | X_{v,t}) − H(X_{t+1} | X_t).
- **Implemented** (`infotheory.py:524-534`). Exactly this, with Gaussian conditional entropies on the same T−1 samples; the constants and the ddof factor cancel. Probe: 0.16167711 equals numpy.
- **Deviations.** None. Candidates: the singular-covariance exception that blanks the whole matrix, and the fact that innovation correlation contributes to SI (by definition).
- **Proposed checks:**
  - Numpy reference.
  - Analytic population SI on gauss_ar1_pair from Σ_X, AΣ_X and Σ_E.
  - iid null.

## Open questions

1. **Bundled jar provenance.** Which upstream commit built the bundled `infodynamics.jar`? The class inventory and build date point to master at about a9b6b3e (2024-04-20), not the v1.6.1 tag. This is unconfirmed without a byte-level rebuild comparison. The Gaussian maths matched in every probe.
2. **Intent behind di_gaussian.** Was `_compute_entropy_rates` meant to estimate H(Y^n), or n × the entropy rate? Neither matches Σ H(Y^m)/m. A decision is needed on whether di_gaussian should be redefined, and the same code path serves the kozachenko and kernel di configurations, which belong to other families.
3. **History length for cce and di.** Which is intended, 5 (code) or 10 (documentation)? The answer changes values by roughly a factor of 2 in scale.
4. **Factor ½ in gc_gaussian.** Should the documentation say "half the log variance ratio (= Gaussian TE)", or should the value be multiplied by 2 to match the Geweke statistic it names?
5. **Direction and redundancy under z-scoring.** Should ce_gaussian keep the "directed" label, given that it is exactly symmetric under default preprocessing? Should je, ce and mi Gaussian all be kept, given that they are mutually affine under z-scoring?
6. **xme intent.** What was "cross-map entropy" intended to be? Without a definition we cannot decide whether the lags 1..k−1 indexing and the missing target-past conditioning are bugs.
7. **MAX_CORR_AIS selection.** Is its observed instability (k = 7–10 chosen for white noise) acceptable for the EEG use case, or should the auto-embedded GC be dropped or replaced by an information criterion on the bivariate VAR?

## Checks

**Setup.** The manifest `audit/checks/manifest/it_gaussian.json` has 158 runs (154 from the first pass plus 4 added after the review: `est|scaled_offset|…|entropy-shift`, `est|scaled_offset|…|norm-invariance`, `je|scaled_offset|…|shift-d2` and `cce|iid_gaussian|4096|…|truth-iid`). References are in `audit/checks/refs/it_gaussian.py` (comparison venv: numpy/scipy only, no JIDT). Probes are in `audit/checks/probes/it_gaussian_probes.py` (project venv). Results are in `audit/checks/out/results/it_gaussian.json`: **157 pass, 1 fail, 0 without reference**. The one failure is `infotheory.di.gaussian|var2_bivariate|1024|pipeline|massey`, which compares pyspi against Massey's canonical directed information. It is designed to fail, and its mismatch is the documented deviation below.

Run time on the pyspi side depends on machine load. The `time_s` values now stored in `out/pyspi/index/it_gaussian.json` come from the review-stage forced re-run, made under load: 158 runs, 53.6 s in total. Its longest runs are the degenerate-contract probe (9.6 s), the P=5, T=4096 auto-embedding probe (`gc auto|iid_gaussian|4096|pipeline|params`, 6.9 s), the normalise-invariance probe (4.7 s), the te_symbolic `k1-labels` probe (2.9 s), the entropy-shift probe (2.0 s) and the T=1024 auto-embedding probe (1.9 s). Every other run takes ≤ 1.4 s. Three earlier forced runs took 45 s, 24.3 s and 31.3 s in total (the last with the 4 added runs); in each of them the T=4096 auto-embedding probe was the longest run (11.1 s, 3.6 s and 5.7 s); in the last of them (the first that included the normalise-invariance probe, one of the 4 added runs) the next longest were the degenerate-contract probe (3.7 s) and the normalise-invariance probe (1.6 s). Those earlier timings are not stored in the index or results files (the index keeps only the latest run of each entry).

Four kinds of check are used:

1. **`numpy` (exact).** The pyspi matrix in both orientations is compared with the `_kernels_infotheory` closed form (slogdet of the ddof=1 covariance), using the same preprocessing. The `pipeline` path z-scores with ddof=1, `kernel` uses raw rows, and `pipeline-detrend` applies `scipy.signal.detrend` and then z-scores. Tolerance is rtol 1e-10, atol 1e-12.
2. **`truth*` (statistical).** The pass rule is |pyspi − E_MC| ≤ 2·SD_MC, where E_MC and SD_MC are the mean and standard deviation of the numpy estimator over 300 fresh draws from the dataset's generating model at the same T. The analytic population value is stored next to it as `ref_population`, together with `ref_pyspi_minus_population_maxabs`. Each run is restricted with `processes` to the pair whose truth is known, so that one draw is not judged against many null entries at once.
3. **`null` (qualitative).** The mean over all pairs is compared with the Monte Carlo distribution of that mean. The analytic χ² mean df/(2N) is reported alongside.
4. **`direction` (qualitative).** The ground truth is x0 → x1. On delayed_noise, x1 is x0 delayed by one sample; on var2_bivariate, x0 drives x1.

**Final-check item: the Gaussian MI closed form agrees with JIDT to ≤ 1e-10.**

- `infotheory.mi.gaussian|var2_bivariate|1024|pipeline|numpy`: max_abs **1.6e-16**.
- All 10 exact MI runs: max_abs ≤ 4.7e-15.
- The direct JIDT calls (entropy up to 10 dimensions, MI, CMI) in `infotheory.est.gaussian|*|kernel|direct-jidt`: max_abs ≤ 5.5e-15 over 72 calls.

### Shared estimator (`infotheory.est.gaussian`): matches, score 9 (anchor 9: canonical up to the ddof convention)

- **Closed forms.** pyspi's own JIDT entropy, MI and CMI calculators were driven directly on raw lag matrices of var2_bivariate, sparse_precision and iid_gaussian. They agree with numpy determinant formulas to ≤ 5.5e-15.
- **ddof.** The covariance uses ddof=1. A ddof=0 estimate would differ by up to 0.0199 nats at T=256 (`ref_ddof0_entropy_maxabs_diff`).
- **Univariate entropies.** After z-scoring, every cached univariate entropy is 1.4189385332046727 to ≤ 4.4e-16 (`…|entropy-constant`, 3 runs).
- **NORMALISE** (`…|var2_bivariate|1024|kernel|normalise`). The entropy calculator reports `true`, and the MI and TE calculators report `false`. H(10x) − H(x) = 2.302585092994045 = ln 10, so no normalisation is applied and the setting has no effect on values. Units are nats throughout.
- **Jar provenance** (`…|var2_bivariate|64|kernel|jar-provenance`). The Gaussian entropy class extends `EntropyCalculatorMultiVariateCommon`, which does not exist in the v1.6.1 tree. Its `getProperty('BIAS_CORRECTION')` is None, and `NOISE_SEED` is stored (42) but never used.
- **Degenerate contract** (`…|shared_envelope|256|pipeline|degenerate`), with x2 a bit-identical copy of x0:
  - je, ce, cce, di and si raise `NonPositiveDefiniteMatrixException`.
  - mi returns +∞ at (0,2).
  - gc_k1, gc auto and te_symbolic return 0.
  - tlmi and xme_gaussian_k1 stay finite.
  - A real `Calculator` restricted to je + mi returns an all-NaN je table and a `failed` report (`pyspi/calculator.py:336-338` at HEAD a1f20c7; the trace cited the pre-a1f20c7 lines 301-303).
- **Trace correction.** xme_gaussian_k10 also raises on shared_envelope. It does so for every narrowband source row, independently of the duplicate pair: the noise-free sums of sinusoids have a singular 9-lag covariance, and numpy's Cholesky also fails (`ref_lag9_cov_row0_pd = 0`). The trace sentence "tlmi and xme stay finite" is true only for k=1. It is corrected in the est.gaussian candidate text and in the xme_k10 `units_and_range`.
- **Scale behaviour on scaled_offset** (raw rows, x = s·D2 + c with s = 0.2, 2, 5, 10; added after the review, which noted that the proposed check had been run on one channel only):
  - `est|scaled_offset|1024|kernel|entropy-shift`: pyspi's cached univariate entropies with normalise=False shift by exactly ln s relative to the unscaled D2 rows: −1.6094379, 0.6931472, 1.6094379, 2.3025851, to max_abs 6.7e-16.
  - `est|scaled_offset|1024|kernel|norm-invariance`: mi_gaussian, tlmi_gaussian and gc_gaussian_k-1 computed with normalise=True and with normalise=False differ by at most 1.4e-15 (mi), 3.3e-16 (tlmi) and 4.4e-16 (gc_k1); both slots equal the numpy closed forms on z-scored rows to max_abs 3.3e-15.
- **Candidates.** All four are confirmed: provenance (exact commit still unknown), NORMALISE (no effect on values), robustness, and the no-op properties. The robustness finding (one duplicated channel makes je, ce, cce, di and si raise and blanks their matrices) is counted once, here, and not again on the individual SPI records.

### je_gaussian: matches, score 9 (ddof convention)

- **Exact checks.** 6/6 numpy runs pass with max_abs ≤ 3.6e-15. These include the raw scaled_offset kernel run at 3.6e-15. The z-scored scaled_offset equals bivariate_gaussian at 1.3e-15.
- **Identity.** je = 2·1.4189385332 − mi_gaussian holds to 4.4e-16 (`je|bivariate_gaussian|1024|pipeline|identity-mi`).
- **Scale shift** (`je|scaled_offset|1024|kernel|shift-d2`, added after the review). With normalise=False, je(scaled_offset) − je(D2) = ln(s_i s_j) on every pair to 1.4e-15 (for example ln(0.2·2) = −0.91629 and ln(5·10) = 3.91202).
- **Analytic at T=4096.**

| Pair | Population | \|pyspi − pop\| | max_z |
|---|---|---|---|
| ρ=0.3 | 2.79072 | 0.0029 | 0.71 |
| ρ=0.8 | 2.32705 | 0.0116 | 0.85 |

- **Symmetry.** max|A − Aᵀ| = 0.0.
- **Candidates.** Both are confirmed: je is affine in mi (redundancy), and an exactly collinear pair blanks the whole je matrix (robustness). `new_finding` rests on the redundancy; the robustness mechanism is counted under est.gaussian.

### ce_gaussian: matches, score 9 (ddof convention)

- **Exact checks.** 6/6 numpy runs pass with max_abs ≤ 3.6e-15.
- **Conditioning order** (`ce|scaled_offset|1024|kernel|order`, raw data). A[0,1] = 2.07866 = H(x1|x0) and A[1,0] = −0.23733. Their difference is 2.3159931 = ln(sd_1/sd_0), with an error of 1.3e-15. The reversed orientation differs by up to 3.91, so A[i,j] = H(x_j | x_i), as traced.
- **Symmetry under the default pipeline.** max|A − Aᵀ| ≤ 2.0e-15 on 5 datasets. The value is exactly H1 − mi: 3.3e-16 on bivariate_gaussian and 0.0 on var2_bivariate, where A[0,1] = A[1,0] = 1.40007.
- **Analytic at T=4096.** ρ=0.3: 0.0029 (z 0.72). ρ=0.8: 0.0116 (z 1.01).
- **Candidates.** Both are confirmed: the matrix is symmetric under the default pipeline, and ce is redundant with mi.

### mi_gaussian: matches, score 10 (anchor 10)

- **Exact checks.** 10/10 pass with max_abs ≤ 4.7e-15. They cover the kernel path, pipeline-detrend on trending, and the invariance run (MI of raw scaled_offset = MI of the unscaled rows).
- **Convergence to `true_mi_nats`** (0.0471553 for ρ=0.3, 0.5108256 for ρ=0.8), `mi|bivariate_gaussian|{256,1024,4096}|pipeline|truth-p01/p23`:

| T | pyspi, ρ=0.3 | pyspi, ρ=0.8 | \|error\|, ρ=0.3 | \|error\|, ρ=0.8 |
|---|---|---|---|---|
| 256 | 0.04721 | 0.56078 | 5.6e-5 | 0.0500 |
| 1024 | 0.04112 | 0.50528 | 0.0060 | 0.0055 |
| 4096 | 0.05004 | 0.49923 | 0.0029 | 0.0116 |

  All six runs are within 2 SD (max_z ≤ 1.07). The SD shrinks as ρ/√T, from 0.05 to 0.0125 for ρ=0.8. The T values are independent draws, so the error is not monotone in T.
- **Null on iid_gaussian.**

| T | pyspi mean | 1/(2N) | z |
|---|---|---|---|
| 256 | 0.00126 | 0.00195 | −0.83 |
| 1024 | 0.00063 | 0.00049 | 0.64 |

  No value is negative.
- **Linear-only scope.** On nonlinear_dependence, pyspi gives 1.1e-4 and 7.7e-4 nats, while KSG gives 1.61 and 3.25 nats.
- **Candidates.** Redundancy is confirmed. Bias is confirmed as plug-in behaviour, which is not a defect.
- **new_finding: false** (changed after the review). mi_gaussian is itself the canonical quantity. The affinity of je and ce to mi is counted on the je and ce records, the SPIs it makes redundant, and is not counted a third time here.

### tlmi_gaussian: matches, score 10

- **Exact checks.** 4/4 numpy runs pass with max_abs ≤ 5.8e-14.
- **delayed_noise at T=4096.** The population value of I(x0(t); x1(t+1)) is ½ ln 101 = 2.30756. |pyspi − pop| ≤ 0.0158 (z 0.96).
- **gauss_ar1_pair at T=4096.** The population values are 0.12664 (0→1) and 0.17382 (1→0), from Σ_XY = AΣ_X. |pyspi − pop| ≤ 0.0160 (z 1.48).
- **Direction.** A[i,j] = I(x_i(t); x_j(t+1)), so the row leads.

| Dataset | A[0,1] | A[1,0] |
|---|---|---|
| delayed_noise | 2.2707 | 1.6e-5 |
| var2_bivariate | 0.1666 | 0.0028 |

- **Candidate.** The supplement says undirected while GitBook says directed. This is confirmed, but it is recorded as an elaboration of REVIEW M8 (tlmi is asymmetric), not as a new finding.

### gc_gaussian_k-1_kt-1_l-1_lt-1: matches, score 9 (anchor 9: trivial documented convention; changed from variant, 8 after the review)

The record's own canonical definition is Gaussian TE = F/2, and the code equals it to floating-point precision. The only mismatch is the supplement naming the statistic "the log-ratio of residual variance" (Geweke F) while also stating its equivalence to Gaussian TE. F = 2·TE is a documented identity (Barnett, Barrett & Seth 2009), a convention of the same kind as log base. Anchor 8 (published variant of canonical) does not fit, because the code is canonical TE, not a variant of it. The documentation mismatch remains a confirmed finding. Rubric reading is nearest-anchor 9 (score_rubric_note), because anchor 10's supplement clause fails and no anchor-9 clause names a factor of 1/2; the wording above and in the heading is the original reasoning for score_0_10.


- **Covariance form.** 9/9 runs pass with max_abs ≤ 5.7e-14.
- **OLS log-ratio form.** The OLS log-ratio and the covariance form agree to 1.0e-15 on var2_bivariate and 4.0e-14 on delayed_noise. pyspi equals 0.5·ln(RSS_r/RSS_f) to ≤ 1.7e-14.
- **Factor ½.** pyspi divided by Geweke's F is exactly 0.5. On var2_bivariate, F(0→1) = 0.29665 and pyspi gives 0.14833. The supplement's "log-ratio of residual variance" is therefore off by a factor of 2 (candidate confirmed).
- **VAR(2) population at T=4096.** TE(0→1) = 0.13258 and TE(1→0) = 0.01052. The reverse value is non-zero because a lag-1 model is misspecified for a VAR(2). |pyspi − pop| = 0.0060 (z 0.77).
- **Null** (iid_gaussian, T=1024). Mean 3.6e-4 against 1/(2(T−1)) = 4.9e-4 (z −0.92).
- **Direction.** A[i,j] = TE(i→j).

| Dataset | A[0,1] | A[1,0] |
|---|---|---|
| var2_bivariate | 0.1483 | 0.0127 |
| delayed_noise | 2.2704 | 1.3e-5 |

### gc_gaussian_k-max-10_tau-max-2: variant, score 7 (anchor 7: matches the supplement's MAX_CORR_AIS description, value-changing against canonical)

The score comes from the embedding selection. The factor ½ is the same documented convention as in the fixed-lag unit (anchor 9 on its own) and does not lower the score further. Rubric reading is nearest-anchor 7 (score_rubric_note), because anchor 8 fails on the per-pair embedding search and anchor 7 holds only up to the factor 1/2; the fixed-lag unit's rubric reading is nearest-anchor 9, since no anchor-9 clause names a factor of 1/2. The wording above is the original reasoning for score_0_10.


- **Re-implementation.** `refs/it_gaussian.py:ais_select` is a from-scratch MAX_CORR_AIS: Gaussian AIS − k/(2N), with N being the candidate's own sample count, and a strict `>`. With Gaussian TE on top, it reproduces the values to ≤ 5.3e-14 (5 runs).
- **Selected embeddings.** JIDT's `getProperty` selections match bitwise for every ordered pair on 9 probe runs.
- **The selections are unstable** (candidate confirmed):

| Process | True order | T=64 | T=256 | T=1024 | T=4096 |
|---|---|---|---|---|---|
| var2_bivariate x0 | AR(2) | k=3 | k=10 | k=9 | k=8 |
| var2_bivariate x1 | — | k=4 | k=5 | k=2 | k=5 |

  On iid_gaussian, the white channels get k = 6, 1, 7, 8, 10 at T=4096, with τ = 2 for one of them.
- **Redundant searches.** Selection depends only on the process, so the 2·P·(P−1) searches can be cached per process with bit-identical results (efficiency candidate confirmed).
- **Direction on delayed_noise.** A[0,k] = 2.267, 2.282 and 2.296 for k = 1, 2, 3, against A[k,0] ≤ 0.0032. The long source order chosen for white x0 reaches the lag-3 and lag-5 couplings that the fixed k=l=1 unit misses.
- **Direction on var2_bivariate.** A[0,1] = 0.1631 against A[1,0] = 0.0011.

### te_symbolic (both identifiers): wrong, score 1 (anchor 1: wrong estimator substituted). Already documented as tests/README.md bug 1, not new.

- **Gaussian TE values.** te_symbolic_k-1 equals Gaussian TE to ≤ 5.7e-14, and is bitwise equal to gc_gaussian_k-1 (`k1-identity-gc1`, rtol = atol = 0). te_symbolic_k-10 equals Gaussian TE with 10 target lags and 1 source lag to ≤ 8.9e-14.
- **Monotone-transform test.** `refs/it_gaussian.py:ste` is a from-scratch Staniek–Lehnertz STE (m=3). Under exp(x), STE does not change at all (change 0.0), while pyspi's value changes by up to 185% on anm and 29% on logistic_maps_chaotic.
- **Values on logistic_maps_chaotic.**

| Direction | STE | pyspi |
|---|---|---|
| 0→1 | 0.0942 | 0.0062 |
| 1→0 | 0.0014 | 8.6e-5 |

- **Label candidate.** The instance labels are symbolic=1, linear=0, while gc_k1 has linear=1. This is confirmed, but it is a metadata consequence of bug 1, so it is not counted as new.

### xme_gaussian_k1: related-quantity, score 5 (anchor 5: non-standard quantity, no literature definition, computed as the code states)

- **Exact checks.** 4/4 numpy runs pass with max_abs ≤ 5.5e-14. The reference is H(x_j(t) | x_i(t−1)) with no target past.
- **delayed_noise at T=4096.** The population value of A[0,1] is −0.88862, and |pyspi − pop| = 0.0159 (z 0.97). For the (0,3) pair the lag is 5, which k=1 does not use, so the population value is H1 and the error is 6.1e-5.
- **k=1 against k=2.** history_length=2 equals history_length=1 on samples t = 2..T−1, to 6.7e-16 and 3.1e-14. On var2_bivariate 0→1, k=1 gives 1.25161 and k=2 gives 1.25208. The only difference is one dropped sample, so k=1 and k=2 coincide in lag set.
- **Candidates.** All three are confirmed: the documentation gap, the k=1/k=2 coincidence, and the missing target-past conditioning.

### xme_gaussian_k10: related-quantity, score 4 (anchor 4, related but different quantity: lags 1..9 instead of the 10 the identifier names; the hyperparameter is applied off by one, not ignored, so not anchor 3; as xme.kernel_k10 and xme.kozachenko_k10)

- **Exact checks.** 4/4 numpy runs with lags 1..9 pass, max_abs ≤ 8.5e-14.
- **Shift test** (`xme|iid_gaussian|4096|pipeline|shift-pattern`). u → u(t−9) gives −0.8957 (analytic −0.8886), while u → u(t−10) gives 1.4146 ≈ H1. A lag-10 coupling is invisible to history_length=10.
- **delayed_noise at T=4096.** Both the lag-1 and the lag-5 pairs are near the population value −0.88862: z 1.09 and 1.56.
- **Robustness.** The SPI raises on noise-free narrowband sources (see est.gaussian).
- **Candidates.** Both are confirmed: the off-by-one lag set and the documentation gap.

### cce_gaussian: variant, score 7 (anchor 7: matches Kramer's quantity but with a value-changing parameter that differs from the supplement)

- **Exact checks.** 5/5 numpy runs pass with max_abs ≤ 2.3e-14. The reference uses x lags 0..m−1, so the present x is included.
- **White pairs at T=4096.** The population value 5(H1 + ½ ln(1−ρ²)) is 6.85892 for ρ=0.3 and 4.54056 for ρ=0.8. The errors are 0.0183 (z 0.57) and 0.0555 (z 0.99).
- **gauss_ar1_pair at T=4096.** The population values from Γ(h) = AʰΣ_X are 5.81321 and 5.56278. The error is 0.147 (z 1.75). This SD is large because the sample variance used in z-scoring enters all five terms.
- **History length.** The configured n is 5 for both cce_gaussian and di_gaussian (`cce|var2_bivariate|64|pipeline|configured-n`); the documentation says 10. On var2_bivariate the n = 10 probe gives 11.949 / 13.108, against 6.125 / 6.631 for n = 5.
- **Direction.** A[i,j] = H(x_j^5 ‖ x_i^5).

| Dataset | A[0,1] | A[1,0] |
|---|---|---|
| var2_bivariate | 6.125 | 6.631 |
| delayed_noise | −1.997 | 7.083 |

- **Independent white pair at T=4096** (`cce|iid_gaussian|4096|pipeline|truth-iid`, added after the review; it replaces the earlier quotation of the T=256 exact-run values 6.97–7.06 as evidence). The population value is 5·H1 = 7.09469. The Monte Carlo mean is 7.09162 (SD 0.00180). pyspi gives 7.09433 (0→1) and 7.09321 (1→0): |pyspi − pop| = 0.0015, max_z 1.51, pass.
- **Candidates.** All three are confirmed: n = 5, the present x included in every term, and the output being a block sum.

### di_gaussian: related-quantity, score 4 (anchor 4: "DI with block-entropy rate")

- **Code form.** 7/7 exact runs pass with max_abs ≤ 2.3e-14.
- **Massey's DI on var2_bivariate** (the run that fails by design): max_abs 0.159.

| Direction | Massey DI₅ | pyspi |
|---|---|---|
| 0→1 | 0.64825 | 0.77460 |
| 1→0 | 0.03690 | 0.19589 |

  pyspi − Massey equals the target-only block term B to 8.9e-16. B is 0.12635 for target x1 and 0.15899 for target x0.
- **Independent white source into the AR(2) target** (T=4096). pyspi gives 0.1742, against the analytic B = 0.17660 (z 1.72). The Massey form gives 0.0017, and the reverse direction gives −8.8e-5. This is spurious directed information (confirmed).
- **White pairs.** Both forms equal 5·MI; errors are 0.0153 and 0.0565 (z ≤ 0.95).
- **History length.** n = 10 gives 1.675 / 0.313, against 0.775 / 0.196 for n = 5.
- **Direction.** var2_bivariate: 0.775 > 0.196. delayed_noise: 9.093 > 0.012.
- **Candidates.** All three are confirmed: the block-entropy formula, the spurious DI, and n = 5.

### si_gaussian_k-1: matches, score 10

- **Exact checks.** 4/4 numpy runs pass with max_abs ≤ 8.9e-16.
- **gauss_ar1_pair at T=4096.** Population SI is 0.09992; the error is 0.0021 (z 0.45).
- **White pairs.** SI equals the contemporaneous MI: 0.0472 and 0.5108, within 0.0029 and 0.0118 (z ≤ 0.98).
- **Null** (iid_gaussian, T=1024). Mean 0.00132 against χ²(3)/(2N) = 0.00147 (z −0.35).
- **Symmetry.** max|A − Aᵀ| = 0.0.
- **Candidates.** Both are confirmed. Robustness: si raises on the duplicate pair. Interpretation: innovation correlation counts toward SI. The interpretation point is a property of Ay's definition, not a deviation.
- **new_finding: false** (changed after the review). The robustness mechanism is counted once under est.gaussian, and the interpretation point is a documentation note, not a defect.

### Changes after the adversarial review

- **new_finding.** Set to false on `infotheory.si.gaussian` and `infotheory.mi.gaussian`. Each finding is now counted once: robustness on est.gaussian, and the je/ce/mi affinity on je and ce.
- **Coverage.** Three probe runs complete the proposed scaled_offset checks: per-channel entropy shifts for all four scales, mi/tlmi/gc_k1 invariance between normalise=True and normalise=False, and the je ln(s_i s_j) shift. A statistical run replaces the cce iid sanity range.
- **Score.** gc_gaussian_k-1 changes from variant, 8 to matches, 9.
- **Timing.** The gc auto `check_cost` now quotes all three measured timings and gives the per-pair cost relative to the fixed-lag unit (about 200× at P=5, T=1024) as the basis for `moderate`. (Superseded: after the review-stage forced re-run the record's `check_cost` quotes four timings of the T = 4096 params probe and about 270× at P=5, T=1024.)

### Corrections to trace-stage fields

- **`infotheory.est.gaussian` robustness candidate.** It now says xme_gaussian_k10 also raises, for a different reason. The Calculator line citation is updated to 336-338 at HEAD.
- **`infotheory.xme.gaussian_k10` `units_and_range`.** The raising behaviour is added.
- **Note intro attribution.** The remaining trace issue (attribution of the `itg_fix/*` probe numbers) is already resolved: the intro paragraph names `itg_fix/mi_calls.py` and `itg_fix/ce_probe.py`.

## Open questions: answered

1. **Which JIDT commit built the bundled jar?**
   - *Partly answered.* The jar is not a v1.6.1 tag build: `jar-provenance` shows the Gaussian entropy class extends `EntropyCalculatorMultiVariateCommon`, which the tag lacks.
   - *Still open:* the exact commit. A byte-level comparison against rebuilds of candidate commits around a9b6b3e (2024-04-20) would settle it.
   - *Value impact:* none found. The 72 direct calls match the closed forms to 5.5e-15.
2. **What did `_compute_entropy_rates` intend, and should di be redefined?**
   - *Answered by the evidence.* The code computes neither H(Y^n) nor n × the entropy rate. It computes Σ H(Y^m)/m, which exceeds Massey's term by B(target) ≥ 0. This produces spurious DI of 0.174 nats from an independent source into the AR(2) row.
   - *Recommendation.* Massey's form, Σ_m H(Y_m|Y^{m−1}) − CCE (`_kernels_infotheory.gaussian_di(form='massey')`), is the canonical quantity. The kozachenko and kernel di configurations share the code path and need the same decision.
   - *Owner.* Whether to change pyspi is the maintainers' decision.
3. **History length for cce and di: 5 or 10?**
   - The code uses 5 for both (`configured-n`). The documentation says 10.
   - Using 10 roughly doubles the values: cce 6.12 → 11.95, di 0.77 → 1.68 on var2_bivariate 0→1.
   - Intent cannot be settled from code. Either the documentation or the `n` default must change.
4. **Factor ½ in gc_gaussian.**
   - The value is exactly F/2 (pyspi/F = 0.5).
   - The cheapest consistent fix is documentation: "half the log residual-variance ratio (= Gaussian transfer entropy)". Doubling the value would match Geweke's F, but it would change every stored value. Rankings are unaffected either way.
5. **Should ce keep the 'directed' label? Should je, ce and mi all stay?**
   - ce is exactly symmetric under the default pipeline (≤ 2e-15), and je and ce are exact affine functions of mi (≤ 4.4e-16).
   - Under z-scoring the three carry identical information, so keeping one of them, mi, loses nothing.
   - The 'directed' label is only meaningful with `normalise=False`.
6. **What is xme meant to be?**
   - *Unanswerable from the code or the documentation.*
   - *What the code computes:* exactly H(x_j(t) | x_i(t−1..t−(k−1))), with no target past.
   - *Whether it is a bug:* k=1 and k=2 coincide, and k=10 misses lag 10. Under any reading where "history length k" means k lags, the k=10 case is an off-by-one.
   - *What would settle it:* the author's intended definition.
7. **Is MAX_CORR_AIS stable enough for EEG?**
   - No, as an order estimator. It selected k = 3, 10, 9 and 8 for a true AR(2) across T = 64..4096, and k from 1 to 10 for white channels.
   - The TE values remain consistent with the chosen orders. On delayed_noise the long source orders even capture lags 3 and 5.
   - For EEG, a fixed order, or an information criterion on the bivariate VAR, would give comparable model orders across pairs. The selection is per process, so caching it per process is a bit-identical speed-up regardless.
8. **Line-number convention (briefing pack vs `grep -n`).**
   - Records cite the supplement section, the page and the `grep -n` line in `supplement.txt`.
   - The checks did not need supplement line numbers beyond those already cited.
   - Settling the convention is a synthesis-stage decision. The recommendation is the `grep -n` line, because it is reproducible from the text file.

## Optimisation

Workflow 3 (group infotheory) is a short, profile-based search for where the time goes and which avenues look promising; nothing in pyspi was changed. Walls below are from `audit/profiles/summary.json` (quiet machine); percentages are shares of the profiled SPI call (`tests/regression/harness.py:compute_spi`) in the cProfile run at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`). That profiling pass ran contended for the JIDT units (profiled walls 1.3-20x the quiet walls), so its walls are not used, only its shares. cProfile cannot see inside Java: JPype dispatches JIDT methods without profiler entries, so the Java time is booked as self time of the calling pyspi frame, and every `opaque_share` in these records (0.0002-0.04) is a lower bound on JIT/JNI time, not an estimate of it. `expected_speedup` is the unit's per-window speedup at 118x1024 as an order of magnitude; risks carry the `unverified-` prefix unless a micro-benchmark diff file backs them. Agreement with today's output means only that a candidate reproduces it.

### Walls

| Unit | 8x1024 fresh | 32x1024 fresh | 118x1024 fresh (measure_mode) | 8x1024 family-warm | 118x1024 family-warm projected (measure_mode) |
|---|---|---|---|---|---|
| `je.gaussian` | 0.0299 s | 0.519 s | 11 s (measured) | 0.0293 s | 7.18 s (extrapolated-pairs-from-32) |
| `ce.gaussian` | 0.0297 s | 0.522 s | 11.6 s (measured) | 0.000795 s | 0.0444 s (extrapolated-pairs-from-32) |
| `cce.gaussian` | 1.41 s | 26.6 s | 370 s projected (extrapolated-pairs-from-32) | 1.49 s | 367 s (extrapolated-pairs-from-32) |
| `xme.gaussian_k1` | 0.115 s | 2.1 s | 43.7 s (measured) | 0.114 s | 30.2 s (extrapolated-pairs-from-32) |
| `xme.gaussian_k10` | 0.126 s | 2.52 s | 59 s (measured) | 0.124 s | 34.9 s (extrapolated-pairs-from-32) |
| `di.gaussian` | 1.52 s | 27.2 s | 378 s projected (extrapolated-pairs-from-32) | 0.0203 s | 10.2 s (extrapolated-pairs-from-32) |
| `si.gaussian` | 0.341 s | 6.41 s | 132 s (measured) | 0.337 s | 89.4 s (extrapolated-pairs-from-32) |
| `mi.gaussian` | 0.0084 s | 0.269 s | 4.59 s (measured) | 0.00829 s | 3.26 s (extrapolated-pairs-from-32) |
| `tlmi.gaussian` | 0.00812 s | 0.149 s | 17.2 s (measured) | 0.0081 s | 2 s (extrapolated-pairs-from-32) |
| `te.gaussian_auto-MAX_CORR_AIS_kmax-10_taumax-2` | 0.699 s | 15 s | 209 s projected (extrapolated-pairs-from-32) | 0.757 s | 205 s (extrapolated-pairs-from-32) |
| `te.gaussian_k-1_kt-1_l-1_lt-1` | 0.0146 s | 0.385 s | 6.05 s (measured) | 0.0142 s | 6.23 s (extrapolated-pairs-from-32) |
| `te.symbolic` | 0.0146 s | 0.388 s | 6.25 s (measured) | 0.0429 s | 15.9 s (extrapolated-pairs-from-32) |

`infotheory.est.gaussian` is infrastructure (no timing of its own; filled from the mi.gaussian profile). The 118x1024 fresh `tlmi.gaussian` wall (17.2 s, one repeat) is out of line with its family-warm projection (2.0 s) and with mi (4.59 s); its 8x1024 fresh wall_cv is 1.57, so treat it as noisy.

### Where the time goes

- **Single-call units (mi, tlmi, gc_k1, te_symbolic).** One JIDT call per ordered pair; the pyspi `bivariate` frame holds 97.7-99.8% of the profiled call as self time (mi 97.9% cum / 94.7% self at `infotheory.py:267`, tlmi 97.7% / 92.6% at `:301`, gc_k1 98.3% / 94.4% and te_symbolic_k-1 99.8% / 99.3% at `:376`); 1-D `JArray` conversions are at most 4.2% (tlmi). The JIDT side is mostly per-call overhead: a micro-benchmark component (`audit/profiles/micro/opp_it_gaussian-1.json`, `components`) puts one Gaussian MI call at T=1024 (initialise, two 1-D conversions, setObservations, compute) at 0.13 ms, of which the two conversions are ~0.01 ms (8%); the rest is JPype dispatch and JIDT's per-call object work. This is why the fresh T exponents are 0.7-0.8.
- **je / ce.** `_compute_joint_entropy` (`:159`) holds 98.7% (je) and 93.5% (ce) with a T x 2 `JArray(JDouble, 2)` conversion per call; ce adds `_compute_entropy` 5.6%. The joint-entropy cache already runs one call per unordered pair.
- **Composite units (si, xme, cce, di).** Two to ten JIDT entropy calls per pair, each with a 2-D conversion. The component timing shows the conversion is the cost: a 1023 x 4 `JArray(JDouble, 2)` conversion takes 1.005-1.009 ms (`components.conversion_2d_by_shape["1023x4"]`, `components.jarray_2d_1023x4_s`) of a 1.047 ms 4-D entropy call, and JIDT's entropy on a prebuilt array takes 0.043 ms. si: `_compute_conditional_entropy` 99.7% cum / 99.4% self (168 calls). xme: `CrossmapEntropy.bivariate` 99.7% (k1) / 99.8% (k10) cum, np.append 1.2% at k10. cce: `_compute_causal_entropy` 100% cum with 55.1% self (np.concatenate iterating Java `double[][]` rows returned by `MatrixUtils.makeDelayEmbeddingVector`: 57,400 `jpype __next__` calls, 2.4%), `_compute_conditional_entropy` 42.2%. di: `_compute_causal_entropy` 94.0% cum (28.0% self, conditional entropies 58.9%, `__next__` 6.9%), `_compute_entropy_rates` 5.9%.
- **gc auto.** `TransferEntropy.bivariate` 99.9% cum / 99.6% self: JIDT's MAX_CORR_AIS search (19 candidates for the target and 19 for the source) and the TE in one Java call; 12.5 ms per ordered pair against 0.26 ms for the fixed k=1 TE.

### Opportunities

- **it_gaussian-1 (replace-library; je, ce, mi, tlmi, gc_k1, te_symbolic, est.gaussian): closed forms from one covariance.** Measured: `audit/profiles/micro/opp_it_gaussian-1.py` / `.json` computes mi, je, ce, tlmi, gc_k1 and te_symbolic_k-10 for all pairs from one ddof=1 covariance of the z-scored rows (lag blocks x(t-m), x(t+1) for tlmi and TE) with a batched slogdet, against the real `harness.compute_spi` call on the same Data at 8x1024 and 32x1024, and times the candidates alone at 118x1024 (`shapes["118x1024_candidate_only"]`). Same-run speedups on the shared machine are 9-28x at 8x1024 and 224-643x at 32x1024 in the latest run (9-29x and 174-454x in the first; the 32x1024 same-run baselines are noisy, so ce and tlmi moved by more than 30% between runs). The 118x1024 timings are the per-window basis: the candidates take mi 3.8 ms, je 3.8 ms, ce 3.6 ms, tlmi 4.7 ms, gc_k1 11.8 ms and te_symbolic_k-10 146 ms, each including Data construction (2.9 ms). Against the quiet summary.json walls at 118x1024 that is mi 1200x fresh (4.59 s measured) / 850x family-warm (3.26 s projected), je 2900x / 1900x (11.0 s / 7.18 s), ce 3200x fresh but only 12x family-warm (after je ce is already a cache lookup, 0.044 s), tlmi ~3600x fresh (17.2 s, one noisy repeat; the record's figure) / 420x family-warm (2.0 s), gc_k1 510x / 530x (6.05 s / 6.23 s), and te_symbolic 101x for both identifiers (146 + 11.8 ms against 15.9 s family-warm, which counts Data construction twice; 102x with it counted once; te_symbolic_k-1 is the gc_k1 closed form). Expected: mi and est.gaussian 800-1200x, je ~1900x (~2900x fresh), ce ~10x family-warm (~3000x fresh), tlmi ~400x, gc_k1 ~500x, te_symbolic ~100x. These ratios count Data construction in the candidate, while the summary.json walls time `compute_spi` on a prebuilt Data, so they move with Data-construction time. A Workflow 4c run (`--split118`, `shapes["118x1024_split"]`) times the math alone on one prebuilt Data (Data construction 3.3 ms on its own) and one fresh one-shot pyspi call per identifier: mi 0.93 ms, je 0.89 ms, gc_k1 9.3 ms, te_symbolic 119.6 ms (k-10) + 9.4 ms (k-1), which against the family-warm bases gives the like-for-like ratios mi 3504x, je 8106x, gc_k1 669x and te_symbolic 123x. With that run's Data construction added the ratios fall to mi 767x, je 1709x and gc_k1 493x, below the with-Data figures above, so the recorded lower bounds are conservative figures that rest on the like-for-like ratios. Two of the bases are themselves low: the pairs-from-32 extrapolations for je (7.18 s) and mi (3.26 s) are 0.65x and 0.71x their single fresh 118x1024 measurements (11.0 s and 4.59 s), while te_symbolic's k-10 identifier, about two thirds of its 15.9 s basis, was never profiled or measured at 118x1024 (the 4c one-shot ratio k-10 / k-1 of 2.1 puts it near 13 s against the extrapolated 10.9 s). te_symbolic's saving holds only while known bug 1 stands: the candidate reproduces today's Gaussian TE output, and fixing bug 1 (a semantic change) replaces the avenue. Output: max_abs_diff <= 1.2e-14 at 8x1024 and <= 6.8e-15 at 32x1024 over all six identifiers in both runs, and <= 5.3e-15 at 118x1024 over the seven identifiers of the 4c run (one-shot baselines), NaN masks equal, not bit-identical: **within-tolerance (measured)**. The relative differences in the JSON are larger on near-zero entries: max_rel_diff at 32x1024 is 2.2e-6 (gc_k1), 8.0e-7 (te_symbolic_k-10), 3.8e-9 (tlmi) and 7.5e-10 (mi), at 8x1024 2.2e-9 (te_symbolic_k-10) or less, and at 118x1024 5.8e-4 (te_symbolic_k-10, |value| <= ~9e-12), 1.9e-5 (gc_k1 and te_symbolic_k-1), 1.3e-5 (tlmi) and 1.2e-5 (mi); the entry carrying each maximum has |value| <= max_abs/max_rel (~4e-10 for gc_k1 at 32x1024). The records quote both figures; the te_symbolic record labels its figures as te_symbolic_k-10 and gives gc_k1's for te_symbolic_k-1 (bitwise gc_gaussian_k-1), whose 2.2e-6 is the unit's largest. The check stage's ddof and NORMALISE findings carry over: ddof=1 throughout (it cancels in mi/tlmi/TE, and je/ce use JIDT's own ddof=1), and NORMALISE has no effect on the Gaussian values (MI/TE calculators report false; the entropy calculator reports true but never normalises).
- **it_gaussian-2 (replace-library; si, xme_k1, xme_k10, cce, di, gc auto): closed forms for the composites.** Same idea with lagged covariances: si from one lag-1 covariance, xme from the source-lag block with and without the target sample, cce/di from one lagged covariance per term (each term m = 1..5 has its own sample window, so five covariances), gc auto from Gaussian AIS of each (k, tau) candidate per channel. Not measured; basis = the profile frames above plus the component timing (each JIDT 2-D entropy call costs ~1 ms, a batched slogdet of a <= 11 x 11 sub-matrix microseconds). Expected: si 50-300x, xme 30-100x, cce/di/gc auto 20-100x. Risk unverified-within-tolerance: the check stage's numpy closed forms reproduce pyspi to <= 8.9e-16 (si), 5.5e-14 / 8.5e-14 (xme k1/k10), 2.3e-14 (cce, di), 5.3e-14 with identical selections (gc auto).
- **it_gaussian-3 (cache-sharing; gc auto): MAX_CORR_AIS once per channel.** The selection depends only on the channel's own series (Gaussian AIS has no jitter; the check stage found identical selections for every ordered pair on 9/9 probe runs), so P searches replace 2P(P-1), and each pair then runs one fixed-embedding TE with the cached (k, k_tau, l, l_tau). Expected 10-25x (the remaining TE with selected embeddings costs a few times the 0.26 ms k=1 TE, against 12.5 ms per pair now). Not measured; unverified-bit-identical. Effort S, and it keeps JIDT. it_gaussian-2 runs the search once per channel itself (Gaussian AIS of every (k, tau) candidate) and computes the TE without JIDT, so the record lists it in `subsumed_by` and build_csv ranks this avenue with no saving on top of it.
- **it_gaussian-4 (vectorise-batch; si, xme_k1, xme_k10, cce, di, est.gaussian): cut JPype marshalling.** `jpype.JArray.of(ndarray)` converts a 1023 x 4 array in 0.173-0.176 ms instead of 1.005-1.009 ms (`components.conversion_2d_by_shape["1023x4"]` and `components.jarray_of_2d_1023x4_s` / `jarray_2d_1023x4_s`), cuts a 4-D JIDT Gaussian entropy call including the conversion from 1.047 ms to 0.216 ms (`jidt_gaussian_entropy_4d_call_incl_conversion_s`, `jidt_gaussian_entropy_4d_call_jarray_of_s`), and gives a bitwise-equal JIDT entropy (component check in `opp_it_gaussian-1.json`, `components`: the it_gaussian-4 change itself, timed on one 1023 x 4 array and one entropy call by the it_gaussian-1 script; no it_gaussian-4 candidate was run on any unit, so the records quote these figures inside a profile basis and do not tag them microbench); building the cce/di delay embeddings in numpy removes the row iteration over Java arrays (cce 55.1% self + 2.4%). Expected ~3x (si, xme), 5-10x (cce), 3-5x (di); unverified-bit-identical (the same doubles reach JIDT). It needs nothing from the cross-cutting xc-jni-overhead (which moves the per-pair loop into Java, 1.01-1.07x): depends_on is empty. The latest run of the micro adds the conversion cost by shape (`components.conversion_2d_by_shape`): `JArray(JDouble, 2)` takes ~1.0 ms for every T x d array from 1023 x 1 to 1014 x 10 (row-by-row), `JArray.of` 0.17-0.22 ms. The same change also applies to the box-kernel units (listed there under this id): every unit that lists it_gaussian-4 carries the same opportunity text, and the part of the change that applies to the unit (which 2-D calls, and the delay embeddings for cce/di only) is named at the end of its speedup_basis. This is the cheap avenue if JIDT is kept; it_gaussian-2 supersedes it: the si, xme_k1, xme_k10, cce and di records list it_gaussian-2 in `subsumed_by`, so build_csv ranks it_gaussian-4 there with no saving on top of it (README "Optimisation columns", ranking rule 4).
- **it_gaussian-5 (cache-sharing; te_symbolic): reuse gc_k1.** While known bug 1 stands, te_symbolic_k-1 is bitwise gc_gaussian_k-1 (check run `k1-identity-gc1`, rtol = atol = 0), so it can be served from the gc_k1 result; removes the k-1 increment (0.37 of 1.16 s at 32x1024 family-warm), ~1.5x for the unit. Fixing bug 1 (a real symbolic estimator) is semantic and supersedes this. It is also contained in it_gaussian-1, which computes te_symbolic_k-1 by the same closed form as gc_gaussian_k-1 (11.8 ms per 118x1024 window, `opp_it_gaussian-1.json`): on top of it this avenue saves at most that call, so the record lists it_gaussian-1 in `subsumed_by` (README "Optimisation columns", ranking rule 4).
- **JNI per-call share (cross-cutting xc-jni-overhead).** For the 1-D single-call units the JNI/JPype share is effectively the whole call: 0.13 ms per Gaussian MI call at T=1024 against microseconds of arithmetic; for the composites the 2-D conversion alone is ~96% of a 4-D entropy call. Both are removed entirely by it_gaussian-1/-2, which the cross-cutting row lists in `subsumed_by` on all twelve units.

### Refuted or narrowed seed items

- **di: cache the entropy-rate sum per target.** `_compute_entropy_rates` is 5.9% of the fresh di profile (under the ~10% bar): refuted for the fresh unit. It is ~100% of the family-warm increment (cce's causal entropies are reused through `data.causal_entropy`), 10.2 s projected at 118x1024, where a per-target cache would cut ~(P-1)x; it_gaussian-2 subsumes it, so it is not listed separately.
- **gc auto: search once per channel** is confirmed (it_gaussian-3).
- **The family in closed form from one covariance** is confirmed and measured for the single-call units (it_gaussian-1) and carried by complexity for the composites (it_gaussian-2).

### Shared versus incremental cost

Three caches on the Data object already share work: `data.joint_entropy` (je -> ce: ce's family-warm increment is 0.8 ms at 8x1024 against 29.7 ms fresh, 0.044 s projected at 118x1024), `data.entropy` (per-channel marginals for ce) and `data.causal_entropy` (cce -> di: di's increment 20 ms vs 1.52 s fresh). Everything else (mi, tlmi, gc, te_symbolic, xme, si, gc auto) is incremental: no state is shared, and te_symbolic_k-1 recomputes gc_k1.

### EEG scaling (retained selection)

Per-window cost at 118 channels = family-warm per-identifier increment x number of EEG identifiers (one identifier each, two for te_symbolic):

| Unit | EEG identifiers | Per-window cost at 118 | Note |
|---|---|---|---|
| je.gaussian | 1 | 7.2 s projected family-warm (11.0 s measured fresh) | |
| ce.gaussian | 1 | 0.044 s projected family-warm | the `data.joint_entropy` cache from je makes ce a lookup when je runs first (fresh 11.6 s) |
| mi.gaussian | 1 | 3.3 s projected family-warm (4.6 s measured fresh) | historical EEG row ~1.8 s |
| tlmi.gaussian | 1 | 2.0 s projected family-warm (17.2 s measured fresh, noisy) | historical ~1.8 s |
| te.gaussian_k-1 (gc_k1) | 1 | 6.2 s projected family-warm (6.05 s measured fresh) | |
| te.symbolic | 2 | 15.9 s projected family-warm: k-1 0.37 s and k-10 0.78 s at 32x1024, x 13.9 pair ratio = 5.2 + 10.9 s | historical 31.6 s per window; no cache between the two identifiers or with gc_k1 |

About 35-55 s per window for the retained Gaussian set; it_gaussian-1 would bring it to about 0.2 s (sum of the measured 118x1024 candidate times: je 3.8 + ce 3.6 + mi 3.8 + tlmi 4.7 + gc_k1 11.8 + te_symbolic 146 + 11.8 ms, Data construction counted in each).
