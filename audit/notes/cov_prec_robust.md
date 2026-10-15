# cov_prec_robust: robust covariance and precision (EllipticEnvelope, MinCovDet)

Stage: define, trace and check. The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does, from scripts in
`/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/cov_prec_robust/probes/`
(p1.py to p4.py), run with the project venv: scikit-learn 1.9.1, scipy 1.17.1, numpy 2.5.4. The regression goldens are
snapshots of current output. They are not used as a reference here.
The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections, including "Candidate findings" below, are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Units (records in `audit/records/`):

| unit_id | identifiers | in EEG selection |
|---|---|---|
| basic.cov.EllipticEnvelope | cov_EllipticEnvelope, cov-sq_EllipticEnvelope | no |
| basic.cov.MinCovDet | cov_MinCovDet, cov-sq_MinCovDet | yes (both) |
| basic.prec.EllipticEnvelope | prec_EllipticEnvelope, prec-sq_EllipticEnvelope | yes (both) |
| basic.prec.MinCovDet | prec_MinCovDet, prec-sq_MinCovDet | yes (both) |

All four units go through one code path and one sklearn fit. Most of the analysis is shared, so it is given once.
The per-unit sections below only add what differs.

## Shared trace

pyspi side:

1. `pyspi/data.py:184-195`. Detrending is optional. Each channel is then z-scored with `scipy.stats.zscore(axis=1, ddof=1)` over the
   **whole** sample, outliers included. The mean and SD used here are not robust.
2. `pyspi/calculator.py:329-335` (HEAD a1f20c7: :329 `multivariate`, :332 `fill_diagonal`, :335 table store). `multivariate(dataset)` is called, the diagonal is set to NaN and the table is stored.
3. `pyspi/statistics/basic.py:9-29`. `Estimators(Undirected, Signed)` adds `-sq` to the identifier, adds the `unsigned` label and sets `issigned=False`.
   No estimator hyperparameters are forwarded.
4. `basic.py:31-44` `_from_cache`. The fit is cached under `data.covariance[estimator_name]`, and this one fit is shared by cov, prec, squared and unsquared.
   `z = data.to_numpy(squeeze=True).T` has shape T x P. All warnings are silenced (`:37-38`). `getattr(sklearn.covariance, est)().fit(z)` runs with
   every default, including `random_state=None`.
5. `pyspi/statistics/basic.py:46-54`. The code reads `covariance_` or `precision_`, then sets its diagonal to NaN **in place on the cached array**. It returns that array,
   or its square. `Undirected.multivariate` (`base.py:132-138`) never runs (REVIEW M1).

sklearn side (`.venv/lib/python3.12/site-packages/sklearn/covariance/`):

- `_elliptic_envelope.py:166-185`. `EllipticEnvelope.fit` is `super().fit(X)` (that is, `MinCovDet.fit`) plus `offset_` (`:184`), the outlier threshold.
  `contamination=0.1` never touches `covariance_` or `precision_`.
- `_robust_covariance.py:770-822` `MinCovDet.fit`. `check_random_state(None)` returns the global `np.random.mtrand._rand`
  (`sklearn/utils/validation.py:1477-1478`). The method then runs `fast_mcd`, `correct_covariance` and `reweight_covariance`.
- `fast_mcd` (`:399-618`) sets h = ceil((n+p+1)/2) (`:476`).
  - For n <= 500 it uses 30 random h-subsets with 2 C-steps each, keeps the best 10 and refines them (`:590-616`).
  - For n > 500 it uses random subsets of about 300 samples with 500 trials in total, then merges (`:511-589`).
  - `_c_step` (`:114-213`) draws its random start at `:129` and runs concentration steps based on `det(MLE cov)`.
- `correct_covariance` (`:824-861`) divides the raw distances by the Croux-Haesbroeck factor c(p, h/n) (`:216-250`, `:858-860`).
- `reweight_covariance` (`:863-925`) keeps rows with d^2 < chi2_{p,0.975}. It computes their MLE covariance (`np.cov(bias=1)`, `_empirical_covariance.py:107`) and multiplies it
  by c(p, 0.975) (`:917-920`). `precision_ = scipy.linalg.pinvh(covariance_)` (`_empirical_covariance.py:213`).

The supplement documents scikit-learn 0.24.1. In that version
(`https://raw.githubusercontent.com/scikit-learn/scikit-learn/0.24.1/sklearn/covariance/_robust_covariance.py`):

- the raw correction is `np.median(dist_)/chi2(p).isf(0.5)` (`:707`);
- the reweighted covariance is stored with **no** consistency factor (`:756`).

## Canonical definition

The MCD is defined in Rousseeuw (1985). FAST-MCD is Rousseeuw & Van Driessen (1999), Technometrics 41:212-223; the supplement cites it as [3] for EllipticEnvelope.

- **Raw MCD.** Take the h-subset that minimises det(S_H). h = floor((n+p+1)/2) gives maximal breakdown. Its mean and its consistency-corrected covariance are the raw estimates.
- **Reweighting.** Hard-reject points with robust d^2 > chi2_{p,0.975}. Recompute the mean and covariance on the rest, with a consistency factor
  c_alpha = alpha / F_{chi2_{p+2}}(chi2_{p,alpha}) (Croux & Haesbroeck 1999, JMVA 71:161-190). Optional small-sample factors are given by Pison et al. 2002, Metrika 55:111.
- **Precision and partial correlation.** The precision is the inverse of that scatter. The partial correlation is -Omega_ij / sqrt(Omega_ii Omega_jj) (Whittaker 1990, ch. 5).
- **Elliptic envelope.** This is an outlier rule (the ellipsoid d^2 <= threshold), not a different covariance estimator. The canonical covariance for both "estimators" is therefore the reweighted MCD.

## basic.cov.MinCovDet (cov_MinCovDet, cov-sq_MinCovDet)

- **Documented** (supplement §1.1.1, pp.3-4): "Due to the z-scoring of the time series, the correlation and covariance
  matrices are equivalent and thus the covariance statistic is within [-1, 1]. We use v0.24.1 of scikit-learn ...
  the minimum covariance determinant (MinCovDet) methods for outlier removal". GitBook basic-statistics.md:21-25 says the same. The `-sq` variant is
  not described.
- **Implemented**: the reweighted FAST-MCD covariance of the z-scored data (trace above). The output is the raw `covariance_`
  entry with no normalisation to a correlation.
- **Deviation from documented**:
  - **Range claim.** The robust scatter of z-scored data has a diagonal that is not 1:
    - contaminated, T=4096: diag [0.193, 0.189], so cov[0,1] = 0.151, while the clean core correlation is about 0.8.
    - iid_gaussian, T=64: diag between 0.43 and 1.17.
    - Values above 1 occur: 1.141 on cointegrated T=256 and 1.008 on trending T=256, both the maximum over 20 seeds. On rank-deficient shared_envelope (x2 == x0) the value is 1.0478.
      In that last case every sample passes reweighting and the MLE covariance is multiplied by c(4, 0.975) = 1.0645.
  - **Version.** Since 0.24.1 the correction and reweighting have changed.
    - p4.py emulates the 0.24.1 steps from scratch on the same raw fit. Median off-diagonal ratio, new/old: 1.0555 for p=5 and 1.0645 for p=4.
    - The reweighted support size can also change: 57 vs 53 on sparse_precision T=64.
    - This emulation is a hypothesis-level reading of the old source. 0.24.1 itself was not run.
  - **Stochastic output** is not mentioned in the supplement. tests/README.md does document it.
- **Deviation from canonical** (variant): FAST-MCD is an approximate search, and here it is seeded from the global RNG. The other differences:
  - h is ceil instead of floor;
  - the reweighted covariance uses the MLE divisor n_w (robustbase uses n_w-1) and has no small-sample factor;
  - the raw factor is asymptotic rather than the median rescaling of RVD 1999;
  - the input is non-robustly z-scored, so the result is a robust covariance in units of the contaminated SD, not a robust correlation.
- **Proposed checks** (see the record):
  1. A from-scratch numpy implementation of the correction and reweight, given the sklearn raw support (tolerance 1e-10). It is
     labelled same-lineage: the arithmetic is independent, but the input comes from the sklearn search, which checks 2 and 3 test.
  2. A from-scratch C-step search with 2000 starts (optimality of the raw log-det).
  3. R robustbase::covMcd 0.99.2 with `initHsets` set to the sklearn raw support and `use.correction=FALSE`, after rescaling the divisor (tolerance).
  4. Identity of EE and MCD under the same seed (same-lineage, exact).
  5. A study of 20-seed and 20-process spread (statistical).
  6. The analytic clean correlation on contaminated and bivariate_gaussian, using the robust correlation C_ij/sqrt(C_ii C_jj) (statistical).
  7. A range scan (qualitative).
  8. A direction and symmetry check.
  9. trending with detrend.

## basic.cov.EllipticEnvelope (cov_EllipticEnvelope, cov-sq_EllipticEnvelope)

- **Documented**: the same §1.1.1 text, listed as "the elliptic envelope (EllipticEnvelope) [3]".
- **Implemented**: `EllipticEnvelope().fit` runs exactly `MinCovDet.fit` (`_elliptic_envelope.py:183`). Then only `offset_` is set (`:184`).
- **Briefing hypothesis 1 confirmed by probe.** Under `np.random.seed(0)` before each fit, `covariance_` and `precision_` are
  bit-identical to MinCovDet in all 16 cases (4 datasets x 4 T). In a Calculator pass the two identifiers use different cache keys,
  so they are fitted at different points of the RNG stream. They then differ only by RNG noise. Each pass is one unseeded draw:
  the first p3.py pass gave 0.0765 at T=64 (sparse_precision), 0.0519 at T=256 (iid_gaussian) and about 0.01 at T=1024.
  Four further fresh-process passes gave per-pass maxima of 0 to 0.146 at T=64, 0 to 0.0337 at T=256 (iid_gaussian T=256:
  0.0176, 0.0212, 0.017, 0.0337) and 0 to 0.0149 at T=1024. The difference is exactly 0 when both reach the same optimum.
- All other deviations and checks are as for cov.MinCovDet.

## basic.prec.MinCovDet (prec_MinCovDet, prec-sq_MinCovDet)

- **Documented** (supplement §1.1.2, p.4): "The precision matrix is the matrix inverse of the covariance matrix ... For
  normalized time-series data, the precision matrix is equivalent to the partial correlation ... and is within [-1, 1]."
- **Implemented**: `precision_ = pinvh(reweighted MCD covariance)` from the same cached fit as cov_MinCovDet, so cov and prec are mutually consistent
  within one run. It is returned raw.
- **Deviation from documented**: precision entries are not partial correlations.
  - Probe (bivariate_gaussian T=256): prec[2,3] = -2.6276, while -Omega/sqrt(Omega_ii Omega_jj) = +0.8126.
  - Values are unbounded: up to 368 on narrowband_f0-0.25 T=64.
  - This affects every prec_* estimator, not only the robust ones. Candidate; not in KNOWN_ISSUES.
  - The version drift and stochasticity are as for cov.
- **Deviation from canonical**: as for cov. In addition, `pinvh` silently returns a pseudo-inverse for singular scatter. On shared_envelope (x2 == x0)
  prec[0,2] = +0.345, and no warning reaches the user because basic.py:37-38 silences warnings.
- **Symmetry**: round-off asymmetry of 8.9e-16 (REVIEW M2).
- **Proposed checks**: as for cov, using inverses. In addition:
  - an analytic Omega of the z-scored process for sparse_precision and bivariate_gaussian (Omega_23 = -2.222, partial correlation 0.8), statistical;
  - inverse consistency and symmetry (tolerance);
  - condition number on singular and narrowband data (qualitative).

## basic.prec.EllipticEnvelope (prec_EllipticEnvelope, prec-sq_EllipticEnvelope)

Identical to prec.MinCovDet by construction (EE fit = MCD fit). Both prec units, with their -sq variants, are in the EEG selection, so
the selection contains four features that are two quantities, each duplicated up to RNG noise.

## Already documented (cited, not new)

- tests/README.md:145: the stochastic marker in the tolerance-policy row ("sklearn robust covariance draws from the global numpy RNG;
  reproducible only via seed_all()"). tests/README.md:134: the CALC_E2E_SKIP of cov*/prec*MinCovDet. tests/README.md:135: the
  shared-cache test that reseeds the 12 stochastic identifiers. tests/README.md:125 only lists *MinCovDet* and *EllipticEnvelope* in
  FORCE_SLOW, which is a timing marker.
- REVIEW M1: the Undirected mirror is bypassed.
- REVIEW M2: precision last-bit asymmetry.

## Candidate findings (to confirm in the check stage)

1. EllipticEnvelope equals MinCovDet; the identifiers are redundant and differ only by RNG position.
2. The magnitude of the seed sensitivity is large, not last-bit: 0.155 to 0.341 at T=64, 0.07 to 0.16 at T=256 and 0.0035 to 0.0087 at
   T=4096. Only the magnitude is a candidate. That the output is stochastic and needs seed_all() is already documented at tests/README.md:145.
3. Version drift from the documented sklearn 0.24.1: the c(p, 0.975) inflation is 1.055 to 1.104 for p = 5 down to 2, and the support can change.
4. cov: the documented "= correlation, within [-1,1]" claim is false for the robust estimator. The scale follows the non-robust z-score, and values above 1 are observed.
5. prec: the documented "= partial correlation, within [-1,1]" claim is false. The value has the opposite sign and is unbounded.
6. Silent pseudo-inverse and suppressed rank or determinant warnings on singular input.
7. The returned unsquared matrix is the cached sklearn attribute, and its diagonal is NaN-ed in place. This is value-neutral in Calculator.
8. Minor variants: h = ceil rather than floor, the MLE divisor in reweighting, and no small-sample factor.

## Open questions

- Which sklearn release introduced the Croux-Haesbroeck raw factor and the reweighting consistency factor? Not established here; only 0.24.1 and 1.9.1 were read.
- Does a full default-config Calculator run reproduce run to run? Hypothesis: yes, for a fixed SPI set and order, because importing
  `pyspi.statistics.misc` runs `np.random.seed(1717)` (`pyspi/lib/ids/numpy_dependence.py:11-12`). This was not probed: the full config needs the JVM.
- Is the remaining FAST-MCD spread at T=64 due to sklearn missing the global MCD optimum (to be answered by the optimality check) or to near-ties between h-subsets?
- Should the audit judge cov_* against "robust covariance" (matches, as a variant) or against the documented "correlation in [-1,1]"? This is a definition-under-review call for synthesis.
- Should robustbase `initHsets` take sklearn's raw support or the final reweighted support? Using the raw support isolates the reweighting step. The robustbase reweighting formula (which factors are
  applied with `use.correction=FALSE`) needs confirming from its source before the tolerance is fixed.

## Checks

Stage: CHECK. Every number below is quoted from `audit/checks/out/results/cov_prec_robust.json` (run ids are
`<unit>|<dataset>|<T>|<path>|<tag>`); 323 runs, 215 pass, 108 fail, 0 without reference. A fail is a reported
mismatch, not a script error. Most fails are qualitative propositions that the evidence refutes (seed independence,
documented range, documented partial correlation, RNG-stream independence, aliasing). The regression goldens were not used.

Files: `audit/checks/manifest/cov_prec_robust.json`, `audit/checks/refs/cov_prec_robust.py` (comparison venv, no sklearn import),
`audit/checks/r/cov_prec_robust.R` (system R 4.6.1, robustbase 0.99.2, rrcov 1.7.5),
`audit/checks/probes/cov_prec_robust_mcd.py` (project venv: fitted state, seed sweep, determinism, fresh processes, robust
correlation, aliasing). Exchange files, the R in/out CSVs and the Monte-Carlo caches are under `audit/checks/out/exchange/cov_prec_robust/`.

### What was compared with what

| tag | pyspi side | reference | kind |
|---|---|---|---|
| state / base | sklearn fit state (probe) / SPI output | from-scratch Croux-Haesbroeck c(p,h/n) on raw distances, chi2_{p,0.975} rejection, MLE divisor, c(p,0.975), `numpy.linalg.inv`, fed with sklearn's raw estimates | same-lineage |
| rdet | SPI output (pipeline) | `robustbase::covMcd(nsamp="deterministic", initHsets = sklearn raw_support_)`, h matched, `use.correction=FALSE`, divisor n_w-1 mapped to n_w | external-pkg |
| rrcov | SPI output (pipeline) | `rrcov::CovMcd` with its own FAST-MCD search (nsamp 500, 20 R seeds), h matched, `use.correction=FALSE`; atol = sklearn 20-seed range + rrcov 20-seed range (wide by construction; `ref_within_sklearn_range_only` records the stricter elementwise bound of the sklearn range alone) | external-pkg |
| stat / statc / rcorr / dir | SPI output / robust correlation (probe) | analytic targets in path units (Sigma_ij/(s_i s_j), Omega_ij s_i s_j, the clean correlation, signs) or the MLE estimate of the same draw, SE from a Monte-Carlo of the dataset design through the from-scratch reweighted FAST-MCD, Bonferroni over P(P-1)/2 entries | analytic / independent-impl |
| opt | sklearn raw h-subset (probe, seed_all(42)) | best log det of 2000 from-scratch C-step runs (1000 random h-subsets, 1000 random (p+1)-subsets) and 20 rrcov seeds | independent-impl |
| eeqmcd | EllipticEnvelope SPI | MinCovDet SPI, same seed | same-lineage |
| seeds / det / rngstream / fresh* / alias | probes and SPI outputs | numpy summaries of pyspi's own outputs (seed independence, determinism, RNG-stream independence, fresh-process reproducibility, aliasing): pyspi self-consistency, qualitative propositions | same-lineage |
| range / singular / breakdown / pcorr | probes and SPI outputs | qualitative propositions (documented range, singular input via numpy matrix_rank of the input, closer to the clean truth than EmpiricalCovariance, documented partial correlation) | literature-only / independent-impl / analytic |

Conventions of the two implementations, as executed:

* sklearn 1.9.1: `raw_covariance_` is the **uncorrected** MLE (divisor h) of the raw h-subset. The factor c(p,h/n) is applied only to `dist_`.
  `covariance_` is the reweighted MLE (divisor n_w) times c(p,0.975), and the factor is applied even when every point is kept.
  h = ceil((n+p+1)/2). Starts are random h-subsets (`_robust_covariance.py:129`); n <= 500 uses 30 starts, keeps the best 10 and refines them.
* robustbase 0.99.2 / rrcov 1.7.5: `raw.cov` = c(p,h/n) x cov(H) with divisor **h-1**. This holds to floating point, not bit-exactly: `ref_r_raw_cov_vs_sklearn_raw_conv_max_rel` is 1.1e-15 to 1.25e-13 over the 28 rdet runs (1.25e-13 on `sparse_precision|1024|pipeline|rdet`, 8.1e-14 on `iid_gaussian|1024|pipeline|rdet`).
  `cov` = `cov.wt` (divisor n_w-1) x c(p,0.975), applied **only if n_w != n**. The default h is floor((n+p+1)/2) (alpha = 0.5). The default
  `use.correction = TRUE` adds the Pison et al. small-sample factors. FAST-MCD starts from random (p+1)-subsets (nsamp = 500), as in Rousseeuw & Van Driessen 1999.

### basic.cov.MinCovDet: variant, score 6

* Arithmetic given the raw subset:
  * state 18/18 (max_rel 1.54e-13 over the whole fitted state) and base 10/10 (max_abs 1.11e-16), same-lineage.
  * External: rdet 7/8 with max_abs 6.66e-16, max_rel 3.51e-15.
  * The failing `basic.cov.MinCovDet|bivariate_gaussian|256|pipeline|rdet` differs by **one** reweighting point (robustbase keeps 245, sklearn 244).
    That point lies in the band [q, q h/(h-1)) created by the different raw-scatter divisor (`ref_disagreements_all_in_divisor_band` = 1).
    This is a convention difference between the packages, recorded and not tuned away (max_abs 0.0118).
* Search, external: rrcov 5/6 within the combined seed range. This atol is wide by construction (sum of two 20-seed ranges), so a pass here is weak
  evidence and not value-level agreement. With the sklearn 20-seed range alone as the elementwise bound (`ref_within_sklearn_range_only`) the count is 4/6:
  `bivariate_gaussian|1024|pipeline|rrcov` then also fails, with 2 off-diagonal entries outside (`ref_n_offdiag_outside_sklearn_range` = 2).
  * `bivariate_gaussian|1024`: 0.0058 vs atol 0.0276. `contaminated|4096`: 0.00037 vs 0.0061.
  * Fail on `contaminated|256`: 0.00576 vs atol 0.00373. All 20 rrcov seeds reach raw log det -6.4858 to -6.4831. All 20 sklearn seeds stay at
    -6.4630 to -6.4591 (seed 42: -6.46297). So rrcov's search finds a lower-determinant h-subset that sklearn's search never reaches here.
* Optimality: 3/14 runs pass. sklearn's seed_all(42) raw subset is the best found (by 2000 from-scratch C-step runs and 20 rrcov seeds) only on
  delayed_noise|64, delayed_noise|1024 and iid_gaussian|1024. The search is not exhaustive: each "best found" is an upper bound on the minimum
  log det, so each gap below is a lower bound (a better search can only make sklearn look worse). Gaps in log det:
  * >= 0.046 (iid|64) and >= 0.128 (sparse|64);
  * 0.012 to 0.023 at T=256;
  * 4.4e-6 to 9.6e-4 at T >= 1024.
  * Several seeds that land on the best subset give identical output (spread 0.0): delayed_noise|64 (10 seeds), iid_gaussian|1024 (6), delayed_noise|1024 (3).
  * rrcov is not exhaustive either. On iid_gaussian|256 no rrcov seed reaches the from-scratch best. On `basic.cov.MinCovDet|sparse_precision|64|pipeline|opt`
    two rrcov seeds (`ref_rrcov_n_seeds_at_best` = 2) reach a log det 0.0059 below the 2000-start search (`ref_scratch_best_gap`).
  * The prec-unit opt runs solve the same raw problem with another search seed and found lower determinants on two datasets:
    iid_gaussian|256 best found -3.25164 (prec) vs -3.25023 (cov), so sklearn's gap is >= 0.0202 (`basic.prec.MinCovDet|iid_gaussian|256|pipeline|opt` ref_gap);
    sparse_precision|64 -4.69293 vs -4.69047, gap >= 0.131. The 3/14 count is unchanged by this.
* Analytic (stat, Monte-Carlo SE, Bonferroni): 11/12 pass.
  * bivariate_gaussian: max raw z 1.00/1.32/1.78 at T=256/1024/4096.
  * contaminated pipeline: target Sigma_clean/(s s) = 0.147 (the non-robust SD of the contaminated series sets the scale); max_abs 0.0036 at T=1024.
  * contaminated kernel: target 0.8, max_abs 0.0197.
  * trending, pipeline-detrend: raw z 1.14. delayed_noise null: raw z 0.78.
  * Fail on the iid_gaussian|1024 null: max_abs 0.124, raw z 3.65. The draw's own Pearson r(0,2) is 0.084, and robustbase reproduces the
    value to 1e-15 (rdet). The fail comes from the draw, not from the implementation.
* Same-draw comparison (statc, MCD minus MLE): 3/4. It fails on iid_gaussian|4096 with raw z 2.95 (max_abs 0.0246, MC SE 0.00995).
  * A scratch diagnostic checked the Monte-Carlo SE of this from-scratch estimator. An sklearn-based Monte-Carlo gave a difference SD of
    0.0191 vs 0.0185 at T=1024 and 0.0088 vs 0.0089 at T=4096. This was a diagnostic only, not a reference.
  * The remaining null exceedances (here and for prec) are borderline tail values on the iid draws. They are recorded as inconclusive
    statistical evidence, not as an implementation fault.
* Robust correlation (rcorr): 0.8087/0.8071/0.7905 at T=256/1024/4096 vs 0.8 (4/4 pass, max_z 1.30).
* Breakdown, 3/3: on the kernel contaminated data, |pyspi - 0.8| / |EmpiricalCovariance - 0.8| = 0.0139/0.0042/0.0032.
* Seeds and range:
  * 20-seed max spread: 0.155/0.068/0.013/0.0027 (iid_gaussian, T=64/256/1024/4096) and 0.257/0.061/0.015/0.0044 (sparse_precision).
  * Documented range [-1,1] violated: 1.141 (cointegrated|256), 1.008 (trending|256), 1.083 (shared_envelope|64).
* Determinism:
  * det 2/2: two seed_all(42) calls are bit-identical.
  * rngstream 0/2: after the other robust SPI the value moves by 0.111 (sparse_precision|64) and 0.0081 (bivariate_gaussian|1024).
  * freshfull: two fresh-process Calculator runs with the default config are identical.
  * freshbasic: with only the 8 robust SPIs they differ by 0.0765. Both processes seed the global RNG from OS entropy, so this size changes between check runs.
* Structure: sq and perm are bitwise; dir 2/2; symmetric exactly (max|A - A.T| = 0.0 over 48 SPI runs).
* Verdict:
  * The quantity is the reweighted FAST-MCD robust covariance, computed exactly as sklearn 1.9.1 defines it.
  * It deviates from the canonical MCD in interpretable ways: seed- and RNG-position-dependent local optima, a weaker start strategy than RVD 1999,
    ceil h, MLE divisor, and c(p,0.975) even when every point is kept.
  * It also deviates from the supplement: it is not a correlation in [-1,1], and the reweighting factor is absent from the documented 0.24.1.
  * Rubric anchor 6: "deviates from both the supplement and canonical in an interpretable way".
  * The family cap ("<= 7 unless R agrees to tolerance") is not binding at 6. The value-level external evidence is rdet: robustbase, started from
    sklearn's raw subset, agrees on 24/28 runs across the four units (max_rel 3.51e-15 cov, 2.97e-13 prec; the 4 fails are the one divisor-band point).
    The rrcov own-search runs (22/24 within the combined seed range, 20/24 with the sklearn range alone) carry little weight, because their atol is
    wide by construction (0.0276 on bivariate_gaussian|1024 cov, 0.313 on bivariate_gaussian|256 prec, 0.525 on contaminated|1024 prec). The score does not rest on them.
* EEG scope (Workflow 4b challenge probes, not check runs; scratchpad `wf4b_mid-eeg/mcd_highp_seeds.py`, `mcd_detwarn.py` and the re-check `wf4b2_mid-eeg/mcd_car_check.py`,
  sklearn MinCovDet on pyspi's z-scored Data, T=1024, 5 global seeds):
  * The checks above use P <= 5 and no average-referenced input. cov_MinCovDet and cov-sq_MinCovDet are retained for EEG, where the workflow runs up to
    about 118 channels and the example config sets `reference: car` (`examples/workflow/preflight.yaml:12`).
  * Seed spread grows with P: max 0.0207 / 0.0187 / 0.0267 / 0.046-0.047 and median 0.009 / 0.008 / 0.010 / 0.018 at P = 8 / 19 / 32 / 64 (iid, no reference),
    against a median |cov| of about 0.023. At EEG P the seed noise is of the order of the null-level values, larger than the 20-seed 0.013-0.015 at P <= 5.
  * With CAR (rank P-1) every h-subset scatter is singular, the exact-fit situation of Rousseeuw and Van Driessen (1999). Every fit raised 18-38 sklearn
    "Determinant has increased; this should not happen" warnings (`_robust_covariance.py:188-195`) at P = 5-64, and none without CAR. pyspi suppresses them
    (`basic.py:37-38`; the prec.MinCovDet record lists this as a candidate). The canonical objective is degenerate on CAR EEG input.
  * The score stays 6: it rests on the synthetic evidence above. These facts limit how far that evidence carries to EEG.

### basic.cov.EllipticEnvelope: variant, score 6

* eeqmcd: 8/8 runs **bitwise** equal to cov_MinCovDet under the same seed (same-lineage self-consistency).
* Its own external runs give the same pattern as MinCovDet: rdet 5/6 (the same divisor-band point) and rrcov 5/6 (the same contaminated|256 fail; 4/6 with the sklearn range alone).
* Analytic and statistical runs: stat 2/2, rcorr 1/1 (0.8071), breakdown 1/1 (0.0042).
* In a Calculator pass the identifier differs from cov_MinCovDet only through RNG position (rngstream: 0.111 at T=64).
* freshbasic happened to give identical values in this pair of fresh processes: both landed on the same local optimum. cov_MinCovDet, run in its own pair of fresh processes with the same configuration, gave different output (observed example max_abs 0.0765 in its stored run). Each process seeds the global RNG from OS entropy, so whether two fresh processes agree, and by how much they differ, can change between check runs for either unit.
* Score 6 as for MinCovDet. The label names the sklearn class whose fitted covariance this is. `contamination` never reaches the output.

### basic.prec.MinCovDet: variant, score 6

* Arithmetic given the raw subset:
  * state 18/18 and base 10/10 (max_rel 1.54e-13).
  * rdet 7/8: max_abs 2.45e-13, max_rel 2.97e-13. The bivariate_gaussian|256 run differs by the same single point (max_abs 0.0244).
* External search: rrcov 6/6 (contaminated|256: 0.106 vs atol 0.119), also 6/6 with the sklearn range alone. The atol reaches 0.313 (bivariate_gaussian|256)
  and 0.525 (contaminated|1024), so this is weak evidence, not value-level agreement.
* Analytic (stat, target Omega_ij s_i s_j): 10/11 pass.
  * bivariate_gaussian raw z 1.72/2.10/1.60; sparse_precision raw z 1.83/2.45/2.25; trending pipeline-detrend 2.00; contaminated kernel 0.38.
  * Fail on the iid_gaussian|1024 null (raw z 3.82).
  * statc 2/4: fails on iid|1024 (2.83) and iid|4096 (3.06).
* Documented definition: pcorr 0/2.
  * pyspi prec(2,3) = -2.480 where the documented partial correlation is +0.8; the Omega_z target is -2.191, max_z 18.0.
  * sparse_precision prec(0,1) = -0.611 vs +0.4 (target -0.542), max_z 14.3.
* range 0/5: 32.7 (cointegrated|256), 12.3 (trending|256), 1.60 (iid|64), 2.28 (shared_envelope|64), 368.3 (narrowband_f0-0.25|64).
* Singular input: shared_envelope|64 has input rank 3 of 4, yet the output is finite with prec(0,2) = 0.446. A pseudo-inverse is returned and no NaN appears.
* Optimality 3/12, with the same gaps as the shared fit (lower bounds; this unit's search found the lower bests quoted above for iid_gaussian|256 and sparse_precision|64).
* Seeds: 20-seed spread 0.817 (iid|64) and 0.803 (sparse|64), falling to 0.0028/0.0055 at T=4096.
* Breakdown 3/3: prec(0,1) is -2.573/-2.280/-2.045 vs the clean -2.222; empirical gives +0.28.
* Symmetric to round-off: 2.66e-15 (REVIEW M2, already documented).
* Score 6: deviates interpretably from the canonical MCD (as for cov), and the supplement's partial-correlation description does not hold (sign and scale).

### basic.prec.EllipticEnvelope: variant, score 6

* eeqmcd 8/8 bitwise equal to prec_MinCovDet.
* rdet 5/6 (max_rel 1.76e-14 on the passing runs); rrcov 6/6 (weak evidence: wide atol, see prec_MinCovDet).
* stat 2/2 (raw z 2.10 and 2.45); pcorr 0/2; singular: finite pseudo-inverse.
* rngstream: 0.134 (sparse_precision|64) and 0.0177 (bivariate_gaussian|1024).
* The EEG selection keeps prec_EllipticEnvelope and prec_MinCovDet (and their -sq variants) as separate features. They are one quantity that differs only by RNG position.

### Candidates

* Confirmed:
  * EllipticEnvelope == MinCovDet: eeqmcd 16/16 bitwise; rngstream shows the RNG-position difference.
  * Seed-spread magnitude, with its mechanism: FAST-MCD local optima. Seeds at the best subset agree exactly.
  * Documented cov range false.
  * Documented prec = partial correlation false.
  * Silent pseudo-inverse on singular input.
  * Aliasing of the cached attribute (alias 4/4 fail the no-aliasing proposition). It is value-neutral in the Calculator: sq, perm and base agree bitwise.
  * Minor conventions: ceil h, MLE divisor, c(p,0.975) when all points are kept.
* Inconclusive: the size of the drift from the documented sklearn 0.24.1. The 1.9.1 side is reproduced numerically, but 0.24.1 was not executed.
* New evidence beyond the trace candidates, folded into the seed-magnitude resolution:
  * sklearn's start strategy (random h-subsets, `_robust_covariance.py:129`) reaches lower-determinant subsets less often than rrcov's (p+1)-subset starts on contaminated data.
  * On contaminated|256 the sklearn seeds reach the best subset 0/20 times, against 11/20 for rrcov. On contaminated|64 sklearn's seed-42 subset misses the best by 0.020 in log det, while all 20 rrcov seeds reach it.

Trace-stage fields corrected in the records:
* stale citations: `pyspi/calculator.py` 226-244 -> 230-264 (:239, :256) and 294/297/300 -> 329/332/335 at HEAD a1f20c7; `tests/README.md` lines
  126/115/116/106 -> 145/134/135/125.
* `direction_convention` now says it was confirmed by the perm and dir runs.

Corrections after the adversarial review (results re-run with `--force`; pass/fail counts unchanged at 215/108):
* The robustbase raw-scatter convention identity is quoted as floating-point agreement (1.1e-15 to 1.25e-13), not 0.0.
* The det / rngstream / freshfull / freshbasic / alias / seeds runs are relabelled `same-lineage` in `refs/cov_prec_robust.py`: they summarise pyspi's own outputs.
* The rrcov runs carry the extra diagnostic `ref_within_sklearn_range_only`; their passes are described as weak evidence.
* Optimality gaps are described as lower bounds from a non-exhaustive search, with the lower bests found by the prec-unit opt runs.
* The freshbasic difference changed from the earlier run to 0.0765 because both processes seed from OS entropy; it is re-quoted.

The trace path for checks 2 and 5 (optimality on the kernel path, seed study on the pipeline path) was changed to the pipeline path for both.
MCD is affine equivariant, so the h-subset problem is the same, and the 20-seed study and the optimality runs then share data.
The "20 fresh-process Calculator runs" were replaced by a 20-seed sweep, with fresh-process evidence from the freshfull/freshbasic pairs.

## Open questions: answered

* **Which sklearn release introduced the Croux-Haesbroeck raw factor and the c(p,0.975) reweighting factor?** Not settled. The checks
  establish only that 1.9.1 applies both: the from-scratch state reproduction needs both factors to reach max_rel 1.54e-13. They also show that the executed
  conventions differ from the 0.24.1 source text. The release would be settled by reading the sklearn changelog or bisecting
  `sklearn/covariance/_robust_covariance.py` between 0.24.1 and 1.9.1. No sklearn install was allowed here.
* **Is a full default-config Calculator run reproducible run to run?** Yes, for the robust SPIs. Evidence:
  * `basic.cov.MinCovDet|sparse_precision|64|pipeline|freshfull` and the EllipticEnvelope twin: two fresh interpreter processes, each building
    `Calculator(dataset=x)` with the default config.yaml (which imports pyspi.statistics.misc and so runs `np.random.seed(1717)`) and computing the
    .statistics.basic block, give bit-identical cov_MinCovDet and cov_EllipticEnvelope.
  * With a config of only the 8 robust SPIs (misc never imported), the two processes differ by 0.0765 (`...|freshbasic`; the processes seed from OS entropy, so the size changes between check runs).
  * Reproducibility therefore rests on the misc import side effect and on a fixed SPI set and order. Changing either moves the value by up to 0.111 at T=64 (rngstream).
  * Not covered: reproducibility of SPIs after the basic block. The full compute was not run.
* **Is the T=64 seed spread caused by sklearn missing the global MCD optimum, or by near-ties between h-subsets?** It comes from missed optima, not ties:
  * Whenever two or more seeds reach the best h-subset, their outputs are identical: `spread_among_seeds_at_best` = 0.0 on delayed_noise|64
    (10 seeds), iid_gaussian|1024 (6) and delayed_noise|1024 (3).
  * At T=64 most seeds stop in other local optima: the log det gap of sklearn's seed-42 subset is at least 0.046 (iid) and at least 0.131 (sparse_precision; best found over the cov and prec opt runs), and 0/20 seeds reach the best on iid|64 and sparse|64.
  * The output spread (0.155/0.257) is the spread across those local optima.
* **Which definition should judge cov_* (and prec_*)?** Both forms are recorded and `definition_under_review` is true. Against the robust
  covariance/precision (canonical MCD) the units are variants that external packages reproduce. Against the documented "correlation /
  partial correlation within [-1, 1]" they fail outright: cov 1.141, cov 0.15 where the robust correlation is 0.81, prec -2.48 where the partial correlation is +0.8.
  The score of 6 reflects deviation from both; synthesis may choose the reading.
* **Which factors does robustbase covMcd apply with use.correction=FALSE, and should initHsets take the raw or the reweighted support?** Answered
  from the deparsed covMcd source in this R and confirmed numerically:
  * raw: c(p,h/n) times the h-subset covariance with divisor h-1;
  * reweighted: `cov.wt` (divisor n_w-1) times c(p,0.975), only when n_w != n;
  * no small-sample factor.
  * initHsets works only with nsamp = "deterministic", and only through robustbase::covMcd: rrcov::CovMcd silently ignores it and runs its six deterministic starts.
  * The raw support is the right input, because covMcd runs C-steps from it. sklearn's raw support is a C-step fixed point (`ref_r_best_equals_init` = 1 and n.csteps = 3 on every rdet run), so robustbase keeps it and the comparison isolates the correction and reweighting.

## Optimisation

Workflow 3, group basic. Walls are from `audit/profiles/summary.json` (quiet machine, all shapes measured, `measure_mode: measured` at 118x1024). Percentages are shares of the profiled `compute_spi` call. Nothing in pyspi was changed.

| unit (fresh) | 8x256 | 8x1024 | 32x1024 | 118x1024 (measured) |
|---|---|---|---|---|
| basic.cov.MinCovDet | 0.023 s | 0.318 s | 0.518 s | 3.49 s |
| basic.prec.MinCovDet | 0.023 s | 0.317 s | 0.517 s | 3.50 s |
| basic.cov.EllipticEnvelope | 0.023 s | 0.321 s | 0.519 s | 3.50 s |
| basic.prec.EllipticEnvelope | 0.023 s | 0.317 s | 0.517 s | 3.48 s |

**Complexity.** sklearn `_robust_covariance.py:fast_mcd` runs C-steps (`_c_step`). Each iteration costs one pinvh (a full eigh, P^3), one empirical covariance of the h-subset and one Mahalanobis pass over the sample. The number of trials depends on T, not P: 30 random starts for T <= 500. For T > 500 there are T//300 subsets with 500/(T//300) starts each (166 at T=1024), then a merged pass with 30 refinements (528 C-steps at T=1024). The measured T_exponent of 1.90 between 8x256 and 8x1024 is this regime switch at T=500, not a T^2 law. The P_exponent of 0.35 comes from 8->32 channels, where sklearn validation is still a large share. Going from 32 to 118 channels implies about 1.46.

**Hotspots.** The 8x1024 profile (`basic.cov.MinCovDet__8x1024.pstats`) has `_c_step` at 98.9%, `empirical_covariance` at 46.8% (of which `check_array` input validation is 26.9%) and pinvh at 17.4%. That mix changes with P. A profile at 118x1024 written by this stage (`audit/profiles/micro/prof_basic.cov.MinCovDet__118x1024.pstats`) shows `scipy/linalg/_decomp.py:eigh` at 57.4% self time (pinvh cumulative 62.5%, 1154 calls), `_c_step` self time at 14.9% (Mahalanobis products), `empirical_covariance` at 13.5% and check_array at 4.9%. The EllipticEnvelope profiles have the same shape: its fit is `MinCovDet.fit` plus `offset_` (`sklearn/covariance/_robust_covariance.py:fit` is 99.9% of the EllipticEnvelope call at 8x1024). The four 118x1024 profiles of this group (EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet) are regenerated by `audit/profiles/micro/prof_basic.py run <dir>`, and `prof_basic.py index` writes `audit/profiles/micro/prof_basic__index.json` (frame shares, opaque_share by the profile_units definition, and the shares of a reproduction run into scratch, which agree within a few points: e.g. EmpiricalCovariance eigh 58.7% kept vs 62.7% reproduced, MinCovDet eigh 57.4% vs 60.2%, GraphicalLassoCV graphical_lasso_path 86.4% vs 86.6%).

**Avenues.**
- `cov_prec_robust-1` (all four units, algorithmic, ~2x, effort M): use a Cholesky factor and a triangular solve for the Mahalanobis distances and log det inside the C-steps, instead of pinvh plus a dense product. The Amdahl bound from the 118x1024 eigh share is 2.3x. Expected risk: unverified-within-tolerance. The support is chosen by `np.argpartition` on the distances, so a near-tie can select a different subset (semantic in rare cases). It needs a pyspi-side copy of fast_mcd. Not measured. On the two EllipticEnvelope units the records list `cov_prec_robust-3` in `subsumed_by`: served from the cached MinCovDet fit, EllipticEnvelope runs no FAST-MCD fit of its own, so build_csv credits this avenue no saving there and counts it on the MinCovDet units (README "Optimisation columns", ranking rule 4).
- `cov_prec_robust-2` (all four units, parallel, 3-7x with 8 workers, effort L): run the independent trials in parallel, with the random permutations pre-drawn in the serial loop order. Each `_c_step` with `initial_estimates=None` consumes exactly one `random_state.permutation` at entry. `fast_mcd` draws `samples_shuffle` and the merge selection outside the trial loops. Pre-drawing therefore keeps the global RNG stream (expected unverified-bit-identical). Per-worker generators would make it rng-stream, and workers left at joblib's default BLAS threads (cpu_count()//n_jobs) could round the pinvh/eigh differently, so the workers need BLAS at 1 thread (`xc-blas-threads`). This is wall-clock parallelism inside one window: it cuts per-window latency, not core-hours, and competes for the same cores with window-level parallelism (`xc-window-parallel`), which already parallelises the 600 windows. The record lists only `xc-blas-threads` in depends_on: `xc-window-parallel` is an alternative use of the same cores, not a prerequisite, and inside one of the ~8 memory-bounded window workers (about 4 physical cores) this pool adds at most about 4x. Not measured.
- `cov_prec_robust-3` (EllipticEnvelope units, cache-sharing, effort S): serve EllipticEnvelope from the cached MinCovDet fit. The check stage found the two fits bit-identical for the same RNG state (p1.py), and pyspi never reads `offset_`. The EllipticEnvelope increment becomes a cache hit, written as '>100x' for those units. The robust family halves from 6.87 s to about 3.43 s per window at 118x1024. Risk: unverified-rng-stream. EllipticEnvelope would take MinCovDet's draw instead of a later-stream draw, so its identifiers become copies of the MinCovDet ones; today they differ only by RNG noise. On the EEG selection that makes the retained prec_EllipticEnvelope and prec-sq_EllipticEnvelope exact duplicates of prec_MinCovDet and prec-sq_MinCovDet, so the avenue amounts to aliasing two of the 185 retained metrics. Not measured. On basic.prec.EllipticEnvelope the basis is its own profile (`basic.prec.EllipticEnvelope__8x1024.pstats`, `_robust_covariance.py:fit` 99.87%) and the fresh 118x1024 wall (3.48 s, representative prec_EllipticEnvelope alone) against the two cache reads it would keep (0.184 ms per window in the family-warm run), about 18,900x. The basic.cov.EllipticEnvelope row credits the same fit for a full-config run (configured identifiers, outside tier 0); the two rows describe different run scopes and are not to be summed.

**Seed check.** The seed said "MinCovDet/EllipticEnvelope: parallelism only (rng-stream)". It is refuted on two counts. First, the profile shows an algorithmic avenue (eigh 57.4% at 118 channels). Second, trial-level parallelism can keep the RNG stream when the permutations are pre-drawn. The seed also asked whether EllipticEnvelope repeats MinCovDet's fit. Confirmed: it does, under a separate cache key. Sharing changes the RNG stream EllipticEnvelope draws from, as described above. Bypassing sklearn validation is refuted for 118 channels: check_array is 26.9% at 8x1024 but 4.9% at 118x1024, below the ~10% threshold.

**Shared versus incremental cost and EEG scaling.** In family-warm mode at 118x1024 the two fits cost 3.44 s (EllipticEnvelope) and 3.43 s (MinCovDet). The six other identifiers are cache reads of about 0.1 ms each, for a family total of 6.87 s. The EEG selection keeps cov_MinCovDet, cov-sq_MinCovDet, prec_MinCovDet, prec-sq_MinCovDet, prec_EllipticEnvelope and prec-sq_EllipticEnvelope. The per-window EEG cost at 118 channels is therefore two fits plus four cache reads, about 6.9 s per window or about 4.1e3 s per 600-window patient. There are no band variants. The historical EEG figure for MinCovDet (2.3 s per window, channel count unknown) is of the same order as one fit here. With `cov_prec_robust-3` the EEG cost halves to one fit. The ranking follows this split: basic.prec.EllipticEnvelope ranks on its fresh projection (`optimisation.eeg_basis: fresh`, 3.48 s per window, 2,085 s per patient instead of the family-warm 0.11 s), because the Covariance configs of the EEG selection have no EllipticEnvelope and an EEG run fits it on prec_EllipticEnvelope, while basic.prec.MinCovDet keeps its family-warm figure (`eeg_basis: family-warm`), because the retained cov_MinCovDet fits first.
