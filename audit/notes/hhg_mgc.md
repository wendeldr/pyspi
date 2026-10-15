# Family hhg_mgc: define, trace and check

Units: `distance.hhg` (identifier `hhg`) and `distance.mgc` (identifier `mgc`). Both live in
`pyspi/statistics/distance.py` and delegate to hyppo 0.5.2. MGC goes on from hyppo into SciPy 1.17.1. Both
take their distance matrices from scikit-learn 1.9.1's `pairwise_distances(metric="euclidean")`. Neither unit is
in the EEG selection.

Probe scripts and downloaded reference sources are in the session scratchpad
(`.../scratchpad/hhg_mgc/probes/` and `.../scratchpad/hhg_mgc/dl/`). They include the SciPy v1.17.1 `_stats.pyx`, the
neurodata/r-mgc `R/MGCLocalCorr.R` and `R/MGCSampleStat.R` (master, fetched 2026-10-14), and the arXiv texts of
Heller et al. 2013 (1201.3522v3) and Shen et al. 2020 (1710.09768). The probes only confirm what the code does. They
are not the numeric checks. Throughout this note, the regression goldens are snapshots of current output and are not
a reference.
The sections before `## Checks` are the trace stage. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Supplement line numbers: in this checkout the headings sit at `supplement.txt:243` (§1.2.3) and `:272`
(§1.2.6). The briefing pack lists them 6 lines later (249 and 278). Both sections are on page 6.

## Shared preprocessing and dispatch

* `Calculator(detrend=False, normalise=True)` (`pyspi/calculator.py:46-53`, `:277`) builds `Data`, which z-scores
  each process along time with `ddof=1` and does not detrend (`pyspi/data.py:184-197`). Data are stored as
  (P, T, 1), so `data.to_numpy()[[i, j]]` gives two (T, 1) arrays (`distance.py:75`, `:109`).
* `hyppo.tools.compute_dist` (`hyppo/tools/common.py:436-503`) calls `sklearn.metrics.pairwise_distances`
  (`:499`). For float64 that computes `-2 X Y^T + XX + YY`, clips at 0 and zeroes the diagonal
  (`sklearn/metrics/pairwise.py:403-421`). For 1-D data the result is not exactly `|x_a - x_b|`: the probe found a
  maximum error of 3.5e-14 at T=64, with small distances carrying a large relative error. The matrix is also not
  bit-symmetric. This matters here because both statistics compare or rank distances.
* Both statistics are affine-invariant per variable, so z-scoring only changes the result through rounding.

## distance.hhg: Heller-Heller-Gorfine

**Documented.** Supplement §1.2.6, p.6, and GitBook `distance-similarity.md:184-199` say: "The
Heller-Heller-Gorfine (HHG) method [16] yields an RKHS-based statistic that uses the ranks of random variables to
obtain sample kernel matrices, rather than their distances (cf. HSIC in Sec. 1.2.5)". The listed keywords are
undirected, nonlinear, unsigned, bivariate, contemporaneous.

**Canonical.** Heller, Heller & Gorfine (2013), *Biometrika* 100(2):503-510, §2. For each ordered pair (i, j),
i ≠ j, the remaining N−2 points are cross-classified by I{d(x_i,x_k) ≤ d(x_i,x_j)} and
I{d(y_i,y_k) ≤ d(y_i,y_j)}. The pair's score is the Pearson chi-square
S(i,j) = (N−2)(A12·A21 − A11·A22)² / (A1·A2·A·1A·2), and the statistic is T = Σ_i Σ_{j≠i} S(i,j). The paper adds:
"For i and j with 0 in at least one of the margins, we set S(i,j) = 0" (arXiv v3 p.3).

**Implemented.** `HellerHellerGorfine(Directed, Unsigned)` (`distance.py:66-77`) calls `HHG().statistic(x, y)`
(`hyppo/independence/hhg.py:151-194`, with `auto=False`), which reaches the numba kernel `_pearson_stat`
(`hhg.py:274-297`). That kernel:

* sets `a = distx[i,:] <= distx[i,j]` (and `b` likewise);
* computes `t11 = sum(a*b) - 2`, removing k=i and k=j, which always satisfy both "≤" conditions;
* computes the other three cells directly;
* sets `S = (n-2)(t12 t21 - t11 t22)^2 / denom` if `denom > 0`, and 0 otherwise.

The returned value is the sum of the off-diagonal S. No p-value is computed, and the fast Hoeffding variant is not
used.

**Deviation analysis.**

* *Kernel vs canonical: matches.* The ≤ tie rule, the exclusion of i and j, the zero-margin rule and the double sum
  are all as in the paper. A from-scratch probe that uses exact distances reproduces pyspi to about 1e-15 relative
  on continuous data, e.g. 2928.4725589238396 vs 2928.4725589238515.
* *Distance inputs (candidate).* The ranks are taken over sklearn dot-product distances, so rounding resolves
  ties and near-ties. At T=256, exact distances and sklearn distances gave identical T on 17 of 19 datasets. The
  exceptions were logistic_maps, at 1.4% relative (its process 1 takes only 8 distinct values), and
  shared_envelope, at 0.43%. At T=64, shared_envelope pair (0,1) differed by exactly 62 = n−2, which is one S(i,j)
  switching on or off. At T=256, the sklearn distances introduced extra exact ties: 180/240 row ties against
  7/135 for exact distances. The root cause is the one in REVIEW.md M2, where it was reported only as last-bit
  asymmetry of pdist. The consequence here, a percent-level change in a rank statistic, is new.
* *Direction (candidate).* S is invariant under swapping x and y: A12 swaps with A21, A1· with A·1, and all the
  counts are integers. The statistic is therefore exactly symmetric, and the probe gave `max|A − A.T| = 0.0` on 5
  datasets. Yet the class is `Directed` and labelled `"directed"`, while `config.yaml:180-186` and the supplement
  say "undirected". pyspi computes both orders: about 6 s per call at T=1024 and an estimated 6 min at T=4096, which
  is O(T³).
* *Documentation (candidate).* The method is a sum of chi-square statistics over rank-of-distance tables. It is
  not an RKHS or kernel statistic, and it does not rank the variables themselves.
* *Scale.* pyspi returns the raw T. Under the null, T is about the number of non-zero-margin pairs, roughly
  N(N−1): on iid data at T=64 the probe gave 2714–4212 against N(N−1) = 4032. For identical 1-D series T reaches
  N(N−2)(N−3), and the probe gave exactly 242048 = 64·62·61 on shared_envelope (0,2). Values cannot be compared
  across series lengths.
* *Hypothesis (kernel arithmetic probed, end-to-end not run).* numba types the t-counts, `denom` and `n` in
  `_pearson_stat` as int64. The numerator `(n-2)*(t12*t21 - t11*t22)**2` is bounded by
  (n−2)(⌊N/2⌋⌈N/2⌉)², with N = n−2, and this bound first exceeds 2^63−1 at n = 10812 (about 1.08e4). At n = 10502
  it is 7.98e18 < 9.22e18. A numba replica of the numerator at n = 10812, with t11 = t22 = 5405 and t12 = t21 = 0,
  wrapped to −9220849508775795366. Such tables occur on identical or near-identical series, so strongly dependent
  pairs at T ≥ 10812 may give wrapped, negative S(i,j). The full O(n³) HHG at that length was not run. This is not
  reachable at T ≤ 4096.

**Proposed checks.** The kernel check uses a from-scratch numpy HHG on exact distances (independent-impl) over
iid_gaussian, bivariate_gaussian, nonlinear_dependence, anm, contaminated and var2_bivariate, with a 1e-12 relative
tolerance. A second kernel check uses the same reference on logistic_maps and shared_envelope; pyspi is expected to
disagree there, which would confirm the rounding candidate. Hyppo with exact precomputed distances serves only as a
same-lineage attribution aid. The other checks are:

* an analytic check: identical or affine copies give N(N−2)(N−3) exactly;
* a direction check on var2_bivariate, logistic_maps, delayed_noise and gauss_ar1_pair: A[i,j] == A[j,i] bit for
  bit, and A[i,j] equals the reference for (x_i, x_j);
* a permutation-null calibration on iid_gaussian;
* scaled_offset invariance;
* a qualitative detrend check on trending.

## distance.mgc: multiscale graph correlation

**Documented.** Supplement §1.2.3, p.6, describes MGC as "a generalization of distance correlation [13] ... the
algorithm truncates the Euclidean distance matrices ... at an optimal threshold. MGC also includes a smoothing
function ... Only this unbiased estimator is included here (as in hyppo)". The supplement cites [14], Shen, Priebe &
Vogelstein, JASA 2020. The GitBook (`distance-similarity.md:251-266`) has the same text, but its reference link
(`:257`) points to "Discovering similar multidimensional trajectories", which is an LCSS paper, not MGC.

**Canonical.** Shen, Priebe & Vogelstein (2020), JASA 115(529):280-291.

* §3.1, eq. (6): column-centred distance matrices, A_ij = D_ij − (1/(n−1)) Σ_s D_sj for i ≠ j, with a zero
  diagonal. Ranks are taken within each column, using minimal (consecutive) rank under ties.
* dCov^{k,l} = Ê(A^k∘B^l′) − Ê(A^k)Ê(B^l′), with Ê = 1/(n(n−1)) Σ_{i≠j}. dCorr is dCov over the square root of
  the product of the local variances, and is set to 0 if a variance is below ε.
* §4.1: R is the largest connected component of the cells with dCorr > max(τ_n, dCorr^{n,n}). Sample MGC is the
  maximum over R. It falls back to dCorr^{n,n} if |R| < 2n or if the maximum does not exceed dCorr^{n,n}.
* Corollary 3: τ_n = 2F⁻¹_{Beta((v−1)/2,(v−1)/2)}(1−0.02/n) − 1, with v = n(n−3)/2.

**Implemented.** `MultiscaleGraphCorrelation(Undirected, Unsigned)` (`distance.py:100-111`) calls `MGC().statistic`
(`hyppo/independence/mgc.py:151-166`). That calls `scipy.stats.multiscale_graphcorr(distx, disty,
compute_distance=None, reps=0)` with warnings suppressed. SciPy then runs:

* `_mgc_stat` (`_mgc.py:383-429`);
* `_local_correlations` in Cython (`_stats.pyx:451-491` at v1.17.1), which does dense column ranks (`:323-332`)
  and column centring (`:337-357`), then computes local covariances with the subtractive term divided by **n²**
  (`:444`);
* `_threshold_mgc_map` (`_mgc.py:432-474`), with **samp_size = n−1**, 4-connected labelling, and the first label
  taken when components tie in size;
* `_smooth_mgc_map` (`_mgc.py:477-525`), whose region-size rule is
  **|R| ≥ ceil(0.02·max(m,n))·min(m,n)**.

T < 5 raises a ValueError (`_mgc.py:331-333`). In pyspi, `Undirected.multivariate` computes every ordered pair and
then copies the upper triangle over the lower (`base.py:132-138`).

A from-scratch prototype (`probes/mgc_scratch.py`, numpy plus the `scipy.ndimage.label` and `scipy.stats.beta`
primitives) reproduced pyspi to 10 decimals on exact distances. That confirms the trace above.

**Deviation analysis.**

* *n² vs n(n−1) (candidate; a paper-vs-reference-code discrepancy).* The authors' R package uses n² as well
  (`MGCLocalCorr.R:167`). The term has no effect at the global scale, where column sums are zero. Elsewhere the
  effect is O(1/n): 0.58183 vs 0.58028 on nonlinear_dependence at T=64, and 0.534318 vs 0.534201 at T=1024.
* *Threshold with n−1 instead of n.* This matches r-mgc (`MGCSampleStat.R:99, 122-126`). No probed value changed.
* *Region-size rule (candidate).* SciPy's rule matches neither the paper (2n) nor r-mgc (`2*min(m,n)`, at
  `MGCSampleStat.R:172`). For continuous data, where m = n = T, the requirement is:

  | T | required region size |
  |---|---|
  | ≤ 50 | n |
  | 51–100 | 2n |
  | 256 | 6n |
  | 1024 | 21n |
  | 4096 | 82n |

  So the rule is stricter than the paper for T > 100, which pushes toward the dCorr fallback. For tied data
  (m < n) it is laxer: on logistic_maps at T=64 (a 64×8 map), pyspi gives 0.3707 where the 2n rule gives 0.3157. In
  the continuous probes the region rule changed nothing, because the regions were large.
* *Distance rounding (candidate).* Rounding changes neighbour ranks. On shared_envelope at T=256, pair (0,1), pyspi
  gives 0.708 while exact distances give 0.769; the optimal scale is (2,2), i.e. nearest neighbours only. On
  logistic_maps at T=256 the values are 0.3176 vs 0.3221. Other datasets differ by ≤ 7e-14.
  The root cause is REVIEW.md M2 (sklearn dot-product distances); the consequence here, a shift of up to 6e-2, is
  new. Both records therefore cite `REVIEW M2 (root cause only; consequence new)` in `already_documented`; the mgc record
  also marks the sign-label class as a new instance of REVIEW M7/M8 (`partly`, the same treatment as dcorr_hsic).
* *Sign (candidate).* MGC is labelled unsigned, yet when no region qualifies it returns the global-scale dCorr,
  which is negative for most independent pairs. The probe minimum was −0.0264 on iid data at T=64.
* *"Unbiased" wording.* The global-scale value is the column-centred dCorr, not hyppo's U-centred unbiased dcorr.
  On iid data at T=256 the probe gave −0.0059473 vs −0.0059357.
* *Minor differences from the paper:* variances are cut off at ≤ 0 rather than ε, and equal-size components are
  resolved by taking the first label rather than their union. Dense ranks agree with the paper's "minimal rank"
  indexing.
* *Symmetry.* bivariate(i,j) and bivariate(j,i) differ by ≤ 1.1e-14, and the published matrix is exactly
  symmetric. Half of the computed pairs are discarded, which matters for optimisation but not for correctness.

**Proposed checks.**

* A from-scratch MGC (independent-impl, exact distances) with the implementation's constants, on continuous data,
  to 1e-10.
* The same reference with the paper's constants and with r-mgc's constants, to quantify each deviation, including
  the region rule on logistic_maps.
* An exact-distance attribution check on logistic_maps and shared_envelope; pyspi is expected to disagree there.
* An analytic global-scale identity: on fallback pairs, pyspi equals Σ A_ab B_ba / √(Σ A_ab A_ba · Σ B_ab B_ba).
  dcor's U-centred value is reported alongside as a related quantity.
* A symmetry and direction check.
* A permutation null with a count of negative values.
* Anchors: an identical copy must give 1, and nonlinear_dependence must give clearly positive values.
* scaled_offset invariance.
* A qualitative detrend check on trending.

## Open questions

1. Should `HellerHellerGorfine` derive from `Undirected`? The output would not change, since the matrix is exactly
   symmetric, and the O(T³) cost would halve. Is the "directed" label relied on anywhere, for example in label-based
   SPI grouping?
2. Should pyspi pass exact 1-D distances to hyppo, for example `np.abs(x - x.T)` with `compute_distance=None`, to
   remove the rounding sensitivity of the rank-based statistics? That would change historical values on tied or
   near-tied data, by up to 0.06 for mgc and 1.4% for hhg in the probes.
3. Which canonical target should the check stage score MGC against: the paper's text (n(n−1), τ_n with n, 2n) or
   the authors' r-mgc reference code (n², n−1, 2·min(m,n))? SciPy matches r-mgc on two of the three points and
   neither on the region rule. The r-mgc sources were taken from GitHub master, not from a tagged CRAN release, so
   the exact CRAN version still needs confirming. Should the region rule be raised upstream with SciPy?
4. The R packages HHG and mgc (CRAN) failed to install in the audit environment (`audit/env/r-packages.txt`).
   Should the external-package cross-check be retried?
5. Is the int64 overflow hypothesis for HHG at T ≥ 10812 (about 1.08e4) worth confirming end to end? The kernel
   arithmetic wraps in a numba replica, but a full run is out of scope for T ≤ 4096.
6. `hyppo/independence/hhg.py:3` imports `random` from `scipy.sparse.construct`. That import is unused and
   deprecated; under SciPy 1.17.1 and 1.18.1 it emits "will be removed in SciPy 2.0.0", which would make
   `import pyspi.statistics.distance` fail on SciPy 2.0. This is a dependency issue rather than a calculation issue.
7. HHG's raw T grows like N²–N³. Should pyspi document that hhg values are length-dependent, or offer a normalised
   form such as T/(N(N−1))?

## Checks

**Setup.** `audit/checks/manifest/hhg_mgc.json` holds 145 runs: 48 for `distance.hhg` and 97 for `distance.mgc`. The
reference is `audit/checks/refs/hhg_mgc.py` (comparison venv), and the probes are in `audit/checks/probes/hhg_mgc_probes.py`
(project venv). `out/results/hhg_mgc.json` records **124 pass, 21 fail, 0 without reference**. 19 of the fails are deliberate
deviation measurements (listed below), not script errors. The other 2 are permutation-null uniformity fails
(`distance.hhg|iid_gaussian|64|kernel|perm`, `distance.mgc|iid_gaussian|256|kernel|perm`), discussed under each unit. An interrupted earlier attempt left drafts of the reference and probe
files. They were reviewed against the SciPy v1.17.1 `_stats.pyx` and `_mgc.py` sources, kept, and every run was recomputed.
pyspi time was 79 s for HHG and 123 s for MGC in the first run, and 63.9 s and 109.2 s in the forced re-run after the `perm` runs were added. The longest single run was `distance.hhg|bivariate_gaussian|1024|kernel|def-p23`:
28.9 s for two ordered calls in the first run (14.4 s per call) and 17.7 s in the re-run, as machine load differed. That is why HHG is run at T=1024 only on one pair.

**Reference independence.** Neither statistic's reference imports hyppo, sklearn or `scipy.stats.multiscale_graphcorr`.

* `hhg_from_dist` follows Heller, Heller & Gorfine (2013) §2 (arXiv v3 l.115-135: tables over the N−2 remaining points, factor
  N−2, S = 0 for a zero margin), with counts from a 2-D cumulative histogram. `--selftest` cross-validates it against the loop
  implementation `_kernels_distance.hhg`, including on tied data.
* `mgc_from_dist` is written in numpy from Shen, Priebe & Vogelstein (2020): eq. 6 centring, dense column ranks, rank-binned
  cumulative local covariances, the Corollary 3 Beta threshold and the largest 4-connected component. It uses only the
  `scipy.stats.beta.ppf` and `scipy.ndimage.label` primitives.
* MGC's three implementation constants (n², n−1 and the region rule) can only be read from the SciPy/r-mgc code lineage. The
  canonical definition is therefore literature-only (Shen et al. 2020; Vogelstein et al. 2019), and the paper's constants are
  measured as deviations.
* The `hyppo-exact` runs feed the delegate itself exact distance matrices. They are a same-lineage attribution aid on the pyspi
  side, not a reference.

**Reference correction made during this stage.** The first draft built the qualitative `order` and `trendqual` null bands as
mean ± z·SD. HHG's null is right-skewed (skewness 0.98 over 2000 Monte Carlo draws at n=256), so that band under-covers: it
flagged the independent anm pair (0,3) at 103654 against an upper limit of 102044. `mc_quantile_band` now uses the empirical quantiles q(0.025/6) and q(1−0.025/6) of 2000 draws (HHG
[38512, 118786], MGC [−0.0072, 0.0373] at T=256). The anm (0,3) pair has a null upper tail of 0.0165. No tolerance was widened
on any computed value.

**Correction made after review: the Monte Carlo band is a Gaussian-marginal null, not an exact null.** The first version of
this note said that HHG, being a function of distance ranks, has a null distribution that depends on n alone. That is true of
its mean, not its spread: in 1-D the ranks of |x_i − x_k| depend on the spacings, so the marginal law matters. The scratchpad
probe `hhg_mgc_check/p_null_marginals.py` (n=64, 3000 independent pairs per law) gives null means 3813 / 3820 / 3840 / 3857
against the analytic 3842.0, but SDs 923 (Gaussian), 1022 (exponential), 1055 (uniform) and 996 (Cauchy), with KS Gaussian
vs uniform p = 0.007. The `order` and `trendqual` bands (and MGC's, whose null also depends on the marginals) are therefore
approximate for the non-Gaussian anm and nonlinear_dependence pairs. No verdict depends on the approximation: the
non-Gaussian dependent pairs lie 13.6× to 45.5× above the HHG band's upper limit (the closest dependent pair, bivariate_gaussian
ρ=0.3 at 1.27×, has Gaussian marginals, for which the band is the right null), and the closest independent pair (anm (0,3), 103654 against 118786)
would stay inside a band widened by the 14% SD excess of the uniform law. The `perm` runs below use the conditional per-pair
permutation null, which does not need the approximation. The `mc_quantile_band` docstring has been corrected.

### distance.hhg: matches, score 9 (canonical reproduced bit for bit, supplement misdescribes it; tie rounding within estimator variance by analogy)

* **Kernel, continuous data.** All 14 `def` runs give **max_abs 0.0**: pyspi equals the from-scratch Heller et al. statistic
  bit for bit. The runs are iid_gaussian, bivariate_gaussian, nonlinear_dependence, anm, contaminated and var2_bivariate at
  T64 and T256 on the kernel path, plus `bivariate_gaussian|256|pipeline|def` and `bivariate_gaussian|1024|kernel|def-p23`.
  `scaled_offset|256|pipeline|scaleinv`, `trending|256|pipeline|trend` and `trending|256|pipeline-detrend|detrend` also give
  0.0.
* **Analytic identical ranking.** `iid_gaussian|64|kernel|copies` and `|256|kernel|copies` give N(N−2)(N−3) = **242048** and
  **16451072** exactly on the identical-copy and affine-copy (2x0+1) pairs. `shared_envelope|64|pipeline|dup` gives 242048.
* **Null level.** The analytic null mean is E[T] = n(n−2)(n−3)/(n−1). It is derived in the refs docstring from the
  hypergeometric table, E[S | margins > 0] = N/(N−1), together with P(no zero margin) = ((n−3)/(n−1))². That gives 3842.0 at
  n=64 and 64514.0 at n=256. The 2000-draw Monte Carlo gives 64814 ± 328 (SE).
  `iid_gaussian|64|pipeline|null` has a pair mean of 3290.4 (max_z 0.96), and `|256|pipeline|null` has 53204.2 (max_z 1.44).
* **Per-pair permutation null (trace proposed check [4], added after review).** The first version of this section replaced
  the proposed 200-permutation null with the analytic E[T] and the Monte Carlo band, and said so only outside the record. The
  proposed check has now been run as proposed (`ref_perm`: 200 permutations of y per pair, from-scratch statistic on exact
  distances, kernel rows); the analytic and Monte Carlo checks above are additional to it.
  * `iid_gaussian|64|kernel|perm` **fails** the pre-set uniformity criterion: KS p 0.0109 for the 10 pair percentiles
    (0.085 to 0.695, mean 0.309). `iid_gaussian|256|kernel|perm` passes: KS p 0.124 (percentiles 0.0 to 0.74, mean 0.2805).
    Neither run flags any pair as dependent: the smallest upper-tail permutation p is 0.308 and 0.264, against 0.005.
  * The deviation is in the lower tail: on both draws the pyspi values sit below their own permutation nulls, consistent
    with the `|null` pair means of 3290.4 and 53204.2 against E[T]. It is not attributable to the computation. pyspi equals
    the from-scratch statistic bit for bit on these rows (`iid_gaussian|{64,256}|kernel|def`, max_abs 0.0). For exchangeable
    rows the permutation percentile is uniform by construction. The calibration probe
    `hhg_mgc_check/p_perm_ks_calibration.py` (400 fresh 5 × 64 iid draws) shows that the criterion fails on 5.75% of null draws
    and reaches KS p ≤ 0.012 on 0.75%; the mean pair percentile is ≤ 0.31 on 2.0% of draws. The two iid_gaussian draws are
    therefore each in the low 1-2% tail. Taken together that is unusual, but the run verdict is recorded as a fail and is
    not reinterpreted. The T=256 draw (`iid_gaussian_T256.npz`, dataset seed 1000256 in `audit/checks/data/MANIFEST.json`) is the one on which dcorr_hsic's four
    `distance.*|iid_gaussian|256|pipeline|null` runs fail: its mean dCor² sits at the 0.15% lower tail of its own permutation
    null (notes/dcorr_hsic.md, "The five failures"). The T=64 fail is on the T64 draw, which passes in dcorr_hsic
    (|z| ≤ 1.73), so it has no such explanation; further draws through pyspi would settle whether anything beyond chance is
    involved there.
  * The pooled permutation-null mean equals the analytic E[T] within Monte Carlo error: z = −0.60 and 1.08 (iid T64,
    T256), −1.12 and 1.21 (gauss_ar1_pair T64, T256). E[T] is exact conditionally on the data as well, because the x margins
    are fixed with exactly two zero-margin j per i and the y margins are uniformly random under the permutation.
  * `gauss_ar1_pair|{64,256}|kernel|perm` pass: pyspi 7945.2 and 315100.7 lie above all 200 permutations (percentile 1.0,
    p_upper 0.005; permutation means 3768.3 and 65778.0). The pair is contemporaneously correlated (stationary correlation
    0.52 from `Sigma_X`), and the permutation ignores the AR(1) autocorrelation, so this is a detection check only.
  * The unit verdict stays `matches`, score 9. The failing run tests the iid draw's percentile, not whether pyspi computes the
    canonical statistic, which the `def` runs settle bit for bit.
* **Ordering (T256).** Dependent pairs lie far above the band: nonlinear_dependence parabola 3286900.5 and circle 1620054.8;
  anm 2510927.2 and 5409301.4; bivariate_gaussian ρ=0.8 1940899.5 > ρ=0.3 150648.4. Every independent pair lies inside the band.
* **Trend.** For the independent trending pairs (1,2) and (1,3), the values are 2086441.8 and 5532123.2 without detrend, and
  61082.4 and 49758.8 after it.
* **Tied data (rounding candidate confirmed).** Against exact distances the `exact` runs fail:
  * logistic_maps T256 kernel: max_rel **4.88e-2** (858472.24 vs 818546.71);
  * logistic_maps T256 pipeline: 1.40e-2;
  * logistic_maps T64 kernel: 9.9e-3;
  * shared_envelope T256 kernel: 3.73e-3, abs 8128 = 32·254;
  * shared_envelope T256 pipeline: 4.32e-3;
  * shared_envelope T64 kernel: abs 62 = n−2, on the bit-identical (0,2) pair, which gives 242110 instead of the analytic 242048.

  The same pairs match the sklearn-mirrored-distance reference (`skdist`, 4/4, max_abs 0.0). hyppo fed exact distances matches
  the exact-distance reference (`hyppo-exact`, 4/4, max_abs 0.0). The distance arithmetic is therefore the entire cause. The
  root cause is REVIEW M2, which is already documented. Its consequence for a rank statistic is new. The trace quoted 1.4% for
  logistic_maps, which is the pipeline value; on raw rows the effect is 4.9% (added to `deviation_from_canonical`). This is a
  floating-point artefact of the dot-product distance formula, not a convention.

  *Scope on digitised recordings (wf4b-2 scratch probe `wf4b2_high-optimiser-other/p_hhg.py`).* Every digitised recording is
  grid-valued, so ties are not a niche case. On an EEG window (001_Baseline.EDF, channels EEG G1-G3, mid-recording; ADC step
  2.66e-7, 341 to 446 distinct values per 1024 samples over six channels) pyspi differs from the canonical statistic computed with
  exact ties on the integer codes by 2.24e-3 / 1.53e-3 / 3.69e-3 relative at T=256 (pairs 01/02/12) and 1.62e-3 / 1.18e-3 /
  1.53e-3 at T=512. The audit's exact-distance reference applied to pyspi's z-scored floats does not recover the canonical ties:
  it differs from pyspi by 4.0e-6 to 4.8e-4 and from the integer-code value by about the same 1e-3 as pyspi, because z-scoring
  breaks some exact ties by rounding. The check design therefore cannot separate canonical from rounded ties on grid-valued data;
  a manifest run on integer codes would be needed (not run here). Continuous data agree to fp rounding in the same probe
  (bivariate_gaussian T64 1.7e-16, nonlinear_dependence T256 8.5e-16 relative; these nonzero figures come from the probe's own summation, and bit-exactness on continuous data rests on the def runs, max_abs 0.0). distance.hhg is not in the retained 185-metric EEG set.
* **Symmetry and direction.** max|A − A.T| = 0.0 on every HHG run. On the direction datasets (`|256|pipeline|dir`), A[i,j] equals
  A[j,i] bit for bit, and equals the from-scratch value for both argument orders:
  * var2_bivariate (x0→x1): 123220.1;
  * logistic_maps_chaotic (x→y): 213720.7;
  * gauss_ar1_pair: 315100.7 (null upper tail 0.0; HHG detects the contemporaneous coupling);
  * delayed_noise: at the null level (e.g. (0,1) 74162.8, tail 0.22), because the copies are lagged white noise.

  HHG carries no direction information despite the `Directed` class and its "directed" label.
* **Candidates.**
  * Confirmed: Directed-but-symmetric; the documentation misdescribes the method as an RKHS/kernel statistic; sklearn rounding
    on tied data; raw T is length-dependent.
  * Inconclusive: the int64 overflow hypothesis, which was not run because it needs n ≥ 10812 at O(n³).
  * `new_finding: true`. The label contradiction and the documentation text are not in KNOWN_ISSUES (REVIEW M9 is a general
    label discussion that does not name hhg). The rounding consequence is new, on a documented root cause.
* **Score mapping.** On continuous data the numeric half of anchor 10 is met (bit for bit against an independent reference) and
  its "matches the pyspi supplement description" half fails (RKHS/ranks-of-variables wording, "undirected" against a Directed
  class). That is not one of anchor 9's trivial value conventions (squared vs not, ddof, log base, sign); 9 is the score
  for "canonical computation reproduced, supplement wrong" (README "Score columns: original, rubric reading, adjusted"; the rubric reading for hhg is anchor 9 through the estimator-variance clause), applied the same way to misc.gpfit.DotProduct and
  misc.coint.johansen_order-0_*. The tie-rounding deviation is placed under the anchor-9 clause "within statistical tolerance
  consistent with estimator variance", by analogy: it runs from 2.6e-4 (shared_envelope T64 kernel) to 4.9e-2 relative (raw
  logistic_maps T256), and 1.2e-3 to 3.7e-3 on the EEG window, while the null sampling SD at n=256 is 14225 on a mean of 64514,
  about 22% (`iid_gaussian|256|pipeline|null`). The sampling SD on dependent data was not measured, so this comparison is an
  analogy, not a measured tolerance. In absolute terms the analogy fits only part of the runs: the largest deviation (`logistic_maps|256|kernel|exact`, max_abs 39926 on the raw logistic maps) is 2.8 times the only SD measured (14225, the iid null at n=256), while shared_envelope T256 (max_abs 8128 kernel, 9188 pipeline, about 0.6 SD), logistic_maps T256 pipeline (11420, 0.8 SD) and the EEG T256 gaps (about 1.1e3 to 4.2e3, at most 0.3 SD) lie within it; the relative comparison above sets a deviation against a relative SD and overstates the fit for the raw logistic maps.

### distance.mgc: variant, score 6 (anchor 6: deviates from both the supplement and the canonical text in an interpretable way)

The score rests on the deviation anchor alone. The same-lineage cap does not apply: `mgc_from_dist` does not import or port
`scipy.stats.multiscale_graphcorr` or hyppo, so `check_method` is independent-impl. It takes three constants (n², n−1 and the
region rule) from SciPy's code; those constants are the thing being described, and the paper's and r-mgc's constants are
measured against them as deviations. (An earlier draft of this heading also invoked the cap, which conflicted with the
independent-impl label.)

* **SciPy-constant reference.** All 19 `def` runs pass with max_abs ≤ **5.8e-14**: six datasets at T64, T256 and T1024 on the
  kernel path, plus `bivariate_gaussian|256|pipeline|def`. `dir`, `ordered`, `trend`, `detrend` and `scaleinv` give ≤ 8.6e-14.
  pyspi computes exactly what SciPy's code states.
* **Global-scale identity.** This was the analytic handle the task named. The optimal-scale probe replays hyppo's path
  (`compute_dist` then `multiscale_graphcorr`, reps=0) and exports `mgc_dict['opt_scale']`, because hyppo's `MGC.statistic`
  keeps only `stat`, at `mgc.py:151-166`.
  * Every bivariate_gaussian pair is at the global scale (n, n): 12/12 at T64, T256 and T1024. So are all 20/20 iid_gaussian
    pairs at T256 (`|optscale`; scale and global map value match the reference to ≤ 2.9e-14).
  * On those pairs pyspi equals the column-centred dCorr^{n,n} = Σ A_ab B_ba / √(Σ A_ab A_ba · Σ B_ab B_ba) to max_abs
    2.6e-15 / 1.0e-14 / 3.8e-14 (`|global`, T64/T256/T1024).
  * It does **not** equal the unbiased (U-centred) dCor² that hyppo's Dcorr returns. The `|udcorr` runs show differences of
    1.79e-3 / 1.64e-4 / 1.23e-5 on bivariate_gaussian and 1.39e-3 / 1.55e-4 / 3.24e-5 on iid_gaussian. That is near-equality of
    order 1/n, passing the 1/n related-quantity tolerance, not identity.
  * The task text's premise, "MGC equals the unbiased dcorr at the global scale", therefore holds only approximately. The
    exact identity is with Shen's column-centred dCorr.
* **Local scales.** nonlinear_dependence T256 uses (42,89) for the parabola and (5,5) for the circle. narrowband_f0-0.1 uses
  (65,26) and (50,48). logistic_maps uses (249,6).
* **Population.** `bivariate_gaussian|{64,256,1024}|pipeline|pop` pass with max_z 0.86 / 0.64 / 0.72. The Monte Carlo means at
  T1024 are 0.0723 (ρ=0.3) and 0.5674 (ρ=0.8), against the population dCor² of 0.0724 and 0.5699 (Székely, Rizzo & Bakirov 2007,
  Thm 7).
* **Anchors.** The shared_envelope (0,2) bit-identical copy gives 1.0 exactly at T64 and T256, and the probe copies give 1
  within 2.2e-15. On `|order` (T256), dependent pairs range from 0.075 to 0.913 and lie above the band, the circle gives 0.477,
  and independent pairs lie inside the band. The trending independent pairs give 0.634 and 0.908 without detrend and −0.0054
  and −0.0042 after it. n=4 raises ValueError, and the n=5 value of −0.34544 matches the reference (`|smalln`).
* **Constants (deviation measurements).**
  * With the paper's constants (n(n−1), threshold with n, region 2n), pyspi differs by:
    * 1.5e-3 on nonlinear_dependence T64 and 5.2e-4 at T256 (these pass at 2e-3);
    * 2.4e-3 on narrowband T64;
    * 0.0564 on logistic_maps T64 and 0.0464 at T256;
    * 0.0610 on narrowband T256;
    * 0.0269 on anm T256.
  * With r-mgc's constants (n², n−1, region 2·min(m,n)), pyspi agrees to ≤ 6.3e-15 wherever the region rule does not bind. The
    rule does bind on two pairs:
    * anm T256 pair (1,3), an independent pair: 0.0319 under the r-mgc rule vs pyspi 0.0051, which is SciPy's stricter
      fallback;
    * narrowband T256 pair (1,2): 0.5364 vs 0.4751.
* **Trace correction.** The trace said the region rule "changed nothing" on continuous data, and that both the paper and r-mgc
  rules give about 0.316 on logistic_maps T64. Both statements are refuted.
  * At T=256, SciPy's rule (6n) rejects regions that 2n and 2·min(m,n) accept, which moves values by up to 0.061.
  * On logistic_maps (a 64×8 map), r-mgc's rule coincides with SciPy's: ceil(0.02·64)·8 = 16 = 2·8.

  `deviation_from_canonical` (c) has been amended accordingly.
* **Rounding.** `shared_envelope|256|pipeline|exact` gives max_abs **6.14e-2** (0.708 vs 0.769), and `|256|kernel|exact`
  4.99e-2. `logistic_maps|256|pipeline|exact` gives 4.47e-3. The `skdist` and `hyppo-exact` runs pass at ≤ 1.2e-14. This is
  confirmed on the already-documented root cause REVIEW M2.
* **Sign.** `iid_gaussian|{64,256,1024}|pipeline|null` pass (max_z 0.67 / 0.75 / 1.98), with 90% / 90% / 80% of the pyspi
  values negative and a minimum of −0.0264. The from-scratch null is negative in 61-68% of draws. This contradicts the
  "unsigned" label and the GitBook glossary's "distance correlation (defined in the range [0,1])" (glossary-of-terms.md:21).
* **Per-pair permutation null (trace proposed check [5], added after review).** The proposed 200-permutation null had been
  replaced by the Monte Carlo null without a not-done entry. It has now been run as proposed (`ref_perm`, iid_gaussian kernel,
  T64 and T256); the Monte Carlo `|null` runs are additional to it. T=1024 was not run in the manifest, because it would take
  about 40 min serially.
  * `iid_gaussian|64|kernel|perm` passes: KS p 0.288, mean percentile 0.411, smallest upper-tail p 0.114.
  * `iid_gaussian|256|kernel|perm` **fails** the pre-set uniformity criterion: KS p 0.0122, mean percentile 0.2475. Again
    the deviation is in the lower tail, and no pair is flagged as dependent (smallest upper-tail p 0.134). This is the same
    T=256 draw on which HHG's percentiles are also low (mean 0.2805). The calibration probe gives a 4.25% fail rate on null
    draws and KS p ≤ 0.012 on 1.0%. A fresh MGC draw at T=1024 (probe `hhg_mgc_check/p_perm_other_draws.py`) gives KS p
    0.180. The fail is recorded as such. It bears on the draw, not on the computation, which the `def` runs match to
    ≤ 5.8e-14. The draw (`iid_gaussian_T256.npz`, dataset seed 1000256) is the same one on which dcorr_hsic's four
    `distance.*|iid_gaussian|256|pipeline|null` runs fail (mean dCor² at the 0.15% lower tail of its own permutation null;
    notes/dcorr_hsic.md, "The five failures"), and lagged_dcorr_mgc traces its one null exceedance to it too. MGC on iid
    data falls back to the global-scale column-centred dCorr, so that shared draw is the common explanation.
  * Sign: 63.3% (T64) and 62.6% (T256) of the 2000 permutation-null values are negative, and 90% of the pyspi values are
    negative in both runs. This adds a conditional-null confirmation of the sign finding above.
* **Symmetry.** The published matrix gives 0.0. Unmirrored `Directed.multivariate` gives ≤ 1.07e-14 (`|ordered`). No direction
  information.
* **Candidates.** All five are confirmed: the unsigned label with negative values, the region rule, n² vs n(n−1) (a
  paper-vs-author-code discrepancy, not a pyspi defect), sklearn rounding, and the "unbiased" wording plus the GitBook LCSS
  link, which was re-read at `distance-similarity.md:257`.
* **Flags.** `new_finding: true`, because the SciPy-specific region rule is not in KNOWN_ISSUES and the sign-label
  contradiction is a new instance of the REVIEW M7/M8 class (which does not name mgc; `already_documented` is `partly`). `definition_under_review: true`, because the paper text, the authors' r-mgc code and SciPy disagree on the
  constants (both forms are recorded above).

**Remaining trace issue (mgc sign candidate citing the glossary).** This is resolved. `deviation_from_documented` (2) and the
first `new_finding_candidates` item of `distance.mgc` now cite GitBook `glossary-of-terms.md:21` (checked: the "Signed" keyword
row). The check-stage `candidate_resolutions` cite it as well.

## Open questions: answered

1. *Should HellerHellerGorfine derive from Undirected?* The evidence supports it. max|A − A.T| = 0.0 on all 44 HHG runs whose references record it (and on the four `|perm` pyspi matrices, read from
   `out/pyspi`), so
   mirroring would leave every output unchanged and halve an O(T³) cost (14.4 s per call at T=1024). The reduced-SPI-set filter
   (`pyspi/utils.py:186-187`) reads the config.yaml labels, which already say "undirected". The class label only feeds the
   grouping helper (`pyspi/base.py:106-107`) and the label printed at initialisation. No label-dependent behaviour would break.
2. *Should pyspi pass exact 1-D distances to hyppo?* The checks quantify the trade-off. With exact distances the delegate
   reproduces the exact-distance definition bit for bit (`hyppo-exact` runs). Historical values would change only on tied or
   near-tied data: HHG by up to 4.9% (raw logistic_maps T256) or 1.4% (z-scored), and MGC by up to 0.061 (shared_envelope T256).
   On all continuous datasets they would be unchanged (`def` runs ≤ 5.8e-14). Whether to change it is a policy decision for the
   maintainers.
3. *Which canonical target for MGC?* Both were measured. pyspi equals SciPy's own constants. It equals the r-mgc constants
   except where the region rule binds (0.027 and 0.061 at T=256). It differs from the paper text by ≤ 2.4e-3 on n² and τ_n, and
   by up to 0.061 on the region rule. The record scores against the paper text, with the author code as the documented variant
   (`definition_under_review: true`). The SciPy rule is the one that matches neither source; raising it upstream with SciPy is
   justified by `distance.mgc|anm|256|kernel|rmgc` and `|narrowband_f0-0.1|256|kernel|rmgc`. The CRAN version of r-mgc was not
   confirmed, because no network fetch was made at this stage.
4. *Retry the R packages HHG and mgc?* This was not attempted: the check stage may not install anything. The from-scratch HHG
   already matches pyspi bit for bit. An external HHG package (CRAN `HHG::hhg.test`, sum.chisq) would raise HHG's
   `check_method` to external-pkg but cannot change its score. For MGC, CRAN `mgc` is the same author lineage as SciPy/hyppo,
   so it would add a third constant set rather than an independent reference.
5. *Is the int64 overflow worth confirming?* Still open. Only a single hyppo `HHG().statistic` call on two identical series of
   length ≈ 1.1e4 would settle it. At O(n³), with 14.4 s at n=1024, that is roughly 4-5 h of compute, outside this stage's
   budget and its T ≤ 4096 scope. The candidate stays inconclusive.
6. *Deprecated `scipy.sparse.construct` import.* Confirmed: line 3 of `hyppo/independence/hhg.py` in this checkout is
   `from scipy.sparse.construct import random`, and the name is unused. This is a dependency-maintenance issue, not a
   calculation issue, and no check is needed.
7. *Document HHG's length dependence or offer a normalised form?* The null mean is exactly n(n−2)(n−3)/(n−1): 3842 at n=64 and
   64514 at n=256. The identical-ranking maximum is n(n−2)(n−3). A normalisation such as T/(n(n−2)(n−3)) would map identical
   rankings to 1 and the null to about 1/(n−1). That is a design choice for the maintainers. The checks only establish that raw
   values are not comparable across window lengths.
8. *Supplement line numbers.* Not resolvable from the evidence here. In this `supplement.txt` the headings sit at 243 (§1.2.3)
   and 272 (§1.2.6). The pack's 249 and 278 are consistently 6 lines later, which suggests a different extraction of the same
   PDF. The records cite the lines of this checkout.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls come from `audit/profiles/summary.json`. Shares are percentages of the profiled `compute_spi` frame. Nothing in pyspi was changed and no micro-benchmark was written for this family (the group's two went to dcorr). Neither unit is in the retained EEG selection, and both are listed in `profiles/infeasible_above_8x1024.txt`.

### Walls

| unit | 8x256 | 8x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|
| hhg (`distance.hhg`) | 5.50 s | 321.6 s | 79285 s | extrapolated-pairs-from-8 (pair count, no fixed cost) |
| mgc (`distance.mgc`) | 0.82 s | 13.7 s | 3377 s | extrapolated-pairs-from-8 |

### hhg

**Complexity: O(P^2 T^3).** `hyppo/independence/hhg.py:_pearson_stat` loops over all T(T-1) sample pairs and, for each one, counts over T samples using four boolean/int temporaries. pyspi runs it for every ordered pair, because `HellerHellerGorfine` is `Directed`. The measured T_exponent is 2.93 (8x256 to 8x1024).

**Profile.** The profiler pass timed out at 8x1024 and at 8x256. In this stage I profiled one warm call at 8x256 with cProfile and saved it to `audit/profiles/micro/distance.hhg__8x256.pstats` (wall 5.8 s, NUMBA/OMP threads 1):

- `hyppo/independence/hhg.py:HHG.statistic` has 98.8% self time. That is the numba `_pearson_stat` kernel, which cProfile cannot see.
- sklearn distances take 1.2%.
- opaque_share is 0.0016 by the profile_units heuristic. That figure is a lower bound: the numba time sits in a Python frame. No profile JSON was written for this pstats; the fix stage recomputed the heuristic independently (tottime of builtin `~` entries / total_tt = 0.00157 of 5.82 s), which agrees.

Opportunities:

- **hhg_mgc-1: one direction (~2x, unverified-bit-identical, effort S).** Swapping x and y swaps t12 and t21 and leaves t11, t22 and the integer denominator unchanged. S is therefore the same in both orders, and the check stage measured max|A - A.T| = 0.0 on every HHG run. Computing i < j and mirroring halves the work. The cross-cutting `xc-undirected-upper` does not cover this unit because the class is `Directed`, so it is listed here.
- **hhg_mgc-2: rank / dominance-count formulation of `_pearson_stat` (20-80x, unverified-bit-identical, effort L).**
  - Method: rank each row of both distance matrices once, then count t11 for every j with a sort plus a Fenwick tree. That is O(T log T) per row and O(T^2 log T) per pair.
  - Output: the counts stay integers and the S formula is unchanged, so the output should be unchanged if ties follow the `<=` rule.
  - Bound: Amdahl caps it at 83x from the 98.8% share. Complexity alone gives T / log2 T, about 100x at T=1024 and about 30x at T=256 before constants.
  - Scaling: the gain grows with T, because the baseline is O(T^3) and this is O(T^2 log T).

Combined with hhg_mgc-1, the 118x1024 window would drop from about 79000 s to roughly 500-2000 s. A plain fused numba loop without temporaries keeps O(T^3). It would give a constant factor (not measured) and leave the unit infeasible.

### mgc

**Complexity: O(P^2 T^2 log T)** per ordered pair (Undirected). It has two parts:

- A column-wise rank transform of each T x T distance matrix: `np.unique` per column, i.e. T sorts.
- scipy's local-correlation map over all (k, l) scales, which is O(T^2).

The measured T_exponent is 2.03.

**Profile (8x1024).**

- `scipy/stats/_mgc.py:_mgc_stat` is 93.4% cumulative and 50.5% self. The compiled `_stats._local_correlations` is hidden in that self time.
- `numpy unique`, called per column from inside it, takes 28.9% (114688 calls; argsort 9.8%).
- `numpy hstack` takes 7.7%.
- hyppo `compute_dist` -> sklearn takes 3.2%.

**hhg_mgc-3: per-channel cache of the distance matrix, its column ranks and centring (~1.5x, unverified-within-tolerance, effort L).** The per-channel part is about 40% of the call (unique 28.9% + hstack 7.7% + distances 3.2%), so Amdahl caps the gain at 1.66x. scipy has no entry point that accepts precomputed ranks, so this needs a port of `_local_correlations`; the port may change summation order. mgc also gains 2x from the cross-cutting `xc-undirected-upper`, since the published matrix uses only the upper triangle. MGC's ordered results differ by up to 1.1e-14, so the halving reproduces the stored matrix only because the lower triangle is overwritten today.

### Seed items refuted or narrowed

- **Distance-matrix cache for mgc (seed a): refuted for this unit.** Distances are only 3.2% of mgc, under the ~10% bar. The per-channel work worth caching is the rank transform (hhg_mgc-3).
- **Seed c: confirmed.** hhg is symmetric by construction: hhg_mgc-1 for the direction, hhg_mgc-2 for the rank formulation.

### Shared versus incremental cost

Family-warm increments at 8x1024 are hhg 324.0 s and mgc 13.4 s, equal to their fresh walls; nothing is shared. Both could read a Data-level per-channel distance cache (see dcorr_hsic-1), but distances are only 1-3% of either unit.
