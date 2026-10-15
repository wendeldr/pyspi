# Family `dcorr_hsic`: define, trace and check

Units: `distance.dcorr.biased-False` (`dcorr`), `distance.dcorr.biased-True` (`dcorr_biased`),
`distance.hsic.biased-False` (`hsic`), `distance.hsic.biased-True` (`hsic_biased`).
Records: `audit/records/<unit_id>.yaml`. Probe scripts (read-only, not part of the repo):
`/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/dcorr_hsic/probe1.py` to `probe5.py`, `popref.py`, `popref2.py`, and (review fixes) `dcorr_hsic_fix/closed_form.py`, `dcorr_hsic_fix/eq3_vs_song.py`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Library versions in the project venv: hyppo 0.5.2, scikit-learn 1.9.1, numba 0.68.0, numpy 2.5.4, scipy 1.17.1.
None of the statements below rely on the regression goldens. They are snapshots of current output, and nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.

## Shared pipeline (all four units)

1. `pyspi/data.py:184-195`: optional `scipy.signal.detrend` (default off), then `zscore(axis=1, ddof=1)` (default on). Data is stored as `(M, T, 1)`.
2. `pyspi/calculator.py:329` (HEAD a42ebc1, unchanged since a1f20c7) calls `multivariate`, and `:332` sets the diagonal to NaN. `pyspi/base.py:132-138` (`Undirected.multivariate`) calls
   `Directed.multivariate` (`base.py:84-94`), which evaluates `bivariate(i, j)` for **every ordered pair**. It then overwrites
   the lower triangle with the upper triangle. So `A[i,j] = A[j,i] = bivariate(x = proc min(i,j), y = proc max(i,j))`.
3. `pyspi/statistics/distance.py`: `x, y = data.to_numpy()[[i, j]]`, two `(T, 1)` arrays. Time order is ignored, and the T samples
   are treated as iid draws of a 1-D variable.
4. hyppo's `Dcorr.statistic` never takes the O(n log n) fast path. `is_fast` is set only inside `Dcorr.test`
   (`dcorr.py:231-237`), so the statistic uses dense n x n matrices.

## dcorr (unbiased) — `distance.dcorr.biased-False`

**Documented** (supplement §1.2.1, p.5): "The sample distance correlation is computed by summing over the entry-wise product of
Euclidean distance matrices. This statistic is biased, however an unbiased estimator can be obtained by first double-centering
the distance matrices [14]." The GitBook `distance-similarity.md` uses the same wording. The GitBook glossary
(`glossary-of-terms.md:21`, "Signed" row of the keyword table) says: "Other SPIs, such as distance correlation (defined in the range [0,1]) are unsigned."
This is the strongest documentary basis for the sign-label candidate below.

**Canonical:** the bias-corrected distance correlation R*_n (Székely & Rizzo 2013, JMVA 117:193-213; Székely & Rizzo 2014,
Ann. Statist. 42(6):2382-2412, U-centring). Each distance matrix is U-centred: row and column sums are divided by (n-2), the grand sum by (n-1)(n-2),
and the diagonal is set to 0. The estimator is (X·Y)_n = Σ_{i≠j} Ã B̃ / (n(n-3)) and R*_n = (X·Y)/√((X·X)(Y·Y)), which estimates **dCor²**.

**Implemented** (`hyppo/independence/dcorr.py`): `statistic` (123-152) → `compute_dist` with sklearn Euclidean distances
(`tools/common.py:497-501`) → `_dcorr` (416-449). `_center_distmat(bias=False)` (272-280) is exactly U-centring.
`_dcov(..., only_dcov=False)` (402) returns the bare sum; the scaling factors cancel in the ratio. The result is
`stat = covar/sqrt(varx*vary)` (443), with **no square root** when bias=False (446-447). It returns 0 if a variance is <= 0 (438-439).
Probe: hyppo 0.25339647074002397 and the from-scratch U-centred R*_n 0.25339647074002286 agree.

**Deviations**
- From canonical: none for the estimator. Note that it is on the squared scale.
- From documented:
  - The unbiased estimator uses U-centring, not "double-centering". Double centring is what the biased variant does.
  - The documentation leaves out the centring step entirely.
  - It does not say that `dcorr` is dCor² while `dcorr_biased` is dCor.
  - Ref [14] is the MGC paper, not Székely & Rizzo.
  - The SPI is labelled "unsigned" (config keywords, and the glossary's "[0,1]" description of distance correlation) but returns negative values.
- Candidate findings (not yet confirmed by a numeric check):
  - **Scale mismatch** between `dcorr` and `dcorr_biased`. On the probe pair, unbiased 0.2534 vs biased 0.5145; the biased value squared is 0.2647.
  - **Negative values**: 64.5% of 200 iid pairs at T=64 were negative, and the pyspi matrix gave −0.0054 on an independent pair.
  - The clamp to 0 on a non-positive variance has low impact. Through pyspi a constant series fails earlier, in the NaN check.
- sklearn computes the distances with a dot-product expansion. They differ from |x_i − x_j| by up to 4e-11 at T=4096, but the
  statistic changes by ≤ 1.3e-15 (probe5). A kernel tolerance of 1e-10 is therefore safe.

**Proposed checks:**
- Kernel, compared with `dcor.u_distance_correlation_sqr` (dcor 0.7 in the comparison venv, independent of hyppo) and with a from-scratch
  U-centring implementation (rtol 1e-10).
- Pipeline on `bivariate_gaussian`, compared with the analytic bivariate-normal dCor² (Székely, Rizzo & Bakirov 2007, Theorem 7(ii),
  checked against the paper text (arXiv 0803.4101, p.18): ρ=0.3 → 0.07236, ρ=0.8 → 0.56987). Values should converge to these squared-scale numbers, not to 0.269 / 0.755.
- `iid_gaussian` null centred on 0.
- Exact symmetry, and no dependence on argument order.
- `scaled_offset` should equal `bivariate_gaussian` (affine invariance).
- On the detrended pipeline, `trending` should equal `bivariate_gaussian` (detrend(D2 + a·t) = detrend(D2)).
- Qualitative detection of the x² and circle pairs.

## dcorr_biased — `distance.dcorr.biased-True`

**Documented:** supplement §1.2.1, p.5, the same text as above (identifier 23).

**Canonical:** the sample dCor of Székely, Rizzo & Bakirov 2007 (Ann. Statist. 35(6):2769-2794, §2.2). The distance matrices are
double-centred, V_n² = (1/n²)ΣA B, and R_n = √(V_n²(X,Y)/√(V_n²(X)V_n²(Y))).

**Implemented:** `_center_distmat(bias=True)` (265-271) double-centres with 1/n means. `_dcorr` takes covar/√(varx·vary) and then
**square-roots** it (446-447). Probe: 0.514514058070257, against the from-scratch dCor 0.5145140580702596.
This confirms the hypothesis that hyppo returns dCor, not dCor², for bias=True.

**Deviations:**
- From canonical: none.
- From documented: the centring step is not mentioned, and the scale mismatch with `dcorr` is not mentioned.
- Expected behaviour: a positive bias under independence. The probe mean was 0.211 at T=64.

**Proposed checks:**
- Kernel, compared with `dcor.distance_correlation` (rtol 1e-10). `distance_correlation_sqr` must not match.
- Pipeline compared with the analytic √dCor²: 0.269 and 0.755, approached from above.
- The null is positive and decreasing in T.
- `dcorr_biased²` should come close to `dcorr`. This documents the scale mismatch numerically.
- Symmetry, affine invariance, and the detrend equality.

## hsic (unbiased) — `distance.hsic.biased-False`

**Documented** (supplement §1.2.5, p.6): "HSIC [15] is an RKHS-based statistic that quantifies a statistical dependence between
random variables via a sample kernel matrix. ... HSIC yields a biased statistic, where the unbiased (and consistent) estimator can
be derived by double centering the kernel distance matrix [15]." [15] is Gretton et al., NIPS 2007. That paper gives the unbiased
estimator as a sum of three U-statistics (eq. 3), and then the biased HSIC_b = m⁻² tr(KHLH) (eq. 4) that it uses for the test.

**Canonical:**
- HSIC itself: Gretton et al. 2005 (ALT, LNCS 3734) and Gretton et al. 2007 (NIPS), with a Gaussian kernel. Gretton 2007 §4 says only
  "kernel size set to the median distance between points in input space". It does not say explicitly that σ is set separately for x and y.
  A separate median per variable is the common reading and common practice, but it is not what [15] states.
- The unbiased estimator: Gretton et al. 2007 eq. (3), written in matrix form as Song et al. 2012 (JMLR 13:1393-1434), eq. (5). Song's form is
  computed on Gram matrices with the diagonal zeroed. A brute-force probe at m=9 gives the two forms equal to 1.8e-16 (`eq3_vs_song.py`).
- HSIC is an unnormalised quantity.

**Implemented:**
1. `hsic.py:117`: `compute_kern(x, y, 'gaussian')`.
2. `tools/common.py:321-333`: the median of the off-diagonal |x_i − x_j| **of x only** gives γ = 1/(2·med²). It is stored in
   kwargs and **reused for y** at 342-343, through sklearn `rbf_kernel` (`pairwise.py:1583-1627`).
3. `hsic.py:118-119`: D = 1 − K/max K. Since max K = 1 (the diagonal), D = 1 − K.
4. `hsic.py:122`: D is passed to `Dcorr(bias=False, compute_distance=None)`. This U-centres D, and U-centring removes the constant,
   so the result is −Ũ(K) with a zero diagonal. It then normalises.

The result is HSIC_u(K,L)/√(HSIC_u(K,K)·HSIC_u(L,L)), with no square root. In the probe this matched Song et al. eq. (5) after normalising
(0.7662620588623132 from both), while raw HSIC_u was 0.0570.
This confirms the second briefing hypothesis: the code takes 1 − K/max K, passes it into Dcorr, and computes the bandwidth from x and reuses it for y.

**Deviations:**
- From canonical:
  1. The value is a normalised (correlation-like) HSIC, not HSIC.
  2. The bandwidth comes from the first argument only. With per-variable medians (common practice) the probe gives 0.7720 instead of 0.7663.
  3. As a result the kernel is asymmetric in (x, y).
- From documented:
  - The normalisation, the bandwidth rule and the shared x-bandwidth are all undocumented.
  - The unbiased estimator is described as "double centering". The code U-centres 1 − K, which equals Gretton 2007 eq. (3) and Song eq. (5)
    before normalisation. Double centring is what the biased variant does. Citing [15] for an unbiased estimator is itself supported (eq. 3).
  - The SPI is on the squared scale while `hsic_biased` is on the root scale.
  - It is labelled "unsigned" (config keywords; glossary-of-terms.md:21) but can be negative.

**Candidate findings:**
- **Order dependence.** On z-scored x0 ~ U(−1,1) vs x0² + noise, bivariate(0,1) = 0.3029 and bivariate(1,0) = 0.3266.
  The pyspi matrix reports the upper-triangle value, 0.302936, so permuting the processes changes the reported value.
  This is distinct from REVIEW M1 (tests/README.md bug 5 before b2f6d7e): this class *is* mirrored, but its kernel is asymmetric.
- **Relative-scale dependence when normalise=False.** Multiplying y by 1, 10 and 0.1 gives 0.791, 0.353 and 0.748 for `hsic`.
  `dcorr` stays at 0.829 in all three cases.
- **Negative values under independence:** 66% of 200 iid pairs at T=64.

**Proposed checks:**
- Kernel, compared with from-scratch numpy (Gaussian Gram with the median of the first argument, Song eq. (5), normalised; rtol 1e-10).
  A second, per-variable-median variant quantifies the bandwidth deviation.
- Pipeline direction and order check on `nonlinear_dependence`, `anm` and `contaminated`, with the processes in their original and in reversed order.
- Pipeline on `bivariate_gaussian`, compared with the population normalised HSIC. This is **provisional**. The constants are an audit
  derivation, not taken from the literature, and they were not yet checked at the trace stage (the check stage re-derived them; see "Open questions: answered", item 2). With a common σ = √2·Φ⁻¹(0.75) = 0.9539 and E[exp(−z'Az/2)] = det(I + AΣ)^−1/2,
  HSIC(ρ) = det(I + 2C_ρ/σ²)^−1/2 + (1 + 2/σ²)^−1 − 2·det(I + C'_ρ/σ²)^−1/2, with C_ρ = [[1,ρ],[ρ,1]] and C'_ρ = [[2,ρ],[ρ,2]].
  The normalised value is HSIC(ρ)/HSIC(1): ρ=0.3 → 0.04553 and ρ=0.8 → 0.47393 (`closed_form.py`, matching the earlier `popref.py`).
  Monte Carlo with Song eq. (5) at n=1500 (20 reps, fixed σ) gave 0.0418 ± 0.0017 and 0.4742 ± 0.0026.
  The check stage must re-derive these constants independently before using them.
- `iid_gaussian` null centred on 0.
- `scaled_offset` vs `bivariate_gaussian`: they differ when normalise=False and are identical on the default pipeline.
- The detrend equality.

## hsic_biased — `distance.hsic.biased-True`

**Documented:** §1.2.5, p.6 (identifier 30).

**Canonical:** HSIC_b = (1/n²) tr(KHLH) (Gretton et al. 2005; Gretton et al. 2007 eq. 4), with a median-heuristic Gaussian kernel.
Gretton 2007 §4 does not say explicitly that the median is taken per variable; that is the common reading.

**Implemented:** the same kernel construction as `hsic`. D = 1 − K is double-centred, which gives −HKH. The output is then
√(HSIC_b(K,L)/√(HSIC_b(K,K)·HSIC_b(L,L))), the square root of centred kernel alignment.
Probe: 0.8773521262252134 from hyppo and 0.8773521262252156 from the from-scratch implementation, while raw HSIC_b was 0.0577.
The result lies in [0, 1] (the Gram matrices are PSD) and has a positive bias under independence.

**Deviations:**
- The normalisation and the square root.
- The bandwidth taken from x and applied to y. With per-variable medians (common practice) the probe gives 0.8805 instead of 0.8774.
- Order dependence: bivariate(0,1) = 0.5561 vs bivariate(1,0) = 0.5771, and the matrix reports 0.55612.
- Relative-scale dependence when normalise=False: 0.890 for y and 0.603 for 10·y.
- None of these appear in the documentation.

**Proposed checks:**
- Kernel, compared with from-scratch numpy (rtol 1e-10).
- The pipeline order and direction check.
- √(population normalised HSIC), provisional with the same caveat as for `hsic`: ρ=0.3 → 0.2134, ρ=0.8 → 0.6884, approached from above.
- The null is positive and decreasing in T.
- The detrend equality.

## Independence of references

*Trace-stage text, partly superseded by the check stage:* R `dHSIC` 2.2 was later installed in `~/R/audit-lib` and used as the
external package for both HSIC units (see "Checks" and open question 6 below). The second bullet describes the situation at trace time only.

- `dcor` 0.7 is installed in the comparison venv and does not import hyppo. It covers both dcorr units.
- No independent HSIC package is available. R `dHSIC` failed to install according to `audit/env/r-packages.txt`, and R `energy` is also not installed.
  The HSIC references are therefore from-scratch numpy implementations, plus the analytic Gaussian population values.
- Nothing in the proposed checks imports hyppo.

## Open questions

1. **Population constants (resolved).** The Székely-Rizzo-Bakirov bivariate-normal closed form is Theorem 7(ii), checked against the paper
   text (arXiv 0803.4101, p.18). The formula gives 0.07236 and 0.56987 at ρ=0.3 and 0.8, and exactly 1 at ρ=1.
2. **Population normalised HSIC.** The values in the analytic reference (0.04553, 0.47393) are an audit derivation (closed form written out
   in the records), not a literature value. They remain unverified until the check stage re-derives them independently.
   The sample bandwidth also fluctuates around 0.9539, which adds variability at small T.
3. **Should the order dependence be reported as a finding?** It is real at kernel level. After pyspi's mirror, its size depends on how different the
   z-scored marginal shapes are. Possible upstream remedies are per-variable medians, or a symmetric choice such as the pooled median. Which of these is
   "correct" is a definitional choice, so the record frames it as a deviation from the usual per-variable median practice, not as a bug.
   Gretton 2007 itself does not state per-variable medians explicitly.
4. **Should the dCor/dCor² scale mismatch count as a hyppo design choice, or as a pyspi labelling problem?** hyppo documents in comments that it
   follows `energy::dcovu`, where only the biased dCov is square-rooted. pyspi's documentation is silent about it.
5. **Labels.** All four units have `Undirected` symmetry, but `dcorr` and `hsic` are labelled "unsigned" while returning negative values.
   This needs a decision on how the label set is used downstream.
6. **Cost.** Both triangles are computed and then one is discarded (`base.py:84-94`, `132-138`). The fast O(n log n) 1-D dCor path is never
   used by `statistic`. These are left for the optimisation stage; they do not change the values.

## Checks

**Setup.** The manifest `audit/checks/manifest/dcorr_hsic.json` has 155 runs (39 dcorr, 41 dcorr_biased, 44 hsic, 31 hsic_biased). References are in `audit/checks/refs/dcorr_hsic.py` (comparison venv: numpy/scipy, `dcor` 0.7, no hyppo/sklearn in the value path; R `dHSIC` 2.2 through `audit/checks/r/dcorr_hsic.R` with CSV exchange under `out/exchange/dcorr_hsic/`). Probes are in `audit/checks/probes/dcorr_hsic_probes.py` (project venv): `unmirrored` returns `bivariate(i, j)` for every ordered pair before pyspi's mirror, and `gamma_used` recovers the Gaussian gamma hyppo applies to the second argument's kernel. Results are in `audit/checks/out/results/dcorr_hsic.json`: **150 pass, 5 fail, 0 without reference**. pyspi-side compute was 454 s in total on a machine shared with other families. The longest single run was `dcorr_biased|iid_gaussian|4096|pipeline|null` (P=5, T=4096) at 72.9 s, above the ~60 s guide. HSIC runs are kept at T ≤ 1024.

The draft from the interrupted attempt was reused and corrected. Changes made in this pass:

- The Monte Carlo cache was regenerated under key version v2.
- `null-trend` now tests pyspi's own spread. The draft tested only the Monte Carlo spread.
- The `trend-contrast` rule was infeasible on the root scale: D2 + 0.3 > 1 for the ρ=0.8 pair. It now requires +0.3 on the independent pairs and an increase on (2,3). This was a reference bug.
- Song eq. 5 and HSIC_b reductions now accumulate in `np.longdouble`. In float64, the three sums of near-one Gram entries lost 1.1e-12 on raw `scaled_offset` A[3,0] (x scale 10, y scale 0.2). pyspi's hyppo route agrees with an 80-bit evaluation to 4.7e-15 (`scratchpad/dcorr_hsic_check/probe_ld.py`), so this was a reference precision bug, not a pyspi mismatch.
- Added runs: the `gamma` probe runs, and `scale` on nonlinear_dependence.

Check types:

1. **`scratch` / `dcor` / `dhsic` / `eq3` (exact, rtol 1e-10, atol 1e-12).** The stored matrix in both orientations is compared with from-scratch formulas, the `dcor` package and R dHSIC. Extras record the distance to the other canonical form.
2. **`truth-p01/p23` (statistical).** The pass rule is |pyspi − population| ≤ 2·SD_MC(T).
   - dCor population values come from Székely, Rizzo & Bakirov 2007 Theorem 7(ii). The closed form equals a quadrature of the three-term dCov² decomposition to 3e-15, and agrees with a 10 × 65536 U-statistic Monte Carlo (`popcheck`).
   - Normalised-HSIC population values use the det(I + AΣ)^−1/2 closed form. It equals a 2-D quadrature of each expectation and agrees with a 10 × 2e6 quadruple Monte Carlo: 0.045722 ± 0.001318 vs 0.045528, and 0.473045 ± 0.000992 vs 0.473935.
3. **`null` / `null-trend` (qualitative).** The mean over 10 iid pairs is compared with the Monte Carlo distribution of that mean, plus the T-scaling.
4. **`unmirrored` / `reversed` / `gamma` (exact).** These establish the argument-order convention and the bandwidth rule.
5. **`affine`, `detrend`, `trend-contrast`, `detected`, `scale`.**

**The five failures.**

- **`iid_gaussian|256|pipeline|null` on all four units** (z −2.03, −2.60, −2.18, −2.85). This is one event, not four: the four SPIs see the same draw.
  - On its own marginals (each row permuted independently, 2000 times), this draw's mean dCor² sits at the **0.15 %** lower tail (z −2.07).
  - Its mean r² is at the 23rd percentile.
  - The generator is plain iid N(0,1): Shapiro p 0.36–0.69, lag-1 autocorrelation ≤ 0.094 in magnitude.
  - The pyspi values on this draw equal the independent references exactly (`iid_gaussian|256|kernel|scratch`, `|dcor`).
  - T = 64, 1024 and 4096 pass for every unit (|z| ≤ 1.73).
  - The criterion was not loosened. These are recorded failures of a pre-set 2-SD rule on an unusual draw, not estimator defects (`scratchpad/dcorr_hsic_check/probe_null256.py`, `probe_fail.py`).
- **`dcorr_biased|nonlinear_dependence|256|pipeline|detected`.** The circle pair (2,3) gives 0.20818. That is above all 200 time permutations (max 0.20804; z_perm 5.43), but only 4.37 SD above the iid Gaussian null, against a pre-set margin of 5. This is a power limit of the V-statistic at T=256 on the circle, not a calculation issue. At T=1024 it passes with z_perm 11.3.

### dcorr (`distance.dcorr.biased-False`): matches, score 9 (anchor 9: squared vs not)

- **Exact checks.** Against the `dcor` package `u_distance_correlation_sqr`: 8/8 pass, max_abs ≤ 7.2e-14. Against from-scratch U-centring: 9/9 pass, max_abs ≤ 7.4e-14.
- **Which form hyppo returns.** The stored value is the **squared** bias-corrected R*_n. Its distance to √R*_n is 0.067–0.250 (`ref_maxabs_vs_dcor`).
- **Population.** Theorem 7(ii) gives 0.072362 at ρ=0.3 and 0.569869 at ρ=0.8:

| T | pyspi (0,1) | pyspi (2,3) | SD_MC (0,1) / (2,3) | max_z |
|---|---|---|---|---|
| 256 | 0.07474 | 0.59984 | 0.031 / 0.039 | 0.76 |
| 1024 | 0.05846 | 0.57756 | 0.014 / 0.020 | 1.00 |
| 4096 | 0.07584 | 0.55453 | 0.0069 / 0.0094 | 1.63 |

  The estimator is centred: Monte Carlo bias ≤ 0.0019.
- **Null.** The mean is centred on 0, and 60–90 % of null pairs are negative (min −0.0277 at T=64). The spread shrinks with log-log slope −0.91 (Monte Carlo −1.00, i.e. a T⁻¹ spread for the U-statistic of dCor²). T=256 is the shared draw failure above.
- **Symmetry.** The kernel is exactly symmetric: unmirrored max|A − Aᵀ| = 0.0, and reversed process order agrees to 1.7e-15.
- **Invariances.** Raw scaled_offset equals bivariate_gaussian to 2.8e-14 (affine invariance holds with normalise=False). The detrend equality holds to 2.6e-14.
- **nonlinear_dependence.** Pearson r is 0.015 for (0,1) and −0.039 for (2,3). dcorr gives 0.2294 and 0.0390, with z_perm 175 and 25.
- **Candidates.** All four are confirmed:
  - The scale mismatch with dcorr_biased (counted here). dcorr_biased² − dcorr = 0.00073 / 0.00038 at T=4096, while dcorr_biased − dcorr = 0.20 / 0.19.
  - The "unsigned" label with negative values. This is a new instance of an already-documented class: REVIEW M7 (negative debiased dspli/dswpli/ppc estimates are valid), REVIEW M8 (pec/pec_log negative despite "unsigned") and EEG_OPTIMIZATION_NOTES.md:169 (review topic "legitimate negative estimates"). None of them names dcorr; the record's `already_documented` now cites them.
  - The documentation's "double-centering" wording. The double-centred ratio is dcorr_biased², 0.08592 vs the stored 0.07474.
  - The zero clamp on constant input: hyppo returns 0.0; pyspi's default pipeline raises on NaN first (`probe_edge.py`).
- `new_finding: true`.

### dcorr_biased (`distance.dcorr.biased-True`): matches, score 9 (anchor 9: matches canonical; not 10 because the supplement description does not match)

- **Exact checks.** Against `dcor.distance_correlation`: 8/8 pass, max_abs ≤ 9.4e-14. Against from-scratch double centring: 9/9 pass. The distance to dCor² is 0.103–0.246, which confirms that hyppo returns **unsquared** dCor_n for bias=True.
- **Population.** The targets are √ of the Theorem 7(ii) values: 0.269001 and 0.754897.

| T | pyspi (0,1) | pyspi (2,3) | max_z |
|---|---|---|---|
| 256 | 0.29312 | 0.77809 | 0.84 |
| 1024 | 0.24778 | 0.76085 | 0.81 |
| 4096 | 0.27670 | 0.74492 | 1.51 |

  The positive bias at ρ=0.3 is +0.018 at T=256 and ≤ 0.0006 at T=4096.
- **Null.** Every value is > 0. The mean is 0.194, 0.090, 0.054 and 0.027 for T = 64 → 4096, with log-log slope −0.46 (Monte Carlo −0.50, i.e. T^−1/2).
- **Symmetry.** Exactly symmetric.
- **Documentation candidate (centring omitted).** Confirmed: the literal uncentred reading gives 0.814 against the stored 0.248 (`probe_edge.py`).
- **Score.** The estimator is fp-exact against `dcor` and the from-scratch double centring, but anchor 10 also requires matching the supplement description. The supplement omits the centring (literal reading 0.814 vs stored 0.248) and presents dcorr/dcorr_biased as biased/unbiased versions of one statistic on different scales. Score 9, as for `distance.hhg` in `hhg_mgc`, where a supplement-text mismatch also gave 9. (Changed from 10 after review.) Rubric reading is anchor 9 for both dcorr units (score_rubric_note), because anchor 9's squared-vs-not clause names the dCor^2 / dCor scale split; distance.hhg's rubric reading is anchor 9 through the estimator-variance clause. The wording above is the original reasoning for score_0_10.
- `new_finding: false`. Both confirmed candidates are counted on the dcorr record.

### hsic (`distance.hsic.biased-False`): variant, score 6 (anchor 6: deviates from canonical and the supplement in an interpretable way)

- **Exact checks.** The stored value is HSIC_u(K,L)/√(HSIC_u(K,K)·HSIC_u(L,L)), checked three ways:
  - From-scratch Song et al. 2012 eq. 5: 9/9 pass, ≤ 2.7e-14.
  - Brute-force Gretton et al. 2007 eq. 3: 2/2 pass, ≤ 6.8e-14.
  - The R route on dHSIC Gram matrices: 5/5 pass. Here `dHSIC::dhsic` (gaussian.fixed, bandwidth = median(dist(x))) gives a raw HSIC_b equal to the Python one to ≤ 1.5e-16, which pins the kernel convention exp(−d²/(2σ²)).
- **Stored vs raw.** The stored value is HSIC normalised by its self-terms. On z-scored data it is 8.4–18.8× the raw HSIC_u. For Gaussian margins the population factor is 1/HSIC(X,X) = 1/0.07724 = 12.9; the observed ratio is 12.7–13.1 on bivariate_gaussian.
- **Bandwidth.** The `gamma` probe shows that hyppo applies the **first argument's** median-heuristic gamma to the second kernel, to 6.2e-16 relative. The per-variable gamma differs by up to 31 % (anm).
- **Order dependence**, after z-scoring:
  - Unmirrored A[0,1] / A[1,0] is 0.3164 / 0.3330 on nonlinear_dependence and 0.4489 / 0.4229 on anm.
  - Reversing the process order changes stored values by up to 0.0166 and 0.0260 (5.8 %).
  - On raw scaled_offset the effect is large: 0.0073 vs 0.0549 for (0,1), and 0.3683 vs 0.5129 for (2,3), against z-scored values of 0.0334 / 0.4878.
- **Population.** The targets are normalised HSIC with σ = √2·Φ⁻¹(0.75) = 0.95387: 0.045528 and 0.473935. pyspi gives 0.05033 / 0.49968 at T=256 and 0.03335 / 0.48776 at T=1024 (max_z 1.20).
- **Null.** The mean is centred on 0, with 70–90 % of pairs negative.
- **Candidates.** All five are confirmed: order dependence; undocumented normalisation; the scale mismatch with hsic_biased (counted here; hsic_biased² − hsic ≤ 0.0024 while hsic_biased − hsic is 0.16–0.24); relative-scale dependence with normalise=False; and the "unsigned" label with negative values. The last is a new instance of the class already documented in REVIEW M7/M8 and EEG_OPTIMIZATION_NOTES.md:169 (none names hsic); `already_documented` now cites them.
- `new_finding: true`. `definition_under_review: true`, because whether the bandwidth is per variable, pooled, or taken from x is a definitional choice that Gretton 2007 does not state. Both the x-only and the per-variable values are recorded (`ref_per_variable_*`, `ref_perVar*`).

### hsic_biased (`distance.hsic.biased-True`): variant, score 6 (same anchor)

- **External check.** dHSIC `dhsic` gives HSIC_b(x,y), HSIC_b(x,x) and HSIC_b(y,y) with the x-bandwidth. √(HSIC_b(x,y)/√(HSIC_b(x,x)·HSIC_b(y,y))) equals pyspi to ≤ 2.4e-14 in 5/5 runs. From-scratch: 9/9 pass.
- **Stored vs raw.** On bivariate_gaussian at T=1024 (pipeline), the stored values are 0.18913 / 0.69937. These are √ of the normalised values 0.03577 / 0.48911, while raw HSIC_b is 0.00276 / 0.03782.
- **Order dependence.** Reversing the process order changes values by up to 0.0146 (nonlinear) and 0.0195 (anm).
- **Population.** √ of normalised HSIC gives 0.213374 and 0.688429. pyspi gives 0.24371 / 0.71089 at T=256 and 0.18913 / 0.69937 at T=1024 (max_z 1.02).
- **Null.** Every value is > 0, with mean slope −0.46.
- **Candidates.** All three are confirmed.
- `new_finding: true`. The order dependence and the undocumented normalisation and square root affect this SPI's own output; the mechanism is shared with hsic through hyppo `compute_kern`.

**Trace corrections.**

- The `implementation_trace` line `pyspi/calculator.py:294 … :297` is now `:329 … :332`. Those are the lines at HEAD a42ebc1 and are unchanged since a1f20c7.
- The trace's Monte Carlo for normalised HSIC at ρ=0.3 (0.0418 ± 0.0017) sits 2.2 se below the closed form. The check-stage Monte Carlo (0.045722 ± 0.001318) and the quadrature confirm 0.045528, so the earlier provisional constants are now re-derived and agree.
- No other trace field was refuted.
- After review: `already_documented` on dcorr and hsic changed from "no" to "partly", citing REVIEW M7, REVIEW M8 and EEG_OPTIMIZATION_NOTES.md:169 for the unsigned-but-negative class and EEG_OPTIMIZATION_NOTES.md:132 for the double work of the generic Undirected path. The mgc, dcorrx and mgcx sign items belong to the `hhg_mgc` and `lagged_dcorr_mgc` families and are not edited here.

## Open questions: answered

1. **Theorem number and formula of the bivariate-normal dCor².** Confirmed numerically. The Theorem 7(ii) expression equals an independent quadrature of dCov² = E|X−X′||Y−Y′| + E|X−X′|·E|Y−Y′| − 2E|X−X′||Y−Y″| to 3e-15 at ρ = 0.3 and 0.8, and equals exactly 1 at ρ=1. A 10 × 65536 U-statistic Monte Carlo gives 0.073086 ± 0.000753 (vs 0.072362) and 0.569129 ± 0.000753 (vs 0.569869) (`dcorr.biased-False|bivariate_gaussian|256|pipeline|popcheck`). The theorem number comes from the trace's reading of the paper and was not re-checked against the PDF here.
2. **Population normalised-HSIC constants.** Re-derived and cross-checked:
   - The det(I + AΣ)^−1/2 closed form equals a direct 2-D quadrature of each expectation (≤ 1e-8 normalised, ≤ 1e-9 raw).
   - A 10 × 2e6 quadruple Monte Carlo gives 0.045722 ± 0.001318 and 0.473045 ± 0.000992.
   - A scratch n=3000 U-statistic check with fixed σ gives 0.0454 ± 0.0024 and 0.4741 ± 0.0028 (`scratchpad/dcorr_hsic_check/popsanity.py`).
   - The sample bandwidth's variability is included in the per-T Monte Carlo SDs used as `se`.
3. **Is the HSIC order dependence a finding or a definitional variant?** Both, recorded as a confirmed finding with `definition_under_review: true`.
   - The mechanism is proven: the `gamma` runs show the y-kernel uses x's gamma to 6.2e-16.
   - The effect after pyspi's default z-scoring is measurable: up to 5.8 % (hsic) and 2.9 % absolute-relative on the largest pair (hsic_biased, anm (0,1)). It is large with normalise=False: up to 100× relative on a near-zero entry, and 0.368 vs 0.513 on scaled_offset (2,3).
   - Which bandwidth rule is "right" (per variable, pooled, or x-only) is a definitional decision. What is not a matter of definition is that an SPI labelled undirected returns a value that depends on process order.
4. **dCor vs dCor² scale split: hyppo design or pyspi labelling?** Both, measured.
   - hyppo states the convention (dcorr.py comment citing `energy::dcovu`), and the stored values follow it exactly: biased = √(V-statistic ratio), unbiased = R*_n.
   - pyspi presents the two as biased and unbiased versions of one statistic, but they differ by about a square root: dcorr_biased² − dcorr ≤ 0.0021 while dcorr_biased − dcorr is 0.16–0.25, and the same holds for hsic.
   - The fix would be in pyspi's documentation or labels (or a √ on one side), not in hyppo.
5. **"unsigned" for the unbiased dcorr and hsic.** The negative values are legitimate: 60–90 % of iid null pairs are negative, as Monte Carlo predicts. As REVIEW M7 recommends for the bias-corrected spectral estimators, the label should describe the population target, which is non-negative, separately from the estimator's range, which is [−1, 1]. Values should not be clipped. This needs the same metadata decision as M7/M8/M9 and the EEG_OPTIMIZATION_NOTES.md:169 topic "legitimate negative estimates". It is recorded as a confirmed, low-severity new instance of that documented class for these two SPIs.
6. **No independent HSIC package.** Superseded: R `dHSIC` 2.2 is now installed in `~/R/audit-lib` and was used as the external package for raw HSIC_b (exact to 1.5e-16) and for hsic_biased's stored value. dHSIC has no unbiased estimator, so hsic's unbiased value is checked against an R re-implementation on dHSIC Gram matrices and against two independent Python forms (Song eq. 5, Gretton eq. 3). R `energy` is still not installed; `dcor` 0.7 covers both dcorr units.
7. **Both triangles computed; fast dCor path unused.** This is unchanged and belongs to the optimisation stage. The double work is already documented (EEG_OPTIMIZATION_NOTES.md:132). The measured costs are dcorr 22 s and dcorr_biased 41 s per P=4, T=4096 matrix, against 0.9 s and 1.3 s at T=1024 (O(T²)). The unmirrored probe shows that dcorr's kernel is exactly symmetric, so for dcorr computing only the upper triangle would be bit-identical. For hsic it would also be identical, because the lower triangle is discarded.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls from `audit/profiles/summary.json` (quiet machine, single thread), shares from the cProfile runs at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`, percentages of the `compute_spi` frame; profiled walls are inflated). Nothing in pyspi was changed; the two micro-benchmarks below hold their candidates inside `audit/profiles/micro/`. "Matches the baseline" means the candidate reproduces today's output, nothing more. None of these units is in the retained EEG selection.

### Walls

| unit | 8x1024 | 32x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|
| dcorr (`distance.dcorr.biased-False`) | 3.70 s | 65.9 s | 918 s | extrapolated-pairs-from-32 (t_pair 66 ms, t_fixed 0) |
| dcorr_biased (`distance.dcorr.biased-True`) | 4.05 s | 70.9 s | 986 s | extrapolated-pairs-from-32 |
| hsic (`distance.hsic.biased-False`) | 5.31 s | 95.9 s | 1335 s | extrapolated-pairs-from-32 |
| hsic_biased (`distance.hsic.biased-True`) | 5.60 s | 101.3 s | 1410 s | extrapolated-pairs-from-32 |

Complexity O(P^2 T^2) for all four: every ordered pair (Undirected over `Directed.multivariate`, both triangles computed, the lower one overwritten) builds two dense T x T matrices. Measured P_exponent 2.07-2.09 and T_exponent 2.41-2.67 (above 2; a cache effect from the T x T temporaries at T=1024 is a plausible cause, not measured). The pair term is the whole cost at 118 channels.

### Where the time goes

- dcorr: `hyppo/independence/dcorr.py:Dcorr.statistic` self time 84.5% (84.4% biased). cProfile cannot see numba, so this is the `_dcorr` kernel: two `_center_distmat` calls (column/row sums, `np.repeat` temporaries) and three `_dcov` products. The rest is `hyppo/tools/common.py:compute_dist` -> sklearn `_euclidean_distances`, 14.6% (14.8%), with `safe_sparse_dot` 8.0%. opaque_share 0.003 is a lower bound: the numba time sits inside a Python frame's tottime.
- hsic: the same hidden `_dcorr` kernel, 60.0% (61.7%), called through `hsic.py:Hsic.statistic`. `compute_kern` takes 30.9% (29.0%): sklearn `rbf_kernel` 16.2% (14.8%) and the median bandwidth by `numpy partition` 9.8% (9.5%). `Hsic.statistic`'s own `1 - K/max(K)` temporaries take 7.1%.

### Opportunities

- **dcorr_hsic-1, per-channel cache (cache-sharing, measured; dcorr and dcorr_biased only).** Each channel's distance matrix, its centred form and its self inner product are computed once per window, leaving one cross product `_dcov(A_i, A_j)` per pair. `audit/profiles/micro/opp_dcorr_hsic-1.py` calls the same hyppo numba functions on the same arrays.
  - At 8x1024: dcorr 4.09 s -> 0.394 s (10.4x), dcorr_biased 4.28 s -> 0.387 s (11.0x). Output is bit-identical at 7x100 and 8x1024 for both bias flags, and also against the 7x100 golden snapshot. The upper-triangle variant (adding the cross-cutting `xc-undirected-upper`) gives 11.6x / 13.0x.
  - The per-component times (about 35-39 ms per channel setup, 1.5-2.0 ms per cross product) project to about 25-32 s per window at 118x1024, against 918 / 986 s: **~30x**. The cache gets better with P because each setup is reused by P-1 pairs.
  - Memory: one cached T x T float64 matrix per channel, about 0.94 GB at 118x1024.
  - The same sklearn distance call appears in dcorr_biased, hsic (`l2`), mgc, hhg and dcorrx/mgcx lag 0. A Data-level cache, like the existing `data.barycenter`, would share it across SPIs.
- **dcorr_hsic-3, per-channel x-side cache for hsic (cache-sharing, not measured; hsic and hsic_biased).** For hsic the bandwidth comes from the first argument and is used for both Gram matrices (`hyppo/tools/common.py:compute_kern`), so only the x side and the squared distances are per channel. The y-side `exp(-gamma_x D2_y)`, its centring and two products stay per pair. Estimate **~2x** from the profile shares (median 9.8%, half of `rbf_kernel`, the l2 distances, and about half of the numba share). This is unverified-bit-identical, because the same sklearn and numba calls run on the same arrays.
  - Until the Workflow 3 recheck this estimate was filed under `dcorr_hsic-1` on the hsic records. It got its own id because the mechanism (x side only), the speedup (~2x against ~30x) and the risk class (unverified, against measured bit-identical) differ from the dcorr cache, and a shared id must carry one opportunity text.
- **dcorr_hsic-2, hyppo's O(T log T) path (algorithmic, measured; dcorr and dcorr_biased only).** Both dcorr records list `dcorr_hsic-1` under its `subsumed_by`: the two are alternatives, because the fast 1-D path builds no distance matrix for `dcorr_hsic-1` to cache, and its measured per-pair cost (about 4.1 ms at 8x1024) is above the about 1.5 ms cross product per pair that `dcorr_hsic-1`'s micro-benchmark leaves, so build_csv ranks it with no saving on top of `dcorr_hsic-1` (README 'Optimisation columns', rule 4). Setting `Dcorr(bias=...).is_fast = True` before `statistic` makes hyppo skip the distance matrices and call `_fast_1d_dcov` three times.
  - Bias mapping is unchanged:
    - bias=False uses the U-statistic denominators n(n-3), n(n-3)(n-2), n(n-3)(n-2)(n-1) and the unsquared ratio, which is today's R*_n on the squared scale.
    - bias=True uses the V-statistic denominators n^2, n^3, n^4 and the final sqrt, which is today's dCor_n.
  - `audit/profiles/micro/opp_dcorr_hsic-2.py`, every ordered pair, at 8x1024:
    - dcorr: 3.89 s -> 0.230 s (16.9x), max_abs_diff 9.6e-15, max_rel_diff 2.1e-10 (on a near-zero entry).
    - dcorr_biased: 4.15 s -> 0.228 s (18.2x), max_abs 8.9e-14.
    - At 7x100 the speedup is only 1.9x; the diffs there and against the golden snapshot are below 7e-15. The ratio grows about as T / log T.
  - Unit speedup at 118x1024 is **~16-18x** (per-pair costs on both sides). Caching each channel's `_fast_1d_dcov(x, x)` would leave one call per pair instead of three, about 45x; that is a complexity estimate, not measured.
  - Within-tolerance. Compared with dcorr_hsic-1 it needs O(T) memory instead of a T x T matrix per channel.
  - Not applicable to hsic: its distances are kernel-induced, not |x_i - x_j|.

### Seed items refuted or narrowed

- A distance-matrix-only cache (seed a as stated) removes 14.6% of dcorr (Amdahl 1.17x). It pays off only when the centred matrix and its self product are cached too, which is dcorr_hsic-1.
- For hsic the kernel matrix is not purely per channel, because the bandwidth comes from the first argument.
- The fast 1-D path (seed b) is confirmed for dcorr and does not apply to hsic.

### Shared versus incremental cost

Family-warm increments at 8x1024: dcorr 3.66 s, dcorr_biased 3.90 s, hsic 5.33 s, hsic_biased 5.61 s, each equal to its fresh wall. Nothing is shared today. After dcorr_hsic-1 and dcorr_hsic-3 only the per-channel Euclidean distance matrix (squared, for hsic's Gram matrix) would be shared by all four. No two configured units share a centred matrix: dcorr U-centres D, dcorr_biased double-centres D, and hsic / hsic_biased centre 1 - K/max K, whose bandwidth comes from the first argument. Halving the pair loop (`xc-undirected-upper`) applies to all four; it is owned by the cross-cutting group.
