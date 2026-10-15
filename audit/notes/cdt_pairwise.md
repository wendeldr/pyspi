# cdt_pairwise: define, trace and check (anm, cds, reci, igci)

Units: `causal.anm`, `causal.cds`, `causal.reci`, `causal.igci`. All four are `Directed` SPIs in
`pyspi/statistics/causal.py`. Each one delegates to `cdt` (installed **0.6.0**; the supplement and
GitBook name v0.5.23, a mismatch already recorded in `DEPENDENCY_MIGRATION.md`). The regression goldens
for these identifiers are snapshots of current output. They are not used as a reference here.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared path

- `pyspi/data.py:184-195`: by default the data are z-scored (ddof=1) and not detrended. The stored
  array is (M, T, 1), so `z[i]` is a (T, 1) column (`data.py:108-118`).
- `pyspi/config.yaml:324-367`: the four classes have empty `configs`, so
  `calculator.py:261-267` instantiates them without arguments. The labels that are reported come
  from the class attributes, not from the config.
- `pyspi/base.py:84-94` `Directed.multivariate`: `A[i, j] = bivariate(i, j)` is computed for every
  ordered pair, with no mirroring. In every unit, process `i` is the hypothesised cause `x` and `j`
  is the hypothesised effect `y`.
- pyspi always calls cdt's single-direction score function. It never calls cdt's `predict_proba`
  decision (score(b,a) − score(a,b)). The only exception is IGCI, whose `predict_proba` is itself
  the single score.
- `calculator.py:327-339`: if the call raises any exception, the whole matrix is set to NaN.

Probe scripts are in
`/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/cdt_pairwise/`
(`p1_anm.py`, `p2_anm.py`, `p3_others.py`, `p4_igci.py`). The datasets come from `audit/checks/data`.

---

## causal.anm: additive noise model

**Documented** (supplement §1.3.1, p.8): "computed by first predicting y from x via a Gaussian
process (with a radial basis function kernel), and then computing the normalized HSIC test statistic
from the residuals (see Sec. 1.2.5 above)". The listed keywords include *nonlinear* and *unsigned*.

**Canonical** (Hoyer et al. 2008, NIPS 21, §4–5): regress y on x by GP regression "using a Gaussian
kernel and independent Gaussian noise, optimizing the hyperparameters". Then test whether the
residual is independent of x with HSIC and a gamma-approximated p-value, and repeat in the reverse
direction. The paper warns that "overfitting may lead one to falsely accept models". As a score:
HSIC_b(residual, x), where lower values are more consistent with x → y.

**Implemented**:
- `causal.py:17-24`: pyspi monkeypatches cdt's `ANM.anm_score` at import (`ANM.anm_score =
  corrected_anm_score`, cdt PR #155). The patched function fits
  `GaussianProcessRegressor(random_state=42).fit(x, y)`, sets
  `r = predict(x).reshape(-1,1) - y` and returns `normalized_hsic(r, x)`. The original cdt
  function (`ANM.py:178-192`) would broadcast (T,) − (T,1) into a (T,T) array.
- sklearn 1.9.1 defaults:
  - `_gpr.py:249`: kernel `C(1.0) * RBF(1.0)`, both with bounds 1e-5..1e5 (`kernels.py:1236, 1511`).
  - `_gpr.py:215`: alpha = 1e-10.
  - There is **no white-noise kernel**.
  - A single L-BFGS-B run (`_gpr.py:299-341, 658-674`).
  - `random_state` is used only for restarts, and `n_restarts_optimizer=0`, so it has no effect
    (probe: max difference 0.0).
- `ANM.py:106-111`: `normalized_hsic` standardises r and x (ddof=0). `FastHsicTestGamma`
  (`ANM.py:69-103`) then works as follows:
  - when T > 200, it takes a deterministic stride subsample of 200 points;
  - it uses median-heuristic Gaussian kernels, σ² = ½·median of the squared distances
    (`ANM.py:53-66`);
  - it returns `(1/m)·Σ Kc∘Lc`, which is **m·HSIC_b** with m = min(T, 200), and 0 if the result is
    not finite.
  - It computes no p-value, despite its name.
  - A from-scratch version reproduced 8.016951686623885 exactly.

**Deviation analysis**:
- *Candidate, high impact.* The marginal-likelihood optimum for a noise-free GP interpolates the
  data. On every pair probed, the RBF length scale went to 1e-5…8.3e-4 (maximum 8.33e-4 on anm_T64
  pair 2 → 3; sklearn ConvergenceWarning:
  "close to the specified lower bound 1e-05"). R² was 1.0, the residual std was about 1e-10, and
  `r ≈ −(1e-10/c²)·y` (median ratio 1.000001).
- The anm score is therefore approximately m·HSIC_b(y, x) of the raw pair, which is a
  near-symmetric dependence measure:
  - On anm_T64, 9 of 12 entries equal plain HSIC(z_j, z_i) to 4 decimals. The exceptions are (2,0),
    (2,1) and (2,3): 0.6091, 0.5530 and 6.314 vs 0.6409, 0.5896 and 7.417. On anm_T256 none of the 12
    agree to 4 decimals, and the maximum relative difference is 0.124 (probe `fix_cdt/anm_count.py`).
  - Across five datasets the maximum relative difference from plain HSIC was 0.09–0.24.
- As a result, the causal ranking is unreliable:
  - anm_T256: A[2,3] = 22.16 > A[3,2] = 20.03.
  - anm_T1024: A[0,1] = 7.98 > A[1,0] = 7.68.
- A GP with a WhiteKernel ranks the true direction lower in the probes (anm_T1024: 0.41 vs 2.38).
- Hypothesis: the score sits at roundoff level in the residual, which may explain the
  documented SciPy 1.18 drift.
- Other deviations:
  - The supplement's cross-reference to §1.2.5 suggests the `hsic` SPI (hyppo: normalised,
    square-rooted). This statistic is different: unnormalised, it scales with m when T < 200, and it
    is computed on a subsample.
  - The class label is `'linear'` (`causal.py:15`), while the config and supplement say
    *nonlinear*.
  - The class-body monkeypatch changes cdt globally for the whole process.
  - Only one direction is returned. Lower means more consistent with i → j.

**Proposed checks**:
1. Kernel, exact: a from-scratch m·HSIC_b.
2. Kernel, qualitative: a from-scratch numpy/scipy GP with the same model specification, to show
   that the collapse comes from the model specification and not from sklearn.
3. Pipeline, tolerance: compare pyspi's A with the symmetric plain-HSIC matrix. A pass supports the
   reduction to HSIC(x, y).
4. Pipeline, direction, on the `anm` dataset: compare with a canonical Hoyer-style reference (RBF +
   noise GP, HSIC with a gamma p-value). This establishes the direction convention, A[i,j] =
   score for i → j with lower meaning causal.
5. Pipeline-detrend, qualitative: sensitivity to input scale.

---

## causal.cds: conditional distribution similarity

**Documented** (supplement §1.3.3, p.8): "the standard deviation of the conditional probability
distribution of y given x, where the distributions are estimated by discretizing the values."

**Canonical** (Fonollosa 2016, arXiv:1601.06680, §2.1 and §2.3):
CDS(X,Y) = sqrt((1/M) Σ_y var_x p_n(y|x)), where the conditional distributions are centred to zero
mean and quantised into 2·maxdev·sfactor+1 levels of width σ/sfactor. The paper uses sfactor = 3 and
maxdev = 3 "for almost all measures". The score is used in both directions as a classifier feature.

**Implemented** (`CDS.py:125-195`, defaults ffactor = 2, maxdev = 3, minc = 12):
- x is standardised, re-standardised on the samples with |x| < 3, then rounded into 13 bins of width
  0.5 SD and clipped to ±6.
- For every x-bin with **more than 12** samples, the y values are centred on the bin mean, divided by
  the global std(y), quantised to −6..6 and turned into a pmf.
- The result is `np.std` of the pmf matrix with columns centred across bins. This equals Fonollosa's
  formula with ddof = 0.
- If no bin qualifies, it returns **0** (`CDS.py:190-191`). If exactly one bin qualifies, the 1 × 13
  matrix minus its own column mean is all zeros, so `np.std` also returns exactly 0.0
  (`CDS.py:193-195`). When all qualifying bins have identical pmfs the result is 0 only up to rounding
  of the column mean: exactly 0.0 for 2 bins, up to ~2e-17 for 3 or 7 bins with non-dyadic pmf entries
  (probe `cleanup/cdt_pairwise/ident_bins.py`).
- A from-scratch numpy version matches to 1.4e-17.

**Deviation analysis**:
- *Candidate.* The output is an exact 0 whenever at most one x-bin has more than 12 samples, and is 0 up
  to rounding when all qualifying bins have identical pmfs. A 0 looks like the most causal possible score, not a
  missing value.
  - anm_T64, all 12 entries: process 0 has no qualifying bin (max count 12); processes 1–3 have
    exactly one (max counts 16, 16, 18).
  - iid_gaussian_T64, 8 of 20 entries: rows 0 and 2 each have exactly one qualifying bin (max counts
    16 and 13).
  - Across all 76 check files (probe `fix_cdt/cds_zero.py`), there are 91 exact zeros, all at T = 64.
    In 90 of them, row process i has at most one qualifying bin. The remaining one,
    narrowband_f0-0.25_T64 A[1,0], has two 32-sample bins with identical y-level counts.
  - The no-bin branch is impossible for T > 156 (13 bins × 12). The single-bin branch is not tied to
    short T: a spike-heavy input at T = 458 (406 samples in one bin, every other bin ≤ 11) returned
    0.0 (probe `fix_cdt/cds_zero3.py`).
- *Candidate.* The supplement wording describes the SD of Y|X. The code computes the variability of
  the conditional distribution shapes across x-bins.
- ffactor = 2 differs from the paper's general sfactor = 3. Whether the original jarfo CDS used 2 is
  an open question.
- ddof = 0 is used, whereas the paper says "sample variance".
- Only one direction is returned. Lower means more consistent with i → j.

**Proposed checks**:
- Kernel, exact: from scratch.
- Kernel, tolerance: the paper variant with sfactor = 3 and ddof = 1.
- Pipeline, direction: on `anm` (probe T=1024: A[0,1] = 0.023 < A[1,0] = 0.110).
- Predicate check for the zero outputs (exact 0.0 for the count part, |A| < 1e-15 for the identical-pmf part). The predicate is: at most one x-bin of z_i has more
  than 12 samples, or all qualifying bins have identical conditional pmfs. Include an inline
  spike-heavy input at T ≥ 256.
- Affine invariance: on `scaled_offset`.

---

## causal.reci: regression error based causal inference

**Documented** (supplement §1.3.4, p.8): "the MSE of the linear regression of the cubic (plus
constant) of x with y".

**Canonical** (Blöbaum et al. 2018, AISTATS, eq. (1) and §4–5): scale both variables to [0,1] and fit
regressions in both directions. Infer X → Y when the MSE of predicting the effect from the cause is
smaller. The paper's function classes are LOG, MON (a·xⁿ + b with n ∈ [2,9]), POLY, SVR and NN. The
MSE is measured on random train/test splits and averaged over runs.

**Implemented**:
- `RECI.py:104-116`: both variables are min-max scaled. `PolynomialFeatures(3)` is applied, with the
  x and x² columns zeroed, which leaves [1, x³]. Then `LinearRegression` and the in-sample MSE.
- This matches numpy `lstsq` on [1, x³] to 1.4e-17.

**Deviation analysis**:
- Against the documentation: none in substance. The min-max scaling and the in-sample MSE are not
  mentioned.
- Against the canonical method:
  - a single fixed monomial (n = 3);
  - no held-out error;
  - only MSE_{i→j} is returned, without the comparison.
- Lower A[i,j] means more consistent with i → j (probe anm_T256: 0.012 vs 0.040).
- The fixed odd cubic shows no clear asymmetry on linear pairs (delayed_noise). This is the cdt
  design, flagged only for interpretation.

**Proposed checks**:
- Kernel, exact: `lstsq`.
- Pipeline, direction: against the ground truth plus a from-scratch canonical RECI (MON family,
  held-out MSE).
- Invariance: on `scaled_offset` and on `trending` with detrend.

---

## causal.igci: information-geometric causal inference

**Documented** (supplement §1.3.2, p.8): "the difference in differential entropies where the
probability density is computed via nearest-neighbor estimators". The listed keywords include
*unsigned*. The class and GitBook name it "Information-Geometric Conditional Independence".

**Canonical** (Daniusis et al. 2010, UAI, §2.3, §3.2, §3.5 eq. (12)): with the Gaussian reference
measure, standardise both variables. Then C_{X→Y} = Ŝ(Y) − Ŝ(X), where
Ŝ = ψ(m) − ψ(1) + 1/(m−1) Σ log|x_{i+1} − x_i|. Infer X → Y if C_{X→Y} < 0.

**Implemented** (`IGCI.py:126-151`, defaults `ref_measure='gaussian'`, `estimator='entropy'`):
- A[i,j] = Ŝ(z_i) − Ŝ(z_j) = **−C_{i→j}**, using StandardScaler (ddof = 0). Zero spacings are
  skipped, but the divisor stays m−1 (`IGCI.py:42-56`).
- It matches a from-scratch eq. (12) to 4.4e-15.
- It is exactly antisymmetric (max |A + Aᵀ| = 0.0).
- Inline deterministic probe (X ~ U(0,1), Y = X³): A[x,y] = 0.77, 0.85 and 0.90 at T = 256, 1024
  and 4096. The analytic −C^G is 0.883.

**Deviation analysis**:
- *Candidate.* The output is signed and antisymmetric (observed −0.49..+0.49), but it is labelled
  `unsigned` (`causal.py:56, 60`; `config.yaml:363`; supplement). Symmetrising or taking absolute
  values downstream would destroy it.
- *Candidate.* The sign convention is undocumented: positive means i → j, the opposite sign to the
  paper's C.
- Otherwise the code matches the canonical Gaussian-reference entropy estimator.
- Minor: the bivariate raw-array path departs from antisymmetry by about 5e-9, and the multivariate
  path does not (0.0). Mechanism (probe `fix_cdt/igci_raw.py`): `parse_bivariate` (`base.py:33-36`)
  builds Data with `add_process`. That z-scores only the first array (`data.py:238`) and appends the
  second raw (`data.py:230-234`). So bivariate(x, y) sees (zscore(x), raw y), and bivariate(y, x) sees
  (zscore(y), raw x). Both pairs reproduce the pyspi values exactly (x ~ U(0,1), y = x³, T = 1024:
  0.8105075201 vs −0.8105075125). After the StandardScaler they differ only at ulp level, which the log
  of small spacings amplifies. With both inputs z-scored, the gap is 0.0. This is the two-array
  preprocessing asymmetry owned by `preproc.two_array_api` (see also `base.pair_handling`). It is
  not specific to igci.

**Proposed checks**:
- Kernel, exact: from-scratch eq. (12).
- Analytic and statistical: the U(0,1) → x³ deterministic case, compared with −C^G = 0.8832. This
  check also establishes the sign convention.
- Exact antisymmetry, on the pipeline path.
- Pipeline, qualitative direction: on `anm`, which lies outside the method's assumptions.

---

## Open questions

1. Did the original jarfo/Fonollosa code use ffactor = 2 for the CDS feature (cdt's default), or the
   sfactor = 3 that §2.1 states for most measures? This needs the original `features.py`.
2. Is the GP length-scale collapse present on real EEG windows? Long T would make it more extreme.
   Does pyspi intend `anm` to be "HSIC of the raw pair"? If not, should a noise kernel (for example
   `C*RBF + WhiteKernel`) be considered? Any such change is a definition change and needs a decision.
3. Does the collapse explain the SciPy 1.18 drift in `DEPENDENCY_MIGRATION.md`? The hypothesis is
   that roundoff-level residuals make the score sensitive to the optimiser's endpoint.
4. Should the zero output of `cds` be reported as NaN when at most one x-bin has more than 12 samples?
   This mostly happens at short T, but it can happen at any T for spiky inputs. It currently looks
   like maximal causal evidence.
5. Should `igci` be relabelled `signed`, and should `anm`'s `'linear'` label be corrected to
   `'nonlinear'`? Both relate to the label discussion in REVIEW M9.
6. (Resolved) The 5e-9 igci antisymmetry gap on the raw-array bivariate path comes from
   `Data.add_process`, which z-scores only the first array. It is tracked under
   `preproc.two_array_api`.
7. What is the cost of `anm` at T = 4096? T = 1024 took about 5 s per ordered pair, which is O(T³)
   per likelihood evaluation. This affects check design and the later optimisation stage.

---

## Checks

Files: `audit/checks/manifest/cdt_pairwise.json` (170 runs), `audit/checks/refs/cdt_pairwise.py` (comparison venv),
`audit/checks/probes/cdt_pairwise_probes.py` (project venv). Results: `audit/checks/out/results/cdt_pairwise.json`,
153 pass and 17 fail, 0 without a reference. Every fail is explained below. Each is either a real mismatch with the
reference or ground truth, or a direction check where the method itself fails on the data (both pyspi and the reference fail).
The references build on the self-tested kernels in `refs/_kernels_distance.py` (`anm_replica`,
`gp_fit_hyperparameters`, `cdt_fast_hsic_gamma`, `cds_ref`, `reci_ref`, `igci_ref`). None of them imports cdt or sklearn.
`refs/cdt_pairwise.py` adds the canonical references: Hoyer et al. ANM with a C·RBF + White GP, a full-sample HSIC and
a gamma p-value; Blöbaum et al. RECI with the MON family and held-out MSE; and Fonollosa CDS with sfactor 3 and ddof 1. The ANM
and RECI canonical forms, including the gamma p-value, were written from recollection and not re-checked against the
papers or hsicTestGamma.m.
Directed decisions are tested on `A.T − A`. The probe `pairdiff` returns that matrix, which is cdt's `predict_proba`
form, so `> 0` means i → j. For igci, A is tested directly.

Calculator citations in the four records were checked against HEAD (a1f20c7). They are already current: `_load_yaml`
251-267 with `except TypeError` at 261, `compute` 329/332/336-339, and warning 354-355. The other remaining trace issues
listed for this family were also fixed before this stage: the CDS identical-pmf wording (exactly 0.0 for ≤ 2 bins, ~1e-17
for ≥ 3), the `base.py:41-43` nit, and the criterion of proposed_checks[3]. The check runs confirm the CDS zero predicate as worded.

### Thread and environment sensitivity of anm (applies to every anm number)

pyspi's anm value is reproducible to the last bit only under a fixed BLAS configuration. In the check run, sklearn's
fitted theta is reproduced exactly only with `OPENBLAS_NUM_THREADS=1`. This was found with the scratch script
`cdt_pairwise_check/inv4.py`: the default multi-threaded OpenBLAS moves log theta by up to 5.6e-7 and the score by ~1e-6.
The reference therefore sets one thread before importing numpy. `pyspi/__init__.py:4` sets `OMP_NUM_THREADS=1` for BLAS
libraries loaded after pyspi. This sensitivity explains why a scratch script got anm_T256 A[0,1] = 8.016948001378065 while
the pipeline (and the trace probe) gives 8.016951686623885.

### causal.anm: verdict related-quantity, score 4 (rubric 4: related but different quantity)

- **Replica at sklearn's theta** (`|theta` probe → `|replica-theta`, exact):
  - 8 of 9 runs match to ≤ 1.6e-15: anm T64/256 on the pipeline and kernel paths, anm T1024 pairs 0-1 and 2-3
    (0.0), and iid_gaussian, bivariate_gaussian and contaminated T256.
  - `causal.anm|nonlinear_dependence|256|pipeline|replica-theta` fails at 7.5e-8 (one pair, (3,1)). The same numpy
    replica run inside the project venv (`|replica-projectenv`) gives 6.6e-16 against pyspi. The mismatch is
    therefore scipy 1.18.1 vs 1.17.1 LAPACK roundoff amplified by the estimator. It is not a replica defect.
- **Replica with an independent L-BFGS-B fit** (`|replica-ownfit`, rtol 1e-5):
  - anm_T64 9.9e-7, anm_T256 9.5e-7, nonlinear_dependence_T256 4.7e-6.
  - bivariate_gaussian_T256 fails at 1.05e-5. The lml gap is 3.5e-6, so the optimum is flat.
  - iid_gaussian_T256 fails at 1.17e-2. For pair (1,2), sklearn's L-BFGS-B stops at lml −362.672, while the independent
    fit from the same start reaches −362.457 (`lml_ref_minus_sklearn_max` 0.215). The theta run differs by 0.498 in
    log length scale.
  - So pyspi's value is defined only up to the optimiser's end point. In 1 of 20 pairs that end point is not the
    likelihood optimum.
  - Theta agreement on the other 8 runs is ≤ 2.6e-4 in log theta.
- **GP collapse** (`|collapse`, 7/7 pass):
  - std(r)/std(y) ≤ 1.3e-9 in both sklearn's fit and the independent fit.
  - Maximum length scale 8.3e-4 (anm_T64). Both pairs of anm_T1024 0-1 sit at the 1e-5 bound.
  - The median of r/(−(α/c)·y) is within 1 ± 3e-6.
  - The collapse is a property of the model specification (C·RBF, alpha = 1e-10, no noise term). sklearn is not the cause.
  - Exactly, r = K(K+αI)⁻¹y − y = −α(K+αI)⁻¹y. The score is therefore m·HSIC_b(x, standardised (K+αI)⁻¹y).
- **anm vs plain m·HSIC_b of the raw pair** (13 rawhsic runs, 10/13 pass: `|rawhsic` 7/10, `|p01-rawhsic` 1/1,
  `|p23-rawhsic` 2/2):
  - Median relative difference 2.5e-7 to 0.084.
  - Three runs fail: contaminated_T64 (median 0.30, max 0.52), contaminated_T256 (0.12, 0.21) and
    nonlinear_dependence_T64 (median 1.2e-6, but one entry 0.85).
  - The failing entries are those where the fitted length scale exceeds some x spacings. For example,
    contaminated_T64 0→1 has l = 4.9e-3 with 31 spacings < 3l. In that case (K+αI)⁻¹ mixes near-coincident points and
    the score departs from HSIC(x, y).
  - The trace-stage statement "essentially m·HSIC_b(y, x)" is refined in `implemented_as` to the exact form above.
- **Direction vs ground truth** (`|direction`, canonical Hoyer reference as value):
  - `causal.anm|anm|256|pipeline|direction`: agreement 0.5. The tanh pair is wrong (A.T − A at (2,3) = −2.13).
  - `causal.anm|anm|1024|pipeline|p01-direction`: agreement 0.0 (D01 = −0.299).
  - `p23-direction`: agreement 1.0 (D23 = 1.22).
  - The canonical reference recovers the truth in all three. At T1024, 0→1 gives 0.265 (p 0.60) and 1→0 gives
    8.88 (p 4e-65). At T256, 2→3 gives 0.500 (p 0.13) and 3→2 gives 0.617 (p 0.059).
  - The canonical reference (`anm_canonical`, `hsic_gamma_test`) was written from recollection of Hoyer et al. and of
    hsicTestGamma.m and was not re-checked against either. The direction decision uses the full-sample statistic
    m·HSIC_b (lower = causal), not the p-value; the quoted p-values come from a recalled gamma-approximation formula and
    are unverified.
- **Orientation on lagged data** (`|orientation`): agreement 1.0 against the independent replica on var2_bivariate and
  logistic_maps_chaotic. Truth is recovered on logistic_maps_chaotic but not on var2_bivariate.
- **200-point subsample** (`|p01-subsample`, `|p23-subsample`):
  - The 200-point replica reproduces pyspi to 3.9e-7 and 9.1e-8.
  - The full-sample statistic is 4.66 to 4.97× larger (m = T instead of 200).
  - pyspi's decisions do not change: pair 0-1 is wrong and pair 2-3 is right either way.
  - The canonical separation widens with the full sample. For pair 0-1 it goes from 0.41 vs 2.38 to 0.27 vs 8.88.
- **T-dependent scale**: the ρ = 0.8 pair of bivariate_gaussian gives 3.24 at T64 (m = 64), 8.81 at T256 and
  8.45 at T1024 (m = 200).
- **random_state**: `|random-state` gives 0.0 on all 12 pairs. This is a pyspi-vs-pyspi determinism check, labelled
  same-lineage.
- **Global patch**: `|patch-global` flags are [1, 1, 1].
- **Labels**: `|labels` flags are [1, 0, 1]. Both are static code-property probes (pyspi introspection against a
  hard-coded expected flag vector), labelled same-lineage. The class labels contain 'linear', and only config.yaml says 'nonlinear'.
- **SciPy 1.18 drift**: the SciPy 1.18 change to anm was already documented in `DEPENDENCY_MIGRATION.md` (SciPy row,
  1.25e-6) and `EEG_OPTIMIZATION_NOTES.md` §review. `causal.anm|anm|256|pipeline|scipy118-drift` measures the mechanism:
  - sklearn 1.9.1 under scipy 1.18.1 moves anm_T256 by a maximum of 1.95e-6 relative, with log theta changing by 4.1e-6.
  - The replica at pyspi's theta under scipy 1.18.1 gives 6.6e-16.
  - A one-ulp change of the fitted constant moves the score by up to 2.2e-6.
  - So the drift arises mainly from the optimiser end point. LAPACK roundoff (the 7.5e-8 case above) and BLAS threading
    contribute at the same order, because the residual is the 1e-10-scale difference K(K+αI)⁻¹y − y, which keeps only
    ~1e-6 relative precision.
- **Scale and detrend**:
  - On raw scaled_offset (kernel path), the independent fit misses by 4.5e-4 (fail, optimiser end point). pyspi itself
    moves 15% relative to the normalised bivariate_gaussian.
  - trending with detrend passes at 3.0e-6 and moves 17% relative to the trend-free draw.

Candidates:
- Confirmed:
  - GP collapse leading to an HSIC-like dependence score.
  - Wrong causal ranking.
  - The drift mechanism.
  - random_state is a no-op.
  - The 'linear' label.
  - The global monkeypatch.
  - The T-dependent scale.
- None of these is in KNOWN_ISSUES. Only the drift itself is already documented, and the new part is its mechanism.

### causal.cds: verdict variant, score 6 (rubric 6: library formula deviates from both the supplement and canonical in an interpretable way)

- **Exact**: `causal.cds|*|exact` passes 23/23 with max abs 2.8e-17, on the kernel path at T64–4096, the pipeline path
  and pipeline-detrend.
- **Direction on anm**: agreement 1.0 at T256/1024/4096. D01 is 0.0215/0.0865/0.0963 and D23 is
  0.0020/0.0268/0.0264. Both lagged datasets also recover x → y.
- **Zero predicate**: 6/6 runs pass.
  - anm_T64: 12 exact zeros, equal to the 12 count-part entries. Qualifying bins per process are [0, 1, 1, 1].
  - iid_gaussian_T64: 8/8. gauss_ar1_pair_T64: 2/2.
  - narrowband_f0-0.25_T64: one exact zero with no count-part entry (identical pmfs).
  - Inline spike input (24 excursions, so ≤ 12 samples on each side of the bulk): A[0,1] = 0.0 at T256 and at T1024
    with a single qualifying bin. The zero is therefore not confined to short series.
- **Invariance and detrend**: scaled_offset (raw) equals the independent reference evaluated on the unscaled draw to
  1.4e-17 (independent-impl; the affine-invariance argument only sets the expectation). Detrend runs are exact.
- **Paper variant** (`|paper-variant`: sfactor 3, ddof 1):
  - Values change by a maximum of 0.36 to 0.58 relative. sfactor alone accounts for 0.39 to 0.50, ddof alone for
    0.08 to 0.12.
  - 3 of 4 runs fail the "no ordering flip" criterion, with 2 or 3 of 6 pairs flipped. Every flip is on an independent
    pair or on the symmetric circle pair (2,3) of nonlinear_dependence. The causal anm pairs keep their ordering.
  - This is a real parameter sensitivity, not a code error.
- **Candidates**:
  - Zero degeneracy: confirmed, and new.
  - Wording: confirmed.
  - ffactor 2 vs 3: inconclusive. The effect is quantified, but the provenance in the original code is not established.
- **Score**: 6, not 7. Anchor 7 needs a match with the supplement variant, and the confirmed wording candidate says the
  supplement does not describe the computed quantity. cdt's formula also departs from Fonollosa's stated parameters
  (sfactor 3, sample variance) by 0.36–0.58 relative with ordering flips on non-causal pairs, and returns an exact 0 when
  at most one x-bin qualifies. Deviating from both in an interpretable way is anchor 6.

### causal.reci: verdict variant, score 7 (rubric 7: matches the supplement variant, value-changing difference from canonical)

- **Exact**: 24/24 runs, max abs 5.6e-17, covering the kernel path at T64–4096, the pipeline path and pipeline-detrend
  (the EEG order).
- **Invariance and detrend**: invariance is exact (6.9e-18; independent reci_ref on the unscaled draw, independent-impl). Detrending moves values by ≤ 1.0e-3.
- **Direction on anm** (canonical MON-family, held-out reference; `reci_canonical` was written from recollection of
  Bloebaum et al. 2018 and not re-checked against the paper, so "canonical" below means this recalled form):
  - T256 and T4096 pass with agreement 1.0.
  - `causal.reci|anm|1024|pipeline|direction` fails with agreement 0.5. pyspi gives D23 = −0.00026 on the tanh pair, the
    same as the cdt-form reference. The canonical RECI gives +0.0007.
  - The cubic pair is right at every T (D01 = 0.0283/0.0291/0.0285).
- **Lagged datasets**: var2_bivariate recovers x → y. logistic_maps_chaotic does not, and the reference agrees with pyspi.
- **Candidate**: the fixed odd monomial is confirmed. It is the definition documented in supplement §1.3.4, so it is
  not counted as a new finding.

### causal.igci: verdict matches, score 9 (rubric 9 by analogy: sign of an antisymmetric quantity, documented only in cdt)

- **Exact**: 22/22 runs, max abs 3.0e-14.
- **Antisymmetry**: 5/5 runs have max|A + A.T| = 0.0 bit for bit. The value is −A.T of pyspi's own matrix, so this is a
  pyspi-vs-pyspi self-consistency check, labelled same-lineage; the score rests on the 22 exact independent-impl runs only (the cubic analytic check is statistical and does not decide it).
- **Analytic check** (`|cubic-analytic`, X ~ U(0,1), Y = X³, 20 seeds):
  - pyspi means are 0.8746/0.8742/0.8741 at T256/1024/4096, against −C^G = 0.88320. |z| is 0.36/0.93/1.51, all within 2 se.
  - The single-seed values in the trace (0.77 at T256) were seed noise.
  - These three runs do not demonstrate convergence: the 20-seed means do not move towards 0.8832 with T and |z| grows,
    so the pass rests on the 20-seed se. A 200-seed probe of igci_ref (seeds 1000..1199;
    `cdt_pairwise_check/igci_bias200.py`, igci_ref equals pyspi to 3.0e-14 in the exact runs) gives mean − 0.88320 =
    −0.0183 (z −2.65), −0.0104 (z −2.70), −0.0051 (z −2.76), −0.0024 (z −2.69) and −0.0002 (z −0.41) at
    T = 256/1024/4096/16384/65536. This is a negative finite-sample bias of the 1-spacing entropy estimator that shrinks
    with T, not a formula difference (README: finite-sample bias is not incorrectness).
  - The positive sign establishes that positive means i → j, which is −C_{i→j}.
- **Raw two-array path** (`|raw-bivariate`): the gap of 7.5885e-9 is reproduced to 1.5e-14 by the reference on
  (zscore(first), raw second). With both arrays z-scored the gap is 0.0, which confirms the `preproc.two_array_api`
  mechanism.
- **Direction on anm**: agreement is 0.5 at every T. The noisy cubic pair is reversed (A01 = −0.200/−0.088/−0.151). The
  reference gives identical values, so this is a method limit, not an implementation deviation. The tanh pair is right.
- **Lagged datasets**: both recover x → y (A01 = 0.021 and 0.946).
- **Range**: values are signed, −0.52..+0.52 on continuous data. On logistic_maps_T1024 they reach ±6.35, because the
  period-4 y has tied values and zero spacings are skipped while the divisor stays m−1. The `units_and_range` field
  was corrected accordingly.
- **Candidates**: all three are confirmed: 'unsigned' label on a signed output (`|labels` flags [1, 0, 1]), the
  undocumented sign convention, and the raw-path gap. A fourth (tie counts on quantised data, below) was added in wf4b-2 and
  is confirmed by scratch probe.
- **check_method** is independent-impl (corrected from analytic in wf4b-2): the score rests on the 22 exact runs against
  `_kernels_distance.igci_ref` (no cdt/sklearn), labelled independent-impl in results. The analytic cubic check is statistical
  and its 20-seed means do not approach 0.8832, as said above.
- **Score mapping.** Anchor 9 lists "sign of a symmetric quantity" as a trivial documented convention. Here the quantity is
  antisymmetric, so negating it is a transpose (A[i,j] = −C_{i→j} = C_{j→i}); the clause is applied by analogy. The orientation
  agrees with pyspi's directed convention (A[i,j] is evaluated with i as the source argument, and positive means i → j). The
  convention is documented only in the delegate, cdt's docstring (`IGCI.py:138`, ">0 if a->b"), not in pyspi's supplement,
  config or GitBook.
- **Tie handling.** Zero spacings are skipped with the divisor kept at m−1 (cdt `eval_entropy`, `IGCI.py:42-56`); eq. (12) is
  undefined at log 0 and the paper is silent. The trace's remark that this is "the same as the original Matlab-style
  estimator" had no cited source and is withdrawn (no original Matlab code was consulted). No exact run covers ties: every
  exact-run dataset is continuous, and logistic_maps (±6.35) is used only for range and antisymmetry.
- **Scope on quantised recordings (wf4b-2 scratch probes `wf4b2_high-optimiser-other/p_igci.py`, `p_igci_ties.py`).** On an
  EEG window (001_Baseline.EDF, six EEG channels, mid-recording, ADC step 2.66e-7) pyspi reproduces an own eq. (12) with the
  skip/(m−1) rule to 8.4e-15 (T512), but the value is mainly a tie-count artefact:
  - 179-244 zero spacings per channel at T512 (578-683 at T1024);
  - the alternative divisor (number of non-zero spacings) changes A by up to 0.488 when max|A| is 0.688;
  - A[i,j] correlates 0.993 (T512) and 0.998 (T1024) with the difference in zero-spacing counts of the two channels;
  - sub-quantum uniform dither, which keeps the signal and breaks the ties, lowers max|A| to 0.236 / 0.169 and changes the
    sign of A on 80% (T512) and 47% (T1024) of pairs.

  causal.igci is not in the retained 185-metric EEG set. An exact manifest run on tied data is still missing (deferred: it
  needs a manifest and reference-script change). `definition_under_review` stays false: eq. (12) is defined for continuous
  data and undefined, not contested, on tied data; both tie rules are recorded in the record's tie candidate.

### Fails in results (17)

| Unit | Failing runs | Reason |
|---|---|---|
| anm | direction T256, p01-direction T1024 | Finding |
| anm | rawhsic: contaminated T64/T256, nonlinear_dependence T64 | The HSIC equivalence is approximate |
| anm | theta and replica-ownfit on iid_gaussian T256 | sklearn's end point is suboptimal |
| anm | replica-ownfit bivariate_gaussian T256 (1.05e-5) and scale scaled_offset T256 | Optimiser end point |
| anm | replica-theta nonlinear_dependence T256 | Cross-environment LAPACK; the project-venv replica passes |
| cds | paper-variant ×3 | Parameter sensitivity |
| reci | direction T1024 | Fixed cubic on the tanh pair |
| igci | direction ×3 | Method limit, shared by the reference |

### Corrections after adversarial review

- `refs/cdt_pairwise.py` now labels the pyspi-vs-pyspi runs as same-lineage: igci `|antisym` (value = −A.T of pyspi's
  own matrix) and anm `|random-state` (determinism of pyspi's score). The static code-property probes are labelled the
  same way: anm `|patch-global` and `|labels`, and igci `|labels`. The cds and reci `|invariance` runs are labelled
  independent-impl, because their value is the independent replica on the unscaled draw. None of these runs carries a
  score: each unit's score rests on its independent-impl runs (for igci the 22 exact runs; its cubic analytic check is statistical and does not decide the score).
- The family was re-run with `--force` after these changes. Pass/fail is unchanged (153 pass, 17 fail), and every number
  quoted above is the same in the new results.json. The pyspi timings changed (anm T1024 pair matrix 11 s, anm total
  218 s).
- The anm rawhsic count is 10/13, not 12/15 as previously quoted.
- cds score 7 → 6 (see the cds section).
- igci: the cubic-analytic runs are now described as within 2 se with a finite-sample bias, not as showing convergence.
- anm and reci: the "canonical" direction references are recalled forms that were not re-checked against their sources.
  The records say so.

## Open questions: answered

1. **CDS ffactor = 2 vs sfactor = 3 in the original jarfo code.** Not answered. It needs the original `features.py`,
   which is not available offline. The checks bound the stakes:
   - sfactor 3 changes CDS values by 39–50% relative.
   - It flips orderings only on independent or symmetric pairs, never on the causal anm pairs
     (`causal.cds|*|paper-variant`).
2. **anm collapse on real EEG, and the intended semantics.**
   - The collapse appeared on every synthetic dataset tested (`|collapse` 7/7), at T up to 1024; the length scale sat at the 1e-5
     bound for 9 of 12 pairs at anm_T64 and for 3 of 4 pairs at anm_T1024. It follows from the model specification (no noise term, alpha =
     1e-10) for any data without exactly duplicated x, so EEG windows are expected to behave the same way. Not run on
     EEG here: a one-window probe of the patient data would settle it.
   - The resulting semantics is m·HSIC_b(x, (K+αI)⁻¹y), close to plain HSIC of the raw pair.
   - Adding a WhiteKernel recovers the canonical behaviour, with the designed direction on all anm checks. That is a
     definition change and needs a decision. This audit does not make it.
3. **Does the roundoff-level residual explain the SciPy 1.18 drift?** Yes, as a mechanism. The residual carries only
   ~1e-6 relative precision, so any last-bit change moves the score by ~1e-6:
   - a one-ulp change of c: 2.2e-6;
   - sklearn under scipy 1.18.1: 1.95e-6, against the documented 1.25e-6;
   - BLAS thread count: ~1e-6;
   - LAPACK build: 7.5e-8 on one pair.
   The drift itself was already documented; this check identifies its mechanism. Run:
   `causal.anm|anm|256|pipeline|scipy118-drift`.
4. **Should cds return NaN instead of 0 when ≤ 1 x-bin qualifies?** That is a design decision. The evidence is that the
   0 is exact, occurs at any T (spike input at T1024), and is indistinguishable from the minimum possible score. It
   reads as maximal causal evidence.
5. **Relabel igci as signed and anm as nonlinear?** The evidence supports both:
   - igci values are signed and exactly antisymmetric (`|antisym`, `|labels`).
   - The anm class label 'linear' contradicts config.yaml and the supplement (`|labels`).
   Changing labels is a metadata decision (REVIEW M9).
6. **The 5e-9 igci gap.** This was resolved at trace stage and is now confirmed numerically (`|raw-bivariate`: 7.5885e-9
   reproduced to 1.5e-14; 0.0 when both arrays are z-scored).
7. **Cost of anm at T = 4096.** In the run stored in `out/pyspi/index/cdt_pairwise.json`, anm took 1.28–4.83 s per P = 4 run
   at T = 256 (23 runs, median 1.73 s) and 9.7–12.3 s for each of the 14 two-process runs (2 ordered pairs) at T = 1024; an
   earlier run took up to 17 s for a two-process T = 1024 run (not stored in the index or results files). That is O(T³) per
   likelihood evaluation, so roughly 64× per pair at T = 4096, about 5–7 minutes per pair. It was not run at T = 4096. The full
   anm budget was 218 s of pyspi time in the stored run and 243 s in the earlier one (not stored in the index or results files).

## Optimisation

Workflow 3 (profile-based search for hotspots and avenues; nothing in `pyspi/` was changed). Walls are from `audit/profiles/summary.json` (quiet machine, OMP_NUM_THREADS=1); percentages are shares of the profiled `compute_spi` frame in the named pstats (cProfile inflates walls). Every expected speedup is the unit's per-window cost at 118x1024, bounded by Amdahl from the profiled share the avenue removes.

| unit | 8x1024 (fresh / family-warm) | 32x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|
| causal.anm | 252.1 s / 253.3 s (8x256: 6.23 s) | infeasible | 62,142 s (projected) | extrapolated-pairs-from-8 (T_exponent 2.67) |
| causal.cds | 0.92 s / 0.93 s | 16.6 s | 231.5 s (projected) | extrapolated-pairs-from-32 (118 skipped-budget) |
| causal.reci | 0.087 s / 0.088 s | 1.53 s | 21.5 s | measured |
| causal.igci | 0.110 s / 0.111 s | 1.93 s | 26.9 s | measured |

Shared versus incremental cost: the four classes share nothing (no cache; each identifier runs its own cdt call per ordered pair), so family-warm equals fresh and the family costs about 62,420 s per window at 118x1024, of which anm is 99.6%. Only reci is in the retained EEG selection: 21.5 s per window at 118 channels (one EEG identifier, measured; the user's historical figure is 12.2 s with an unknown channel count).

### causal.anm
Profile `audit/profiles/causal.anm__8x256.pstats` (8x256; the 8x1024 cProfile run timed out), opaque share 0.241 (lower bound). `sklearn/gaussian_process/_gpr.py:log_marginal_likelihood` is 91.8% cumulative over 1,861 evaluations for 56 pairs (about 33 per pair): `scipy/linalg/_decomp_cholesky.py:_cho_solve` 27.5% (K^-1 = cho_solve(L, eye) for the gradient), the RBF kernel call `kernels.py:__call__` (line 1530) 22.8% inside the Product kernel 33.2%, `_cholesky` 12.6%, `c_einsum` 8.4%. The cdt HSIC on the 200-point subsample is 1.9%. Complexity O(P^2 n_eval T^3); T_exponent 2.67 between 8x256 and 8x1024.
- gpfit-2 (shared with misc.gpfit.*): process pool over ordered pairs, one BLAS thread per worker; expected 10-30x per-window latency on its own (at most ~4x inside a window worker). The ceiling is the 32 physical cores of the Threadripper 3970X (64 logical CPUs; SMT adds little to Cholesky-bound fits); there is no core-hour saving. Window-level parallelism (xc-window-parallel) competes for the same cores and gives the same throughput without code in the unit. The two are alternatives, not factors that multiply, so the record does not list it in depends_on; inside one of the ~8 memory-bounded window workers (about 4 physical cores each) the pool lowers the unit's latency by at most about 4x. Not measured: complexity and profile only. Output risk unverified-bit-identical (independent deterministic fits; the ~1e-6 BLAS-thread sensitivity recorded in the checks is unchanged with one thread per worker).

### causal.cds
Profile `audit/profiles/causal.cds__8x1024.pstats`, opaque share 0.191. `cdt/causality/pairwise/CDS.py:cds_score` 100.0%; `discretized_sequence` 69.8%, of which `pandas/core/series.py:__setitem__` 53.2%: the boolean-mask clipping `x[x > vmax] = vmax` on a pandas Series first tries the mask as a label, and `pandas/core/indexes/base.py:get_loc` builds a KeyError message through `Series.__repr__` (29.0% of the call) before falling back. `count_unique` (Python `set()` over a Series, recomputed for y in every qualifying x-bin) is 12.2% self.
- cdt_pairwise-2: numpy/numba re-implementation of cds_score with per-channel discretisation computed once per Data; expected 100-1000x. Not measured (profile share plus complexity). Output risk unverified-within-tolerance: pandas reductions become numpy reductions (different summation order), and a last-bit change at an np.round half-way point would move one sample to a neighbouring bin.

### causal.reci
Profile `audit/profiles/causal.reci__8x1024.pstats`, opaque share 0.204. `cdt/.../RECI.py:b_fit_score` 99.7%; sklearn input validation dominates: `sklearn/utils/validation.py:check_array` 44.9% (840 calls, 15 per pair), `preprocessing/_data.py:minmax_scale` 37.8% (two per pair), `LinearRegression.fit` 20.6%, `mean_squared_error` 10.2%. T_exponent 0.01: the per-call overhead (about 1.5 ms per pair) hides the O(T) arithmetic.
- cdt_pairwise-1 (seed confirmed): min-max scale each channel once and compute all ordered-pair OLS fits on [1, x^3] from one P x P cross-moment matrix. Micro-benchmark `audit/profiles/micro/opp_cdt_pairwise-1.py` / `.json`: 22.1 s -> 6.1 ms at 118x1024 (3621x, one-call baseline), 682x at 32x1024, 64x at 8x1024 (candidate time includes building a fresh Data); max_abs_diff 8.3e-17, max_rel_diff 4.0e-15, NaN masks equal at 118x1024; max_rel 1.9e-15 against the 7x100 golden snapshot. The candidate matches the baseline within floating-point tolerance (output_risk within-tolerance); this says nothing about whether the quantity itself is the canonical RECI. The seed target of about 12 s -> 0.05 s per window is met with margin at the measured 21.5 s. Like-for-like figure (Workflow 4c, `shapes["118x1024_split"]`, written by `--split118`): the summary.json walls time `compute_spi` on a prebuilt Data, so the comparable candidate cost is the math alone, 3.10 ms (Data construction 3.19 ms on its own), 6983x against the family-warm ranking basis of 21.66 s (extrapolated-pairs-from-8; the fresh 118x1024 measurement, 21.52 s, is within 0.7%). Counting Data construction in the candidate gives 3547x (Workflow 3 run) or 3122x (4c run), a figure that moves with Data-construction time; the record keeps ~3500x as a conservative lower bound, and the rank does not depend on the choice.

### causal.igci
Profile `audit/profiles/causal.igci__8x1024.pstats`, opaque share 0.219. `IGCI.py:predict_proba` 99.7%: `eval_entropy` 54.4% (41.7% self in the Python loop over sorted spacings, `builtins.sorted` 12.6%) and the module-level `StandardScaler.fit_transform` 44.1%.
- cdt_pairwise-3: A[i, j] = S(z_i) - S(z_j) (record: exactly antisymmetric), so evaluate the scaler and the entropy once per channel and fill an outer difference: 27,612 evaluations per window at P=118 become 118; expected ~100x (Amdahl bound 333x). Not measured. Output risk unverified-bit-identical (same functions on the same vectors, only fewer calls).

### Refuted or narrowed seeds
- "cds/igci numba for their Python loops": for igci the loop (eval_entropy self 41.7%) is real but the larger win is removing the pair loop altogether (cdt_pairwise-3); numba alone would leave the StandardScaler's 44.1% per pair. For cds the Python-level cost is pandas indexing, not an arithmetic loop, so the avenue is a numpy/numba rewrite with per-channel caching (cdt_pairwise-2), not a numba port of the existing loop.
- anm: only pair-level parallelism was considered, as briefed; a custom LML objective (gpfit-3) would also apply but removes at most about 25% of the GP time.
