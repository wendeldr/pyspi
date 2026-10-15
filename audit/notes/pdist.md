# pdist family: define, trace and check

Units: `distance.pdist.{euclidean, cityblock, cosine, chebyshev, canberra, braycurtis}` (identifiers `pdist_<metric>`), one class `PairwiseDistance` in `pyspi/statistics/distance.py:28-40`.

Versions in this checkout: scikit-learn 1.9.1, SciPy 1.17.1, NumPy 2.5.4. Probe scripts are in the session scratchpad (`pdist/probe1.py` to `probe4.py`). Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths. The sections before `## Checks` are the trace stage: statements there based on probes give the probe value, a small read-only confirmation of what the code does, and everything else there was a hypothesis for the check stage. The numeric check results, verdicts and scores are in `## Checks` and in the records. Those hypotheses and the candidates are resolved in the records' `candidate_resolutions` and in `## Checks`.

## What pyspi documents

- **Supplement:** pdist is **not** in the supplement. Supplement §1.2 "Distance similarity (26 SPIs)", p.5 (heading at line 209), numbers SPIs 22–47: dcorr, dcorrx, mgc, mgcx, hsic, hhg, dtw, lcss, softdtw and bary. None of them is pdist.
- **GitBook:** the only documentation is the "Pairwise Distance" section of `distance-similarity.md` (lines 297–321). It says: "Pairwise distance between time-series, computed using _sklearn's_ pairwise distance function using the following distance metrics: Euclidean, Cityblock, Cosine, Chebyshev, Canberra, Braycurtis". The keywords are "undirected, nonlinear, unsigned, unordered". The page gives no formula and does not mention preprocessing or normalisation. The glossary (`glossary-of-terms.md`) does not define "unordered".

## Shared pipeline (all six units)

1. `pyspi/calculator.py:277` (HEAD a1f20c7) builds `Data(..., normalise=self._normalise, detrend=self._detrend)`. The `__init__` defaults at `calculator.py:48-50` are `detrend=False, normalise=True`.
2. `pyspi/data.py:184-195` optionally runs `scipy.signal.detrend(axis=1)`, then `zscore(data, axis=1, ddof=1)` on each process. After this step each row has mean 0 and `sum z_t^2 = T-1`.
3. `pyspi/calculator.py:329-332` (HEAD a1f20c7) calls `multivariate(dataset)` (:329), then `np.fill_diagonal(S, nan)` (:332).
4. `pyspi/statistics/distance.py:38-40` runs `pairwise_distances(data.to_numpy(squeeze=True), metric=self._metric)` on an (M, T) array. Each process is one vector in R^T, and the distance is taken between rows. This method overrides `Undirected.multivariate` (`base.py:132-138`), so the generic triangle mirror never runs (REVIEW M1).
5. `sklearn/metrics/pairwise.py:2309` is `pairwise_distances`. There are two routes:
   - **sklearn-native** (euclidean, cityblock, cosine): `:2463-2464` looks the metric up in `PAIRWISE_DISTANCE_FUNCTIONS` (`:1919-1931`). Then `:2494` calls `_parallel_pairwise`, which sets `Y = X` at `:1977-1978` and calls `func(X, X)` at `:1981`.
   - **SciPy** (chebyshev, canberra, braycurtis): `:2476-2484` runs `check_pairwise_arrays`, which sets X = Y at `:175`. Then `:2490-2491` returns `squareform(scipy.spatial.distance.pdist(X, metric))`. In SciPy, `distance.py:1816` is `pdist`, `:2133-2138` resolves the metric alias, and the work is done by the compiled `_distance_pybind.pdist_<metric>`.

In every unit, `A[i,j] = d(z_i, z_j)` for z-scored rows i and j. The metric is symmetric, so A carries no direction. The diagonal is NaN after `Calculator.compute`.

Probe on a 64×1024 correlated dataset: every scipy-route metric and cityblock is bit-identical to the SciPy `pdist` loop and exactly symmetric. Cosine differs from SciPy by at most 4.4e-16 and is exactly symmetric. Euclidean has `max|A - A.T| = 7.1e-15` (REVIEW M2).

## Per unit

### distance.pdist.euclidean

- **Canonical:** `d = sqrt(sum_t (x_t - y_t)^2)` (Deza & Deza 2016, *Encyclopedia of Distances*, "Euclidean metric").
- **Implemented:** `euclidean_distances` → `_euclidean_distances`, which computes `D^2 = ||x||^2 + ||y||^2 - 2 x·y`:
  - row norms at `pairwise.py:389`;
  - `YY = XX.T` at `:393-394`;
  - the BLAS product at `:409-411`;
  - clamp at 0 at `:413-416`;
  - true diagonal set to 0 at `:420-421`;
  - square root at `:426`.
- **On z-scored input** the value is exactly `sqrt(2 (T-1) (1 - r))`, with r the Pearson correlation. The probe deviation was 0.0. This confirms the briefing hypothesis with the (T-1) factor, which follows from ddof=1.
- **Deviations:**
  - The formula is the same as the canonical one.
  - (a) Numerical: the dot-product expansion cancels badly for near-identical rows. In `shared_envelope`, x2 is an exact copy of x0. `pdist_euclidean[0,2]` is 1.19e-7 at T=64, 4.13e-7 at T=256, 0.0 at T=1024 and 9.54e-7 at T=4096, while the direct difference gives 0.0. The relative error reaches 4.5e-2 when 1−r ≈ 5e-15. *Candidate:* this is related to M2, but M2 only documents last-bit reciprocal asymmetry.
  - (b) Because of z-scoring, this is a monotone function of r. *Candidate:* the "nonlinear" label is not supported.
  - (c) The value scales as sqrt(T−1), so it is not comparable across window lengths. It is also a dissimilarity: larger values mean less coupling.
- **Already documented:** REVIEW M1, M2.
- **Proposed checks:**
  - numpy from-scratch kernel (tolerance);
  - the analytic identity `sqrt(2(T-1)(1-r))` on bivariate_gaussian, iid_gaussian and sparse_precision;
  - the population value `sqrt(2(T-1)(1-rho))` (statistical);
  - shared_envelope x0 vs x2: expected 0 by definition (exact);
  - symmetry and direction on var2_bivariate and gauss_ar1_pair;
  - scaled_offset invariance;
  - trending with pipeline-detrend;
  - time-permutation invariance;
  - delayed_noise (qualitative).

### distance.pdist.cityblock

- **Canonical:** `d = sum_t |x_t - y_t|` (Deza & Deza 2016, "Manhattan metric").
- **Implemented:** `manhattan_distances` (`pairwise.py:1055`, numpy branch `:1109`) → `scipy.spatial.distance.cdist(X, X, 'cityblock')`, which reaches `_distance_pybind.cdist_cityblock` through `distance.py:2922-2926` and `:1681-1686`.
- **Deviations:**
  - None in the formula. The probe matched SciPy pdist exactly.
  - *Candidate (interpretation):* the value grows linearly with T. For Gaussian pairs, `cityblock/T → 2 sqrt((1-rho)/pi)`; the probe at T=4e5 gave 0.5045 vs 0.5046 for rho=0.8. On Gaussian data it is asymptotically a function of rho only.
- **Already documented:** M1.
- **Proposed checks:** the common set, plus the population value `2 sqrt((1-rho)/pi)` (statistical).

### distance.pdist.cosine

- **Canonical:** `1 - x·y / (||x|| ||y||)` (Salton & McGill 1983 for cosine similarity; Deza & Deza 2016).
- **Implemented:** `cosine_distances` (`pairwise.py:1136`):
  - `cosine_similarity` normalises the rows and takes the dot product (`:1703`, `:1750`, `:1756`);
  - then `clip(1 - S, 0, 2)` at `:1174-1177`;
  - the diagonal is set to 0 at `:1178-1181`.
- **On z-scored rows** the value equals `1 - r`. The probe deviation was 7.8e-16, which confirms the briefing hypothesis.
- **Deviations:**
  - None in the formula.
  - *Candidate:* the measure duplicates Pearson correlation up to an affine map, so "nonlinear" is not supported.
  - *Candidate:* it is non-negative but still sign-sensitive: values in (1, 2] mean negative correlation, and y = −x gives exactly 2.0.
  - With `normalise=False` it becomes the uncentred cosine and depends on offsets. The probe changed from 0.379 to 0.710 under x → 3x+5.
- **Already documented:** M1.
- **Proposed checks:** the common set, plus the analytic identity `1 - r` (tolerance 1e-12) on bivariate_gaussian, iid_gaussian, sparse_precision and contaminated.

### distance.pdist.chebyshev

- **Canonical:** `max_t |x_t - y_t|` (Deza & Deza 2016, "Chebyshev metric").
- **Implemented:** SciPy `pdist` 'chebyshev' (`distance.py:1674-1679`; Python reference at `:1041`) followed by `squareform` (`:2154`).
- **Deviations:**
  - None in the formula. The probe matched exactly.
  - *Candidate (interpretation):* this is a single-sample extreme statistic. Its expectation grows with T (the expected maximum of T half-normals), and it is dominated by outliers. The range is bounded by `2(T-1)/sqrt(T)` (Samuelson bound for ddof=1 z-scores).
- **Already documented:** M1.
- **Proposed checks:**
  - the common set;
  - the population expectation `integral_0^inf (1 - erf(t/(2 sqrt(1-rho)))^T) dt` checked against a Monte-Carlo spread (statistical);
  - contaminated vs bivariate_gaussian pair (2,3) (qualitative).

### distance.pdist.canberra

- **Canonical:** `sum_t |x_t - y_t| / (|x_t| + |y_t|)`, with 0/0 := 0.
  - Lance & Williams 1966, *Computer J.* 9:60–64, and 1967, *Aust. Comput. J.* 1:15–20, originally for non-negative data.
  - Deza & Deza 2016 for the |x|+|y| form.
- **Implemented:** SciPy `pdist` 'canberra' (`distance.py:1667-1672`; Python reference at `:1150`; the 0/0 rule is documented at `:1178`), then `squareform`. Probe: [0,1,2] vs [0,1,3] gives 0.2, so 0/0 counts as 0.
- **Deviations:**
  - The formula matches the generalised form.
  - *Candidate (domain):* on z-scored signed data, every opposite-sign sample adds exactly 1 whatever its amplitude, and samples near zero get maximal weight. y = −x gives exactly T (256.0 at T=256).
  - For independent Gaussian pairs, `canberra/T → 1/2 + ln2/pi = 0.72064` (the probe at T=4e5 gave 0.71987). For general rho, the population value is a 1-D quadrature over the bivariate-normal angle density `sqrt(1-rho^2)/(2 pi (1 - rho sin 2θ))`. The probe agreed to about 2e-3 at T=4e5 for rho in {0, 0.3, 0.8, −0.5}.
- **Already documented:** M1.
- **Proposed checks:**
  - the common set;
  - the quadrature population value (statistical);
  - edge inputs (x, −x) = T and exact zeros (exact).

### distance.pdist.braycurtis

- **Canonical:** `sum |x - y| / sum (x + y)` for non-negative abundances, with range [0,1].
  - Bray & Curtis 1957, *Ecol. Monogr.* 27:325–349.
  - Legendre & Legendre 2012, coefficient D14.
- **Implemented:** SciPy `pdist` 'braycurtis' (`distance.py:1660-1665`; Python reference at `:1102`) computes `sum|x-y| / sum|x+y|`. The SciPy docstring itself says the result is in [0,1] only "if all coordinates are positive".
- **Deviations:**
  - *Candidate (domain):* on z-scored signed data the range is [0, +inf]. Probe values at T=4096 were 1.75 for rho=−0.5, 4.39 for −0.9, 14.2 for −0.99 and 45.1 for −0.999. y = −x gives +inf with no warning recorded.
  - Asymptotically `BC → sqrt((1-rho)/(1+rho))`. The probe at T=4e5 gave 0.33337 vs 0.33333 for rho=0.8. This is a monotone function of rho for Gaussian pairs.
  - *Probed through Calculator:* a configfile with the braycurtis and canberra configs, on data `[x, −x, 2x+1]` (x iid N(0,1), T=256), runs `compute()` to completion. The summary counts both SPIs as successful with 0 NaNs, no divide warning is raised, and `calc.table['pdist_braycurtis']` holds +inf at (0,1) and (1,0).
  - The z-scored affine copies are not exact (`z0 − z2` and `z1 + z2` have max abs 4.4e-16). So (x, 2x+1) gives 1.77e-17 instead of 0, and (−x, 2x+1) gives 5.67e16 instead of +inf. Rounding in z-scoring decides between ~1e16 and inf.
  - Propagation of inf into downstream analyses has not been probed.
- **Already documented:** M1.
- **Proposed checks:**
  - the common set;
  - the population value `sqrt((1-rho)/(1+rho))` (statistical);
  - the (x, −x) edge case, expected +inf (exact).

## Common proposed checks (in every record)

All references are numpy or from-scratch, or use scipy.integrate/special primitives. None imports sklearn or scipy.spatial.

1. **Kernel:** numpy loop formula vs `multivariate(Data(x, normalise=False))`, rel 1e-12, on iid_gaussian, bivariate_gaussian, nonlinear_dependence and anm.
2. **Direction convention:** on var2_bivariate, gauss_ar1_pair and iid_gaussian, `A == A.T`. Bit-identical except euclidean, where only last-bit differences are expected (M2).
3. **Invariance to positive scale and offset:** scaled_offset vs bivariate_gaussian through the pipeline.
4. **Detrend:** trending vs bivariate_gaussian through pipeline-detrend.
5. **Time-permutation invariance** ("unordered").
6. **delayed_noise** (qualitative): lagged copies look like independent noise, because only lag 0 is used.

## Open questions

- Should the "nonlinear" label stay for euclidean and cosine? On default input they are exact functions of Pearson r. Cityblock and Bray–Curtis are asymptotically functions of rho on Gaussian data. This is a labelling-policy question.
- What does "unordered" mean in pyspi? It is not in the glossary. The pdist metrics are contemporaneous (lag-0 pairing) and invariant to joint time permutation.
- Should pdist outputs carry a "dissimilarity" orientation flag? Larger values mean weaker coupling, unlike most SPIs. Users who combine SPIs may need it.
- Should pyspi document, or guard against, Bray–Curtis returning inf and leaving [0,1] on signed data, and Canberra on signed data? Both metrics were designed for non-negative data.
- Is the ~1e-7 absolute error of `pdist_euclidean` for identical or near-identical channels acceptable, given that a direct-difference implementation gives 0? This matters mainly for duplicated channels and for storage-exactness comparisons (see M2).
- Euclidean, cityblock, canberra and chebyshev depend on T. Should the docs warn that values are not comparable across window lengths?
- The supplement and GitBook give no formula. Should the audit treat the SciPy definitions (especially the |x+y| form of Bray–Curtis) as the documented definition?

## Checks

Check stage. Manifest: `audit/checks/manifest/pdist.json`, with 256 runs, about 41 for each metric. References: `audit/checks/refs/pdist.py`, run in the comparison venv. Probes: `audit/checks/probes/pdist_derived.py`. Results: `audit/checks/out/results/pdist.json`, with **252 pass, 4 fail, 0 without reference**. All 4 failures are euclidean runs the trace stage predicted (dup ×3, near ×1). Run IDs are written `distance.pdist.<metric>|<dataset>|<T>|<path>|<tag>`. Every pyspi run took less than 0.03 s.

### What was compared with what (all six metrics)

| tag | pyspi side | reference | criterion |
|---|---|---|---|
| `def` | kernel (raw rows): bivariate_gaussian T 64/1024/4096; iid_gaussian, nonlinear_dependence, anm T 1024. pipeline: bivariate_gaussian T 64–4096; scaled_offset, nonlinear_dependence, contaminated, sparse_precision T 1024. pipeline-detrend: bivariate_gaussian | `_kernels_basic_misc.pdist_metric`, a from-definition numpy implementation, on the rows pyspi receives. For pipeline paths the reference applies its own ddof=1 z-score; for pipeline-detrend it first does its own lstsq linear detrend. Independent: no sklearn and no scipy.spatial | exact, rtol = atol = 1e-12 |
| `cdist` | pipeline bivariate_gaussian 1024 | `scipy.spatial.distance.cdist` (scipy 1.18.1 in the comparison venv). **Same-lineage**, recorded only as such | exact 1e-12 |
| `ident` (euclidean, cosine) | pipeline bivariate_gaussian, iid_gaussian, sparse_precision, contaminated at T 1024; bivariate_gaussian at T 4096 | analytic: `1 - r` and `sqrt(2(T-1)(1-r))`, where r is a from-scratch Pearson correlation of the raw rows | rtol 1e-10, atol 1e-12 |
| `scaleinv`, `detrendinv`, `perm` | scaled_offset (pipeline); trending (pipeline-detrend); a joint time permutation (probe) | the from-definition distance on the z-scored bivariate_gaussian draw. For `detrendinv` the draw is first detrended. For `perm` the rows are left unpermuted | rtol 1e-10 |
| `trend-nodetrend` | trending through the plain pipeline | from-definition on the z-scored trending rows. The change against the untrended draw is recorded | exact 1e-12 |
| `sym`, `sym-wide64` | var2_bivariate, gauss_ar1_pair, iid_gaussian; a 64-row mixture of iid_gaussian (probe) | pyspi's own matrix: NaN diagonal and max\|A − A.T\|. Bit-exact symmetry is expected, except euclidean, where last-bit differences are allowed (REVIEW M2). pyspi is judged against itself, so these runs are labelled **same-lineage** (self-consistency) in results.json | qualitative |
| `pop`, `pop-negated` | bivariate_gaussian T 64/256/1024/4096; iid_gaussian T 4096; negated rows (probe; ρ → −ρ) at T 1024/4096 | analytic population value at the dataset's true ρ, with se taken from the Monte-Carlo sd of 2000 z-scored Gaussian pairs | statistical. se = sd × z₁₋₀.₀₂₅/ₘ/2, where m is the number of unique pairs. This gives a Bonferroni family-wise 5 % test, declared before the runs. Raw z values (`ref_max_z_raw`) are also stored |
| `delayed-null` | delayed_noise T 1024 | Bonferroni 95 % band of the Monte-Carlo ρ = 0 null | qualitative |
| `dup` | shared_envelope T 64–4096 (x2 is a bit-identical copy of x0) | d(x0, x2) = 0 by definition | exact 1e-12 |
| `near` | rows z(x0 + ε x1) for ε = 1e-3, 1e-5, 1e-7, 1e-9 (probe) | cancellation-free direct forms. Cosine uses \|u/\|u\| − v/\|v\|\|²/2 | rtol 1e-6, atol 1e-12 |
| `edge` | kernel on (x, −x) and on two rows that share exact zeros (probe) | from-definition, with canberra 0/0 := 0 | exact |

Population values used by `pop`:
- euclidean: `sqrt(2(T-1)(1-ρ))`
- cityblock: `T·2 sqrt((1-ρ)/π)`
- cosine: `1-ρ`
- chebyshev: `∫₀^∞ 1 - erf(t/(2 sqrt(1-ρ)))^T dt`, by quadrature
- canberra: T times a quadrature over the bivariate-normal angle density. At ρ=0 this reproduces 1/2 + ln2/π = 0.72064 to 1e-16
- braycurtis: `sqrt((1-ρ)/(1+ρ))`

### distance.pdist.euclidean: matches, score 9 (check_method analytic)

**What agrees**
- **From-definition:** 15/15 pass. Kernel max_rel 1.29e-15, pipeline 9.5e-16, pipeline-detrend 2.27e-15.
- **Identity `sqrt(2(T-1)(1-r))`:** 5/5 pass, max_rel 1.21e-15 (`…|ident`). This confirms the ddof=1 convention.
- **cdist (same-lineage):** max_rel 1.05e-15.
- **Invariances:** scaleinv 9.5e-16, detrendinv 2.09e-15, perm 5.24e-16.
- **Population (`pop`):** 7/7 pass, max_z 1.41. The relative error falls with T: 0.0898 (T=64), 0.0543 (256), 0.0180 (1024), 0.0132 (4096).

**Where it fails** (the trace candidate, now confirmed)
- `…|shared_envelope|{64,256,4096}|pipeline|dup` fails. pyspi gives d(x0, x2) = 1.19e-07, 4.13e-07 and 9.54e-07 where the answer is exactly 0. At T=1024 it gives 0.0.
- `…|iid_gaussian|1024|pipeline|near` fails:

  | ε | relative error |
  |---|---|
  | 1e-3 | 2.4e-11 |
  | 1e-5 | 7.0e-07 |
  | 1e-7 | 4.9e-02 |
  | 1e-9 | 1.0 (pyspi returns 0.0 for a true distance of 3.18e-08, because of the clamp at 0) |

  The largest absolute error is 1.56e-07.

**Why**
- The cause is sklearn's dot-product expansion (`sklearn/metrics/pairwise.py:409-416`).
- sklearn's own docstring warns about it (`pairwise.py:284-286`: "potentially suffers from 'catastrophic cancellation'").
- It is not listed in KNOWN_ISSUES. REVIEW M2 documents only the last-bit reciprocal asymmetry, which comes from the same expansion.

**Symmetry**
- M2 is reproduced and is already documented. `…|iid_gaussian|1024|pipeline|sym-wide64` gives max\|A − A.T\| = 7.1e-15 (relative 1.1e-16) on 132 of 2016 pairs.
- On P ≤ 5 the matrix is exactly symmetric.

**Score**
- Score 9, rubric anchor "matches canonical … within statistical tolerance / trivial convention". The formula is canonical.
- It is not scored 10 because agreement fails the ≤ 1e-8 relative bar for near-duplicate rows.

### distance.pdist.cityblock: matches, score 10 (independent-impl)

- **From-definition:** 15/15 pass, max_rel 3.55e-15.
- **cdist:** bit-identical, as expected because pyspi's `manhattan_distances` calls scipy cdist.
- **Invariances:** scaleinv 2.38e-15, detrendinv 7.9e-16, perm 1.58e-15.
- **Population `T·2 sqrt((1-ρ)/π)`:** 7/7 pass, max_z 1.64. The relative error falls from 0.131 at T=64 to 0.017 at T=4096. The pass uses the family-wise Bonferroni band (2 se = 2 × 1.404 sd for the 10 pairs of iid_gaussian). The per-pair raw z reaches 2.31 on `…|iid_gaussian|4096|pipeline|pop`, so a per-pair 2-sd band would fail that run.
- **Duplicates and edges:** dup gives 0.0 at every T. Edge d(x, −x) = 1680.899291219274 against the reference 2 Σ|x| = 1680.8992912192734 (max_abs 9.1e-13, inside exact 1e-12; `…|iid_gaussian|1024|kernel|edge`).
- **Candidates:** both confirmed (T-linear scaling and dissimilarity orientation; asymptotically a monotone function of ρ for Gaussian pairs, 7/7 pop and pop-negated runs). Neither is a formula deviation. Both are properties of the canonical definition, so `new_finding` is false.

### distance.pdist.cosine: matches, score 10 (analytic)

- **Identity `1 − r`:** 5/5 pass, max_abs 4.44e-16.
- **From-definition:** 15/15 pass, max_rel 6.5e-15.
- **cdist:** max_abs 2.2e-16.
- **Population:** 7/7 pass, max_z 1.40.
- **Near-identical rows:** passes at atol 1e-12 (rtol 1e-6) against the cancellation-free form \|u/\|u\| − v/\|v\|\|²/2, with max_abs 3.79e-16. The relative error is 2.3e-10, 2.2e-06, 7.7e-02 and 1.0 at ε = 1e-3, 1e-5, 1e-7 and 1e-9. At ε = 1e-9 pyspi returns 0.0 against 4.95e-19. So the per-value 1e-8 relative bar is missed for ε ≤ 1e-5.
  - The cancellation belongs to the canonical `1 − u·v/(|u||v|)` form itself. Evaluated from definition (the audit kernel `pdist_metric`) on the same rows, it misses the stable form in the same way (rel 5.4e-02 at ε = 1e-7, −2.2e-16 at ε = 1e-9) and agrees with pyspi to ≤ 4.4e-16 absolute (probe `wf4b_high-eeg/fix_cosine_near.py`). A cancellation-free reformulation avoids it.
  - Score 10 therefore reads "floating-point tolerance" as rtol plus an absolute floor at the [0, 2] value scale. Euclidean is scored 9 for a different reason: its canonical form is a direct difference, pyspi's dot-product expansion departs from it, and the absolute error (1.56e-07) fails atol 1e-12 on the near and dup runs.
  - pdist is absent from the supplement, so the anchor-10 supplement clause is met only through the GitBook description. This applies to the whole family and is left to synthesis.
- **Candidates:** both confirmed.
  - It is an affine map of Pearson r, so "nonlinear" does not describe it on default input.
  - Values above 1 encode negative correlation. In `…|bivariate_gaussian|1024|pipeline|pop-negated` the ρ = −0.8 pair gives 1.797, and the ρ = −0.3 pair gives 1.281. At the kernel edge, d(x, −x) = 2.0.

### distance.pdist.chebyshev: matches, score 10 (independent-impl)

- **From-definition:** bit-identical in all 14 kernel and pipeline runs. Detrend gives 3.9e-16.
- **cdist:** bit-identical.
- **Population (quadrature):** 7/7 pass, max_z 1.82. The relative errors of 0.13–0.23 reflect the large sd of an extreme statistic. The pass uses the family-wise Bonferroni band (2 se = 2 × 1.319 sd for 6 pairs, 2 × 1.404 sd for 10 pairs). The per-pair raw z exceeds 2 in three runs: 2.56 (`…|iid_gaussian|4096|pipeline|pop`), 2.22 (`…|bivariate_gaussian|4096|pipeline|pop-negated`) and 2.01 (`…|bivariate_gaussian|256|pipeline|pop`). A per-pair 2-sd band would fail them.
- **Growth with T:** the population expectation grows with T. At ρ = 0 it is 3.67, 4.31, 4.87 and 5.38 for T = 64, 256, 1024 and 4096.
- **Outliers:** `…|contaminated|1024|pipeline|contam` gives 4.718, against 2.238 for the clean pair (ratio 2.11).
- **Candidate:** confirmed (interpretation only). It describes properties of the canonical definition (T-dependence, outlier sensitivity, dissimilarity orientation), so `new_finding` is false.

### distance.pdist.canberra: matches, score 9 (independent-impl)

**Agreement**
- **From-definition:** 15/15 pass, max_rel 1.69e-15.
- **cdist:** bit-identical.
- **Population:** 7/7 pass, max_z 1.45.

**Signed-data behaviour**
- **Edge:** d(x, −x) = 1024.0 = T, exactly. Shared exact zeros follow the 0/0 := 0 rule; pyspi and the reference both give 551.7.
- **Signed-affine (pipeline):** S01 = 256.0 = T, exactly, with no RuntimeWarning.
- **Effect on z-scored (signed) data:** every sample of opposite sign contributes exactly 1, whatever its amplitude. Samples near zero carry the most weight. The value stays inside [0, T].

**Score and candidates**
- Score 9: the canonical formula holds up to the documented |x| + |y| generalisation to signed data (Deza & Deza). Lance & Williams defined it for non-negative data.
- Both candidates are confirmed.

### distance.pdist.braycurtis: variant, score 7 (independent-impl), definition_under_review

**Agreement**
- **From-definition (SciPy form `Σ|x−y| / Σ|x+y|`):** 15/15 pass, max_rel 5.28e-15.
- **cdist:** bit-identical.
- **Population `sqrt((1-ρ)/(1+ρ))`:** 7/7 pass, max_z 1.35.

**Behaviour on signed data**
- **Range above 1:** the negated runs show values outside [0, 1]. At ρ = −0.8 the population value is 3.0; pyspi gives 3.054 at T=1024 and 2.933 at T=4096.
- **Silent +inf:**
  - `…|iid_gaussian|256|pipeline|signed-affine` gives S01 = S10 = +inf.
  - In the same run, S12 = 2.66e16 for the pair (−x, 2x+1). z-score rounding decides between this value and +inf; the reference's own z-score gives the same 2.66e16.
  - `…|signed-warn` records 0 RuntimeWarnings.
- **Labels:** `signed-affine` tests pyspi's entries against analytic constants (+inf, 0) and is labelled analytic. `signed-warn` and `calc-downstream` only observe pyspi itself (warning count, Calculator output) and are labelled same-lineage.
- **Downstream:** `…|calc-downstream` shows that +inf survives `Calculator.compute` and `_rmmin` (`pyspi/calculator.py:361-366`). The rank-based SPI–SPI Spearman matrix from `_get_correlation_df` (`calculator.py:427-460`) stays finite (all 1.0).

**Score and candidates**
- Score 7: pyspi matches the documented sklearn/SciPy variant fp-exactly. It differs in a value-changing way from the canonical non-negative index (Bray & Curtis 1957; Legendre & Legendre D14). On non-negative data the two forms coincide.
- Both candidates are confirmed.

### Common observations (all six)

**Symmetry and direction**
- Every metric except euclidean is exactly symmetric (max\|A − A.T\| = 0.0), including cosine on the 64-row probe. This is a self-consistency check of pyspi's matrix against its own transpose (same-lineage label); no score rests on it.
- The diagonal is NaN in every run.
- No direction is encoded. var2_bivariate (x0 → x1) gives a symmetric matrix.
- The delayed_noise copies sit inside the ρ = 0 null band for every metric. For cosine the values are 1.024, 0.992 and 0.946, inside [0.922, 1.083]. The metrics are lag-0 only.

**Preprocessing**
- Positive scale and offset are removed exactly by z-scoring (scaleinv).
- A linear trend is removed exactly by pipeline-detrend (detrendinv).
- Without detrend, the trend dominates. For trending pair (1,3), which is independent before the trend, cosine is 0.0028 against 0.98 untrended, and euclidean is 2.41 against 44.77 (`…|trending|1024|pipeline|trend-nodetrend`).

**Sign of correlation**
- Every metric gives its largest values for anti-correlated pairs (`pop-negated`). All six are dissimilarities that depend on the sign of the correlation, although the SPIs are labelled "unsigned".

### Trace-stage corrections

- `implementation_trace` in all six records now cites `pyspi/calculator.py` at HEAD a1f20c7: `:277` (Data construction), `:48-50` (defaults) and `:329-335` (multivariate, the NaN diagonal and the table store). These replace the stale `:264`, `:48` and `:294-297`.
- The trace text "(to be confirmed empirically in the check stage)" in `direction_convention` has been replaced by a pointer to the sym runs.
- No trace claim was refuted.
- The trace probe figure of 7.1e-15 for euclidean asymmetry is reproduced exactly, on a different 64-row input.

### Review corrections (after the adversarial review)

- **cityblock candidate 2 evidence:** said "9/9 within the Bonferroni band"; results.json holds 7 cityblock pop / pop-negated runs, all pass. Now 7/7.
- **cityblock edge:** the record and this note wrote d(x, −x) = 1681.0. The stored value is 1680.899291219274 (reference 1680.8992912192734). Corrected.
- **Reference labels:** `sym`, `sym-wide64`, `signed-warn` and `calc-downstream` judge pyspi against itself. `refs/pdist.py` now labels them same-lineage; the family was re-run with `--force` (252 pass, 4 fail, the same 4 euclidean runs; all quoted numbers unchanged). `signed-affine` keeps the analytic label because its pass rule compares pyspi's entries with analytic constants (canberra(x, −x) = T, braycurtis(x, −x) = +inf, d(x, 2x+1) = 0; `refs/pdist.py:ref_signed_affine`). No verdict or score rested on these runs.
- **new_finding:** set to false for cityblock and chebyshev. Their confirmed candidates are interpretation points about the canonical definitions, not deviations. The other four units keep `new_finding: true` on substantive candidates (euclidean dup/near failures and the r-identity; cosine 1 − r; canberra and braycurtis signed-domain behaviour).
- **Statistical runs:** closeness for cityblock and chebyshev now states that the per-pair raw z exceeds 2 and that the pass rests on the family-wise band. For the other four units the largest raw z is below 2 (euclidean 1.97, cosine 1.96, canberra 1.92, braycurtis 1.90).

### Infrastructure note

- `pyspi_outputs.py` rebuilds the merged `out/pyspi/index.json` after every run. It reads every family index file, so it crashed twice with `JSONDecodeError` when another family's index file was being written at the same moment.
- The crash comes from concurrency with other families, not from pdist.
- I worked around it by re-running. Cached runs are skipped, so the third pass completed all 256 runs with 0 errors.

## Open questions: answered

1. **Should "nonlinear" be kept for pdist_euclidean and pdist_cosine?**
   - On default input both are exact functions of Pearson r:
     - cosine = 1 − r, max_abs 4.44e-16;
     - euclidean = sqrt(2(T−1)(1−r)), max_rel 1.21e-15 (`…|ident`).
   - cityblock and braycurtis agree with their ρ-only population limits on Gaussian data (`…|pop`).
   - The label does not describe euclidean or cosine on the default (normalise=True) path. With normalise=False, cosine becomes the uncentred cosine, and that variant is not covered by this audit's runs.
   - Whether to drop the label is a labelling-policy decision for pyspi. The evidence supports dropping it for euclidean and cosine.
2. **What does "unordered" mean?**
   - Empirically, every pdist metric is invariant to a joint permutation of time (`…|perm`, max_rel ≤ 2.2e-15; for euclidean, pyspi permuted against pyspi unpermuted differs by 2.5e-14) and uses only lag-0 pairs (`…|delayed-null`).
   - "Unordered" therefore fits the reading "sample order is irrelevant".
   - The glossary still does not define it. Only pyspi's authors can settle the intended meaning.
3. **Should pyspi document or guard Bray–Curtis and Canberra on signed data?**
   - Confirmed:
     - braycurtis is unbounded (3.0 in population at ρ = −0.8);
     - it returns +inf silently for y = −x;
     - for an affine pair, rounding decides between +inf and 2.66e16;
     - canberra counts every sign disagreement as exactly 1.
   - The evidence supports at least a documentation note. Whether a guard is wanted (for example, a warning or recording the SPI as failed) is a design choice.
4. **Is the ~1e-7 error of pdist_euclidean on duplicated channels acceptable?**
   - Quantified: 1.19e-07 to 9.54e-07 absolute on exact duplicates. The relative error reaches 4.9e-02 at ε = 1e-7, and the value is clamped to 0.0 at a true distance of 3.2e-08.
   - Relative to the value scale (about 2 sqrt(T−1)), the error is about 1e-8. It matters only for duplicate detection, for exact-zero tests and for byte-exact storage comparisons (M2).
   - A direct-difference implementation (scipy `cdist`, or `pdist_metric`) gives 0.0 exactly.
5. **Dissimilarity orientation and T-dependence**
   - Confirmed for all six metrics: larger values mean less similar, and anti-correlated pairs score highest.
   - euclidean grows as sqrt(T−1). cityblock and canberra grow linearly in T. The chebyshev expectation grows slowly (3.67 → 5.38 from T=64 to 4096 at ρ = 0).
   - cosine and braycurtis do not depend on T in expectation.
   - This is a documentation issue for anyone who compares values across window lengths or combines SPIs.
6. **Should SciPy's definitions be treated as the documented definition?**
   - The GitBook names sklearn's `pairwise_distances` with these metric names, and sklearn hands chebyshev, canberra and braycurtis to SciPy. So the SciPy formulas are the documented definitions.
   - The audit records both forms only for braycurtis (`definition_under_review: true`). There the SciPy |x+y| denominator differs from the canonical Σ(x+y) on signed data.
   - For canberra, the |x| + |y| form is the standard generalisation.
7. **Does +inf from braycurtis survive Calculator.compute and break downstream normalisation?**
   - It survives `compute` and `_rmmin`: inf − min = inf (`…|calc-downstream`).
   - It does not break pyspi's own SPI–SPI correlation step. `_get_correlation_df` uses Spearman ranks of |values|, and +inf ranks as the largest value, so the matrix stays finite.
   - Downstream code outside pyspi that z-scores or min-max scales the table would produce NaN. That case was not run.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls come from `audit/profiles/summary.json`, all measured directly at every shape, including 118x1024. Nothing in pyspi was changed. All six units are in the retained EEG selection, one identifier each, with no band variants.

### Walls

| unit | 8x1024 | 32x1024 | 118x1024 per window (measured) |
|---|---|---|---|
| pdist_euclidean | 0.56 ms | 0.58 ms | 1.38 ms |
| pdist_cityblock | 0.49 ms | 0.91 ms | 7.52 ms |
| pdist_cosine | 0.70 ms | 0.76 ms | 1.77 ms |
| pdist_chebyshev | 0.52 ms | 0.69 ms | 3.44 ms |
| pdist_canberra | 0.54 ms | 0.84 ms | 5.84 ms |
| pdist_braycurtis | 0.53 ms | 0.78 ms | 5.12 ms |

**Complexity: O(P^2 T) in one call** (`pyspi/statistics/distance.py:PairwiseDistance.multivariate` -> sklearn `pairwise_distances` -> compiled sklearn/scipy kernels). At EEG sizes the fixed per-call cost of about 0.5 ms dominates: P_exponent is 0.02-0.45 and T_exponent about 0.1.

### Where the time goes (profiled call of 1-2 ms)

- sklearn's `_param_validation.py:wrapper` is 94-96% cumulative and `pairwise_distances` 53-66%. Both are wrappers.
- The deepest frame that explains the most time at 8x1024 is input validation, `sklearn/metrics/pairwise.py:check_pairwise_arrays` -> `sklearn/utils/validation.py:check_array`: euclidean 29.6%, cityblock 33.4%, cosine 22.3%, chebyshev 30.5%, canberra 32.6%, braycurtis 31.6% (hotspot_1).
- The metric itself (hotspot_2):
  - euclidean: `_euclidean_distances` 15.8%. `euclidean_distances` (46.0%) includes the validation call.
  - cityblock: `scipy/spatial/distance.py:cdist` (third-party:scipy) 5.5%, inside `manhattan_distances` (42.8%).
  - cosine: `sklearn/preprocessing/_data.py:normalize` 19.3%, inside `cosine_distances` (57.7%).
  - chebyshev, canberra and braycurtis: `scipy/spatial/distance.py:pdist` (third-party:scipy) 16.1-16.6%, with the compiled kernel at 2.8-3.5%.
- At 8x1024, validation is about half of each millisecond call. That does not hold at 118x1024. Validation is a fixed cost of roughly 0.15-0.3 ms, while the compute grows with P^2. It is then roughly 10-20% of euclidean (1.38 ms) and cosine (1.77 ms), and 2-9% of chebyshev, canberra, braycurtis and cityblock (3.4-7.5 ms).

### Opportunities

None listed. pdist is already a single compiled call per window. The six units together cost about **24 ms per window** at 118x1024 (sum of the measured family-warm increments: 1.34 + 7.31 + 1.35 + 3.32 + 5.66 + 4.97 ms). That is about 14 core-seconds per 600-window patient, against roughly 185 s per window historically for the whole EEG metric set.

Two avenues remain, but neither is material at these walls:

- Calling scipy `pdist` + `squareform` directly would skip sklearn's validation, about half of the call at 8x1024, but only a few percent of cityblock, canberra and braycurtis at 118x1024.
- cityblock goes through scipy `cdist(X, X)` and computes both triangles; `pdist` would compute one.

### EEG cost

The per-window EEG cost at 118 channels is the measured increment per identifier, as in the table above: about 24 ms per window for all six. No cache is involved; each identifier is an independent call.
