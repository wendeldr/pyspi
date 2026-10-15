# dtw_family: define, trace and check

Units: `distance.dtw.constraint-None`, `distance.dtw.constraint-itakura`, `distance.dtw.constraint-sakoe_chiba`,
`distance.lcss.constraint-None`, `distance.lcss.constraint-itakura`, `distance.lcss.constraint-sakoe_chiba`,
`distance.softdtw` (identifiers `softdtw`, `softdtw_constraint-itakura`, `softdtw_constraint-sakoe-chiba`).

Delegate: tslearn 0.9.0 (`tslearn/__init__.py:2`), numpy/numba backend. Probe scripts:
`/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/dtw_family/probe{1..6}.py`;
review fixes in `.../scratchpad/dtw_fix/probe_fix.py`, `probe_softdtw_err.py`, `tsl.py` and `dtai.py`.
The sections before `## Checks` are the trace stage; their probes only confirm what the code does. The regression goldens are
snapshots of current output. They are cited only as observations and are not a reference.
The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared pipeline (all seven units)

1. `Calculator.compute` calls `multivariate(dataset)` (`pyspi/calculator.py:329`).
2. Preprocessing (`pyspi/data.py:184-195`): optional `scipy.signal.detrend` (default off), then `zscore(axis=1, ddof=1)`
   (default on). From here on the series are dimensionless.
3. `Undirected.multivariate` (`pyspi/base.py:132-138`) calls `Directed.multivariate` (`base.py:84-94`). That method
   computes `bivariate(i, j)` for **both** orders and leaves the diagonal NaN. The lower triangle is then overwritten
   by the upper, so `A[i,j] = A[j,i] = f(z_min(i,j), z_max(i,j))`. All three functions are argument-symmetric for
   equal lengths (probe: raw `soft_dtw(z0,z1) - soft_dtw(z1,z0) = 0.0`; the Itakura mask equals its transpose at
   equal length), so the mirroring only matters at the last bit. It also doubles the work, which is relevant to the
   optimisation stage.
4. `TimeWarping.bivariate` (`pyspi/statistics/distance.py:174-177`) calls
   `self._simfn(z[i], z[j], global_constraint=...)`. No radius, slope, eps or gamma is ever passed, so every
   hyperparameter is a tslearn default.
5. Identifier spelling. `distance.py:161` replaces `_` by `-`, giving `*_constraint-sakoe-chiba`. The supplement and
   GitBook list `*_constraint-sakoe_chiba`. This is cosmetic.

Cross-family observation (preprocessing infrastructure, hypothesis-level relevance here): when an SPI is called as
`spi.bivariate(x, y)` with raw arrays, `parse_bivariate` (`base.py:33-36`) builds `Data()` and appends the processes
one at a time. Only the first goes through `set_data` and is z-scored. The second is `np.append`-ed raw
(`data.py:230-234`). Probe: x, y ~ 5 + 3N(0,1) were stored with means [0.0, 4.783] and stds [1.0, 2.995]. Kernel
checks must therefore pass pre-z-scored arrays or go through `Data`/`multivariate`. This is not in KNOWN_ISSUES.

## DTW (`dtw`, `dtw_constraint-itakura`, `dtw_constraint-sakoe-chiba`)

**Documented** (supplement §1.2.7, p.6-7; GitBook "Dynamic Time Warping"): "DTW finds the minimum distance between
two time series through alignment (shifting and dilating of the sequences)". The SPI uses "the Sakoe-Chiba band [17]
and the Itakura parallelogram [18] global constraints" and is computed with tslearn. No width or slope is documented.

**Canonical**: Sakoe & Chiba 1978 (IEEE TASSP 26(1):43-49), symmetric form:
g(i,j) = min{g(i,j-1)+d, g(i-1,j-1)+2d, g(i-1,j)+d}, D = g(I,J)/(I+J), where d is a non-squared distance. Adding the
adjustment window |i-j| ≤ r gives the Sakoe-Chiba band. Itakura 1975 (IEEE TASSP 23(1):67-72) uses local slope
constraints of 1/2..2, and their reachable region is the parallelogram.

**Implemented** (trace):

- `distance.py:187` sets `_simfn = tslearn.metrics.dtw`. The function is `tslearn/metrics/_dtw.py:30-163`.
- Inputs pass through `to_time_series(remove_nans=True)` and are mapped by `GLOBAL_CONSTRAINT_CODE` (`_masks.py:10`).
- `_dtw_generic` (`_dtw.py:391-400`) computes the mask (`_masks.py:310-361`) and the accumulated matrix
  (`_dtw.py:324-344`), then returns `sqrt(cum_sum[-1,-1])`.
- The local cost is the squared difference. All three moves have weight 1 ("symmetric1"), and there is no length
  normalisation. On a T=50 seed-0 iid pair, tslearn equals a from-scratch DP exactly (6.566820109835115).
  dtaidistance 2.5.1 `dtw.distance(use_c=False)` (audit comparison venv, no tslearn import) gives the same value on that
  pair and on a second pair, T=80 seed 5 (5.972081410007747 for both, difference 0.0). dtaidistance `window=2` equals the
  tslearn radius-1 band on both pairs (8.061528067978553 and 8.12914957772889). `window=1` gives 10.13 and 11.37.
- The Sakoe-Chiba radius defaults to **1** (`_masks.py:339-341`). The mask is |i-j| ≤ 1 (`_masks.py:13-29`) and
  covers 3T-2 cells: 4.6% of the grid at T=64 and 0.07% at T=4096.
- The Itakura max slope defaults to **2.0** (`_masks.py:346-348`). The mask is a global parallelogram with bounds
  rounded to 2 decimals (`_masks.py:89-140`) and covers about 1/3 of the grid at every T.

**Deviations**:

- *From documented*: none in substance. The band width (1 sample) and slope (2) are undocumented, and so is the
  squared-cost/sqrt convention.
- *From canonical* (all three units):
  - squared cost with a final sqrt instead of the sum of |a-b|. Probe: 6.567 vs 36.674 for the abs-cost unweighted DP.
  - diagonal weight 1 instead of 2.
  - no division by I+J, so the value grows with T.
- *Sakoe-Chiba unit*: r is fixed at 1 regardless of T. Candidate: this unit stays close to the lock-step Euclidean
  distance (probe T=50: 8.06 vs unconstrained 6.57 vs Euclidean 10.13). It cannot absorb delays above 1 sample.
- *Itakura unit*: the constraint is the global parallelogram only. Itakura's local slope rule is not applied, so long
  horizontal or vertical runs are allowed inside the region.
- Candidate (minor): the 2-decimal rounding of the parallelogram bounds may admit or exclude boundary cells
  differently from exact arithmetic at large T.
- Candidate (interpretation): unconstrained DTW on z-scored noise barely separates independent from anti-correlated
  pairs. Probe (T=1024): 21.95 vs 22.24.

**Range**: ≥ 0, 0 iff exact alignment, unnormalised, symmetric, diagonal NaN.

**Proposed checks**:

- From-scratch numpy DP, tolerance 1e-12, on iid_gaussian, delayed_noise, narrowband_f0-0.25 and logistic_maps.
- External-package check with dtaidistance 2.5.1 (`window=2` for the band).
- Analytic bounds (optimality): dtw(x,x)=0, dtw_none ≤ dtw_constrained ≤ ‖x−y‖₂, and dtw ≥ sqrt(Σ_i min_j (x_i−y_j)²).
- Exact-rational Itakura mask compared cell by cell with tslearn's mask.
- Itakura-local-slope and Sakoe-1978-normalised variants as qualitative convention-gap checks.
- delayed_noise with delays 1/3/5 for the 1-sample band.
- scaled_offset vs bivariate_gaussian (z-score invariance).
- trending through pipeline-detrend.
- Direction/symmetry check: A == A.T exactly, NaN diagonal, and A[i,j] equal to the reference on (z_min, z_max).

## LCSS (`lcss`, `lcss_constraint-itakura`, `lcss_constraint-sakoe-chiba`)

**Documented** (supplement §1.2.8, p.7; GitBook "Longest Common Subsequence"): "similarity between continuous
subsections ... subject to distance thresholds and alignment constraints. We use the default threshold from the tslearn
package (ϵ = 1), and include the three global constraints".

**Canonical**: Vlachos, Kollios & Gunopulos 2002 (ICDE, 673-684). LCSS_{δ,ε} counts matches where |a_n−b_m| < ε and
|n−m| ≤ δ, and the similarity is S1 = LCSS / min(n, m). The definition numbering was not re-checked against the PDF.

**Implemented** (trace):

- `distance.py:197` sets `_simfn = tslearn.metrics.lcss` (`dtw_variants.py:2383-2499`, eps=1.0).
- The mask comes from `_masks.compute_mask` (imported at `dtw_variants.py:11`), which is **boolean**.
- `_njit_lcss` (`dtw_variants.py:2321-2345`) returns `acc[-1][-1] / min(l1, l2)`.
- `njit_lcss_accumulated_matrix` (`dtw_variants.py:2232-2267`) uses the match rule `sqrt(sq) <= eps` at line 2260.
  Line 2259 gates each cell with **`if numpy.isfinite(mask[i - 1, j - 1]):`**.

**Candidate new finding (confirmed by probes, not in KNOWN_ISSUES)**: in tslearn 0.8.0, 0.8.1 and 0.9.0 the numpy LCSS
path ignores the global constraint.

- `np.isfinite(False)` is `True`, so every cell is evaluated.
- Release range (WebFetch of the GitHub tags). In v0.5.2, v0.6.4 and v0.7.0 the masks are float arrays
  (`numpy.full(..., numpy.inf)`, admissible cells 0.0), so the same `isfinite` test excludes masked cells. In v0.8.0
  the masks move to `tslearn/metrics/_masks.py` and become boolean (`backend.full((sz1, sz2), False)`), while v0.8.0,
  v0.8.1 and v0.9.0 keep the `isfinite` test. tslearn `main` now tests `if mask[i - 1, j - 1]:` in
  `njit_lcss_accumulated_matrix`. The bug is therefore fixed upstream, but v0.9.0 is still the latest tag.
- pyspi requirement history (git): ef478a8 (2021-11-29) `tslearn>=0.5.2`; e66057f (2022-02-10) `tslearn==0.5.2`;
  0943868 (2024-05-18) bare `tslearn` (`requirements.txt:20`, `setup.py:29` at a2c3300^); a2c3300 changes the bare
  `tslearn` to `tslearn>=0.9,<0.10` (`pyproject.toml:41`). The test lock `tests/requirements-test-lock.txt` already held
  `tslearn==0.9.0` when it was created (30556b9, line 66, before a2c3300). The migration therefore did not change the
  locked tslearn version, and the snapshot goldens were already produced with the ignored mask.
- The tslearn version behind published pyspi results cannot be read from the pins. Under the `==0.5.2` pin
  (2022-02-10 to 2024-05-18) the mask is honoured. An install from the unpinned period could resolve to whatever
  release was current (0.8.0 or later ignores the mask). Hypothesis only: published `lcss_constraint-*` values may
  differ between installs.
- The non-numpy branch (`lcss_accumulated_matrix`, `dtw_variants.py:2306`, `if mask[...]`) honours the mask.
- Probe on a delayed pair (T=60):
  - tslearn returns 0.9167 for None, itakura and sakoe_chiba alike.
  - a from-scratch masked LCSS gives 0.9167 / 0.0 / 0.70.
  - the tslearn torch backend reproduces the masked values (T=64: 0.797 and 0.766, against 0.8125 from the numpy
    backend).
- The pyspi 4×128 matrices for the three identifiers are bit-identical. As an observation only, the snapshot files
  (small/full) also hold identical matrices for the three identifiers.

**Other deviations / candidates**:

- The match test is `<=` where Vlachos uses strict `<`. This matters only on exact ties.
- eps=1 is absolute, which means 1 SD after z-scoring. Unconstrained LCSS means δ = ∞.
- Candidate (low dynamic range): independent z-scored noise scores about 0.80 at every T, and x vs −x scores about
  0.80 too (probe T=64/256/1024: 0.797/0.805/0.809 and 0.828/0.797/0.807). The snapshot matrices span 0.84-0.97.
- Candidate (Itakura, intended behaviour, if the mask were honoured): at T=128 the first row of tslearn's parallelogram
  admits only column 0, and the last row and last column admit only the corner cell. Masked cells stay 0, so the score
  is exactly 0 iff the final samples differ by more than eps. A from-scratch masked DP confirmed this on all 400 seeded
  independent pairs. If the final z-scored samples are approximately N(0,1), their difference is approximately N(0,2),
  so P(score = 0) = 1 − (2Φ(1/√2) − 1) = 0.48. Sampled fractions scatter around this value (30-pair runs: 50% and 43%;
  400-pair run: 47%).

**Range**: similarity in [0, 1], higher = more similar despite the "distance" module, quantised in steps of 1/T,
symmetric.

**Proposed checks**:

- From-scratch LCSS DP (exact rational values), plus a strict-`<` variant.
- Masked from-scratch references for the band (|i−j| ≤ 1) and the exact parallelogram. Expected outcome: pyspi differs
  from the masked reference and equals the unconstrained one.
- Same-lineage comparison of the three pyspi matrices (expected bit-identical) and of the torch-backend tslearn.
- Statistical saturation check on iid_gaussian and bivariate_gaussian.
- z-score invariance on scaled_offset.
- Direction/symmetry check as for DTW.

## Soft-DTW (`softdtw` + two constraint identifiers)

**Documented** (supplement §1.2.9, p.7; GitBook "Soft Dynamic Time Warping"): "smoothed formulation of DTW ... via the
tslearn package (with the default hyperparameter, γ = 1), and includes the same global constraints as DTW".

**Canonical**: Cuturi & Blondel 2017 (ICML, PMLR 70:894-903), Algorithm 1:
r_ij = ‖x_i−y_j‖² + min^γ(r_{i−1,j−1}, r_{i−1,j}, r_{i,j−1}), where min^γ = −γ log Σ exp(−a/γ).

**Implemented** (trace):

- `distance.py:205-208` overrides `bivariate` and calls `tslearn.metrics.soft_dtw(z[i], z[j])`. It passes neither a
  constraint nor gamma, and `soft_dtw` (`softdtw_variants.py:530`) has no constraint argument.
- The cost matrix is `SquaredEuclidean.compute` (`softdtw_variants.py:1208-1216`), which squares sklearn 1.9.1
  `euclidean_distances` (`pairwise.py:376-426`, dot-product expansion with sqrt).
- The recursion is `_njit_soft_dtw` (`soft_dtw_fast.py:100-125`), using `_njit_softmin3` with fastmath (`:14-48`).
- `SoftDTW.compute` (`softdtw_variants.py:1111-1135`) returns R[T,T].
- Probe: −2.205017147170598 against a from-scratch −2.2050171471705955.

**Deviations**:

- *From documented*: the Sakoe-Chiba and Itakura variants do not exist. The three identifiers are identical. This is
  already documented (tests/README.md bug 3; `tests/unit/test_known_bugs.py:111-131`).
- The class `name` duplicates "Dynamic time warping" (cosmetic).
- *From canonical*: none for the value.

**Candidates**:

- Interpretation: pyspi reports the raw soft-DTW value, not the soft-DTW divergence of Blondel, Mensch & Vert 2021.
  The value is mostly negative, with sdtw(x,x) ≠ 0, and is dominated by a length-dependent entropic term. Probe at
  T=1024: x,x = −717.3, independent = −189.1, x,−x = −188.4. (Check stage: this is a property of the canonical,
  documented quantity, and it does not contradict the "unsigned" keyword: the supplement only lists the keyword axes
  (supplement.txt:126-128), and the GitBook glossary (glossary-of-terms.md:21) is mixed, tying the keyword first to the range and then to
  whether the sign indicates how A and B co-fluctuate; the audit adopts the sign-meaning sentence; "distance" is the literature category.)
- Numerical (minor): the cost matrix uses the dot-product expansion. The absolute cell error is of order 1e-14 on
  unit-scale data (1.07e-14 on z-scored data) and 2.9e-8 with a 1e4 offset when normalise=False. The relative error of
  a cell is that absolute error divided by (x_i − y_j)², so it is unbounded as a cell approaches 0. For y = x + 1e-3
  noise at T=256, the maximum diagonal relative error ranged from 1.2e-7 to 2.1e-3 across 20 seeds.

**Proposed checks**:

- From-scratch soft-DTW with logsumexp, tolerance 1e-10.
- Same-lineage exact identity of the three identifiers (re-confirming bug 3).
- A γ→0 sanity check of the reference against dtw², using the bound dtw² − γ log|paths| ≤ sdtw ≤ dtw².
- Qualitative sign, range and divergence comparison.
- Direction/symmetry check.

## Open questions

1. Which tslearn version produced the published pyspi results? The requirement was `>=0.5.2` (2021-11), `==0.5.2`
   (2022-02 to 2024-05), then unpinned until a2c3300. If it was below 0.8.0, `lcss_constraint-*` values in published
   work were constrained. The in-repo test lock already held `tslearn==0.9.0` before a2c3300, so the migration did not
   change the locked version. The boolean-mask change is in v0.8.0, and the fix is on tslearn `main` but unreleased.
2. Is a 1-sample Sakoe-Chiba radius intended? tslearn's default makes `dtw_constraint-sakoe-chiba` nearly lock-step at
   EEG lengths. A width proportional to T (e.g. 10%) would match common practice but would change the SPI.
3. Should `lcss_constraint-itakura` be repaired to honour the mask, given that the intended masked LCSS under
   tslearn's parallelogram is itself degenerate (0 unless the final samples match)?
4. Should softdtw report the divergence (`cdist_soft_dtw_normalized`) instead, given the negative, T-dominated raw
   values?
5. Should the parse_bivariate raw-array normalisation asymmetry (second process not z-scored) be routed to the
   preprocessing/infrastructure family as a new finding? It affects every bivariate SPI called with two arrays.
6. Vlachos 2002 definition numbering and the precise Sakoe-Chiba 1978 table/equation numbers were cited from memory
   and should be checked against the PDFs.

## Checks

Manifest `audit/checks/manifest/dtw_family.json` (345 runs), references `audit/checks/refs/dtw_family.py` (comparison venv, no
tslearn or pyspi import), probes `audit/checks/probes/dtw_family_probes.py` (tslearn masks, unmirrored `Directed.multivariate`,
derived rows, raw two-array calls, tslearn torch-backend LCSS, soft-DTW at other gamma). Results in
`audit/checks/out/results/dtw_family.json`: 300 pass, 45 fail, 0 without reference. Every fail is an expected deviation
documented below: 12 `canon` runs (DTW convention), 23 LCSS constraint runs (ignored mask), 8 soft-DTW `masked` runs (known
bug 3), the soft-DTW `offset` run (1.0e-8 relative) and the sakoe-chiba `null-masked` run. All DP checks were run at T <= 1024.
Only the mask probes ran at T = 4096. The reference DPs vectorise over anti-diagonals. The Monte Carlo nulls use a batched
two-diagonal DP that the script checks against the kernels at start-up. The regression goldens were not used.

Conventions used throughout. `dp` references compute every ordered pair separately, so both orientations of the stored
matrix are compared. `unmirrored` probes call `Directed.multivariate` on the SPI, which lets tslearn compute both orders
without the upper-to-lower copy. Null runs compare the mean over pairs with a 400-pair Monte Carlo of independent z-scored
Gaussian pairs, with a Bonferroni-scaled SE.

### distance.dtw.constraint-None: variant, score 7 (rubric 7: matches the supplement variant, differs from canonical in a value-changing way)

- **From-scratch DP.** `_kernels_distance.dtw_dp` gives squared cost, weight-1 steps and a final sqrt. All 16 `dp` runs are bit-identical (max_abs 0.0): kernel T64/256/1024 on iid_gaussian, delayed_noise, narrowband_f0-0.25 and logistic_maps, plus pipeline T1024 on 4 datasets.
- **dtaidistance 2.5.1** (external package). The C path gives max_abs 0.0 in 4 runs. The C path equals the pure-python path (0.0 at T256).
- **Convention against Sakoe & Chiba 1978.** The 1978 symmetric form uses |a-b| cost, diagonal weight 2 and divides by n+m. Both values are recorded (`canon`, 4 runs, fail by design):
  - iid_gaussian T1024 pair (0,1): pyspi 22.50, 1978 form 0.417, |.|-cost weight-1 sum 623.8.
  - On this unit's 4 `canon` runs the pyspi/1978 ratio is 23.96-55.18 and the Spearman rank correlation over pairs is 0.77-0.96. The constrained units have their own ranges (itakura 25.6-62.2 and 0.71-0.95; sakoe_chiba 24.4-51.1 and 0.76-0.94).
  - The disagreement goes beyond scale. The squared cost changes which path is optimal and how pairs rank.
- **Analytic bounds** hold on every pair (`bounds`): ||x-y|| >= dtw >= sqrt(max row/column minima) and dtw_None <= dtw_constrained.
- **Null.** T256: 11.19 vs Monte Carlo 11.24 +- 0.31 (max_z 1.32). T1024: 22.04 vs 21.84 +- 0.25 (max_z 1.51). The value grows like sqrt(T).
- **Scale and trend.**
  - scaled_offset through the pipeline equals the z-scored bivariate_gaussian reference to 3.6e-15.
  - On the kernel path, raw/z-scored ratios are 3.0-15.5, so DTW without z-scoring carries the amplitude.
  - trending without detrend changes values by up to 279%. pipeline-detrend restores them to within 0.19% (exact against the own detrend reference, 4.6e-14).
- Symmetry: max|A - A.T| = 0.0, also on the unmirrored probe. On delayed_noise the dtw/Euclidean ratios are 0.084 / 0.098 / 0.104 for delays 1 / 3 / 5.
- **Candidates.**
  - Low discriminative range: **confirmed** (`null-neg`). At T1024, dtw(x,-x) is 22.25 against 22.09 for independent pairs (+0.7%), while the Euclidean gap is +41.5%. The anti-correlated pair is 1.6 null SD above the null mean. This is a property of unconstrained DTW on noise, so new_finding is false.
  - parse_bivariate: **confirmed** (`rawcall`, exact). It is owned by preproc.two_array_api.
- definition_under_review: true. The DTW cost convention is genuinely split between the 1978 form and the Euclidean-over-path form used by ML libraries, and both are recorded.

### distance.dtw.constraint-itakura: variant, score 7

- **DP and mask.** The 16 `dp` runs are bit-identical. tslearn's parallelogram equals an exact-integer parallelogram cell by cell at T 64/256/1024/4096 (`mask`, n_cells_differ 0; one third of the grid).
- **Bounds and null.** Bounds and nesting hold. Null max_z is 1.16 (T1024) and 1.08 (T256).
- **Convention against Sakoe & Chiba 1978** (`canon`, 4 runs, fail by design): pyspi/1978 ratio 25.6-62.2 (62.16 on distance.dtw.constraint-itakura|delayed_noise|1024|kernel|canon), Spearman 0.71-0.95 (0.714 on |delayed_noise|256|kernel|canon).
- **Global vs local constraint.** The global-only parallelogram is compared with one symmetric local slope rule inside the same region (`local-slope`): no two consecutive (0,1) and no two consecutive (1,0) moves, weight-1 steps. Its local slopes are 1/2..2, the range of the Sakoe-Chiba P = 1 pattern, but horizontal-vertical corners are allowed. pyspi <= local on every pair. This rule raises the value by a median 18% on iid_gaussian (T1024: 17.7-19.6%, median 18.5%) and 0.7-4.5% on delayed_noise. These gaps are specific to the rule chosen. Itakura 1975's own local rule is asymmetric (the reference index advances 0, 1 or 2 per test frame, with no two consecutive 0 advances). It is a different DP and was not computed. An earlier docstring in `refs/dtw_family.py` called the implemented rule "Sakoe-Chiba P=1/2"; that label was wrong and has been corrected (values unchanged).
- **Candidates.**
  - Rounding: **refuted**. For equal lengths and slope 2 every bound is a multiple of 0.5, so 2-decimal rounding is exact.
  - parse_bivariate: confirmed, owned by preproc.two_array_api.

### distance.dtw.constraint-sakoe_chiba: variant, score 6 (rubric 6: deviates from both the documented description and canonical practice in an interpretable way)

- **DP, dtaidistance and mask.**
  - The 16 `dp` runs are bit-identical.
  - dtaidistance `window=2` equals tslearn radius 1 (4 runs, 0.0). `window=1` equals the Euclidean distance to 2e-14, so radius r maps exactly to window r+1.
  - The band is |i-j| <= 1: 190 / 3070 / 12286 cells at T 64/1024/4096, i.e. 3T-2 cells (4.6% / 0.29% / 0.073% of the grid).
- **Delay check** (`delay`, T1024). dtw/||z0-zk|| is 0.084 for delay 1 but 0.724 and 0.731 for delays 3 and 5. On iid_gaussian dtw/Euclidean is 0.70-0.74, against 0.48-0.51 unconstrained.
- **Convention against Sakoe & Chiba 1978** (`canon`, 4 runs, fail by design): pyspi/1978 ratio 24.4-51.1, Spearman 0.76-0.94.
- **Why 6 and not 7 like itakura.** Both units use the supplement's wording ("the Sakoe-Chiba band [17]", "the Itakura parallelogram [18]", no width or slope given), and a radius-1 band is still a Sakoe-Chiba band. The difference is what the supplement says the constrained DTW does (supplement.txt:283-286, GitBook distance-similarity.md:150). It finds the minimum distance "through alignment (shifting and dilating of the sequences)", and the constraints are there "to prevent pathological warpings". With r = 1 the band also prevents ordinary shifts. On the same delayed_noise data, delays of 3 and 5 samples keep dtw/||z0-zk|| at 0.724 / 0.731 under the band (distance.dtw.constraint-sakoe_chiba|delayed_noise|1024|pipeline|delay), against 0.098 / 0.123 under the parallelogram (distance.dtw.constraint-itakura|delayed_noise|1024|pipeline|delay). The itakura unit does the documented shifting and differs from canonical practice only by convention (rubric 7). The sakoe_chiba unit also departs from the documented description (rubric 6).
- **Candidates.**
  - 1-sample band: **confirmed, new finding** (not in KNOWN_ISSUES). The width is undocumented and fixed at every T, so the SPI is a slightly relaxed lock-step distance.
  - parse_bivariate: confirmed, owned elsewhere.

### distance.lcss.constraint-None: matches, score 9 (rubric 9: '<=' vs '<' tie convention)

- **From-scratch LCSS.** All 17 `dp` runs are bit-identical. The strict '<' rule of Vlachos et al. gives identical values on iid_gaussian T256/T1024 and on the tied logistic_maps data (T1024, pipeline).
- **Null.** Means 0.792 / 0.804 / 0.812 vs Monte Carlo 0.788 / 0.803 / 0.810 at T 64/256/1024 (max_z <= 1.86).
- **Candidates.**
  - Low dynamic range: **confirmed** as a property of eps = 1, not as a code deviation.
    - lcss(x,-x) is 0.7988 against 0.8105 for independent pairs (-1.4%), just below the Monte Carlo 1% quantile of 0.8018.
    - The rho 0.3 pair scores 0.8145 (z 1.3) and the rho 0.8 pair 0.9023 (z 25).
  - parse_bivariate: confirmed (`rawcall` 0.613 / 0.793 vs 0.809 both z-scored), owned by preproc.two_array_api.

### distance.lcss.constraint-itakura and distance.lcss.constraint-sakoe_chiba: wrong, score 3 (rubric 3: identifier hyperparameter ignored)

- **Bit identity.** Both constrained matrices are bit-identical to pyspi lcss in the 6 `same` runs. Every `dp` run has pyspi_eq_unmasked = 1, and all 16 `unmasked` runs are exact.
- **Against the masked LCSS** (the documented quantity), 13 of 16 (itakura) and 10 of 16 (sakoe_chiba) `dp` runs fail:
  - iid_gaussian T1024 pipeline pair (0,1): pyspi 0.8037 vs masked 0.0 (itakura) and 0.7061 (sakoe_chiba).
  - delayed_noise pairs differ by up to 0.28 under the band.
  - logistic_maps and (for the band) narrowband_f0-0.25 are the runs where the mask does not bind.
- **tslearn's own torch backend** equals the from-scratch masked reference exactly (`torch`, max_abs 0.0 for both units) and differs from the numpy value. This rules out a reference-side mask error.
- **Ignored mask: confirmed, new finding** (not in KNOWN_ISSUES). The trace-stage citation stands: `numpy.isfinite(bool mask)`, dtw_variants.py:2259. The release range 0.8.0-0.9.0 rests on the trace-stage reading of the tags; only 0.9.0 was executed.
- **Itakura degeneracy** (relevant if the mask is ever honoured): **confirmed**. The fraction is analytic. tslearn's parallelogram admits only the corner cell in the last row and the last column, so the masked LCSS is 0 if and only if |z_x[T-1] - z_y[T-1]| > eps. For independent z-scored pairs that fraction is 2(1 - Φ(1/√2)) = 0.4795. Probe `scratchpad/dtw_family_check/lcss_itakura_zero_frac.py`: the equivalence held on 60/60 random pairs at T1024 and 60/60 at T256, and the final-sample criterion gave 0.484 (se 0.0035) over 20000 pairs at T1024. The 400-pair Monte Carlo null at T1024 (`null-masked`) has 0.445 exact zeros, an estimate with se about 0.025 (1.4 se below 0.4795; the trace estimate was 48%), mean 0.449, SD 0.402. The `null-masked` statistical test passes for itakura only because that bimodal null is so wide. For sakoe_chiba it fails (masked null 0.715 +- 0.009 against pyspi 0.804-0.819, max_z 8.2), which shows again that the band is not applied.

### distance.softdtw: wrong, score 3 (rubric 3 names the softdtw constraints)

- **Scoring rule.** The 3 applies to `softdtw_constraint-itakura` and `softdtw_constraint-sakoe-chiba`. The representative identifier `softdtw` (units.yaml) would rate 10 on its own, so the CSV row "softdtw, wrong, 3" describes the unit's lowest-scoring identifiers, not the representative. The unit is scored on its lowest-scoring identifier, the same rule that gives `distance.bary.euclidean` 0 (there the lowest-scoring identifier is also the representative). A one-line statement of this rule in `audit/README.md` is left to synthesis.

- **`softdtw` itself.**
  - The 16 `dp` runs agree with Cuturi & Blondel Algorithm 1 (direct cost, logsumexp) to max_abs 2.3e-13 and max_rel 2.5e-14.
  - tslearn with gamma 0.001 and 0.1 agrees with the reference to <= 1.1e-14 relative, inside the analytic band dtw^2 - gamma log3 (2T-1) <= sdtw <= dtw^2.
- **Constraint identifiers.** The 32 `ident` runs are bit-identical to softdtw (KNOWN_ISSUES bug 3, cited, not new). All 8 `masked` runs differ from the documented constrained soft-DTW: by 2.5-22.6 for itakura and 57-1303 for sakoe-chiba.
- **Symmetry.** tslearn's soft_dtw is symmetric only to the last bits: unmirrored max|B - B.T| is 1.1e-13 at T1024. The mirrored pyspi matrix is exactly symmetric.
- **Candidates.**
  - Raw value, not the divergence: **confirmed**, as a property of the canonical, documented quantity (Cuturi & Blondel 2017 eq. 2 / Alg. 1; supplement.txt:305-310), not a code deviation. new_finding is false, as for the DTW and LCSS low-dynamic-range candidates above. The 'unsigned' keyword is defined by whether the sign indicates how fluctuations in A correspond to B (supplement.txt:126-128; GitBook glossary), not by range, so the negative values do not contradict it.
    - At T1024 every z-scored value is negative (-453.6 to -174.1), with self values around -712 to -718.
    - The divergence is >= 0 (11-549), and the Spearman correlation between raw value and divergence is 0.44-0.60.
    - The anti-correlated pair (-174.0) lies within 1.5 null SD of the null (-194.6 +- 13.6).
  - Dot-product cost: **confirmed and minor**. The error is 3.1e-16 relative on near-identical z-scored rows and 1.0e-8 relative (3.0e-7 absolute) with a +1e4 offset on the kernel path. Only the second fails the 1e-10 tolerance, and that case needs normalisation turned off.
  - parse_bivariate: confirmed, owned elsewhere.

### Trace-stage corrections and remaining issues

- The two remaining trace issues ask for the pyspi-vs-pyspi scaled_offset checks in `distance.dtw.constraint-None` and `distance.lcss.constraint-None` to be relabelled `same-lineage`. Both records already carried `reference_kind: same-lineage` when this stage started (file time 19:27), so no further edit was needed. The checks were run both ways:
  - `scaleinv-self`, same-lineage: pyspi on scaled_offset vs pyspi on bivariate_gaussian.
  - `scaleinv`, independent: pyspi vs a from-scratch reference on the z-scored draw.
- softdtw `units_and_range` says "symmetric in its arguments". This holds only to 1.1e-13 in tslearn. The mirroring makes the pyspi matrix exact. Nothing else in the trace fields was refuted.

## Open questions: answered

1. **Which tslearn version produced published pyspi results?** Not answerable from the code or the checks. The numbers here show what is at stake: with 0.9.0 the constrained LCSS equal lcss bit for bit, while a mask-honouring implementation gives different values. Examples: sakoe_chiba 0.706 vs 0.804 on iid pairs, and itakura exactly 0 for a fraction 2(1 - Φ(1/√2)) = 0.48 of independent pairs (400-pair Monte Carlo estimate 0.445, se about 0.025). Running tslearn 0.5.2/0.7.0 in a separate environment on the same data would settle the version boundary. Nothing was installed here.
2. **Is a 1-sample Sakoe-Chiba radius intended?** The code passes no radius, so it is tslearn's default. The checks show its effect: delays of 3 and 5 samples keep 72-73% of the Euclidean distance, and noise pairs keep 70-74%. Whether this is intended is a design decision for the maintainers. A T-proportional radius would change every value of this SPI.
3. **Should lcss_constraint-itakura be repaired?** A repair would make it degenerate: masked values are exactly 0 whenever the final samples differ by more than eps, which happens for a fraction 2(1 - Φ(1/√2)) = 0.48 of independent z-scored pairs (400-pair Monte Carlo estimate at T1024: 0.445, se about 0.025), because the parallelogram pins both end points. A repair would therefore need either a different window or dropping the identifier. lcss_constraint-sakoe-chiba would repair cleanly, to the Vlachos delta = 1 window.
4. **Should softdtw report the divergence?** The evidence favours it. Raw values are all negative and dominated by the length-dependent self term, and the divergence ranks pairs differently (Spearman 0.44-0.60). This is a design choice, not a contested standard: the raw value is the canonical Cuturi & Blondel 2017 quantity and the one the supplement describes (supplement.txt:307-308: 'computed via the tslearn package (with the default hyperparameter, γ = 1)'), while the divergence (Blondel, Mensch & Vert 2021) is a later, different quantity, so definition_under_review stays false. The 'unsigned' keyword, defined by whether the sign indicates how A and B co-fluctuate, is consistent with the raw value.
5. **parse_bivariate routing.** Already settled: the preprocessing_base family confirmed it as preproc.two_array_api (`preproc.two_array_api|scaled_offset|256|pipeline|spi-dtw`). Here it was re-measured for dtw, lcss and softdtw (`rawcall`, exact).
6. **Vlachos 2002 definition numbering and Sakoe-Chiba 1978 equation/table numbers.** Not checked against the PDFs; no copy is available in this environment. The numerical content (S1 = LCSS/min(n,m); the symmetric P = 0 form with diagonal weight 2 and normalisation by n+m) is what was implemented and compared. Reading the two papers would settle the numbering.
7. **Itakura 2-decimal rounding at T = 1024/4096.** No differences: 0 cells differ at T 64/256/1024/4096. For equal lengths every bound is a half-integer, so the rounding is exact.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls come from `audit/profiles/summary.json` and shares are percentages of the profiled `compute_spi` frame at 8x1024. Nothing in pyspi was changed and no micro-benchmark was written for this family. Not in the retained EEG selection.

### Walls

| unit | 8x1024 | 32x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|
| dtw | 0.187 s | 3.39 s | 47.5 s | measured |
| dtw_constraint-itakura | 0.183 s | 3.38 s | 46.6 s | measured |
| dtw_constraint-sakoe-chiba | 0.039 s | 0.70 s | 9.49 s | measured |
| lcss | 0.349 s | 6.18 s | 87.8 s | measured |
| lcss_constraint-itakura | 0.432 s | 7.95 s | 113.2 s | measured |
| lcss_constraint-sakoe-chiba | 0.338 s | 6.13 s | 88.4 s | measured |
| softdtw (3 identifiers) | 2.54 s fresh / 7.77 s family-warm | not run | 626 s / 1916 s | extrapolated-pairs-from-8 |

**Complexity: O(P^2 T^2).** Each ordered pair runs a numba DP over the full T x T grid plus a T x T mask; TimeWarping is Undirected over `Directed.multivariate`, so both orders are computed. The measured T_exponent between 8x256 and 8x1024 (0.8 dtw, 1.2 lcss, 1.8 softdtw, -0.2 sakoe) understates T^2 for two reasons:

- a fixed per-call Python cost;
- a ~50 ms overhead visible at 8x256 in several units.

P_exponent is 2.07-2.10, and the pair term dominates at 118. On the first call in a process, tslearn's `@njit(nogil=True)` kernels compile, because they have no `cache=True`: `t_cold_s` is 4.16 s for dtw against 0.19 s warm. That cost is per process, not per window.

### Where the time goes

- **dtw: `tslearn/metrics/_dtw.py:dtw` self time** (numba `_njit_dtw`: mask build and accumulated matrix, hidden from cProfile).
  - Shares: 92.5% (None), 88.7% (itakura), 83.0% (sakoe).
  - `tslearn/backend/backend.py:instantiate_backend` takes the rest: 5.6% / 8.5% / 13.4%. It formats each input array with `str()` to detect a torch tensor; numpy arrayprint alone is 4.6% / 7.0% / 10.9%.
  - For itakura the hidden numba frame includes a per-call mask build: `tslearn/metrics/_dtw.py:_dtw_generic` (line 398) calls `_njit_compute_mask` inside every `_njit_dtw` call. A scratch timing at T=1024 (NUMBA/OMP threads 1, tslearn's njit functions called directly, minimum of 20 calls; not saved under `micro/`) gives:
    - itakura: mask 1.74 ms, DP with the mask cached 1.59 ms (same value), full `_njit_dtw` 3.48 ms, public `dtw()` 3.56 ms. summary.json gives 0.183 s / 56 pairs = 3.27 ms per call. The mask build is about half of each itakura call.
    - None: mask 0.013 ms, DP 3.32 ms. sakoe_chiba: mask 0.015 ms, DP 0.63 ms. The mask build is negligible for these two.
- **lcss: `tslearn/metrics/dtw_variants.py:lcss` self time** (numba `njit_lcss_accumulated_matrix`).
  - Shares: 87.3% / 79.4% / 92.0%.
  - `compute_mask`: 5.4% / 14.2% / 2.7%.
  - `instantiate_backend`: 5.4% / 4.8% / 4.0%.
- **softdtw.**
  - `tslearn/metrics/softdtw_variants.py:SoftDTW.compute` is 51.9% self (numba recursion).
  - `SoftDTW.__init__` -> `SquaredEuclidean.compute` -> sklearn `_euclidean_distances` is 45.4%: sklearn 29.3% plus astype casts 10.6%.

### Opportunities

- **dtw_family-1: batched numba kernel over the unordered pairs (all six dtw/lcss units).** Call tslearn's njit accumulated-matrix kernels directly from a loop over i < j and build the constraint mask once per window. This skips the per-call Python layer.
  - Serial gain is small for dtw, dtw_constraint-sakoe-chiba and the three lcss units: **~1.1-1.2x** (Amdahl from 5-19% removable).
  - For dtw_constraint-itakura it is **~2x**. Building the mask once per window also removes the per-call Itakura mask build, 1.74 of 3.48 ms per call (Amdahl about 2.4x together with the 8.5% Python layer).
  - The real value is a `prange` over pairs (nogil kernels). That is counted under the cross-cutting `xc-window-parallel`, and the one-direction halving under `xc-undirected-upper`.
  - The unmirrored probes show bit-symmetric dtw and lcss, so upper-only reproduces the stored matrix.
  - Unverified-bit-identical (same kernel and per-cell arithmetic). Effort M.
- **dtw_family-2: banded DP for constrained DTW.** Visit only the admissible cells: 3T-2 cells for Sakoe-Chiba radius 1 (0.29% of the grid at T=1024), and the parallelogram rows for Itakura (about 1/3). Cells outside the mask stay +inf, as they do now, so the output should not change (unverified-bit-identical).
  - dtw_constraint-sakoe-chiba: **~5x** alone (Amdahl 5.9x from the 83.0% kernel share). Together with dtw_family-1 it is 20-100x.
  - dtw_constraint-itakura: **~2.5x**. With the mask cached, the Itakura DP takes 1.59 ms against 3.32 ms unconstrained. The two walls are nearly equal (0.183 vs 0.187 s at 8x1024) only because the per-call mask build (1.74 ms) adds back what the mask saves. A banded DP that derives the parallelogram row bounds directly removes both costs. One that still builds the T x T mask per call would gain only about 1.1x. It overlaps dtw_family-1 on the mask part; both together give about 2.5-3x.
  - Not applicable to lcss while tslearn ignores the lcss mask (see below).
- **dtw_family-3: one soft-DTW computation for the unit's three identifiers (~3x, effort S).**
  - `SoftDynamicTimeWarping.bivariate` (`pyspi/statistics/distance.py:206-208`) calls `soft_dtw(z[i], z[j])` without the constraint, so the three identifiers repeat one computation. Family-warm increments at 8x1024 are 2.58 / 2.59 / 2.60 s.
  - Unverified-bit-identical with today's output. It stops being valid once the constraint is honoured (the correctness finding, score 3).
- **dtw_family-4: fuse the squared-difference cost into the soft-DTW recursion (1.5-1.8x).** This replaces sklearn's dot-product expansion and the casts, which are 45.4% of the call. The Amdahl bound is 1.83x, reached only if the fused squared difference costs nothing inside the recursion. Unverified-within-tolerance: sklearn computes clamp(x^2 + y^2 - 2xy, 0), not (x - y)^2. Effort M.
- **dtw_family-5: lcss constrained units reuse the unconstrained lcss matrix (1000-10000x, effort S; CONDITIONAL on lcss being computed in the same run, which holds in the default config, otherwise no gain).** tslearn's numpy backend ignores the boolean mask (`numpy.isfinite(mask)` is True everywhere), so both constrained units equal lcss exactly today. This is listed only as a statement about today's redundant computation. Fixing the mask is a semantic change, and it would also make the Sakoe-Chiba lcss band O(T), so it is cheaper as well. On the two constrained lcss units the records list dtw_family-5 in `subsumed_by` for dtw_family-1, and the cross-cutting `xc-undirected-upper` lists it too: once those units copy lcss, no per-pair computation is left, so build_csv credits neither avenue a saving there (README "Optimisation columns", ranking rule 4).

### Seed items refuted or narrowed

- **"One direction" is the cross-cutting `xc-undirected-upper`.** symmetry_observed is bit-symmetric for dtw and lcss. softdtw is symmetric only to 1.1e-13, but the stored matrix already uses only the upper triangle. The cross-cutting row is recorded `independent_of` dtw_family-2 and dtw_family-3 because they act on different factors (`audit/profiles/crosscutting.yaml`): it halves the number of bivariate calls, dtw_family-2 cuts the cost of each call, and dtw_family-3 serves one identifier from another identifier's per-pair result, which the remaining i<j loop still reads once per pair; so it keeps its own ~2x on their residual.
- **"One numba prange kernel over all unordered pairs" is refuted as a serial win for dtw, dtw_constraint-sakoe-chiba and the three lcss units.** Their per-call layer is only 5-19% of the call, and their value is parallelism, counted under `xc-window-parallel`.
- **It holds as a serial win for dtw_constraint-itakura (~2x).** The batched kernel builds the Itakura mask once per window instead of once per call, and that build is about half of each call (scratch timing above). The pstats cannot show it because the mask is built inside the njit frame (`dtw` self 88.7%).

### Shared versus incremental cost

Family-warm increments at 8x1024: dtw 0.19 s, itakura 0.19 s, sakoe 0.04 s, lcss 0.35 s, lcss-itakura 0.44 s, lcss-sakoe 0.35 s, and softdtw 2.58 + 2.59 + 2.60 s.

Nothing is shared today. The only sharing available without changing values is:

- softdtw's three identifiers (dtw_family-3);
- the two constrained lcss units copying lcss (dtw_family-5), which lasts only as long as tslearn's mask bug.
