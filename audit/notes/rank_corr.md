# Family `rank_corr`: define, trace and check

Units: `basic.spearmanr` (identifiers `spearmanr`, `spearmanr-sq`) and `basic.kendalltau` (identifiers `kendalltau`, `kendalltau-sq`).
Records: `audit/records/basic.spearmanr.yaml`, `audit/records/basic.kendalltau.yaml`.
Environment probed: scipy 1.17.1, numpy 2.5.4, project venv. Probe scripts are in the session scratchpad under `rank_corr/` (probe1–probe4) and, for the post-review corrections, `rank_corr_fix/` (p_labels.py, p_recip.py, p_recip2.py). Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

The trace stage did not run any numeric checks (they are in "## Checks" below). The probe outputs quoted below are small read-only confirmations of what the code does. They do not verify correctness. The regression goldens are snapshots of current output and are not used as a reference here. The verdicts and scores are in "## Checks" and in the records, and candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in "## Checks".

## Shared pipeline (both units)

1. `Calculator._load_yaml` (`pyspi/calculator.py:230-267`, instantiation at `:256`, HEAD a1f20c7) instantiates `SpearmanR(squared=True/False)` and `KendallTau(squared=True/False)` from `pyspi/config.yaml:74-97`. The `labels:` block in the YAML is never read, so each instance's labels come from the class attribute.
2. `Data` preprocessing (`pyspi/data.py:184-195`): optional `scipy.signal.detrend` (default off), then `zscore(axis=1, ddof=1)` (default on). Any NaN after this step raises `ValueError` (`data.py:199-203`). A constant channel z-scores to NaN, so it is rejected here and never reaches SciPy's `ConstantInputWarning` path.
3. `Calculator.compute` → `Undirected.multivariate` (`pyspi/base.py:132-138`) → `Directed.multivariate` (`base.py:88-94`). This loops over **all ordered pairs** (i ≠ j) and calls `bivariate(data, i, j)`. Then `A[li] = A.T[li]` overwrites the lower triangle with the upper one, and `calculator.py:332` sets the diagonal to NaN (`:329` calls `multivariate`, HEAD a1f20c7).
4. `bivariate` (`basic.py:151-157`, `175-181`) slices `data.to_numpy()[[i, j]]`, which gives two `(T, 1)` arrays, and calls SciPy. The `-sq` variants square the scalar result.

Effect of preprocessing: both statistics depend only on ranks. A positive affine map (z-score) leaves them unchanged, and `tests/README.md` line 131 records that n1d0 and n0d0 give identical values. Probe 3 found no change in 500 tied integer datasets after z-scoring. Detrending subtracts a fitted line, which is not monotone, so it changes the ranks. On `trending_T1024`, pair (0,1) gives Spearman −0.0368 raw and 0.2570 detrended. Pair (2,3) gives −0.9826 raw and 0.7908 detrended.

## basic.spearmanr

**Documented.** Supplement §1.1.3, p.4 (supplement.txt:172-178): "Spearman's ρ is a nonparametric measure of rank correlation … capture non-linear (but monotonic) relationships … implemented via function spearmanr in SciPy [6], and has a value in [−1, 1]." Keywords: undirected, nonlinear, signed, bivariate, contemporaneous. The GitBook text (`basic-statistics.md:126-140`) is the same. `table-of-spis.md:35` lists `spearmanr` and `spearmanr-sq`. The supplement enumerates only "17. spearmanr" and never defines the squared variant.

**Canonical.** Spearman (1904), Am J Psychol 15:72–101. ρ_S is the Pearson correlation of ranks. With ties the standard choice is mid-ranks (Kendall & Gibbons 1990, ch. 3). The shortcut 1 − 6Σd²/(n(n²−1)) holds only when there are no ties.

**Implemented.** `scipy.stats.spearmanr` (`_stats_py.py:5168`):
- column-stacks the two series (`:5358`);
- returns NaN if n ≤ 1 (`:5364`), if an input is constant (`:5370-5378`), or if there is a NaN (`:5389-5399`);
- computes average ranks with `rankdata` (`:5405`; `_rankdata` gives mid-ranks at `:10068-10070`);
- computes the Pearson correlation of the ranks with `np.corrcoef` (`:5406`; numpy `_function_base_impl.py:3029-3043`, which includes a clip to [−1, 1]);
- returns `rs[1,0]` (`:5419-5422`).

pyspi discards the p-value.

Probe 1 compared pyspi against a from-scratch mid-rank Pearson:
- continuous data: 0.4279951980792317 in both;
- tied data: 0.35395114622767243 in both.

**Deviations.**
- Values: none, either against the documentation or against the canonical tie-corrected ρ_S.
- `spearmanr-sq` returns ρ_S². It is listed in the GitBook table but has no definition in the supplement, and it is not a standard named quantity.
- Numerics: with ties, `spearmanr(x, y)` and `spearmanr(y, x)` can differ in the last bits. The ranks of x and y have different mid-rank SDs, and `np.corrcoef` divides by the two SDs one after the other (`_function_base_impl.py:3036-3038`), so swapping the inputs changes the rounding. On continuous data both rank vectors are permutations of 1..n, the SDs are bit-identical, and the order does not matter. Probe `p_recip.py` (2000 random pairs of `np.round(N(0,9))`, n in [20, 400)) found 691 mismatches on tied data, with a maximum of 5.55e-17 (at most 2 ulp of the smaller value), and 0 on continuous data. The Undirected mirror hides this in the output matrix.
- Metadata: the class labels contain `linear` (`basic.py:140`) while the documentation and `config.yaml` say `nonlinear`. This is a candidate finding. In a fresh interpreter `SpearmanR().labels` is `['basic', 'unordered', 'rank', 'linear', 'undirected', 'signed']`. After a `squared=True` instance has been created, a new default instance carries `[..., 'signed', 'unsigned', 'signed']`, which is the bug 4 mutation. An earlier version of this note quoted the mutated list as if it came from a fresh instance. The class labels carrying both `signed` and `unsigned` is already documented as tests/README bug 4 / REVIEW M8.

**Proposed checks** (details in the record):
1. Kernel check against a from-scratch mid-rank Pearson on `bivariate_gaussian`, `nonlinear_dependence`, `contaminated` and `iid_gaussian`, plus a tied variant made by rounding inside the check. Tolerance 1e-12.
2. Statistical check against the bivariate-normal population value ρ_S = (6/π)·arcsin(ρ/2), which is 0.28756 and 0.78594 for ρ = 0.3 and 0.8.
3. Exact invariance between `scaled_offset` and `bivariate_gaussian`.
4. `pipeline-detrend` on `trending` against a from-scratch reference that detrends with lstsq.
5. Symmetry and direction check on `delayed_noise` and `var2_bivariate`. Exact symmetry holds at pipeline level only.
6. Reciprocal check on tied data: a few ulp at kernel level, exact at pipeline level.
7. Qualitative check that ρ_S ≈ 0 for the non-monotone dependence in `nonlinear_dependence`.
8. Exact check that `-sq` equals the square of the signed value.

## basic.kendalltau

**Documented.** Supplement §1.1.4, p.4: "Kendall's τ [1] … becoming more mathematically tractable in the event of ties [7]. The method is implemented via function kendalltau in SciPy [6], and has a value in [−1, 1]." Reference [1] is Kendall 1938, Biometrika 30, 81. The documentation does not name a variant. `kendalltau-sq` appears only in the GitBook table.

**Canonical.**
- Kendall (1938) defines τ_a = (C − D)/(n(n−1)/2).
- Kendall (1945, Biometrika 33:239–251) defines the tie-adjusted τ_b = (C − D)/√((n0 − n1)(n0 − n2)).
- Knight (1966) gives the O(n log n) counting algorithm.

**Implemented.** `scipy.stats.kendalltau` (`_stats_py.py:5530-5535`, with the `_axis_nan_policy` wrapper) runs with the default `variant='b'`:
- ravels the inputs (`:5650`);
- converts to dense ranks (`:5671-5678`);
- counts discordant pairs with the compiled `_kendall_dis`, a Fenwick tree (scipy v1.17.1 `_stats.pyx`, about line 83);
- counts ties (`:5662-5689`) and returns NaN if one series is entirely tied (`:5691`);
- computes `con_minus_dis` (`:5699`) and τ_b = `con_minus_dis / sqrt(tot−xtie) / sqrt(tot−ytie)` (`:5701`);
- clips the result (`:5710`).

Probe 1 compared pyspi against a from-scratch O(n²) τ_b:
- continuous data: pyspi, τ_b and τ_a are all 0.2897959183673469;
- tied data: pyspi 0.30084768027508835 equals τ_b, while τ_a is 0.22285714285714286.

**Deviations.**
- Against the documentation: none in value, since SciPy's default is what the documentation names.
- Against the cited canonical paper (Kendall 1938, τ_a): the implementation is τ_b. The two agree when there are no ties and diverge when there are ties. On coarse integer data at T = 4096, probe 3 gave τ_b 0.38386 vs τ_a 0.33287.
- Against τ_b: none.
- Numerics: swapping x and y swaps the order of the two divisions at `:5701`. Probe 3 found 666 of 2000 tied random pairs where the reciprocal calls differed, by at most 1.1e-16. The Undirected mirror hides this in the output matrix. This is not specific to Kendall. An earlier version of this note contrasted it with "0 mismatches for Spearman", but that Spearman probe used only continuous data. On tied data Spearman shows the same effect (see above). The re-probe `p_recip.py` gave 691 of 2000 for each statistic, on different pairs.
- Metadata: same as Spearman (`linear` vs `nonlinear` is a candidate; bug 4 is already documented).

**Proposed checks:**
1. Kernel check against a from-scratch O(n²) τ_b with tied variants. It must match τ_b, not τ_a.
2. Statistical check against Greiner's relation τ = (2/π)·arcsin ρ, which gives 0.19397 and 0.59033.
3. Exact invariance check.
4. `pipeline-detrend` on `trending`.
5. Symmetry and direction check.
6. Reciprocal check to within a few ulp at kernel level, and exact at pipeline level.
7. Exact `-sq` identity.
8. Optional external cross-check against R `cor(method="kendall")`.

## Briefing hypothesis

"kendalltau is tau-b; spearman is Pearson on ranks; both symmetric so the Undirected double computation is pure waste."
- **Confirmed by code and probes.** τ_b comes from `_stats_py.py:5701` and matches the from-scratch τ_b on tied data. Spearman is `rankdata`(average) followed by `corrcoef`, and matches the from-scratch mid-rank Pearson.
- **Double computation.** This is already documented at `EEG_OPTIMIZATION_NOTES.md:132`, which notes that the generic Undirected path computes both ordered directions before one triangle is overwritten. The new part here is the count for this family: with P = 4, probe 2 counted 12 `bivariate` calls where 6 would be enough, and the output was exactly symmetric. With ties, the reciprocal values of both Spearman and Kendall can differ by a few ulp, so the duplicate work is not literally bit-identical. It is still discarded by the mirror, so for these values it is pure waste.

## Infrastructure observation (not specific to this family)

When `parse_bivariate` is called with raw arrays (`base.py:33-36`), it z-scores only the first array (via `Data.set_data`) and appends the second raw (`data.py:230-234`). Probe 4: row 0 had mean/std 0/1, row 1 had 4.38/4.763. This has no effect on rank statistics, but it could affect other bivariate SPIs that are called on arrays directly. The underlying `Data.add_process` behaviour is already locked by `tests/unit/test_data.py:74-88` (`test_add_process_appends_raw_unnormalised`), and unit `preproc.two_array_api` owns it. What it means for SPI values reached through `parse_bivariate` is not in KNOWN_ISSUES.

## Open questions

1. Does the EEG export carry ties (integer ADC counts or limited decimal precision)? If it does, τ_b and τ_a differ materially, and the cited reference (Kendall 1938, τ_a) does not describe what is computed.
2. Should the `linear` class label be changed to `nonlinear`, the keyword the documentation and config use? Or should labels be derived from config? This is a metadata decision, related to REVIEW M9.
3. Is `-sq` (ρ², τ²) intended as an "unsigned strength" variant? It has no documented definition.
4. Data with more than one realisation (R > 1) would pass `(T, R)` blocks to `spearmanr`, which then returns a matrix instead of a scalar. This is not exercised by the Calculator and is unconfirmed (hypothesis).
5. The array-input normalisation asymmetry in `parse_bivariate` (above) needs a decision by the infrastructure owner.

## Checks

Files: `audit/checks/manifest/rank_corr.json` (134 runs, 67 per unit), `audit/checks/refs/rank_corr.py` (comparison venv, no `scipy.stats` import), `audit/checks/probes/rank_corr_probes.py` (project venv), `audit/checks/r/rank_corr.R` (R 4.6.1 `stats::cor`). Results: `audit/checks/out/results/rank_corr.json`, 134 pass, 0 fail, 0 without reference. Run ids below are written without the `basic.<unit>|` prefix where the unit is clear.

Corrections to trace-stage fields made at this stage: `pyspi/calculator.py` citations updated to HEAD a1f20c7 (`_load_yaml` is now `:230-267`, with instantiation at `:256`; `compute` calls `multivariate` at `:329` and fills the diagonal at `:332`; before, they were `:239-246`, `:294`, `:297`). `tests/README.md` line 112 is now line 131. The `config.yaml:74-97` and `:88-93` citations named in the remaining trace issues already pointed at the cited lines in both records and in this note, so nothing changed there. No trace-stage value claim was refuted.

How the checks are built:
- **Kernel path.** `Data(normalise=False)`. Each ordered pair is compared separately with the from-scratch kernels in `_kernels_basic_misc` (`spearman`: average ranks by `np.sort`/`np.searchsorted`, then Pearson; `kendall_tau_b`: O(n²) sign products). Criterion `exact`, atol 1e-12, rtol 0.
- **Tied input.** `probes/rank_corr_probes.py:ties_kernel` / `ties_pipeline` round x to 1 decimal with `np.round(x, 1)`, and the reference applies the same rounding. That leaves 13 to 80 distinct values per row (`ref_min_distinct_values`), so tau-b's tie correction and the mid-ranks are both exercised.
- **Pipeline path.** The pyspi matrix is computed on z-scored data, and the reference is computed on the **raw** rows. A pass therefore also checks rank invariance to z-scoring. Separate qualitative runs check that the pipeline and kernel matrices are bit-identical.
- **Unmirrored calls.** `probes/...:directed_ties` and `directed_kernel` call `Directed.multivariate` directly, so reciprocal calls are visible before the Undirected mirror hides them.
- **Population value.** Analytic expectation under the bivariate normal. Spearman: Moran 1948, E r_S = 6/(π(n+1))[asin ρ + (n−2) asin(ρ/2)], which is exact for finite n. Kendall: E τ = (2/π) asin ρ. Analytic SE: exact null SD for ρ = 0; for ρ ≠ 0, the Fisher-z delta method (Bonett & Wright 2000 for Spearman, Fieller, Hartley & Pearson 1957 for Kendall). A scratch Monte Carlo (3000 replicates at n = 64, 256 and 1024, ρ = 0, 0.3 and 0.8; session scratchpad `rank_corr_check/mc2.py`) gave empirical SD / analytic SE between 0.94 and 1.00, so the bands are slightly conservative. The Monte Carlo means matched both expectations to within 0.002. The stored `se` is 1.5 × SE, so compare.py's 2·se band equals the record's 3-SE criterion. `ref_max_z_1se` gives z in units of one SE. The analytic SE is a substitution for what the trace stage proposed. Kendall's proposed check named jackknife standard errors, and Spearman's named the Fieller-Hartley-Pearson var z = 1.06/(n−3). The analytic form was used because it is closed-form, deterministic, and validated by the Monte Carlo above. For Spearman, the Bonett-Wright SE is 1.11 times the FHP SE at ρ_S = 0.786 and 0.99 times it at ρ_S = 0.288. With the FHP SE the largest population z would be 1.71 (pair (2,3), T = 4096) instead of 1.54, still inside the band. The 3-SE band was declared in `proposed_checks` before any run and was not changed afterwards.

### basic.spearmanr: verdict matches, score 10

Rubric anchor 10: matches the canonical tie-corrected ρ_S to floating-point precision on an independent reference and an external package, and matches the supplement's description (scipy `spearmanr`, range [−1, 1]).
- **Kernel, continuous input.** `bivariate_gaussian` T = 64/256/1024/4096, `nonlinear_dependence` and `contaminated` T = 256/1024/4096, `iid_gaussian` T = 256/1024: 12/12 pass, max_abs ≤ 1.1e-16.
- **Tied input** (8 runs, kernel and pipeline): max_abs ≤ 1.1e-16. On the dependent pairs the untied shortcut 1 − 6Σd²/(n(n²−1)) differs from pyspi. In `|bivariate_gaussian|1024|kernel|ties-vs-untied-formula`, pair (2,3) gives pyspi 0.79011 against 0.79032 from the shortcut (min difference 2.09e-4). On tied `nonlinear_dependence` the difference reaches 1.0e-2. pyspi therefore uses mid-rank Pearson, not the shortcut.
- **External.** R `stats::cor(method="spearman")` against pyspi: max_abs 5.6e-17 (continuous), 3.5e-18 and 0 (tied) in `|...|R-cor` and `|...|R-cor-ties`.
- **Population value.** `|bivariate_gaussian|T|pipeline|population`: max_z_1se is 1.26, 0.88, 1.04 and 1.54 at T = 64, 256, 1024 and 4096. For pair (0,1), |error| is 0.150, 0.0071, 0.0253 and 0.0065 against SE 0.120, 0.059, 0.029 and 0.015. The error tracks the SE, which is consistent with convergence. Null `iid_gaussian`: max_z_1se 2.94 (T = 1024) and 1.96 (T = 4096). The T = 1024 run is a near-miss. Pair (0,2) has r_S = 0.09193 against null SD 0.03127, so it passes only because the declared band is 3 SE (2.94 against 3.00). With 10 pairs treated as independent, the chance of max |z| > 2.94 is about 3%. The same pair has Pearson z = r√(n−1) = 2.68 in the data, and on this dataset pyspi equals the from-scratch kernel (`|iid_gaussian|1024|kernel|exact`, max_abs 6.9e-18). The excursion therefore belongs to the sample and does not come from pyspi.
- **Invariance.** The pipeline matrix is bit-identical to the kernel matrix in 5/5 runs. `scaled_offset` is bit-identical to `bivariate_gaussian` on both paths.
- **Detrend.** `pipeline-detrend` matches the lstsq-detrend reference (max_abs 2.2e-16). On `trending` T = 1024, pair (2,3) is −0.98255 on the raw pipeline and 0.79084 after detrending.
- **Squared variant.** `spearmanr-sq` equals the kernel value squared (max_abs 2.2e-16) and is bit-identical to the signed matrix squared (`sq-relabel`). Observed range [1.2e-6, 0.653].
- **Symmetry.** In every Calculator-path run, max|A − A.T| = 0. With the mirror removed, on tied `bivariate_gaussian` T = 1024, 2 of 6 pairs differ by 5.55e-17 (1 ulp; `recip-ties`); on tied `contaminated` T = 4096, 1 of 1 differs by 1.39e-17; on continuous data, 0 of 6 differ.
- **Direction.** None. On `var2_bivariate`, A[0,1] = A[1,0] = 0.17152, with the sign of the zero-lag correlation 0.2107 (sign agreement 1.0). On `delayed_noise`, every pair is within 1.71 null SD of 0 even though the series are exact lagged copies.

### basic.kendalltau: verdict matches, score 10

Rubric anchor 10: matches the tie-corrected τ_b (Kendall 1945) on an independent O(n²) reference and on R, and matches the supplement's "kendalltau in SciPy".
- **Kernel, continuous input.** 12/12 pass, max_abs ≤ 1.1e-16.
- **Tied input.** 8/8 pass, max_abs ≤ 1.1e-16.
- **τ_b, not τ_a.** In `|bivariate_gaussian|1024|kernel|ties-vs-untied-formula`, pyspi gives 0.18134 and 0.61192, equal to τ_b (max_abs 8.7e-19). τ_a gives 0.17628 and 0.59470. At T = 4096, pyspi gives 0.20453 and 0.59646 against τ_a 0.19881 and 0.57948.
- **External.** R `cor(method="kendall")` against pyspi: max_abs 2.2e-19 (continuous), 1.1e-16 and 0 (tied). R against the from-scratch kernel: at most 1.1e-16. This is consistent with R's `cor` computing τ_b.
- **Population value.** max_z_1se is 1.27, 0.79, 1.06 and 1.41 at T = 64, 256, 1024 and 4096. For pair (0,1), |error| is 0.105, 0.0014, 0.0176 and 0.0052 against SE 0.082, 0.040, 0.020 and 0.010. Null: 2.95 (T = 1024) and 1.97 (T = 4096). The T = 1024 run is the same near-miss on the same pair (0,2): τ = 0.06164 against null SD 0.02087, which is 2.95 against the 3-SE band and about a 3% event over 10 pairs. On this dataset pyspi equals the kernel (`|iid_gaussian|1024|kernel|exact`, max_abs 0). Spearman and Kendall on the same draw are strongly correlated, so the two near-misses are one sample event, not two.
- **Invariance.** Same results as Spearman: pipeline bit-identical to kernel, `scaled_offset` bit-identical to `bivariate_gaussian`.
- **Detrend.** max_abs 3.5e-18 (T = 256) and 1.7e-18 (T = 1024). On `trending` T = 1024, pair (2,3) is −0.87879 raw and 0.59666 detrended.
- **Squared variant.** `-sq` equals the kernel value squared (max_abs 1.7e-16) and is bit-identical to the signed value squared. Range [1.4e-7, 0.374].
- **Symmetry.** Pipeline asymmetry is 0 everywhere. With the mirror removed, on tied input 1 of 6 pairs differs by 1.11e-16 (1 ulp); continuous input gives 0 of 6, and tied `contaminated` T = 4096 gives 0 of 1.
- **Direction.** None. On `var2_bivariate`, A[0,1] = A[1,0] = 0.11514. On `delayed_noise`, every pair is within 1.71 null SD of 0.
- **Interpretation.** The supplement cites Kendall 1938 (τ_a), while the value is τ_b. The two are identical without ties. With ties they differ by 5e-3 to 1.7e-2 on the rounded data here. The documented implementation ("kendalltau in SciPy") is followed exactly, so this is recorded in `what_changed` and not scored as a deviation.

### Candidates (both units)

| Candidate | Outcome | Evidence |
|---|---|---|
| Class label `linear` contradicts the documented and config keyword `nonlinear` | **confirmed**; not in KNOWN_ISSUES (`new_finding: true`, metadata only) | `|iid_gaussian|64|kernel|labels`: fresh `SpearmanR()`/`KendallTau()` labels contain `linear` and not `nonlinear`, and so do the Calculator-built instances |
| Bug 4, `signed` + `unsigned` (cited, already documented) | reproduced | Same run. In a Calculator built from a two-config YAML, the `-sq` and plain instances share **one** list object that carries both `signed` and `unsigned`. A default instance created after a squared one carries `signed` twice. Already documented: tests/README.md bug 4 and REVIEW M8. |
| Double work: P(P−1) calls where P(P−1)/2 suffice | confirmed (quantification of EEG_OPTIMIZATION_NOTES.md:132, already documented) | `|bivariate_gaussian|256|pipeline|call-count`: 12 calls for P = 4, output exactly symmetric |
| Reciprocal last-bit differences on tied data | confirmed, value-neutral in the pipeline | `recip-ties` / `recip-continuous` runs above |
| `nonlinear` covers monotone dependence only | confirmed | `|nonlinear_dependence|{1024,4096}|kernel|monotone-only`. Spearman \|r\| ≤ 0.0295 and Kendall ≤ 0.0155 (bands 0.094 and 0.063 at T = 1024), while the dependence witnesses are ρ_S(\|x0\|, x1) = 0.974, τ = 0.866, and both statistics give −1.0 for (x2², x3²). The supplement itself says "non-linear (but monotonic)". |
| Spearman with R > 1 realisations | confirmed | `basic.spearmanr|bivariate_gaussian|1024|kernel|realisations`: `bivariate` returns a 4×4 matrix and `multivariate` raises. Kendall (not a trace candidate, checked anyway): pools the realisations, giving τ_b of the concatenated samples, 0.17637 against 0.15079 and 0.20172 for the two realisations separately |
| Kendall citation τ_a vs computed τ_b | confirmed | `ties-vs-untied-formula` runs above |
| `parse_bivariate` array input z-scores only the first array | confirmed, no effect on these units | `|scaled_offset|256|kernel|array-input`: row 1 is left raw (mean 1.0117, std 1.9969) and the value is identical to Data input. Owned by `preproc.two_array_api`. |

Nothing was refuted. No check failed.

## Open questions: answered

1. **Does the EEG export contain ties?** Not answered here. No EEG data was examined. What the checks do show: at 1-decimal resolution on unit-variance data (13 to 80 distinct values per row), τ_b and τ_a differ by 5e-3 to 1.7e-2, while the mid-rank Spearman and the untied shortcut differ by 2e-4 to 1e-2. pyspi computes the tie-corrected forms in both cases, so ties change *which canonical form the citation describes*, not whether the value agrees with a reference. To settle it, count distinct values per channel in one stored EEG window, before and after the workflow's detrend and z-score. Detrending breaks exact ties, because the fitted line differs per sample. z-scoring preserves them.
2. **Should `linear` become `nonlinear`, or should labels come from the config?** This is a metadata decision, not a numerical one. The checks confirm the mismatch: the class says `linear`, while the config and supplement say `nonlinear`, and the config `labels:` block is never read. They also confirm that both statistics respond only to monotone dependence (`monotone-only` runs), which is what the supplement's "non-linear (but monotonic)" says. Related to REVIEW M9. A fix belongs with the bug 4 fix, since both live in the same `__init__`.
3. **Are the `-sq` variants documented "unsigned strength" SPIs?** Numerically they are exactly the square of the signed statistic (`sq-relabel`, bit-identical) with range [0, 1]. Neither the supplement nor the GitBook defines them, and nothing in the checks can settle the intent. Only the maintainers can answer it.
4. **Data with R > 1 realisations.** Confirmed for Spearman: `bivariate` returns a 2R×2R matrix and `multivariate` raises. For Kendall, the realisations are silently pooled and τ_b is computed over the concatenated samples, which is neither the per-realisation value nor their average (`realisations` runs). The Calculator path never builds R > 1 from a 2-D array, so EEG outputs are unaffected.
5. **Asymmetric normalisation in `parse_bivariate`.** Confirmed: row 0 is z-scored and row 1 is appended raw. Rank statistics are unaffected, with identical values via array input and via Data input. The decision belongs to the infrastructure family (`preproc.two_array_api`).

## Optimisation

Workflow 3, group basic. Walls are from `audit/profiles/summary.json` (quiet machine, all shapes measured, `measure_mode: measured` at 118x1024). Percentages are shares of the profiled `compute_spi` call. Nothing in pyspi was changed. Micro-benchmark: `audit/profiles/micro/opp_rank_corr-1.py` / `.json`.

| unit | mode | 8x1024 | 32x1024 | 118x1024 (measured) |
|---|---|---|---|---|
| basic.spearmanr | fresh (spearmanr-sq) | 0.021 s | 0.356 s | 4.97 s |
| basic.spearmanr | family-warm (both identifiers) | 0.041 s | 0.715 s | 10.00 s |
| basic.kendalltau | fresh (kendalltau-sq) | 0.018 s | 0.322 s | 4.51 s |
| basic.kendalltau | family-warm (both identifiers) | 0.037 s | 0.642 s | 9.03 s |

**Complexity.** Both units call scipy once per ordered pair: P(P-1) = 13806 calls at 118 channels, through `pyspi/base.py:Directed.multivariate`, and the lower triangle is then overwritten by the upper one. Each call costs O(T log T). The measured P_exponent is 2.06; the T_exponent of 0.19-0.21 shows that per-call overhead still dominates at T <= 1024. At 118x1024 the pair term is the whole cost: 0.36 ms (Spearman) and 0.33 ms (Kendall) per call.

**Hotspots.**
- Spearman (`basic.spearmanr__8x1024.pstats`): `scipy/stats/_stats_py.py:rankdata` is 64.6% of the call (112 rankings for 56 ordered pairs; ndarray.argsort 15.6% self), `numpy corrcoef` (2x2 per pair) 13.1%, and the rest is wrapper overhead.
- Kendall (`basic.kendalltau__8x1024.pstats`): hotspot_1 is the kendalltau body, `scipy/stats/_stats_py.py:kendalltau`, at 47.5% of the call (argsort 11.4%, count_rank_tie 8.8%, compiled `_kendall_dis` hidden in its 12.5% self time); this is the frame `rank_corr-2` replaces. hotspot_2 is the overhead of the enclosing `scipy/stats/_axis_nan_policy.py:axis_nan_policy_wrapper` outside that body: 96.6% cumulative - 47.5% = 49.1%, of which the `inspect` signature machinery is 24.4% (getfullargspec 16.2% over 112 calls plus signature 8.1% over 56 calls; their shared inner `_signature_from_callable` is 19.8%), wrapper self time 5.2%, `_contains_nan` 4.5%, `_get_nan` 3.5% and `xp_promote` 2.9%.

**Avenues.**
- `rank_corr-1` (spearmanr, vectorise-batch, 300-500x, effort S): rank each channel once and take one `np.corrcoef` of the P x T rank matrix. Measured in `opp_rank_corr-1.json` (OMP_NUM_THREADS=1, min of 2-3 repeats, Data construction timed on both sides):

  | shape | identifier | baseline | candidate | speedup |
  |---|---|---|---|---|
  | 118x1024 | spearmanr | 5.10 s | 0.010 s | 510x |
  | 118x1024 | spearmanr-sq | 5.07 s | 0.014 s | 370x |
  | 32x1024 | both | | | 97-121x |
  | 8x1024 | both | | | 13x |

  The candidate output is bit-identical to the baseline at 7x100, 8x1024, 32x1024 and 118x1024, and matches the 7x100 golden snapshot (max_abs_diff 0). One detail matters for any vectorisation. pyspi squares each pair's numpy scalar, which goes through C pow. A first version squared the array (x*x) and differed by 1 ulp on 8 of the 13924 entries at 118x1024; squaring through the scalar path restored bit identity.
- `rank_corr-2` (kendalltau, rewrite, 10-30x, effort M): one compiled all-pairs tau-b kernel over i<j, with each channel's sort order and tie counts computed once. scipy already uses Knight's O(T log T) merge-sort path, so the gain is in the constant: the nan-policy wrapper is about 50% of the call, there is one argsort per channel instead of two per call, and half the calls disappear. The kernel replaces the whole per-pair call, so the frame that bounds it is `pyspi/base.py:28:parsed_function` at 99.30% of the call (Amdahl bound 143x at 8x1024; `base.py:84` multivariate 99.59%, 246x), not the nan-policy wrapper alone (96.6%, 29x). With tied samples the per-pair (x, y) lexicographic order still needs per-pair work inside each run of tied x values, so the one-argsort-per-channel saving is complete only for tie-free channels; that extra cost depends on the tie structure of the data and was not measured. The i<j halving that `xc-undirected-upper` gives on its own is inside this estimate. Expected unverified-bit-identical: tau-b comes from integer counts and one float expression, which can be reproduced. With ties xtie != ytie, so the bits depend on which channel is passed as x, and the kernel must keep the orientation of the kept triangle. Not measured; the group's two micro-benchmarks went to Spearman and xcorr.
- `rank_corr-3` (both units, cache-sharing, ~2x, effort S): cache the signed matrix on Data so the -sq identifier is a cache read. The family-warm walls show no sharing today: 5.01 s + 4.99 s for Spearman and 4.52 s + 4.52 s for Kendall at 118x1024. Expected unverified-bit-identical if the square goes through the numpy scalar path (see above). Once `rank_corr-1` is in, this saves only ~10 ms for Spearman. For Kendall the ~2x applies to the configured pair of identifiers only: the EEG selection keeps kendalltau-sq alone (`audit/units.yaml` eeg_identifiers, `audit/env/retained_metrics.txt`), so `rank_corr-3` gives no EEG gain for Kendall; the EEG gain is Spearman's (two retained identifiers). Each record lists its unit's rewrite in `depends_on` (`rank_corr-1` for Spearman, `rank_corr-2` for Kendall), so build_csv ranks `rank_corr-3` on the residual after that rewrite (README "Optimisation columns", ranking rule 4) instead of on today's cost. The kendalltau record also sets `eeg_applicable: false` on `rank_corr-3`, so that row is ranked outside tier 0, on the configured pair of identifiers, and is left out of kendalltau's tier-0 stacking (README ranking rule 2).

**Seed check.** Seed "spearman = rank once + one corrcoef": confirmed and measured (bit-identical, ~370-510x at 118x1024). Seed "kendall single direction": confirmed. The result is exactly symmetric (check stage: unmirrored Directed.multivariate gives 0/6 differing pairs) and the second triangle is discarded, so computing i<j only halves the calls. That is the cross-cutting `xc-undirected-upper` avenue, and `rank_corr-2` includes it. Seed "does scipy already use its O(T log T) path": yes (argsort, mergesort argsort, `_kendall_dis`), so no algorithmic change of order is available; only the constant and the wrapper overhead can be removed.

**Shared versus incremental cost and EEG scaling.** There is no cache in this family; each identifier pays the full pair loop. The EEG selection keeps spearmanr, spearmanr-sq and kendalltau-sq. The per-window EEG cost at 118 channels is therefore 5.0 + 5.0 + 4.5 = about 14.5 s per window, or about 8.7e3 s per 600-window patient. There are no band variants. The ranking uses the same selection: basic.kendalltau ranks on its retained identifier (the kendalltau-sq family-warm increment, 4.52 s per window, 2,711 s per patient instead of the configured 5,420 s; `basis_scope` retained-identifiers), and basic.spearmanr on both identifiers (5,998 s, family-warm; the fresh projection, 2,979 s, times spearmanr-sq only). The historical EEG figure for spearmanr (16.6 s per window, channel count unknown) is the same order as the 10 s for both Spearman identifiers here. With `rank_corr-1` and `rank_corr-2` the family would drop to about 0.2-0.5 s per window, dominated by Kendall.
