# Family `ids`: define, trace and check

Unit: `misc.ids` (identifier `ids`; parameters `terms=6`, `pnorm='max'`, `bandwidth=0.5`).
Record: `audit/records/misc.ids.yaml`.
Environment probed: numpy 2.5.4, tqdm 4.70.1, project venv. The probe scripts are in the session scratchpad under `ids_probe/` (probe1–probe5). The upstream code was downloaded for comparison to `ids_upstream/`, and the paper full text (PMC12403096, Europe PMC XML) to `ids_paper/`. Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

The trace stage (the sections before "## Checks") did not run any numeric checks; the numeric checks are reported under "## Checks". The trace-stage probe numbers are small read-only confirmations of what the code does. They do not verify correctness. The regression goldens are snapshots of current output and are not used as a reference here. The verdict and score are under "## Checks" and in the record, and candidates and hypotheses raised in the trace sections are resolved in the record's `candidate_resolutions` and under "## Checks".

## misc.ids — InterDependence Score (IDS-max)

### Documented definition
The supplement has no section on this SPI; it was added after the supplement was written. The only documentation is GitBook `miscellaneous.md:117-127`:
- keywords "undirected, nonlinear, unsigned";
- "a fast and scalable measure of dependence that captures both linear and various non-linear dependencies … Based on the Hilbert-Schmidt Independence Criterion (HSIC), the IDS offers a finite-dimensional approximation of the universal kernels that can detect any type of dependence";
- "normalized to a range of [0, 1] with a score of 0 indicating independence";
- the implementation is "based on the code made publicly available" at `aradha/interdependence_scores`.

### Canonical definition
Radhakrishnan, Jain, Uhler & Lander 2025, *PNAS* 122(34):e2509860122.
- Eq. 3 gives the Gaussian kernel `exp(-(x-y)^2/(2B^2))`.
- Eq. 6 gives its Taylor feature map `phi_m(x) = exp(-x^2/(2B^2)) x^m / (sqrt(m!) B^m)`. The first k terms are kept.
- Eq. 11 defines the correlations `rho_ml = Corr(phi_m(x), phi_l(y))`.
- Eq. 12 defines `IDS_p = ((1/k^2) Σ |rho_ml|^p)^(1/p)`. `IDS_inf` is the maximum.
- The reported implementation uses IDS_inf, B = 1 and k = 6, after the data are "normalized … by min-max scaling to the range [0, 8]". The authors add that "it may be worth exploring other choices of power, feature map, and normalization".
- The paper gives IDS a meaning through permutation P-values.

### What is implemented (trace)
1. `config.yaml:1394-1404` sets the configuration. `Calculator._load_yaml` (`calculator.py:230-267`) imports `.statistics.misc` at line 239. On first import this runs `pyspi/lib/ids/numpy_dependence.py:11-12`, `np.random.seed(1717)`. The config loop (251-267) then instantiates the SPI at line 256 and stores it under `ids` at 257.
2. `data.py:184-203` z-scores each process (ddof=1). Detrending is off by default. Any NaN raises an error, so a constant channel never reaches IDS.
3. `misc.py:194-200` overrides `Undirected.multivariate`, so there is no mirroring. It reshapes the data to `(T, P)` and calls `compute_IDS` (`dependence.py:9-31`). That function always calls the numpy backend; upstream would prefer torch when it is installed.
4. `numpy_dependence.py:16-23` builds the term-major features `exp(-0.5 z^2) z^k / sqrt(k!)` for k = 0..5. Lines 25-26 and 63-64 centre them.
5. Lines 76-78 compute the full `(KP x KP)` `np.corrcoef` and reshape it to `(K, P, K, P)`. numpy `_function_base_impl.py:3029-3047` does `cov` (a symmetric `dot`), divides by `stddev[:, None]` and then by `stddev[None, :]`, and clips to [-1, 1].
6. Lines 80-84 apply `nan_to_num → abs → amax over axes (0, 2)`. Lines 85-92 handle p = 2 and p = 1. Any other `p_norm` returns the raw 4-D array.

The vendored `numpy_dependence.py` is a verbatim copy of upstream `ids/numpy_dependence.py` (commit 3c8d62b, 2025-03-18). The diff shows only a header docstring added and the `__main__` demo removed.

Probes:
- A from-scratch numpy reference agrees to ≤ 8.9e-16 on 7 datasets × 4 lengths.
- The diagonal is 1.
- max |A − A^T| ≤ 1.1e-16. The matrix is symmetric up to the last bit, but not exactly.
- `Calculator` gives the same output as a direct `multivariate` call (difference 0.0).

### Deviation analysis
- **Parameters vs paper.** Here `bandwidth` is the coefficient of z², so 0.5 corresponds to paper B = 1. The B^m and sqrt(m!) column scalings cancel inside a correlation. The code agrees with a from-scratch Eq. 6 to ≤ 4.4e-16 for bandwidth 0.25, 0.5 and 1.0. With its defaults it is the paper's IDS_inf with k = 6 and B = 1.
- **Preprocessing (variant).** pyspi z-scores; the paper min-max scales to [0, 8]. The Gaussian damping is not invariant to scale or shift, so the values differ a lot. At T = 4096: nonlinear_dependence (0,1) gives 0.943 vs 0.485 and (2,3) gives 0.993 vs 0.313; contaminated gives 0.949 vs 0.592. The paper allows other normalisations, so this is a variant choice. The GitBook does not say which one pyspi uses.
- **"0 indicates independence".** This holds only asymptotically. On iid_gaussian the off-diagonal means are 0.235, 0.113, 0.065 and 0.032 at T = 64, 256, 1024 and 4096. A from-scratch Monte Carlo null has mean 2.06/√T and a 95 % quantile of 0.370 at T = 64. Autocorrelated series inflate the null further: on `trending` without detrend, cross pairs reach 0.98–0.996.
- **Not a generalisation of |Pearson|.** For a Gaussian pair the population IDS (from quadrature) is 0.2641 at ρ = 0.3 and 0.7349 at ρ = 0.8. The identity feature is not among the damped features. Exact linear relations reach 1 only because z-scoring makes the two series identical up to sign.
- **Zero-lag only.** delayed_noise copies at lags 1, 3 and 5 score at the null level (0.03–0.05 at T = 4096).
- **Code-level candidates.** These are hypotheses with evidence and are not yet confirmed as findings:
  1. *Global RNG reseed.* After `np.random.seed(123); Calculator()`, the global state equals `seed(1717)` (probe True). This silently replaces the user's seed for global-RNG SPIs in other families: MinCovDet and EllipticEnvelope through `basic.py:39` with `random_state=None`, and tslearn sgddtw. It happens only on the first import in a process. The behaviour comes from upstream.
  2. *Undocumented null bias* (above).
  3. *Preprocessing differs from the paper's reported setting* (above).
  4. *Last-bit asymmetry.* There is no mirroring, and `corrcoef` divides in a fixed order. This is the REVIEW M2 class of issue, but IDS is not listed there.
  5. *Unsupported `p_norm`.* A value of 3 or `'2'` returns a `(6, P, 6, P)` array. `Calculator.compute` never stores it. For P ≠ 6, `fill_diagonal` (calculator.py:332) raises `ValueError`. For P = 6, `fill_diagonal` accepts the array, but the table assignment at calculator.py:335 raises `AssertionError` in pandas. Either way the except at calculator.py:336-339 sets the whole `ids` block to NaN (337) and records the error and warning text (338-339); the failure is recorded as a `MetricReport` with status `failed` (calculator.py:342-346) and the warning is emitted at calculator.py:354-355, after the optional callback (347-353). No 4-D result is stored. Probe: a Calculator run with `pnorm` 3 or `'2'` gives an all-NaN block for both P = 4 and P = 6. Only a non-default config can reach this.

### Proposed checks
(See the record for the full specifications.)
- **Kernel, exact.** A from-scratch damped-Taylor-feature Pearson-max implementation, on 7 datasets.
- **Kernel, statistical.** Gauss–Hermite population values 0.2641 and 0.7349 on bivariate_gaussian.
- **Pipeline, statistical.** Monte Carlo null on iid_gaussian (mean ≈ 2.06/√T).
- **Pipeline, direction/index.** Symmetry ≤ 1e-15, and the dependent pairs of nonlinear_dependence scoring > 0.9.
- **Pipeline, exact.** Invariance under scaled_offset vs bivariate_gaussian, and under a sign flip.
- **Pipeline-detrend, qualitative.** On `trending`, the bivariate_gaussian structure should reappear.
- **Kernel, exact.** Equivalence with paper Eq. 6 across bandwidths.
- **Kernel, qualitative.** Size of the preprocessing effect compared with min-max scaling to [0, 8].
- **Exact.** The RNG state after `Calculator()` equals `seed(1717)`.

## Open questions
- Should pyspi document, or change, the z-score preprocessing relative to the paper's min-max [0, 8] setting? Which convention should a "matches" verdict be judged against?
- Should the module-level `np.random.seed(1717)` be removed from the vendored copy? Removing it would change the RNG stream of the stochastic SPIs (MinCovDet, EllipticEnvelope, sgddtw) in Calculator runs, so their goldens would shift. Is the cross-family impact large enough to call it a finding?
- Is there a dependent pair with population IDS-max = 0 at k = 6 (a dependence invisible to the truncated features)? This would refute the GitBook's "any type of dependence" in the strict sense. The probe did not search for one; this is a hypothesis.
- Should the identifier encode `terms`, `pnorm` and `bandwidth`, so that non-default configurations do not collide on `ids`? This relates to the parameter-specific labels issue in the EEG notes §review and REVIEW M9.
- How much does autocorrelation inflate the null on the real EEG-like inputs? This would set the significance thresholds a user needs.

## Checks

**Setup.** The manifest `audit/checks/manifest/ids.json` has 69 runs. References are in `audit/checks/refs/ids.py`, which runs in the comparison venv (numpy/scipy only) and never imports `pyspi.lib.ids`. Probes are in `audit/checks/probes/ids_probes.py`, which runs in the project venv. Its top level is numpy-only, so the reference script can load it by path to rebuild the constructed sample. Results are in `audit/checks/out/results/ids.json`: **65 pass, 4 fail, 0 without reference**. The four failures are the `paper-minmax08` runs, which compare pyspi's default with the paper's reported preprocessing. They are designed to fail, and their max_abs measures the size of the preprocessing effect.

This check stage re-ran the whole family with `--force`. It started from the draft manifest, references and probes left by the interrupted earlier attempt, which were reviewed and kept. One addition was made: a Monte Carlo under the constructed hidden-dependence law, so that that check no longer rests on a single draw.

Cost: every SPI run takes ≤ 0.005 s at T = 4096. The longest probe is `rng-cross-spi` at 14 s, because it starts seven fresh interpreters that each run MinCovDet. A full forced run takes about 75 s on the pyspi side and 60 s on the reference side.

Seven kinds of check were run:

1. **`kids` (exact, rtol 1e-10 / atol 1e-12).** pyspi is compared with `_kernels_basic_misc.ids` on the same prepared rows: raw on the kernel path, z-scored (ddof=1) on the pipeline path, and linearly detrended then z-scored on pipeline-detrend. The kernel builds the transform, computes Pearson correlations with explicit dot products and norms (no `np.corrcoef`), and takes the max over the term blocks. As an extra key, each run also compares the kernel with `refs/ids.py:eq6_ids`, an own per-pair transcription of paper Eq. 6, 11 and 12.
2. **Analytic.** Identical rows give 1. The other analytic checks are sign-flip invariance and z-scoring invariance on scaled_offset.
3. **`eq6-bw*` and `pnorm1/2` (exact).** Non-default constructor parameters, set through a probe, are compared with Eq. 6/12, using B = sqrt(1/(2·bandwidth)).
4. **`truth-*`, `null*` and `hidden-dependence` (statistical).** The rule is |pyspi − MC mean| ≤ 2 se. Here se is the MC standard deviation of the statistic; for the null runs it is multiplied by a familywise factor over the pairs. The truth runs also report the population value from Gauss–Hermite quadrature.
5. **`structure` and `zero-lag` (qualitative).** Each run checks four things: max|A − Aᵀ| ≤ 1e-15, a NaN diagonal, dependent pairs above their bound, and independent pairs below it. The bound is a per-pair permutation (1 − 0.05/m) quantile from 1000 shuffles.
6. **`minmax08` / `paper-minmax08`.** These use the paper's reported setting: min-max to [0, 8], B = 1.
7. **Code-level checks.**
   - `pnorm-unsupported`: what happens with a pnorm value the code does not handle.
   - `rng-*`: fresh interpreters, judged against the legacy MT19937 stream `np.random.RandomState(seed)`.
   - `symmetry`: max|A − Aᵀ| over every square output in the family.

### misc.ids: variant, score 7 (anchor 7: matches the documented implementation, differs from the paper's reported setting in a value-changing way, here the input normalisation)

**The estimator matches the paper's IDS_inf with k = 6 and B = 1 to floating-point precision.**
- 14 kernel-path `kids` runs pass with max_abs ≤ 7.8e-16, and 10 pipeline and pipeline-detrend runs pass with max_abs ≤ 2.2e-15. The largest, 2.2e-15, is `trending|1024|pipeline-detrend`, where the own lstsq detrend replaces scipy's.
- The own Eq. 6/11/12 transcription agrees with the kernel mirror to ≤ 1.0e-15.
- Bandwidths 0.25, 0.5 and 1.0 match Eq. 6 with B = 1.414, 1.0 and 0.707 to ≤ 4.4e-16. So `bandwidth` is the coefficient of z², and the default is paper B = 1.
- pnorm 1 and 2 match Eq. 12 to 1.1e-16.

**Analytic checks.**
- Rows [x0, x0, −x0, 2.5·x0 − 1] (`bivariate_gaussian|256|pipeline|identical`) give 1 in every off-diagonal entry with max_abs 0.0. So does the bit-copy pair x0, x2 of shared_envelope (`…|identical-pair`).
- Sign flip: 1.1e-16 on both paths.
- `scaled_offset|1024|pipeline|invariance` equals the z-scored bivariate_gaussian to 5.6e-17. On the raw kernel path the same data differ by 0.628 (`scaled_offset|1024|kernel|kids`, `ref_kernel_minus_bivariate_zscored_maxabs`): IDS is not scale invariant without z-scoring.

**Population values.** Gauss–Hermite quadrature gives 0.26408 at ρ = 0.3 and 0.73485 at ρ = 0.8; the argmax term pair is (3, 3). These confirm the trace values.

| Pair | T = 256 | T = 1024 | T = 4096 | max_z |
|---|---|---|---|---|
| ρ = 0.3, pop 0.2641 | 0.2746 (MC 0.2814) | 0.2617 (MC 0.2670) | 0.2697 (MC 0.2641) | 0.39 |
| ρ = 0.8, pop 0.7349 | 0.7617 (MC 0.7380) | 0.7492 (MC 0.7349) | 0.7303 (MC 0.7347) | 0.93 |

All 6 truth runs pass. The MC mean converges to the population value: at ρ = 0.3 the gap is 0.0173, 0.0029 and 0.00004 at T = 256, 1024 and 4096.

**Null.** These are the iid_gaussian runs with 2000 MC draws; all 4 pass, with max_z ≤ 1.50.

| T | MC mean | MC mean·√T | MC q95 | pyspi off-diagonal mean |
|---|---|---|---|---|
| 64 | 0.2579 | 2.06 | 0.368 | 0.2354 |
| 256 | 0.1273 | 2.04 | 0.183 | 0.1135 |
| 1024 | 0.0648 | 2.07 | 0.094 | 0.0647 |
| 4096 | 0.0324 | 2.07 | 0.047 | 0.0324 |

Autocorrelation raises the null a lot (extra keys on `iid_gaussian|1024|pipeline|null`):
- For AR(1) at T = 1024, the MC mean is 0.069, 0.132 and 0.359 for φ = 0.5, 0.9 and 0.99, and the q95 is 0.100, 0.192 and 0.557.
- For independent random walks, the MC mean is 0.568, which is 8.8 times the iid mean. pyspi gives 0.465 on cointegrated x2, x3 (`cointegrated|1024|pipeline|null-randomwalk`, pass, z 0.62).

**Structure.** All 6 runs pass.
- nonlinear_dependence: (0,1) and (2,3) give 0.945 and 0.985 at T = 1024, and 0.943 and 0.993 at T = 4096.
- anm: 0.853 and 0.968.
- bivariate_gaussian: 0.262 and 0.749.
- Every independent pair is ≤ 0.071 and at most 0.84 of its permutation bound.
- On trending, detrending recovers the D2 structure: 0.263 and 0.750, with cross pairs ≤ 0.077. Without detrending, the cross pairs reach 0.996.

**Preprocessing (the reason for 7).**
- pyspi run on rows min-max scaled to [0, 8] reproduces the paper setting to ≤ 3.3e-16 (4 `minmax08` kernel runs).
- pyspi's default z-scored path differs from that setting by up to:
  - 0.680 on nonlinear_dependence (0.943 / 0.993 vs 0.485 / 0.313);
  - 0.357 on contaminated (0.949 vs 0.592);
  - 0.178 on anm;
  - 0.054 on bivariate_gaussian (0.270 / 0.730 vs 0.225 / 0.677).

  All four are at T = 4096 and are the 4 designed failures.
- The paper allows other normalisations, so this is a variant. Neither the GitBook nor the code says which normalisation pyspi uses.
- Not 9: the difference changes values and is not documented. Not 10: there is no supplement section, and pyspi's default path does not reproduce the paper's reported setting.

**Hidden dependence (answers open question 3).**
- Construction (`probes/ids_probes.py:hidden_construction`): g is cos(8x) with its projection on 1 and on the six Eq. 6 features (B = 1) removed under N(0, 1), then normalised to unit variance. The joint density is φ(x)φ(y)(1 + ε g(x)g(y)) with ε = 0.474.
- Properties: the marginals stay N(0, 1). Every population Cov(φ_k(X), φ_l(Y)) is 0, so IDS_p = 0 for every p. Yet Corr(g(X), g(Y)) = ε = 0.474.
- Cross-check by an independent trapezoid rule: orthogonality 1.1e-16, mean 1.6e-16, E g² = 1.0000000.
- pyspi values: 0.0436 on the kernel path and 0.0435 on the pipeline path, against null MC means of 0.0325 and 0.0324 (pass, z 1.30 and 1.38).
- Own MC of 300 draws under this law: mean IDS 0.0322 and 0.0317, against null means of 0.0325 and 0.0324 (z −0.62 and −1.26). In the pyspi sample, Corr(g(X), g(Y)) is 0.491 (z 31.4).
- So IDS with k = 6 is blind to this dependence. The GitBook's "can detect any type of dependence" describes the universal-kernel limit, not the truncated statistic.

**Symmetry.** Over 65 square outputs the maximum of max|A − Aᵀ| is 2.2e-16, and 52 of the outputs are not bitwise symmetric (`iid_gaussian|4096|pipeline|symmetry`; the worst is `nonlinear_dependence|256|kernel|kids`). This corrects the trace value of ≤ 1.1e-16 in `implemented_as`, `units_and_range` and `direction_convention`.

**Direction.** IDS is undirected, and A[i,j] = A[j,i] at zero lag. The anm causal pairs score 0.853 and 0.968 in both orientations. The delayed_noise lagged copies stay at the null level: max 0.046, against a permutation-bound median of 0.055 (`delayed_noise|4096|pipeline|zero-lag`).

**Candidates.** All five are confirmed. Only the first is a new finding.
1. **Global RNG reseed: confirmed; the consequence for user seeds is new.** The import side effect itself is already noted in the test harness: `tests/regression/harness.py:96-98` says that importing `pyspi.statistics.misc` calls `np.random.seed(1717)` and that `seed_all` must run after it (the checked cov_prec_robust note, in its answer at `notes/cov_prec_robust.md:338-343`, states that default-config reproducibility of the robust SPIs rests on this reseed). KNOWN_ISSUES.md does not list it, and no repository text says explicitly that a user seed set before the first `Calculator()` is discarded for the global-RNG SPIs. That consequence is what the runs below establish.
   - `rng-import`: after `np.random.seed(123)`, importing `pyspi.lib.ids` leaves the global state equal to that of `seed(1717)`; the first draw is 0.35547 = `RandomState(1717)`. Importing `pyspi.calculator` alone leaves the stream untouched (0.69647 = `RandomState(123)`). One IDS compute consumes no random numbers: the next draw is 0.26495, the second draw of `RandomState(1717)`. IDS is deterministic: seeds 1 and 2 give a difference of 0.0.
   - `rng-calculator`: the first `Calculator()` in a process reseeds to 1717. A second one keeps the user's seed (0.22199 = `RandomState(5)`).
   - `rng-cross-spi`: cov_MinCovDet gives sum|A| = 2.30186 under user seeds 1 and 2 set before `Calculator()`. That equals the no-ids config under seed 1717. Without ids, seeds 1 and 2 give 2.64277 and 2.24323. So a seed set before construction is silently replaced for the global-RNG SPIs. A seed set after construction (as `seed_all()` does in tests) is respected. Computing ids before MinCovDet does not shift the stream MinCovDet sees: 2.74185 in both configs with seed 7.
   - The size of the MinCovDet seed spread is documented in the checked cov_prec_robust family (for example `basic.cov.MinCovDet|iid_gaussian|1024|pipeline|seeds`). sgddtw was not run here; it draws from the same global stream (tests/README.md:146).
2. **Undocumented null bias: confirmed.** See the null table above. This is estimator behaviour, and finite-sample bias is not incorrectness (README). The deviation is in the GitBook wording "a score of 0 indicating independence".
3. **Preprocessing differs from the paper: confirmed.** These are the `paper-minmax08` runs. It is a variant.
4. **Last-bit asymmetry: confirmed, max 2.2e-16.** This is the same class as REVIEW M2, which does not list IDS. The size is negligible.
5. **Unsupported pnorm: confirmed.** `bivariate_gaussian|256|pipeline|pnorm-unsupported`: for pnorm 3 and '2', compute_IDS returns a 4-D array. The Calculator block is all NaN and the MetricReport is `failed`, at both P = 4 (ValueError from fill_diagonal) and P = 6 (AssertionError from the table store). Only a non-default config can reach this.

`new_finding` is true on the strength of candidate 1, limited to its user-facing consequence (a seed set before the first `Calculator()` is discarded for MinCovDet and the other global-RNG SPIs); the import-time reseed alone is documented at `tests/regression/harness.py:96-98`.

## Open questions: answered

1. **Paper min-max [0, 8] or pyspi z-score as the reference for "matches"?**
   - The checks judge the formula and the preprocessing separately. The formula (Eq. 6/11/12, k = 6, B = 1, IDS_inf) matches to ≤ 1e-15.
   - The preprocessing is z-scoring, which differs from the paper's reported setting by up to 0.68.
   - Because the paper leaves the normalisation open, the verdict is variant (7), not wrong. The standard formula is not contested, so `definition_under_review` is false.
   - The GitBook should state that pyspi feeds z-scored data and that the paper reports min-max [0, 8]. Settling which one is preferable for time series would need a power comparison under the two normalisations; that comparison was not run.
2. **Remove the module-level `np.random.seed(1717)`?**
   - The checks confirm the side effect and its reach: a user seed set before the first `Calculator()` is discarded for MinCovDet, EllipticEnvelope and sgddtw. IDS itself uses no random numbers (deterministic, no draws), so removing the line cannot change any IDS value.
   - Removing it would change the stream that those stochastic SPIs see only when a user seeds before construction or does not seed at all.
   - Effect on the regression goldens: not tested here, so this part stays open. The code path suggests no change for the per-SPI goldens, because they are generated and tested with `seed_all()` called after the statistics modules are imported and immediately before each compute (`tests/regression/generate_goldens.py:82`, `tests/regression/harness.py:282-284`, `tests/regression/test_spi_goldens.py:188, 204`; the harness docstring at `harness.py:96-98` states this ordering is deliberate). What would settle it: regenerate the goldens with the line removed and compare. The goldens are snapshots of current output, not references, so such a comparison would show only whether the stream changed.
   - The impact is cross-family and silent, so it is recorded as a finding.
3. **Is there a dependent law with population IDS-max = 0 at k = 6?** Yes. See "Hidden dependence" above. The constructed law has population IDS_p = 0 for every p while Corr(g(X), g(Y)) = 0.474, and pyspi scores it at the independence null level at T = 4096. The GitBook's "can detect any type of dependence" does not hold for the truncated statistic.
4. **Should the identifier encode terms, pnorm and bandwidth?**
   - The checks show that these parameters change values: bandwidth 0.25, 0.5 and 1.0 and pnorm 1 and 2 give different matrices, each matching its own Eq. 6/12 value. All of them would be stored under `ids`.
   - Whether to encode them is a design decision covered by the parameter-specific labels item in the EEG notes §review and by REVIEW M9; it is already documented and not raised as new here.
   - The unsupported-pnorm path fails into an all-NaN block rather than colliding.
5. **How much does autocorrelation inflate the null, and what thresholds should users apply?**
   - At T = 1024 the iid null has mean 0.065 and q95 0.094. With AR(1) φ = 0.9, which is roughly a low-pass EEG band, the q95 is 0.192. With φ = 0.99 it is 0.557, and independent random walks have a mean of 0.568.
   - So no fixed threshold works across inputs. Users need surrogates that preserve autocorrelation: block or phase-randomised permutations of one series, recomputed per pair.
   - Measuring this on real EEG would need the EEG workflow data, which these checks did not use.

## Optimisation

Workflow 3 (profile-based search for hotspots and avenues; nothing in `pyspi/` was changed). Walls from `audit/profiles/summary.json`: 0.001 s at 8x1024, 0.006 s at 32x1024, 0.029 s (fresh) / 0.028 s (family-warm) per window at 118x1024, measured; about 18 s per 600-window patient. Not in the EEG selection. One call per window, no pair loop, nothing shared with other families.

Hotspots: at 8x1024 (`audit/profiles/misc.ids__8x1024.pstats`, opaque share 0.10) `pyspi/lib/ids/numpy_dependence.py:transform` 42.5% (42.3% self), `center` 25.6%, `numpy corrcoef` 20.3%. At 118x1024 (`audit/profiles/micro/misc.ids__118x1024.pstats`, profiled by this group, opaque share 0.089) the O(K^2 P^2 T) correlation dominates: `numpy corrcoef` 52.4% (cov 50.3%), `transform` 33.7%, `center` 6.3%. Measured power-law P_exponent 1.19, T_exponent 0.54.

Opportunity:
- ids-1: build z^k exp(-B z^2) by repeated multiplication from one exp factor and drop the redundant `center()` before `np.corrcoef` (which centres again); removable share at most 0.4 at 118x1024, expected ~1.3x, i.e. about 10 ms per window. Not measured. Output risk unverified-within-tolerance (different rounding of the powers and of the centring). The corrcoef itself cannot shrink: every cross-term correlation of the 6 x 6 feature blocks is needed for the max. Low priority; listed for completeness.

Note for parallel runners: importing `pyspi.lib.ids` reseeds the global numpy RNG (`np.random.seed(1717)` at numpy_dependence.py:11-12, already recorded in the checks); any worker-pool design for other units that consume the global RNG must import pyspi in each worker the same way to keep their streams.
