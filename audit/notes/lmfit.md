# lmfit family: define, trace and check

Units: `misc.lmfit.Ridge`, `misc.lmfit.Lasso`, `misc.lmfit.SGDRegressor`, `misc.lmfit.ElasticNet`, `misc.lmfit.BayesianRidge` (one identifier each, `lmfit_<model>`, all in the EEG selection).
Class: `LinearModel(Directed, Unsigned)` in `pyspi/statistics/misc.py:106-126`. Delegate: scikit-learn 1.9.1 (`sklearn.linear_model`), with numpy 2.5.4 and scipy 1.17.1.
The probe scripts are in the session scratchpad under `lmfit/probe1.py` to `probe4.py`, and the record generator is `lmfit/write_records.py`. The regression goldens are snapshots of current output, and nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Documented definition (shared)

Supplement §1.6.1, p.18. The heading is at supplement.txt line 856, and the section runs to line 868. The briefing pack gives lines 874-887, which is 18 lines off.

> "Linear regression is commonly used for establishing independence through model fits (e.g., see additive noise models in Sec. 1.3.1). As such, we use a number of linear models and record the mean squared error (MSE) of a regression of y on x. The following models (with the default parameters) from scikit-learn [2] are included: stochastic gradient descent regression with a squared loss (ordinary least squares fit) function (denoted by modifier SGDRegressor); Ridge regression, which uses ℓ2-norm regularization (Ride); the Elastic-Net model [76, 77], which uses both ℓ1 and ℓ2-norm regularization (ElasticNet); and the Bayesian Ridge regressor uses a gamma distribution prior (with λ1 = λ2 = 10⁻⁶) for the ℓ2-norm regularizor in Ridge regression (BayesianRidge). This yields four SPIs: 215–218."

The keywords are: directed, linear, unsigned, bivariate, contemporaneous. The GitBook page `spi-descriptions/miscellaneous.md` ("Linear Model Fit") repeats the text, including the typo "Ride", and lists the same four identifiers. **`lmfit_Lasso` is documented nowhere.** It exists only because of `pyspi/config.yaml:1275`.

## Shared implementation (trace)

1. **Preprocessing.** `pyspi/data.py:184-195`: optional `scipy.signal.detrend`, then `zscore(axis=1, ddof=1)` (default `normalise=True`).
2. **Matrix assembly.** `pyspi/base.py:84-94` `Directed.multivariate` loops over every ordered pair: `A[i, j] = bivariate(data, i=i, j=j)`, with no mirroring. `pyspi/calculator.py:327-339` (HEAD a1f20c7) fills the diagonal with NaN (:332); if any pair raises, the whole matrix becomes NaN (:336-339).
3. **The SPI.** `misc.py:117-126`:
   - `z = data.to_numpy()`, so `z[i]` is a (T, 1) design matrix.
   - If the estimator's signature has `random_state`, it is built with `random_state=42`. That applies to Ridge, Lasso, ElasticNet and SGDRegressor. BayesianRidge has no such parameter and gets no argument.
   - `.fit(z[i], ravel(z[j]))`, then `mean_squared_error(mdl.predict(z[i]), ravel(z[j]))`.

   So **predictor = process i, target = process j**. The MSE is **in-sample**: there is no hold-out and no lag. The swapped argument order to `mean_squared_error` is harmless.
4. **sklearn path.** Ridge, Lasso, ElasticNet and BayesianRidge share the centring, intercept and prediction steps below; SGDRegressor takes its own path (fourth bullet).
   - `_preprocess_data` (`_base.py:188-199`) centres X and y. Ridge calls it from `_BaseRidge.fit` (`_ridge.py:964`), Lasso and ElasticNet through `_pre_fit` (`_coordinate_descent.py:1212` → `_base.py:872`), BayesianRidge from its `fit` (`_bayes.py:257`).
   - `_set_intercept` (`_base.py:318-334`) sets an unpenalised intercept (`_ridge.py:1017`, `_coordinate_descent.py:1283`, `_bayes.py:370`).
   - Prediction for these four ends in `LinearModel._decision_function` (`sklearn/linear_model/_base.py:291-299`), X @ coef_ + intercept_. `LinearModel.predict` (`:301-316`) returns it; Ridge reaches it through `Ridge.predict` (`_ridge.py:1265-1279`, `super().predict`), Lasso and ElasticNet through `MultiOutputLinearModel.predict` (`_base.py:341-355`) and the `ElasticNet._decision_function` override (`_coordinate_descent.py:1301-1317`), which calls the base method for dense X. BayesianRidge overrides `predict` (`_bayes.py:374-403`); with the default `return_std=False` it returns `self._decision_function(X)` (`:396-398`).
   - SGDRegressor uses none of these. Nothing in `_stochastic_gradient.py` calls `_preprocess_data` or `_set_intercept`, so sklearn does not centre X or y: `_plain_sgd` updates the intercept together with the weight (`_sgd_fast.pyx.tp:549-554`) and `_fit_regressor` stores it (`_stochastic_gradient.py:1794`, default `average=False`). Its `predict` is `BaseSGDRegressor.predict` (`_stochastic_gradient.py:1697-1710`), which returns `BaseSGDRegressor._decision_function` (`:1678-1695`), X @ coef_.T + intercept_, raveled. The fit chain is in the SGDRegressor section below.
   - MSE (`metrics/_regression.py:586-588`), with divisor T, for all five units.

**Key algebraic fact.** After ddof=1 z-scoring, every process has Σx = 0 and Σx² = T−1, so Σxy = r(T−1). Every deterministic fit below is therefore a function of (r_ij, T) only, and so is its in-sample MSE. The matrix is symmetric under the default preprocessing, even though the class is `Directed` and labelled "directed". The probe measured max |A − Aᵀ| ≤ 7.8e-16 for Ridge, Lasso, ElasticNet and BayesianRidge on six datasets × four T.

**Direction convention.** From the code, A[i, j] is the MSE of predicting j from i, and lower means more predictable. It can only be seen empirically with `normalise=False`. Then column j scales with var(x_j). On scaled_offset_T1024 (raw), the column values track std_j² = 0.0395, 4.058, 24.78 and 97.89.

**Shared candidates.**
1. The "directed" label is nominal under the default preprocessing.
2. The documented motivation ("establishing independence through model fits", ANM) does not match the quantity. The SPI is a raw in-sample MSE, which measures linear predictability. No residual-independence test is computed.

## misc.lmfit.Ridge

- **Canonical.** Hoerl & Kennard 1970, *Technometrics* 12(1):55-67: β*(k) = (X′X + kI)⁻¹X′y. scikit-learn minimises ‖y − Xw‖² + α‖w‖² on centred data with an unpenalised intercept, at α = 1. The independent source for the unpenalised intercept and centring is Hastie, Tibshirani & Friedman 2009, *The Elements of Statistical Learning*, 2nd ed., §3.4.1 (section and wording from memory, not re-checked). Scale convention: Hoerl & Kennard state k for X in correlation form (X′X = 1), where k = 1 would give w = r/2; pyspi applies α = 1 to ddof=1 z-scores (X′X = T−1), giving w = r(T−1)/T. This is the documented "default parameters" choice, not a deviation.
- **Implemented.** `Ridge(random_state=42)`:
  1. `Ridge.fit` (`_ridge.py:1229-1263`) calls `_BaseRidge.fit` (`:919-1019`).
  2. `resolve_solver_for_numpy` (`:867-877`) picks "cholesky".
  3. The cholesky branch of `_ridge_regression` (`:746-761`) calls `_solve_cholesky` (`:215-227`), which computes w = Σxy/(Σx² + 1).

  `random_state` is inert: the probe found identical predictions for seeds 42 and 7 and for the default.
- **Closed form on z-scored data.** w = r(T−1)/T and MSE = ((T−1)/T)(1 − r²(T²−1)/T²). The probe's assertion held to ≤ 1e-12 for all 552 ordered pairs (14 datasets × T ∈ {64, 256, 1024, 4096}). The pack hypothesis "w = r(T−1)/T" is **confirmed**.
- **Deviations.**
  - Documented: the "Ride" typo; the nominal "directed" label; "default parameters" plus an inert `random_state`.
  - Canonical: none for the estimator.
  - The SPI is an affine, decreasing function of r². This is a candidate finding that it is redundant with squared Pearson correlation.
- **Range.** [(T−1)/T³, (T−1)/T). Identical series give 2.40e-4 at T=64.
- **Proposed checks.**
  - From-scratch numpy ridge on the kernel, exact to ≤ 1e-12.
  - The analytic r² formula on the pipeline, exact.
  - The detrend pipeline on trending and cointegrated.
  - Symmetry, exact.
  - A raw-scale direction test against var0(x_j) − cov0² T(Sxx + 2)/(Sxx + 1)². This formula agrees with pyspi on scaled_offset_T1024 pair (2, 3): 35.6348282303112 both.

## misc.lmfit.Lasso

- **Canonical.** Tibshirani 1996, *JRSS-B* 58(1):267-288. scikit-learn's objective is (1/(2T))‖y − Xw‖² + α‖w‖₁ (`_coordinate_descent.py:1334`). For one centred predictor the solution is w = S(Σxy/T, α)/(Σx²/T).
- **Implemented.**
  1. `Lasso` sets `l1_ratio=1.0` (`:1501-1514`) and calls `ElasticNet.fit` (`:1096-1294`).
  2. `enet_path` scales the penalty to l1_reg = α·T (`:781-784`).
  3. `_cd_fast.enet_coordinate_descent` (`_cd_fast.pyx:243`) checks the duality gap at the initial w = 0. That check already passes (`:391-400`), because |xᵀy| = |r|(T−1) < T, so the solver returns after 0 iterations.

  The probe found `n_iter_` 0, `coef_` [0.], `dual_gap_` 0.0, even for an r = 0.82 pair.
- **Result.** A[i, j] = mean(z_j²) = (T−1)/T **exactly, for every pair and every dataset**. The probe gave 0.984375, 0.99609375, 0.9990234375 and 0.999755859375, with max deviation 0.0. Even identical series give (T−1)/T. The pack hypothesis is **confirmed**, and in a stronger form: the condition |r|(T−1)/T ≤ α·l1_ratio = 1 always holds, so the SPI is degenerate rather than merely degenerate for weak pairs.
- **Without normalisation** (`normalise=False`), the coefficient is non-zero only if |cov0(x_i, x_j)| > 1. That makes the output depend on the raw scale. On scaled_offset_T1024, only the high-variance pair (2, 3) is fitted. The other 10 of 12 entries equal var0 of the target.
- **Deviations.**
  - Documented: the SPI is undocumented (the supplement says "four SPIs").
  - Canonical: the estimator agrees with the canonical lasso in the checks, but α = 1 makes the fitted model the intercept-only model on standardised data.
- **Proposed checks.**
  - Analytic constant (T−1)/T, exact, on the pipeline and the detrend pipeline.
  - From-scratch soft-threshold on raw scaled_offset, tolerance 1e-8 because the solver stops on a gap tolerance.
  - Symmetry.
  - The raw-scale direction test. It is clean for Lasso because inactive entries equal var0(x_j) exactly.

## misc.lmfit.ElasticNet

- **Canonical.** Zou & Hastie 2005, *JRSS-B* 67(2):301-320. This is the naive elastic net; their recommended estimator rescales by (1 + λ₂). The coordinate-descent solution is Friedman, Hastie & Tibshirani 2010, *J Stat Softw* 33(1), which is supplement ref [76]. Supplement ref [77] (Andersen et al., interior-point methods for cone programming) is not the algorithm used. That is a citation candidate.
- **Implemented.** `ElasticNet(alpha=1, l1_ratio=0.5, random_state=42)` uses the same coordinate-descent path, with l1_reg = l2_reg = 0.5·T. On z-scored data, with c = (T−1)/T, the exact minimiser is w = sign(r)·max(|r|c − 0.5, 0)/(c + 0.5) with MSE = c(1 − 2wr + w²). sklearn does not always return it:
  1. The solver first evaluates the duality gap at w = 0 (`_cd_fast.pyx:391-400`). With s = xᵀy = r(T−1) and l1_reg = 0.5·T, that gap is 0.5·yᵀy·(1 − min(1, 0.5T/|s|))² (`dual_gap_formulation_A`, `:138-159`).
  2. If the gap is ≤ tol·yᵀy = 1e-4·(T−1), the solver returns w = 0 with `n_iter_` 0. That happens whenever |r| ≤ 0.5·T/((T−1)(1 − √(2e-4))): 0.51522 at T=64, 0.50916 at 256, 0.50767 at 1024 and 0.50730 at 4096.
  3. Otherwise one cyclic update reaches the closed form and the next gap check stops (`n_iter_` 2).

  The exact-minimiser threshold is lower: 0.5·T/(T−1), which is 0.50794, 0.50196, 0.50049 and 0.50012. In the band between the two, the exact w lies between 0 and about 0.0048, and its MSE is up to about 4.8e-3 below (T−1)/T. pyspi returns (T−1)/T there.

  Probe `lmfit_fix/fix1.py` (14 datasets × 4 T, 552 ordered pairs):
  - Against the pure closed form, max |A − ref| = 3.50e-3. Six pairs fall in the band. One example is sparse_precision T=64 pair (1, 0), r = 0.5130: the closed form gives 0.981006, and pyspi gives 0.984375 = 63/64.
  - Against the closed form with the tol exit modelled, max |A − ref| = 2.8e-16.

  The synthetic sweep `fix2.py` (r step 1e-5) finds sklearn's switch from w = 0 to w ≠ 0 at the modelled bound, with 1 of 1604 boundary points misclassified. An earlier probe (`probe1.py`, six datasets) reported agreement with the pure closed form to ≤ 2.2e-16 only because none of its pairs fell in the band. The adversarial review found the band (`fable_wl/lm2.py`, `lm3.py`).
- **Consequences** (the pack hypothesis is **confirmed**):
  - Every pair with |r| ≤ 0.5·T/((T−1)(1 − √(2e-4))), about 0.507 to 0.515, gets the constant (T−1)/T. The SPI then jumps down by about 4.8e-3. In the probe, iid_gaussian, delayed_noise and nonlinear_dependence were entirely constant, and the ρ=0.3 pair of bivariate_gaussian was constant too.
  - Identical series cannot go below c(1 − (c − 0.5)/(c + 0.5))² ≈ 4/9. The probe gave 0.4468 at T=64 and 0.4445 at T=4096.
- **Deviations.** The SPI is censored at both ends. The "directed" label is nominal. The estimator itself is canonical (naive elastic net), but the solver's tol-based initial exit returns w = 0 in the band above the exact threshold. That is a solver-tolerance artefact with a jump discontinuity of about 4.8e-3, and it is a candidate finding.
- **Proposed checks.**
  - The from-scratch closed form with the initial tol exit modelled, exact (≤ 1e-12), on the kernel (now including sparse_precision and logistic_maps, which have pairs in the band) and the detrend pipeline. Report the deviation from the pure closed form separately: up to about 4.8e-3, and non-zero only in the band.
  - The raw-scale closed form, tolerance.
  - Symmetry.
  - The direction test.

## misc.lmfit.BayesianRidge

- **Canonical.** MacKay 1992, *Neural Computation* 4(3):415-447, gives the evidence fixed-point updates γ, λ = γ/‖w‖² and α = (N − γ)/SSE. Tipping 2001, *JMLR* 1:211-244, adds the Gamma hyperpriors. sklearn cites both (`_bayes.py:154-160`).
- **Implemented.** `BayesianRidge()`, with no `random_state`:
  1. Initialisation: α₀ = 1/var0(y), λ₀ = 1 (`_bayes.py:270-282`).
  2. SVD of X (`:293-305`).
  3. Loop over `_update_coef_` (`:405-425`) and the updates at `:330-332`, with all four hyperprior parameters equal to 1e-6.
  4. Stop when Σ|Δw| < 1e-3 (`:335`). The coefficient is then recomputed once (`:345-349`).

  The effective ridge penalty λ/α depends on the data. It was 0.484 on bivariate_gaussian_T256 pair (2, 3), against Ridge's fixed value of 1. A from-scratch numpy implementation of the same recursion agreed to ≤ 2.2e-16. The pack hypothesis "evidence iterations" is **confirmed**.
- **Deviations.**
  - Documented: the supplement mentions only λ1 = λ2 = 1e-6. It does not mention α1 = α2 = 1e-6 or the evidence estimation. The "directed" label is nominal.
  - Canonical: none for the estimator. The stopping rule is loose, so the result is a near fixed point. The hypothesis that this has only a second-order effect on the MSE is untested.
- **Range.** (0, (T−1)/T]. Identical series give 1e-19 to 1e-26.
- **Proposed checks.**
  - The from-scratch recursion, exact. This is the same algorithm written as independent code.
  - Maximise the log evidence over a numpy grid or with an optimiser, using an optimality criterion (≤ 1e-4 MSE) to quantify early stopping.
  - The detrend pipeline.
  - Symmetry.
  - The direction test.

## misc.lmfit.SGDRegressor

- **Canonical target.** The documentation says "ordinary least squares fit", so the canonical value is the OLS in-sample residual mean square, ((T−1)/T)(1 − r²) on z-scored data. The optimiser follows Robbins & Monro 1951 and Bottou 2010.
- **Implemented.** `SGDRegressor(random_state=42)` with sklearn defaults:
  - squared loss (CyHalfSquaredError), l2 penalty α = 1e-4
  - invscaling η = 0.01/t^0.25
  - shuffle each epoch from a seeded RNG
  - stop when the running epoch objective has not improved by more than tol = 1e-3 for 5 epochs

  The call chain is `_stochastic_gradient.py:1640-1676` → `_fit` (`:1580-1638`) → `_fit_regressor` (`:1712-1794`) → `_sgd_fast.pyx.tp` `_plain_sgd` (`:274`; updates at `:487-554`; stop at `:606-626`). It ran only 7 epochs at T ≥ 1024 and raised no ConvergenceWarning.
- **Deviations from OLS** (probe):
  - The excess MSE over the exact OLS value is up to 2.9e-3 at T=64 and up to about 5.8e-4 at T=4096 in this trace probe. These trace-probe figures are superseded by the check stage (record `closeness` and `what_changed`): over all datasets the excess reaches 1.98e-3 at T=4096 (sparse_precision), about 8x the null r² scale 1/T.
  - Coefficients are shrunk at T=64: 0.376 against OLS 0.420.
  - At T=4096 the intercept is 0.017 instead of about 0.
  - In 49 of 552 ordered pairs the MSE exceeds (T−1)/T. That is worse than predicting the mean, and 34 of those pairs are at T=4096. The maximum was 1.0003.
  - The result is deterministic for a fixed seed. Across seeds 0-9 it varies by up to about 4.6e-4.
  - Max |A − Aᵀ| is 1.43e-3, on sparse_precision T=4096 (`lmfit_fix/fix1.py`, all 14 datasets × 4 T). Every other dataset/T maximum is ≤ 4.7e-4. `probe3.py` printed only its top 8 rows by OLS excess, so it missed this maximum, which the review found. The asymmetry is optimiser noise, not direction.
  - At T ≥ 1024 this noise is comparable to the null-pair spread of r² (about 1/T) in the trace probe; the check stage found the excess about 8x 1/T at T=4096 and below 1/T at T=1024 only at pyspi's seed 42 (see the Checks section).

  These are candidates: the result does not match the documented OLS fit, and its value depends on the pyspi seed override.
- **Proposed checks.**
  - **Optimality** against from-scratch OLS: A ≥ OLS − 1e-12, with the excess reported per T.
  - **Statistical:** count the entries above (T−1)/T.
  - **Same-lineage** seed sweep, used only to measure sensitivity.
  - **Symmetry tolerance:** asymmetry ≤ 2e-3 but non-zero, on iid_gaussian, bivariate_gaussian, anm and sparse_precision. The per-dataset maxima in `fix3.py` are 1.9e-4, 4.1e-4, 2.7e-4 and 1.43e-3.
  - **Detrend pipeline.**
  - **Direction test** on raw data, with a tolerance.

## Open questions

1. Should lmfit_Lasso be classified as degenerate? It is a data-independent constant (T−1)/T under every default preprocessing path. It is also undocumented, so a further question is whether to drop it from the EEG selection or re-parameterise it, for example with α scaled to the data or chosen by cross-validation. That is outside this stage.
2. For ElasticNet, the documentation says "with the default parameters", and pyspi does use sklearn's defaults. Is the censoring (constant for |r| up to about 0.51, floor about 4/9) intended? Nothing in the supplement suggests the authors knew the fixed penalty dominates on standardised data.
3. For SGDRegressor at T=4096, is the non-convergence robust across sklearn versions? The stopping rule and the invscaling schedule have changed historically. This is a hypothesis, not tested. It matters for storage reproducibility, because the value depends on the seed and the shuffle RNG stream.
4. Is lmfit_BayesianRidge strictly monotone in |r| on z-scored data? If so it is redundant with r², like Ridge. That is a hypothesis to test with a dense r grid.
5. The pack's supplement line range (874-887) does not match supplement.txt, where the heading is at line 856. Other families' packs may carry the same 18-line offset.
6. Multi-realisation data (R > 1) would make `z[i]` (T, R) and `ravel(z[j])` length T·R. A shape error would then turn the whole matrix NaN. This was not probed, and it is probably unreachable through the Calculator.

## Checks

Files: `audit/checks/manifest/lmfit.json` (644 runs), `audit/checks/refs/lmfit.py` (comparison venv, numpy/scipy only, no scikit-learn import), and the probes `audit/checks/probes/lmfit_params.py`, `lmfit_grid.py` and `lmfit_sgd.py` (project venv). Results are in `audit/checks/out/results/lmfit.json`: **611 pass, 33 fail, 0 without a reference**. Every failure is a real mismatch described below. None is a reference error. Run IDs follow `<unit>|<dataset>|<T>|<path>|<tag>`.

All five units share three facts:
- On the default path (z-scored, ddof=1) every matrix is symmetric to floating point, except SGDRegressor, whose asymmetry is optimiser noise.
- On raw data (the `kernel` path), A[i, j] is the fit that predicts j from i. For the four deterministic units, the from-scratch value with predictor i and target j matches pyspi to ≤ 3.7e-16 relative. SGDRegressor is discussed below. The transposed orientation is off by 3.8% (var2_bivariate) to 2477× (scaled_offset). Runs: `*|1024|kernel|direction`.
- Lower values mean the target is more linearly predictable. On the direction datasets (delayed_noise, var2_bivariate, logistic_maps_chaotic, anm), the default path gives A[i, j] = A[j, i], so these SPIs carry no causal direction.

### misc.lmfit.Ridge: matches, 10

What was compared:
- **Analytic closed form.** ((T−1)/T)(1 − r²(T²−1)/T²) on all 21 datasets × 4 T (`|pipeline|closedform`): 84/84 exact, max_abs 2.19e-15.
- **From-scratch ridge on raw data** (`_kernels_basic_misc.ridge_mse`, `|kernel|scratch`): 33/33, max_abs 4.55e-13 at values of about 4e3, max_rel 3.5e-14.
- **Detrend path** (`|pipeline-detrend|detrend`): 12/12, max_abs 1.1e-16.
- **Direction runs:** 4/4.
- **Symmetry:** asym_max is 7.8e-16 on the pipeline path and 5.0e-15 on the detrend path.

Anchor 10: the result matches canonical Hoerl–Kennard ridge on analytic and independent references, and it matches the supplement's "MSE of a regression of y on x".

Confirmed candidates (new; none is in KNOWN_ISSUES):
- the "directed" label is nominal under z-scoring;
- the SPI is an affine function of r²;
- the documented independence and ANM rationale does not describe the quantity computed;
- the "Ride" typo, at supplement.txt:863.

### misc.lmfit.Lasso: degenerate, 0

What was compared:
- **Analytic constant (T−1)/T on every dataset.** Runs `|pipeline|constant` (84) and `|pipeline-detrend|constant` (12): 96/96 exact, max deviation 3.55e-15. That is 864 of 864 off-diagonal entries, including the identical pair, where |r|(T−1)/T = 0.99976 < α = 1. **No run was non-constant.**
- **Params probe** (`|pipeline|params`, 3 runs): `n_iter_` 0 and `coef_` 0 on every pair, so sklearn exits at the initial duality-gap check.
- **Raw data** (`|kernel|raw`, 15/15, max_rel 2.2e-16 against `enet_sklearn_tol`): the fit is active only when |cov0| > α. `n_active` equals `n_abs_cov0_gt_alpha_l1` in every run: scaled_offset|1024 has 2 of 12, cointegrated|1024 has 12 of 12, and bivariate_gaussian, anm and var2_bivariate have 0.

Anchor 0: the output is a constant. All four candidates are confirmed: degenerate, undocumented, scale-dependent on raw data, and rationale vs quantity. All are new.

Practical consequence for the EEG data: lmfit_Lasso is in the retained 185 (`audit/env/retained_metrics.txt:82`), and on the EEG workflow preprocessing (detrend then z-score, the `|pipeline-detrend|constant` runs) it is the same constant (T−1)/T, so it stores no information.

### misc.lmfit.ElasticNet: variant, 6

What was compared:
- **From-scratch minimiser with sklearn's initial duality-gap exit modelled** (`refs/lmfit.py:enet_sklearn_tol`, `|pipeline|exitmodel` and its detrend variant): 96/96 exact, max_abs 2.8e-16.
- **Params probe:** `n_iter_` is 0 or 2 exactly as modelled (5/5).
- **Raw data:** 15/15.
- **Direction runs:** 4/4.

The pure exact minimiser (`_kernels_basic_misc.elasticnet_1d_mse`, `|pipeline|pure`) fails on exactly the three dataset/T combinations that have pairs in the band 0.5T/(T−1) < |r| ≤ 0.5T/((T−1)(1−√2e-4)):

| Run | max_abs | r |
|---|---|---|
| logistic_maps\|4096 | 3.50e-3 | −0.5053 |
| sparse_precision\|64 | 3.37e-3 | 0.5130 |
| sparse_precision\|1024 | 1.38e-3 | 0.5025 |

Outside the band, max_abs_outside_band is ≤ 1.1e-16. The two control runs pass.

The synthetic r grid (`|iid_gaussian|<256,4096>|pipeline|grid`) agrees exactly with the exit-modelled reference. It places the last constant value at r = 0.50915, against the modelled bound 0.509161 (T=256), and at 0.50725, against 0.507296 (T=4096). The jump there is 4.87e-3 and 4.83e-3. At r = 1 the value is 0.44502 and 0.44448, the ~4/9 floor.

On the default path:
- 540 of 696 entries equal (T−1)/T;
- 36 of 84 runs are entirely constant, among them iid_gaussian, delayed_noise, var2_bivariate and logistic_maps_chaotic at every T.

Anchor 6: the result deviates from canonical in an interpretable way. There is a solver-tolerance band, plus censoring from the fixed penalty on standardised data, although sklearn's documented defaults are used. All five candidates are confirmed: censoring, the 4/9 floor, the nominal direction label, citation [77] (supplement.txt:2201-2203, interior-point cone programming), and the tolerance artefact. All are new.

### misc.lmfit.BayesianRidge: matches, 9

What was compared:
- **From-scratch MacKay/Tipping recursion** (`_kernels_basic_misc.bayesian_ridge_mse`): `|pipeline|recursion` 96/96 at max_abs 2.2e-16, and `|kernel|scratch` 27/27 at max_abs 1.4e-14. Identical-pair values of about 1e-19 to 1e-26 are compared in absolute terms.
- **Params probe** (`|pipeline|params`, 5/5): the same `n_iter_` (2 to 8) and the same alpha_ and lambda_ to 1.7e-15 relative.
- **Evidence optimality** (`|pipeline|evidence`, direct maximisation of the log evidence with Gamma(1e-6) hyperpriors): 15/20 pass the a-priori 1e-4 bound. The maximum |pyspi − MSE at the maximiser| converges across T:

  | T | max deviation |
  |---|---|
  | 64 | 8.5e-4 |
  | 256 | 3.1e-4 |
  | 1024 | 6.3e-5 |
  | 4096 | 1.0e-5 |

  pyspi usually sits below the optimum (signed_min −8.5e-4), because sklearn stops before lambda has grown fully. The maximiser equals the fully converged fixed point to ≤ 2.1e-9. The five failures (T=64: 2; T=256: 3) are a real effect of sklearn's tol = 1e-3 stopping rule. It stays far below the null sampling spread of the MSE, about √2/T.
- **r grid** (`|iid_gaussian|<256,1024>|pipeline|grid`, 2001 points): exact. The SPI is **not strictly monotone** in r. It has 6 increases at T=256 (max 1.39e-5) and 2 at T=1024 (max 5.1e-6), each where the iteration count changes.

Anchor 9: the estimator matches; the deviation from the evidence maximiser is a sklearn-default convention and lies within statistical tolerance. All four candidates are confirmed: the nominal direction label, being a function of (r, T), the partly documented prior, and rationale vs quantity. All are new. The monotonicity part of candidate 2 is refuted, as described above.

### misc.lmfit.SGDRegressor: variant, 5

What was compared:
- **OLS from scratch, tolerance |A − OLS| ≤ 1/(10T)**, fixed before running as one tenth of the null mean of r² (`|pipeline|ols`): 24/48 pass. The excess over OLS by T:

  | T | max excess | failed runs |
  |---|---|---|
  | 64 | 2.93e-3 (cointegrated) | 8/12 |
  | 256 | 1.56e-4 | 0/12 |
  | 1024 | 4.13e-4 | 5/12 |
  | 4096 | 1.98e-3 (sparse_precision) | 11/12 |

  The excess does not shrink with T.
- **Lower bound A ≥ OLS** (`|optimality`, including detrend): 54/54. Despite the run name this is not an optimality test: OLS with an intercept minimises the in-sample MSE over all linear predictors, so every linear fit, any SGD iterate included, passes it. It is a sanity bound; the closeness-to-optimum test is the 1/(10T) tolerance above.
- **Entries above the trivial bound (T−1)/T:** 0/120, 2/120, 11/120 and 30/120 at T = 64 to 4096. The maximum is 1.00032.
- **Determinism** (`|determinism`, same lineage: pyspi compared with itself, so it is a self-consistency check, not a correctness reference): the output does not change across global seeds 42, 0 and 12345 (difference 0.0), because pyspi passes random_state=42.
- **Seed sensitivity** (`|seedsweep`, same lineage): the spread across random_state 0 to 19 is up to 1.78e-3 (iid_gaussian|4096).
- **Asymmetry:** asym_max is up to 1.43e-3 (sparse_precision|4096).
- **Raw direction:** 3/4 runs pass. `cointegrated|1024|kernel|direction` matches the OLS orientation on 11 of 12 pairs. SGD on unscaled random walks is up to 2.66× the OLS MSE there.

Anchor 5 ("stochastic/optimiser unit checked only qualitatively"): no reference reproduces pyspi's SGD iterate. The independent checks only bound it (never below OLS, at most about 3e-3 above), and the closeness-to-optimum test fails 24/48, so anchor 7 ("an optimiser-based unit passing optimality checks") does not apply; the A ≥ OLS runs are a lower bound, not an optimality check. Anchor 6 would be the next candidate. Anchor 9 fails because the excess reaches about 8 × the null r² scale 1/T at T = 4096 (1.98e-3 vs 2.4e-4; at T = 1024 it stays below 1/T at pyspi's random_state=42, 4.13e-4 vs 9.8e-4, but not at every seed: the seedsweep run iid_gaussian|1024, random_state 0-19, reaches an excess of 1.10e-3). check_method is independent-impl, so the same-lineage cap does not apply. (Workflow 4b challenge: the earlier wording here, "passes only the qualitative and optimality checks; the family cap is 6", used anchor-7 language and a cap the rubric does not define.) All five candidates are confirmed: worse than the mean predictor, not the OLS fit, seed dependence and spurious asymmetry, the nominal direction label, and rationale vs quantity. All are new.

### Trace-stage corrections and kernel issue

- In all five records, the `implementation_trace` citations of `pyspi/calculator.py` now point at HEAD a1f20c7 (package code at HEAD a42ebc1 is identical outside pyspi/workflow/, so the line numbers hold): `_load_yaml` 230-267 (instantiation at 252-256) and `compute` 329-339 (they were 239-250 and 292-302).
- In the BayesianRidge `deviation_from_canonical`, the untested hypothesis "second order effect on the MSE" is replaced by the measured numbers.
- The "Ride" typo is at supplement.txt:863. The Bayesian prior sentence is at line 864.
- Kernel issue in `_kernels_basic_misc.elasticnet_1d_mse`: at the check stage its docstring said sklearn's coordinate descent "reaches [the exact minimiser] to its tol=1e-4". That is false in the band described above, where the mismatch reaches 3.5e-3. The family reference `refs/lmfit.py:enet_sklearn_tol` therefore models the initial gap exit, and the kernel function is used only as the "pure" comparison. The docstring was revised on 2026-10-15. It now says the function gives "the exact minimiser, NOT always what sklearn returns", describes sklearn's initial duality-gap exit (w = 0 when the gap at w = 0 is ≤ tol·||y_c||², tol = 1e-4), gives the band a < |s| ≤ a/(1 − √(2·tol)) with an MSE up to about 4.8e-3 below sklearn's (T−1)/T there, and points to `refs/lmfit.py:enet_sklearn_tol` for the exit.

## Open questions: answered

1. **Is lmfit_Lasso degenerate?** Yes. It returned exactly (T−1)/T on 864 of 864 entries across 21 datasets × 4 T and the detrend path, with `n_iter_` 0 everywhere. Verdict degenerate, score 0. Whether to drop it from the EEG selection or re-parameterise it (for example with α scaled to the data or chosen by cross-validation) is a decision for the maintainers, not something the evidence settles.
2. **Is ElasticNet's censoring intended?** The code cannot answer this, and the supplement gives no sign of it. The evidence shows how far it reaches: 540 of 696 default-path entries are constant, and so is every entry on the direction datasets var2_bivariate and logistic_maps_chaotic. Only the authors' intent, or a documented α choice, would settle it. Until then the SPI distinguishes only pairs with |r| above about 0.51.
3. **Is SGDRegressor stable across scikit-learn versions?** This could not be tested. Both venvs carry scikit-learn 1.9.1, and installing other versions is out of scope. Within 1.9.1 the value is deterministic: global seeds have no effect because random_state=42 is fixed. It is seed-sensitive at up to 1.78e-3, and its excess over OLS does not shrink with T. Running `probes/lmfit_sgd.py` under other scikit-learn versions on the same data would settle it.
4. **Is BayesianRidge strictly monotone in |r|?** No. It is an exact function of (r, T), but the tol = 1e-3 stop produces small increases (≤ 1.39e-5 at T=256) wherever the iteration count changes. Apart from those points it decreases in |r|, so it is nearly but not exactly rank-equivalent to r².
5. **Is the pack's supplement line range wrong?** Yes. §1.6.1 runs from line 856 to line 868, so the pack's 874-887 is off by 18 lines. Other packs were not checked here.
6. **What does R > 1 (multi-realisation) data do?** Not probed. No check in this stage exercises it, so it remains open.

## Optimisation

Workflow 3 (profile-based search for hotspots and avenues; nothing in `pyspi/` was changed). Walls from `audit/profiles/summary.json` (quiet machine, OMP_NUM_THREADS=1); percentages are shares of the profiled `compute_spi` frame in `audit/profiles/misc.lmfit.<model>__8x1024.pstats`.

| unit | 8x1024 (fresh) | 32x1024 | 118x1024 per window (fresh / family-warm) | measure_mode |
|---|---|---|---|---|
| misc.lmfit.Ridge | 0.043 s | 0.751 s | 10.5 s / 10.4 s | measured |
| misc.lmfit.Lasso | 0.046 s | 0.797 s | 11.1 s / 11.1 s | measured |
| misc.lmfit.ElasticNet | 0.046 s | 0.803 s | 11.2 s / 11.2 s | measured |
| misc.lmfit.BayesianRidge | 0.049 s | 0.855 s | 12.1 s / 12.0 s | measured |
| misc.lmfit.SGDRegressor | 0.062 s | 1.085 s | 15.1 s / 15.0 s | measured |

Shared versus incremental cost: nothing is shared (each identifier refits every ordered pair), so the family costs 59.7 s per window at 118x1024 (family-warm increments 10.4 + 11.1 + 11.2 + 12.0 + 15.0 s). All five identifiers are in the retained EEG selection, one EEG identifier each and no band variants: EEG cost at 118 channels = 59.7 s per window (the user's historical figure is about 44 s with an unknown channel count).

Hotspots (common pattern): the solver numerics are a small share; sklearn's per-call machinery dominates. `sklearn/utils/validation.py:check_array` 42.1% (Ridge), 36.8% (Lasso), 36.3% (ElasticNet), 39.4% (BayesianRidge), 23.9% (SGD); `sklearn/metrics/_regression.py:mean_squared_error` 15.5-21.3%; `inspect.signature` 10.4-13.5% in all, of which 5.3-6.2% is called per pair from `pyspi/statistics/misc.py:120` and the rest sits inside sklearn's `_validate_params` (get_params -> _get_param_names) and the validate_params wrapper of `mean_squared_error`; `sklearn/base.py:_validate_params` 8.4-14.2%. The numerics: `_ridge_regression` 9.3%, `enet_path` 9.9% / 9.8%, BayesianRidge `_update_coef_` 5.2% inside `_bayes.py:fit` 48.1%, and the SGD Cython loop shows as `_stochastic_gradient.py:_fit_regressor` 25.9% (17.2% self, opaque). Opaque shares 0.19-0.21. T_exponent 0.01-0.19, P_exponent 2.06: per-pair constant cost, about 0.75-1.1 ms per ordered pair.

Opportunities:
- lmfit-1 (Ridge; seed confirmed): batched closed form from one centred Gram matrix. Micro-benchmark `audit/profiles/micro/opp_lmfit-1.py` / `.json`: 10.9 s -> 3.8 ms at 118x1024 (2879x, one-call baseline), 524x at 32x1024, 44x at 8x1024 (candidate time includes building a fresh Data); max_abs_diff 8.9e-16, max_rel_diff 9.1e-16, NaN masks equal at 118x1024; max_rel 3.1e-15 against the 7x100 golden snapshot. Output risk within-tolerance (the candidate matches the baseline; nothing here says the SPI is a good statistic: on z-scored data it is an affine function of r^2, per the check stage).
- lmfit-2 (Lasso, ElasticNet; seed kept, unmeasured): soft-threshold closed form from the same Gram matrix, reproducing sklearn's duality-gap exit at w = 0; expected 1000-3000x. The batched form replaces the whole ordered-pair loop: base.py:multivariate is 99.967% (Lasso) and 99.968% (ElasticNet) of the profiled call, an Amdahl bound of about 3,000x at 8x1024. The remainder is a fixed cost (pairs fit t_fixed 1.0-1.15 ms of 11.1-11.2 s), so the bound is about 1e4x at 118x1024. The range itself comes from the replacement cost: lmfit-1's structure measured 3.5-3.8 ms per 118x1024 window, and an estimated 4-11 ms (not a timing; no candidate code exists) is allowed here for the soft threshold and the gap test. An earlier version cited only bivariate 99.7%, whose bound (about 380x) is below the claim. Output risk unverified-within-tolerance, with a discrete jump of up to 4.8e-3 possible for a pair at the exit-band edge (ElasticNet check). For Lasso at alpha = 1 on the default z-scored path every pair lies inside the exit band (|x^T y| = |r|(T-1) < T = l1_reg), so the edge cannot arise there; it matters for ElasticNet and for non-default inputs. On the default z-scored path Lasso is the constant (T-1)/T (check stage), so its whole 11.1 s per window computes a constant.
- lmfit-3 (BayesianRidge; seed kept, unmeasured): vectorised evidence recursion with a per-pair stopping mask; expected 300-1000x. base.py:multivariate is 99.958%, a bound of 2,390x at 8x1024 and about 9e3x at 118x1024 (t_fixed 1.27 ms of 12.0 s). The replacement is O(P^2 T) moments plus O(P^2 n_iter) masked updates, estimated at 12-40 ms per window (sklearn's n_iter_ has mean 3.96, median 4 and max 6 over the 13,806 ordered pairs of the synthetic 118x1024 window in a Workflow 4c throwaway probe, `scratchpad/wf4c_b2_lmfit_rank_corr/br_niter.py` run as `br_niter.py 118x1024`, which prints the distribution and saves no file, so a masked loop runs at most 6 array passes). Output risk unverified-within-tolerance (the check stage's from-scratch recursion matches sklearn to <= 2.2e-16 including n_iter_). Batching changes the float order of the moment sums, and through the discrete stopping rule (sum|delta w| < 1e-3) a pair whose convergence test sits within float noise of the tolerance could stop one iteration earlier or later than sklearn, moving its coefficient by up to about 1e-3 and the MSE by a similar order.
- lmfit-4 (all five): keep sklearn's solvers but hoist inspect.signature to __init__, run under sklearn.config_context(skip_parameter_validation=True, assume_finite=True) and compute the MSE with numpy. Per-unit profiled shares (overlapping, inflated by cProfile) of inspect.signature / mean_squared_error / _validate_params / _assert_all_finite: Ridge 10.4 / 20.4 / 8.4 / 9.6%, Lasso 11.8 / 21.3 / 10.6 / 8.4%, ElasticNet 12.0 / 20.9 / 10.9 / 8.2%, BayesianRidge 11.2 / 18.7 / 9.9 / 8.9%, SGDRegressor 13.5 / 15.5 / 14.2 / 6.9%. Of each inspect.signature share only 5.3-6.2% is the per-pair call at misc.py:120 that the hoist removes; the rest sits inside _validate_params and the mean_squared_error validation wrapper, so it is already in those shares. An unprofiled throwaway check at 8x1024 (56 ordered pairs, OMP_NUM_THREADS=1, not saved as a micro-benchmark, max_abs_diff 0.0 for every model) gave 1.90x (Ridge), 1.78x (Lasso, ElasticNet), 1.76x (BayesianRidge) and 1.57x (SGDRegressor). Workflow 4c re-ran an equivalent throwaway check for SGDRegressor (`scratchpad/wf4c_b2_lmfit_rank_corr/lmfit4_check.py`, which prints its timings and saves no file): 1.58x and 1.59x at 8x1024, 1.57x at 32x1024, bit-identical on these finite arrays. Expected ~1.9x / ~1.8x / ~1.8x / ~1.8x / 1.5-1.6x (the SGDRegressor record's range: its lower bound 1.5x sits below every timing); an earlier '~2x for all' overstated SGD. Output risk unverified-bit-identical (same solver code on the same arrays; assume_finite=True skips the finite-value check, so on NaN or inf input sklearn raises today while the stripped call would proceed, and the bit-identical claim covers finite input only). With lmfit-4 alone, SGDRegressor would keep about 15.0 / 1.57 = 9.6 s per window at 118x1024. This is the only non-parallel avenue for SGDRegressor and a cheap fallback for the others. On Ridge, Lasso, ElasticNet and BayesianRidge it is contained in lmfit-1 / lmfit-2 / lmfit-3, which replace the sklearn fit whose overhead it strips, so those four records list it in `subsumed_by` and build_csv ranks it there with no saving on top of the closed form (README "Optimisation columns", ranking rule 4).
- lmfit-5 (SGDRegressor): process pool over ordered pairs, expected 10-30x per-window latency on its own (at most ~4x inside a window worker). The ceiling is the 32 physical cores (Threadripper 3970X, 64 logical CPUs), and there is no core-hour saving. Window-level parallelism (xc-window-parallel) competes for the same cores: the two are alternatives, not factors that multiply, so the record does not list it in depends_on. With the ~8 memory-bounded window workers running, about 4 physical cores per worker remain, so inside a window worker the pool lowers this unit's latency by at most about 4x and patient throughput stays bounded by the core count. The record marks it `speedup_kind: throughput`, so build_csv ranks it after every per-window row (README "Optimisation columns", ranking rule 3). Output risk unverified-bit-identical: every fit builds its own RandomState from random_state=42 (misc.py:122), so no global RNG stream is consumed and scheduling cannot change a pair's shuffle order. (The briefing labelled SGD parallelism rng-stream; the code does not support that.)

Refuted or narrowed seeds: none refuted. The closed-form seeds hold for Ridge (measured) and are expected to hold for Lasso, ElasticNet and BayesianRidge; SGDRegressor has no closed form (the stopped SGD iterate is the definition), so it gets overhead trimming and parallelism only.
