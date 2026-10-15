# gpfit family: define, trace and check

Family `gpfit` contains two units, both from the class `GPModel` (`pyspi/statistics/misc.py:129-147`):

| unit_id | identifier | kernel |
|---|---|---|
| `misc.gpfit.DotProduct` | `gpfit_DotProduct` | `ConstantKernel() + WhiteKernel() + DotProduct()` |
| `misc.gpfit.RBF` | `gpfit_RBF` | `ConstantKernel() + WhiteKernel() + RBF()` |

Records: `audit/records/misc.gpfit.DotProduct.yaml` and `audit/records/misc.gpfit.RBF.yaml`. The read-only
probe scripts are in the session scratchpad under `gpfit/`:

- `p1_kernel.py`: kernel structure
- `p2_fits.py`: fits, own posterior mean, restarts
- `p3_scan.py`: 18 datasets × T ∈ {64, 256, 1024}
- `p4_mech.py`: degeneracy, closed form, scale
- `p5_env.py`, `p6_loo.py`: interpolation and LOO
- `p7_opt.py`: multi-start survey
- `p8_best_loo.py`: LOO of the multi-start optima
- `gpfit_fix/q.py` (sibling directory): MSE vs elementwise posterior-mean agreement, jittered closed form

Versions: scikit-learn 1.9.1, SciPy 1.17.1, NumPy 2.5.4.

The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records.
Anything marked *candidate* or *hypothesis* in the trace sections is resolved in the records'
`candidate_resolutions` and in `## Checks`. The regression
goldens are snapshots of current output; they are not used here as a reference.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared documentation

Supplement §1.6.2, p.18 (supplement.txt lines 870-879). The GitBook page
`spi-descriptions/miscellaneous.md` (lines 46-63) repeats it word for word:

> "Similar to the linear model fits, we also use Gaussian process [78] model fits as a nonparametric measure of
> influence of x on y. Here, we use a combination of kernels (with parameters chosen from default settings)
> from the scikit-learn package [2] and compute the MSE of their fit. The fits are computed for the
> dot-product kernel with inhomogenity parameter σ0 = 1 (DotProduct) and the radial basis function (RBF)
> kernel with length scale l = 1 (RBF). Each of these kernels are separately combined with the constant
> kernel (with a constant of 1.0) and the white kernel (with a noise level of 1)"

The keywords are *directed, nonlinear, unsigned, bivariate, contemporaneous*. These match
`config.yaml:1280-1290`. The class labels at `misc.py:132` are `misc, unsigned, unordered, normal, nonlinear,
directed`. Reference [78] is Williams & Rasmussen, "Gaussian processes for regression", NIPS 8. The
supplement dates it 1995; the proceedings appeared in 1996.

## Shared canonical definition

No paper defines a pairwise "GP fit MSE" statistic, so the canonical objects are:

- **GP regression.** From Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*:
  - Alg. 2.1: the posterior mean is f̄ = K_f (K_f + σ_n² I)⁻¹ y.
  - §5.4.1, eqs. 5.8-5.9: hyperparameters are set by type-II maximum likelihood (ML-II). The book notes
    that the marginal likelihood can have several local optima, each corresponding to a different
    interpretation of the data.
  - §5.4.2, eq. 5.12: the leave-one-out residual is r_t = [K⁻¹y]_t / [K⁻¹]_tt. This is the standard
    predictive error.
- **Kernels.**
  - Squared exponential with signal variance (R&W eq. 2.31): k = σ_f² exp(−(x−x')²/(2l²)) + σ_n² δ.
  - Dot-product kernel (§4.2.2): k = σ₀² + x·x'.
  - A GP with the dot-product kernel is the same model as Bayesian linear regression (§2.1, eqs.
    2.8-2.11). In one dimension with an N(0, 1) slope prior, the slope's posterior mean is
    b = x'y / (x'x + σ_n²).

The quantity the documentation describes is the training (in-sample) MSE of the posterior mean of y regressed
on x.

## Shared trace (pyspi → scikit-learn)

1. Preprocessing (`pyspi/data.py:184-197`). Detrending is applied only if `detrend=True`; the default is
   False. Z-scoring with `ddof=1` is the default. As a result every process has mean 0 and Σx² = T−1.
   `Data.to_numpy()` (`data.py:108-118`) returns shape (M, T, 1), so `z[i]` is a (T, 1) column.
2. `Calculator._load_yaml` (`calculator.py:230`, config loop `251-267`, constructor call `256`) builds
   `GPModel(kernel=...)`. `Calculator.compute` (`calculator.py:295`; `multivariate` call `329`,
   `np.fill_diagonal` `332`, table store `335`, `except` `336-339`, warning `354-355`; lines confirmed at HEAD
   a42ebc1, file unchanged since a1f20c7) calls `multivariate`, sets the diagonal to NaN, and turns any exception into a
   whole-NaN matrix.
3. `Directed.multivariate` (`base.py:84-94`) computes `A[i, j] = bivariate(data, i=i, j=j)` for every
   ordered pair. Nothing is mirrored.
4. `GPModel.__init__` (`misc.py:134-137`) sets `kernel = ConstantKernel() + WhiteKernel() + <kernel>()`.
   The terms are combined with `Kernel.__add__` (`kernels.py:364-367`), which builds a `Sum`
   (`kernels.py:838-871`). Every hyperparameter starts at 1.0 with bounds (1e-5, 1e5):
   - ConstantKernel: `kernels.py:1236`
   - WhiteKernel: `kernels.py:1366`
   - RBF: `kernels.py:1511`
   - DotProduct: `kernels.py:2159`

   The constant is **added**, not multiplied, so the RBF/DotProduct part has no amplitude hyperparameter.
5. `GPModel.bivariate` (`misc.py:139-147`) runs
   `GaussianProcessRegressor(kernel=self._kernel).fit(z[i], ravel(z[j]))`, then `predict(z[i])`, then
   `mean_squared_error`. Warnings are silenced only around `fit`.
6. sklearn `fit` (`_gpr.py`):
   - Constructor defaults (211-231): `alpha=1e-10`, `n_restarts_optimizer=0`, `normalize_y=False`,
     `random_state=None`.
   - The kernel is cloned (249), so pyspi's template is never changed. A probe confirmed this.
   - One L-BFGS-B run starts from the log of the initial values, θ = 0 (312-320, 658-674). Restarts at
     322-333 are skipped.
   - `_check_bounds_params` warns when a hyperparameter ends at a bound (338). pyspi suppresses these
     warnings.
   - The log marginal likelihood and its gradient are at 541-656. The final Cholesky factor and α come
     from 346-368.
7. sklearn `predict` (`_gpr.py:444-454`) returns `K(X, X_train) @ alpha_`. `WhiteKernel(X, Y)` returns
   zeros when Y is given (`kernels.py:1418-1419`), so the white term is left out of the prediction. The
   result is the de-noised posterior mean, as in R&W Alg. 2.1.
8. `mean_squared_error` (`_regression.py:524`, 586-588) is called with the arguments as
   (y_pred, y_true). MSE is symmetric in its arguments, so the order does not matter.

The MSE of a from-scratch posterior mean at sklearn's fitted hyperparameters matches pyspi's MSE to ≤ 1.9e-15
absolute in 48 probes (`p2_fits.py`). The posterior-mean vectors themselves differ elementwise by up to 4.3e-13 (RBF) and
1.1e-13 (DotProduct), from `np.linalg.solve` versus sklearn's Cholesky (`gpfit_fix/q.py`).
So the GP algebra holds no surprises. The interesting behaviour is in the model
specification, the hyperparameter search and the choice of score.

Direction: `A[i, j]` is the MSE of predicting process **j** (target) from process **i** (input). Lower
means i predicts j better. The convention comes from the code (`misc.py:145-147`, `base.py:91-93`). The
RBF unit confirms it empirically. On `nonlinear_dependence_T256` (x1 = x0² + noise),
`A[0,1] = 0.026` and `A[1,0] = 0.996`.

## misc.gpfit.DotProduct

**Documented.** A dot-product kernel with σ0 = 1, combined with a constant of 1.0 and white noise of 1. It
is described as "nonparametric" and labelled nonlinear and directed.

**Canonical.** A GP with k = c + σ₀² + x x' plus noise s is Bayesian linear regression (R&W §2.1, §4.2.2).
Its posterior-mean slope is b = x'y / (x'x + s).

**Implemented.**

- All three hyperparameters are re-estimated by ML-II. The documented 1.0 values are only the starting
  point.
- Z-scored data have zero mean, so the intercept direction is orthogonal to y. As a result c and σ₀ never
  affect the prediction, and the optimiser drives both to the 1e-5 lower bound. The fitted kernel is
  always printed as `0.00316**2 + WhiteKernel(noise_level≈1−r²) + DotProduct(sigma_0=1e-05)`.
- The only hyperparameter left is s. pyspi's value equals the closed form mean((y − b x)²) with
  b = x'y/(x'x + s) to ≤ 3e-14 relative. Examples:
  - `bivariate_gaussian_T64` 0→1: 0.8107422442473634 against 0.8107422442473563.
  - `anm_T256` 0→1: 0.333256243521899 against 0.333256243521898.
- The value also equals ((T−1)/T)(1 − r²) to within 6.2e-5 relative at T=64, 3.8e-6 at T=256 and 6.6e-9
  at T=1024 (iid), across 18 datasets.
- The matrix is symmetric to ≤ 8.1e-15 absolute.

**Deviations.**

- *From the documentation:*
  - The parameters are fitted, not fixed at the stated values.
  - The unit is linear, not "nonparametric" or "nonlinear". On `nonlinear_dependence_T256` (x1 = x0²)
    every entry is 0.994-0.996, the same as for independent pairs.
  - It is effectively undirected.
  - "MSE of their fit" means the in-sample MSE.
- *From the canonical model:*
  - c and σ₀ are two redundant, non-identifiable constant terms.
  - The slope prior is fixed at N(0, 1), which gives ridge shrinkage of O(1/T).
  - The score is a training MSE, not a predictive error or a likelihood.

**Candidates.**

1. DotProduct is a deterministic, symmetric transform of r², so the labels *nonlinear* and *directed*
   are nominal. This matches the `misc.lmfit.*` records.
2. The bound warnings are suppressed, and the documented constant and σ₀ values never apply.
3. `random_state` has no effect, and the fit is deterministic: 15 restarts reach the same likelihood
   and MSE.

**Proposed checks.** Details are in the record.

- *Kernel tolerance check:* a from-scratch Bayesian linear regression with scalar ML-II over s. Expected
  difference ≤ 1e-8 relative.
- *Pipeline analytic check:* ((T−1)/T)(1 − r²). Expected ≤ 1e-4 relative at T=64 and ≤ 1e-5 at T ≥ 256,
  with |A − Aᵀ| ≤ 1e-12.
- *Direction test* on `nonlinear_dependence` and `anm`. The expected result is that the unit carries no
  direction information.
- *`normalise=False` on `scaled_offset`:* exposes the convention that j is the target. In the probe,
  `A[0,1] = 3.64` and `A[1,0] = 0.036` for process scales 0.2 and 2.0.
- *`trending` with detrending:* checks that the shared-trend inflation goes away.

## misc.gpfit.RBF

**Documented.** An RBF kernel with l = 1, combined with a constant of 1.0 and white noise of 1. The MSE of
the fit is reported.

**Canonical.** An SE kernel with a signal variance σ_f² and a noise term, hyperparameters set by the
ML-II maximum, and the posterior-mean residual. A leave-one-out residual can be used if a predictive error
is wanted.

**Implemented.** The kernel is k = c + w δ + exp(−(x−x')²/(2l²)), fitted from a single start at θ = 0.
The RBF amplitude is fixed at 1. Observed behaviour:

- **Gaussian dependent pairs.** l ≈ 2-5, w ≈ 1 − r², and A is within 6.3e-2 relative of
  ((T−1)/T)(1 − r²) at T=256.
- **Independent pairs.** l goes to 2e3-1e5, w ≈ 1, and the prediction is about 0. So A = (T−1)/T:
  0.9961 at T=256 for 16 of 20 `iid_gaussian` entries, and 0.984375 = 63/64 at T=64.
- **Nonlinear functional pairs.** A is low in the functional direction. On `nonlinear_dependence_T256`,
  `A[0,1] = 0.026` and `A[1,0] = 0.996`. On `anm_T256`, `A[0,1] = 0.215 < A[1,0] = 0.314` and
  `A[2,3] = 0.065 < A[3,2] = 0.092`.

**Deviations from the documentation.** The parameters are fitted. The result depends on the local optimum
the single start reaches. The value is an in-sample MSE.

**Deviations from the canonical model.**

1. There is no σ_f², because the constant is added rather than multiplied.
2. The single start does not find the ML-II maximum.
3. The score is a training MSE.

**Candidates.** These are the main findings of this family.

1. **The ML-II maximum is degenerate, so the single start matters.** In the multi-start survey
   (`p7_opt.py`, 10 restarts), sklearn finds a higher log marginal likelihood than pyspi's single start
   on:
   - every `iid_gaussian` pair at T=64 and at T=256 (20/20 each);
   - 12/12 `delayed_noise_T64` pairs;
   - 11/12 `nonlinear_dependence_T64` pairs;
   - 8/12 `bivariate_gaussian` and `anm` pairs.
   - overall, 144 of 188 ordered pairs across 9 datasets × T ∈ {64, 256}.

   On pairs with no dependence, the better optimum interpolates: w sits at the 1e-5 bound,
   l < 1e-3 (below the input spacing), and the MSE is about 1e-10. The LOO MSE stays about 1. For
   example, on `iid_gaussian_T64` 0→1 the in-sample MSE is 9.84e-11 and the LOO MSE is 0.984. The
   single start itself converges properly (`CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`, 21
   iterations). pyspi's sensible-looking values on independent data therefore rest on the starting point
   θ = 0 and on the optimiser *not* finding the global ML-II optimum.
2. **The mechanism (hypothesis, supported by a numeric match to 1.1e-8 relative).** Because the RBF amplitude is
   fixed at 1, an RBF with l below the input spacing has a Gram matrix of about I. It is then
   indistinguishable in likelihood from unit-variance white noise, but `predict` treats it as signal.
   So f̄ ≈ y/(1 + w′), with w′ = w + 1e-10 (sklearn's jitter alpha), and MSE = (w′/(1 + w′))² (T−1)/T =
   9.960937e-11 for the fitted w = 1e-5 and T = 256. That matches the restart value 9.960938e-11 on
   `delayed_noise_T256` 0→1 to 1.1e-8 relative. Without the jitter the formula gives 9.960738e-11 (2.0e-5 relative), so the
   jitter is needed for the match (`gpfit_fix/q.py`).
3. **The single start can also miss real structure.**
   - *Missed real structure.* On `narrowband_f0-0.1_T64` 1→0, pyspi reports 0.483 (LOO 0.52). A fit
     with a log marginal likelihood 58.4 higher has MSE 0.023 and LOO 0.056, so the dependence is real
     and the reported value misses it.
   - *Interpolating local optimum.* On `shared_envelope_T64` 0→1, pyspi's own single start lands in an
     interpolating local optimum. It reports A = 1.1e-8, but the LOO MSE of that fit is 0.50. A better
     optimum (likelihood 11.5 higher) gives MSE 0.099 and LOO 0.146. At T=256 the same pair gives
     A = 5.3e-11 (LOO 0.374), against 0.539 in the reverse direction. This dataset is noise-free and
     exactly periodic, with repeated (x0, x1) pairs.
   - *Consequence.* The asymmetry |A − Aᵀ| (up to 0.99) can reflect which basin each direction reached,
     not predictability.
4. **Independent pairs collapse to (T−1)/T.** The value carries information only when the fit leaves
   this mode.
5. **Warnings are hidden.** The bound warnings for c and l are suppressed, and `random_state` has no
   effect.

**Proposed checks.** Details are in the record.

- *Kernel tolerance check.* A from-scratch numpy/scipy GP with the same parameterisation, the same
  bounds and the same start θ = 0. Expected ≤ 1e-6 relative when both land in the same basin. A basin
  disagreement is reported separately.
- *Optimality check.* A from-scratch multi-start ML-II. It should show that pyspi's fit is a local
  optimum below the best, and measure how many values change.
- *Exact identity check, same-lineage* (the hyperparameters come from sklearn). The identity is
  MSE = (w + 1e-10)² ‖α‖²/T. The interpolation formula and the LOO MSE are computed alongside it.
- *Direction check* on `nonlinear_dependence` and `anm`. This establishes that A[i, j] is the MSE of j
  predicted from i.
- *Gaussian-expectation tolerance check:* ((T−1)/T)(1 − r²) on `iid_gaussian`, `bivariate_gaussian`,
  `sparse_precision` and `gauss_ar1_pair`.
- *Qualitative check* on `trending` with detrending and on `scaled_offset`.

**Cost.** On P = 5 at T = 1024, one `multivariate` call took 54 s (DotProduct) and 91 s (RBF) while
other probes were running. The cost is O(T³) per likelihood evaluation, so T = 4096 is expected to be
about 60× slower and is optional in checks.

## Open questions

1. The supplement lists parameter values (σ0 = 1, l = 1, constant 1.0, noise 1). Were they meant to be
   fixed (`*_bounds="fixed"`), with ML-II being an unintended sklearn default? If they were fixed,
   DotProduct would still be linear. RBF would become a fixed-bandwidth kernel smoother with no
   interpolation mode.
2. Should the RBF score use a held-out or leave-one-out error (R&W eq. 5.12) instead of the training
   MSE? With the training MSE, the value can drop to 1e-8 to 1e-11 without any predictive skill.
3. Should the documentation say that ConstantKernel is added rather than multiplied? This leaves the RBF
   with no amplitude hyperparameter, and it is the hypothesised root of the degenerate optimum. A
   `ConstantKernel() * RBF() + WhiteKernel()` specification would be the standard one (R&W eq. 2.31).
   Not tested.
4. How often does pyspi's own single start land in the interpolating basin on real data? Only
   `shared_envelope` showed it among 18 synthetic datasets at T ≤ 256. Pairs with repeated or
   near-duplicate inputs, such as quantised or periodic signals, may be more exposed (hypothesis).
5. The BLAS and SciPy versions can change which basin L-BFGS-B ends in near the saddle between modes.
   This could make gpfit_RBF fragile across environments, in the same way as the documented SciPy 1.18
   drift for `anm`. Not tested.

## Checks

**Setup.**

- Manifest: `audit/checks/manifest/gpfit.json`, 163 runs (generator: scratchpad `gpfit_check/make_manifest.py`; the 11 `crossproc`,
  `fixedhp` and `amplitude` runs were appended after the review).
- References: `audit/checks/refs/gpfit.py`, run in the comparison venv. Every correctness reference is numpy/scipy only and uses
  `_kernels_basic_misc` (`gp_posterior_mean`, `gp_log_marginal_likelihood`, `gp_fit`), its own weight-space Bayesian linear
  regression (`blr_ml2`, `ridge_cf`) or its own LML for the C·RBF + W kernel (`amp_lml`, `amp_fit`). None imports scikit-learn, except
  the `scipy118` runs, which are labelled same-lineage.
- Threads: the reference script pins every BLAS to one thread before importing numpy. A first version ran the references with the
  comparison venv's default 64-thread OpenBLAS. At a fixed thread count the reference output is bit-identical across processes, but
  the ill-conditioned shared_envelope fits move with the thread count. In a throwaway probe (scratchpad `gpfit_check/fix/xp_ref.py`,
  not regenerated by `run_checks.py`), the shared_envelope T64 reference values shifted by up to 3.1e-7 (`scratch`) and 4.7e-9
  (`scipy118`) relative between 1 and 64 threads. Two runs at different thread settings therefore gave different
  `scratch` and `scipy118` numbers (e.g. shared_envelope T64 `scipy118` 2.7e-11 vs 4.7e-9). With the pin the numbers below are
  regenerable. The pyspi side needs no pin: `pyspi/__init__.py:4` sets `OMP_NUM_THREADS=1` before `audit/checks/pyspi_outputs.py`
  loads SciPy, so SciPy's OpenBLAS runs on one thread there (measured by `crossproc`, below).
- Probes: `audit/checks/probes/gpfit_probes.py`, run in the project venv.
  - `params`: sklearn's fitted `kernel_.theta`, LML, MSE, L-BFGS-B stop reason and iteration count, `n_restarts_optimizer`, and the number of
    draws from `gp._rng`.
  - `postmean`: theta, LML, MSE and the posterior-mean vector for pair 0→1.
  - `determinism`, `warnings_count`.
  - `crossproc`: pyspi recomputed in five fresh project-venv processes from the raw rows under different import orders and BLAS
    thread counts.
  - `fixedhp`: sklearn with `optimizer=None`, i.e. the documented hyperparameter values held fixed.
  - `amplitude`: sklearn's standard `ConstantKernel() * RBF() + WhiteKernel()` with 10 restarts (open question 3).
- Results: `audit/checks/out/results/gpfit.json`, **135 pass, 28 fail, 0 without reference**. None of the 28 failures is a script
  error. They fall into three groups:
  - Measurements of pyspi's behaviour (22): 9 DotProduct `kernel|scratch` runs (raw-data stopping point of the optimiser), and 11 RBF
    `multistart` plus 2 RBF `params` runs (optimality).
  - A refuted expectation (5 RBF `vsridge` runs). The record's proposed check expected the RBF MSE never to exceed the linear fit.
    It does exceed it on 30 of 50 pairs. This refutes the expectation; it is not a pyspi defect.
  - An alternative-kernel probe (1 RBF `amplitude` run, nonlinear_dependence). It tests a property of C·RBF + W, not of pyspi's
    kernel. It fails where the input spacing is below the length-scale bound (see below).
- Budget: T ≤ 1024, with P ≤ 3 (DotProduct) or P = 2 (RBF) at T = 1024. The pyspi side took 587 s in total. The longest run was
  `misc.gpfit.RBF|sparse_precision|1024|pipeline|gaussexp` at 42.9 s (under concurrent load). The whole forced re-run took about 18
  minutes wall time. T = 4096 was not run.
- Versions: the project venv has scikit-learn 1.9.1 and SciPy 1.17.1; the comparison venv has SciPy 1.18.1.
- Orientation: every reference matrix is M[i, j] = f(input x_i → target x_j).

### misc.gpfit.DotProduct: matches, score 9 (nearest-anchor reading: anchor 10's numeric half met on an independent reference, its supplement half fails)

- **Posterior arithmetic** (analytic `ridgecf`; independent-impl `posterior` and `postmean`). These runs use sklearn's fitted theta, so they
  check the posterior algebra exactly but not the hyperparameter fit.
  - `ridgecf`: closed-form penalised least squares with slope penalty w' and intercept penalty w'/(c + σ0²). This is the Bayesian
    linear regression posterior mean (R&W §2.1). It reproduces pyspi to max_rel 7.4e-15 on z-scored data (anm, bivariate_gaussian,
    iid_gaussian, nonlinear_dependence, T256) and 2.0e-13 on raw cointegrated T256, where c and σ0 do matter.
  - `posterior`: 5/5 pass, max_rel 0 (bit-identical in this run, with the reference BLAS on one thread).
  - `postmean`: f̄, LML and MSE differences are 0 (bivariate_gaussian T256, cointegrated T256 raw).
- **ML-II on z-scored data.**
  - `scratch` (global BLR type-II ML over (c + σ0², w)): 8/8 pass, max_rel 2.4e-9.
  - `params`: 4/4 pass. The polish gain is ≤ 1.8e-12, the projected gradient ≤ 9.7e-6, and every fit stopped on PGTOL.
  - `multistart` (21 starts per pair): 2/2 pass, max LML gain 3.2e-12.
  - Here the local optimum is the global one.
- **ML-II on raw data** (kernel path).
  - `scratch` fails 9 of 13 runs: max_rel 1.35e-4 on anm T256, 1.2e-6 on cointegrated T256, ≤ 8.6e-7 elsewhere.
  - The `single` runs isolate the cause. An own L-BFGS-B start from θ = 0 (`_kernels_basic_misc.gp_fit`, no sklearn) reproduces pyspi
    to ≤ 2.4e-9 (5/5 pass). Its gap to the global optimum is pyspi's gap: anm T256 sits 7.5e-5 below in LML with the MSE 1.35e-4
    relative off; cointegrated T256 sits 6.0e-6 below with 1.2e-6.
  - `misc.gpfit.DotProduct|cointegrated|256|kernel|params`: 8/12 fits stopped on the factr test, with θ moving up to 1.13 along the flat
    c + σ0² direction under a tight polish.
  - This is the optimiser's stopping point, not a model error. It does not occur under the default z-scoring.
- **Closed form.** `closedform`: 14/14 pass. The relative difference to ((T−1)/T)(1 − r²) is 5.4e-5 to 6.1e-5 at T = 64, ≤ 3.8e-6 at
  T = 256 and ≤ 6.9e-8 at T = 1024 (trending detrended: 6.8e-8). The convergence follows the O(s/T) ridge shrinkage.
- **Symmetry and direction.**
  - Under z-scoring, max|A − Aᵀ| ≤ 8.1e-15. On anm T256, A[0,1] = A[1,0] = 0.333256. On nonlinear_dependence T256 (x1 = x0²), both
    entries are 0.995776, the same as for independent pairs.
  - On raw scaled_offset T256, A[i, j] scales with var(x_j): A[0,3] = 97.30 and A[3,0] = 0.0392. The own oriented fits agree on 12/12
    pairs, which establishes that column j is the target.
- **Documented values held fixed** (`fixedhp`, 2/2 pass). The own posterior mean at θ = 0 equals sklearn with `optimizer=None`
  (max_rel 0). On bivariate_gaussian the fixed value differs from pyspi's fitted value by ≤ 9.2e-6 absolute (2.8e-5 relative) at
  T = 256 and ≤ 1.5e-4 (4.7e-4 relative) at T = 64. It equals the ridge (alpha = 1) closed form to ≤ 3.3e-14.
  - The two versions separate only on correlated pairs. The fitted noise level is about 1 − r², which is close to the documented 1
    when r is near 0. A scratch probe (`wf4b2_high-optimiser-other/p_gpfit.py`, own 1-D ML-II and fixed s = 1, no sklearn) gives
    a fixed-vs-pyspi relative gap of 1.5e-10 on nonlinear_dependence T256 (all |r| ≤ 0.05), inside rtol 1e-8, against 4.7e-4 on
    bivariate_gaussian T64, 1.5e-4 on anm T256 and 7.4e-6 on an EEG window (001_Baseline, 3 channels, T256). The `fixedhp` runs use
    bivariate_gaussian, whose pair correlations run from 0.06 to 0.83 at T64 (|r| 0.058-0.827) and from 0.02 to 0.82 at T256 (|r| 0.018-0.821); both contain strongly correlated pairs, so they do separate the two versions.
- **Other checks.**
  - `affine` (scaled_offset vs bivariate_gaussian): 1.0e-15.
  - `determinism`: bit-identical within one process.
  - `warnings`: 24 bound warnings on 12/12 pairs, none escape `GPModel.bivariate`.
  - `scipy118`: max_rel 1.7e-15.
- **Candidates.** All three are confirmed:
  1. The unit is a linear, symmetric function of r².
  2. Bound warnings are hidden. Refinement: c and σ0 sit at the 1e-5 bound on every z-scored pair but not on raw data (cointegrated
     T256: c 1/12, σ0 3/12).
  3. random_state is a no-op: 0 RNG draws and `n_restarts_optimizer` 0.
- **Score.** Score 9. The first version gave 7 and called anchor 7 a cap for ML-II units, but the README states no such cap. Anchor
  7 describes an optimiser unit that is checked only by optimality conditions. This unit has stronger checks. On the default
  (z-scored) path an independent global ML-II (`blr_ml2`, no sklearn) reproduces pyspi to 2.4e-9, and the posterior is fp-exact
  against the analytic ridge form. `check_method` is therefore independent-impl: the scratch runs decide the score, while the
  analytic `ridgecf` runs take sklearn's theta and the analytic closed form agrees only to 6.1e-5. It is not 10, because the
  supplement misdescribes the computation:
  - the documented values are fitted, not fixed (≤ 9.2e-6 at T = 256, ≤ 1.5e-4 at T = 64, `fixedhp`);
  - "nonparametric", "nonlinear" and "directed" do not fit a symmetric linear function of r².

  Mapping to the anchors: the numeric half of anchor 10 is met (2.4e-9 ≤ 1e-8 relative against an independent reference) and its
  "matches the pyspi supplement description" half fails. None of the anchor-9 clauses names this case literally (it is not a
  trivial value convention, an estimator-variance tolerance or a MATLAB/author reference), and the README has no anchor for
  "canonical computation reproduced, supplement wrong". 9 is the nearest-anchor reading (README "Score columns: original, rubric reading, adjusted") for that case, applied the same way to
  misc.coint.johansen_order-0_*. distance.hhg also fails anchor 10's supplement clause, but over a different supplement text
  (an RKHS kernel statistic on ranks of the variables), and its rubric reading is `anchor` 9 through anchor 9's
  estimator-variance clause (tie rounding on tied data; `records/distance.hhg.yaml` score_rubric_basis and score_rubric_note,
  README "Score columns: original, rubric reading, adjusted"), not a nearest-anchor mapping.

  `definition_under_review` stays false. The canonical objects (GP posterior mean, ML-II, R&W Alg. 2.1 and eqs. 5.8-5.9) are
  standard and not contested; what has no literature form is only the pairwise "GP fit MSE" wrapper, which is described above as
  the in-sample MSE of that posterior mean. The supplement's "parameters chosen from default settings" with σ0 = 1 and noise 1
  is a documentation discrepancy (deviation (a)), not a second competing standard. The class labels
  are metadata, as for kendalltau and pdist.cosine, so anchor 3 ("label does not describe the computation", used for identifier
  hyperparameters) does not apply. The raw-data stopping point (≤ 1.35e-4 relative, non-default path) is an optimiser tolerance.
  `new_finding` is true for candidate 1. It is not in KNOWN_ISSUES.md; REVIEW M9 discusses label structure only in general.

### misc.gpfit.RBF: variant, score 6 (anchor 6: deviates from supplement and canonical in an interpretable way)

- **Posterior arithmetic** (independent).
  - `posterior`: 12/12 pass. max_rel is 3.1e-11 (T64) and 5.3e-12 (T256) on shared_envelope, where the values are as small as
    5e-11, and ≤ 1.7e-14 elsewhere.
  - `postmean`: 3/3 pass. f̄ agrees elementwise to ≤ 1.9e-14 and the LML to ≤ 2.1e-12.
- **Single start.** `scratch` uses an own L-BFGS-B from θ = 0 in the sklearn box with default options. It passes 14/14 within rtol 1e-6.
  max_rel is 2.1e-7 on shared_envelope T64 raw, 4.3e-9 on shared_envelope T256 raw, and ≤ 1.7e-11 elsewhere. Raw rows, trending pipeline and detrended trending are all
  included. pyspi always reaches the same basin as an independent single start.
- **Local optimality** (`params`): 10/12 pass. Two runs fail marginally: anm T256 (polish gain 1.85e-3) and delayed_noise T256
  (1.31e-3). sklearn stops on the factr test on a near-flat l → 1e5 plateau: θ moves up to 5.8 log units while the MSE changes by
  ≤ 6.7e-6 relative. Over all 12 runs the MSE changes by ≤ 1.2e-4 under the polish. `n_restarts_optimizer` is 0 and no RNG draws are
  made.
- **Global optimality** (`multistart`): 0/11 pass.
  - The own multi-start (θ = 0 plus 20 log-uniform starts at every T) finds a higher LML (gain > 1e-3) on **100 of 128** ordered
    pairs. The value changes by > 1e-3 on 99 of them. The trace stage proposed this check on the kernel (raw) path. It was run on the
    pipeline path, which is pyspi's default. A first version used only 10 starts at T = 256; the counts above use 20.

    | Dataset | T | Pairs with higher LML |
    |---|---|---|
    | iid_gaussian | 64 | 20/20 |
    | iid_gaussian | 256 | 6/6 |
    | delayed_noise | 64 | 12/12 |
    | delayed_noise | 256 | 12/12 |
    | nonlinear_dependence | 64 | 11/12 |
    | nonlinear_dependence | 256 | 11/12 |
    | anm | 256 | 8/12 |
    | bivariate_gaussian | 256 | 8/12 |
    | narrowband_f0-0.1 | 64 | 4/6 |
    | shared_envelope | 64 | 2/12 |
    | shared_envelope | 256 | 6/12 |

  - The best fit interpolates on 71 pairs (w at 1e-5, l below the input spacing, MSE about 1e-10). For example, iid_gaussian T256 0→2
    goes from 0.996094 to 9.96e-11, with LML +2.773 and LOO 0.9961.
  - On the 12 best fits with l ≤ 0.1 × the minimum input spacing, the MSE equals (w'/(1 + w'))² mean(y²), with w' = w + 1e-10, to
    ≤ 5.5e-11 relative. This confirms the mechanism candidate numerically.
  - The largest LML gains are 419.4 (shared_envelope T256 1→0, interpolating, LOO 0.225 against pyspi's 0.600) and 58.4
    (narrowband_f0-0.1 T64 1→0).
- **Residual identity and LOO** (`identity`, same-lineage): 4/4 pass, max_rel 2.3e-11.
  - shared_envelope T64 0→1: in-sample 1.106e-8, LOO 0.4995.
  - shared_envelope T256: 5.26e-11, LOO 0.374.
  - iid_gaussian T256: every entry is 0.996094 = (T−1)/T, LOO 1.0039.
  - nonlinear_dependence T256 0→1: 0.0260, LOO 0.0275.
- **Gaussian expectation** (`gaussexp`, analytic ((T−1)/T)(1 − r²)): 8/8 pass. The error shrinks with T on every dataset:

  | Dataset | max_rel at T256 | max_rel at T1024 |
  |---|---|---|
  | bivariate_gaussian | 0.063 | 5.1e-4 |
  | sparse_precision | 0.0157 | 1.6e-3 |
  | gauss_ar1_pair | 7.7e-3 | 2.3e-4 |
  | iid_gaussian | 1.1e-3 | 4.5e-5 |

  No value falls below 0.01 on these datasets.
- **Versus the linear fit** (`vsridge`): 1/6 pass.
  - The Ridge(alpha = 1) MSE is the analytic closed form, equal to pyspi's `misc.lmfit.Ridge|*|*|pipeline|closedform` outputs to
    ≤ 2.2e-15. Those runs pass in `out/results/lmfit.json` at ≤ 2.4e-14 relative.
  - RBF's in-sample MSE **exceeds** the linear MSE on 30 of 50 pairs: iid_gaussian 6/6, bivariate_gaussian 10/12,
    nonlinear_dependence 8/12, anm T256 4/12, sparse_precision 2/6. The largest excess is 0.0063 absolute (anm T256), or 1.8% relative
    (bivariate_gaussian T256).
  - RBF is lower by up to 0.970 on nonlinear_dependence. Both anm T1024 pairs are below Ridge.
  - The expectation "RBF ≤ linear" is therefore refuted. ML-II prefers the l → ∞ "no dependence" mode on weak pairs, which fits worse
    in-sample than ridge.
- **Direction** (`direction`, 5/5 pass).
  - A[i, j] = MSE of x_j predicted from x_i.
  - nonlinear_dependence T256: A[0,1] = 0.0260 and A[1,0] = 0.9961.
  - anm T256: 0.2146 < 0.3140 (x0→x1) and 0.0652 < 0.0918 (x2→x3).
  - anm T1024: 0.2424 < 0.3595.
  - Raw scaled_offset: A[i, j] scales with var(x_j).
  - The own oriented fits agree on every distinguishable pair.
  - logistic_maps_chaotic T256 (x0 → x1, lagged coupling): A[1,0] = 0.913 < A[0,1] = 0.975. The asymmetry is contemporaneous
    predictability, not causal direction.
- **Documented values held fixed** (`fixedhp`, 3/3 pass). The own posterior mean at θ = 0 equals sklearn with `optimizer=None`
  (max_rel 0). Fixing the values changes pyspi's output substantially:
  - shared_envelope T64 A[0,1]: 0.661 instead of 1.1e-8;
  - iid_gaussian T256: 0.970-0.992 instead of 0.996094 everywhere (difference up to 0.026);
  - nonlinear_dependence T256 A[0,1]: 0.0387 instead of 0.0260.
- **Standard amplitude kernel** (`amplitude`, C·RBF + W, 2/3 pass). The reference uses its own LML for that kernel. At l = 1e-5 the
  RBF Gram is close to I, so the LML depends on C + w only. It compares the two ends of that ridge: interpolating (w = 1e-5, MSE
  1.0e-10) and white noise (C = 1e-5, MSE 0.996).
  - The two ends tie to ≤ 2.3e-4 in LML on all 12 iid_gaussian and delayed_noise pairs. They also tie (≤ 6.2e-8) on the 6
    nonlinear_dependence pairs whose input is x0 or x1.
  - The run fails on the 6 nonlinear_dependence pairs whose input is x2 or x3 (differences up to 11.2). Those inputs have minimum
    spacings of 6.4e-6 and 6.8e-6, below the 1e-5 length-scale bound, so the Gram at l = 1e-5 is not close to I there.
  - Under pyspi's C + W + RBF, the interpolating mode beats the flat no-signal mode by 2.7733 to 2.7735 on every iid_gaussian and
    delayed_noise pair. This is about ½ log T = 2.7726, the Occam cost of the forced unit-amplitude constant on z-scored data.
- **Other checks.**
  - `affine`: 2.4e-12.
  - `determinism`: bit-identical within one process on iid_gaussian T256 and shared_envelope T64.
  - `crossproc` (3/3 pass): five fresh processes on shared_envelope T64 and T256 and iid_gaussian T256.
    - Two processes with `pyspi_outputs.py`'s import order (numpy, then pyspi, then sklearn) reproduce the stored output bit for bit
      (12/12, 12/12, 6/6). So do a process that imports pyspi first and one with `OPENBLAS_NUM_THREADS=1`.
    - When `scipy.linalg` is imported before pyspi, SciPy's OpenBLAS keeps its 64 threads. The values then move by 2.8e-9
      (shared_envelope T64), 4.1e-11 (T256) and 1.0e-15 (iid_gaussian T256) relative, with no basin change.
  - `warnings`: 11 bound warnings on 10/12 pairs, none escape.
  - `scipy118`: 5/5, max_rel 2.7e-11 (shared_envelope T64, 6/12 entries bit-identical), with both sides' SciPy OpenBLAS on one
    thread. There is no basin change between SciPy 1.17.1 and 1.18.1. A SciPy-version effect below about 1e-9 cannot be
    separated from thread-count effects of the same size (`crossproc`).
- **Candidates.** All seven are confirmed:
  1. Single-start local optimum; the ML-II optimum is degenerate.
  2. The mechanism, to 5.5e-11.
  3. Interpolation without skill reached by pyspi's own start (shared_envelope).
  4. Missed structure (narrowband).
  5. (T−1)/T collapse.
  6. Hidden warnings.
  7. random_state no-op. The result is deterministic given the data and the BLAS thread count, and reproducible across processes
     for a fixed import order (`crossproc`).
- **Score.** Score 6, not 7: the optimiser unit fails the global optimality check, and the value depends on the start rather than on
  the ML-II maximum, which is degenerate in this parameterisation. It is not 5, because the posterior and the single-start path are
  checked quantitatively, not only qualitatively. `new_finding` is true for candidates 1-4. None is in KNOWN_ISSUES.md.

### Trace-stage corrections made in the check stage

- Both records and this note cited `calculator.py` at pre-a1f20c7 lines. These now read: `_load_yaml` def 230, config loop 251-267
  (constructor call 256); `compute` def 295, `multivariate` 329, `np.fill_diagonal` 332, table store 335, `except` 336-339, warning
  354-355. The lines were confirmed at HEAD a42ebc1; calculator.py is unchanged since a1f20c7.
- DotProduct candidate 2 said c and σ0 go to the lower bound "in every probed fit". That holds under z-scoring only (see above).
- Both `direction_convention` fields now record that the convention was confirmed by the check runs.

## Open questions: answered

1. **Were the documented values meant to be fixed?** The code and the evidence cannot settle intent. The supplement's "parameters chosen
   from default settings" fits both readings. The `fixedhp` runs measure what is at stake: own kernels at θ = 0, equal to sklearn with
   `optimizer=None`.
   - DotProduct (`misc.gpfit.DotProduct|bivariate_gaussian|{64,256}|pipeline|fixedhp`): fixing the values changes the output by
     ≤ 9.2e-6 at T = 256 and ≤ 1.5e-4 at T = 64. The fixed version equals the ridge (alpha = 1) closed form to ≤ 3.3e-14, i.e.
     lmfit Ridge(alpha = 1) on z-scored data.
   - RBF (`misc.gpfit.RBF|{iid_gaussian,nonlinear_dependence}|256|pipeline|fixedhp`, `|shared_envelope|64|pipeline|fixedhp`): fixing
     them gives a fixed-bandwidth smoother, so the interpolation mode disappears. shared_envelope T64 A[0,1] would be 0.661, not 1.1e-8.
     iid_gaussian T256 entries would range over 0.970-0.992, not sit at (T−1)/T. nonlinear_dependence T256 A[0,1] would be 0.0387, not
     0.0260.
   - A maintainer decision, or `*_bounds="fixed"` in `GPModel.__init__`, would settle it.
2. **Held-out or LOO error instead of in-sample MSE?** The evidence supports it. LOO (R&W eq. 5.12) keeps the skill-less interpolating fit
   at 0.4995 (shared_envelope T64) where the in-sample value is 1.1e-8. On genuine structure LOO stays close to the in-sample value
   (nonlinear_dependence 0.0275 vs 0.0260). This is a definitional choice for the maintainers; the computation is in
   `refs/gpfit.py:resid_identity`.
3. **Does C·RBF + White remove the degenerate optimum?** No. It changes a strict preference into a tie (`amplitude` runs on
   iid_gaussian, delayed_noise and nonlinear_dependence at T256). A first version of this answer said "yes", based on a throwaway sklearn
   probe in which 0 of 24 restarted fits interpolated. That showed only which end of a tie sklearn's restarts reached.
   - In pyspi's C + W + RBF, the interpolating mode beats the flat no-signal mode by 2.7733 to 2.7735 in LML (≈ ½ log T), so ML-II strictly
     prefers it.
   - In C·RBF + W, the interpolating fit (MSE 1.0e-10) and pure white noise (MSE 0.996) lie on one LML ridge. They tie to ≤ 2.3e-4
     wherever the input spacing exceeds the 1e-5 length-scale bound (18 of 24 pairs).
   - sklearn's 10 restarts (random_state 0) returned no interpolating fit (0/24, MSE 0.487-0.996 on iid pairs). The own 21-start
     optimum is interpolating on 5 of 24 pairs and beats the tie by more than 1e-3 on 18. sklearn's result is up to 9.0 below the own
     optimum in LML.
   - An in-sample score therefore stays degenerate and start-dependent with the standard kernel too. A predictive (LOO) score
     (question 2) is what removes the problem.
4. **How often does pyspi's single start interpolate?** Across the 22 RBF pipeline dataset/T combinations in this check, only
   shared_envelope (T64 and T256) has entries below 1e-3, 4 of 12 at each T.
   - Two of them, (0,2) and (2,0), are the bit-identical copy pair. Their LOO is about 0, so the low value is legitimate.
   - The other two, 0→1 and 2→1, lack skill (LOO 0.50 and 0.37).
   - Real quantised or periodic recordings were not tested.
5. **Can SciPy or BLAS versions change the basin?** Not on the tested data.
   - SciPy 1.17.1 → 1.18.1, at matched BLAS thread counts, changes values by ≤ 2.7e-11 on 5 datasets, including both shared_envelope
     sizes and narrowband (`scipy118`).
   - The OpenBLAS thread count changes values by up to 2.8e-9 (`crossproc`). Which count applies depends on import order:
     `pyspi/__init__.py:4` sets `OMP_NUM_THREADS=1`, and this reaches SciPy's OpenBLAS only if `scipy.linalg` was not loaded before
     `import pyspi`.
   - Neither effect moved a value across basins here. The factr stops on near-flat plateaus (`params`: θ moves up to 5.8 log units
     under a polish) mean that environment changes can move such values at the 1e-6 to 1e-4 level. Different BLAS builds were not
     tested.
6. **T = 4096 cost.** Measured per fit: 2.8 s (DotProduct) and 5.6 s (RBF) at T = 1024, under concurrent load. With O(T³) per LML
   evaluation, T = 4096 would take minutes per fit and roughly an hour or more for a P = 5 RBF matrix. T = 4096 was not run and should
   stay optional.

## Optimisation

Workflow 3 (profile-based search for hotspots and avenues; nothing in `pyspi/` was changed). Both units are listed in `profiles/infeasible_above_8x1024.txt`; their 118 values are pair-count scalings of the 8x1024 wall with no fixed cost. The 8x1024 cProfile runs timed out; the profiles are at 8x256 (`audit/profiles/misc.gpfit.<kernel>__8x256.pstats`).

| unit | 8x256 | 8x1024 (fresh / family-warm) | 32x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|---|
| misc.gpfit.DotProduct | 2.66 s | 131.7 s / 131.9 s | infeasible | 32,461 s (projected) | extrapolated-pairs-from-8 (T_exponent 2.81) |
| misc.gpfit.RBF | 6.08 s | 254.4 s / 253.9 s | infeasible | 62,711 s (projected) | extrapolated-pairs-from-8 (T_exponent 2.69) |

Shared versus incremental cost: nothing is shared between the kernels; each fits every ordered pair. Neither unit is in the EEG selection. Per 600-window patient: 1.9e7 s (DotProduct) and 3.8e7 s (RBF).

Hotspots: `sklearn/gaussian_process/_gpr.py:log_marginal_likelihood` is 91.2% (DotProduct, 898 evaluations for 56 pairs) and 94.7% (RBF, 1,748 evaluations) of the profiled call. Inside it `scipy/linalg/_decomp_cholesky.py:_cho_solve` 33.5% / 29.7% (sklearn forms K^-1 = cho_solve(L, eye) for the gradient), `_cholesky` 17.0% / 13.7%, `numpy dstack` (gradient tensor) 10.7% / 10.3%, `c_einsum` 9.3% / 8.1%; for RBF the Sum kernel call `kernels.py:__call__` (line 838) is 30.9% (RBF 16.2%, squareform 8.2%). Opaque shares 0.172 / 0.239. Prediction is under 1%.

Opportunities:
- gpfit-1 (DotProduct): Constant + White + DotProduct is Bayesian linear regression, K = (c + sigma_0^2) 11^T + x x^T + w I (rank two plus a diagonal; c and sigma_0^2 enter only as their sum), so the ML-II objective, its gradient, the posterior-mean prediction (linear in x) and the in-sample MSE all follow from six sufficient statistics per pair. The avenue replaces the whole per-pair fit/predict/mean_squared_error loop (pyspi/statistics/misc.py:139-147) with one vectorised optimiser over all 13,806 two-parameter objectives. Amdahl: base.py:multivariate is 99.998% of the profiled call at 8x256 (about 5e4x); `_gpr.py:fit` alone is 98.2% (55x), so fit alone is not enough. Expected 1000-10000x, assuming the batched optimiser (about 3-30 s per window against 32,461 s, i.e. 2.35 s per ordered pair). A per-pair scipy L-BFGS-B over the O(1) objective (about 10 ms per pair, estimate) would give only about 200x. Not measured. Output risk unverified-within-tolerance: same objective, different optimiser path; the check stage found pyspi equal to an independent global ML-II optimum to <= 2.4e-9 on the default path.
- gpfit-2 (both, shared with causal.anm): process pool over ordered pairs with one BLAS thread per worker; expected 10-30x per-window latency on its own (at most ~4x inside a window worker). The ceiling is the 32 physical cores (Threadripper 3970X, 64 logical CPUs): SMT adds little to Cholesky-bound fits, and 32 concurrent 8 MB kernel matrices exceed the shared L3. There is no core-hour saving. Window-level parallelism (xc-window-parallel) competes for the same cores and gives the same throughput. The two are alternatives, not factors that multiply, so the record does not list it in depends_on; inside one of the ~8 memory-bounded window workers (about 4 physical cores each) the pool lowers the unit's latency by at most about 4x. Output risk unverified-bit-identical (independent deterministic fits).
- gpfit-3 (RBF): custom LML and gradient for the same L-BFGS-B run, K^-1 from LAPACK dpotri on the existing factor and the trace terms as elementwise sums without the dstack/squareform tensor; removable share about 0.25, expected ~1.3x. Output risk unverified-within-tolerance (the RBF ML-II surface is degenerate on weakly dependent pairs; independent single-start runs agree with pyspi only to <= 2.1e-7, so the end point can move at that level).

Seed check: "gpfit: parallel only" holds for RBF apart from the modest gpfit-3; for DotProduct the profile plus the check-stage algebra point to the closed form (gpfit-1), which removes the O(T^3) term instead of spreading it over cores.
