# Family cov_prec_glasso: graphical-lasso covariance and precision (define, trace and check)

Units: `basic.cov.GraphicalLasso` (cov_GraphicalLasso, cov-sq_GraphicalLasso), `basic.cov.GraphicalLassoCV` (cov_/cov-sq_GraphicalLassoCV), `basic.prec.GraphicalLasso` (prec_/prec-sq_GraphicalLasso), `basic.prec.GraphicalLassoCV` (prec_/prec-sq_GraphicalLassoCV). Records: `audit/records/<unit_id>.yaml`. Probe scripts (read-only): scratchpad `cov_prec_glasso/probe1.py` to `probe7.py`. Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Versions executed: scikit-learn 1.9.1, numpy 2.5.4, scipy 1.17.1. The supplement cites scikit-learn v0.24.1.

The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Statements marked *hypothesis* or *candidate* in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`. The regression goldens are snapshots of current output and are not used as a reference.

## Shared pipeline (all four units)

1. **Preprocessing**: `pyspi/data.py:184-195` optionally detrends (default off) and then z-scores each channel with `ddof=1`.
2. **Instantiation**: `pyspi/statistics/basic.py:9-72` `Estimators(Undirected, Signed)`, with subclasses `Covariance` (kind `covariance`) and `Precision` (kind `precision`). `squared=True` adds the `-sq` prefix and makes `issigned()` return False.
3. **Fit**: `basic.py:31-44 _from_cache` builds `z = data.to_numpy(squeeze=True).T` with shape (T, P). It calls `sklearn.covariance.<estimator>().fit(z)` with **library defaults**, with **all warnings suppressed** (`:37-38`). The fitted object is cached in `data.covariance[<estimator name>]`, so cov, cov-sq, prec and prec-sq share one fit. A raised exception is not cached.
4. **Output**: `basic.py:46-54 multivariate` returns `mycov.covariance_` or `mycov.precision_` itself, not a copy. It first runs `np.fill_diagonal(..., nan)` on that array, which mutates the cached estimator. The `-sq` variants return `np.square`. This method overrides `Undirected.multivariate`, so no mirroring happens (REVIEW M1).
5. **Calculator**: `calculator.py:329-339` (HEAD a1f20c7: :329 `multivariate`, :332 `fill_diagonal`, :335 table store, :336-339 `except`) sets the diagonal to NaN again. Any exception becomes an all-NaN entry.
6. **sklearn**:
   - `GraphicalLasso.fit` (`_graph_lasso.py:547-587`) computes `S = empirical_covariance(X)`, which is `np.cov(bias=1)` (`_empirical_covariance.py:104-107`). Because the data were z-scored with ddof=1, **S_ii = (T-1)/T**, not 1.
   - `_graphical_lasso` (`:69-206`) runs Friedman's block coordinate descent with an **off-diagonal-only** l1 penalty (`:43-65`). It starts from 0.95·S off the diagonal (`:90-103`) and warm-starts the inner lasso (`:134-156`, `_cd_fast.pyx:1092`, pre-loop early exit at `:1173-1182`). It stops when |dual gap| < 1e-4 or after 100 sweeps (`:178-201`).
   - It returns its running `covariance_` and `precision_` iterates (`:206`). These are exactly symmetric because rows and columns are written together (`:169-177`), but they are **not exact inverses of each other**.

## basic.cov.GraphicalLasso

**Documented** (supplement §1.1.1, p.3-4): "Due to the z-scoring of the time series, the correlation and covariance matrices are equivalent and thus the covariance statistic is within [−1, 1] … the Lasso technique, which uses an ℓ1-regularization to sparsify the covariance matrix GraphicalLasso". The keywords are "undirected, linear, signed, multivariate, contemporaneous". The GitBook page `basic-statistics.md` gives "undirected, nonlinear, unsigned, bivariate" for Covariance, which contradicts this.

**Canonical**: Friedman, Hastie & Tibshirani 2008, *Biostatistics* 9(3):432–441:

  Θ̂ = argmin −log det Θ + tr(SΘ) + ρ‖Θ‖₁, with the diagonal penalised, and Σ̂ = W = Θ̂⁻¹.

The off-diagonal-only variant is Yuan & Lin 2007 (*Biometrika* 94:19–35). The KKT conditions are:
- W_ij = S_ij + α·sign(Θ_ij) on the support;
- |W_ij − S_ij| ≤ α off the support;
- W_ii = S_ii when the diagonal is not penalised.

**Implemented**: W = `covariance_` from sklearn GraphicalLasso with α = 0.01 (an absolute value on the (T−1)/T-scaled correlation matrix). The diagonal is not penalised: W_ii = S_ii exactly in the probe. The diagonal is then set to NaN, and `-sq` returns W².

**Deviations**:
- *Documented*:
  - The penalty sparsifies the **precision**, not the covariance. In the probe on sparse_precision T=1024, W had no exact zeros while Θ had 2.
  - "Covariance equals correlation" holds only up to the factor (T−1)/T and the shrinkage of at most α.
  - The executed library is 1.9.1, not the documented 0.24.1. A fetched copy of the 0.24.1 source shows the inner CD used `max_iter` directly, while 1.9.1 uses `max(20, max_iter)`, `do_screening`, and a pre-loop exit (`:1173-1182`). In 0.24.1 `enet_coordinate_descent_gram` (`_cd_fast.pyx:470-621` at tag 0.24.1) the first gap is computed inside the loop after a full coordinate pass (`:571-607`, break `:608-610`), with no screening, so an n_iter=0 return cannot occur there. The overall numeric effect versus 0.24.1 has not been measured; the 1.9.1 exit causes the stall below.
  - The GitBook keywords are inconsistent with the supplement.
  - The `-sq` variant is undocumented.
- *Canonical*: this is the Yuan–Lin variant (unpenalised diagonal) and the optimisation is approximate. See the candidates below.

**Candidates**:
- **Silent non-convergence.** Over the 76 audit data files, 14 fits emit a ConvergenceWarning that pyspi suppresses, and one raises an error (trending T=4096: FloatingPointError, so the whole matrix becomes NaN; with detrend=True the fit succeeds). Against a tight-tolerance refit, the covariance differs by at most 1.3e-4, except in the stall below (8.7e-4).
- **Stall on an easy problem (sklearn 1.9.1).** For var2_bivariate T=4096 (P=2), the returned W₀₁ equals the starting value 0.95·S₀₁ (0.206438) instead of the closed-form optimum soft(S₀₁, 0.01) = 0.207303.
  - Mechanism: the inner CD returns with n_iter 0 because the warm-start gap (3.4e-7) is at most enet_tol·‖row‖² (`_cd_fast.pyx:1173-1182`; probe: 3.378e-7 ≤ 4.722e-6). The outer dual gap stays at −3.7e-4 for 100 sweeps.
  - sklearn 0.24.1 has no pre-loop exit (see Deviations). For P=2 the inner lasso has one coordinate, so one pass gives the soft-threshold solution; that 0.24.1 returns the optimum here is inferred from its source, not run.
  - Setting enet_tol=1e-6 or 1e-8 (inner coefficient 0.207354 in 2 iterations) or mode='lars' recovers the optimum.
- **cov and prec are not exact inverses.** Check-stage numbers (`|kkt` runs, `ref_inv_err`, T=1024 pipeline unless stated): max|inv(W) − Θ| is 9.5e-6 to 2.44e-4 for GraphicalLasso and 5.9e-6 to 2.62e-4 for GraphicalLassoCV on sparse_precision/bivariate_gaussian (all four T, largest at bivariate_gaussian T=4096); on gauss_ar1_pair, logistic_maps and logistic_maps_bidir it is 1.6e-13, 1.9e-13 and 2.2e-16 for GraphicalLasso, 1.7e-13 for the converged GraphicalLassoCV fit (gauss_ar1_pair), and 1.2e-12 / 2.5e-8 for the non-converged (100-sweep) GraphicalLassoCV fits on logistic_maps / logistic_maps_bidir; it reaches 0.408 (GraphicalLasso, shared_envelope T=1024) and 0.509 (GraphicalLassoCV, narrowband_f0-0.25 T=4096). This refutes the briefing hypothesis that the covariance is computed from the inverse of the precision: both come from the solver.
- **The cached estimator is mutated** to have a NaN diagonal.
- **Scale mismatch**: ddof=1 z-scoring followed by a divide-by-T covariance.

**Proposed checks** (details in the record):
1. Analytic P=2 closed form (var2_bivariate, gauss_ar1_pair, logistic_maps(_bidir), contaminated). In the probe it already agrees to 1e-13 on 18 of 20 files.
2. A from-scratch numpy ADMM reference (Boyd 2011 §6.5) with an off-diagonal penalty, on sparse_precision, bivariate_gaussian, iid_gaussian, delayed_noise, nonlinear_dependence, var2_chain and scaled_offset. Criterion: tolerance 1e-3·max.
3. An objective/KKT optimality certificate on the ill-conditioned sets (narrowband, shared_envelope, cointegrated, trending, var2_bivariate).
4. Exact symmetry and permutation equivariance, to establish the A[i,j] convention.
5. Failure modes with and without detrending.
6. The [−(T−1)/T, (T−1)/T] range.

## basic.prec.GraphicalLasso

**Documented** (supplement §1.1.2, p.4): "The precision matrix is the matrix inverse of the covariance matrix … For normalized time-series data, the precision matrix is equivalent to the partial correlation … and is within [−1, 1]."

**Canonical**: the same graphical-lasso Θ̂. The partial correlation is ρ_ij|rest = −Θ_ij / √(Θ_ii Θ_jj).

**Implemented**: the raw `precision_` iterate Θ (α = 0.01) with the diagonal set to NaN; `-sq` returns Θ².

**Deviations**:
- *Documented*:
  - The value is **not** a partial correlation. Its sign is opposite and its magnitude is unbounded. For bivariate_gaussian T=256, prec[2,3] = −2.385 while the partial correlation is +0.811. Values reach |Θ_ij| ≈ 49 on shared_envelope.
  - It is not the exact inverse of the reported covariance.
  - The penalty sparsifies this matrix, not the covariance.
  - The version differs (0.24.1 documented, 1.9.1 executed), and `-sq` is undocumented.
- *Canonical*: the Yuan–Lin variant with approximate optimisation; the reported quantity is a precision entry, not a partial correlation.

**Candidates**: the same as for the covariance unit (non-convergence, the P=2 stall with a precision error of 9.9e-4, the inverse-consistency gap, mutation, the FloatingPointError leading to NaN, and scale).
- REVIEW M2 (last-bit precision asymmetry) does **not** reproduce here: Θ is bit-exactly symmetric. This is consistent with REVIEW counting twelve, not sixteen, precision variants with residuals (*hypothesis*: the four missing variants are these units).

**Proposed checks**:
1. The P=2 closed form Θ = inv(W) with a soft-thresholded W.
2. The ADMM reference.
3. The optimality certificate.
4. Symmetry and permutation tests.
5. Failure modes.
6. A qualitative comparison against the from-scratch partial correlation, expected to show the documented definition is not met.
7. Support recovery against `precision_true` on sparse_precision.

## basic.cov.GraphicalLassoCV and basic.prec.GraphicalLassoCV

**Documented**: the same text as above, with the regularisation "chosen through cross validation with five splits".

**Canonical**: the same estimator, with α chosen by K-fold CV of the Gaussian likelihood. There is no single canonical CV protocol.

**Implemented** (`_graph_lasso.py:916-1129`):
- The splitter is KFold(5), **unshuffled**, so the folds are contiguous time blocks.
- The initial grid is 4 log-spaced α values in [0.01, 1]·max|S_offdiag|, refined over 4 rounds, for 16 α values in total.
- Training-fold fits are warm-started along the path with max_iter = 10, and their warnings are suppressed.
- The test score is the Gaussian log-likelihood, with each test fold centred on its own mean.
- Ties go to the smaller α. α = 0 is scored but can never be selected.
- The final fit at α_ uses max_iter = 100.

**Deviations**:
- The "five splits" text is confirmed.
- Every point listed for the fixed-α units also applies here.
- The CV rule is specific to sklearn.

**Candidates**:
- **α_ varies enormously.** It ranges from 1.3e-8 (delayed_noise T=256) to α_max (iid_gaussian T=64), because the grid extends downward by 0.01× per round.
  - At α_ = α_max every off-diagonal of both W and Θ is exactly 0, which gives a constant SPI matrix.
  - Entries that should be zero can instead be roundoff of about 3.5e-18. This relates to REVIEW R7, where a float32 cast flushes such values to zero.
- **Many silent failures.** 22 final refits emit a ConvergenceWarning, and 3 raise a FloatingPointError in the final refit (shared_envelope T=256 and T=1024, narrowband_f0-0.1 T=64), giving NaN matrices. With detrend=True, shared_envelope T=256 still fails.
- **Large values on rank-deficient data.** |Θ| reaches about 95 on narrowband, and max|inv(W) − Θ| reaches 0.51.
- **Near-duplicate of the fixed-α units on Gaussian data.** On sparse_precision, α_ is 0.0097 at T=64 and 0.0085 at T=1024, close to the fixed 0.01 (but 0.024 at T=256 and 0.0049 at T=4096).

**Proposed checks**:
1. The P=2 closed form at the α_ read from `data.covariance['GraphicalLassoCV'].alpha_`.
2. ADMM at α_.
3. The optimality certificate.
4. An independent numpy re-implementation of the selection protocol, with a tolerance of one refined-grid step on α.
5. Symmetry and permutation tests.
6. Failure modes.
7. The degenerate all-zero output.

## Open questions

1. Does sklearn 0.24.1 (the documented version) return the closed-form optimum on var2_bivariate T=4096? Its source has no pre-loop inner-CD exit, so the 1.9.1 stall mechanism cannot occur there, but the output has not been run; that needs a 0.24.1 environment, which is not installed.
2. Should prec_* be reported as partial correlations, as the documentation says, or should the documentation be corrected? Either choice changes values or text.
3. Is a fixed absolute α = 0.01 appropriate when S is (T−1)/T-scaled, and should the silenced ConvergenceWarning be surfaced or the tolerances tightened? Doing so changes outputs (rel ≈ 1e-4, up to 5e-3).
4. For the CV units, is unshuffled contiguous KFold acceptable for autocorrelated series, and should α = 0 or α ≥ α_max (a constant all-zero matrix) be guarded against?
5. Do the cache mutation (NaN diagonal) and the uncached failure (four refits per failing estimator) matter to any downstream consumer? *Hypothesis*: they are value-neutral for the four SPIs.
6. Is the REVIEW M2 count of twelve precision variants with residuals exactly the 16 minus these four glasso variants? This should be confirmed against the storage data.

## Checks

**Files.** Manifest `audit/checks/manifest/cov_prec_glasso.json` (499 runs, run_id `<unit>|<dataset>|<T>|<path>|<tag>`), references `audit/checks/refs/cov_prec_glasso.py` (comparison venv, no sklearn import), R script `audit/checks/r/cov_prec_glasso.R` (R 4.6.1, glasso 1.11), probes `audit/checks/probes/cov_prec_glasso_fit.py` (project venv). Results: `audit/checks/out/results/cov_prec_glasso.json`, **414 pass, 85 fail, 0 without reference**. Compute: about 3.3 min for the whole family (pyspi side 78 s; T up to 4096 everywhere because a fit takes milliseconds to 0.5 s). The regression goldens were not used.

**Conventions fixed before comparing.**
- Problem: Θ̂ = argmin −log det Θ + tr(SΘ) + α Σ_{i≠j}|Θ_ij|, S = divide-by-T covariance of the array pyspi passes (raw on `kernel`, ddof=1 z-scored on `pipeline`, detrended then z-scored on `pipeline-detrend`). Every reference recomputes S from the raw dataset and checks it against pyspi's stored processed input (`ref_S_vs_pyspi_input_maxabs`).
- R glasso solves Friedman 2008 eq. 1 (maximise log det Θ − tr(SΘ) − ρ‖Θ‖₁). That is the same objective scale as sklearn's `_objective`/`_dual_gap` (`_graph_lasso.py:43-65`) with ρ = α: there is no factor 2 and no factor T. `penalize.diagonal=FALSE` is the sklearn convention. Each `|rglasso` run also fits `penalize.diagonal=TRUE`, which has W_ii = S_ii + α (`ref_R_pendiag_diag_shift_minus_alpha` ≤ 1.1e-16). It differs from pyspi off the diagonal by up to 3.4e-3 (cov) and 0.16 (prec) on well-conditioned data (`ref_maxabs_vs_R_penalize_diagonal_TRUE`; cov: 3.447e-3 on `basic.cov.GraphicalLassoCV|sparse_precision|256|pipeline|rglasso`, 2.196e-3 for GraphicalLasso at α = 0.01 on `basic.cov.GraphicalLasso|sparse_precision|4096|pipeline|rglasso`). So pyspi uses the unpenalised diagonal.
- KKT (Friedman 2008 eq. 2, diagonal unpenalised): `_kernels_basic_misc.glasso_kkt_residual(S, Θ, α)` evaluates the conditions at W = inv(Θ). That is the right certificate for the precision output. For the covariance output, `ref_kkt` evaluates the same conditions on pyspi's reported W, using the support and sign of pyspi's Θ (`kkt_W`). The pass threshold is 1e-4. sklearn's tol and enet_tol are also 1e-4, but they act on the dual gap, so every run reports the residual, sklearn's final dual gap and `n_iter_`.
- For GraphicalLassoCV, every reference uses the `alpha_` exported by a `full_fit` probe on the same dataset/T/path. In 33/33 `|fit` runs the probe's off-diagonals are bit-identical to the plain SPI runs. Following the family rule, this is a self-consistency check at sklearn's chosen α, with the score capped at 7.

### basic.cov.GraphicalLasso — variant, score 7 (rubric 7: optimiser-based unit passing optimality checks)

- **Analytic P=2 closed form** (W12 = soft(S12, α)), `|cf` runs: 16/19 pass, and converged fits agree to ≤ 2.25e-12. Fails:
  - `basic.cov.GraphicalLasso|var2_bivariate|4096|pipeline|cf`: 8.652e-4. pyspi W12 = 0.20643823, which equals the 0.95·S12 starting point to 9.6e-12; the optimum is 0.20730340. sklearn reports n_iter 100 and dual gap −3.73e-4. The same data on the raw `kernel` path converge exactly.
  - Early stops: `|var2_bivariate|1024|pipeline|cf` 1.333e-5 (one sweep, dual gap 7.8e-5 < tol) and `|logistic_maps_chaotic|1024|pipeline|cf` 5.71e-5.
- **Independent ADMM**, `|admm`: 17/17 within 1e-3·max|ref|, max abs 5.64e-5.
- **R glasso**, `|rglasso`: 10/11 pass, max abs 7.86e-5. The fail is the var2_bivariate T=4096 stall (8.65e-4).
- **KKT on W**, `|kkt`: 34/37 runs ≤ 1e-4 (well-conditioned max 4.22e-5). Fails: cointegrated T=256 1.33e-4, the stall 8.65e-4, and trending T=4096 FloatingPointError.
- **Non-convergence.** 13 of 36 fits end non-converged. W still stays within 1.33e-4 of the ADMM optimum on all of them except the stall. The objective excess reaches 0.031 (shared_envelope T=1024).
- **Scale.** W_ii − S_ii ≤ 4.4e-16 and W_ii − (T−1)/T ≤ 8.9e-16.
- **Statistical**, `|stat` (sparse_precision, kernel): 3/4 pass. At T=1024, max_z is 2.29 because the sample covariance itself is 3.20 sd from cov_true on entry (2,3) (`ref_sample_S_max_z_sd`). That is a property of the data draw, not of the estimator.
- **Range**, `|range`: 6/6; |W_ij| ≤ (T−1)/T and |W − S| ≤ α + 4e-5.
- **Other checks:** symmetry is bit-exact on 65 runs; permutation 3/3 (≤ 3.8e-5); `-sq` exact; failure modes 6/6.
- **Why 7, not 9:** the external author package agrees within solver tolerance on well-conditioned data. But the solver silently returns its starting point on an easy P=2 input, silently stops without converging on ill-conditioned inputs, and can return an all-NaN matrix.

### basic.prec.GraphicalLasso — variant, score 6 (rubric 6: deviates from the supplement in an interpretable way and fails optimality on ill-conditioned inputs)

- **Closed form**, `|cf`: 16/19 pass (≤ 3.32e-13). The fails are the same three inputs as for cov: the stall 9.852e-4, then 1.43e-5 and 5.79e-5.
- **ADMM**: 17/17 pass, max abs 2.36e-4.
- **R glasso**: 8/11 pass. Well-conditioned data agree to ≤ 2.35e-4. Fails: shared_envelope T=1024 0.149, narrowband_f0-0.25 T=256 0.101, and the stall.
- **KKT on Θ**, `|kkt`: only 16/37 runs ≤ 1e-4.
  - bivariate_gaussian T=4096 has 2.35e-4 although sklearn's own dual gap is 7.3e-6.
  - Ill-conditioned inputs reach 0.212.
  - max|Θ − Θ_ADMM| reaches 0.267 on shared_envelope T=256, where max|Θ_ij| = 49.4.
- **Statistical**, `|stat`: 4/4 pass (max_z ≤ 1.55). Nearest-neighbour entries are negative at every T. The 12 non-neighbour entries have 0, 0, 2 and 6 exact zeros at T = 64, 256, 1024 and 4096.
- **Documented partial correlation**, `|pcorr`: 0/4. The sign is opposite on 100% of nonzero entries, and the maximum difference is 3.21 (bivariate_gaussian T=256). pyspi returns the raw precision entry.
- **Other checks:** symmetry is bit-exact on 63 runs; permutation ≤ 4.4e-5.

### basic.cov.GraphicalLassoCV — variant, score 6 (family cap 7; self-consistency at sklearn's alpha_)

- **α selection.** alpha_ ranges from 3.05e-8 (var2_bivariate T=1024) to 0.2069 = alpha_max (iid_gaussian T=64).
- **CV selection**, `|cvsel`: 6/6 independent numpy re-scorings select exactly sklearn's alpha_ over its own grid. They use contiguous KFold(5), exact ADMM fold fits, the test fold centred on its own mean, and ties going to the smaller α. sklearn's 10-iteration fold fits shift mean scores by ≤ 2.84e-4.
- **Closed form at alpha_**, `|cf`: 16/19 pass. Fails:
  - `basic.cov.GraphicalLassoCV|logistic_maps_bidir|1024|pipeline|cf`: 7.06e-3. W12 equals the 0.95·S12 start to 1.5e-7, so this is a stall in the final refit.
  - var2_bivariate T=64 pipeline 3.59e-3 and T=256 pipeline 4.95e-4: 100 sweeps, not converged.
- **ADMM**: 14/17. The three fails (2.7e-6 to 6.9e-5 absolute) are at solver-tolerance level. They exceed 1e-3·max|ref| because alpha_ is near alpha_max, which leaves only small entries.
- **R glasso**: 8/8, max abs 5.64e-5.
- **KKT on W**: 32/37. Fails: logistic_maps_bidir 7.06e-3, logistic_maps 6.35e-3, trending T=4096 4.61e-4, and two FloatingPointErrors (shared_envelope T=256 and T=1024). 17 of 35 final refits are not converged, and none of them produces a warning the caller can see.
- **Degenerate output**, `|degen`: 6/6 consistent. alpha_ = alpha_max on iid_gaussian T=64/256/4096 and on nonlinear_dependence T=1024, giving a constant all-zero matrix. Two entries are roundoff values of 1.4e-17, 3.5e-18 and 6.9e-18 instead of exact 0.
- **Failure modes**, `|fail`: 7/8. `basic.cov.GraphicalLassoCV|shared_envelope|1024|pipeline-detrend|fail` also raises FloatingPointError, which the trace did not predict.
- **Other checks:** symmetry is bit-exact on 68 runs; permutation ≤ 3.8e-5; range 6/6; `-sq` exact.
- **Score 6 (below the cap of 7):** the final refits fail optimality on easy P=2 inputs by up to 7e-3, and three inputs give all-NaN output (shared_envelope T=256 and T=1024, the latter also after detrending; narrowband_f0-0.1 T=64).

### basic.prec.GraphicalLassoCV — variant, score 6 (family cap 7)

- **Selection evidence** is the same as for the cov unit.
- **Closed form**, `|cf`: 15/19. logistic_maps_bidir T=1024 is off by 2.28e-2 (the stall); var2_bivariate T=64/256 by 4.42e-3 and 5.91e-4; var2_bivariate T=256 kernel by 6.2e-7.
- **ADMM**: 14/17, the same small-entry fails as the cov unit.
- **R glasso**: 8/8, max abs 2.53e-4.
- **KKT on Θ**: 16/37.
  - bivariate_gaussian T=4096 has 2.53e-4 with sklearn dual gap 1.0e-5.
  - Narrowband reaches 0.232.
  - max|Θ − Θ_ADMM| reaches 0.336 (narrowband_f0-0.4 T=1024, where max|Θ_ij| = 74.9).
- **Statistical**: 4/4 pass.
- **Partial correlation**, `|pcorr`: 0/4; the sign is opposite on 83–100% of nonzero entries.
- **Degenerate output**: 6/6 consistent.

### Candidates

All candidates are confirmed; none is refuted. Per-record evidence is in `candidate_resolutions`, and none of the candidates is in KNOWN_ISSUES.md. The REVIEW M1/M2 items are cited there as already documented.

- **Silent non-convergence.** 13 GraphicalLasso and 17 GraphicalLassoCV non-converged fits. `ref_warnings_seen` is 0 through the HEAD Calculator hook `pyspi.calculation._capture_metric_warnings`, while a direct sklearn fit of the same array emits ConvergenceWarning (`ref_convergence_warnings_direct` ≥ 1).
- **The P=2 stall.** Confirmed for GraphicalLasso on var2_bivariate T=4096 and newly also for the GraphicalLassoCV final refit on logistic_maps_bidir T=1024.
- **The cov/prec inverse gap.** Confirmed; see the corrected numbers in the GraphicalLasso section above.
- **Cache mutation.** Confirmed and value-neutral: the `|side` runs give bit-identical outputs whichever identifier runs first.
- **FloatingPointError.** Gives an all-NaN SPI and four refits per failing estimator.
- **(T−1)/T scale.** Confirmed.
- **CV candidates.** The α range, the degenerate all-zero output with roundoff, the protocol, and near-redundancy with GraphicalLasso on Gaussian data are all confirmed. On sparse_precision T=1024, cov differs by 1.5e-3 and prec by 6.6e-3.

### Trace-stage corrections

- The GraphicalLassoCV failure candidate is extended: shared_envelope T=1024 also fails with `detrend=True`.
- The inverse-consistency bullet above is replaced with check-stage numbers.
- The GraphicalLassoCV records were checked for the copied identifier mismatch (`cov_GraphicalLasso and prec_GraphicalLasso`, `d.covariance['GraphicalLasso']`). Both records already name `cov_GraphicalLassoCV` / `prec_GraphicalLassoCV` and `data.covariance['GraphicalLassoCV']`.
- No trace-stage definition was refuted.
- The `pyspi/calculator.py` citations in the four records and in step 5 above are moved to HEAD a1f20c7: `_load_yaml` instantiation :251-264, `multivariate` call :329, `fill_diagonal` :332, table store :335, `except` :336-339, warning :354-355.
- The proposed sym/perm and failure-mode checks in the four records now carry `reference_kind: same-lineage` instead of `analytic`, because they compare pyspi with itself or with a prediction about pyspi's own behaviour.

### Reference notes

- Reference kinds in results.json: `|sym`, `|perm`, `|fail`, `|side`, `|cf-sq`, `|sym-sq` and `|fit` compare pyspi output with pyspi output (or with the trace prediction about pyspi's behaviour) and are labelled `same-lineage`. They support only the symmetry, equivariance, failure-mode and `-sq` statements. The verdicts and scores rest on `|cf` (analytic), `|admm`, `|kkt`, `|cvsel` (independent-impl) and `|rglasso` (external-pkg). `|range`, `|degen`, `|pcorr` and `|stat` stay `analytic`: each tests pyspi's output against a condition derived on paper and evaluated on the independently recomputed S. Relabelling changed no value and no pass/fail (414 pass, 85 fail before and after).
- The ADMM stops on the KKT residual (≤ 1e-12·max(1, max|S|)) rather than on an absolute primal/dual residual. An absolute rule stopped too early on raw scaled_offset, where Θ entries are about 1e-2 and S about 1e2.
- The reported `ref_admm_kkt` is ≤ 7.4e-11 on all z-scored inputs. It is 1.1e-8 and 4.8e-7 on raw scaled_offset (relative to max|S| ≈ 98), against a criterion of 0.039 there.

## Open questions: answered

1. **sklearn 0.24.1 and the stall.** Not settled. No 0.24.1 environment exists. The checks show that 1.9.1 returns its starting point on z-scored var2_bivariate T=4096 (`|var2_bivariate|4096|pipeline|cf`, 8.652e-4) but converges on the raw data (`|kernel|cf`, 0.0). The same kind of stall also occurs in a GraphicalLassoCV final refit (logistic_maps_bidir T=1024). To settle it, run `GraphicalLasso().fit` under sklearn 0.24.1 on the z-scored var2_bivariate T=4096 array and compare with the closed form 0.20730340.
2. **prec_* as partial correlations or corrected documentation.** The evidence fixes what is computed: the raw precision entry Θ_ij. Its sign is opposite to the partial correlation on 83–100% of nonzero entries, and it is unbounded: |Θ_ij| reaches 49.5 for GraphicalLasso (`basic.prec.GraphicalLasso|shared_envelope|1024|pipeline|kkt`) and 94.6 for GraphicalLassoCV (`basic.prec.GraphicalLassoCV|cointegrated|1024|pipeline|kkt`; 91.1 on shared_envelope T=64, 74.9 on narrowband_f0-0.4 T=1024), against the documented [−1, 1]. Both forms are recorded and `definition_under_review: true` is set on the two prec records. The choice between them is a maintainer decision and changes either values or text.
3. **Surfacing the warning or tightening tolerances.** 30 of the 71 non-erroring fits checked end non-converged without a visible warning. Against the exact optimum (ADMM, and R glasso at thr=1e-12):
   - covariance outputs are within 1.33e-4, except the stalls (8.65e-4 GraphicalLasso, 7.06e-3 GraphicalLassoCV);
   - precision outputs miss by up to 0.27 (GraphicalLasso) and 0.34 (GraphicalLassoCV) on ill-conditioned data.
   Tightening tolerances would move the outputs by these amounts. Whether to do so is a policy decision.
4. **A fixed absolute α = 0.01 on (T−1)/T-scaled S.** The scale is confirmed: W_ii = (T−1)/T to 8.9e-16. With unit-variance data, the effective penalty is α·T/(T−1), i.e. 1.6% larger at T=64 and 0.02% at T=4096. The code and supplement do not say whether a fixed absolute α is intended, so this cannot be settled from the evidence.
5. **Contiguous KFold and α ≥ alpha_max.**
   - The contiguous protocol is what sklearn runs. An independent re-scoring with contiguous folds reproduces alpha_ on 6/6 datasets.
   - alpha_ = alpha_max occurs on iid_gaussian T=64/256/4096 and on nonlinear_dependence T=1024, giving constant all-zero matrices (with 1e-17 roundoff entries).
   - Whether contiguous folds are acceptable for autocorrelated series was not tested. To settle it, compare alpha_ under contiguous and blocked/shuffled folds on var2_bivariate and gauss_ar1_pair.
   - A guard against α ≥ alpha_max is a design decision.
6. **The cache mutation and repeated refits.** Value-neutral for these four SPIs. In `|side` runs:
   - the cached estimator's diagonals are NaN after `multivariate`;
   - the returned array is the cached array;
   - all four identifiers give bit-identical outputs whichever runs first.
   On failing inputs the estimator is refitted 4 times (`n_fits` 4), which costs only time. Consumers that read `data.covariance[...]` directly were not surveyed.
7. **REVIEW M2's twelve precision variants.** Confirmed from the storage data. `docs/storage_comparison/explore/symmetry_105_64.json` lists exactly twelve asymmetric precision identifiers: prec/prec-sq × {EllipticEnvelope, EmpiricalCovariance, LedoitWolf, MinCovDet, OAS, ShrunkCovariance}. The four glasso precision identifiers are symmetric (eq_frac 1.0, rel_asym_max 0.0). `reports/headroom_w64/headroom.jsonl` stores them in mode `sym` for patients 004, 105 and 111, against `sym_res` for the other twelve. This agrees with the bit-exact symmetry in every check run.

## Optimisation

Workflow 3, group basic. Walls are from `audit/profiles/summary.json` (quiet machine, all shapes measured, `measure_mode: measured` at 118x1024). Percentages are shares of the profiled `compute_spi` call. Nothing in pyspi was changed.

| unit (fresh) | 8x1024 | 32x1024 | 118x1024 (measured) |
|---|---|---|---|
| basic.cov.GraphicalLasso | 1.8 ms | 4.6 ms | 0.058 s |
| basic.prec.GraphicalLasso | 1.8 ms | 4.4 ms | 0.058 s |
| basic.cov.GraphicalLassoCV | 0.091 s | 0.493 s | 5.20 s |
| basic.prec.GraphicalLassoCV | 0.091 s | 0.497 s | 5.23 s |

**Complexity.** sklearn `_graph_lasso.py:_graphical_lasso` sweeps over the P columns. For each column it copies the (P-1)x(P-1) sub-covariance and solves a lasso on that Gram matrix with the Cython `cd_fast.enet_coordinate_descent_gram`, so a sweep costs about P^3. T enters only through the empirical covariance, which is why the measured T_exponent is about 0.01. GraphicalLassoCV adds 5 folds x 4 refinements x 4 alphas through `graphical_lasso_path`, plus a final refit. Its measured P_exponent is 1.22 from 8->32 channels; 32->118 implies about 1.8.

**Hotspots.** At 8x1024, GraphicalLasso spends 52.2% of its call in `_graphical_lasso` (34.8% self) and 11.9% in check_array. GraphicalLassoCV has `graphical_lasso_path` at 87.7% and `_graphical_lasso` at 81.3% (61.9% self). This stage also wrote profiles at 118x1024 (`audit/profiles/micro/prof_basic.cov.GraphicalLasso__118x1024.pstats` and `prof_basic.cov.GraphicalLassoCV__118x1024.pstats`). In them `_graphical_lasso` is 88.6% self time for GraphicalLasso (709 column subproblems). For GraphicalLassoCV it is 89.1% self time (98.8% cumulative, 81 calls, 81066 column subproblems at about 62 us each), and `graphical_lasso_path` under joblib's sequential loop is 86.4%. The self time hides the Cython solver, so `opaque_share` (0.10-0.15) is a large underestimate here. The visible per-column helpers (np.errstate context manager, check_random_state, builtins.max, np.dot) add up to only about 4%. At 118x1024 the final GraphicalLassoCV refit hit max_iter=100 without converging (ConvergenceWarning 'did not converge after 100 iteration: dual gap: -5.065e-04', alpha_ 0.0724, n_iter_ 100; a sklearn refit on the same window with warnings recorded, saved as `graphical_lasso_cv_refit` in `audit/profiles/micro/prof_basic__index.json`). The four 118x1024 profiles of this group (EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet) are regenerated by `audit/profiles/micro/prof_basic.py run <dir>`, and `prof_basic.py index` writes `audit/profiles/micro/prof_basic__index.json` (frame shares, opaque_share by the profile_units definition, and the shares of a reproduction run into scratch, which agree within a few points: e.g. EmpiricalCovariance eigh 58.7% kept vs 62.7% reproduced, MinCovDet eigh 57.4% vs 60.2%, GraphicalLassoCV graphical_lasso_path 86.4% vs 86.6%). This matches the silent non-convergence described above.

**Avenues.**
- `cov_prec_glasso-1` (GraphicalLassoCV units, parallel, ~3x, effort S): pass `n_jobs` to GraphicalLassoCV so the 5 folds of each refinement run in parallel. pyspi forwards no estimator arguments today. The Amdahl bound with 5 workers on the 86.4% fold share is 1/(0.136 + 0.864/5) = 3.2x; refinements and the final refit stay serial. Expected unverified-bit-identical: KFold without shuffling, no RNG in the path, and joblib returns results in submission order. That holds only with worker BLAS pinned to 1 thread: joblib `_parallel_backends.py:_prepare_worker_env` sets each unset `*_NUM_THREADS` variable to cpu_count()//n_jobs, so with only OMP_NUM_THREADS=1 exported the loky workers would run OpenBLAS with 64//5 = 12 threads, and threaded BLAS calls may round differently (within-tolerance). The avenue therefore depends on `xc-blas-threads`. It cuts per-window wall time, not core-hours, and competes for cores with `xc-window-parallel`. The record lists only `xc-blas-threads` in depends_on: `xc-window-parallel` is an alternative use of the same cores, not a prerequisite, and inside one of the ~8 memory-bounded window workers (about 4 physical cores) the Amdahl bound is 1/(0.136 + 0.864/4) = 2.8x. Not measured.
- `cov_prec_glasso-2` (all four units, rewrite, 1-3x, effort L): one compiled glasso sweep (numba/C), or a compiled solver, in place of the per-column Python loop. The speedup is uncertain because cProfile cannot split `_graphical_lasso` self time between the Python per-column work and the hidden solver. Expected unverified-within-tolerance; fits that end at max_iter can move by more than rounding. Not measured.

**Seed check.** GraphicalLassoCV `n_jobs` over folds is confirmed as `cov_prec_glasso-1` (fold share 86.4% at 118x1024). GraphicalLasso (fixed alpha) costs 0.058 s per window, so even a perfect avenue saves under 35 s per 600-window patient.

**Shared versus incremental cost and EEG scaling.** One fit per estimator name serves cov, cov-sq, prec and prec-sq. In the family-warm run at 118x1024, GraphicalLasso costs 0.057 s and GraphicalLassoCV 5.28 s; the six other identifiers are about 0.1 ms each, for a family total of 5.34 s. The EEG selection keeps cov-sq_GraphicalLasso, prec_GraphicalLasso and prec-sq_GraphicalLasso, and all four GraphicalLassoCV identifiers. That is two fits per window, about 5.34 s at 118 channels or about 3.2e3 s per 600-window patient. There are no band variants.
