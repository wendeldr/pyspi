# phi family: define, trace and check

The `phi` family has four units. They come from one class, `IntegratedInformation`
(`pyspi/statistics/infotheory.py:537-573`), crossed over `phitype` (`star` or `Geo`) and
`normalization` (0 or 1), with `delay = 1`:

| unit_id | identifier |
|---|---|
| `infotheory.phi.star_norm-0` | `phi_star_t-1_norm-0` |
| `infotheory.phi.star_norm-1` | `phi_star_t-1_norm-1` |
| `infotheory.phi.Geo_norm-0` | `phi_Geo_t-1_norm-0` |
| `infotheory.phi.Geo_norm-1` | `phi_Geo_t-1_norm-1` |

The records are in `audit/records/infotheory.phi.*.yaml`. The probe scripts are read-only and live in
the session scratchpad under `phi/`:

- `probe1.py` to `probe6.py`;
- `gen_records.py`, which writes the records;
- `mtb/`, which holds pyspi's vendored copy of the PhiToolbox `.m` files, extracted with
  `git show 76ab2bb^:...` (see "Vendored copy versus upstream" below);
- `run.m`, which runs them in Octave 8.4.

The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records.
Anything marked *candidate* or *hypothesis* in the trace sections is resolved in the records'
`candidate_resolutions` and in `## Checks`. The
regression goldens are snapshots of current output. They are not used as a reference anywhere here.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared pipeline (all four units)

1. **Preprocessing.** The Calculator wraps the data in `Data(normalise=True, detrend=False)` by default
   (`calculator.py:277`; defaults at `calculator.py:48-50`). It z-scores each channel with `ddof=1` (`data.py:193-195`).
2. **Matrix assembly.** `Undirected.multivariate` (`base.py:132-138`) calls `bivariate(i, j)` for every
   ordered pair. It then overwrites the lower triangle with the upper one, so `A[i,j] = A[j,i]`. The
   value comes from rows `[min, max]`, and the lower-triangle calls are computed and then thrown away.
3. **`bivariate`** (`infotheory.py:549-573`) builds `X = data[[i, j]]` (2×T) and the partition
   `P = [1, 2]`, which puts each channel in its own part. That is the only bipartition, so no
   minimum-information-partition search takes place. It then calls `pyspi.lib.phi_native.phi_comp`. Any
   exception is logged and returns NaN.
4. **Covariances.** `cov_comp` (`phi_native.py:46-77`) takes the past window `x[:, :T-1]` and the present
   window `x[:, 1:]`, centres each by its own mean, and divides by T−2. This gives `Cov_X`, `Cov_Y` and
   `Cov_XY = E[X(t−1) X(t)']`. It is a line-for-line port of `Cov_comp.m`.
5. **`logdet`** (`phi_native.py:21-43`) uses a Cholesky factorisation. If Cholesky fails, it falls back
   to `eigvals`, keeps only the eigenvalues above 1e-12, and returns their log-sum as a complex-dtype
   number. The MATLAB `logdet.m` instead used `log(det(X/c)) + n log c`.

The native module (`phi_native.py`, by L. Lin, commit eb9d471) replaced the Octave call
`octave.phi_comp(Z, P, params, options)` that earlier pyspi versions used. That older wrapper
hard-coded `tau = 1`; the native one passes `delay`, which is 1 in every config. Octave 8.4 is
installed on this machine. The vendored toolbox copy was run on three probe inputs and agrees with the
native port to 1e-10 or better, except for the Φ* optimiser stop discussed below (difference 9e-11).

**Vendored copy versus upstream.** The `.m` files pyspi shipped (added in ef478a8, removed in 76ab2bb)
are not the published toolbox in every file. I diffed them (`diff -w`) against upstream
`oizumi-lab/PhiToolbox` master, fetched 2026-10-14:

- `Phi/phi_G_Gauss.m`, `Phi/phi_G_Gauss_AL.m` and `utility/Gauss/logdet.m` are identical apart from
  whitespace.
- `Phi/phi_star_Gauss.m` differs. Upstream calls `minFunc(@I_s_I_s_d, beta, Options)` with a nested
  `I_s_I_s_d` that returns the analytic gradient. The vendored copy calls `fminunc(fh, beta, Options)`
  (line 108) on `utility/I_s_I_s_d.m`. It keeps the minFunc header (lines 29-31) and the comment
  `% set options of minFunc` (line 99).

So "Octave" below always means Octave running pyspi's vendored copy, not the published toolbox.

## infotheory.phi.star_norm-0 (Φ*)

**Documented.** Supplement §1.4.10, p.12: "Φ∗ is a proxy of integrated information [53]. It is an
undirected measure that uses the concept of mismatched decoding in information theory [54] and can be
considered as the amount of loss of information due to the disconnection between two variables." The
listed keywords are "undirected, nonlinear, unsigned, bivariate, time-dependent". The GitBook page
`information-theory.md` (lines 111-133) has the same text.

**Canonical.** Oizumi, Amari, Yanagawa, Fujii & Tsuchiya (2016), *PLoS Comput Biol* 12:e1004654:

- Φ* = I(X_{t−τ}; X_t) − I*.
- I* = max_β Ĩ(β).
- Ĩ(β) = −Σ_y p(y) log Σ_x p(x) q(y|x)^β + Σ p(x,y) log q(y|x)^β.
- The mismatched decoder is q(y|x) = Π_k p(y_k | x_k).

The equation numbers I cite are Eq. 8 (decoder), Eq. 9 (Φ*) and Eq. 20 (Ĩ(β), with I* at its maximum).
In the Methods section "Analytical computation of Φ* under the Gaussian assumption" the numbers are:

- Eq. 23: Gaussian I;
- Eq. 24: the conditional covariance;
- Eq. 25: the closed form of Ĩ(β);
- Eqs. 26-27: Q and R;
- Eq. 28: the block-diagonal covariances;
- Eqs. 29-33: dĨ/dβ.

These come from a fetch of the article page, where the equations are images. They agree with the
vendored code's comment "2nd term of eq. (26)" (`phi_star_Gauss.m:87`, `I_s_I_s_d.m:2`). The same
Methods text says that I* is "a convex function with respect to β", that dĨ/dβ = 0 "is a quadratic
equation with respect to β", and that "β can be analytically computed without resorting to numerical
optimization". I quote these statements but have not re-derived them. "Convex" sits oddly with a
maximisation. The paper states 0 ≤ Φ* ≤ I.

**Implemented** (`phi_native.py:148-246`):

- I is computed as H(Y) − H(Y|X) from Gaussian log-determinants.
- The blocks of the per-channel decoder are X_D, YX_D and C_D_cond.
- `i_s_objective` evaluates the Gaussian closed form:
  I_s(β) = ½ log|Σ_X⁻¹ + β A_Dᵀ D⁻¹ A_D| + ½ log|Σ_X| + ½ tr(Σ_Y R) − βN/2.
- β is maximised by `scipy.optimize.minimize(method='L-BFGS-B')`. The search starts at β = 1, has no
  bounds and no analytic gradient (so it uses 2-point finite differences), and keeps scipy's default
  `gtol = 1e-5` (`_lbfgsb_py.py:294-296`).
- The result is Φ* = I − I_s.

I re-derived Ĩ(β) for the Gaussian case from scratch: compute E_x[q^β] as a Gaussian integral, then take
the expectation over y. The result matches `i_s_objective` term by term. The constant βN/2 comes from
tr(D⁻¹ E[(y − A_D x)(y − A_D x)ᵀ]) = N.

On the population covariances of `gauss_ar1_pair`, pyspi gives Φ* = 0.025288134449 with β = 0.8644. A
from-scratch bounded maximiser gives the same value. I = 0.446290796.

**Deviations.**

- **Documented vs code.** The supplement keyword "nonlinear" (also in `config.yaml:616-622`) conflicts
  with the purely second-order Gaussian estimator. The class label correctly says "linear". The config
  labels are never applied at runtime: `calculator.py:253-259` instantiates the class, and the instance
  keeps the class labels.
- **Optimiser stop (candidate).** When the finite-difference gradient at β = 1 is below 1e-5, L-BFGS-B
  stops at β = 1 without taking a step. One probe recorded `nit 0, nfev 2, jac 9.7e-6`. I_s is then
  under-maximised, so Φ* comes out too high. I compared 136 pairs (8 datasets × T ∈ {64, 256, 1024,
  4096}). Every pyspi − reference difference was ≥ −9e-16, and every large gap had `nit 0`:
  - the largest absolute error was 4.65e-8, on `narrowband_f0-0.25` T256, pair (1,2). There Φ* = 3.891
    and β stayed at 1.0 where the reference optimum is 0.922;
  - the next largest were 3.3e-8 and 2.8e-8, both on narrowband T1024;
  - the largest relative error was 6.4e-4, on `iid_gaussian` T1024, pair (2,3): 1.50073e-05 against
    1.49977e-05. Its absolute gap, 9.55e-9, is the figure an earlier probe reported as its maximum. That
    was a misreading: the same probe's subset also held narrowband T1024 pair (0,1) at 3.35e-8.

  Octave running the vendored copy stops the same way (1.50072e-05), so the behaviour is inherited from
  that copy. I have not checked whether upstream minFunc with the analytic gradient would also stop at
  β = 1.
- **Hypothesis check: analytic gradient (confirmed for the published toolbox).** The briefing says the
  MATLAB original had an analytic gradient. That is true of upstream PhiToolbox:
  `minFunc(@I_s_I_s_d, beta, Options)` uses the gradient that `I_s_I_s_d` returns as its second output.
  pyspi's vendored copy swapped in `fminunc(fh, beta, Options)` (`phi_star_Gauss.m:108`) with no
  `GradObj`. Octave's `fminunc.m:130,147` defaults `GradObj` to `"off"`, so the vendored path used finite
  differences. The native finite-difference L-BFGS-B therefore reproduces the vendored copy, but it is a
  behavioural change relative to the published toolbox. The size of that change has not been measured.
  The other part of the hypothesis, that `inv(C_D_cond)` is recomputed on every evaluation, is confirmed
  in both versions (`phi_native.py:136`, `I_s_I_s_d.m:6`). That affects speed only.
- **β domain.** β is unconstrained. The paper does not state its domain, but says dĨ/dβ = 0 is
  quadratic in β, so β can be computed analytically. pyspi, like PhiToolbox, optimises numerically
  instead. All probe optima fell between 0.64 and 1.33.
- **Degenerate inputs (candidate).** The `logdet` fallback silently drops eigenvalues ≤ 1e-12, including
  negative ones. On an exact one-step-delay pair:
  - after z-scoring, eig(Cov_Y|X) = [−2.2e-16, 1.0015]. Cholesky fails and Φ* = 0.000636+0j;
  - on the raw series the small eigenvalue is +2.2e-16. Cholesky succeeds and Φ* = 18.03.

  The guards `np.isinf(H_cond)` and `np.iscomplex(H_cond)` can never fire on this path.

  A bit-identical copy (`shared_envelope` pair (0,2), still bit-identical after the Data z-score)
  behaves differently on the Calculator/Data path, and the outcome depends on T:
  - **T = 64, 256, 1024:** `scipy.linalg.inv(Cov_X)` (`phi_native.py:203`) raises LinAlgError, and the
    result is NaN with UserWarning "Matrix inversion failed in phi_star computation".
  - **T = 4096:** Cov_X is only ill-conditioned (LinAlgWarning). A complex logdet reaches L-BFGS-B, which
    raises `TypeError: '<' not supported between instances of 'complex' and 'float'`. The class's
    `except` (`infotheory.py:571-573`) logs it and returns NaN.

  The value 0.0657 I quoted earlier comes only from the raw-array call `spi.bivariate(x[0], x[2])` at
  T1024. That call leaves the second array un-z-scored, so the two channels are affinely related rather
  than identical. The canonical value is +∞ (deterministic prediction).

**Units.** The value is in nats and symmetric. It is invariant to per-channel affine maps: on
`scaled_offset` with `normalise=False` it equals the `bivariate_gaussian` value to 1e-12. On independent
data it has a positive finite-sample bias.

**Proposed checks** (details in the record):

1. A kernel check on the population covariances against the from-scratch Ĩ(β) maximiser (tolerance
   1e-8).
2. A kernel check on sample covariances for 8 datasets, reporting the maximum relative error and
   whether it is one-sided.
3. A Monte-Carlo check of Ĩ(β) at fixed β values. This tests the closed form itself.
4. Statistical convergence of the pipeline on `gauss_ar1_pair` towards 0.0252881.
5. The bounds 0 ≤ Φ* ≤ I on every dataset.
6. A null check: `iid_gaussian` values should shrink like 1/T.
7. Direction and symmetry on `var2_bivariate` and `var2_chain`: A == Aᵀ bitwise, and the X→Y coupling
   produces no asymmetry.
8. A same-lineage comparison against Octave running pyspi's vendored `.m` files (fminunc). This is not
   the published minFunc path.
9. Qualitative documentation of the degenerate cases.
10. The scale-invariance check on `scaled_offset`.
11. A published-algorithm reference: maximise with the analytic derivative (Eqs. 29-33, the same algebra
   as `I_s_I_s_d.m:12-16`), and separately solve dĨ/dβ = 0 directly. Expect the pyspi gap to be ≤ ~5e-8
   and one-sided. Also report whether an explicit-gradient L-BFGS would stop at β = 1.

## infotheory.phi.star_norm-1

This unit is Φ* as above, divided by `(N_c−1)·min_k h_gauss(Cov_X[k,k])` (`phi_native.py:235-244`, a
port of `phi_star_Gauss.m:114-126`). With two parts, the divisor is the smaller of the two past-window
Gaussian differential entropies, ½ log(2πe σ²).

**Documented.** "Both measures are optionally normalized (divided) by entropy" (supplement §1.4.10,
p.12, line 564). The supplement does not say which entropy.

**Canonical.** Balduzzi & Tononi (2008) normalise by (m−1)·min_k H^max(M_k) in order to find the minimum
information partition. Oizumi et al. (2016, Discussion) warn that for continuous variables this factor
"can be negative because entropy can be negative for continuous variables" and that "we need a
different normalization factor when we deal with continuous variables".

**Candidates.**

- **Negative values.** The divisor is negative whenever a past variance is below 1/(2πe) ≈ 0.0585. With
  `normalise=False`, the `scaled_offset` pair (0,1) (scale 0.2) gives −0.000834. That is a negative value
  for an `Unsigned` SPI. The same pair gives +0.000115 after z-scoring.
- **Redundancy with norm-0.** With the default z-scoring, the divisor is about 1.4185 for every pair: the
  `var2_chain` T1024 ratio was 1.4185084 for both pairs. So norm-1 is essentially a constant rescaling of
  norm-0 and ranks pairs identically.
- **No search.** Because pyspi only evaluates one bipartition, this normalisation never serves the
  purpose it was designed for, which is choosing among partitions.

**Extra checks.**

- An exact check that norm-1 equals norm-0 divided by the from-scratch minimum past entropy (relative
  error ≤ 1e-12).
- A qualitative check of the sign flip on unnormalised `scaled_offset`.

**Already documented (partial).** REVIEW M8 (`docs/storage_comparison/REVIEW.md:198`) documents that
Gaussian differential entropy can be negative, for `je`/`ce`. The negative-divisor candidate is the
same `h_gauss` mechanism, applied to the past marginals. The records set `already_documented` to this
cross-reference. What M8 does not cover is the phi-specific consequence: negative values from an
`Unsigned` SPI.

## infotheory.phi.Geo_norm-0 (Φ_G)

**Documented.** Supplement §1.4.10, p.12: "ΦG is a measure of integrated information derived from
information geometry [55]. It is an undirected measure that quantifies the divergence between the actual
probability distribution of a system and an approximated probability distribution where influences
among elements are statistically disconnected. Here we implement the Gaussian-distribution model [53]."

**Canonical.** Oizumi, Tsuchiya & Amari (2016), *PNAS* 113:14817. I read the arXiv:1510.04455 v1 text:

- Eq. 8: q(Y[P_i]|X) = q(Y[P_i]|X[P_i]).
- Eq. 9: Φ_G = min_q D_KL(p‖q).
- In the Gaussian AR case, Y = AX + E and the disconnected model has A′_ij = 0 for i ≠ j (Eq. 17).
- Eqs. 18-20: Σ(X)′ = Σ(X); (Σ(X)(A−A′)ᵀ Σ(E)′⁻¹)_ii = 0; Σ(E)′ = Σ(E) + (A−A′) Σ(X) (A−A′)ᵀ.
- Eq. 21: Φ_G = ½ log |Σ(E)′| / |Σ(E)|.
- The paper states 0 ≤ Φ_G ≤ I(X;Y) and Φ_G ≤ SI.

**Implemented** (`phi_native.py:249-391`, a port of `phi_G_Gauss.m` and `phi_G_Gauss_AL.m`):

- A = Cov_YX Cov_X⁻¹ and Σ_E = Cov_Y − Cov_YX Cov_X⁻¹ Cov_XY.
- With K = 2 parts, the `'AL'` method is selected.
- The augmented Lagrangian runs with μ = 2, α = 0.9, γ = 1.01 and maxiter 1e5. It uses a closed-form A_p
  update in the eigenbasis.
- An eigenvalue clamp at 1e-12 was added on line 303. It is not in the MATLAB version.
- The loop stops when the change in logdet(Σ_E′) between iterations is below 1e-10.
- The result is Φ_G = ½(logdet Σ_E′ − logdet Σ_E).

On the population covariances of `gauss_ar1_pair`, pyspi gives 0.0874723233. Direct minimisation over
diagonal A′ gives the same.

**Deviations.**

- **Non-optimal stop (candidate).** The stopping rule tests only the change in the objective. The
  multiplier μ grows geometrically, so in a flat valley the iterate can stall at a feasible point that is
  not stationary. On `delayed_noise` T64, pair (0,1):
  - the AL returns 2.3439385 at a = [−2.618, 0.029];
  - Nelder-Mead and BFGS both reach 2.3406849 at a = [−3.413, 0.022];
  - the gradient at the AL point is [8.2e-3, 8.3e-5] and the Hessian eigenvalues are [0.010, 9.0];
  - the AL ran 2681 iterations, ending with μ = 1.35e6 and constraint violation 5.6e-33.

  That is a 0.14 % overestimate. Octave running the vendored copy (phi_G files identical to upstream) returns 2.34393854875, so this is inherited from
  PhiToolbox. Two other probe cases were off by 2e-5 and 4e-6. Every other probe case agreed with the
  global minimum to about 1e-8.
- **Hypothesis check: eigenvalue clamp.** The clamp is confirmed absent in MATLAB, in both the vendored
  and upstream `phi_G_Gauss_AL.m`. Because Σ_E′ ⪰ Σ_E, it can only act when Σ_E is numerically singular.
  On `delayed_noise` T64, pair (0,1):
  - the smallest eigenvalue of Σ_E is 0.00832;
  - the smallest eigenvalue of any iterate Σ_E′, over all 2681 AL iterations, is 0.00936 (this is the
    0.0094 quoted earlier, from probe3's running minimum);
  - the smallest eigenvalue of the final Σ_E′ is 0.1009.

  So the clamp never acted there. In the three cases compared, pyspi and Octave agree to better than
  1e-10.
- **Degenerate inputs.** These share the `logdet` problem with Φ*. On the Calculator/Data path a
  bit-identical copy gives:
  - **T ≤ 1024:** NaN with UserWarning "Singular covariance matrix in phi_G computation", because
    `solve(Cov_X, Cov_XY)` raises (`phi_native.py:374`);
  - **T = 4096:** a finite negative −0.3466, numerically −ln(2)/2, with LinAlgWarning. That breaks
    Φ_G ≥ 0.

  The 0.2350 quoted earlier is the raw-array-path value, `spi.bivariate(x[0], x[2])` at T1024.
- **Labels.** The same "nonlinear" versus "linear" conflict as for Φ*.

**Observation, not a defect.** On several datasets Φ_G exceeds Φ*. For example, on `gauss_ar1_pair` the
population values are Φ_G = 0.0875 and Φ* = 0.0253. The papers I read give no ordering between the two
measures (open question).

**Proposed checks.**

1. A kernel tolerance check against the population global minimum.
2. A kernel optimality check on sample covariances for 8 datasets, using a multi-start direct minimiser
   and the Eq. 19 stationarity residual. Flag any case where pyspi − reference > 1e-6.
3. The bounds Φ_G ≤ min(I, SI).
4. Statistical convergence on `gauss_ar1_pair`.
5. The null check on `iid_gaussian`.
6. Direction and symmetry.
7. A same-lineage comparison against Octave.
8. The degenerate cases.
9. Scale invariance (agreement to 1e-8, the AL tolerance).

## infotheory.phi.Geo_norm-1

This unit is Φ_G divided by the minimum past-window Gaussian entropy (`phi_native.py:343-352`, a port of
`phi_G_Gauss_AL.m:107-119`). It has the same candidates as `star_norm-1`: with `normalise=False` the
`scaled_offset` pair (0,1) gives −0.00106, and with z-scoring the divisor is about 1.4185, so the unit
is a near-constant rescaling of `Geo_norm-0`. The extra checks are the same as for `star_norm-1`.

## Direction convention (all units)

All four units are undirected. The off-diagonal entries satisfy A[i,j] = A[j,i] bitwise, because of
the mirror at `pyspi/base.py:136-137`. The diagonal is NaN, so `np.array_equal(A, A.T)` is False; the
comparison to use is `np.array_equal(A, A.T, equal_nan=True)`, which is True (var2_chain T1024).

Before mirroring, the pair computed in the other row order differs only by optimiser tolerance. On the
Calculator/Data path (both rows z-scored, `phi_comp` on rows [0,1] versus [1,0], `gauss_ar1_pair`
T4096):

- the Φ* gap is −5.0e-13;
- Φ_G is bit-identical (0.08925335492965669).

The gaps I quoted earlier (7e-13 and 1.3e-9) came from the raw-array calls `spi.bivariate(x0, x1)` and
`(x1, x0)`. Those calls z-score only the first array. The proposed
`direction` checks on `var2_bivariate` and `var2_chain` confirm empirically that unidirectional coupling
leaves no asymmetry.

## Already documented

None of the findings above appears in `KNOWN_ISSUES.md`. Two entries come close but differ:

- REVIEW M8 mentions negative Gaussian entropies for `je`/`ce`. It covers the mechanism behind the
  norm-1 candidate but not the phi normalisation itself. The two norm-1 records cross-reference it in
  `already_documented`.
- REVIEW M1 (tests/README.md bug 5 before b2f6d7e) covers "undirected" SPIs that bypass the mirror. `IntegratedInformation` does not bypass
  it.

## Open questions

1. **Preprocessing on the array path (outside this family, hypothesis).** `parse_bivariate` with two raw
   arrays builds `Data()` through `add_process`. The first process is z-scored by `set_data`, but the
   second is appended raw. A probe gave means [0, −2.516] and SDs [1, 6.98]. This affects any norm-1 or
   other scale-dependent SPI called as `spi.bivariate(x, y)`, but not the Calculator path. It belongs to
   the base/data audit.
2. **Should negative norm-1 values be NaN-ed or documented?** The PLoS CB authors call for a different
   continuous-variable normalisation. Since pyspi evaluates only the atomic bipartition, the
   normalisation never serves its original purpose of selecting a partition.
3. **Domain of β (partly answered).** The paper does not state a domain. Its Methods text does say
   that dĨ/dβ = 0 "is a quadratic equation with respect to β" and that β "can be analytically computed
   without resorting to numerical optimization". Check 11 of `star_norm-0` uses this as an exact β
   reference. Still open: the paper calls I* "convex" in β although it is maximised.
4. **Ordering of Φ_G and Φ*.** Is there a theoretical ordering between them? The probes show
   Φ_G > Φ* on several datasets.
5. **How often the AL stalls.** How common is the non-optimal AL stop on realistic data, especially for
   strongly predictable pairs where the valley is flat? The check stage should report the distribution of
   gaps over all datasets and all four T values.
6. **Output for deterministic pairs.** Should exactly deterministic or duplicated channels
   return NaN, +inf, or a capped value? Today the output depends on the call path and on T:
   - exact-delay pairs depend on the sign of a roundoff eigenvalue;
   - bit-identical copies give NaN at T ≤ 1024;
   - at T = 4096, Φ* gives NaN (TypeError) and Φ_G gives −0.3466.

## Checks

All numbers below come from `audit/checks/out/results/phi.json` (355 runs, family `phi`), produced by
`audit/checks/run_checks.py --family phi`: 326 pass, 29 fail, 0 without a reference. All 29 failures are
Phi_G runs and are genuine mismatches against the independent optimum, not script errors. Files:

- `audit/checks/manifest/phi.json`;
- `audit/checks/refs/phi.py`: comparison venv, numpy/scipy only, no pyspi import;
- `audit/checks/probes/phi_probes.py`: project venv;
- `audit/checks/octave/phi.m`: Octave 8.4.0 running pyspi's vendored PhiToolbox copy from `git 76ab2bb^`,
  with `addpath` on `Phi/`, `utility/` and `utility/Gauss/`.

Run ids have the form `<unit>|<dataset>|<T>|<path>|<tag>`. Every run is cheap: the slowest pyspi run took
0.9 s, and the Monte Carlo reference (`mc`, about 8 min) dominates the reference step.

**Reference labels.**

- **Octave is labelled `same-lineage`.** The task text called it the "external-pkg" reference, but
  `phi_native.py` is a line-by-line port of these `.m` files. The independence rule therefore makes Octave
  same-lineage, as gwtau already does for its authors' MATLAB loop. Octave serves only to show which
  behaviours are inherited from PhiToolbox.
- **The independent references are written from scratch:**
  - `StarRef`: the Gaussian tilde-I(beta) and its analytic derivative, with the root found by
    `scipy.optimize.brentq`.
  - `matlab_IsId`: a transcription of `utility/I_s_I_s_d.m`, also solved with brentq. It agrees with
    `StarRef` to 4.4e-16, and its root agrees with the `StarRef` optimum to 2.9e-13 or better.
  - `geo_ref`: the global minimum of the PNAS Eq. 21 objective over diagonal A'. It uses a grid, then
    BFGS with the Eq. 19 gradient, then a Nelder-Mead polish.

**Inputs passed exactly as pyspi passes them.** In every Octave job and every reference the inputs are
partition Z = [1 2], `params.tau = 1` (the `delay` default; the pre-eb9d471 Octave wrapper hard-coded
tau = 1), `options.type_of_dist = 'Gauss'` and `options.type_of_phi = 'star'|'Geo'`.

**How norm 0 and norm 1 differ.** Only `options.normalization` changes. For norm-1, phi is divided by
(N_c − 1)·min_k H_gauss(Cov_X[k,k]), with N_c = 2. This is the same AL or L-BFGS run as norm-0 followed by
one division. The `ratio` runs show it is an exact identity (max_abs 6.9e-18).

### infotheory.phi.star_norm-0: matches, score 9 (optimiser unit within optimiser tolerance)

**Score rationale.** Score 9 rests on the rubric clause "within statistical tolerance consistent with
estimator variance", and only on independent references: the brentq optimum of the from-scratch Ĩ(β), the
population covariances and the Monte Carlo of Eq. 20.
- The largest gap, 4.65e-8 (`narrowband_f0-0.25|256|pipeline|indep`), is 5.6e-4 of the smallest Monte
  Carlo SD of the Φ* estimator in this family (8.3e-5, `iid_gaussian|4096|pipeline|null`).
- The run with the largest relative gap (6.4e-4) is `iid_gaussian|1024|pipeline|indep`. Its absolute gap,
  9.55e-9, is 3.2e-5 of the null SD at that T (2.95e-4).
- The clause is applied by analogy. The gap is a one-sided optimiser bias (L-BFGS-B leaves β at 1 when the
  gradient there is below its tolerance; pyspi − optimum lies in [−3.6e-14, 4.65e-8]), not estimator noise. It is
  admitted because it is at most 5.6e-4 of the smallest sampling SD of the estimator.
- It is not 10, because the agreement is not fp-exact (relative error up to 6.4e-4).
- It is not 7 (the optimality-only anchor), because the canonical optimum is computed independently.
- The Octave run of pyspi's vendored PhiToolbox is same-lineage. That cap is 6, so the Octave match does not
  contribute to the score; the same reasoning applies to Geo_norm-0.

- **Population covariances.**
  - D15 `gauss_ar1_pair` (`|pop`): pyspi 0.025288134449. The gap to the brentq optimum is 4.3e-13, and
    beta is 0.86440 (pyspi − ref 1.1e-6).
  - I = 0.44629079610, which equals ½ log|Σ_X|/|Σ_E| to 2e-16.
  - D5 `var2_bivariate`, lag-1 model from the companion Lyapunov solution (`ref_Gamma0_vs_truth_maxabs`
    5.6e-17): 0.127276, gap 2.2e-11.
  - Octave on the same covariances is within 5.4e-14.
- **The closed form against the Eq. 20 definition** (`|gauss_ar1_pair|1024|kernel|mc`): a Monte Carlo
  outer expectation (n = 2e5) with a 60×60 Gauss-Hermite inner integral (stable against 40×40 to 8.8e-13).
  At beta = 0.5, 1 and 2 the result agrees with pyspi's own objective within 0.52 Monte Carlo SE. The
  closed form equals pyspi's objective to 8.9e-16.
- **Sample covariances.** All 37 `indep`/`detrend` runs pass (8 datasets × 4 T on the pipeline path,
  4 datasets on the kernel path, and `trending` on pipeline-detrend).
  - pyspi − ref lies in [−3.6e-14, 4.65e-8].
  - The largest absolute gap is 4.65e-8 (`narrowband_f0-0.25|256`).
  - The largest relative gap is 6.4e-4 (`iid_gaussian|1024`, absolute 9.55e-9).
  - All 37 `details` runs pass the optimality criterion; their gaps lie in [−4.9e-13, 4.65e-8].
  - On the 8-dataset set (136 pairs), 67 pairs ended with nit = 0, meaning beta stayed at 1.0. Every gap
    above 5.9e-10 is one of these nit = 0 pairs, and at beta = 1 their |dĨ/dβ| was at most 9.7e-6.
  - **The optimiser candidate is confirmed** and reproduces the trace figures exactly.
- **Published gradient path.** Octave's `fminunc` with `GradObj` on (consuming the analytic gradient of
  `I_s_I_s_d.m`) is at most 4.5e-8 from the optimum, the same as the finite-difference path. scipy
  L-BFGS-B with the analytic `jac` stops at nit = 0 on 69 of 136 pairs. The early stop is therefore a
  first-order-tolerance effect. The finite-difference substitution is real, but its numerical effect is
  nil.
- **Same-lineage comparison.** All 16 Octave runs pass, with max_abs 4.5e-8.
- **Convergence to the population value.** `gauss_ar1_pair` pyspi − population is 0.0597, 0.0073, 0.0074
  and 0.0057 at T = 64, 256, 1024 and 4096 (max_z 1.61 against R = 300 fresh draws). `var2_bivariate`
  gives 0.0339, 0.0197 and −0.0078 (T = 256, 1024, 4096).
- **Null.** On `iid_gaussian` the mean over pairs is 0.0209, 0.00169, 0.000706 and 0.000221 for
  T = 64…4096. T × mean is 1.34, 0.43, 0.72 and 0.90, so the bias shrinks roughly as 1/T. Every run passes
  the Monte Carlo test, and the minimum value is 5.7e-6 (never negative).
- **Bounds.** All 16 pass 0 ≤ Φ* ≤ I. The smallest margin is I − Φ* = 6.8e-8 (narrowband T1024).
- **`nullpop`.**
  - A = diag(0.5, 0.3) with Σ_E = I gives 0 (6.4e-13).
  - The same A with an instantaneous noise correlation of 0.5 gives Φ* = 0.0382 in both pyspi and the
    reference. This is a property of Φ*: the mismatched decoder ignores noise correlation.
- **Invariance.** `scaled_offset` on the kernel path equals the `bivariate_gaussian` optimum to 1.1e-10.
- **Degenerate inputs (confirmed, with a correction).** On `shared_envelope` (0,2) the result is NaN at
  every T. The exact-delay construction gives −0.00110 (z-scored) and −0.0215 (raw) at `delayed_noise`
  T1024. Both values are negative, while the canonical value is +∞. The trace probe used a different series
  and got +0.000636 and 18.03. The output therefore depends on the sign of a roundoff eigenvalue and can
  break Φ* ≥ 0. I corrected the record's `units_and_range` and `deviation_from_canonical` to say so.

### infotheory.phi.star_norm-1: variant, score 7

- 21/21 `indep` runs pass (gap ≤ 2.4e-8).
- The ratio identity holds exactly. Under z-scoring the divisor is 1.3701–1.4251 at T64 and 1.4168–1.4194
  at T1024 (unit variance gives 1.41894).
- **Sign flip.** With `normalise=False`, `scaled_offset` pair (0,1) gives −0.000834 and the divisor is
  −0.196; 3 of the 6 pairs are negative. pyspi agrees with the independent value to 1.6e-10. **The sign
  flip is confirmed.** The mechanism is partly documented (REVIEW M8), but the phi consequence is not in
  KNOWN_ISSUES.
- **Score rationale.** The unit matches the supplement wording but differs from canonical Φ* in a
  value-changing way. `definition_under_review` is true: Oizumi et al. 2016 themselves call for a
  different normalisation for continuous variables.

### infotheory.phi.Geo_norm-0: variant, score 6

- **Population covariances.** D15 gives 0.0874723233 (gap 3.1e-11); D5 gives 0.142013 (gap 2.5e-10).
  `nullpop` gives 0 and 0: Φ_G vanishes with noise correlation only, unlike Φ*.
- **Over-estimates: the trace candidate is confirmed and is larger than the trace found.** The AL stops at
  a feasible but non-stationary point:
  - `delayed_noise`: 3.25e-3, 7.6e-4, 2.0e-5 and 1.65e-5 at T = 64, 256, 1024 and 4096;
  - `narrowband_f0-0.25`: 4.0e-6 at T64 and 1.1e-6 at T4096;
  - `shared_envelope` (0,1), processes [0,1,3]: **1.107 nats** at T64 (pyspi 6.5319 against a global
    minimum of 5.4252, i.e. 20 %), 1.15e-3 at T256 and 2.2e-5 at T1024.
- **Under-estimates: not a trace candidate.** On weakly coupled pairs at T = 4096 the AL stops before the
  constraint is met; the largest off-diagonal |A_p| at the stop is between 6.6e-4 and 9.8e-4:
  - `cointegrated`: −3.46e-4 (relative error about 1: pyspi is about 0);
  - `bivariate_gaussian`: −4.6e-6;
  - `iid_gaussian`: −2.7e-6.

  The same mechanism makes the raw-scale output scale-dependent. On raw `scaled_offset` the gap is −2.34e-5
  (relative 0.28), while the z-scored pipeline matches to 3.4e-10. This contradicts the trace statement
  that other cases agree to about 1e-8. The record's `deviation_from_canonical` now carries a check-stage
  correction.
- **Counts.** Over the 136 pipeline pairs of the 8-dataset set, 6 pairs over-estimate by more than 1e-6
  and 3 under-estimate by more than 1e-6. Adding `shared_envelope` makes it 9 of 148 over-estimates.
- **Inherited from PhiToolbox.** Octave on the vendored `phi_G_Gauss_AL.m`, which is identical to
  upstream, reproduces every failing case to 3.4e-10 or better (16/16 runs).
- **The 1e-12 eigenvalue clamp** never changes a value compared with Octave.
- **Convergence, null and bounds all pass.**
  - Convergence: `gauss_ar1_pair` max_z 0.70; `var2_bivariate` max_z 0.97.
  - Null: T × mean is 0.45–1.38.
  - Bounds: 16/16 satisfy Φ_G ≤ min(I, SI). The smallest margin is SI − Φ_G = 3.9e-7.
- **Degenerate inputs.** NaN at T ≤ 1024 and −0.34657 at T4096 on the copy. The exact-delay pair gives
  −0.00062 (z-scored) and −0.0200 (raw).
- **Score rationale.** The output deviates from the canonical constrained minimum in a regime-dependent way,
  by up to 20 % on strongly predictable pairs and by up to 100 % relative on near-null pairs at large T.
  The authors' code gives the same numbers, but that is a same-lineage result and cannot raise the score
  above 6.

### infotheory.phi.Geo_norm-1: variant, score 6

- 17/21 `indep` runs pass. The 4 failures are the norm-0 stalls divided by about 1.42: `delayed_noise`
  T64 2.28e-3, T1024 1.4e-5 (both paths), and `narrowband` T64 2.9e-6.
- The ratio identity holds exactly.
- On raw `scaled_offset`, pair (0,1) gives −0.00106 with divisor −0.196, so the sign flip is reproduced.
  That value is still 1.19e-4 away from the independent value because of the AL scale dependence (the
  `negative` run fails on value).
- `definition_under_review` is true, as for star_norm-1.

### Symmetry and direction (all units)

- Off the diagonal, A[i,j] == A[j,i] bitwise on every run, and `np.array_equal(A, A.T, equal_nan=True)`
  holds on the mirror runs.
- The unmirrored `Directed.multivariate` probe computes both row orders. The orders differ by at most
  2.3e-13 for star and 1.6e-14 for Geo (T1024 on `var2_bivariate`, `var2_chain` and `delayed_noise`).
- The coupling directions in `var2_bivariate`, `var2_chain` and `delayed_noise` leave no trace: on
  `var2_bivariate` T1024, A01 = A10 = 0.147014 (star) and 0.160150 (Geo).

### Trace issues fixed

The three remaining trace issues listed for this family were already corrected in the records and in this
note:

1. the provenance of the 9.55e-9 figure (above, and in the star records' first candidate);
2. the numbering of the proposed-checks list (1–11, in order);
3. the "Already documented (partial)" paragraph, which now sits below the "Extra checks" list.

I re-checked all three.

## Open questions: answered

1. **Two-array preprocessing.** Answered outside this family by `preproc.two_array_api` (status checked).
   `preproc.two_array_api|scaled_offset|{64,1024}|pipeline|cap-contract` confirms that row 1 is appended
   raw. The phi checks use the Calculator/Data path throughout, so they are unaffected. The
   scale-dependence of norm-1 and of the Phi_G AL means that `spi.bivariate(x, y)` on raw arrays will not
   match the Calculator value for these four units either.
2. **Negative norm-1 values.** These are confirmed (−0.000834 for star and −0.00106 for Geo on raw
   `scaled_offset`). The checks cannot settle whether they should become NaN or be documented, because
   that is a policy choice. The evidence supports documenting that norm-1 is meaningful only after
   z-scoring, where it is a near-constant rescaling by 1.37–1.43. An alternative is to NaN the output
   when the divisor is ≤ 0.
3. **Domain of β and the "quadratic" claim.** For the atomic bipartition, dĨ/dβ·det(Q⁻¹)² is a
   **quartic** polynomial in β, and dĨ/dβ·det(Q⁻¹) is not a polynomial at all. So, in this
   parametrisation, the stationarity condition is not a quadratic in β. Evidence:
   - the `pop` runs report `ref_deriv_numerator_poly_degree` 4;
   - `scratchpad/phi_check/beta_poly.py`;
   - the quartic has exactly one positive real root, which is the optimum (0.86440 for D15, 0.96505 for
     D5);
   - Ĩ is concave in β (its second differences are below 0 on [0, 6], and `ref_deriv_decreasing_on_0_6`
     is 1). The paper's "convex" presumably refers to −Ĩ.

   The brentq root on the derivative is unique and serves as the exact β reference. What would settle
   the paper's statement is reading Eqs. 29–33 (they are images) to see whether "quadratic" refers to a
   different reduction.
4. **Ordering of Φ_G and Φ\*.** Neither ordering holds in general.
   - On the population covariances Φ_G > Φ* for D15 (0.0875 vs 0.0253) and for D5 (0.1420 vs 0.1273).
   - With noise correlation only, Φ_G = 0 < Φ* = 0.0382 (`nullpop`).
   - On sample covariances Φ_G > Φ* in 72 of 156 pair-runs (`ref_n_pairs_geo_gt_star` in the Geo `indep`
     runs).
5. **How often the AL stalls.** Over 136 pipeline pairs (8 datasets × 4 T), 6 pairs over-estimate by more
   than 1e-6 (all on `delayed_noise` or `narrowband`, i.e. strongly predictable pairs). 3 pairs
   under-estimate by more than 1e-6 (near-null pairs at T4096). 8 pairs over-estimate by more than 1e-8
   (12 of 148 once `shared_envelope` is included). Adding
   `shared_envelope` produces the largest error seen, 1.107 nats. Octave reproduces all of these.
6. **Deterministic pairs.** The checks confirm the output is inconsistent:
   - NaN for an exact copy at T ≤ 1024;
   - at T4096, NaN for star and −ln(2)/2 for Geo;
   - small negative values for an exact one-step delay in this check's series, where the trace's series
     gave positive values.

   The canonical value is +∞. The checks cannot choose the policy, but a guard that returns NaN (or +∞)
   whenever Cov(Y|X) or Cov_X is numerically singular would remove the sign dependence on roundoff.
7. **Process note (git status).** This is not a question about phi. The check stage modified no file
   outside the phi audit artefacts listed above.

## Optimisation

Workflow 3 (group infotheory) is a short, profile-based search for where the time goes and which avenues look promising; nothing in pyspi was changed. Walls below are from `audit/profiles/summary.json` (quiet machine); percentages are shares of the profiled SPI call (`tests/regression/harness.py:compute_spi`) in the cProfile run at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`). That profiling pass ran contended for the JIDT units (profiled walls 1.3-20x the quiet walls), so its walls are not used, only its shares. cProfile cannot see inside Java: JPype dispatches JIDT methods without profiler entries, so the Java time is booked as self time of the calling pyspi frame, and for the JIDT units every `opaque_share` (0.0002-0.04) is a lower bound on JIT/JNI time, not an estimate of it. The phi units run no Java: their opaque_share (0.17-0.25) is numpy/scipy C time. `expected_speedup` is the unit's per-window speedup at 118x1024 as an order of magnitude; risks carry the `unverified-` prefix unless a micro-benchmark diff file backs them. Agreement with today's output means only that a candidate reproduces it.

### Walls

| Unit | 8x1024 fresh | 32x1024 fresh | 118x1024 fresh (measure_mode) | 8x1024 family-warm | 118x1024 family-warm projected (measure_mode) |
|---|---|---|---|---|---|
| `phi.star_norm-0` | 0.0987 s | 1.67 s | 22.3 s (measured) | 0.0997 s | 22.3 s (measured) |
| `phi.star_norm-1` | 0.102 s | 1.71 s | 22.9 s (measured) | 0.102 s | 23 s (measured) |
| `phi.Geo_norm-0` | 0.289 s | 4.92 s | 64 s (measured) | 0.291 s | 62.7 s (measured) |
| `phi.Geo_norm-1` | 0.29 s | 4.97 s | 63.7 s (measured) | 0.293 s | 62.8 s (measured) |

All four were measured directly at 118x1024.

### Where the time goes

The fresh T exponents are 0.05-0.07: the per-pair cost is Python/numpy/scipy call overhead on 2x2 and 4x4 matrices, not arithmetic on T samples (`cov_comp`, the only O(T) part, is 4.1% of phi_star and 1.7% of phi_G).

- **phi_star.** norm-0 profile (`infotheory.phi.star_norm-0__8x1024.pstats`): `phi_native.py:phi_star_gauss` 95.1% cum; inside it `scipy.optimize.minimize` (L-BFGS-B on the scalar beta) 71.6%, `scipy/optimize/_numdiff.py:approx_derivative` 40.3% (2-point finite-difference gradient: 270 calls for 56 pairs; 24.0% of the call is the 270 extra `i_s_objective` evaluations it makes, 16.3% numdiff bookkeeping), `scipy.linalg.solve` 34.3% (988 calls), `logdet` 16.2% (numpy cholesky 8.6%), `scipy.linalg.inv` 5.3% (596 calls, 540 of them `inv(C_D_cond)` inside the objective), `_prepare_scalar_function` 14.9%. norm-1 profile (`infotheory.phi.star_norm-1__8x1024.pstats`, same path plus the normalisation): `phi_star_gauss` 95.4%, minimize 69.2%, approx_derivative 33.8% (extra evaluations 22.2%, bookkeeping 11.6%), scipy solve 37.8%, logdet 14.5% (cholesky 8.5%), inv 4.9%, cov_comp 4.1%. opaque_share 0.23 / 0.25; self time by owner numpy 0.48 / 0.47, scipy 0.37 / 0.37, pyspi 0.14 / 0.15, so `hotspot_owner` is third-party:numpy (the owner with the largest self-time share; the hotspot_1 frame itself is pyspi).
- **phi_G (Geo).** norm-0 profile: `phi_native.py:phi_g_gauss_al` 96.0% cum / 29.7% self: ~51 augmented-Lagrangian iterations per pair (2848 numpy solves / 56 pairs), each with one `eigh` (2904 = 56 initial + 2848 in-loop; 10.7%), one `solve` (8.5%), `np.ix_` indexing (11.6%) and two Cholesky `logdet`s (5808 = 2 x 2904; 28.8%, cholesky 15.1%), plus `np.diag`/`sum` bookkeeping. norm-1 profile (`infotheory.phi.Geo_norm-1__8x1024.pstats`): `phi_g_gauss_al` 96.5% cum / 31.8% self, logdet 25.6% (5920 calls, 112 more for the normalisation; cholesky 13.3%), np.ix_ 12.4%, eigh 10.5%, solve 9.2%. opaque_share 0.19 / 0.17; self time by owner numpy 0.63 / 0.61, pyspi 0.36 / 0.38, so `hotspot_owner` is third-party:numpy.

### Opportunities

- **phi-1 (algorithmic; star norm-0/1): analytic gradient and hoisted inverse.** Give the beta search dI_s/dbeta from the same Q_inv solve (the archived MATLAB PhiToolbox optimiser had an analytic gradient) and move `inv(C_D_cond)` out of `i_s_objective`. `approx_derivative` is 40.3% (norm-0) / 33.8% (norm-1), but 24.0% / 22.2% of the call is the extra objective evaluations it makes, which an analytic derivative replaces with its own (cheaper or comparable) work; what certainly goes is the numdiff bookkeeping (16.3% / 11.6%) and the repeated inverse (5.3% / 4.9%). Range: 1.28-1.84x for norm-0 and 1.20-1.63x for norm-1 (lower end: the derivative costs one objective; upper end: Amdahl cap with the extra evaluations removed), written 1.3-1.8x and 1.2-1.6x. Risk unverified-within-tolerance: the optimiser path and stopping point move; today's pyspi sits within 4.65e-8 of the optimum on 37/37 check runs, which is the size of difference to expect. Not measured. The records list phi-2 in `subsumed_by` on star norm-0, and phi-3 and phi-2 on star norm-1 (a compiled per-pair computation or a cached norm-1 increment leaves no numdiff work for phi-1), so build_csv credits phi-1 no saving on top of them (README "Optimisation columns", ranking rule 4).
- **phi-2 (rewrite-numba-rust-cpp; star norm-0/1) and phi-4 (rewrite-numba-rust-cpp; Geo norm-0/1): compile or batch the 2x2 kernel.** Two avenues with separate ids because they rewrite different code: phi-2 compiles the whole per-pair phi_star computation (`cov_comp`, the 2x2 conditional covariances and a scalar Newton/Brent search on beta with analytic derivatives), phi-4 compiles `phi_g_gauss_al`'s augmented-Lagrangian loop (numba, or closed-form 2x2 eigendecompositions); either can instead batch over stacked (n_pairs, 2, 2) arrays with a convergence mask. For the two-element partition everything is 2x2/4x4. Caps from each unit's own profile: 1/(1-0.960) = 25x (Geo norm-0), 1/(1-0.965) = 28x (Geo norm-1), 1/(1-0.951) = 20x (star norm-0), 1/(1-0.954) = 22x (star norm-1); expected 5-20x (Geo, phi-4) and 5-15x (star, phi-2). Risk unverified-within-tolerance; for Geo (phi-4), the 1e-10 stopping rule on the logdet change can stop one iteration earlier or later under different rounding, and the check stage found non-stationary stops (reproduced by Octave/PhiToolbox) that move Phi_G by 3.25e-3 (delayed_noise T=64) up to 1.1 nats (shared_envelope T=64), so near such cases a rewrite can differ by more than fp tolerance. Not measured. On the norm-1 units the records list phi-3 in `subsumed_by` for phi-2 (star norm-1) and phi-4 (Geo norm-1): once the norm-1 identifier reads the cached unnormalised phi, there is no norm-1 optimisation left to compile, so build_csv credits them no saving on top of phi-3.

- **phi-3 (cache-sharing; star norm-1, Geo norm-1): one optimisation for both normalisations.** Normalisation is the last step of `phi_star_gauss` and `phi_g_gauss_al`, so a per-(pair, type, tau) cache of the unnormalised phi lets the second identifier skip the covariance and optimisation work (phi_star_gauss 95.4% and phi_g_gauss_al 96.5% of the norm-1 profiled calls; caps 22x / 28x; the normalisation itself adds 112 logdet calls per 56 pairs), expected 10-20x for the norm-1 increment when norm-0 runs first, the family's phi cost roughly halved. Unverified-bit-identical: the normalised branch divides the same float. Not measured.

### Refuted seed items

- **Hoist inv(C_D_cond) out of the objective** on its own: `scipy.linalg.inv` is 5.3% of the profiled phi_star call (under the ~10% bar); folded into phi-1.
- **Batch the covariances across pairs:** `cov_comp` is 4.1% (star) and 1.7% (Geo): refuted.
- **Analytic gradient** is confirmed but bounded at 1.2-1.8x (phi-1: part of approx_derivative's share is objective evaluations the analytic derivative replaces rather than removes); the call overhead (phi-2) is the larger avenue.
- **Symmetry:** A[i,j] == A[j,i] bitwise (check stage), and Undirected computes both orders; halving belongs to the cross-cutting xc-undirected-upper (2x on top of the above) and is not listed per unit; that row is recorded `independent_of` phi-1 to phi-4 because they act on different factors (`audit/profiles/crosscutting.yaml`): it halves the number of bivariate calls, phi-1, phi-2 and phi-4 cut the cost of each call, and phi-3 serves one identifier from another identifier's per-pair result, which the remaining i<j loop still reads once per pair; so it keeps its own 2x on their residual.

### Shared versus incremental cost

Nothing is shared today: each identifier recomputes the covariances and its optimisation (family-warm walls equal fresh walls: star 0.0997 vs 0.0987 s, Geo 0.291 vs 0.289 s at 8x1024). norm-0 and norm-1 differ only in the final division by (N_c - 1) x min H_p (`phi_native.py:235-244`, `:343-352`), which is what phi-3 exploits. No phi SPI is in the retained EEG selection.
