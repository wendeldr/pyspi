# cov_prec_simple: define, trace and check

Family: `basic.cov.{EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance}` and
`basic.prec.{EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance}`. Each unit covers the plain and
`-sq` identifiers. Stage: checked. The sections before `## Checks` are the trace stage; the numbers quoted there come from small
read-only probes (scratchpad `cov_prec_simple/probe1-6.py`, project venv: scikit-learn 1.9.1, SciPy 1.17.1,
NumPy 2.5.4) that confirm what the code does. The regression goldens are snapshots of current output and are not used as a reference anywhere here.
The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared pipeline (all eight units)

1. `pyspi/data.py:59-71`: by default `Data` uses `normalise=True` and `detrend=False`. At `pyspi/data.py:184-195`,
   each channel is z-scored with `scipy.stats.zscore(..., ddof=1)`.
2. `pyspi/calculator.py:327-339` (HEAD a1f20c7: :329 `multivariate` call, :332 `fill_diagonal`, :335 table store,
   :336-339 `except`, warning :354-355) calls `spi.multivariate(dataset)` and then sets the diagonal to NaN.
3. `pyspi/statistics/basic.py:31-44` (`Estimators._from_cache`) builds `z = data.to_numpy(squeeze=True).T` with shape
   (T x P). It then runs `getattr(sklearn.covariance, estimator)().fit(z)` with **all sklearn defaults** and caches the
   fitted object in `data.covariance[estimator]`. `cov_X` and `prec_X` share that object.
4. `pyspi/statistics/basic.py:46-54` reads `matrix = mycov.covariance_` or `mycov.precision_` and runs
   `np.fill_diagonal(matrix, np.nan)` **in place on the cached sklearn attribute**. It returns `np.square(matrix)` for
   `-sq` and otherwise returns the cached array object itself.
5. `Estimators(Undirected, Signed)` overrides `multivariate`, so the triangle mirroring in `Undirected.multivariate`
   (`pyspi/base.py:132-138`) never runs. This is already documented as REVIEW M1.
6. Every sklearn estimator centres each column on its sample mean and computes `np.cov(X.T, bias=1)`, i.e. it divides
   by T (`_empirical_covariance.py:107`). `_set_covariance` (`_empirical_covariance.py:196-215`) stores
   `precision_ = scipy.linalg.pinvh(covariance_)` at fit time (`:213`). `pinvh` (`scipy/linalg/_basic.py:2006-2023`)
   is a Moore-Penrose pseudo-inverse computed with `eigh`. It drops eigenvalues with `|s| <= P*eps*max|s|` and gives no
   warning.

Key consequence of the defaults: the z-score divides by T-1 but the covariance divides by T. On the default
pipeline, `S = (T-1)/T * R`, where R is the Pearson correlation matrix. The probe confirmed
`max |C - (1-s)(T-1)/T r| <= 4.4e-16` for all four estimators (bivariate_gaussian, T=1024). This confirms briefing
hypothesis 1.

## basic.cov.EmpiricalCovariance

- **Documented** (supplement §1.1.1, p.3-4): "Due to the z-scoring of the time series, the correlation and covariance
  matrices are equivalent and thus the covariance statistic is within [−1, 1]." The MLE estimator is called
  EmpiricalCovariance.
- **Canonical**: the MLE covariance `(1/T) Σ (x_t - x̄)(x_t - x̄)^T` (Anderson 2003, §3.2).
- **Implemented**: the canonical MLE. On the pipeline its off-diagonal equals `(T-1)/T * r`. Example: C[2,3] = 0.79671
  against r = 0.79749 at T=1024; at T=64 the factor is 0.984. With `normalise=False` the result is the raw covariance
  (probe on scaled_offset: 0.1125, equal to `np.cov(bias=True)`) and has no bound of 1.
- **Deviations**:
  - From the docs: the value is "equivalent to correlation" only up to the factor (T-1)/T.
  - The `-sq` variants are undocumented. The probe confirms they are exactly `np.square` of the plain matrix.
  - The GitBook keyword line (`basic-statistics.md:17`: "nonlinear, unsigned, bivariate") contradicts both the
    supplement and the config.
  - None from the canonical definition.
- **Proposed checks**:
  - Kernel path against a from-scratch NumPy `Xc Xc^T/T`.
  - Pipeline path against the analytic `(T-1)/T * np.corrcoef`.
  - Statistical convergence to rho = 0.3 / 0.8 on bivariate_gaussian.
  - Exact symmetry: 0 of 76 dataset files were asymmetric in the probe.

## basic.cov.LedoitWolf

- **Documented**: "the Ledoit-Wolf method for optimizing the shrinkage coefficient" (supplement §1.1.1, ref [4]).
- **Canonical**: Ledoit & Wolf 2004, JMVA 88:365-411, §3: `S* = (b²/d²) m I + (a²/d²) S`, with
  `b² = min(b̄², d²)` and the normalised Frobenius norm.
- **Implemented**: `_shrunk_covariance.py:584-613` → `_ledoit_wolf` (`:25-43`) → `ledoit_wolf_shrinkage`
  (`:297-400`). A from-scratch LW 2004 implementation reproduced sklearn's shrinkage to 10 digits (0.1842118924,
  0.01415005694, 0.2503679232). On the pipeline the off-diagonal is `(1-s)(T-1)/T r`.
- **Deviations**:
  - From the docs: the value is a correlation shrunk by `(1-s)(T-1)/T`.
  - The `-sq` variants are undocumented.
  - None from the canonical definition, apart from the standard sample-mean centring.
  - Behaviour note: on iid_gaussian the shrinkage clips to 1 at every T, so every off-diagonal is exactly 0.0.
- **Proposed checks**: a from-scratch LW (independent-impl, tolerance), the analytic pipeline identity, the exact zero
  result on iid data, and symmetry.

## basic.cov.OAS

- **Documented**: "the oracle approximating shrinkage ... if the data are Gaussian" (supplement §1.1.1, ref [5],
  Chen et al. 2010).
- **Canonical**: Chen, Wiesel, Eldar & Hero 2010, IEEE TSP 58(10), Eq. 23. I read the formula from arXiv:0907.4698v1:
  `rho = min([(1-2/p) tr S² + tr² S] / [(n+1-2/p)(tr S² - tr² S/p)], 1)`.
- **Implemented**: `_oas` (`_shrunk_covariance.py:46-94`) computes
  `rho = min((tr S² + tr² S)/((n+1)(tr S² - tr² S/p)), 1)`. The comment at lines 70-71 says "The factor 2 / p is
  omitted since it does not impact the value of the estimator for large p".
- **Deviation from canonical (candidate)**: at EEG-like small p the omission does matter.
  - bivariate_gaussian (p=4): 0.18388 (sklearn) vs 0.16058 (Eq. 23) at T=64, and 0.014569 vs 0.012728 at T=1024.
  - sparse_precision (p=5): 0.21778 vs 0.19936 at T=64.
  - So the off-diagonals are scaled by a (1-rho) that is a few percent smaller than Eq. 23 gives.
  - Relative to Eq. 23 the sklearn shrinkage is +14.5% (p=4) and +9.2% (p=5) at T=64.
  - sklearn documents the omission itself. The OAS class docstring (`_shrunk_covariance.py:754-758`) and the `oas()`
    docstring (`:655-659`) say the formula "differs from Eq. 23" and that 2/p is dropped because it is small "for a
    large p".
  - scikit-learn 0.24.1, the version named in supplement §1.1.1 (p.4), computes the same num/den
    (0.24.1 `sklearn/covariance/_shrunk_covariance.py:487-492`). The comment at line 487 reads "formula from Chen et
    al.'s **implementation**", the `oas()` docstring at 466-467 says "slightly modified compared to the one given in
    the article", and the class docstring at 508-512 describes the omitted 2/p. So the documented pyspi version
    computed the same value. This is a documented upstream approximation, not a version change. The remaining
    candidate is only that the "large p" justification does not hold at small p.
- **Other deviations**: the same issues as for LW (shrunk correlation, undocumented `-sq`, exact zeros on iid data).
- **Proposed checks**:
  - Two from-scratch references: Eq. 23 (expected to differ) and the sklearn variant (expected to match to 1e-12).
    Together they isolate the 2/p effect.
  - The pipeline identity, and symmetry.

## basic.cov.ShrunkCovariance

- **Documented**: "a basic shrinkage covariance estimator with a fixed shrinkage coefficient of 0.1"
  (supplement §1.1.1).
- **Canonical**: no literature paper defines this estimator. The definition is sklearn's `0.9 S + 0.1 (tr S/p) I`
  (Pedregosa et al. 2011), which is the LW 2004 convex-combination family with a fixed coefficient.
- **Implemented**: `ShrunkCovariance.fit` (`_shrunk_covariance.py:255-283`) → `shrunk_covariance` (`:109-156`), with
  the default `shrinkage=0.1` (`:248-252`). On the pipeline the value is `0.9 (T-1)/T r`, so |value| <= 0.9 always.
  Example: C[2,3] = 0.71704 against r = 0.79749.
- **Deviations**: the docs' claim that covariance is equivalent to correlation is clearly false for this estimator (a
  fixed factor of 0.9). None from the sklearn definition.
- **Proposed checks**: the analytic kernel and pipeline identities (tolerance 1e-12), and symmetry.

## basic.prec.* (all four estimators)

- **Documented** (supplement §1.1.2, p.4): "the matrix inverse of the covariance matrix ... For normalized
  time-series data, the precision matrix is equivalent to the partial correlation ... and is within [−1, 1]."
- **Canonical**: the precision (concentration) matrix `Θ = Σ̂^{-1}` (Dempster 1972). The partial correlation is
  `-Θ_ij/√(Θ_ii Θ_jj)` (Whittaker 1990, ch. 5).
- **Implemented**: the cached fit's `precision_ = pinvh(covariance_)`, with the diagonal set to NaN. There is no
  normalisation and no sign flip. On the pipeline:
  - EmpiricalCovariance: `Θ = T/(T-1) R^{-1}`.
  - Shrinkage estimators: `Θ = T/(T-1) [(1-s)R + sI]^{-1}`.
- **Deviation from documented (candidate, material; confirms briefing hypothesis 2)**: the output is not the partial
  correlation. It has the opposite sign and its magnitude is unbounded.
  - bivariate_gaussian, T=1024, pair (2,3): prec_EmpiricalCovariance = -2.193, while the partial correlation from the
    same matrix is +0.797. The other estimators give -2.061 (LW), -2.057 (OAS) and -1.482 (Shrunk).
  - cointegrated, T=1024: max |prec| = 95.3 (EmpiricalCovariance), 61.5 (LW), 49.1 (OAS) and 4.26 (Shrunk).
  - The largest value over all 76 dataset files differs by estimator: 3042.8 (EmpiricalCovariance), 563.1 (LW) and
    341.9 (OAS), all at trending_T4096, and 4.71 (Shrunk) at shared_envelope_T64.
- **Scaling note (not a deviation)**: the supplement's partial-correlation sentence is stated "For normalized
  time-series data", so the z-scoring is part of the documented definition. As a consequence, on the pipeline the
  value is the inverse of the (shrunk) correlation matrix times T/(T-1), not the inverse of the raw-units covariance.
  - sparse_precision, T=4096 (EmpiricalCovariance): the kernel path gives -0.393 (dataset precision_true -0.4) and the
    pipeline gives -0.539.
  - `D Θ_kernel D` with `D = diag(sqrt(S_ii))` gives -0.5392 for the same pair, so the pipeline value is the kernel
    precision rescaled by the channel standard deviations.
  - This matters only when pipeline values are compared with a known raw-units precision matrix.
- **prec_EmpiricalCovariance, singular case (candidate)**: when S is singular, pinvh silently returns a pseudo-inverse.
  - shared_envelope T=256 has x2 bit-identical to x0, and eig(S) includes 0.
  - prec[0,2] = +0.338 there, the same order as unrelated pairs.
  - With P=8 > T=5 the output is finite (max 0.75).
  - The shrinkage estimators are full rank in this case and return large magnitudes for the duplicated pair: LW -27.8,
    OAS -17.7, Shrunk -4.68.
- **Symmetry**: equal only up to last-bit roundoff. The largest |A-Aᵀ| over the 76 files was 5.7e-14 for
  EmpiricalCovariance, LW and OAS, and 8.9e-16 for Shrunk. It was nonzero in 60 (EmpiricalCovariance), 49 (LW),
  47 (OAS) and 60 (Shrunk) of the 76 files.
  This is already documented as REVIEW M2 (and M1 for the missing mirror). It is not new.
- **OAS / LW**: the prec units inherit the covariance findings: the 2/p omission for OAS, and the exact zeros on iid
  data.
- **Proposed checks**:
  - Kernel path: `np.linalg.inv` (NumPy LU, not SciPy pinvh) of a from-scratch estimate, tolerance 1e-9.
  - A qualitative check against the analytic partial correlation, documenting the sign and range.
  - sparse_precision convergence on the kernel path.
  - shared_envelope singular case: compare to `numpy.linalg.pinv` for EmpiricalCovariance, and record the magnitudes
    for the shrinkage estimators.
  - A symmetry tolerance check (direction criterion).

## Cross-cutting candidates

- **In-place mutation and aliasing.** `basic.py:49-50` sets the diagonal of the cached sklearn `covariance_` /
  `precision_` to NaN and returns that same array.
  - The probe showed `dat.covariance[est].covariance_` diagonal = `[nan nan nan nan]` after the call, and "returned is
    cache: True".
  - No current output changes: `precision_` is computed at fit time, and no other module reads `data.covariance`.
  - Any later consumer of the cached fit would see the corrupted diagonal: `get_precision`, `score`, `mahalanobis`, or
    a caller mutating the returned matrix.
- **Squared variants.** `-sq` is exactly the elementwise square (confirms briefing hypothesis 3). It is labelled
  unsigned and is not documented in the supplement or GitBook.
- **Not in this family.** The supplement still lists EllipticEnvelope, MinCovDet, GraphicalLasso and GraphicalLassoCV.
  They share this code path but belong to other units.

## Open questions

1. OAS: should dropping Chen et al.'s 2/p terms count as a deviation (verdict `variant`), or as acceptable sklearn
   behaviour? sklearn documents the omission, which favours treating it as a known approximation. The effect size at p=19-64 (typical EEG montages) has not been measured. Expectation (hypothesis): the
   relative difference in shrinkage scales roughly with 2/p, so it is about 3-10%.
2. Answered: scikit-learn 0.24.1 used the same OAS formula (0.24.1 `_shrunk_covariance.py:487-492`, same num/den
   with the 2/p terms omitted). The OAS shrinkage formula is therefore not a source of difference from historical
   pyspi outputs. Other version differences were not checked.
3. For prec on singular inputs (duplicated or bridged EEG channels, average reference giving rank P-1), should pyspi
   warn or return NaN? An average-referenced montage makes S exactly singular, so prec_EmpiricalCovariance would be a
   pseudo-inverse for every pair. This is a hypothesis to test on a rank-deficient dataset; none is in the MANIFEST
   apart from shared_envelope.
4. Should prec be reported as partial correlation (normalised and sign-flipped) to match the docs, or should the docs be
   corrected? This is a definition decision, not a bug fix.
5. Could the in-place diagonal mutation affect EllipticEnvelope/MinCovDet in other families, if their downstream use
   reads `covariance_` after a `cov_` call? Not traced here.

## Checks

Stage: checked. Manifest `audit/checks/manifest/cov_prec_simple.json` (491 runs). References:
`audit/checks/refs/cov_prec_simple.py`, which runs in the comparison venv and never imports sklearn. Probes:
`audit/checks/probes/cov_prec_simple_probes.py`. Results are in `audit/checks/out/results/cov_prec_simple.json`:
**451 pass, 40 fail, 0 without reference.** Every failure is accounted for:

- 36 failures are documented-claim runs that are expected to fail. Each one records the size of a gap:
  - `doc-corr`: cov against plain Pearson r.
  - `doc-pcorr`: prec against the partial correlation.
  - `eq23`: OAS against Chen et al. Eq. 23.
- 4 failures are statistical runs on one dataset draw, `sparse_precision_T1024`, kernel path, cov units only (see
  the statistical criterion below).

The manifest generator is scratchpad `cov_prec_simple_check/gen_manifest.py`. The scratchpad scripts `summ.py` and
`agg.py` print the per-run tables used below.

### Design

- **Independent estimators.** The estimators come from `_kernels_basic_misc`:
  - `empirical_cov`: `Xc Xc^T / T`.
  - `ledoit_wolf`: LW 2004 eq. 14 with the literal per-sample bbar^2 sum.
  - `oas(paper=False)`: sklearn's formula without the 2/p terms.
  - `oas(paper=True)`: Chen et al. Eq. 23.
  - `shrunk_cov(0.1)`.

  For precision, the inverse is NumPy's LU inverse (`np.linalg.inv`), not the eigh/pinvh route that pyspi takes. The
  difference from a pinvh mirror is reported as `ref_max_rel_inv_vs_pinvh_mirror`, and it is at most 1e-9 on the
  well-conditioned runs.

  The preprocessing is re-implemented in NumPy:
  - z-score with ddof=1;
  - least-squares linear detrend, then z-score, for `pipeline-detrend`.
- **Paths.**
  - `kernel` isolates the estimator.
  - `pipeline` and `pipeline-detrend` catch the normalisation assumptions: the (T-1)/T factor, the shrinkage being
    recomputed on z-scored data, and the D Θ D scaling of precision.
- **Inputs.** Besides the MANIFEST datasets, the checks use:
  - two EEG-like P = 19 stacks of independent datasets:
    - `stackA` (weakly correlated): iid_gaussian + bivariate_gaussian + sparse_precision + gauss_ar1_pair +
      var2_chain;
    - `stackB` (strongly correlated): delayed_noise + narrowband 0.1 + narrowband 0.25 + var2_bivariate +
      bivariate_gaussian + var2_chain;
  - average-referenced inputs (`x - x.mean(0)`), which have rank P-1 by construction.
- **Statistical criterion (`truth` and `null` runs).**
  - The reference value is the Monte Carlo mean of the same independent estimator, applied on the same path to
    N = 1000 Gaussian draws from the dataset's population model (`cov_true`). The se is the Monte Carlo sd.
  - The se is multiplied by z_{1-0.05/(2m)}/2, with m = P(P-1)/2 pairs: 1.32 for P = 4 and 1.40 for P = 5. compare.py's
    rule "every pair within 2 se" is then a familywise 5% test, not a 5% test per pair.
  - Each truth run also reports `mc_familywise_p`: the fraction of Monte Carlo replicates whose largest standardised
    pair deviation is at least the observed one.
  - **The 4 failing statistical runs** are `basic.cov.{EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance}|sparse_precision|1024|kernel|truth`.
    - Each has max_z 2.26, max |dev|/sd 3.17 and `mc_familywise_p` = 0.014.
    - The realised T=1024 file has entry (2,3) 3.2 sd above its expectation (diagonal entries (2,2) and (3,3) are
      +2.9 and +2.0 sd).
    - The same file passes the exact reference for all four estimators. The T=256 and T=4096 files pass, and so do
      the four precision units on the same file.
    - This is a property of the draw. It is not an estimator mismatch, and the criterion was not loosened.
- **Reference kinds of the supporting runs.** The verdicts rest on the `ref`, `sq-ref`, `sym`, `singular`, `avgref`
  and stack runs (independent-impl, or analytic for the pseudo-inverse runs). The other runs are labelled as follows:
  - `truth` and Monte Carlo `null` runs: **independent-impl**. The Monte Carlo pushes population draws through the
    independent `_kernels_basic_misc` estimator, so it is no more than an independent implementation.
  - LedoitWolf/OAS `null` runs: **analytic**. The reference is the exact zero matrix that follows from the clipped
    shrinkage, with no Monte Carlo.
  - `mutation` and `mutation-others` runs: **same-lineage**. They compare pyspi with pyspi computed in another call
    order.

  An adversarial review found that earlier versions of `refs/cov_prec_simple.py` labelled the `truth`, Monte Carlo
  `null`, `mutation` and `mutation-others` runs `analytic`. The labels were corrected and the family re-run with
  `--force`. The outcome was the same: 451 pass, 40 fail, with the same failing runs. No verdict or score depends on
  these runs.
- **Tolerances.**
  - Covariance: exact, rtol 1e-10 / atol 1e-12.
  - Precision: rtol 1e-10 with atol = 1e-12·max|Θ|. This is the norm-wise accuracy of a computed inverse.
    - The largest elementwise relative difference in any precision comparison is 1.0e-9.
    - It occurs at a near-zero entry of `basic.prec.EmpiricalCovariance|iid_gaussian|1024|kernel|stackB` (cond 750).
    - Its absolute difference is 1.3e-11, which is inside atol.
  - Pseudo-inverse runs use the same tolerance.

### basic.cov.EmpiricalCovariance: matches, 9

- **Independent reference.** 33/33 value runs pass, with max abs 1.22e-15 (`ref`, `sym`, `singular`, `avgref`,
  `stack*`). `np.cov(bias=True)` differs from the reference by 0.0 on every kernel run.
- **Pipeline.** `doc-corr` gives pyspi/r = 0.984375 at T=64 and 0.9990234 at T=1024 on every pair. That is exactly
  (T-1)/T: `basic.cov.EmpiricalCovariance|bivariate_gaussian|{64,1024}|pipeline|doc-corr`, max rel 1/T.
- **Squared variant.** `cov-sq` is bitwise `np.square` of the plain output (`sq-ref`).
- **Convergence (bivariate_gaussian, pipeline).** The distance to the population correlation is 0.142, 0.0658, 0.0363
  and 0.0161 at T = 64, 256, 1024 and 4096. That is about 1/sqrt(T), and all four runs pass the familywise rule.
- **Null (iid_gaussian).** 4/4 pass.
- **Symmetry.** Bitwise symmetric on all 51 matrix runs.
- **Score.** Rubric anchor 9: canonical, up to the trivial (T-1)/T convention, against the supplement's "equivalent
  to correlation".

### basic.cov.LedoitWolf: matches, 9

- **Independent reference.** 33/33 value runs pass, with max abs 3.67e-15 and max rel 2.24e-12.
- **Fitted shrinkage.** The fitted `shrinkage_` (probe) matches the kernel on 26/26 runs, max abs 4.4e-16, over
  P = 4/5/19, T = 64-4096, kernel and pipeline.
- **Pipeline.** pyspi/r = (1-s)(T-1)/T: 0.8030414 at T=64 (s = 0.184) and 0.9848872 at T=1024.
- **Null.** iid_gaussian gives s = 1.0 and exact 0.0 off-diagonals at all four T (`null` runs). delayed_noise
  (zero-lag uncorrelated) also gives s = 1.0.
- **Convergence (bivariate_gaussian).** 0.136, 0.062, 0.0358, 0.016.
- **Symmetry.** Bitwise symmetric.
- **Score.** Anchor 9. The shrinkage is the estimator the supplement names. The departure from "equivalent to
  correlation" is the shrinkage itself plus the (T-1)/T factor.

### basic.cov.OAS: variant, 7

- **Agreement with sklearn's variant.** pyspi equals the no-2/p variant: 33/33 value runs, max abs 1.22e-15; fitted
  shrinkage 26/26, max abs 1.1e-16.
- **Gap to Eq. 23, shrinkage** (sklearn rho relative to the Eq. 23 rho, from the `shrink` and `*stack*-shrink` runs):

  | P | T | sklearn rho vs Eq. 23 rho |
  |---|---|---|
  | 4 (bivariate_gaussian) | 64-4096 | +14.4% to +14.5% |
  | 5 (sparse_precision) | 64-4096 | +9.6% to +10.3% |
  | 19 (stackA) | any | +0.61% to +0.71% |
  | 19 (stackB) | any | +0.67% to +0.69% |

- **Gap to Eq. 23, off-diagonals** (`eq23` runs, which are expected to fail):

  | P | T | path | relative change |
  |---|---|---|---|
  | 4 | 64 | kernel | 2.9% |
  | 4 | 1024 | kernel | 0.19% |
  | 19 | 64 | kernel | 0.68% |
  | 19 | 1024 | pipeline | 0.055% |

- **Analytic bound.** With a = tr S² ≤ b = tr² S,
  `rho_sklearn / rho_eq23 - 1 <= (2/p) a / ((1-2/p) a + b) <= 1/(p-1)`. That is 33% at p=4, 5.6% at p=19 and 1.6% at
  p=64. The bound is reached when one common component dominates S. The weakly correlated stacks sit near the
  O(1/p²) end.
- **Score.** Anchor 7: it matches the variant that the supplement's sklearn 0.24.1 also ran, but differs from the
  canonical Eq. 23 in a way that changes values.

### basic.cov.ShrunkCovariance: matches, 9

- **Independent reference.** 33/33 value runs pass, max abs 1.11e-15.
- **Pipeline.** pyspi/r = 0.9 (T-1)/T: 0.8859375 at T=64 and 0.8991211 at T=1024. So |value| ≤ 0.9, and the value is
  never "equivalent to correlation".
- **Symmetry.** Bitwise symmetric.
- **Score.** Anchor 9. The supplement describes exactly this fixed-0.1 estimator.

### basic.prec.EmpiricalCovariance: matches (canonical precision), 7, definition under review

- **Independent reference.** 34/34 value runs pass against the LU inverse, max abs 1.3e-11.
- **Partial correlation.** `doc-pcorr` compares against -Θ_ij/sqrt(Θ_ii Θ_jj):
  - The sign is opposite on every pair (frac_opposite_sign 1.0 on bivariate_gaussian, sparse_precision and
    cointegrated at T=1024).
  - bivariate_gaussian (2,3): prec -2.193 vs partial correlation +0.7972.
  - bivariate_gaussian (0,1): -0.3065 vs +0.2816.
  - cointegrated: max |prec| 95.34.
  - Re-normalising pyspi's own value reproduces the partial correlation to 4.1e-14.
- **Singular input** (shared_envelope, x2 == x0):
  - pyspi equals `numpy.linalg.pinv`, max rel 4.3e-15 at T=256 and 4.1e-14 at T=1024. Meanwhile `np.linalg.inv`
    raises "Singular matrix".
  - The duplicated pair (0,2) is +0.3379 at T=256 and +0.3338 at T=1024.
- **Average reference** (rank P-1):
  - pyspi equals the pseudo-inverse with the structural null eigenvalue removed, max rel 9.2e-16 at P=4 and 4.5e-13
    at P=19 (stackB).
  - The null eigenvalue is 2.6e-16 at P=4 and -1.0e-15 at P=19. pinvh drops it. The LU inverse would reach 2.4e15 and
    1.1e16.
- **Scaling.** On the pipeline, prec = D1 Θ_kernel D1 with D1 = diag(sd, ddof=1), to 3.0e-14. At sparse_precision
  T=4096, pair (0,1) is -0.5393 on the pipeline, -0.3925 on the kernel path, and -0.4 in `precision_true`.
- **Kernel convergence to `precision_true`.** The distance is 0.141, 0.0595 and 0.0209 at T = 256, 1024 and 4096.
  All pass, with the Monte Carlo mean including the Wishart bias T/(T-p-2).
- **Symmetry.** Equal only up to the last bit: max |A-A.T| 5.68e-14 (`trending|1024|pipeline|ref`), nonzero in 53 of
  54 runs. This is REVIEW M2.
- **Score.** Anchor 7. Anchor 7 literally names the mirror case: the supplement is matched and canonical differs.
  This family reads the anchor symmetrically: the value matches exactly one of the two definitions and differs from
  the other in a value-changing way. Here the value matches the canonical precision fp-exactly (independent LU
  inverse, max abs 1.3e-11). It differs from the supplement's stated quantity ("equivalent to the partial correlation
  ... within [-1, 1]") in sign and range. The same convention gives 7 to prec.LedoitWolf and prec.ShrunkCovariance.
  prec.OAS gets 6 because it differs from both definitions (the no-2/p shrinkage, below). The robust and glasso prec
  units (cov_prec_robust, cov_prec_glasso) also get 6: they differ from both, through the FAST-MCD scatter and the
  glasso iterate that is not at the optimum. **Synthesis should confirm this reading of anchor 7.** Rubric text
  that covers "matches canonical, documented quantity wrong" would settle it.
  **Workflow 4b challenge (2026-10-15): reading kept.** The challenger weighed the opposite reading: the supplement's
  operative definition is "the matrix inverse of the covariance matrix", which the code meets exactly, and the
  partial-correlation sentence is only a gloss; on that reading prec.OAS would equal cov.OAS (7) and prec.LedoitWolf,
  prec.EmpiricalCovariance and prec.ShrunkCovariance would be 9. It upheld the scores here instead, because the
  supplement asserts a concrete value property ("equivalent to the partial correlation ... within [-1, 1]") that fails
  in a value-changing, non-monotone way (opposite sign on every pair, unbounded up to 49.1), whereas cov.OAS keeps 7
  because its documentation gap is a monotone factor. The rubric text in `audit/README.md` is unchanged; this note is
  where the symmetric reading is recorded. **Settled at synthesis by the score columns:** README rule R4 ('code canonical,
  documentation different') reads prec.EmpiricalCovariance, prec.LedoitWolf and prec.ShrunkCovariance at `score_rubric` 9
  (nearest-anchor: anchor 10's supplement clause fails), while `score_0_10` keeps the 7 assigned under the symmetric reading for
  provenance; prec.OAS stays 6 in both columns, because it also differs from the canonical value.

### basic.prec.LedoitWolf: matches, 7; basic.prec.ShrunkCovariance: matches, 7

- **Independent reference.** 34/34 value runs pass each, max abs 4.28e-12 (LW) and 3.02e-14 (Shrunk).
- **Partial correlation.** The sign is opposite on every pair.
  - LedoitWolf: (2,3) is -2.061 vs +0.7859; cointegrated max 61.5.
  - ShrunkCovariance: (2,3) is -1.482 vs +0.7174; cointegrated max 4.258.
- **Duplicated channel.** The estimate is full rank, so the inverse is ordinary. The duplicated pair (0,2) is large
  and negative:

  | estimator | T=256 | T=1024 |
  |---|---|---|
  | LedoitWolf | -27.81 | -100.8 |
  | ShrunkCovariance | -4.681 | -4.67 |

- **LedoitWolf null.** iid_gaussian gives exact 0.0 at every T.
- **Symmetry.** Last-bit only: max 1.42e-14 (LW) and 8.88e-16 (Shrunk).

### basic.prec.OAS: variant, 6

- This unit has the cov.OAS 2/p deviation and the prec deviation from the documented definition together.
- **Gap to the inverse of the Eq. 23 estimate** (off-diagonals, median / max):

  | P | T | path | median | max |
  |---|---|---|---|---|
  | 4 | 64 | kernel | 5.1% | 12.7% |
  | 4 | 1024 | kernel | 0.44% | 1.7% |
  | 19 (stackB) | 1024 | pipeline | 0.21% | 20.5% (a near-zero entry) |

- **Partial correlation.** (2,3) is -2.057 vs +0.7856; cointegrated max 49.12.
- **Duplicated pair (0,2).** -17.69 at T=256 and -68.19 at T=1024.
- **Score.** Anchor 6: deviates from both the supplement and the canonical estimator, in interpretable ways.
- **EEG-sized P.** The canonical (2/p) part is small there: the shrinkage excess is bounded by 1/(p-1) (5.6% at p=19,
  0.9% at p=118; +0.61-0.71% observed on the P=19 stacks, off-diagonal median 0.21% at T=1024). In EEG use the score
  therefore rests mainly on the documentation gap (sign and range against the stated partial correlation). A Workflow
  4b challenge probe (scratchpad `wf4b_mid-eeg/oas_prec.py`) reproduced the P=4 numbers on the pipeline path: shrinkage
  +14.5%, pyspi equal to the inverse of the no-2/p estimate to 2.4e-15, opposite sign to the partial correlation on every pair.

### Candidates

Every trace candidate is **confirmed**. None of them is in KNOWN_ISSUES.md: REVIEW M1 (the mirror bypass) and M2
(last-bit asymmetry) are cited as already documented. Evidence is the run_ids in each record's
`candidate_resolutions`. The candidates are:

1. **In-place mutation and aliasing** (`mutation` runs).
   - The returned array *is* the cached `covariance_` or `precision_`, and the cache diagonal is NaN afterwards.
   - pyspi outputs are bitwise order independent. This holds for all four estimators here and also for EllipticEnvelope,
     MinCovDet, GraphicalLasso and GraphicalLassoCV (`basic.cov.EmpiricalCovariance|bivariate_gaussian|256|pipeline|mutation-others`).
   - The cached fit's own `score()` and `mahalanobis()` return NaN after a prec call. The hazard is latent: nothing in
     pyspi reads `data.covariance` outside `basic.py:31-54`.
2. **(T-1)/T and (1-s) factors against "equivalent to correlation"** (`doc-corr` runs).
3. **Undocumented `-sq`**, which is exactly `np.square` (`sq-ref`).
4. **GitBook keyword line** (`basic-statistics.md:17`). Confirmed by reading the documentation.
5. **OAS 2/p omission at small p** (`eq23` and `shrink` runs).
6. **LW/OAS clip to exact zeros on null data** (`null` runs).
7. **prec is not partial correlation** (`doc-pcorr` runs).
8. **Silent pinvh pseudo-inverse** (`singular` and `avgref` runs). The P > T sub-claim was not re-run.
9. **Pipeline precision scaling** (`scaling` runs). The wording is corrected: the T/(T-1) factor of the trace-stage
   sentence applies only when D uses ddof=0 standard deviations. With ddof=1 standard deviations the identity is
   Θ_pipe = D1 Θ_kernel D1 with no factor. For the shrinkage estimators, the D Θ D identity is not exact, because the
   shrinkage is recomputed on the z-scored input. Example: LedoitWolf bivariate_gaussian T=1024 has s = 0.014312 on
   the kernel path and 0.01415 on the pipeline.

### Trace-stage fixes

- The issues listed for this family were already fixed when the check stage started, and I re-checked each one:
  - The LW/OAS/Shrunk `deviation_from_documented` fields carry their own pair values: -2.061, -2.057 and -1.482.
  - No `486-491` citation remains.
  - The note header names probe1-6.
- No trace-stage record field was changed. The scaling-note correction above is recorded in the `candidate_resolutions`
  of the prec units.

### Harness note

`pyspi_outputs.py` crashed twice with `JSONDecodeError` in `load_index()`. The cause was reading another family's
index file while that family's concurrent run was half-way through writing it: `write_index` is not atomic. Re-running
skipped the cached outputs and completed the family. No shared script was edited.

## Open questions: answered

1. **Is the OAS 2/p omission a variant or acceptable sklearn behaviour?**
   - Recorded as a **variant** (cov 7, prec 6). sklearn documents it, and the supplement's sklearn 0.24.1 also ran
     it.
   - Effect size, measured:

     | p | shrinkage vs Eq. 23 | off-diagonals |
     |---|---|---|
     | 4 | +14.4% | ≤ 2.9% |
     | 5 | +10% | not measured separately |
     | 19 (stacked P=19 arrays) | +0.61% to +0.71% | ≤ 0.68% (kernel) / 1.07% (pipeline) at T=64, ≤ 0.23% at T=256, 0.055% at T=1024 |

   - The analytic bound is 1/(p-1): 5.6% at p=19 and 1.6% at p=64. Strongly volume-conducted EEG, where one component
     dominates, sits nearer the bound than the stacks tested here.
   - For EEG-sized p the "large p" justification therefore holds to within 0.23% of the off-diagonals at T ≥ 256
     (1.07% at T=64 on the pipeline). At p ≤ 5 it does not hold.
2. **Did scikit-learn 0.24.1 use the same OAS formula?**
   - Answered at the trace stage from the hash-verified 0.24.1 source (`_shrunk_covariance.py:487-492`, same num/den).
   - It was not reproduced numerically: 0.24.1 is not installed, and installing it is out of scope here.
   - A 0.24.1 environment run on the `shrink` runs' inputs would settle it numerically.
3. **Is prec_EmpiricalCovariance a pseudo-inverse on average-referenced EEG?**
   - Tested: yes. On an average-referenced input (rank P-1), `prec_EmpiricalCovariance` is the Moore-Penrose
     pseudo-inverse for every pair, with no warning:
     - `basic.prec.EmpiricalCovariance|bivariate_gaussian|1024|pipeline|avgref`, P=4;
     - `basic.prec.EmpiricalCovariance|iid_gaussian|1024|pipeline|stackB-avgref`, P=19.
   - pinvh's cutoff, P·eps·max|w| (about 9e-15 at P=19), dropped the computed null eigenvalue (≤ 1.0e-15 in
     magnitude). The values look ordinary, so nothing flags them.
   - The shrinkage estimators stay full rank on the same input. Their precision matches the LU inverse (`avgref`
     runs).
   - Not tested: whether the null eigenvalue can land above the cutoff at larger P or worse scaling. If it did, the
     output would be about 1e15-1e16 instead of a pseudo-inverse.
4. **Should prec become partial correlation, or should the docs change?**
   - Still a definition decision; `definition_under_review: true` on the prec units.
   - The evidence fixes the relation exactly: prec_ij = -pcorr_ij·sqrt(Θ_ii Θ_jj), reproduced to 4.1e-14. The sign is
     opposite on 100% of pairs, and the magnitude reaches 95 (cointegrated).
   - Either the docs must say "unnormalised precision entry, opposite sign to partial correlation, unbounded", or the
     code must return -Θ_ij/sqrt(Θ_ii Θ_jj). Maintainers have to choose.
5. **Does the in-place diagonal mutation affect EllipticEnvelope, MinCovDet or GraphicalLasso?**
   - Tested: no output effect. With fixed seeds, cov, cov-sq, prec and prec-sq of EllipticEnvelope, MinCovDet,
     GraphicalLasso and GraphicalLassoCV are bitwise identical whether computed on a fresh Data or on a shared Data, in
     either order (`basic.cov.EmpiricalCovariance|bivariate_gaussian|256|pipeline|mutation-others`).
   - `precision_` is computed at fit time, so a covariance call cannot reach it.
   - Only external use of the cached fit (`score`, `mahalanobis`, `get_precision`) sees the NaN diagonal.

## Optimisation

Workflow 3, group basic. Walls are from `audit/profiles/summary.json` (quiet machine, fresh mode = the representative identifier alone; all four shapes measured, `measure_mode: measured` at 118x1024). Percentages are shares of the profiled `compute_spi` call. Nothing in pyspi was changed. The units are cheap: every number below is milliseconds per window.

| unit | 8x1024 | 32x1024 | 118x1024 (measured) | family-warm increment at 118 |
|---|---|---|---|---|
| basic.cov.EmpiricalCovariance | 0.8 ms | 1.0 ms | 3.8 ms | 3.7 ms (fit) + 0.1 ms (cov-sq) |
| basic.cov.LedoitWolf | 0.9 ms | 1.1 ms | 4.7 ms | 4.4 ms + 0.1 ms |
| basic.cov.OAS | 0.7 ms | 0.9 ms | 3.8 ms | 3.4 ms + 0.1 ms |
| basic.cov.ShrunkCovariance | 0.9 ms | 1.1 ms | 3.9 ms | 3.7 ms + 0.1 ms |
| basic.prec.* (fresh) | 0.7-0.8 ms | 0.9-1.1 ms | 3.6-4.7 ms | ~0.1 ms per identifier (cache hit on the cov fit) |

**Where the time goes.** The 8x1024 profiles are dominated by fixed sklearn overhead: `sklearn/utils/validation.py:check_array` is 31-36% of the call and `scipy/linalg/_basic.py:pinvh` 10.8-13.8% cumulative (the `eigh` inside it is 4.9-6.4%; the records quote both). That is not what matters at 118 channels. A profile at 118x1024 (`audit/profiles/micro/prof_basic.cov.EmpiricalCovariance__118x1024.pstats`, written by this stage with the same seeded harness call) shows `scipy/linalg/_decomp.py:eigh` (inside pinvh, called by sklearn `_empirical_covariance.py:_set_covariance` to store `precision_`) at 58.7% of the call (pinvh 63.0%), `np.cov` 15.4%, and check_array 10.6%. A scratch timing at 118x1024 (OMP_NUM_THREADS=1) gave pinvh 2.17 ms, np.cov 0.49 ms and full fits of 2.9-3.8 ms for the four estimators, so the O(P^3) eigh is the dominant term for all of them. The four 118x1024 profiles of this group (EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet) are regenerated by `audit/profiles/micro/prof_basic.py run <dir>`, and `prof_basic.py index` writes `audit/profiles/micro/prof_basic__index.json` (frame shares, opaque_share by the profile_units definition, and the shares of a reproduction run into scratch, which agree within a few points: e.g. EmpiricalCovariance eigh 58.7% kept vs 62.7% reproduced, MinCovDet eigh 57.4% vs 60.2%, GraphicalLassoCV graphical_lasso_path 86.4% vs 86.6%). Measured exponents (P 0.13-0.22, T 0.08-0.14) only reflect the fixed overhead between 8 and 32 channels.

**Avenues.**
- `cov_prec_simple-1` (all eight units, algorithmic, effort M; records: ~2x on basic.cov EmpiricalCovariance, OAS and ShrunkCovariance, ~1.7x on basic.cov.LedoitWolf, whose own Amdahl bound is 1.9x and whose figure with the 0.21 ms Cholesky inverse left is 4.52/2.56 = 1.77x, 1-2x on basic.prec EmpiricalCovariance, OAS and ShrunkCovariance and 1-1.7x on basic.prec.LedoitWolf. The prec lower bounds are 1x because Covariance runs before Precision in an EEG run and each prec unit reads the fit its basic.cov sibling cached, so the pinvh this avenue makes cheaper is paid, and credited, on the cov unit (README 'Optimisation columns', rule 4); the upper ends hold only where the prec unit fits the estimator itself. On the four basic.cov units the records list `cov_prec_simple-2` under `subsumed_by`, which applies outside tier 0 only, since that avenue has `eeg_applicable: false`): compute the precision with a Cholesky factorisation instead of pinvh's eigh, by fitting with `store_precision=False` and inverting in `basic.py:Estimators._from_cache`. A Cholesky inverse of the 118x118 matrix took 0.21 ms in the scratch timing, against 2.17 ms for pinvh. The Amdahl bound is per unit: the 2.17 ms pinvh over each unit's own measured 118x1024 wall gives 2.3-2.5x for EmpiricalCovariance, OAS and ShrunkCovariance (pinvh ~55-60% of 3.6-3.9 ms) but 1.9x for LedoitWolf, whose extra shrinkage pass makes it 4.7 ms (pinvh ~46%); the 58.7% eigh share is from the EmpiricalCovariance profile. Expected risk differs by kind. The four basic.cov.* units read only `covariance_`, which sklearn `_set_covariance` assigns (line 210) before and independently of the precision step (line 213), so their output cannot change: unverified-bit-identical. The four basic.prec.* units are unverified-within-tolerance on full-rank input (a different factorisation, ~1e-15 relative); on a rank-deficient covariance pinvh returns a pseudo-inverse and Cholesky fails, so a pinvh fallback is needed there. The inverse must be taken in `_from_cache` at fit time, before any identifier's in-place `np.fill_diagonal` (basic.py:49-50). Not measured.
- `cov_prec_simple-2` (cov.* units only, cache-sharing, effort S; records: ~2x on EmpiricalCovariance, OAS and ShrunkCovariance and ~1.9x on LedoitWolf, `eeg_applicable: false`, so its rows rank outside tier 0 on the configured identifiers: the Precision unit of the same estimator reads the cached fit and the EEG selection keeps both, so in an EEG run the pinvh moves to the prec identifier instead of disappearing (README 'Optimisation columns', rule 2)): make the precision lazy, so cov and cov-sq do not pay for a precision they never read. The precision must be computed from a copy of `covariance_` taken at fit time: `Estimators.multivariate` (basic.py:47-50) runs `np.fill_diagonal(matrix, np.nan)` on the cached estimator's own `covariance_`, so after a cov identifier has run (the config order and the EEG selection) the cached matrix has a NaN diagonal, and inverting it lazily would make the prec/prec-sq output fail or come out NaN (semantic). The 'Cross-cutting candidates' section above records this in-place aliasing. With the fit-time copy the lazy pinvh is the same call on the same bits, so the expected class is unverified-bit-identical. The per-unit Amdahl bound is the same as for `cov_prec_simple-1` (1.9-2.4x). There is no gain when the prec sibling is also computed, and the EEG selection keeps both. Not measured.

**Refuted or minor.** Bypassing sklearn input validation is a large share at 8x1024 (31-36%) but 10.6% at 118x1024, worth well under 1 ms per window, so it is not listed. Sharing one empirical covariance across EmpiricalCovariance, OAS and ShrunkCovariance would save np.cov (15.4%, ~0.5 ms) per estimator at most. That is also left out.

**Shared versus incremental cost and EEG scaling.** One fit per estimator name is cached on the Data object and serves cov, cov-sq, prec and prec-sq (each later identifier costs ~0.1 ms). The family-warm run at 118x1024 takes 0.016 s for all 16 identifiers. The EEG selection keeps all 16, so the per-window EEG cost at 118 channels is 4 fits (3.4-4.4 ms each) plus 12 cache reads, about 16 ms per window or about 10 s per 600-window patient. There are no band variants. These units are negligible next to the rest of the selection; the avenues above are listed for completeness.
