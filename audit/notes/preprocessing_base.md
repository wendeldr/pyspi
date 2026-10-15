# preprocessing_base: define, trace and check

Family: `preprocessing_base`. Units: `preproc.detrend`, `preproc.zscore`, `preproc.two_array_api`, `base.pair_handling`.
Records: `audit/records/<unit_id>.yaml` (status `checked`). Environment: scipy 1.17.1, numpy 2.5.4, project venv.
Probe scripts (read-only, outputs quoted below) are in
`/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/preproc_base/` (`p1_data.py` to `p7_sanity.py`).
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`, for example `preproc_base/p1_data.py`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.
The regression goldens are snapshots of current output and are not used as a reference anywhere here.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records.
Statements marked **candidate** or **hypothesis** in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.

All four units are shared infrastructure, so every SPI computed through `Calculator` passes through `preproc.zscore` (default on), optionally `preproc.detrend`, and either `base.pair_handling` or a class-specific `multivariate` override.

---

## preproc.detrend

**Documented.** The supplement has no detrending rule. §1 (line 95, p.3) only says "we standardize (z-score) the MTS along the time axis". The GitBook Calculator API page (`api-reference/pyspi.calculator.calculator.md:19`) says: "Detrend the dataset along the time axis before normalising (if enabled), default=False." The Data API page (`pyspi.data.data.md:29`) says "default=True", but its own signature (line 31) and the code (`pyspi/data.py:62`) say False. The docs never say the detrend is linear.

**Canonical.** Least-squares removal of a first-order polynomial a + b·t (Bendat & Piersol 2010, *Random Data*, 4th ed., trend-removal section). The residual has zero mean and zero correlation with t.

**Implemented.** The order in `Data.set_data` is:

1. Reorder to (p, s, r) (`data.py:177`).
2. Truncate (`:179-182`).
3. Run `scipy.signal.detrend(data, axis=1)` (`:184-191`).
4. Z-score (`:193-197`).
5. NaN check (`:199-203`).

scipy (`_signaltools.py:4298-4304`) fits the design `[arange(1,N+1)/N, 1]` with `linalg.lstsq` for each column, in a single segment, and subtracts the fit. `Calculator` forwards `detrend` only when it builds the Data itself (`calculator.py:269-277`). float32 input stays float32 and integer input is promoted to float64. The probe matches a `numpy.polyfit` degree-1 residual to 5.3e-15. For (p,s,r) input it detrends each replication separately.

**Deviation analysis.**
- **Residual:** none from canonical for non-degenerate channels.
- **Order:** the historical stores used z-score then detrend (already documented in `EEG_OPTIMIZATION_NOTES.md` 42-43). The two orders differ exactly by a per-row constant c = std(x)/std(resid). The probe gives per-row ratios of 1.137, 1.389 and 3.691, equal to the predicted c. So switching the order changes only scale-sensitive SPIs (cov, pdist, dtw, lmfit, ...) and leaves scale-invariant ones unchanged. The GitBook finance tutorial still shows the historical order.
- **Candidate: flat or exactly linear channel with detrend=True.** The floating-point residual (std 1e-19 to 2e-13) passes scipy's zero-variance guard `std <= |eps*mean|` (`_stats_py.py:2932`), because after detrending the mean is about 1e-17. The channel is then z-scored into a unit-variance artefact with a few levels and strong autocorrelation:
  - `T= 1024 const 3.0 -> OK std=1.000 uniq=9 lag1ac=+1.00`
  - `T= 4096 linear .37t+1.1 -> OK std=1.000 uniq=13`

  The behaviour depends on the data. `const 0.0` always raises, and `const 3.0` raises at T=64. Without detrend, a constant channel raises. This matters for flat EEG channels in the detrend=True workflow.
- **Candidate: detrend errors are swallowed.** `data.py:186-189` only prints. With `normalise=False` and an `inf` sample, the data is accepted untrended and the inf stays in (`p6_inf.py`). The default `normalise=True` path still raises later.
- **Candidate: documentation.** The Data API page gives the wrong default.

**Proposed checks.**
- **(kernel, independent-impl, tolerance):** compare with a from-scratch closed-form OLS residual on `trending`, `scaled_offset` and `iid_gaussian`. Expect ≤1e-12 relative difference.
- **(pipeline-detrend, analytic, tolerance):** `trending` and `bivariate_gaussian` should give identical output after detrending. The probe shows 1.8e-15 at T=64 and 3.1e-14 at T=1024, compared with an O(1) difference without detrending. Extend this to cheap deterministic SPIs.
- **(pipeline-detrend, analytic, exact):** on `trending`, check that the historical and current orders differ by the per-row constant c. Scale-invariant SPIs should agree and scale-sensitive ones should differ.
- **(qualitative):** replace a row of `iid_gaussian` with a constant or an exact ramp, at every T, to confirm or refute the flat-channel candidate.

## preproc.zscore

**Documented.** Supplement §1 (line 95), p.3: "Prior to evaluating each statistic, we standardize (z-score) the MTS along the time axis." Supplement §1.1.1 (line 136-139), p.3: "Due to the z-scoring of the time series, the correlation and covariance matrices are equivalent". The ddof is not stated anywhere.

**Canonical.** The standard score z = (x − x̄)/s. Sources differ on whether s is the population or the sample standard deviation. With the sample SD, |z| ≤ (T−1)/√T (Shiffler 1988).

**Implemented.** `zscore(data, axis=1, nan_policy='omit', ddof=1)` (`data.py:195`), which calls `zmap(a,a)` (`_stats_py.py:2760, 2913-2936`). It subtracts the mean, divides by the ddof=1 SD, and sets rows to NaN when std ≤ |eps·mean|. pyspi then raises on any NaN (`data.py:199-203`). The probe matches (x−mean)/std(ddof=1) to 4.4e-16.

**Deviation analysis.**
- **Non-degenerate rows:** none from canonical. ddof=1 is a convention choice.
- **Candidate: the supplement's "covariance = correlation" claim is off by (T−1)/T.** With ddof=1 z-scores, sklearn's 1/T MLE gives cov = (T−1)/T · r. The probe gives 0.984375 at T=64 and 0.99902344 at T=1024, and the diagonal is (T−1)/T. This crosses into another family and should be confirmed in `cov_prec_simple`.
- **Candidate: misleading error message.** A constant channel raises "contains non-numerics (NaNs)" although the input has no NaN.
- **Candidate: dead option.** `nan_policy='omit'` has no effect because any NaN is rejected afterwards.

**Proposed checks.**
- **(kernel, independent-impl, tolerance):** compare with a numpy z-score on `scaled_offset`, `iid_gaussian` and `contaminated`. Also check the max-|z| bound.
- **(pipeline, analytic, tolerance):** `scaled_offset` and `bivariate_gaussian` should give identical results, through Data and through every deterministic SPI. The probe shows 8.9e-16 for Data.
- **(pipeline, analytic, exact):** check cov_EmpiricalCovariance/corrcoef = (T−1)/T.
- **(qualitative):** check the error message for a zero-variance channel.

## preproc.two_array_api

**Documented / canonical.** pyspi's own contract (supplement §1, p.3) is that every process is z-scored before any statistic. Following it, `spi.bivariate(x, y)` should equal `spi.multivariate(Data(vstack([x,y])))[0,1]`. No external literature applies.

**Implemented.** `parse_bivariate` (`base.py:27-49`) builds `Data()` and calls `add_process` twice.
- The first call goes through `set_data` (`data.py:237-238`), so x is z-scored.
- The second call appends y raw with `np.append` (`data.py:231-234`): no z-score, and no detrend at all on this path.

`parse_multivariate` (`base.py:51-68`) does the same for a list. Only the first element is z-scored. The probe input was 4x+10 and 3y−5. The resulting row means were [0, −4.71] and the ddof=1 SDs were [1, 2.95].

**Deviation (candidate, partly documented).** `tests/unit/test_data.py:74-88` already locks the raw append in `Data.add_process`. Its effect on SPI values is not documented:

| Example | Data path | Direct-array path | Z-scored inputs |
|---|---|---|---|
| dtw | 9.5958 | `bivariate(xs, ys)` = 58.4698 | 9.5958 |
| cov | 0.607 | `multivariate([xs, ys])` = 1.789 | — |

`spearmanr` and, on this input, `xcorr_max_sig-False` were unaffected. `Calculator` is not affected.

**Candidate: unset i/j.** `base.py:45` builds `Warning('i and j not set.')` but never raises or emits it. With 3 processes and no i/j, the probe emitted 0 warnings. dtw failed with a numba TypingError, xcorr and spearmanr failed with IndexError, and a toy SPI received (None, None). It has not been tested whether any SPI silently returns a number in this case (hypothesis).

**Minor.** `parse_univariate` rejects a raw 1-D array with a RuntimeError about `dim_order`.

**Proposed checks.**
- **(kernel, independent-impl, exact):** capture the array the SPI sees. The candidate predicts row 0 is z-scored and row 1 is bit-identical to the input.
- **(pipeline, same-lineage, qualitative):** for the 141 identifiers that use the generic loops, compare `bivariate(x_raw, y_raw)` with the Data path on `scaled_offset` rows (2,3). Also confirm that z-scored inputs restore agreement.
- **(qualitative):** for every bivariate-capable SPI with no i/j on a 3-process `iid_gaussian`, record whether a warning is emitted and whether it raises or returns a value.

## base.pair_handling

**Documented.**
- Supplement §3.1 (line 1445-1446), p.29: "undirected statistics have repeated entries in the upper and lower diagonals of the MPIs".
- GitBook glossary (`spis/glossary-of-terms.md:27`): rows are sources and columns are targets (`table['di_gaussian'][0,1]` is 0→1).
- Developer guide (`development/development/incorporating-new-spis.md:376-378`): Undirected computes "only entries below the diagonal (lower triangle)" and copies "the lower triangular values to the upper triangle".

**Canonical.** For a directed SPI, A[i,j] = s(x_i→x_j). For an undirected SPI, A[i,j] = A[j,i] = s(x_i,x_j), which needs one evaluation per unordered pair.

**Implemented.**
- `Directed.multivariate` (`base.py:84-94`) loops j (outer) and i≠j (inner), setting A[i,j] = bivariate(i=i, j=j). That is P(P−1) calls.
- pyspi bivariates read `src, targ = z[[i, j]]`, e.g. TransferEntropy at `infotheory.py:383`, whose docstring says "from i->j". So A[i,j] = i→j.
- `Undirected.multivariate` (`base.py:132-138`) runs the same full loop and then sets `A[tril] = A.T[tril]`. It keeps the **upper** triangle (i<j) and discards the P(P−1)/2 lower-triangle evaluations. These are partly computed before the kept ones, because j is the outer loop.
- Three mirror classes cache each pair under both (i,j) and (j,i): CrossCorrelation (`basic.py:109-113`), Cointegration (`misc.py:62-97`) and JointEntropy (`infotheory.py:160-177`). For them the kept call bivariate(i<j) is a cache hit that returns the first-visited bivariate(j,i) value. Probe `p10_cache_order.py` showed that loop A[0,1] equals an isolated bivariate(1,0) exactly for xcorr_max_sig-True, coint_aeg and je_kozachenko.
- `Calculator.compute` (`calculator.py:329-339`) sets the diagonal to NaN. Any exception blanks the whole SPI.

The probe used a toy SPI that returns 10i+j:
- Undirected gives `[[nan 1 2 3] [1 nan 12 13] [2 12 nan 23] [3 13 23 nan]]` with 12 calls for P=4, starting (1,0), (2,0), (3,0).
- The 285 identifiers split into 50 Undirected-mirror, 108 Undirected with an override (bypass, REVIEW M1; tests/README.md bug 5 before b2f6d7e), 91 Directed loop, and 36 Directed with their own `multivariate`.
- Direction convention, confirmed for one SPI only: `gc_gaussian_k-1_kt-1_l-1_lt-1` on `delayed_noise` rows 0-1 gives A[0,1]=2.259 and A[1,0]=0.003, with x0 leading. For every other directed SPI, row = source is a reading of the code only. The check stage must not treat it as established; each directed family has to confirm it for its own SPI.

**Deviation analysis.**
- **Double work (already documented; quantification only).** `EEG_OPTIMIZATION_NOTES.md:132` already notes that the generic Undirected path computes both ordered directions before one triangle is overwritten and lists this as an optimisation candidate. The new part here is the count: P(P−1)/2 bivariate evaluations are discarded for the 50 Undirected-mirror identifiers. For xcorr and coint, probe `p8_rng_cache.py` counted 6 underlying `scipy.signal.correlate` / `stattools.coint` calls for the 12 ordered pairs at P=4, so the expensive step already runs once per unordered pair (je caches the same way). The cost of the cache-hit calls was not timed. Whether removing them saves time for these classes remains a hypothesis for the optimisation stage.
- **Candidate: the developer guide describes the wrong triangle.** It says the lower triangle is computed and copied up. The code computes everything and copies the upper triangle down.
- **RNG-stream coupling: not found (probe), replaced by a cache-order candidate.** Probe `p9_mirror_rng.py` ran all 50 Undirected-mirror identifiers at P=3, T=100. None advanced the global numpy RNG, none changed between `np.random.seed(0)` and `seed(12345)`, and all reran identically. For `mi_kraskov_NN-4` (JIDT KSG noise, NOISE_SEED=42), an isolated bivariate(0,1) equalled the loop A[0,1]. Other P, T and non-mirror SPIs were not tested.
- **Candidate: call order matters through the cache.** For 17 of the 50 identifiers (6 xcorr, 10 coint, je_kozachenko), an isolated bivariate(0,1) differs from the loop A[0,1], because the cache returns the first-visited (j,i) evaluation. Examples: xcorr_max_sig-True 0.2013 vs 0.1266; coint_aeg_tstat_trend-c -9.3894 vs -9.0604; je_kozachenko differs by 8.5e-8 (`p10_cache_order.py`). Reordering the loop or skipping the j>i calls would change these values. The xcorr case is already in the `basic.xcorr.*` records; the coint and je cases are not.
- **Candidate: label/base mismatch.** Barycenter (16 identifiers) is labelled "undirected" but inherits Directed. Its output is symmetric only through its (i,j)/(j,i) cache. The stored value comes from the i>j ordering, the opposite of Undirected's i<j.
- The bypass of the mirror is already documented (REVIEW M1). Only the counts are new here.

**Proposed checks.**
- **(kernel, analytic, exact):** toy index-encoding SPIs on `iid_gaussian`, checking the values and the call counts.
- **(pipeline, independent-impl, direction):** establish the meaning of A[i,j] empirically on `var2_bivariate`, `delayed_noise` and `logistic_maps`, using a from-scratch numpy OLS Granger log-variance ratio against `gc_gaussian`. Note that the MANIFEST's VAR matrix uses the row = target convention, so the expected pyspi result is A[0,1] > A[1,0] for var2. For logistic_maps (y = process 1 drives x) the expectation is A[1,0] > A[0,1].
- **(pipeline, analytic, exact):** measure max|A−Aᵀ| for each identifier on `iid_gaussian` and `bivariate_gaussian`, grouped by `multivariate` owner.
- **(pipeline, analytic, exact):** count calls and compare values computed from upper-triangle calls only. This quantifies the halving opportunity. The expected result from p9 is an exact match for 33 identifiers and a mismatch for the 17 cached-ordering ones; this has to be confirmed on `iid_gaussian` at the audit P and T.

---

## Open questions

1. **Which bug 5?** At trace time KNOWN_ISSUES.md listed "bug 5" as "Undirected spectral SPIs not symmetrised (MRO bypass)", but `tests/README.md` item 5 now describes PLI signed antisymmetry. Which reference should records cite for the mirror bypass? The trace stage cited REVIEW M1 and KNOWN_ISSUES bug 5 together (answered below).
2. **Flat channels with detrend=True.** Should these be rejected, flagged, or left as they are? The EEG pipeline uses detrend=True, and the outcome currently depends on T and the constant's value. Do saved windows with flat or saturated channels exist?
3. **Covariance vs correlation.** Is the (T−1)/T factor between cov_EmpiricalCovariance and Pearson r (ddof=1 z-score with the 1/T MLE) intended? The supplement claims they are equivalent. This should be decided together with the cov_prec_simple family.
4. **Barycenter label.** Should Barycenter's "undirected" label be kept even though the class is Directed? And which ordering of (i,j) should define its value for the DBA/soft-DTW modes, where the barycenter may depend on input order (hypothesis)?
5. **Removing the double work.** Probe p9 found no RNG-stream coupling among the 50 Undirected-mirror identifiers. But for xcorr, coint and je_kozachenko the stored value is the cached (j,i) evaluation. If the loop visits only i<j, these 17 values change. Is that acceptable, and which ordering should define them?
6. **Unset i/j.** Does any bivariate-capable SPI silently return a finite value when i=j=None? This is untested beyond dtw, xcorr, spearmanr and a toy SPI.

---

## Checks

Manifest `audit/checks/manifest/preprocessing_base.json` (179 runs, regenerated by the scratch script `preprocessing_base_check/gen_manifest.py`). The pyspi side is `audit/checks/probes/preprocessing_base_probes.py`, because every unit here is infrastructure. The references are in `audit/checks/refs/preprocessing_base.py`. Results are in `audit/checks/out/results/preprocessing_base.json`: **131 pass, 48 fail, 0 without reference** (after the review fixes: 6 flat-channel runs at P=3 and P=2 added, all failing; mod-infotheory now passes under the revised unset-i/j criterion, see preproc.two_array_api). `run_checks.py` exits 1 whenever any comparison fails, which here is by design; all three steps reported 0 errors.

Almost every failure is deliberate. For each candidate, the reference encodes the documented or canonical expectation, so a fail means pyspi deviates from it. The remaining failures are already-documented behaviour (antisymmetric spectral classes, RNG-dependent robust covariance and sgddtw) or a direction convention other than the glossary's. Each failure is listed below.

Trace-stage corrections made in this stage:
- Every `calculator.py` citation now points to HEAD a1f20c7: `__init__` :48-56, `load_dataset` :269-277, `compute` :295-358 with `multivariate` at :329, diagonal at :332, exception at :336-339. This applies to the records and to this note (§base.pair_handling).
- `base.pair_handling.already_documented` now cites KNOWN_ISSUES item 5 as it currently reads. That item records both the old wording of tests/README bug 5 (the mirror bypass, still REVIEW M1) and its current wording (PLI antisymmetry). It also adds tests/README.md:134 and :145-146 for the RNG-order effects.
- The trace count of visiting-order-dependent identifiers changes from 17 to 18 bitwise. Only 10 of these differ beyond rounding (see below).
- The trace statement "without detrend, a constant channel raises" holds only for some constants (see preproc.detrend).
- Adversarial-review corrections: the flat-channel table now covers P=5, 3 and 2, because the outcome depends on the other rows; the unset-i/j NaN returns were relabelled from "silent" to "reported only through the logging module"; preproc.detrend moved from score 10 to 9; the narrowed scope of three proposed checks is now stated as not done; the zscore trace proposal no longer uses the goldens' tolerance classes as a criterion (rtol 1e-9 instead).

### preproc.detrend — matches, score 9 (independent-impl; rubric 9: trivial convention)

**Comparison.** `Data(x, normalise=False, detrend=True)` was compared with the closed-form OLS residual r = x − x̄ − b(t − t̄) in numpy.
- All 12 detonly runs pass (`preproc.detrend|{trending,scaled_offset,iid_gaussian}|{64,256,1024,4096}|kernel|detonly`), with atol = 1e-12·max|x|. The largest absolute difference is 8.53e-14, on trending at T=4096. For scaled_offset and iid_gaussian it is ≤3.6e-15.

**Full order.** The order truncate → detrend → z-score (ddof=1) was checked against `zs(ols_detrend(x[:P-1, :T-7]))`.
- All 6 trunc runs and all 6 order-full runs pass at exact tolerance 1e-12. The largest absolute difference is 1.39e-13 (`trending|4096|pipeline-detrend|trunc`).
- So truncation does happen before detrending.

**Invariance.** Data(trending, detrend=True) equals the numpy-preprocessed bivariate_gaussian draw. The 3 inv-data runs agree to 8.88e-14. The 4 inv-SPI runs (cov_EmpiricalCovariance, spearmanr, mi_gaussian, dtw against independent kernels) agree to a relative difference of 1.91e-12.
- Control: without detrending, cov and spearmanr differ from the detrended draw by 1.59 and 1.61 (`nodetrend-*`).

**Historical order.** On trending, zscore(detrend(x)) = c·detrend(zscore(x)) with c = std(x)/std(resid).
- The pyspi/historical ratio is constant per row. Its spread is ≤2.8e-10 at T=1024, and it equals c = [1.0018, 14.67, 6.07, 29.88].
- `order-hist-spis`: cov_EmpiricalCovariance differs between the two orders by up to 0.792 (T=1024). The difference is exactly the factor c_i c_j (relative deviation 1.6e-14). spearmanr is identical (difference 0.0).
- So the historical stores differ from new calculations only in scale-sensitive SPIs, and the size of the difference is known exactly.

**Candidates.**
- **Confirmed: a flat or linear channel passes when detrend=True, and whether it passes depends on floating-point rounding.** All 10 flat runs fail the rejection expectation: 4 `flat` runs with all P=5 rows of iid_gaussian, and 3 `flat-P3` plus 3 `flat-P2` runs that use the first 3 or 2 rows of the same draw at T = 64, 256, 1024.
  - P=5: at T=64, const 3.0, const 1e-3 and the ramp are accepted. At T=256, const 3.0 and the ramp are accepted and const 1e-3 is rejected. At T=1024 and 4096 all three are accepted.
  - P=3 and P=2: at T=64, const 1e-3 and the ramp are accepted and const 3.0 is rejected. At T=256 and 1024 all three are accepted.
  - So the same replacement at the same T can be accepted at one P and rejected at another (const 3.0 at T=64; const 1e-3 at T=256). scipy.signal.detrend solves all rows in one multi-right-hand-side lstsq (`scipy/signal/_signaltools.py:4303`), so the rounding residual of the flat row depends on the other rows. The reviewer's probe `preprocessing_base_check/fable_probe2.py` shows a residual std of exactly 0.0 at T=64, P=3, const 3.0 (rejected) against 3.06e-16 at P=5 (accepted). The per-T pattern above belongs to these arrays; it is not a rule about T or the constant.
  - Only const 0.0 (always rejected) and the ramp (always accepted) behave the same in every run.
  - Each accepted channel is stored with std 1.0, only 2–26 distinct values, and lag-1 autocorrelation of 0.62–0.999.
  - Additional observation: without detrending, a constant 1e-3 channel is also accepted at T ≥ 256 (`raised_without_detrend = [1,1,0,0]`, the same at every P because that path z-scores each row on its own). The scratch probe `p_const_nodetrend.py` shows it is stored as a constant −0.998 row with zero variance. scipy's zero-variance test is a floating-point comparison, so the outcome depends on the constant's value and on T.
- **Confirmed: detrend errors are swallowed.** In the `swallow` run, an inf with normalise=False is accepted. The inf stays in the data and the added 0.05·t trend is kept (stored slope 0.0493). With normalise=True the input is still rejected, so the default path is safe.
- **Confirmed: documentation default.** The GitBook Data API page (`pyspi.data.data.md:29`) says detrend defaults to True. The code says False. This comes from reading the docs, not from a numeric run.

**Score.** 9, not 10. The residual is fp-exact against the independent OLS reference. But the documentation states neither the detrend type (linear is scipy's default) nor a consistent default, and the flat-channel and swallowed-error paths deviate from the canonical expectation of rejection. preproc.zscore is scored 9 for its undocumented ddof convention, and the same anchor applies here. The score was lowered from 10 after the adversarial review.

### preproc.zscore — matches, score 9 (independent-impl; rubric 9: ddof convention)

**Comparison.** Data(x) was compared with numpy (x − mean)/std(ddof=1).
- All 12 zs runs pass at exact tolerance 1e-12, with a largest absolute difference of 4.4e-16.
- Row means are ≤1e-16. The ddof=1 variance differs from 1 by ≤4.4e-16. The largest |z| is 2.3–5.0, well inside the Shiffler bound (T−1)/√T.
- Invariance: scaled_offset ≡ bivariate_gaussian holds to 1.33e-15 for the array and to a relative difference of 1.14e-12 for 4 SPIs.

**Candidates.**
- **Confirmed: covariance equals (T−1)/T · r, not r.** The cov-ratio runs pass against (T−1)/T · corrcoef (relative difference 5.5e-15). The ratio is 0.984375, 0.99902344 and 0.99975586 at T = 64, 1024 and 4096. The cov-supp-claim runs fail against the supplement's "equivalent" claim, with a largest relative difference of exactly 1/T. The cov_prec_simple family found the same factor for the SPI.
- **Confirmed: misleading error message.** A constant row raises "contains non-numerics (NaNs) in processes: [1]" (`errmsg`).
- **Confirmed: `nan_policy='omit'` has no effect.** One NaN sample is rejected, although scipy's omit would leave the other rows finite (`nanpolicy`).

**Not done.** The trace stage proposed the invariance check for every deterministic SPI identifier computed through `Calculator(normalise=True)`. It was run for 4: cov_EmpiricalCovariance, spearmanr, mi_gaussian and dtw (`inv-*`). For any SPI that reads only the stored array, invariance follows from the inv-data runs (the stored arrays agree to 1.33e-15), but the other identifiers' values were not compared.

### preproc.two_array_api — wrong, score 3 (independent-impl; README anchor 3 by analogy: "the label does not describe the computation", with the documented z-scoring contract as the label)

**Score.** No README anchor is written for infrastructure units. Anchor 3 reads "An identifier hyperparameter is ignored or the label does not describe the computation". Here the documented contract (supplement.txt:101-102, "Prior to evaluating each statistic, we standardize (z-score) the MTS") plays the role of the label, and the direct-array path does not honour it for any array after the first. It also makes argument order matter: dtw gives 159.48 or 68.86 depending on which array comes first. Anchors 2, 1 and 0 do not fit: no sign changes against a canonical quantity, no estimator is substituted and no output is degenerate. Each SPI computes its own formula on wrongly scaled input.

**Captured input.** A toy SPI captured the array it receives from `bivariate(x, y)`.
- Row 0 is z-scored. Row 1 is bit-identical to the raw y: mean 4.68 and sd 9.90 at T=1024.
- `cap-mechanism` passes with largest absolute difference 0. `cap-contract` fails with a largest absolute difference of 35.0.
- List input to `multivariate([…])` behaves the same way (`list-mechanism` passes, `list-contract` fails).

**Effect on SPI values.** Each SPI was compared with a kernel run on z-scored inputs, in both orientations:

| SPI | Direct raw path | Reference | Data path deviation | Direct z-scored path deviation |
|---|---|---|---|---|
| dtw (Undirected) | 159.48 / 68.86 | 8.714 | 0.0 | 0.0 |
| lmfit_Ridge (Directed) | 31.71 / 7.72 | 0.3245 | 0.0 | 0.0 |
| cov_EmpiricalCovariance (`multivariate([x, y])`) | 8.085 | 0.818 | ≤2.2e-16 | ≤2.2e-16 |

The scale-invariant controls, spearmanr and gc_gaussian, agree on all three paths (deviation ≤9.6e-16).

**Candidates.**
- **Confirmed: unset i/j is never reported as such.** Calls were made for all 285 config identifiers on 3 processes.
  - 0 Python warnings were emitted. `pyspi/base.py:45` builds a `Warning` object and discards it.
  - cds and igci return finite numbers (0.0132 and 0.0) with no exception, Python warning or logging record (`mod-causal` fails). With i = j = None, `z[None]` selects the whole array.
  - 10 infotheory identifiers return NaN without an exception or Python warning: mi_gaussian, mi_kraskov_NN-4, mi_kernel_W-0.25, the three tlmi identifiers and the four phi identifiers. Each emits one logging record (`n_logging_records` = 10 on `mod-infotheory`). For mi_* it is WARNING "MI calcs failed. Maybe check input data for Cholesky factorisation?", for tlmi_* the time-lagged equivalent, and for phi_* ERROR "Integrated information computation failed: only integers, slices …" (`pyspi/statistics/infotheory.py:281`, `:315`, `:571`). So these failures are reported through the logging module, although the message names a different cause. An earlier version of this note called them silent because the probe counted only Python warnings. The probe now also captures logging records, and the criterion is "raise, Python warning or logging record", which `mod-infotheory` passes.
  - The other 273 raise, 144 of them with NotImplementedError.
- **Confirmed: a raw 1-D array is rejected.** Passing one to a parse_univariate method raises RuntimeError (`univariate-1d`).

All four candidates are confirmed, and none is in KNOWN_ISSUES. `tests/unit/test_data.py:74-88` locks only the raw append inside `Data.add_process`.

**Not done.** The trace stage proposed the same-lineage comparison of the Data path with the direct path for all 141 identifiers whose class uses the generic loops. It was run for 5: dtw, lmfit_Ridge, cov_EmpiricalCovariance, spearmanr and gc_gaussian. For the other 136, the statement that every later array reaches the SPI raw rests on the captured-array runs (`cap-*`, `list-*`), because every SPI uses the same parse decorators. Their SPI values were not compared, so which of them change value, and by how much, was not measured.

### base.pair_handling — matches, score 9 (analytic; rubric 9: the convention differs from the developer guide)

**Toy SPI and call order.** A toy SPI was checked analytically (`toy`): all values match exactly. For P=5 there are 20 calls, of which 10 are overwritten, and the first calls are (1,0), (2,0), (3,0), (4,0).
- With a real SPI, `calls-spearmanr` makes the same 20 calls, and its values match the kernel to 1.4e-17.
- `calls-gc_gaussian` on var2 matches the kernel independently: pyspi A[0,1] = 0.148325 equals TE(0→1), and A[1,0] = 0.012653 equals TE(1→0), to a relative difference of 3e-14.

**Symmetry (iid_gaussian, T=256, all config identifiers of every Undirected class).**
- The 15 mirror classes and Barycenter are bit-exactly symmetric (`sym-mirror-*`, `sym-directed-*`).
- Among the 16 bypass classes, 12 are symmetric. Precision (9.7e-17) and IDS (2.8e-17) differ from symmetry only in the last bit (REVIEW M2).
- Four are antisymmetric: CoherencePhase 1.60, PLI 0.40, wPLI 0.39 and multitaper PSI 24.1. These are already documented: REVIEW M1, M3, M4 and M5, tests/README bug 5 (PLI), and KNOWN_ISSUES item 5. They are not new.
- Wavelet PSI comes out exactly symmetric. Its axis handling is already documented in EEG notes §review (see the wavelet_psi family).

**Visiting order (`upper-*`).** Values built from only the i<j calls were compared with the loop's values.
- 32 of the 50 mirror identifiers are bit-identical.
- These differ: xcorr 6/6 (up to 0.081), coint_aeg 3/3 (up to 7.41) and je_kozachenko (1.6e-7).
- The 8 coint_johansen identifiers differ by ≤2.6e-13, which is rounding from the column order.

**Variant order (`grp-*`, one shared Data, forward vs reversed).**
- These groups give identical results: xcorr, coint, spectral coherence, spectral phase-sync, barycenter, ccm and je/ce/cce gaussian.
- cov/prec is the exception. The EllipticEnvelope and MinCovDet identifiers differ by up to 0.0364 because they draw from the global RNG (already documented at tests/README.md:134 and :145).
- sgddtw shifts by 1.8e-3 after one extra RNG draw (`rng-sgddtw`; already documented at tests/README.md:134 and :146).
- The exception expected for coint aeg does not occur when the order of the *variants* changes. coint aeg depends on the order in which *channels* are visited (`upper-misc.Cointegration`), not on variant order.

**Direction table (`dir-*`).** Each SPI was read against the glossary's "row = source, larger = stronger". Process 0 drives process 1 in every dataset used.

| SPI | var2_bivariate | delayed_noise (rows 0, 1) | Meaning of A[i,j] |
|---|---|---|---|
| gc_gaussian (TE) | 0.148 > 0.013 | 2.27 > 1.3e-5 | i → j (row = source) |
| psi_multitaper | +136 | +694 | > 0 when row leads |
| gd_multitaper | all-NaN (REVIEW M6) | +1.011 | delay in samples by which the row leads |
| dtf_multitaper | A[1,0] 0.269 > A[0,1] 0.143 | A[0,1] 0.817 > 0.171 | row = target on var2. The delayed_noise result is the innovation-variance artefact described in the spectral_directed note. |
| dcorrx_maxlag-1 | A[1,0] > A[0,1] | A[1,0] > A[0,1] | j → i (row = target) |
| xme_gaussian k10 / k1 | A[0,1] < A[1,0] | A[0,1] < A[1,0] | conditional entropy: a lower value means the row informs the column |
| ccm_E-1_mean (T=256) | +0.037 | +0.050 | On logistic_maps_chaotic, A[1,0] 0.218 > A[0,1] 0.182: high when the column drives the row (CCM convention). The var2 and delayed_noise gaps are small and have the other sign. |
| anm | A[0,1] > A[1,0] | A[1,0] > A[0,1] | No consistent orientation (cubic pair A[0,1] > A[1,0], tanh pair the reverse) |
| coint_aeg | symmetric (mirrored) | symmetric | no direction; the isolated kernel is asymmetric (−24.85 vs −25.18) |
| lmfit_Ridge | difference 2e-16 | difference 1e-16 | no direction (contemporaneous) |

10 of 25 direction runs pass. Only TE, PSI and group delay follow the glossary on every dataset where they give a value. In the first reference version, lmfit's rounding-level difference passed on its sign. That was a reference defect: differences ≤1e-12 relative are now recorded as "no resolvable direction" (qualitative fail).

**Candidates.**
- **Double work.** Confirmed as a quantification only, because EEG_OPTIMIZATION_NOTES.md:132 already documents it. This is not a new finding.
- **Developer guide describes the wrong triangle.** Confirmed by the toy run and `incorporating-new-spis.md:376-378`.
- **Cache visiting-order coupling.** Confirmed (above). The xcorr part is in the basic.xcorr.* records. The coint and je parts are not in KNOWN_ISSUES.
- **Barycenter labelled "undirected" but built on Directed.** Confirmed: it is symmetric only through the cache, at distance.py:211 and :215.

**Not done.** Two proposed checks were narrowed:
- Proposed check 2 asked for a from-scratch OLS Granger reference on var2_bivariate, delayed_noise and logistic_maps. It ran as the kernel Gaussian-TE value comparison on var2_bivariate and iid_gaussian only (`calls-gc_gaussian`). delayed_noise has a direction-only run (`dir-gc_gaussian_*`, no kernel value comparison). gc_gaussian was not run on logistic_maps; ccm on logistic_maps_chaotic was used for the driven-map direction instead.
- Proposed check 4 asked for call counts for every Directed-loop and Undirected-mirror identifier. Counts were taken for spearmanr, gc_gaussian and the toy SPI. The 50 mirror identifiers were covered by the upper-only value runs (`upper-*`), not by call counts.

## Open questions: answered

1. **Which bug 5?** KNOWN_ISSUES.md item 5 has since been rewritten. It now records the current tests/README wording (PLI signed antisymmetry, README:206) and notes that the earlier wording, the MRO bypass of `Undirected.multivariate`, is still documented as REVIEW M1. Records should cite **REVIEW M1** for the mirror bypass and "tests/README.md bug 5" only for PLI antisymmetry. The base.pair_handling record now does this.

2. **Flat channels with detrend=True.** These are not rejected consistently, and the outcome is decided by floating-point rounding.
   - With detrend: acceptance depends on T, on the constant and on the other rows of the array, because scipy solves all rows in one lstsq. Const 3.0 at T=64 is accepted with P=5 rows and rejected with P=3 or P=2. Const 1e-3 at T=256 is rejected with P=5 and accepted with P=3 or P=2. The ramp was accepted and constant 0.0 rejected in all 10 runs (`flat`, `flat-P3`, `flat-P2`).
   - Without detrend: constant 1e-3 is accepted at T ≥ 256 and rejected at T=64, independently of P.
   - Accepted channels become few-valued, unit-variance artefacts.
   - Whether to reject or flag them is a decision for the project. An explicit check such as ptp(x) == 0, or residual std ≤ k·eps·max|x| before z-scoring, would make the outcome independent of T and of the constant's value.
   - Whether saved EEG windows contain such channels was not checked. That needs a scan of the stored windows' raw ptp or std, which is outside this family's synthetic data.

3. **Covariance vs correlation.** The factor (T−1)/T is confirmed exactly (cov-ratio, cov-supp-claim), here and in cov_prec_simple. Nothing in the code or the supplement says it is intended. It follows from combining ddof=1 z-scoring with sklearn's 1/T MLE. The supplement's "equivalent" holds only to a relative error of 1/T. Whether to call it a convention or to correct it is a decision for the project. The scale is monotone and becomes negligible at EEG window lengths: 2.4e-4 at T=4096.

4. **Barycenter.** All 16 identifiers are exactly symmetric through the cache (`sym-directed-distance.Barycenter`). The stored value comes from the first-visited ordering, z[[b, a]] for a < b. The barycenters family note reports DBA as order-invariant, and finds that swapping inputs changes the sgddtw barycenter (by up to 0.177 pointwise). Which ordering should define the value is a labelling decision. Two options would make the matrix independent of visiting order: inherit Undirected, which keeps bivariate(a, b) with a < b, or order the pair explicitly inside `bivariate`. The label can stay "undirected" only if the class's output is made order-independent.

5. **Removing the double work.** For 32 of the 50 mirror identifiers, visiting only i < j gives bit-identical matrices (`upper-*`). The 8 Johansen identifiers change at rounding level only (≤2.6e-13). Ten would change materially: 6 xcorr (≤0.081), 3 coint_aeg (≤7.41) and je_kozachenko (1.6e-7).
   - No RNG coupling was found in the mirror group. The variant-order runs for xcorr, coint and je are identical, and the trace probe p9 found no global-RNG consumption.
   - The RNG-sensitive identifiers sit elsewhere: EllipticEnvelope and MinCovDet in cov/prec, which bypass the mirror, and sgddtw in Barycenter, a Directed class. Removing the double work does not touch them.
   - Whether the 10 changed values are acceptable depends on which orientation the project wants to define the stored value. The i < j orientation matches what an isolated `bivariate(i, j)` returns, so the change would remove the current visiting-order dependence.

6. **Unset i/j.** Yes, two do. cds and igci return finite numbers (0.0132 and 0.0) with no exception, Python warning or logging record (`mod-causal`). 10 infotheory identifiers return NaN without an exception or Python warning, but each emits a logging record (WARNING for mi_* and tlmi_*, ERROR for phi_*) whose message names a different cause (`mod-infotheory`, `n_logging_records` = 10). None of the 285 identifiers emits a Python warning, and none reports the unset i/j itself.

## Optimisation

Workflow 3 is a short search for where the time goes and which avenues look promising. Nothing in pyspi was changed. The infrastructure units in this family have no timing or profile of their own, so their numbers come from the consumer profiles and from two micro-benchmarks written by this group:
- `audit/profiles/micro/opp_xc-blas-threads.py` / `.json`: a BLAS thread sweep. It also times Data construction at 118x1024 and probes the JVM + OpenBLAS segfault in subprocesses.
- `audit/profiles/micro/opp_xc-jni-overhead.py` / `.json`: the JIDT per-call cost.

Both ran while five other audit groups were computing, so the walls are indicative only.

This group also owns the cross-cutting rows in `audit/profiles/crosscutting.yaml`, which are summarised at the end of this section.

### preproc.detrend and preproc.zscore

**No profile frame.** `profile_units.py` builds the `Data` before `prof.runcall` (`profile_units.py:712-718`), so `Data.set_data`, `scipy.signal.detrend` and `scipy.stats.zscore` never appear in a unit pstats. A scan of every `*.pstats` found zscore frames only inside the kraskov_DCE units, at 0.3-0.5% of their calls. Those frames are the per-pair re-normalisation of those estimators, not this preprocessing.

**Measured cost.** `opp_xc-blas-threads.json` (`data_construction_118x1024`) times one 118x1024 window's `Data(raw, dim_order='ps', normalise=True, detrend=True)` plus `to_numpy`, which covers truncation, detrend, z-score and the NaN check:
- 2.96 ms at 1 BLAS thread.
- 3.3 ms at 4 or 8 threads (no gain).
- That is about 0.0013% of the 229.1 s retained-selection window.
- Detrend and z-score were not timed separately.

**Complexity.** Both steps are O(P T). `detrend` solves one `lstsq` on the T x 2 design for all P rows (`pyspi/data.py:184-187`), and `zscore` follows at `data.py:195`. Preprocessing runs once per window. It does not scale with the number of SPIs or EEG band identifiers.

**`preprocessing_base-2` (shared by both units).**
- **Change:** replace scipy `detrend` + `zscore(nan_policy='omit')` with one numpy pass (closed-form OLS slope and intercept, then mean and ddof=1 std).
- **Expected speedup:** 1-3x on the 3 ms. The absolute saving is under 3 ms per window, so this is not a target. It is listed only to record that preprocessing was looked at.
- **Basis:** `complexity:` (both steps are already O(P T); only scipy's lstsq set-up and the `nan_policy` handling go away). `opp_xc-blas-threads.json` holds only the baseline timing of Data construction, not a candidate, so it is cited as the measured cost and not as a `microbench:` basis.
- **Risk:** `unverified-within-tolerance`. It is the same OLS detrend and ddof=1 z-score, re-associated, so ordinary channels change at rounding level. Degenerate-input caveat: flat or exactly linear channels are accepted or rejected on last-bit residuals (preproc.detrend record, probe `p2_flat.py`), so a re-associated detrend could flip that outcome for such channels.
- **Effort:** S.

**Refuted.** Preprocessing as an optimisation target (≤0.002% of a window).

### preproc.two_array_api

No opportunity. This path is `parse_bivariate`, which builds `Data()` and calls `add_process` twice (`pyspi/base.py:28-37`). It is O(T) per call and is not used by `Calculator` or the EEG workflow, so it costs 0 s per window. The change it does need (z-scoring the second array; verdict `wrong`) is a correctness fix, not an optimisation.

### base.pair_handling

**Loop overhead.** `Directed.multivariate` + `Undirected.multivariate` + `parse_bivariate` together take 0.0005-1.86% of the profiled call (median 0.065%) across the 84 loop-unit pstats at 8x1024. Examples: basic.spearmanr 0.48%, infotheory.mi.gaussian 1.02%, distance.gwtau 1.86%. A "pair-loop/wrapper overhead" avenue is therefore refuted (<10%).

**The discarded half.** In the 26 Undirected-mirror units without per-pair state, the bivariate wrapper `pyspi/base.py:28:parsed_function` accounts for 96.2-100% of the profiled SPI call at 8x1024. Half of those calls are overwritten at `base.py:136-137`. Amdahl gives 1/(1 − share/2): 1.93x for gwtau (96.2%), 1.97x for mi.gaussian (98.3%), 1.99x for spearmanr (99.4%) and kendalltau (99.3%), and 2.0x for the dcorr, hsic, mgc, dtw, lcss, softdtw, si, mi-kraskov/kernel and phi units (99.5-100%).
- Three of these units are in the retained EEG selection: basic.spearmanr (10.0 s per 118x1024 window, family-warm, measured), basic.kendalltau (9.03 s) and infotheory.mi.gaussian (3.26 s). Together they are 22.3 s of the 229.1 s window on their configured identifiers, where halving them would save about 11 s (~5%). On their retained identifiers they are 17.8 s, because the EEG selection keeps kendalltau-sq alone (basic.kendalltau 4.52 s instead of 9.03 s), so halving them would save about 8.9 s per window (~4% of the 221.1 s tier-0 window in the table below). The ranking credits that saving to the unit rewrites, not to this avenue: `xc-undirected-upper` is `subsumed_by` rank_corr-1, rank_corr-2 and it_gaussian-1 on these three units, so `family:undirected-upper/retained` ranks on the about 283 s per patient those unit avenues leave and saves none of it (see "How build_csv ranks these rows" below).
- The largest absolute saving outside the selection is in the distance units:
  - distance.mgc: 3305 s per window projected, extrapolated-pairs-from-8.
  - distance.softdtw: 1916 s family-warm, extrapolated-pairs-from-8.
  - distance.hsic: 1336-1416 s, extrapolated-pairs-from-32.

**Excluded units.**
- basic.xcorr.* (4 units), misc.coint.* (5) and infotheory.je.* (3) are excluded. For these, the first-visited (j,i) call fills the (i,j) cache. The underlying computation therefore already runs once per unordered pair: 28 `scipy.signal.correlate` / `statsmodels coint` / JIDT entropy calls for 56 bivariate calls at 8x1024, for example `scipy/signal/_signaltools.py:correlate` n=28, 72.1% of the xcorr_max_sig-False call.
- Visiting i<j only would also change their stored values: xcorr by up to 0.081, coint_aeg by up to 7.41, je_kozachenko by 1.6e-7, and coint_johansen by ≤2.6e-13 (check runs `base.pair_handling|iid_gaussian|256|pipeline|upper-*`).
- No mirror unit draws from the global RNG (probe p9 above).
- mi.kraskov_DCE fills `data.theiler` for all pairs on its first call, independent of visiting order.

**No unit opportunity.** The only avenue for this unit is the i<j-only loop in `Undirected.multivariate` (`pyspi/base.py:132-138`). It is recorded once, as `xc-undirected-upper` in `audit/profiles/crosscutting.yaml` (row `family:undirected-upper`), and is not repeated here under a second id, so that Workflow 4 does not count it twice. The base class has no window cost of its own (no summary.json entry), and its own loop overhead (above) is refuted as an avenue. The ~2x per scope unit assumes the dropped (j,i) call costs the same as the kept (i,j) call; the two directions were not timed separately. The bit-identical expectation rests on the check-stage `upper-*` runs (max_abs 0.0 for all 30 scope identifiers), which used one dataset (iid_gaussian, T=256) and are a check-stage diff, not a micro-benchmark.

### Shared versus incremental cost

Preprocessing is a once-per-window cost (~3 ms) shared by every SPI. The pair loop has no shared part. Its cost is the SPI's per-pair work times P(P−1): 13806 calls per SPI at P=118, of which 6903 are discarded in the mirror units.

### Cross-cutting rows (`audit/profiles/crosscutting.yaml`, owned by this group)

| Row | Opportunity | Headline evidence | Expected | Risk |
|---|---|---|---|---|
| `family:undirected-upper` | `xc-undirected-upper` | 26 units; bivariate wrapper 96.2-100% of each profiled call, half discarded; assumes (i,j) and (j,i) cost the same (not timed separately) | ~2x per unit | unverified-bit-identical (check runs max_abs 0.0) |
| `family:window-parallel` | `xc-window-parallel` | 229.1 s per window serial (38.2 h per patient), the configured identifiers of the 53 retained units; the tier-0 unit bases of README ranking rule 1 (kendalltau on its retained identifier, plv on its retained build, prec.EllipticEnvelope and xcorr.max_sig-False on their fresh projections) sum to 221.1 s, still without the untimed spectral bands. Peak RSS 3409 MB (spectral phase-sync family at 118x1024) gives 8 uniform workers in 31 GB (~8x). Split pools (6 spectral at ≤3.41 GB + 11 others at ≤0.64 GB) could give ~16x (not measured). Pair-level splitting is capped at 229.1/92.0 = 2.5x per window, because 92.0 s come from multivariate-override units | ~8x patient wall-clock (split pools ~16x is an unmeasured upper scenario; per-window cost unchanged, so not comparable with per-window unit speedups) | unverified-rng-stream (MinCovDet/EllipticEnvelope draw from the unreseeded global RNG, `pyspi/workflow/_run.py:344`) |
| `family:jni-overhead` | `xc-jni-overhead` | JNI floor 14.9 µs per call sequence (T=16); 0.21 s per ordered Gaussian SPI per window; 1.1 s per window over the retained Gaussian JIDT SPIs (~3% of their 34.6 s). At T=1024 the cost is Java-side `setObservations` work (~0.55 ms in a tight loop, 0.17-0.33 ms in situ), not crossing overhead | 1.01-1.07x | unverified-bit-identical |
| `family:blas-threads` | `xc-blas-threads` | gpfit_DotProduct pair (the only gpfit identifier timed; RBF was not) 1.94 s → 1.45 s (4 threads, 1.34x), 1.38 s (8 threads, 1.40x), max_abs_diff 7.8e-16; GraphicalLassoCV 118x1024 and Data construction no gain. JVM + OpenBLAS: `numpy.linalg.solve` 1500x1500 is killed by SIGSEGV after `jpype.startJVM` with 2 or 8 BLAS threads, completes with 1 thread or without the JVM; GEMM, batched 8x8 solve and eigh complete with 8 threads under the JVM (OpenBLAS 0.3.34, OpenJDK 21.0.12.1) | ~1.4x (gpfit only) | within-tolerance (micro diff) |
| `family:gpu-spectral` | `xc-gpu-spectral` | 10 retained spectral units; spectral.gd is outside the scope (its coherency share, 18.7% per band, is what `spectral_csm-shared-csm` removes, and the rest is per-pair Python callbacks the GPU does not move). At 118x1024 the spectral.plv call is 96.4% `_expectation_cross_spectral_matrix` (`fcn` 78.5%, `_complex_inner_product` 11.0%; `micro/prof_spectral.plv__118x1024.pstats`). The CSM is 1.14 GB complex128 and fits a 6 GB RTX 2060 in frequency blocks | 2-10x (projection; `speedup_kind: projection`) | unverified-within-tolerance |
| `family:gpu-distance` | `xc-gpu-distance` | dcorr `hyppo statistic` 99.1%; per-channel 1024² matrices total 0.99 GB; after `dcorr_hsic-1` the dcorr per-pair work is one GEMM-shaped reduction. hsic is not GEMM-shaped: `compute_kern` takes the RBF bandwidth from the x channel, so after `dcorr_hsic-3` (x-side cache) each pair still needs exp(−γ_x D²_y) over T² entries, its centring and two inner products (batched elementwise work per x channel); FP64 is slow on this GPU | 3-10x (projection from the dcorr profile; none for hsic; `speedup_kind: projection`) | unverified-within-tolerance |
| `family:calc-vs-storage` | none | Calculation 229.1 s per window. Storage durable write is 174-812 ms per 118-channel metric per 600 windows (`tests/storage/PRODUCTION_RESULTS.md`), i.e. 0.05-0.25 s per window for 185 metrics (0.02-0.11% of the calculation) | n/a | n/a |

**How build_csv ranks these rows** (README "Optimisation columns", ranking rules). Rows without an explicit `projected_patient_s` are split by the tier of the units they cover, each part with its own units' basis only: `family:undirected-upper/retained` (spearmanr, kendalltau, mi.gaussian on their retained identifiers, kendalltau-sq alone for kendalltau; 10,662 s per patient, 17.8 s per window), `/eeg-fc-family` (6 units) and `/other` (17 units); `family:jni-overhead/retained` (6 retained Gaussian JIDT units, 20,737 s) and `/eeg-fc-family` (6 units). `xc-undirected-upper` is `subsumed_by` rank_corr-1, rank_corr-2, it_gaussian-1 and it_gaussian-2, which compute every pair without the bivariate loop, so it gets no saving on those four units (all three retained ones among them), nor on distance.lcss.constraint-itakura and -sakoe_chiba, where it is `subsumed_by` dtw_family-5. It is recorded `independent_of` it_kernel-1/-2, it_kozachenko-1/-2, it_gaussian-4, phi-1 to phi-4, dtw_family-2/-3 and rank_corr-3 because they act on different factors (`audit/profiles/crosscutting.yaml`): it halves the number of calls, it_kernel-2, it_kozachenko-2, it_gaussian-4, phi-1, phi-2, phi-4 and dtw_family-2 cut the cost of each call, it_kernel-1 and it_kozachenko-1 move the per-channel terms out of the pair loop, and phi-3, dtw_family-3 and rank_corr-3 serve one identifier from another identifier's per-pair result or cached matrix; so it keeps its ~2x on the residual those avenues leave. `xc-jni-overhead` is `subsumed_by` it_gaussian-1/-2 on all twelve units. `xc-window-parallel` and `xc-blas-threads` are `speedup_kind: throughput` and rank after every per-window row, together with the pair-level pools that compete for the same cores (lmfit-5, gpfit-2, ccm-1, cov_prec_robust-2, cov_prec_glasso-1). The two GPU rows have distinct ids (`xc-gpu-spectral`, `xc-gpu-distance`) and are ranked on the residual after the caches they name in `depends_on`; both carry `speedup_kind: projection` (no GPU code ran), so each ranks after the other per-window rows of its tier. `xc-gpu-spectral` carries a `scope_speedup` of 1x on spectral.cohmag, spectral.icoh and spectral.psi, the later consumers of the shared coherency, so the one shared build is credited on spectral.phase alone; `xc-gpu-distance` has 1x on distance.hsic.*, which has no GPU projection of its own. The pair-level pools are alternatives to `xc-window-parallel`, so none lists it in `depends_on`; `xc-window-parallel` lists `xc-blas-threads` only for BLAS at 1 thread in JVM workers, a deployment precondition. On every unit, avenues with no link are stacked by lower-bound speedup and each is ranked on the residual the earlier ones leave (ranking rule 4), so the savings of one unit's rows never add up past its basis.

**Refuted seed items (cross-cutting).**
- **JNI per-call overhead as a major cost.** The floor is 1.4-6% of each Gaussian JIDT unit (<10%). The row is kept only to record the measured bound; the real cost is the Java-side estimator work, which the it_gaussian closed-form avenue removes.
- **BLAS threads as a general speed-up.** No gain on GraphicalLassoCV or on preprocessing, and 1.4x on 8 cores for gpfit, which is worse use of those cores than window workers.
- **Storage as a bottleneck.** ≤0.11% of the calculation.

**Why `family:blas-threads` carries an opportunity.** The JVM + OpenBLAS segfault is reported as a finding (evidence list of the row). The row's opportunity `xc-blas-threads` exists because the task defines it as a shared id owned by this group, and other groups cite it in `depends_on` (basic.cov/prec MinCovDet, EllipticEnvelope and GraphicalLassoCV, distance.dcorrx.maxlag-10). Its ~1.4x applies to gpfit only, which is outside the retained selection, and it is not a recommendation to raise BLAS threads in JVM processes.

**EEG scaling.** None of this family's units has EEG identifiers. Preprocessing runs once per window, whatever the number of bands.
