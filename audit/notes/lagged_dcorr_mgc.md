# lagged_dcorr_mgc: define, trace and check

Units: `distance.dcorrx.maxlag-1`, `distance.dcorrx.maxlag-10` (class `CrossDistanceCorrelation`) and `distance.mgcx.maxlag-1`, `distance.mgcx.maxlag-10` (class `CrossMultiscaleGraphCorrelation`), all in `pyspi/statistics/distance.py`. Delegate: hyppo 0.5.2 (`hyppo.time_series.DcorrX` / `MGCX`). MGC itself comes from scipy 1.17.1 `scipy.stats.multiscale_graphcorr`. Pairwise distances come from sklearn 1.9.1.

Status: checked (numeric checks in the `## Checks` section at the end; sections 1-6 are the trace stage). The probe numbers in sections 3-4 come from small read-only scripts in the scratchpad (`lagged_dcorr_mgc/probe1.py` to `probe4.py`). They support the trace but do not count as checks. The regression goldens for these identifiers are snapshots of current output and serve as no reference here. The verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## 1. Documented definition

- dcorrx, supplement §1.2.2, p.5: "The cross-distance correlation [11] quantifies the independence between two univariate time series based on distance correlation [13] ... This measure is the average of lagged distance correlations between the past of time series x to the future of time series y, up to a given lag." The listed keywords are "undirected, nonlinear, unsigned, bivariate, time-dependent".
- mgcx, supplement §1.2.4, p.6: "defined similarly to dcorrx (Sec. 1.2.2) but uses lagged MGCs instead of lagged distance correlations." The keywords are the same.
- The GitBook page `distance-similarity.md` repeats both texts. Its key reference is the hyppo paper, arXiv:1907.02088.
- The GitBook glossary (`glossary-of-terms.md:21,27`) says distance correlation is "defined in the range [0,1]". For directed SPIs it says "The rows in this table reflect the sources while the columns reflect the targets".
- In `pyspi/config.yaml:191-213` the config entries carry the label "undirected". The runtime class labels are "directed" and "unsigned" (`distance.py:119,139`).

## 2. Canonical definition

Source: Shen, Chung, Mehta, Xu and Vogelstein, 2024, "Independence Testing for Temporal Data", *TMLR* 05/2024, arXiv:1908.06486v5, Sec. 2.2, Algorithm Steps 1-3, p.3. The hyppo docstrings cite an earlier version of this paper.

- Step 1. For each l = 0..L, compute τ_n(X, Y_{-l}) on the n-l lagged pairs {(X_{1+l}, Y_1), ..., (X_n, Y_{n-l})}. This tests whether X_t depends on the past Y_{t-l}.
- Step 2. The optimal lag is L* = argmax_l ((n-l)/n) τ_n(X, Y_{-l}).
- Step 3. The statistic is T_n = Σ_{l=0}^{L} ((n-l)/n) τ_n(X, Y_{-l}). It is a weighted **sum** and includes lag 0.
- Choice of τ_n. For DCorrX, τ_n is the unbiased distance correlation (Székely & Rizzo 2014, *Ann. Stat.* 42(6); original dcorr from Székely, Rizzo & Bakirov 2007, *Ann. Stat.* 35(6)). For MGCX, τ_n is MGC (Vogelstein et al. 2019, *eLife* 8:e41690; Shen, Priebe & Vogelstein 2020, *JASA* 115(529)). The paper lists both as admissible at line 379 of the extracted text.

## 3. What is implemented (trace)

1. **pyspi.**
   - `Directed.multivariate` (`pyspi/base.py:84-94`) sets A[i,j] = `bivariate(data, i, j)`.
   - `CrossDistanceCorrelation.bivariate` (`distance.py:125-131`) takes x = z[i] and y = z[j], each of shape (T,1) and z-scored by `Data`. It returns `DcorrX(max_lag).statistic(x, y)[0]`.
   - `CrossMultiscaleGraphCorrelation.bivariate` (`distance.py:145-151`) does the same with `MGCX` and drops `opt_lag` and `opt_scale`.
2. **hyppo `compute_stat`** (`time_series/_utils.py:85-108`).
   - Line 88 builds Euclidean distance matrices D_x and D_y of the series.
   - Line 92 instantiates the test with `compute_distance="euclidean"`.
   - Line 93 computes the lag-0 term on the raw series, with weight 1.
   - Lines 98-102 compute, for j = 1..M, `slice_distx = distx[j:n, j:n]` (the future of x = process i) and `slice_disty = disty[0:n-j, 0:n-j]` (the past of y = process j). The term is `(n-j)/n * indep_test.statistic(slice_distx, slice_disty)`.
   - Line 106 returns `np.sum(dep_lag)`.
3. **Lag-term inputs.** `Dcorr.statistic` (`independence/dcorr.py:141-149`) and `MGC.statistic` (`independence/mgc.py:151-166`) call `compute_dist(..., "euclidean")` whenever `is_distance` is False. `is_distance` is set only when `compute_distance` is falsy (`dcorr.py:116-118`, `mgc.py:128-130`). The sliced distance matrices are therefore treated as data matrices: each time point becomes its (n-j)-vector of distances, and the Euclidean distances between those rows are used. In short, the lag terms use a distance of distances.
4. **Dcorr kernel.**
   - `_center_distmat(bias=False)` (`dcorr.py:262-281`) applies U-centring.
   - `_dcorr` (`dcorr.py:417-449`) returns cov/sqrt(varx·vary). There is no clipping in the unbiased case, so the value can be negative. It returns 0 if a variance is ≤ 0.
5. **MGC kernel.** `scipy.stats.multiscale_graphcorr(distx, disty, compute_distance=None, reps=0)` (`scipy/stats/_mgc.py:300-380`) leads to:
   - `_mgc_stat` (`_mgc.py:383-429`)
   - `_local_correlations` (compiled; https://github.com/scipy/scipy/blob/v1.17.1/scipy/stats/_stats.pyx)
   - `_threshold_mgc_map` (`_mgc.py:432-474`)
   - `_smooth_mgc_map` (`_mgc.py:477-525`). It returns the global-scale entry `stat_mgc_map[m-1][n-1]` (`_mgc.py:502`) unless a large enough significant region holds a larger local correlation. The docstrings give the statistic a range of [-1, 1] (`_mgc.py:396, 493`).
   - `multiscale_graphcorr` raises `ValueError("MGC requires at least 5 samples ...")` when n < 5 (`_mgc.py:331-333`). Every lag slice has n-j samples, so mgcx needs T ≥ max_lag + 5. Probe (`lagged_dcorr_mgc_fix/probe_smalln.py`): T = 5 raises and T = 6 returns a value for max_lag = 1; T = 14 raises and T = 15 returns a value for max_lag = 10.
   - The trace ends at this compiled boundary. The installed scipy ships no `.pyx`, so the from-scratch MGC check (check 3 below) is what tests the local-correlation map.
   - hyppo's `Dcorr` and its unbiased U-centring are never executed for mgcx: `MGC.statistic` (`mgc.py:151-166`) calls scipy directly.
6. **Extra work in mgcx only.** `MGCX.statistic` then calls `compute_scale_at_lag` (`mgcx.py:113-115`, `_utils.py:111-126`). That runs `MGC().test(..., reps=0)`, which calls `multiscale_graphcorr` once directly (`mgc.py:254`) and once more through `perm_test`'s observed statistic (`independence/base.py:130`, `tools/common.py:730`). pyspi discards the result.

### Probe evidence

Setup: n = 64, x_t = y_{t-1} + 0.3·noise, `probe1.py`.

| Statistic | hyppo output | Hand-built dist-of-dist sum | Canonical sum (slices passed as precomputed distances, or dcorr of `x[1:]`, `y[:-1]`) |
|---|---|---|---|
| DcorrX(max_lag=1) | 0.7465127 | 0.7465127 (exact) | 0.8046051 |
| MGCX(max_lag=1) | 0.7468173 | 0.7468173 (exact) | 0.8045958 |

## 4. Deviation analysis

### Against the canonical definition (Shen et al. 2024)

These parts match:
- the (n-l)/n weights
- inclusion of lag 0
- summation rather than averaging
- the pairing (X_t, Y_{t-l})
- the choice of unbiased dcorr or MGC

Deviation (candidate, supported by both code and probe): for every l ≥ 1 the lag statistic is computed on distances between rows of the sliced distance matrices, not on distances between the lagged samples.

- The lag-1 term moves from 0.8166 to 0.7585 in the toy probe.
- The lag-0 term is unaffected.
- For maxlag-10 this affects 10 of the 11 terms.

Hypothesis: the distance-profile embedding is an injective, monotone-like re-embedding for 1-D data. If so, the statistic still detects lagged dependence, but its values differ from the published statistic. Whether the paper's consistency guarantees carry over is not established.

### Against the documentation (supplement §1.2.2 and §1.2.4, GitBook)

1. **Sum, not average.** The docs say "average", but the output is the weighted sum over M+1 lags. Values above 1 occur:
   - T256 probe: dcorrx_maxlag-1 up to 1.43; mgcx_maxlag-1 up to 1.69.
   - dcorrx_maxlag-10 up to 6.34; mgcx_maxlag-10 up to 7.97 (trace probe on the sum datasets; the check runs reach 7.116 and 9.008 on narrowband_f0-0.25, see Checks).
2. **Lag 0 is included.** It is not mentioned in the docs.
3. **Direction.** The code pairs the future of process i with the past of process j, so A[i,j] measures j → i (rows are targets). This is the reverse of the supplement's "past of x to future of y" (with x = z[i]) and of the glossary's "rows = sources".
   - Probe, `delayed_noise` T256 (x0 leads x1 by 1 sample): dcorrx_maxlag-1 gives A[1,0] = 0.968 and A[0,1] = -0.001.
   - Probe, `var2_bivariate` T256 (X → Y): A[1,0] = 0.220 and A[0,1] = 0.042 (dcorrx_maxlag-1).
   - Probe, `delayed_noise` T256, first two processes (`fable_review/probe_lagged.py`): mgcx_maxlag-1 gives A[1,0] = 0.9681 and A[0,1] = -0.0012; mgcx_maxlag-10 gives A[1,0] = 0.9670 and A[0,1] = 0.0002.
4. **Labels.** The supplement keywords and `config.yaml` say "undirected", while the class is `Directed` and labelled "directed".
5. **Sign.** The SPIs are labelled "unsigned" and the glossary gives distance correlation a [0,1] range, yet values are negative on independent data. `iid_gaussian` T64 probe, off-diagonal minimum and count of negative entries out of 20:
   - dcorrx_maxlag-1: -0.031 (13); dcorrx_maxlag-10: -0.099 (10).
   - mgcx_maxlag-1: -0.0318 (13); mgcx_maxlag-10: -0.0972 (8).

   The mechanism differs between the two classes:
   - For dcorrx, the unbiased `_dcorr` returns cov/sqrt(varx·vary) with no clipping (`dcorr.py:438-447`).
   - For mgcx, the negative values come from scipy's MGC statistic, a local correlation in [-1, 1]. When no significant region is selected, scipy returns the global-scale entry (`_mgc.py:502`), and that entry takes both signs. On the lag-0 distance matrices alone, scipy MGC is negative on 18 of 20 `iid_gaussian` T64 pairs (minimum -0.0264). The glossary's [0,1] statement concerns distance correlation, so for mgcx only the "unsigned" label is contradicted. The sign mechanism inside the compiled `_local_correlations` was not traced.
6. **Distance of distances.** No document mentions this construction.
7. **Efficiency (mgcx only).** There are M+3 `multiscale_graphcorr` evaluations per ordered pair where M+1 are needed. The probe counted 4 for M=1 and 13 for M=10. The extra calls do not change the output and do not consume the global RNG.

KNOWN_ISSUES.md has no entry for these SPIs. M7 and M8 cover legitimately negative estimators for *other* SPIs; the negative dcorrx/mgcx values are a new instance of that class, so every unit carries `already_documented: "partly"` (the same treatment as dcorr_hsic). The class name "bias-corrected" fits mgcx too: SciPy column-centres both distance matrices with "the centering and unbiased modification" (`scipy/stats/_mgc.py:199-201`; `hyppo/independence/mgc.py:82-84`). Each `new_finding_candidates` entry for the sign now says in its own text that it is an instance of this class and not a new finding by itself. Everything else listed above is absent from KNOWN_ISSUES.md.

## 5. Proposed checks (summary; full list in the records)

1. **Kernel trace check, `tolerance`.**
   - dcorrx: an independent from-scratch reimplementation of the as-implemented formula, with U-centred dcorr and lag terms on row distances of the sliced distance matrices. Expected agreement: within 1e-10.
   - mgcx: `scipy.stats.multiscale_graphcorr` fed the same inputs. This reference is labelled **same-lineage**.
2. **Kernel canonical check, `tolerance`.** A from-scratch Shen et al. 2024 T_n using unbiased dcorr (or a from-scratch MGC) on the lagged sample pairs.
   - Datasets: `delayed_noise`, `var2_bivariate`, `gauss_ar1_pair`, `logistic_maps`, `iid_gaussian`.
   - Expected: lag-0 terms identical, totals different. Report the size of the difference and whether the A[i,j] vs A[j,i] ordering survives.
3. **From-scratch MGC check (mgcx), `tolerance`.** Following Shen, Priebe & Vogelstein 2020 and Vogelstein et al. 2019, applied to the lag-0 term on `nonlinear_dependence`, `bivariate_gaussian` and `iid_gaussian`.
4. **Direction check, pipeline, `direction`.** On `delayed_noise`, `var2_bivariate`, `logistic_maps` and `gauss_ar1_pair`. Expected: A[target, source] > A[source, target], which establishes that A[i,j] means j → i.
5. **Null check, pipeline, `statistical`.** On `iid_gaussian` at T in {64, 256, 1024, 4096}. Expected: values centred near 0, a spread that shrinks with T, and negative values present.
6. **Range check, pipeline, `qualitative`.** On `narrowband_f0-0.1` and `shared_envelope`. Expected: values above 1 and never above the bound Σ_j (n-j)/n.
7. **Invariance check, pipeline, `tolerance`.** `scaled_offset` should equal `bivariate_gaussian` within 1e-10. This compares pyspi with pyspi, so it is labelled **same-lineage** (self-consistency). It can establish invariance but not what A[i,j] means.
8. **Call-count check (mgcx), kernel, `exact`.** Expect M+3 calls, with identical output when the scale step is skipped.

## 6. Upstream status

The distance-of-distances construction is still present in current upstream hyppo. Source: neurodata/hyppo `main`, `hyppo/time_series/_utils.py`, fetched from raw.githubusercontent.com on 2026-10-14.
- `compute_stat` still runs `indep_test = indep_test(compute_distance=compute_distance, **kwargs)` and then `stat = indep_test.statistic(slice_distx, slice_disty)` on slices of the distance matrices. This is identical to the installed 0.5.2 (`_utils.py:85-108`).
- Upstream `Dcorr.statistic` and `MGC.statistic` still call `compute_dist` whenever `is_distance` is False.
- `compute_scale_at_lag` (called only by MGCX) has an `if not compute_distance: compute_distance = "precomputed"` branch. The same branch is in the installed 0.5.2 (`_utils.py:114-115`), so it is not an upstream difference. It never triggers on the pyspi path, because `compute_distance="euclidean"` is truthy, so it cannot rescue the lag terms from the distance-of-distances construction.

Upstream `main` differs from 0.5.2 in how it handles small samples. Its `MGC.statistic` wraps `multiscale_graphcorr` in `except (IndexError, ValueError): stat = 0.0`, and `compute_scale_at_lag` catches `ValueError`. A lag slice with fewer than 5 samples therefore gives 0.0 upstream, where 0.5.2 raises.

## Open questions

- Does the distance-of-distances statistic keep the consistency and validity guarantees of Shen et al. 2024? It is still a dcorr between two metric embeddings of the time points, but the embeddings depend on the whole sample. This is a hypothesis for the check stage, not a claim.
- Is the j → i direction intended by pyspi? The supplement's wording suggests the authors meant i → j. A pipeline direction check should settle the empirical convention before any downstream use of row and column semantics.
- With multi-realisation `Data` (R > 1), z[i] is (T, R) and hyppo treats each time point as an R-dimensional sample. This path was not probed.
- For MGC, scipy's `_local_correlations` is compiled. Its source was cited from the v1.17.1 tag on GitHub but was not opened in this checkout. The trace ends at that boundary; the from-scratch MGC check is what will test it.

## Checks

**Setup.**
- Manifest: `audit/checks/manifest/lagged_dcorr_mgc.json`, 187 runs (dcorrx_maxlag-1 49, dcorrx_maxlag-10 49, mgcx_maxlag-1 46, mgcx_maxlag-10 43). 20 of them were added after the adversarial review (see "Review corrections" below).
- References: `audit/checks/refs/lagged_dcorr_mgc.py` (comparison venv; no hyppo or sklearn; `scipy.stats.multiscale_graphcorr` only in the runs tagged `scipy`, which are labelled same-lineage). The dcorrx replica (`_kernels_distance.hyppo_dcorrx_replica`) is a numpy port of hyppo's `compute_stat` and `_dcorr` arithmetic, so its runs (replica, lagterms, scaleinv, multireal) are labelled same-lineage too. The mgcx replica keeps the same lag slicing but computes MGC from scratch, and stays independent-impl.
- Probes: `audit/checks/probes/lagged_dcorr_mgc_probes.py` (project venv).
- Results: `audit/checks/out/results/lagged_dcorr_mgc.json`: **144 pass, 43 fail, 0 without reference**. 42 of the failures are the `mehta` runs, which are designed to fail: they compare pyspi with the canonical statistic, and the size of the gap is the deviation reported below. The remaining failure is one null exceedance; it is not an independent chance event but the unusual iid_gaussian T256 draw that dcorr_hsic also flags, seen through the lag-0 dCor² term (see dcorrx_maxlag-1).
- Cost: the pyspi side took 754 s in total in the first run and 412 s in the review re-run with `--force` (sum of `time_s` in `out/pyspi/index/lagged_dcorr_mgc.json`), on a machine shared with several other families. dcorrx_maxlag-10 accounts for 614 s and 294 s. Its longest runs are the T=1024 lagterms probe on delayed_noise (93.6 s, then 42 s; it computes the statistic twice) and the 5-process T=1024 null run (81 s, then 31 s). mgcx is kept at T ≤ 256 and P ≤ 3, as the brief requires: MGCX costs 0.05-0.1 s (L=1) and 0.26-0.29 s (L=10) per ordered pair at T=256.

**What hyppo computes** (established from the source in `.venv/lib/python3.12/site-packages/hyppo/`, and reproduced numerically below):
- `time_series/_utils.py:85-108` `compute_stat`:
  - line 88 builds the Euclidean distance matrices D_x and D_y of the two series;
  - line 92 builds the test with `compute_distance="euclidean"`;
  - line 93 computes the lag-0 term on the raw series, with weight 1;
  - lines 98-102: for j = 1..M, `slice_distx = distx[j:n, j:n]` (the LATER samples of x = z[i]) and `slice_disty = disty[0:n-j, 0:n-j]` (the EARLIER samples of y = z[j]); the term is `(n-j)/n * indep_test.statistic(slice_distx, slice_disty)`;
  - line 105 sets `opt_lag = argmax`, and line 106 returns `stat = np.sum(dep_lag)`.

  The statistic is therefore a **sum over lags 0..M**, not a maximum or an average, with factor (n-j)/n. It pairs x_t with y_(t-j): **y (the column process j) is the one shifted into the past.**
- `independence/dcorr.py:116-118` and `mgc.py:128-130` set `is_distance` only when `compute_distance` is falsy. `dcorr.py:144-147` and `mgc.py:154-157` therefore recompute Euclidean distances between the ROWS of the sliced distance matrices (distance of distances).
- hyppo's own docstring (`time_series/dcorrx.py:69-72`) states DcorrX_n = Σ_j (n-j)/n Dcorr_n(x[j:n], y[0:(n-j)]), the series-slice statistic of Mehta et al. 2019 and Shen et al. 2024. The code does not compute that formula for j ≥ 1.
- `time_series/mgcx.py:113-115` then calls `compute_scale_at_lag` (`_utils.py:111-126`), which runs a fresh `MGC().test(slice_distx, slice_disty, reps=0)`. That makes one `multiscale_graphcorr` call directly (`mgc.py:254`) and one more through `perm_test`'s observed statistic (`independence/base.py:130`, `tools/common.py:729-730`). The result is `opt_scale`, which pyspi discards (`distance.py:150`).

Check tags used for every unit:

| tag | what is compared | criterion |
|---|---|---|
| replica | pyspi vs the as-implemented statistic rebuilt without hyppo (kernel, pipeline, pipeline-detrend) | tolerance 1e-10 |
| mehta | pyspi vs the canonical series-slice statistic | tolerance 1e-10, designed to fail |
| lagterms | hyppo's own per-lag weighted terms and the statistic vs the replica's | tolerance 1e-10; the per-lag gap to Mehta is reported as extra keys |
| dir | A − Aᵀ vs the generating truth | sign at every claimed [target, source] entry |
| null | every off-diagonal entry vs the Bonferroni band of the Monte Carlo null (empirical quantiles) | statistical |
| neg / sum / smalln (+ calls for mgcx) | sign on iid data / value above 1 within the bound Σ(n−l)/n / failure threshold / call count | qualitative |
| scaleinv | scaled_offset through the pipeline vs the replica on the z-scored bivariate_gaussian draw | tolerance |
| multireal (dcorrx) | R = 2 realisations vs the replica with 2-D samples | tolerance |
| scipy, lag0, mgcmap (mgcx) | SciPy on the same inputs (same-lineage); lag-0 term and pyspi mgc vs from-scratch MGC; compiled MGC map vs `mgc_local_corr` | tolerance |

**Review corrections.** An adversarial review found that the canonical (`mehta`) comparison covered only 8 datasets per dcorrx unit and 5 per mgcx unit, so the quoted maximum gap and the quoted maximum value were maxima over a subset. The review's numbers were reproduced and the check was widened:

- 16 `mehta` runs were added (narrowband_f0-0.25, nonlinear_dependence, narrowband_f0-0.1 and shared_envelope at T=256 for each unit; p012 for mgcx where the replica or sum run used it). Every dataset of the replica and sum runs except trending now has a canonical comparison.
- 4 `lagterms` runs were added on narrowband_f0-0.25 T256.
- `ref_mehta` now also stores the signed range of pyspi − Mehta, the fraction of entries above the canonical value, and the per-lag terms of the pair with the largest gap. `ref_lagterms` stores the signed per-lag range.
- The largest gaps are now on narrowband_f0-0.25, and the gap has no fixed sign. The figures below replace the earlier ones, and the records were updated to match.
- The dcorrx replica runs are relabelled same-lineage (see Setup), and the null exceedance is traced to the dcorr_hsic draw (see dcorrx_maxlag-1).

The scores do not change. The finding was already "the lag terms are a different statistic". The new runs show that the difference is larger than first stated and can go either way.

**Reference correction during the check.** The first null reference used a symmetric mean ± z·SD band. It failed one dcorrx_maxlag-10 entry in the upper tail (0.0606 against a normal limit of 0.0566).

- The Monte Carlo null is right-skewed: skewness is 0.4 (L=10) to 1.9 (mgcx L=1).
- A 4000-replicate scratch Monte Carlo (`scratchpad/lagged_dcorr_mgc_check/tail.py`) put the empirical 99.875% quantile at 0.0631.
- A three-moment Pearson III band was tried next. Its fitted lower bound was too tight and gave two lower-tail failures.
- The final reference uses **empirical Monte Carlo quantiles**: 8000 replicates for dcorrx and 2400 for mgcx at T ≤ 256. At T=1024 (dcorrx only), the T=256 standardised quantiles are rescaled with the mean and SD of a 200- or 80-replicate T=1024 Monte Carlo; the skewness at 256 and 1024 is similar (1.19 vs 1.24 for L=1, 0.42 vs 0.32 for L=10). The band was not widened to obtain passes, and the one remaining exceedance is reported.

### dcorrx_maxlag-1 (`distance.dcorrx.maxlag-1`): related-quantity, score 3 (README anchor 3: "label does not describe the computation (… mgcx lag handling)")

- **As implemented: exact.** 13/13 replica runs pass with max_abs ≤ 6.5e-14. The reference is `_kernels_distance.hyppo_dcorrx_replica` (same-lineage: a port of hyppo's code path, so it shows that pyspi computes what the code states, nothing more): U-centred dCor² of the series at lag 0, plus U-centred dCor² of the Euclidean distances between rows of the sliced distance matrices at lag 1, with sklearn-mirrored arithmetic. These runs include the tied logistic_maps (4.8e-14), the pipeline and pipeline-detrend paths, and trending (1.7e-14).
- **Per-lag terms.** The 7 lagterms runs agree to ≤ 6.5e-14. The statistic equals the sum of the terms to 0.0 and differs from their mean by up to 0.583.
- **Canonical (Mehta / Shen, series slices).** The 12 mehta runs fail as designed:
  - lag-0 terms are identical (≤ 6.2e-16; 5.0e-11 on tied logistic_maps, from exact vs sklearn distance arithmetic);
  - totals differ by 0.0019 (iid T1024) to 0.0794 (gauss_ar1_pair T1024, 15.6% relative); narrowband_f0-0.25 gives 0.0741, var2_bivariate 0.0673 (T1024) and 0.0616 (T256), shared_envelope 0.0512, narrowband_f0-0.1 0.0262;
  - the sign of pyspi − Mehta is not fixed: every entry is below the canonical value on var2_bivariate and gauss_ar1_pair, every entry is above it on logistic_maps (+0.0077 to +0.0116), and both signs occur on shared_envelope (−0.0512 to +0.0350) and narrowband_f0-0.1 (−0.0190 to +0.0262);
  - per-lag on var2_bivariate T1024: lag-1 term pyspi 0.1705 vs Mehta 0.2378;
  - on delayed_noise the true-delay (d=1) term is 0.9746 vs 0.9824 (ratio 0.9921, T1024).
- **Direction.** 8/8 dir runs pass. A[1,0] − A[0,1] is:

  | dataset | pyspi | Mehta |
  |---|---|---|
  | delayed_noise T1024 | 0.9753 | 0.9834 |
  | var2_bivariate T1024 | 0.1679 | 0.2337 |
  | gauss_ar1_pair T1024 (claim 1 → 0) | 0.0762 | 0.0963 |
  | logistic_maps_chaotic T1024 | 0.0169 | 0.0194 |

  var2_chain T1024 gives 0.1737 and 0.1907 on (1,0) and (2,1). **A[i,j] = past of j → future of i: rows are targets.**
- **Null** (iid_gaussian, pipeline). Two of the three null runs pass:

  | T | band | result |
  |---|---|---|
  | 64 | [−0.059, 0.143] | pass, max_z 1.44 |
  | 256 | [−0.0140, 0.0379] | **fail** |
  | 1024 | [−0.0037, 0.0097] | pass, max_z 1.67 |

  The Monte Carlo SD shrinks 0.0316 → 0.0078 → 0.0020, about 1/T. The T=256 entry A[3,0] = −0.0143 sits just below the band. Its empirical per-entry two-sided p is 0.002, or 0.04 after Bonferroni, and pyspi equals the replica on this draw (`…|iid_gaussian|256|kernel|replica`, 1.1e-16).
  - This is not an independent chance event. The lag-0 term is dCor² on the same iid_gaussian T256 draw that the dcorr_hsic family flagged. It equals that family's `distance.dcorr.biased-False` pyspi matrix to 1.8e-16 (`scratchpad/lagged_dcorr_mgc_check/null256_shared.py`).
  - On that draw the mean dCor² sits at the 0.15% lower permutation tail: dcorr_hsic runs `distance.dcorr.biased-False|iid_gaussian|256|pipeline|null` (pyspi mean −0.00368, z −2.03) and the other three `distance.*|iid_gaussian|256|pipeline|null` runs; see notes/dcorr_hsic.md.
  - A[3,0] = −0.0143 is the lag-0 term −0.0066 plus the lag-1 term −0.0077. On this draw 18/20 lag-0 terms and 14/20 lag-1 terms are negative.
  - The exceedance is that unusual draw seen through a statistic whose lag-0 term is dCor². It is not an estimator defect and not a new finding.
- **Other checks.**
  - neg: 13/20 entries are negative at T=64 (min −0.0311).
  - sum: max 1.426 (narrowband_f0-0.1) and 1.237 (shared_envelope), both above 1 and below the bound 1.996.
  - scaleinv: 7.4e-15.
  - smalln: n = 2, 3 raise ZeroDivisionError and n = 4 returns a value, so T ≥ L + 3.
  - multireal: R = 2 equals the replica with 2-D samples to 5.6e-16 and differs from the per-realisation average by 0.0112.
- **Symmetry.** max|A − Aᵀ| = 0.975 on delayed_noise and 0.0033-0.011 on iid data.
- **Independent from-scratch probe (Workflow 4b).** `scratchpad/wf4b_low-lagged-distance/canon_probe.py` uses numpy only, without hyppo or the audit reference code. On the stored pyspi inputs its distance-of-distances construction reproduces pyspi to 1.9e-14 (gauss_ar1_pair T1024), 1.3e-14 (var2_bivariate T1024) and 2.5e-10 (tied logistic_maps T256). Its series-slice statistic gives pyspi − canonical in [−0.0794, −0.0593], [−0.0673, −0.0015] and [+0.0077, +0.0116] on the same runs, as the mehta runs do. Unlike the same-lineage replica, this is independent support for the as-implemented formula. Workflow 4b-2 re-ran it with the same output (`scratchpad/wf4b2_low-lagged-distance/canon_rerun.out`).
- **Candidates.** All 5 are confirmed: distance of distances, sum not average, direction reversed relative to the glossary, negative values despite the "unsigned" label, and "undirected" keywords.

### dcorrx_maxlag-10 (`distance.dcorrx.maxlag-10`): related-quantity, score 3 (anchor 3)

- **As implemented: exact.** 13/13 replica runs pass with max_abs ≤ 9.7e-14 (T up to 1024, 5 processes; same-lineage reference). The 7 lagterms runs agree to ≤ 6.5e-14. The statistic equals the sum of the 11 terms to ≤ 8.9e-16 and differs from their mean by up to 6.469.
- **Canonical.** The 12 mehta runs fail as designed:
  - lag 0 is identical;
  - totals differ by 0.0046 (iid T1024) to **1.2163** (`…|narrowband_f0-0.25|256|kernel|mehta`, pair (2,0): pyspi 7.1156 vs Mehta 5.8994, 20.6% above). The same run has pair (1,0) at 4.0819 vs 4.8177 (−0.7357);
  - other runs: gauss_ar1_pair T1024 0.2604 below (24.4%), shared_envelope 0.2096 (−0.2096 to +0.0214), narrowband_f0-0.1 0.1089 below, logistic_maps 0.1009 above on every entry (3.5% of a total of about 2.9), var2_bivariate 0.0920 (T1024) and 0.0951 (T256) below on every entry;
  - per lag on narrowband_f0-0.25 (`…|narrowband_f0-0.25|256|kernel|lagterms`), pair (2,0), lags 0..10:

    | | terms |
    |---|---|
    | pyspi | 0.1765, 0.9897, 0.4204, 0.9820, 0.4165, 0.9742, 0.4133, 0.9664, 0.4107, 0.9586, 0.4073 |
    | Mehta | 0.1765, 0.9874, 0.1754, 0.9797, 0.1737, 0.9724, 0.1723, 0.9644, 0.1712, 0.9565, 0.1699 |

    The even-lag terms are about 2.4 times the canonical ones. Over all pairs of that run, the per-lag pyspi − Mehta ranges from −0.0756 to +0.2450. So the construction can inflate some lag terms and attenuate others; the delayed_noise and var2_bivariate examples below show only attenuation.
- **Per-lag max|pyspi − Mehta|, delayed_noise T1024.** Lags 0..10: 1.8e-16, 0.0078, 0.0142, 0.0078, 0.0140, 0.0073, 0.0011, 0.0011, 0.0013, 0.0011, 0.0029.
  - At the true delays d = 1, 3, 5 (pairs (1,0), (2,0), (3,0)) the pyspi terms are 0.9746, 0.9732, 0.9725, against Mehta 0.9824, 0.9809, 0.9799 (ratio 0.992 at each delay).
  - The reverse terms are |·| ≤ 0.0009.
  - On var2_bivariate T1024 the gap is largest at lag 1 (0.0673; 0.1705 vs 0.2378) and ≤ 0.011 at lags 2-10.
- **Direction.** 8/8 dir runs pass. delayed_noise T1024 covers all six claimed pairs (lags 1-5), with antisymmetric parts 0.950-0.976 (Mehta 0.963-0.983). var2_bivariate T1024 gives 0.2100 (Mehta 0.2987), and gauss_ar1_pair T1024 gives 0.2735 (Mehta 0.3753; the population lagged-dCor² sum favours 1 → 0 by 0.172).
- **Null.** 3/3 runs pass:

  | T | band | max_z |
  |---|---|---|
  | 64 | [−0.188, 0.265] | 1.51 |
  | 256 | [−0.046, 0.068] | 1.78 |
  | 1024 | [−0.010, 0.015] | 1.82 |

  The Monte Carlo SD is 0.075, 0.018 and 0.0040.
- **Other checks.**
  - neg: 10/20 entries are negative at T=64 (min −0.0986).
  - sum: max 6.337 (shared_envelope) and 5.252 (narrowband_f0-0.1), against a bound of 10.785. The largest value in the family is 7.116 (`…|narrowband_f0-0.25|256|kernel|replica`). An average over 11 lags could not exceed 1.
  - smalln: n = 11, 12 raise and n = 13 returns a value, so T ≥ L + 3 = 13.
  - multireal: equals the 2-D-sample replica to 5.6e-15 and differs from the realisation average by 0.054.
- **Symmetry.** max|A − Aᵀ| = 0.976 on delayed_noise and 0.0099-0.038 on iid data.
- **Independent from-scratch probe (Workflow 4b).** The same probe (`scratchpad/wf4b_low-lagged-distance/canon_probe.py`) reproduces pyspi with the distance-of-distances construction to 5.7e-13 on `…|narrowband_f0-0.25|256|kernel|mehta` and 1.7e-14 on `…|gauss_ar1_pair|1024|kernel|mehta`. Its series-slice statistic gives pyspi − canonical in [−0.7357, +1.2163] (pair (2,0): 7.1156 vs 5.8994) and [−0.2604, −0.1587], as the mehta runs do. Workflow 4b-2 re-ran it with the same output.
- **Candidates.** All 5 are confirmed.

### mgcx_maxlag-1 (`distance.mgcx.maxlag-1`): related-quantity, score 3 (anchor 3)

- **MGC from scratch.** `refs/lagged_dcorr_mgc.py:mgc_local_corr` and `mgc_stat` implement column centring with 1/(n−1), dense column ranks, truncated local covariances, the beta threshold, the largest connected component and the smoothed maximum.
  - They reproduce SciPy's compiled local-correlation map to ≤ 3.8e-14 on 5 mgcmap runs. One of these is the tied logistic_maps map of shape (256, 8); min, max or average ranks do not reproduce that shape.
  - They reproduce the full statistic to ≤ 3.3e-14 on 9 replica runs, including the pipeline and pipeline-detrend paths. The reference is therefore independent-impl.
  - The same-lineage SciPy (1.18.1) computation on the same inputs agrees to 0.0 on 3 runs.
- **Lag 0 vs the mgc unit.** On 3 runs, the mgcx lag-0 term MGCX(max_lag=0) equals the pyspi mgc SPI (`distance.mgc`) to ≤ 2.7e-15 and the from-scratch MGC to ≤ 9.1e-15. The hhg_mgc family is now checked: on the same draws the lag-0 term equals its pyspi matrices `distance.mgc|bivariate_gaussian|256|kernel|def`, `distance.mgc|nonlinear_dependence|256|kernel|def` and `distance.mgc|iid_gaussian|256|kernel|def` to ≤ 2.7e-15 (`scratchpad/lagged_dcorr_mgc_check/lag0_vs_hhg_mgc.py`). Those runs pass against that family's independent MGC to ≤ 2.7e-14. MGC is symmetric in its arguments to ≤ 2.7e-15.
- **Per-lag terms.** The 4 lagterms runs agree to ≤ 3.3e-14. The statistic equals the sum of the terms exactly and differs from their mean by up to 0.834.
- **Canonical.** The 9 mehta runs fail as designed:
  - lag 0 is identical (0.0);
  - totals differ by 0.0020 (nonlinear_dependence) to 0.0741 (narrowband_f0-0.25, pair (1,0): 0.8163 vs 0.8904, 8.3% below); var2_bivariate gives 0.0615 (lag-1 term 0.1725 vs 0.2340), gauss_ar1_pair 0.0571 (15.1%), narrowband_f0-0.1 0.0438, shared_envelope 0.0284;
  - the gap takes both signs: −0.0074 to +0.0438 on narrowband_f0-0.1 and −0.0177 to +0.0284 on shared_envelope, while every var2_bivariate, gauss_ar1_pair and logistic_maps entry is below the canonical value;
  - delayed_noise d=1: 0.9695 vs 0.9779 (ratio 0.991).
- **Direction.** 5/5 dir runs pass: delayed_noise 0.9693, var2_bivariate 0.1783, gauss_ar1_pair 0.0928, logistic_maps_chaotic 0.0869, and var2_chain 0.2166 and 0.3104.
- **Null.** 2/2 runs pass: T=64 band [−0.055, 0.152], max_z 1.56; T=256 band [−0.013, 0.040], max_z 1.74. The Monte Carlo mean is slightly positive (0.0042 and 0.0010), and the SD is 0.040 and 0.0099.
- **Other checks.**
  - neg: 6/6 entries are negative at T=64 (min −0.0318).
  - sum: max 1.689 (shared_envelope) and 1.461 (narrowband_f0-0.1), against a bound of 1.996.
  - smalln: n = 2-5 raise ValueError and n = 6 returns a value.
  - calls: 4 `multiscale_graphcorr` calls per pair where 2 are needed. The value is identical, the global RNG is untouched, and the extra calls are 50% of the per-pair time at T=256.
- **Symmetry.** max|A − Aᵀ| = 0.969 on delayed_noise and 0.0061-0.014 on iid data.
- **Candidates.** All 6 are confirmed. The extra MGC test is efficiency only.

### mgcx_maxlag-10 (`distance.mgcx.maxlag-10`): related-quantity, score 3 (anchor 3)

- **As implemented.** 10/10 replica runs pass with max_abs ≤ 7.0e-14. The scipy runs agree to ≤ 1.1e-16, and the 5 lagterms runs to ≤ 7.0e-14; the statistic differs from the mean of its terms by up to 8.189. On 2 lag0 runs the lag-0 term equals the mgc unit to ≤ 8.3e-16, and the hhg_mgc pyspi matrix `distance.mgc|bivariate_gaussian|256|kernel|def` to 3.2e-16.
- **Canonical.** The 9 mehta runs fail as designed.
  - Totals differ by 0.0206 (iid) to **0.7349** (`…|narrowband_f0-0.25|256|kernel|mehta`, pair (1,0): 4.0807 vs 4.8156, 15.3% below). Pair (2,0) of the same run is +0.1560 above.
  - Other runs: shared_envelope 0.1236 below, narrowband_f0-0.1 0.1050 above on every entry, gauss_ar1_pair 0.0984 below (20.0%), var2_bivariate 0.0908 below, logistic_maps 0.0692 below (the dcorrx gap on the same data is above).
  - Per lag on `…|narrowband_f0-0.25|256|kernel|lagterms`, pyspi − Mehta ranges from −0.0755 to +0.0323. The even-lag inflation is much smaller than for dcorrx.
- **Delays.** On delayed_noise p012 and p023, the true-delay terms for d = 1, 3, 5 are 0.9695, 0.9668, 0.9546, against Mehta 0.9779, 0.9738, 0.9627.
- **Direction.** 6/6 dir runs pass, including p023 for the lags 3 and 5.
- **Null.** 2/2 runs pass: T=64 max_z 1.07; T=256 max_z 1.99, band [−0.040, 0.078]. On the T=256 draw all 6 entries are negative (min −0.0400), inside the band.
- **Other checks.**
  - neg: 2/6 entries are negative at T=64.
  - sum: max 7.973 (shared_envelope) and 6.921 (narrowband_f0-0.1), against a bound of 10.785. The largest value in the family is 9.008 (`…|narrowband_f0-0.25|256|kernel|replica`).
  - smalln: n = 11-14 raise and n = 15 returns a value.
  - calls: 13 calls per pair where 11 are needed, 12% of the per-pair time.
- **Symmetry.** max|A − Aᵀ| = 0.967 on delayed_noise and 0.026-0.22 on iid data.
- **Candidates.** All 6 are confirmed.

### Scores and findings

- **Score.** All four units get 3. README anchor 3 names "mgcx lag handling" as its example. The `max_lag` label and the supplement promise lagged distance correlations (or MGCs) of the series, but every lag ≥ 1 term is the statistic of a sample-dependent distance-profile embedding. The documented "average" is a weighted sum. dcorrx shares the construction exactly, so it gets the same score.
- **Why not lower.** The statistic is computed exactly as the code states: the replica is fp-exact (same-lineage for dcorrx, independent MGC for mgcx). The lag-0 term is canonical, and the direction ordering of the canonical statistic is preserved on every truth-claimed pair.
  - On the 42 mehta runs, sign(A − Aᵀ) of pyspi and of the canonical statistic differ only at pairs with no truth claim, where both antisymmetric parts are at most 0.020 in absolute value: iid draws, shared_envelope (1,3), narrowband_f0-0.1 (0,1) for mgcx_maxlag-10, nonlinear_dependence (0,2), and delayed_noise (1,2) where both values are below 0.001 (`scratchpad/lagged_dcorr_mgc_check/mehta_order.py`).
  - The larger gaps found after the review (up to 1.216 on a total) change how large the deviation is, not its kind. Anchor 2 ("sign can differ") is not met for the values or for the truth-claimed directions.
- **New finding** (true on all four records). The distance-of-distances lag construction is not in KNOWN_ISSUES.md. The other confirmed candidates are documentation mismatches that are also absent from KNOWN_ISSUES.md: sum vs average, direction vs the glossary, "undirected", and the extra MGC test. "Unsigned" is a new instance of the class documented in REVIEW M7/M8 (KNOWN_ISSUES.md lines 31-32), which do not name dcorrx or mgcx, so `already_documented` is "partly" on all four records, as for dcorr_hsic.
- **Definition under review** (false on all four records). Shen et al. 2024 Alg. Step 3 fixes the lag construction: a weighted sum with lag 0 included. The supplement's "average" is a documentation mismatch, not a contested standard. For mgcx every MGC term, lag 0 included, inherits distance.mgc's contested minimum-region rule (`definition_under_review: true` on that record); this does not change the score.
- **Trace corrections.**
  - `direction_convention` in all four records now says "confirmed by the check-stage dir runs" in place of "to be confirmed".
  - `units_and_range` cites the smalln runs. dcorrx: T ≥ L + 3 from the U-centring divisor n − 2; mgcx: T ≥ L + 5.
  - The three remaining trace issues from the trace review were already corrected in the records by the interrupted earlier attempt, and they were re-checked here. No record or note still says "raises for n <= 3", "needs n > 3" or "only compute_scale_at_lag switches", and section 6 of this note states that the `precomputed` branch exists in both versions and is unreachable for pyspi:
    - the dcorrx_maxlag-10 small-n rule (T ≥ 13);
    - the dcorrx_maxlag-1 mechanism (ZeroDivisionError from the U-centring);
    - the upstream "precomputed" wording.

## Open questions: answered

1. **Is the distance-of-distances construction still in current upstream hyppo?** Yes. Section 6 records the trace-stage fetch of neurodata/hyppo `main` (2026-10-14): `compute_stat` and the `is_distance` logic are unchanged. This stage did not fetch it again. The numeric checks confirm the installed 0.5.2 behaviour: 45 replica runs and 23 lagterms runs, all fp-exact.
2. **Does the distance-of-distances statistic keep the guarantees of Shen et al. 2024?** Not settled. The evidence available:
   - Under independence the statistic is centred: for dcorrx the U-centred numerator has mean 0 by exchangeability, and the Monte Carlo means are |mean| ≤ 7e-4 for dcorrx and ≤ 0.015 for mgcx.
   - Its spread shrinks about as 1/T.
   - Every true lag is detected. At the delayed_noise delays the terms are 99.1-99.3% of the canonical terms, but on var2_bivariate the lag-1 term is only 72% of the canonical one (0.1705 vs 0.2378).
   - The distortion is not only attenuation. On narrowband_f0-0.25 T256 (`distance.dcorrx.maxlag-10|narrowband_f0-0.25|256|kernel|lagterms`), the even-lag dcorrx terms of pair (2,0) are about 2.4 times the canonical ones (0.42 vs 0.17), while the odd-lag terms agree to about 0.002. The total exceeds the canonical value by 1.216, and another pair of the same run is 0.736 below.
   - So on a narrowband process the statistic weights lags differently from the canonical statistic, and that weighting depends on the data. This does not decide consistency, since both statistics can still be consistent with different finite-sample values, but it does mean the per-lag terms and the argmax lag (`opt_lag`) cannot be read as canonical lagged dCor² values.
   - A heuristic argument: for 1-D data, each time point's distance profile is an injective embedding into a Euclidean (Hilbert) space, and distance covariance characterises independence there. The embedding depends on the whole sample, however, so the published consistency proof does not carry over directly. A proof, or a power study over a family of alternatives, would settle it.
3. **Is the j → i direction what pyspi intends?** The empirical convention is settled: A[i,j] = past of j → future of i, with rows as targets, on all 27 dir runs (agreement 1.0). Whether this is intended cannot be settled from the code. The supplement's "past of x to the future of y" and the glossary's "rows reflect the sources" both read the other way, so either the documentation or the argument order in `distance.py:128-130` / `148-150` needs to change.
4. **Multi-realisation Data (R > 1).** Probed with `multireal`: R = 2 realisations of T/2 samples, `Data(dim_order='psr')`.
   - pyspi feeds z[i] as a (T, R) array, and hyppo treats each time point as an R-dimensional sample. The output equals the replica with 2-D samples to 5.6e-16 (L=1) and 5.6e-15 (L=10).
   - It is not the average of per-realisation statistics: the differences are 0.011 and 0.054.
   - The lag slicing runs along time inside each realisation's columns jointly.
5. **SciPy `_local_correlations` is compiled.** It is now tested numerically. The from-scratch map reproduces SciPy's compiled map to ≤ 3.8e-14 on 5 mgcmap runs, including tied data. That match needs minimal **dense** column ranks: other rank conventions change the map shape on logistic_maps. The full statistic, with SciPy's threshold and smoothing in `_mgc.py:432-525`, agrees to ≤ 7e-14. The `.pyx` source itself was still not opened, because the installed wheel does not ship it.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls come from `audit/profiles/summary.json` and shares are percentages of the profiled `compute_spi` frame. Nothing in pyspi was changed and no micro-benchmark was written for this family. The per-component costs quoted below come from the dcorr cache micro-benchmark, `audit/profiles/micro/opp_dcorr_hsic-1.json`. Not in the retained EEG selection. All four units are genuinely directed (max|A - A.T| up to 0.98), so neither halving avenue applies.

### Walls

| unit | 8x256 | 8x1024 | 32x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|---|
| dcorrx_maxlag-1 | 0.27 s | 7.11 s | 194.0 s | 2701 s fresh / 1743 s family-warm | extrapolated-pairs-from-32 (fresh) / from-8 (family-warm) |
| dcorrx_maxlag-10 | 1.41 s | 46.8 s | not run | 11541 s | extrapolated-pairs-from-8 |
| mgcx_maxlag-1 | 3.33 s | 58.8 s | not run | 14507 s | extrapolated-pairs-from-8 |
| mgcx_maxlag-10 | 9.96 s | 175.3 s | not run | 43222 s | extrapolated-pairs-from-8 |

For dcorrx_maxlag-1, the per-pair cost rose from 127 ms at 8x1024 to 196 ms at 32x1024 (fresh P_exponent 2.39). The two 118 projections therefore differ by 1.55x; the cause was not investigated.

**Complexity: O(P^2 (T^2 + L T^3)).** For each lag j >= 1, `hyppo/time_series/_utils.py:compute_stat` applies sklearn Euclidean distances to the rows of the (T-j) x (T-j) distance-matrix slices. That is a GEMM, O(T^3), and it is the distance-of-distance construction recorded as a correctness finding; this stage does not change it. mgcx adds O(T^2 log T) column rank sorts per MGC call. Measured T_exponent: 2.36 for dcorrx-1, 2.52 for dcorrx-10, 2.07 for mgcx-1 and 2.07 for mgcx-10. At 118x1024 the P^2 pair term is the whole cost. Within a pair, a two-term fit a T^2 + b T^3 to the 8x256 and 8x1024 walls puts the L T^3 term at about 70% of dcorrx_maxlag-10's per-pair cost at T=1024 (36% at T=256, where its profile shows `safe_sparse_dot` 26.1%) and at about half of dcorrx_maxlag-1's. This is a fit, not a measurement at T=1024.

### Where the time goes

- **dcorrx_maxlag-1 (8x1024).**
  - `hyppo/independence/dcorr.py:Dcorr.statistic` is 93.4% cumulative, of which 62.2% is self time: the hidden numba `_dcorr`.
  - `compute_dist` -> sklearn takes 36.9%. Of that, `safe_sparse_dot` is 27.6%: the lag-1 distance-of-distance GEMM.
  - `compute_stat` computes `distx, disty` at line 88. The lag-0 `statistic(x, y)` then rebuilds them, which is three `compute_dist` calls per pair.
  - Caveat: the profiled `compute_spi` total is 24.2 s, against a measured 8x1024 wall of 7.07-7.11 s. That is 3.4x, against 1.4x for dcorr (5.13 vs 3.70 s) and 1.2x for mgcx_maxlag-1 (70.1 vs 58.8 s). The shares above may therefore be distorted by the profiling run; the cause was not investigated. The order of magnitude of lagged_dcorr_mgc-2 does not rest on them alone.
- **dcorrx_maxlag-10 (profiled at 8x256; the 8x1024 profile timed out).**
  - `hyppo/tools/common.py:compute_dist` takes 67.5%. `pairwise_distances` itself is 57.6%; the rest is sklearn's param-validation wrapper and hyppo glue. Inside it, `_euclidean_distances` is 44.2%, with `safe_sparse_dot` 26.1% (the GEMM), and input validation `check_pairwise_arrays` is 11.0%. That validation share is over-weighted at T=256 relative to T=1024.
  - The hidden `_dcorr` is 32.3% self.
- **mgcx_maxlag-1 (8x1024).**
  - `scipy/stats/_mgc.py:_mgc_stat` is 82.3% cumulative over 224 MGC evaluations, four per ordered pair; `numpy unique` (rank transforms) is 31.2%.
  - `compute_scale_at_lag` is 51.4%. It is the discarded scale search: `MGC.test` 50.6%, its `perm_test` observed statistic 21.0%.
- **mgcx_maxlag-10 (profiled at 8x256).**
  - `_mgc_stat` is 84.8% and `numpy unique` 56.5%.
  - `compute_scale_at_lag` is 16.3%.

### Opportunities

- **lagged_dcorr_mgc-1: drop MGCX's discarded scale search (mgcx only; unverified-bit-identical, effort S).** Call hyppo's `compute_stat(x, y, MGC, 'euclidean', max_lag)` directly. `MGCX.statistic` then runs `compute_scale_at_lag` only to return `opt_scale`, which `pyspi/statistics/distance.py:150` discards.
  - mgcx_maxlag-1: **~2x** (Amdahl 2.06x from 51.4%).
  - mgcx_maxlag-10: **~1.2x** (16.3% at 8x256; at 8x1024 the same test is about 29 s of 174 s by difference with maxlag-1).
  - RNG: `MGC.test(reps=0)` -> `hyppo/tools/common.py:perm_test` calls `np.random.randint(..., size=0)` with `random_state=None`. That leaves the legacy global RandomState unchanged (scratch check: `np.random.random()` after `compute_scale_at_lag` equals the value drawn without it), so dropping the call does not shift the global RNG stream. The class stays unverified-bit-identical, not rng-stream.
- **lagged_dcorr_mgc-2: lag-major per-channel cache (unverified-bit-identical for dcorrx; unverified-within-tolerance for mgcx, which needs the hhg_mgc-3 rank-cache port; effort L).**
  - What is cached: each channel's distance matrix, and per lag its leading and lagging distance-of-distance matrices, which depend on one channel only. Also cached are their centred forms and self products (dcorrx) or column ranks (mgcx).
  - Per-channel builds drop from 2P(P-1)(L+1) to P(2L+1), and the duplicate distances at line 88 disappear. Each pair keeps one cross term per lag.
  - Memory: processing lag by lag keeps it at about 2P T^2 floats (1.9 GB at 118x1024).
  - Estimates:
    - dcorrx_maxlag-1: **10-40x**. That is 13806 pairs x 2 lags x ~1.5 ms cross products plus 118 x 3 GEMM-sized setups, about 40-80 s, against 1743-2701 s.
    - dcorrx_maxlag-10: **10-30x**. The 13806 x 11 cross products alone are about 230 s, against 11541 s.
    - mgcx_maxlag-1: **~1.3x**. Distances and ranks in `compute_stat` are about 20-25% of the call.
    - mgcx_maxlag-10: **~2x**. unique 56.5% + distances 12.4%, mostly per channel and lag.
  - The dcorrx GEMMs are BLAS calls, so thread settings matter (`xc-blas-threads`, cross-cutting).
- **lagged_dcorr_mgc-3: derive maxlag-1 from maxlag-10 (dcorrx_maxlag-1, mgcx_maxlag-1; unverified-bit-identical, effort S).** `compute_stat` returns `np.sum(dep_lag)`, and the first two terms are the same operations for max_lag 1 and 10. So the maxlag-1 value is the sum of the first two terms of maxlag-10's list, and the maxlag-1 unit costs almost nothing: **1000-10000x**, CONDITIONAL on the maxlag-10 identifier being computed in the same run, which holds in the default config (otherwise no gain). The remaining cost is a two-term sum and pyspi's dispatch per ordered pair, well under 1 s per window. Today the family-warm increments are 7.07 s (dcorrx-1) and 57.3 s (mgcx-1) at 8x1024. In the ranking the maxlag-1 records list lagged_dcorr_mgc-3 in `subsumed_by` for lagged_dcorr_mgc-1 and -2 (mgcx_maxlag-1) and for lagged_dcorr_mgc-2 (dcorrx_maxlag-1): once the maxlag-1 value is a partial sum of maxlag-10's terms, nothing of its own computation is left for them, so build_csv credits them no saving on those two units (README "Optimisation columns", ranking rule 4). On the maxlag-10 units lagged_dcorr_mgc-1 and -2 keep their figures.

### Seed items refuted or narrowed

- **Seed d: "the extra discarded MGC test" applies to mgcx only.** DcorrX has no scale search. For mgcx-10 the gain is ~1.2x, not ~2x.
- **Distance-of-distance recomputation: kept as is.** It is a semantic finding. Only caching it per channel is proposed.
- **The fast 1-D dcorr path (dcorr_hsic-2) cannot serve the lag terms.** Their "points" are (T-j)-dimensional rows of distance matrices. It would only cover the lag-0 term, a small share for L=10.

### Shared versus incremental cost

Family-warm increments at 8x1024: dcorrx_maxlag-1 7.07 s, dcorrx_maxlag-10 46.8 s, mgcx_maxlag-1 57.3 s, mgcx_maxlag-10 174.1 s. Each equals its fresh wall, so nothing is shared today. With lagged_dcorr_mgc-2 and -3:

- the per-channel distance and distance-of-distance matrices for lags 1..10 would be shared by all four units;
- each maxlag-1 unit becomes a partial sum of its maxlag-10 sibling.
