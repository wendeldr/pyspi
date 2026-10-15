# coint family: define, trace and check

Units: `misc.coint.johansen_order-0_ardiff-10`, `misc.coint.johansen_order-0_ardiff-1`, `misc.coint.johansen_order-1_ardiff-10`, `misc.coint.johansen_order-1_ardiff-1` and `misc.coint.aeg`.

- Each Johansen unit covers two identifiers: `max_eig_stat` and `trace_stat`. Both read one cached fit.
- The `aeg` unit covers three identifiers: `trend-c_autolag-aic`, `trend-ct_autolag-aic` and `trend-ct_autolag-bic`, all with `maxlag-10`.
- Class: `Cointegration(Undirected, Unsigned)` in `pyspi/statistics/misc.py:22-103`.
- Delegates: statsmodels 0.15.0 (`coint_johansen` and `stattools.coint`), with numpy 2.5.4.
- Probe scripts: session scratchpad, `coint/probe1.py` to `probe8.py`, `urca_probe.R` and `gen_records.py`.
- Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.
- The regression goldens are snapshots of current output. Nothing here treats them as a reference.
- The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.

## Documented definition (shared)

Supplement §1.6.3, p.19 (heading at supplement.txt line 885, text 886-899):

> "If two time series are individually integrated but some linear combination of them has a lower order of integration, then the series are said to be 'cointegrated' [79]. We implement statistics for quantifying the cointegration of bivariate time series from two tests included in v0.12.0 of statsmodels [75]: the Augmented Engle-Granger (AEG, modifier aeg) [80] two-step test and the Johansen test (johansen) [81]."

The keywords are "undirected, linear, unsigned, bivariate, time-dependent". The GitBook page `spi-descriptions/miscellaneous.md:13-41` repeats the text and marks `coint_aeg_tstat_trend-ct_autolag-aic_maxlag-10` as the representative "cointegration" SPI. That is also the only coint config in `sonnet_config.yaml:166-181`.

Reference [81] is Pesaran & Shin 1999, "An autoregressive distributed lag modelling approach to cointegration analysis". That is the ARDL bounds approach, not the Johansen test. Hypothesis: this is a citation error in the supplement and on the GitBook (the GitBook's second key reference links to the same chapter).

## Shared plumbing (trace)

1. **Config and labels.** `pyspi/config.yaml:1293-1358` defines eleven configs. `Calculator._load_yaml` (`calculator.py:252-259`) never reads the YAML `labels` block, which says "linear". Every instance therefore carries the class labels `["misc","unsigned","temporal","undirected","nonlinear"]` (`misc.py:26`, confirmed by a probe).
2. **Preprocessing.** `data.py:184-195`: optional `scipy.signal.detrend`, then `zscore(ddof=1)`.
3. **Pair order.** `base.py:132-138` `Undirected.multivariate` calls `Directed.multivariate` (`base.py:85-94`), whose outer loop is over j and inner loop over i≠j. For a pair a<b the first call is therefore `bivariate(i=b, j=a)`. Afterwards the lower triangle is overwritten from the upper.
4. **Cache.** `misc.py:54-60, 62-95`: the cache key is `(method, det_order, k_ar_diff)` or `(method, autolag, maxlag, trend)`. The statistic is not part of the key. The first-computed orientation is stored under both `(i,j)` and `(j,i)` (`misc.py:95`).
5. **Errors.** Any exception in `multivariate` makes the Calculator store an all-NaN matrix for that SPI (`calculator.py:336-337`); a failed MetricReport is recorded and the warning, built at `:339`, is emitted at `:355`.

## Johansen units (`misc.coint.johansen_order-{0,1}_ardiff-{1,10}`)

**Documented** (supplement.txt:896-899):

> "we output both the maximum eigenvalue (modifier max_eig_stat) and the trace (trace_stat) of the vector error correction model ... a constant (order-0) or constant and linear trend (order-1) and fixed autoregressive lags of 1 (ardiff-1) and 10 (ardiff-10)."

**Canonical.** Johansen 1988 (JEDC 12:231-254), Johansen 1991 (Econometrica 59:1551-1580), and Johansen & Juselius 1990 for the deterministic cases.

1. R0 and R1 are the residuals of Δy_t and y_{t−1} after regressing on the lagged differences and the deterministic terms.
2. S_ij = m⁻¹ Σ R_i R_jᵀ.
3. The eigenvalues λ₁ ≥ λ₂ solve |λS11 − S10 S00⁻¹ S01| = 0.
4. trace(0) = −m Σ ln(1−λ_i) and λ-max(0) = −m ln(1−λ₁).

**Implemented** (`misc.py:79-87` → `statsmodels/tsa/vector_ar/vecm.py:594-723`):

1. `coint_johansen(z[[i,j]].T, det_order, k_ar_diff)` is called with columns [z_hi, z_lo].
2. The function detrends the levels by OLS on `np.vander(linspace(-1,1,T), det_order+1)` (`:650-653`, `:670`).
3. With f=0 it demeans the differences, the lagged differences and the lagged levels on the effective sample (`:665-684`). The level regressor is `endog[1:T-k]`, which is y_{t−k} (the Johansen-1988 form). Because the differences at lags 1..k are partialled out, this gives the same eigenvalues as y_{t−1}. The from-scratch match below confirms it.
4. It forms the moment matrices and takes `eig(inv(Skk) Sk0 inv(S00) S0k)` (`:686-692`).
5. With m = T−k−1 (`:714`), `lr1 = −m Σ log(1−a)` (trace) and `lr2 = −m log(1−a_i)` (`:716-718`).
6. pyspi keeps element [0] of each, which is the test of r=0. Critical values are discarded.

Swapping the column order changes the value by at most 1.3e-11 relative (probe8), so the unit carries no direction. The identity trace − max_eig = −m ln(1−λ₂) holds to 1.5e-14.

**Deviation analysis.**

- **order-0 (unrestricted constant).** These units match the canonical statistic. A from-scratch numpy reduced-rank regression with Δy_t and y_{t−1} partialled on [1, lagged differences] agrees to ≤4.2e-13 relative (probe4). R urca 1.3 `ca.jo(ecdet="none", K=k+1, spec="transitory")` agrees to the 6 printed decimals:

  | Dataset (T=256, pair (1,0)) | statsmodels and urca "none": maxeig / trace | urca "const" (restricted constant): maxeig / trace |
  |---|---|---|
  | cointegrated, k=1 | 105.293744 / 111.013941 | 105.293780 / 112.303727 |
  | cointegrated, k=10 | 27.483770 / 35.158291 | 27.487023 / 36.310711 |

  The documented phrase "constant" does not say which Johansen case is meant. The implemented one is the unrestricted constant (model H1(r)), which differs from the restricted-constant case in the last column.
- **order-1 (candidate deviation).** statsmodels removes a full-sample linear trend from the levels before differencing. It never puts t into the partial regressions. The result matches neither the restricted-trend case (H*(r), urca `ecdet="trend"`, which equals my from-scratch case-4 regression to 6 decimals) nor the unrestricted-trend case. All figures are on cointegrated at T=256:

  | Pair | k | statsmodels: maxeig / trace | restricted trend: maxeig / trace | trace gap |
  |---|---|---|---|---|
  | (1,0) | 1 | 105.347364 / 109.648779 | 105.347370 / 111.330040 | 1.5% |
  | (1,0) | 10 | 27.185651 / 32.915367 | 27.511696 / 35.644458 | 8.3% |
  | (2,0) | 1 | 13.4316 / 16.8835 | 13.5735 / 18.4528 | 9.3% |
  | (2,0) | 10 | 24.9979 / 28.8501 | 25.0466 / 31.5702 | 9.4% |

  At T=1024 the gaps are 2e-6 to 1.7e-2 relative. Pipeline detrending does not change the order-1 values (≤4.9e-11). It changes the order-0 values a lot: up to 422 for the k=1 trace statistic on trending at T=1024.
- **Documentation wording.** The outputs are rank-0 LR statistics, not "the maximum eigenvalue" or "the trace of the VECM". "ardiff" counts lagged differences, so the VAR order is k+1.
- **Stationarity confound (candidate, conceptual).** The documented premise that the series are I(1) is never tested. Independent white noise (iid_gaussian, T=256, pair (1,0)) gives a larger trace statistic than the truly cointegrated random-walk pair:

  | Configuration | iid_gaussian trace | cointegrated (1,0) trace |
  |---|---|---|
  | order-0, k=1 | 203.18 | 111.01 |
  | order-0, k=10 | 50.40 | 35.16 |
  | order-1, k=1 | 205.73 | 109.65 |
  | order-1, k=10 | 54.03 | 32.92 |

  On z-scored stationary EEG the statistic will mostly reflect how persistent or stationary each channel is (hypothesis).
- **Robustness (candidate).** What happens with an exactly collinear pair depends on the input.
  - Bitwise-identical channels (shared_envelope, where x2 == x0) raised numpy `LinAlgError` ("Singular matrix") in all four configs at T=256 and T=1024. The Calculator then blanks the whole matrix.
  - A pair z and 3z+1 (z a normalised channel, `coint_johansen` called directly) raised in 29 of 64 probed cases (4 datasets × 2 T × 2 channels × 4 configs). The other 35 returned without error, often with out-of-range values: negative eigenvalues, negative statistics (cointegrated T=256 ch0, order-1, k=10: trace −79.73), or eigenvalues near 1 with very large statistics (shared_envelope T=256 ch0, order-0, k=10: trace 2222.44). These values would enter the matrix silently (probe `coint_fix/p_collin.py`).
  - Both stats are ≥ 0 with trace ≥ max-eig only on non-degenerate pairs.
- **Numerics (value-neutral).** numpy 2.5.4 `eig` returns complex128 even for real spectra, so every call emits a `ComplexWarning` at `vecm.py:717-718`. The imaginary parts were exactly 0 on every probed pair.

**Proposed checks** (details in the YAML):

- the from-scratch reduced-rank regression (kernel, tolerance 1e-9; for order-1 also a literal numpy re-implementation of the statsmodels algorithm, plus the canonical case 4 to quantify the gap)
- urca `ca.jo` (external; agreement expected for order-0, a quantified disagreement expected for order-1)
- a statistical test on cointegrated against the is_cointegrated truth and the MacKinnon-Haug-Michelis critical values, with a Monte Carlo for size
- the stationarity confound (qualitative)
- direction: A == Aᵀ and column-order invariance
- affine invariance (scaled_offset, normalise on and off)
- pipeline-detrend on trending
- the degenerate shared_envelope case
- the trace/max-eig identity

## AEG unit (`misc.coint.aeg`)

**Documented** (supplement.txt:892-894):

> "we use a lag that is inferred via either AIC (aic) or BIC (bic), with a maximum lag of 10 and obtain a t-statistic (tstat) of the unit-root test on the residuals. The time series is first detrended by assuming either a constant (c) or a constant and linear trend (ct)."

**Canonical.** Engle & Granger 1987 (Econometrica 55:251-276), Said & Dickey 1984 and MacKinnon 1991/2010.

1. Regress y0 by OLS on [1, (t), y1].
2. On the residuals û, run Δû_t = ρû_{t−1} + Σγ_iΔû_{t−i} + e_t with no deterministic term.
3. The statistic is the t-ratio on ρ. More negative means stronger evidence of cointegration.
4. The statistic depends on which variable is the regressand (Phillips & Ouliaris 1990).

**Implemented** (`misc.py:69-78` → `statsmodels/tsa/stattools/_stattools.py:2571-2717`):

1. `coint(z[i], z[j], autolag, maxlag=10, trend)` fits `OLS(y0, add_trend(y1, trend))` (`:2686-2688`).
2. If R² < 1−100√eps, it calls `adfuller(resid, maxlag, autolag, regression="n")` (`:2690-2699`). Otherwise it returns −inf with a CollinearityWarning (`:2700-2706`).
3. Inside `adfuller`, a common-sample lag search runs over p = 0..10 (`:386-407`, `_autolag :147-154`). The information criteria are aic = −2llf + 2k and bic = −2llf + ln(n)k (`linear_model.py:2143-2174`).
4. It re-fits with the chosen p on the largest sample (`:420-427`) and returns `tvalues[0]` (`:433-436`).
5. pyspi returns the t-statistic unchanged. The comment "Return the negative t-statistic" at `misc.py:99` does not match.

A from-scratch numpy implementation reproduces pyspi to ≤4.2e-13 on all three configs and every probed dataset and T (probe5). A variant without the re-fit differs by up to 2.27.

**Deviation analysis.**

- **Index-order dependence (candidate; pack hypothesis, probes consistent with it).** Under `multivariate` the higher-index channel is always the dependent variable, because of the loop order and the two-key cache. The kernel is strongly asymmetric:

  | Dataset | Pair | Config | pyspi | Reverse orientation |
  |---|---|---|---|---|
  | cointegrated, T=256 | (0,2) | c/aic | −1.114 | −2.294 |
  | cointegrated, T=256 | (2,3) | c/aic | −2.474 | −1.750 |
  | trending, T=4096 | (0,3) | c/aic | −0.365 | −61.24 |
  | iid_gaussian, T=4096 | (0,4) | c/aic | −66.05 | −27.69 |

  For iid_gaussian the two orientations select different AIC lags (0 and 5). Reordering channels can change an entry by about 61.
- **API inconsistency (candidate).** On a two-channel Data built from trending (0,3), `bivariate(D)` (y0 = channel 0) returns −61.24, while `multivariate(D)[0,1]` returns −0.365.
- **Sign and labels.** The class is `Unsigned` (`issigned()` is False) and the comment speaks of a negative t-statistic, yet the raw, usually negative, t-statistic is returned. Its orientation is opposite to the Johansen identifiers. The class label says "nonlinear" while the documentation and the config say "linear".
- **Stationarity confound.** iid white-noise pairs give −17.6 to −10.0 at T=256 and −66 to −46 at T=4096, comparable to or below the cointegrated pair (−16.25 at T=256).
- **Collinear pairs** give −inf (shared_envelope A[0,2]). Downstream code has to handle the infinity.
- "First detrended" is implemented as deterministic terms inside the cointegrating regression. For the residuals the two are equivalent (Frisch–Waugh).

**Proposed checks:**

- the from-scratch Engle-Granger (kernel, 1e-10)
- urca `ur.df` on the lm residuals (external; the sample convention may differ, which is a hypothesis)
- a direction check in both orientations plus a channel-permuted pipeline run (expected to show order dependence)
- a statistical test on cointegrated against MacKinnon 2010 critical values, with a Monte Carlo
- the stationarity confound
- affine invariance
- pipeline-detrend (ct configs invariant to 1.8e-14; the c config changes by up to 32.5 on trending at T=1024)
- shared_envelope giving −inf

## Open questions

- Did statsmodels 0.12.0, the version the supplement documents, compute identical `coint_johansen` and `coint` values? Lag-search details (the re-fit on the largest sample) and the 0.15 code were not diffed against 0.12.
- Is the det_order=1 pre-detrending scheme intended to approximate a specific published case? It descends from LeSage's MATLAB `johansen.m`, by recollection and unverified. Should the audit judge order-1 against the restricted-trend case or against the unrestricted-trend case?
- The MacKinnon-Haug-Michelis (1999) and MacKinnon (2010) critical values quoted in the checks are from memory and must be re-read from the sources before the statistical checks use them.
- Does urca `ur.df` with `selectlags` re-fit on the largest sample like statsmodels? If not, the external AEG comparison needs a matching sample rule.
- How the stationarity confound plays out on real EEG (persistence versus coupling) is a hypothesis for the check stage.
- Whether −inf (AEG) or an all-NaN matrix (Johansen) occurs in saved EEG outputs was not checked. Duplicated or re-referenced channels would trigger them.

## Checks

**Setup.** The manifest `audit/checks/manifest/coint.json` has 395 runs: 72 or 73 per Johansen unit and 121 for `misc.coint.aeg`. It was regenerated from a draft left by an interrupted attempt. Every pyspi output was recomputed with `--force`.

- References: `audit/checks/refs/coint.py`, run in the comparison venv with numpy/scipy only. It never imports statsmodels or pyspi.
- External reference: `audit/checks/r/coint.R` (R 4.6.1, urca 1.3.3), with CSV exchange under `out/exchange/coint/`.
- Probes: `audit/checks/probes/coint_probes.py`, in the project venv. It defines calc, onefit, collinear, cplx, meta, api, crit and size.
- Results: `audit/checks/out/results/coint.json` has **334 pass, 61 fail, 0 without reference**.
  - 59 failures are runs designed to differ, marked "EXPECTED TO DIFFER" in their notes. Their gap is the measurement:
    - 18 `rev` (reverse aeg orientation)
    - 6 `orderdep` (visiting order)
    - 16 `canon` (order-1 vs textbook case 4)
    - 16 `urca` (order-1 vs ca.jo ecdet 'trend')
    - 3 `trendinv` (order-0 and aeg trend 'c' on a linear trend)
  - The other 2 failures are real outcomes of the truth check: `misc.coint.johansen_order-0_ardiff-10|cointegrated|64|pipeline|stat-{me,tr}`. See below.
- Cost: the summed pyspi `time_s` was 169 s. The slowest plain SPI run took 0.6 s (aeg, T=4096). The collinear probes took ≤ 10 s each.
- The regression goldens played no part in any check.

The check kinds are:

1. **`own` (exact, rtol 1e-10).** Shared kernels `_kernels_basic_misc.johansen` (statsmodels placement of the deterministic terms) and `engle_granger_tstat`. Each is applied in pyspi's stored orientation: column 0 / y0 = the higher index.
2. **`canon` (Johansen).** A textbook reduced-rank regression written in the reference file (`rrr_case`: R0 and R1 residuals, `scipy.linalg.eigh(S10 S00⁻¹ S01, S11)`). It covers case 2 (restricted constant), case 3 (unrestricted constant), case 4 (restricted trend) and case 5 (unrestricted trend).
3. **`urca` / `urcadt`.**
   - Johansen: `ca.jo` for every ecdet × spec, plus ecdet "none" on full-sample OLS-detrended levels (`none_dt`) and ecdet "none" with dumvar t (`none_ut`).
   - aeg: `lm` residuals plus `ur.df(type="none", selectlags="Fixed")` at the lag the statsmodels rule selects. urca's own AIC/BIC search is reported alongside.
4. **Truth and tables.**
   - `stat`: pair (0,1) of `cointegrated` against statsmodels' tabulated 5% values (probe `crit`).
   - `size`: 400 independent random-walk pairs pushed through pyspi itself. The per-replicate values are checked exactly; rejection rates are extra keys.
   - `cvmc`: a Monte Carlo of the textbook case behind each table.
5. **Propositions (qualitative).** `confound`, `calc`, `collinear`, `degen`, `cplx`, `meta`, `onefit`, `api`.
6. **Invariances.**
   - `affine`: scaled_offset against bivariate_gaussian.
   - `trendinv`: trending against bivariate_gaussian.
   - `perm`: channels reversed, kernel on the permuted data.
   - `ident`: trace − max_eig = −m ln(1−λ₂).

**Which statistic each identifier stores.**

- `coint_johansen_max_eig_stat_*` stores λ-max(0) = −m ln(1−λ₁).
- `coint_johansen_trace_stat_*` stores trace(0) = −m Σ ln(1−λᵢ), with m = T−k−1.
- Both are the tests of rank 0, element [0] of statsmodels' arrays, read from one cached fit.
- The `ident` runs reproduce trace − max_eig = −m ln(1−λ₂) to ≤ 2.2e-15 relative.
- The `onefit` runs show the trace SPI makes no additional `coint_johansen` call: 6 calls for 6 pairs, then 0.

### Johansen order-0 units (`misc.coint.johansen_order-0_ardiff-1`, `…_ardiff-10`): matches, score 9

Score 9 is the nearest-anchor reading for "canonical computation reproduced, supplement wrong" (README "Score columns: original, rubric reading, adjusted"). The numeric half of anchor 10 is
met (fp-exact against the textbook RRR and R urca, both independent of statsmodels), and its "matches the pyspi supplement
description" half fails: the supplement calls the outputs the "maximum eigenvalue" and "trace" of the VECM and the parameter
"autoregressive lags", and cites [81] Pesaran & Shin (ARDL). The anchor-9 examples are value conventions, so this is a mapping,
not a literal anchor-9 clause; misc.gpfit.DotProduct is mapped the same way. distance.hhg also fails anchor 10's supplement
clause, but over a different supplement text (an RKHS kernel statistic on ranks of the variables), and its rubric reading is
`anchor` 9 through anchor 9's estimator-variance clause (tie rounding on tied data; `records/distance.hhg.yaml`
score_rubric_basis and score_rubric_note), not a nearest-anchor mapping.

- **Which references are independent.** The own kernel `_kernels_basic_misc.johansen` imports no statsmodels but places the
  deterministic terms "EXACTLY as statsmodels" (its docstring; the det_order block, `if det_order > -1`, at
  vecm.py:665-684), so it is a structural
  transcription and independent only under the import rule. The independent-impl label rests on `rrr_case` (case 3) and R urca.
- **Variant margins.** The urca `ref_maxrel_const_*` figures (6.8e-5 to 0.17) are per-run maxima over pairs. Per pair, the
  nearest alternative deterministic case still differs from pyspi by at least 8.1e-7 relative (restricted constant, iid_gaussian
  T1024) and 3.7e-6 (no deterministic terms, cointegrated T4096) in the trace statistic (k=10; wf4b-2 scratch probe
  `wf4b2_high-optimiser-other/p_coint.py`). That margin is thin on iid T1024 but far above the rtol 1e-9 / 1e-8 criteria. On an
  EEG pair (001_Baseline, T512) R urca `ca.jo(ecdet="none", K=11)` gives 6.0418587745 / 8.8743131766 against pyspi's 6.041859 /
  8.874313 (wf4b scratch probe `p_coint_urca.R`).

- **Kernel.** 19/19 exact passes per unit. The largest max_rel is 1.35e-12 (k=1, cointegrated T=4096) and 2.5e-13 (k=10).
- **Textbook case 3.** 8/8 passes per unit at tolerance 1e-9 (max_rel 5.2e-14 and 2.0e-12).
- **urca.** `ca.jo(ecdet="none", K=k+1)` gives 8/8 passes (max_rel 3.3e-13 and 3.5e-12).
- **Matching ecdet/spec for det_order 0** is ecdet "none" with either spec. spec "longrun" gives the same statistics as "transitory" (`maxrel_none_longrun` 0), and [x_lo, x_hi] equals [x_hi, x_lo] in urca.
- **Other ecdet values do not match.** ecdet "const" (restricted constant) differs by up to 0.21 (k=1) and 0.17 (k=10) relative on cointegrated. ecdet "trend" differs by up to 0.60 and 0.72.
- **Truth (`stat`).** The cointegrated pair is far above the 95% values (14.26 max-eig, 15.49 trace) at T ≥ 256:
  - k=1, trace: 111.01, 386.05 and 1628.1 at T=256, 1024 and 4096.
  - k=10, trace: 35.16, 101.76 and 343.45.
  - At T=64 with k=10 (m=53), the cointegrated pair gives 12.81 (max-eig) and 12.92 (trace), below the 95% values. This is lack of power, not an implementation fault: the kernel and urca runs agree with pyspi on every value. The two `stat` runs fail and stay failed.
  - k=1 at T=64 passes (43.33).
- **Tables (`cvmc`, `misc.coint.johansen_order-0_ardiff-1|cointegrated|256|pipeline|cvmc`).** statsmodels' det_order-0 table is the case-3 distribution for walks with drift. Monte Carlo at T=4000 with 2000 reps gives a 95% trace quantile of 15.516 (se 0.33), against the table's 15.494; max z is 1.07.
- **Size.** On driftless walks the case-3 statistic is oversized:
  - trace rejects 0.100 and 0.135 (k=1) and 0.1225 and 0.150 (k=10) at T=256 and 1024, against a nominal 0.05 (se 0.011);
  - max-eig rejects 0.0625 to 0.105.
  - This is a property of the test. pyspi discards critical values, so it affects interpretation, not values.
- **Candidates.** Six, all confirmed as observed behaviour (details in the records). The first is an interpretation note, not a finding (see Findings):
  - stationarity confound;
  - collinear robustness;
  - complex dtype (value-neutral);
  - "nonlinear" label;
  - documentation wording and citation;
  - method-spelling AttributeError.
- **Why 9 and not 10.** The value is the canonical LR statistic, fp-exact on one independent reference (the textbook RRR, `rrr_case`) and one external reference (R urca); the own kernel also agrees, but it is a structural transcription of statsmodels (see 'Which references are independent' above) and does not count as independent. The supplement's "autoregressive lags of 1/10" are lagged differences (VAR order k+1), and "maximum eigenvalue"/"trace of the VECM" name the LR statistics.

### Johansen order-1 units (`misc.coint.johansen_order-1_ardiff-1`, `…_ardiff-10`): variant, score 7

Score anchor 7: matches the supplement's "constant and linear trend" in a non-canonical, value-changing way.

- **Kernel.** 19/19 exact passes per unit (max_rel 2.3e-13 and 3.0e-12).
- **The ecdet/spec combination that reproduces det_order 1 does not exist natively.** None of ecdet none, const or trend matches, with either spec. statsmodels' numbers are reproduced exactly by **ecdet "none" on levels linearly detrended by full-sample OLS**: `urcadt` 8/8 passes per unit, max_rel 9.6e-14 and 7.6e-14. The textbook case 3 on pre-detrended levels also reproduces them (≤ 1.6e-14 and 7.1e-12).
- **Closest native case: ecdet "trend" (restricted trend, case 4).** All 16 `urca` and 16 `canon` runs fail, as designed. The max_rel against case 4 is:

  | Unit | Statistic | T=256 | T=1024 | T=4096 |
  |---|---|---|---|---|
  | k=1 | max-eig | 0.025 | 0.018 | 0.26 |
  | k=1 | trace | 0.124 | 0.085 | 0.34 |
  | k=10 | max-eig | 0.012 | 0.019 | 0.28 |
  | k=10 | trace | 0.195 | 0.053 | 0.38 |

  Example, T=256 pair (1,0), k=10: pyspi gives 27.186 (max-eig) and 32.915 (trace); case 4 gives 27.512 and 35.644.
- **Overall closest case: unrestricted trend (case 5, urca `none_ut`).** The gap to case 5 shrinks with T:
  - trace max_rel 2.0e-3, 1.2e-3 and 4.8e-4 (k=1) at T=256, 1024 and 4096;
  - 0.024, 5.7e-3 and 7.1e-3 (k=10).
- **Tables (`cvmc`, `misc.coint.johansen_order-1_ardiff-1|cointegrated|256|pipeline|cvmc`).** statsmodels' det_order-1 table is the case-5 distribution for walks with a quadratic trend. The 95% trace quantile is 18.727 (se 0.36) against 18.3985; max z is 1.31. At T=1000 the Monte Carlo sat about +0.4 above the table, which is finite-T drift. The run therefore uses T=4000; the tolerance was not changed.
- **Size.** On driftless walks the order-1 trace rejects 0.19 to 0.2225 at the tabulated 5% value. The Monte Carlo 95% quantile is 23.2 to 24.1, against 18.40. This is why 4 of 5 null pairs exceed the 5% value at T=1024 in `stat-tr` (the shared walks make those pairs dependent).
- **Trend handling.** det_order 1 removes linear trends exactly: `trendinv` passes (≤ 1.7e-15). The pipeline-detrend values equal the kernel-path values.
- **Correction of the trace text** (appended to `deviation_from_canonical` in both records). "Differences shrink with T" is true for the cointegrated pair only. On non-cointegrated pairs the case-4 gap persists, and the statistic tracks case 5 instead.
- **Candidates.** Seven, all confirmed as observed behaviour. The new one is the det_order-1 scheme itself, refined as above. The stationarity confound is an interpretation note, not a finding (see Findings).

### AEG unit (`misc.coint.aeg`): variant, score 6

Score anchor 6: an interpretable deviation from both the supplement ("undirected") and canonical practice (orientation chosen by substance).

- **Kernel.** 36/36 exact passes (max_rel 1.6e-13).
- **urca.** `ur.df` Fixed at the selected lag gives 12/12 passes (max_rel 6.5e-14). The selected lag comes from this file's re-implementation of the statsmodels rule: a common-sample search over 0..10, then a re-fit on the largest sample.
- **urca's own selection.** `selectlags="AIC"/"BIC"` differs from pyspi by up to 10.63 (iid T=1024) and picks another lag in 2 to 10 of the 6 to 10 pairs. ur.df searches lags 1..10 only (`for (i in 2:(lags))`), on the common sample, without a re-fit. So it can never choose lag 0.
- **Symmetry and visiting order.**
  - The stored matrix is bit-exactly symmetric (max|A − Aᵀ| = 0 over 107 plain runs).
  - The underlying statistic is not symmetric. Reversing the orientation (`rev`) changes the cointegrated-data entries by up to 1.53, 1.10 and 0.93 (trend c / ct-aic / ct-bic) at T=256, and by 1.04, 1.23 and 1.23 at T=1024.
  - Reversing the channel order (`orderdep`) changes all 6 pairs by the same maxima. `perm` passes exactly with the rule "y0 = the higher permuted index".
  - `rev` on iid_gaussian T=256 reproduces base.pair_handling's measurement of 7.41: 7.414, 7.415 and 1.710 here, against `base.pair_handling|iid_gaussian|256|pipeline|upper-misc.Cointegration` (max_abs 7.41, 7.41 and 1.71).
  - On trending T=4096 the reversal changes the c-config entry by 60.87.
- **API inconsistency.** `api-c-p03` (exact pass of the reference vector): bivariate(D) = −61.239 with y0 = z0, while multivariate(D)[0,1] = −0.3646 with y0 = z3. After bivariate runs on the same Data, multivariate returns −61.239.
- **Truth (`stat`).** 12/12 pass. The (0,1) entry for trend c is −8.24, −16.25, −30.22 and −63.63 at T=64, 256, 1024 and 4096, against MacKinnon 5% values of about −3.4.
- **Size.** 0.04 to 0.075 at the MacKinnon 5% value (se 0.011).
- **Confound.** iid white-noise pairs give a median of −28.14 at T=1024 (cointegrated pair: −30.22). delayed_noise gives −32.93.
- **Trend handling.** The ct configs are trend-invariant (≤ 4.4e-16). Trend c on trending differs by up to 32.45, by design.
- **Collinear pairs and sign.** Duplicated channels give −inf (6/6 `degen`, 16/16 `collinear`). The sign is the raw negative t-statistic.
- **Candidates.** Seven, all confirmed as observed behaviour. The stationarity confound is an interpretation note, not a finding (see Findings).

### Symmetry and direction (all units)

- **Johansen.** Every plain run gives max|A − Aᵀ| = 0.0. The column-order swap changes values by ≤ 5.2e-12 relative, and the channel reversal by ≤ 1.5e-11 absolute. This agrees with base.pair_handling's ≤ 2.6e-13 visiting-order differences for the 8 Johansen identifiers (`base.pair_handling|iid_gaussian|256|pipeline|upper-misc.Cointegration`). The Johansen identifiers encode no direction.
- **aeg.** The stored orientation follows the channel index, not causality. On var2_bivariate (x0 → x1) the stored value is −25.18 and the reverse is −24.85. On delayed_noise the stored value is −31.99 and the reverse is −32.06.

### Remaining trace issues

All six were already fixed in the records and in this note when the check stage started. They were re-checked against the current checkout (HEAD a42ebc1; calculator.py lines unchanged since a1f20c7):

- `calculator.py:252-259` is `_load_yaml`'s config loop.
- `calculator.py:295` is `compute`.
- `calculator.py:327-339` is the try/except: except at 336, NaN at 337, warning built at 339.
- The warning is emitted at 355, after the `MetricReport` at 342.
- `tests/README.md:130` is the preprocessing-variant list.
- `tests/README.md:521-531` are the coint run-status rows.

### Findings

The confirmed findings below are not in KNOWN_ISSUES.md:

1. **det_order-1 scheme.** order-1 Johansen is LeSage's pre-detrending scheme. It differs from the restricted-trend case by up to 38% on null pairs at T=4096 and approximates the unrestricted-trend case.
2. **aeg orientation.** The aeg value depends on channel order and entry point.
3. **Collinear Johansen pairs.** A duplicated channel blanks the whole Johansen matrix (8 of 11 coint SPIs fail on shared_envelope). Exactly collinear pairs can instead return out-of-range values silently: 3 to 6 of 16 cases per config.
4. **aeg −inf.** aeg returns −inf for collinear pairs.
5. **Metadata and documentation.** Mislabels: the "nonlinear" label, the Unsigned class with a raw negative t, and citation [81]. The latent method-spelling AttributeError.

The complex-dtype note is value-neutral (max |imag| = 0).

**Interpretation note (not a finding): stationarity confound.** Independent stationary pairs score as strongly as, or more strongly than, a truly cointegrated pair, for both methods (`confound` runs). This is the standard behaviour of cointegration tests applied to I(0) series: every linear combination of two stationary series is stationary, so the Johansen r = 0 hypothesis and the Engle-Granger unit-root hypothesis are rejected regardless of coupling. It is an applicability limit for z-scored stationary data such as EEG, not a value defect, and it does not support `new_finding` in any record. `new_finding` stays true on the findings above: the collinear-pair defects for every unit, the det_order-1 scheme for order 1, and the orientation dependence and −inf for aeg. The candidate resolutions keep outcome "confirmed" for the observed behaviour, and their evidence says it is an interpretation note.

## Open questions: answered

1. **Were the values identical in statsmodels 0.12.0?** Inconclusive, not answered: no execution against 0.12.0 was possible. The source reading below is circumstantial.
   - The source diff is not stored under `audit/`. It exists only in the session scratchpad (`coint_check/sm012/`), so it cannot be re-inspected from the audit tree.
   - The 0.12.0 sources were read from the v0.12.0 tag on GitHub: `statsmodels/tsa/vector_ar/vecm.py`, `tsa/stattools.py` and `regression/linear_model.py`, saved under the session scratchpad `coint_check/sm012/`.
   - They were diffed against the installed 0.15.0 for `coint_johansen`, `coint`, `adfuller`, `_autolag` and OLS `aic`/`bic`.
   - The differences are cosmetic: formatting, `stacklevel`, the `"nc"` → `"n"` rename, the result-object plumbing, and a new ValueError for constant input in `adfuller`. The new ValueError is unreachable here because `coint` returns −inf first.
   - No value-changing edit was found. The detrend, `lagmat`, residual, `eig` and statistic code, and the autolag search with re-fit, are line-for-line the same, and `aic` = −2 llf + 2(df_model + k_constant) in both.
   - Running 0.12.0 on the same inputs and comparing with the stored outputs would settle it. It was not installed, as the task forbids installs. Until then the question stays open; the scores do not depend on it, because every check compares with the installed 0.15.0.
2. **Should order-1 be judged against the restricted-trend or the unrestricted-trend case?** Against both. The evidence favours the unrestricted-trend case (case 5) as the target statsmodels approximates:
   - the gap to case 5 shrinks with T (≤ 1.2e-3 and ≤ 1.1e-2 at T=4096);
   - the gap to case 4 does not shrink on null pairs;
   - the tabulated det_order-1 critical values are case-5 quantiles (`cvmc`).
   The scheme descends from LeSage's `johansen.m`. statsmodels' `coint_tables.py:1-35` reproduces LeSage's header ("p = order of time polynomial in the null-hypothesis … p = 1, for constant plus time-trend"; tables from MacKinnon, Haug & Michelis via `johdist.f`). The records quote both gaps.
3. **Critical values.** They were not re-read from the papers. Instead:
   - Johansen: statsmodels' copies (`coint_tables.py` `c_sjt`/`c_sja`, via the `crit` probe) were reproduced by an in-check Monte Carlo of the textbook case 3 (det_order 0, max z 1.07) and case 5 (det_order 1, max z 1.31).
   - AEG: the MacKinnon (2010) values from `adfvalues.mackinnoncrit` give 4% to 7.5% rejection on 400 driftless null pairs (se 1.1%).
   - The values used are −3.43, −3.36, −3.34 and −3.34 (trend c, T=64, 256, 1024 and 4096) and −3.93 to −3.78 (trend ct).
   - Reading MHM 1999 / MacKinnon 2010 directly would settle the literature side.
4. **Does urca `ur.df` with selectlags re-fit like statsmodels?** No.
   - urca 1.3.3 searches lags 1..10 only, never 0, on the common sample, and then fits the chosen lag on that same sample without a re-fit.
   - That is why the external check uses `selectlags="Fixed"` at the statsmodels-selected lag (exact, 12/12).
   - urca's own selection differs by up to 10.63.
5. **How much of the value reflects persistence or stationarity on real EEG?**
   - On the synthetic stationary data the answer is "most of it". Independent white noise gives Johansen trace statistics 1.2 to 2.2 times the cointegrated pair's, growing with m (median/m ≈ 0.81 to 0.88 for k=1, 0.175 for k=10).
   - Every independent pair rejects "no cointegration" at 5%, for every config (`confound` runs).
   - AEG gives −28 (iid) against −30 (cointegrated) at T=1024.
   - Real EEG was not run (out of scope for the synthetic suite). A run on recorded EEG comparing within-channel persistence (e.g. lag-1 autocorrelation) with the statistic would settle it.
6. **Do saved EEG outputs contain −inf or all-NaN coint matrices?** Not checked; no saved EEG outputs were inspected.
   - The mechanism is confirmed on synthetic data: a bit-identical duplicate channel gives an all-NaN Johansen matrix and −inf aeg entries.
   - An exactly scaled copy can also give silent out-of-range Johansen values.
   - Scanning the stored EEG result files for NaN-only `coint_johansen_*` tables and −inf `coint_aeg_*` entries would settle it.

## Optimisation

Workflow 3 (profile-based search for hotspots and avenues; nothing in `pyspi/` was changed). Walls from `audit/profiles/summary.json` (quiet machine, OMP_NUM_THREADS=1); percentages are shares of the profiled `compute_spi` frame in `audit/profiles/misc.coint.<unit>__8x1024.pstats`.

| unit | 8x1024 (fresh) | 32x1024 | 118x1024 per window (fresh / family-warm) | measure_mode |
|---|---|---|---|---|
| johansen_order-0_ardiff-10 | 0.045 s | 0.786 s | 10.95 s / 11.01 s | measured |
| johansen_order-0_ardiff-1 | 0.023 s | 0.408 s | 5.88 s / 5.69 s | measured |
| johansen_order-1_ardiff-10 | 0.047 s | 0.827 s | 11.52 s / 11.54 s | measured |
| johansen_order-1_ardiff-1 | 0.026 s | 0.448 s | 6.23 s / 6.25 s | measured |
| aeg (3 identifiers) | 0.146 s (family-warm 0.43 s) | 2.66 s (family-warm 7.95 s) | 37.0 s / 110.8 s | measured |

Shared versus incremental cost: within a Johansen unit the trace_stat identifier is a cache hit after max_eig_stat (family-warm increment 0.008-0.009 s at 118x1024), and each unordered pair is computed once (the cache stores (i, j) and (j, i)). Across units nothing is shared. The three aeg identifiers have different cache keys (trend, autolag), so each recomputes everything: 37.1 s (c, aic), 37.3 s (ct, aic), 36.4 s (ct, bic) per window. Family total about 146 s per window at 118x1024; no coint identifier is in the EEG selection.

Hotspots:
- Johansen: `statsmodels/tsa/vector_ar/vecm.py:coint_johansen` 97.8-98.7%; `vecm.py:detrend` (statsmodels OLS per regression, 112 fits) 44.0% / 67.5% / 46.1% / 69.3% (order-0 ardiff-10, order-0 ardiff-1, order-1 ardiff-10, order-1 ardiff-1) and `vecm.py:resid` (numpy pinv/svd) 40.8% / 13.3% / 39.9% / 12.3%; numpy svd 18.5-39.7% in total. Opaque shares 0.117-0.208. T_exponent 0.09-0.34: statsmodels object overhead still dominates at T=1024.
- aeg: `statsmodels/tsa/stattools/_stattools.py:coint` 99.7%, `adfuller` 92.7%, `_autolag` 81.6%; 364 statsmodels OLS fits for 28 pairs (13 per pair): OLS `__init__` 52.5% (data handling, `_handle_constant` 45.5% runs matrix_rank/svd on every design), OLS `fit` 35.7% (pinv_extended 22.2%). Opaque share 0.208.

Opportunities:
- coint-1 (four Johansen units; seed "batched regressions and a vectorised 2x2 generalised eigenproblem" kept): one Gram matrix of every channel's detrended level, difference and lagged-difference columns per window, per-pair S00/S0k/Skk by Schur complements, all 2x2 generalised eigenproblems in one vectorised call. Expected 10-50x. The batched computation replaces the whole pair loop: base.py:multivariate is 99.951 / 99.926 / 99.955 / 99.893% (o0a10 / o0a1 / o1a10 / o1a1), Amdahl bounds of about 2,040 / 1,350 / 2,200 / 937x at 8x1024. Replacing only the per-pair coint_johansen call (98.65 / 97.92 / 98.71 / 97.79%) would be bounded at 74 / 48 / 78 / 45x. An earlier version quoted 71x for every record. Not measured. Output risk unverified-within-tolerance (same reduced-rank regression, different factorisation; the check stage already sees column-order changes of <= 5.2e-12 relative).
- coint-2 (aeg): the 11 candidate ADF regressions are nested in the lag order on a common sample, so one QR gives every candidate's RSS and AIC/BIC; numpy instead of statsmodels OLS objects for the first stage and the re-fit. Expected 3-8x (removable share about 0.88). Output risk unverified-within-tolerance, with the caveat that an AIC/BIC near-tie could select another lag (a discrete change).
- coint-3 (aeg): share the trend='ct' first stage and candidate fits between the aic and bic identifiers (adfuller with regresults=True keeps every candidate fit); expected ~1.35x on the unit's 110.8 s per window (the record's figure). Output risk unverified-bit-identical (same fits, different criterion read off them).
