# Family `ccm`: convergent cross-mapping (define, trace and check)

Units: `causal.ccm.E-None`, `causal.ccm.E-1`, `causal.ccm.E-10`. Each unit has three identifiers (`_mean`, `_max`, `_diff`). All three statistics share one cached curve per E (`data.ccm[E]`).
Records: `audit/records/causal.ccm.*.yaml`. Probe scripts: `/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/ccm/probes/` (p2 direction, p3 E selection, p4 short series and parameters, p5 from-scratch reproduction, p6 library schedule, p7 E-None vs E-10).
The regression goldens are snapshots of current output. This note does not use them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared definition and trace (all three units)

### Documented definition
Supplement §1.3.5, p.8-9:

> "there is a causal influence from time series x → y if the Takens time-delay embedding of y can be used to predict the observations of x"

It continues: "we use 21 uniformly sampled library sizes between this minimum and maximum", followed by "summary statistics of the mean (modifier mean) and the maximum (max) across the curves". The supplement gives the keywords as "directed, nonlinear, unsigned, bivariate, time-dependent" and lists six SPIs, with no `diff`.

The GitBook page `spi-descriptions/causal-inference.md` repeats this text and adds "mean (`mean`), maximum (`max`) and difference (`diff`) across the curves", but never defines `diff`. Both sources say convergence is not measured explicitly.

### Canonical definition
The canonical definition comes from Sugihara et al. 2012 (Science 338:496-500 and its Supplementary Materials), which builds on the simplex projection of Sugihara & May 1990 (Nature 344:734).

1. Build the shadow manifold M_X from delay vectors x(t) = (X(t), X(t-τ), …, X(t-(E-1)τ)).
2. For each t, find the E+1 nearest library neighbours of x(t), excluding t itself.
3. Weight the neighbours by u_i = exp(-d_i/d_1), with w_i = u_i/Σu.
4. Form the cross-map estimate Ŷ(t)|M_X = Σ w_i Y(t_i).
5. Compute ρ(L) = corr(Y, Ŷ|M_X) over random libraries of size L.

The inference rule is: if ρ(Y|M_X) rises with L and converges, then Y causes X. The criterion is convergence, not a summary of the curve.

### What the code computes
pyspi first z-scores each process (`data.py:193-195`). In `pyspi/statistics/causal.py`:
- **Library sizes** (lines 127-131): `lowerE = ceil(2E/10)*10`, `upperE = floor((T-E-1)/10)*10` and `inc = int((upperE-lowerE)/21)`. These go to pyEDM as the string `"lowerE upperE inc"`.
- **Cross-mapping** (lines 118-144): for each unordered pair i<j, pyspi makes one call `pyEDM.CCM(E, columns=proc_i, target=proc_j, sample=100, seed=42)`. It stores `ccmf[i,j,:] = LibMeans["proc_i:proc_j"][:22]` and `ccmf[j,i,:] = LibMeans["proc_j:proc_i"][:22]`.
- **Statistics** (lines 156-161): `mean = nanmean(ccmf, axis=2)`, `max = nanmax(ccmf, axis=2)` and `diff = nanmean(ccmf - ccmf^T, axis=2)`. Because no curve value is NaN, `diff = mean - mean^T`.

pyEDM 1.15.2.0 contains cppEDM 1.15.2 (2023-11-27); the version was confirmed with `parameterList=True`. The relevant cppEDM source at tag v1.15.2:
- `API.cc` L635-708: library and prediction set are all rows.
- `Parameter.cc` L198-291: the library-size string `"start stop inc"` expands to `(stop-start)//inc + 1` sizes. An increment below 1 throws.
- `Parameter.cc` L297-316: `knn = E+1`.
- `Parameter.cc` L643-721: the first E-1 partial rows are dropped.
- `CCM.cc` L139-409: for each size, 100 libraries are drawn uniformly without replacement with `std::default_random_engine(42)`. The prediction set is all rows; the curve value is the mean ρ over the 100 libraries.
- `CCM.cc` L426-434 and L475-491: the second mapping swaps column and target. Column names are `col:target`, meaning "embed col, predict target".
- `EDM_Neighbors.cc` L74-365 and L388-448: Euclidean distances. Only the prediction row itself is excluded (L153-155), because `exclusionRadius = 0`.
- `Simplex.cc` L34-167: weights `exp(-d/d_min)` with a floor of 1e-6 (L44, L76-80). When the nearest distance is exactly 0, zero-distance neighbours get weight 1 and the rest get the floor (L57-70).
- Ties: tie detection in `EDM_Neighbors.cc` L276-345 runs only when `parameters.method == Method::Simplex` (L282). CCM builds its parameters with `Method::CCM` (`API.cc` L659) and reuses them unchanged in `colToTarget`/`targetToCol` (`CCM.cc` L21-26) and in the per-sample local `SimplexClass` (`CCM.cc` L342). So `ties`/`anyTies` stay false, the tie expansion in `Simplex.cc` L93-157 never runs in CCM, and CCM keeps exactly E+1 neighbours: the first E+1 entries of `std::sort` by distance only (`DistanceCompare`, `Common.cc` L10-13; `EDM_Neighbors.cc` L226). `std::sort` is not stable, so which rows are kept among equal distances is implementation-defined.
- `Common.cc` L100-219: Pearson ρ, set to 0 (not NaN) when a variance is 0.

Defaults used: `tau = -1` (lag of 1 sample) and `Tp = 0` (contemporaneous).

**Trace confirmation.** Probe p5 reimplements the cross-map from scratch in numpy (E+1 neighbours, no tie expansion), with the full deterministic library (`random=False`). On tie-free data it reproduces pyEDM to about 1e-16:
- `proc0:proc1` = 0.972222487974537 from pyEDM and 0.9722224879745365 from numpy (E=1).
- E=3 gives the same agreement.

The agreement does not hold when neighbour distances tie. The review probe `fable_dt3/fm2.py` (in the scratchpad, T=256, `random=False`) used the same plain E+1 reference:
- Every column of `var2_bivariate`, `iid_gaussian` and `nonlinear_dependence`, and `logistic_maps` `proc0:proc1`, agree to within 3.3e-16 at E=1 and E=10. None of these rows has a tie.
- `logistic_maps` `proc1:proc0` embeds process 1, the period-4 driver with only 4 distinct values. Every prediction row has a tie at the (E+1)-th neighbour distance (256 of 256 at E=1, 247 of 247 at E=10). pyEDM gives 0.412996812927705 against 0.408547833137669 for the reference at E=10, and 0.258036085716451 against 0.016209343992146 at E=1.

The re-check probe `fable_recheck_cdt_ccm_gwtau/ccm_tie_probe.py` (same data, full library) adds a tie-expanded reference that implements cppEDM's Simplex tie rule. `pyEDM.Simplex(lib=pred=all, Tp=0)` reproduces it (0.0 difference at E=10, 1.7e-16 at E=1), but `pyEDM.CCM` does not: 0.412996812927705 against 0.438997955936925 at E=10, and 0.258036085716451 against 0.449407364219300 at E=1. Tie-free columns match both references to within 3.3e-16.

So on `logistic_maps` the whole `ccmf[1,0]` curve depends on the order `std::sort` gives to equal distances. CCM does not expand ties (`EDM_Neighbors.cc` L282), and no order-independent reference can reproduce this column.

### Direction convention (candidate, to be confirmed)
`A[i,j]` summarises ρ(x_j | M_{x_i}), the skill of predicting j from i's embedding. Under Sugihara's reading, a large `A[i,j]` is evidence that j causes i (j → i). pyspi's generic `Directed` convention is the opposite: `A[i,j] = bivariate(i, j)` (`base.py:91-93`), and transfer entropy computes "from i->j" (`infotheory.py:377-383`). So ccm's orientation is the transpose of the TE/GC orientation.

Probe p2 used `logistic_maps` at T=256, where process 1 drives process 0:

| E | mean, A[0,1] | mean, A[1,0] | diff, A[0,1] |
|---|---|---|---|
| 1 | 0.5484 | 0.2913 | +0.2571 |
| 2 | 0.7415 | 0.3226 | +0.4188 |

Caveat: in this dataset the uncoupled driver y has r=3.5, which gives a period-4 orbit (EmbedDimension ρ = 1.0 at every E for process 1). Because of that orbit, every prediction row of `A[1,0]` has tied neighbour distances, so the column depends on `std::sort` order among equal distances (CCM does not expand ties; see the trace confirmation), and no order-independent reference reproduces it. The check stage should also use a chaotic-driver system.

### Shared deviations and candidates
- **Library schedule.** The curve has 22 points, not 21. At short T it is truncated: cppEDM returns more sizes and pyspi keeps the first 22. In probe p4, `"20 50 1"` (T=64, E=10) gave 31 sizes, and pyspi kept 20..41 rather than 20..50. Analytically:
  - T=64, E=1: keeps 10..52 of 10..60.
  - T=256, E=10: keeps 20..230 of 20..240.
  - For T ≥ 1024, all 22 sizes are kept.
- **Short-series failure.** When `inc = 0`, cppEDM throws `RuntimeError Parameters::Validate(): CCM librarySizes increment 0 is invalid.` (probe at T=50 and T=55 with E=10). Analytically this happens for T ≤ 60 (E=10, and therefore E-None) and T ≤ 41 (E=1). `Calculator.compute` then stores an all-NaN matrix (`calculator.py:336-339`, warning at 354-355; HEAD a42ebc1, file unchanged since a1f20c7). The comment at `tests/unit/test_known_bugs.py:88` gives "70 observations" as the approximate minimum.
- **No temporal exclusion.** Only the row itself is excluded, and τ is fixed at 1 sample. Hypothesis: for autocorrelated or oversampled signals such as EEG, temporally adjacent neighbours inflate ρ.
- **RNG.** `std::default_random_engine` and `std::uniform_int_distribution` are implementation-defined. Hypothesis: values may differ between C++ standard libraries. Both directions and every pair use seed 42, so both directions share the same library draws.
- **Diagonal.** A direct `.multivariate` call returns 0 on the diagonal (`ccmf` starts as zeros). Only the Calculator sets it to NaN.
- **Signed vs unsigned labels.** The documented keyword and `config.yaml` say "unsigned", but the class is `Signed`. The code is the consistent side, because ρ and `diff` are negative in practice.
- **`diff`.** It is not defined in the supplement. The code defines it as mean over L of ρ(x_j|M_i) − ρ(x_i|M_j), which is antisymmetric. This is a pyspi construct, not part of Sugihara 2012.
- **Convergence not tested.** Mean and max over L replace the convergence test. This is documented. The mean includes the small-L part of the curve.

## causal.ccm.E-None

- **Documented:** "we infer the embedding dimension to be the maximum of the two univariate delay embeddings that best predicted each time series" (supplement §1.3.5, p.9).
- **Implemented** (`causal.py:98-111`, `CoreEDM.py:459-526`, cppEDM `Eval.cc` L132-207): for each process, pyspi runs `EmbedDimension` with Tp=1, lib = pred = rows 10..N-10, and E = 1..10. It then takes `embed_df.max()["E"]`, the column maximum of E, which is always 10. This is known bug 2 (tests/README.md). For each pair, E = max(10, 10) = 10.
- **Probe p3** (T=256) compares the E pyspi picks with the E that maximises ρ:

  | Series | pyspi E | argmax-ρ E |
  |---|---|---|
  | logistic x | 10 | 1 |
  | iid, process 0 | 10 | 9 |
  | iid, process 1 | 10 | 3 |
  | var2, process 1 | 10 | 7 |

- **Probe p7:** on `var2_bivariate` at T=256, `ccm_E-None_*` is bitwise equal to `ccm_E-10_*` for `mean` and `diff`. This restates known bug 2 and is recorded under `already_documented`, not as a new candidate. The `EmbedDimension` runs (10 simplex fits per process) are computed and then discarded; their cost belongs to the optimisation stage.
- **Deviations:**
  - From the documentation: the chosen E is the maximum, not the best-ρ E (already documented).
  - From the canonical method: E is fixed at 10 instead of being chosen by simplex forecast skill.
  - Interpretation, not a sourced deviation: the documented rule uses one E = max(E_i, E_j) for both directions. The supplement (p.9) attributes that rule to the Sugihara 2012 Supplementary Materials and the pyEDM documentation, and pyEDM takes one E per CCM call. A per-direction E (the E of the embedded variable) is a possible alternative, but no Sugihara 2012 passage prescribing it was located.
- **Proposed checks:**
  - Shared checks (below).
  - A from-scratch simplex E-selection (Tp=1) must equal pyEDM's ρ(E) table. It will confirm that pyspi's E (10) differs from argmax-ρ.
  - Bitwise equality with E-10.

## causal.ccm.E-1

- **Implemented:** E=1, knn=2. There is no delay coordinate, so each cross-map is a 2-nearest-neighbour, exponentially weighted regression of x_j(t) on x_i(t). Library sizes run from 10 to floor((T-2)/10)*10.
- **Deviation from canonical:** this is not a Takens reconstruction (that needs E > 2d). It scores static, instantaneous functional dependence.
- **Candidate:** probe p5 used iid data with x1 = x0² + noise. It gave ρ(x1|x0) = 0.972 and ρ(x0|x1) = −0.032, a large asymmetric "causal" score with no dynamics in the data.
- **Proposed check:** run on `nonlinear_dependence` and expect `A[0,1] ≈ 1` and `A[1,0] ≈ 0` (qualitative).

## causal.ccm.E-10

- **Implemented:** E=10, τ=1, knn=11. Library sizes run from 20 to floor((T-11)/10)*10. E-None produces identical numbers.
- **Deviation from canonical:** E and τ are fixed rather than chosen from the data.
- **Behaviour note (not a code defect):** probe p7 used `var2_bivariate` at T=256, a stochastic linear system where process 0 drives process 1.
  - Result: mean = [[0, 0.3208], [0.0774, 0]] and diff[0,1] = +0.2434.
  - Under the code orientation this reads as 1 → 0, which is the wrong direction. This matches known CCM limitations for stochastic or strongly forced linear systems.
- **Proposed check:** a from-scratch cross-map with and without a Theiler window, run on `gauss_ar1_pair` and `narrowband_f0-0.1`. It measures how much of the skill comes from adjacent-time neighbours.

## Proposed checks (shared; details in the records)

| # | What | Datasets | Path | Reference | Criterion | Expected |
|---|---|---|---|---|---|---|
| 1 | Cross-map kernel | logistic_maps, nonlinear_dependence, var2_bivariate | kernel | from-scratch numpy, full library with `random=False`; E+1 neighbours, zero-distance weight rule, no tie expansion | exact | < 1e-12 on tie-free columns; `logistic_maps` `proc1:proc0` (ties) depends on `std::sort` order, not reproducible by an order-independent reference, reported separately |
| 2 | Random-library curve | logistic_maps, iid_gaussian | kernel | numpy RNG libraries; optional libstdc++ minstd_rand0 and uniform_int replica for an exact match | statistical | within 4√2·SE |
| 3 | Aggregation and schedule | iid_gaussian (T=64 exercises truncation), var2_bivariate | pipeline | numpy on `data.ccm[E]` plus the analytic schedule | exact | diff = mean − meanᵀ; first-22 schedule |
| 4 | Direction convention | logistic_maps, logistic_maps_bidir; var2_bivariate with TE/GC for pyspi's convention | pipeline | Sugihara criterion on known coupling | direction | A[0,1] > A[1,0], i.e. the transpose of the TE orientation |
| 5 | Null | iid_gaussian T≥1024 | pipeline | analytic ρ→0 | statistical | small values, diff ≈ 0 |
| 6 | Affine invariance | bivariate_gaussian vs scaled_offset | kernel | analytic | tolerance | equal within 1e-10 |

## Open questions
1. **Intended orientation.** Is ccm meant to follow pyspi's `A[i,j] = i → j` convention? If so, the current matrix is transposed: by the code orientation, a large `A[i,j]` is evidence for j → i. The check stage must establish this against TE/GC on the same data before it is called a finding.
2. **Chaotic-driver test data.** `logistic_maps` has a period-4 driver (r_y = 3.5, uncoupled). A direction check on it is weak. Should the check stage generate Sugihara's chaotic setting (r_x = 3.8, r_y = 3.5, β_xy = 0.02, β_yx = 0.1)? That is not in MANIFEST.
3. **Fixing E-None.** If E-None is fixed to argmax-ρ, should each direction use its own embedded variable's E (an interpretation, not a sourced canonical rule) or max(E_i, E_j) (documented, and attributed by the supplement to the Sugihara 2012 Supplementary Materials)?
4. **The 22-point truncation.** Is it intended (21 intervals → 22 points)? At short T it silently caps the largest library below `upperE`.
5. **Cross-platform reproducibility.** The C++ RNG is implementation-defined, so CCM values may differ across platforms. A second implementation-defined element is the order of equal distances after `std::sort` (not stable), which picks the E+1 neighbours on tied columns such as `logistic_maps` `proc1:proc0`, because CCM does not expand ties (`EDM_Neighbors.cc` L282). This is unconfirmed: only one Linux build was examined. pyEDM 2.x is known to change values (DEPENDENCY_COMPATIBILITY_REVIEW.md:88-90).
6. **Temporal neighbours.** How much does the missing Theiler window inflate ρ on EEG-like autocorrelated data (hypothesis)?

## Checks

Files:
- Manifest: `audit/checks/manifest/ccm.json` (115 runs).
- Probes: `audit/checks/probes/ccm_probes.py` (project venv).
- References: `audit/checks/refs/ccm.py` (comparison venv; it never imports pyEDM).

Results are in `audit/checks/out/results/ccm.json`: 102 runs pass, 13 fail, and every run has a reference. All 13 failures are explained below. Eight are direction runs (method behaviour), four are single-seed statistical excursions, and one is the expected known-bug-2 run.

Throughout, `A[i,j] = rho(x_j | M_{x_i})`. pyspi z-scores the input (`pipeline`); the `kernel` path passes the raw array straight to pyEDM.

The reference in every exact run is `_kernels_distance.simplex_crossmap`: a simplex projection from the papers, with E+1 neighbours ordered by stable argsort, weights `exp(-d/d1)` floored at 1e-6, and no tie expansion. The statistical references use `refs/ccm.py:XMap`. It does the same arithmetic on a precomputed distance matrix and equals `simplex_crossmap` to 0.0 in every run (`ref_xmap_vs_kernel_max_abs`).

### Delegate kernel (exact, independent-impl)

The probe `kernel_xmap` calls `pyEDM.CCM(random=False, libSizes="<L>", sample=1)` directly, using the DataFrame layout of `causal.py:91-96`. It reads both columns of one call.

- **logistic_maps_chaotic and logistic_maps_sugihara, T=256.** E ∈ {1, 2, 3, 4, 10}, with L ∈ {100, full}. All 20 runs pass at 1e-10. The largest error is max_abs 1.94e-14 (`causal.ccm.E-10|logistic_maps_chaotic|256|kernel|xmap-E10-L100`). No prediction row has a tied (E+1)-th distance.
  - At E=2 with the full library, `y xmap x` = A[1,0] = 0.8824 and `x xmap y` = A[0,1] = 0.2909. This is the kernel selftest's direction.
  - E=3 gives 0.8479 vs 0.2694 (chaotic).
  - E=10 gives 0.3930 vs 0.3714.
- **Other datasets.** var2_bivariate, nonlinear_dependence (4 processes) and iid_gaussian at E = 1 and 10, plus nonlinear_dependence at T=1024 with E=1: all pass, max_abs ≤ 3.4e-15.
- **Exclusion radius.** With `exclusionRadius = 10` (cppEDM rule |pred row − lib row| ≤ X, `EDM_Neighbors.cc` L173-178), the gauss_ar1_pair and narrowband_f0-0.1 runs also match `XMap(excl=10)` exactly (≤ 4.4e-16).
- **Affine invariance (tolerance, analytic).** pyEDM on raw scaled_offset, processes 2 and 3 (scale 5 and 10, offsets −2 and 5), equals the kernel on the same bivariate_gaussian draw. max_abs is 6.7e-16 at E=1 and 1.2e-15 at E=10.
- **Tied fixture logistic_maps (qualitative).** The tie-free column A[0,1] matches to 1.3e-14. The tied column A[1,0] is tied on all 256 rows at E=1 and all 247 rows at E=10.
  - At E=1, pyEDM gives 0.2580 against the stable reference's 0.0162.
  - At E=10, pyEDM gives 0.41455 on the raw input (`...|kernel|tied-E10-Lfull`) and 0.41300 on the z-scored input (`...|pipeline|tied-E10-Lfull`). The stable reference gives 0.40855 on both.
  - So on tied data the value depends on the rounding of the input, through the order `std::sort` leaves equal distances in. CCM is not affine-invariant there.
  - The kernel workflow's libstdc++ introsort emulation (scratchpad `kernels/probe_ccm_stdsort.py`) reproduces pyEDM's tied column to ≤ 1.1e-13. That confirms the mechanism.
  - The trace-stage figure 0.412996812927705 was the z-scored value.

### Library-size grid and wrapper conventions (exact / qualitative, analytic)

The probe `schedule` intercepts the `pyEDM.CCM` call that pyspi makes. The schedule it records equals `refs/ccm.py:schedule` exactly (max_abs 0.0) at T = 64, 256 and 1024 for E = 1, 10 and None.

| T | E | libSizes string | cppEDM expands to | kept by pyspi |
|---|---|---|---|---|
| 64 | 1 | `10 60 2` | 26 sizes | 10..52 |
| 64 | 10 | `20 50 1` | 31 sizes | 20..41 |
| 256 | 1 | `10 250 11` | 22 sizes | all (10..241) |
| 256 | 10 | `20 240 10` | 23 sizes | 20..230 |
| 1024 | 1 | `10 1020 48` | 22 sizes | all (10..1018) |
| 1024 | 10 | `20 1010 47` | 22 sizes | all (20..1007) |

The grid always has 22 points. `inc = floor((upperE − lowerE)/21)` always yields at least 22 sizes, and pyspi keeps the first 22.

The runs `short-E1-T41`, `short-E10-T60` and `short-ENone-T60` pass. cppEDM throws "increment 0 is invalid" at T=41 (E=1) and at T=60 (E=10 and None), and computes normally at T=42 and T=61. The runs `diag-*` pass: a direct `multivariate` call returns a diagonal of exactly 0.

### Aggregation of the cached curve (exact, analytic)

The probe `aggregate` computes `ccm_E-*_mean`, `_max` and `_diff` on one Data object and returns them together with the cached curve `data.ccm[E]` they read, as (P, P, 3 + 22). `refs/ccm.py:run_agg` recomputes the three statistics in numpy from that curve: `nanmean(c, 2)`, `nanmax(c, 2)` and `nanmean(c - c^T, 2)` (`causal.py:152-163`). This is proposed check 3; the earlier draft of this stage ran only its schedule half.

| Run | Data | Result |
|---|---|---|
| `causal.ccm.E-1|iid_gaussian|64|pipeline|agg-E1` | 5 processes, T=64 (truncated grid 10..52) | pass, max_abs 0.0 |
| `causal.ccm.E-1|var2_bivariate|256|pipeline|agg-E1` | T=256 | pass, max_abs 0.0 |
| `causal.ccm.E-10|iid_gaussian|64|pipeline|agg-E10` | 5 processes, T=64 (truncated grid 20..41) | pass, max_abs 0.0 |
| `causal.ccm.E-10|var2_bivariate|256|pipeline|agg-E10` | T=256 | pass, max_abs 0.0 |
| `causal.ccm.E-None|iid_gaussian|64|pipeline|agg-ENone` | as E-10 | pass, max_abs 0.0 |
| `causal.ccm.E-None|var2_bivariate|256|pipeline|agg-ENone` | as E-10 | pass, max_abs 0.0 |

In every run the curve has 22 entries, `diff` equals `mean − mean^T` to ≤ 5.6e-17 (`ref_diff_vs_mean_minus_meanT`), and `diff + diff^T` is exactly 0.

**Symmetry observed** (max|A − A.T| on the pyspi outputs of the check runs; `diff` is exactly antisymmetric, max|A + A.T| = 0.0, in every diff, stats3 and agg run):

| Unit | mean | max |
|---|---|---|
| E-1 | 0.0071 (iid T=256) to 0.9699 (nonlinear_dependence `static-mean`) | 0.0009 (iid T=256) to 0.2801 (chaotic T=64 `stats3-E1`) |
| E-10 | 0.0034 (chaotic T=256) to 0.6432 (chaotic T=1024 `stats3-E10`) | 0.0147 (chaotic T=256) to 0.7684 (chaotic T=1024 `stats3-E10`) |
| E-None | 0.0034 (chaotic T=256) to 0.3981 (iid T=64 `agg-ENone`) | 0.0147 (chaotic T=256) to 0.4336 (iid T=64 `agg-ENone`) |

The earlier record text quoted one range for mean and max together and missed the max end (0.7684 for E-10) and, for E-1, the static-mean run.

### Wrapper (statistical, independent-impl)

**Reference construction.** For each pyspi identifier (sample=100, seed 42, pyspi's grid), the reference draws 500 random libraries per size from the same N−E+1 embedding rows that cppEDM samples from (`CCM.cc` L175-176). It draws without replacement and shares each library between both directions, as cppEDM's two engines with the same seed do. It then bootstraps 4000 curves, each the average of 100 libraries per size. The reference value is the bootstrap mean, and se = bootstrap sd × √(1 + 100/500).

**Single-seed results.**
- The family has 58 statistical cells (23 runs; the E-None runs duplicate E-10). Four exceed 2 se, and the signed z values have mean 0.10 and sd 0.98 (ddof 1; 0.97 with ddof 0).
- E-10 and E-None (9 runs each, T=256) all pass, with max |z| 1.15. Example: chaotic mean is 0.2844 / 0.2810 from pyspi against 0.2833 / 0.2801 from the reference.
- E-10 at T=64 passes (|z| ≤ 1.48). At T=1024 (`stats3-E10`), 5 of 6 cells have |z| ≤ 0.99; mean A[1,0] gives 0.59207 against 0.59318, |z| 2.06.
- E-1: 8 of 11 runs pass. The excursions are:
  - iid stat-mean: |z| 2.99 (−0.0306 vs −0.0340).
  - chaotic stat-mean: |z| 2.16.
  - sugihara stat-max: |z| 2.04.

**Bias test.** A single seed cannot tell a fluctuation from a bias, so the probe `seeds` repeats pyspi's exact call with seeds 1..16 (1600 libraries per size). The reference has 2000 draws per size and an analytic se.
- `seeds-E1-S16` iid: −0.02414 / −0.03339 vs −0.02417 / −0.03292 (|z| 0.08, 1.34).
- chaotic: 0.17971 / 0.21851 vs 0.17944 / 0.21787 (|z| 0.70, 1.55).

So the random-library path has no systematic bias, and the four seed-42 excursions are sampling fluctuation. I did not widen the tolerance; those four runs stay marked as fails in results.json.

**Curves.** The pre-declared rule is max|z| ≤ 4 and χ²(44) p > 0.001 over 2 × 22 entries. All four curve runs pass: chaotic E=1 2.29 / 0.53; sugihara E=1 2.50 / 0.49; chaotic E=10 1.76 / 0.78; sugihara E=10 2.32 / 0.45.

**Library-size grid used.** The reference uses pyspi's exact grid (`refs/ccm.py:schedule`). The `schedule` runs check that grid against the string pyspi actually passes.

### E=None

- `embed-ENone` (exact): the from-scratch Tp=1 simplex table for E = 1..10 equals pyEDM.EmbedDimension to ≤ 1.0e-15 on chaotic, var2 and iid. The library is rows 9+(E−1)..N−12 and the predictions are rows 9..N−11 (0-based); these index ranges were found by matching the output, not from source.
  - pyspi passes E = 10 for every pair.
  - The argmax-ρ E per process is [1, 2] (chaotic), [10, 7] (var2) and [9, 3] (iid).
  - This confirms known bug 2 (tests/README.md), which is already documented.
- `eqE10-mean` (same-lineage): on chaotic and sugihara, ccm_E-None_mean is bitwise equal to ccm_E-10_mean.
- `stat-mean-docE` fails, as expected under known bug 2. The documented rule gives E = 2 on chaotic, and the reference at E = 2 is A = [[·, 0.1992], [0.6873, ·]]. pyspi gives 0.2844 / 0.2810 (max_z 408.6). At the documented E the direction x → y is clearly present (A[1,0] ≫ A[0,1]); at pyspi's E = 10 it is not.

### E=1 behaviour

`static-mean` passes its pre-declared rule (A[0,1] > 0.7 and |A[1,0]| < 0.2). On nonlinear_dependence, where samples are iid and x1 = x0² + noise, A[0,1] = 0.9672 and A[1,0] = −0.0027. The kernel with the full library gives 0.9768 / −0.0073 at T=256 and 0.9763 / −0.0219 at T=1024.

### Direction (direction criterion, analytic, CONVENTIONS.md D8)

Under the CCM criterion a cause c → effect e should give diff[e, c] > 0.

| Unit | Dataset (T=256 unless noted) | diff[e,c] | Outcome |
|---|---|---|---|
| E-1 | chaotic (x → y), processes in order | +0.0360 | pass |
| E-1 | chaotic, processes swapped | +0.0360 | pass |
| E-1 | sugihara | +0.2566 | pass |
| E-1 | bidir | −0.0997 | fail |
| E-1 | var2 | −0.0369 | fail |
| E-10 | chaotic (both orders) | −0.0034 | fail |
| E-10 | sugihara | −0.1086 | fail |
| E-10 | bidir | +0.1233 | pass |
| E-10 | var2 | −0.2434 | fail |
| E-10 | chaotic, T=1024 (stats3) | +0.6432 | right direction |
| E-None | chaotic, sugihara | as E-10 | fail |

What these results establish:
- **Orientation.** The swapped-input runs give exactly the transposed matrix, and the T=1024 and E=2 kernel results agree. Together they settle the orientation: **row = target**. A large A[i,j] means j → i.
- **This contradicts the glossary.** `glossary-of-terms.md:25-27` says rows are sources, and TE is row = source (base.pair_handling `calls-gc_gaussian`). This is a confirmed finding not listed in KNOWN_ISSUES.md.
- **The failed direction runs are method behaviour.** The kernel shows the same asymmetries, e.g. var2 at E=10 with the full library gives 0.3054 vs 0.1282. The causes are E=10 over-embedding a one-dimensional map at T=256, and CCM's known weakness on stochastic linear coupling.

### Verdicts

| Unit | Verdict | Score | Reason |
|---|---|---|---|
| causal.ccm.E-1 | variant | 6 | The kernel is fp-exact against an independent implementation, the curve is within sampling error and the aggregation is exact. The output still deviates from both the supplement and canonical CCM in interpretable ways (anchor 6). (1) It reports the supplement's mean/max summary, not convergence, and `diff` is pyspi's own construct. (2) The grid has 22 library sizes, not the supplement's 21 between the bounds, and is cut short of the maximum when cppEDM expands to more than 22. (3) Rows are targets, the transpose of the documented rows-are-sources convention (`glossary-of-terms.md:25-27`), so a matrix read as documented gives the direction backwards. |
| causal.ccm.E-10 | variant | 6 | Same reasons as E-1 (at T=256 the grid is 20..230, not 20..240). |
| causal.ccm.E-None | related-quantity | 4 | Anchor "CCM max-E selection"; known bug 2. |

Score revision after the check review: E-1 and E-10 were first scored 7 (anchor "matches the supplement variant"). Points (2) and (3) are confirmed deviations from the documentation, so the output does not match the supplement variant either. Both units are now at 6. The kernel being fp-exact does not change this.

All three units have `new_finding: true`, for the row = target orientation.

Candidate outcomes:
- **Confirmed:**
  - Orientation.
  - 22-size and truncated grid.
  - Short-series failure; the exact thresholds are T ≥ 42 for E=1 and T ≥ 61 for E=10/None.
  - No Theiler window, quantified: on the AR(1) pair a 10-row window lowers ρ by 0.129 / 0.110, which is 24-36 % of the skill. On the sinusoid the change is < 1e-4.
  - Zero diagonal from direct calls.
  - Signed vs unsigned labels.
  - E-1: static dependence is scored as causality.
  - E-10: behaviour on var2.
- **Inconclusive:**
  - Cross-platform reproducibility. In-build sensitivity of tied columns is confirmed.
  - E-None: the per-direction E design question.
- **Refuted:** none.

**Trace-stage corrections.**
- Calculator citations are refreshed to HEAD (`calculator.py:327-339`, 332, 336-339, 354-355). The file is unchanged between a1f20c7 and HEAD a42ebc1.
- `direction_convention` now carries the check-stage confirmation.
- The tie-mechanism issues listed for this family were already corrected in the records and in this note before this stage (`EDM_Neighbors.cc` L282 gate). I re-read and kept them.

## Open questions: answered

1. **Intended orientation.** Answered. ccm stores row = target: a large A[i,j] means j → i. On the swapped chaotic input, diff[effect, cause] = +0.0360 in both orders (E=1), and at T=1024 E=10 A[1,0] = 0.592 vs A[0,1] = −0.051. This is the transpose of the glossary convention (`glossary-of-terms.md:25-27`) and of TE / gc_gaussian (row = source, base.pair_handling runs). Whether to transpose the output or document the convention is for the maintainers.
2. **Chaotic-driver data.** Settled by the data stage. `logistic_maps_chaotic` (x → y only) and `logistic_maps_sugihara` (the Fig. 1 parameters) are now in MANIFEST. Both are tie-free (0 tied rows in every kernel run) and were used for every direction and statistical check here.
3. **Per-direction E vs max(E_i, E_j).** Not decidable numerically. The checks show only that the two rules would differ on every dataset tried (argmax E [1, 2], [10, 7], [9, 3]) and that either would differ from pyspi's fixed 10. At the documented E = 2 the chaotic direction is recovered (0.687 vs 0.199). The maintainers have to choose the rule.
4. **The 22-point truncation.** Characterised exactly (see the table above). The grid always has 22 points, never 21. Truncation happens whenever cppEDM expands to more than 22 sizes: at T=64 for both E, and at T=256 for E=10 (largest library 230 instead of 240). At T=1024 nothing is cut. The code cannot tell us whether 22 points are intended. It allocates `ccmf` with `nlibs + 1 = 22` slots on purpose (`causal.py:116-117`), but the supplement says 21. Only the maintainers can settle which was meant.
5. **Cross-platform reproducibility.** Partly answered.
   - Ties: cppEDM's tie expansion never runs on the CCM path. The kernel docstring and the `EDM_Neighbors.cc` L282 gate establish this, and every tie-free column here matches the order-independent kernel to ≤ 1.94e-14.
   - On tied data, the neighbour choice follows libstdc++'s introsort order, which an emulation reproduces to 1.1e-13. It changes with input rounding even within this build: 0.41455 on the raw logistic_maps input and 0.41300 z-scored, against 0.40855 for the stable reference.
   - A different standard library (libc++, MSVC) could therefore change tied columns. It could also change the random libraries, since `std::default_random_engine` and `uniform_int_distribution` are implementation-defined.
   - Settling it needs a run on a second toolchain, which was not available.
   - For tie-free data the random-library path is statistically indistinguishable from an independent sampler (16-seed bias test, |z| ≤ 1.55).
6. **Temporal neighbours.** Quantified on two signals at E=10, T=256, full library.
   - AR(1) pair: a 10-row Theiler window lowers ρ from 0.363 / 0.455 to 0.234 / 0.345.
   - Pure sinusoid: unchanged (< 1e-4), because its nearest neighbours lie on other cycles anyway.
   - For EEG-like broadband autocorrelated signals expect behaviour closer to the AR(1) case. A run on real or EEG-like data would settle the size.
7. **Minimum length (test comment "about 70").** The exact thresholds are T ≥ 42 for E=1 and T ≥ 61 for E=10 and E=None (`short-*` runs, pass). The comment at `tests/unit/test_known_bugs.py:88` is approximate.

## Optimisation

Workflow 3 (profile-based search for hotspots and avenues; nothing in `pyspi/` was changed). Walls from `audit/profiles/summary.json`; CCM was re-measured quietly at 7x100 and 8x256 with a 7x100 warm-up (PROFILE_WARMUP_SHAPE), and 8x1024 was skipped-budget, so summary.json carries no 118 projection. The cProfile run of profile_units timed out at 4x256 (its 8x1024 warm-up alone takes about half an hour per identifier); this group profiled each unit itself at 4x256 after a 3x100 warm-up (`audit/profiles/micro/causal.ccm.<E>__4x256.pstats` and `__profile.json`, OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1, written with cProfile around harness.compute_spi).

| unit | 7x100 (fresh / family-warm) | 8x256 (fresh / family-warm) | 118x1024 per window | basis |
|---|---|---|---|---|
| causal.ccm.E-None | 14.51 s / 14.52 s | 100.4 s / 100.0 s | about 2.8e5 s (ESTIMATE) | per-pair T exponent 1.75 |
| causal.ccm.E-1 | 12.38 s / 12.33 s | 93.3 s / 93.1 s | about 3.0e5 s (ESTIMATE) | per-pair T exponent 1.84 |
| causal.ccm.E-10 | 14.45 s / 14.38 s | 99.2 s / 99.6 s | about 2.8e5 s (ESTIMATE) | per-pair T exponent 1.75 |

Estimate method (labelled estimate, not a measurement): per-ordered-pair cost at 7x100 (wall / 42) and at 8x256 (wall / 56) gives a T exponent of 1.84 (E-1), 1.75 (E-10, E-None); the 8x256 per-pair cost scaled to T=1024 with that exponent is 21.4 s (E-1), 20.2 s (E-10), 20.1 s (E-None) per ordered pair, times 13,806 ordered pairs at P=118. The two shapes use different data (7x100 is the golden cml7 set, 8x256 synthetic), and the check stage measured one E=10 pair at T=1024 at 85 s alone, which implies a T exponent of about 2.3 between 256 and 1024 and about 5.9e5 s per window; the table values are therefore a lower-end estimate. Per 600-window patient: about 1.7e8-3.5e8 s per unit. The code loops over unordered pairs with one pyEDM.CCM call each (both directions), so pair scaling is quadratic even though units.yaml lists pair_basis multivariate.

Hotspots (all three units): `~:<built-in method pyBindEDM.CCM>` (C++ cppEDM via `pyEDM/CoreEDM.py:CCM`) is 99.7% (E-1), 99.9% (E-10) and 98.4% (E-None) of the profiled SPI call at 4x256, six calls (one per unordered pair); E-None adds `pyBindEDM.EmbedDimension` 1.5% (four calls, one per channel). Opaque share 0.997-0.999: cProfile sees nothing inside the C++ code, so the split between neighbour search and cross-map arithmetic is a complexity argument, not a profile fact.

Shared versus incremental cost: mean, max and diff of one E share one cached ccmf (family-warm increments 0.0001 s for _max and _diff after _mean). E-None and E-10 do not share, although the check stage found them bitwise equal: the cache key is the requested E (None vs 10). The family-warm run follows pyspi's config order (E-None, then E-1, then E-10; pyspi/config.yaml:378-393, `causal.ccm.E-None__8x256__family-warm.json`): ccm_E-None_mean costs 100.0 s first and ccm_E-10_mean still costs 99.6 s afterwards. (An earlier version of this note had the order reversed.)

Opportunities:
- ccm-1 (all three): process pool over the unordered-pair loop (pyspi/statistics/causal.py:118-144); each call is seeded (seed=42). Each call already runs its two cross-map directions in two std::threads (cppEDM CCM.cc:80-90, CCM_THREADED); the probe `audit/profiles/micro/probe_ccm_threads.py` / `.json` measured CPU/wall 2.00 for one call at T=256 (3.65 s wall, 7.29 s CPU, OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1). On the 32 physical cores (Threadripper 3970X, 64 logical CPUs) that leaves at most about 16 concurrent calls: expected 8-16x per-window latency on its own, with no core-hour saving. Window-level parallelism (xc-window-parallel) competes for the same cores and gives the same throughput. The two are alternatives, not factors that multiply, so the record does not list it in depends_on; inside one of the ~8 memory-bounded window workers (about 4 physical cores each) the pool runs about two two-thread calls at a time, at most about 2x. The pair-level split is not measured. Output risk unverified-bit-identical.
- ccm-2 (E-10): key the cache by the resolved embedding (the per-channel E actually used) instead of the requested E. In the config order E-None computes first, so ccm_E-10_* becomes the cache hit: its `_from_cache` is 99.999% of the profiled call at 4x256 (pyBindEDM.CCM 99.93%, Amdahl bound about 9e4x), and only nanmean/nanmax over the cached curve remains. Expected >1000x for causal.ccm.E-10 at 118x1024. Output risk unverified-bit-identical, from the check-stage bitwise equality; the avenue disappears if the E-selection rule (known bug 2) is changed. The saving goes to whichever of E-None and E-10 runs second on a Data. In the config order that is E-10, so ccm-2 is listed only on causal.ccm.E-10; E-None would gain (Amdahl 66x at 4x256, CCM 98.4% against EmbedDimension 1.5%) only if E-10 ran first. An earlier version credited ccm-2 to E-None from a misread run order.
- ccm-3 (all three): with seed 42, the same library sizes and the same E, the random libraries and hence the neighbour sets and weights of a call depend only on the source column. The cppEDM source used by the check stage confirms this: CrossMap seeds its own std::default_random_engine with the seed in each direction (CCM.cc:114, 203) and draws row indices from uniform_int_distribution(RNGStart, RNGStop) (CCM.cc:175-176, 220) into an unordered_set (CCM.cc:274-285), so the draws depend on the seed, the library-row count (N, E, tau) and the library size, not on the data. Therefore a re-implementation could search neighbours once per source channel and reuse them for all targets: expected 10-50x, effort L. Output risk unverified-rng-stream: reproducing cppEDM's library draws outside cppEDM is the hard part, and draws that differ change the curves within sampling error.

Semantic note: `sample=100` (random libraries per size) and the 22 library sizes are part of the definition; reducing either would be a semantic change and is not proposed. Trimming the libSizes string to the 22 kept sizes was considered and refuted at T=1024: the rounding yields exactly 22 sizes there (E=1: 10..1018 step 48; E=10: 20..1007 step 47), so nothing is computed and discarded (one extra size is discarded only at small T, e.g. 240 at T=256).
