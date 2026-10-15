# Barycenters family (`distance.bary.*`): define, trace and check

Units: `distance.bary.euclidean`, `distance.bary.dtw`, `distance.bary.sgddtw`, `distance.bary.softdtw`.
Each unit covers four identifiers, `bary[-sq]_<mode>_{mean,max}`. That makes 16 in `pyspi/config.yaml:255-311`. Only the Euclidean unit is in the EEG selection.
Library: tslearn 0.9.0 (scipy 1.17.1, scikit-learn 1.9.1, numba 0.68.0). Probe scripts are in
`/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/scratchpad/barycenters/probe{1..4}.py`.
The sections before `## Checks` are the trace stage: their probe numbers describe what the code does on the audit datasets. They are not verifications. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Shared pyspi path (all four units)

* `pyspi/statistics/distance.py:211-263` defines `class Barycenter(Directed, Signed)`. `__init__` maps `mode` to a tslearn function (lines 218-227). `squared=True` adds a `x**2` preprocessing step and the identifier prefix `bary-sq` (230-234). The statistic is `np.nanmean` or `np.nanmax` of the (squared) barycenter (236-239). A third statistic, `max_time` (`np.argmax`, 240-241), exists but no config uses it.
* `bivariate` (247-263) calls the library with **library defaults** on `z[[i, j]]`. It caches the result on the `Data` object under both `(i, j)` and `(j, i)` (line 261), keyed only by `mode`. So within one Calculator run, the mean, max and squared variants of a mode share one barycenter per pair (`calculator.py:329` passes the same `self.dataset` to every SPI).
* `Directed.multivariate` (`base.py:85-94`) loops with `j` outer and `i` inner. For a pair `{a<b}` the barycenter is therefore computed from `z[[b, a]]`; probe3 shows cache insertion order `(1,0), (0,1), (2,0), ...`. The output matrix is exactly symmetric because of the cache, not because of `Undirected` mirroring.
* Preprocessing: `Data` defaults to `normalise=True`, `detrend=False` (`data.py:59-71`). It applies `scipy.stats.zscore(axis=1, ddof=1)` (`data.py:193-195`). Each row then has mean ~1e-17 and `sum z^2 = T-1`.
* Labels: the class labels are `['distance','signed','undirected','temporal','nonlinear']` and `issigned()` is True. The supplement, GitBook and config all say "unsigned". The config labels are never attached (`calculator.py:252-266`).

### Documented definition
Supplement §1.2.10, p.7: "A barycenter (or Fréchet mean) [21, 22] is a (univariate) time series that minimizes the sum of squared distances between MTS ... we compute the raw barycenters then summarize them by taking their mean (modifier mean) and maximum (max), yielding seven SPIs." The seven listed are `bary_euclidean_max`, `bary_dtw_{mean,max}`, `bary_softdtw_{mean,max}` and `bary_sgddtw_{mean,max}`. The supplement omits `bary_euclidean_mean` and every `bary-sq_*`.
The GitBook page (`distance-similarity.md:41-75`) lists 12 identifiers, including `_max_time`, and no squared variants. Refs [21] Petitjean et al. 2011 and [22] Schultz & Jain 2018 (supplement lines 2030-2034).

### Canonical status of the summaries
The barycenter (Fréchet mean) is canonical. Reducing it to a scalar by its time-mean or time-max is a pyspi construction with no literature counterpart. Every "canonical definition" below therefore covers the barycenter step only.

---

## distance.bary.euclidean

**Documented:** the "(unwarped) Euclidean distance (euclidean)" barycenter (§1.2.10).
**Canonical:** the Fréchet mean under squared Euclidean distance, `argmin_b Σ_k ||b − x_k||² = (x+y)/2` (Fréchet 1948; arithmetic mean).
**Implemented:** `tslearn/barycenters/euclidean.py:42-44` computes `numpy.average(X_, axis=0, weights=ones(2))`, i.e. exactly `(z_i + z_j)/2`.
**Analytic consequences on z-scored input** (confirmed by probe1 on bivariate_gaussian T=256):
* `bary_euclidean_mean` = 0 up to rounding (observed −1.04e-17 .. 1.73e-17). **Candidate: degenerate SPI.** It is in config and in the EEG selection, but it carries no information. tests/README.md:147 only records it as "~1e-16" rounding noise in the tolerance table.
* `bary-sq_euclidean_mean` = (1 + r)(T − 1)/(2T), with r the Pearson correlation. Probe1 matched this to 6 decimals for all 6 pairs. **Candidate:** the identifier is a monotone transform of Pearson r.
* `bary_euclidean_max` = max_t (z_i+z_j)/2 ≥ 0, and `bary-sq_euclidean_max` = (max_t |z_i+z_j|/2)². These are extreme-value statistics of the summed series.

**Deviations:** the barycenter matches the canonical definition exactly. Deviations from the documentation are the labels (signed vs "unsigned", Directed base vs "undirected"), undocumented identifiers (the 8 squared variants and `bary_euclidean_mean`), and `max_time` being documented but not configured. Under `normalise=False`, the mean statistic would measure offsets ((mean x + mean y)/2), not coupling.

**Proposed checks:** an analytic zero for the mean (atol 1e-12, pipeline), the analytic `(1+r)(T−1)/(2T)` formula, a from-scratch numpy z-score-and-average for all four identifiers (rtol 1e-12), a from-scratch detrend + z-score reference (`trending`, `scaled_offset`), a pyspi-vs-pyspi (same-lineage) scale/offset invariance check, a statistical population check at T=4096 against (1+ρ)/2 ∈ {0.65, 0.90} on `bivariate_gaussian`, and the symmetry and direction check.

---

## distance.bary.dtw

**Documented:** "alignment via expectation maximization (dtw)".
**Canonical:** the DTW Fréchet mean `argmin_b (1/n) Σ_k DTW(b, x_k)²`, computed by DBA (Petitjean et al. 2011, Pattern Recognit. 44:678) or the equivalent Majorize-Minimize mean algorithm (Schultz & Jain 2018, Pattern Recognit. 74:340, Alg. 2). The problem is non-convex, so these algorithms return a local minimiser.
**Implemented:** `tslearn.barycenters.dtw_barycenter_averaging` (`dba.py:485-618`, `621-737`) with `n_init=1`, `max_iter=30`, absolute `tol=1e-5` and init = arithmetic mean (`_init_avg`, `dba.py:23-25`). Each iteration runs the DTW path from the barycenter to each series (`_mm_assignment`, `dba.py:229-305`; `_njit_dtw_path`, `metrics/_dtw.py:325-344, 415-443`, squared local cost, no constraint) and then the averaging update `diag(1/ΣV_k) Σ W_k x_k` (`dba.py:400-434`). It stops when the cost change is below tol or the cost increases (`dba.py:729-734`). The routine is deterministic and order-invariant (probe2).
**Deviations:** the documented "expectation maximization" is loose wording for MM. Compared with Petitjean's practice, the run uses a mean init and a single init. **Candidate:** on `narrowband_f0-0.25`, DBA returns its Euclidean initialisation unchanged: after 2 iterations the cost equals the init cost (37.6819 at T=256). There `bary_dtw_*` would equal `bary_euclidean_*`. **Candidate:** `bary_dtw_mean` is near-degenerate: the largest |mean| over pairs is up to ~0.09 at T=64 (0.051–0.090 across `bivariate_gaussian`, `delayed_noise`, `narrowband_f0-0.25`, `logistic_maps`, `iid_gaussian`) and 0.004–0.012 at T=1024 except `logistic_maps` (0.071), against `bary_dtw_max` of 1.2–3.2.
**Proposed checks:**
* A from-scratch numpy DBA with the same init and stopping rule (atol 1e-10).
* An optimality comparison of F(b) against the Euclidean mean, each input series, and a 20-start from-scratch DBA.
* The analytic DBA(x, x) = x.
* A qualitative comparison with `bary_euclidean_*` on `delayed_noise` and `narrowband_f0-0.25`.
* The symmetry and direction check.

---

## distance.bary.sgddtw

**Documented:** "alignment via a subgradient descent algorithm (sgddtw)".
**Canonical:** the same DTW Fréchet function minimised by the stochastic subgradient mean algorithm (Schultz & Jain 2018; Alg. 3 in arXiv:1701.06393v1, where Alg. 1 is the batch subgradient mean and Alg. 2 the majorize-minimize mean).
**Implemented:** `tslearn.barycenters.dtw_barycenter_averaging_subgradient` (`dba.py:740-884`). `random_state=None` resolves to the **global** numpy RNG (`dba.py:838`; `sklearn/utils/validation.py:1477-1478`). The barycenter starts at the arithmetic mean. Each epoch shuffles the 2 series and applies `b ← b − 2η(V_k b − W_k x_k)` per series (`dba.py:437-482`). η falls from 0.05 to 0.005 during epoch 0 only and is constant afterwards (`dba.py:874-875`). Stopping (`dba.py:876-883`) compares the cost of the **last processed single series**, `dist²·w[0]/Σw` (`dba.py:862-863`), between epochs, and breaks on any increase with a ConvergenceWarning.
**Deviations / candidates:**
* **Under-convergence (candidate, not in the documented-issue list).** The stopping statistic is a noisy single-sample cost. Probe4 measured F(b) at three points on two pairs:

  | Dataset (pair) | tslearn stop | 30 epochs, no early stop | MM-DBA |
  |---|---|---|---|
  | bivariate_gaussian T=256 (1,0) | 57.77 (4 epochs) | 50.26 | 44.69 |
  | delayed_noise T=1024 | 331.46 | 185.01 | 135.86 |

  The ConvergenceWarning fired in all 9 probe2 runs.
* **Stochasticity and cache order.** Candidate: whichever sgddtw identifier runs first fixes the barycenter for the other three (they reuse the cache), and the Directed loop computes pair {a<b} from z[[b, a]]. Already documented (tests/README.md:134 and :146): the value depends on the global RNG, hence on RNG draws consumed by SPIs computed earlier. An unseeded repeat changed the mean statistic by up to 1.1e-3; seed 0 vs seed 1 barycenters differ pointwise by up to 1.18.
* **Input-order dependence for a fixed seed.** Swapping the inputs changed the barycenter by up to 0.177 pointwise. The cache hides this in the matrix.
* `_mm_assignment` is given the full weights vector (cost uses `weights[0]`). This is harmless with uniform weights.

**Proposed checks:**
* A from-scratch SSG driven by the same seeded permutations (atol 1e-10).
* An optimality comparison against from-scratch MM-DBA and against SSG without early stopping.
* Seed-spread statistics over 50 seeds.
* The analytic SSG(x, x) = x.
* The symmetry and direction check, including cold-cache swapped order.

---

## distance.bary.softdtw

**Documented:** "alignment via a differential loss function (softdtw)". §1.2.9 states γ = 1 as the soft-DTW default.
**Canonical:** `argmin_b Σ_k λ_k dtw_γ(b, x_k)` with the soft-min over alignments (Cuturi & Blondel 2017, ICML, PMLR 70:894-903, §4). This is not a divergence: dtw_γ(x, x) ≠ 0 (Blondel, Mensch & Vert 2021, AISTATS).
**Implemented:** `tslearn.barycenters.softdtw_barycenter` (`softdtw.py:38-117`) with γ = 1, unit weights (an objective sum, not a mean), init = Euclidean barycenter, and scipy L-BFGS-B with `tol=1e-3` (ftol = gtol, `scipy/optimize/_minimize.py:677-681`) and `maxiter=50`. `res.x` is returned without checking `success`. The objective and gradient come from `_softdtw_func` (`softdtw.py:19-35`), the forward and backward recursions (`metrics/soft_dtw_fast.py:15-39, 101-125, 205-234`) and the squared-Euclidean Jacobian (`soft_dtw_fast.py:318-340`).
**Deviations / candidates:**
* **Shrinkage bias (candidate).** With γ = 1 on z-scored data, the barycenter of (x, x) is not x: max|b − x| = 1.249 and var(b) = 0.704 vs 0.996 (probe3). On narrowband pairs, max(b) is 0.83-0.86 vs 1.31-1.34 for DBA (probe2). So `bary_softdtw_max` is systematically lower than the DTW-barycenter max. Objective values are negative.
* **Loose optimiser tolerance, no success check (candidate).** Every probe run converged (nit 4-13, T ≤ 1024). T=4096 is untested.
* **Cost.** About 2.5-3.3 s per pair at T=1024 (probe2). This is for the optimisation stage.
* tests/README.md bug 3 (`SoftDynamicTimeWarping` ignores `global_constraint`) does not apply here, because Barycenter exposes no constraint.

**Proposed checks:**
* A from-scratch numpy soft-DTW objective and gradient with the same L-BFGS-B setup (objective/gradient rtol 1e-10, statistics rtol 1e-6).
* A finite-difference gradient check.
* A tight-tolerance re-optimisation (optimality).
* A γ sweep {1, 0.1, 0.01} on (x, x) documenting the bias.
* The symmetry and direction check.

---

## Open questions

1. Did the pyspi authors drop `bary_euclidean_mean` from the supplement's seven SPIs because it is identically zero? If so, why is it still in `config.yaml` and the EEG selection? (Hypothesis: yes; nothing in the code or docs says so.)
2. Should the squared variants be treated as intended SPIs? They are configured but documented nowhere, and `bary-sq_euclidean_mean` duplicates Pearson r.
3. Is the sgddtw early stop a tslearn bug or an intended heuristic? Schultz & Jain 2018 run a fixed number of epochs as far as we recall (unconfirmed: we did not re-read the paper for this). tslearn's comparison of a last-sample cost looks unintended. This needs checking against the paper and the tslearn issue tracker.
4. Does DBA's stall at the Euclidean init (narrowband) reflect a true local minimum or an artefact of tie-breaking in the path backtrack? The from-scratch DBA check should settle this.
5. Does soft-DTW L-BFGS-B still converge within 50 iterations at T=4096? This is untested and expensive (O(T²) per evaluation).
6. Should the family be labelled signed (the code) or unsigned (all documentation)? Only the mean statistics can be negative.

---

## Checks

Check stage at HEAD a1f20c7. The files are `audit/checks/manifest/barycenters.json` (337 runs), `audit/checks/refs/barycenters.py`, which runs in the comparison venv and never imports tslearn, and `audit/checks/probes/barycenters_probes.py`, which runs in the project venv. Results are in `audit/checks/out/results/barycenters.json`. Run ids have the form `<unit>|<dataset>|<T>|<path>|<tag>|<identifier>[|p<processes>][|seed<s>]`. In the text, `*` stands for a dataset, T or identifier.

The references are built as follows:
* **Euclidean:** analytic formulas, plus numpy z-scoring and closed-form detrending.
* **DTW, DBA, soft-DTW:** the shared kernels in `refs/_kernels_distance.py` (`dtw_acc`, `dtw_path`, `dba`, `soft_dtw_acc`, `soft_dtw_grad_D`, `soft_dtw_barycenter_objective`). For multi-start runs, `barycenters.py:dba_from` is the same loop as the kernel's `dba` with an explicit initial barycenter.
* **SSG:** `barycenters.py:ssg` is written from Schultz & Jain. It reproduces tslearn's step schedule and stopping rule (`tslearn/barycenters/dba.py:838-884`) and is driven by `numpy.random.RandomState(seed)`. This is the legacy stream that `np.random.seed(seed)` resets. The reference replays the global-RNG stream across pairs in `Directed.multivariate` order: pair {a<b} is computed from `z[[b, a]]`, in the order a = 0.., b = a+1...
* **Soft-DTW barycenter:** the kernel objective and gradient, passed to scipy L-BFGS-B with tslearn's settings (tol 1e-3, maxiter 50, init = pointwise mean, unit weights). scipy is 1.18.1 in the comparison venv and 1.17.1 in the project venv.

Self-consistency runs compare pyspi with pyspi and are labelled same-lineage: `*|sym|*`, `*|invariance|*`, `*|det|*` and `*|vs-eucl|*`. They can establish symmetry, invariance or determinism only. They cannot establish what A[i,j] means.

Outcome: 337 runs, 329 pass, 8 fail, 0 without a reference. The failing runs are listed under their units.

Current line citations: `pyspi/calculator.py:252-266` (SPI construction from class labels) and `:329` (every SPI's `multivariate` receives the shared `self.dataset`); `tests/README.md:134` (sgddtw skipped in the end-to-end Calculator comparison), `:146` (stochastic tolerance row) and `:147` (`bary_euclidean_mean ~1e-16`). The trace-stage text cited `calculator.py:238-244/289-294` and `tests/README.md:115/127/128`. These were updated in the records.

Trace-stage fields corrected by this stage:
* `distance.bary.sgddtw` `units_and_range`: the bound |mean| ≤ 0.0162 came from nine single probe pairs. It is replaced by check-run ranges: 0.0012-0.0128 on the pipeline path at T = 1024, and up to 0.242 on raw kernel-path input.
* `deviation_from_documented`: now says the RNG dependence is documented in tests/README.md:134/146, not undocumented.
* The narrowband DBA stall applies to 1 of 3 pairs (stated in the candidate resolution).
* `distance.bary.sgddtw` `canonical_ref` and `deviation_from_canonical`: the stochastic subgradient method is Alg. 3 of Schultz & Jain (arXiv:1701.06393v1), not Alg. 1, and tslearn's update steps by 2η where Alg. 3 steps by η, so "matches" was replaced by "is SSG except for the doubled step" (corrected after the adversarial review).
The relabelling of the self-consistency checks as same-lineage, and the split of Euclidean check [3], were already in the records when this stage started.

### distance.bary.euclidean

**Analytic zero.** `bary_euclidean_mean` is identically zero on every path that z-scores. The check covered 84 runs: all 21 datasets at T = 64, 1024 and 4096 on the pipeline path, and T = 1024 on pipeline-detrend (`distance.bary.euclidean|*|pipeline|zero|bary_euclidean_mean`, `...|pipeline-detrend|zero|...`). The largest value was max|A| = 2.52e-16 (`...|logistic_maps|1024|pipeline|zero|...`). Every run is within atol 1e-12 of the analytic 0, because mean_t (z_i + z_j)/2 = (mean z_i + mean z_j)/2 = 0.

This SPI is in config.yaml and in the EEG selection. The supplement's seven-SPI list omits it. The repository's own storage screen of real EEG windows, `docs/storage_comparison/explore/properties_105_64.json`, shows the same picture: vmin -1.03e-15, vmax 1.00e-15, unique fraction 0.0016. KNOWN_ISSUES.md does not list this issue. tests/README.md:147 mentions the value only as rounding noise in a tolerance table. **Confirmed, new finding.**

**Pearson duplicate.** `bary-sq_euclidean_mean` equals (1 + r)(T − 1)/(2T) with max_rel 1.36e-15 (`*|formula|*`, 12/12 runs on bivariate_gaussian, sparse_precision, iid_gaussian and contaminated at T = 64/256/1024). It is therefore an affine transform of Pearson r. **Confirmed.**

**Population check.** At T = 256, 1024 and 4096 (`distance.bary.euclidean|bivariate_gaussian|{256,1024,4096}|pipeline|popstat|bary-sq_euclidean_mean`), pyspi's value is compared with (1 + ρ)/2 · (T − 1)/T. The standard error is sd(r)/2 = (1 − ρ²)/(2√T). It is inflated by 2.865/2, so the compare script's 2-se rule becomes a two-sided Bonferroni test at α = 0.05 over the 12 ordered off-diagonal entries that the compare script tests (raw |z| ≤ Φ⁻¹(1 − 0.05/24) = 2.865). Over the 6 unordered pairs this is α = 0.025; the α = 0.05 value over 6 pairs, Φ⁻¹(1 − 0.05/12), would be 2.638. The compare script's max_z is therefore on the inflated se and understates the raw z by the factor 1.4325. The reference also reports raw_max_z on the unscaled se.

| T | max_z (inflated se) | raw_max_z (unscaled se) | max_abs |
|---|---|---|---|
| 256 | 0.74 | 1.06 | 0.0329 |
| 1024 | 0.81 | 1.16 | 0.0182 |
| 4096 | 0.72 | 1.03 | 0.0081 |

All three pass, with raw z of about 1 against the critical value 2.865. max_abs falls roughly as 1/√T.

**From-scratch statistics.** All four identifiers, on the kernel path at T = 256 and the pipeline path at T = 1024, on bivariate_gaussian, delayed_noise, narrowband_f0-0.25 and nonlinear_dependence: 32/32 pass with max_abs 0 (`*|scratch|*`).

**Detrend path.** On trending and scaled_offset at T = 256 and 1024, compared with a closed-form least-squares detrend followed by z-scoring: 16/16 pass, max_abs 4.97e-14 (`*|detrend|*`).

**Kernel path (normalise = False).** The mean statistic is the average of the two offsets. `distance.bary.euclidean|scaled_offset|1024|kernel|offset|bary_euclidean_mean` matches (mean x_i + mean x_j)/2 to 4.4e-16. The sample means are [0.014, 0.987, −2.112, 4.678], against design offsets [0, 1, −2, 5]. **Confirmed.**

**Invariance (same-lineage).** scaled_offset equals bivariate_gaussian for all four identifiers.

**Symmetry and direction.** max|A − A.T| = 0 on var2_bivariate, delayed_noise, logistic_maps_chaotic and anm (`*|sym|bary_euclidean_max`). The cold-cache ordered-pair matrix (`...|delayed_noise|1024|pipeline|cold|bary_euclidean_max`) matches the reference exactly, with max|M − M.T| = 0. A[i,j] carries no direction.

**Verdict: degenerate, score 0** (rubric anchor 0, "constant"), on the representative identifier `bary_euclidean_mean`. Under README rule R1 (representative identifier) a row's `score_adjusted` and `verdict_adjusted` describe the unit's representative, and every other identifier whose own anchor differs is listed in `identifier_exceptions` with its anchor and reason; here the representative is degenerate, so `score_adjusted` stays 0. (Before R1 was adopted this note said that the audit scores a multi-identifier unit on its lowest-scoring identifier and left a README statement of the rule to synthesis. R1 replaced that reading: `distance.softdtw` now has `score_adjusted` 9 on its representative `softdtw`, with its two constraint identifiers listed at 3 in `identifier_exceptions`; see notes/dtw_family.md.) The barycenter and the other three identifiers are fp-exact against analytic or independent references, and the record's `identifier_exceptions` lists them:
* `bary_euclidean_max`, which is the supplement's SPI, at 10.
* `bary-sq_euclidean_mean` at 9: computed exactly, but it duplicates Pearson r, and no document defines the squared variants, so anchor 10's supplement clause fails (the R4 reading).
* `bary-sq_euclidean_max` at 9, exact, for the same reason.

The per-identifier picture is also recorded in `what_changed`.

The adversarial review proposed scoring the unit 10 for its calculation and recording `bary_euclidean_mean` only as a degenerate-identifier finding. The score stays 0. `audit/units.yaml:1230` fixes `representative: bary_euclidean_mean` for this unit, and `build_csv.py` writes that representative (IDENTITY columns, line 66; filled at line 222) on the same row as the score columns, so a 10 would put "bary_euclidean_mean, 10" on the row and describe a constant SPI in the EEG selection as fp-exact against canonical. At the time of the review the CSV could not carry a per-identifier score; the `identifier_exceptions` column, added later with the score columns, now does, and holds the three identifiers above. To keep the row from being misread the other way, `what_changed` opens by saying that the 0 applies to the representative identifier and not to the barycenter calculation. What is new here is the identification of the zero as structural (mean_t (z_i + z_j)/2 = 0 on every z-scoring path, including the EEG workflow's detrend then z-score) and the three other confirmed candidates (bary-sq_euclidean_mean as an affine transform of Pearson r, the Signed class against the documented "unsigned", and the offset average under normalise=False); the near-zero value itself was already observed in tests/README.md:147 as rounding noise absorbed by atol. The label candidate (Signed class versus documented "unsigned") is confirmed as a mismatch between the class label and the documents. The values (the mean statistics take both signs, the max statistics are ≥ 0 on z-scored data) describe the range only; the audit reads the keyword by the GitBook glossary's sign-meaning sentence (`glossary-of-terms.md:21`, whether the sign indicates how A and B co-fluctuate; the same entry also mentions the range, so the glossary is mixed), as notes/dtw_family.md does for softdtw, and under that reading the range does not decide the label. The `max_time` candidate is confirmed statically: there is no `max_time` in `config.yaml`.

### distance.bary.dtw

**DBA replication.** pyspi's DBA matches the from-scratch DBA (`_kernels_distance.dba`, with tslearn's init and stopping rule mirrored) on 52/52 statistic runs, max_rel 4.2e-14. The runs cover the kernel path at T = 64 and 256 (four identifiers) and the pipeline path at T = 64, 256 and 1024 (`distance.bary.dtw|*|dba|*`). On the raw barycenters it matches on 8/8 runs, max_abs 4.4e-16 (`*|dba-bc|*`).

Iteration counts are 2-12 per pair. No run reached max_iter = 30. The same arithmetic tie-break (diagonal, then up, then left) reproduces every path, so no tie differences appeared. The analytic case DBA(x, x) = x holds exactly on 4/4 runs (`distance.bary.dtw|iid_gaussian|256|kernel|xx|*|p0-0`).

**Optimality.** The runs are `distance.bary.dtw|{bivariate_gaussian,delayed_noise,narrowband_f0-0.25}|256|pipeline|opt|bary_dtw_mean`. They report F(b) = mean_k DTW(b, x_k)² at pyspi's barycenter, at the Euclidean init and at each input series. They also report the best of 8 from-scratch DBA starts: each input series, and 6 starts at mean + 0.5·N(0,1), each run for up to 100 iterations.

F(pyspi) ≤ F(init) holds on every pair, so all three runs pass the declared descent criterion. But pyspi's barycenter is a poor local minimiser (gap = F(pyspi)/F(best of 8 starts) − 1):

| Dataset | Gap to best of 8 starts | Notes |
|---|---|---|
| bivariate_gaussian | 0.12-0.56 | |
| delayed_noise | 1.85-44.5 | Pair (0,1): F 49.1 vs 1.08. Returning one raw input series would give F = 2.14. |
| narrowband_f0-0.25 | 0.13 / 13.5 / 0.12 | |

The returned value therefore depends on the initialisation rule. Petitjean et al. start from an input sequence; tslearn starts from the mean and uses n_init = 1. **Candidate 4 confirmed; new finding.** It is not in KNOWN_ISSUES.md.

**Stall at init (narrowband).** DBA returns the Euclidean init unchanged on narrowband_f0-0.25 pair (0,1) only, which is 1 of 3 pairs, at every T (`stalled_at_init` [1, 0, 0] in `*|narrowband_f0-0.25|*|dba|*`). For that pair:
* The DTW paths from the init to both series are diagonal, with 0 exact ties in the backtrack.
* The smallest change in F over 20 random perturbations of norm 1e-3 and 1e-2 is +1e-6 > 0.

So the init is a strict local minimum, not a tie-breaking artefact. The 8-start search finds a 12.9% lower cost.

The qualitative run `distance.bary.dtw|narrowband_f0-0.25|1024|pipeline|vs-eucl|bary_dtw_max` **fails**. Its expectation was equality with `bary_euclidean_max` on every pair, but equality holds on 1 of 3 pairs: 1.3317 = 1.3317, while 1.521 vs 0.864 and 1.340 vs 0.628 differ. Candidate 1 is therefore confirmed only for some pairs. The delayed_noise run passes: `bary_dtw_max` > `bary_euclidean_max` on 6/6 pairs (2.67-3.03 vs 2.09-2.49).

**bary_dtw_mean range.** The largest |bary_dtw_mean| over pairs on the pipeline path is:

| T | Range | Exception |
|---|---|---|
| 64 | 0.051-0.090 | |
| 256 | 0.005-0.063 | |
| 1024 | 0.004-0.012 | logistic_maps 0.071 |

By comparison, `bary_dtw_max` is 1.20-3.04 at T = 1024. The statistic is near zero but not identically zero. **Candidate 2 confirmed.** This range replaces the single-pair numbers used at the trace stage.

**Symmetry and direction.** max|A − A.T| = 0 on all four direction datasets. Cold-cache ordered pairs on delayed_noise at T = 256 give max|M − M.T| = 0 and match the reference. DBA with mean init is order-invariant, so A[i,j] carries no direction.

**Verdict: variant, score 7** (rubric 7: "matches the pyspi-supplement variant but differs from canonical in a value-changing way"). The supplement names the algorithm that runs, "alignment via expectation maximization (dtw)", which is tslearn's DBA/majorize-minimize mean. The independent DBA reproduces it (52/52 statistic runs, max_rel 4.2e-14; raw barycenters max_abs 4.4e-16). DBA is a local method in Petitjean et al. too, and the supplement specifies no initialisation. The value-changing difference is from the canonical target, the global DTW Fréchet mean: a single mean-initialised run lands in a local minimum whose cost is up to 44.5× above the best of 8 starts (delayed_noise). The descent criterion F(pyspi) ≤ F(init), which passed 3/3, is only an objective decrease (anchor 5 wording), so the score rests on the first clause of anchor 7, not on the optimality clause.

The check stage first scored this unit 6, reading the supplement's "minimizes the sum of squared distances" as a requirement for the global minimum. The adversarial review argued that anchor 6 needs a deviation from the supplement as well as from canonical. pyspi runs exactly the documented algorithm, so the score was raised to 7. This differs from softdtw (score 6): DBA stops when its own update no longer lowers the cost by more than tol 1e-5 (2-12 iterations, never at max_iter 30), while tslearn's soft-DTW L-BFGS-B stops before a stationary point of its own objective.

### distance.bary.sgddtw

**SSG replication.** The from-scratch SSG replays the global-RNG stream and reproduces pyspi bit for bit, up to rounding:
* 33/33 statistic runs, max_rel 9.6e-14 (`distance.bary.sgddtw|*|ssg|*`, including `...|seed43`).
* 6/6 raw-barycenter runs, max_abs 2.2e-16.
* 100 seeded barycenters, max_abs 2.2e-16 (`*|seeds50|*`).
* The analytic case SSG(x, x) = x holds exactly on 4/4 runs.

So the sgddtw value is exactly verifiable once the RNG stream is known. The family brief expected otherwise.

**RNG-stream dependence.** In `distance.bary.sgddtw|bivariate_gaussian|256|kernel|rngdep|bary_sgddtw_max`, another sgddtw SPI first consumes the global RNG; this is what an earlier SPI does in a Calculator run. The reference that replays the consumer's draws matches with max_abs 0. pyspi's value moves by up to 0.0278 (1.5% relative) from the plain seed-42 run.

**Determinism.** Two `seed_all(42)` computations are bit-identical (`*|det|*`, same-lineage).

**Shared cache.** In `*|cache|bary_sgddtw_max`, the second sgddtw identifier on the same Data object returns the first identifier's barycenter (max_abs 0 against the seed-42 reference) and consumes no RNG draws (flag 0). **Candidate 2 confirmed.**

The RNG dependence itself is already documented (tests/README.md:134/146) and is cited, not claimed. The new part is the mechanism: the barycenter is drawn once per Data object, by whichever sgddtw SPI runs first, from z[[b, a]].

**Seed spread.** Over seeds 0..49, for pair (0,1) at T = 256 on the pipeline path:

| Dataset | SD of mean statistic | SD of max statistic | Range of max |
|---|---|---|---|
| bivariate_gaussian | 3.0e-4 | 8.0e-3 | 0.0168 |
| iid_gaussian | 1.8e-3 | 2.9e-3 | |

**Stopping rule.** tslearn compares the pre-update DTW² of the single series processed last in each epoch. On bivariate_gaussian at T = 256 (kernel path) all 6 pairs stopped on the "cost increased" branch after 3-5 epochs (`ref_stopped_on_increase`).

**Objective versus DBA** (qualitative, `*|vs-dba|*`, rubric cap 5 for this check). The declared criterion is that F(sgddtw) < F(Euclidean init) on every pair; all 4 runs pass. The relative gap F(sgddtw)/F(DBA) − 1 is:

| Run | sgddtw | 30-epoch SSG without early stop |
|---|---|---|
| bivariate_gaussian, T = 256 | 0.02-0.43 | 0.006-0.15 |
| delayed_noise, T = 256 | −0.05 to 3.62 | |
| logistic_maps, T = 256 | 1.86 | 0.80 |
| delayed_noise [0,3], T = 1024 | 0.995 | 0.995 (the run used all 30 epochs) |

F(sgddtw) exceeds F(DBA) on 13 of 14 pairs. The early stop leaves the barycenter under-converged. **Candidate 1 confirmed; new finding** (not in KNOWN_ISSUES.md).

**Comparison with the paper.** I read Schultz & Jain, arXiv:1701.06393v1:
* Sec. 3 proposes termination by (T1) a maximum number of iterations, or (T2) no improvement in the Fréchet function F evaluated over all samples.
* Sec. 5.1 runs a fixed 50 epochs, with the same linear η schedule from 0.05 to 0.005 over the first epoch.
* Alg. 3 line 8 updates z ← z − η(V z − W x). tslearn uses 2η, which doubles the step.

In arXiv:1701.06393v1, Alg. 1 (line 557 of the extracted text) is the batch subgradient mean, with the update z ← z − η(2/N) Σ_k (V^(k) z − W^(k) x^(k)). The stochastic method is Alg. 3 (line 798). The trace-stage `canonical_ref` and `deviation_from_canonical` of `distance.bary.sgddtw` and the docstrings and `reference_detail` of `refs/barycenters.py:ssg`/`h_ssg` cited "Alg. 1". They now cite Alg. 3 of arXiv v1, so every `*|ssg|*` result carries the corrected `reference_detail`. I did not check the published Pattern Recognit. 74 version, which may number the algorithms differently.

**Input-order dependence.** Each cold-cache ordered pair is computed with `np.random.seed(42)` before the call (`*|order|bary_sgddtw_max`). The routine is not order-invariant: max|M − M.T| = 0.0486 on bivariate_gaussian and 0.379 on delayed_noise at T = 256 on the kernel path. The from-scratch SSG reproduces the ordered-pair matrix (max_abs 2.2e-16). The cache hides this asymmetry in the SPI matrix. **Candidate 3 confirmed.** Candidate 4 (the `weights[0]` cost) is confirmed harmless: the reference uses dist² / n and reproduces every stopping decision.

**Symmetry and direction.** max|A − A.T| = 0 on all four direction datasets, because of the cache. A[i,j] carries no direction.

**Verdict: variant, score 5** (rubric 5: a non-standard estimator, SSG with last-sample early stopping, computed exactly as the code states). The exact replication would allow a higher score on implementation grounds. The score reflects that the quantity is not the documented Fréchet mean, and that it depends on the global RNG stream.

### distance.bary.softdtw

**Replication.** The from-scratch objective plus scipy L-BFGS-B reproduces pyspi to fp precision, even across the two scipy versions:
* statistics on 19/19 runs, max_abs 3.7e-13, max_rel 2.4e-13 (`distance.bary.softdtw|*|sdtw|*`: kernel T = 64 with four identifiers on four datasets, pipeline T = 256 for `bary_softdtw_max`), with 4-12 L-BFGS-B iterations;
* raw barycenters on 4/4 runs, max_abs 7.0e-14;
* tslearn's objective and gradient at the Euclidean init on 4/4 runs, max_abs 2.8e-13 (`*|obj0|*`);
* tslearn's gradient against central differences of the kernel objective on 2/2 runs, max_abs 2.4e-8 (`*|fd|*`).

**Optimality.** This check **fails** on 7 of 10 runs (`distance.bary.softdtw|*|pipeline|opt|bary_softdtw_mean*`). The rule, declared before running, was a relative objective gap of at most 1e-3 to a tight L-BFGS-B re-optimisation (ftol = gtol = 1e-12) started from pyspi's barycenter. After the adversarial review the T = 1024 re-optimisation, previously capped at 40 iterations, was allowed up to 500 iterations. Every re-optimisation at T = 64, 256 and 1024 now reaches scipy's convergence test (`reopt_converged` 1, final max|grad| ≤ 1.6e-4). The T = 1024 runs needed 48 and 49 iterations, and their gaps agree with the capped values to 4 significant figures. So at T ≤ 1024 the gap measures the distance to the local minimum next to pyspi's point. It does not measure the distance to the global minimum.

| T | bivariate_gaussian and delayed_noise: relative gap | Notes |
|---|---|---|
| 64 | 2.2e-5 to 0.0138 | converged re-optimisation (12-31 iterations) |
| 256 | 2.5e-4 to 0.0058 | converged re-optimisation (24-46 iterations) |
| 1024 | 0.00118 and 0.00313 | Pair [0,1]; converged re-optimisation (48 and 49 iterations) |
| 4096 | ≥ 0.00103 | `distance.bary.softdtw|bivariate_gaussian|4096|pipeline|opt|bary_softdtw_mean|p0-1`; one Armijo step along −grad, so only a lower bound. The distance to the nearby minimum at T = 4096 is not measured. |

The gradient at the returned point is still 2-27% of the gradient at the init. Re-optimisation moves `bary_softdtw_max` by up to 0.051 and `bary_softdtw_mean` by up to 0.030 (bivariate_gaussian T = 64, pair (2,3)). narrowband_f0-0.25 passes at T = 64, 256 and 1024 (gap ≤ 6.0e-5).

The cause is tslearn's `tol=1e-3` (ftol = gtol) with `maxiter=50`. It returns `res.x` without a convergence check (`tslearn/barycenters/softdtw.py:113-115`). **Candidate 2 confirmed; new finding**, not in KNOWN_ISSUES.md. The effect already appears at T = 64.

**Shrinkage with γ = 1.** For the barycenter of two identical raw series (`*|xx|*`, `*|gamma|*`, processes [0,0]), pyspi matches the reference to 7.3e-14, and the result is far from x:

| Series | max\|b − x\| | var(b) vs var(x) | max(b) vs max(x) |
|---|---|---|---|
| iid_gaussian | 1.315 | 0.732 vs 1.015 | |
| narrowband sinusoid | 1.136 | 0.0036 vs 0.511 | 0.312 vs 1.142 |

The deviation shrinks as γ → 0. With γ = 1, 0.1 and 0.01 it is 1.315 / 0.234 / 0.022 for iid_gaussian and 1.136 / 0.0015 / 0 for the sinusoid. This is a property of the soft-DTW definition (soft-DTW is not a divergence), so it is **confirmed but recorded as canonical behaviour**, not as an implementation error. It does mean that `bary_softdtw_max` is not comparable with the DTW-barycenter max.

**Cost.** pyspi takes 2.6-3.2 s per pair at T = 1024 and 66.4 s for one pair at T = 4096. That is about 25× the T = 1024 time, more than the O(T²) factor of 16, because more iterations are needed. **Candidate 3 confirmed.**

**Symmetry and direction.** max|A − A.T| = 0 on the four direction datasets. Cold-cache ordered pairs on delayed_noise at T = 64 match the reference to 4.1e-12, with max|M − M.T| = 0. A[i,j] carries no direction.

**Verdict: variant, score 6** (rubric 6). The implementation is exact, but the optimiser stops short of the soft-DTW barycenter by a value-changing margin. Anchor 7 would require the optimality checks to pass.

### Shared infrastructure notes
* `audit/checks/pyspi_outputs.py:65-77` (`load_index`/`write_index`) reads every family's index file without locking. The first barycenters run aborted with `JSONDecodeError` while another family was writing its index. A plain re-run recovered, and only cached runs were affected. The script is read-only for this stage, so this is reported and not fixed.
* `_kernels_distance.dba` and `dtw_path` needed no workaround. The tie order (diagonal, up, left; first index wins) reproduced every tslearn path. Two pieces were added locally in `refs/barycenters.py` and are not kernel defects:
  * `dba_from` (an explicit init, for multi-start runs);
  * `ssg` (the kernel module has no SSG).

## Open questions: answered

1. **Was `bary_euclidean_mean` dropped from the supplement because it is identically zero?** The evidence settles the premise: it is identically zero on every z-scoring path and every dataset (max|A| 2.52e-16, 84 runs), and also on real EEG in the repository's storage screen (±1.0e-15). The authors' intent cannot be settled from code or documents. Nothing in the repository explains why it remains in `config.yaml` and the EEG selection. Only the pyspi authors could answer this.
2. **Are the 8 `bary-sq_*` identifiers intended SPIs?** They are computed exactly as the code states (fp-exact against independent references in all four units). No document defines them. `bary-sq_euclidean_mean` = (1 + r)(T − 1)/(2T) exactly (max_rel 1.4e-15), so it duplicates Pearson r. Whether they are intended remains a documentation question. Their scoring is uneven across the four units: distance.bary.euclidean lists its two bary-sq identifiers at 9 in `identifier_exceptions` (anchor 10's supplement clause fails for undocumented variants), while bary.dtw, bary.softdtw and bary.sgddtw list none; whether anchor 7's supplement-variant clause holds for the bary-sq DTW identifiers is left to the record owner (`_handoff/score_columns_open_readings.md`, distance.bary.dtw).
3. **Is the sgddtw last-sample stop an upstream bug or a heuristic?** It is not the paper's rule. Schultz & Jain (arXiv:1701.06393v1) terminate on (T1) an iteration budget or (T2) no improvement in the full Fréchet function, and run 50 fixed epochs in their experiments. The step schedule (0.05 → 0.005, linear over the first epoch, then constant) matches their Sec. 5.1. tslearn additionally doubles the step: its update uses 2η, against η in Alg. 3. The measured effect is a stop after 3-5 epochs on Gaussian pairs and a cost 2-43% above DBA. An upstream issue search was not done (no network search beyond the paper).
4. **Is DBA's narrowband stall a genuine local minimum or a tie artefact?** It is a genuine strict local minimum. On narrowband pair (0,1) the init paths are diagonal with 0 backtrack ties, every perturbation raises F (min +1e-6), and the from-scratch DBA with the same tie order stalls identically. It is not the global minimum: an 8-start DBA finds a 12.9% lower cost. It occurs on 1 of 3 narrowband pairs, not on all of them.
5. **Does soft-DTW L-BFGS-B converge at T = 4096?** On the one pair run (bivariate_gaussian [0,1], 66.4 s on the pyspi side), the returned point is at least 0.10% above an objective value reached by a single line-search step. The gradient ratio is 0.27. So pyspi's optimiser stops short at T = 4096, as it already does at T = 64-1024, where converged re-optimisations give gaps up to 0.0138 (T = 64) and 0.00313 (T = 1024). How far short it stops at T = 4096 is not measured. Full re-optimisation at T = 4096 was not run, because each evaluation of the from-scratch objective is O(T²) and the run would exceed the family budget. A converged L-BFGS-B run from pyspi's point at T = 4096 would settle it.
6. **Signed or unsigned?** Measured: the mean statistics take both signs. Examples are `bary_dtw_mean` −0.051 to 0.033 (bivariate_gaussian T = 64) and `bary_euclidean_mean` on the kernel path −1.05 to 2.83. The max statistics on z-scored data were ≥ 0 in every run (minimum off-diagonal 0.97 in the symmetry runs). The `signed` class label is accurate for the mean statistics. The documentation's "unsigned" is accurate only for the max statistics on z-scored input.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls come from `audit/profiles/summary.json` and shares are percentages of the profiled `compute_spi` frame at 8x1024. Nothing in pyspi was changed and no micro-benchmark was written for this family.

Only `distance.bary.euclidean` is in the retained EEG selection, with all four of its identifiers. `Barycenter` is `Directed`, but it caches each pair's barycenter under (i, j) and (j, i) on the Data object (`data.barycenter[mode]`). The work is therefore already per unordered pair, and the mean/max/squared identifiers of a mode reuse one barycenter.

### Walls

| unit | 8x1024 | 32x1024 | 118x1024 per window | measure_mode |
|---|---|---|---|---|
| bary.euclidean | 0.0061 s | 0.103 s | 1.52 s (fresh, representative); 1.93 s (family-warm, 4 identifiers) | measured / extrapolated-pairs-from-8 |
| bary.dtw | 3.23 s | 53.3 s | 738 s | extrapolated-pairs-from-32 (t_fixed 0.24 s) |
| bary.sgddtw | 2.32 s | 41.8 s | 581 s | extrapolated-pairs-from-32 |
| bary.softdtw | 56.1 s | not run | 13831 s | extrapolated-pairs-from-8 |

Complexity per unit:

- bary.euclidean: O(P^2 T), but dominated by per-pair Python overhead. Its T_exponent of -1.57 is an artefact of a ~50 ms 8x256 overhead.
- bary.dtw and bary.sgddtw: O(P^2 n_iter T^2). DTW paths are O(T^2), and the path matrices are dense T x T. T_exponent is 2.45 (dtw) and 2.20 (sgddtw).
- bary.softdtw: O(P^2 n_eval T^2). About 11 L-BFGS-B evaluations per pair at 8x1024, each running two soft-DTW forward and backward passes. T_exponent is 1.75.

### Where the time goes

- **bary.euclidean (16 ms profiled).** The arithmetic is cheap; per-pair overhead dominates.
  - `tslearn/utils/utils.py:to_time_series_dataset` takes 65.9%. Of that, `instantiate_backend` is 50.6%, and its numpy arrayprint (`str()` of the input) is 45.1%.
  - nanmean takes 13.4% and `np.average` 12.7%.
- **bary.dtw.**
  - `tslearn/barycenters/dba.py:_subgradient_valence_warping` is 48.1% cumulative and 30.8% self: a Python loop that sets `w_k[i, j] = 1.` on a fresh `numpy.zeros((T, T))` per series per iteration. `numpy.zeros` alone is 11.2%.
  - `_mm_assignment` (numba DTW path, hidden) is 28.7%.
  - `_mm_update_barycenter` is 19.7%: dense `dot` 13.3% overall and `numpy.diag(1/V)` 5.7%.
- **bary.sgddtw.**
  - `_subgradient_valence_warping` is 44.0% (self 24.9%).
  - `_mm_assignment` is 39.9%.
  - `_subgradient_update_barycenter` is 13.0%.
- **bary.softdtw.**
  - `tslearn/metrics/softdtw_variants.py:SoftDTW.grad` is 56.8% (self 50.7%, numba backward pass).
  - `SoftDTW.compute` is 28.6% (numba forward pass).
  - The glue is about 16%: astype 4.7%, safe_sparse_dot 3.8%, vstack 3.0%, hstack 2.4%, jacobian_product 1.1%, sklearn in-place 1.0%.

### Opportunities

- **barycenters-1: path-index accumulation in place of dense T x T path matrices (bary.dtw ~2x, bary.sgddtw ~2x; effort M).**
  - Change: a numba pass over each DTW path gives the valence vector and the warped sums in O(T). It replaces `numpy.zeros((T, T))`, the Python fill loop, `w_k.dot(x_k)` and `numpy.diag(1/V).dot(...)`.
  - Size: the removable share is 48.1% + most of 19.7% for dtw (Amdahl 2.5-3x) and 44.0% + most of 13.0% for sgddtw (~2.2x). The DTW paths remain.
  - Risk: unverified-within-tolerance, because the order of the sums over path entries changes. DBA's |cost change| < 1e-5 stopping test could occasionally stop at a different iteration.
  - sgddtw consumes the global RNG (one `rng.permutation(2)` per epoch). This change keeps that consumption the same, but any pair-parallel or reordered execution would change the stream (rng-stream).
- **barycenters-2: vectorise bary.euclidean over pairs (30-100x, unverified-bit-identical, effort S).**
  - Change: form (z_i + z_j)/2 for all i < j in one array and reduce each row with nanmean/nanmax (of the square for bary-sq_*). This is the same element-wise arithmetic and the same contiguous-row reduction.
  - Size: P^2 T/2 = 7.1M element operations per statistic at 118x1024, tens of milliseconds, against 1.52 s measured.
  - bary_euclidean_mean is degenerate (score 0): on z-scored rows it is (mean z_i + mean z_j)/2, rounding noise around 0. Its optimisation is trivial, and a constant 0 would differ only by that noise (within-tolerance).
- **barycenters-3: fuse the squared-Euclidean cost and Jacobian into the soft-DTW barycenter objective (~1.2x, unverified-within-tolerance, effort M).**
  - Size: removes the ~16% glue. The two numba recursions (85%) stay.
  - Risk: L-BFGS-B with ftol 1e-3 can turn rounding differences into a different stopping iterate. The check stage found the returned point is not at the optimum.
  - Beyond this, only process-level pair parallelism is left (`xc-window-parallel`).

### Seed items refuted or narrowed

- **"DBA/sgddtw: process parallelism only" is refuted.** 44-48% of each call is a Python loop over dense path matrices (barycenters-1). For bary.softdtw the seed holds: 85% is in the numba recursions.
- **"bary_euclidean vectorised over pairs" is confirmed** (barycenters-2), and the mean statistic is degenerate.

### EEG cost (retained selection)

bary.euclidean has four EEG identifiers: the same four configured identifiers, with no band variants. Family-warm increments at 8x1024, scaled by pair count to 118x1024:

| identifier | 8x1024 | 118x1024 | |
|---|---|---|---|
| bary_euclidean_mean | 6.13 ms | 1.51 s | first: computes and caches the barycenters |
| bary_euclidean_max | 0.34 ms | 0.08 s | reads the Data cache |
| bary-sq_euclidean_mean | 0.96 ms | 0.24 s | reads the Data cache |
| bary-sq_euclidean_max | 0.38 ms | 0.09 s | reads the Data cache |

That is about **1.9 s per window** (family-warm projection 1.93 s; fresh measured 1.52 s for the first identifier), about 0.3 core-hours per 600-window patient. The Data-level barycenter cache already makes the three later identifiers 5-16% of the first. barycenters-2 would bring the whole set to well under 0.1 s per window.

### Shared versus incremental cost

The first identifier of each mode carries the barycenter cost. At 8x1024 those are bary_dtw_mean 3.11 s, bary_sgddtw_mean 2.31 s, bary_softdtw_mean 56.5 s and bary_euclidean_mean 6.1 ms. The other three identifiers per mode cost 0.3-1.0 ms each, through cache hits.
