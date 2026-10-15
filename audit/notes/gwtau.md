# gwtau family: define, trace and check

Family `gwtau` contains one unit, `distance.gwtau` (identifier `gwtau`, class
`GromovWasserstainTau`, no hyperparameters). The record is `audit/records/distance.gwtau.yaml`.
Probe scripts (read-only) live in the session scratchpad under `gwtau/` (`probe1.py`, `probe2.py`,
`probe3.py`). Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Statements marked *hypothesis* or *candidate* in the trace sections are resolved in the record's `candidate_resolutions` and in `## Checks`.

## distance.gwtau

### Documented definition

The supplement has no section for this SPI: `supplement.txt` contains no "Gromov", "Wasserstein" or
"gwtau" entry. It predates the SPI. The only documentation is the GitBook page
`spi-descriptions/distance-similarity.md` (lines 164-180), which lists the keywords *undirected,
nonlinear, unsigned, unordered, distance*. The page says that GWτ "represents each time series as a
metric space and comput[es] the distances from the start of each time series to every point. These
distance distributions are then compared using the Wasserstein distance ... making it robust to shifts
and perturbations". It cites Kravtsova et al. (2023). The class docstring
(`pyspi/statistics/distance.py:267`) reads "Gromov-Wasserstain distance (GWTau)". The misspelling also
appears in the class name and in `config.yaml:313-314`.

### Canonical definition

Kravtsova, McGee & Dawes (2023), *Bull. Math. Biol.* 85:77, doi:10.1007/s11538-023-01175-y (open access
as PMC10326159):

- A 1-D series becomes the planar trajectory f(t) = (t, h(t)) with points x_i = f(t_i), i = 1..n
  (Def. 2, Eqs. 6-7).
- Each point has uniform mass 1/n. In Def. 2 this measure is an unnumbered display after Eq. 9.
- The intrinsic distance is the polygonal arc length d_X(x_i, x_j) = Σ_{k=i}^{j-1} ‖f(t_{k+1}) − f(t_k)‖
  (Eqs. 8-9).
- The root is r_X = f(t_1).
- GWτ_p(X, Y) = inf over couplings of (Σ |d_X(r_X, x) − d_Y(r_Y, y)|^p μ(x, y))^{1/p} (Eqs. 5 and 10).
  Unlike GW, there is no factor 1/2.
- Prop. 1(a): GWτ = W_p between the push-forwards of the root-distance distributions.
- Prop. 1(b): GW ≤ GWτ, so GWτ is an upper bound on GW. Mémoli's TLB is a lower bound.
- The experiments use p = 2.
- Closed forms: Eq. 11 (equal sizes, sorted matching) and Eq. 12 (unequal sizes, quantile-overlap
  weights λ_ij), both in the proof of Prop. 1(e).

Equation labels were read from the Europe PMC JATS full text of PMC10326159 (the `<label>` of each
display formula; the measure display has none). The journal PDF was not consulted separately.

The authors' code is at github.com/kravtsova2/GWtau, in `GWtau_tutorial/vec_geo_dist.m`,
`wass_sorted.m` and `gwtau_tutorial_main.m`. That code:

- uses `time = (0:n-1)'`;
- returns only the n−1 cumulative distances, leaving out the root;
- sorts in descending order.

### What is implemented (trace)

1. `pyspi/calculator.py:277` builds `Data(..., normalise=True, detrend=False)` by default (defaults at
   `calculator.py:50`; lines at HEAD a42ebc1).
   `pyspi/data.py:184-195` applies the optional detrend, then `zscore(ddof=1)`.
2. `pyspi/calculator.py:329` (inside `Calculator.compute`, 327-339) calls `multivariate`, which is `Undirected.multivariate`
   (`pyspi/base.py:132-138`). It calls `Directed.multivariate` (`base.py:85-94`), which computes every
   ordered pair i ≠ j, and then mirrors the upper triangle into the lower one.
3. `distance.py:317-322`: `bivariate` takes the rows `x, y = data.to_numpy()[[i, j]]` (each of shape
   (T, 1)) and calls `gwtau(x, y)`.
4. `distance.py:304-315`: `gwtau` builds the trajectory `column_stack([arange(T), x])`. Time is the
   sample index with unit step; amplitude is in SD units.
5. `distance.py:273-277`: `vec_geo_dist` computes `cumsum(norm(diff(traj)))`. This gives T−1 values
   L_x(k) = Σ_{t<k} √(1 + Δx_t²), k = 1..T−1, without the root.
6. `distance.py:279-302`: `wass_sorted` sorts both vectors in descending order.
   - Equal lengths: it returns √(mean (L_x − L_y)²).
   - Unequal lengths: a vectorised λ coupling. pyspi cannot reach this branch, because `Data` stores
     equal-length rows and two 1-D arrays of different lengths raise
     `ValueError: cannot reshape array of size 80 into shape (1,100,1)` in `parse_bivariate` → `Data`.

The code is a line-for-line Python port of the authors' MATLAB functions.

Probe confirmations:

- A from-scratch `sqrt(mean((L_i-L_j)^2))` on z-scored data reproduces `A[0,1]` with a difference of
  exactly 0.0 (T = 256).
- `wass_sorted` with unequal lengths equals an independent quantile-function W₂ (0.3499664167975089 vs
  0.34996641679750884).

### Deviation analysis

**Against the paper**

- *Root exclusion (candidate, shared with the authors' code).* The paper puts uniform mass on all n
  points, including the root at distance 0. pyspi uses T−1 points. For equal lengths this is an exact
  constant factor: pyspi = canonical × √(T/(T−1)). Probe at T = 256: canonical / pyspi =
  0.998044963916957 = √(255/256). The factor is negligible for ranking at fixed T, but it is a real
  definitional difference.
- *Time axis.* The time coordinate is always the sample index, and it is paired with z-scored amplitude.
  This implicitly fixes a time:amplitude scale of 1 sample : 1 SD. The paper allows real, unequally
  spaced times and does not prescribe normalisation. The authors' tutorial uses the same index time.
- p is fixed at 2, the paper's experimental choice. Descending sort of both vectors is equivalent to
  Eq. 11's ascending sort.

**Against the documentation**

- *"unordered" label (candidate).* No documentation defines either "unordered" or "temporal".
  - The GitBook keyword table (`spis/glossary-of-terms.md:21`) defines 12 keywords, including
    Contemporaneous and Time-dependent, but neither of these two.
  - `development/development/incorporating-new-spis.md:430,546` describes labels only as descriptive
    keywords used for filtering. That page uses gwtau's own labels as its worked example.
  - So this is not a deviation from a documented definition. It is an inconsistency with how sibling
    SPIs use the labels.
  - The statistic depends on temporal order through Δx_t. Probe: one common random permutation of both
    series changes the value from 13.527 to 5.126 (T = 256).
  - The order-dependent distance SPIs carry the label "temporal": `distance.py:119` (dcorrx), `:139`
    (mgcx), `:156` (dtw, lcss, softdtw), `:215` (bary), and `basic.py:78` (xcorr).
- *"distances from the start ... to every point".* The code leaves out the start point itself.
- *"robust to shifts" (candidate).* The GitBook does not say what kind of shift it means. There are
  two separate points.
  - **The claim contradicts the source paper for time shifts.** [Corrected at the check stage: the paper
    also says slightly shifted trajectories get a small distance, so the GitBook sentence is an incomplete
    paraphrase, not a contradiction; see "Corrections to trace-stage fields".] Kravtsova et al. (2023), section
    "Empirical Results on Synthetic, Model-Based, and Publicly Available Real-World Data", say: "GWτ is
    able to distinguish between shifts since a shifted version of a trajectory has a different
    distribution of intrinsic distances, and the larger the shift, the easier it is for GWτ to
    distinguish between the trajectories". They call capturing time shifts "a desirable property".
    Shift sensitivity is intended behaviour, whatever the sampling.
  - **The paper's narrower claim fails on sampled data.** The paper's only robustness statement is that
    "slightly shifted versions" of oscillations "will be matched by an optimal transport routine
    producing a small distance". A phase shift relative to the sampling grid breaks this.
  - The only shift the value is exactly invariant to is a constant offset x → x + c.
  - Delayed white noise stays small compared with independent pairs: 3.6-6.5 vs 22-53 at T = 4096. A
    phase-shifted sampled sinusoid does not.
  - On `narrowband_f0-0.25`, the π/4-shifted pair gives 16.2 at T = 256 and 267.9 at T = 4096. The
    −π/2-shifted pair gives 0.53 and 2.1.
  - At f0 = 0.25, the noise-free z-scored series have |Δz| = √2 at every step for phases 0 and −π/2.
    For phase π/4 the steps alternate between 0 and 2. Arc length per two steps is therefore
    2√3 = 3.464 against 1 + √5 = 3.236.
  - This linear drift predicts about 0.114·T/√3 ≈ 269.6 at T = 4096.
  - Noise-free probe (`gwtau_fix/probe_shift.py`): the z-scored phase-0 and π/4 cosines give 16.50 at
    T = 256 and 269.27 at T = 4096, against predictions of 16.85 and 269.63. The −π/2 copy gives about
    1e-13.
  - This grid effect comes from the canonical construction on sampled data, not from a porting error.
    It is separate from the paper's intended sensitivity to time shifts.

**Structural observations (candidates, inherited from the definition)**

- Every segment has length ≥ 1, so the cumulative vectors are strictly increasing (probe: minimum step
  1.0000647). The sort is therefore a no-op, and gwtau =
  √(mean_k (Σ_{t<k} [g(Δx_t) − g(Δy_t)])²) with g(u) = √(1 + u²). It is an RMS of the running
  difference in local roughness. It is not an order-free comparison of distributions.
- The value is not normalised by length. Under independence it grows roughly as √T: the mean over 20 iid
  pairs is 4.09, 8.67, 15.2 and 32.6 at T = 64, 256, 1024 and 4096.
- It is a pseudometric: gwtau(z, −z) = 0 exactly, and gwtau(z, z + c) ≈ 1e-14.
- With z-scoring it is invariant to per-series affine maps. On `scaled_offset` the matrix equals the
  `bivariate_gaussian` matrix to the printed digits.
- With `normalise=False` it depends on the raw units: on `scaled_offset` at T = 1024, row 0 is
  929-6022 raw against 9.6-21.0 z-scored.

### Direction convention

The SPI is undirected. A[i, j] = gwtau(z_i, z_j) = A[j, i]. The formula is exactly symmetric, and
`Undirected.multivariate` mirrors the upper triangle (`base.py:136-137`). In probes, A equals A.T bitwise
on 5 datasets, and `bivariate(i=0, j=1)` equals `bivariate(i=1, j=0)`. Nothing here is already
documented in KNOWN_ISSUES.md.

### Proposed checks (full list in the record)

1. **Pipeline vs from-scratch numpy (independent-impl, tolerance 1e-12).** Reference: z-score, cumsum of
   √(1 + Δz²), then RMS difference. Datasets: iid_gaussian, delayed_noise, var2_bivariate,
   narrowband_f0-0.25 and trending, at every T.
2. **Root-inclusion analytic factor (kernel, exact).** canonical / pyspi = √((T−1)/T).
3. **Same-lineage transcription of the MATLAB loops (kernel, tolerance).** Covers both branches.
   Labelled same-lineage.
4. **Unequal-length branch vs quantile-function W₂ (kernel, independent-impl).**
5. **Direction / symmetry (pipeline, direction).** A == A.T bitwise. On var2_bivariate the X→Y and Y→X
   cells match.
6. **Affine invariance under z-scoring (pipeline, tolerance).** scaled_offset ≡ bivariate_gaussian.
   Contrast with `normalise=False` (qualitative).
7. **Invariances (kernel, qualitative).** Sign flip and constant offset give 0. A joint permutation
   changes the value, which refutes "unordered".
8. **Null √T scaling (pipeline, statistical).** The log-log slope should be about 0.5 on iid_gaussian.
9. **Shift robustness (pipeline, qualitative).** Compare narrowband_f0-0.25 pairs (0,1) and (0,2) across
   T, and delayed_noise against iid_gaussian.

## Open questions

- Should pyspi follow the paper's Def. 2, with the root included (factor √((T−1)/T)), or the authors'
  MATLAB code, which pyspi currently mirrors? The check stage can quantify this, but the choice is
  definitional.
- Should the "unordered" keyword be changed to "temporal" for gwtau? This is a labelling decision. It
  links to REVIEW M9's label discussion, but M9 does not name gwtau.
- The implicit time:amplitude scale (unit sample step against z-scored amplitude) acts as a hidden
  hyperparameter. Would a sampling-rate or time-scale parameter, as the paper's real t_i would allow, be
  appropriate for EEG at different sampling rates? Hypothesis: gwtau values for the same signals
  resampled at a different rate are not comparable.
- Is the phase/sampling-grid sensitivity (narrowband_f0-0.25) significant for EEG, where oscillations are
  close to rational fractions of fs? This remains a hypothesis until the check stage quantifies it.
- The √T growth under the null means gwtau values are not comparable across recordings of different
  lengths. Should the documentation say so?

## Checks

All numbers below are quoted from `audit/checks/out/results/gwtau.json` (run ids are given with the
`distance.gwtau|` prefix omitted). The family manifest has 114 runs, and all 114 pass with a reference.
The files are:

- manifest: `audit/checks/manifest/gwtau.json`
- references: `audit/checks/refs/gwtau.py`
- probes: `audit/checks/probes/gwtau_probes.py`
- Octave transcription: `audit/checks/octave/gwtau.m`

Every pyspi run takes at most 0.017 s. The regression goldens were not used; they are snapshots, not a
reference.

### distance.gwtau: variant, score 8 (anchor 8: constant factor such as T/(T−1))

**What was compared with what.**

- **Definition, exact (46 `def` runs).** pyspi's full matrix, both orientations, was compared with a
  from-scratch GWτ that never imports pyspi: `_kernels_distance.gwtau`, which is `root_arc_lengths` plus
  `wasserstein2_1d`. That 1-D W₂ is computed on merged quantile breakpoints, not by sort-and-pair.
  Coverage: 21 datasets on the kernel, pipeline and pipeline-detrend paths, T = 64 to 4096.
  max_rel 1.2e-13, max_abs 1.7e-12.
- **Paper Def. 2 with the root included (`canon`, 6 runs).** canonical/pyspi = √((T−1)/T) holds to
  2.2e-16 at every T. pyspi is larger than the paper's value by 0.79 % at T = 64, 0.20 % at 256,
  0.049 % at 1024 and 0.012 % at 4096.
- **External package (`pot`, 5 runs).** POT 0.9.7.post1 `ot.wasserstein_1d(p=2)` on the same
  arc-length samples gives max_rel 1.9e-15.
- **Unequal-length branch.** pyspi's Calculator cannot reach this branch, so a probe calls
  `GromovWasserstainTau.gwtau` directly on 7 prefix-length pairs. Results:
  - quantile W₂: 1.3e-15 (`iid_gaussian|256|kernel|unequal-quantile`)
  - POT: 1.7e-14
  - Octave: 1.5e-15
- **Authors' code, same-lineage.** `octave/gwtau.m` is a loop transcription of the authors'
  `vec_geo_dist.m` and `wass_sorted.m`. Both files were re-read on 2026-10-15; the transcription only
  moves the authors' in-loop `res=sqrt(my_sum)` out of the loop, which does not change the value. It
  agrees with pyspi to 6.5e-15. This only confirms the port; it does not verify the definition.
- **Closed forms for sampled sinusoids.** `sinusoid_steps` computes the mean, SD and increments of a
  z-scored sampled sinusoid from Dirichlet sums, with no sample differencing.
  - f = 0.1 against f = 0.4 (`sin-freq`): max_abs 1.4e-11. The leading-order law
    |s̄(0.1) − s̄(0.4)|·T/√3 (s̄ = 1.1699 and 2.0521 mean steps) is within 6.1e-5 relative of the
    (0,1) value at T = 4096.
  - The noisy dataset rows `narrowband_f0-0.1` stacked over `narrowband_f0-0.4` (`xfreq`) are exact
    (2.2e-16). They lie within 0.6 % of the noise-free cross-frequency closed form at T = 4096.
  - The f = 0.25 phase grid (`sin-grid25`) gives 4.6e-11, and the EEG-like frequencies (`sin-eeg`)
    give 2.1e-11.
- **Invariances.**
  - On the pipeline path, scaled_offset equals bivariate_gaussian to 3.4e-14 relative (`affine`,
    T = 64, 1024 and 4096).
  - gwtau(x, −x) and gwtau(x, x + 5) are ≤ 4.8e-14 on the raw kernel path (`invar`).
  - gwtau(x, 3x − 2) is 0.0 after z-scoring but 1198.4 on the raw path.
  - The bit-identical pair in shared_envelope gives exactly 0.0 (`ident`).
  - Linear detrending recovers the untrended value to 5.0e-14 (`detrendinv`).
- **Symmetry.**
  - max|A − Aᵀ| = 0.0 on every `sym` run.
  - The `unmirrored` probe calls `base.Directed.multivariate`, which computes bivariate(i, j) and
    bivariate(j, i) separately. Its matrices are bitwise symmetric and equal to the reference
    (1.9e-16). The mirroring step in `Undirected.multivariate` therefore changes nothing.
  - On var2_bivariate (x → y), var2_chain, delayed_noise, logistic_maps_chaotic and anm, both
    orientations are identical: the value carries no direction.
- **Null.**
  - Mean over the 10 iid_gaussian pairs: 3.42, 8.87, 19.65 and 36.55 at T = 64, 256, 1024 and 4096.
    The Monte Carlo means are 4.32, 8.74, 17.33 and 34.95. Every entry lies inside the empirical
    Bonferroni band (`null`).
  - The log-log slope is 0.570 against 0.502 for the Monte Carlo means (`slope`).
  - The analytic E[gwtau²] = γ₀T/2 + γ₁(T−2) treats the running sum of g(Δx) − g(Δy), with
    g(u) = √(1 + u²), as a random walk with 1-dependent increments: E g(U) = 1.6045,
    Var g(U) = 0.4256, lag-1 covariance 0.1021. It gives 39.9, 160.8, 644.5 and 2579.0, against Monte
    Carlo values of 39.1, 159.4, 652.7 and 2558.8 (|z| ≤ 1.66, `null-m2`).
- **Delays.** Delayed white noise stays below the 1 % null quantile (`delay`):
  - T = 1024: maximum 2.99, against a 1 % quantile of 6.42.
  - T = 4096: maximum 6.47, against 12.79.
- **Order dependence.** A common random time permutation changes the value (`perm-change`):
  - iid_gaussian: by 9.0-73.8 %
  - var2_bivariate: by 6.9 %
  - narrowband_f0-0.1: by 33-69×

  The `perm-change` pass flags compare pyspi with pyspi (permuted against original rows), so these runs
  are labelled same-lineage. The independent evidence for order dependence is:
  - pyspi equals the from-scratch kernel on the permuted rows (`perm-def`, 2.5e-16) and on the original
    rows (`def`);
  - the kernel alone, with no pyspi value, changes by the same amounts (`perm-change`
    kernel_rel_change_min 0.0905, 0.0691 and 33.24; the kernel and pyspi relative changes agree to
    ≤ 7.1e-15).

  Joint time reversal changes the median value by 28 % (`rev`).

**Changes to the draft references from the interrupted attempt.** These recalibrate statistical bands to
the skew of the null distribution. They do not loosen a tolerance on a pyspi discrepancy: on the same rows
pyspi equals the from-scratch value exactly.

- **`null`.** The null distribution is right-skewed (skew 1.14-1.24). The draft's band was
  mean ± z·sd: its lower bound was negative, and its upper bound at T = 4096 was 79.9, below the empirical
  99.75 % quantile of 97.8. The band is now the empirical Bonferroni interval.
- **`null-m2` (post hoc change of criterion).** The draft rule was "observed pair mean of A² within 2 MC
  sd of the analytic value". It flagged the T = 256 draw. With the current Monte Carlo it gives
  z = 2.64 there (`drafted_rule_z`, `drafted_rule_ok` = 0): pyspi's mean A² is 404.5 against an analytic
  160.8, with Monte Carlo sd 92.3. The other three T give z = −0.85, 0.17 and 0.62. The pair-mean
  statistic has skew 1.3-1.5, and 404.5 is at the 98th Monte Carlo percentile, inside the 98.75 %
  Bonferroni band.
  - The new rule was chosen after the drafted rule flagged this draw, so it is a post hoc change of
    criterion and should be read that way. The drafted-rule result stays in results.json.
  - Both rules test whether the data draw is typical of the null. Neither tests pyspi's arithmetic:
    pyspi equals the from-scratch kernel exactly on these rows (`def` runs on the kernel path).
  - The rule was split into two parts. (i) The analytic formula must agree with the Monte Carlo mean.
    (ii) The observed draw must lie inside the empirical band.
  - The T = 256 percentile is reported as it is.
  - The kernel rows of that draw have sample variances 0.85-1.22.

**Additions to the draft.**

- **Sampling-rate probe (`fs`, `fs-ratio`).** The same 16 s continuous pair was sampled at 16, 32, 64,
  128 and 256 Hz.
  - pyspi gives 5.80, 4.37, 2.46, 1.27 and 0.64. These are exact against the reference.
  - The paper's Eq. 8 with real times t_k = k/fs and the root included gives 6.48, 9.43, 9.51, 9.62
    and 9.66.
  - Ratio of the 256 Hz value to the 32 Hz value: pyspi 0.147, real-time form 1.025.
- **EEG-frequency grid probe (`sin-eeg`, `eeg-grid`).**
- **null-m2 at all four T.**

**Verdict.** The computation matches the authors' MATLAB GWτ exactly. It differs from the paper's
Def. 2, and from the GitBook's "to every point" description, only by the constant factor √(T/(T−1)), so
the score is 8. This is not a 10, because the canonical uniform measure includes the root. The supplement
has no gwtau entry.

**Candidates.** All six trace candidates are confirmed; candidate 5 only in a narrowed form:

1. **The "unordered" label.** Confirmed, and new: KNOWN_ISSUES has only the general REVIEW M9 label
   discussion. The statistic is order dependent (perm-change), while the order-invariant pdist SPIs
   carry the same "unordered" keyword (pdist family `…|bivariate_gaussian|1024|pipeline|perm`,
   max_abs ≤ 1.8e-12).
2. **Root exclusion.** Confirmed: exact factor √(T/(T−1)). This is new as an audit statement, and it is
   inherited from the authors' code.
3. **The sort is a no-op.** Confirmed: every segment length is ≥ 1, and the closed-form references that
   never sort match to 1e-11. gwtau is therefore the RMS over time of the running difference of the two
   cumulative arc lengths. This is an interpretation, not a defect.
4. **√T growth under the null.** Confirmed by the slope and the analytic E[gwtau²]. This is a property
   of the canonical definition.
5. **"Robust to shifts".** Confirmed only in a narrowed form. The check-stage evidence mostly supports
   the GitBook sentence in realistic cases:
   - Delayed white noise stays below the 1 % null quantile (`delay`).
   - The f = 10/256 phase pairs do not grow with T (0.0846 → 0.0847, `eeg-grid`).
   - At f = 0.2 the phase pairs grow slowly (0.56 → 1.09 for the π/4 pair).

   The failure is limited to the grid-commensurate case f0 = fs/4:
   - At f0 = 0.25, the π/4 pair grows from 16.18 to 267.91 between T = 256 and 4096 (ratio 16.56). The
     noise-free closed form gives 17.03 and 269.81, with a drift of 0.1140 per step. The −π/2 pair goes
     from 0.53 to 2.12.
   - At T = 4096 the π/4 pair (267.91) is far above the upper end of the iid null band (102.86,
     `iid_gaussian|4096|pipeline|null`). The paper's own claim that slightly shifted oscillations get a
     small distance therefore fails in this case.
   - The trace candidate's other half, that the GitBook sentence "contradicts the paper", is withdrawn.
     The paper text was re-read (see "Corrections to trace-stage fields"). It says both that larger
     shifts are distinguished and that slight shifts give small distances. The unqualified GitBook
     sentence is therefore an incomplete paraphrase, not a contradiction.
6. **Preprocessing dependence.** Confirmed: the raw-to-z-scored ratio is 75.8-624.2 (`rawunits`), and
   the sampling-rate probe adds a dependence on fs.

`new_finding: true`, for candidates 1, 2, 5 (narrowed form) and the sampling-rate dependence. None of them is in
KNOWN_ISSUES.md.

### Corrections to trace-stage fields

- The calculator citations were refreshed to HEAD a42ebc1. The task text named a1f20c7, but the checkout
  is one commit later; `pyspi/calculator.py` lines were read at a42ebc1:
  - `Data(...)`: line 277
  - defaults: line 50
  - `Calculator.compute` block: 327-339, with `multivariate` at 329, `fill_diagonal` at 332, the table
    store at 335 and the `except` at 336-339
  - warning: 354-355
  - `_load_yaml`: def at 230, config loop at 251-267
- The `base.py` (85-94, 132-138) and `config.yaml` (313-320) citations were re-checked and are unchanged.
- One trace claim was corrected. New-finding candidate 5, point (i), said the GitBook's "robust to
  shifts" contradicts Kravtsova et al. (2023). The section "Empirical Results on Synthetic, Model-Based,
  and Publicly Available Real-World Data" was re-read in the Europe PMC JATS full text (PMC10326159,
  fetched 2026-10-15; the journal PDF was not opened). It contains both quotes verbatim:
  - "GWτ is able to distinguish between shifts since a shifted version of a trajectory has a different
    distribution of intrinsic distances, and the larger the shift, the easier it is for GWτ to
    distinguish between the trajectories"
  - "Slightly shifted versions of qualitatively similar trajectories will be matched by an optimal
    transport routine producing a small distance between such trajectories"

  The paper therefore claims small distances for slight shifts and separation for larger shifts. The
  record's candidate text and resolution were rescoped to match.
- The `perm-change` runs were relabelled from independent-impl to same-lineage, because their pass flag
  compares pyspi with pyspi. A kernel-only repeat was added. The outcome and the score are unchanged.

## Open questions: answered

- **Paper Def. 2 (root included) or the authors' MATLAB code?** The difference is now quantified as the
  exact factor √(T/(T−1)): 0.79 % at T = 64 and 0.012 % at T = 4096 (`canon` runs).
  - It does not change rankings at a fixed T.
  - It matters only when values are compared across different T, and there the √T null growth is much
    larger anyway.
  - The choice remains definitional. Following Def. 2 needs one prepended 0 in `vec_geo_dist`.
  - For equal lengths, the paper-form value is pyspi × √((T−1)/T).
- **Change "unordered" to "temporal"?** The evidence supports it.
  - gwtau changes by 6.9 % to 69× under a joint time permutation.
  - The pdist SPIs that share the "unordered" keyword are invariant to ≤ 1.8e-12.
  - The order-dependent siblings carry "temporal".
  - The decision belongs to the M9 label discussion, which should name gwtau.
- **Hidden time:amplitude scale and sampling rate.** The hypothesis is confirmed (`fs-ratio`).
  - For the same 16 s of signal, pyspi's value falls by a factor of 6.8 between 32 Hz and 256 Hz:
    4.37 → 0.64, roughly ∝ 1/fs at high rates.
  - The paper's form with real times converges: 9.43 → 9.66.
  - gwtau values from recordings at different sampling rates are therefore not comparable. A
    time-step parameter (dt = 1/fs in the trajectory (t, x)) would make the value
    sampling-rate-consistent.
- **Phase-grid sensitivity for EEG.** The effect is the q-th Fourier harmonic of g(c cos θ) for
  f = p/q, so it falls off quickly with the denominator q (`sin-eeg`, `eeg-grid`). Per-step drift between
  phase-shifted copies:
  - 0.114 at f = 1/4
  - 2.6e-4 and 5.2e-4 at f = 1/5 (50 Hz line noise at fs = 250 Hz)
  - 0 to rounding at f = 10/256 (10 Hz alpha at fs = 256 Hz)

  Over T = 256 → 4096, the f = 10/256 phase pairs stay at 0.0846 → 0.0847. The f = 0.2 pairs grow
  0.56 → 1.09 (π/4) and 0.64 → 1.72 (−π/2). The effect is therefore negligible for broadband EEG and for
  alpha at these rates, and small but visible for line noise at fs/4 or fs/5. The extreme case is
  fs = 4 × the line frequency (f0 = 0.25).
- **Document the √T null growth?** Yes; the evidence is clear.
  - The mean null value is 4.32, 8.74, 17.33 and 34.95 at T = 64, 256, 1024 and 4096 (Monte Carlo,
    z-scored).
  - The slope is 0.50, and the analytic E[gwtau²] = γ₀T/2 + γ₁(T−2) agrees.
  - gwtau is not comparable across recording lengths unless it is divided by √T.
- **Equation numbers against the journal PDF.** Not settled here. The check used only the
  definitions: Eq. 8 arc length, uniform measure over all n points, and W₂ by sorted matching.
  Spot-checking the labels needs the Springer PDF, which was not opened in this stage.

## Optimisation

Workflow 3 (optimisation search, 2026-10-15). Walls come from `audit/profiles/summary.json` and shares are percentages of the profiled `compute_spi` frame at 8x1024. Nothing in pyspi was changed and no micro-benchmark was written. Not in the retained EEG selection.

### Walls

| shape | wall | measure_mode |
|---|---|---|
| 8x1024 | 5.4 ms | measured |
| 32x1024 | 92 ms | measured |
| 118x1024 | 1.31 s per window | measured |

**Complexity: O(P^2 T log T).** Each ordered pair computes two arc-length vectors (`gwtau` stacks time and value with column_stack, then `vec_geo_dist`: diff, norm, cumsum) and two descending sorts, then takes the RMS difference (`wass_sorted`). Undirected over `Directed.multivariate` means both orders are computed. P_exponent is 2.04 and T_exponent 0.37: per-call overhead dominates at T <= 1024.

### Where the time goes (7.7 ms profiled)

- `pyspi/statistics/distance.py:GromovWasserstainTau.vec_geo_dist` takes 53.6% (numpy linalg.norm 32.7%, cumsum 9.0%, diff 7.8%). The trajectory `column_stack` (7.0%) is called from `GromovWasserstainTau.gwtau`, not from `vec_geo_dist`.
- `wass_sorted` takes 22.0% (np.sort 10.5%).
- The owner is pyspi. opaque_share is 0.49: the time is in numpy builtins.

### Opportunities

- **gwtau-1: per-channel cache of the arc-length vector and its descending sort (3-50x, unverified-bit-identical, effort S).**
  - Both depend on one channel only. Each channel would be processed once per window, P = 118 times instead of 2P(P-1) = 27612 times at 118: two arc-length vectors and two sorts per ordered pair, over P(P-1) = 13806 ordered pairs.
  - Each pair keeps only sqrt(mean((a_i - a_j)^2)) over the cached sorted vectors: the same operations, so the output should not change.
  - The low end keeps pyspi's per-pair dispatch and removes only the per-channel work: vec_geo_dist 53.6% + column_stack 7.0% + sort 10.5%, about 71%, so the Amdahl bound is about 3.5x. Only the high end, which bypasses the per-pair Python loop and vectorises the RMS differences, goes beyond that. A dot-product expansion over pairs would be within-tolerance instead.
  - The pair-loop halving is the cross-cutting `xc-undirected-upper`. The unmirrored probe is bitwise symmetric.

### Seed items

- **"gwtau vectorised arc lengths and sorting" is confirmed.** The arc lengths, the trajectory column_stack and the sort are about 71% of the call (53.6 + 7.0 + 10.5%).

### Shared versus incremental cost

Single identifier: the family-warm increment equals the fresh wall (5.5 ms at 8x1024, 1.31 s at 118x1024).
