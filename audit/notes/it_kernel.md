# it_kernel family: define, trace and check

Family `it_kernel` covers the JIDT box-kernel estimator in `pyspi/statistics/infotheory.py`. It has ten
configured SPIs and one infrastructure unit:

| unit_id | identifier | class | kernel width |
|---|---|---|---|
| infotheory.je.kernel | je_kernel_W-0.5 | JointEntropy | 0.5 |
| infotheory.ce.kernel | ce_kernel_W-0.5 | ConditionalEntropy | 0.5 |
| infotheory.cce.kernel | cce_kernel_W-0.5 | CausalEntropy | 0.5 |
| infotheory.xme.kernel_k1 | xme_kernel_W-0.5_k1 | CrossmapEntropy | 0.5 |
| infotheory.xme.kernel_k10 | xme_kernel_W-0.5_k10 | CrossmapEntropy | 0.5 |
| infotheory.di.kernel | di_kernel_W-0.5 | DirectedInfo | 0.5 |
| infotheory.si.kernel | si_kernel_W-0.5_k-1 | StochasticInteraction | 0.5 |
| infotheory.mi.kernel | mi_kernel_W-0.25 | MutualInfo | 0.25 |
| infotheory.tlmi.kernel | tlmi_kernel_W-0.25 | TimeLaggedMutualInfo | 0.25 |
| infotheory.te.kernel_k-1_kt-1_l-1_lt-1 | te_kernel_W-0.25_k-1 | TransferEntropy | 0.25 |
| infotheory.est.kernel | (infrastructure) | JIDTBase + JIDT kernel package | n/a |

Each unit has a record in `audit/records/<unit_id>.yaml`. The read-only probes are in the session scratchpad
under `it_kernel/probes/`:

- `boxref.py`: a from-scratch numpy box-kernel reference.
- `probe1_props.py` to `probe7_more.py`: the probes cited below.
- `it_kernel_fix/p1_frame_all.py` to `p4_pickle.py` (scratchpad `it_kernel_fix/`): probes added after the
  adversarial review (copy path, kernel properties, DI Gaussian plug-in, pickle round trip).

The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

**JIDT build.** The jar `pyspi/lib/jidt/infodynamics.jar` is labelled 1.6.1 (`version-1.6.1.txt`), but it is
a post-1.6.1 master build. All entries are dated 2024-04-21, and the jar contains
`EntropyCalculatorMultiVariateCommon.class`, which does not exist at the v1.6.1 tag. At runtime
`EntropyCalculatorMultiVariateKernel`'s superclass is `EntropyCalculatorMultiVariateCommon` and it stores
`NOISE_SEED`. This matches master commit a9b6b3e (2024-04-20), the same provenance the it_kraskov,
it_gaussian and it_kozachenko records give. Of the kernel-path sources, `EntropyCalculatorMultiVariateKernel.java`
and `MutualInfoMultiVariateCommon.java` differ between the v1.6.1 tag and a9b6b3e. `KernelEstimatorMultiVariate`,
`MutualInfoCalculatorMultiVariateKernel`, `TransferEntropyCalculatorKernel`, `KernelEstimatorTransferEntropy`,
`TransferEntropyCommon` and `MatrixUtils` are byte-identical. Line numbers cited as `L…` refer to a9b6b3e for
the two files that differ (and for `EntropyCalculatorMultiVariateCommon`), and are valid for both revisions for
the identical files. Anything called a *candidate* or *hypothesis* in the trace sections is resolved in
the records' `candidate_resolutions` and in `## Checks`. The regression goldens are snapshots of current output and were not used as a reference.

**Main trace result.** On var2_bivariate (T = 256), the from-scratch numpy reference reproduces all ten
SPIs to within 2.3e-14:

- je: 8.9e-16
- ce: 1.8e-15
- mi: 5.6e-17
- tlmi: 1.1e-16
- te: 0
- xme_k1: 2.2e-15
- xme_k10: 2.3e-14
- si: 9.8e-15
- cce: 8.9e-16
- di: 8.9e-16

The reference uses these conventions:

- The box is closed: |dx| ≤ w.
- The norm is the max-norm.
- Each point counts itself.
- The half-width is w_d = width × the n−1 sample SD of each column passed to JIDT.

So the formulas below are what the code computes.

## Shared estimator (infotheory.est.kernel)

**Documented.** Supplement §1.4, p.9 (supplement.txt:405-443; pack lines 414-453) says: "Kernel estimation (kernel) uses a box
kernel method with a specific kernel width (default width of 0.5 standard deviations (denoted by the
modifier W-0.5))". GitBook `information-theory.md:14` repeats this. The supplement cites JIDT v1.5. The GitBook
and the jar's label say v1.6.1; the jar contents are the later master build a9b6b3e (see above). The documentation describes the Theiler window (DCE) for KSG only.

**Canonical.** The box (step) kernel of Kantz & Schreiber (1997, *Nonlinear Time Series Analysis*):
p̂(x_b) = C_b / (N ∏_d 2r_d), with C_b = #{t : ‖x_t − x_b‖_max ≤ r}. Plugging this into −E log p gives the
resubstitution entropy estimator (Ahmad & Lin 1976; review in Beirlant et al. 1997). Schreiber (2000, PRL
85:461) uses the same kernel for transfer entropy.

**Implemented (trace).**

1. The Calculator z-scores every series with ddof = 1 (`calculator.py:277`, `data.py:193-195`).
2. `JIDTBase._setup` (`infotheory.py:84-93`) sets `KERNEL_WIDTH`. It also sets `BIAS_CORRECTION=false` and
   `NOISE_SEED=42`.
   - `BIAS_CORRECTION` is not a property of any kernel class and is ignored (`getProperty` returns None on all
     three).
   - `NOISE_SEED` is stored by `EntropyCalculatorMultiVariateKernel` (through
     `EntropyCalculatorMultiVariateCommon.setProperty` L115-148) and by `MutualInfoCalculatorMultiVariateKernel`
     (through `MutualInfoMultiVariateCommon.setProperty` L245-280); both read back '42'. Seeded noise is added
     only when `NOISE_LEVEL_TO_ADD` is non-zero (Common L200-211; MI Common L627-642), and pyspi leaves it at
     the default 0. So the seed is stored but numerically inert. Setting `NOISE_LEVEL_TO_ADD=1e-8` with seed 42
     changes the kernel entropy by -1.5e-9 bits; the seed alone changes nothing (`p2_props.py`).
   - `TransferEntropyCalculatorKernel` ignores `NOISE_SEED` (`TransferEntropyCommon.setProperty` L316-336
     leaves unknown names unset); `getProperty` returns None.
3. `_getcalc` (`infotheory.py:103-134`) selects `EntropyCalculatorMultiVariateKernel`,
   `MutualInfoCalculatorMultiVariateKernel` or `TransferEntropyCalculatorKernel`.
4. In JIDT, `KernelEstimatorMultiVariate.setObservations` (L189-242) sets
   `kernelWidthsInUse[d] = width × MatrixUtils.stdDev(data, d)`. `stdDev` divides by n−1 (L2436-2447). The
   kernel volume is ∏ 2w_d.
5. `getProbabilityFromBins` (L732-738) and `getProbabilityComparingToAll` (L811-817) both return
   count / N / volume.
6. `stepKernel` (L959-967) rejects a point only when |dx_d| > w_d. The box is therefore closed.
7. The query's own bin is always counted, so the point counts itself.
8. Entropy is −mean ln p / ln 2 (`EntropyCalculatorMultiVariateKernel` L108-125 in a9b6b3e; the observations
   reach the estimator through the inherited `setObservations` and the override `finaliseAddObservations`
   L102-106).
9. MI is mean log(p_xy / (p_x p_y)) (L178-200). The volumes cancel.
10. TE is mean log((C_nps/C_ps)/(C_np/C_p)) (`TransferEntropyCalculatorKernel` L295-321).

pyspi never sets a Theiler window for these configurations, and no noise is added (noise level 0).

**Established by probe (`probe3_ties.py`).** I ran JIDT directly on the grid [0,1,2,3,10] with absolute
width 1:

- H = 2.287943094599. This is exactly the closed-box value with counts [2,3,3,2,1]. An open box would give
  3.3219.
- An isolated point gives a finite entropy: count 1, i.e. self-inclusion.
- MI on the same grid is 1.2879430945989, which is also the closed-box value.

Setting `FORCE_KERNEL_COMPARE_TO_ALL` gives bit-identical TE, so the bin search is exact.

**Deviations.** These are properties of the estimator, not bugs:

- The estimator is resubstitution: no leave-one-out, no bias correction, no Theiler window.
- "Width" in the documentation means half-width. The box side is 2 × width × SD.

With W-0.5 on unit-SD columns the volume is exactly 1, so the kernel entropies equal −mean log2(C/N) and lie
in [0, log2 N].

**Candidate (copy path, confirmed by probe).** A `CalculatorFrame` computes every W-0.5 kernel SPI with width
0.25 while the identifier still reads W-0.5.

- `CalculatorFrame.init_from_list` deep-copies the base Calculator for every dataset (`calculator.py:556-562`,
  `copy.deepcopy` at :559).
- `JIDTBase.__deepcopy__` (`infotheory.py:77-82`) builds `type(self)()` and deep-copies every attribute.
  `copy.deepcopy` of the JPype proxy returns a new Java object of the same class in its default state
  (`KERNEL_WIDTH` 0.25, `NOISE_SEED` 'NONE'), and `_setup` is not re-run.
- Pickling is not affected: `__setstate__` (`infotheory.py:71-75`) rebuilds the calculators through `_getcalc`,
  which re-applies `_setup` (`p4_pickle.py`: width 0.5 kept).

Probe `p1_frame_all.py`, bivariate_gaussian T = 256, A[0,1], Calculator vs CalculatorFrame:

| SPI | Calculator | CalculatorFrame |
|---|---|---|
| je_kernel_W-0.5 | 3.9822834 | 3.7487119 (= direct W-0.25 je) |
| ce_kernel_W-0.5 | 1.9366210 | 1.7266038 (= direct W-0.25 ce) |
| cce_kernel_W-0.5 | 3.4050629 | −1.7929180 |
| xme_kernel_W-0.5_k1 | 1.9814678 | 1.7793581 |
| xme_kernel_W-0.5_k10 | 0.0037641 | −0.9962359 |
| di_kernel_W-0.5 | 5.8235130 | 8.6877201 |
| si_kernel_W-0.5_k-1 | 0.9962347 | 3.2876895 |
| mi / tlmi / te (W-0.25) | identical | identical |

var2_bivariate T = 256 shows the same pattern. The W-0.25 SPIs are unchanged only because 0.25 is JIDT's
default width. The effect is larger than the kraskov TE copy bug, and the frame values of cce and xme_k10 are
negative, outside the [0, log2 N] range that holds for W-0.5 on the Calculator path.

**Candidate (general pyspi quirk).** Calling `bivariate(x, y)` with raw arrays z-scores only `x`.
`Data.add_process` (`data.py:219-240`) sends the first array through `set_data` and appends later arrays
raw. On scaled_offset (T = 256), JointEntropy then differs by exactly log2(SD(y)) = 3.30519. MI, TLMI and TE
are unaffected because the box widths scale with each column's SD (diff 0 / 1e-16).

**Proposed checks.**

- numpy reference against direct JIDT calls for d = 1..10. A ddof=0 or open-box variant of the reference
  must disagree, which pins down the convention.
- Analytic tie-grid test with exact criterion.
- Large-sample limit h_box(ε) = −∫φ log2[(Φ(x+ε)−Φ(x−ε))/(2ε)] dx: 2.04935 bits at ε = 0.5 and 2.04725 bits
  at ε = 0.25, against the true 2.04710.
- Bin-path equivalence (labelled same-lineage).
- Raw-array versus Data path.
- CalculatorFrame versus Calculator for all ten SPIs, with property read-back on the copied calculators and a
  pickle round trip (same-lineage, exact).
- Property read-back and perturbation for `NOISE_SEED`, `NOISE_LEVEL_TO_ADD` and `BIAS_CORRECTION` on the three
  kernel classes (same-lineage, exact).

## Joint entropy (infotheory.je.kernel)

- **Documented.** Supplement §1.4.1, p.10 (supplement.txt:444-450; pack 454-460): "quantifies the uncertainty over the paired
  observations". The formula is lost in the text extraction. Keywords: undirected, contemporaneous.
  GitBook :139-157.
- **Canonical.** h(X,Y) = −E log p(X,Y) (Cover & Thomas 2006, Ch. 8).
- **Implemented.** `JointEntropy.bivariate` (`infotheory.py:232-234`) calls `_compute_joint_entropy`
  (:159-177). This concatenates z_i and z_j into a (T, 2) array, runs JIDT kernel entropy at width 0.5,
  caches the value under ('kernel', 0.5), and fills [i,j] and [j,i]. `Undirected.multivariate` then mirrors
  the matrix.
- **Value.** Because the volume is 1, the result is −mean log2(C_b/T), bounded in [0, log2 T] and never
  negative.
- **Deviations.**
  - Documented: none, apart from the half-width wording.
  - Canonical: only the estimator bias. On iid_gaussian the means are 3.805, 4.004, 4.070 and 4.089 bits
    at T = 64, 256, 1024 and 4096. The large-sample box limit is 4.0987 and the truth is 4.0942.
- **Checks.**
  - numpy reference: tolerance 1e-12.
  - Box-limit convergence on iid_gaussian.
  - Box limit for correlated Gaussians on bivariate_gaussian: numerical rectangle probabilities.
  - Exact symmetry (direction check).
  - Affine invariance on scaled_offset.
  - Bound check: 0 ≤ JE ≤ log2 T.

## Conditional entropy (infotheory.ce.kernel)

- **Documented.** Supplement §1.4.2, p.10 (supplement.txt:451-458; pack 461-468): "uncertainty over the observations in y in the
  context of simultaneously observing x". Keyword: undirected. GitBook :46-64 also says undirected.
- **Canonical.** h(Y|X) = h(X,Y) − h(X).
- **Implemented.** `ConditionalEntropy.bivariate` (`infotheory.py:246-250`) computes JE(i,j) − H(i)
  (`_compute_entropy`, :137-156). So A[i,j] = ĥ(X_j | X_i): row i conditions, column j is the target. The
  conditioning order is already documented as REVIEW M8.
- **Value.** With W-0.5 the value is −mean log2(C_xy/C_x) ≥ 0.
- **Deviations.**
  - Documented: the keyword "undirected" contradicts the Directed class and the asymmetric output. On
    nonlinear_dependence (T = 1024), H(x1|x0) = 0.8608 and H(x0|x1) = 0.9217. This is a *candidate*. It
    may overlap REVIEW M9, which discusses label structure but does not name ce.
  - Canonical: estimator only.
- **Checks.**
  - numpy reference.
  - Direction on nonlinear_dependence: expect A[0,1] < A[1,0].
  - Box limit or Gaussian truth on bivariate_gaussian: 1.310 bits for ρ = 0.8 and 1.979 bits for ρ = 0.3.
  - Bounds.
  - Invariance on scaled_offset.

## Mutual information (infotheory.mi.kernel)

- **Documented.** Supplement §1.4.3, p.10 (supplement.txt:459-466; pack 469-476): "undirected measure of the (potentially
  nonlinear) dependence between paired observations". Identifier mi_kernel_W-0.25. GitBook :161-180.
- **Canonical.** I = E log p(x,y)/(p(x)p(y)) (Cover & Thomas, Ch. 8). Box-kernel version as in Moon et al.
  1995 and Kantz & Schreiber.
- **Implemented.** `MutualInfo.bivariate` (`infotheory.py:267-283`) calls `initialise(1,1)` and
  `setObservations(z_i, z_j)`. JIDT then computes mean log2(N C_xy/(C_x C_y)) at width 0.25 (L178-200).
  `normaliseData` is a no-op (L166-171); normalisation happens only through the widths. No noise is added
  and there is no Theiler window. A bare `except` returns NaN with a message about "Cholesky", which does
  not apply to this estimator.
- **Deviations.**
  - Documented: none. The 0.25 width is stated in the identifier and is also JIDT's default.
  - Canonical: positive small-sample bias. On iid_gaussian the means are 0.739, 0.272, 0.093 and 0.030
    bits at T = 64, 256, 1024 and 4096. On bivariate_gaussian at T = 4096, MI = 0.734 for ρ = 0.8 (true
    0.737) and 0.097 for ρ = 0.3 (true 0.068).
- **Checks.**
  - numpy reference.
  - Gaussian truth (statistical).
  - Null bias.
  - nonlinear_dependence (qualitative). Probe: 1.646 and 1.205 against cross pairs of 0.02-0.03.
  - Symmetry (direction).
  - Raw-array invariance.

## Time-lagged mutual information (infotheory.tlmi.kernel)

- **Documented.** Supplement §1.4.4, p.10 (supplement.txt:467-475; pack 477-485): "undirected measure of the dependence between
  time series x and a time-lagged instance of the series y ... only lag-one". GitBook :235-254 calls it
  "directed".
- **Canonical.** I(X_{t−1}; Y_t) (Schreiber 2000; Kaiser & Schreiber 2002).
- **Implemented.** `infotheory.py:301-317` passes src = z_i[:-1] and targ = z_j[1:] to the same kernel MI at
  width 0.25. A[i,j] = I(X_i,t−1; X_j,t).
- **Direction probe.** On delayed_noise (T = 1024), A[0,1] = 2.647 and A[1,0] = 0.064.
- **Deviations.**
  - Documented: the keyword conflict. The asymmetry is already documented (REVIEW M8). The disagreement
    between the supplement and the GitBook is a candidate.
  - Canonical: estimator bias only. On iid the means are 0.775, 0.290, 0.097 and 0.028 bits.
- **Checks.**
  - numpy reference.
  - Direction on delayed_noise.
  - Gaussian lag-1 MI from the stationary covariances of gauss_ar1_pair and var2_bivariate.
  - Null bias.

## Transfer entropy (infotheory.te.kernel_k-1_kt-1_l-1_lt-1)

- **Documented.** Supplement §1.4.5, pp.10-11 (supplement.txt:476-499; pack 486-510): TE "from a source time series x to a target
  time series y, based on the Takens time-delay embedding", with modifiers l, k, lt and kt. Identifier
  te_kernel_W-0.25_k-1. GitBook :206-231.
- **Canonical.** Schreiber 2000, Eq. 4, with the max-norm kernel estimator, k = l = 1 and lag 1.
- **Implemented.**
  - The constructor (`infotheory.py:326-367`) forwards only `k_HISTORY=1` (`_K_HISTORY_PROP_NAME`
    at `infotheory.py:358`). For kernel it skips the `k_TAU`, `l_HISTORY` and `l_TAU` calls.
  - JIDT `TransferEntropyCommon.setProperty` (L316-336) would throw "Unsupported property" for those
    properties (probe confirmed).
  - `bivariate` (:376-391) calls `initialise()` and then `setObservations(z_i, z_j)`.
  - JIDT builds y_{t−1} (past), y_t (next) and x_{t−1} (source) (L254-292).
  - The widths are 0.25·SD(past) and 0.25·SD(source). The next-value box reuses the past width
    (`KernelEstimatorTransferEntropy` L151-153).
  - The counts come from callbacks with closed comparisons (L145-159).
  - TE = mean log2((C_nps/C_ps)/(C_np/C_p)). A[i,j] = TE(i → j).
- **Direction probes.**
  - delayed_noise: 2.571 vs 0.501.
  - var2_bivariate at T = 4096: 0.348 vs 0.187.
  - logistic_maps (y → x): A[1,0] = 0.066 > A[0,1] = 0.016.
- **Deviations.**
  - The hypothesis that "kernel TE accepts only k_HISTORY" is **confirmed**. l_history, k_tau and l_tau
    are dropped without a warning and do not appear in the identifier. This has no numerical effect for
    the configured l = 1.
  - Outside the configured units: `auto_embed_method` with kernel is accepted but ignored (k stays 1),
    yet the identifier reads `_k-max-4`. The config entry is commented out.
  - The shared next/past width is a minor estimator detail.
  - The null bias is large: 1.440, 1.045, 0.522 and 0.196 bits at T = 64 to 4096.
- **Checks.**
  - numpy reference (probe diff 0).
  - Direction on three datasets.
  - Gaussian TE from stationary covariances (statistical).
  - Null bias.
  - Parameter-forwarding test (exact).

## Cross-map entropy, k = 1 and k = 10 (infotheory.xme.kernel_k1, infotheory.xme.kernel_k10)

**Documented.** Neither the supplement nor the GitBook SPI pages describe this SPI. `table-of-spis.md:73`
lists "Cross-Map Entropy" with the identifiers `xme_kernel_k1` and `xme_kernel_k10`. The code produces
`xme_kernel_W-0.5_k1` and `xme_kernel_W-0.5_k10`.

**Canonical.** I found no published "cross-map entropy". What the code computes is the conditional entropy
h(Y_t | X_{t−1..t−m}). The name suggests convergent cross mapping (Sugihara et al. 2012), which this is not.

**Implemented.** `CrossmapEntropy.bivariate` (`infotheory.py:405-426`) computes targ_future = y[k:] and
src_past = x[k−1:−1]. It then appends further lags with `for i in range(2, k)`. The result is
H(joint) − H(src_past). The loop yields max(1, k−1) lags:

- k = 1 gives one lag: ĥ(Y_t | X_{t−1}). This equals ĥ(Y_t) − Î(X_{t−1}; Y_t).
- k = 2 also gives one lag.
- k = 10 gives 9 lags, x_{t−1}..x_{t−9}.

A[i,j] = ĥ(X_j,t | X_i past). A *low* value means i predicts j. On delayed_noise, xme_k1 gives A[0,1] =
0.180 and A[1,0] = 2.024.

**Deviation candidates.**

- (a) Off-by-one: history_length 10 builds 9 lags. The probe reference with lags 1..9 matched pyspi to
  2.3e-14.
- (b) Degeneracy of k10 at W-0.5. The fraction of points whose 10-D joint box contains only the point itself
  is 100% at T = 256, 99.6% at T = 1024 and 98.7% at T = 4096. xme_k10 therefore stays at about 0 whatever
  the coupling: −0.003 to 0.033 on iid data. On var2_bivariate the ordering even reverses: A[0,1] = 0.024 vs
  A[1,0] = 0.009.
- (c) The SPI is undocumented and its documented identifiers do not match the produced ones.

**Checks.**

- numpy references, including a 10-lag variant.
- Count-saturation diagnostics.
- Direction on delayed_noise and var2_bivariate. For k10 the expected outcome is "no reliable ordering".
- Box limit 2.049 bits on iid for k1.
- Gaussian conditional entropy on gauss_ar1_pair.
- A k = 2 test of the off-by-one.

## Causally conditioned entropy (infotheory.cce.kernel)

**Documented.** Supplement §1.4.7, p.11 (supplement.txt:513-525; pack 524-536): "a sum of conditional entropies (of y given both
the past of x and the past of y) with increasing history lengths ... we restrict the history length to 10".
GitBook :24-42.

**Canonical.** Kramer 1998: H(Y^n‖X^n) = Σ_m H(Y_m | Y^{m−1}, X^m). The conditioning includes the current
X_m. See also Massey 1990 and Quinn et al. 2011.

**Implemented.** `infotheory.py:439-454` loops m = 1..**5**. The class default is n = 5, and the config
(`config.yaml:425-437`) passes no n. Each term is:

- Yp = `makeDelayEmbeddingVector`(y, m−1)[:-1]. For m = 1 this is a zero-width array.
- Xp = `makeDelayEmbeddingVector`(x, m), i.e. x_t..x_{t−m+1}.
- term_m = H([y_t, Yp, Xp]) − H([Yp, Xp]), through `_compute_conditional_entropy` (:180-193).

The value is cached under ('kernel', 0.5, 5) and shared with DI. A[i,j] = H(X_j‖X_i).

**Deviations.**

- Documented: n = 5, not 10 (*candidate*). The documented "past of x" actually includes the current x_t.
  That matches Kramer, not the prose.
- Canonical: the formula matches Kramer. The estimator collapses in 6-10 dimensions. For iid data the true
  value is 10.24 bits. The pyspi per-term values at T = 4096 are [2.039, 1.934, 1.232, 0.277, 0.024],
  summing to 5.51. At T = 64, 256 and 1024 the sums are 2.22, 3.42 and 4.50.

**Direction probe.** On delayed_noise, A[0,1] = 2.405 and A[1,0] = 7.169.

**Checks.**

- numpy reference.
- iid truth (statistical, documents the collapse).
- Direction.
- Gaussian block value, plus an n = 10 variant (qualitative).

## Directed information (infotheory.di.kernel)

**Documented.** Supplement §1.4.8, p.11 (supplement.txt:526-538; pack 537-550): "difference between the conditional entropy of y
given its own past, and causally conditioned entropy ... up to a history length of 10". GitBook :68-86.

**Canonical.** Massey 1990: I(X^n → Y^n) = H(Y^n) − H(Y^n‖X^n), with H(Y^n) = Σ_m H(Y_m|Y^{m−1}). It is 0
when X has no causal or contemporaneous influence on Y.

**Implemented.** `infotheory.py:487-510` computes R − CCE. Here R = Σ_{m=1}^{5} Ĥ(Y^m)/m over m-sample
target blocks.

**Deviation candidates.**

- (a) The definition differs from Massey's. R is not H(Y^5); the two agree only for an i.i.d. target. For a
  stationary Gaussian AR(1) target, the gap R − H(Y^5) = −0.6417·log2(1−φ²) bits: 0.266 bits at φ = 0.5
  and 1.537 bits at φ = 0.9. The gap is positive whenever the target is autocorrelated. It follows from the
  code (`infotheory.py:493-499`) by algebra. A Gaussian plug-in of both target terms on a simulated AR(1)
  target (T = 200000, `p3_di_gauss.py`) gives 0.000, 0.261 and 1.532 bits at φ = 0, 0.5 and 0.9. An earlier
  kernel probe (R − H(Y^5) = 0.79, 0.83 and 1.34 bits at T = 1024) does not isolate the definition: at φ = 0
  the definitional gap is 0, so the 0.79 bits there is the kernel estimator's dimension-dependent bias. This
  gap does not depend on the estimator, so it should also affect di_gaussian and di_kozachenko (other
  families; *hypothesis* there).
- (b) The history length is 5, not the documented 10.
- (c) Kernel magnitude is degenerate. Between independent white-noise series DI is 5.66, 5.85, 5.27 and
  4.58 bits at T = 64, 256, 1024 and 4096, against a truth of 0.

The direction ordering is still right on delayed_noise (7.419 vs 2.647). The numpy reference reproduces the
pyspi formula to 8.9e-16.

**Checks.**

- numpy reference.
- Analytic Gaussian gap between the two target-entropy forms on the target of var2_bivariate (process 0,
  an AR(2) that process 1 does not drive) and on gauss_ar1_pair.
- iid truth 0 (statistical). A canonical-form box-kernel reference separates the two deviations.
- Direction.

## Stochastic interaction (infotheory.si.kernel)

- **Documented.** Supplement §1.4.9, p.12 (supplement.txt:539-549; pack 551-561): "difference between the joint entropy of the
  bivariate process and the individual entropies of each univariate process ... conditioned on the history
  ... one-step". GitBook :184-202.
- **Canonical.** Ay 2001/2015 (Entropy 17:2432); Ay & Wennekers 2003; Barrett & Seth 2011:
  SI = Σ_k H(M^k_t|M^k_{t−1}) − H(M_t|M_{t−1}).
- **Implemented.** `infotheory.py:524-534` computes H(x_t|x_{t−1}) + H(y_t|y_{t−1}) −
  H(x_t,y_t|x_{t−1},y_{t−1}). The terms are kernel conditional entropies on 2-D/1-D and 4-D/2-D arrays
  over T−1 samples.
- **Symmetry.** Max-norm box counts do not depend on column order, so the result is exactly symmetric, and
  `Undirected.multivariate` mirrors it anyway.
- **Deviations.**
  - Documented: none. The sign follows the canonical individual-minus-joint form. The "_k-1" suffix is the
    delay.
  - Canonical: only the estimator null bias: 1.766, 0.934, 0.407 and 0.150 bits on iid at T = 64 to 4096.
- **Checks.**
  - numpy reference.
  - Gaussian SI on gauss_ar1_pair: 0.5·log2(s_x s_y / det Σ_E).
  - Null bias.
  - Symmetry (direction).

## Open questions

1. JIDT version (resolved after review). The jar is labelled 1.6.1 but is the post-1.6.1 master build
   a9b6b3e (see "JIDT build" above). The numerical kernel code is byte-identical between the two revisions,
   so the numerical reproduction is unaffected; the entropy-class trace and the property behaviour now cite
   a9b6b3e.
2. The DI target-entropy formula, Σ_m H(Y^m)/m, sits in the shared `DirectedInfo` code. The Gaussian and
   Kozachenko families should confirm the same gap analytically (hypothesis). Is it intentional, perhaps as
   an averaged entropy-rate estimate? The supplement text does not support it.
3. CCE and DI use history length 5 but are documented as 10. Is the documentation stale, or should the
   config pass n = 10? With the kernel estimator at W-0.5, n = 10 would be even more degenerate.
4. Should xme be treated as an undocumented SPI with an off-by-one (history_length = k gives k−1 lags), or
   is "k" meant as "k−1 lags plus the target"? There is no reference definition to settle this.
5. The kernel SPIs that use embeddings of 6 or more dimensions (cce, di, xme_k10) are degenerate at
   W-0.5 for T ≤ 4096. Should the check stage score them "degenerate" rather than "variant"?
6. The supplement page line numbers in the pack (e.g. §1.4.1 at 454) are about 10 lines later than the
   headings in the current `supplement.txt` (§1.4.1 at line 444). The citations here use section and page,
   with the pack's line ranges.
7. The CE keyword "undirected" in the documentation: does REVIEW M9 already cover it? It is not named
   there explicitly.

## Checks

**Setup.** The manifest `audit/checks/manifest/it_kernel.json` has 222 runs. References are in
`audit/checks/refs/it_kernel.py` (comparison venv: numpy/scipy only, no JIDT); it composes the shared
box kernels of `audit/checks/refs/_kernels_infotheory.py` (`entropy_box_kernel`, `mi_box_kernel`,
`te_box_kernel`, `cce_embedding`, `xme_embedding`, `embed`). Probes are in
`audit/checks/probes/it_kernel_probes.py` (project venv). Results are in
`audit/checks/out/results/it_kernel.json`: **215 pass, 7 fail, 0 without reference** (123 exact,
52 qualitative, 47 statistical runs). Every value is in **bits**. The pyspi side of the whole family
takes 86 s (sum of `time_s`); the longest run is the CalculatorFrame probe of di (12.1 s). The
reference side takes about 3 to 6 minutes, dominated by O(T²) counting for cce/di at P = 4, T = 1024.
The box estimator is O(T²) on both sides, so no run uses T = 4096.

The seven failures, all explained below and none a mismatch between pyspi and the reference
estimator:

- Six statistical runs fail the per-entry band |pyspi − Monte Carlo mean| ≤ 2 SD:
  `est|iid_gaussian|64|…|entropy-1d-truth` (max_z 2.93, 10 entries), `est|iid_gaussian|1024|…|entropy-1d-truth`
  (2.26, 10 entries), `est|nonlinear_dependence|1024|…|entropy-1d-uniform` (2.30),
  `te|var2_bivariate|256|…|truth` (2.48), `tlmi|delayed_noise|1024|…|truth-p01` (2.16) and
  `xme_k10|iid_gaussian|256|…|truth-iid` (2.05). Each of these datasets has an exact run (or an exact
  1-D entropy run) on which the numpy estimator reproduces pyspi to ≤ 1e-12, so the pyspi value is the
  box estimator's value on that draw. Over all 47 statistical runs, 8 of 130 entries lie beyond 2 SD
  (5.92 expected for a 2-SD band; binomial P(X ≥ 8) = 0.241) and none beyond 3 SD (max z 2.93): the
  `ref_calib_*` keys that `refs/it_kernel.py:attach_calibration` stores on
  `est|iid_gaussian|1024|pipeline|entropy-1d-truth` (family-wide, not specific to that run; the same
  summary prints with `python refs/it_kernel.py --calibration`). The binomial probability treats
  entries as independent, which they are not within a run, so it is descriptive. The
  failures are tail draws under multiple comparison, not deviations.
- `xme_k10|var2_bivariate|1024|pipeline|direction` fails: A[0,1] = 0.0238 > A[1,0] = 0.0094, the
  reversed ordering that the trace predicted from count saturation (see xme_k10 below).

Check kinds:

1. **`numpy` (exact).** The pyspi matrix in both orientations against the box-kernel composition on
   the same preprocessing (pipeline: z-score ddof=1; kernel path: raw rows, JIDT widths follow each
   column's N−1 SD; pipeline-detrend: `scipy.signal.detrend` then z-score). rtol 1e-10, atol 1e-12.
2. **`truth*` (statistical).** Monte Carlo mean and SD of the same box estimator under the dataset's
   generating model at the same T (100 to 400 draws; KD-tree counting in width-scaled coordinates,
   `fast_*` in the reference, identical counts except at exact |dx| = w ties). The Gaussian population
   value and, where it has a closed numerical form, the large-sample box limit (`h_box1`, `h_box2` by
   quadrature; the uniform limit equals the edge formula of the `_kernels_infotheory` selftest) are
   stored as extras. Box-kernel estimates of differential entropy/MI/TE are biased at practical T; the
   population comparison is convergence information only and is not scored.
3. **`null` (qualitative).** The mean over pairs on iid data against the Monte Carlo distribution of
   that mean.
4. **`direction` (qualitative).** Ground truth x0 → x1 (delayed_noise lag 1, var2_bivariate,
   logistic_maps_chaotic, logistic_maps_sugihara); for ce the identity A[i,j] − A[j,i] = H(x_j) − H(x_i).
5. **Probes** for what the SPI path cannot show: direct JIDT calls, the tie grid, properties,
   CalculatorFrame/pickle, raw-array calls, non-config parameters (n = 10, history_length = 2, TE
   l/k_tau/l_tau, auto-embedding).

**Convention re-check (edge rule, self-inclusion, widths).** `est|iid_gaussian|64|kernel|tie-grid`
drives JIDT with NORMALISE=false on grids whose box counts change exactly at |dx| = w. All 15 values
equal the closed-box numpy counts to 8.9e-16: H([0,1,2,3,10], w = 1) = 2.2879430946 (counts
[2,3,3,2,1]; an open box gives 3.3219281); w = 1 − 1e-6 gives 3.3219266522 and w = 1 + 1e-6 gives
2.2879445373; H([0,100], w = 0.5) = 1.0 bit (finite only because the point counts itself); on the
k·0.1 grid (k = 0..30) at w = 0.1, where float rounding puts some neighbour differences just above
and some just below 0.1, JIDT's bin search returns 1.7959839412, equal to the numpy compare-to-all
count (open box 1.9116). The `_kernels_infotheory` module note (revised 2026-10-15) says a bin-search miss of
a pair at exactly |dx| == w that rounding puts two bins apart is "a theoretical limit only, not reproduced in
the it_kernel check"; this grid did not trigger it, and `est|var2_bivariate|1024|pipeline|force-all` shows bin
search and FORCE_KERNEL_COMPARE_TO_ALL bit-identical (0.0) and equal to numpy (5.0e-16) on continuous data.
`est|*|kernel|direct-jidt` (30 calls each on iid_gaussian T256 and var2_bivariate T1024: entropy d = 1..10, MI
1+1 and 2+2, TE k = 1 and 2, widths 0.5 and 0.25) agree to ≤ 2.6e-13 with half-width W × N−1 SD; a ddof=0
width differs by 0.0287 / 0.0071 bits and nats by 2.39 / 3.56, so ddof and log base are pinned.

### Shared estimator (`infotheory.est.kernel`): matches, score 9 (anchor 9: half-width, N−1 SD, bits)

- Box kernel exactly as Kantz & Schreiber, closed max-norm box, self-inclusive counts, as above.
- **Properties** (`est|iid_gaussian|256|pipeline|props`, same-lineage): entropy and MI classes read
  NOISE_SEED 42 and NOISE_LEVEL_TO_ADD 0; BIAS_CORRECTION is unknown to all three classes; the TE class
  knows neither noise property; NORMALISE is true for entropy and TE. NOISE_SEED alone and
  BIAS_CORRECTION=true change H and MI by 0.0; noise 1e-8 changes H by 4.1e-10, noise 0.01 by 7.6e-4
  (H) and −4.3e-3 (MI): the seeded noise path is live but unused.
- **Copy path** (`est|bivariate_gaussian|256|pipeline|copy-props`): `copy.deepcopy` reads
  KERNEL_WIDTH 0.25 and NOISE_SEED 'NONE' for all ten kernel SPIs; a pickle round trip keeps the
  configured width. The per-unit `|frame` runs show the consequence (below).
- **Raw arrays** (`est|scaled_offset|256|kernel|raw-array`, `|1024|…`): `bivariate(x, y)` with raw
  arrays z-scores only x; je and ce shift by exactly log2 sd(y) = 3.305191 / 3.307281, mi/tlmi/te are
  unchanged (≤ 2.2e-16). The trace named only je; ce is affected the same way (correction, recorded in
  the candidate resolution).
- **1-D convergence**: N(0,1) at W-0.5 1.979 / 2.035 / 2.044 bits at T = 64 / 256 / 1024 (box limit
  2.04935, truth 2.04710); uniform 1.929 / 1.893 at T = 256 / 1024 (box limit 1.92028, truth 1.79248).
- Candidates: all six confirmed. New finding: the copy path (and the raw-array quirk, a general
  Data.add_process behaviour).

### je_kernel_W-0.5: matches, score 9

12/12 exact runs, max_abs ≤ 4.1e-14 (kernel path on scaled_offset 8.9e-15 and var2_bivariate 1.2e-14,
pipeline-detrend 6.2e-15, tied logistic_maps 9.8e-15). Kernel path on raw scaled_offset = je(z-scored
D2) + log2(sd_i sd_j) to 7.1e-15 (NORMALISE: counts invariant, volume scales). All entries inside
[0, log2 T]. Fresh unmirrored fills are exactly symmetric (max|D − D.T| = 0.0). Convergence: iid pairs
3.741 / 3.989 / 4.066 at T = 64 / 256 / 1024 vs box limit 4.0987 and truth 4.0942; rho = 0.8 pair 3.004
/ 3.306 / 3.396 vs box limit 3.3980, truth 3.3572 (all max_z ≤ 1.86). CalculatorFrame: 3.7487119189
(bivariate_gaussian) and 3.6838644230 (var2_bivariate) equal the numpy width-0.25 values (≤ 4.9e-15),
against Calculator 3.9822833545 / 3.9458115241. Four candidates confirmed; new finding: copy path.

### ce_kernel_W-0.5: matches, score 9 (conditioning order REVIEW M8)

10/10 exact runs, max_abs ≤ 3.0e-14; kernel path on raw scaled_offset = ce(z-scored D2) + log2 sd_j
(1.2e-14). The asymmetry is exactly the marginal-entropy difference: A[i,j] − A[j,i] = H(x_j) − H(x_i)
to fp precision on var2_bivariate, delayed_noise and nonlinear_dependence (`ref_antisym_identity_err`
6.2e-15 / 3.1e-15 / 5.8e-15, `ce|<dataset>|1024|pipeline|direction`) (A[0,1] = 0.8608 < A[1,0] = 0.9217
because x1 = x0² + noise has the lower marginal box entropy, 1.8325 vs 1.8934). Never negative (minimum
0.5 on narrowband). Convergence: rho = 0.8 pair 1.055 / 1.262 / 1.344 vs box limit 1.3486, truth 1.3101.
CalculatorFrame 1.7266038499 / 1.6579346797 = width-0.25 numpy. The 'undirected' keyword conflict is
confirmed and recorded as an elaboration of REVIEW M8/M9; new finding: copy path.

### mi_kernel_W-0.25: matches, score 9

10/10 exact runs, max_abs ≤ 1.6e-15; raw scaled_offset on the kernel path equals MI of the unscaled
D2 rows (6.7e-16); fresh fills symmetric (0.0). Convergence: rho = 0.3 pair 0.777 / 0.297 / 0.122 at
T = 64 / 256 / 1024 vs truth 0.0680 (box limit 0.0679); rho = 0.8 pair 1.297 / 0.898 / 0.775 vs truth
0.7370 (box limit 0.7338). Null: 0.693 / 0.242 / 0.099 vs Monte Carlo 0.762 / 0.280 / 0.090.
nonlinear_dependence: 1.646 and 1.205 vs cross pairs ≤ 0.032 (Gaussian MI 0.00015 / 0.0011).
CalculatorFrame identical (0.25 is JIDT's default). Candidates confirmed (bias, bare except by code
reading at `infotheory.py:279-283`, copy-path scope); no new finding.

### tlmi_kernel_W-0.25: matches, score 9 (asymmetry REVIEW M8)

8/8 exact, max_abs ≤ 2.2e-15. Direction: delayed_noise 2.647 vs 0.064, var2_bivariate 0.305 vs 0.101
(population 0.222 vs 0.0017), logistic_maps_chaotic 0.144 vs 0.032. The delayed_noise pair has
population 0.5·log2(101) = 3.329 bits, but the 0.25-SD box caps the estimate at about 2.65 (pyspi 2.647,
Monte Carlo 2.665 at T = 1024); this run fails the 2-SD band at z = 2.16 (tail draw, see Setup). Null:
0.715 / 0.270 / 0.094. Candidates confirmed (keyword conflict as an M8 elaboration, bare except at
`infotheory.py:313-317`, copy-path scope); no new finding.

### te_kernel_W-0.25_k-1: matches, score 9

10/10 exact, max_abs ≤ 1.3e-15, with JIDT's next-value box reusing the past width; giving the next
value its own SD would change values by ≤ 0.0049 bits. Direction: delayed_noise 2.571 vs 0.501,
var2_bivariate 0.577 vs 0.484 (T = 1024) and 1.122 vs 0.851 (T = 256), logistic_maps_chaotic 0.134 vs
0.018, logistic_maps_sugihara 0.147 vs 0.031. Convergence: var2_bivariate population 0.191 / 0.015 vs
pyspi 0.577 / 0.484; null 1.512 / 1.062 / 0.502 at T = 64 / 256 / 1024. Forwarding: five
(l_history, k_tau, l_tau) variants equal numpy k = l = 1 (3.3e-16) under one identifier; setProperty
raises for l_HISTORY/k_TAU/l_TAU; auto-embedding gives identifier `_k-max-4`, k_HISTORY 1 and the
fixed-k value (0.0). New finding: silent parameter drop and the misleading auto-embed identifier for
estimator='kernel' (no effect on the configured unit).

### xme_kernel_W-0.5_k1: related-quantity, score 5

8/8 exact, max_abs ≤ 2.7e-14 for H(x_j(t) | x_i(t−1)). history_length = 2 equals history_length = 1
on samples t = 2..T−1 (6.2e-15); a genuine two-lag variant would give 1.746 / 1.923 instead of 1.787 /
2.012. Direction (low = predictable): delayed_noise 0.180 vs 2.024, var2_bivariate 1.787 vs 2.012,
logistic_maps_chaotic 1.079 vs 1.801. Convergence: iid 1.738 / 1.976 / 2.023 vs box limit 2.049;
gauss_ar1_pair 1.863 / 1.740 vs Gaussian 1.864 / 1.796. Anchor 5 (non-standard, undocumented quantity
computed as coded), as infotheory.xme.gaussian_k1. New finding: undocumented SPI, k1 = k2, copy path.

### xme_kernel_W-0.5_k10: related-quantity, score 4

7/7 exact with source lags 1..9 (max_abs ≤ 1.2e-12). The intended ten-lag variant differs by 6.2e-15 on
iid T256 (`ref_ten_lag_variant_maxabs_diff`; every count is 1, so 9 and 10 lags give the same value to
fp precision), 0.008-0.015 bits on Gaussian data at T = 1024 and 0.06-0.07 on logistic maps. At W-0.5 ≥
99% of points are alone in their 10-D joint box (0.990 delayed_noise, 0.995 var2_bivariate, 0.998 iid
T1024), so |values| ≤ 0.024 bits on every stochastic dataset against a 2.047-bit truth, and the
var2_bivariate ordering is reversed (the one direction failure). On logistic_maps_chaotic, a
low-dimensional deterministic system, the counts are not saturated (4.3% and 0% of points alone in the
joint box for 0→1 / 1→0; scratch measurement with `refs/it_kernel.py:_frac1`, not a manifest run) and x
→ y is recovered (0.977 < 1.777). Scored on formula agreement at anchor 4 (related but different quantity:
H(x_j(t) | x_i(t-1..t-9)) instead of the 10 lags the identifier names; the hyperparameter is applied
off by one, not ignored, so anchor 3 does not apply), as infotheory.xme.gaussian_k10 and
infotheory.xme.kozachenko_k10. The check-stage review had lowered it to 3 to match a temporary
kozachenko_k10 value that the it_kozachenko score policy later restored to 4; Workflow 4b put it back
to 4. The saturation is recorded as convergence evidence. New finding: off-by-one, saturation, copy path (frame −0.996236 = numpy width
0.25).

### cce_kernel_W-0.5: variant, score 7

7/7 exact, max_abs ≤ 7.9e-13 (Kramer form with the present x in every term); n = 10 probe = numpy
n = 10 (5.7e-13); configured n = 5 (documented 10). The terms of order m ≥ 3 collapse: iid per-term at
T = 1024 2.012, 1.702, 0.675, 0.099, 0.012 bits vs 2.047 each; sums 2.389 / 3.443 / 4.495 at T = 64 /
256 / 1024 vs 10.235; gauss_ar1_pair 4.549 vs Gaussian 8.387. With n = 10 the extra terms add −0.005
(4.379 vs 4.384 on var2_bivariate): the documented history length would not change the kernel value.
Direction: delayed_noise 2.405 < 7.169, var2_bivariate 4.384 < 4.659, logistic_maps_chaotic 1.361 <
4.184. Anchor 7 as infotheory.cce.gaussian (n = 5 vs 10); the collapse is finite-sample behaviour.
New finding: n = 5 and copy path (frame −1.792918, negative because the 0.25 box volume is < 1).

### di_kernel_W-0.5: related-quantity, score 4

8/8 exact (pyspi form), max_abs ≤ 8.0e-13. `di|var2_bivariate|1024|pipeline|massey-identity`: pyspi =
box-kernel Massey DI_5 + B(target) to 6.7e-14 (`ref_identity_maxabs`) (Massey 4.317 / 4.015 vs pyspi
5.199 / 4.860; B = 0.882 / 0.846), the block-rate definition confirmed on this estimator. On the
independent pair (white source, AR(2) target) the box B of the AR(2) target is 0.846 but the box B of a
white series is 0.860, against a Gaussian population B of 0.255 for the AR(2) target: with this
estimator the definitional excess is swamped by the dimension-dependent bias of the 1..5-D box
entropies; the estimator-free confirmation is
`infotheory.di.gaussian|var2_bivariate|4096|pipeline|indep-truth` (it_gaussian, already checked).
Independent white pairs: 5.55 / 5.83 / 5.31 bits at T = 64 / 256 / 1024 against a population 0 (box
Massey form 3.50 / 4.27 / 4.45: most of it is cce collapse). n = 10: 11.52 / 11.18. The ordering
survives: delayed_noise 7.419 > 2.647, var2_bivariate 5.199 > 4.860, logistic_maps_chaotic 1.675 >
1.612. Anchor 4 ('DI with block-entropy rate'), as infotheory.di.gaussian. New finding.

### si_kernel_W-0.5_k-1: matches, score 9

7/7 exact, max_abs ≤ 7.4e-14; fresh fills symmetric (0.0). gauss_ar1_pair population 0.144 vs pyspi
0.728 / 0.322 at T = 256 / 1024; null 1.743 / 0.930 / 0.400. Ay's SI with the box kernel, individual
minus joint. New finding: copy path (frame 3.287689 = numpy width 0.25 vs Calculator 0.996235).

### Copy path summary (new finding, all W-0.5 kernel SPIs)

`CalculatorFrame.init_from_list` deep-copies the base Calculator (`pyspi/calculator.py:556-559`, `copy.deepcopy` at :559) and
`JIDTBase.__deepcopy__` (`pyspi/statistics/infotheory.py:77-82`) deep-copies the JPype proxies, which
come back in JIDT's default state. Every `|frame` run's CalculatorFrame slice equals the independent
numpy value at width 0.25 (≤ 1.4e-13), while the Calculator and pickle slices equal the configured
width: je 3.98228 → 3.74871, ce 1.93662 → 1.72660, cce 3.40506 → −1.79292, xme_k1 1.98147 → 1.77936,
xme_k10 0.00376 → −0.99624, di 5.82351 → 8.68772, si 0.99623 → 3.28769 (bivariate_gaussian T = 256,
A[0,1]). mi, tlmi and te are unaffected only because 0.25 is JIDT's default. On that path the W-0.5
hyperparameter is ignored (rubric anchor 3 would apply there); the scores above are for the Calculator
path.

### Trace fields corrected

- `infotheory.je.kernel` and `infotheory.ce.kernel` copy-path candidates: the parenthetical now gives
  the width-0.25 value of each dataset (je 3.7487119189 bivariate_gaussian, 3.6838644230
  var2_bivariate; ce 1.7266038499, 1.6579346797) and cites the check runs instead of the scratch probe
  (remaining trace issue).
- `infotheory.est.kernel` raw-array candidate: ce shifts by log2 sd(y) as well as je (resolution text).
- All eleven records: the `implementation_trace` citations of `pyspi/calculator.py` now point at HEAD
  (`Data(...)` at :277, `multivariate` at :329, `np.fill_diagonal` at :332, `__init__` defaults at
  :48-51); the trace-stage lines :264 / :294 / :297 / :47-48 predate commit a1f20c7.
- Check-stage review: identities that hold to fp precision are quoted with the stored residuals
  (ce `ref_antisym_identity_err` ≤ 6.2e-15, di `ref_identity_maxabs` 6.7e-14, xme_k10
  `ref_ten_lag_variant_maxabs_diff` 6.2e-15 on iid T256) instead of 0.0; the family calibration is
  stored in results (`ref_calib_*`); xme_k10 moves from score 4 to 3 (anchor 3, label/lag-set mismatch). Workflow 4b put it back to 4
  (anchor 4, related but different quantity) for consistency with the gaussian and kozachenko k10 siblings.

## Open questions: answered

1. **JIDT version.** Not settled byte-for-byte, and not needed for values: every kernel path agrees
   with the independent numpy box kernel to ≤ 2.6e-13 (`est|*|direct-jidt`, all `|numpy` runs), and
   the property read-back matches the a9b6b3e master build (NOISE_SEED stored on the entropy and MI
   classes). A rebuild of a9b6b3e compared class-by-class with the bundled jar would settle provenance.
2. **DI target term.** Confirmed as a definition, estimator-independent: the it_gaussian family
   established the analytic gap (`infotheory.di.gaussian|var2_bivariate|1024|pipeline|massey-identity`
   and `|4096|pipeline|indep-truth`), and here `di|var2_bivariate|1024|pipeline|massey-identity`
   shows pyspi = Massey DI_5 + B(target) exactly for the kernel estimator. Nothing in the code or the
   supplement says it is intended as an averaged entropy-rate estimate; only the authors can say.
3. **History length 5 vs 10.** The code and config use 5 (`cce|var2_bivariate|64|pipeline|configured-n`).
   For the kernel estimator n = 10 changes cce by −0.005 bits (terms 6-10 are ~0) and roughly doubles
   di (11.52 vs 5.20) through the saturated block-entropy terms, so n = 10 would be no more informative
   at W-0.5. Whether the documentation is stale is for the authors.
4. **xme history_length.** The code gives k − 1 source lags for k ≥ 2 and one lag for k = 1, 2
   (`xme_k1|…|k2-equals-k1-shifted`, `xme_k10|*|numpy`). There is no reference definition; the records
   treat it as an off-by-one relative to the identifier (the identifier says 10, the lag set is 1..9).
5. **Score cce/di/xme_k10 as degenerate?** No. The README scores formula agreement and reports
   finite-sample bias separately. cce keeps informative low-order terms and the right ordering on all
   three direction datasets (variant 7); di keeps the ordering on a ~5-bit bias floor (related-quantity
   4); xme_k10 is effectively constant (~0) on stochastic data at T ≤ 1024 and reverses var2_bivariate,
   but it is informative on logistic_maps_chaotic and it is not constant by construction, so it keeps
   the formula score (related-quantity 4, anchor 4 for the k10 off-by-one lag set) with the saturation recorded in `what_changed` and in this
   note for the synthesis.
6. **Supplement line offsets.** Remapped in the citation-freshness pass (Workflow 4a): every record and every Documented paragraph of this note now cite
   the `supplement.txt` section range first and keep the pack range after it (e.g. §1.4 `supplement.txt:405-443;
   pack lines 414-453`, §1.4.9 `539-549; 551-561`). The current `supplement.txt` headings sit 9-12 lines earlier
   than the pack's.
7. **ce 'undirected' keyword and REVIEW M9.** M9 is a general label-structure discussion that does not
   name ce. The check shows the asymmetry is exactly H(x_j) − H(x_i) (`ce|*|direction`), and M8 already
   documents ce = H(i,j) − H(i). The keyword conflict is recorded as an elaboration of M8/M9, not as a
   new finding.

## Optimisation

Workflow 3 (group infotheory) is a short, profile-based search for where the time goes and which avenues look promising; nothing in pyspi was changed. Walls below are from `audit/profiles/summary.json` (quiet machine); percentages are shares of the profiled SPI call (`tests/regression/harness.py:compute_spi`) in the cProfile run at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`). That profiling pass ran contended for the JIDT units (profiled walls 1.3-20x the quiet walls), so its walls are not used, only its shares. cProfile cannot see inside Java: JPype dispatches JIDT methods without profiler entries, so the Java time is booked as self time of the calling pyspi frame, and every `opaque_share` in these records (0.0002-0.04) is a lower bound on JIT/JNI time, not an estimate of it. `expected_speedup` is the unit's per-window speedup at 118x1024 as an order of magnitude; risks carry the `unverified-` prefix unless a micro-benchmark diff file backs them. Agreement with today's output means only that a candidate reproduces it.

### Walls

| Unit | 8x1024 fresh | 32x1024 fresh | 118x1024 fresh (measure_mode) | 8x1024 family-warm | 118x1024 family-warm projected (measure_mode) |
|---|---|---|---|---|---|
| `je.kernel` | 0.114 s | 1.94 s | 37.3 s (measured) | 0.111 s | 27.4 s (extrapolated-pairs-from-8) |
| `ce.kernel` | 0.153 s | 2.1 s | 38.2 s (measured) | 0.0399 s | 9.85 s (extrapolated-pairs-from-8) |
| `cce.kernel` | 9.12 s | - | 2.25e+03 s projected (extrapolated-pairs-from-8) | 8.83 s | 2.18e+03 s (extrapolated-pairs-from-8) |
| `xme.kernel_k1` | 0.565 s | - | 139 s projected (extrapolated-pairs-from-8) | 0.559 s | 138 s (extrapolated-pairs-from-8) |
| `xme.kernel_k10` | 1.02 s | - | 251 s projected (extrapolated-pairs-from-8) | 1.01 s | 250 s (extrapolated-pairs-from-8) |
| `di.kernel` | 11.4 s | - | 2.81e+03 s projected (extrapolated-pairs-from-8) | 3.92 s | 967 s (extrapolated-pairs-from-8) |
| `si.kernel` | 2.18 s | 32.5 s | 447 s projected (extrapolated-pairs-from-32) | 1.9 s | 468 s (extrapolated-pairs-from-8) |
| `mi.kernel` | 0.369 s | 6.67 s | 142 s (measured) | 0.362 s | 89.3 s (extrapolated-pairs-from-8) |
| `tlmi.kernel` | 0.366 s | 6.61 s | 139 s (measured) | 0.36 s | 88.7 s (extrapolated-pairs-from-8) |
| `te.kernel_k-1_kt-1_l-1_lt-1` | 0.262 s | 4.36 s | 80.1 s (measured) | 0.255 s | 63 s (extrapolated-pairs-from-8) |

`cce`, `di` and `xme` (k1, k10) are in `infeasible_above_8x1024.txt` (pair-count projections from 8x1024). `infotheory.est.kernel` is infrastructure (filled from the je.kernel profile).

### Where the time goes

All time is inside JIDT's box-kernel estimator, booked as self time of the pyspi frame: je `_compute_joint_entropy` 99.5%, ce `_compute_joint_entropy` 90.5% + `_compute_entropy` 9.4%, mi / tlmi / te `bivariate` 99.9%, si `_compute_conditional_entropy` 99.8% (168 calls), xme `bivariate` 99.9% (k1) / 100.0% (k10), cce `_compute_conditional_entropy` 94.2% (+ `_compute_causal_entropy` self 5.1%), di `_compute_conditional_entropy` 58.4% (via cce 64.8%) + `_compute_entropy_rates` 35.1%. JIDT's `KernelEstimatorMultiVariate` counts closed-box neighbours using bins of width w x SD; the measured T exponents (1.5-1.9 for d <= 3, ~1.0 for the 10-D cce terms) show it is close to quadratic in T. The estimators are single-threaded: no `Thread`, `NUM_THREADS` or `availableProcessors` reference exists in any `infodynamics/measures/continuous/kernel` class of the bundled jar (string scan), unlike the KSG classes.

### Opportunities

- **it_kernel-1 (cache-sharing; si, xme_k1, xme_k10, di): per-channel entropy terms.** si's H(x_t|x_{t-1}) and H(y_t|y_{t-1}) (4 of 6 calls per ordered pair), xme's H(source past) (1 of 2 calls) and di's target entropy-rate sum (5 calls per pair) depend on one channel; caching them at Data level runs each P times instead of P(P-1). cProfile cannot split the conditional-entropy frame, so the si and xme shares are weighted by dimension: ~2x (si), ~1.5x (xme_k1), ~1.9x (xme_k10). di's share is measured: 35.1%, cap 1.54x fresh, and ~(P-1)x of the family-warm increment (967 s projected) when cce runs first. Unverified-bit-identical (the same JIDT call on the same array; the box kernel adds no noise). The figures hold on today's JIDT cost: the records rank this avenue after it_kernel-2 (`depends_on`) with `residual_speedup` 1-2x (si), 1-1.5x (xme_k1, di) and 1-1.9x (xme_k10), lower bound 1x, because the bitsets' gain differs with the number of ANDed rows and the per-channel terms' share of that residual is not known, so no saving is credited after it_kernel-2.
- **it_kernel-2 (algorithmic; all kernel units and est.kernel): neighbourhood bitsets.** Max-norm box counts are products of per-dimension indicators, so per-(channel, width, lag window) T x T bitsets of |x(t) - x(s)| <= w_d make every joint count a popcount of ANDed rows (T x T/64 = 16k word operations per 2-D count at T=1024, ~10-50 us, against 4-200 ms per pair in JIDT), and the neighbourhood search is shared by all P-1 partners and all composed terms. Expected 10-50x for d <= 3 (je, ce, mi, tlmi, te, si, xme_k1, est.kernel) and 5-20x where up to ten lag windows are ANDed per pair (xme_k10, cce, di). The bitsets replace every JIDT call of the unit and the Java delay embeddings, so the Amdahl caps use all of those frames: ce `_compute_joint_entropy` 90.5% + `_compute_entropy` 9.4% = 99.9% (cap ~1000x); cce `_compute_conditional_entropy` 94.2% + `_compute_causal_entropy` self 5.1% (Java embeddings + np.concatenate) = 99.3% (cap ~140x); di conditional entropies 58.4% + `_compute_entropy_rates` 35.1% + causal self 5.8% = 99.3% (cap ~140x). With the conditional-entropy frame alone the caps would be 10.5x (ce), 17x (cce) and 2.4x (di), below the stated ranges. Risk unverified-within-tolerance: counts are exact integers if the closed-box test and the per-column widths (w x ddof=1 SD of each embedded column) match JIDT; the log-sum order changes. The check stage's all-pairs numpy counts reproduce JIDT to <= 2.6e-13. Effort L. Not measured.
- **it_gaussian-4 (vectorise-batch; je, ce, si, xme_k1, est.kernel; shared id owned by it_gaussian, with the same opportunity text on every unit and the part that applies here named in each speedup_basis; the cce/di delay-embedding part does not apply to these units): `JArray.of` for the 2-D observations.** Pass the T x d arrays of `_compute_joint_entropy`, `_compute_conditional_entropy` and `CrossmapEntropy.bivariate` with `jpype.JArray.of` (one buffer copy, 0.17-0.22 ms) instead of `JArray(JDouble, 2)` (row by row, ~1.0 ms at T=1024 for any column count; `opp_it_gaussian-1.json` `components.conversion_2d_by_shape`). The conversion is 18-20% of these units' quiet per-call cost at 118x1024 and the change removes 15-17%: ~1.2x. Basis: the self-time frames above (which hold conversion and Java time together) + that component timing; unverified-bit-identical (the same float64 buffer reaches JIDT; the Gaussian component check gives a bitwise-equal entropy). Effort S. Not measured at unit level. The je, ce, si and xme_k1 records list it_kernel-2 in `subsumed_by`: the bitset counts replace every JIDT box-kernel call, so no 2-D conversion is left and build_csv credits this avenue no saving on top of it.

### Refuted seed items

- **NUM_THREADS and pair-level JVM parallelism:** the kernel estimators are single-threaded (string scan above), so there is no thread setting to fix; process-level parallelism is the cross-cutting xc-window-parallel.
- **Symmetric quantities once per unordered pair (je, mi):** je already runs once per unordered pair through `data.joint_entropy`. mi.kernel computes both orders and is exactly symmetric (fresh unmirrored fills max|D - D.T| = 0.0, check stage); si.kernel likewise. Halving these belongs to xc-undirected-upper (2x) and is not listed per unit; that cross-cutting avenue is recorded `independent_of` it_kernel-1, it_kernel-2 and it_gaussian-4 because they act on different factors (`audit/profiles/crosscutting.yaml`): it halves the number of bivariate calls, it_kernel-2 and it_gaussian-4 cut the cost of each call, and it_kernel-1 moves the per-channel terms out of the pair loop; so it keeps its own 2x on their residual.
- **JNI share (narrowed; now listed as it_gaussian-4 for je, ce, si, xme_k1 and est.kernel):** the latest run of `opp_it_gaussian-1.py` times the 2-D conversion by shape (`components.conversion_2d_by_shape`): `JArray(JDouble, 2)` ~1.0 ms for every T x d array from 1023 x 1 to 1014 x 10, `JArray.of` 0.17-0.22 ms. Against the quiet per-call costs at 118x1024 that is 19% of je (one 1024 x 2 conversion per 5.4 ms unordered-pair call, 37.3 s measured), 18% of ce fresh, 19% of si (six conversions per 32.4 ms ordered pair) and 20% of xme_k1 (two per 10.1 ms): above the ~10% bar, so the analyst's 'under the bar' was wrong, and the cross-cutting xc-jni-overhead (Java-side pair loop for the Gaussian SPIs) explicitly leaves the kernel/Kozachenko units out of scope. The avenue is therefore the shared it_gaussian-4 (`JArray.of` instead of the row-by-row conversion), removing ~15-17% of these units (~1.2x), unverified-bit-identical (the same float64 buffer reaches JIDT). It stays under the bar for xme_k10 (two conversions per 18.2 ms pair: 11%, the change removes 9%), mi/tlmi/te (1-D inputs) and cce/di (100-300 ms per pair).
- **di entropy-rate cache** is confirmed (35.1%, it_kernel-1).

### Shared versus incremental cost

`data.joint_entropy` makes ce a lookup after je (family-warm 0.040 s against 0.153 s fresh at 8x1024) and `data.causal_entropy` makes di reuse cce's causal entropies (3.92 s against 11.4 s; what remains is the per-pair entropy-rate sum). mi, tlmi, te, si and xme are incremental. No kernel SPI is in the retained EEG selection.
