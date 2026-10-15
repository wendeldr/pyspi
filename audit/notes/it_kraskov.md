# it_kraskov family: define, trace and check

Units (11): `infotheory.mi.kraskov`, `infotheory.mi.kraskov_DCE`, `infotheory.tlmi.kraskov`, `infotheory.tlmi.kraskov_DCE`, `infotheory.te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4`, `infotheory.te.kraskov_DCE_auto-MAX_CORR_AIS_kmax-10_taumax-4`, `infotheory.te.kraskov_DCE_k-2_kt-1_l-1_lt-1`, `infotheory.te.kraskov_DCE_k-1_kt-1_l-1_lt-1`, `infotheory.te.kraskov_k-1_kt-1_l-1_lt-1`, and the two infrastructure units `infotheory.est.ksg1` (the shared JIDT KSG estimator) and `infotheory.theiler_window` (the DCE window heuristic).

All nine SPIs live in `pyspi/statistics/infotheory.py` (`JIDTBase` 9-220, `MutualInfo` 253-283, `TimeLaggedMutualInfo` 286-317, `TransferEntropy` 320-391) and delegate to JIDT through jpype 1.7.1 (OpenJDK 21). The probe scripts are in the session scratchpad under `it_kraskov/probes/` (`p1_props.py` to `p14_other.py`, plus the from-scratch reference `ksg_ref.py`). The regression goldens are snapshots of current output; nothing here treats them as a reference.
The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records. Candidates and hypotheses raised in the trace sections are resolved in the records' `candidate_resolutions` and in `## Checks`.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

## Which JIDT is this?

`pyspi/lib/jidt/version-1.6.1.txt` labels the jar 1.6.1 (22/08/2023), and the GitBook says v1.6.1. The supplement says v1.5. The jar is neither. Its `MutualInfoMultiVariateCommon.class` has the strings `NOISE_SEED`, `noiseSeedSet` and `setSeed`, and the jar contains `EntropyCalculatorMultiVariateCommon.class`. In the v1.6.1 tag source the jitter generator is `new Random()` with no seed (`MutualInfoMultiVariateCommon.java:519-529`), and `setProperty` (`:201-229`) has no `NOISE_SEED`. The jar's content is the main evidence. The property arrived on JIDT master in commit 2a18cd0, whose message reads "Adding property to allow used to set random seed for noise addition to data in MI, CMI and wrapped estimators". The common class came in 3d1f3fe ("Gathering common functionality of EntropyMultiVariate estimators into a Common class"), and a9b6b3e added `NOISE_SEED` to the entropy estimators. All 337 jar entries are dated 2024-04-21 (`unzip -l`). The commit HTML pages give hashes and messages but no dates. The dates come from the GitHub REST API (`https://api.github.com/repos/jlizier/jidt/commits/<sha>`, where author date = committer date): 2a18cd0 is 2024-04-16T04:08:04Z, 3d1f3fe is 2024-04-20T07:57:38Z and a9b6b3e is 2024-04-20T08:52:10Z. The master listing from 2024-04-15 to 2024-05-31 contains only these three commits, so a9b6b3e is the last master commit before the jar's entry date. JIDT line numbers in this note are therefore at commit a9b6b3e unless marked v1.6.1 (`https://github.com/jlizier/jidt/blob/a9b6b3e77ca8c051f1031003474e2e35dd4a3269/java/source/infodynamics/...`). The `it_gaussian` and `it_kozachenko` notes reach the same conclusion.

The match was checked numerically as well as by date. A from-scratch Python reimplementation, which regenerates the jitter with a pure-Python `java.util.Random`, reproduces pyspi to ≤ 3e-15 for MI and to 12 printed decimals for conditional MI and auto-embedding (probes p5 to p7).

This matters for reproducibility. pyspi sets `NOISE_SEED=42` (`infotheory.py:91`). The shipped jar honours it, but the real v1.6.1 would ignore it, leaving the jitter unseeded.

## Shared machinery (`infotheory.est.ksg1`)

**Documented.** Supplement §1.4, p.9 (pack lines 414-453): "The Kraskov–Stögbauer-Grassberger (KSG) technique (ksg) combines nearest-neighbor estimators for mutual information based measures (default of four nearest neighbors, indicated by modifier NN-4)".

**Canonical.** Kraskov, Stögbauer & Grassberger 2004, Phys. Rev. E 69, 066138, algorithm 1: Î = ψ(k) − ⟨ψ(n_x+1) + ψ(n_y+1)⟩ + ψ(N), using the max-norm distance ε(i) to the k-th neighbour in the joint space and strict marginal counts. For conditional MI, Frenzel & Pompe 2007, PRL 99, 204101: ψ(k) − ⟨ψ(n_xz+1) + ψ(n_yz+1) − ψ(n_z+1)⟩. Equation numbers are from memory.

**Implemented.** I probed the properties on all nine SPIs (p1):

| Property | Value |
|---|---|
| `k` | 4 |
| `NORM_TYPE` | `MAX_NORM` |
| `NORMALISE` | `true`: per-column mean/std with divisor N−1 (`MatrixUtils.java:2490-2502`, `:2663-2672`) |
| `NOISE_LEVEL_TO_ADD` | 1e-8: Gaussian, added after normalisation, row by row (source, then dest, then cond), from `java.util.Random(42)` (`MutualInfoMultiVariateCommon.java:626-642`; `ConditionalMutualInfoMultiVariateCommon.java:766-785`) |
| `NUM_THREADS` | 64 (= `availableProcessors()`) |
| `ALG_NUM` | 1 (TE) |
| `DYN_CORR_EXCL` | 0, unless DCE sets it |
| `BIAS_CORRECTION` | not a property of these classes (`getProperty` returns None), so pyspi's `BIAS_CORRECTION=false` is a no-op |

The core loops are `MutualInfoCalculatorMultiVariateKraskov1.java:60-123` and `ConditionalMutualInfoCalculatorMultiVariateKraskov1.java:75-270`. The final averages are computed in `MutualInfoCalculatorMultiVariateKraskov.java:518-522` and `ConditionalMutualInfoCalculatorMultiVariateKraskov.java:539-546`.

The exclusion rule is in `KdTree.java:751-753` and `UnivariateNearestNeighbourSearcher.java:497-530`: a point is excluded when |dt| ≤ W, so W=0 excludes only the point itself. JIDT throws when N ≤ k + 2W (`MutualInfoCalculatorMultiVariateKraskov.java:298-303`; CMI `:316`). pyspi catches that per pair and returns NaN.

**Deviation analysis.**
- The estimator is textbook KSG1. The additions are standardisation, the 1e-8 jitter, and ψ(N) left unadjusted for the Theiler window (JIDT marks this as a TODO at `Kraskov1.java:102-104`).
- Candidate: **thread-count dependence.** Per-thread partial sums are added in thread order. Results differ by up to 1.6e-14 between `NUM_THREADS=1` and the 64-core default (p2). Repeat runs on the same machine are bit-identical.
- Candidate: **the jitter decides values on tied data.** In logistic_maps (T=256) the r=3.5 driver is period-4 with 4 distinct values. Pair 1->0 MI is 0.226622 / 0.226587 / 0.230899 for seeds 42/43/44, and the TE auto-embedding picks a different source embedding for each seed (p12). Continuous data are unaffected.
- Candidate: **values are negative on independent data** although the SPIs are labelled `unsigned`. This is an estimator property; the minima are listed per unit below.

## infotheory.mi.kraskov

**Documented.** Supplement §1.4.3, p.10 (pack lines 469-476): "Mutual information [33] is an undirected measure of the (potentially nonlinear) dependence between paired observations of x and y". The GitBook lists it at `information-theory.md:160-180`.

**Canonical.** I(X;Y) = ∫∫ p log p/(p_x p_y) (Cover & Thomas 2006 §8.5), estimated by KSG1 under an i.i.d. sampling assumption.

**Implemented.**
1. `config.yaml:507-508` sets the parameters.
2. `Data.set_data` z-scores each channel (`data.py:184-195`).
3. `Undirected.multivariate` (`base.py:132-138`) first calls `Directed.multivariate` (`base.py:84-94`), which runs `bivariate` for every ordered pair i ≠ j, lower triangle included. It then overwrites the lower triangle with the transpose (`A[li] = A.T[li]`). So A[i,j] = A[j,i] = bivariate(min, max), and each unordered pair is estimated twice with one result thrown away.
4. `MutualInfo.bivariate` (`infotheory.py:267-283`) passes the full-length x_i and x_j to `MutualInfoCalculatorMultiVariateKraskov1` and returns `computeAverageLocalOfObservations()`. A bare `except` returns NaN, with a warning about "Cholesky factorisation" that does not apply to KSG.

**Deviations.**
- None in the estimator. A brute-force KSG1 reproduces pyspi to ≤ 3e-15 on bivariate_gaussian, delayed_noise, var2_bivariate, gauss_ar1_pair and narrowband_f0-0.1 at T=256 (p5).
- KSG1 is symmetric (p2: bivariate(0,1) = bivariate(1,0) exactly).
- Candidate (efficiency, correctness-neutral): because of the double computation above, half the KSG runs for MI and MI_DCE are discarded. Computing i<j only would give the same matrix at half the cost.
- Undocumented: nats, the jitter and seed, negative values (minimum −0.167 at T=64), and the JIDT version.
- Treating autocorrelated samples as i.i.d. is outside the estimator's assumptions. On gauss_ar1_pair at T=256 the estimate is 0.1195 against an analytic 0.1578.

**Proposed checks.**
- Kernel tolerance ≤ 1e-12 against the from-scratch KSG1 (independent: numpy + scipy.special.digamma + the Python LCG, no JIDT).
- Statistical convergence to the analytic Gaussian MI on bivariate_gaussian (0.0472, 0.5108 nats) and gauss_ar1_pair (0.1578 nats).
- iid_gaussian null.
- scaled_offset vs bivariate_gaussian invariance (≤ 1e-10).
- Exact symmetry.
- External-package comparison with `sklearn.feature_selection.mutual_info_regression(n_neighbors=4)`, tolerance 0.01 nats where both are positive.

## infotheory.mi.kraskov_DCE and infotheory.theiler_window

**Documented.** Supplement §1.4, p.9-10: "...dynamic correlation exclusion (also known as a Theiler window) [35, 36] ... here, we set the window as the product of the autocorrelation functions of both time series (a heuristic for the autocorrelation time based on Bartlett's formula [10, 37])". The same text is on GitBook `information-theory.md:20`.

**Canonical.**
- The window itself is from Theiler 1986, Phys. Rev. A 34, 2427.
- The autocorrelation time of a pair, from Bartlett 1946 (supplement [37]) and Cliff et al. 2021 (supplement [10]), is τ = 1 + 2 Σ_{k≥1} ρ_x(k) ρ_y(k), the inflation factor for the variance of a sample cross-correlation.
- A Theiler window of roughly τ samples excludes neighbours with |t − t'| ≤ W.

The exact Bartlett/Cliff forms are from memory; see the open questions.

**Implemented.** `_set_theiler_window` (`infotheory.py:195-216`) runs on the first DCE call for a Data object. It computes, for every pair, W = 2 · np.dot(acf(x_j), acf(x_i)), using `utils.acf` (`pyspi/utils.py:47-60`): the biased sample ACF of the z-scored series over all lags 0..T−1. The matrix is cached as `data.theiler` and shared by every DCE SPI on that Data. Each pair then gets `DYN_CORR_EXCL = int(W[i,j])`. The window is pair-specific but not recomputed per pair, which refines the pack hypothesis.

**Deviations (candidates, probe p3 on 11 datasets × 4 T × 2 preprocessing variants).**
- **Off by one (pack hypothesis confirmed).** The raw value is exactly Bartlett's 1 + 2Σ_{k≥1} plus 1 (difference 1.00 on every pair), then truncated by `int()`. White noise gives W ∈ {1, 2} (iid_gaussian T=1024: [2,2,1,1,2,2,2,2,1,1]). gauss_ar1_pair at T=1024 gives 5 where Bartlett gives 4.14.
- **Not an autocorrelation time for oscillatory or trending input.** Because all T−1 lags are summed, W grows linearly with T:

  | Dataset | T=64 | T=256 | T=1024 | T=4096 |
  |---|---|---|---|---|
  | narrowband_f0-0.1 (≈ T/3, since 2 Σ (1−k/T)² cos²ωk ≈ T/3) | 22 | 85 | 339 | 1353 |
  | logistic_maps | 14 | 39 | 208 | 665 |

  At T=4096 other datasets reach W = 1403 (trending without detrend; 2 with detrend), 1277 (cointegrated) and 1084 (shared_envelope). With W ≈ T/3, about two thirds of the series is excluded from every neighbour search.
- **Documented "product of the ACFs" vs implemented "2 × inner product over all lags".**
- **Effect on values.** MI rises sharply where W is large. For narrowband_f0-0.1 at T=256, pair (0,1) goes from 1.448 to 2.239 (p5). For logistic_maps at T=64 the minimum goes from 0.186 to 0.720 (p13). No MI NaN occurs at the audit sizes, because 4 + 2W < T whenever W ≤ T/3.
- **Efficiency.** Each channel's ACF is recomputed inside the pair loop with an O(T²) `np.correlate`.

**Proposed checks.**
- Exact integer agreement of `data.theiler` with a from-scratch lag-loop ACF.
- Tolerance comparison of the raw value with the analytic τ + 1 for white noise and the AR/VAR datasets.
- W/T → 1/3 on narrowband.
- Detrend vs no detrend on trending and cointegrated.
- For MI_DCE: the kernel check with the window, and the DCE vs non-DCE error against the analytic MI on gauss_ar1_pair (probe at T=256: 0.1195 without, 0.1688 with, 0.1578 analytic).

## infotheory.tlmi.kraskov and infotheory.tlmi.kraskov_DCE

**Documented.** Supplement §1.4.4, p.10 (pack lines 477-485): "Keywords: undirected ... Time-lagged mutual information [35] is an undirected measure of the dependence between time series x and a time-lagged instance of the series y. We include statistics for only lag-one mutual information". The GitBook (`information-theory.md:235-251`) calls it "directed".

**Canonical.** Lag-1 lagged mutual information I(X_t; Y_{t+1}) (Schreiber 2000, supplement [35], treats time-delayed MI as the predecessor of TE), estimated by KSG1.

**Implemented.** `TimeLaggedMutualInfo(JIDTBase, Directed)` is at `infotheory.py:286-317`. `bivariate(i, j)` sets `src = x_i[:-1]` and `targ = x_j[1:]`, then runs KSG1 MI on N = T−1 pairs. All ordered pairs are computed and none is mirrored. The DCE variant uses the same window as above, computed on the unshifted series.

**Deviations.**
- The lag convention (pack hypothesis) is confirmed: A[i,j] = I(x_i(t); x_j(t+1)). The from-scratch KSG1 on those slices matches pyspi to 12 decimals on 10 ordered pairs (p5).
- The output is asymmetric: delayed_noise at T=256 gives A[0,1] = 2.168 and A[1,0] = 0.048. That asymmetry is already documented (REVIEW M8, recorded for `tlmi_gaussian`).
- The new documentation point is that the supplement's "undirected" keyword and text contradict both the code's `directed` label and the GitBook.
- The DCE effects are as for MI. For narrowband_f0-0.25 at T=64 the minimum over pairs goes from 0.344 to 1.208.

**Proposed checks.**
- Kernel tolerance ≤ 1e-12.
- **Direction (empirical A[i,j] convention).** Expected: A[0,1] ≫ A[1,0] on delayed_noise; A[0,1] > A[1,0] on var2_bivariate; A[1,0] > A[0,1] on gauss_ar1_pair (analytic 0.1738 vs 0.1266).
- Statistical convergence to the analytic Gaussian lagged MI: gauss_ar1_pair 0->1 = 0.1266 and 1->0 = 0.1738; delayed_noise 0->1 = ½ log 101 = 2.3076 (pyspi gives 2.168 at T=256).
- iid null.
- Asymmetry qualitative check.

## Transfer entropy, fixed embeddings

Three units: `te.kraskov_k-1_kt-1_l-1_lt-1`, `te.kraskov_DCE_k-1_kt-1_l-1_lt-1` and `te.kraskov_DCE_k-2_kt-1_l-1_lt-1`.

**Documented.** Supplement §1.4.5, p.10-11 (pack lines 486-510): "Transfer entropy [35] is a measure of information transfer from a source time series x to a target time series y, based on the Takens time-delay embedding ... The fixed parameters are minimal values, i.e., typically with a fixed target embedding length of 1 (denoted by k-1) or 2 (denoted by k-2)."

**Canonical.** Schreiber 2000, Eq. (4): T_{X→Y} = I(Y_{t+1}; X_t^{(l)} | Y_t^{(k)}). Embedding delays and the source-target delay u follow Wibral et al. 2013. Estimation uses Frenzel-Pompe KSG CMI.

**Implemented.**
1. `TransferEntropy.__init__` (`infotheory.py:326-367`) sets `k_HISTORY`, `k_TAU`, `l_HISTORY` and `l_TAU`. `DELAY` keeps the JIDT default of 1.
2. `bivariate(i, j)` (`:376-391`) calls `setObservations(x_i, x_j)` on `TransferEntropyCalculatorKraskov`, which uses the KSG1 CMI.
3. JIDT builds the samples in `TransferEntropyCalculatorViaCondMutualInfo.java:585-619`:
   - destPast = (y_t, …, y_{t−(k−1)kt})
   - destNext = y_{t+1}
   - sourcePast ends at t+1−u = t
   - start = max((k−1)kt, (l−1)lt + u − 1), so N = T − 1 − start
4. The CMI is I(sourcePast; destNext | destPast), in nats.

**Deviations.**
- None in the estimator. The from-scratch CMI on that layout matches pyspi to 12 decimals on five datasets at T=256 (p5).
- These are finite-history TEs. var2_bivariate is order 2, so its k=1 TE is not the full-history TE.
- `config.yaml:565` comments the k-2 entry as "auto-embedding on target-only", but no auto-embedding is configured.
- With DCE, the window changes the value on oscillatory input. For k=1, narrowband_f0-0.1 at T=64 has a minimum over pairs of 0.218 without and 0.295 with DCE. The much lower k=2 value (0.013) reflects the two-sample target history predicting a sinusoid, not the window.
- **Direction.** A[i,j] = TE(x_i → x_j), with row = source. Delayed_noise at T=256 (k=1): A[0,1] = 1.616, A[1,0] = −0.016.

**Proposed checks.**
- Kernel tolerance ≤ 1e-12.
- Direction on delayed_noise, var2_bivariate, gauss_ar1_pair and logistic_maps.
- Analytic Gaussian TE for the same embedding. For gauss_ar1_pair with k=l=1: 1->0 = 0.0524, 0->1 = 0.0271. var2_bivariate is computed from its autocovariances.
- iid null.
- For the k-2 DCE unit, a CalculatorFrame comparison (see below).

## Transfer entropy, auto-embedding (MAX_CORR_AIS)

Two units: `te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4` and `te.kraskov_DCE_auto-MAX_CORR_AIS_kmax-10_taumax-4`.

**Documented.** "The optimized parameters are inferred by choosing the embedding parameters (up to a maximum embedding length of 10, denoted by k-max-10, and maximum time delay of 2, tau-max-2) that maximize ... active information storage (see the MAX_CORR_AIS method of [32])." The GitBook repeats tau-max-2.

**Canonical.** Schreiber TE with embeddings chosen by maximum AIS (Garland, James & Bradley 2016, PRE 93, 022221, which JIDT cites).

**Implemented.**
- In `TransferEntropyCalculatorViaCondMutualInfo.preFinaliseAddObservations` (`:701-855`), an `ActiveInfoStorageCalculatorKraskov(1)` (`:740`) receives every stored property (`:752-754`), including `k`, `NOISE_SEED` and the pair's `DYN_CORR_EXCL`.
- It searches k = 1..10 and τ = 1..4, with τ only 1 when k = 1 (`ActiveInfoStorageCalculatorViaMutualInfo.java:579-614`). It keeps the strict maximum of the KSG1 AIS, with no extra bias correction for KSG (`:636-639` returns 0).
- This is done first for the target (k, kt) and then for the source (l, lt), and repeated on every `setObservations`, so for every ordered pair.
- A from-scratch search reproduces both the chosen embedding and the TE to 12 decimals on var2_bivariate, gauss_ar1_pair and logistic_maps at T=256 (p6).

**Deviations and candidates.**
- **tau-max.** Documented 2; implemented 4 (`config.yaml:556`, identifier `tau-max-4`). (Corrected at the check stage: only the prose says 2; the identifier lists printed with it say 4, which the code matches.)
- **Embedding varies by pair and T (pack hypothesis confirmed).** On var2_bivariate 0->1 the chosen (k,kt,l,lt) is (3,3,1,1) at T=64, (10,1,1,1) at T=256 and (3,1,2,1) at T=1024 (p4). On cml7 (first 3 channels, T=100) it is (1,1,1,1) and equals the fixed k=1 TE (p11). That cml7 case is already in tests/README.md "worth knowing".
- **DCE feeds into the embedding search.** gauss_ar1_pair 1->0 at T=256 selects (2,3,5,1) without DCE and (1,1,2,2) with it. narrowband_f0-0.1 at T=256 with W=85 selects (10,4,10,4) for every pair.
- **DCE NaN at small T.** The k=10, τ=4 AIS candidate has N = T − 37. JIDT requires N > 4 + 2W, and one failing candidate aborts the search (`:609-612`). At T=64 this gives all-NaN pairs for the narrowband series (6/6) and logistic_maps (2/2), and 6/12 pairs for shared_envelope (p13). None at T=256.
- **Efficiency.** Without DCE, a channel's embedding does not depend on its partner, but it is recomputed for each of its 2(P−1) pairs.

**Proposed checks.**
- Kernel tolerance with a from-scratch MAX_CORR_AIS search.
- Exact read-back of the chosen (k, kt, l, lt) for each pair.
- Direction.
- iid null.
- Qualitative NaN pattern for the DCE unit.
- CalculatorFrame comparison (next section).

## New candidate across the family: copies lose the calculator configuration

pickle and `copy.deepcopy` both keep the identifier, but they lose different settings.

- **Pickle.** `JIDTBase.__getstate__` (`infotheory.py:60-69`) drops the JIDT objects. `TransferEntropy.__setstate__` (`:369-374`) rebuilds `_calc` with `_getcalc`, which applies only `_setup` (`:84-93`: k = prop_k, `BIAS_CORRECTION`, `NOISE_SEED` 42). The `AUTO_EMBED_METHOD`, `k_HISTORY` and `l_HISTORY` properties set in `__init__` (`:344-367`) are lost. k and the seed are kept. After a pickle round trip, `getProperty` returns `k_HISTORY = '1'` and `AUTO_EMBED_METHOD = 'NONE'` (p9).
- **Deepcopy.** `JIDTBase.__deepcopy__` (`:77-82`) builds `type(self)()` and then sets every attribute to a deep copy of the original's. A deep copy of a JPype JIDT object is a new Java object with JIDT constructor defaults, not a copy of pyspi's properties, and `_setup` is not run on this path. So the copy also loses `NOISE_SEED` (it reads `'NONE'`, and the 1e-8 jitter is unseeded) and `prop_k` (it reverts to JIDT's default k = 4). `getProperty` read-back on the copies is in `it_kraskov_fix/p_te_copy.py`.

`CalculatorFrame.init_from_list` deep-copies the base Calculator (`calculator.py:556-563`, `copy.deepcopy` at `:559`), so it takes the deepcopy path. Probe p10 and `p_te_copy.py` compare the paths on var2_bivariate at T=256:

| Identifier | Calculator A[0,1] | deepcopy / CalculatorFrame A[0,1] | pickle A[0,1] | Copy computes |
|---|---|---|---|---|
| `te_kraskov_NN-4_k-max-10_tau-max-4` | 0.060599 | 0.133643 | 0.133643 | the k=1 TE (0.133643) |
| `te_kraskov_NN-4_DCE_k-2_kt-1_l-1_lt-1` | 0.159244 | 0.134372 | 0.134372 | k=1 with DCE |

Consequences for the rest of the family:

- **Tie-free data.** MI, TLMI and the k=l=1 TE units give the same values in the copies (diff 0.0 on var2_bivariate). Their embeddings match only because the JIDT defaults happen to be k_HISTORY = l_HISTORY = 1, and the lost seed does not move any neighbour count.
- **Tied data.** The deep copies are unseeded, so their values vary from run to run. Pickle copies reproduce the original. On logistic_maps at T=256, pair 1->0 (channel 1 has 4 distinct values), the original gives the same value on every run, but three deep copies (`r6_tied.py`, `p_tied_dce.py`) give:

  | SPI | original | three deep copies |
  |---|---|---|
  | mi | 0.226622 | 0.230475 / 0.236390 / 0.237750 |
  | tlmi | 0.278902 | 0.270293 / 0.274896 / 0.279048 |
  | te_k1 | 0.115063 | 0.113357 / 0.111490 / 0.114040 |
  | mi_DCE | 0.603752 | 0.592057 / 0.606156 / 0.603638 |
  | tlmi_DCE | 0.650270 | 0.648670 / 0.645450 / 0.650404 |
  | te_DCE_k1 | 0.078301 | 0.083693 / 0.079346 / 0.080654 |
  | te_auto | 0.101410 (pickle 0.115063) | 0.114476 / 0.109197 / 0.111046 |

  The review's run of `r6_tied.py` gave different deep-copy values (for example mi 0.234155 / 0.233200 / 0.231582), as expected without a seed.
- **prop_k ≠ 4.** This does not affect the nine NN-4 units, but any `mi_kraskov_NN-k` with k ≠ 4 changes under deepcopy. `mi_kraskov_NN-6` on bivariate_gaussian at T=256, pair (0,1), goes from 0.039668 to 0.028352 (`r4_deepcopy.py`).

Outside this family, the same deepcopy path changes `gc_gaussian_k-max-10_tau-max-2` (0.16556 → 0.1585) and `te_symbolic_k-10_kt-1_l-1_lt-1` (p14). It also resets `KERNEL_WIDTH` to 0.25 for kernel SPIs (`je_kernel_W-0.5` 3.982283 → 3.748712, `r4_deepcopy.py`). Those belong to it_gaussian and it_kernel and are not checked here beyond these probes. The regression tests use the plain `Calculator` path, which does not copy SPIs, so the goldens are not affected by this, though that does not make them a reference.

## Open questions

1. **Jar provenance.** Which exact commit built the jar? The jar's content (NOISE_SEED in the MI and entropy Common classes, `EntropyCalculatorMultiVariateCommon.class`) and the API commit dates point to master a9b6b3e (2024-04-20), but this is unconfirmed without a byte-level rebuild. Should the audit environment record the jar as "JIDT master snapshot 2024-04-20" rather than 1.6.1? With the real v1.6.1, `NOISE_SEED` would be ignored.
2. **Bartlett and Cliff formulas.** The τ = 1 + 2Σρ_xρ_y form and its attribution to Bartlett 1946 and Cliff et al. 2021 are cited from memory. Cliff et al. may use a (1 − k/T)-weighted sum over sample ACFs, which would change the "+1" statement only in its finite-T interpretation, not the code finding. The KSG Eq. (8) and Schreiber Eq. (4) numbers are also from memory.
3. **Intended window rule.** Was the DCE heuristic meant to sum only a few lags, or to use a 1/e or first-zero autocorrelation time? The documentation's "product of the autocorrelation functions" does not say. The ∝T behaviour on oscillatory input is the main practical consequence.
4. **Jitter and dataset D8.** The logistic_maps jitter-sensitivity and NaN results depend on the current periodic r_y=3.5 driver. Task #16 plans to change D8 to a chaotic map, after which those probes must be rerun. Are ties common in the EEG data (ADC quantisation)? If so, jitter sensitivity applies there too.
5. **Thread count in reproducibility tests.** Should `NUM_THREADS` be pinned to 1 so that golden comparisons across machines are exact? Current differences are ≤ 2e-14.
6. **ψ(N) with a window.** Is leaving ψ(N) unadjusted under a Theiler window acceptable for the audit's "canonical" verdict, or should the check stage treat it as a variant? JIDT itself flags it as an open TODO.
7. **Copy bug scope.** Does any user-facing workflow (CalculatorFrame, pickling for multiprocessing, saved calculators) feed EEG results? If so, which stored values were produced through a copy, and by which path? Through pickle, the auto-embedding and k-2 TE (and GC) values are wrong. Through deepcopy, which CalculatorFrame uses, those values are wrong and, in addition, every mi/tlmi/te kraskov value ran with unseeded jitter, any NN-k with k ≠ 4 ran with k = 4, and kernel SPIs ran with width 0.25. The unseeded jitter matters only if the EEG data contain ties (see question 4).

## Checks

All numbers below are from `audit/checks/out/results/it_kraskov.json` (224 runs: 213 pass, 11 fail, 0 without a reference; the 11 failures are the findings and statistical excursions named below; manifest `audit/checks/manifest/it_kraskov.json`, references `audit/checks/refs/it_kraskov.py`, probes `audit/checks/probes/it_kraskov_probes.py`). The pyspi side took 311 s in total on a loaded 64-core machine (largest single run 12 s); the references a few minutes. The manifest and references were extended after an adversarial review (theiler population runs at T=256/4096, an NN-6 copy run, te_DCE_auto copy runs, tlmi direction and time-reversal runs, and DCE guard runs) and the whole family was re-run with `--force`. T is at most 1024 except for four cheap convergence runs at T=4096 (two `mi.kraskov` pairs, one `te_k1` and one `te_DCE_k2` run, each at most 1.4 s) and two cheap window-rule population runs at T=4096; the auto-embedding units stay at T <= 256 with P <= 3 (P = 4 for one T=64 NaN-pattern run).

**The reference.** The main reference is a from-scratch KSG estimator that does not import JIDT or jpype. It z-scores every column with the N−1 std. It then adds the 1e-8 jitter from a pure-Python port of `java.util.Random`: the 48-bit LCG plus the polar `nextGaussian`, seeded with NOISE_SEED at every `finaliseAddObservations`, filling rows in order and, within a row, source/var1 columns, then dest/var2, then cond (`MutualInfoMultiVariateCommon.java` a9b6b3e:626-642, `ConditionalMutualInfoMultiVariateCommon.java`:766-785). Finally it runs the shared kernel's `mi_ksg1`/`cmi_ksg1` with `normalise=False` (cKDTree max norm, k = 4, strict counts, exclusion |s−t| ≤ W, ψ(N) unadjusted), with NaN when N ≤ k + 2W. In the JVM the port matches `java.util.Random.nextGaussian` to ≤ 8.9e-16 over 20 001 draws for seeds 42, 43 and 7 (scratchpad `it_kraskov_check/jr_test.py`). TE uses the kernel's `te_embedding` (JIDT layout). Auto-embedding uses a from-scratch MAX_CORR_AIS search: k 1..10, τ 1..4, the τ loop broken at k = 1, the strict maximum of the uncorrected KSG1 AIS, and NaN if any candidate trips the guard. The DCE window uses an explicit lag-loop ACF.

Because the jitter is regenerated exactly, the comparison is exact even on the tied `logistic_maps` fixture, and for every seed. That settles the earlier "unreproducible jitter" caveat for KSG.

### infotheory.est.ksg1: matches, 9
- JIDT's MI and CMI calculators driven directly (`direct`, W ∈ {0, 1, 2, 5, 20, 85}): max_abs 3.6e-15 (var2_chain T=1024) and 5.3e-15 (delayed_noise rows 0-2 T=256).
- Across the 64 `*|ksg` runs of the nine SPIs: ≤ 7.1e-15.
- **DYN_CORR_EXCL semantics** (`semantics` probe on a 32-point ramp and a 40-point pair). Univariate and KdTree counts are 31 − 2W for W = 0..6. The nearest admissible neighbour sits at |s−t| = W+1. The calculator raises at N = 4 + 2W and computes at N = 5 + 2W. The MI column matches the "|s−t| ≤ W excluded" rule to 1.3e-15; the alternative "< W" rule gives different values (W=1: −0.0559 vs −0.0075). So the excluded offsets are |s−t| ≤ W, and W = 0 excludes only the point itself. Source: `KdTree.java` a9b6b3e:752 and :1125, `UnivariateNearestNeighbourSearcher.java`:278-346 and :503-525; guard at `MutualInfoCalculatorMultiVariateKraskov.java`:298 and `ConditionalMutualInfoCalculatorMultiVariateKraskov.java`:316.
- **Properties** (`props`). NUM_THREADS defaults to `availableProcessors()` = 64 for both the MI and the CMI calculator (`MutualInfoCalculatorMultiVariateKraskov.java`:133, `ConditionalMutualInfoCalculatorMultiVariateKraskov.java`:155). `BIAS_CORRECTION` reads back as None. NOISE_SEED is unset on a fresh calculator and 42 after `_setup`.
- **Score.** 9 (rubric: canonical up to a convention): the jitter decides tied data, and ψ(N) is not adjusted under a window.
- **Candidates.** All five confirmed: provenance (NOISE_SEED is honoured; the exact master commit is still unproven), thread dependence, tie/jitter dependence, the BIAS_CORRECTION no-op, and negative values (an estimator property).

### infotheory.mi.kraskov: matches, 10
- **Exactness.** The `ksg` runs agree to ≤ 4.4e-15 (7/7), including the tied logistic_maps at 8.9e-16.
- **Kernel without any jitter.** The shared kernel `mi_ksg1` gives 3.6e-15 on bivariate_gaussian and nonlinear_dependence (`nojitter`). This agrees with the kernel selftest, whose INFO line gives the 1e-8 jitter effect on `mi_ksg1` as +0.00e+00 nats on tie-free bivariate_gaussian data (that figure is in the selftest output, not in the docstring).
- **External package.** `sklearn==1.9.1 mutual_info_regression(n_neighbors=4)` matches to 0.0–1.3e-15 on four positive-MI pairs (`sklearn`), so the tolerance is left at 1e-12.
- **Symmetry and invariance.** The unmirrored ordered matrices are exactly symmetric (max|O−O.T| = 0.0, `ordered`). scaled_offset invariance holds to 3.6e-15.
- **Convergence to −½ ln(1−ρ²).** All |z| ≤ 1.31 against the Monte Carlo SD of the same estimator:

  | Pair (truth) | T=256 | T=1024 | T=4096 |
  |---|---|---|---|
  | (0,1), 0.0472 | 0.0284 | 0.0333 | 0.0595 |
  | (2,3), 0.5108 | 0.5379 | 0.4990 | 0.5084 |

- **Autocorrelated input.** gauss_ar1_pair gives 0.1395 against a population value of 0.1578. The MC mean is 0.1572, so this draw is within sampling error and the estimator is close to unbiased here.
- **Null.** iid_gaussian at T=256: mean −0.0186 (MC −0.0007 ± 0.0112, z −1.60), minimum −0.055, 80% of pairs negative.
- **Finding (copy path).** `copy` fails as expected on tied logistic_maps (A[0,1] = 0.2301): CalculatorFrame and deepcopy run with unseeded jitter, so each run gives a non-reproducible draw (here 0.2300 and 0.2354; an earlier run of the same check gave 0.2248). Calculator and pickle are exact. `copyk6` shows the prop_k loss on tie-free data: MutualInfo(kraskov, prop_k=6) gives 0.5290 through Calculator and pickle (<= 5.3e-15 from the k=6 reference), but 0.5379 through CalculatorFrame and deepcopy, which is the k=4 value (2.7e-15). `copypropsk6` reads k = 6 / 4 / 6 for original / deepcopy / pickle. Before these runs the prop_k half rested only on the trace-stage probe `r4_deepcopy.py`.

### infotheory.mi.kraskov_DCE: variant, 6
- **Exactness.** 10/10 `ksg` runs agree to ≤ 7.1e-15 for windows W = 1..339, including detrended trending input.
- **Effect of the window.** For narrowband_f0-0.1 at T=1024, the estimate goes from 1.442 to 2.254 with W = 339. For logistic_maps at T=256 it goes from 0.230 to 0.606 (W = 39).
- **gauss_ar1_pair** (W = 5): pyspi 0.1475. The DCE estimator's MC mean is 0.1697, against 0.1572 without DCE and a population value of 0.1578. Here the window adds about +0.012 of expected bias instead of removing bias.
- **Null.** z −0.93.
- **Score.** 6 (rubric: deviates from both the supplement and canonical in an interpretable way). The KSG part is exact; the value is set by the window rule, which `infotheory.theiler_window` scores 6 because it is not the documented autocorrelation-time heuristic. A unit whose value depends on that rule cannot also match the supplement variant (anchor 7), so all four fixed DCE consumers move from 7 to 6, in line with te_DCE_auto (review item).

### infotheory.theiler_window: variant, 6
- **Exact reproduction.** The raw `data.theiler` values match the from-scratch lag loop exactly (12/12 runs, max_abs 0.0, no integer mismatches).
- **Bartlett + 1 (candidate resolved).** raw − (1 + 2Σ_{k≥1} r_i r_j) = 1.0000 on all 11 pairs:

  | Dataset (T=1024) | Bartlett | pyspi raw | W |
  |---|---|---|---|
  | var2_bivariate | 1.6398 | 2.6398 | 2 |
  | iid_gaussian (10 pairs) | 0.940–1.040 | 1.940–2.040 | 1 or 2 |

  Truncating the sum at T/5 lags gives 1.42 for var2_bivariate.
- **Population (Bartlett τ + 1).** On cross-correlated pairs the expectation of the rule sits above τ + 1, and the excess does not shrink as T grows. So this is a non-vanishing (asymptotic) upward bias of the all-lag rule, not a finite-sample bias: the rule is an inconsistent estimator of Bartlett's factor (review item; my earlier wording "finite-sample bias" was wrong). Each lag k beyond the correlation scale contributes E[r_i(k) r_j(k)] ≈ Cov(r_i(k), r_j(k)) ≈ (T−k)/T² · C with C = Σ_m ρ_ij(m)² (Bartlett's covariance of lagged products for a Gaussian pair). Summing all T−1 lags gives 2 Σ_k (T−k)/T² · C → C, so E[raw] → τ + 1 + C (`refs/it_kraskov.py:pop_cross_sq`). For independent channels C = 0 and the rule has no such excess.

  | Dataset | T | τ + 1 | MC mean ± SD of the rule | MC excess | limit τ + 1 + C | pyspi raw |
  |---|---|---|---|---|---|---|
  | gauss_ar1_pair | 256 | 3.63 | 4.802 ± 1.319 | 1.18 | 5.197 | 4.358 (z 0.34) |
  | gauss_ar1_pair | 1024 | 3.63 | 5.061 ± 0.697 | 1.44 | 5.197 | 5.144 (z 0.12) |
  | gauss_ar1_pair | 4096 | 3.63 | 5.167 ± 0.350 | 1.54 | 5.197 | 4.741 (z 1.21) |
  | var2_bivariate | 256 | 2.34 | 2.705 ± 0.209 | 0.36 | 2.749 | 3.421 (z 3.42, fail) |
  | var2_bivariate | 1024 | 2.34 | 2.739 ± 0.119 | 0.39 | 2.749 | 2.640 (z 0.83) |
  | var2_bivariate | 4096 | 2.34 | 2.749 ± 0.061 | 0.40 | 2.749 | 2.643 (z 1.75) |
  | iid_gaussian | 1024 | 2.00 | 2.001 ± 0.038 | 0.001 | 2.000 | 2.024 (z 0.60) |

  The var2_bivariate T=256 run fails the 2-SD criterion: pyspi 3.421 against MC 2.705 ± 0.209 (z 3.42). pyspi matches the lag-loop rule exactly on this draw. The rule's distribution is right-skewed (skew 0.71), and in a 2000-draw probe (scratchpad `it_kraskov_check/theiler_tail.py`) 0.55% of draws reach 3.421. I read it as a tail draw of this dataset and did not loosen the criterion. The reviewer's own Monte Carlo reached T = 16384 (mean 5.21 for gauss_ar1_pair), consistent with the limit 5.197.
- **Growth with T.** narrowband_f0-0.1 gives 22.1 / 85.4 / 339.2 / 1352.9 at T = 64 / 256 / 1024 / 4096. That is within 1.2% of 2a² Σ(1−k/T)² cos²(ωk), which is about T/3.
- **Detrending.** Trending pairs go from 340–350 to 2.0–2.7. Cointegrated pairs stay at 38–111 (from 158–206).
- **Candidates.** All five confirmed. The efficiency point (ACF recomputed per pair) is confirmed by reading `infotheory.py:202-209`.

### infotheory.tlmi.kraskov: matches, 9; infotheory.tlmi.kraskov_DCE: variant, 6
- **Exactness.** 6/6 (tlmi) and 11/11 (tlmi_DCE, including four T=64 guard-margin runs) `ksg` runs agree to ≤ 6.2e-15.
- **Direction (measured).** A[i,j] = I(x_i(t); x_j(t+1)), with the row being the earlier channel.

  | Dataset (T=1024) | A[0,1] | A[1,0] | Population |
  |---|---|---|---|
  | delayed_noise | 2.323 | −0.004 | |
  | var2_bivariate | 0.164 | −0.032 | |
  | gauss_ar1_pair | 0.139 | 0.196 | 0.127 / 0.174 |

- **Convergence** (all |z| ≤ 0.65). On delayed_noise (0,1) pyspi gives 2.3233 against ½ ln 101 = 2.3076. On gauss_ar1_pair the DCE estimator's expectation (MC 0.1388 / 0.1845) sits further above the population than the non-DCE one (0.1309 / 0.1771).
- **Direction on logistic_maps_chaotic** (x → y only; replaces the tied `logistic_maps` fixture named in the proposed check, as CONVENTIONS.md asks): A[0,1] 0.822 vs A[1,0] 0.251 (tlmi), 0.823 vs 0.252 (tlmi_DCE), both pass.
- **Time reversal** (`reversal`, probe `time_reversed`). For y = x reversed, I(y_i(t); y_j(t+1)) uses the sample pairs (x_j(s), x_i(s+1)), so A(y) = A(x).T up to the jitter, which follows the sample order. pyspi gives max|A(y).T − A(x)| = 7.1e-15 on delayed_noise and 2.7e-15 on iid_gaussian, but 0.0063 on tied logistic_maps (tlmi_DCE: 5.3e-15, 7.1e-15, 0.0057). The from-scratch reference reproduces both layers on all six runs (max_abs <= 6.2e-15), and without jitter the reversal is exact on the tied data too (8.9e-16). So "equals the reversed-time transpose only approximately" holds exactly as far as the jitter goes.
- **tlmi_DCE guard** (proposed qualitative check). At T=64 (N = 63) the rule gives at most W = 22 on narrowband_f0-0.1/0.25, logistic_maps and shared_envelope. The guard margin N − (4 + 2W) is at least 15, and no pair is NaN (4/4 `ksg` runs exact). The rule's window stays near T/3 or below on these inputs, while the guard needs W >= (N − 4)/2 ≈ T/2, so it cannot trip here. To show the guard itself, `guard` rebuilds the SPI with a numeric window: W = 29 computes, and W = 30 gives NaN on all 6 ordered pairs (pyspi logs a warning per pair), both layers matched by the reference to 3.1e-15.
- **Score.** 9 for tlmi.kraskov because of the supplement's "undirected" keyword, which contradicts the directed, asymmetric output (max|A−A.T| = 2.33). That candidate is confirmed. The asymmetry itself is already documented (REVIEW M8). 6 for tlmi_DCE, for the same reason as mi_DCE: the value depends on the window rule scored 6.

### TE, fixed embeddings
- **te_k-1 (matches, 10).** 6/6 `ksg` runs agree to ≤ 3.1e-15. Convergence on gauss_ar1_pair towards 0.0271 / 0.0524:

  | T | pyspi 0→1 / 1→0 | z |
  |---|---|---|
  | 256 | −0.0021 / 0.0630 | 0.91 |
  | 1024 | 0.0444 / 0.0601 | 0.97 |
  | 4096 | 0.0253 / 0.0511 | 0.21 |

  delayed_noise (0,1) gives 2.0097 against 2.3076. The MC mean is 2.0505, so the gap is KSG bias at large MI (z 1.75). Direction passes 4/4, with logistic_maps_chaotic x→y at 0.565 vs −0.074. All four copy paths agree on tie-free data.
- **te_DCE_k-1 (variant, 6).** 11/11 `ksg` runs agree to ≤ 2.9e-15, including cointegrated with W = 158–205. On narrowband (W = 339), A[1,0] goes from 0.717 to 0.675. Guard (proposed check): at T=64 the rule gives W <= 22 on narrowband_f0-0.1/0.25, logistic_maps and shared_envelope, margin N − (4 + 2W) >= 15, no NaN. With a numeric window, W = 29 computes and W = 30 gives NaN on all 6 pairs (`guard`, reproduced to 6.7e-16). Score 6 as for mi_DCE.
- **te_DCE_k-2 (variant, 6).** 7/7 `ksg` runs agree to ≤ 1.1e-15. Score 6 as for mi_DCE; in addition every copy path computes k=1 (below).
  - **Convergence.** Against the k=2 Gaussian TE: gauss_ar1_pair z 0.11.
  - **Statistical failure.** var2_bivariate (population 0.1241) fails the 2-SD criterion at T=1024: 0.1600 vs MC 0.1203 ± 0.0196, z 2.02. That draw is exact against the from-scratch estimator (2.2e-16), and the same draw is also high for te_k1 (z 1.26). T=256 (z 1.39) and T=4096 (0.1351, z 0.94) pass. I read it as a sampling excursion and did not loosen the criterion.
  - **Finding (copy path).** `copy` fails: every copy gives the k=1 value, 0.1344, instead of 0.1592.
  - **Documentation slip.** The `config.yaml` comment "auto-embedding on target-only" does not match the fixed k=2 that is configured.

### TE, auto-embedding (MAX_CORR_AIS)
- **te_auto (variant, 7).**
  - **Exactness.** 5/5 `ksg` runs agree to ≤ 1.1e-15. The read-back (k, kt, l, lt) is identical to the from-scratch search on 5/5 `embed` runs.
  - **Chosen embeddings.**

    | Dataset | Pair | (k, kt, l, lt) |
    |---|---|---|
    | var2_bivariate T=64 | 0→1 | (3,3,1,1) |
    | var2_bivariate T=256 | 0→1 | (10,1,1,1) |
    | gauss_ar1_pair T=256 | 0→1 | (5,1,2,3) |
    | gauss_ar1_pair T=256 | 1→0 | (2,3,5,1) |
    | narrowband_f0-0.1 T=256 | 0→1 | (2,1,4,1) |
    | logistic_maps T=256 | 1→0 | (1,1,7,2) |

  - **Fixed path reproduces auto.** The fixed-embedding pyspi TE with those values equals the auto-embedding value exactly (difference 0.0 on 4/4 `fixed` runs). This supports the optimisation note: the embedding search can be cached per channel and the TE recomputed with the fixed path.
  - **Seeds.** On tied input, seeds 42/43/44 give 0.1014 / 0.0891 / 0.0442 with different source embeddings, each reproduced exactly.
  - **Finding (copy path).** `copy` and `copyprops` fail. The copies compute 0.1336 (the k=1 value) instead of 0.0606. AUTO_EMBED_METHOD is lost by both pickle and deepcopy, and NOISE_SEED by deepcopy.
  - **Selection criterion.** How MAX_CORR_AIS compares with a canonical criterion is literature-only. Ragwitz & Kantz 2002 (PRE 65, 056201) minimise local prediction error; Garland, James & Bradley 2016 maximise AIS. The supplement documents MAX_CORR_AIS, so this is the supplement's variant, value-changing against a canonical criterion (anchor 7).
  - **tau-max (candidate refuted as a code deviation).** The documentation contradicts itself. The supplement prose (§1.4.5, `supplement.txt:489`) and the GitBook prose (`information-theory.md:216`) say tau-max-2, but the SPI lists printed with them name `te_kraskov_NN-4_k-max-10_tau-max-4` and `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` (`supplement.txt:495-496`, `information-theory.md:222-223`). The code (`config.yaml:556`, :562) matches those identifiers. tau-max-2 is the gc_gaussian setting (`config.yaml:601-602`). What remains is a prose slip inside the documentation, which is no longer used as a deviation-from-supplement argument for either auto unit.
- **te_DCE_auto (variant, 6).**
  - **Exactness.** 8/8 `ksg` runs pass, with the NaN masks identical. The T=64 NaN pattern is all pairs of narrowband_f0-0.1 (W=22), both pairs of logistic_maps (W=14), and 6/12 pairs of shared_envelope, exactly where T − 37 ≤ 4 + 2W.
  - **The window enters the embedding search.**

    | Dataset (T=256, window) | Without DCE | With DCE |
    |---|---|---|
    | gauss_ar1_pair (W=4) | (5,1,2,3) / (2,3,5,1) | (2,2,1,1) / (1,1,2,2) |
    | narrowband_f0-0.1 (W=85) | (2,1,4,1) | (10,4,10,4) |
    | logistic_maps 1→0 (W=39) | (1,1,7,2) | (1,1,9,4) |

  - **Fixed path.** `fixed` passes 4/4.
  - **Finding (copy path), now run for this unit.** On var2_bivariate T=256, A[0,1] is 0.0571 through Calculator (matching the reference to 2.2e-16), but 0.1344 / 0.1344 / 0.1344 through CalculatorFrame / deepcopy / pickle (the k=l=1 DCE TE on the same data is 0.1344). `copyprops`: AUTO_EMBED_METHOD is lost by both copies (read-back 0), NOISE_SEED by deepcopy. Previously this unit's copy candidate rested on the shared code path and the non-DCE runs.
  - **Score.** 6 (unchanged): the window rule (scored 6) feeds the embedding search. The tau-max-2 argument is withdrawn (see above).

### Directions, symmetry, null (all directed units)
- **Directions.** Every direction run passes (28/28), including the two tlmi runs on logistic_maps_chaotic. For TE, A[i,j] = TE(x_i → x_j) with the row as the source. For TLMI the row is the earlier channel.
- **Symmetry.** MI and MI_DCE are exactly symmetric. TLMI and TE are asymmetric.
- **Null.** All 9 null runs are consistent with the Monte Carlo null (|z| ≤ 1.60). Negative values are common: minima range from −0.031 to −0.070.

### Trace-stage fields corrected
- **`infotheory.est.ksg1`, third proposed check.** The NUM_THREADS / NOISE_SEED self-consistency check was relabelled from `reference_kind: analytic` to `same-lineage`, since it compares pyspi with pyspi.
- **Remaining trace issue (CalculatorFrame-vs-Calculator `reference_kind`).** This was already `same-lineage` in all three kraskov TE records when this stage started, so it needed no edit. The check-stage copy runs compare every copy path against the independent from-scratch value, and `copyprops` is labelled same-lineage.
- **logistic_maps values.** The trace-stage probe numbers on logistic_maps (e.g. mi 1→0 0.226622 for seed 42) differ from the values on the current data file (0.2301). The trace probes evidently predate the present D8 files. Every logistic_maps number in the records and in this section comes from the current files.
- **Direction dataset.** For directed checks I used `logistic_maps_chaotic` (x → y only) in place of the tied `logistic_maps` fixture, as data/CONVENTIONS.md asks. This now covers the two tlmi units as well as the TE units.
- **tau-max.** The trace-stage deviation "documented 2, implemented 4" in `te.kraskov_auto` and `te.kraskov_DCE_auto` (`deviation_from_documented` (1) in each) was corrected: the documentation is self-inconsistent and the code matches its identifiers (see TE auto-embedding above). Both auto records mark the candidate refuted.
- **Reference kinds of the Monte Carlo runs.** `truth`, `null` and `pop` runs now carry `reference_kind: independent-impl` in results.json, because the pass criterion uses the Monte Carlo mean and SD of the from-scratch estimator (or rule). The analytic population value is still quoted (`population`, `population_kind: analytic`).
- **Window-rule bias.** `infotheory.theiler_window` closeness and what_changed, and Open question 3 below, now say "non-vanishing (asymptotic) upward bias" instead of "finite-sample bias".
- **DCE scores.** mi_DCE, tlmi_DCE, te_DCE_k-1 and te_DCE_k-2 go from 7 to 6 (see mi_DCE).

### Kernel issues
- **Jitter caveat.** At the check stage the module note of `_kernels_infotheory.py` said the seeded JIDT jitter "cannot be reproduced in numpy". A 40-line pure-Python `java.util.Random` port reproduces it to the last bit that matters (`refs/it_kraskov.py:JavaRandom`). I used the port and left the kernel unchanged. The module note was revised on 2026-10-15. It now says "That stream CAN be reproduced", names the ports `refs/it_kraskov.py:JavaRandom` and `refs/it_kozachenko.py:JavaRandom`, and says the kernel's own `noise_level`/`seed` jitter (numpy `default_rng` normals) is only for quantifying the jitter effect; to mirror JIDT, the ported stream is added to the inputs and `noise_level=0` is passed.
- **Source citations.** The kernel cites v1.6.1-tag line numbers (for example the unseeded `new Random()`). The shipped jar behaves like master a9b6b3e (seeded), so this family cites a9b6b3e lines. Since the 2026-10-15 revision the kernel's CITATION VERSION note says this itself: its JIDT lines are at the v1.6.1 tag unless marked a9b6b3e, and each file that changed after the tag has its a9b6b3e lines in brackets.

## Open questions: answered

1. **Jar provenance.** Numerically, the jar honours NOISE_SEED: three seeds reproduce exactly and give different values on tied data. So it is not the v1.6.1 tag, which ignores the seed. The specific master commit (a9b6b3e by date) is still unconfirmed; a byte-level rebuild of that commit compared with the shipped class files would settle it. Recording the dependency as "JIDT master snapshot (post-1.6.1, NOISE_SEED present)" is accurate on the evidence here.
2. **Bartlett/Cliff forms and equation numbers.** This stage re-derived the identity the code implies: raw = 1 + (1 + 2Σ_{k≥1} r_i r_j), with Bartlett's factor in the variance-inflation form used by `theiler_window_bartlett`. It did not check the exact forms in Bartlett 1946 / Cliff et al. 2021 or the KSG Eq. 8 / Schreiber Eq. 4 numbers against the papers. That still needs a reading of the papers. The code finding does not depend on it.
3. **Intended window rule.** The code cannot answer this. The evidence shows what the implemented rule does: it is about T/3 for a sinusoid (within 1.2% of the closed form), and on cross-correlated pairs it carries an upward bias that does not vanish with T: its expectation tends to τ + 1 + Σ_m ρ_ij(m)² (gauss_ar1_pair MC 5.17 at T=4096 against τ + 1 = 3.63 and the limit 5.20). A lag-truncated sum behaves differently (var2_bivariate 1.42 at T/5 lags against 1.64 over all lags). The authors' intent (or Cliff et al.'s code) would settle it.
4. **Jitter and D8; EEG ties.** logistic_maps is still the tied period-4 fixture, and the seed spreads on it are now quantified: 0.0035 (mi) to 0.057 (te_auto). The jitter is reproduced exactly, so tied values are deterministic for a given seed. Whether EEG data contain ties (ADC quantisation) was not checked. It would only matter for kraskov SPIs, and none is in the retained-185 EEG set (`pyspi/workflow/retained185.yaml` has only gaussian/symbolic infotheory estimators).
5. **NUM_THREADS in reproducibility tests.** The thread-count effect is ≤ 1.6e-14 (mi 7.1e-15, mi_DCE 1.07e-14, tlmi 1.6e-14, te_k1 2.4e-15, te_auto 1.8e-15), and repeat runs on one machine are bit-identical. Pinning NUM_THREADS=1 is only needed if regression comparisons across machines must be bit-exact; a 1e-12 tolerance absorbs it.
6. **ψ(N) under a window.** Treated as a JIDT convention. It holds `est.ksg1` at 9 rather than 10, and it is not what drives the DCE units' variant scores; the window-size rule is.
7. **Copy-bug scope.** The mechanism and magnitudes are confirmed: auto TE 0.0606 → 0.1336 and k=2 DCE TE 0.1592 → 0.1344 under any copy, and unseeded jitter under deepcopy (tied MI 0.2301 → a non-reproducible draw, 0.2300 in this run and 0.2248 in an earlier one), and k = 4 instead of k = 6 for an NN-6 MI (0.5290 → 0.5379). The EEG workflow (`pyspi/workflow/_run.py:107`) builds one plain `Calculator` and reuses it across windows through `load_dataset`, with no CalculatorFrame, deepcopy or pickle. Its retained metric set contains no kraskov SPI. So stored EEG results from this workflow do not pass through either copy path for this family. Results produced earlier through CalculatorFrame or pickled calculators cannot be assessed from the repository.

## Optimisation

Workflow 3 (group infotheory) is a short, profile-based search for where the time goes and which avenues look promising; nothing in pyspi was changed. Walls below are from `audit/profiles/summary.json` (quiet machine); percentages are shares of the profiled SPI call (`tests/regression/harness.py:compute_spi`) in the cProfile run at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`). That profiling pass ran contended for the JIDT units (profiled walls 1.3-20x the quiet walls), so its walls are not used, only its shares. cProfile cannot see inside Java: JPype dispatches JIDT methods without profiler entries, so the Java time is booked as self time of the calling pyspi frame, and every `opaque_share` in these records (0.0002-0.04) is a lower bound on JIT/JNI time, not an estimate of it. `expected_speedup` is the unit's per-window speedup at 118x1024 as an order of magnitude; risks carry the `unverified-` prefix unless a micro-benchmark diff file backs them. Agreement with today's output means only that a candidate reproduces it.

### Walls

| Unit | 8x1024 fresh | 32x1024 fresh | 118x1024 fresh (measure_mode) | 8x1024 family-warm | 118x1024 family-warm projected (measure_mode) |
|---|---|---|---|---|---|
| `mi.kraskov` | 1.02 s | - | 251 s projected (extrapolated-pairs-from-8) | 0.898 s | 221 s (extrapolated-pairs-from-8) |
| `mi.kraskov_DCE` | 1.09 s | - | 270 s projected (extrapolated-pairs-from-8) | 1.14 s | 280 s (extrapolated-pairs-from-8) |
| `tlmi.kraskov` | 0.99 s | - | 244 s projected (extrapolated-pairs-from-8) | 1.03 s | 254 s (extrapolated-pairs-from-8) |
| `tlmi.kraskov_DCE` | 1.1 s | - | 271 s projected (extrapolated-pairs-from-8) | 1.05 s | 259 s (extrapolated-pairs-from-8) |
| `te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4` | 38.8 s | - | 9.55e+03 s projected (extrapolated-pairs-from-8) | 39.9 s | 9.84e+03 s (extrapolated-pairs-from-8) |
| `te.kraskov_DCE_auto-MAX_CORR_AIS_kmax-10_taumax-4` | 40 s | - | 9.85e+03 s projected (extrapolated-pairs-from-8) | 39.3 s | 9.7e+03 s (extrapolated-pairs-from-8) |
| `te.kraskov_DCE_k-2_kt-1_l-1_lt-1` | 1.18 s | - | 291 s projected (extrapolated-pairs-from-8) | 1.31 s | 323 s (extrapolated-pairs-from-8) |
| `te.kraskov_DCE_k-1_kt-1_l-1_lt-1` | 0.794 s | - | 196 s projected (extrapolated-pairs-from-8) | 1.05 s | 259 s (extrapolated-pairs-from-8) |
| `te.kraskov_k-1_kt-1_l-1_lt-1` | 0.905 s | - | 223 s projected (extrapolated-pairs-from-8) | 0.77 s | 190 s (extrapolated-pairs-from-8) |

All Kraskov units are in `infeasible_above_8x1024.txt`; their 118x1024 values are pair-count projections of the 8x1024 wall with no fixed cost (`extrapolated-pairs-from-8`). `infotheory.est.ksg1` and `infotheory.theiler_window` are infrastructure (filled from the mi.kraskov and mi.kraskov_DCE profiles).

### Where the time goes

Every unit is one JIDT call per ordered pair, and the pyspi `bivariate` frame holds 99-100% of the profiled call as self time (mi 99.9%, mi_DCE 96.1% self with `_set_theiler_window` 3.8% cum, tlmi 99.9%, te k-1 99.9%, te auto 100.0%; te DCE auto 99.2% at 8x256 because its 8x1024 profile timed out). cProfile cannot split the Java call. The walls can: the T exponents between 8x256 and 8x1024 are -0.13 to 0.23 (mi 1.07 s at 8x256, 1.02 s at 8x1024) and a 7x100 pair already costs 10.6 ms, so the per-call cost is not the N log N neighbour search but JIDT's per-call thread fan-out: pyspi never sets NUM_THREADS, so every KSG call is split over `availableProcessors()` = 64 threads (process thread count 76, RSS ~1.8 GB in these runs). The auto-embedding units run ~75 such calls per pair (2 x 37 AIS candidates + the TE): 692 ms per pair.

### Opportunities

- **it_kraskov-1 (parallel; mi, mi_DCE, tlmi, tlmi_DCE and est.ksg1; refuted for all five TE units: te k-1, te DCE k-1/k-2, te auto and te DCE auto): NUM_THREADS=1 per calculator.** Measured: `audit/profiles/micro/opp_it_kraskov-1.py` / `.json` sets `NUM_THREADS=1` on the SPI's calculator and compares with the default call on the same Data (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, JIDT availableProcessors 64, all recorded in the JSON). The script has now run three times on the shared machine (the analyst's run, the reviewer's rerun and the fix agent's rerun; the JSON keeps the latest at top level and the other two under `earlier_runs`, with a `run_variation_note`). The output differences reproduce exactly in all three (max_abs_diff 1.6e-14 mi and mi_DCE, 2.3e-14 tlmi, 2.9e-15 te k-1, 1.1e-15 te auto; NaN masks equal), and so do the single-thread walls at 8x256 (mi 0.029-0.036 s) and for te auto at the labelled 3x256 shape (1.173-1.176 s). The single-thread walls at 8x1024 do not: mi 0.136 / 0.270 / 0.723 s, tlmi 0.135 / 0.344 / 0.634 s, mi_DCE 0.155 / 0.427 / 1.491 s, te k-1 0.507 / 0.888 / 2.088 s, with process CPU above the wall in the slow runs (JVM background threads), and the default-thread baselines in the same runs were 2.3-4.3 s against quiet walls of ~1 s. The same-run speedups the analyst quoted (22x, 22x, 28x) are therefore not reproducible (the reruns gave 4.7-10.8x, 3.6-7.9x, 2.9-6.6x). Against the quiet default walls in summary.json (mi 1.02, tlmi 0.99, mi_DCE 1.09, te k-1 0.91 s) the single-process gain is 1.4-7.5x (mi), 1.6-7.3x (tlmi), 0.7-7.0x (mi_DCE) and 0.4-1.8x (te k-1), the high end being the least contended run (process CPU equal to the wall); te auto gives 3.3-4.6x against its same-run defaults and ~3.1x against the quiet 8x256 per-pair cost scaled to 6 pairs (3.7 s), but only at the labelled 3x256 shape. At T=1024 the sign flips: the script's `--only 3x1024` mode (both auto identifiers at 3x1024, 6 ordered pairs, merged into the JSON under `shapes["3x1024"]` and logged in `merged_runs`, load 0.7-2.3) gives single-thread 7.11 s against a same-run default of 4.83 s for te auto (0.68x) and 7.46 s against 4.69 s for te DCE auto (0.63x), and 0.58x / 0.57x against the quiet defaults (summary.json 8x1024 fresh walls scaled to 6 pairs, 4.15 / 4.28 s). Expected per window: 1-8x (mi, est.ksg1), 1-7x (mi_DCE, tlmi, tlmi_DCE). For every TE unit the avenue is refuted as a per-window gain (~1x for te k-1 and te DCE k-1/k-2, below 1x for te auto and te DCE auto; see the refuted items below) and their records no longer list it. The larger and steadier effect is CPU: per call the single-thread run uses 2-38x fewer core-seconds (e.g. mi 0.17-0.90 s vs 3.2-5.9 s, te auto 1.1-1.3 s vs 10.7-11.2 s at 3x256 and 7.6 s vs 18.8 s at 3x1024), which is the throughput gain once windows run in separate processes (depends_on xc-window-parallel). Output: **within-tolerance (measured)** for mi, mi_DCE and tlmi (per-thread digamma partial sums are added in another order; the check stage measured the same <= 1.6e-14; the TE units, where the avenue is refuted, give 2.9e-15 for te k-1, 1.1e-15 / 2.0e-15 for te auto at 3x256 / 3x1024 and 2.0e-15 for te DCE auto at 3x1024); unverified-within-tolerance for tlmi_DCE (same code path, not run; its record carries tlmi's figure on a complexity basis, not tagged microbench).
- **it_kraskov-2 (cache-sharing; te auto): MAX_CORR_AIS once per channel.** Without DCE the AIS search sees only the channel's series, k and NOISE_SEED 42, so P searches replace 2P(P-1) (13806 x 74 -> 118 x 37 KSG AIS calls at P=118) and each pair runs one fixed-embedding TE with the cached (k, k_tau, l, l_tau). The check stage already showed that pyspi's fixed-embedding TE with the read-back embedding equals the auto value exactly (`ref_fixed_minus_auto_pyspi = 0.0` on 4/4 `...|fixed` runs), so the expected risk is unverified-bit-identical. Expected 10-40x for te auto (remaining cost: one KSG TE per pair, 16 ms at k = l = 1 and more for the larger selected embeddings, against 692 ms now). Not measured.
- **it_kraskov-4 (cache-sharing; te DCE auto): MAX_CORR_AIS once per (channel, Theiler window).** A different avenue from it_kraskov-2, with its own id: for te DCE auto the pair's Theiler window enters the search (gauss_ar1_pair selects (2,2,1,1) with DCE and (5,1,2,3) without), so a cache keyed on the channel alone would be semantic and the key must be (channel, W[i,j]). The saving is 2(P-1) divided by the number of distinct window values per channel, data-dependent: 1-10x. Unverified-bit-identical (the check stage's fixed-embedding runs for this unit pass 4/4). Not measured.
- **it_kraskov-3 (cache-sharing; theiler_window): one ACF per channel.** `_set_theiler_window` calls `utils.acf` (z-score + `np.correlate` 'full', O(T^2)) for both channels of every i<j pair: P(P-1) calls once per Data. In the mi_DCE profile acf is 56 calls, 0.023 s of the 5.13 s call (0.45%, ~0.41 ms per call). The frame's 3.3% self time there (0.171 s) is outside acf (the loop and the 56 JIDT `setProperty` calls, which JPype dispatches without cProfile entries) and the cache does not touch it. A self time that large appears in 2 of the 5 DCE profiles (mi_DCE at 8x1024; te DCE auto at 8x256, 0.299 s, 0.72%) and not in the other three (tlmi_DCE and te DCE k-1 / k-2 at 8x1024: 0.0023-0.0053 s, 0.06-0.08%). `audit/profiles/micro/probe_theiler_window_calls.py` / `.json` times every call without cProfile (five DCE units, 3 fresh processes x 2 SPI calls each; run2 at top level, run1 under `earlier_runs`): after the window-building first call, a call takes a median of 0.02-0.05 ms, apart from stalls. In te DCE auto at 8x256 every SPI call has them (run2: 3-7 above 10 ms; the largest 132-208 ms; 0.44-1.26 s per SPI call, 1.1-3.3% of its 37-43 s); in the four 8x1024 units they are rare (one stall in 1 of 24 SPI calls per run: 137 ms tlmi_DCE in run1, 155 ms mi_DCE in run2) and every other call stays at or below 0.18 ms. They are not this frame's work: the probe's `precall` case times a trivial Java call (`getProperty`) just before each `_set_theiler_window` call in te DCE auto, and the stalls move to that call (4-6 per SPI call in 5 of 6) while the frame stays below 0.06 ms. So they land on the first Java call of a pair, a JVM-side wait whose cause (GC, safepoints) was not measured, and neither this cache nor any other change to `_set_theiler_window` removes them. At P=118 the window build is ~13806 x 0.41 ms = ~5.7 s per window (3.7-10.6 s over the 0.27-0.77 ms per acf call the probe measured without cProfile at T=1024); caching the P vectors and keeping the per-pair `np.dot` gives the same windows from 118 acf calls (~0.05 s). ~100x for the infrastructure unit's own cost; unverified-bit-identical.

### Refuted or narrowed seed items

- **DCE window ACF once per channel, as a unit-level avenue for the DCE SPIs:** the ACF work (`utils.acf`) is 0.45% of the profiled mi_DCE call (0.023 s of 5.13 s; `_set_theiler_window` is 3.8% cum, but 3.3% of that is the frame's own self time, not the ACF, and of the size of the single stalls that land on the first Java call of a pair; that self time is large in 2 of the 5 DCE profiles (also te DCE auto at 8x256, 0.72%) and 0.06-0.08% in the other three; without cProfile they are rare in mi/tlmi/te DCE k-1/k-2 and 0.44-1.26 s per SPI call in te DCE auto, `probe_theiler_window_calls.json`, see it_kraskov-3) and about 2% of the quiet mi_DCE wall at 8x1024 (1.09 s); at 118x1024 ~5.7 s of acf against 270 s projected for mi_DCE is also ~2%. data.theiler is built once per Data and shared by all six DCE SPIs, so the avenue is refuted for those units and kept only for `infotheory.theiler_window` (it_kraskov-3).
- **NUM_THREADS=1** is confirmed as a CPU/throughput measure; as a single-process wall gain it is load-dependent (1-8x for the MI units across three runs), absent for the fixed-embedding TE units and a slowdown for the auto-embedding TE units at T=1024, so it_kraskov-1 is **refuted as a per-window avenue for all five TE units (te k-1, te DCE k-1, te DCE k-2, te auto, te DCE auto)** and removed from their records (each keeps a comment above its `opportunities:` key): te k-1 at 8x1024 (`opp_it_kraskov-1.json`) gives single-thread 0.507 / 0.888 / 2.088 s in three runs against the quiet default wall of 0.91 s = 1.8x / 1.0x / 0.4x, ~1x; the DCE k-1/k-2 units use the same ConditionalMutualInfoCalculatorMultiVariateKraskov path and were not run. The lower CPU per call (0.47-3.47 s single-thread vs 3.2-7.8 s default) matters only for throughput with windows in separate processes (xc-window-parallel). For te auto and te DCE auto the 3.3-4.6x holds at the labelled 3x256 shape only. At 3x1024 (`opp_it_kraskov-1.json` `shapes["3x1024"]`, this identifier for each) single-threading makes both slower: te auto 7.11 s vs 4.83 s same-run default = 0.68x, te DCE auto 7.46 s vs 4.69 s = 0.63x, and 0.58x / 0.57x against the quiet defaults (summary.json 8x1024 fresh 38.75 / 40.0 s scaled to 6 pairs = 4.15 / 4.28 s). The default wall is overhead-bound (T exponents 0.09 / 0.13 between 8x256 and 8x1024) while the single-thread cost grows with T. The lower CPU per call (7.6 / 8.0 s single-thread vs 18.8 / 19.9 s default at 3x1024) is kept only as throughput with windows in separate processes (xc-window-parallel).
- **Auto-embedding once per channel** is confirmed for te auto (it_kraskov-2) and narrowed to a (channel, W) key for te DCE auto (it_kraskov-4).
- **Symmetric MI computed twice** (Undirected runs both orders; the unmirrored ordered matrix is exactly symmetric, check stage) belongs to the cross-cutting xc-undirected-upper and is not listed per unit.

### Shared versus incremental cost

Only `data.theiler` is shared (one window matrix per Data for all DCE SPIs). Every other Kraskov SPI is incremental: family-warm walls equal fresh walls within noise (e.g. mi 0.90 vs 1.02 s, te auto 39.9 vs 38.8 s at 8x1024). No Kraskov SPI is in the retained EEG selection.
