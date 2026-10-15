# it_kozachenko family: define, trace and check

This family has eight units: six composed SPIs (`je`, `ce`, `cce`, `xme_k1`, `xme_k10`, `di`, `si`, all
with `estimator='kozachenko'`) plus the shared estimator `infotheory.est.kozachenko`. The records are in
`audit/records/infotheory.*.kozachenko*.yaml`.

Probe scripts (read-only) are in the session scratchpad under `it_kozachenko/probes/`:

| Probe | What it checks |
|---|---|
| `probe1.py` | properties, labels, determinism, KL reference |
| `probe2.py` | jar class reflection |
| `probe3.py` | each composition vs a from-scratch KL |
| `probe4.py` | DI bias, null behaviour |
| `probe5.py` | JIDT nearest-neighbour routines vs cKDTree |
| `probe6.py` | per-unit effect at T=4096, threshold scan |
| `probe7.py` | direction conventions on structured datasets |
| `probe8.py`* | DI bias: seed spread of the KL gap vs Gaussian plug-in B, T=1024 |
| `probe9.py`* | `je` near-tie values on shared_envelope rows 0–2 |
| `probe10.py`* | naive-NN routine: wrong distances only where rho > 1, value in (1, rho) |

\* These three are in `it_kozachenko_fix/` in the session scratchpad, not under `it_kozachenko/probes/`.
The noise-magnitude probes cited under Deviation 2 are in `cleanup/it_kozachenko/`.

The JIDT sources used are in `it_kozachenko/jidt_src/` (v1.6.1 tag) and `it_kozachenko/jidt_a9b6/` (commit
a9b6b3e77c, which matches the shipped jar).

The sections before `## Checks` are the trace stage; their probe numbers are small read-only confirmations of what the code does. The numeric check results, verdicts and scores are in `## Checks` and in the records.
Statements marked *candidate* or *hypothesis* in the trace sections are resolved in the records'
`candidate_resolutions` and in `## Checks`. The regression
goldens play no role here.
Probe scripts cited by scratchpad-relative paths (`<dir>/<file>`) are preserved in `audit/checks/probes/scratch_probes_2026-10-15.tar.xz` under the same relative paths; they were throwaway probes and may reference scratch-only paths.

Supplement line numbers: the briefing pack's line numbers are about 9 lines later than this checkout of
`supplement.txt`. I cite the actual heading lines: §1.4 at 405, §1.4.1 at 444, §1.4.2 at 451, §1.4.7 at 513,
§1.4.8 at 526 and §1.4.9 at 539. Page footers are at lines 438 (p.9), 486 (p.10), 535 (p.11) and 585 (p.12).

## Shared estimator: infotheory.est.kozachenko

**Documented.** Supplement §1.4, p.9, says only: "The Kozachenko–Leonenko technique (kozachenko) is a
nearest-neighbour approach that is suitable when measures can only be constructed using entropy or joint
entropy (not mutual information)". The supplement names JIDT v1.5; the GitBook names v1.6.1.

**Canonical.** Kozachenko & Leonenko (1987), *Probl. Inf. Transm.* 23(2):95–101, in the digamma form of
Kraskov, Stögbauer & Grassberger (2004), *PRE* 69:066138, Eq. (20), with k = 1:

Ĥ = −ψ(1) + ψ(N) + log c_d + (d/N) Σ log ε_i

Here ε_i is twice the nearest-neighbour distance and c_d is the volume of the unit-diameter ball. For the
Euclidean norm, c_d = π^{d/2}/Γ(1+d/2)/2^d.

**Implemented (trace).**

1. `JIDTBase.__init__` (`infotheory.py:28-36`) calls `_getcalc('entropy')` (`103-110`), which builds
   `EntropyCalculatorMultiVariateKozachenko`.
2. `_setup` (`84-93`) sets `BIAS_CORRECTION=false` and `NOISE_SEED=42`.
3. Data arrive z-scored with ddof=1 (`calculator.py:277` at HEAD, `data.py:193-195`).

The shipped jar is not the v1.6.1 tag:

- `version-1.6.1.txt` says 22/08/2023, but the jar entries are dated 2024-04-21.
- Reflection shows the class extends `EntropyCalculatorMultiVariateCommon`, which does not exist at the
  tag.
- The source matches commits 3d1f3fec86 and a9b6b3e77c (2024-04-20).

At that commit:

- **Noise.** The constructor sets noise to 1e-8 (Kozachenko `:72-76`). `Common.setProperty` honours
  `NOISE_SEED` (`:133-139`). `finaliseAddObservations` adds N(0, 1e-16) noise from `Random.setSeed(42)`
  on every call (`:200-211`). Results are deterministic (probe1: three repeat calls bit-identical).
- **NORMALISE.** It defaults to `true` (`Common:66`) but nothing reads it. probe3 gives
  H(10x) − H(x) = 2.302585092994038 = log 10, so the property is inert.
- **Distance.** The calculation uses Euclidean distances (`EuclideanUtils.computeMinEuclideanDistances`),
  not the max-norm. The `sdTerm` arithmetic gives log c_d for the Euclidean unit-diameter ball; I checked
  d = 1..5 by hand.
- **Digamma.** ψ(N) − ψ(1) is the exact harmonic number for N < 10000.

With noise disabled, JIDT equals a from-scratch scipy `cKDTree` KL estimate to 0.0 (d=1) and 3.6e-15
(d=2).

**Deviations.**

1. **Large; candidate third-party defect.** For more than 2000 observations JIDT switches
   (`EuclideanUtils.java:30,69-75` at the JIDT v1.6.1 tag, identical at a9b6b3e77c) to `computeMinEuclideanDistancesNaive`. Its early-exit test at line 96,
   `(thisDistance < minDistance)`, compares a running *squared* sum with an *unsquared* minimum. Once the
   minimum exceeds 1, a candidate is abandoned after a partial sum, and the partial square root is accepted
   as its distance. Both the abandonment and the acceptance need a running minimum above 1, so only points
   whose true nearest-neighbour distance is above 1 can be wrong, and a wrong value always lies between 1
   and the true distance. With d = 1 the loop always completes, so d = 1 is immune. probe10 found no
   exception over d ∈ {1,2,3,5,6,10}, N ∈ {2001, 4096}, scales 0.05, 1 and 10 and two seeds. At scale 0.05
   nothing is wrong even at d = 10; at scale 10, d = 2 has 159–184 wrong points. On z-scored data the
   number of points with distance > 1 rises with d and falls with N. That explains why je, ce and xme_k1
   (d ≤ 2) are unaffected in practice while the d ≥ 3 terms of cce, di, si and xme_k10 are hit. probe5
   (iid normal data, noise off) shows how many nearest-neighbour distances are wrong and the effect on the
   entropy:

   | d | N | wrong distances | Entropy: JIDT | Entropy: reference |
   |---|---|---|---|---|
   | 2 | 4096 | 0 / 4096 | | |
   | 3 | 4096 | 10 / 4096 | | |
   | 5 | 4096 | 188 / 4096 | | |
   | 6 | 4096 | 572 / 4096 | | |
   | 10 | 4096 | 3998 / 4096 | 9.82 | 14.19 |

   The switch is visible as a discontinuity at d = 10: N = 2000 gives 14.25 (the exact-search value, which agrees with the reference), while N = 2001 gives
   9.14. The fast path (N ≤ 2000) is exact. The code is unchanged since 2014.
2. **Tie-breaking noise.** The noise changes values by about 1e-8 to 5e-6 nat on continuous z-scored data
   in probe3 (T = 256). The d = 1 term varies more with the data: over 10 AR(1) seeds it reaches 9.5e-5 nat
   at T = 256 and 2.7e-4 nat at T = 1024, which accounts for the whole probe8 DI residual; d = 2..5 stay
   ≤ 2e-7 (`cleanup/it_kozachenko/probe_noise_T.py`, `probe_noise_d1_probe8data.py`).
   On tied data it decides the result: rounded data give −14.9 nat with noise and −inf without.
3. **Labels.** All kozachenko variants are labelled `linear` (`infotheory.py:53-54` else-branch). The
   documentation says `nonlinear`.

**Proposed checks.**

- A cKDTree KL reference for d ∈ {1,2,3,5,6,10} at all T.
- A direct comparison of the naive and fast routines against cKDTree around N = 2000.
- Analytic Gaussian entropies.
- The scale law H(cX) = H(X) + d log c, plus a determinism check.
- A ties demonstration.

## infotheory.je.kozachenko

**Documented.** §1.4.1, p.10: "quantifies the uncertainty over the paired observations"; keywords
undirected and nonlinear.

**Canonical.** h(X,Y) (Cover & Thomas 2006, §8.4), estimated by KL with d = 2.

**Implemented.** `_compute_joint_entropy` (`infotheory.py:159-177`) passes the concatenated (T, 2) z-scored
pair to the estimator. The result is cached at both [i,j] and [j,i], and `Undirected.multivariate` mirrors
the matrix. probe3 matches a from-scratch reference to 4e-15 with noise off and 4e-8 with noise on.

**Deviations.** Only at the estimator level. The d = 2 naive-distance effect is negligible: at T = 4096 the
value is 2.821, the same as the reference. Values can be negative (REVIEW M8). On near-tied data the value
is set by the noise: shared_envelope rows 0–2 have 181 distinct values to 1e-9, and every pair among them
gives −28.809 to 3 decimals (probe9: −28.8093911, −28.8093892 and −28.8093878 for pairs (0,1), (0,2) and
(1,2); they differ at the 1e-6 level through the noise layout).

**Checks.** cKDTree reference; Gaussian analytic value; a symmetry/direction check (exactly symmetric);
the near-tie datasets flagged as degenerate.

## infotheory.ce.kozachenko

**Documented.** §1.4.2, p.10: "uncertainty over the observations in y in the context of simultaneously
observing x"; keyword *undirected*.

**Canonical.** h(Y|X) = h(X,Y) − h(X).

**Implemented.** A[i,j] = H_KL(z_i, z_j) − H_KL(z_i) = H(x_j | x_i) (`infotheory.py:246-250`). probe3
matches the reference to ≤ 3.6e-15 with noise off.

**Deviations.**

- The class is Directed, but the documentation keyword is *undirected*. REVIEW M8 already documents the
  conditioning order; the keyword conflict is an additional candidate.
- Label `linear`.
- The two KL estimates are made separately, so there is no bias cancellation. This is a property of the
  construction.

**Direction.** A[i,j] = H(column | row). For the anm tanh pair, H(x3|x2) = 0.104 < H(x2|x3) = 0.311. For the
cubic pair the two cells are close and reversed, so the convention should be fixed by matching the exact
reference, not by the semantic ordering.

## infotheory.cce.kozachenko

**Documented.** §1.4.7, p.11: a sum of conditional entropies of y given the past of y and of x, with
"history length to 10".

**Canonical.** Kramer (1998): H(Y^n‖X^n) = Σ_i H(Y_i | Y^{i−1}, X^i).

**Implemented.** n = 5 (`infotheory.py:435`; `config.yaml` does not override it). For i = 1..5
(`439-454`), each term is:

- the target's past: `makeDelayEmbeddingVector(targ, i−1)[:-1]`. For i = 1 this is a (T+1)×0 array,
  checked in probe3;
- the source's past and present: `makeDelayEmbeddingVector(src, i)`;
- the target's present: `targ[i−1:]`;
- Term_i = H([Yf, Yp, Xp]) − H([Yp, Xp]).

Under stationarity this is Kramer's quantity truncated at n = 5. probe3 matches a from-scratch reference to
≤ 7e-15.

**Deviations.**

1. Documented n = 10, implemented n = 5 (candidate).
2. For T ≥ 2005 the naive-distance defect lowers the value sharply (probe6, T = 4096, noise off):

   | Dataset | pyspi | Reference |
   |---|---|---|
   | var2_bivariate | 4.379 | 6.218 |
   | iid_gaussian | 4.484 | 7.078 |

   At T = 1024 the two agree.
3. Label `linear`.

**Direction.** A[i,j] has source i and target j. On var2_bivariate (x0 drives x1) at T = 1024,
A[0,1] = 6.212 < A[1,0] = 6.710.

## infotheory.xme.kozachenko_k1 and infotheory.xme.kozachenko_k10

**Documented.** Nothing. There is no supplement section and no GitBook description; `table-of-spis.md`
line 73 only lists the name "Cross-Map Entropy".

**Canonical.** None found. The closest standard quantity is the lagged conditional entropy
H(Y_t | X_{t−1..t−k}). The name evokes convergent cross mapping (CCM), but nothing is cross-mapped.

**Implemented** (`infotheory.py:405-426`). A[i,j] = H_KL([source past, target present]) − H_KL(source past).

- **k = 1.** One lag, H(y_t | x_{t−1}). Matches the reference to ≤ 7e-15 and is unaffected by the T = 4096
  defect.
- **k = 10.** The columns are `src[k-1:-1]` plus `src[k-m:-m]` for m = 2..k−1, which gives lags 1..9: **9
  lags, not 10** (candidate off-by-one). probe3 shows the k = 10 design pairs target time 10 with source
  times 9..1, and k = 2 would also give a single lag.

The joint has d = 10. For T ≥ 2011 (more than 2000 rows) the defect applies:

| T | Rows | pyspi | Reference |
|---|---|---|---|
| 2010 | 2000 | 1.4637 | 1.4637 |
| 2011 | 2001 | −0.2084 | 1.4626 |
| 4096 (var2_bivariate) | 4086 | −0.048 | 1.329 |

**Direction.** A[i,j] = H(x_j(t) | x_i past). On delayed_noise, A[0,1] = −0.937 (x1 is x0 delayed by one
sample) against A[1,0] = 1.479.

## infotheory.di.kozachenko

**Documented.** §1.4.8, p.11: "difference between the conditional entropy of y given its own past, and
causally conditioned entropy … history length of 10".

**Canonical.** Massey (1990) and Kramer (1998): I(X^n→Y^n) = H(Y^n) − H(Y^n‖X^n) ≥ 0, with
H(Y^n) = Σ_k H(Y_k | Y^{k−1}).

**Implemented** (`infotheory.py:487-510`). A[i,j] = Σ_{k=1}^{5} H_KL(Y^{(k)})/k − CCE_5(src = i, targ = j).
The first term is a sum of per-symbol block entropies, **not** H(Y^5) (candidate). For a stationary target,
H_k/k does not increase with k, so Σ H_k/k ≥ H_n. The difference is a bias B = Σ H_k/k − H_n ≥ 0, which is
nonzero under independence whenever the target is autocorrelated. For a Gaussian AR(1) target with
coefficient φ:

B = 1.2833 × (−½ log(1 − φ²))

That is 0.185 nat at φ = 0.5 and 1.066 nat at φ = 0.9. On data, the gap between pyspi and the canonical
KL estimate is, sample by sample, the KL plug-in of B: Σ_k H_KL(Y^k)/k − H_KL(Y^5) on the same estimates
(probe8: agreement ≤ 2.7e-4 nat, the 1e-8 noise). At T = 1024 one realisation of that gap is noisy and
seed-dependent, so it is not a measurement of B. probe8 drew 10 seeds of independent AR(1) pairs at
T = 1024 and also computed the Gaussian plug-in of B (½ log det of the sample k-lag covariances):

| φ | KL gap, mean (sd) | KL gap, range | Gaussian plug-in B, mean (sd) | Population B |
|---|---|---|---|---|
| 0 | 0.040 (0.046) | −0.039 to 0.116 | 0.001 (0.001) | 0 |
| 0.5 | 0.200 (0.069) | 0.096 to 0.318 | 0.190 (0.020) | 0.185 |
| 0.9 | 1.036 (0.148) | 0.881 to 1.414 | 1.082 (0.081) | 1.066 |

The earlier single draws (probe4: 0.034, 0.322, 1.111) fall inside these ranges. The check stage should
quantify B with the Gaussian plug-in and test the KL gap only against the exact sample identity.

Other deviations:

- n = 5 instead of the documented 10.
- The naive-distance defect at T ≥ 2005. At T = 4096 on iid_gaussian, pyspi gives 2.614 against a
  reference of 0.028; independent white pairs jump from −0.07 at T = 1024 to 2.48 at T = 4096.
- Label `linear`.

The composition is estimator-agnostic, so di_gaussian and di_kernel presumably share the B term
(hypothesis for those families).

**Direction.** A[i,j] is DI from i to j. On var2_bivariate, A[0,1] = 0.614 > A[1,0] = 0.134. On
delayed_noise, A[0,1] = 5.76.

## infotheory.si.kozachenko

**Documented.** §1.4.9, p.12: the joint vs individual entropies, each conditioned on a one-step history.

**Canonical.** Ay (2015) and Barrett & Seth (2011): SI = Σ_k H(X^k_t | X^k_{t−1}) − H(X_t | X_{t−1}) ≥ 0.

**Implemented** (`infotheory.py:524-534`). τ = 1, individual minus joint (the canonical sign). It needs six
KL estimates in d = 1, 2 and 4. probe3 matches the reference to ≤ 1.3e-15.

**Deviations.** The formula is canonical; the problems are in the estimator:

- The null value is sign-varying. iid_gaussian pair (0,1) gives −0.31 (T = 64), −0.23 (256), +0.23 (1024)
  and +0.13 (4096).
- gauss_ar1_pair at T = 1024 gives 0.025, against a Gaussian population value of 0.0999 computed from the
  known A and Σ_E.
- The defect effect at T = 4096 is small: 0.168 vs 0.166.
- Label `linear`.

**Direction.** Symmetric. Direct calls in the two orders differ only at noise level.

## Pack hypotheses: outcome

- "KL estimator with max-norm, NORMALISE=true and 1e-8 noise (NOISE_SEED=42); output in nats":
  - **max-norm: refuted.** The estimator uses the Euclidean norm.
  - **NORMALISE=true: the property reads true but is inert.** Normalisation comes only from pyspi's
    z-scoring.
  - **1e-8 noise: confirmed.**
  - **NOISE_SEED=42: honoured by the shipped jar,** but it would not be by the v1.6.1 tag source. (Check stage: honoured on the Calculator and pickle paths only; a CalculatorFrame deep copy reads 'NONE', see "Copy (CalculatorFrame) path" under Checks.)
  - **nats: confirmed.**
- "Same composed quantities as the gaussian family": the compositions are written once in `infotheory.py`
  and only `_entropy_calc` changes. So the gaussian/kernel variants share the n = 5, xme off-by-one and DI
  Σ H_k/k constructions (hypothesis for those families). The naive-distance defect is specific to the KL
  calculator.

## Open questions

1. **Upstream status of the defect.** Is the `computeMinEuclideanDistancesNaive` defect known upstream, and
   is it fixed in any JIDT release? Do other JIDT classes on pyspi's path call it? My hypothesis is that
   only the Kozachenko calculator does, because the KSG classes use KdTree with max-norm.
2. **Jar provenance.** Which JIDT commit was the jar built from exactly? The evidence points at or after
   a9b6b3e77c (2024-04-20). Should the regression environment pin the tag or this build? With the tag
   source, NOISE_SEED would be ignored and outputs would not be reproducible at the 1e-6 level.
3. **Intent of the DI target term.** Was `Σ H(Y^k)/k` meant as an entropy rate H(Y^n)/n, or as
   H(Y^n) = Σ_k [H_k − H_{k−1}]? Neither matches the code, and the check stage should quantify both
   readings.
4. **History lengths.** Is n = 5 for cce/di (documented 10) and the 9-lag xme_k10 intentional?
5. **Datasets for check design.** logistic_maps row 1 (r = 3.5) is a period-4 orbit with 4 distinct
   values, and shared_envelope rows 0–2 repeat to within 1e-9 (181 distinct values). KL results on these
   are set by the noise, so they should not be used as quantitative references for this family;
   logistic_maps_bidir has no ties.
6. **T = 4096 in the check grid.** The check stage should not treat T = 4096 agreement or disagreement as
   estimator convergence for d ≥ 3. Every multi-dimensional unit crosses the N = 2000 switch there.

## Checks

**Setup.** The manifest `audit/checks/manifest/it_kozachenko.json` has 182 runs (166 from the first check pass plus 16 added after the adversarial review: the copy-path runs, the ce ordering and nonlinear_dependence runs, and the di block-term runs on white and AR(1) targets). References are in `audit/checks/refs/it_kozachenko.py` (comparison venv: numpy/scipy only, no JIDT, no pyspi); probes are in `audit/checks/probes/it_kozachenko_probes.py` (project venv); population helpers and simulators are imported from `audit/checks/refs/it_gaussian.py`. Results (`audit/checks/out/results/it_kozachenko.json`, re-run with `--force`): **163 pass, 19 fail, 0 without reference, no script errors**. The 16 added runs all pass, and the 19 failures are the same runs as before. Every failure is listed below with its cause; none comes from a reference error. The pyspi side took 291 s in total (182 runs, under machine load; longest cce CalculatorFrame on shared_envelope 21 s, di massey-identity on iid_gaussian T=1024 14 s, nn-routines 12 s); the references took about 4 min wall, mostly the Monte Carlo runs. T = 4096 is used for every unit, with P = 2 for cce/di/xme_k10/si (3–5 s per run); those runs document the > 2000-row defect and are not read as convergence.

**Three entropy back-ends.**

1. **kl (tolerance).** The canonical Kozachenko–Leonenko estimate, `_kernels_infotheory.entropy_kozachenko` (cKDTree, Euclidean, Kraskov 2004 eq. 20, k = 1), without jitter. pyspi carries JIDT's 1e-8 jitter, so the criterion is a tolerance: atol = 5e-4 nat, rtol 0. Each kl run also stores the jitter residual measured on the reference side (`ref_jitter_residual_maxabs`, |jrand − kl|). Over the 30 kl runs at T ≤ 1024 its maximum is **2.7e-4 nat** (`si|logistic_maps_bidir|1024|pipeline|kl`), and 5e-4 is about twice that. At T = 4096 it reaches 1.1e-3 (`cce`/`di` on iid_gaussian), but those runs fail by about 2.6 nat for another reason.
2. **jrand (exact, rtol = atol = 1e-10).** The same canonical estimate on the input plus JIDT's jitter, reproduced exactly. JIDT adds `1e-8 * random.nextGaussian()` row-major to every observation with `java.util.Random` reseeded with 42 on every call (`EntropyCalculatorMultiVariateCommon.java` a9b6b3e7 :200-211). `java.util.Random` is a documented 48-bit LCG with a polar Box–Muller `nextGaussian`; `refs/it_kozachenko.py:JavaRandom` reimplements it. The column order of every JIDT call is mirrored, because the jitter layout depends on it: for je and ce the joint is `[z_max, z_min]`, because `Directed.multivariate` visits (i = b, j = a) first. This contradicted the check-stage kernel module note, which said the stream cannot be reproduced. Since its 2026-10-15 revision the note says the stream can be reproduced and names this port (see "Kernel issues" below).
3. **jidt (exact, same-lineage).** jrand plus a vectorised port of `EuclideanUtils.computeMinEuclideanDistancesNaive` for > 2000 rows. It is used only to show why the T = 4096 values differ. It is never used for scoring.

Every composed SPI also has a **noiseoff** run: the probe `spi_noiseoff` sets `NOISE_LEVEL_TO_ADD=0` on the SPI's own calculator, and the result is compared exactly with kl. This isolates the composition from the jitter.

**Statistical runs.** The reference value is the population value of the generating model. The se is the SD of a Monte Carlo (100–300 draws) of the canonical estimator of the same composition at the same T; the pass rule is |pyspi − population| ≤ 2 se. Each run also stores:

- the MC mean (population plus KL bias) and `z_vs_mc_mean`;
- `canonical_on_data`, the canonical estimate on the same data. When this equals pyspi and the run still fails, the failure is a property of that realisation (or of the estimator), not a pyspi deviation.

**Score policy (applied after review).** A composed unit is scored on its composition, using the formula anchor that the it_gaussian and it_kernel siblings use. The JIDT > 2000-row nearest-neighbour defect is in the shared estimator. It is scored once, on `infotheory.est.kozachenko`, and carried on cce, di, si and xme_k10 as an inherited finding (`what_changed`, `new_finding`) with no deduction. Path-dependent defects (the CalculatorFrame copy path below) are carried as findings and not scored, as in it_kernel. The first pass lowered cce (7 → 5), di (4 → 3) and xme_k10 (4 → 3) for the inherited defect but not si (2.2e-3 at T4096) or je/ce. That was inconsistent, so those three are now back at their formula anchors.

**Copy (CalculatorFrame) path, added after review.** `CalculatorFrame.init_from_list` deep-copies the base Calculator (`pyspi/calculator.py:556-559`, `copy.deepcopy(base_calc)` at line 559). `JIDTBase.__deepcopy__` (`pyspi/statistics/infotheory.py:77-82`) builds `type(self)()` and deep-copies every attribute without re-running `_setup`, and the JPype calculator proxy comes back in JIDT's default state.
- **Properties** (`infotheory.est.kozachenko|var2_bivariate|64|pipeline|copy-props`, pass): for all seven identifiers NOISE_SEED reads 42 on the original, 'NONE' after `copy.deepcopy` and 42 after a pickle round trip (`__setstate__` → `_getcalc` → `_setup`). NOISE_LEVEL_TO_ADD reads 1e-8 on all three. So the frame-path calculator still adds the 1e-8 jitter, but from an unseeded `Random`.
- **Values** (`<unit>|var2_bivariate|1024|pipeline|frame` for all seven SPIs, plus je and cce on shared_envelope T1024; all pass). In every run the Calculator and pickle slices equal the seeded replica (≤ 3.8e-13), and two separate CalculatorFrame objects differ from the Calculator and from each other:

  | Run | frame A − Calculator | frame B − Calculator | frame A − frame B |
  |---|---|---|---|
  | je var2 | 6.3e-8 | 7.7e-8 | 1.4e-8 |
  | ce var2 | 3.4e-4 | 1.9e-4 | 1.6e-4 |
  | cce var2 | 2.1e-4 | 4.7e-5 | 2.6e-4 |
  | xme_k1 var2 | 5.0e-4 | 5.1e-5 | 4.5e-4 |
  | xme_k10 var2 | 1.3e-8 | 6.8e-9 | 6.6e-9 |
  | di var2 | 3.0e-4 | 2.2e-4 | 1.7e-4 |
  | si var2 | 2.1e-4 | 1.1e-4 | 1.0e-4 |
  | je shared_envelope | 0.061 | 0.042 | 0.078 |
  | cce shared_envelope | 0.360 | 0.376 | 0.480 |

  (max |·| over the matrix). On continuous data the frame values remain KL estimates with another jitter draw (frame − canonical KL ≤ 3.6e-4, within the 5e-4 jitter allowance), so the effect is non-reproducibility at the d = 1 jitter level. On the near-tied shared_envelope rows the frame values differ by up to 0.48 nat between two runs on the same data. The reviewer's probe (scratchpad `verify_it_kozachenko_it_kernel/p_copy2.py`) found the same effect.
- **Finding.** CalculatorFrame outputs of every kozachenko SPI are not reproducible. The mechanism is the one it_kernel confirmed for KERNEL_WIDTH (`infotheory.est.kernel|bivariate_gaussian|256|pipeline|copy-props`). It is not in KNOWN_ISSUES.md, so it is recorded as a new finding candidate (confirmed) on all eight records.

### Shared estimator `infotheory.est.kozachenko`: wrong, score 1 (anchor 1: above 2000 rows a non-nearest-neighbour distance is substituted)

- **≤ 2000 rows: fp-exact.** With noise off, JIDT equals canonical KL to 7.1e-15 (`est|iid_gaussian|1024|kernel|direct-noiseoff`, d ∈ {1, 2, 3, 5, 6, 10}) and 4.4e-15 (`…|var2_bivariate|1024|kernel|direct-noiseoff`). With pyspi's properties it equals the jitter replica to 6.2e-15 (`…|direct-jrand`, iid T1024 and bivariate T256). The jitter itself moves d = 1 entropies by 1.1e-5 to 4.4e-5 nat and d ≥ 2 by ≤ 5.1e-8. Scale law H(cX) − H(X) = d ln c to 1.0e-13 (`…|scale-law`). Calls are bit-identical and equal the replica (`…|determinism`).
- **> 2000 rows: wrong nearest neighbours (confirmed defect).** On iid T4096 with noise off (`…|4096|kernel|direct-noiseoff` fails), pyspi − canonical is:

  | d | 1 | 2 | 3 | 5 | 6 | 10 |
  |---|---|---|---|---|---|---|
  | pyspi − canonical (nat) | 1.8e-14 | −2.1e-4 | −9.0e-4 | −0.038 | −0.157 | −4.334 |

  The port reproduces all six to 2.0e-14 (`…|direct-naive-noiseoff`). Called directly on 4087 rows (`…|nn-routines`), the naive routine returns wrong distances at 0 (d = 1), 1 (d = 2), 4 (d = 3), 176 (d = 5) and 3981 (d = 10) points. At scale 10, d = 2 gives 176 wrong points; at scale 0.05, d = 10 gives 0. Every wrong value lies in (1, true distance); the fast routine equals cKDTree; the dispatch switches between N = 2000 and 2001. The scale law fails above 2000 rows for d ≥ 2 (d = 5, c = 1000: −26.6 nat off, `…|scale-law-naive`).
- **Population d·½ ln(2πe)** (`…|truth`):
  - T256 passes (max_z 1.07).
  - T1024 fails at max_z 2.10 (d = 2: 2.9513 vs 2.8379, MC SD 0.054). Canonical KL on the same data equals pyspi, so this is the draw.
  - T4096 fails at max_z 89 (d = 10: 9.819 vs 14.189; MC mean 14.162). This is the defect, detected without the from-scratch code.
- **Properties** (`…|properties`):
  - NORMALISE reads true but is inert (H(10x) − H(x) = ln 10 to 1e-12).
  - NOISE_SEED = 42 is honoured; BIAS_CORRECTION is None.
  - The superclass is `EntropyCalculatorMultiVariateCommon` (post-v1.6.1 build, as in `infotheory.est.gaussian|var2_bivariate|64|kernel|jar-provenance`).
- **Ties.** A row rounded to 0.5 steps gives −14.668 with jitter (the replica gives the same) and −∞ without (`…|ties`).
- **Labels.** All seven identifiers carry 'linear', not 'nonlinear' (`…|labels`).
- **Copy path.** NOISE_SEED is honoured on the Calculator path and after a pickle round trip, but not after `copy.deepcopy` (`…|var2_bivariate|64|pipeline|copy-props`). The determinism result above therefore holds for the Calculator path only.
- **Verdict.** The verdict is wrong, score 1. In the > 2000-row regime JIDT replaces the nearest-neighbour distance with a different (partial-sum) distance, so the output is not the KL estimate. That is rubric anchor 1 (wrong estimator substituted). The first pass gave 3, which has no rubric basis: anchor 3 covers an ignored hyperparameter or a label that does not describe the computation. The ≤ 2000-row computation on its own would rate 9 (jitter convention). Under the score policy above, this is the only unit scored for the defect.
- **EEG reach (context, not scored).** No `*_kozachenko` identifier is in `audit/env/retained_metrics.txt`, and the default 0.5 s window (`docs/workflow.md:72`) gives more than 2000 rows only above 4 kHz, so the defect does not reach the stored default EEG run.

### je: matches, score 9 (anchor 9: canonical up to the jitter convention)

- **kl:** 5/5 pass, max_abs 1.3e-7. **jrand:** 6/6 exact (incl. `scaled_offset` kernel and `trending` pipeline-detrend), max_abs 1.2e-14. **noiseoff:** 0.0.
- **T4096 (var2):** 2.82078501 vs canonical 2.82078503; d = 2 escapes the defect.
- **Population 2H1 + ½ ln(1 − ρ²)** at T 256/1024/4096 for both pairs: 6/6 pass, max_z ≤ 1.82.
- **Symmetry:** max|A − A.T| = 0.0 on every run. The order probe is exact; direct calls in the two column orders differ by 4.2e-8.
- **Ties** (`je|shared_envelope|1024|pipeline|ties`). Rows 0–2 give −28.809 and shift by 9.2103 nat (= 2 ln 100) when the jitter goes from 1e-8 to 1e-6, i.e. the value is fixed by the jitter level. Pairs with row 3 move ≤ 7.8e-5.
- **`je|logistic_maps|1024|pipeline|ties` fails.** The pair moves by 3.9e-3 nat, above the 1e-3 I set before running. It is mildly jitter-sensitive, about 2400 times (three to four orders of magnitude) less than shared_envelope. The trace expected the logistic_maps pairs to move by many nats, and that expectation is refuted.
- **Copy path** (`je|var2_bivariate|1024|pipeline|frame`, `je|shared_envelope|1024|pipeline|frame`, both pass). The frame slices differ from the Calculator by ≤ 7.7e-8 on var2 and by 0.061 on shared_envelope (A[0,1] −28.8094 vs −28.7913 / −28.7975).
- **Kernel path on scaled_offset** (`je|scaled_offset|1024|kernel|shift-d2`). pyspi minus the unscaled D2 value equals canonical KL on raw rows to 5.5e-8, with shifts of 0.016–3.91 nat. A NORMALISE z-score would give 0, so NORMALISE is not applied. The analytic ln(s_i s_j) is matched only to 0.030, because KL is not equivariant under unequal per-axis scaling. My first reference used the analytic value as the target, which was a reference error; the canonical KL is now the target and the analytic value is stored.
- **Candidates:** 'linear' label confirmed. Jitter-set value on near-tied rows: confirmed for shared_envelope, while logistic_maps is only mildly sensitive (above). Copy path confirmed.

### ce: matches, score 9

- **kl:** 5/5, max_abs 1.4e-4 (the d = 1 marginal term). **jrand:** 5/5, 1.7e-14. **noiseoff:** 2.7e-15. **T4096:** 3.7e-5 (jitter only).
- **Population H1 + ½ ln(1 − ρ²):** 6/6 pass, max_z ≤ 1.27.
- **Orientation (anm):** A equals the H(z_j | z_i) replica to 8.9e-15 and differs from its transpose by 0.262. Tanh pair: A[2,3] = 0.104 < A[3,2] = 0.311. Cubic pair: 0.668 vs 0.615.
- **Ordering by ρ** (`ce|bivariate_gaussian|{256,1024,4096}|pipeline|rho-order`, all four rows, 3/3 pass). Both ρ = 0.8 cells lie below both ρ = 0.3 cells at every T: 0.899/0.937 vs 1.358/1.355 at T256, 0.940/0.999 vs 1.392/1.412 at T1024, and 0.935/0.925 vs 1.342/1.344 at T4096 (population 0.908 vs 1.372). Margins are 0.419 / 0.393 / 0.406 nat, and the replica matches to ≤ 3.3e-14.
- **Orientation on nonlinear_dependence** (`ce|nonlinear_dependence|1024|pipeline|direction`, pass). pyspi equals the H(z_j | z_i) replica to 2.8e-14 and differs from its transpose by 0.410. For x1 = x0² + 0.05e, A[0,1] = H(x1|x0) = −0.354 < A[1,0] = H(x0|x1) = −0.313 (margin 0.042). The circle pair is nearly symmetric (−4.0945 vs −4.0932).
- **Symmetry:** asymmetric even after z-scoring (0.023 on var2, 0.262 on anm, 0.410 on nonlinear_dependence), unlike ce_gaussian.
- **Copy path** (`ce|var2_bivariate|1024|pipeline|frame`, pass): the frames differ from the Calculator by ≤ 3.4e-4.
- **Candidates:** 'linear' label confirmed. The 'undirected' keyword vs asymmetric Directed values is confirmed, but as an elaboration of REVIEW M8 (ce = H(i,j) − H(i), the conditioning order) and M9 (label structure), as it_kernel records it. It is not a new finding of this family; ce's `new_finding` rests on the 'linear' label and the copy path. Copy path confirmed.

### xme_k1: related-quantity, score 5 (anchor 5: non-standard quantity computed as the code states)

- **kl:** 5/5, max_abs 1.8e-4. **jrand** + k2 + shift: exact ≤ 2.1e-14. **noiseoff:** 1.8e-15.
- **Population on delayed_noise:** 5/6 pass; T4096 pair 0-1 gives −0.919 vs −0.889. `xme_k1|delayed_noise|1024|pipeline|truth-p03` fails at max_z 2.01 (1.5446 vs H1 = 1.4189). Canonical KL on the same data equals pyspi and z vs MC is 2.13, so this is the draw.
- **history_length = 2:** equals the k = 2 replica and differs from k = 1 by ≤ 2.0e-3 (same lag set {1}; cross-estimator with `infotheory.xme.gaussian_k1|var2_bivariate|1024|pipeline|k2-numpy`).
- **Direction:** 2/2.
- **Copy path** (`xme_k1|var2_bivariate|1024|pipeline|frame`, pass): the frames differ from the Calculator by ≤ 5.0e-4.
- **Candidates:** documentation/naming confirmed; label confirmed; copy path confirmed.

### xme_k10: related-quantity, score 4 (anchor 4, related but different quantity: 9 lags under history_length = 10, applied off by one rather than ignored, so not anchor 3; as xme.gaussian_k10 and xme.kernel_k10; the defect regime is inherited, not deducted)

- **T ≤ 1024:** kl 4/4 pass (1.0e-8), jrand exact (2.8e-14), noiseoff 1.8e-14.
- **Lag set** (`xme_k10|iid_gaussian|1024|pipeline|shift-pattern`). For the lag-10 copy, pyspi gives 1.50506, equal to canonical KL on the 9-lag design (1.50506) and far from the 10-lag design (0.689). The lag-9 copy gives 0.640. My first version of this check used the Gaussian population sign (lag-9 value < 0). The Monte Carlo shows that KL in d = 10 cannot reach it, so the check was re-derived as a 9-lag vs 10-lag design comparison on pyspi's own sample (t = 10..T−1).
- **KL bias.** The population runs fail 0/4: canonical KL's MC mean is 0.607 (T1024) and 0.777 (T256) for a relation whose population value is −0.889, a bias of +1.5 to +1.7 nat in d = 10. pyspi is consistent with that distribution (z vs MC 1.05–2.59). This is an estimator property, not a composition error.
- **Defect.**
  - `xme_k10|iid_gaussian|4096|pipeline|threshold`: truncations T′ = 2008–2010 (≤ 2000 rows) equal canonical KL; T′ = 2011 gives −0.1976 vs 1.4489.
  - `xme_k10|var2_bivariate|4096|pipeline|kl` fails: −0.048/−0.113 vs 1.329/1.433; port 5.3e-15.
- **Direction:** delayed_noise 6/6 pairs, var2 1/1.
- **Copy path** (`xme_k10|var2_bivariate|1024|pipeline|frame`, pass): the frames differ from the Calculator by ≤ 1.3e-8 (all terms have d ≥ 2).
- **Candidates:** off-by-one confirmed; defect confirmed (inherited, scored on est); documentation confirmed; label confirmed; copy path confirmed.

### cce: variant, score 7 (anchor 7: the n = 5 variant, as cce_gaussian and cce_kernel; the T ≥ 2005 defect is inherited, not deducted)

- **T ≤ 1024:** kl 5/5 (1.4e-4), jrand 5/5 (3.7e-14), noiseoff 2.7e-14.
- **T4096** (`cce|var2_bivariate|4096|pipeline|kl` and `cce|iid_gaussian|4096|pipeline|kl` fail):
  - var2: 4.379/4.686 vs 6.218/6.653;
  - iid: 4.484/4.400 vs 7.078/7.007;
  - the port reproduces both to 1.4e-14.
- **Population on gauss_ar1_pair (5.813/5.563):**
  - T256 passes (max_z 1.08) and T1024 passes (0.58).
  - `cce|gauss_ar1_pair|4096|pipeline|truth` fails at max_z 13.2 (4.506/4.370; MC mean 5.847/5.610). This is the defect seen against an analytic value.
- **White pair:** 7.306 vs 5H1 = 7.095 (z 1.8).
- **n = 10 probe:** 12.97/13.95 vs n = 5 6.21/6.71; `configured-n` gives 5/5.
- **Direction:**
  - var2: 6.212 < 6.710;
  - delayed_noise: 5/5 pairs with a delay difference inside lags 0–4;
  - logistic_maps_bidir: A[0,1] = −15.28 < A[1,0] = −8.62, i.e. ordered by the target's own predictability, not by the coupling. This ordering is recorded, not asserted, because the record asked only for orientation on that dataset.
- **Copy path** (`cce|var2_bivariate|1024|pipeline|frame`, `cce|shared_envelope|1024|pipeline|frame`, both pass): the frames differ from the Calculator by ≤ 2.1e-4 on var2 (A[0,1] 6.211631 vs 6.211839 / 6.211638) and by 0.360 / 0.376 on shared_envelope. The two frames differ from each other by 0.480.
- **Candidates:** n = 5 confirmed (cross-estimator, as `infotheory.cce.gaussian|var2_bivariate|64|pipeline|configured-n`); defect confirmed (inherited, scored on est); label confirmed; copy path confirmed.

### di: related-quantity, score 4 (anchor 4: DI with block-entropy rate, as di_gaussian and di_kernel; the defect is inherited, not deducted)

- **T ≤ 1024:** kl 5/5 (1.9e-4), jrand 5/5 (4.3e-14), noiseoff 2.8e-14.
- **Block term (cross-estimator, confirmed for KL).**
  - `di|var2_bivariate|1024|pipeline|massey` fails by design: pyspi 0.614/0.134 vs Massey 0.449/−0.181.
  - `…|massey-identity`: pyspi = Massey + B(target) to 3.7e-14, with B_KL = 0.165 (target x1) and 0.315 (target x0). For comparison, the Gaussian plug-in gives 0.126/0.159 and the population 0.138/0.177 (closed-form side: `infotheory.di.gaussian|var2_bivariate|4096|pipeline|indep-truth`, B = 0.1766, and `infotheory.di.gaussian|var2_bivariate|1024|pipeline|massey`).
  - Independent random walks (`di|cointegrated|1024|pipeline|massey-identity`): B_KL = 2.52/2.77 against Massey 0.51/0.41.
  - White targets (`di|iid_gaussian|1024|pipeline|massey-identity`, five rows, 20 cells, pass). The identity holds to 7.1e-14. The gap pyspi − Massey lies in [−0.105, 0.070] (mean 0.009), so |gap| ≤ 0.15 as the trace expected. The Gaussian plug-in B is in [−0.0053, 0.0096], against a population B of 0.
  - AR(1) targets (`di|gauss_ar1_pair|1024|pipeline|massey-identity`, pass). The identity holds to 3.4e-14. B_KL = 0.331 (target 0) / 0.302 (target 1), the Gaussian plug-in gives 0.358 / 0.361, and the population B (`refs/it_gaussian.py:pop_di_excess_univariate`) is 0.351 / 0.305.
- **Independent white → AR(2) pair** (`di|var2_bivariate|1024|pipeline|indep-truth` fails at max_z 2.04). pyspi gives −0.112 against population B = 0.177. Canonical KL's MC mean is 0.057 (KL bias −0.12, SD 0.14), and pyspi is at z 1.35 from it. The Massey-form KL value on the same pair is −0.427. At T = 1024 the KL bias partly hides B on this pair; the identity run shows that B is present.
- **T4096** (`di|iid_gaussian|4096|pipeline|kl` and `di|var2_bivariate|4096|pipeline|kl` fail):
  - iid: 2.613/2.704 vs 0.028/0.106;
  - var2: 2.409/2.096 vs 0.574/0.133;
  - port 5.2e-14.
- **Population pyspi-form DI on gauss_ar1_pair:** 2/2 pass.
- **n = 10:** 0.579/−0.466 vs 0.614/0.134.
- **Direction:** 3/3 (logistic_maps_bidir: A[1,0] = 2.163 > A[0,1] = 0.674).
- **Copy path** (`di|var2_bivariate|1024|pipeline|frame`, pass): the frames differ from the Calculator by ≤ 3.0e-4.
- **Candidates:** all four confirmed (the defect is inherited and scored on est); copy path confirmed.

### si: matches, score 9 (anchor 9: within statistical tolerance)

- **kl:** 4/4 (2.7e-4). **jrand:** exact 4.1e-14. **noiseoff:** 7.1e-15.
- **T4096** (`si|var2_bivariate|4096|pipeline|kl` fails): 0.16774 vs 0.16557, i.e. 2.2e-3 from the d = 4 terms through the defect; port 3.2e-14. Small against the estimator's SD (0.058 at T4096).
- **Population 0.0999 on gauss_ar1_pair:** 3/3 pass. T1024 gives 0.025 with MC mean 0.123 and SD 0.118, so the trace's "large relative bias" is sampling variance, not bias.
- **bivariate_gaussian ρ = 0.8, T1024** (`si|bivariate_gaussian|1024|pipeline|truth-p23` fails at max_z 2.52): 0.802 vs 0.511. Canonical on the same data equals pyspi, so this is the draw.
- **Null on iid_gaussian:**
  - T256: mean −0.046, 5/10 pairs negative;
  - T1024: mean 0.088, 3/10 negative;
  - both pass against the MC distribution of the mean.
- **Symmetry:** 0.0.
- **T4096 defect.** The 2.2e-3 shift is inherited from est and is not deducted (score policy). It is small against the MC SD.
- **Copy path** (`si|var2_bivariate|1024|pipeline|frame`, pass): the frames differ from the Calculator by ≤ 2.1e-4.
- **Candidates.** The trace candidate 'nonzero, sign-varying null values and a large relative bias' is recorded as refuted, with the two parts split in the evidence:
  - the sign-varying nulls are observed, but they lie inside the canonical estimator's Monte Carlo distribution, so they are sampling variance and not a deviation;
  - the bias part is refuted (MC mean 0.123 vs population 0.0999, SD 0.118).
  - The 'linear' label and the copy path are confirmed.

### Failures (19), by cause

| Cause | Runs |
|---|---|
| > 2000-row NN defect (finding) | `cce` var2/iid T4096 kl, `cce` gauss_ar1 T4096 truth, `di` var2/iid T4096 kl, `xme_k10` var2 T4096 kl, `si` var2 T4096 kl, `est` iid T4096 direct-noiseoff and truth |
| Block-entropy DI, by design (finding) | `di` var2 T1024 massey |
| d = 10 KL bias for a near-deterministic relation (estimator property) | four `xme_k10` delayed_noise truth runs |
| Single realisation 2.0–2.5 SD from the population, with canonical KL on the same data equal to pyspi | `est` iid T1024 truth (2.10), `si` bivariate p23 T1024 (2.52), `xme_k1` delayed p03 T1024 (2.01), `di` indep-truth (2.04; z vs MC 1.35) |
| Mild jitter sensitivity above my pre-set 1e-3 | `je` logistic_maps ties (3.9e-3) |

### Corrections to trace-stage text

- The trace's "large relative bias" for si on gauss_ar1_pair is refuted. The MC mean of canonical KL is 0.123 against a population of 0.0999; the gap at T1024 is sampling variance (SD 0.118).
- The est record's jitter magnitude is extended with check-stage numbers (see the next list).
- The trace's open question 5 expected the logistic_maps pairs to move by many nats with the jitter. The measured move is 3.9e-3 nat, so this part is refuted (`je|logistic_maps|1024|pipeline|ties`).
- The est record's 'NOISE_SEED honoured' (provenance candidate) now carries the qualifier 'Calculator and pickle paths only'. A deepcopy reads 'NONE' (`est|var2_bivariate|64|pipeline|copy-props`).

### Trace-stage issues fixed

1. **Probe table.** The probe8–10 rows and their `it_kozachenko_fix/` location were already in the table (lines 18–23); I re-checked them and changed nothing.
2. **Stale citations.** The pre-a1f20c7 calculator.py line numbers (48 and 264; 264, 294 and 297) in all eight `implementation_trace` lists now point at HEAD: `:277` (Data(..., normalise=self._normalise)), `:329` (multivariate) and `:332` (fill_diagonal). The je/ce NOISE_SEED citation now names `infotheory.py:91`, the setProperty line. `audit/checks/citations.py` reports no stale, token-miss or out-of-range citations in the family's records and note.
3. **Jitter magnitude.** `infotheory.est.kozachenko` `deviation_from_canonical` (a) now ends with the check-stage jitter numbers:
   - d = 1: 1.1e-5 to 4.4e-5 nat;
   - composed SPIs: up to 2.7e-4 at T ≤ 1024 and 1.1e-3 at T = 4096;
   - d ≥ 2: ≤ 5.1e-8.

### Kernel issues (shared `_kernels_*` modules, not edited at the check stage; docstrings revised on 2026-10-15)

- **The JIDT jitter can be reproduced.** At the check stage `_kernels_infotheory.py` said JIDT's seeded jitter "cannot be reproduced in numpy". It can: `java.util.Random` is a specified LCG. My replica agrees with the JVM stream to 4.4e-16, and with JIDT entropies to ≤ 6e-15. Workaround: `refs/it_kozachenko.py:JavaRandom` plus a row-major jitter matrix, fed to `entropy_kozachenko(noise_level=0)`. The module note was revised on 2026-10-15. It now says "That stream CAN be reproduced" and gives this workaround (`refs/it_kozachenko.py:jitter` fills row-major; pass `noise_level=0`).
- **No JIDT nearest-neighbour dispatch in the kernel.** The kernel's `entropy_kozachenko` uses exact neighbours only, so it does not model JIDT above 2000 rows (for d ≥ 2). Workaround: the port `refs/it_kozachenko.py:nn_naive_jidt`, labelled same-lineage. Since 2026-10-15 the function's docstring states this limit in a SCOPE note and points to the port.
- **The task brief's premise about NORMALISE is wrong.** The task brief described JIDT's NORMALISE=true as z-scoring inside JIDT. The kernel docstring ("JIDT never normalises") and the runs both say otherwise.

## Open questions: answered

1. **Upstream status of the naive-routine defect.**
   - **Other users on pyspi's path:** none. Searching the class files of the shipped jar shows that only `EntropyCalculatorMultiVariateKozachenko` and `EuclideanUtils` itself reference `computeMinEuclideanDistances*`. The KSG classes reference `EuclideanUtils` only for norms, and search with `KdTree`/`NearestNeighbourSearcher`. So no other pyspi SPI hits the defect.
   - **Upstream master:** JIDT master `EuclideanUtils.java`, fetched 2026-10-15, still has `MAX_TIMESTEPS_FOR_FAST_DISTANCE = 2000` and the loop condition `(d < dimensions) && (thisDistance < minDistance)`, so the defect is unfixed upstream.
   - **Not settled:** I did not search JIDT's issue tracker.
2. **Jar provenance.**
   - **Exact commit:** not settled beyond "post-tag, at or after 3d1f3fe / a9b6b3e7" (`infotheory.est.gaussian|…|jar-provenance`; superclass seen again in `est|…|properties`).
   - **Pinning:** on the Calculator path, and after a pickle round trip, the shipped build's seeded jitter makes every kozachenko output reproducible to ≤ 6e-14 by an independent replica. The regression environment should therefore pin this jar, by checksum, rather than the v1.6.1 tag, whose unseeded `Random` would make outputs vary at the 1e-5 to 1e-4 nat level for d = 1 terms (the measured jitter effect).
   - **Pinning does not cover the CalculatorFrame path.** There the deep-copied calculator reads NOISE_SEED 'NONE' (`est|var2_bivariate|64|pipeline|copy-props`), so frame outputs vary between runs even with the pinned jar: by up to 5.0e-4 nat on var2_bivariate and up to 0.48 nat on the near-tied shared_envelope rows (`cce|shared_envelope|1024|pipeline|frame`). Reproducibility there needs a pyspi-side fix: re-run `_setup` after the copy, as `__setstate__` does.
3. **Intent of DI's target term.** On var2_bivariate T1024 (jitter replica, scratch `it_kozachenko_check/oq3.py`), for 0 → 1 / 1 → 0:

   | Reading | Target term | DI |
   |---|---|---|
   | pyspi Σ H_k/k | 6.826 / 6.844 | 0.614 / 0.134 |
   | Massey H(Y⁵), chain sum | 6.660 / 6.530 | 0.449 / −0.181 |
   | Joint H_KL(Y⁵) | 6.660 / 6.521 | 0.449 / −0.190 |
   | Rate H(Y⁵)/5 | 1.332 / 1.304 | −4.88 / −5.41 |

   The rate reading is incompatible with a block CCE and is almost certainly not intended. The code's name `entropy_rate_sum` and its comment ("entropy for an i-dimensional embedding") suggest the author meant to sum entropy rates. Only the Massey reading gives a DI that is zero for independent pairs. The code does not reveal the intent.
4. **n = 5 and 9 lags.** Both behaviours are confirmed (configured-n, n10, shift-pattern) and are estimator-independent (same results in it_gaussian). Whether they are intentional can only be settled by the authors; the code has no comment on either.
5. **Dataset suitability.** Confirmed for shared_envelope; refuted for logistic_maps, where the effect is three to four orders of magnitude smaller than the trace expected.
   - On shared_envelope rows 0–2, je is set by the jitter: it shifts by 2 ln 100 (9.2103 nat) for a 100× jitter change (`je|shared_envelope|1024|pipeline|ties`).
   - On logistic_maps, the trace expected the pairs to move by many nats. They move by 3.9e-3 nat (`je|logistic_maps|1024|pipeline|ties`, which fails its pre-set 1e-3 rule). The period-4 row is tied only in one dimension, and je in two dimensions stays nearly jitter-insensitive. So logistic_maps is only mildly unsuitable.
   - Quantitative and direction checks used logistic_maps_bidir, anm, var2 and delayed_noise instead.
6. **T = 4096.** Every T = 4096 run is reported as a defect run. je, ce and xme_k1 (d ≤ 2) agree with canonical KL at T = 4096 to jitter level (≤ 3.7e-5), and their T = 4096 population runs are valid convergence evidence. For cce, di, xme_k10, si and est with d ≥ 3, T = 4096 measures the defect, not convergence.

## Optimisation

Workflow 3 (group infotheory) is a short, profile-based search for where the time goes and which avenues look promising; nothing in pyspi was changed. Walls below are from `audit/profiles/summary.json` (quiet machine); percentages are shares of the profiled SPI call (`tests/regression/harness.py:compute_spi`) in the cProfile run at 8x1024 (`audit/profiles/<unit>__8x1024.pstats`). That profiling pass ran contended for the JIDT units (profiled walls 1.3-20x the quiet walls), so its walls are not used, only its shares. cProfile cannot see inside Java: JPype dispatches JIDT methods without profiler entries, so the Java time is booked as self time of the calling pyspi frame, and every `opaque_share` in these records (0.0002-0.04) is a lower bound on JIT/JNI time, not an estimate of it. `expected_speedup` is the unit's per-window speedup at 118x1024 as an order of magnitude; risks carry the `unverified-` prefix unless a micro-benchmark diff file backs them. Agreement with today's output means only that a candidate reproduces it.

### Walls

| Unit | 8x1024 fresh | 32x1024 fresh | 118x1024 fresh (measure_mode) | 8x1024 family-warm | 118x1024 family-warm projected (measure_mode) |
|---|---|---|---|---|---|
| `je.kozachenko` | 0.425 s | 6.64 s | 89.9 s (measured) | 0.401 s | 98.9 s (extrapolated-pairs-from-8) |
| `ce.kozachenko` | 0.532 s | 6.6 s | 81.2 s (measured) | 0.0452 s | 11.2 s (extrapolated-pairs-from-8) |
| `cce.kozachenko` | 12.2 s | - | 3.01e+03 s projected (extrapolated-pairs-from-8) | 11.4 s | 2.81e+03 s (extrapolated-pairs-from-8) |
| `xme.kozachenko_k1` | 1.77 s | - | 436 s projected (extrapolated-pairs-from-8) | 1.64 s | 404 s (extrapolated-pairs-from-8) |
| `xme.kozachenko_k10` | 2.54 s | - | 625 s projected (extrapolated-pairs-from-8) | 2.5 s | 616 s (extrapolated-pairs-from-8) |
| `di.kozachenko` | 16.7 s | - | 4.13e+03 s projected (extrapolated-pairs-from-8) | 4.31 s | 1.06e+03 s (extrapolated-pairs-from-8) |
| `si.kozachenko` | 5.58 s | - | 1.38e+03 s projected (extrapolated-pairs-from-8) | 4.99 s | 1.23e+03 s (extrapolated-pairs-from-8) |

`si`, `xme` (k1, k10), `cce` and `di` are in `infeasible_above_8x1024.txt` (pair-count projections from 8x1024). `infotheory.est.kozachenko` is infrastructure (filled from the je.kozachenko profile).

### Where the time goes

All time is inside JIDT's `EntropyCalculatorMultiVariateKozachenko`, booked as self time of the pyspi frame: je `_compute_joint_entropy` 99.8%, ce `_compute_joint_entropy` 67.7% + `_compute_entropy` 32.2%, si `_compute_conditional_entropy` 99.8%, xme `bivariate` 99.9% (k1 and k10), cce `_compute_conditional_entropy` 92.8% (+ `_compute_causal_entropy` self 6.3%), di `_compute_conditional_entropy` 67.3% (via cce 72.5%) + `_compute_entropy_rates` 27.4%. The nearest-neighbour search is pairwise (`EuclideanUtils.computeMinEuclideanDistances`, the fast routine for <= 2000 rows): measured T exponents 2.0-2.3 for d <= 4. The class is single-threaded (no `Thread`, `NUM_THREADS` or `availableProcessors` reference in the bundled jar).

### Opportunities

- **it_kozachenko-2 (replace-library; all units and est.kozachenko): kd-tree nearest neighbours.** A scipy cKDTree k=1 Euclidean query on the same normalised, jittered observations replaces N^2 = 1.05e6 distance evaluations per call at T=1024 with ~N log N queries; expected 5-20x for d <= 4 (je, ce, si, xme_k1, est) and 2-5x where the embedding reaches 9-10 dimensions (xme_k10, cce, di), where kd-trees lose most of their advantage. For di the kd-tree replaces both the conditional entropies (67.3%) and the entropy-rate calls (27.4%), 94.7% of the profiled call (Amdahl cap 19x); the conditional entropies alone would cap it at 3.1x. Risk unverified-within-tolerance: the check stage's cKDTree replica with the `java.util.Random(42)` jitter port reproduces JIDT to <= 1.2e-14 (je), 3.7e-14 (cce) and 2.1e-14 (xme_k1) at <= 2000 rows. Above 2000 rows JIDT switches to the naive routine whose distances differ (est.kozachenko finding), so there the change is semantic. Not measured.
- **it_kozachenko-1 (cache-sharing; si, xme_k1, xme_k10, di): per-channel entropy terms.** As it_kernel-1. Brute force costs N^2 d per call, so the dimension weighting gives ~2x (si), ~1.5x (xme_k1), ~1.9x (xme_k10); di's measured share is 27.4% (cap 1.38x fresh), and ~(P-1)x of the family-warm increment (1060 s projected) when cce runs first. Unverified-bit-identical: the jitter is re-seeded with NOISE_SEED 42 at every setObservations, so a cached call returns the same value. Not measured. The figures hold on today's JIDT cost: the records rank this avenue after it_kozachenko-2 (`depends_on`) with `residual_speedup` 1-2x (si), 1-1.5x (xme_k1), 1-1.9x (xme_k10) and 1-1.4x (di), lower bound 1x, because the kd-tree's gain differs with dimension and the per-channel terms' share of that residual is not known, so no saving is credited after it_kozachenko-2.

### Refuted seed items

- **NUM_THREADS and pair-level JVM parallelism:** the estimator is single-threaded; process-level parallelism is the cross-cutting xc-window-parallel.
- **Symmetric quantities once per unordered pair:** je already runs once per unordered pair; si.kozachenko is symmetric only up to the jitter layout (fresh bivariate(i,j) vs (j,i) differ by 3.1e-8, est.kozachenko check), so halving it under xc-undirected-upper would be within-tolerance, not bit-identical. That cross-cutting avenue is recorded `independent_of` it_kozachenko-1 and it_kozachenko-2 because they act on different factors (`audit/profiles/crosscutting.yaml`): it halves the number of bivariate calls, it_kozachenko-2 cuts the cost of each call, and it_kozachenko-1 moves the per-channel terms out of the pair loop; so it keeps its own 2x on their residual.
- **JNI share:** the 2-D conversion (~1.0 ms per call for every T x d shape, `opp_it_gaussian-1.json` `components.conversion_2d_by_shape`) is under 10% of every unit's quiet per-call cost at 118x1024: je 7.7% (one conversion per 13.0 ms unordered-pair call, 89.9 s measured), ce 8.5% (11.8 ms), si 6% (six per 99.7 ms), xme_k1 6.3% (two per 31.6 ms), xme_k10 4.4%, cce/di under 1% (218-299 ms per pair). Refuted as a unit-level avenue here (it is listed as it_gaussian-4 for the faster box-kernel units).
- **di entropy-rate cache** is confirmed (27.4%, it_kozachenko-1).

### Shared versus incremental cost

`data.joint_entropy` makes ce a lookup after je (family-warm 0.045 s against 0.53 s fresh at 8x1024) and `data.causal_entropy` makes di reuse cce (4.31 s against 16.7 s). si and xme are incremental. No Kozachenko SPI is in the retained EEG selection.
