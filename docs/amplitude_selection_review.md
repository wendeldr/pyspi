# Systematic amplitude-selection review

The review covered all **200 starting outputs**. After approved redundancy
removals, the main normalized selection contained **195 outputs**. The subsequent
[GraphicalLasso review](workflow_performance_review.md#graphical-lasso-run-removal)
removes seven more, leaving **188**. The alternative selects
**12 existing outputs**, using the same signals, reference, filters and 0.5-second
nonoverlapping windows, with **detrend=true, zscore=false**. This replaces the
earlier incomplete review of nine hand-selected outputs. Every starting main-run
identifier is accounted for in [the inventory](../audit/amplitude_review.csv).

Selection follows the requested distinction: omit unchanged outputs and simple
amplitude reweightings; retain interpretable changes beyond those, provided
they are numerically usable and inexpensive. This is a feature-selection
decision, not evidence that any candidate distinguishes EZ across patients.
No estimator formulas or parameters changed. The five main-run removals were
approved with a requirement to choose representatives by **standalone cost**,
without assuming other metrics will remain in the eventual final selection.

## Coverage of all 200 outputs

| Group | Outputs | Alternative decision |
| --- | ---: | --- |
| Scale-invariant calculations | 161 | Omit identical results under the same reference. |
| Empirical/robust covariance and shrunk covariance, ordinary/squared | 10 | Omit amplitude weighting; robust fits also fail at voltage scale. |
| Empirical/robust precision, ordinary/squared | 6 | Omit inverse-amplitude scaling or singular-matrix effects; robust fits also fail in volts. |
| Fixed GraphicalLasso, Lasso/ElasticNet and BayesianRidge | 6 | Omit unit-dependent regularization/prior effects and target-power weighting. |
| GraphicalLassoCV covariance/precision, ordinary/squared | 4 | Defer: unstable under equivalent units and CAR; expensive additional fit. |
| Signed-data Bray-Curtis | 1 | Defer: denominator can vanish for opposite signals. |
| Euclidean, cityblock, Chebyshev and Canberra distances | 4 | Include: relative channel amplitudes change pairwise differences. |
| Euclidean barycenter maximum and maximum of its square | 2 | Include: unequal amplitudes change cancellation and the largest joint excursion. |
| LedoitWolf, OAS and ShrunkCovariance precision, ordinary/squared | 6 | Include: relative variances change the regularized multichannel fit beyond rescaling. |
| **Total** | **200** | **12 included; 183 omitted; 5 deferred.** |

The 161 invariant outputs comprise 132 spectral outputs (all 12 selected
classes and all 11 bands/configurations), 12 cross-correlation summaries
(including significant-only selection), 3 rank correlations, 6 Gaussian/symbolic
information outputs, 6 PEC variants, RECI, and cosine distance. Positive channel
scales cancel through ranks, energy normalization, coherency, phase ratios,
entropy ratios, ordinal patterns, envelope correlations, min–max scaling or
cosine normalization respectively. Ordinary/squared and band variants were
checked individually; no selected identifier was inferred from an unchecked
class.

## Included outputs and their meaning

For detrended channel samples x and y:

| Output | Definition / extra information |
| --- | --- |
| Euclidean | sqrt(sum((x-y)^2)); large differences contribute more. |
| Cityblock | sum(abs(x-y)); total absolute difference. |
| Chebyshev | max(abs(x-y)); largest pointwise difference. |
| Canberra | sum(abs(x-y)/(abs(x)+abs(y))); relative differences, with 0/0 contributing zero. Common gain cancels. Sign disagreements contribute one; small samples can have substantial relative weight. |
| Barycenter maximum | max((x+y)/2); largest positive joint excursion. |
| Squared barycenter maximum | max(((x+y)/2)^2); largest joint excursion of either sign, squared. This differs from square(max((x+y)/2)). |
| Three ordinary precision outputs | Off-diagonal inverse of each fitted shrunk covariance matrix. |
| Three squared precision outputs | Square each fitted precision entry **before** averaging across windows. |

Distances and barycenter extrema are sensitive to signal excursions, including
artifacts and epileptiform events. Their inclusion does not label all large
excursions as useful connectivity. They are existing features evaluated under
the requested amplitude-preserving preprocessing. Euclidean/cityblock/Chebyshev
have voltage units; the squared barycenter has squared-voltage units. Canberra
is dimensionless. Use consistent physical units throughout the cohort.

Shrunk covariance has the form S* = (1-lambda) S + lambda mu I, where mu is the
average channel variance. ShrunkCovariance fixes lambda at 0.1; LedoitWolf and
OAS estimate it using their existing definitions. After z-scoring, the diagonal
variances are equal. Without z-scoring, shrinkage pulls unequal variances toward
a common level, so its proportional effect differs between channels. Inverting
this matrix changes the fitted relationships beyond multiplying each normalized
pair value by channel scale factors. This is a statistical regularization
assumption, not a measured physiological noise floor. See the
[estimator definitions](https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance).

These precision outputs are **not partial correlations**: they have inverse
squared-voltage units and are not bounded by [-1,1]. Squared versions have inverse
fourth-power units. The wrapper preserves the estimator's precision entries;
partial correlation would require a sign change and diagonal normalization.
This selection does not introduce that different definition.

In the eight-channel CAR probe, even after removing pairwise amplitude factors,
ordinary LedoitWolf/OAS/Shrunk precision differed from the normalized result by
39%, 40% and 55% in relative matrix norm respectively. Equivalent volts versus
microvolts inputs agreed after the expected output-unit conversion, with
relative errors below 5e-13 for all six outputs in the full-rank and CAR probes.
Thus the retained difference is more than a unit conversion or solver artifact.

## Why the other amplitude-sensitive outputs are omitted

**Covariance weighting (10).** Empirical covariance transforms as
C_raw[i,j] = sd[i] sd[j] C_normalized[i,j]. A well-defined affine-equivariant
robust fit has the same transformation. Shrunk covariance's off-diagonal is
simply (1-lambda) times empirical covariance; LedoitWolf/OAS may change that
window-wide multiplier when variances differ. These weights can change final
window averages. They are omitted under the user's preference against this
amplitude-weighted coupling view, not asserted to be redundant after averaging.
MinCovDet also rejected nonzero voltage-scale data as having zero covariance in
the probe: sklearn's support check uses an absolute `allclose(..., 0)` tolerance.
CAR additionally disrupts the robust search's usual full-rank assumptions.

**Empirical/robust precision (6).** For an invertible covariance, inverse
covariance transforms by the reciprocal pairwise scale factors. CAR introduces
an exact linear dependence, and the Moore-Penrose pseudoinverse generally does
not obey that transformation under unequal channel rescaling. The empirical
CAR probe differed by 5.4% after removing those factors. That is a choice of
singular-matrix representation, not newly established conditional connectivity.
MinCovDet and EllipticEnvelope also failed in volts in both probes. Their
same-seed fits agree; independent random searches must not be mistaken for
amplitude information. Existing audit evidence is in
[covariance/precision](../audit/notes/cov_prec_simple.md) and
[robust fits](../audit/notes/cov_prec_robust.md).

**Fixed penalties and Bayesian prior units (6).** All three selected fixed
GraphicalLasso outputs became zero off-diagonal in volts. Lasso(alpha=0.1) and
ElasticNet returned target variance, with no predictor-specific information.
BayesianRidge did not collapse, but its in-sample error carries target-power
weighting, and its fixed hyperpriors/tolerances introduce additional unit
dependence. It is not claimed to be an exact scaled copy: after output-unit
correction its volts/microvolts results differed by about 1.7e-4 and 6.8e-5 in
relative norm in the two probes. Those prior/solver changes are not a justified
amplitude feature. Choosing different penalties or priors would need a separate
definition/units decision; none were silently retuned.

**GraphicalLassoCV (4).** Unlike fixed-alpha GraphicalLasso, its ideal
cross-validated solution should respect a common unit conversion after alpha
and outputs are converted. The current solver failed on one CAR scaling and
gave materially different solutions at other scalings: ordinary CAR precision
differed by approximately 99% between volts and microvolts after output-unit
correction. It is therefore deferred regardless of cost. The prior audit also
measured about 5.28 s per 118-channel fit shared across all four outputs—roughly
53 minutes ×600 or 61 hours ×41,400. These are historical timings, not a new
raw-data benchmark; that unstable expensive fit was not repeated at full size.
See [the existing audit](../audit/notes/cov_prec_glasso.md) and
[upstream guidance](https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance).

**Bray-Curtis (1).** SciPy evaluates sum(abs(x-y))/sum(abs(x+y)). For y=-x the
denominator is zero; for y=-0.999x the value is 1999. Such values can dominate
means across windows. This existing signed-input issue is documented in the
[distance audit](../audit/notes/pdist.md) and follows directly from
[SciPy's definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html).
It is excluded from this alternative. Its existing main-run inclusion is left
for the separate signed-distance definition review; no new denominator or
absolute-signal transform was substituted here.

## Previously removed normalized redundancies

The six removals that reduced the previous 205-output selection, together with
the added Gaussian order-4 output, to 200 were also reconsidered:

- Ordinary barycenter mean stays zero with detrending/centering.
- Squared barycenter mean combines channel variances and raw covariance, so it
  adds the marginal-power/amplitude-weighting view already excluded above.
- Gaussian JE/CE add marginal log-standard-deviation terms to normalized MI;
  those terms remain separable after averaging windows.
- Both Ridge variants retain penalties calibrated for normalized inputs.
  Applying them in volts changes the regularization strength and produces
  target-power-dominated errors; reparameterizing them is a different decision.

None is restored to this alternative. Their library implementations remain
available. No additional algorithms were introduced.

## Fastest independent representatives in the main run

The user approved removing five redundant main-run outputs, choosing the
representatives independently rather than favoring cached work from another
metric. `tests/benchmark_redundant_representatives.py` measures fresh Data and
fresh fits for each output on 118-channel × 1,024-sample synthetic CAR windows,
one numerical thread, with preprocessing/imports/constructors/I/O excluded.
Fast alternatives have 11 alternating-order passes. Robust alternatives have
three matched-seed passes. Receipt: `tests/redundant_representative_timings.json`.

| Independent alternative | Median/window | ×600 | ×41,400 | Main decision |
| --- | ---: | ---: | ---: | --- |
| Cosine distance | **1.58 ms** | 0.95 s | 1.09 min | Keep |
| Ordinary empirical covariance | 4.92 ms | 2.95 s | 3.40 min | Remove |
| Ordinary ShrunkCovariance | 4.94 ms | 2.96 s | 3.41 min | Remove |
| Squared empirical covariance | **4.76 ms** | 2.85 s | 3.28 min | Keep |
| Squared ShrunkCovariance | 5.35 ms | 3.21 s | 3.69 min | Remove |
| MinCovDet precision | **2.92 s** | 29.2 min | 33.6 h | Keep ordinary/squared |
| EllipticEnvelope precision | 3.08 s | 30.8 min | 35.4 h | Remove ordinary/squared |

For centered windows of fixed length N and ddof=1 z-scoring,
C_empirical = (N-1)/N * (1 - cosine_distance). Off-diagonal ShrunkCovariance is
0.9*C_empirical; its squared output is 0.81*C_empirical^2. These affine
relationships survive averaging across windows. The 118-channel check matched
the identities within 1.5e-15. **Mean squared covariance is retained:** averaging
cosine alone does not determine the average of the squared relationships.

EllipticEnvelope runs MinCovDet's fit and computes an outlier threshold that
these precision outputs never use. The three matched-seed precision matrices
were bit-identical, so their squared matrices are identical as well. Independent
random searches can produce different estimates; that is not a different
estimator. Timings overlap, so the observed 5% difference is not a reliable
per-fit speedup claim. MinCovDet avoids the unused threshold step; eliminating
the second fit is the material saving. Both emit the existing CAR robust-fit
warnings, recorded in the receipt. This comparison does not resolve those
previously audited singular-data concerns.

These choices prioritize the requested future standalone selection. Ordinary
empirical covariance would be cheap if another metric had already fitted it,
but that was deliberately not used to choose the representative. The current
195-output run still needs several of these shared fits for other outputs;
do not add standalone timing differences to predict its total speedup.

Removed main identifiers: `cov_EmpiricalCovariance`, `cov_ShrunkCovariance`,
`cov-sq_ShrunkCovariance`, `prec_EllipticEnvelope`, `prec-sq_EllipticEnvelope`.
Their amplitude decisions remain in the 200-row inventory for traceability.

## Verification and runtime

`tests/review_amplitude_selection.py` resolved the starting 200-output config and
executes every output on eight synthetic channels × 1,024 samples, with both
full-rank and CAR inputs. Each case compares normalized inputs, unequal channel
amplitudes, volts and microvolts. Seeds are fixed for the scaling comparison;
warnings, failures, finite masks and differences are retained in
`tests/amplitude_full_review_results.json`. All 161 invariant outputs had the
same unavailable-entry masks and maximum finite absolute difference 3.34e-14;
each had at least one finite off-diagonal value in every control. Failures of excluded estimators are recorded
as failures, not counted as evidence of invariance or useful amplitude behavior.
The numerical receipt retains the starting config hash. To reproduce the same
inventory after the five removals, extract `a95e475:pyspi/workflow/reviewed.yaml`
to a temporary file and supply it with `--config`; the default audits the current
selection instead. The 200-output investigation was not repeated for removals.

`tests/benchmark_amplitude_selection.py` measures the entire 12-output set with
shared fits on synthetic 118-channel × 1,024-sample CAR windows. Five alternating
normalized/unnormalized passes follow an untimed warmup. Every output was finite
off-diagonal in all passes. Receipt: `tests/amplitude_full_selection_timings.json`.

| Workload | Median/window | ×600 | ×600×69 |
| --- | ---: | ---: | ---: |
| Same 12 outputs, normalized | 86.9 ms | 52.1 s | 59.9 min |
| Complete additional unnormalized set | **88.6 ms** | **53.1 s** | **61.1 min** |
| Four unnormalized distances | 46.4 ms | 27.9 s | 32.0 min |
| Two unnormalized barycenter extrema | 27.4 ms | 16.4 s | 18.9 min |
| Six unnormalized precision outputs | 14.8 ms | 8.9 s | 10.2 min |

Medians of separate groups need not sum exactly to the whole-family median.
The three squared precision outputs together add about 0.93 ms after their
ordinary fits; squared barycenter maximum adds about 0.45 ms after the shared
extrema calculation. Raw and normalized fits use separate Data objects, so the
88.6 ms is the full additional calculation cost, not a cache hit on normalized
fits. The normalized comparison measures these 12 outputs alone, not all 200.
These one-thread serial estimates exclude preprocessing, imports and I/O and
do not measure patient heterogeneity, clinical accuracy or parallel throughput.

Independent costs were also measured for every amplitude candidate using
`--standalone-only`, which appends results without repeating completed family
timings. Each output gets fresh Data and no other metric's cached fit. Medians:
Euclidean 0.94 ms, cityblock 16.89 ms, Chebyshev 9.54 ms, Canberra 18.71 ms;
barycenter maximum 24.16 ms and squared maximum 23.33 ms; ordinary/squared
LedoitWolf precision 5.01/5.04 ms, OAS 3.68/3.68 ms, ShrunkCovariance
3.87/3.84 ms. All standalone projections are in the receipt. Small differences
between paired variants reflect timing variation; both barycenter variants
perform the same shared extrema calculation when run alone.

The affected main-selection inventory test passed (one test); the final example
schema and both selections were checked against the full inventory. No unrelated
regressions or goldens were rerun or changed. The user's brief calculation-review
CSV is preserved. The per-output
inventory here is separate from that concise tracking CSV.

## Reference comparisons

Scale invariance assumes the reference and underlying signals stay the same.
CAR versus a local montage changes the signals themselves, so normalized
connectivity can change too. Compare a matched subset of the main normalized
selection under both references for that question. This amplitude selection
can also be compared under both references, with the same retained contacts
and windows, once a physically justified local mapping has been specified.
