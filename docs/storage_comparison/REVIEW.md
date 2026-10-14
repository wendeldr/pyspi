# Storage implementation and metric interpretation review

Reviewed 2026-10-13 against commit `ae55c26949d1144337e804a5ec1aaf7dd432cda3`.

**Current status:** production integration and the adversarial test review are complete. R1/R6 are fixed, R5 uses the agreed size-target/warning contract, and R3 is fixed in the separate analysis helper. Experimental R4 tolerance packing and precision conversion are excluded from production. The [storage suite](../../tests/storage/README.md) has 439 passing cases with no expected failures; [the test review](../../tests/TEST_REVIEW.md) records current calculation/storage verification. Scientific decisions remain separate, with review in progress and an audit due when complete. The original findings and prototype-stage follow-ups below are historical evidence, not the current implementation status.

At the original review checkpoint, the lossless triangle and XOR representation was sound on the tested inputs, and the measured float64 size savings had supporting data. **That prototype was not ready to adopt unchanged:** verification missed deleted data, some claims exceeded what was measured, and mathematical discrepancies needed individual decisions. No implementation, tests, labels, or existing patient outputs were changed during that review.

## Evidence and scope

Scope clarification (2026-10-14): this was a targeted investigation of suspicious matrix relationships, labels, and selected definitions, not a comprehensive mathematical review of all pyspi calculations. The calculation regression tests subsequently implemented and reviewed preserve existing behavior and detect changes; they do not certify that behavior as mathematically correct.

- The existing `test_pspi_store.py` passed its 115 generated round-trip cases, including subset reads, float64 means, both codecs, and a flipped-byte check. Additional disposable-store probes preserved arbitrary float32 and float64 bit patterns through both codecs.
- Targeted probes below reproduce defects absent from those cases. They ran without editing the implementation. Zarr's background event loop hung in the restricted sandbox; rerunning the storage probes outside it succeeded. Those sandbox attempts were stopped.
- Existing benchmark rows and summaries were checked, including the 555-array, 64-window codec screen and full conversions of patients 004 and 105. They were not all rerun.
- An independent read of four middle windows of all 185 metrics in each of patients 004, 105, and 111 checked actual signs, transpose relationships, missing labels, and all-NaN results. Seven relevant metrics per patient were additionally inspected over 16 middle windows. These reads decoded the existing Zstd chunk files directly and did not write to the source stores.
- Controlled calculation probes used this repository with `/home/dan/anaconda3/envs/pyspi/bin/python`: NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2, spectral-connectivity 1.1.2, and MNE 0.23.0. Relevant historical spectral source in `pyspi_1.1.1.1` also uses the all-frequency-pairs PSI implementation. This connects current behavior to a plausible historical dependency; it does not prove the exact environment of every original calculation.
- The complete original calculation suite was not run: importing `tests/test_SPIs.py` itself starts the full calculation benchmark. Its assertion function was examined and exercised independently instead.

## Implementation findings

### R1 Missing data passes verification and becomes zeros

**High priority before production use.** `pspi_store.py:109` uses `fill_value=0`. Zarr permits absent chunks and supplies the fill value. `verify()` at line 216 reads what is present without checking an inventory of expected shard files or completed metrics.

Reproduction: write a nonzero 2 × 2 × 3 metric, delete its data shard while retaining metadata, then call `verify()` and `load()`. Verification returns `{}` and loading returns twelve zeros. A deleted residual shard can similarly turn an asymmetric matrix into a mirrored one. Deleting a complete metric directory also removes it from verification's directory listing.

CRC32C protects the contents of chunks that exist. It does not establish completeness. An interrupted write or copy can therefore appear valid. The suggested use of a metric group's existence as a resume checkpoint is unsafe with the current implementation: metadata is created before its data is written (`pspi_store.py:103–113`). A production design needs a defined completion/inventory contract and tests that reject missing required data.

### R2 The reported write totals omit substantial required work

**Material performance reporting issue.** `bench_containers.py:215` times packing and residual selection separately, before the timed container write at line 227. `analyze_containers.py:73` defines the displayed write total as write plus fsync only.

| Patient | Packing and selection | Recommended write plus fsync | Combined | Recreated baseline write plus fsync |
| --- | ---: | ---: | ---: | ---: |
| 004 | 4.226 s | 6.717 s | 10.942 s | 130.094 s |
| 105 | 72.195 s | 37.370 s | 109.564 s | 217.848 s |

The format still looks useful. However, the 37-second figure is not the complete cost of taking patient 105's full matrices and storing them. The baseline also reconstructs a previously packed matrix inside its timed write, so this table does not establish a fair final production speed ratio either. Source loading, calculation, and a consistent direct-input baseline remain separate costs.

`pspi_store._residual_smaller()` compresses the complete upper triangle, complete residual, and complete full matrix to choose a representation; writing then encodes the selected representation again. This is valid for losslessness but expensive. The selection uses a whole-array proxy and default codec level rather than the eventual chunked encoding and requested level, so it also cannot guarantee the smallest on-disk representation. Selection strategy can be simplified independently of whether exact residuals are retained.

### R3 Antisymmetric means are not universally bit-identical

**Reproduced edge-case correctness defect.** `pspi_store.py:268–269` averages the upper series and mirrors that mean using a rule that leaves zeros unchanged. That rule is suitable for the stored zero convention, but it does not always describe the result of averaging a negated series.

For upper values `[smallest_positive_float64, 0, 0]` and lower values `[-smallest_positive_float64, 0, 0]`, loading is bit-exact. The true lower mean is `-0.0`; `Results.mean()` returns `+0.0`. The same shortcut exists in `containers.py:64–68`. This has negligible numerical impact for the EEG examples, but falsifies the unconditional bit-exact claim. Float32 means also deliberately accumulate as float64; the comparison must explicitly use `np.nanmean(x.astype(np.float64), axis=2)`, rather than NumPy's default float32 accumulation.

### R4 Tolerance packing mishandles infinities

**Reproduced defect in the optional lossy experiment, not the recommended exact writer.** `packing.analyse()` measures relative error only where both entries are finite. `pack(..., structure="tol")` then checks NaN positions but does not require matching finite/infinite status and infinity signs.

With upper `+inf`, lower `7.0`, and tolerance `1e-6`, packing selects `sym` and reconstructs the lower as `+inf`. The recorded finite-entry error is zero because that pair was excluded. Tolerance packing must not be reused for an approved roundoff-only policy in its current form.

### R5 The shard cap is conditional, not guaranteed

`pspi_store.py:106–109` splits the window dimension only. If a single row of packed matrix entries exceeds `max_shard_values`, the minimum shard still exceeds the limit. A small reproduction with a symmetric 5 × 5 × 1 input and cap 10 values creates a 15-value shard. The same issue scales to sufficiently large channel counts with the default cap. Current cohort matrices are below this case; the claim that the cap always handles arbitrarily large matrices is incorrect.

Decision agreed and implemented 2026-10-14: use a size target with a one-window minimum and an identifying `ShardSizeWarning` if one payload row exceeds that target. Permit the larger shard; do not split within a matrix or restrict channel count. The target measures uncompressed stored values, not an absolute compressed-file size. Chunk/shard sizing respects the target when a window fits. The original strict-cap expected failure was replaced by ordinary tests for the agreed contract, including both codecs, dtypes, every packing mode, target boundaries, empty payloads, and invalid configuration before writes.

### R6 Window selection silently coerces invalid input

`Results._arrays()` at line 169 casts selections to integer before validation. `windows=1.9` reads window 1; `True` also reads window 1; `[True, False, True]` reads `[1, 0, 1]` rather than applying a boolean mask. Empty selections, duplicate indices, reordered selections, and negative indices worked in the probes. A production API should explicitly define supported selection types and reject ambiguous inputs.

Decision agreed and implemented 2026-10-14: accept integer indices and one-dimensional integer selections, preserving negative indices, order, duplicates, and empty selections. Reject floats and booleans with a clear error, including boolean elements mixed into an otherwise integer selection. `windows` selects existing saved results; it does not specify duration or overlap. Validate original elements and bounds before conversion, so timestamps and unsigned overflow cannot select unintended windows. All four original R6 expected failures now pass as ordinary tests, with additional type, shape, and boundary checks.

### R7 Float32 tables do not measure the advertised fallback policy

The reference writer implements an opt-in float32 range guard. The screen and container benchmark instead call `astype(np.float32)` unconditionally (`screen.py:99`, `bench_containers.py:213`). Thus the reported float32 sizes and cohort projection describe direct casting, not the mixed float32/float64 policy described in the report.

The saved 64-window screen records finite values becoming zero, with maximum relative error 1.0:

| Patient | Metric | Values flushed to zero |
| --- | --- | ---: |
| 105 | `cov_GraphicalLassoCV` | 2,394 |
| 105 | `cov-sq_GraphicalLassoCV` | 3,622 |
| 111 | `cov_GraphicalLassoCV` | 1,814 |
| 111 | `cov-sq_GraphicalLassoCV` | 7,684 |

The guard would retain these arrays as float64, but that guarded size and error result has not been measured by this harness. The report's blanket per-value relative-error bound of `6e-8` is false for the direct-cast measurements. Float64 findings are unaffected. Float32 remains a separate scientific choice.

### R8 Storage mode names are being mistaken for scientific classifications

The XOR mechanism can encode **any** lower triangle relative to a symmetric or antisymmetric prediction. It does not prove approximate symmetry. In the actual saved rows, `tlmi_gaussian` selects `sym_res` for all three patients even though its directional difference is substantial. The report's “16 near-symmetric metrics” are actually 15 numerical-symmetry candidates plus this directed metric.

Likewise, “directed matrices store every entry” is not the implemented rule. A directed matrix may use a residual when that compresses better, and an antisymmetric directional matrix may need only a triangle. These are reasonable storage choices; the documentation's interpretation is wrong. All-NaN matrices also satisfy symmetry tests vacuously and cannot establish a metric's mathematical symmetry.

### R9 Compression diagnostics cannot identify scientific noise

Approximately balanced zero/one frequencies in low mantissa bits do not establish that those bits lack scientific information. Marginal bit balance also does not prove independence or an information-theoretic compression limit. `headroom.py` supplies comparisons among tested compressors, not evidence that values can be discarded. It examines the upper/full float payload and omits XOR residuals from its split-half comparison (`headroom.py:76–91`).

The report should distinguish poor compression by the tested methods from roundoff established through a metric's mathematics and implementation. Its statements about “noise” and exhausted lossless headroom are stronger than the evidence.

## Question 1 What the residual preserves and whether dropping it helps

For a symmetric prediction, the residual is the integer XOR of the stored lower float's bits and its mirrored upper float's bits. Reconstruction XORs it back. Antisymmetric prediction first flips the upper value's sign bit, with an explicit zero/NaN convention. The diagonal is stored directly. This is an ordinary reversible prediction technique, not an attempt to estimate or recover a hidden signal.

There are three distinct residual cases in the existing data:

| Case | Mathematical interpretation | Consequence of dropping the residual |
| --- | --- | --- |
| Twelve precision variants, `pdist_euclidean`, `pec`, `pec_log` | Fifteen mathematically symmetric metrics; reciprocal differences are consistent with numerical roundoff | Choosing one triangle can be scientifically defensible after a specific policy is agreed; it changes historical bits |
| Full-band `phase` | Phase branch behavior followed by arithmetic averaging, with a reproducible offset | Not a rounding residual; dropping it changes the existing band summary |
| `tlmi_gaussian` | Genuine lag-direction dependence | Dropping it destroys information |

The existing tolerance screen removes residuals for the first group while retaining the latter two. For float64 pcodec in window-major order:

| Patient | Extra compressed-byte savings over exact packing, 64 windows |
| --- | ---: |
| 004 | 402,708 bytes, 0.736% |
| 105 | 5,112,124 bytes, 0.862% |
| 111 | 18,787,436 bytes, 0.891% |
| Combined | 24,302,268 bytes, 0.882% |

These are codec-screen payload sizes, not a measured full-cohort migration. They support “less than 1% extra space savings,” rather than another factor of two.

Removing validated roundoff residuals also removes some encoding, decoding, and lower-triangle reconstruction work. The net production speed improvement has not been isolated. In the saved full patient-105 run, the fifteen candidates consume 24.257 seconds of packing/selection and 4.234 seconds of writing/fsync; those figures include work that could be reduced even while preserving the residual. They are not a measured speedup from dropping it.

**Decision confirmed 2026-10-14:** preserve exact historical values and retain residuals. The measured extra space savings are too small to justify discarding values. No roundoff-dropping policy is selected, and this decision does not require deciding whether particular residual bits contain scientific information. Float32 remains a separate decision.

## Question 2 Mathematical relationships and labels

### M1 Why the generic undirected class does not enforce these outputs

`Undirected.multivariate()` mirrors one triangle at `pyspi/base.py:132–138`. Spectral, covariance/precision, pairwise-distance, and envelope-correlation classes override that method. Python resolves their override first. `NonparametricSpectralMultivariate.multivariate()` at `spectral.py:85–102` aggregates the dependency's returned matrix and never calls the symmetry-enforcing base method. `Unsigned` only reports `issigned() == False`; it does not apply an absolute value. `Calculator.compute()` stores the returned matrix after replacing the diagonal with NaN.

### M2 Precision, Euclidean distance, and envelope correlation

Precision is an inverse or pseudoinverse of a symmetric covariance estimate, so its mathematical matrix is symmetric. The scikit-learn path uses SciPy `pinvh`; numerical matrix products can leave reciprocal entries different in their final bits. Controlled calls reproduced asymmetry in `Precision` without any storage layer.

Euclidean distance is symmetric by definition. Its dot-product implementation can be numerically asymmetric; [scikit-learn explicitly documents this](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html). Unorthogonalized envelope correlation is also a symmetric correlation; MNE's row-wise normalization performs its divisions in opposite orders for reciprocal entries. Controlled calls reproduced both effects.

In 16 saved middle windows, maximum absolute differences for these examples were approximately:

| Metric | 004 | 105 | 111 |
| --- | ---: | ---: | ---: |
| `prec_EmpiricalCovariance` | 1.14e-13 | 5.82e-11 | 1.46e-11 |
| `prec_ShrunkCovariance` | 2.22e-15 | 2.22e-15 | 4.00e-15 |
| `pec` | 3.22e-15 | 7.38e-15 | 3.03e-15 |
| `pec_log` | 5.55e-16 | 2.64e-15 | 6.11e-16 |
| `pdist_euclidean` | 7.11e-15 | 1.42e-14 | 1.42e-14 |

The larger relative discrepancies occur near zero. These are not independent directed measurements. An undirected label is appropriate; exact storage versus an explicitly canonicalized symmetric output remains a policy decision.

### M3 PLI and wPLI are signed variants here

The dependency returns `E[sign(Im(Sij))]` for PLI and `E[Im(Sij)] / E[abs(Im(Sij))]` for wPLI. Swapping channels conjugates the cross-spectrum and reverses the numerator's sign. The sign-flipped relationship is expected mathematics, not floating-point noise. Current wrapper calls reproduced negative antisymmetric outputs while their labels said unsigned/undirected.

The conventional unsigned versions take a magnitude. The [version-1.1.2 dependency source](https://spectral-connectivity.readthedocs.io/en/v1.1.2/_modules/spectral_connectivity/connectivity.html#Connectivity.phase_lag_index) explicitly distinguishes its signed PLI from the original unsigned version. For a band summary, `abs(mean(signed_measure))` and `mean(abs(signed_measure))` can differ because frequencies may have opposite lag signs. Existing band averages cannot generally recover the latter.

**Decision needed:** retain and explicitly identify the signed variants, or compute separately named conventional unsigned variants with an agreed frequency aggregation. Simply mirroring the same signed value into both triangles is incorrect. Antisymmetric packing retains their information exactly. Lead/lag sign is not by itself proof of causal influence.

### M4 Coherence phase and the full-band offset

Coherence phase is the argument of complex coherency. Conjugation reverses its angle modulo `2π`; the sign describes relative phase, so an unsigned label is inappropriate. At the branch cut, both directions may be represented by `+π`, which is equivalent to `-π` as an angle.

The pyspi wrapper takes an arithmetic mean of principal angles over selected frequencies. In version 1.1.2, the nonnegative-frequency slice includes DC and excludes the even-length FFT's Nyquist bin. A negative real DC cross-spectrum contributes `+π` in both directions, producing an apparent antisymmetry offset of `2π / number_of_bins`. A controlled nondegenerate signal pair reproduced the offset; excluding DC removed it in that probe.

Saved data confirms two offsets, not a universal `2π/512`:

- 105 and 111: `0.012271846303085... = 2π/512`.
- 004: `0.024543692606170... = 2π/256`.

Patient metadata records 1024 Hz for 004 and 2048 Hz for 105/111, consistent with 512 versus 1024 samples in a half-second window and those bin counts. This is an explanation of the observed offset, not a resampling change. The report's DC/Nyquist description is inaccurate for the inspected dependency.

**Decision needed:** preserve the historical arithmetic band summary, define a branch/endpoint convention, or introduce a circular/complex phase summary with a new identifier. A circular mean is a different band statistic, not a transparent storage fix. The offset residual must not be treated as numerical roundoff.

### M5 PSI has a definition discrepancy beyond its labels

PSI is a signed directional measure. Its antisymmetry is compatible with that purpose; signed/directed metadata is appropriate. The [original PSI paper, equation 3](https://arxiv.org/pdf/0712.2352), combines coherency at frequencies separated by a fixed frequency step. Its optional statistical normalization is a separate choice.

The inspected spectral-connectivity 1.1.2 implementation instead uses `_inner_combination()` at `connectivity.py:1107–1113`, summing **every unordered frequency pair** within the selected band. It is not the adjacent-frequency raw PSI formula. The historical local environment contains the same implementation. Relabeling it alone would not make its values standard PSI.

Evidence from the same four unit coherencies with phases `[0.6925794593, -0.3522172326, 2.7390760913, 2.2241494312]`:

- Adjacent-frequency formula: `-1.3070153772`.
- Dependency's all-pairs formula: `+1.1167942658`.

Thus even the sign can differ, not just a scale factor. On a controlled delayed-noise example, the actual pyspi wrapper returned approximately `616.6003`, while an adjacent-bin calculation from the same coherency gave `3.7563`. The [MNE PSI documentation](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.phase_slope_index.html) also identifies PSI as directional.

**Decision needed:** retain the historical all-pairs variant under an explicit identity for comparability, introduce the standard raw PSI, or replace the old definition after an agreed migration. Existing historical matrices alone cannot reconstruct the missing frequency-resolved coherency. No definition has been changed.

### M6 Group delay can be directional and antisymmetric

The dependency fits phase slope for one channel pair and fills the reverse slope with its negative; delay is slope divided by `2π`. This is expected directional lead/lag structure. It does not require storing two independent edge values. The signed delay/slope therefore conflicts with unsigned metadata, while directed metadata is reasonable.

In the four-window saved-data survey, every “symmetric directed group-delay” case was all-NaN. This is evidence of missing results, not evidence that valid delays are symmetric. The dependency's separate `rvalue` output is mirrored unchanged; it is a different returned statistic and should not inherit a delay interpretation automatically. The retained 185 include delay variants; diagnosing their failed bands is deferred.

### M7 Negative debiased estimates are valid

`dspli`, `dswpli`, and `ppc` are symmetric but can be negative because finite-sample bias correction subtracts a term. For example, the implemented dsPLI is `(K * PLI**2 - 1) / (K - 1)`, which is negative when the squared observed PLI is small. The PPC expression has the same type of subtraction. The saved sample contained negative values for all three families, and a controlled PPC calculation reproduced them.

These negatives do not encode an edge direction and should not be converted with `abs` or clipped as a label repair. Undirected is consistent with their matrix relationship. Metadata should distinguish an unsigned population target from the possible numerical range of its estimator.

### M8 Other signedness and metadata issues

- `pec` and `pec_log` can have negative correlations. The current `absolute=False` default and MNE path preserve those signs; the unsigned label does not describe their range. Orthogonalized absolute variants need their own parameter-specific metadata. This is separate from their tiny reciprocal roundoff.
- Gaussian joint and conditional entropy can be negative: differential entropy is not constrained to be nonnegative. The conditional calculation is asymmetric (`H(i,j) - H(i)`), so directed here describes conditioning order, not a causal assertion. The saved negative values are not evidence of an `abs` omission.
- `SpearmanR.__init__()` and `KendallTau.__init__()` mutate a shared class-level labels list with `+=` (`basic.py:147–149`, `171–173`). Instantiating squared and unsquared variants reproduced both signed and unsigned labels on the same shared list. Both Spearman arrays' saved metadata contain that contradiction. This is a concrete metadata-state bug, not an unusual scientific formulation.
- Current metadata for `gc_gaussian_k-1_kt-1_l-1_lt-1`, `xcorr_max_sig-False`, and `xcorr-sq_max_sig-False` lacks pyspi labels. Gaussian transfer entropy/Granger is directed in the current implementation; the cross-correlation summary is mirrored by its implementation. Missing metadata is not an undirected label. The older saved `symmetry_105_64.json` flags for Gaussian GC disagree with current metadata and should not be treated as current provenance.
- `tlmi_gaussian` compares lagged variables in an ordered pair. Its 16-window maximum reciprocal differences were 0.189, 0.656, and 0.671 for 004, 105, and 111. These are substantial, unlike roundoff. A `sym_res` storage mode does not make it undirected.

### M9 Label structure for discussion

Two distinct concepts would avoid a pseudo-undirected catch-all: the scientific interpretation of an edge and its matrix relationship. Candidate fields could describe `symmetric`, `antisymmetric`, or `general` relationships separately from signedness, estimator range, direction/conditioning meaning, parameters, and band aggregation. Undefined/all-NaN results need their own status. Storage may exploit redundancy in any of these scientific classes, but it should not assign their meaning from observed symmetry alone.

## Original recommendation: testing before implementation changes

The review of questions 1 and 2 can be completed without first rewriting the suite. **Before fixing storage, changing metric definitions, relabeling, or optimizing calculations, strengthening tests is the logical next implementation piece.** That can be focused rather than attempting a complete 185-metric scientific validation first.

The existing calculation benchmark is inadequate as a regression gate. `test_SPIs.py:71–72` replaces NaNs with zeros, and numerical disagreements end by calling a logger rather than failing. An isolated invocation with an all-NaN new matrix and a reference matrix filled with 100 returned successfully and merely logged the differences. The file also computes the entire benchmark during import, making small targeted runs expensive.

Proposed test priorities, pending discussion:

1. **Storage integrity:** missing shard, missing residual, deleted metric, truncated file/index, valid-JSON metadata corruption, interrupted write, completion state, overwrite refusal, and unchanged source inputs. Assert explicit detection and identification of unavailable data; partial-recovery behavior remains to be specified. A checksum test alone is insufficient.
2. **Numerical storage behavior:** exact bits for special values and NaN payloads, signed/subnormal zeros, float32 accumulation semantics, every packing mode, one channel/one window, noncontiguous inputs, multiple shards, cap boundaries, ordered/duplicate/empty selections, and rejected ambiguous selections. Add the reproduced anti-mean and infinity/tolerance cases.
3. **Independent mathematical oracles:** covariance symmetry; phase conjugation and branch endpoints; PLI/wPLI signed and unsigned definitions including the placement of `abs`; standard versus historical PSI; delayed-signal orientation; and legitimate negative bias-corrected estimates. Compare formulas from the same intermediate inputs so preprocessing and estimator differences cannot explain a mismatch.
4. **Integration and regression gates:** explicitly assert expected NaN masks, finite-result presence, parameter and channel order, detrend-then-z-score behavior, cache consistency across metric order, and independent label lists. Separate fast deterministic tests from slow, optional-dependency benchmarks. Make genuine discrepancies fail rather than merely printing a warning, while retaining per-metric tolerances where justified.

## Discussion queue at the prototype checkpoint

Status updated 2026-10-14: the integrity contract and exact residual retention are settled. The remaining scientific items are targeted questions, not a completed comprehensive mathematical review.

1. Completed: establish and implement the focused storage test and completeness contract. Production integration is pending.
2. Decided: retain exact residuals, including candidate reciprocal roundoff and the phase/TLMI differences. No tolerance-based dropping is selected.
3. Decide the intended PSI definition and historical comparability.
4. Decide signed versus conventional unsigned PLI/wPLI and band aggregation.
5. Decide coherence-phase aggregation and endpoint handling.
6. Decide parameter-specific labels, legitimate estimator negatives, and independent matrix-relationship metadata.
7. Reassess end-to-end write performance after the chosen representation and selection strategy are measured consistently. Float32 remains deferred unless explicitly selected.

The full expensive calculation/failure investigation, broader parallel execution work, and output migration remain outside this review.

## Historical follow-up discussion and scope

The user confirmed the distinction between stored calculation results (including calculated NaNs) and unavailable stored data. Agreed partial reads preserve requested shape and original window positions, retain recoverable entries even within a partially readable matrix, fill unavailable entries with NaN, and identify availability separately. Missing or unreadable requested data emits a warning; a NaN actually saved by a calculation does not.

Writes must report failure if interrupted or unsuccessful and must not mark incomplete output complete. Partial recovery must not conceal a writer defect. Copy verification remains the copying tool's responsibility; a general copy/repair framework is outside the proposed fix. The missing-file defect was reproduced by deliberately damaging temporary test stores, not discovered as corruption of an existing patient output. The recovery API and completion mechanism remain to be specified before implementation.

The user requires storage code to concern storage. The existing `Results.mean()` is an optional optimization, not a requirement for decoding the format. For a multiwindow metric, `np.nanmean(results.load(metric), axis=2)` already computes the ordinary time mean. The specialized helper works on packed arrays to avoid reconstructing every mirrored matrix entry before reducing time, and arranges the reduction to match a particular NumPy summation order. It still loads the packed arrays into memory; it is not a streaming time reduction.

The recommended scope is ordinary storage reads/writes, reconstruction, selection, integrity, and identification of unavailable data. Time averaging belongs in downstream analysis or a separate optional analysis helper, with its own tests if retained. Comparing analysis access costs remains useful benchmark evidence, but the specialized mean implementation and its edge-case defect need not become production storage responsibilities. Removing or relocating that helper would not change stored values or the format. No implementation change has been made.

The user accepted retaining the optimized mean temporarily for later relocation if worthwhile, and authorized focused main-repository tests before storage integration. The initial [storage regression suite](../../tests/storage/README.md) had 144 passing cases and 16 strict expected failures reproducing R1, R4, R5, and R6, before prototype changes. Subsequent discussion settled the [partial-recovery and completion contract](../../tests/storage/INTEGRITY_PLAN.md), and the user authorized its full implementation.

The initial integrity implementation completed with 273 passes and seven strict expected failures. The 2026-10-14 selection/shard-size follow-up brings the suite to 381 passes and two strict expected failures for experimental tolerance packing. Packing coverage remains 100%; reader/writer and integrity-helper statement coverage are approximately 99% and 98%. The original 115 synthetic round trips also passed. New-format metadata declares metric inventory and layouts, missing/unreadable reads warn and preserve recoverable entries with availability masks, and verification identifies incomplete/deleted output. The optional mean rejects unavailable input instead of reducing fill zeros. Timings and the deliberate legacy-format boundary are recorded in [INTEGRITY_RESULTS.md](../../tests/storage/INTEGRITY_RESULTS.md). No mathematical definitions, labels, precision policy, or existing patient outputs changed.
