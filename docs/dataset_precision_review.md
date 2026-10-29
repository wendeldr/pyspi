# Dataset integration and output precision results

The workflow now reads the built data-lake contact list, writes through
seeg-dataset, and separates recording/window/reference/run outputs. Matrices
remain raw per-window results. See [configuration and reading](dataset_output.md).

## Precision decision

Use **float32** for the working normalized, amplitude and candidate selections.
No float64 override is justified by the sampled results. This is bounded evidence,
not a guarantee about every future recording; overflow remains nonfatal and
recoverable from the diagnostic log.

The previous exceptions were GraphicalLassoCV covariance and squared covariance:
very small finite values flushed to zero in patients 105/111. These estimators
are no longer selected, following the earlier solver/runtime review. There was
no reason to rerun that completed investigation. Its warning about misleading
relative error near zero remains valid; see the
[historical precision review](storage_comparison/REVIEW.md#r7-float32-tables-do-not-measure-the-advertised-fallback-policy).

Current checks used three cached, real 118-channel × 1,024-sample EEG windows
from patient 001, CAR and linear detrending, with current estimators: all 188
reviewed outputs and all 12 unnormalized amplitude outputs. Workflow sampling-rate
resolution was applied, including the 12 cross-correlation variants. Their first
standalone-constructor pass omitted that rate; only those variants were recalculated.
Candidate outputs reused the completed three-patient (40/118/249 retained contacts),
CAR/Laplacian pilot through the documented identifier mapping, including the pilot
superset containing the selected candidates. No candidate estimators were rerun.

| Selection | Finite values checked | Maximum float32 relative conversion error | Overflow / underflow to zero |
|---|---:|---:|---:|
| Reviewed, 188 outputs | 7,538,460 | 5.955e-8 | 0 / 0 |
| Amplitude, 12 outputs | 497,016 | 5.955e-8 | 0 / 0 |
| Candidate pilot, both references | 4,226,178 | 5.959e-8 | 0 / 0 |

There were no nonzero sign reversals or finite-to-nonfinite changes in later time
means. Mean comparisons accumulated float64 on both sides; **the stored arrays
were never time-averaged**. Large inverse-covariance values have large absolute
rounding errors despite small relative errors: the amplitude sample reached
2.35e21, with maximum absolute conversion error 6.55e13. This is about the expected
float32 relative precision, not overflow or proof of an estimator error.

The report counts input NaNs/infinities independently of conversion events.
Established coefficient/angle bounds were checked for 142 reviewed outputs with
1e-12 roundoff allowance; none violated those bounds. Other estimators were not
assigned invented bounds. No values were clipped. This is a storage-precision
review, not a repeat of every estimator's scientific validation.

Float16 is supported but is **not** the working default. Some reviewed outputs
underflow/overflow, and the unnormalized precision family frequently overflows.
Original-value recovery logs make that case slower and larger than float64.

Per-metric errors, counts, mean comparisons and source/config hashes are in
[the precision receipt](../tests/precision_review/results.json).

## Actual storage measurements

Matched three-window payloads, 118 channels, actual bounded writer and shared
reader, including conversion, durable batch publication, diagnostics and disk
metadata. Each measurement ran in a fresh process. Three-window batches had two
repetitions in reverse dtype order; the table shows medians. Timings include
filesystem-cache/host variability and are not cohort throughput projections.

| Selection / dtype | Stored file bytes | Write seconds | Read seconds | Output peak RSS MiB |
|---|---:|---:|---:|---:|
| Reviewed float64 | 23,215,105 | 5.758 | 2.871 | 578.3 |
| Reviewed float32 | 11,316,410 | 5.291 | 2.963 | 535.2 |
| Reviewed float16 | 6,111,328 | 5.434 | 2.712 | 515.5 |
| Amplitude float64 | 1,850,775 | 0.337 | 0.089 | 387.8 |
| Amplitude float32 | 846,196 | 0.257 | 0.077 | 381.3 |
| Amplitude float16 | 14,372,193 | 1.556 | 0.061 | 410.5 |

Float32 reduced stored file sizes by **51% for the reviewed selection and 54% for amplitude**.
Sizes sum compressed payload and metadata file lengths; filesystem block allocation
and directory/inode overhead can differ.
Read time did not consistently improve; the writer still has per-metric metadata
and durability overhead. Float16 amplitude diagnostics alone occupied 14,183,400
bytes because they preserve the original overflowing values and positions.

Buffering matters more for write speed than dtype alone in this bounded sample.
For the reviewed float32 payload, one-window batches took 15.388 seconds for the
same three windows versus 5.291 seconds together, and used 12,389,022 versus
11,316,410 bytes. The one-window comparison had one repetition. Retain bounded
batching rather than writing each metric/window as a separate committed batch.
The target remains configurable; this three-window experiment does not establish
an optimal batch size for every recording.

RSS includes imports and cached input results: pre-conversion RSS was about 487 MiB
for the reviewed trial and 373 MiB for amplitude. The receipt also records whole-
process peaks through readback. These measurements cover output processes, not
full-record EEG preprocessing or all concurrent workers. No calculation speedup
is attributed to reduced storage precision.

See [timing and memory receipts](../tests/precision_review/storage_timings.json)
and [reproduction instructions](../tests/precision_review/README.md).

## Integration and verification

A temporary dataset containing a byte-identical copy of patient 001 metadata was
used for a real two-window, six-metric run (118 contacts, target 2048 Hz). Covariance,
Spearman, Gaussian Granger, coherence magnitude, PSI and PEC all succeeded. Each
loaded float32 value matched the explicitly converted calculation result bit-for-
bit, channel order matched the authoritative metadata, and original/copy metadata
remained unchanged. Elapsed time was 10.38 s including full-record preparation;
this limited selection is not a cohort estimate. The temporary output was removed.
[Integration receipt](../tests/precision_review/eeg_integration.json).

Focused coverage includes reference/preprocessing coexistence, conflicting
metadata order, missing labels, explicit bipolar mappings, local-reference masks,
float16/32/64 output, overflow value recovery, serial/spawn paths, partial reads,
interruption/retry/resume, and metadata preservation. The affected workflow suite
passed 264 cases (252 initially; 12 size-estimator cases passed after a targeted
fix). Additional metadata-edge checks passed 3 and fixed-dtype failure/retry checks
passed 11. Shared run API tests passed 6; the data-lake metadata suite passed 27. Shared
storage precision and adapter/recovery checks are recorded in
[the progress/evidence log](dataset_integration_progress.md).

Both working candidate examples passed read-only preflight against the actual
built metadata: CAR 10 metrics/118 usable contacts; Laplacian 9 metrics/92 usable
contacts on the same 118 saved axes. Existing recordings, patient metadata and
historical connectivity were not rewritten. The downstream localization checkout
named in the data-lake documentation is absent from this workspace, so no claim
is made about testing that separate application; the shared reader/API is tested
independently and the data lake has no connectivity consumer to update.
