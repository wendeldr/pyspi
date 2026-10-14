# Storage completion and partial recovery plan

Status: implemented and tested, following the agreed recovery behavior. This piece fixes the missing-data integrity problem R1 in [the review](../../storage_comparison/REVIEW.md). The candidate remains in `storage_comparison/`; the regression tests remain in the main repository. See [INTEGRITY_RESULTS.md](INTEGRITY_RESULTS.md) for validation and measured overhead.

## Agreed behavior

A successful write produces complete output. A failed or interrupted write reports failure and cannot mark its output complete. A partial read preserves the requested dimensions, window order, and every recoverable matrix entry. Unavailable entries become NaN, with separate availability information. Missing or unreadable requested data emits a warning. A NaN actually stored by the calculation remains available data and does not trigger that warning.

Copy verification stays with copying tools. This piece adds completion tracking and bounded data recovery, without a copy, repair, or job-runner framework. Matrix definitions, precision policy, residual selection, compression, shard-size policy, mean optimization, and production integration remain separate decisions.

## Reader interface

Keep the existing array-returning call:

```python
values = results.load(metric, windows=[0, 5, 10])
```

Add one explicit option for the availability array:

```python
values, available = results.load(
    metric, windows=[0, 5, 10], return_availability=True
)
```

`available` is boolean and has exactly the same shape and selected window order as `values`. It is True for entries reconstructed from successfully read data, including stored NaNs and infinities. It is False for entries whose required stored data is missing or fails validation. Scalar-window and single-matrix reads return matching two-dimensional arrays; empty and duplicate selections retain their existing shapes and ordering.

Data-only reads avoid allocating a full boolean matrix stack. Internally, unavailable chunk ranges are sufficient to fill the affected entries and assemble a warning. Materialize the full mask only when requested. Do not keep a mutable global or reader-level “last read status.”

Emit one `PartialReadWarning`, a `UserWarning` subclass, per affected load call, identifying the metric, original window ranges, matrix region, and missing/unreadable cause. Callers can use ordinary Python warning filters to treat it as an error. Healthy selected reads should not scan or warn about unrelated metrics or windows. Configuration/programming errors, unsupported formats, and invalid user selections remain exceptions rather than being converted into partial data.

## Minimal completion record

Add one small root JSON manifest for the local filesystem store. It records each requested metric's name, declared shape, dtype, packing mode/required arrays when known, and write state. Retain enough independent shape information to return an identified all-unavailable result if a declared metric directory is later deleted. Required array geometry can be checked against each array's Zarr metadata.

`write_results()` registers its provided metric set before beginning the writes, so an interrupted batch still identifies the requested metrics that were not reached. A standalone `write_metric()` registers that metric before writing it. The writer cannot declare metrics from an external configuration that was never provided to it; calculation orchestration will supply that scope when integrated later.

Write manifest updates through a sibling temporary file followed by `os.replace`, with appropriate file/directory flushing. Use a single writer per recording path for this piece, consistent with the current prototype; coordination of parallel writers belongs to later execution work. This avoids introducing lock management or multiple generations of a recording.

Metric states are `pending`, `writing`, and `complete`. The state changes to complete only after all required arrays have been written, required shard/inner-chunk presence has been checked, and files have been flushed. No second full decode of every value is required to publish a write: codec correctness is covered by the tests, while the explicit verifier remains available for a full checksum read.

An I/O exception propagates to the caller. The previously recorded pending/writing state remains incomplete, even if updating an error message would also fail. A crash after data is written but before completion is published remains identifiable as uncommitted output. Reading never marks or repairs it as complete. Existing per-metric directories can be retained; their existence alone no longer establishes completion. An explicitly requested overwrite is also marked incomplete before replacing data and must never leave a stale complete state after failure.

Version the new prototype completion contract explicitly. Existing patient Zarr stores and old comparison outputs remain untouched. Older prototype metadata cannot establish which wholly deleted metrics once existed. It must not silently receive the new completeness guarantee: require the new manifest for the new integrity-aware interface and give an explicit unsupported/legacy-format error when it is absent. Do not add automatic migration or a second compatibility reader in this piece; the previous prototype remains available in Git for experimental-output reproduction.

## Detect missing data before Zarr substitutes fill values

Reuse Zarr's codecs and CRC checks. With the installed Zarr 3.4.1, the public per-array configuration is:

```python
array = zarr.open_array(path, mode="r").with_config(
    {"read_missing_chunks": False}
)
```

A disposable-store probe confirmed that this raises `ChunkNotFoundError` for an absent shard. Passing `config=...` to `open_array()` does not configure an already existing array in this installed version, so the plan must not rely on that argument or change global Zarr settings.

Strict reads alone are insufficient for absent inner chunks. The [Zarr sharding specification](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html) represents those as empty entries in the shard index and allows them to read as fill values. Another probe, with a valid rechecksummed index containing an intentionally absent inner chunk, confirmed that even strict reads returned zeros. The prototype writes every in-bounds chunk explicitly, including zero-valued chunks, so absence is unexpected for this format.

Add a narrow check of required shard-index entries using the actual metadata, index encoding, and checksum. Check only entries intersecting the declared array shape and requested windows: padded slots beyond the final window are not missing data. Use existing checksum/codec facilities; do not implement an alternative numeric decoder. Cache each small index within the read operation. Unsupported index encodings raise an explicit format error rather than being guessed. This check is specific to the candidate writer's layout, not a general validator of every valid Zarr store.

Check that packing mode, required array names, dtype, shape, and chunk geometry agree with the declared metric. Inconsistent metadata is reported as an integrity issue. If sufficient validated metadata is unavailable to establish the output shape or decoding rules, raise a clear metadata error instead of inventing dimensions or guessing how to decode bytes.

## Recovery and reconstruction

Keep healthy reads batched, using the existing block-sized reads and Zarr's decoding. On a storage/decode failure, retry at inner-chunk boundaries to preserve other independently verifiable chunks. Missing required index entries can be identified before decoding. Catch storage and codec failures at their I/O boundaries; do not blanket-catch the entire load/reconstruction function.

The checksum is the recovery boundary: a failed chunk is unavailable as a unit. An unreadable shard index makes that shard unavailable because its chunk locations cannot be trusted. No byte scanning or repair is attempted.

| Representation | Availability dependencies |
| --- | --- |
| Full | Each matrix entry requires its full-payload chunk |
| Symmetric/antisymmetric triangle | Upper and mirrored lower entries require their shared upper-payload chunk; diagonal requires its own stored upper entry |
| Triangle plus residual | Upper entries and diagonal require the upper payload; lower entries require both the corresponding upper payload and residual |

Thus a missing residual preserves the upper triangle and diagonal while marking the affected lower triangle unavailable. A missing upper chunk also prevents exact reconstruction of its dependent lower entries, even if the residual survives. Never mirror approximately, assume a lost residual is zero, or apply an absolute value. Original values, signs, diagonal bits, and NaN payloads remain unchanged wherever data is available.

## Verification and the temporary mean helper

Keep `verify(path)` returning `{metric: diagnostic}` and its CLI returning a nonzero exit code for incomplete or unreadable output. Enumerate declared metrics from the manifest so deleted directories cannot disappear from verification. Check write state and metadata consistency, required shard/index entries, and decoded chunk checksums. Reuse the same read/integrity helpers as loading, without constructing all NxN outputs just to verify them.

The existing optional mean helper must not bypass these checks and calculate from substituted fill zeros. Its reads should fail clearly on unavailable input, with an identifying warning, while healthy behavior stays unchanged. Partial-data averaging remains the caller's explicit analysis operation using `np.nanmean(values, axis=2)` and availability information. Do not expand, relocate, or repair the helper's numerical shortcuts in this integrity piece.

## Tests and completion criteria

1. Turn the nine R1 expected failures into ordinary passing tests. The unrelated R4/R5/R6 expected failures remain visible.
2. Add tests for missing inner chunks with an otherwise valid shard index, including harmless padded slots; root manifest inventory and schema errors; and state transitions interrupted before/during/after data writing.
3. Assert full requested shape/order, NaN placeholders, and exact availability masks for full, triangle, and both residual modes, using both codecs. Include selected, duplicate, scalar, and empty window requests.
4. Assert that missing or unreadable requested data warns, saved NaNs do not, and a healthy selected window remains readable without warnings when other windows are damaged. Verify surviving entries bit-for-bit.
5. Assert that read/verify do not modify outputs, failed writes do not claim completion, and verification reports missing declared metrics. Keep source patient stores entirely outside the tests.
6. Run the focused suite and branch coverage. Make a bounded comparison of healthy full/subset reads and writes, including flush/completion overhead, so index checks and status tracking are assessed against the common case. Save measurements with the test artifacts; do not run EEG calculations or a cohort conversion.

After these checks, review the diff and measured overhead before selecting production integration or further storage work.
