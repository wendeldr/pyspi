# Storage integrity implementation results

Completed 2026-10-13. This implements the agreed [integrity contract](INTEGRITY_PLAN.md) for R1 in [the review](../../storage_comparison/REVIEW.md). The implementation remains a candidate in `storage_comparison/`. No patient store, scientific calculation, packing policy, or production pyspi dependency was changed.

Follow-up 2026-10-14: the agreed window-selection and shard-size behaviors are now implemented. The current suite has **381 passes and two expected failures** confined to experimental tolerance packing. The original integrity measurements below retain their 2026-10-13 context; the final section records the follow-up.

## Behavior

The new `pspi-store/0.2` format adds `pspi-manifest.json` beside the root Zarr metadata. It declares requested metrics and their decoding layouts independently of the metric directories. A batch registers its complete supplied metric inventory before writing its first result. Each metric moves through `pending`, `writing`, and `complete`; completion follows payload writes, required chunk-presence checks, and file/directory flushes. Manifest publication uses a flushed sibling file and atomic replacement. Write errors propagate.

Ordinary reads still return an array. Availability is optional:

```python
values = results.load("metric", windows=[0, 5, 10])
values, available = results.load(
    "metric", windows=[0, 5, 10], return_availability=True
)
```

Both arrays have the same requested shape and order. Missing or unreadable entries become NaN with availability False and one identifying `PartialReadWarning`. Saved calculation NaNs and infinities have availability True and do not warn. Data-only reads avoid allocating a full matrix availability mask. Recoverable entries retain their original bits.

Recovery respects dependencies: losing a residual affects only the strict lower triangle, while losing an upper payload also affects its mirrored/dependent entries. An invalid numeric chunk loses that chunk; an invalid shard index loses that shard. Healthy selections avoid unrelated damaged chunks. Empty selections request no payload and do not warn about unrequested damage.

The reader uses Zarr's codecs and checksums. A small CRC-checked shard-index check prevents absent inner chunks from silently turning into fill zeros; Zarr's strict-read setting alone does not detect that case. Out-of-shape padded slots are ignored. A healthy read returns Zarr's decoded block directly, avoiding an extra payload copy. Metadata is checked against the independent declaration; readable values can survive damage to the root/metric header, with an identifying warning.

`verify()` enumerates declared metrics, so deleting a directory cannot remove it from verification. It checks completion, metadata, required chunks, and decoded checksums without reconstructing every NxN matrix. Its CLI exits nonzero for problems. Read/verify never modify outputs. The temporary mean helper uses the checked reader and rejects partial input with a warning and exception; its numerical algorithm is unchanged.

## Validation

The isolated suite has **273 passes and seven strict expected failures**, in approximately 22 seconds with coverage enabled. All nine original R1 cases now pass. The original storage script separately passed **115 bit-exact synthetic round trips**, including full/subset/scalar reads and its existing mean cases.

Fault tests cover both codecs, both dtypes, every representation, missing inner chunks with valid rechecksummed indexes, padded slots, missing outer shards, payload/index corruption, truncation, invalid offsets, deleted metrics, damaged headers, malformed declarations, unsafe path names, and unsupported formats. They assert exact survivor bits, masks, shape/order, one warning per affected call, unchanged outputs, and propagation of programming errors.

Write fault tests cover packing, array layout, partial residual writes, payload writing, data flush, completion publication, and publication-directory flush, including overwrites. An interrupted three-metric batch retains `complete`, `writing`, and `pending` entries as appropriate. A deliberately removed inner chunk after payload writing prevents completion.

| Module | Statements | Branches |
| --- | ---: | ---: |
| Packing | 104/104 | 24/24 |
| Reader/writer | 310/315 | 111/114 |
| Integrity helper | 248/254 | 88/94 |

Combined statement/branch coverage is approximately 98%; no code is excluded. The seven deferred R4/R5/R6 cases were also run with expected-failure handling disabled: all seven failed for their recorded assertions. The independent storage CI workflow is configured but has not run remotely.

## Bounded healthy-path timings

Compared against `pspi_store.py` from commit `ae55c26949d1144337e804a5ec1aaf7dd432cda3`, using the same current codec environment and unchanged packing code. The baseline writer received an external file/directory flush because it did not flush itself. Timings include packing, adaptive representation selection, encoding, writing, and durability work; they exclude synthetic input generation and EEG calculations.

Three repetitions used alternating before/after order on an AMD Ryzen Threadripper 3970X, 64 logical CPUs, Python 3.12.3, NumPy 2.5.4, Zarr 3.4.1, uv 0.8.17. Stores were in `/tmp` on an ext4 NVMe filesystem, separate from the patient-data filesystem. Reads used a warm OS cache. Full, symmetric, and residual forms were tested with pcodec and Zstd at 40×40×64, 118×118×96, 249×249×32, and 118×118×600.

Median milliseconds for the 600-window cases; each cell is **before → after**:

| Form | Codec | Durable write | Full read | Four selected windows |
| --- | --- | ---: | ---: | ---: |
| Full | pcodec | 1005.4 → 1027.2 | 124.1 → 113.5 | 7.8 → 8.6 |
| Full | Zstd | 869.5 → 886.1 | 124.2 → 119.6 | 10.3 → 10.8 |
| Symmetric | pcodec | 176.6 → 189.8 | 139.8 → 157.7 | 8.3 → 9.3 |
| Symmetric | Zstd | 157.4 → 169.6 | 142.4 → 160.7 | 9.5 → 11.6 |
| Residual | pcodec | 881.7 → 900.7 | 172.5 → 187.6 | 12.6 → 15.3 |
| Residual | Zstd | 941.6 → 961.3 | 173.4 → 189.7 | 13.4 → 15.7 |

Across all 24 cases, the median write differences ranged from −0.5 to +24.7 ms, full-read differences from −10.6 to +18.3 ms, and selected-read differences from −1.6 to +3.1 ms. Negative differences indicate measurement variation or changes to block handling, not an established integrity-related speedup. The extra declaration added 876–1584 bytes for a one-metric store; payload representation and compressed values were unchanged.

A separate batch with **185 tiny 4×4×7 metrics** exposed manifest growth without running a large conversion. Median durable batch-write time increased from **2.21 to 6.09 seconds**; file bytes increased from 374,383 to 528,712. This approximately four-second fixed cost matters for very small batches. It does not establish an end-to-end patient processing ratio. The batch remains fully readable and passes verification.

Raw samples and environment metadata are saved locally in [integrity_benchmark.json](.artifacts/integrity_benchmark.json), with console rows in `.artifacts/integrity_benchmark.log`. Reproduce with the locked test environment:

```bash
mkdir -p tests/storage/.artifacts
git show ae55c26949d1144337e804a5ec1aaf7dd432cda3:storage_comparison/pspi_store.py \
  > tests/storage/.artifacts/baseline_pspi_store.py
uv run --project tests/storage --locked python tests/storage/benchmark_integrity.py \
  --baseline tests/storage/.artifacts/baseline_pspi_store.py \
  --output tests/storage/.artifacts/integrity_benchmark.json
```

## Limits and remaining decisions

- The completion contract supports a local POSIX filesystem and one writer per recording. Concurrent-writer coordination, automatic retry/repair, and job scheduling are deferred. Tests inject I/O failures; they do not simulate physical power loss or a permanently unavailable filesystem during rollback.
- Format 0.2 requires its manifest. Older prototype stores without one raise an explicit unsupported-format error; they are not migrated or rewritten. The previous implementation remains in Git for reproducing old experiments. Existing patient Zarr stores were never converted by this piece.
- If the manifest cannot establish shape/decoding rules, reading raises an explicit metadata/format error. It does not guess an output shape. Recovering bytes beyond a failed checksum is outside scope.
- The isolated tools now require Python ≥3.12 and Zarr ≥3.4.1. Production pyspi support is unchanged; dependency choices must be settled during integration.
- At the initial 2026-10-13 checkpoint, R4 tolerance/infinity handling, R5 wide-window shard caps, and R6 selection coercion remained expected failures. R5/R6 are addressed by the 2026-10-14 follow-up below. The temporary mean helper's R3 signed-zero edge case remains. No mathematical definitions, signedness labels, roundoff-dropping policy, or residual selection strategy changed.

The integrity piece is complete. Production integration, relocating the mean helper, and archiving the comparison folder require their own agreed scope.

## Follow-up: selection and shard-size contract, 2026-10-14

The user selected exact residual preservation, integer-only window selection, and a shard-size target with a one-window minimum and warning. Those selection and sizing corrections are implemented in the candidate; production integration has not begun.

`Results.load(..., windows=...)` validates original elements before converting indices. Floats, booleans (including mixed integer/boolean lists), timestamps, and other noninteger values raise `TypeError`. Out-of-range and multidimensional selections raise `IndexError`. Large signed/unsigned integers are checked before int64 conversion. Valid integer indices/sequences retain negative indexing, shape, order, duplicates, and empty selections, including matching availability masks.

`max_shard_values` retains its existing argument name but is documented as an uncompressed stored-value target. Inner chunk length now also respects that target. If a single payload row exceeds it, the writer emits `ShardSizeWarning` with metric/array, values/bytes, and target, then stores one window per shard. No within-matrix split or channel-count limit is introduced. Empty payloads and payloads exactly at the target do not warn. Invalid size settings are rejected before modifying standalone or batch output. Promoting a size warning to an exception leaves the write incomplete and identifiable.

Validation: **381 passing tests, two strict expected failures**, approximately 29 seconds with coverage enabled. The four R6 expected failures are now ordinary passing tests; the R5 strict-cap test was replaced with checks of the selected target/warning behavior. Both codecs, dtypes, every packing mode, and integrity recovery pass. The original script also passed its 115 synthetic round trips during this follow-up. Coverage is 104/104 packing statements and 24/24 branches; 345/350 reader/writer statements and 129/132 branches; and 248/254 integrity-helper statements and 88/94 branches. Combined statement/branch coverage remains approximately 98%.

Only the optional lossy tolerance experiment retains expected failures (R4). Its algorithm, exact residual packing, scientific calculations, dependencies, the temporary mean helper, and existing patient stores were unchanged by this follow-up. Production integration and mean-helper separation remain the next discussion.
