# Lossless connectivity storage comparison

## Scope

Compare a bounded set of lossless representations of existing connectivity results. The output is a reproducible comparison of disk size, file counts, memory use, and read/write costs that can guide the later output writer and batching decisions.

This is the first piece from [the overall discussion](../EEG_OPTIMIZATION_NOTES.md). It uses already calculated matrices and preserves their float64 values. Calculation optimization, the runtime estimator, EEG preprocessing utilities, and difficult failure diagnosis remain later work.

Status (2026-10-13): comparison complete; results and recommendation are in [REPORT.md](REPORT.md). All benchmark code, environment files, reports, and generated data live under `storage_comparison/`. Existing patient stores were only read.

The work went beyond the plan below in these ways:
- The screen used 64 middle windows instead of 16.
- 004 and 105 were converted at full size in every container, plus a 10-metric subset of 111.
- float32 was evaluated as well as float64, at the user's request.
- Further native codecs (OpenZL, Vortex, DuckDB) and a lossless-headroom check were added.
- New tests: corruption, interrupted transfer, chunk size, threads, and peak memory.
- `pspi_store.py` is a reference reader/writer for the recommended format.

## Input sample

Read from `/media/dan/Data/outputs/calculated_seeg_patient_data` without modifying it.

| Case | Patient store | Channels from inspected metadata |
| --- | --- | ---: |
| Small | `004.zarr` | 40 |
| Typical | `105.zarr` | 131 |
| Large | `111.zarr` | 249 |

Initially sample 16 consecutive windows near the middle of each recording, covering all 185 saved metrics. This samples the distribution of metric values without loading full patient results or rerunning calculations. Verify shapes, dtypes, and codec settings during the initial inventory rather than assuming every store matches the inspected example.

Process one metric at a time. Patient selection and window selection should be configurable; these example sizes are not limits. Expand the sample only if the initial results are sensitive to the chosen windows or fail to represent an important case.

## Representations to compare

Use a small number of deliberate variants rather than an exhaustive codec search.

| Representation | Purpose |
| --- | --- |
| Existing full-matrix Zarr layout and Zstd settings | Establish the current compressed baseline on the same sample. |
| Full matrices with chunks spanning several windows | Test the effect of batching windows on size, file counts, and reading across time. |
| Full matrices with a time-first layout and lossless shuffle/bitshuffle compression | Test layout and floating-point compression opportunities while preserving every value. |
| Compact matrices in Zarr, with full-matrix fallback | Test exact symmetry packing with the same candidate codecs. |
| Bounded Zarr shards using the promising payload/layout | Measure the tradeoff between independently readable chunks, file count, and write cost. |
| Independent compressed NumPy blocks with a small index | Provide a simple portable alternative container, using matching payloads and compression settings where possible. |

Start with moderate compression settings. Keep codec, payload, and container effects identifiable in the report; compare matched settings where possible and label combined changes. Higher-cost settings can be considered if the initial results make them worthwhile.

For symmetry packing, store the diagonal and one triangle only when actual float64 bit patterns reconstruct exactly. Preserve all other matrices in full. Do not infer eligibility solely from a metric's directed/undirected label, and do not alter asymmetric values to make them fit a compact representation.

## Validation

- Reconstruct the original NxN matrices and verify float64 bit patterns against the sampled source arrays, including NaNs, infinities, signed zeros, and diagonal entries.
- Preserve matrix orientation, channel order, metric identity, and source window indices. Record the representation and codec configuration needed to decode each result.
- Check subsets of windows and metrics through the loader as well as complete sampled arrays.
- Treat existing invalid values as source values to preserve. Failure cleanup and recalculation are separate from this comparison.
- Keep benchmark artifacts separate from the source stores. Do not migrate or rewrite the cohort as part of this piece.

## Measurements

Record these for each case and representation, with per-metric detail where useful:

- Stored bytes including required metadata, allocated filesystem space, and file counts.
- Size relative to the current compressed baseline and the uncompressed float64 payload.
- Encoding and writing time, and reading, decoding, and NxN reconstruction time.
- Time and memory for the common access pattern of loading one selected metric across time and calculating its time average.
- Source loading time separately, so repeated source reads do not unfairly penalize individual variants.
- Peak process memory, thread settings, package versions, sample selection, and the machine/filesystem context.

Report whether reads are cached and whether write timings include flushing to durable storage. Keep quick access measurements distinct from claims about cold disk throughput. Full-cohort size projections should be labeled as extrapolations from the sample, not measured conversion results.

## Deliverables

- A terminal benchmark utility with a minimal, reproducible environment separate from the full pyspi calculation stack.
- Machine-readable results containing the sample and settings, plus a concise comparison for review.
- Lossless reconstruction validation results and enough reader code to assess the compact representations.
- A recommendation describing measured size and access tradeoffs. The final production format and any cohort conversion are separate decisions.

## Execution sequence for this piece

1. Create the small benchmark environment and inventory the selected source arrays.
2. Check one small case to validate the readers, writers, and comparison method.
3. Run the bounded sample comparison, retaining progress and partial measurements.
4. Review the results and expand only the comparisons whose uncertainty could change the storage decision.

Relevant references are the [Zarr performance guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/) and the current source array metadata.
