# Dataset integration and storage precision

Active goal: user's revised attachment `pasted-text-1.txt`, 2026-10-28.

## Requirements tracked

- Metadata: read authoritative `patient_info/electrode` in stored order, match EDF
  uniquely, use the configured data-lake destination without rebuilding metadata.
- Layout: recording/window/reference/run identity, including full reference and
  preprocessing settings, prevents collisions and incompatible resume. Preserve
  contact axes, explicit transformed-channel mappings, and availability masks.
- Precision: required global `output_dtype` (examples float32), optional exact
  identifier overrides float16/32/64, validation before computation, stable dtype
  across batches/retries. Conversion at output boundary only. User decision: allow finite overflow to
  signed infinity; log metric/window/output channel positions without stopping.
- Storage: one engine owned by seeg-dataset, bounded writes, diagnostics, durable
  batch publication and recovery. Shared reader usable without pyspi calculators.
- Evidence: investigate historical GraphicalLassoCV underflow, assess current
  reviewed/amplitude/candidate selections including final time means. Separate
  estimator validity from quantization; discuss meaningful loss.
- Benchmarks: matched f16/f32/f64 write/read size/time/process memory; temporary
  representative EEG integration with exact post-conversion comparison.
- Delivery: focused integration/regression tests, working examples/documentation,
  report precision exceptions, no historical migration or cohort production run.

## Initial findings

- pyspi uses `patient/calculations/run/batches`; seeg-dataset currently uses
  `patient.zarr/connectivity/window/raw`, with no reference/run dimension.
- The data lake writes `dataset_dir/{pid}.zarr`; corrected `config_redo.toml`
  intentionally has no destination. Do not invent one or invoke the builder.
- seeg-dataset's uncommitted pre-existing work was committed with explicit user
  permission as `7ea8a9d`. Source baseline preserved, including existing modes.
- Engines are identical except documentation; consolidate rather than duplicate.
- Installed PCodec directly encodes float16 and uint16; exact end-to-end testing
  is required before claiming support.
- Historical precision exceptions: GraphicalLassoCV ordinary/squared covariance
  in patients105/111 had tiny finite values flush to zero. Previously reported
  uniform relative-error bound was incorrect; audit absolute impact instead.

This checklist is not completion evidence. Record actual tests and benchmark
artifacts as implementation proceeds.

## Completed precision foundation

- seeg-dataset `db786bf` adds exact native f16/f32/f64 storage. All 388 affected
  storage cases pass (385 initially plus three corrected error-text assertions);
  separate patient API tests passed. Includes all 65,536 binary16 patterns,
  pcodec/Zstd and full/triangle/residual layouts.
- pyspi shares that engine through module adapters; 174 public adapter/recovery
  tests passed. Local dependency lock validates offline without third-party upgrades.
- Required global output dtype, per-metric overrides and dtype-aware buffers are
  implemented. 60 initial config/precision/estimate tests passed. 81 execution
  and estimate cases passed; one stale-calibration diagnostic assertion exposed
  redundant mismatch reporting, corrected with a focused recheck.
- User clarified: ALL finite conversion overflow is allowed and recorded with
  metric/window/zero-based output row+column positions. Existing infinities are
  not counted as conversion overflow. Original finite values are logged alongside
  positions and tested for bit-exact JSON recovery. 13 focused tests passed, including saved
  logs, later windows and resume after overflow.

The foundation checkpoint above preceded the metadata integration and precision
benchmarks recorded below.

## Final integration evidence

- Shared `seeg_dataset._runs` owns run addresses, authoritative contact access,
  transformed-axis validation, durable JSON/journals, and the composite reader.
  pyspi output modules delegate to it; no second numerical storage engine remains.
- Dataset mode resolves an explicit directory or the data lake's `dataset_dir`,
  reads included contacts/EDF history in exact order, and refuses contradictions
  before preparing samples. Working examples now use this mode.
- Dataset recording/window/reference/run paths distinguish full reference and
  preprocessing settings; explicit run names still undergo contract checks.
  Source/output axes and local-reference masks remain accessible to readers.
- Full affected workflow regression: 252 passed initially, then only the 12
  failed size-estimator cases reran after replacing its remaining old itemsize
  call; all 12 passed. Three additional ambiguous/duplicate/exclusion mapping
  cases passed. Later strict dtype-publication/read/resume checks passed 11
  focused failure/retry/overflow regressions; no unrelated suites rerun.
- Shared API tests: 6 passed; expanded explicit mapping validation: 1 targeted
  recheck passed. Independent import check: 1 passed. Data-lake metadata tests:
  27 passed with no source metadata changes. Standalone source data-lake builder
  code did not need changes; integration instructions were added.
- Precision receipt covers 188 reviewed outputs, 12 amplitude outputs, and the
  existing three-patient/two-reference candidate pilot. Float32 caused zero
  sampled finite overflow, underflow-to-zero, nonzero sign reversal or finite
  time-mean loss. Largest relative conversion error was below 5.96e-8.
  Established bounds were checked for 142 reviewed outputs without violations.
  The old GraphicalLassoCV exceptions are absent from current selections.
- Actual bounded writer/read benchmark: 18 fresh-process trials, f16/f32/f64,
  three real 118-channel windows per selection, diagnostics included. Float32
  reduced reviewed/amplitude disk bytes by 51%/54%. Three-window buffering was
  faster/smaller than one-window commits; float16 amplitude recovery logs made
  that setting larger and slower. See the report for timing/memory limits.
- Real EEG validation: 118 contacts, 2 windows, 6 metrics. Raw saved float32 bits
  match independently converted callback arrays; original and copied metadata
  hashes match. Temporary output removed. No cohort calculation or migration.
- Working CAR/Laplacian examples passed read-only preflight against actual
  metadata: 10/9 metrics, 118 retained axes, 118/92 reference-available contacts.
- Source localization repository named in data-lake documentation is not present
  here; no external consumer was fabricated or claimed tested. The shared reader
  is verified without pyspi, and existing data-lake reader assumptions are tested.

Evidence and limitations: `docs/dataset_precision_review.md`,
`tests/precision_review/{results,storage_timings,eeg_integration,example_preflight}.json`.
All output remains raw per-window. Mean comparisons belong only to precision
validation; downstream analysis retains responsibility for later aggregation.
