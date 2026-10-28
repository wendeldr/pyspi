# Dataset integration and storage precision

Active goal: user's revised attachment `pasted-text-1.txt`, 2026-10-28.

## Requirements and evidence still needed

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

## Current findings

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

Metadata/run-layout integration and representative precision/size benchmarks
remain incomplete. No claim of final float32 suitability yet.
