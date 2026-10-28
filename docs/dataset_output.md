# Dataset metadata, references, and output precision

The workflow saves **raw matrices for every window**. Averaging across windows
belongs to later analysis. Precision-validation reports compare those later
averages only to quantify conversion error; they do not replace saved windows.

## Input and destination

Install the shared storage dependency (`uv sync --locked --extra storage` in
this checkout, with sibling `../seeg-dataset`). The data lake owns selection and
metadata construction; pyspi consumes the already built result:

```yaml
version: 1
recordings:
  - path: /path/to/001_Baseline.EDF
    patient_id: "001"
    recording_id: baseline
dataset:
  data_lake_config: /home/dan/git/seeg-data-lake/config.toml
output_dtype: float32
metrics:
  preset: reviewed
sampling_rate_hz: 2048
bandpass: {low_hz: 0.5, high_hz: 500}
reference: car
window: {length_seconds: 0.5, step_seconds: 0.5}
preprocessing: {detrend: true, zscore: true}
```

Alternatively use `dataset: {directory: /path/to/built/dataset}`. Choose exactly
one destination source. Relative paths in a data-lake TOML resolve against that
file. Its `dataset_dir` must be set; the corrected `config_redo.toml` intentionally
has no destination yet, so selecting it currently fails clearly. Pyspi never
invokes the metadata builder. Omit `output_directory` and standalone `metadata`
in dataset mode.

For each explicit patient ID, read `dataset_dir/<id>.zarr/patient_info/electrode`.
This exact list determines both source contact axes and their order. The built
`edf_history` supplies the original signal labels. Missing, duplicate, ambiguous,
or inconsistent mappings stop preflight before signal preparation or output.
Older metadata without that table uses the existing strict contact-label matcher;
ambiguity still fails. An explicit inline channel selection must agree exactly
with the metadata, including order. No new QC exclusions or sorting are inferred.

## Reference and preprocessing identity

```text
001.zarr/
  patient_info/                         existing metadata, unchanged
  connectivity/
    baseline/                          recording ID
      0p5s-step0p5s/                    length and step in seconds
        car/                           reference method
          detrend-zscore-<signature>/   run configuration
            zarr.json
            recording.json
            run.json
            batches/00000000/
              attempts.jsonl
              attempt-<id>/
                reports.json
                results.zarr/<metric>/
```

`none`, `car`, `laplacian`, and other supported references occupy separate
reference groups. Metric identifiers remain mathematical names. A default run
signature includes the complete reference settings, contact pool/order, filters,
sampling rate, and detrend/zscore settings, so different local definitions or
preprocessing can coexist. New reference parameters participate in the same
signature. Optional `run_name` supplies a readable run name within the
recording/window/reference group; reuse with incompatible settings is rejected.
For multiple metric selections under otherwise identical settings, give each an
explicit distinct run name.

The configuration, exact metrics/parameters, dtypes, source identity, channel
mapping, and code/dependency identities are checked on resume. A fresh run never
overwrites an existing run. Historical `connectivity/<window>/raw` results are
left untouched. The shared dataset package owns the compact engine and run
reader; pyspi orchestrates preparation and calculation.

Local-reference endpoint/gap masks retain the authoritative axes. Inspect
`reader.recording['reference_available']` for that channel mask. Bipolar outputs
carry new names and an explicit `channel_mapping` to original contacts; original
patient metadata stays unchanged. Full reference details are retained separately
from that identity mapping.

## Storage precision and buffering

`output_dtype` is required: `float16`, `float32`, or `float64`. Working examples
choose float32. Override selected metrics by their exact readable identifier:

```yaml
output_dtype: float32
output_dtype_overrides:
  cov_sq_empirical: float64
output_batch_mib: 32
```

This illustrates the syntax; current validation does not require that override.
Unknown/unselected names and unsupported dtypes fail before calculation. Dtype is
fixed for each metric throughout a run and is independent of estimator arithmetic.
Conversion occurs at output collection; float16 really uses 16-bit payloads.
Both pcodec and Zstd preserve the converted bits, including subnormals, signed
zeros and NaN payloads.

The buffer target accounts for the sum of actual per-metric item sizes. One
window is the minimum. Calculation caches, worker transfers, continuous EEG,
compression scratch, and diagnostics are additional memory. The target is not
a complete-process RAM cap. Each committed batch is durable and independently
readable; interrupted or explicitly retried batches preserve existing completion
and failure behavior.

Any finite conversion overflow becomes signed infinity **without stopping the
run**. The diagnostic event records the metric, zero-based window, original and
output dtypes, counts, output row/column `positions`, and paired `original_values`.
These values survive JSON parsing back to float64 exactly, permitting recovery.
Positions refer to the saved channel list, including any unavailable contacts.
Existing NaNs/infinities are not counted as conversion overflow. Underflow can
round a tiny value to zero; the precision review reports its magnitude separately.

## Reading downstream

```python
from seeg_dataset import list_runs, open_run

paths = list_runs('/path/to/dataset/001.zarr')
reader = open_run(paths[0])  # Select the intended recording/reference/run explicitly.
metric = reader.metrics[0]
raw, readable = reader.load(metric, windows=[5, 1, 5], return_availability=True)
channels = reader.recording['channels']
reference = reader.recording['reference']
statuses = reader.statuses(metric)
diagnostics = list(reader.diagnostics(metric))
```

Raw arrays are `(output_channels, output_channels, windows)`; scalar window
selection returns a matrix. The readable mask describes storage availability,
not whether an estimate is finite. A successfully calculated NaN has status
`success`; calculation failures are stored NaNs with status `failed`; unfinished
or unreadable data warn and are unavailable. Reference availability is its
separate explicit mask.

Later analyses may accumulate in float64 regardless of storage dtype. They must
choose how to treat infinities and failed/unavailable estimates: the reader does
not silently filter them or pre-average data. The overflow records can reconstruct
individual original finite values before a later reduction if needed.

The [precision and storage review](dataset_precision_review.md) contains measured
float32 errors, write/read/size/memory comparisons, and the real EEG integration
receipt. Working metric selections remain 188 normalized, 12 amplitude, and
10 CAR / 9 Laplacian additional candidates; this integration adds no estimators.

The new composite run schema is `seeg-connectivity-run/1`. Old standalone
`pyspi-eeg-run/1` indexes are not silently migrated or interpreted as new runs;
their compact per-batch payloads remain accessible through the storage reader.
