# Serial EEG workflow and EDF preflight

Preflight validates inputs and describes the proposed calculation. It reads EDF
headers and file sizes, resolves metric constructors/dependencies, and reports
channel order and full-window sample ranges. It does not load EEG samples,
preprocess, calculate connectivity, resample, create output directories, or
write into existing patient stores. The separate `run` command loads/prepares
continuous EEG, calculates windows serially, and writes bounded result batches.
Parallelization and changes to the separate data lake remain later work.

Calculator's [completion reporting and timing](calculation_reporting.md) are
used by the runner to persist diagnostics and write completed values.

Cross-correlation uses the workflow's target sampling rate to convert its
configurable lag limit (default ±125 ms) to samples. A conflicting metric `fs`
is rejected. See [calculation definitions](calculation_definitions.md) for
minimum/peak variants, squaring order and significant-lag selection. The
retained185 preset keeps its existing four cross-correlation requests;
new variants can be selected in a custom metric configuration.

## Terminal usage

Use the root locked environment. Edit the paths in
[the example configuration](../examples/workflow/preflight.yaml), then run:

```bash
uv sync --locked --extra storage
uv run --no-sync python -m pyspi.workflow preflight examples/workflow/preflight.yaml
uv run --no-sync python -m pyspi.workflow preflight examples/workflow/preflight.yaml --json
```

The default report summarizes every recording. `--json` includes the ordered
channel mapping, metric identifiers and exact configured parameters, and every
zero-based `[start, stop)` sample range. Logs from dependency initialization and
warnings go to stderr so stdout remains valid JSON.

Exit codes are 0 for ready inputs, 1 when resampling is required, and 2 for
invalid configuration/input or unavailable requested dependencies. Ready means
header/configuration checks passed; it does not establish signal quality,
estimator convergence, or mathematical correctness.

## Configuration

Version 1 accepts these fields; unknown fields and duplicate YAML keys fail.
Paths are relative to the workflow YAML file unless absolute. They are expanded
and resolved without creating directories.

```yaml
version: 1
edf_glob: /path/to/baseline_edfs/*.[Ee][Dd][Ff]
metrics:
  preset: retained185
sampling_rate_hz: 2048
bandpass:
  low_hz: 0.5
  high_hz: 500
reference: car
window:
  length_seconds: 0.5
  step_seconds: 0.5
preprocessing:
  detrend: true
  zscore: true
output_directory: /path/to/new_results
```

- Supply exactly one of `edf_directory`, `edf_glob`, or a nonempty `recordings`
  list. Directories select immediate `.edf` files case-insensitively. List items
  can be paths or the recording mappings below. Directory/glob order is sorted;
  explicit recording order is preserved. A source cannot appear twice, and
  `(patient_id, recording_id)` pairs must be unique. IDs default to the filename
  stem; no project-specific patient-number parsing is assumed.
- `metrics` selects either a preset (`retained185`, `all`, `fast`, `sonnet`,
  `fabfour`) or `config: path/to/metrics.yaml` using pyspi's existing SPI YAML
  structure. The resolved report preserves configuration order.
- `sampling_rate_hz`, `bandpass.low_hz`, `bandpass.high_hz`, and `reference` are
  required. Omitted/null values fail; there is no inferred/default rate or
  reference. Bandpass edges must satisfy `0 < low < high < target_rate / 2`.
- Window length defaults to 0.5 seconds; step defaults to the chosen length.
  Smaller steps give overlap; larger steps leave gaps. Both must be positive
  and span whole samples at the target rate. There is no silent rounding.
- Preprocessing defaults to detrending followed by z-scoring within each window.
  These flags are recorded only; preflight does not apply them.
- `output_directory` is optional for preflight and required for execution.
  Preflight reports it without creating or writing it; `run` creates the patient
  output layout described below.

## Fixed channel selection and metadata mapping

A data-lake metadata pass can provide the selected channels before calculation:

```yaml
version: 1
recordings:
  - patient_id: example_patient
    recording_id: baseline
    path: /path/to/recording.EDF
    channels:
      - label: EEG B2
        name: B2
      - index: 0
        name: A1
metrics:
  preset: retained185
sampling_rate_hz: 2048
bandpass: {low_hz: 0.5, high_hz: 500}
reference: car
```

Channel entries can be exact EDF labels, zero-based integer signal indices, or
mappings containing exactly one of `label`/`index` and optionally an output
`name`. Indices refer to the original EDF signal header, including its annotation
slots. Selections retain their supplied order. Source channels and output names
must be unique. Ambiguous labels require explicit indices; no case folding,
prefix stripping, prime normalization, or alphabetical sorting is applied.
Duplicate/empty EDF labels can be disambiguated with indices and unique names.

Without a selection, only nonannotation signals containing the literal `EEG `
substring, case-insensitively, are selected. `SkippedChannelsWarning` lists all
skipped signal names. Untyped/misnamed contacts require explicit metadata or
selection. Annotation signals cannot be selected as calculation channels.

Bad-channel decisions apply to the entire recording. The data lake decides the
exclusions; preflight consumes the resulting fixed list. The report retains each
selected source index, original EDF label, output name, native rate, and sample
count. This is the identity/order contract for aligning later matrices with
electrode metadata. Clinical metadata and edge-direction interpretation are not
inferred here.

## Sampling and window boundaries

Native rates come from samples per EDF data record divided by record duration,
before any reader could implicitly resample. Annotation rates and unselected
channels do not affect the selected signal-rate check. Ordinary continuous EDF
and EDF+C are supported. BDF and discontinuous EDF+D are rejected; the latter
requires a loader that understands its time annotations. Unknown record counts
(`-1`) are inferred from whole data records using file size. Truncated or
inconsistent file geometry fails before calculation.

Every selected signal must ultimately match the configured rate. A mismatch,
including mixed rates within one EDF, is reported as `requires_resampling`.
That recording's `windows` is null: sample indices for the resampled signal
are confirmed against MNE's actual resampled output during execution. Matching
recordings still receive their full window plan. Up/downsampling is required
before calculating mismatched inputs; the preparation helper uses MNE
polyphase. No metric `fs` or frequency-band parameters are automatically
rewritten from the EDF rate.

Full windows start at sample 0 and advance by the integer step. Stops are
exclusive; window order is chronological and independent of metric order.
An incomplete final window is discarded with `IncompleteWindowWarning`, without
a special persistent tail record. Recordings shorter than one window report
zero windows and warn. There are no arbitrary channel/patient count bounds.

## Metric validation and the retained preset

Preflight checks module/class identity, constructor arguments (including named
parent arguments behind `**kwargs`), supported backend names/choices where
constructors defer them, finite JSON-compatible parameters, and duplicate
resolved identifiers. Constructor failures propagate as identified errors;
there is no default-constructor fallback or omission of requested metrics.
Null/absent `configs` explicitly requests default construction; an empty list
is rejected. Data-dependent admissibility and convergence cannot be checked
without samples and remain outside preflight.

The existing Java availability check starts JIDT when a requested SPI requires
it, including a JIDT class whose YAML omitted its Java dependency. This constructs
calculators but supplies no observations and runs no estimator. Other metric
imports may initialize their normal third-party caches. NumPy/Python RNG state
and mutable SPI class label lists are restored after inspection; labels remain
unvalidated scientific metadata.

The packaged [retained185.yaml](../pyspi/workflow/retained185.yaml) matches the
185 identifiers in all three archived patient surveys. The historical timing
`used.yaml` contains 186 different requests: four significant-only
cross-correlation variants instead of the two stored unrestricted max variants,
and no stored Gaussian GC configuration. The preset aligns those choices and
otherwise preserves the historical parameters, including full-precision bands.
This reconstructs the requested inventory; it does not prove the exact original
run's environment, defaults, preprocessing, or numerical outputs.

Current verbose identifiers are retained. Simpler names and a separate parameter
representation remain a later discussion near the end of EEG optimization.

## Python access and verification

```python
from pyspi.workflow import preflight

report = preflight("workflow.yaml")
ready = report.ready
metadata = report.as_dict()  # includes explicit sample ranges
summary = report.as_dict(include_ranges=False)
for recording in report.recordings:
    if recording["windows"] is not None:
        for start, stop in recording["windows"].ranges():
            pass  # lazy range iteration; no signal data is loaded
```

Run the focused tests with:

```bash
uv sync --locked --extra testing --extra storage
uv run --no-sync python -m pytest tests/workflow --require-java
```

Preflight tests cover malformed/truncated headers, header-only read guards, native/mixed
rates, up/downsampling requirements, ambiguous channel identities, exact window
boundaries/overlap, strict configuration/metric errors, seeded-state restoration,
the retained inventory, and the real terminal entry point. Signal Data
construction and Calculator computation are prohibited in the preflight tests. The
original preflight-only checkpoint is recorded in
[verification.json](../tests/workflow/verification.json). It predates the required
preparation fields; current preflight/execution verification is recorded in
[execution_results.json](../tests/workflow/execution_results.json).

## EEG preparation and serial execution

```bash
uv run --no-sync python -m pyspi.workflow run examples/workflow/preflight.yaml
uv run --no-sync python -m pyspi.workflow run examples/workflow/preflight.yaml --resume
uv run --no-sync python -m pyspi.workflow run examples/workflow/preflight.yaml --resume --retry-failures
```

For an explicitly bounded trial use `--max-windows 2`. This limits calculation
and output, while continuous preparation still processes the selected channels
of the entire recording. The limit is part of resume compatibility. Stdout is
JSON listing run paths; Calculator progress and warnings go to stderr. A
handled interruption exits 130; handled configuration and filesystem errors
exit 2. Unexpected errors propagate, with their traceback saved in the session
once a run has been initialized.
Finishing with recorded metric failures still exits 0: inspect diagnostics.

The continuous preparation order is fixed:

1. Select and order retained channels.
2. Apply native-rate MNE FIR harmonic notches.
3. Resample mismatches using MNE polyphase, before windowing.
4. Apply continuous MNE zero-phase FIR bandpass at the target rate.
5. Apply the configured MNE reference to actual data.
6. Window chronologically, then use Calculator's existing detrend-then-z-score
   preprocessing per window (`ddof=1`). All requested metrics share that window's
   Data and intermediate caches. Every new window gets a fresh Data/cache.

`notch` accepts `enabled` (default true), `line_frequency_hz` (60), `width_hz`
(1), and `transition_hz` (1). Width is the full stopband width; MNE splits the
transition setting equally between the edges. All harmonics whose complete
stopband/transition fits strictly inside native Nyquist are used, including
1020 Hz at native 2048 Hz. Mixed-rate signals are loaded in native-rate groups
so reader resampling cannot occur before this step. EDF physical units are
converted to SI by MNE, normally volts.

Both notch and bandpass use MNE FIR `phase=zero`, `fir_design=firwin`, Hamming
window, automatic length, and reflected boundary padding. Bandpass transitions
are MNE's automatic values. At 2048 Hz, 500 Hz is the passband edge and the upper
transition ends at 625 Hz. Effective taps, transitions, native-rate notch lists,
reference pool, rates, settings, and emitted preparation warnings are saved.
Short recordings can produce MNE filter-length warnings. Upsampling cannot
restore frequencies missing at acquisition; low-rate recordings can still be
resampled to the common target and use the same final bandpass.

`reference` is explicitly `none`, `car`, or a mapping:

```yaml
# Average named retained contacts; names are output names from the selection.
reference:
  method: channels
  channels: [A1, A2]
# Other supported forms:
# {method: mapping, mapping: {A1: [A2], B1: [B2]}}
# {method: bipolar, anode: [A1], cathode: [A2], names: [A1-A2], drop_refs: true}
# {method: rest, forward: /path/to/matching-forward-solution.fif}
```

CAR averages only retained channels. Named/mapped/bipolar references likewise
use retained names; missing contacts fail before calculation. Bipolar outputs
follow MNE order (remaining original channels, then supplied pairs), with their
derivation recorded. REST requires a separately prepared matching MNE forward
solution. No custom referencing or electrode-selection rule is implemented.
With one retained channel, CAR makes a zero signal; normalisation then fails
and is recorded as window-preprocessing failure rather than changing reference.

## Standalone selection metadata

```bash
uv run --no-sync python -m pyspi.workflow export-selection \
  /path/to/009.zarr /path/to/009_Baseline.EDF /path/to/009-selection.json \
  --patient-id 009 --recording-id baseline
```

Add `metadata: /path/to/009-selection.json` to the workflow. A multi-recording
file uses the same schema and combines the `recordings` entries:

```json
{"schema":"pyspi-channel-selection/1","recordings":[
  {"path":"/path/to/009_Baseline.EDF","patient_id":"009","recording_id":"baseline",
   "channels":[{"index":4,"name":"A1"},{"index":1,"name":"B2"}]}
]}
```

Metadata paths are relative to the metadata file. Every selected EDF needs a
matching entry; conflicting inline IDs/channels fail. Export reads only the
legacy `patient_info/electrode` array with Zarr `mode=r`, preserving its included
contacts and order. It matches real EDF headers after stripping the leading EEG
prefix, removing whitespace, normalizing primes, and case folding. Missing,
duplicate, or ambiguous matches fail; contact numbers are never coerced.
It neither reconstructs exclusion rules nor uses the historical `in_edf_file`
flag. Export requires a new destination outside the source store. The runner
does not depend on legacy Zarrs once this standalone file exists. Clinical and
anatomical metadata remain with the data lake.

## Patient output, reads, and memory

```text
output/patient/
  metadata/baseline.json
  calculations/baseline/
    run.json
    batches/00000000/
      attempts.jsonl
      attempt-<unique-id>/
        reports.json
        results.zarr/
```

The run directory defaults to `recording_id`; optional `run_name` selects another
name so additional configurations can coexist. A run name must be unique within
each patient in an invocation. Metadata/result links are relative; copy the
whole patient folder to preserve them. Inputs' absolute paths are retained as
provenance. Existing outputs are never overwritten by a fresh run. Existing EDFs
and legacy patient stores are untouched; output inside an existing Zarr fails.
Metric identifiers remain unchanged, with module/class/parameters saved in the
run index. Channel axes preserve native pyspi matrix orientation; scientific
source/destination interpretation remains per-metric review work.

```python
import numpy as np
from pyspi.workflow import open_run

results = open_run("output/patient/calculations/baseline")
metric = results.metrics[0]  # configured order
values, available = results.load(metric, return_availability=True)  # N x N x time
channels = results.recording["channels"]
mean = np.nanmean(values, axis=2)
subset = results.load(metric, windows=[5, 1, 5])
status = results.statuses(metric)  # success, failed, unfinished
reports = list(results.diagnostics(metric))
```

Single-window selections return `N x N`; all-window loads remain `N x N x time`,
even with one saved window. Negative/duplicate/empty integer selections retain
the production reader's semantics. Other metrics are not decoded. Missing or
unreadable committed data, and requested unfinished batches, warn and yield
unavailable NaNs. Successfully calculated NaNs/infinities are available values;
recorded calculation failures contain stored NaNs with status `failed`.
`diagnostics`/`statuses` describe committed reports. Pending batch attempts can
be inspected in `attempts.jsonl` but do not claim available numerical output.
The packed `pyspi.analysis.time_mean` helper still targets an individual
production `Results` store; use NumPy reduction for the composite run reader.

`output_batch_mib` defaults to 64 and targets uncompressed matrix buffers across
all metrics (`N² * metric_count * 8` bytes per window). A single window is the
minimum and exceeding the target warns. Packing/compression scratch, current
Calculator table/caches, and continuous selected EEG are additional memory.
The target is not an overall RAM cap. Output arrays never accumulate across
the whole patient. Loading all windows of one metric explicitly allocates that
metric's dense array. Preparing the continuous recording remains proportional
to retained contacts times recording duration; no streaming EDF filtering,
automatic concurrency, or algorithm-memory limiter is supplied.

## Checkpoints, failures, compatibility, and timings

A batch becomes visible only after the existing production writer finishes,
diagnostics are flushed, and the run index is atomically published. `--resume`
keeps committed batches and recomputes the unfinished batch. In a multi-recording
invocation it also starts recordings that had not been reached before the
interruption. At least one configured recording must have an existing run.
Its durable failed
attempts are skipped unless `--retry-failures` is also supplied. Explicit retries
replace only failed metric/windows; other values survive exactly. A retry writes
a new batch generation and publishes it before deleting the old generation.
An interruption during retry leaves the prior committed result readable.
Unreferenced attempt directories are removed after that batch is committed
again. One writer per run is enforced with a POSIX advisory lock.

Compatibility checks include source path/size/mtime/header hash, selected order,
target/preparation/reference settings, window grid/limit, metric identities and
parameters, batch target, Python/package versions, and pyspi source hash.
Committed batch declarations/reports must match the run. This is not a complete
EDF checksum or a payload scan of every previously saved chunk. Production read
integrity checks still apply; no automatic corruption repair is added.

No seed is introduced/reset and no estimator is changed. Skipping committed
windows changes RNG consumption relative to a fresh process/full rerun.
Recomputed unfinished stochastic results can differ; committed values remain
exact. Internally swallowed estimator errors that return NaNs remain successful
calculations under Calculator's existing behavior.

Per-metric reports retain Calculator's existing elapsed time, warnings and
exceptions. Emitted per-window preprocessing warnings are recorded as well;
unhandled run errors retain their traceback in the session. Each run's `sessions` records preparation (loading/filtering/
resampling/reference), per-window preprocessing, calculation wall time,
finished-metric elapsed sum, output, and recording elapsed time. Calculation
wall time includes callbacks/progress; output includes callback buffering/logging
and durable packing/writing, so these fields overlap and should not be summed.
Recording elapsed includes setup/publication for that recording. Shared
preflight/Calculator setup is recorded separately as `invocation_setup_seconds`.
Root `execution-*.json` summaries record complete successful invocation wall
time including shared setup. Interrupted/failed sessions retain elapsed time
and exception identity. The summary's own final write is outside its timer.
An uncatchable termination cannot save a final wall time. Resume identifies its
unfinished session as interrupted with unknown (`null`) elapsed time; already
flushed metric reports and committed batches remain usable.

The runner adds coarse batching and a reader/index around `pyspi.storage`; it
does not refactor the production format or retain every completed matrix before
a batch commits. Files per patient therefore increase with smaller batches.
Representative measurements and verification are recorded in
[execution_results.json](../tests/workflow/execution_results.json).
