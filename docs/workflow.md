# EEG workflow and EDF preflight

Preflight validates inputs and describes the proposed calculation. It reads EDF
headers and file sizes, resolves metric constructors/dependencies, and reports
channel order and full-window sample ranges. It does not load EEG samples,
preprocess, calculate connectivity, resample, create output directories, or
write into existing patient stores. The separate `run` command loads/prepares
continuous EEG, calculates windows with a configured worker count, and writes
bounded result batches.

Calculator's [completion reporting and timing](calculation_reporting.md) are
used by the runner to persist diagnostics and write completed values.

Cross-correlation uses the workflow's target sampling rate to convert its
configurable lag limit (default ±125 ms) to samples. A conflicting metric `fs`
is rejected. See [calculation definitions](calculation_definitions.md) for
minimum/peak variants, squaring order and significant-lag selection. The
reviewed preset includes unrestricted and significant-only max/mean,
ordinary/squared, plus signed minimum and strongest-magnitude summaries.

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

Before a long run, ask for a planning estimate of its duration and output size:

```bash
uv run --no-sync python -m pyspi.workflow estimate workflow.yaml
uv run --no-sync python -m pyspi.workflow estimate workflow.yaml --json
```

The first call measures a calibration beside the configuration
(`workflow.yaml.calibration.json`, about a minute for the reviewed selection
on a 118-contact recording) and later calls reuse it while the metric set,
window and preprocessing still match. See
[Calibration and runtime estimates](#calibration-and-runtime-estimates).

## Configuration

Version 1 accepts these fields; unknown fields and duplicate YAML keys fail.
Paths are relative to the workflow YAML file unless absolute. They are expanded
and resolved without creating directories.

```yaml
version: 1
edf_glob: /path/to/baseline_edfs/*.[Ee][Dd][Ff]
metrics:
  preset: reviewed
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
- `metrics` selects either a preset (`reviewed`, `all`, `fast`, `sonnet`,
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
  preset: reviewed
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

The packaged [reviewed.yaml](../pyspi/workflow/reviewed.yaml) contains 188
requests derived from the 185 identifiers in the archived patient surveys.
It removes ten group-delay band requests, adds 22 signed PLI/wPLI requests
matching the existing bands, and adds eight cross-correlation requests (signed
minimum/peak without selection and six significant-only summaries). Factorized
directed additions are excluded because of their runtime; corrected methods
remain available in explicit configurations. The subsequent regression/Gaussian
review removes six redundant outputs, replaces the Lasso and symbolic TE
settings, and adds Gaussian Granger order 4. The systematic amplitude review
removes five further normalized redundancies, selecting representatives by
standalone runtime. See [the regression/Gaussian review](regression_gaussian_review.md)
and [the amplitude selection review](amplitude_selection_review.md) for the
complete decisions and benchmark receipts.
The later CAR resource review drops seven GraphicalLasso/GraphicalLassoCV
outputs for solver failures and runtime, reducing 195 to 188. The existing
four-window serial pilot spent 9.23 seconds/window on these requests (including
failed refits), about 42% of measured metric work. See the
[removal evidence](workflow_performance_review.md#graphical-lasso-run-removal).
MinCovDet and EllipticEnvelope now also share one seeded FAST-MCD fit in the
library: the latter only adds an outlier threshold that covariance and precision
outputs do not use. The two redundant EllipticEnvelope precision requests remain
excluded from the 188-output selection.
The example workflow selects `reviewed`; the former `retained185` preset name
has been replaced, so update custom workflow YAMLs that use that name.

Execution defaults to one window worker and one numerical thread. Set
`execution.workers`, `execution.threads_per_worker`, and
`execution.spectral_cache_mib` explicitly to compare resource settings. The
spectral cache defaults to 256 MiB per workspace; 1024 MiB is a larger-cache
candidate that can fuse more phase reductions. These limits exclude temporary
arrays, transforms, other calculation caches and process/library overhead.
Values and preprocessing are unchanged by the cache policy.

Workers use spawned processes, initialize their own calculators/JVM when
needed, and map prepared signals read-only. At most one outstanding job/result
per worker is admitted; only the parent publishes journals and output batches.
Completed batches remain resumable after an interruption. In parallel execution,
a window's diagnostics reach the parent when that window finishes; interrupted
in-flight windows are recomputed on resume. Resource settings are recorded per
session and may change on resume. Random estimators without fixed seeds can
still vary with scheduling; the reviewed robust fits use a fixed seed.
One numerical thread per worker remains the default. The earlier CAR pilot
found convergence-sensitive GraphicalLasso results at two threads; those outputs
are now excluded. The other outputs agreed within the recorded tolerance, but
the two-thread trial showed no overall speed benefit. Prefer window workers
until a new profile demonstrates a benefit from additional numerical threads.

Use `python -m pyspi.workflow profile CONFIG --max-windows 4 --report trial.json`
with a new output location for an end-to-end measurement including continuous
preparation, calculation, process-tree RSS/PSS and output writing. The
`estimate-resources` command derives a conservative CPU/memory worker limit from
measured parent/worker peaks. See [the resource review](workflow_performance_review.md)
for the measurement protocol and limitations. It does not launch a cohort or
guarantee a hard process-memory ceiling.

The historical timing
`used.yaml` contains 186 different requests: four significant-only
cross-correlation variants instead of the two stored unrestricted max variants,
and no stored Gaussian GC configuration. The original inventory reconstruction
aligned those choices and preserved full-precision bands; the reviewed config
then applies the additions/removals above. This does not prove the exact original
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
The later parallel/resource review, complete calculation/storage suite and
CAR pilot are recorded in
[resource_results.json](../tests/workflow/resource_results.json).

## EEG preparation and execution

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
5. Apply the configured reference to actual data.
6. Define windows chronologically, then use Calculator's existing detrend-then-z-score
   preprocessing per window (`ddof=1`). All requested metrics share that window's
   Data and intermediate caches. Every new window gets a fresh Data/cache.
   Parallel completion order can differ; stored window axes retain chronological order.

Metrics with explicit `continuous_bands` requirements add shared continuous
band views after rereferencing and before window slicing. For example,
`PowerEnvelopeCorrelation(band=[30, 70])` uses 30–70 Hz input rather than the
broadband window. Each band must fit inside the configured broad passband and
below Nyquist. A band identical to the workflow passband reuses that signal.
Otherwise filtering uses the same FIR settings, in channel blocks covering
the entire time axis. Distinct bands are calculated once per recording.

Additional views are temporary `.npy` files mapped read-only by workers.
`execution.scratch_directory` optionally selects their filesystem and the
parallel input scratch location. By default, scratch is under the output
directory; standalone preparation without an output directory uses OS temp.
Use a disk-backed filesystem for large recordings: `/tmp` can be RAM-backed.
Scratch files are removed when their prepared recording/worker pool is released.
Estimate `prepared_bytes` includes the logical band views; `band_file_bytes`
identifies their file storage. Neither value is a measurement of resident RAM.

Band PEC follows normal window preprocessing. The exploratory
`PhaseAmplitudeCoupling` class follows its prototype's definition: window
Hilbert transforms of continuously filtered samples without extra window
detrending/z-scoring, source phase on rows and target amplitude on columns.
Its amplitude is centered inside the estimator. These distinct preprocessing
choices intentionally prevent incompatible transform sharing. Definitions,
limitations and benchmark receipts are in [the candidate review](additional_pairwise_review.md).

For standalone calculations, attach aligned, pre-window-preprocessing band
arrays with `Data.set_filtered_bands({(30, 70): samples}, sampling_rate_hz=2048)`
or the matching `Calculator.load_dataset(..., filtered_bands=..., sampling_rate_hz=...)`
keywords. The library does not silently filter a short window when bands are
missing. Reattach bands after changing the main Data.

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
solution. No automatic electrode-selection rule is implemented.
With one retained channel, CAR has no usable output; it is recorded as a
window-preprocessing failure rather than changing reference.

For a strict contact-centered Laplacian, supply physical shaft order explicitly:

```yaml
reference:
  method: laplacian
  shafts:
    A: [A1, A2, A3, A4, A5]
    B: [B1, B2, B3, B4]
```

Lists must include intervening physical contacts even when they were excluded
from the retained selection. Each available output is the central contact
minus the mean of its immediately preceding and following contacts. Endpoints
and contacts missing either neighbor retain their output labels but all their
pairwise results are NaN, including when z-scoring is disabled. There is no
one-neighbor, CAR, or original-signal fallback. Unassigned retained contacts and
duplicate layout entries are errors. Layouts are never inferred by sorting
names. Use a recording-specific layout when shaft identities differ.

`method: shaft_average` accepts the same layout and subtracts the mean of all
retained contacts on that shaft, including the target. A shaft with fewer than
two retained contacts is unavailable. This option is for explicit exploratory
comparisons, not an automatic replacement for CAR.

CAR and shaft references record coefficients, unavailable reasons, shared-source
pairs, and proportional pairs in preparation metadata. The recorded null
correlation assumes independent equal-variance source signals; it describes
reference geometry, not measured connectivity. Calibration and runtime estimates
count usable contacts for computation and all output contacts for storage.

### One alternative to the normalized run

[`examples/workflow/amplitude_sensitivity.yaml`](../examples/workflow/amplitude_sensitivity.yaml)
sets **detrend=true, zscore=false**, retains the main run's 0.5-second
nonoverlapping windows and filter settings, and selects 12 outputs from
[`pyspi/workflow/amplitude_sensitivity.yaml`](../pyspi/workflow/amplitude_sensitivity.yaml).
Replace the input, channel-selection metadata and output paths before preflight.
This template does not change the main run or launch any calculations.

The [systematic review](amplitude_selection_review.md) covers **all 200 starting
main-run outputs**, with one decision per identifier in
[`audit/amplitude_review.csv`](../audit/amplitude_review.csv). The alternative
retains four distances (Euclidean, cityblock, Chebyshev, Canberra), two Euclidean
barycenter maxima (ordinary and square-before-maximum), and six precision outputs
(LedoitWolf, OAS, ShrunkCovariance, each ordinary and squared).

The review omits 161 scale-invariant outputs and 22 amplitude-sensitive outputs
whose extra behavior is unwanted amplitude weighting, unit-dependent penalties
or singular/numerical effects. Five more are deferred: four GraphicalLassoCV
outputs with numerical/runtime problems, and signed-data Bray-Curtis with a
potentially vanishing denominator. Previously removed normalized redundancies
were also reconsidered. Five additional normalized redundancies were removed
using standalone timings to choose representatives: that review left
**195 outputs**, retaining cosine distance, squared empirical covariance and
MinCovDet precision over their equivalent alternatives. The subsequent
GraphicalLasso removal leaves **188 main outputs**; the amplitude selection
remains **12**.

The six included precision outputs change beyond pairwise amplitude rescaling
because shrinkage acts on unequal raw variances. They are fitted inverse
covariance entries, not bounded partial correlations. The distances and extrema
also retain relative amplitude information lost by z-scoring. These are
exploratory features, not established EZ biomarkers. Keep the same
EDF-to-volts convention across comparisons.

Detrending already removes the mean as well as a linear trend. Ordinary
barycenter mean remains zero. Fixed regression penalties calibrated for
normalized inputs are not carried into voltage-unit data.

Full inventory checks used eight-channel full-rank and CAR inputs with unequal
amplitudes and volts/microvolts unit controls. The 12-output benchmark at
118 channels × 1,024 samples measured **88.6 ms/window, 53.1 seconds ×600,
61.1 minutes ×41,400**. These are serial projections with one numerical thread,
excluding preprocessing, imports and I/O. Receipts and per-family increments
are linked from the complete review; no clinical discrimination was tested.

CAR is the runnable reference in the template. A local neighbor-average
(Laplacian) montage is a candidate comparison using the strict
`reference.method: laplacian` support above. It requires confirmed same-shaft
adjacency and masks endpoints and gaps. The older general-purpose `mapping`
option leaves unlisted channels unchanged; do not use a partial mapping to
represent a complete local montage. Any chosen local
montage needs matching output/contact-label interpretation and its own output
directory. Shared contacts in local references can themselves induce pairwise
relationships, so a change in connectivity is not by itself evidence that one
reference reveals the true network. See the
[SEEG reference comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648/) and
[MNE's mapping interface](https://mne.tools/stable/generated/mne.set_eeg_reference.html).
Reference changes can also change scale-invariant connectivity outputs. To
test that question, compare a matched subset of the normalized main selection
under each reference; the amplitude selection does not replace that comparison.

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
to retained contacts times recording duration. Spawned workers map that prepared
input read-only, while each keeps its own window data/calculation caches. Result
transfers and worker imports add memory beyond output buffers. Streaming EDF
filtering and a hard process-memory cap are not supplied. Choose concurrency
from measured peaks and throughput using the resource tools described above.

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

## Calibration and runtime estimates

`estimate CONFIG` prints, per recording, the channel count, planned windows,
seconds per window, preparation, compute, output and total time, and the
expected stored size, followed by cohort totals and the ten costliest metrics.
`--json` prints the same figures. Only the configuration is required: worker
count, threads and spectral cache come from its `execution` block.

The numbers come from a calibration measured with the configuration's own
metric set. `estimate` measures one automatically when none exists beside the
configuration or the existing one no longer matches its metric identifiers,
window length, detrend/z-score or threads; `calibrate CONFIG` rebuilds it on
demand with optional `--channels`, `--windows-per-point`, `--budget-seconds`
and `--synthetic`. Any metric configuration the workflow accepts can be
calibrated, including Java information estimators; the Calculator is built
exactly as a run builds it.

Calibration prepares the recording with the most output channels exactly as a
run would, then times the first 2, 4, 8, …, 128 and finally all of its channels
through the Calculator on a few windows from the middle of the recording.
Measuring at the full count matters: whole-matrix estimators have a fixed cost
that dominates at small counts and a steep tail at large ones, so power laws
fitted on small counts under-predict by several times. Every other recording
in the cohort is interpolated between measured counts (log-log linear). A
recording above the largest measured count is extrapolated with the local
exponent clamped to [0, 3] and reported as such. The ladder stops early when
the next count is predicted to exceed the budget (default 600 seconds), which
keeps calibration finite for costly families such as Kraskov transfer entropy,
convergent cross mapping or wavelet coherence; those estimates then extrapolate.

Estimates deliberately err long and large. Each ladder point records the
slowest of its timed windows, and the stages carry fixed headroom (25%
preparation, 15% compute, 40% output, 10% stored size), because a run that
finishes early costs nothing while an overrun costs a schedule. On the
verification recording below the measured figures sat 10–25% under the
uninflated prediction. Per recording, with N output channels, W planned
windows and the configured worker count, before headroom:

- compute = W × seconds-per-window(N) ÷ (workers × 0.9 efficiency), plus half
  a window per batch commit per idle worker when workers > 1;
- preparation scales the calibration recording's measured preparation time by
  native channel-samples;
- output time = batches × measured commit cost + raw matrix bytes × measured
  seconds per byte, where batches follow `output_batch_mib`;
- stored size = raw matrix bytes × each metric's measured on-disk ratio (which
  already reflects upper-triangle packing and compression) plus the diagnostic
  reports.

Per-metric times are additive only for one metric set, because shared
covariance and spectral caches charge the first metric in a family; this is why
a changed metric set triggers a new calibration rather than a subset sum.
Robust covariance and graphical-lasso fits are iterative and data dependent, so
synthetic data (`--synthetic`) misstates them and omits preparation entirely;
use it only without a usable EDF. Spectral-cache size, host load, resumed runs
and other patients' conditioning all move real durations. A bounded
`profile --max-windows` run remains the way to check an estimate.
