# Workflow configuration and EDF preflight

Preflight validates inputs and describes the proposed calculation. It reads EDF
headers and file sizes, resolves metric constructors/dependencies, and reports
channel order and full-window sample ranges. It does not load EEG samples,
preprocess, calculate connectivity, resample, create output directories, or
write into existing patient stores. Execution, checkpoints, parallelization,
and the data-lake integration remain subsequent goals.

Calculator's [completion reporting and timing](calculation_reporting.md) are
implemented separately, providing diagnostics and a callback for the future
runner without starting EDF execution.

## Terminal usage

Use the root locked environment. Edit the paths in
[the example configuration](../examples/workflow/preflight.yaml), then run:

```bash
uv sync --locked
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
window:
  length_seconds: 0.5
  step_seconds: 0.5
preprocessing:
  detrend: true
  zscore: true
output_directory: /path/to/new_results
```

- Supply either `edf_glob` or a nonempty `recordings` list. Glob order is sorted;
  explicit recording order is preserved. A source cannot appear twice, and
  `(patient_id, recording_id)` pairs must be unique. IDs default to the filename
  stem; no project-specific patient-number parsing is assumed.
- `metrics` selects either a preset (`retained185`, `all`, `fast`, `sonnet`,
  `fabfour`) or `config: path/to/metrics.yaml` using pyspi's existing SPI YAML
  structure. The resolved report preserves configuration order.
- `sampling_rate_hz` is the intended calculation rate, not a hardcoded 2048 Hz
  assumption. If omitted/null, a rate is inferred only when every selected
  signal in every recording has the same native rate.
- Window length defaults to 0.5 seconds; step defaults to the chosen length.
  Smaller steps give overlap; larger steps leave gaps. Both must be positive
  and span whole samples at the target rate. There is no silent rounding.
- Preprocessing defaults to detrending followed by z-scoring within each window.
  These flags are recorded only; preflight does not apply them.
- `output_directory` is optional and is reported without being created or
  written. The agreed patient-folder layout is a later writer responsibility.

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
```

Channel entries can be exact EDF labels, zero-based integer signal indices, or
mappings containing exactly one of `label`/`index` and optionally an output
`name`. Indices refer to the original EDF signal header, including its annotation
slots. Selections retain their supplied order. Source channels and output names
must be unique. Ambiguous labels require explicit indices; no case folding,
prefix stripping, prime normalization, or alphabetical sorting is applied.
Duplicate/empty EDF labels can be disambiguated with indices and unique names.

Without a selection, all nonannotation signals are considered EEG unless their
label begins with a recognized non-EEG prefix followed by a space (`ECG`, `EKG`,
`EOG`, `EMG`, `BIO`, `RESP`, `STIM`, `MISC`, `TEMP`, `SAO2`, `CO2`, `GSR`). EDF
does not have a mandatory channel-type field: untyped labels are assumed EEG.
Provide an explicit selection when this assumption is insufficient. Annotation
signals cannot be selected as calculation channels.

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
cannot be guaranteed until the resampling method/output is defined. Matching
recordings still receive their full window plan. Up/downsampling is required
before calculating mismatched inputs; its method and location remain for later
discussion. No metric `fs` or frequency-band parameters are automatically
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
uv sync --locked --extra testing
uv run --no-sync python -m pytest tests/workflow --require-java
```

Tests cover malformed/truncated headers, header-only read guards, native/mixed
rates, up/downsampling requirements, ambiguous channel identities, exact window
boundaries/overlap, strict configuration/metric errors, seeded-state restoration,
the retained inventory, and the real terminal entry point. Signal Data
construction and Calculator computation are prohibited in this suite. The
verification receipt is [tests/workflow/verification.json](../tests/workflow/verification.json).
