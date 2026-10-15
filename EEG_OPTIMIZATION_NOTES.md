# EEG and sEEG optimization discussion

Updated 2026-10-14.

The goal is to make pyspi easier to run for this EEG/sEEG workload and investigate improvements to total processing time and output size. Changes may include Python orchestration, lower-level implementation changes, or faster compatible dependencies. Improvements must be assessed against the actual workload and scientific interpretation.

**Current state:** production storage and the adversarial test review are complete. `pyspi.workflow` now connects strict configuration/header-only preflight, read-only electrode-selection export, continuous MNE preparation, serial windowed Calculator execution, bounded production storage batches, diagnostics/timings, and coarse resume with explicit failure retries. Usage and limitations are in [the workflow guide](docs/workflow.md); execution evidence is in [its receipt](tests/workflow/execution_results.json), with the [original preflight receipt](tests/workflow/verification.json) preserved as a historical checkpoint. Calculator exposes [per-metric completion/failure reports, timing, and an optional callback](docs/calculation_reporting.md). Parallelization remains later work. Scientific/numerical review is being carried out separately and will be audited when complete. Production storage is in `pyspi.storage`, with an explicit Calculator adapter and separate `pyspi.analysis.time_mean` helper for individual production stores; composite workflow runs support ordinary NumPy time reduction. Calculation, storage, and workflow tests share the root lock. See the [earlier full test review](tests/TEST_REVIEW.md), [storage API](docs/storage.md), and [initial production validation and timings](tests/storage/PRODUCTION_RESULTS.md). The [comparison archive](docs/storage_comparison/README.md) preserves historical evidence. Existing EDFs and patient stores remain unchanged; no cohort recalculation or conversion is authorized.

## Working agreement

- Discuss before planning or coding, and ask clarification questions one at a time.
- Plan and implement one selected piece at a time. The broader points below are a backlog, not a request to implement everything together.
- Use local verification. This project does not use remote CI; the unused GitHub Actions test, dataset-generation, and publishing workflows were removed at the user's request.
- Preserve the latest decisions in this note as the work develops so context compaction does not lose them.
- Avoid making global rules where the right decision depends on the metric, data, hardware, or measured benefit.
- A complete benchmark of all 185 metrics is not a prerequisite for progress. Some existing single-window timings are already very long.
- The earlier metric review was targeted at suspicious signs, transpose relationships, labels, and selected definitions. It was not a comprehensive mathematical validation of pyspi's calculations.
- Calculation regression tests are implemented and have undergone an adversarial review. They detect changes to existing behavior, including known issues. These baselines do not establish mathematical correctness; intentional corrections require separate supporting evidence and explicit baseline updates.

## Data and workload

- Input recordings are EDFs, generally about five minutes per patient.
- The main optimization case is 0.5-second windows with zero overlap, approximately 600 windows per five-minute recording. Different window lengths and overlap must remain possible.
- Calculate all channel pairs for every selected metric and window.
- A rough typical channel range is 50–300. This is a sizing estimate, not a limit. Avoid arbitrary restrictions on channel or patient counts, including for inputs with only two channels or thousands of channels.
- A run might contain one patient or roughly 25–100 patients. Efficient execution should support both individual recordings and batches.
- This computer may not be the eventual calculation machine. Resource use and configuration should remain portable.

## Calculation priorities and scientific choices

- The retained 185 metrics in the existing outputs are an initial focus. The calculations in the exploratory feature directory are also targets for checking availability and possible optimization or integration.
- Functional connectivity and measures commonly used in EEG/sEEG have priority over other measures. This applies to candidates outside the 185 and relevant methods not yet implemented here.
- Prefer preserving existing mathematical definitions, parameters, and estimators while optimizing their implementations.
- Consider changes to definitions, parameters, or estimators individually when implementation optimization appears insufficient and a substantial benefit is plausible. Discuss the scientific consequences and comparability for the specific metric.
- Python is acceptable. C, C++, Rust, Cython, or existing alternative implementations are also acceptable when they provide correct, worthwhile improvements. There is no requirement to rewrite code that is already efficient.
- GPU use is an option to evaluate per workload and metric family. Previous GPU tests were slower, but the exact implementation, configuration, and cause are uncertain.

## Sampling and preprocessing

- The project normally records at 2048 Hz, with a few exceptions. Sampling rate must remain configurable rather than hardcoded to 2048 Hz.
- Recordings not at the explicitly configured target rate are upsampled or downsampled with MNE polyphase in the EEG-preparation helper before windowing. Preflight checks native rates and identifies preparation needed without reading signal payloads. Target rate is required rather than inferred.
- Use pyspi's existing preprocessing with both operations enabled: **detrend first, then z-score, within each window**.
- Historical output metadata records the old order of z-scoring and then detrending. That historical order is not the desired behavior for new calculations.
- The planned runner accepts a fixed, externally supplied channel selection/order for the recording. Deciding which electrodes are bad belongs to the data lake or preprocessing utilities. The user has now included filtering and MNE-based rereferencing in the upcoming workflow; custom reference implementations and other EEG preprocessing remain later work.

## Execution and resource use

- Prefer a terminal command driven by a configuration file. Notebook support is not a priority.
- Assume the machine is dedicated to the job. Aim for high useful throughput while respecting available memory and avoiding resource exhaustion.
- User-configurable resource upper bounds can be a later objective. Coordinate worker concurrency with threads used internally by numerical libraries rather than blindly maximizing worker counts.
- PBS provided useful external memory allocations and time limits, but a local workflow should be considered for this project. Installing a PBS system is not a requirement already chosen for this work.
- The performance target includes **loading, preprocessing, calculation, and writing output**, not just an isolated algorithm call.
- A normal run uses a configured collection of metrics. Preserve opportunities to share loaded data and intermediate calculations within metric families.
- Batching and parallelization granularity remain to be measured. Window batches are a candidate; no single strategy has been selected as universally best.
- Retain completed metric/window work on interruption unless that requires substantial extra code complexity and coarser checkpoints are significantly simpler. Previously failed calculations are retried only on explicit request; successfully calculated NaNs count as completed work.
- Use the root uv lock for calculation and production storage dependencies. Separate clean environments are useful for installation verification; the archived storage comparison is historical rather than an active environment.

## Dependency integration investigation, 2026-10-14

The historical [compatibility review](DEPENDENCY_COMPATIBILITY_REVIEW.md) found that the original NumPy <2 limit dates to a 2024 tslearn incompatibility that was not reproduced in its probes. Its frozen default suite passed on Python 3.12.3/NumPy 2.5.4 with 549 calculation passes (15 skipped, 10 expected failures) and 381 prototype storage passes (two experimental expected failures). That investigation required temporary `np.in1d` → `np.isin` substitutions and identified the wavelet exception-namespace change; both compatibility paths were subsequently addressed in the migration.

That investigation established feasibility, not mathematical validation. The user subsequently authorized the [dependency migration](DEPENDENCY_MIGRATION.md), completed on Python 3.12/NumPy 2.5.4 with one root lock for calculation and storage tests. MNE/MNE-Connectivity and CDT upgrades passed the unchanged default, small/slow, and full/slow goldens. At the migration checkpoint, the default suite had 563 passes, 15 skips, and 10 expected failures; each slow tier had 576 passes and 11 expected failures, and prototype storage had 381 passes and two experimental expected failures. Those are historical counts; the subsequent [test review](tests/TEST_REVIEW.md) records the expanded current suites. Clean editable and wheel installations, dependency reinstallation, and runtime-only installation were verified. SciPy 1.18 changed ANM beyond the existing tolerances, so SciPy 1.17 is retained. Spectral-connectivity 1.1.2 and pyEDM 1.15.2.0 preserve the existing frequency conventions and CCM estimator; their newer numerical implementations still require individual review. The setuptools bound excludes 82 and newer. The packaged compatibility fixes cover the spectral dependency's obsolete NumPy calls, the wavelet exception namespace, and the full-tier CCM scalar-conversion issue. Existing patient outputs and golden values remain unchanged.

Review follow-up: the actual local Java 21 JPype runtime and successful default run are recorded in [review_verification.json](tests/compatibility/review_verification.json). The redundant storage-test project, lock, and virtual environment were removed. The documented pip CPU command was actually installed and used for calculations in a fresh environment. Pytest's fault-handler plugin is disabled for calculations after a reproduced JPype signal-handler conflict and successful controlled verification. Production integration has now replaced prototype imports; current evidence is in [production_verification.json](tests/storage/production_verification.json). Calculation implementations, original regression functions, goldens, and existing patient stores remain unchanged. The user subsequently confirmed that this project uses local verification, so the unused GitHub Actions workflows were removed.

## Timing priorities and the future estimator

- The previous cohort run of about 72 patients reportedly took roughly 15 days. Treat that as approximate historical context.
- Existing proposal timings concern one window. The patient and channel count are uncertain; do not trust estimated contact counts from the proposal.
- Those timings were probably intended to exclude data loading and output writing. Whether metrics were run independently or in groups is uncertain.
- Existing timing notes describe cached incremental costs and some rapid failures. Low recorded time does not by itself establish a successful, independently cheap calculation.
- A whole patient pass taking an hour or less would be ideal, but there is no rigid performance target to enforce now.
- The discussed 24-hour figure concerned one metric over all windows of one patient, approximately 600 windows. It is rough prioritization context, not an automatic cutoff, exclusion rule, or runtime kill switch.
- A slow baseline can still be worth investigating if optimization could substantially improve it. Assess effort and exclusions individually using their effect on total processing time and scientific priority.
- For now, configured calculations may run to completion without a time limit.
- Use short, representative comparisons for the specific change under consideration. Include small and larger cases when useful because scaling can differ by algorithm.
- Save timing information as work proceeds, with enough configuration and hardware context to interpret it. Include loading/writing costs and output validity where relevant.

**The runtime estimator is a later feature built from accumulated measurements, not the first task or a reason to wait for every metric to finish.** It should eventually accept a configuration and an input directory or representative sizes, expected patient/window counts, and available compute. Around 600 windows per patient is a convenient example rather than a fixed assumption.

Saved measurements and optional calibration on a representative sample are both acceptable. A rough range and conservatively biased planning estimate are useful; perfect accuracy is unnecessary. Curves or equations versus compute are desirable where supported, with shared work, memory limits, and measured parallel scaling accounted for.

## Completed storage comparison and agreed policy

- Use existing calculated results so storage investigation does not require expensive EEG recalculation.
- Retain every window's results. Storing only time averages is insufficient.
- Common downstream use is averaging across time for each patient/metric, then reading selected metrics for a patient. Efficient access to subsets of metrics matters.
- Prefer exact preservation of stored values. Float32, quantization, or other reductions in precision require separate evidence and discussion; do not assume they are acceptable for this comparison.
- Decision confirmed 2026-10-14: retain exact residuals. The measured additional disk savings from discarding candidate reciprocal roundoff (about 0.7–0.9% in the inspected samples) are insufficient to justify changing stored values. This decision does not require classifying residual bits as scientific signal or noise.
- Storage-reader selection agreed and implemented 2026-10-14: `Results.load(..., windows=...)` accepts integer indices and one-dimensional integer selections; preserves negative indices, order, duplicates, and empty selections; and rejects floats and booleans with a clear error. This selects saved result windows, not EEG window duration or overlap. Original elements are validated before conversion; large unsigned integers cannot wrap into a valid negative index.
- Shard sizing agreed and implemented 2026-10-14: treat the configured uncompressed size as a target, with a one-window minimum. When a single stored payload row for a window exceeds the target, allow the larger shard and emit an identifying `ShardSizeWarning`. Do not split within a matrix or impose a channel-count limit for this piece. This is not a guaranteed compressed-file byte limit. Chunk sizing now respects the target when a window does fit. Both size settings must be positive integers and are validated before modifying output.
- A custom compact representation and a loader that reconstructs NxN matrices are acceptable if the disk savings are substantial. The storage implementation need not be Python as long as downstream access remains practical.
- Zarr is convenient and portable, but other formats may be considered. pyspi does not need to reproduce the project's complete Zarr metadata schema.
- Favor manageable independent files or pieces for transfers between computers and drives. The user experienced severe corruption problems when copying enormous HDF5 files and values the easier transfer of partitioned Zarr stores.
- Transfer verification and checksums can remain the copier's responsibility. Measured: a plain `rsync -a` re-run does not detect silent bit damage on a copy, because size and mtime are unchanged. The candidate format's per-chunk CRC32C detects it when the copy is read and names the metric, without needing the source.
- The completed comparison covered lossless codec/filter settings, chunk layouts, sharding, and compact matrix representations. Further measurements should address bounded questions arising during integration or optimization.
- Symmetry packing is a candidate only where reconstruction preserves the actual values. Labels alone are insufficient evidence that a stored matrix can be mirrored exactly.
- Measure savings against the existing compressed output, along with read/write time, file counts, and memory use. Fewer stored entries do not guarantee the same proportional reduction in already-compressed disk size.
- Validate exact reconstruction, including NaNs, infinities, channel/window order, and diagonal values. Existing inputs remain reference data; comparison artifacts belong in separate locations.
- Outcome (2026-10-13; see the [archived storage report](docs/storage_comparison/REPORT.md)). The measured candidate:
  - Exact per-array structure packing: a triangle for bit-exact symmetric or antisymmetric matrices, otherwise full storage or a triangle plus an exact XOR residual. A residual can encode substantial directional differences; its presence does not establish approximate symmetry.
  - Zarr v3 arrays with CRC32C-checked chunks, grouped into shards and compressed with pcodec. Shard sizes follow the agreed target/one-window contract above.
  - Size at float64: 0.66–0.70 of the current layout per patient; about 450–460 GiB of the 717 GiB cohort (projected).
  - Direct-cast float32 experiment: about 0.31 of the old size; the review found underflow and an unmeasured fallback policy. No precision conversion was adopted.
  - About 180× fewer files in the compared patient cases. The [correctness review](docs/storage_comparison/REVIEW.md) qualifies the original write totals, mean claims, and scientific interpretations; production timings include packing and durability work.
  - Exact residual preservation is selected and now used by the production API. Float32 conversion and lossy packing were omitted; supplied native float32 is preserved without conversion.

Production integration (2026-10-14): the installed wheel passed 419 storage/analysis cases with approximately 98% combined coverage, and actual Calculator single/windowed round trips were bit-exact. The default calculation tier retained 563 passes, 15 skips, and 10 expected failures. Bounded synthetic benchmarks include packing and compression selection in durable write totals; these are storage costs, not cohort runtime estimates. Scientific definition/label discussions remain pending individually; integration does not settle them. The local calculation workflow remains a separate future piece.

The subsequent [adversarial test review](tests/TEST_REVIEW.md) strengthens recorded-error identity, real-valued comparisons, required golden files, catalogue/metadata checks, and generator determinism/input consistency. It adds independent references, nondegenerate spectral/wavelet boundary cases, controlled stochastic shared-cache comparisons, and independent storage payload/block-boundary checks. Ten temporary runtime mutations were detected. `--require-java` supports complete local verification without silently skipping unavailable Java. The unjustified proposed PLI symmetry assertion was removed; its finite, signed antisymmetric current behavior is tested, and the scientific decision remains pending. Production code, definitions, labels, dependency locks, and existing goldens remain unchanged. Current test/coverage evidence is tracked in [test_review_results.json](tests/test_review_results.json).

## Failures and invalid results

- Investigate failures among the retained 185 first. Excluded metrics can be considered eventually.
- Reported symptoms include missing values, NaNs, and infinities. The user has no saved error logs.
- There were relatively few failures, mainly group delay and one linear CV variant whose exact identifier still needs identifying.
- Proper failure handling and NaN output are needed throughout. Preserve diagnostics that distinguish failures from intentionally undefined entries such as diagonals; do not silently replace invalid values with zeros.
- Difficult diagnosis can come last. Straightforward fixes can happen earlier, but nonconvergence or genuinely undefined results may be acceptable for the moment.
- During storage comparisons, preserve the existing values exactly, including invalid values. Storage testing is not a cleanup or recalculation step.

## EEG preparation utilities

- Preparation lives beside data loading rather than inside metric implementations or storage.
- MNE-based rereferencing is now part of the upcoming workflow, with explicit configuration required, including an explicit choice of none. CAR is the initial project setting; custom reference implementations remain later work. Other MNE reference modes may need additional channel/model metadata.
- The runner's planned input contract accepts fixed channel inclusions/order supplied externally. Existing project Zarr files contain bad-electrode information; deciding those exclusions and generating the selection belong to the data lake/preprocessing utilities. Without metadata, select labels containing `EEG ` case-insensitively and warn with skipped signal names.
- Additional EEG preprocessing beyond the agreed filtering/resampling/referencing remains later discussion.

## Read only findings relevant to later work

- The output directory contains 72 patient Zarr stores with metadata declaring 185 metrics. A survey of one representative array per patient found 40–249 channels and 504–610 windows. These are observed sizes, not imposed limits.
- In the fully inspected `001.zarr` connectivity group, all 185 metric arrays have shape `(118, 118, 609)`, float64 values, Zstd compression, and chunks shaped `(118, 118, 1)`. The store already uses Zarr format 3.
- In the current [generic undirected implementation](pyspi/base.py), both ordered directions of each pair are calculated before one triangle is overwritten. Avoiding redundant work is a concrete optimization candidate for methods using that path; correctness, shared-cache effects, and total runtime benefit still need checking.
- The current [spectral implementation](pyspi/statistics/spectral.py) caches results for parameter variants. Grouping decisions should preserve useful shared work.
- The [calculator](pyspi/calculator.py) catches metric exceptions and fills that metric's result with NaNs. Structured reports now distinguish those failures from successful NaN outputs and expose timing, emitted warnings, and completed matrices to a callback. Persisting diagnostics and coordinating durable writes remain later execution work.
- The current [data preprocessing](pyspi/data.py) supports the desired detrend-then-z-score order.
- GPU options exist through [Spectral Connectivity](https://spectral-connectivity.readthedocs.io/en/latest/) and [JIDT](https://github.com/jlizier/jidt/wiki/GPU). JIDT's guide describes GPU parallelism primarily across surrogates and warns that single-value calls often gain little. Current [pyspi information-theory methods](pyspi/statistics/infotheory.py) use single-value calls. This is a possible explanation to investigate, not an established diagnosis of the user's old GPU tests.

## Project paths and references

| Item | Location |
| --- | --- |
| Working repository | `/media/dan/Data/git/pyspi` |
| Raw EDF recordings | `/media/dan/Data/data/baseline_patients/baseline_edfs` |
| Existing calculated patient outputs | `/media/dan/Data/outputs/calculated_seeg_patient_data` |
| Additional exploratory calculations | `/media/dan/Data/git/exploratory/top_features/further_feature_exploration` |
| Historical timing discussion and plotting scripts | `/media/dan/Data/git/PHD-Writing/proposal/supporting_analyses/aim_2_timing` |
| Timing CSV referenced by those scripts | `/media/dan/Data/data/connectivity/pyspi_testing/timings_output.csv` |
| Historical metric configuration referenced by those scripts | `/media/dan/Data/data/connectivity/pyspi_testing/used.yaml` |

Relevant documentation includes the [pyspi FAQ](https://time-series-features.gitbook.io/pyspi/installing-and-using-pyspi/usage/faq), [Zarr performance guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/), [Spectral Connectivity documentation](https://spectral-connectivity.readthedocs.io/en/latest/), and [JIDT GPU guide](https://github.com/jlizier/jidt/wiki/GPU).

## Completed foundations and consolidated remaining work

Updated after production storage integration, the adversarial test review, and local workflow discussion, 2026-10-14. Earlier task lists duplicated these pieces; the sequence below consolidates them.

Completed:

- Exact residual retention is settled. No roundoff dropping or precision reduction is selected.
- The storage integrity/completion contract, partial recovery with identifying warnings and availability, strict integer window selection, and shard-size target with a one-window minimum are implemented and tested.
- Calculation regression tests and the storage integrity suite are in the main repository. These detect behavior changes; they are not comprehensive mathematical validation.
- The adversarial test review and targeted test improvements are documented in [TEST_REVIEW.md](tests/TEST_REVIEW.md), with passing-control mutation checks and measured coverage. Scientific decisions remain separate from regression preservation.
- The [dependency migration](DEPENDENCY_MIGRATION.md) is complete: Python 3.12/NumPy 2.5.4, upgraded MNE/MNE-Connectivity and CDT, packaged compatibility fixes, and one root dependency lock. SciPy 1.17, pyEDM 1.15.2.0, and spectral-connectivity 1.1.2 are retained with documented reasons. Local full/slow and installation checks passed. Unused GitHub Actions workflows were removed; verification is local.
- Production storage is integrated into `pyspi.storage`, including single/windowed matrices, an explicit Calculator adapter, native precision, ordering/metadata, integrity/recovery, selection, and shard sizing. Initial integration counts, bounded benchmarks, and installation evidence are in [production results](tests/storage/PRODUCTION_RESULTS.md); the subsequent [test review](tests/TEST_REVIEW.md) records the expanded suites and latest verification.
- Time averaging is separate in `pyspi.analysis.time_mean`; its antisymmetric subnormal signed-zero defect is fixed and tested. Lossy tolerance packing and precision conversion are excluded from production. The comparison implementation, reports, scientific findings, and evidence are [archived](docs/storage_comparison/README.md). Existing patient stores remain untouched.
- The serial local workflow is implemented and validated: strict configuration/preflight, read-only selection export, continuous MNE preparation, detrend-then-z-score windows, grouped calculations, bounded exact production output, diagnostics/timings, and coarse resume with explicit failure retries. Target sampling rate, both filter edges, and reference choice are required. The existing storage format was reused without an append/refactor project. See [execution results](tests/workflow/EXECUTION_RESULTS.md).

Remaining, in recommended order:

1. **Audit the separate scientific/numerical review, then discuss findings individually.** Check its evidence and independent mathematical tests before accepting corrections. Outstanding topics include standard versus historical PSI, spectral band forwarding, wavelet PSI axes, signed PLI/wPLI, phase aggregation/endpoints, parameter-specific/shared labels, legitimate negative estimates, CCM embedding selection, and numerical changes from newer spectral-connectivity, pyEDM and SciPy/ANM. MNE/CDT upgrades are completed; regression agreement does not establish mathematical correctness. Do not change definitions or goldens without the agreed scientific decision.
2. **Establish representative end-to-end performance baselines.** Measure short, typical and larger-channel workloads using selected metric families; include loading, preparation, calculation, packing/writing, file count and peak memory. Existing validation uses bounded samples and does not establish full-patient runtime or an optimal batch size. Avoid waiting for every expensive metric or a full cohort before identifying bottlenecks.
3. **Optimize selected calculations and shared work.** Prioritize the retained 185, relevant exploratory features and common EEG/sEEG measures. Evaluate algorithm/code improvements, shared intermediates, faster compatible implementations, compiled code and GPU options when measurements justify them. Include workflow/output costs and the extra files from small batches. Preserve the selected mathematics unless a separate change is agreed; assess expensive outliers individually.
4. **Choose parallelization and resource control from measurements.** After mathematical validation and calculation optimization have progressed, compare patient-level and window-batch execution. Account for estimator/BLAS threads, use spawned processes where JPype requires them, bound in-flight work and choose concurrency from CPU/RAM availability and measured working sets. Keep ordering, single-writer output and recovery correct. Arbitrary channel/patient bounds and automatic 24-hour termination are not part of the design.
5. **Develop the runtime estimator alongside optimization.** Use saved timings and optional short calibration to estimate a conservative range from configuration, cohort/window counts and available compute. The estimator is not a prerequisite or a reason to wait for all metrics to finish.
6. **Simplify calculation names and parameter representation near the end of EEG optimization.** Discuss simple metric names with explicit parameters in metadata/configuration or a companion file. Preserve unambiguous parameter variants and update relevant interfaces/tests/documentation after the naming decision. Current identifiers remain unchanged until then.

Later or separately scoped work: data-lake/prediction-cache integration, custom
rereferencing, project-specific bad-electrode selection utilities, additional
metrics and difficult failure diagnosis. MNE references and externally supplied
fixed channel selections already work. Straightforward failures can be addressed
when encountered. Precision reduction and conversion of existing patient stores
each require separate justification and discussion.

Current assignments: scientific/numerical review is being carried out separately.
This agent will audit that work when it is complete. The local workflow stages
are recorded below. The user subsequently authorized the combined serial
execution goal after settling EEG preparation and the complexity limit for
resume. This permits bounded validation, not cohort execution or parallelization.

## Local workflow planning discussion

- Latest scope discussion: the user selected bounded output/resume, serial
  execution, and workflow validation/documentation as the next work. Defer
  parallelization until scientific validation and calculation optimization
  have progressed. The combined goal was accepted through the attached goal
  text after the preparation decisions below were settled.
  Subsequent restriction: resumability is optional if it cannot be added
  simply. Do not undertake a major storage refactor or rewrite for resume.
  Prefer reusing the existing writer and a coarse completion boundary; omit
  resume if even that requires substantial complexity.
- Agreed input direction: one calculation command accepts raw EDFs from an
  explicit file list or a directory, with optional matching metadata. It calls
  a separate EEG-preparation helper rather than putting EDF-specific operations
  in Calculator. Without metadata, select signal channels whose labels contain
  the literal `EEG ` case-insensitively. Do not automatically include untyped
  or misnamed channels. Emit a warning listing the skipped signal-channel names
  when metadata is absent. With metadata, use its explicit selection and order.
  Preflight and execution now use this same selection rule.
- Resampling order/location is now agreed: the runner's separate EEG-preparation
  helper resamples mismatched recordings before calculation windowing.
  Detrending followed by z-scoring remains per calculation window. The
  bounded implementation uses coarse output batches below.
- The user accepted this preparation order, adding rereferencing before windows:
  select/order channels → notch at native rates → resample to the required
  target → bandpass the continuous recording → rereference → calculation
  windows → detrend then z-score each window → grouped metrics. MNE handles
  the supported reference operations; custom implementations remain later work.
  Reference choice must be explicit in config, including an explicit none.
  CAR is the initial configured value, not an implicit default. The user agreed
  that CAR averages exactly the retained calculation channels, excluding all
  omitted contacts. Record this reference pool with the results. A separate
  CAR reference pool is not part of the initial workflow.
- Use MNE's existing resampling functionality with explicit `method="polyphase"`,
  as agreed after discussing FFT versus polyphase. Apply it before windowing
  and record the method and effective preparation settings with the results.
  Do not introduce a custom resampler as an assumed requirement.
- The user has now explicitly included EEG filtering in the preparation scope:
  remove very low/high frequencies and 60 Hz line interference and its valid
  harmonics. The historical band was 0.5–300 Hz; the new upper passband edge
  is agreed at 500 Hz. The user accepts that the few lower-rate source recordings
  lack some of that bandwidth and does not want this to block using 500 Hz for
  the cohort. The initial project band is 0.5–500 Hz. Both lower and upper
  filter edges must be explicitly supplied in the configuration and remain
  changeable; neither has an implicit default, matching the explicit target-rate
  policy. Apply the filters to continuous data before calculation windowing.
  Notch policy and initial width are agreed below. Preparation order is now
  agreed above. The user accepted MNE's standard zero-phase FIR bandpass
  (`firwin`, Hamming) with
  automatically chosen filter length and transition widths, recording the
  effective design. At 2048 Hz with a 500 Hz upper passband edge, the automatic
  upper transition extends to 625 Hz; the edge is not a hard cutoff.
  The 250–500 Hz fast-ripple range provides a research reason to preserve the
  extra bandwidth, not evidence that every broadband connectivity estimator
  will improve: https://academic.oup.com/brain/article/147/7/2496/7603217.
  A read-only check of six EDF headers (001, 004, 009, 037, 038, 041) found
  blank prefilter fields, so their hardware filtering is not established by
  those headers. No signal data or source files were modified in this check.
- The user authorized a representative spectral inspection to choose notch
  settings. The read-only [line-noise survey](docs/eeg_line_noise_survey.md)
  sampled 12 patients, 144 historically included contacts, and three 60-second
  sections per patient at native rates. It covers all 60 Hz harmonics available
  through 960 Hz, with 5,040 contact/section/harmonic observations. Odd harmonics
  were generally stronger; even harmonics varied rather than being universally
  absent. Higher harmonics above 500 Hz also occur. Blank cells for 512 Hz input
  represent unavailable bands above native Nyquist, not measured missing peaks.
  A temporary-array comparison on seven patients found 1 Hz full-width FIR
  notches sufficient to suppress the sampled central peaks (minimum measured
  suppression 51.7 dB); 2 Hz notches also worked. The user accepted the
  recommendation: use 1 Hz full stopband widths initially, retain configurable
  settings, make the base line frequency configurable, and
  generate all harmonics whose notch and transition fit below Nyquist at the
  filtering stage. At 2048 Hz, valid 60 Hz harmonics extend through 1020 Hz with
  these widths. The earlier survey stopped at 960 Hz because its ±6 Hz
  background comparison could not fit around 1020 Hz below Nyquist. This covers
  a low-pass transition band and adapts to upper-cutoff changes; frequencies
  already removed by low-pass filtering make some notches redundant. MNE's
  standard FIR bandpass design is implemented in the preparation helper. Source EDFs/Zarrs were not
  modified, and no source connectivity arrays were read.
- Target sampling rate must be explicitly supplied: no default and no native-rate
  inference. The existing YAML `sampling_rate_hz` field is the initial
  interface; missing/null values must fail validation before calculation.
  For this project, specify `sampling_rate_hz: 2048`. This supersedes the
  preceding proposal to default to 2048 Hz. Preflight and execution now reject
  missing/null target rates. Metric-specific frequency parameters remain separate and need
  explicit validation as part of the workflow/scientific-review boundary.
- Existing patient Zarrs at
  `/media/dan/Data/outputs/calculated_seeg_patient_data` must not be modified.
  A small standalone metadata export generated read-only from the existing
  Zarrs is now agreed as the initial input interface. It identifies the
  patient/recording and supplies the selected electrode names in their required
  order; broader patient metadata remains with the data lake. The calculation
  runner consumes this export, rather than depending on the legacy Zarr schema.
  The future data-lake metadata producer should target the same input contract
  so it can survive changes to connectivity storage. The user has selected the
  existing Zarr calculated electrode list and its order as the initial
  authoritative selection. Do not rebuild that selection from bad flags or
  anatomical rules. Map those selected contacts to actual EDF headers, rather
  than assuming the provenance inventory has the same membership or order.
  Investigation of the original exclusion rules is being delegated by the
  user to another agent and does not block this workflow discussion.
- A read-only inspection of metadata JSON in `001.zarr` confirmed separate
  electrode, EDF-channel-name, inclusion, and bad-channel arrays. No array
  payloads or connectivity values were read. An optional single-patient
  conversion test can be considered after the new metadata/output interface is
  settled, with all converted output written separately; conversion has not
  been requested or started.
- Follow-up QC check read the four small selection/provenance arrays across all
  72 stores, opening Zarr in read-only mode. Across 9,924 calculated contacts
  and 10,439 provenance contacts, 66 contacts in 31 patients were flagged bad;
  none were marked included or present in the calculated contact lists. Every
  calculated contact matched provenance, and inclusion flags agreed with the
  calculation lists, with no normalization collisions. Of the 515 provenance
  contacts excluded from calculation, 449 were not flagged bad, so the historical
  selection remains more restrictive than merely removing recorded bad flags
  from that provenance inventory. These are checks of stored flags, not an
  independent reassessment of signal quality. Size/mtime checks found no change
  in the 535 existing files under the inspected arrays. The per-patient report
  is `/tmp/pyspi-bad-channel-metadata-check.json`; no connectivity arrays or EDF
  signal payloads were read. Reusing the stored electrode list/order is therefore
  the recommended initial metadata input, subject to EDF channel mapping.
- Example follow-up: patient `089` has three historical exclusions. `IP'13`
  and `IP'14` are flagged bad and have use=0/use2=0 in the original QC review.
  `G'12` is not flagged bad (use=1/use2=1, blank rev), is present in the EDF,
  and contains finite, nonconstant samples over the full recording. All three
  are absent from `089_chnames.csv`; the Zarr contact order matches that CSV.
  `G'12` lacks all five atlas assignments in the source XYZ CSV, while included
  neighbor `G'11` has AAL/AAL2 assignments. Missing anatomical labels are a
  possible historical exclusion reason, but the stored reason is only
  `not_in_chnames`; the original selection rule has not been established.
  This check also confirmed a data-lake provenance defect: `in_edf_file` is
  assigned from historical inclusion, so it is false for all three contacts
  despite their actual presence in the EDF. Future channel matching must use
  actual EDF headers rather than that flag. No source metadata was corrected.
  Detailed read-only findings: `/tmp/pyspi-excluded-contacts-089.json`.
- Smaller example `009`: 71 calculated contacts and 87 EEG-prefixed EDF
  channels. Its 18 exclusions comprise four recorded bad contacts (`C'7`,
  `I'1`, `I'2`, `I'5`), two XYZ contacts absent from both the EDF and QC review
  (`A'7`, `A'8`), and 12 contacts present in the EDF with use=1/use2=1/blank rev
  but no AAL/AAL2 or MICCAI assignments (`G'10`, `R'8`, `I'9`, `I'10`, `O'4`,
  `O'5`, `O'6`, `P'6`, `S'7`, `S'8`, `S'9`, `S'10`). `O'4` has Brainnetome
  and Harvard Oxford labels; `O'5` has a Harvard Oxford label, so absence of
  AAL/MICCAI does not mean every atlas is blank. All 71 included contacts have
  at least one AAL or MICCAI assignment. This supports an anatomical-selection
  hypothesis but does not establish the historical preprocessing rule. The
  source-data comparison is `/tmp/009_excluded_contact_comparison.csv` and the
  metadata report is `/tmp/pyspi-excluded-contacts-009.json`. Existing source
  files and Zarrs remain untouched.
- Logical output should remain connectivity matrices with a known corresponding
  channel/window order. Reporting the actual order used is acceptable so the
  project's own Zarr metadata can use it. Project-specific patient metadata is
  outside this calculation workflow.
- The user needs easy linkage of patient/calculation metadata and
  per-patient, per-metric `(channels, channels, time)` access for prediction
  caches. A metadata pass may need to precede calculation so known bad channels
  are excluded from the input rather than removed afterward.
- Bad-channel exclusions apply to the entire recording, as confirmed by the
  user. Use one fixed selected channel set and order across all metrics and
  windows in a recording/run. Window-specific channel removal is unnecessary
  for this workflow.
- Read-only inspection of `exploratory/stronger_networks/build_cache.py` found
  that it assumes the connectivity axes, `patient_info/electrode`, and SOZ
  arrays are already aligned. It removes infinities and selected extreme values
  before taking a NaN-ignoring mean. `pyspi.analysis.time_mean` exists, but does
  not perform those project-specific cleaning steps. The cache's direct reads
  of old dense Zarr arrays will need an adapter for the new packed stores.
- The data lake's current `build_patient_info` takes included channels/order
  from historical channel-name CSVs and records bad-channel QC afterward. A new
  metadata-first path must separate channel selection from those historical
  calculation artifacts. Proposed boundary, still for discussion: data lake
  supplies recording identity and an ordered selection mapped to original input
  channels; pyspi records the actual axes, windows, configuration, and
  preprocessing used. Data lake joins patient metadata by identity and validates
  alignment. The agreed patient-folder layout below keeps metadata/results
  together through relative links while preserving the packed representation.
- Existing cache scripts and repositories may be rewritten; their file layouts
  are examples of usage, not compatibility requirements. Additional examples
  inspected were `exploratory/linear-vs-nonlinear/build_cache.py` (all 185 time
  means plus metadata) and the older IJCNN camera-ready data-preparation scripts
  (HDF5 input, dense matrices or derived graph caches).
- User-facing output layout agreed: one top-level folder per patient, holding
  patient metadata and a calculations subfolder. For the current dataset this
  normally contains one recording/run result store with all selected metrics
  and windows; additional calculation runs can coexist within the patient
  folder. Keep metadata/result links relative so they travel together. The
  illustrative calculation-store names were placeholders. Exact configuration
  and identities must be stored explicitly, rather than inferred by parsing
  directory names. Final naming and the simpler metric/parameter representation
  are deferred to discussion near the end of EEG optimization.
- An axis-direction discrepancy is visible in the current downstream loaders:
  `stronger_networks/data.py` uses `means[source, destination]`, whereas
  `linear-vs-nonlinear/data.py` uses `means[destination, source]`. This inspection
  does not establish which is mathematically correct. Preserve native matrices;
  require documented, validated per-metric direction when building graph edges.
  Channel identity/order alone does not settle directional interpretation.
- Discard a trailing incomplete window by default. Emit a short warning so the
  user knows it happened; no special persistent tail record is needed.
- The patient-folder interface is agreed; internal window batches/checkpoint
  files remain an execution design choice. Preserve per-recording/per-metric
  matrix access and ordering while comparing those implementation options.
- Interruption policy preference: retain completed work, including individual
  metric/window results. Coarser checkpoints with some recomputation are an
  acceptable alternative if retaining each result adds substantial code
  complexity and recomputing is significantly simpler. Assess that tradeoff
  before selecting an implementation; do not equate finishing computation with
  durable storage. Checkpoint/write costs belong in end-to-end measurements.
  The user subsequently made the complexity limit explicit: skip resumability
  rather than perform a major refactor/rewrite to support it. Fine-grained
  metric/window recovery is therefore not a required outcome. A possible
  lightweight approach is bounded window-batch stores written through the
  existing production API, a small run index for logical per-metric access,
  and reuse only of completed batches; unfinished batches may be recalculated.
  This coarse design is now implemented; bounded real trials record its file
  count, bytes and timing overhead. Recorded calculation failures remain explicit.
- Scope clarification for the revised goal: a mandatory comparison implemented
  as two competing writers, separate instrumentation of every output stage,
  and a new stochastic-seeding scheme are not agreed requirements. Reuse the
  current writer where practical, capture existing per-metric timings plus
  preparation/output/total elapsed time, and preserve existing estimator
  randomness. Do not promise identical stochastic recomputation after resume.
- Resume must not automatically retry previously failed calculations. Record
  failed attempts and their diagnostics; retry them only when explicitly
  requested. A successful calculation returning NaNs is completed work, not a
  failure inferred from the values. Missing/uncommitted output after an
  interrupted write is unfinished work, not a recorded calculation failure.
- A read-only, header-only inspection found 72 EDFs: 67 uniformly at 2048 Hz,
  two at 1024 Hz, and three at 512 Hz. No file had mixed ordinary-signal rates
  or an EDF+D discontinuity marker. Native signal counts were 71–263 and
  durations 252–305.1875 seconds; these include channels not yet classified or
  excluded. No EEG sample payloads were read in this inspection. Mismatched
  inputs require resampling in the EEG-preparation helper before windowing;
  MNE polyphase has been selected, and preflight itself does not resample.
- The current storage writer consumes a whole supplied metric array; it has no
  streaming window append API. Accumulating every result for 185 metrics and
  600 windows would require about 11.52 GiB at 118 channels or 74.43 GiB at
  300 channels before extra working memory. Bounded output and checkpointing
  need an explicit design before large runs.
- Multiprocessing must respect JPype's [spawn requirement](https://jpype.readthedocs.io/en/latest/userguide.html#multiprocessing).
  EDF sampling checks must use native signal rates, since
  [MNE can automatically upsample mixed-rate signals](https://mne.tools/stable/generated/mne.io.read_raw_edf.html).
  These are planning constraints, not changes to calculation definitions.

## Workflow implementation stages

Serial workflow acceptance: the final checkout and noneditable wheel each pass
177 workflow cases. The full-size calculation tier passed 924 cases with ten
known expected failures at its 174-workflow-case checkpoint; the later three
ordering/recovery cases are included in focused verification. All 439 storage
cases pass. Bounded installed-wheel trials at native 512/1024/2048 Hz match a
fresh Calculator for every tested window bit-for-bit, preserve committed output
on resume, and leave accessed source EDF/Zarr files unchanged. Evidence and
scope are in [execution results](tests/workflow/EXECUTION_RESULTS.md) and
[the machine-readable receipt](tests/workflow/execution_results.json).

The implemented checkpoint boundary is one bounded window batch. A small run
index supplies per-metric `(N, N, time)` access across existing production
stores. Unfinished successful work can be recomputed; durable failures require
an explicit retry. Cohort resume also starts unvisited recordings. Continuous
EEG preparation and estimator caches remain additional RAM outside the output
matrix-buffer target. Synthetic measurements confirm that smaller batches add
substantial file-count and fixed I/O overhead; batch sizing is configurable and
is not claimed optimal. Scientific corrections, concurrency and algorithm
optimization remain separate work.

Stages 1–4 are implemented together under the accepted serial-workflow goal.
Later stages remain a plan for discussion, not authorization to implement
every stage. Scientific changes remain with
the separate review; audit its completed work before depending on any changed
definitions. Existing EDFs and patient stores remain reference inputs.

1. **Configuration, input contract, and preflight (implemented).** The
   [workflow guide](docs/workflow.md) documents the implemented contract and
   [verification receipt](tests/workflow/verification.json). Original scope:
   define a small workflow
   configuration around pyspi's existing metric configuration. Accept recording
   identity/source, an optional externally supplied ordered channel selection
   and name mapping, windows/step, and an output location. The data lake remains
   responsible for deciding which electrodes are bad and supplying project
   metadata. Check native EDF signal rates before a reader can resample them,
   resolve channels uniquely, and compute full-window sample ranges without
   reading sample payloads or running metrics. Validate actual resolved metric
   parameters, dependencies, order, and duplicate identifiers rather than
   accepting Calculator's fallback/skipping behavior. Reconcile the historical
   config's 186 declared configurations with the retained 185 identifiers;
   changes to estimators are not part of that reconciliation. Identify sampling
   mismatches requiring resampling to the configured target; its implementation
   is now MNE polyphase in the preparation helper. Specify integer sample conversion and
   window boundaries; discard a partial tail with a warning only. A preflight
   command reports the selected inputs, axes, metrics, and window counts; it is
   not the later runtime estimator. Verify with small fixtures and read-only
   checks against the current EDF headers.

2. **Calculation reporting and timing (implemented).** Calculator exposes
   immutable per-metric reports with success/failure, emitted warnings, exception
   details, and monotonic elapsed time. An optional completion callback receives
   a detached matrix on success and None on failure; consumer errors propagate
   without becoming metric exceptions. Existing numerical behavior,
   preprocessing defaults, diagonal masking, and compute's None return are
   preserved. Successful NaNs remain successful. Reports are in-memory only;
   window identity and persistence are supplied by the serial runner. See the
   [reporting guide](docs/calculation_reporting.md) and
   [verification receipt](tests/calculation_reporting_results.json).

3. **Bounded output and lightweight resume (implemented).** Reuse the existing
   production writer where practical without accumulating a patient's full
   result set. The user explicitly rejects a major storage refactor/rewrite
   for resumability; coarse checkpoints are acceptable, and resume may be
   omitted if it is still substantially complex. Inspect the existing code
   before choosing a design; implementing two competing writers is not a
   requirement. Measure bounded I/O, memory, temporary space, file-count, and
   finalization costs as applicable. Retain exact production packing and logical
   per-metric `(N, N, time)` access under the agreed patient-folder layout.
   If resume is implemented, reject incompatible runs and reuse only committed
   work. Distinguish successful values, recorded calculation failures, and
   unfinished writes; failed calculations are retried explicitly. Verify output
   reconstruction/order, bounded working memory, and interruption handling;
   test resume compatibility/recovery only for the supported resume behavior.

4. **Serial end-to-end runner (implemented).** Connect EDF loading, fixed channel selection,
   the agreed MNE filtering/resampling/referencing sequence, windowing,
   detrend-then-z-score preprocessing, grouped calculations, durable output,
   and lightweight optional resume through terminal commands. Use a fresh Data/cache per
   window and share that window's intermediates across metrics. Compare reader
   preload/memmap/lazy choices as needed; avoid repeated decoding and needless
   full-recording copies. Maintain a bounded output queue and actual saved axis
   metadata, with separate source-channel identity and project name mapping.
   Preserve all windows' values; cleaning, averaging, and prediction caches
   remain downstream. Record existing per-metric times plus preparation,
   output, and total wall time; subdivide further only where practical. Validate a small
   complete recording and bounded representative real-data samples against
   direct Calculator output, including a nonalphabetical channel selection,
   overlapping windows, one-window output, and any supported resume behavior.
   Preserve existing estimator randomness and document limits on stochastic
   recomputation rather than adding an unrequested seeding scheme or promising
   identical recomputed values. Reuse of SPI instances across windows must
   be justified by agreement with fresh construction.

5. **Measured parallelization and resource control.** Establish the serial
   baseline, then compare patient-level and window-batch execution on short
   typical and larger-channel examples using successful representative metric
   families. Bound profiling so progress does not wait for all 185 expensive
   calculations. Include loading and durable output in comparisons. Use spawned
   processes for JPype, initialize worker thread limits before numerical
   imports, and account for BLAS/OMP and estimator-internal concurrency. Choose
   automatic concurrency from available CPUs/memory and measured working sets;
   bound in-flight work and keep one writer per result path. Resource detection
   and graceful handling reduce exhaustion risk; third-party peak allocations
   cannot be guaranteed from array dimensions alone. Test serial/parallel
   ordering, seeded numerical behavior, worker failures, and recovery. Do not
   impose arbitrary channel/patient counts or the discussed 24-hour kill limit.

6. **Validate and document the usable workflow (serial implementation complete; repeat for future parallel/optimization changes).** Run the relevant numerical
   regression and storage suites plus targeted runner integrations. Demonstrate
   selective metric loading, matching metadata/channel/window axes, downstream
   time averaging, portable relative links, and clean interruption/resume.
   Document a retained-185 example configuration, required resampling and
   metadata-selection inputs, diagnostics, explicit retries, and measured
   limitations. Updating the data-lake/cache repositories remains a separate
   integration task after this contract is settled. Save timing evidence for
   the later estimator; do not make the estimator or naming redesign a
   prerequisite for this workflow.

First goal text:

> Implement and test workflow configuration and EDF preflight only. Resolve
> recording identity, externally supplied fixed channel selection/order, the
> retained metric configuration, native sampling-rate checks, and exact
> full-window sample ranges. Provide a terminal preflight report without
> calculating connectivity, resampling, or modifying existing patient data.
> Reject ambiguous channels, duplicate metrics, invalid parameters, and unmet
> requested dependencies instead of silently substituting or omitting work.
> Keep execution, checkpoint writing, parallelization, and naming redesign for
> separate goals.

First-goal verification: **130 tests passed** in the checkout and against an
installed wheel outside the checkout, with Java required and signal Data/
connectivity computation prohibited. Workflow statement coverage is 98.78% and
branch coverage is 96.88%. Header-only preflight of the actual 72-recording
cohort resolved 185 metrics: 67 recordings match 2048 Hz, five require resampling,
and matching recordings have 504–610 full half-second windows. Source sizes and
modification times were unchanged. The retained preset matches all three saved
metric inventories; its parameter provenance and reconstruction limits are
documented in [the workflow guide](docs/workflow.md). Calculation implementations,
existing configurations, golden values, and the dependency lock are unchanged.
See [verification.json](tests/workflow/verification.json) for scoped evidence.

## Storage integration context

The [main-repository storage suite](tests/storage/README.md) has 439 passing cases with no skips or expected failures after the adversarial review (419 at initial integration). The implemented recovery behavior preserves requested shapes and window positions, retains every recoverable matrix entry, fills unavailable entries with NaN, and identifies availability separately. Missing or unreadable requested data emits a warning; NaNs actually saved by a calculation do not emit that warning.

The new format declares requested metrics before writing and publishes completion after payload, presence checks, and flushing. Interrupted writes remain identifiable; verification detects missing required data and deleted declared metrics. Copy verification stays with copying tools; no copying or repair system was added. The [focused integrity plan](tests/storage/INTEGRITY_PLAN.md) and [historical implementation results](tests/storage/INTEGRITY_RESULTS.md) record the contract and earlier measurements. The final production tiny 185-metric batch took 5.70 seconds median, as recorded in [production results](tests/storage/PRODUCTION_RESULTS.md); this is not a full-patient timing estimate.

Time averaging now belongs to the separate analysis module. Exact residual retention and window/shard behavior are settled, and production integration and archival are complete. Scientific reports and benchmark evidence remain preserved. Serial local execution is implemented. Metric-definition decisions, targeted calculation/end-to-end optimization, parallelization and difficult failure diagnosis remain separate work. Timing evidence now develops with the workflow, with the runtime estimator added when enough useful measurements exist.
