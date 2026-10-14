# EEG and sEEG optimization discussion

Updated 2026-10-14.

The goal is to make pyspi easier to run for this EEG/sEEG workload and investigate improvements to total processing time and output size. Changes may include Python orchestration, lower-level implementation changes, or faster compatible dependencies. Improvements must be assessed against the actual workload and scientific interpretation.

**Current scope:** production storage is integrated into `pyspi.storage`, with an explicit Calculator adapter and separate `pyspi.analysis.time_mean` helper. Exact values/native precision, ordering/metadata, integrity/recovery, selection, and shard-sizing behavior are retained. The [storage suite](tests/storage/README.md) tests the installed package; calculation and storage tests share the root lock, with only required production storage codecs. The [comparison archive](docs/storage_comparison/README.md) retains original reports, scientific findings, source/tests, raw evidence, and generated result copies. See [production validation and timings](tests/storage/PRODUCTION_RESULTS.md). Existing patient Zarr stores remain unchanged. No patient EEG recalculation or cohort conversion was performed in this piece.

## Working agreement

- Discuss before planning or coding, and ask clarification questions one at a time.
- Plan and implement one selected piece at a time. The broader points below are a backlog, not a request to implement everything together.
- Use local verification. This project does not use remote CI; the unused GitHub Actions test, dataset-generation, and publishing workflows were removed at the user's request.
- Preserve the latest decisions in this note as the work develops so context compaction does not lose them.
- Avoid making global rules where the right decision depends on the metric, data, hardware, or measured benefit.
- A complete benchmark of all 185 metrics is not a prerequisite for progress. Some existing single-window timings are already very long.
- The earlier metric review was targeted at suspicious signs, transpose relationships, labels, and selected definitions. It was not a comprehensive mathematical validation of pyspi's calculations.
- Other agents are building calculation regression tests to detect changes to existing behavior, including behavior with known issues. These baselines do not establish mathematical correctness; intentional corrections require separate supporting evidence and explicit baseline updates.

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
- For now, prepare a uniform sampling rate outside this workflow. A lightweight sampling-rate consistency check before calculations is desirable.
- Use pyspi's existing preprocessing with both operations enabled: **detrend first, then z-score, within each window**.
- Historical output metadata records the old order of z-scoring and then detrending. That historical order is not the desired behavior for new calculations.
- Rereferencing, channel selection, and additional EEG preprocessing are deferred to later loading/preprocessing utilities.

## Execution and resource use

- Prefer a terminal command driven by a configuration file. Notebook support is not a priority.
- Assume the machine is dedicated to the job. Aim for high useful throughput while respecting available memory and avoiding resource exhaustion.
- User-configurable resource upper bounds can be a later objective. Coordinate worker concurrency with threads used internally by numerical libraries rather than blindly maximizing worker counts.
- PBS provided useful external memory allocations and time limits, but a local workflow should be considered for this project. Installing a PBS system is not a requirement already chosen for this work.
- The performance target includes **loading, preprocessing, calculation, and writing output**, not just an isolated algorithm call.
- A normal run uses a configured collection of metrics. Preserve opportunities to share loaded data and intermediate calculations within metric families.
- Batching and parallelization granularity remain to be measured. Window batches are a candidate; no single strategy has been selected as universally best.
- Prefer resumability so an interrupted run can reuse completed work.
- Use the root uv lock for calculation and production storage dependencies. Separate clean environments are useful for installation verification; the archived storage comparison is historical rather than an active environment.

## Dependency integration investigation, 2026-10-14

The [compatibility review](DEPENDENCY_COMPATIBILITY_REVIEW.md) found that the original NumPy <2 limit dates to a 2024 tslearn incompatibility that was not reproduced with current tslearn. A frozen default calculation suite passed on Python 3.12.3/NumPy 2.5.4 together with current storage dependencies: 549 calculation passes (15 skipped, 10 expected failures) and 381 storage passes (two experimental expected failures). This required two `np.in1d` → `np.isin` substitutions in a temporary copy of spectral-connectivity 1.1.2; production source, existing environments, dependencies, and goldens were not changed. A NumPy-2 exception namespace in wavelet code also needs migration attention.

That investigation established feasibility, not mathematical validation. The user subsequently authorized the [dependency migration](DEPENDENCY_MIGRATION.md), completed on Python 3.12/NumPy 2.5.4 with one root lock for calculation and storage tests. MNE/MNE-Connectivity and CDT upgrades passed the unchanged default, small/slow, and full/slow goldens. The final default suite has 563 passes, 15 original skips, and 10 expected failures; each slow tier has 576 passes and 11 expected failures with no ordinary skips. Storage has 381 passes and two existing experimental expected failures. Clean editable and wheel installations, dependency reinstallation, and runtime-only installation were verified. SciPy 1.18 changed ANM beyond the existing tolerances, so SciPy 1.17 is retained. Spectral-connectivity 1.1.2 and pyEDM 1.15.2.0 preserve the existing frequency conventions and CCM estimator; their newer numerical implementations still require individual review. The setuptools bound now correctly excludes 82 and newer. The packaged compatibility fixes cover the spectral dependency's obsolete NumPy calls, the wavelet exception namespace, and the full-tier CCM scalar-conversion issue. Existing patient outputs and golden values remain unchanged.

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

## Lossless storage comparison as the first target

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
  - Exact per-array structure packing: a triangle for bit-exact symmetric or antisymmetric matrices, a triangle plus an XOR residual for near-symmetric ones, and full storage otherwise.
  - Zarr v3, with one CRC32C-checked shard file per array, compressed with pcodec.
  - Size at float64: 0.66–0.70 of the current layout per patient; about 450–460 GiB of the 717 GiB cohort (projected).
  - Size at float32: about 0.31.
  - About 180× fewer files, with faster writes and reads.
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

## Deferred EEG loading utilities

- Later preprocessing utilities may live beside data loading rather than inside metric implementations.
- Desired reference schemes include common average, bipolar, and possibly others.
- Later channel selection should support inclusions/exclusions and bad electrodes. Existing project Zarr files contain bad-electrode information; otherwise, all EEG channels can be the default when no selection is supplied.
- Additional EEG preprocessing belongs in this later discussion. These utilities are outside the current storage piece.

## Read only findings relevant to later work

- The output directory contains 72 patient Zarr stores with metadata declaring 185 metrics. A survey of one representative array per patient found 40–249 channels and 504–610 windows. These are observed sizes, not imposed limits.
- In the fully inspected `001.zarr` connectivity group, all 185 metric arrays have shape `(118, 118, 609)`, float64 values, Zstd compression, and chunks shaped `(118, 118, 1)`. The store already uses Zarr format 3.
- In the current [generic undirected implementation](pyspi/base.py), both ordered directions of each pair are calculated before one triangle is overwritten. Avoiding redundant work is a concrete optimization candidate for methods using that path; correctness, shared-cache effects, and total runtime benefit still need checking.
- The current [spectral implementation](pyspi/statistics/spectral.py) caches results for parameter variants. Grouping decisions should preserve useful shared work.
- The [calculator](pyspi/calculator.py) catches metric exceptions and fills that metric's result with NaNs. Durable diagnostics and timing reports remain topics for later execution work.
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

Updated after production storage integration, 2026-10-14. Earlier task lists duplicated storage integration, scientific review, the local workflow, and optimization; the sequence below consolidates them.

Completed:

- Exact residual retention is settled. No roundoff dropping or precision reduction is selected.
- The storage integrity/completion contract, partial recovery with identifying warnings and availability, strict integer window selection, and shard-size target with a one-window minimum are implemented and tested.
- Calculation regression tests and the storage integrity suite are in the main repository. These detect behavior changes; they are not comprehensive mathematical validation.
- The adversarial test review and targeted test improvements are documented in [TEST_REVIEW.md](tests/TEST_REVIEW.md), with passing-control mutation checks and measured coverage. Scientific decisions remain separate from regression preservation.
- The [dependency migration](DEPENDENCY_MIGRATION.md) is complete: Python 3.12/NumPy 2.5.4, upgraded MNE/MNE-Connectivity and CDT, packaged compatibility fixes, and one root dependency lock. SciPy 1.17, pyEDM 1.15.2.0, and spectral-connectivity 1.1.2 are retained with documented reasons. Local full/slow and installation checks passed. Unused GitHub Actions workflows were removed; verification is local.
- Production storage is integrated into `pyspi.storage`, including single/windowed matrices, an explicit Calculator adapter, native precision, ordering/metadata, integrity/recovery, selection, and shard sizing. The installed wheel passed 419 storage/analysis cases, and the default calculation tier retained 563 passes, 15 skips, and 10 expected failures. Documentation, bounded benchmarks, and installation evidence are in [production results](tests/storage/PRODUCTION_RESULTS.md).
- Time averaging is separate in `pyspi.analysis.time_mean`; its antisymmetric subnormal signed-zero defect is fixed and tested. Lossy tolerance packing and precision conversion are excluded from production. The comparison implementation, reports, scientific findings, and evidence are [archived](docs/storage_comparison/README.md). Existing patient stores remain untouched.

Remaining, in recommended order:

1. **Scientific and numerical review, one finding at a time.** Resolve standard versus historical PSI, spectral band forwarding, wavelet PSI axis selection, signed PLI/wPLI, phase aggregation/endpoints, parameter-specific labels and shared label state, legitimate negative estimators, and automatic CCM embedding selection. Evaluate newer spectral-connectivity and pyEDM, and SciPy/ANM drift, through separate numerical decisions. MNE/CDT upgrades are completed; their regression agreement does not replace independent mathematical checks. Add independent oracles for each proposed correction before changing definitions or expected results. This remains targeted review work, not a completed audit of all 185 calculations.
2. **Local calculation workflow.** Build a terminal/configuration workflow for loading, configurable windows/overlap, sampling-rate consistency checks, detrend-then-z-score preprocessing, grouped calculations/shared intermediates, memory-aware batching, resumability, diagnostics, timing capture, and production storage output. Benchmark representative workloads before choosing concurrency. Resampling remains external initially; arbitrary channel/window/patient bounds are not part of the design.
3. **Calculation and end-to-end optimization.** Prioritize the retained 185, relevant exploratory features, and EEG/sEEG measures. Evaluate shared computation, batching, existing faster implementations, compiled code, and GPU options where measurements justify them. Include loading and output costs. Preserve selected metric definitions unless a separate change is agreed.
4. **Runtime estimator.** Build estimates from accumulated timings and optional short calibration, allowing configuration, cohort/window counts, and available compute to inform a conservative range. The estimator is not a prerequisite or a reason to wait for all metrics to finish.
5. **Later utilities and expanded scope.** Rereferencing, channel inclusion/exclusion and bad-electrode handling, additional metrics, and difficult calculation failures remain later work. Straightforward failures can be addressed when encountered; distinguish unavailable output from calculated NaNs and nonconvergence. Precision reduction and conversion of existing patient stores each require separate justification and discussion.

Recommended next goal, pending the user's choice:

> Independently validate the PSI implementations used by pyspi. Trace their formulas, frequency-band handling, wavelet result axes, sign conventions, and labels. Compare current behavior against reference equations using small reproducible examples and independent tests. Document confirmed defects, intentional variants, historical compatibility implications, and correction options in a tracked report. Update project tracking, then pause for discussion before changing calculation implementations, labels, existing goldens, or patient outputs.

## Storage integration context

The [main-repository storage suite](tests/storage/README.md) has 439 passing cases with no skips or expected failures after the adversarial review (419 at initial integration). The implemented recovery behavior preserves requested shapes and window positions, retains every recoverable matrix entry, fills unavailable entries with NaN, and identifies availability separately. Missing or unreadable requested data emits a warning; NaNs actually saved by a calculation do not emit that warning.

The new format declares requested metrics before writing and publishes completion after payload, presence checks, and flushing. Interrupted writes remain identifiable; verification detects missing required data and deleted declared metrics. Copy verification stays with copying tools; no copying or repair system was added. The [focused integrity plan](tests/storage/INTEGRITY_PLAN.md) and [historical implementation results](tests/storage/INTEGRITY_RESULTS.md) record the contract and earlier measurements. The final production tiny 185-metric batch took 5.70 seconds median, as recorded in [production results](tests/storage/PRODUCTION_RESULTS.md); this is not a full-patient timing estimate.

Time averaging now belongs to the separate analysis module. Exact residual retention and window/shard behavior are settled, and production integration and archival are complete. Scientific reports and benchmark evidence remain preserved. Metric-definition decisions, local execution, targeted calculation optimization, and difficult failure diagnosis remain separate work. Timing capture should develop alongside those pieces, with the runtime estimator added when enough useful measurements exist.
