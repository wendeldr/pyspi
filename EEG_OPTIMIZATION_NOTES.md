# EEG and sEEG optimization discussion

Updated 2026-10-13.

The goal is to make pyspi easier to run for this EEG/sEEG workload and investigate improvements to total processing time and output size. Changes may include Python orchestration, lower-level implementation changes, or faster compatible dependencies. Improvements must be assessed against the actual workload and scientific interpretation.

**Current scope:** implement and run a small lossless storage comparison using existing calculated outputs, following [the focused plan](storage_comparison/PLAN.md). Keep its code, environment, reports, and generated data under `storage_comparison/` so the experiment can be cleaned up after an output format is selected. Existing patient Zarr stores remain unchanged. No EEG recalculation or cohort conversion is part of this piece.

## Working agreement

- Discuss before planning or coding, and ask clarification questions one at a time.
- Plan and implement one selected piece at a time. The broader points below are a backlog, not a request to implement everything together.
- Preserve the latest decisions in this note as the work develops so context compaction does not lose them.
- Avoid making global rules where the right decision depends on the metric, data, hardware, or measured benefit.
- A complete benchmark of all 185 metrics is not a prerequisite for progress. Some existing single-window timings are already very long.

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
- An isolated uv or conda environment with compatible, pinned dependencies is acceptable when needed. The storage comparison can use a small separate environment without requiring the full calculation dependency stack.

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
- A custom compact representation and a loader that reconstructs NxN matrices are acceptable if the disk savings are substantial. The storage implementation need not be Python as long as downstream access remains practical.
- Zarr is convenient and portable, but other formats may be considered. pyspi does not need to reproduce the project's complete Zarr metadata schema.
- Favor manageable independent files or pieces for transfers between computers and drives. The user experienced severe corruption problems when copying enormous HDF5 files and values the easier transfer of partitioned Zarr stores.
- Transfer verification and checksums can remain the copier's responsibility.
- Candidate comparisons include lossless codec/filter settings, chunk layouts, bounded sharding, and compact matrix representations. The exact initial variants and sample sizes remain to be chosen in the focused plan.
- Symmetry packing is a candidate only where reconstruction preserves the actual values. Labels alone are insufficient evidence that a stored matrix can be mirrored exactly.
- Measure savings against the existing compressed output, along with read/write time, file counts, and memory use. Fewer stored entries do not guarantee the same proportional reduction in already-compressed disk size.
- Validate exact reconstruction, including NaNs, infinities, channel/window order, and diagonal values. Existing inputs remain reference data; comparison artifacts belong in separate locations.
- Outcome (2026-10-13; see the [storage report](storage_comparison/REPORT.md)). The measured candidate:
  - Exact per-array structure packing: a triangle for bit-exact symmetric or antisymmetric matrices, a triangle plus an XOR residual for near-symmetric ones, and full storage otherwise.
  - Zarr v3, with one CRC32C-checked shard file per array, compressed with pcodec.
  - Size at float64: 0.66–0.70 of the current layout per patient; about 450–460 GiB of the 717 GiB cohort (projected).
  - Size at float32: about 0.31.
  - About 180× fewer files, with faster writes and reads.
  - The production format is still to be chosen after review.

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

## Next discussion

Review the [storage comparison report](storage_comparison/REPORT.md) and its measured tradeoffs, then select a production output format. Local execution, targeted calculation optimization, and difficult failure diagnosis remain later topics. Timing capture should develop alongside those pieces, with the runtime estimator added when enough useful measurements exist.
