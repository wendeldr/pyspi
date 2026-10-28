# CAR workflow resource measurements

Recorded 2026-10-28. Four consecutive 0.5-second nonoverlapping windows, 118
retained contacts, patient 001, 2048-Hz prepared data, CAR, detrend and z-score,
195 selected outputs. Continuous preparation reads the full 304.75-second EDF;
the native recording rate is 1024 Hz. The output-buffer target is 128 MiB.

| Receipt | Workers | Numerical threads/worker | Spectral cache MiB |
| --- | ---: | ---: | ---: |
| `bounded_serial.json` | 1 | 1 | 256 |
| `expanded_serial.json` | 1 | 1 | 1024 |
| `bounded_parallel.json` | 2 | 1 | 256 |
| `threaded_serial.json` | 1 | 2 | 256 |
| `final_native_parallel.json` | 2 | 1 | 256 |

The first four cases precede lazy optional-dependency initialization. The final
case verifies the resulting JVM-free parent/workers. Every profile contains its
actual code/config hashes, timing stages and sampled process-tree memory. Local
input/output paths are retained as provenance; signal data and calculated
connectivity arrays are not included here.
The exact workflow YAMLs are saved in `configs/`; their hashes match the
receipts. Change their machine-specific input/output paths before reusing them.

`comparison.json` compares the first three alternatives against bounded serial,
metric by metric. Cache and worker changes match all 195 outputs. Two numerical
threads change seven graphical-lasso outputs. `final_native_comparison.json`
compares the final profile's stored outputs against that same serial reference;
all 195 match exactly, with matching NaN/infinity masks and statuses.
`phase_timings.json` aggregates the eight phase families from their persisted
metric reports; report hashes and per-window values identify the source evidence.

These are single four-window trials on a shared host. They do not establish
cohort throughput, timing dispersion or scaling to 32 cores. RSS double-counts
shared pages; PSS apportions them. Sampling can miss short peaks. See the
[resource review](../../../docs/workflow_performance_review.md) for projections,
chosen defaults and limitations, and the
[verification receipt](../resource_results.json) for completed tests.

To reproduce with an appropriately configured workflow and a new output path:

```sh
.venv/bin/python tests/workflow/benchmark_resources.py workflow.yaml --out /tmp/new-resource-trial --windows 4
```

The helper checks source/config hashes before reusing a completed case, so it
does not silently rerun or mix code versions. The workflow profiler records
actual storage writes as well as computation. Run outside any sandbox that
blocks the selected EDF or Zarr output.
