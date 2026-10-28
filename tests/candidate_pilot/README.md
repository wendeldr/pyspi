# Bounded additional-candidate pilot

These scripts and receipts support [the assessment](../../docs/additional_pairwise_pilot.md).
They do not run a predictive model or select using outcome labels. Patient IDs
004/001/111 span 40/118/249 retained contacts. Raw EEG windows and result matrices
used for assessment are local, uncommitted NPZ files; receipts contain aggregate
statistics, timings, provenance and contact/reference metadata.

`configs/` contains the **eleven-output benchmark** inputs, not the final
ten-CAR / nine-Laplacian selections. Full recording preparation is necessary even for a
bounded window trial. Always pass a window limit to the workflow profiles.
The final review examples are `examples/workflow/additional_candidates_*.yaml`.
The EDF and coordinate paths are machine-specific; preserve the matching
selection and verified physical layout if adapting them.

## Reproduction

Run from the repository root, with its numerical environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/pyspi-mpl _MNE_FAKE_HOME_DIR=/tmp/pyspi-mne
mkdir -p /tmp/pyspi-candidate-pilot
.venv/bin/python tests/candidate_pilot/benchmark.py \
  --config tests/candidate_pilot/configs/001-car.yaml \
  --local-config tests/candidate_pilot/configs/001-laplacian.yaml \
  --out tests/candidate_pilot/001-results.json \
  --cache /tmp/pyspi-candidate-pilot --fast-reference
```

Repeat the bounded benchmark for 004 and 111, replacing both config paths and
the receipt name; `--fast-reference` was used only for 001. It compares sixteen
usable contacts from the last reference (Laplacian) against Hyppo's fast path.
Do not rerun completed measurements unless code/input changes justify it.
The small NPZ cache may use `/tmp`; full continuous scratch must use a disk
filesystem (the configs point at `.candidate-scratch` in the repository).

```bash
.venv/bin/python tests/candidate_pilot/assess.py \
  --cache /tmp/pyspi-candidate-pilot \
  --out tests/candidate_pilot/assessment.json --replicates 8
.venv/bin/python -m pyspi.workflow profile \
  tests/candidate_pilot/configs/001-car.yaml --max-windows 12 \
  --report tests/candidate_pilot/workflow-serial.json
.venv/bin/python -m pyspi.workflow profile \
  tests/candidate_pilot/configs/001-car-w2.yaml --max-windows 12 \
  --report tests/candidate_pilot/workflow-parallel.json
.venv/bin/python -m pyspi.workflow profile \
  tests/candidate_pilot/configs/001-kernel-w1.yaml --max-windows 12 \
  --report tests/candidate_pilot/kernel-serial.json
.venv/bin/python -m pyspi.workflow profile \
  tests/candidate_pilot/configs/001-kernel-w2.yaml --max-windows 12 \
  --report tests/candidate_pilot/kernel-parallel.json
.venv/bin/python tests/candidate_pilot/check_outputs.py
.venv/bin/python tests/candidate_pilot/summarize.py
```

Zarr required execution outside the sandbox on the measured machine. Preserve
receipt paths until `check_outputs.py` finishes. The generated workflow stores
are disposable after equality checks; they are not committed. Repeated profiles
need a fresh run name/output location if the earlier run still exists.

`cohort_counts.py` reads the historical mmap cache's patient/contact metadata
and checks coordinate layout consistency. It never uses feature values or
outcome/EZ labels in a calculation:

```bash
.venv/bin/python tests/candidate_pilot/cohort_counts.py \
  --cache /run/media/dan/DansFlash/ram_needs_rma_git/top_features/patient_means_185.pt \
  --coordinates /home/dan/data/IEEG/metadata/xyz \
  --out tests/candidate_pilot/cohort_counts.json
```

The recorded count model covers 72 CAR selections and 71 verified local layouts.
Patient 090's coordinate-order reversal is reported, not repaired by assumption.
`reference_layouts.json` gives the concrete pilot-layout provenance and source
hashes. Pilot EDF indices must agree with their recorded output contact names.

## Receipt interpretation

- `*-results.json`: three real windows per metric/reference; standalone costs,
  actual family increments, finite coverage, preprocessing and sampled PSS.
- `assessment.json`: top-three baseline rank similarities, three-window probe
  observations, and all 24 per-control summaries (three windows × eight
  replicates). These are descriptive controls, not significance tests.
- `*-serial.json`, `*-parallel.json`: real twelve-window workflow profiles,
  including preparation, writing, process-tree memory and worker startup.
- `output_equivalence.json`: thirteen exact stored-array/mask/status comparisons.
- `timing_summary.json`: contact-count interpolation and stage projections;
  sampled min/max windows are not statistical confidence bounds. Standalone
  costs remain visible even when a family shares work.

Source hashes in timing receipts identify the implementation measured. The
assessment uses current implementations whose production source is unchanged
from those receipts. The seed is recorded. The selected baseline panel is not
an exhaustive comparison against all 188 existing outputs.

Two focused filter/reference commutation cases and three surrogate invariant
checks passed; they live in `tests/workflow/execution/test_band_candidates.py`
and `tests/unit/test_candidate_surrogates.py`. Earlier independent estimator
tests and affected regressions are recorded in the implementation ledger.

## Final-panel and allocation checks

The common nine-output family measurement reuses the same
cached windows and checks every result against the eleven-output receipt.
The allocation comparison fills the separate memory-comparison requirement;
it does not repeat the runtime benchmark or claim memory savings from the
native path. Imports/compilation are warmed before tracing. These commands
need no continuous filtering or Zarr output:

```bash
.venv/bin/python tests/candidate_pilot/selected_family.py \
  --cache /tmp/pyspi-candidate-pilot \
  --out tests/candidate_pilot/selected_family.json
.venv/bin/python tests/candidate_pilot/distance_memory.py \
  --cache /tmp/pyspi-candidate-pilot \
  --out tests/candidate_pilot/distance_memory.json
```

`selected_family.json` records actual increments and complete family costs for
all three patients/both references, plus 162 exact array comparisons. Its
selection hash identifies the nine-output config at measurement time (later
comment changes do not change its metric definitions). `distance_memory.json`
compares tracked allocations of the old two-direction wrapper, unique-pair
Hyppo fast path, and the shared native implementation. The memory instrument
has explicit native-allocation/RSS limits; instrumented runtimes are not used.
