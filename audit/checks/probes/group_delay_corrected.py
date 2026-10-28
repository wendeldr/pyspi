"""Focused GD timing or EEG missingness; never modify source recordings.

Timing compares the committed pre-change kernel with the current kernel on
fresh workspaces. EEG uses the already documented pilot preprocessing and
12 label-blind contacts from five recordings; it is not a reproduction of the
unknown historical pipeline or an EZ evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyspi._spectral as current
from pyspi._group_delay import REASONS
from pyspi.data import Data
from spectral_window_smoothing import connection, read_eeg


def receipt():
    return dict(source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                              for p in ("pyspi/_group_delay.py", "pyspi/_spectral.py",
                                        "pyspi/statistics/spectral.py", str(Path(__file__).relative_to(ROOT)))})


def timing():
    revision = "cb60808"
    source = subprocess.check_output(["git", "show", f"{revision}:pyspi/_spectral.py"],
                                     cwd=ROOT, text=True)
    baseline = types.ModuleType("gd_pre_significance_change")
    exec(compile(source, "<committed GD baseline>", "exec"), baseline.__dict__)
    sparse = np.random.default_rng(8026).normal(size=(118, 1024))
    sparse[1] = .8*np.roll(sparse[0], 3) + .2*sparse[1]
    dense = sparse.copy()
    for ch in range(1, len(dense)):
        dense[ch] = .9*np.roll(sparse[0], ch % 8) + .1*sparse[ch]
    bands = [(0, 1024), (0, 512), (512, 1024)]

    def run(module, raw):
        conn = module.spectral_workspace(Data(raw.copy(), normalise=False), 2048)
        return [conn.measure("group_delay", frequencies_of_interest=band)[0] for band in bands]

    report = receipt() | dict(baseline_revision=revision, channels=118, samples=1024,
                              fs=2048, bands=bands, numerical_threads=1, repeats=3, workloads={})
    for name, raw in (("sparse", sparse), ("dense", dense)):
        for module in (baseline, current):
            run(module, raw[:3])
        times = {"before": [], "after": []}
        finite = {}
        for repeat in range(3):
            items = [("before", baseline), ("after", current)]
            if repeat % 2:
                items.reverse()
            for label, module in items:
                start = time.perf_counter()
                matrices = run(module, raw)
                times[label].append(time.perf_counter() - start)
                finite[label] = [int(np.isfinite(m).sum()//2) for m in matrices]
        report["workloads"][name] = dict(seconds=times, finite_pairs_by_band=finite,
                                         median_seconds={k: float(np.median(v)) for k, v in times.items()})
        print(name, report["workloads"][name], flush=True)
    report["limits"] = "Fresh Data, FFT/coherency, selection, regression, all three bands. No I/O or other metrics. Definitions intentionally differ; not an equivalence benchmark."
    return report


def eeg(root):
    bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 70),
             (70, 150), (150, 250), (1, 70), (1, 250), (70, 250)]
    report = receipt() | dict(bands=bands, recordings=[], step_s=.5, fs=2048,
                              nw=3, contacts=12, windows_per_recording=20)
    for patient in ("001", "009", "041", "096", "111"):
        print("Preparing", patient, flush=True)
        raw, metadata = read_eeg(root / f"{patient}.zarr")
        total = raw.shape[1]//1024
        windows = np.unique(np.linspace(0, total-1, 20, dtype=int))
        pairs = np.triu_indices(raw.shape[0], 1)
        rows = [dict(band=band, reasons={name: 0 for name in REASONS[1:]},
                     unsupported_windows=0, errors=[], finite_windows_per_pair=np.zeros(len(pairs[0]), int))
                for band in bands]
        start = time.perf_counter()
        for w in windows:
            conn = connection(raw[:, w*1024:(w+1)*1024], 3)
            for row in rows:
                try:
                    result = conn.group_delay_details(tuple(row["band"]))
                except ValueError as exc:
                    if "at least three frequency bins" not in str(exc):
                        raise
                    row["unsupported_windows"] += 1
                    if not row["errors"]:
                        row["errors"].append(str(exc))
                    continue
                codes = result[3][0][pairs]
                for code, name in enumerate(REASONS[1:], start=1):
                    row["reasons"][name] += int(np.sum(codes == code))
                row["finite_windows_per_pair"] += np.isfinite(result[0][0][pairs])
        for row in rows:
            row["finite_windows_per_pair"] = row["finite_windows_per_pair"].tolist()
        report["recordings"].append(dict(metadata=metadata, selected_windows=windows.tolist(),
                                          calculation_seconds=time.perf_counter()-start, bands=rows))
        print(patient, [(r["band"], r["reasons"]["estimated"], r["unsupported_windows"]) for r in rows], flush=True)
    report["limits"] = "20 evenly spaced existing half-second windows, 66 sampled pairs per recording. Preprocessing follows the earlier pilot, not verified historical preprocessing. No EZ labels."
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("timing", "eeg"))
    parser.add_argument("--root", type=Path, default=Path("/home/dan/data/IEEG/connectivity"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = timing() if args.part == "timing" else eeg(args.root)
    output = args.output or ROOT / f"audit/briefing/group_delay_corrected_{args.part}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
