"""Opt-in bounded real-data acceptance check; never modifies source recordings.

Run outside pytest with matching --edf/--legacy pairs and a NEW --output folder.
Only exported selections are read from Zarr; connectivity payloads are untouched.
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import yaml

from pyspi.calculator import Calculator
from pyspi.workflow import export_selection, load_config, open_run, run
from pyspi.workflow._prepare import prepare_recording


def stats(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", action="append", required=True)
    parser.add_argument("--legacy", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contacts", type=int, default=4)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()
    if len(args.edf) != len(args.legacy) or args.contacts < 1 or args.windows < 1:
        parser.error("matching source pairs and positive contact/window counts are required")
    args.output.mkdir(parents=True, exist_ok=False)
    metric_file = args.output / "metrics.yaml"
    metric_file.write_text(yaml.safe_dump({
        ".statistics.basic": {"Covariance": {"configs": [{}]}, "Precision": {"configs": [{}]},
                              "SpearmanR": {"configs": [{}]}},
        ".statistics.spectral": {"CoherenceMagnitude": {"configs": [{"fs": 1, "fmin": .01, "fmax": .4}]},
                                 "PhaseLagIndex": {"configs": [{"fs": 1, "fmin": .01, "fmax": .4}]}}}, sort_keys=False))
    records = []
    for i, (edf, legacy) in enumerate(zip(args.edf, args.legacy)):
        edf, legacy = Path(edf).resolve(), Path(legacy).resolve()
        files = [edf, legacy / "zarr.json", *sorted((legacy / "patient_info" / "electrode").rglob("*"))]
        before = {str(p): stats(p) for p in files if p.is_file()}
        selection = args.output / f"{legacy.stem}-full-selection.json"
        document = export_selection(legacy, edf, selection, patient_id=legacy.stem, recording_id="baseline")
        exported_contacts = len(document["recordings"][0]["channels"])
        document["recordings"][0]["channels"] = document["recordings"][0]["channels"][:args.contacts]
        bounded = args.output / f"{legacy.stem}-bounded-selection.json"
        bounded.write_text(json.dumps(document))
        config = args.output / f"{legacy.stem}.yaml"
        config.write_text(yaml.safe_dump({"version": 1, "output_dtype": "float64", "recordings": [str(edf)],
            "metadata": str(bounded), "metrics": {"config": str(metric_file)}, "sampling_rate_hz": 2048,
            "bandpass": {"low_hz": .5, "high_hz": 500}, "reference": "car",
            "window": {"length_seconds": .5, "step_seconds": .25 if i % 2 else .5},
            "output_directory": str(args.output / "results"), "output_batch_mib": .001}))
        started = perf_counter()
        [path] = run(config, max_windows=args.windows)
        elapsed = perf_counter() - started
        reader = open_run(path)
        cfg = load_config(config)
        prepared = prepare_recording(cfg.recordings[0], cfg)
        with redirect_stdout(sys.stderr):
            for window in range(args.windows):
                calc = Calculator(configfile=str(metric_file), detrend=True, normalise=True)
                start = window * reader.recording["windows"]["step_samples"]
                stop = start + reader.recording["windows"]["length_samples"]
                calc.load_dataset(prepared.data[:,start:stop].copy())
                calc.compute()
                for metric in reader.metrics:
                    expected = calc.table[metric].to_numpy()
                    actual, available = reader.load(metric, window, return_availability=True)
                    assert available.all()
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        before_resume = {m: reader.load(m).copy() for m in reader.metrics}
        run(config, resume=True, max_windows=args.windows)
        for metric, expected in before_resume.items():
            np.testing.assert_array_equal(reader.load(metric).view(np.uint64), expected.view(np.uint64))
        unchanged = all(stats(Path(p)) == value for p, value in before.items())
        assert unchanged
        manifest = json.loads((path / "run.json").read_text())
        records.append({"patient_id": legacy.stem, "exported_contacts": exported_contacts,
                        "calculated_contacts": len(prepared.channel_names), "windows": args.windows,
                        "metrics": reader.metrics, "native_rate_groups": prepared.details["native_rate_groups"],
                        "sampling_rate_hz": 2048, "step_seconds": float(cfg.step_seconds),
                        "wall_seconds_including_setup": elapsed, "session": manifest["sessions"][0],
                        "batch_count": len(manifest["batches"]),
                        "output_files": sum(p.is_file() for p in path.rglob("*")),
                        "output_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
                        "direct_calculator_bit_exact": True, "resume_bit_exact": True,
                        "reference_uses_fresh_calculator_each_window": True,
                        "source_stats_unchanged": unchanged, "run": str(path)})
    receipt = {"schema": "pyspi-real-workflow-validation/1", "records": records,
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "limits": "Bounded contacts/windows, entire continuous preparation; not a 185-metric cohort benchmark or scientific validation."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
