"""Measure the final selection using cached pilot windows, without refiltering.

The standalone/full-eleven benchmarks remain authoritative for their workloads.
This records actual increments for the selected nine-output family and checks
its values against the saved eleven-output results. No new EEG is sampled.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmark import apply_reference, data_for, projected
from summarize import interpolate
from pyspi.workflow import load_config
from pyspi.workflow._metrics import resolve_metrics
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow._reference import reference_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selection = ROOT / "pyspi/workflow/additional_candidates.yaml"
    specs = resolve_metrics(selection, sampling_rate_hz=2048)
    metrics = [getattr(importlib.import_module(s.module), s.class_name)(**s.parameters) for s in specs]
    receipt = {"selection": str(selection.relative_to(ROOT)),
               "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
               "metric_order": [m.identifier for m in metrics], "patients": {},
               "limits": "Three previously sampled windows per patient/reference; fresh window Data/cache. "
                         "Actual selected-family increments with one numerical thread, excluding imports, "
                         "continuous filtering, rereferencing, disk loading and output. "
                         "Cohort interpolation is a planning projection, not a measured cohort run."}
    with numerical_threads(1):
        for pid in ("004", "001", "111"):
            with np.load(args.cache / f"{pid}-inputs.npz") as loaded:
                names = loaded["channel_names"].tolist()
                bands = tuple(map(tuple, loaded["bands"]))
                raw = [(base, {band: loaded[f"band_{i}"][w] for i, band in enumerate(bands)})
                       for w, base in enumerate(loaded["base"])]
                windows = loaded["windows"].tolist()
            entry = receipt["patients"][pid] = {}
            for method in ("car", "laplacian"):
                cfg = load_config(ROOT / f"tests/candidate_pilot/configs/{pid}-{method}.yaml")
                plan = reference_plan(names, cfg.reference)
                valid = np.flatnonzero(plan[1])
                inputs = [(apply_reference(base, names, cfg.reference, plan),
                           {b: apply_reference(v, names, cfg.reference, plan) for b, v in views.items()})
                          for base, views in raw]
                if not receipt.get("warmup_seconds"):
                    before = perf_counter()
                    with redirect_stdout(io.StringIO()):
                        warm, _ = data_for(*inputs[0], valid[:3], cfg, bands)
                        for metric in metrics:
                            metric.multivariate(warm)
                    receipt["warmup_seconds"] = perf_counter()-before
                    del warm
                records = []
                with np.load(args.cache / f"{pid}-{method}-results.npz") as expected:
                    for index, ((base, views), window) in enumerate(zip(inputs, windows)):
                        before = perf_counter()
                        data, usable = data_for(base, views, valid, cfg, bands)
                        setup = perf_counter()-before
                        increments, errors = {}, {}
                        for metric in metrics:
                            before = perf_counter()
                            with redirect_stdout(io.StringIO()):
                                actual = metric.multivariate(data)
                            increments[metric.identifier] = perf_counter()-before
                            reference = expected[metric.identifier][index][np.ix_(usable, usable)]
                            np.testing.assert_array_equal(actual, reference)
                            errors[metric.identifier] = float(np.nanmax(abs(actual-reference)))
                        records.append({"window": window, "setup_seconds": setup,
                                        "increments_seconds": increments,
                                        "total_seconds": setup+sum(increments.values()),
                                        "max_absolute_difference": errors})
                        del data
                entry[method] = {"usable_contacts": len(valid), "windows": records,
                                 **projected(np.median([r["total_seconds"] for r in records]))}
                print(pid, method, round(entry[method]["seconds_per_window"], 4), flush=True)
    cohort = json.loads((Path(__file__).parent / "cohort_counts.json").read_text())["patients"]
    receipt["cohort_projection"] = {}
    for method in ("car", "laplacian"):
        points = [(v[method]["usable_contacts"], v[method]["seconds_per_window"])
                  for v in receipt["patients"].values()]
        counts = [p[f"{method}_contacts"] for p in cohort if p[f"{method}_contacts"] is not None]
        receipt["cohort_projection"][method] = {"patients": len(counts),
            "seconds": 600*sum(interpolate(points, n) for n in counts)}
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
