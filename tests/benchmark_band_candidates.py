"""Measure prepared band candidates on a few real windows; not a cohort run.

Example: .venv/bin/python tests/benchmark_band_candidates.py --config workflow.yaml
  --out receipt.json --scratch /path/on/disk/scratch
The source workflow supplies selection, reference and preprocessing. This script
only calculates broadband PEC and the three band candidates, regardless of its
metric configuration. Use one numerical thread. Scratch should be on disk,
not tmpfs: three band files can exceed the available RAM.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pyspi.data import Data
from pyspi.statistics.misc import PhaseAmplitudeCoupling, PowerEnvelopeCorrelation
from pyspi.workflow import load_config
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow._prepare import prepare_recording
from pyspi.workflow._reference import available_indices


def projection(seconds):
    return {"seconds_per_window": float(seconds), "seconds_600": float(seconds*600),
            "seconds_43200": float(seconds*43200)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.windows < 1 or len(cfg.recordings) != 1:
        raise ValueError("select one recording and at least one window")
    cfg = replace(cfg, execution={**cfg.execution, "scratch_directory": str(args.scratch.resolve())})
    metrics = [PowerEnvelopeCorrelation(), PowerEnvelopeCorrelation(orth=True, absolute=True),
               PowerEnvelopeCorrelation(band=(30, 70)),
               PowerEnvelopeCorrelation(band=(30, 70), orth=True, absolute=True), PhaseAmplitudeCoupling()]
    bands = tuple(sorted({band for metric in metrics for band in metric.continuous_bands}))
    receipt = {"configuration": str(args.config.resolve()), "source": str(cfg.recordings[0].path),
               "patient_id": cfg.recordings[0].patient_id, "reference": cfg.reference,
               "sampling_rate_hz": float(cfg.sampling_rate_hz), "detrend": cfg.detrend, "zscore": cfg.zscore,
               "load_average_before": list(os.getloadavg()), "standalone": {}, "family": [],
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [ROOT / "pyspi/_bands.py", ROOT / "pyspi/_envelope.py",
                                     ROOT / "pyspi/statistics/misc.py", ROOT / "pyspi/workflow/_bands.py",
                                     ROOT / "pyspi/workflow/_prepare.py"]},
               "limits": "Few windows from one real recording; one numerical thread. Standalone includes fresh Data, "
                         "band copies and its own caches. Family increments include each metric's lazy preprocessing "
                         "and computation after common Data setup. Full continuous preparation is measured separately; "
                         "output writing and spawned throughput are excluded. Cohort times are serial projections, "
                         "not full cohort measurements. Process peak includes preparation and all candidates."}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")

    with numerical_threads(1):
        before = perf_counter()
        prepared = prepare_recording(cfg.recordings[0], cfg, bands=bands)
        receipt["continuous_preparation_seconds"] = perf_counter()-before
        receipt["preparation_details"] = {key: value for key, value in prepared.details.items() if key != "reference_geometry"}
        geometry = prepared.details.get("reference_geometry")
        if geometry is not None:
            receipt["reference_geometry_summary"] = {"available": geometry["available"],
                "unavailable_reasons": geometry["unavailable_reasons"],
                "shared_source_pair_count": len(geometry["shared_source_pairs"]),
                "proportional_pairs": geometry["proportional_pairs"]}
        valid = available_indices(prepared.details, len(prepared.channel_names))
        length, step = int(cfg.window_seconds*cfg.sampling_rate_hz), int(cfg.step_seconds*cfg.sampling_rate_hz)
        count = 1 + (prepared.data.shape[1]-length)//step
        windows = np.unique(np.rint(np.linspace(0, count-1, min(count, args.windows))).astype(int))
        receipt.update(channels=len(prepared.channel_names), channel_names=prepared.channel_names,
                       usable_channels=len(valid), samples_per_window=length,
                       windows=windows.tolist(), available_windows=count,
                       prepared_band_file_bytes=sum(view.nbytes for view in prepared.bands.values() if view is not prepared.data))
        save()

        def data(window, requested):
            start, stop = int(window)*step, int(window)*step+length
            with redirect_stdout(io.StringIO()):
                result = Data(prepared.data[valid, start:stop].copy(), normalise=cfg.zscore, detrend=cfg.detrend)
            if requested:
                result.set_filtered_bands({band: prepared.bands[band][valid, start:stop] for band in requested},
                                          sampling_rate_hz=float(cfg.sampling_rate_hz))
            return result

        for metric in metrics:
            records = []
            for window in windows:
                before = perf_counter()
                with redirect_stdout(io.StringIO()):
                    value = metric.multivariate(data(window, metric.continuous_bands))
                seconds = perf_counter()-before
                offdiag = ~np.eye(len(valid), dtype=bool)
                finite = np.isfinite(value[offdiag])
                records.append({"window": int(window), "seconds": seconds,
                                "finite_fraction": float(finite.mean()),
                                "mean": float(np.nanmean(value)) if finite.any() else None})
            receipt["standalone"][metric.identifier] = {"windows": records,
                **projection(np.median([r["seconds"] for r in records]))}
            save()
        for window in windows:
            before = perf_counter()
            current = data(window, bands)
            setup = perf_counter()-before
            increments = {}
            for metric in metrics:
                before = perf_counter()
                with redirect_stdout(io.StringIO()):
                    metric.multivariate(current)
                increments[metric.identifier] = perf_counter()-before
            receipt["family"].append({"window": int(window), "input_setup_seconds": setup,
                                       "increments_seconds": increments, "total_seconds": setup+sum(increments.values())})
        receipt["family_projection"] = projection(np.median([r["total_seconds"] for r in receipt["family"]]))
        receipt["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        receipt["load_average_after"] = list(os.getloadavg())
        save()
        print(json.dumps({"out": str(args.out), "continuous_preparation_seconds": receipt["continuous_preparation_seconds"],
                          "process_peak_rss_mib": receipt["process_peak_rss_mib"],
                          "standalone_seconds": {key: value["seconds_per_window"] for key, value in receipt["standalone"].items()}}))


if __name__ == "__main__":
    main()
