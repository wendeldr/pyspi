"""Bounded matched-reference pilot; imports/continuous preparation are timed separately.

Use --config and --local-config for the same patient/selection, then --out for
the JSON receipt and --cache for local, uncommitted raw-window/result NPZ files.
Continuous filtering is shared before applying fixed linear references to the
sampled windows. This commutation is validated separately; full workflow output
and spawned throughput are measured with pyspi.workflow._profile instead.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from pyspi.data import Data, InvalidChannelsError
from pyspi.workflow import load_config
from pyspi.workflow._bands import required_bands
from pyspi.workflow._metrics import resolve_metrics
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow._prepare import prepare_recording
from pyspi.workflow._reference import apply_shaft_reference, reference_details, reference_plan
from pyspi.workflow._resources import ProcessTreeMemory


def projected(seconds):
    return {"seconds_per_window": float(seconds), "seconds_600": float(seconds*600),
            "seconds_43200": float(seconds*43200)}


def apply_reference(values, names, reference, plan):
    if reference["method"] == "car":
        return values-values.mean(axis=0)
    return apply_shaft_reference(values, names, reference, plan)


def data_for(base, bands, valid, cfg, requested):
    valid = valid.copy()
    with redirect_stdout(io.StringIO()):
        while len(valid) >= 2:
            try:
                data = Data(base[valid].copy(), normalise=cfg.zscore, detrend=cfg.detrend)
                if requested:
                    data.set_filtered_bands({band: bands[band][valid] for band in requested},
                                            sampling_rate_hz=float(cfg.sampling_rate_hz))
                return data, valid
            except InvalidChannelsError as exc:
                valid = np.delete(valid, exc.processes)
    raise ValueError("fewer than two usable channels")


def summary(values, full_count):
    selected = values[~np.eye(len(values), dtype=bool)]
    finite = selected[np.isfinite(selected)]
    return {"finite_usable_fraction": float(len(finite)/len(selected)),
            "finite_retained_fraction": float(len(finite)/(full_count*(full_count-1))),
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std()) if len(finite) else None,
            "quantiles": np.quantile(finite, [0, .25, .5, .75, 1]).tolist() if len(finite) else [],
            "negative_fraction": float(np.mean(finite < 0)) if len(finite) else None}


def fast_reference(base, cfg):
    from hyppo.independence import Dcorr
    from pyspi.statistics.distance import DistanceCorrelation

    x = data_for(base[:16], {}, np.arange(min(16, len(base))), cfg, ())[0].to_numpy(realisation=0)
    result = {}
    for biased in (False, True):
        model = Dcorr(bias=biased)
        model.is_fast = True  # Library's one-dimensional coefficient path; no permutations/p-values.
        before = perf_counter()
        model.statistic(x[0, :, None], x[1, :, None])
        warmup = perf_counter()-before
        before = perf_counter()
        expected = np.full((len(x), len(x)), np.nan)
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                expected[i, j] = expected[j, i] = model.statistic(x[i, :, None], x[j, :, None])
        library_seconds = perf_counter()-before
        with redirect_stdout(io.StringIO()):
            data = Data(x.copy(), normalise=False)
        before = perf_counter()
        actual = DistanceCorrelation(biased=biased).multivariate(data)
        native_seconds = perf_counter()-before
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        result[str(biased)] = {"channels": len(x), "samples": x.shape[1], "library_seconds": library_seconds,
                              "native_seconds": native_seconds, "warmup_seconds": warmup,
                              "max_absolute_difference": float(np.nanmax(abs(actual-expected))),
                              "library": "Hyppo Dcorr.statistic with is_fast=True, each unordered pair once"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fast-reference", action="store_true")
    args = parser.parse_args()
    configs = [load_config(args.config), load_config(args.local_config)]
    cfg = configs[0]
    if len(cfg.recordings) != 1 or any(c.recordings != cfg.recordings for c in configs):
        raise ValueError("both configurations must describe the same single recording and retained selection")
    specs = resolve_metrics(cfg.metric_config, sampling_rate_hz=float(cfg.sampling_rate_hz))
    metrics = [getattr(importlib.import_module(s.module), s.class_name)(**s.parameters) for s in specs]
    bands = required_bands(specs, cfg)
    pid = cfg.recordings[0].patient_id
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"patient_id": pid, "configurations": [str(c.source.relative_to(ROOT)) for c in configs],
        "source": str(cfg.recordings[0].path), "sampling_rate_hz": float(cfg.sampling_rate_hz),
        "detrend": cfg.detrend, "zscore": cfg.zscore, "references": {}, "load_average_before": list(os.getloadavg()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT / "pyspi").rglob("*.py")},
        "limits": "Three windows spanning one recording. Standalone is cold per metric; family increments "
                  "use one Data/cache in listed order. Continuous preparation is shared before fixed linear "
                  "rereferencing of sampled windows. No output writing or spawned throughput in these times. "
                  "Cohort projections assume this patient's size for every person; they are not cohort forecasts."}

    def save():
        args.out.write_text(json.dumps(receipt, indent=2)+"\n")

    with numerical_threads(1), ProcessTreeMemory() as memory:
        before = perf_counter()
        prepared = prepare_recording(cfg.recordings[0], replace(cfg, reference={"method": "none"}), bands=bands)
        receipt["shared_continuous_preparation_seconds"] = perf_counter()-before
        receipt["preparation_details"] = prepared.details
        names = prepared.channel_names
        length, step = int(cfg.window_seconds*cfg.sampling_rate_hz), int(cfg.step_seconds*cfg.sampling_rate_hz)
        count = 1 + (prepared.data.shape[1]-length)//step
        windows = np.unique(np.rint(np.linspace(0, count-1, min(3, count))).astype(int))
        raw = [(prepared.data[:, w*step:w*step+length].copy(),
                {band: source[:, w*step:w*step+length].copy() for band, source in prepared.bands.items()}) for w in windows]
        receipt.update(channel_names=names, channels=len(names), windows=windows.tolist(), samples_per_window=length,
                       metric_order=[m.identifier for m in metrics], available_windows=count)
        cache = {"windows": windows, "channel_names": np.array(names), "bands": np.array(bands),
                 "base": np.stack([base for base, _ in raw])}
        for i, band in enumerate(bands):
            cache[f"band_{i}"] = np.stack([views[band] for _, views in raw])
        np.savez(args.cache / f"{pid}-inputs.npz", **cache)
        del prepared, cache
        before = perf_counter()
        with redirect_stdout(io.StringIO()):
            warm = Data(np.random.default_rng(15).normal(size=(3, 128)), normalise=True)
            warm.set_filtered_bands({band: np.random.default_rng(i).normal(size=(3, 128))
                                    for i, band in enumerate(bands)}, sampling_rate_hz=float(cfg.sampling_rate_hz))
            for metric in metrics:
                metric.multivariate(warm)
        receipt["calculation_warmup_seconds"] = perf_counter()-before
        del warm
        save()
        for reference_cfg in configs:
            reference = reference_cfg.reference
            method = reference["method"]
            plan = reference_plan(names, reference)
            geometry = reference_details(names, plan)
            valid = np.flatnonzero(plan[1])
            start = perf_counter()
            inputs = [(apply_reference(base, names, reference, plan),
                       {band: apply_reference(values, names, reference, plan) for band, values in views.items()})
                      for base, views in raw]
            entry = {"reference": reference, "usable_contacts": len(valid), "available": plan[1].tolist(),
                "unavailable_reasons": geometry["unavailable_reasons"],
                "shared_source_pairs": len(geometry["shared_source_pairs"]), "proportional_pairs": geometry["proportional_pairs"],
                "reference_seconds_for_sampled_windows": perf_counter()-start, "standalone": {}, "family": []}
            receipt["references"][method] = entry
            results = {}
            for metric in metrics:
                measurements = []
                requested = getattr(metric, "continuous_bands", ())
                for w, (base, views) in zip(windows, inputs):
                    start = perf_counter()
                    with redirect_stdout(io.StringIO()):
                        data, usable = data_for(base, views, valid, cfg, requested)
                        values = metric.multivariate(data)
                    elapsed = perf_counter()-start
                    measurements.append({"window": int(w), "seconds": elapsed, "usable_contacts": len(usable),
                                         **summary(values, len(names))})
                    del data
                entry["standalone"][metric.identifier] = {"windows": measurements,
                    **projected(np.median([m["seconds"] for m in measurements]))}
                save()
                print(pid, method, metric.identifier, round(entry["standalone"][metric.identifier]["seconds_per_window"], 4), flush=True)
            for w, (base, views) in zip(windows, inputs):
                start = perf_counter()
                data, usable = data_for(base, views, valid, cfg, bands)
                setup = perf_counter()-start
                increments = {}
                for metric in metrics:
                    start = perf_counter()
                    with redirect_stdout(io.StringIO()):
                        values = metric.multivariate(data)
                    increments[metric.identifier] = perf_counter()-start
                    full = np.full((len(names), len(names)), np.nan)
                    full[np.ix_(usable, usable)] = values
                    results.setdefault(metric.identifier, []).append(full)
                entry["family"].append({"window": int(w), "setup_seconds": setup, "increments_seconds": increments,
                                        "total_seconds": setup+sum(increments.values())})
                del data
            entry["family_projection"] = projected(np.median([w["total_seconds"] for w in entry["family"]]))
            np.savez(args.cache / f"{pid}-{method}-results.npz", **{key: np.stack(value) for key, value in results.items()})
            save()
        if args.fast_reference:
            receipt["hyppo_fast_comparison"] = fast_reference(inputs[0][0][valid], cfg)
        receipt["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        receipt["load_average_after"] = list(os.getloadavg())
    receipt["process_tree_memory"] = memory.as_dict()
    save()


if __name__ == "__main__":
    main()
