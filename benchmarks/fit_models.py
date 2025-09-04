import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression


def _collect_points(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for a in record.get("attempts", []):
        if a.get("status") != "ok":
            continue
        c = a["c"]
        t = a["t"]
        mean = a.get("mean")
        if mean is None or mean <= 0:
            continue
        xs.append([math.log(c), math.log(t)])
        ys.append(math.log(mean))
    if not xs:
        return np.empty((0, 2)), np.empty((0,))
    return np.array(xs), np.array(ys)


def fit_power_law(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Any]:
    if xs.shape[0] < 2:
        return {"fit_status": "insufficient_data"}
    try:
        model = HuberRegressor().fit(xs, ys)
        pred = model.predict(xs)
        resid = ys - pred
        r2 = 1.0 - (np.sum(resid**2) / np.sum((ys - ys.mean()) ** 2))
        params = {
            "log_k": float(model.intercept_),
            "alpha": float(model.coef_[0]),
            "beta": float(model.coef_[1]),
        }
        return {
            "type": "power_law",
            "params": params,
            "metrics": {
                "r2": float(r2),
                "rmse_log": float(np.sqrt(np.mean(resid**2))),
                "resid_std_log": float(np.std(resid, ddof=1)) if xs.shape[0] > 2 else float(np.std(resid)),
                "n": int(xs.shape[0]),
            },
            "fit_status": "ok",
        }
    except Exception as e:
        return {"fit_status": f"failed: {e}"}


def _merge_results(datasets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for data in datasets:
        for rec in data.get("results", []):
            key = rec.get("identifier") or f"{rec['key']['module']}::{rec['key']['measure']}::{rec['key']['config_index']}"
            if key not in by_key:
                by_key[key] = {"identifier": key, "attempts": []}
            by_key[key]["attempts"].extend(rec.get("attempts", []))
    return [v for v in by_key.values()]


def main():
    # ap = argparse.ArgumentParser(description="Fit runtime models from one or more raw timings JSONs")
    # ap.add_argument("--input", required=True, nargs="+", help="Path(s) to raw timings JSON")
    # ap.add_argument("--out", default=None, help="Output models JSON path")
    # args = ap.parse_args()

    # input_paths = args.input
    headpath = "/media/dan/Data/git/pyspi/benchmarks/out"

    input_paths = ["raw_timings_basic_full.json",
"raw_timings_basic_small.json",
"raw_timings_causal_small.json",
"raw_timings_distance_pw_small.json",
"raw_timings_distance_small.json",
"raw_timings_distance_subset.json",
"raw_timings_dry.json",
"raw_timings_full_small.json",
"raw_timings_infotheory_small.json",
"raw_timings_misc_small.json",
"raw_timings_small_r0.json",
"raw_timings_small_r1.json",
"raw_timings_spectral_cohm_small2.json",
"raw_timings_spectral_cohm_small3.json",
"raw_timings_spectral_cohm_small.json",
"raw_timings_spectral_subset.json",
"raw_timings_wavelet_small2.json",
"raw_timings_wavelet_small.json"]

    input_paths = [os.path.join(headpath, p) for p in input_paths]

    out_path = "/media/dan/Data/git/pyspi/benchmarks/out/test.json"

    datasets = []
    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            datasets.append(json.load(f))

    merged = _merge_results(datasets)

    models: Dict[str, Any] = {}
    for rec in merged:
        xs, ys = _collect_points(rec)
        fit = fit_power_law(xs, ys)
        key = rec.get("identifier")
        models[key] = fit

    out = {
        "source": [os.path.abspath(p) for p in input_paths],
        "timestamp": datasets[0].get("run_metadata", {}).get("timestamp") if datasets else None,
        "models": models,
    }
    # default_out = os.path.abspath(args.input[0]).replace("raw_timings_", "models_")
    # out_path = args.out or default_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
