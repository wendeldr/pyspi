import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression

EPS = 1e-19


def _collect_points(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs = {}
    for a in record.get("attempts", []):
        if a.get("status") != "ok":
            continue
        c = a["c"]
        t = a["t"]
        # mean = a.get("mean")
        # if mean is None or mean <= 0:
        #     continue
        if (c, t) not in xs:
            xs[(c, t)] = []
        # ys.append(np.log(mean))
        data = a.get("times_sec", [])
        xs[(c, t)].extend(data)
    
    return xs


def fit_power_law(data: Dict[Tuple[float, float], List[float]]) -> Dict[str, Any]:
    # xs = np.array([[np.log(c), np.log(t)] for (c, t) in data.keys()])
    # ys = np.array([np.nanmean([np.log(d+EPS) for d in data[(c, t)]]) for (c, t) in data.keys()])
    xs = []
    ys = []
    for (c, t), yvals in data.items():
        for y in yvals:
            # xs.append([np.log(c), np.log(t)])
            # ys.append(np.log(y+EPS))
            xs.append([c, t])
            ys.append(y+EPS)
    xs = np.array(xs)
    ys = np.array(ys)

    plt.scatter(xs[:, 0]+np.random.randn(xs.shape[0])*0.01, ys)
    plt.savefig("scatter_c.png")
    plt.close()
    plt.scatter(xs[:, 1]+np.random.randn(xs.shape[0])*0.01, ys)
    plt.savefig("scatter_t.png")
    plt.close()

    if xs.shape[0] < 2:
        return {"fit_status": "insufficient_data"}
    try:
        model = HuberRegressor().fit(xs, ys)
        pred = model.predict(xs)
        resid = ys - pred
        r2 = 1.0 - (np.sum(resid**2) / np.sum((ys - ys.mean()) ** 2))
        # Example: predict for c in 0..99, t fixed at 50
        # x = np.column_stack([np.arange(0, 250), np.full(100, 50)])
        # Generate exactly 1000 well-distributed (c, t) pairs over the ranges
        num_samples = 1000
        grid_size = int(np.sqrt(num_samples))
        c_vals = np.linspace(1, 250, grid_size).astype(int)
        t_vals = np.linspace(1, 2048, grid_size).astype(int)
        cc, tt = np.meshgrid(c_vals, t_vals)
        x = np.column_stack([cc.ravel(), tt.ravel()])
        
        # Ensure we have exactly num_samples points
        if len(x) > num_samples:
            x = x[:num_samples]
        elif len(x) < num_samples:
            # If we have fewer points, pad with additional random samples
            remaining = num_samples - len(x)
            c_extra = np.random.randint(1, 251, remaining)
            t_extra = np.random.randint(1, 2049, remaining)
            x_extra = np.column_stack([c_extra, t_extra])
            x = np.vstack([x, x_extra])
        y = model.predict(x)
        plt.plot(np.arange(num_samples), y)
        plt.savefig("scatter_pred.png")
        plt.close()

        # Create 3D surface plot with multiple views in one figure
        fig = plt.figure(figsize=(20, 12))
        
        # Define views: (elevation, azimuth, title)
        views = [
            (30, 45, 'Isometric View'),
            (0, 0, 'X-axis View (along c)'),
            (0, 90, 'Y-axis View (along t)'),
            (90, 0, 'Z-axis View (top-down)')
        ]
        
        for i, (elev, azim, title) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            surf = ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap='viridis', alpha=0.8)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('c')
            ax.set_ylabel('t')
            ax.set_zlabel('predicted time')
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("scatter_pred_surface_multi_view.png", dpi=150, bbox_inches='tight')
        plt.close()
        # p = model.predict(x)
        # plt.scatter(xs[:, 0]+np.random.randn(xs.shape[0])*0.01, ys)
        # plt.plot(x[:, 0][:xs[:, 0].max()], p[:xs[:, 0].max()])
        # plt.savefig("scatter_c_pred.png")
        # plt.close()
        # plt.scatter(xs[:, 1]+np.random.randn(xs.shape[0])*0.01, ys)
        # plt.plot(x[:, 0], p)
        # plt.savefig("scatter_t_pred.png")
        # plt.close()

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
    # Group results by config_path first, then by key signature
    by_config_and_key: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    identifier_map: Dict[Tuple[str, str, str, int], str] = {}
    
    for data in datasets:
        config_path = data.get("run_metadata", {}).get("config_path", "unknown")
        
        for rec in data.get("results", []):
            # Create key signature from module, measure, config_index
            key_info = rec["key"]
            key_sig = (config_path, key_info["module"], key_info["measure"], key_info["config_index"])
            
            # Track identifier if it exists (successful runs have identifiers)
            if rec.get("identifier"):
                identifier_map[key_sig] = rec["identifier"]
            
            # Merge attempts for this key signature
            if key_sig not in by_config_and_key:
                by_config_and_key[key_sig] = {"attempts": []}
            by_config_and_key[key_sig]["attempts"].extend(rec.get("attempts", []))
    
    # Create final results with proper naming
    results = []
    for key_sig, data in by_config_and_key.items():
        config_path, module, measure, config_index = key_sig
        
        # Use identifier if available, otherwise fall back to module::measure::config_index format
        final_identifier = identifier_map.get(key_sig, f"{module}::{measure}::{config_index}")
        
        results.append({
            "identifier": final_identifier,
            "attempts": data["attempts"]
        })
    
    return results


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
        data = _collect_points(rec)
        fit = fit_power_law(data)
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
