"""Interpolate measured candidate costs over the historical 72-person selection.

Three measured patient sizes bracket all CAR counts. Local counts use verified
coordinate layouts only; an unresolved patient's upper scenario keeps every
retained contact. Ranges propagate the sampled min/max window times, not a
statistical confidence interval or a guarantee on unmeasured patients.
"""

import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def interpolate(points, count):
    points = sorted(points)
    if count <= points[0][0]:
        return points[0][1]
    if count >= points[-1][0]:
        return points[-1][1]
    for (lo, left), (hi, right) in zip(points, points[1:]):
        if lo <= count <= hi:
            fraction = math.log(count/lo)/math.log(hi/lo)
            return math.exp(math.log(left)+fraction*math.log(right/left))


def main():
    measurements = [json.loads((ROOT / f"{pid}-results.json").read_text()) for pid in ("004", "001", "111")]
    cohort = json.loads((ROOT / "cohort_counts.json").read_text())["patients"]
    result = {"patients": len(cohort), "windows_per_patient": 600, "metrics": {}, "families": {},
        "unverified_local_patients": [p["patient_id"] for p in cohort if p["laplacian_contacts"] is None],
        "limits": "Log-log interpolation between measured 40/118/249-contact patients (local sizes differ). "
                  "This is a planning model, not measured cohort runtime. Historical retained counts may change. "
                  "Min/max scenarios use sampled window extremes, not confidence bounds. Output/continuous "
                  "preparation/startup are separate. Shared increments depend on the recorded family order."}
    for metric in measurements[0]["metric_order"]:
        item = {}
        for reference in ("car", "laplacian"):
            refs = [r["references"][reference] for r in measurements]
            points = [(r["usable_contacts"], r["standalone"][metric]["seconds_per_window"]) for r in refs]
            lower = [(r["usable_contacts"], min(w["seconds"] for w in r["standalone"][metric]["windows"])) for r in refs]
            upper = [(r["usable_contacts"], max(w["seconds"] for w in r["standalone"][metric]["windows"])) for r in refs]
            incremental = [(r["usable_contacts"], median(w["increments_seconds"][metric] for w in r["family"])) for r in refs]
            known = [p[f"{reference}_contacts"] for p in cohort if p[f"{reference}_contacts"] is not None]
            missing = [p["car_contacts"] for p in cohort if p[f"{reference}_contacts"] is None]
            total = 600*sum(interpolate(points, n) for n in known)
            item[reference] = {"patients_with_counts": len(known), "mean_seconds_per_window": total/(600*len(known)),
                "mean_seconds_600": total/len(known), "cohort_seconds": total,
                "sampled_window_range_seconds": [600*sum(interpolate(curve, n) for n in known) for curve in (lower, upper)],
                "incremental_cohort_seconds": 600*sum(interpolate(incremental, n) for n in known),
                "unverified_additional_seconds_upper_scenario": 600*sum(interpolate(points, n) for n in missing),
                "measured_points": points}
        item["combined_known_seconds"] = sum(item[r]["cohort_seconds"] for r in ("car", "laplacian"))
        result["metrics"][metric] = item
    for reference in ("car", "laplacian"):
        points = [(r["references"][reference]["usable_contacts"],
                   r["references"][reference]["family_projection"]["seconds_per_window"]) for r in measurements]
        result["families"][reference] = {"cohort_seconds": 600*sum(interpolate(points, p[f"{reference}_contacts"])
            for p in cohort if p[f"{reference}_contacts"] is not None), "measured_points": points}
    profiles = {}
    for family, prefix in (("all_candidates", "workflow"), ("kernel_only", "kernel")):
        profiles[family] = {}
        for mode in ("serial", "parallel"):
            receipt = json.loads((ROOT / f"{prefix}-{mode}.json").read_text())
            session = receipt["runs"][0]["session"]
            # Extrapolate stages once per recording rather than multiplying full
            # preparation/startup by every sampled window.
            windows = receipt["windows"]
            fixed = (session["preparation_seconds"] + session["invocation_setup_seconds"]
                     + session.get("parallel_input_setup_seconds", 0.))
            variable = session["calculation_wall_seconds"] + session["output_seconds"]
            if mode == "serial":
                variable += session["window_preprocessing_seconds"]
            profiles[family][mode] = {"observed_windows": windows, "observed_elapsed_seconds": receipt["elapsed_seconds"],
                "preparation_seconds": session["preparation_seconds"], "output_seconds": session["output_seconds"],
                "calculation_wall_seconds": session["calculation_wall_seconds"],
                "metric_elapsed_seconds": session["metric_elapsed_seconds"],
                "stage_projection_seconds_600_same_patient": fixed+600*variable/windows,
                "process_tree_memory": receipt["process_tree_memory"]}
        profiles[family]["observed_calculation_speedup"] = (profiles[family]["serial"]["calculation_wall_seconds"] /
                                                          profiles[family]["parallel"]["calculation_wall_seconds"])
    result["workflow_profiles"] = profiles
    (ROOT / "timing_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print({k: {"car_hours": round(v["car"]["cohort_seconds"]/3600, 2),
               "local_verified_hours": round(v["laplacian"]["cohort_seconds"]/3600, 2)} for k, v in result["metrics"].items()})
    print({k: v["observed_calculation_speedup"] for k, v in profiles.items()})


if __name__ == "__main__":
    main()
