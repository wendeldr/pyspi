import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Optional


def predict_time(model: Dict[str, Any], c: int, t: int) -> Optional[Dict[str, float]]:
    if not model or model.get("fit_status") != "ok":
        return None
    if model.get("type") != "power_law":
        return None
    p = model["params"]
    log_k = p["log_k"]
    alpha = p["alpha"]
    beta = p["beta"]
    log_time = log_k + alpha * math.log(c) + beta * math.log(t)
    time_est = math.exp(log_time)
    resid_std = model.get("metrics", {}).get("resid_std_log")
    if resid_std is None:
        return {"estimate": time_est, "ci_low": time_est, "ci_high": time_est}
    # 95% log-normal band
    ci_low = math.exp(log_time - 1.96 * resid_std)
    ci_high = math.exp(log_time + 1.96 * resid_std)
    return {"estimate": time_est, "ci_low": ci_low, "ci_high": ci_high}


def main():
    ap = argparse.ArgumentParser(description="Estimate runtimes from saved models")
    ap.add_argument("--models", required=True, help="Path to models JSON")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--config", required=False, help="Optional config YAML to enumerate labels (unused for prediction)")
    args = ap.parse_args()

    with open(args.models, "r", encoding="utf-8") as f:
        models = json.load(f).get("models", {})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["measure", "estimated_seconds", "ci_low", "ci_high"])
        for ident, model in models.items():
            pr = predict_time(model, args.c, args.t)
            if pr is None:
                continue
            w.writerow([
                ident,
                f"{pr['estimate']:.6f}",
                f"{pr['ci_low']:.6f}",
                f"{pr['ci_high']:.6f}",
            ])


if __name__ == "__main__":
    main()
