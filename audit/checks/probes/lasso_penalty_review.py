"""Check Lasso nondegeneracy on label-blind, preprocessed EEG windows.

This is a parameter feasibility check, not clinical validation or penalty
optimization. It reuses the reviewed EDF preprocessing and patient/contact
selection. No patient observations or labels are written to the report.
"""

import contextlib
import datetime
import io
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyspi-regression-matplotlib")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from pyspi.data import Data
from spectral_window_smoothing import read_eeg


def main():
    root = Path("/home/dan/data/IEEG/connectivity")
    patients = []
    for patient in ("001", "041", "096"):
        print("Preparing", patient, flush=True)
        raw, metadata = read_eeg(root / f"{patient}.zarr", contacts=12)
        starts = np.linspace(0, raw.shape[1]//1024-1, 12, dtype=int)*1024
        results = {alpha: [] for alpha in (.1, .05, .01)}
        for start in starts:
            with contextlib.redirect_stdout(io.StringIO()):
                data = Data(raw[:, start:start+1024], detrend=True, normalise=True)
            x = data.to_numpy(realisation=0)
            x -= x.mean(axis=1, keepdims=True)
            n = x.shape[1]
            covariance = x @ x.T/n
            variance = np.diag(covariance)
            offdiag = ~np.eye(len(x), dtype=bool)
            for alpha in results:
                coefficient = np.sign(covariance)*np.maximum(abs(covariance)-alpha, 0)/variance[:, None]
                mse = variance[None, :] - 2*coefficient*covariance + coefficient**2*variance[:, None]
                values = mse[offdiag]
                results[alpha].append({
                    "window_index": int(start//1024),
                    "nonzero_slope_fraction": float(np.mean(coefficient[offdiag] != 0)),
                    "mse_quantiles": np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
                    "mse_range": float(np.ptp(values)),
                })
        patients.append({"metadata": metadata, "penalties": {str(a): r for a, r in results.items()}})
        print(patient, {a: [r["nonzero_slope_fraction"] for r in rows] for a, rows in results.items()}, flush=True)
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": str(root), "patients": patients,
        "protocol": "12 deterministic random retained contacts per patient; full retained-contact CAR, notch/bandpass/resample from reviewed reader; 12 evenly spaced nonoverlapping 0.5s windows at 2048Hz, detrend then ddof=1 normalize.",
        "definition": "Exact one-predictor Lasso with intercept and alpha in MSE/2 + alpha*abs(slope). Independent covariance formula.",
        "limits": "Selected patients/contacts/windows; no EZ labels, prediction analysis, or penalty optimality claim. Runtime is measured separately at 118 channels.",
    }
    destination = ROOT / "audit/briefing/lasso_penalty_review.json"
    destination.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
