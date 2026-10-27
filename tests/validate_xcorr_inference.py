"""Simulation audit of approximate xcorr inference (not a calibration proof).

Run: .venv/bin/python -B tests/validate_xcorr_inference.py
Uses the production workspace on independent pairs, with 3072 burn-in samples
before each 1024-sample window. Counts a detection if ANY searched lag survives.
"""

import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy import signal, stats

from pyspi._xcorr import CrossCorrelationWorkspace


def main():
    rng = np.random.default_rng(601)
    results = []
    repetitions, batch, samples, lag = 1000, 100, 1024, 256
    for kind in [0, .5, .9, .95, .99, -.8, "mixed", "bandpass", "narrowband", "coupled"]:
        count = 0
        for _ in range(repetitions // batch):
            raw = rng.normal(size=(batch, 2, 4096))
            if isinstance(kind, (int, float)):
                raw = signal.lfilter([1], [1, -kind], raw, axis=-1)
            elif kind == "mixed":
                raw[:, 0] = signal.lfilter([1], [1, -.8], raw[:, 0], axis=-1)
                raw[:, 1] = signal.lfilter([1], [1, .8], raw[:, 1], axis=-1)
            elif kind in ("bandpass", "narrowband"):
                band = [2, 80] if kind == "bandpass" else [10, 20]
                raw = signal.sosfilt(signal.butter(4, band, btype="bandpass", fs=2048, output="sos"), raw, axis=-1)
            else:
                raw = signal.lfilter([1], [1, -.8], raw, axis=-1)
                raw[:, 1, 20:] += .8 * raw[:, 0, :-20]
            # One workspace holds the batch; only the original independent
            # pairs are evaluated. No inference is performed across pairs.
            source = raw[..., -samples:].reshape(2 * batch, samples, 1)
            work = CrossCorrelationWorkspace(source)
            i = np.arange(0, 2 * batch, 2)
            j = i + 1
            curves = work._curves(i, j, lag)
            _, selected, status, _ = work._summaries(i, j, curves, lag, .05)
            assert np.isin(status, ["ok", "no_significant_lags"]).all()
            count += int(np.count_nonzero(selected))
        interval = stats.binomtest(count, repetitions).proportion_ci()
        row = {"kind": kind, "fraction_any_lag_selected": count / repetitions,
               "pairs": repetitions, "detected": count,
               "binomial_95_percent_interval": [interval.low, interval.high]}
        results.append(row)
        print(row, flush=True)
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "seed": 601, "samples": samples, "sampling_rate_hz": 2048,
        "maximum_lag_samples": lag, "alpha": .05, "burn_in_samples": 3072,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("pyspi/_xcorr.py", "tests/validate_xcorr_inference.py")},
        "interpretation": "Independent stationary simulated pairs except coupled. Any-lag rate is per pair, not per channel matrix. Approximate inference, often conservative for high autocorrelation; these simulations do not establish validity for arbitrary EEG or nonstationary signals.",
        "measurements": results,
    }
    (ROOT / "tests/xcorr_inference_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
