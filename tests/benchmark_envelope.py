"""Complete six-variant PEC workload against the original MNE delegate path."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mne_connectivity as mnec
import numpy as np

from pyspi.data import Data
from pyspi.statistics.misc import PowerEnvelopeCorrelation


SPECS = [(False, False, False), (True, False, False), (False, True, False),
         (True, True, False), (True, False, True), (True, True, True)]


def run(raw, baseline):
    with contextlib.redirect_stdout(io.StringIO()):
        data = Data(raw.copy(), normalise=False)
        results = []
        for orth, log, absolute in SPECS:
            if baseline:
                result = mnec.envelope_correlation(data.to_numpy(realisation=0)[None],
                    orthogonalize="pairwise" if orth else False, log=log, absolute=absolute,
                    verbose=False).combine("mean").get_data(output="dense").squeeze()
                np.fill_diagonal(result, np.nan)
            else:
                result = PowerEnvelopeCorrelation(orth=orth, log=log, absolute=absolute).multivariate(data)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT/"tests/envelope_refactor_results.json")
    args = parser.parse_args()
    raw = np.random.default_rng(731).normal(size=(118, 1024))
    raw[1] = .8*np.roll(raw[0], 3)+.2*raw[1]
    for baseline in (False, True):
        run(raw[:4, :128], baseline)
    seconds = {"before": [], "after": []}
    outputs = {}
    for repetition in range(args.repeats):
        for baseline in ((True, False) if repetition % 2 == 0 else (False, True)):
            start = time.perf_counter()
            values = run(raw, baseline)
            elapsed = time.perf_counter()-start
            label = "before" if baseline else "after"
            seconds[label].append(elapsed)
            outputs[label] = values
            print(f"{label} repeat {repetition+1}: {elapsed:.6f}s", flush=True)
    # Typical nondegenerate values should agree; independent definitions and
    # corrected degeneracy handling are validated in test_envelope.py.
    differences = []
    for old, new in zip(outputs["before"], outputs["after"]):
        np.testing.assert_allclose(new, old, atol=2e-11, rtol=1e-9)
        differences.append(float(np.nanmax(abs(old-new))))
    median = {key: float(np.median(value)) for key, value in seconds.items()}
    report = dict(shape=list(raw.shape), seed=731, specifications=SPECS, raw_seconds=seconds,
        median_seconds=median, speedup=median["before"]/median["after"],
        extrapolated_sequential_seconds={key: {"one_window": value, "600_windows": value*600,
                                              "600_windows_x_69": value*600*69} for key, value in median.items()},
        maximum_absolute_differences=differences,
        protocol="Fresh Data/cache each run; six variants sharing the same window; original MNE delegate including combine(mean) versus refactor. Three alternating repeats by default, one numerical-library thread; no I/O. Synthetic 118x1024 timing is not a patient runtime prediction.",
        environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mne", "mne-connectivity")},
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                       ("pyspi/_envelope.py", "pyspi/statistics/misc.py", "tests/benchmark_envelope.py")})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(median_seconds=median, speedup=report["speedup"])), flush=True)


if __name__ == "__main__":
    main()
