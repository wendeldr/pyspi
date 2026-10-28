"""Compare tracked allocations for the three distance-correlation paths.

Warm imports/compilation first, then trace fresh eight-contact calculations.
Tracemalloc covers Python and registered NumPy/Numba allocations, not every
native allocator or process RSS. Instrumented times are not runtime benchmarks.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from hyppo.independence import Dcorr

from benchmark import apply_reference, data_for
from pyspi.data import Data
from pyspi.statistics.distance import DistanceCorrelation
from pyspi.workflow import load_config
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow._reference import reference_plan


def evaluate(x, method):
    if method == "native":
        with redirect_stdout(io.StringIO()):
            return DistanceCorrelation().multivariate(Data(x.copy(), normalise=False))
    result = np.full((len(x), len(x)), np.nan)
    for i in range(len(x)):
        for j in range(len(x)) if method == "old_wrapper" else range(i+1, len(x)):
            if i == j:
                continue
            metric = Dcorr(bias=False)
            metric.is_fast = method == "hyppo_fast"
            result[i, j] = metric.statistic(x[i, :, None], x[j, :, None])
    i, j = np.tril_indices(len(x), -1)
    result[i, j] = result[j, i]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(ROOT / "tests/candidate_pilot/configs/001-laplacian.yaml")
    with np.load(args.cache / "001-inputs.npz") as loaded:
        names = loaded["channel_names"].tolist()
        plan = reference_plan(names, cfg.reference)
        base = apply_reference(loaded["base"][0], names, cfg.reference, plan)
    data, _ = data_for(base, {}, np.flatnonzero(plan[1])[:8], cfg, ())
    x = data.to_numpy(realisation=0)
    receipt = {"patient": "001", "reference": "laplacian", "shape": list(x.shape),
               "definition": "bias-corrected normalized squared distance correlation",
               "limits": __doc__, "paths": {}}
    outputs = {}
    with numerical_threads(1):
        for method in ("old_wrapper", "hyppo_fast", "native"):
            evaluate(x[:2, :64], method)
            tracemalloc.start()
            outputs[method] = evaluate(x, method)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            receipt["paths"][method] = {"peak_tracked_bytes": peak,
                                        "remaining_tracked_bytes": current}
    for method, result in outputs.items():
        np.testing.assert_allclose(result, outputs["old_wrapper"], rtol=1e-10, atol=1e-12)
        receipt["paths"][method]["max_absolute_difference"] = float(np.nanmax(abs(result-outputs["old_wrapper"])))
    args.out.write_text(json.dumps(receipt, indent=2)+"\n")
    print(receipt["paths"])


if __name__ == "__main__":
    main()
