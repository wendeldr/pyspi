"""Bounded synthetic production-writer batch-size benchmark (no estimators)."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyspi.storage import open_results, write_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new temporary output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for n in (50, 128, 300):
        rng = np.random.default_rng(2026+n)
        source = {}
        for metric in range(12):
            value = rng.normal(size=(n,n,8))
            if metric % 3 != 0:
                upper = np.triu_indices(n, 1)
                value[upper[1], upper[0]] = value[upper] * (-1 if metric % 3 == 2 else 1)
            for window in range(8):
                np.fill_diagonal(value[:,:,window], np.nan)
            source[f"metric_{metric}"] = value
        for batch_size in (1,4,8):
            root = args.output / f"n{n}-b{batch_size}"
            started = perf_counter()
            for start in range(0,8,batch_size):
                write_results(root / f"{start}", {m: a[:,:,start:start+batch_size] for m,a in source.items()})
            elapsed = perf_counter() - started
            started = perf_counter()
            first = next(iter(source))
            restored = np.concatenate([open_results(root / f"{start}").load(first)
                                       for start in range(0,8,batch_size)], axis=2)
            read_elapsed = perf_counter() - started
            np.testing.assert_array_equal(restored.view(np.uint64), source[first].view(np.uint64))
            files = [p for p in root.rglob("*") if p.is_file()]
            records.append({"channels": n, "metrics":12, "windows":8, "batch_windows":batch_size,
                            "write_seconds_including_packing_and_durability":elapsed,
                            "selective_metric_read_seconds":read_elapsed, "files":len(files),
                            "bytes":sum(p.stat().st_size for p in files), "exact_selected_metric":True})
    receipt = {"schema":"pyspi-bounded-output-benchmark/1", "records":records,
               "limits":"One repeat, synthetic directed/symmetric/antisymmetric Gaussian matrices; no calculator, preprocessing, or cohort-runtime claim."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
