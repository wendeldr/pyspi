"""Read historical group-delay outputs without changing the input stores.

Count strict-upper-triangle entries, excluding the diagonal, across every
window in the selected recordings. Check shard/chunk presence so absent data
cannot silently count as finite fill values. This checks saved results; it
does not recompute group delay or establish the cause of a saved NaN.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import numcodecs.zarr3  # Register the original stores' numeric codecs.

from pyspi.storage._integrity import CheckedArray


def inspect(root, patients):
    results = []
    for patient in patients:
        store = root / patient / "0.5s.pspi"
        metrics = sorted(store.glob("gd_*"))
        if not metrics:
            raise ValueError(f"No group-delay outputs found in {store}")
        for metric in metrics:
            started = time.perf_counter()
            meta = json.loads((metric / "zarr.json").read_text())["attributes"]["pspi"]
            if meta["mode"] not in ("sym", "anti") or meta["arrays"] != ["upper"]:
                raise ValueError(f"Unexpected packing: {metric}")
            spec = json.loads((metric / "upper" / "zarr.json").read_text())
            reader = CheckedArray(metric / "upper", "upper", meta, spec)
            i, j = np.triu_indices(meta["n"])
            off_diagonal = i != j
            counts = dict(total=0, finite=0, nan=0, infinite=0, all_nan_windows=0)
            finite_windows_per_pair = np.zeros(off_diagonal.sum(), dtype=np.int64)
            for first in range(0, meta["windows"], reader.chunk_shape[0]):
                last = min(first + reader.chunk_shape[0], meta["windows"])
                block, available, issues = reader.read(np.arange(first, last))
                if not available.all() or issues:
                    raise OSError(f"Incomplete/unreadable source data: {metric}: {issues}")
                block = block[:, off_diagonal]
                finite = np.isfinite(block)
                nan = np.isnan(block)
                counts["total"] += block.size
                counts["finite"] += int(finite.sum())
                counts["nan"] += int(nan.sum())
                counts["infinite"] += int(np.isinf(block).sum())
                counts["all_nan_windows"] += int(nan.all(axis=1).sum())
                finite_windows_per_pair += finite.sum(axis=0)
            attrs = meta["attrs"]["source_zarr_attributes"]
            row = dict(patient=patient, metric=metric.name, channels=meta["n"],
                       windows=meta["windows"],
                       band_hz=[attrs["frequency_min_hz"], attrs["frequency_max_hz"]],
                       counts=counts,
                       pairs_with_any_finite_window=int((finite_windows_per_pair > 0).sum()),
                       finite_window_fraction_per_pair_quantiles=np.quantile(
                           finite_windows_per_pair / meta["windows"], [0, .25, .5, .75, 1]).tolist(),
                       read_seconds=time.perf_counter() - started)
            results.append(row)
            print(f"{patient} {row['band_hz']}: "
                  f"{100 * counts['nan'] / counts['total']:.4f}% NaN, "
                  f"{counts['all_nan_windows']}/{meta['windows']} all-NaN windows", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patients", nargs="+", default=["001", "009", "041", "096", "111"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "audit/briefing/group_delay_missingness.json")
    args = parser.parse_args()
    rows = inspect(args.root, args.patients)
    report = dict(source=str(args.root), source_access="read-only", patients=args.patients,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note="Every window; diagonal excluded. Saved NaNs do not identify their cause.",
                  results=rows)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
