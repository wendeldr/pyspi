"""Refresh only the approved ElasticNet correction and exact SGD replacement.

Independent model tests must pass first. Preserve every other numerical array,
record and historical provenance; update catalogue order for the renamed fit.
"""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]

import numpy as np
from regression import generate_goldens as generator, harness


OLD = "lmfit_SGDRegressor"
NEW = "lmfit_Ridge_alpha-0-0001_scale-mean"
CHANGED = {"lmfit_ElasticNet", NEW}


def main():
    harness.ensure_jvm()
    entries, by_id = harness.catalogue()
    sources = ["pyspi/_moments.py", "pyspi/_regression.py", "pyspi/statistics/misc.py",
               "pyspi/config.yaml", "audit/checks/probes/refresh_regression_goldens.py"]
    provenance = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": "Approved exact ElasticNet minimizer; exact per-sample L2 fit replaces SGD optimizer.",
        "validation": "tests/unit/test_regression_refactor.py: 53 passed before regeneration",
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
    }
    prepared = {}
    for size in ("small", "full", "preproc"):
        meta = harness.load_meta(size)
        old = harness.load_values(size)
        values = {key: value.copy() for key, value in old.items()}
        new_meta = copy.deepcopy(meta)
        assert OLD in meta["calculator_order"] and NEW not in meta["calculator_order"]
        new_meta["calculator_order"] = [NEW if key == OLD else key for key in meta["calculator_order"]]
        current_order = [entry["identifier"] for entry in entries]
        assert new_meta["calculator_order"] == current_order
        updated = []
        if size != "preproc":
            assert OLD in values
            del values[OLD]
            del new_meta["spis"][OLD]
            for key in sorted(CHANGED):
                value, record, problem = generator.run_one(by_id[key], harness.load_raw(meta["dataset"]), meta["dataset"], 1)
                assert problem is None and record["status"] == "ok", (key, record, problem)
                values[key] = value
                new_meta["spis"][key] = {**generator.catalogue_fields(by_id[key]), **record}
                updated.append(key)
        unrelated = set(old)-CHANGED-{OLD}
        assert all(values[key].tobytes() == old[key].tobytes() for key in unrelated)
        assert all(new_meta["spis"][key] == meta["spis"][key] for key in meta["spis"] if key not in CHANGED | {OLD})
        new_meta.setdefault("partial_regenerations", []).append({
            **provenance, "updated": updated, "removed": [OLD] if OLD in old else [],
            "unrelated_arrays_preserved": len(unrelated),
        })
        prepared[size] = values, new_meta
    for size, (values, meta) in prepared.items():
        destination = harness.golden_dir(size)
        # The preprocessing arrays have not changed at all; preserve its npz.
        if size != "preproc":
            np.savez_compressed(destination/"values.npz", **values)
        (destination/"meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True)+"\n")
        print(size, "updated", meta["partial_regenerations"][-1]["updated"], flush=True)


if __name__ == "__main__":
    main()
