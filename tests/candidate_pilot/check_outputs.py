"""Verify real stored results across worker counts and standalone/family use."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from pyspi.workflow import open_run


def main():
    folder = Path(__file__).resolve().parent
    readers = {}
    for label in ("workflow-serial", "workflow-parallel", "kernel-serial", "kernel-parallel"):
        receipt = json.loads((folder / f"{label}.json").read_text())
        readers[label] = open_run(receipt["runs"][0]["path"])
    checks = []
    for left, right in (("workflow-serial", "workflow-parallel"), ("kernel-serial", "kernel-parallel"),
                        ("workflow-serial", "kernel-serial")):
        a, b = readers[left], readers[right]
        assert a.recording["channels"] == b.recording["channels"]
        for metric in b.metrics:
            x, y = a.load(metric), b.load(metric)
            assert set(a.statuses(metric)) == set(b.statuses(metric)) == {"success"}
            assert np.array_equal(np.isnan(x), np.isnan(y))
            assert np.array_equal(np.isinf(x), np.isinf(y))
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)
            checks.append({"left": left, "right": right, "metric": metric, "shape": list(x.shape),
                           "max_absolute_difference": float(np.nanmax(abs(x-y))), "masks_and_statuses_equal": True})
    (folder / "output_equivalence.json").write_text(json.dumps({"checks": checks}, indent=2)+"\n")
    print(len(checks), "real output comparisons passed")


if __name__ == "__main__":
    main()
