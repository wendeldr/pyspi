"""Read only contact metadata from the existing exploration's mmap cache.

No EEG samples, EZ labels, outcomes, or feature values enter this calculation.
Coordinate layouts are checked for complete numbering and forward shaft order;
unverified layouts are reported rather than silently compressed across gaps.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def canonical(name):
    return re.sub(r"\s+", "", re.sub(r"^(EEG|ECG)\s+", "", name)).replace("-", "~")


def check_layout(names, path):
    rows = list(csv.DictReader(path.open()))
    coordinates = {}
    for row in rows:
        name = canonical(row["Electrode"])
        if name in coordinates:
            raise ValueError(f"duplicate coordinate name {name}")
        coordinates[name] = np.array([float(row[k]) for k in ("X", "Y", "Z")])
    missing = sorted(set(names)-coordinates.keys())
    if missing:
        raise ValueError(f"retained coordinates missing: {missing}")
    groups = {}
    for name, point in coordinates.items():
        match = re.fullmatch(r"(.+?)(\d+)", name)
        if match:
            groups.setdefault(match[1], {})[int(match[2])] = (name, point)
        elif name in names:
            raise ValueError(f"retained contact has no shaft/number identity: {name}")
    available, unavailable = [], {}
    for shaft, items in groups.items():
        kept = [name for name, _ in items.values() if name in names]
        if not kept:
            continue
        numbers = sorted(items)
        if numbers != list(range(min(numbers), max(numbers)+1)):
            raise ValueError(f"coordinate numbering has a gap on {shaft}")
        points = np.stack([items[i][1] for i in numbers])
        if not np.isfinite(points).all():
            raise ValueError(f"nonfinite coordinates on {shaft}")
        if len(points) > 1:
            axis = points[-1]-points[0]
            if not (np.diff(points, axis=0) @ axis > 0).all():
                raise ValueError(f"coordinate order reverses on {shaft}")
        for n, (name, _) in items.items():
            if name not in names:
                continue
            if n == min(numbers) or n == max(numbers):
                unavailable[name] = "shaft_endpoint"
            elif items[n-1][0] not in names or items[n+1][0] not in names:
                unavailable[name] = "missing_adjacent_contact"
            else:
                available.append(name)
    if len(available)+len(unavailable) != len(names):
        raise ValueError("layout did not account for every contact")
    return {"laplacian_contacts": len(available), "unavailable": unavailable,
            "coordinate_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    document = torch.load(args.cache, weights_only=True, mmap=True)
    contacts = [{"patient_id": f"{int(p['pid']):03}", "contacts": [canonical(n) for n in p["contacts"]]}
                for p in document["patients"]]
    receipt = {"source": str(args.cache), "source_bytes": args.cache.stat().st_size,
               "contact_metadata_sha256": hashlib.sha256(json.dumps(contacts, sort_keys=True).encode()).hexdigest(),
               "patients": [], "limits": "Historical retained contact selection; verify against the eventual run's selection. "
                 "Coordinates confirm layout consistency, not localization accuracy. Only verified layouts supply local-reference counts."}
    for patient in contacts:
        pid, names = patient["patient_id"], patient["contacts"]
        entry = {"patient_id": pid, "car_contacts": len(names)}
        path = args.coordinates / f"{pid}_LabelXYZAtlases.csv"
        try:
            entry.update(check_layout(names, path), layout_status="verified")
        except (OSError, ValueError, KeyError) as exc:
            entry.update(layout_status="needs_review", reason=str(exc), laplacian_contacts=None)
        receipt["patients"].append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2)+"\n")
    print({"patients": len(contacts), "verified_layouts": sum(p["layout_status"] == "verified" for p in receipt["patients"])})
    print([(p["patient_id"], p["reason"]) for p in receipt["patients"] if p["layout_status"] != "verified"])


if __name__ == "__main__":
    main()
