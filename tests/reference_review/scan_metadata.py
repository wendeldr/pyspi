"""Read-only cohort validation of the approved contact-preserving reference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pyspi.workflow import _adaptive_reference
from pyspi.workflow._reference import reference_details, reference_plan


def scan(directory):
    patients, errors = [], []
    rules, sizes = Counter(), Counter()
    for path in sorted(directory.glob("*.zarr")):
        try:
            ref = _adaptive_reference.resolve_reference(
                SimpleNamespace(reference={"method": "adaptive_local"}, dataset_directory=directory),
                SimpleNamespace(patient_id=path.stem))
            names = list(ref["contact_numbers"])
            plan = reference_plan(names, ref)
            counts = Counter(r["rule"] for r in ref["rules"].values())
            rules.update(counts)
            shaft_sizes = {shaft: len(contacts) for shaft, contacts in ref["shafts"].items()}
            sizes.update(shaft_sizes.values())
            source = {"names": names, "shafts": ref["shafts"]}
            patients.append({"patient_id": path.stem, "contacts": len(names),
                "shaft_sizes": shaft_sizes, "rules": dict(counts), "available": int(plan[1].sum()),
                "metadata_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                "proportional_pairs": reference_details(names, plan)["proportional_pairs"]})
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"patient_id": path.stem, "error": str(exc)})
    return {"source": str(directory), "patients": len(patients), "shafts": sum(sizes.values()),
            "contacts": sum(p["contacts"] for p in patients), "rules": dict(rules),
            "shaft_size_histogram": dict(sorted(sizes.items())), "errors": errors,
            "implementation_sha256": hashlib.sha256(Path(_adaptive_reference.__file__).read_bytes()).hexdigest(),
            "per_patient": patients}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.dataset.resolve()):
        parser.error("receipt must be outside the source dataset")
    result = scan(args.dataset)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_patient"}, indent=2))
    raise SystemExit(bool(result["errors"]))
