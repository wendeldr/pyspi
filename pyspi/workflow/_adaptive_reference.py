"""Patient-specific local reference with the reviewed, fixed fallback rules.

Contact numbers define shaft order, gaps and relative distances under the
equal-spacing assumption. Coordinates and signal correlations are not used.
"""

import re

import numpy as np

from ._config import WorkflowError


def build_adaptive_reference(names, groups, *, patient_id):
    """Resolve each contact's reference once, preserving the supplied axis."""
    names, groups = list(names), list(groups)
    if (not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names) or len(groups) != len(names)):
        raise WorkflowError("adaptive_local requires unique contacts and aligned shaft labels")
    shafts, numbers = {}, {}
    for i, (name, group) in enumerate(zip(names, groups)):
        if not isinstance(group, str) or not group:
            raise WorkflowError("adaptive_local requires nonempty electrodegroup labels")
        # Match the authoritative group exactly; no lexical sorting or EDF-name
        # normalization. Zero padding is allowed, duplicate numeric IDs are not.
        suffix = name[len(group):] if name.startswith(group) else ""
        if not re.fullmatch(r"[0-9]+", suffix) or int(suffix) < 1:
            raise WorkflowError(f"adaptive_local: {name!r} is not a numbered contact of shaft {group!r}")
        number = int(suffix)
        members = shafts.setdefault(group, {})
        if number in members:
            raise WorkflowError(f"adaptive_local: duplicate contact number {number} on shaft {group!r}")
        members[number] = i
        numbers[name] = number

    rules, ordered = {}, {}
    for group, members in shafts.items():
        sequence = sorted(members)
        indices = [members[number] for number in sequence]
        ordered[group] = [names[i] for i in indices]
        for position, (number, i) in enumerate(zip(sequence, indices)):
            neighbors = ([indices[position - 1]] if position else [])
            if position + 1 < len(indices):
                neighbors.append(indices[position + 1])
            gaps = {names[j]: abs(numbers[names[j]] - number) - 1 for j in neighbors}
            distances = {name: gap + 1 for name, gap in gaps.items()}
            if len(indices) <= 2:
                kind, reason = "car", "at_most_two_good_shaft_contacts"
                refs, weights = list(range(len(names))), [1 / len(names)] * len(names)
            elif any(gap >= 2 for gap in gaps.values()):
                kind, reason = "shaft_average", "two_or_more_missing_contacts_to_neighbor"
                refs, weights = indices, [1 / len(indices)] * len(indices)
            elif len(neighbors) == 1:
                kind, reason = "bipolar", "end_of_retained_shaft"
                refs, weights = neighbors, [1.]
            elif any(gaps.values()):
                kind, reason = "weighted_laplacian", "one_missing_contact_to_neighbor"
                left, right = (distances[names[j]] for j in neighbors)
                refs, weights = neighbors, [right / (left + right), left / (left + right)]
            else:
                kind, reason = "laplacian", "immediate_neighbors"
                refs, weights = neighbors, [.5, .5]
            rules[names[i]] = {
                "rule": kind, "reason": reason,
                "reference_weights": {names[j]: float(w) for j, w in zip(refs, weights)},
                "missing_contacts_to_neighbors": gaps, "neighbor_distances_in_spacings": distances,
            }
    return {"method": "adaptive_local", "definition": "pyspi-adaptive-local/1",
            "patient_id": patient_id, "spacing_assumption": "equal physical spacing within each shaft",
            "shafts": ordered, "contact_numbers": numbers,
            "rules": rules}


def resolve_reference(cfg, recording):
    """Resolve patient-specific adaptive or explicitly metadata-based shaft references."""
    reference = cfg.reference
    shaft_average = reference["method"] == "shaft_average"
    if reference["method"] != "adaptive_local" and not shaft_average:
        return reference
    if "patient_id" in reference:
        if reference.get("patient_id") != recording.patient_id:
            raise WorkflowError("resolved reference belongs to a different patient")
        return reference
    if shaft_average and reference.get("shafts") != "metadata":
        return reference
    if cfg.dataset_directory is None:
        raise WorkflowError(f"{reference['method']} requires dataset metadata; configure dataset.directory or data_lake_config")
    from seeg_dataset import open_patient, patient_contacts

    patient = cfg.dataset_directory / f"{recording.patient_id}.zarr"
    if not patient.resolve().is_relative_to(cfg.dataset_directory.resolve()):
        raise WorkflowError("patient path escapes dataset directory through a symlink")
    names = patient_contacts(patient, patient_id=recording.patient_id)
    info = open_patient(patient)["patient_info"]
    try:
        groups = info["electrodegroup"][:].tolist()
        if shaft_average:
            if len(groups) != len(names) or any(not isinstance(g, str) or not g for g in groups):
                raise WorkflowError("shaft_average requires aligned nonempty electrodegroup labels")
            shafts = {}
            for name, group in zip(names, groups):
                shafts.setdefault(group, []).append(name)
            return {"method": "shaft_average", "definition": "pyspi-metadata-shaft-average/1",
                    "patient_id": recording.patient_id, "shafts": shafts}
        return build_adaptive_reference(names, groups, patient_id=recording.patient_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(f"{recording.patient_id}: cannot resolve {reference['method']} metadata: {exc}") from exc


def adaptive_plan(names, reference):
    rules = reference.get("rules", {})
    if set(names) != set(rules):
        raise WorkflowError("adaptive reference contacts differ from retained channels; resolve patient metadata first")
    index = {name: i for i, name in enumerate(names)}
    weights = np.eye(len(names))
    for name, rule in rules.items():
        for source, weight in rule["reference_weights"].items():
            weights[index[name], index[source]] -= weight
    available = np.any(weights != 0, axis=1)
    return weights, available, ["available" if ok else "zero_reference_row" for ok in available]


def apply_adaptive_reference(source, names, reference):
    """Share CAR and shaft means; avoid a dense channels-by-recording product."""
    index = {name: i for i, name in enumerate(names)}
    out = np.empty_like(source)
    # Accumulate one mean at a time, without copying an entire shaft array.
    averages = {}
    for name, rule in reference["rules"].items():
        if rule["rule"] in ("car", "shaft_average"):
            key = tuple(rule["reference_weights"].items())
            averages.setdefault(key, []).append(name)
        else:
            i = index[name]
            out[i] = source[i]
            for neighbor, weight in rule["reference_weights"].items():
                out[i] -= weight * source[index[neighbor]]
    for refs, targets in averages.items():
        mean = np.zeros(source.shape[1], dtype=source.dtype)
        for name, weight in refs:
            mean += weight * source[index[name]]
        for name in targets:
            out[index[name]] = source[index[name]] - mean
    return out
