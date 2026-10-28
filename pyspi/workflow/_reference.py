"""Explicit contact topology and linear-reference provenance.

Shaft lists describe physical order, including contacts omitted from the EEG
selection. Never compress a gap to construct a Laplacian. Unavailable outputs
keep their identities; the workflow masks them before any metric is evaluated.
"""

import numpy as np

from ._config import WorkflowError


def reference_plan(names, reference):
    """Return linear weights and availability for CAR/explicit shaft references."""
    method = reference["method"]
    if method not in ("car", "laplacian", "shaft_average"):
        return None
    n = len(names)
    weights = np.zeros((n, n))
    reasons = ["available"] * n
    index = {name: i for i, name in enumerate(names)}
    if method == "car":
        weights[:] = -1 / n
        weights[np.diag_indices(n)] += 1
    else:
        shafts = reference["shafts"]
        locations = {name: (shaft, position) for shaft, contacts in shafts.items()
                     for position, name in enumerate(contacts)}
        missing = set(names) - locations.keys()
        if missing:
            raise WorkflowError(f"retained contacts absent from explicit shaft layout: {sorted(missing)}")
        for i, name in enumerate(names):
            shaft, position = locations[name]
            contacts = shafts[shaft]
            if method == "laplacian":
                if position == 0 or position == len(contacts) - 1:
                    reasons[i] = "shaft_endpoint"
                    continue
                neighbors = contacts[position - 1], contacts[position + 1]
                if any(neighbor not in index for neighbor in neighbors):
                    reasons[i] = "missing_adjacent_contact"
                    continue
                weights[i, i] = 1
                weights[i, [index[c] for c in neighbors]] = -.5
            else:
                retained = [index[c] for c in contacts if c in index]
                if len(retained) < 2:
                    reasons[i] = "insufficient_shaft_contacts"
                    continue
                weights[i, retained] = -1 / len(retained)
                weights[i, i] += 1
    available = np.array([r == "available" for r in reasons])
    # With only one CAR channel, the resulting signal is identically zero.
    available &= np.any(weights != 0, axis=1)
    for i in np.flatnonzero(~available):
        if reasons[i] == "available":
            reasons[i] = "zero_reference_row"
    return weights, available, reasons


def reference_details(names, plan):
    """Compact, JSON-safe membership and shared-source diagnostics.

Null correlations assume independent, equal-variance source signals. They
describe mixing geometry, not estimates of physiological connectivity.
"""
    weights, available, reasons = plan
    support = weights != 0
    gram = weights @ weights.T
    norm = np.sqrt(np.diag(gram))
    pairs, proportional = [], []
    for i in np.flatnonzero(available):
        for j in range(i + 1, len(names)):
            if not available[j]:
                continue
            shared = int(np.count_nonzero(support[i] & support[j]))
            if not shared:
                continue
            correlation = float(gram[i, j] / (norm[i] * norm[j]))
            pairs.append({"channels": [names[i], names[j]], "shared_sources": shared,
                          "independent_equal_variance_correlation": correlation})
            if np.isclose(abs(correlation), 1, rtol=0, atol=32 * np.finfo(float).eps):
                proportional.append([names[i], names[j]])
    return {"source_channels": list(names), "available": available.tolist(),
            "unavailable_reasons": {names[i]: reasons[i] for i in np.flatnonzero(~available)},
            "coefficients": {name: {names[j]: float(weights[i, j])
                                    for j in np.flatnonzero(support[i])}
                             for i, name in enumerate(names)},
            "shared_source_pairs": pairs, "proportional_pairs": proportional,
            "null_assumption": "independent equal-variance source signals; geometry only"}


def apply_shaft_reference(source, names, reference, plan):
    """Apply local references in O(channels * samples), without dense mixing."""
    weights, available, _ = plan
    out = np.zeros_like(source)
    if reference["method"] == "laplacian":
        for i in np.flatnonzero(available):
            neighbors = np.flatnonzero(weights[i] < 0)
            out[i] = source[i] - (source[neighbors[0]] + source[neighbors[1]]) * .5
    else:
        index = {name: i for i, name in enumerate(names)}
        for contacts in reference["shafts"].values():
            retained = [index[c] for c in contacts if c in index]
            if len(retained) < 2:
                continue
            # Accumulate one recording-length vector, not a copied shaft array.
            mean = np.zeros(source.shape[1], dtype=source.dtype)
            for i in retained:
                mean += source[i] / len(retained)
            for i in retained:
                out[i] = source[i] - mean
    return out


def available_indices(details, count):
    availability = details.get("reference_geometry", {}).get("available", [True] * count)
    if len(availability) != count:
        raise WorkflowError("reference availability disagrees with output channel count")
    return np.flatnonzero(availability)
