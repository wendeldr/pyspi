"""Read-only migration of legacy electrode order into a standalone contract."""

import json
from pathlib import Path
import re

from ._config import WorkflowError, safe_id
from ._edf import read_header


def _contact_key(value):
    # Only the export matches historical project names. The runner consumes
    # explicit EDF indices and never silently normalizes arbitrary selections.
    value = re.sub(r"^\s*EEG\s*", "", value, flags=re.IGNORECASE)
    value = value.translate(str.maketrans({"’": "'", "′": "'", "`": "'"}))
    return re.sub(r"\s+", "", value).casefold()


def export_selection(store, edf, destination, *, patient_id=None, recording_id=None):
    """Export just ``patient_info/electrode``, never connectivity or exclusions.

    Opens the legacy Zarr with mode='r'. Every contact must match exactly one
    actual EDF header signal after documented prefix/space/prime normalization.
    No numeric contact coercion, anatomical rule, or provenance flag is used.
    The destination must be new and outside the source store.
    """
    import zarr

    store, destination = Path(store).resolve(), Path(destination).resolve()
    if destination.is_relative_to(store):
        raise WorkflowError("metadata export must be outside the source Zarr")
    header = read_header(edf)
    try:
        root = zarr.open_group(str(store), mode="r")
        contacts = root["patient_info/electrode"][:].tolist()
    except Exception as exc:
        raise WorkflowError(f"cannot read electrode order from {store}: {exc}") from exc
    if not contacts or any(not isinstance(c, str) or not c.strip() for c in contacts):
        raise WorkflowError("legacy electrode array must contain nonempty strings")
    lookup = {}
    for signal in header.signals:
        if not signal.annotation:
            lookup.setdefault(_contact_key(signal.label), []).append(signal)
    channels, used = [], set()
    for contact in contacts:
        matches = lookup.get(_contact_key(contact), [])
        if len(matches) != 1:
            raise WorkflowError(f"contact {contact!r}: {len(matches)} EDF matches; supply an explicit manual mapping")
        signal = matches[0]
        if signal.index in used:
            raise WorkflowError(f"duplicate exported electrode/source: {contact!r}")
        used.add(signal.index)
        channels.append({"index": signal.index, "name": contact})
    document = {"schema": "pyspi-channel-selection/1", "recordings": [
        {"path": str(header.path),
         "patient_id": safe_id(patient_id if patient_id is not None else store.stem, "patient_id"),
         "recording_id": safe_id(recording_id if recording_id is not None else header.path.stem, "recording_id"),
         "channels": channels, "source_metadata": {"path": str(store), "array": "patient_info/electrode"}}]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return document


def dataset_selection(patient, header, patient_id):
    """Consume built metadata without recreating QC, exclusion, or sorting rules."""
    from seeg_dataset import open_patient, patient_contacts

    contacts = patient_contacts(patient, patient_id=patient_id)
    info = open_patient(patient)["patient_info"]
    label_map = None
    if "edf_history" in info:
        history = info["edf_history"]
        names = history["contact"][:].tolist()
        labels = history["edf_label"][:].tolist()
        used = history["used"][:].tolist()
        if len({len(names), len(labels), len(used)}) != 1 or any(type(v) is not bool for v in used):
            raise WorkflowError("invalid EDF history columns in patient metadata")
        label_map = {}
        for name, label, selected in zip(names, labels, used):
            if not selected:
                continue
            if name in label_map:
                raise WorkflowError(f"duplicate included contact in EDF history: {name!r}")
            label_map[name] = label
        if set(label_map) != set(contacts):
            raise WorkflowError("included EDF history contacts differ from patient_info/electrode")
    channels, used_indices = [], set()
    for contact in contacts:
        if label_map is not None:
            matches = [s for s in header.signals if not s.annotation and s.label == label_map[contact]]
        else:
            # Older built metadata lacks an EDF history table. Match labels with
            # the existing explicit export convention, but never guess ambiguity.
            matches = [s for s in header.signals if not s.annotation and _contact_key(s.label) == _contact_key(contact)]
        if len(matches) != 1:
            raise WorkflowError(f"metadata contact {contact!r}: {len(matches)} EDF matches")
        signal = matches[0]
        if signal.index in used_indices:
            raise WorkflowError(f"metadata contacts map to the same EDF signal: {contact!r}")
        channels.append({"index": signal.index, "name": contact})
        used_indices.add(signal.index)
    return tuple(channels)
