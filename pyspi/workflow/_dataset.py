"""Dataset addresses and mappings; no independent patient metadata builder."""
import hashlib
import json
from pathlib import Path

from seeg_dataset import initialize_run, run_path


def dataset_address(cfg, recording, entry):
    from pyspi._naming import number

    patient = cfg.dataset_directory / f"{recording.patient_id}.zarr"
    window = f"{number(float(cfg.window_seconds))}s-step{number(float(cfg.step_seconds))}s"
    reference = entry.get("reference", cfg.reference)
    # Capture complete settings, including dictionaries for future references.
    settings = {"reference": reference, "detrend": cfg.detrend, "zscore": cfg.zscore,
                "bandpass": cfg.bandpass, "notch": cfg.notch,
                "sampling_rate_hz": float(cfg.sampling_rate_hz),
                "channels": entry["channels"], "output_channels": entry["output_channels"]}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()[:16]
    preprocessing = f"{'detrend' if cfg.detrend else 'nodetrend'}-{'zscore' if cfg.zscore else 'noscale'}"
    parts = {"recording": recording.recording_id, "window": window,
             "reference": reference["method"], "run": cfg.run_name or f"{preprocessing}-{signature}"}
    return patient, run_path(patient, **parts), parts


def output_channel_mapping(cfg, entry):
    names = [c["name"] for c in entry["channels"]]
    mapping = {name: [name] for name in entry["output_channels"] if name in names}
    if cfg.reference["method"] == "bipolar":
        mapping.update({name: [a, b] for name, a, b in
                        zip(cfg.reference["names"], cfg.reference["anode"], cfg.reference["cathode"])})
    return mapping


def initialize_dataset_run(cfg, recording, entry):
    patient, path, parts = dataset_address(cfg, recording, entry)
    initialize_run(patient, **parts, source_channels=[c["name"] for c in entry["channels"]],
                   output_channels=entry["output_channels"], channel_mapping=output_channel_mapping(cfg, entry))
    return path
