"""Continuous MNE preparation, separate from Calculator and its window caches."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ._config import WorkflowError
from ._edf import read_header, select_channels
from ._reference import apply_shaft_reference, reference_plan, reference_details


FIR = dict(method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
           filter_length="auto")


def harmonics(rate, notch):
    if not notch["enabled"]:
        return []
    # MNE's transition_hz is split equally on the two stopband edges.
    margin = (notch["width_hz"] + notch["transition_hz"]) / 2
    line = notch["line_frequency_hz"]
    return [k * line for k in range(1, int(float(rate) / 2 / line) + 1)
            if margin < k * line < float(rate) / 2 - margin]


def reference_names(names, reference):
    """Validate retained-channel references and determine the resulting axes."""
    method = reference["method"]
    if method in ("laplacian", "shaft_average"):
        reference_plan(names, reference)
    required = []
    if method == "channels":
        required = reference["channels"]
    elif method == "mapping":
        required = list(reference["mapping"]) + [r for refs in reference["mapping"].values() for r in refs]
    elif method == "bipolar":
        required = reference["anode"] + reference["cathode"]
        if any(a == c for a, c in zip(reference["anode"], reference["cathode"])):
            raise WorkflowError("a bipolar pair must use different retained channels")
        if set(reference["names"]) & set(names):
            raise WorkflowError("bipolar output names must differ from retained source names")
    missing = set(required) - set(names)
    if missing:
        raise WorkflowError(f"reference channels absent from retained selection: {sorted(missing)}")
    if method != "bipolar":
        return list(names)
    retained = [n for n in names if not reference["drop_refs"] or n not in required]
    return retained + reference["names"]


@dataclass
class PreparedRecording:
    data: np.ndarray
    channel_names: list[str]
    details: dict
    elapsed_seconds: float


def prepare_recording(recording, cfg, *, channels=None):
    """Load only retained signals, notch at each native rate, then align rates.

    Mixed-rate EDFs are opened once per native-rate group so MNE cannot silently
    FFT-resample them before our native-rate notches/polyphase conversion.
    Preparation holds continuous selected signals in RAM; result buffering is
    bounded separately. MNE converts EDF physical units to SI (normally volts).
    """
    import mne

    started = perf_counter()
    header = read_header(recording.path)
    selection = select_channels(header, recording.channels) if channels is None else channels
    names = [name for _, name in selection]
    output_names = reference_names(names, cfg.reference)
    target = float(cfg.sampling_rate_hz)
    # Header-only open establishes MNE's public unique-name mapping, including
    # duplicate labels selected through original EDF signal indices.
    with mne.io.read_raw_edf(str(recording.path), preload=False, infer_types=False,
                             stim_channel=None, exclude_after_unique=True, verbose="ERROR") as probe:
        ordinary = [s for s in header.signals if not s.annotation]
        if len(probe.ch_names) != len(ordinary):
            raise WorkflowError("MNE channel header disagrees with EDF signal identities")
        reader_names = {s.index: name for s, name in zip(ordinary, probe.ch_names)}
    expected_samples = max(1, round(float(header.records * header.record_seconds) * target))
    prepared = np.empty((len(selection), expected_samples), dtype=np.float64)
    groups = {}
    for position, (signal, _) in enumerate(selection):
        groups.setdefault(signal.rate, []).append((position, signal))
    group_details = []
    for rate, members in groups.items():
        include = [reader_names[s.index] for _, s in members]
        with mne.io.read_raw_edf(str(recording.path), include=include, preload=False,
                                 infer_types=False, stim_channel=None, exclude_after_unique=True,
                                 verbose="ERROR") as raw:
            if not np.isclose(raw.info["sfreq"], float(rate), rtol=0, atol=1e-10):
                raise WorkflowError("MNE loaded signals at a rate different from their native EDF rate")
            values = raw.get_data(picks=include)
        if values.shape[1] != header.records * members[0][1].samples_per_record:
            raise WorkflowError("MNE sample count disagrees with EDF header")
        if not np.isfinite(values).all():
            raise WorkflowError(f"{recording.path}: retained EEG signals contain nonfinite samples")
        freqs = harmonics(rate, cfg.notch)
        if freqs:
            values = mne.filter.notch_filter(values, float(rate), freqs,
                        notch_widths=cfg.notch["width_hz"], trans_bandwidth=cfg.notch["transition_hz"],
                        n_jobs=1, copy=False, verbose="ERROR", **FIR)
        if rate != cfg.sampling_rate_hz:
            values = mne.filter.resample(values, up=target, down=float(rate), method="polyphase",
                                        n_jobs=1, verbose="ERROR")
        if values.shape[1] != expected_samples:
            raise WorkflowError("resampled native-rate groups have inconsistent sample counts")
        prepared[[p for p, _ in members]] = values
        group_details.append({"native_sampling_rate_hz": float(rate), "notch_frequencies_hz": freqs,
                              "channels": [names[p] for p, _ in members],
                              "resampled": rate != cfg.sampling_rate_hz})
        del values
    low, high = cfg.bandpass
    # Design once explicitly to record effective taps/transitions, then apply
    # the same public MNE FIR design to the entire continuous selected signal.
    taps = mne.filter.create_filter(None, target, low, high, verbose="ERROR", **FIR)
    prepared = mne.filter.filter_data(prepared, target, low, high, copy=False,
                                      n_jobs=1, verbose="ERROR", **FIR)
    raw = mne.io.RawArray(prepared, mne.create_info(names, target, "eeg"), copy="auto", verbose="ERROR")
    method = cfg.reference["method"]
    plan = reference_plan(names, cfg.reference)
    if method == "car":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif method in ("channels", "mapping"):
        raw.set_eeg_reference(cfg.reference[method], projection=False, verbose="ERROR")
    elif method == "rest":
        forward = mne.read_forward_solution(cfg.reference["forward"], verbose="ERROR")
        raw.set_eeg_reference("REST", forward=forward, projection=False, verbose="ERROR")
    elif method == "bipolar":
        raw = mne.set_bipolar_reference(raw, cfg.reference["anode"], cfg.reference["cathode"],
                    ch_name=cfg.reference["names"], drop_refs=cfg.reference["drop_refs"],
                    copy=False, verbose="ERROR")
    if raw.ch_names != output_names:
        raise WorkflowError("MNE rereferencing produced an unexpected channel order")
    # get_data() gives an owned public array, independent of the temporary Raw.
    data = raw.get_data()
    raw.close()
    if method in ("laplacian", "shaft_average"):
        data = apply_shaft_reference(data, names, cfg.reference, plan)
        # Zero placeholders are never sent to metrics: the explicit mask is
        # applied even when window z-scoring is disabled.
    if not np.isfinite(data).all():
        raise WorkflowError("EEG preparation produced nonfinite samples")
    details = {"sampling_rate_hz": target, "samples": data.shape[1], "units": "SI (EDF physical units)",
               "native_rate_groups": group_details, "reference": cfg.reference,
               "reference_pool": names if method == "car" else None,
               "bandpass": {"low_hz": low, "high_hz": high, **FIR, "taps": len(taps),
                            "lower_transition_hz": min(max(low * .25, 2), low),
                            "upper_transition_hz": min(max(high * .25, 2), target / 2 - high)},
               "notch": {**cfg.notch, **FIR}, "resampling_method": "polyphase"}
    if plan is not None:
        details["reference_geometry"] = reference_details(names, plan)
    return PreparedRecording(data, output_names, details, perf_counter() - started)
