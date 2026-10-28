"""Read-only EDF preflight; no signal loading, preprocessing, or calculation."""

from dataclasses import dataclass, replace
from fractions import Fraction
from importlib.metadata import version

from ._config import Configuration, WorkflowError, load_config
from ._edf import read_header, select_channels
from ._metrics import metric_input_problems, resolve_metrics
from ._windows import plan_windows
from ._prepare import reference_names
from ._reference import reference_plan
from ._bands import required_bands


@dataclass(frozen=True)
class PreflightReport:
    configuration: Configuration
    sampling_rate_hz: Fraction
    metrics: tuple
    recordings: tuple[dict, ...]

    @property
    def ready(self):
        return all(recording["status"] == "ready" for recording in self.recordings)

    def as_dict(self, *, include_ranges=True):
        cfg = self.configuration
        recordings = []
        for entry in self.recordings:
            data = dict(entry)
            windows = data.pop("windows")
            data["windows"] = windows.as_dict(include_ranges=include_ranges) if windows is not None else None
            recordings.append(data)
        return {"schema": "pyspi-preflight/1", "ready": self.ready,
                "pyspi_version": version("pyspi"), "configuration": str(cfg.source),
                "metric_config": str(cfg.metric_config),
                "execution": cfg.execution.copy(),
                "output_dtype": cfg.output_dtype,
                "output_dtype_overrides": cfg.output_dtype_overrides.copy(),
                "sampling_rate_hz": float(self.sampling_rate_hz),
                "window": {"length_seconds": float(cfg.window_seconds), "step_seconds": float(cfg.step_seconds)},
                "preprocessing": [operation for enabled, operation in
                                  ((cfg.detrend, "detrend"), (cfg.zscore, "zscore")) if enabled],
                "preparation": {"bandpass": {"low_hz": cfg.bandpass[0], "high_hz": cfg.bandpass[1]},
                                "notch": cfg.notch, "reference": cfg.reference,
                                "continuous_bands_hz": required_bands(self.metrics, cfg),
                                "resampling_method": "polyphase"},
                "dataset_directory": str(cfg.dataset_directory) if cfg.dataset_directory is not None else None,
                "output_directory": str(cfg.output_directory) if cfg.output_directory is not None else None,
                "metrics": [metric.as_dict() for metric in self.metrics], "recordings": recordings}


def preflight(config) -> PreflightReport:
    """Validate config/metric constructors and plan already matching-rate EDFs.

    A mismatch yields requires_resampling and no calculation sample ranges.
    Resampled ranges are confirmed against the prepared signal during execution.
    """
    cfg = config if isinstance(config, Configuration) else load_config(config)
    selected = []
    resolved_recordings = []
    for recording in cfg.recordings:
        header = read_header(recording.path)
        if cfg.dataset_directory is not None:
            from ._metadata import dataset_selection
            patient = cfg.dataset_directory / f"{recording.patient_id}.zarr"
            if not patient.resolve().is_relative_to(cfg.dataset_directory.resolve()):
                raise WorkflowError("patient path escapes dataset directory through a symlink")
            channels_from_metadata = dataset_selection(patient, header, recording.patient_id)
            if recording.channels is not None:
                explicit = select_channels(header, recording.channels)
                canonical = select_channels(header, channels_from_metadata)
                if [(s.index, n) for s, n in explicit] != [(s.index, n) for s, n in canonical]:
                    raise WorkflowError("inline channels conflict with authoritative patient metadata order")
            recording = replace(recording, channels=channels_from_metadata)
        resolved_recordings.append(recording)
        try:
            channels = select_channels(header, recording.channels)
        except WorkflowError as exc:
            raise WorkflowError(f"{recording.patient_id}/{recording.recording_id} ({header.path}): {exc}") from exc
        selected.append((recording, header, channels))
    cfg = replace(cfg, recordings=tuple(resolved_recordings))
    target = cfg.sampling_rate_hz
    # Validate the intended sample grid even when every source needs resampling.
    for seconds in (cfg.window_seconds, cfg.step_seconds):
        if (seconds * target).denominator != 1:
            raise WorkflowError("window length/step must each span a whole number of samples at the target sampling_rate_hz")
    metrics = resolve_metrics(cfg.metric_config, sampling_rate_hz=float(target))
    required_bands(metrics, cfg)
    from ._precision import resolve_dtypes
    resolve_dtypes(cfg, [metric.identifier for metric in metrics])
    recordings = []
    for recording, header, channels in selected:
        output_names = reference_names([name for _, name in channels], cfg.reference)
        reference = reference_plan(output_names, cfg.reference)
        available = reference[1].tolist() if reference is not None else [True] * len(output_names)
        matching = all(signal.rate == target for signal, _ in channels)
        windows = None
        if matching:
            samples = header.records * channels[0][0].samples_per_record
            windows = plan_windows(samples, target, cfg.window_seconds, cfg.step_seconds,
                                   recording=f"{recording.patient_id}/{recording.recording_id}")
        recordings.append({"patient_id": recording.patient_id, "recording_id": recording.recording_id,
                           "path": str(header.path), "status": "ready" if matching else "requires_resampling",
                           "duration_seconds": float(header.records * header.record_seconds),
                           "output_channels": output_names,
                           "reference_available": available,
                           "channels": [{"source_index": signal.index, "source_label": signal.label,
                                         "name": name, "native_sampling_rate_hz": float(signal.rate),
                                         "native_samples": header.records * signal.samples_per_record}
                                        for signal, name in channels], "windows": windows})
    # Refuse metrics whose parameters cannot produce an estimate for these
    # windows (for example a frequency band with no bins) before reading samples.
    window_samples = int(cfg.window_seconds * target)
    problems = {}
    for count in sorted({sum(r["reference_available"]) for r in recordings}):
        for problem in metric_input_problems(metrics, window_samples, count):
            problems.setdefault(problem, count)
    if problems:
        raise WorkflowError(
            f"{len(problems)} metric(s) cannot be estimated from {window_samples}-sample windows at "
            f"{float(target):g} Hz. Remove them from the metric configuration or change their parameters:\n  "
            + "\n  ".join(problems))
    return PreflightReport(cfg, target, metrics, tuple(recordings))
