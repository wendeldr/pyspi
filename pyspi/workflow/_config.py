"""Strict, data-only configuration for EDF preflight."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml


class WorkflowError(ValueError):
    """Invalid configuration or input; no calculation should start."""


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently replacing configuration."""


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in result:
                raise WorkflowError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise WorkflowError("YAML keys must be strings") from exc
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return yaml.load(stream, Loader=UniqueLoader)
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise WorkflowError(f"cannot read configuration {path}: {exc}") from exc


def fields(value, allowed, where):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise WorkflowError(f"{where} must be a mapping with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise WorkflowError(f"{where}: unknown fields {sorted(unknown)}")
    return value


def text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{where} must be a nonempty string")
    return value


def positive_number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkflowError(f"{where} must be a positive finite number")
    try:
        result = Fraction(str(value))
    except ValueError as exc:
        raise WorkflowError(f"{where} must be a positive finite number") from exc
    if result <= 0:
        raise WorkflowError(f"{where} must be a positive finite number")
    return result


def _path(value, base, where):
    path = Path(text(value, where)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class Recording:
    path: Path
    patient_id: str
    recording_id: str
    channels: tuple | None


@dataclass(frozen=True)
class Configuration:
    source: Path
    recordings: tuple[Recording, ...]
    metric_config: Path
    sampling_rate_hz: Fraction | None
    window_seconds: Fraction
    step_seconds: Fraction
    detrend: bool
    zscore: bool
    output_directory: Path | None


def load_config(path) -> Configuration:
    """Resolve paths relative to the YAML file, without opening EDF samples."""
    path = Path(path).resolve()
    base = path.parent
    cfg = fields(read_yaml(path), {"version", "recordings", "edf_glob", "metrics",
                 "sampling_rate_hz", "window", "preprocessing", "output_directory"}, "workflow")
    if type(cfg.get("version")) is not int or cfg["version"] != 1:
        raise WorkflowError("workflow version must be 1")
    if ("recordings" in cfg) == ("edf_glob" in cfg):
        raise WorkflowError("provide exactly one of recordings or edf_glob")
    if "edf_glob" in cfg:
        import glob

        pattern = text(cfg["edf_glob"], "edf_glob")
        pattern = str(Path(pattern).expanduser())
        if not Path(pattern).is_absolute():
            pattern = str(base / pattern)
        raw = [{"path": p} for p in sorted(glob.glob(pattern))]
    else:
        raw = cfg["recordings"]
    if not isinstance(raw, list) or not raw:
        raise WorkflowError("recordings must select at least one EDF")
    recordings, identities, sources = [], set(), set()
    for index, entry in enumerate(raw):
        where = f"recordings[{index}]"
        fields(entry, {"path", "patient_id", "recording_id", "channels"}, where)
        source = _path(entry.get("path"), base, f"{where}.path")
        patient = text(entry.get("patient_id", source.stem), f"{where}.patient_id")
        recording = text(entry.get("recording_id", source.stem), f"{where}.recording_id")
        if (patient, recording) in identities:
            raise WorkflowError(f"duplicate patient/recording identity: {(patient, recording)!r}")
        if source in sources:
            raise WorkflowError(f"recording source selected more than once: {source}")
        identities.add((patient, recording))
        sources.add(source)
        channels = entry.get("channels")
        if channels is not None and (not isinstance(channels, list) or not channels):
            raise WorkflowError(f"{where}.channels must be a nonempty list or null")
        recordings.append(Recording(source, patient, recording,
                                    tuple(channels) if channels is not None else None))
    metrics = fields(cfg.get("metrics"), {"config", "preset"}, "metrics")
    if ("config" in metrics) == ("preset" in metrics):
        raise WorkflowError("metrics needs exactly one of config or preset")
    if "config" in metrics:
        metric_config = _path(metrics["config"], base, "metrics.config")
    else:
        preset = metrics["preset"]
        presets = {"all": "config.yaml", "fast": "fast_config.yaml",
                   "sonnet": "sonnet_config.yaml", "fabfour": "fabfour_config.yaml",
                   "retained185": "workflow/retained185.yaml"}
        if not isinstance(preset, str) or preset not in presets:
            raise WorkflowError(f"unknown metrics preset {preset!r}; choose {list(presets)}")
        metric_config = Path(__file__).resolve().parents[1] / presets[preset]
    window = fields(cfg.get("window", {}), {"length_seconds", "step_seconds"}, "window")
    length = positive_number(window.get("length_seconds", 0.5), "window.length_seconds")
    step = positive_number(window["step_seconds"], "window.step_seconds") if "step_seconds" in window else length
    preprocessing = fields(cfg.get("preprocessing", {}), {"detrend", "zscore"}, "preprocessing")
    detrend, zscore = preprocessing.get("detrend", True), preprocessing.get("zscore", True)
    if type(detrend) is not bool or type(zscore) is not bool:
        raise WorkflowError("preprocessing detrend/zscore must be booleans")
    rate = cfg.get("sampling_rate_hz")
    rate = positive_number(rate, "sampling_rate_hz") if rate is not None else None
    output = cfg.get("output_directory")
    output = _path(output, base, "output_directory") if output is not None else None
    return Configuration(path, tuple(recordings), metric_config, rate, length, step,
                         detrend, zscore, output)
