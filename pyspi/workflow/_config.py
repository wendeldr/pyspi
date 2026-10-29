"""Strict, data-only configuration for the serial EEG workflow."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml


from seeg_dataset import DatasetError as WorkflowError


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


def safe_id(value, where):
    value = text(value, where)
    if value in (".", "..") or any(c in value for c in ("/", "\\", "\0")):
        raise WorkflowError(f"{where} must be a single filesystem name")
    return value


def reference_config(value, base):
    if isinstance(value, str):
        value = {"method": value}
    fields(value, {"method", "channels", "mapping", "anode", "cathode", "names", "drop_refs", "forward", "shafts"}, "reference")
    method = value.get("method")
    options = {"none": set(), "car": set(), "channels": {"channels"},
               "mapping": {"mapping"}, "bipolar": {"anode", "cathode", "names", "drop_refs"},
               "rest": {"forward"}, "laplacian": {"shafts"}, "shaft_average": {"shafts"},
               "adaptive_local": set()}
    if not isinstance(method, str) or method not in options:
        raise WorkflowError(f"reference.method must be one of {list(options)}")
    fields(value, {"method"} | options[method], "reference")
    result = dict(value)
    if method in ("laplacian", "shaft_average"):
        shafts = value.get("shafts")
        if not isinstance(shafts, dict) or not shafts:
            raise WorkflowError("reference.shafts must map shaft names to physically ordered contact lists")
        seen = set()
        for shaft, contacts in shafts.items():
            text(shaft, "reference shaft name")
            if not isinstance(contacts, list) or not contacts:
                raise WorkflowError("each shaft must contain a nonempty ordered contact list")
            for contact in contacts:
                text(contact, "reference shaft contact")
                if contact in seen:
                    raise WorkflowError(f"contact {contact!r} occurs more than once in reference.shafts")
                seen.add(contact)
    for key in ("channels", "anode", "cathode", "names"):
        if key in options[method] and key != "names":
            if not isinstance(value.get(key), list) or not value[key]:
                raise WorkflowError(f"reference.{key} must be a nonempty list")
        if key in value:
            if not isinstance(value[key], list) or not value[key]:
                raise WorkflowError(f"reference.{key} must be a nonempty list")
            result[key] = [text(v, f"reference.{key}") for v in value[key]]
    if method == "mapping":
        mapping = value.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise WorkflowError("reference.mapping must be a nonempty mapping")
        for target, refs in mapping.items():
            text(target, "reference.mapping target")
            if not isinstance(refs, list) or not refs:
                raise WorkflowError("each reference.mapping value must be a nonempty list")
            for ref in refs:
                text(ref, "reference.mapping channel")
    if method == "bipolar":
        if len(result["anode"]) != len(result["cathode"]):
            raise WorkflowError("bipolar anode/cathode lengths must agree")
        result.setdefault("names", [f"{a}-{c}" for a, c in zip(result["anode"], result["cathode"])])
        if len(result["names"]) != len(result["anode"]) or len(set(result["names"])) != len(result["names"]):
            raise WorkflowError("bipolar names must be unique and match the pair count")
        result.setdefault("drop_refs", True)
        if type(result["drop_refs"]) is not bool:
            raise WorkflowError("reference.drop_refs must be a boolean")
    if method == "rest":
        result["forward"] = str(_path(value.get("forward"), base, "reference.forward"))
    return result


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
    sampling_rate_hz: Fraction
    window_seconds: Fraction
    step_seconds: Fraction
    detrend: bool
    zscore: bool
    output_directory: Path | None
    bandpass: tuple[float, float]
    notch: dict
    reference: dict
    run_name: str | None
    output_batch_mib: float
    metadata: Path | None
    execution: dict
    output_dtype: str
    output_dtype_overrides: dict
    dataset_directory: Path | None = None


def load_config(path) -> Configuration:
    """Resolve paths relative to the YAML file, without opening EDF samples."""
    path = Path(path).resolve()
    base = path.parent
    cfg = fields(read_yaml(path), {"version", "recordings", "edf_glob", "metrics",
                 "edf_directory", "metadata", "sampling_rate_hz", "window", "preprocessing",
                 "bandpass", "notch", "reference", "run_name", "output_batch_mib", "output_directory",
                 "execution", "output_dtype", "output_dtype_overrides", "dataset"}, "workflow")
    if type(cfg.get("version")) is not int or cfg["version"] != 1:
        raise WorkflowError("workflow version must be 1")
    if sum(key in cfg for key in ("recordings", "edf_glob", "edf_directory")) != 1:
        raise WorkflowError("provide exactly one of recordings, edf_glob or edf_directory")
    if "edf_directory" in cfg:
        directory = _path(cfg["edf_directory"], base, "edf_directory")
        if not directory.is_dir():
            raise WorkflowError(f"edf_directory is not a directory: {directory}")
        raw = [{"path": str(p)} for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() == ".edf"]
    if "edf_glob" in cfg:
        import glob

        pattern = text(cfg["edf_glob"], "edf_glob")
        pattern = str(Path(pattern).expanduser())
        if not Path(pattern).is_absolute():
            pattern = str(base / pattern)
        raw = [{"path": p} for p in sorted(glob.glob(pattern))]
    elif "edf_directory" not in cfg:
        raw = cfg["recordings"]
    if not isinstance(raw, list) or not raw:
        raise WorkflowError("recordings must select at least one EDF")
    metadata = _path(cfg["metadata"], base, "metadata") if "metadata" in cfg else None
    metadata_entries = {}
    if metadata is not None:
        document = fields(read_yaml(metadata), {"schema", "recordings"}, "metadata")
        if document.get("schema") != "pyspi-channel-selection/1" or not isinstance(document.get("recordings"), list):
            raise WorkflowError("metadata must use pyspi-channel-selection/1 with a recordings list")
        for entry in document["recordings"]:
            fields(entry, {"path", "patient_id", "recording_id", "channels", "source_metadata"}, "metadata recording")
            source = _path(entry.get("path"), metadata.parent, "metadata.path")
            if source in metadata_entries:
                raise WorkflowError(f"duplicate metadata source {source}")
            if not isinstance(entry.get("channels"), list) or not entry["channels"]:
                raise WorkflowError("metadata recording must supply a nonempty channels list")
            metadata_entries[source] = entry
    recordings, identities, sources = [], set(), set()
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            entry = {"path": entry}
        where = f"recordings[{index}]"
        fields(entry, {"path", "patient_id", "recording_id", "channels"}, where)
        source = _path(entry.get("path"), base, f"{where}.path")
        if metadata is not None:
            if source not in metadata_entries:
                raise WorkflowError(f"no matching metadata for {source}")
            matched = metadata_entries[source]
            for key in ("patient_id", "recording_id", "channels"):
                if key in entry and key in matched and entry[key] != matched[key]:
                    raise WorkflowError(f"{where}.{key} conflicts with metadata")
            entry = {**matched, **entry}
        patient = safe_id(entry.get("patient_id", source.stem), f"{where}.patient_id")
        recording = safe_id(entry.get("recording_id", source.stem), f"{where}.recording_id")
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
                   "reviewed": "workflow/reviewed.yaml"}
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
    rate = positive_number(cfg.get("sampling_rate_hz"), "sampling_rate_hz (required)")
    band = fields(cfg.get("bandpass"), {"low_hz", "high_hz"}, "bandpass (required)")
    low = positive_number(band.get("low_hz"), "bandpass.low_hz (required)")
    high = positive_number(band.get("high_hz"), "bandpass.high_hz (required)")
    if not low < high < rate / 2:
        raise WorkflowError("bandpass must satisfy 0 < low_hz < high_hz < sampling_rate_hz/2")
    notch = fields(cfg.get("notch", {}), {"line_frequency_hz", "width_hz", "transition_hz", "enabled"}, "notch")
    notch = {"line_frequency_hz": float(positive_number(notch.get("line_frequency_hz", 60), "notch.line_frequency_hz")),
             "width_hz": float(positive_number(notch.get("width_hz", 1), "notch.width_hz")),
             "transition_hz": float(positive_number(notch.get("transition_hz", 1), "notch.transition_hz")),
             "enabled": notch.get("enabled", True)}
    if type(notch["enabled"]) is not bool:
        raise WorkflowError("notch.enabled must be a boolean")
    if notch["width_hz"] + notch["transition_hz"] >= notch["line_frequency_hz"]:
        raise WorkflowError("notch stopbands/transitions must be separated between harmonics")
    reference = reference_config(cfg.get("reference"), base)
    output = cfg.get("output_directory")
    output = _path(output, base, "output_directory") if output is not None else None
    dataset = cfg.get("dataset")
    dataset_directory = None
    if dataset is not None:
        fields(dataset, {"directory", "data_lake_config"}, "dataset")
        if ("directory" in dataset) == ("data_lake_config" in dataset):
            raise WorkflowError("dataset needs exactly one of directory or data_lake_config")
        if "directory" in dataset:
            dataset_directory = _path(dataset["directory"], base, "dataset.directory")
        else:
            import tomllib
            lake_config = _path(dataset["data_lake_config"], base, "dataset.data_lake_config")
            try:
                with lake_config.open("rb") as stream:
                    lake = tomllib.load(stream)
                dataset_directory = _path(lake.get("dataset_dir"), lake_config.parent,
                                          "data-lake dataset_dir (required)")
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise WorkflowError(f"cannot read data-lake config: {exc}") from exc
        if metadata is not None:
            raise WorkflowError("dataset mode reads patient metadata directly; omit selection metadata")
        if output is not None and output != dataset_directory:
            raise WorkflowError("output_directory conflicts with the dataset destination; omit it")
        output = dataset_directory
    execution = fields(cfg.get("execution", {}),
                       {"workers", "threads_per_worker", "spectral_cache_mib", "scratch_directory"}, "execution")
    scratch = (_path(execution["scratch_directory"], base, "execution.scratch_directory")
               if "scratch_directory" in execution else None)
    execution = {"workers": execution.get("workers", 1),
                 "threads_per_worker": execution.get("threads_per_worker", 1),
                 "spectral_cache_mib": float(positive_number(
                     execution.get("spectral_cache_mib", 256), "execution.spectral_cache_mib"))}
    for key in ("workers", "threads_per_worker"):
        if type(execution[key]) is not int or execution[key] < 1:
            raise WorkflowError(f"execution.{key} must be a positive integer")
    if scratch is not None:
        execution["scratch_directory"] = str(scratch)
    from ._precision import dtype_name
    output_dtype = dtype_name(cfg.get("output_dtype"), "output_dtype (required)")
    overrides = cfg.get("output_dtype_overrides", {})
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not k for k in overrides):
        raise WorkflowError("output_dtype_overrides must map metric identifiers to dtypes")
    overrides = {key: dtype_name(value, f"output_dtype_overrides.{key}")
                 for key, value in overrides.items()}
    return Configuration(path, tuple(recordings), metric_config, rate, length, step,
                         detrend, zscore, output, (float(low), float(high)), notch, reference,
                         safe_id(cfg["run_name"], "run_name") if "run_name" in cfg else None,
                         float(positive_number(cfg.get("output_batch_mib", 64), "output_batch_mib")), metadata,
                         execution, output_dtype, overrides, dataset_directory)
