"""Exact reader/writer for compact pyspi results with explicit completion tracking.

Layout (a Zarr v3 hierarchy; one directory per recording, one group per metric)::

    results.pspi/                 group, attrs: {"pspi_store": {...}}
      <metric>/                   group, attrs: {"pspi": {mode, n, windows, dtype, labels, ...}}
        upper/  or  full/         array (windows, values), shard files with crc32c-checked inner chunks
        residual/                 exact XOR correction when it compresses better
      pspi-manifest.json          declared metrics, layouts, and durable write states

Structure is detected per array at write time (never from metric names):
  sym       bit-exactly symmetric                -> diagonal + upper triangle
  anti      bit-exactly sign-flipped mirror      -> diagonal + upper triangle
  sym_res / anti_res  triangle + XOR residual (no scientific symmetry assumption)
  full      every entry

Reads return (n, n, windows) arrays, or (n, n) for results written as a single matrix.
Unavailable entries are NaN with a PartialReadWarning. ``return_availability=True``
also returns a same-shape boolean mask; stored NaNs remain available.
``verify(path)`` (or ``python -m pyspi.storage verify <path>``) reads every checksummed chunk, e.g.
after a copy, and names the metrics that fail so only those directories need copying again.
Local POSIX filesystems and a single writer per recording are supported.
"""

from __future__ import annotations

import math
import json
import os
import operator
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import numcodecs
import zarr
import zarr.codecs.numcodecs as zn
from zarr.codecs import Crc32cCodec, ZstdCodec

from ._packing import UINT, Packed, pack, unpack
from ._integrity import (
    SCHEMA, MANIFEST, MODES, CheckedArray, Issue, MetadataError, PartialReadWarning,
    StorageFormatError, StorageIntegrityError, describe, metric_name, metric_sources,
    read_json, read_manifest, sync_tree, validate_entry, write_manifest,
)

TARGET_CHUNK_VALUES = 256 * 1024          # values per independently decoded, crc-checked chunk
MAX_SHARD_VALUES = 128 * 1024 * 1024      # target per shard (1 GiB raw float64), minimum one window
READ_BLOCK_VALUES = 4 * 1024 * 1024       # full loads unpack this many stored values at a time


class ShardSizeWarning(UserWarning):
    """One window's stored payload exceeds the uncompressed shard-size target."""


def _positive_value_count(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _codec_chain(codec: str, level: int | None):
    if level is not None:
        if isinstance(level, (bool, np.bool_)):
            raise TypeError("level must be an integer")
        level = operator.index(level)
    if codec == "pcodec":
        if level is not None and not 0 <= level <= 12:
            raise ValueError("pcodec level must be between 0 and 12")
        return zn.PCodec(level=8 if level is None else level), [Crc32cCodec()]
    if codec == "zstd":
        return "auto", [ZstdCodec(level=3 if level is None else level), Crc32cCodec()]
    raise ValueError(f"unknown codec {codec!r} (use 'pcodec' or 'zstd')")


def _residual_smaller(codec: str, level: int | None = None):
    """Keep the residual form only if it encodes smaller than the full matrix."""
    encoder = (numcodecs.PCodec(level=8 if level is None else level) if codec == "pcodec"
               else numcodecs.Zstd(level=3 if level is None else level))
    def enc(a):
        return encoder.encode(np.ascontiguousarray(a).ravel())

    def ok(res: Packed, full: Packed) -> bool:
        return sum(len(enc(a)) for a in res.arrays().values()) < sum(len(enc(a)) for a in full.arrays().values())
    return ok


def _ensure_store(path):
    path = Path(path)
    if (path / MANIFEST).exists():
        manifest = read_manifest(path)
        if _root_issues(path, 0):
            raise MetadataError("root group schema differs from completion manifest")
        return manifest
    if (path / "zarr.json").exists():
        raise StorageFormatError("existing store has no completion manifest; automatic migration is disabled")
    root = zarr.open_group(str(path), mode="a")
    root.attrs["pspi_store"] = {"schema": SCHEMA}
    # The root group must be durable before the manifest can refer to it.
    sync_tree(path)
    manifest = {"schema": SCHEMA, "metrics": {}}
    write_manifest(path, manifest)
    return manifest


def _shape(values):
    single = values.ndim == 2
    if values.ndim not in (2, 3) or values.shape[0] != values.shape[1] or values.shape[0] < 1:
        raise ValueError(f"expected (n, n) or (n, n, windows), got {values.shape}")
    return single, values.shape[0], 1 if single else values.shape[2]


def _check_existing(path, manifest, metric, overwrite):
    if not overwrite and ((Path(path) / metric).exists() or
                          manifest["metrics"].get(metric, {}).get("state") == "complete"):
        raise FileExistsError(f"{metric} already stored")


def _root_issues(path, windows):
    try:
        header = read_json(Path(path) / "zarr.json")
        if header.get("node_type") != "group" or \
           header.get("attributes", {}).get("pspi_store", {}).get("schema") != SCHEMA:
            raise MetadataError("root group schema differs from completion manifest")
        return []
    except (MetadataError, AttributeError) as exc:
        return [Issue("root metadata", 0, windows, str(exc))]


def _json_mapping(value, name):
    """Validate and detach JSON metadata before any output is modified."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    def check_keys(item):
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError(f"{name} requires string JSON object keys")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)
    check_keys(value)
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _prepare_metric(metric, values, labels, attrs, codec, level, chunk, shard):
    metric_name(metric)
    values = np.asarray(values)
    single, n, windows = _shape(values)
    if values.dtype not in UINT:
        raise ValueError("values must have native float64 or float32 dtype; storage does not convert precision")
    _codec_chain(codec, level)
    _positive_value_count(chunk, "target_chunk_values")
    _positive_value_count(shard, "max_shard_values")
    if isinstance(labels, str):
        raise TypeError("labels must be an iterable of strings, not one string")
    labels = list(labels)
    if any(not isinstance(label, str) for label in labels):
        raise TypeError("labels must contain strings")
    meta = dict(schema=SCHEMA, mode=None, n=n, windows=windows, single_matrix=single,
                dtype=values.dtype.str, codec=codec, level=None if level is None else operator.index(level),
                arrays=[], labels=labels, attrs=_json_mapping({} if attrs is None else attrs, "attrs"))
    return values, meta


def write_metric(root, metric: str, values: np.ndarray, *, labels: Iterable[str] = (),
                 codec: str = "pcodec", level: int | None = None,
                 attrs: Mapping | None = None, overwrite: bool = False,
                 target_chunk_values: int = TARGET_CHUNK_VALUES, max_shard_values: int = MAX_SHARD_VALUES) -> dict:
    """Write one metric: ``values`` is (n, n) or (n, n, windows). Returns the stored metadata.

    Input must have native float32 or float64 dtype, which is preserved exactly.
    No dtype conversion is performed. ``attrs`` holds JSON-compatible user metadata.

    ``target_chunk_values`` and ``max_shard_values`` count uncompressed stored
    values. Shards target the latter size, with a one-window minimum. A window
    that exceeds it emits ShardSizeWarning and is stored intact in its own shard.
    This target is not a guarantee about compressed file bytes.
    """
    values, meta = _prepare_metric(metric, values, labels, attrs, codec, level,
                                    target_chunk_values, max_shard_values)
    single, n, windows = meta["single_matrix"], meta["n"], meta["windows"]
    if single:
        values = values[:, :, None]
    target_chunk_values = _positive_value_count(target_chunk_values, "target_chunk_values")
    max_shard_values = _positive_value_count(max_shard_values, "max_shard_values")
    data = np.ascontiguousarray(values)
    path = Path(root)
    manifest = _ensure_store(path)
    _check_existing(path, manifest, metric, overwrite)
    entry = {"state": "pending", "meta": meta, "array_metadata": {}}
    manifest["metrics"][metric] = entry
    write_manifest(path, manifest)
    p = pack(data, structure="exact", residual_ok=_residual_smaller(codec, level))
    meta.update(mode=p.mode, arrays=sorted(p.arrays()))
    entry["state"] = "writing"
    write_manifest(path, manifest)
    root = zarr.open_group(str(path), mode="a", attributes=None)
    if metric in root:
        del root[metric]
    g = root.create_group(metric, attributes={"pspi": meta})
    arrays = {}
    for key, arr in p.arrays().items():
        w, cols = arr.shape
        k = max(1, min(w, target_chunk_values // cols, max_shard_values // cols))
        per_shard = max(1, min(math.ceil(w / k), max_shard_values // (k * cols)))   # chunks per shard file
        if w and cols > max_shard_values:
            warnings.warn(
                f"{metric}/{key}: one window has {cols} stored values "
                f"({cols * arr.dtype.itemsize} uncompressed bytes), exceeding the "
                f"max_shard_values target of {max_shard_values} values "
                f"({max_shard_values * arr.dtype.itemsize} bytes); writing one window per shard",
                ShardSizeWarning, stacklevel=2,
            )
        ser, comp = _codec_chain(codec, level)
        kwargs = dict(shape=arr.shape, dtype=arr.dtype, chunks=(k, cols), shards=(per_shard * k, cols),
                      compressors=comp, filters=[], fill_value=0, config={"write_empty_chunks": True})
        if ser != "auto":
            kwargs["serializer"] = ser
        arrays[key] = g.create_array(key, **kwargs)
        entry["array_metadata"][key] = json.loads(json.dumps(arrays[key].metadata.to_dict()))
    # Publish all decoding declarations before the first payload write.
    write_manifest(path, manifest)
    for key, arr in p.arrays().items():
        arrays[key][:] = arr
    for key, expected in entry["array_metadata"].items():
        checked = CheckedArray(path / metric / key, key, meta, expected)
        for start in range(0, p.windows, _block_windows(arrays[key])):
            _, issues = checked.presence(np.arange(start, min(start + _block_windows(arrays[key]), p.windows)))
            if issues:
                raise StorageIntegrityError(f"{metric}: {describe(issues)}")
    sync_tree(path / metric)
    entry["state"] = "complete"
    try:
        write_manifest(path, manifest)
    except OSError:
        # If publication succeeded but its final directory flush failed, do not
        # leave a visible complete claim for an operation that reported failure.
        entry["state"] = "writing"
        try:
            write_manifest(path, manifest)
        except OSError:
            pass  # Preserve the original I/O failure if the filesystem is unavailable.
        raise
    return meta


def write_results(path, results: Mapping[str, np.ndarray], *, labels: Mapping[str, Iterable[str]] | None = None,
                  recording: Mapping | None = None, attrs: Mapping[str, Mapping] | None = None,
                  codec: str = "pcodec", level: int | None = None, overwrite: bool = False,
                  target_chunk_values: int = TARGET_CHUNK_VALUES,
                  max_shard_values: int = MAX_SHARD_VALUES) -> None:
    """Write metrics in mapping order, declaring the complete batch before writing.

    ``recording`` holds shared JSON metadata; ``attrs`` and ``labels`` are keyed
    by metric. Each input retains its own dtype and shape. A failed batch is
    not rolled back: the manifest identifies completed and unfinished metrics.
    """
    pending = {}
    prepared = {}
    options = dict(codec=codec, level=level, overwrite=overwrite,
                   target_chunk_values=target_chunk_values, max_shard_values=max_shard_values)
    _codec_chain(codec, level)
    _positive_value_count(target_chunk_values, "target_chunk_values")
    _positive_value_count(max_shard_values, "max_shard_values")
    if recording is not None:
        recording = _json_mapping(recording, "recording")
    for mapping, name in ((labels, "labels"), (attrs, "attrs")):
        if mapping is not None and (not isinstance(mapping, Mapping) or set(mapping) - set(results)):
            raise ValueError(f"{name} must be keyed only by supplied metrics")
    for metric, values in results.items():
        values, meta = _prepare_metric(metric, values, (labels or {}).get(metric, ()),
                                       (attrs or {}).get(metric), codec, level,
                                       target_chunk_values, max_shard_values)
        prepared[metric] = values
        pending[metric] = {"state": "pending", "meta": meta, "array_metadata": {}}
    manifest = _ensure_store(path)
    for metric in pending:
        _check_existing(path, manifest, metric, overwrite)
    manifest["metrics"].update(pending)
    write_manifest(path, manifest)
    if recording is not None:
        set_recording_info(path, recording)
    for metric, values in prepared.items():
        meta = pending[metric]["meta"]
        write_metric(path, metric, values, labels=meta["labels"], attrs=meta["attrs"], **options)


def set_recording_info(path, info: Mapping) -> None:
    """Store JSON-serialisable recording metadata on the root group (merged with existing)."""
    info = _json_mapping(info, "recording")
    _ensure_store(path)
    root = zarr.open_group(str(path), mode="a")
    store_attrs = dict(root.attrs.get("pspi_store", {"schema": SCHEMA}))
    store_attrs["recording"] = {**store_attrs.get("recording", {}), **dict(info)}
    root.attrs["pspi_store"] = store_attrs
    with (Path(path) / "zarr.json").open("rb") as stream:
        os.fsync(stream.fileno())


class Results:
    """Read access to a results store. Metadata reads are cheap; data is read per metric."""

    def __init__(self, path):
        self.path = Path(path)
        read_manifest(self.path)

    @property
    def metrics(self) -> list[str]:
        """Declared metrics in insertion order, including incomplete metrics."""
        return list(read_manifest(self.path)["metrics"])

    @property
    def recording(self) -> dict:
        root = zarr.open_group(str(self.path), mode="r")
        return dict(root.attrs.get("pspi_store", {}).get("recording", {}))

    def info(self, metric: str) -> dict:
        """Declared layout/user metadata and state (pending, writing, complete)."""
        entry = read_manifest(self.path)["metrics"][metric]
        return {**validate_entry(entry), "state": entry["state"]}

    def _sources(self, metric):
        entry = read_manifest(self.path)["metrics"][metric]
        meta, sources, issues = metric_sources(self.path / metric, entry)
        issues.extend(_root_issues(self.path, meta["windows"]))
        return meta, sources, issues

    @staticmethod
    def _selection(meta, windows):
        count = meta["windows"]
        if windows is None:
            return np.arange(count)
        # Validate original elements: array conversion can hide mixed booleans
        # or convert nanosecond timestamps into integers. Check bounds before
        # int64 conversion so large unsigned indices cannot wrap to negatives.
        dimension = np.ndim(windows)
        if dimension > 1:
            raise IndexError("window selection must be scalar or one-dimensional")
        if isinstance(windows, np.ndarray):
            selected = windows.reshape(-1)
        else:
            selected = (windows,) if dimension == 0 else windows
        indices = []
        for value in selected:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("windows must contain integer indices; booleans are not supported")
            try:
                index = operator.index(value)
            except TypeError as exc:
                raise TypeError("windows must contain integer indices; floats and other types are not supported") from exc
            if index >= count or index < -count:
                raise IndexError("window selection outside the declared result")
            indices.append(index + count if index < 0 else index)
        return np.asarray(indices, dtype=np.int64)

    def _arrays(self, metric, windows=None):
        meta, sources, issues = self._sources(metric)
        indices = self._selection(meta, windows)
        out = {key: np.empty((len(indices), source.shape[1]), dtype=source.dtype) for key, source in sources.items()}
        for key, source in sources.items():
            step = max(1, READ_BLOCK_VALUES // max(1, source.shape[1]))
            for start in range(0, len(indices), step):
                block, _, errors = source.read(indices[start:start + step])
                out[key][start:start + step] = block
                issues.extend(errors)
        if issues:
            message = f"{metric}: {describe(issues, limit=8)}"
            warnings.warn(message, PartialReadWarning, stacklevel=2)
            raise StorageIntegrityError(message)
        return meta, out

    def load(self, metric: str, windows=None, *, return_availability=False):
        """Return original entries; warn and insert NaN only where storage is unavailable.

        With return_availability=True, return (values, boolean availability) with
        matching dimensions. Stored NaNs/infinities are available, not missing.
        ``windows`` selects saved windows using an integer or a one-dimensional
        integer sequence; None loads all windows. Negative indices, duplicates,
        order, and empty selections are preserved. Floats and booleans raise
        TypeError. Out-of-range or multidimensional selections raise IndexError.
        """
        meta = self.info(metric)
        indices = self._selection(meta, windows)
        n, w = meta["n"], len(indices)
        data = np.empty((n, n, w), dtype=meta["dtype"])
        available = np.ones(data.shape, dtype=bool) if return_availability else None
        issues = []
        if w:
            meta, sources, issues = self._sources(metric)
            step = max(1, READ_BLOCK_VALUES // max((s.shape[1] for s in sources.values()), default=1))
            lower = np.tril_indices(n, -1) if (meta.get("mode") or "").endswith("_res") else None
            for start in range(0, w, step):
                selected = indices[start:start + step]
                arrays, good = {}, {}
                for key, source in sources.items():
                    arrays[key], good[key], errors = source.read(selected)
                    issues.extend(errors)
                if not arrays:
                    data[:, :, start:start + len(selected)] = np.nan
                    if available is not None:
                        available[:, :, start:start + len(selected)] = False
                    continue
                block = unpack(Packed(meta["mode"], n=n, windows=len(selected), dtype=meta["dtype"], **arrays))
                primary = good["full" if meta["mode"] == "full" else "upper"]
                bad = np.flatnonzero(~primary)
                block[:, :, bad] = np.nan
                if available is not None:
                    available[:, :, start + bad] = False
                if lower is not None:
                    bad_lower = np.flatnonzero(~(primary & good["residual"]))
                    block[lower[0][:, None], lower[1][:, None], bad_lower[None, :]] = np.nan
                    if available is not None:
                        available[lower[0][:, None], lower[1][:, None], (start + bad_lower)[None, :]] = False
                data[:, :, start:start + len(selected)] = block
            if issues:
                warnings.warn(f"{metric}: {describe(issues, limit=8)}", PartialReadWarning, stacklevel=2)
        scalar_window = windows is not None and np.ndim(windows) == 0
        if scalar_window or (meta["single_matrix"] and windows is None):
            data = data[:, :, 0]
            if available is not None:
                available = available[:, :, 0]
        return (data, available) if return_availability else data

def verify(path) -> dict[str, str]:
    """Read every stored chunk of every metric; return {metric: error} for failures (empty if clean).

    Data chunks carry CRC32C checksums, so damaged bytes raise instead of decoding to wrong values.
    The small zarr.json metadata files are not checksummed; damage there usually fails to parse.
    """
    bad = {}
    try:
        manifest = read_manifest(path)
    except (MetadataError, StorageFormatError) as exc:
        return {"__store__": str(exc)}
    root_errors = _root_issues(path, 0)
    if root_errors:
        bad["__store__"] = describe(root_errors)
    for metric, entry in sorted(manifest["metrics"].items()):
        try:
            meta, sources, issues = metric_sources(Path(path) / metric, entry)
            for source in sources.values():
                if source.error and meta["windows"] == 0:
                    issues.append(Issue(source.key, 0, 0, source.error))
                step = max(1, READ_BLOCK_VALUES // max(1, source.shape[1]))
                for start in range(0, meta["windows"], step):
                    _, _, errors = source.read(np.arange(start, min(start + step, meta["windows"])))
                    issues.extend(errors)
            if issues:
                bad[metric] = describe(issues)
        except (MetadataError, StorageFormatError, OSError) as exc:
            bad[metric] = f"{type(exc).__name__}: {exc}"
    return bad


def _block_windows(a) -> int:
    """Whole inner chunks' worth of windows holding about READ_BLOCK_VALUES stored values."""
    k = a.chunks[0]
    return k * max(1, READ_BLOCK_VALUES // (k * max(1, a.shape[1])))


def open_results(path) -> Results:
    return Results(path)
