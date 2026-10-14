"""Local-filesystem completion records and checked reads for pyspi storage.

One writer owns a recording path. Zarr remains responsible for numeric decoding;
the small shard-index check prevents absent inner chunks becoming fill values.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import google_crc32c
import numpy as np
import zarr

SCHEMA = "pspi-store/1.0"
MANIFEST = "pspi-manifest.json"
MODES = {"full": ("full",), "sym": ("upper",), "anti": ("upper",),
         "sym_res": ("residual", "upper"), "anti_res": ("residual", "upper")}
# Catch decoder/I/O failures only at the read boundary, never reconstruction bugs.
READ_ERRORS = (OSError, ValueError, RuntimeError)


class StorageFormatError(ValueError):
    """The supplied store/layout does not implement this completion contract."""


class MetadataError(ValueError):
    """Required metadata is unreadable or inconsistent."""


class StorageIntegrityError(OSError):
    """Required stored data is unavailable or a write is not complete."""


class PartialReadWarning(UserWarning):
    """A requested result includes missing, unreadable, or uncommitted output."""


def metric_name(name):
    if not isinstance(name, str) or not name or name in (".", "..", "zarr.json", MANIFEST, "__store__") or \
       "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("metric must be a single nonempty path component")
    return name


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise MetadataError(f"{path}: {exc}") from exc


def read_manifest(path):
    path = Path(path) / MANIFEST
    if not path.is_file():
        raise StorageFormatError(f"{path}: completion manifest missing; legacy/unregistered store is unsupported")
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise StorageFormatError(f"{path}: expected schema {SCHEMA}")
    if not isinstance(value.get("metrics"), dict):
        raise MetadataError(f"{path}: metrics must be a mapping")
    for name in value["metrics"]:
        try:
            metric_name(name)
        except ValueError as exc:
            raise MetadataError(f"{path}: invalid metric name") from exc
    return value


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_manifest(path, manifest):
    """Flush before atomic publication; the recording directory is single-writer."""
    path = Path(path)
    text = json.dumps(manifest, allow_nan=False, separators=(",", ":"))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path, prefix=".pspi-manifest-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path / MANIFEST)
        sync_directory(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sync_tree(path):
    """Persist one metric's files and directory entries before its completion record."""
    path = Path(path)
    directories = {path, path.parent}
    for item in path.rglob("*"):
        if item.is_dir():
            directories.add(item)
        elif item.is_file():
            with item.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)


def validate_entry(entry):
    if not isinstance(entry, dict) or entry.get("state") not in ("pending", "writing", "complete"):
        raise MetadataError("invalid metric write state")
    meta = entry.get("meta")
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise MetadataError("invalid metric schema/metadata")
    if type(meta.get("n")) is not int or meta["n"] < 1 or \
       type(meta.get("windows")) is not int or meta["windows"] < 0 or \
       type(meta.get("single_matrix")) is not bool:
        raise MetadataError("invalid declared matrix shape")
    if meta["single_matrix"] and meta["windows"] != 1:
        raise MetadataError("single matrix must declare one window")
    if meta.get("dtype") not in (np.dtype("float32").str, np.dtype("float64").str):
        raise MetadataError("invalid declared result dtype")
    mode = meta.get("mode")
    if mode is None and entry["state"] == "pending":
        return meta
    if not isinstance(mode, str) or mode not in MODES or meta.get("arrays") != list(MODES[mode]):
        raise MetadataError("packing mode and required arrays disagree")
    specs = entry.get("array_metadata", {})
    if not isinstance(specs, dict):
        raise MetadataError("invalid array metadata mapping")
    if entry["state"] == "complete" and set(specs) != set(MODES[mode]):
        raise MetadataError("complete metric lacks required array declarations")
    return meta


def columns(meta, key):
    n = meta["n"]
    return n * n if key == "full" else n * (n + (1 if key == "upper" else -1)) // 2


@dataclass(frozen=True)
class Issue:
    array: str
    start: int
    stop: int
    reason: str


def describe(issues, limit=None):
    unique = list(dict.fromkeys(issues))
    shown = unique if limit is None else unique[:limit]
    regions = {"full": "full matrix", "upper": "upper payload (diagonal and both triangles)",
               "residual": "residual (strict lower triangle)"}
    messages = [f"{regions.get(i.array, i.array)} windows [{i.start}, {i.stop}): {i.reason}" for i in shown]
    if len(shown) < len(unique):
        messages.append(f"{len(unique) - len(shown)} additional affected ranges (use verify for details)")
    return "; ".join(messages)


class CheckedArray:
    """One operation's Zarr reader and small, cached shard indexes; rows are windows."""

    def __init__(self, path, key, meta, expected):
        self.path, self.key = Path(path), key
        self.shape = (meta["windows"], columns(meta, key))
        self.dtype = np.dtype(meta["dtype"] if key != "residual" else f"u{np.dtype(meta['dtype']).itemsize}")
        self.error = None
        self.indexes = {}
        self.array = None
        if expected is None:
            self.error = "array declaration was not published before the write stopped"
            return
        try:
            self._configure(expected)
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            raise MetadataError(f"{key}: invalid declared array layout") from exc
        try:
            actual = read_json(self.path / "zarr.json")
            if actual != expected:
                raise MetadataError("array metadata differs from its completion declaration")
        except MetadataError as exc:
            self.error = str(exc)
            return
        try:
            self.array = zarr.open_array(str(self.path), mode="r").with_config({"read_missing_chunks": False})
        except OSError as exc:
            self.error = str(exc)

    def _configure(self, expected):
        if not isinstance(expected, dict):
            raise MetadataError(f"{self.key}: invalid declared array metadata")
        if expected.get("node_type") != "array" or expected.get("zarr_format") != 3 or \
           expected.get("shape") != list(self.shape) or expected.get("data_type") != self.dtype.name:
            raise MetadataError(f"{self.key}: array shape/dtype disagrees with the matrix")
        grid = expected["chunk_grid"]
        encoding = expected["chunk_key_encoding"]
        if grid["name"] != "regular" or encoding["name"] != "default":
            raise StorageFormatError("only the regular/default chunk layout is supported")
        self.separator = encoding["configuration"]["separator"]
        if self.separator not in ("/", "."):
            raise StorageFormatError("unsupported chunk-key separator")
        codecs = expected["codecs"]
        if len(codecs) != 1 or codecs[0]["name"] != "sharding_indexed":
            raise StorageFormatError("expected one sharding_indexed codec")
        config = codecs[0]["configuration"]
        index_codecs = config["index_codecs"]
        if index_codecs != [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]:
            raise StorageFormatError("unsupported shard-index encoding")
        inner_codecs = config["codecs"]
        if not inner_codecs or inner_codecs[-1]["name"] != "crc32c" or \
           any(c["name"] not in ("bytes", "zstd", "numcodecs.pcodec", "crc32c") for c in inner_codecs):
            raise StorageFormatError("unsupported or unchecked numeric codec chain")
        self.shard_shape = tuple(grid["configuration"]["chunk_shape"])
        self.chunk_shape = tuple(config["chunk_shape"])
        if len(self.shard_shape) != 2 or len(self.chunk_shape) != 2 or \
           any(type(v) is not int or v <= 0 for v in self.shard_shape + self.chunk_shape) or \
           self.chunk_shape[1] != self.shape[1] or self.shard_shape[1] != self.shape[1] or \
           self.shard_shape[0] % self.chunk_shape[0]:
            raise MetadataError("invalid window-major chunk/shard geometry")
        self.location = config.get("index_location", "end")
        if self.location not in ("start", "end"):
            raise StorageFormatError("unsupported shard-index location")
        self.slots = self.shard_shape[0] // self.chunk_shape[0]
        self.index_size = self.slots * 16 + 4

    def _index(self, shard):
        if shard in self.indexes:
            return self.indexes[shard]
        key = self.separator.join(("c", str(shard), "0"))
        try:
            with (self.path / key).open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                if size < self.index_size:
                    raise StorageIntegrityError("truncated shard index")
                stream.seek(0 if self.location == "start" else size - self.index_size)
                raw = stream.read(self.index_size)
            if len(raw) != self.index_size or google_crc32c.value(raw[:-4]) != int.from_bytes(raw[-4:], "little"):
                raise StorageIntegrityError("shard-index checksum mismatch")
            index = np.frombuffer(raw[:-4], dtype="<u8").reshape(self.slots, 2)
            result = (index, size, None)
        except OSError as exc:
            result = (None, 0, f"{key}: {exc}")
        self.indexes[shard] = result
        return result

    def presence(self, indices):
        """Check only requested in-bounds chunks, including index offsets and length bounds."""
        ok = np.ones(len(indices), dtype=bool)
        issues = []
        if not len(indices):
            return ok, issues
        if self.error:
            ok[:] = False
            return ok, [Issue(self.key, int(indices.min()), int(indices.max()) + 1, self.error)]
        k = self.chunk_shape[0]
        for chunk in np.unique(indices // k):
            positions = indices // k == chunk
            index, size, error = self._index(int(chunk) // self.slots)
            if not error:
                offset, length = map(int, index[int(chunk) % self.slots])
                maximum = 2**64 - 1
                if offset == maximum and length == maximum:
                    error = "required inner chunk is missing"
                else:
                    begin = self.index_size if self.location == "start" else 0
                    end = size if self.location == "start" else size - self.index_size
                    if offset < begin or length <= 0 or offset + length > end:
                        error = "inner chunk offset/length is outside the data section"
            if error:
                ok[positions] = False
                issues.append(Issue(self.key, int(chunk) * k, min((int(chunk) + 1) * k, self.shape[0]), error))
        return ok, issues

    def read(self, indices):
        ok, issues = self.presence(indices)
        positions = np.flatnonzero(ok)
        if not len(positions):
            return np.zeros((len(indices), self.shape[1]), dtype=self.dtype), ok, issues
        selected = indices[positions]
        try:
            if len(selected) == 1 or np.all(np.diff(selected) == 1):
                block = self.array[int(selected[0]):int(selected[-1]) + 1]
            else:
                block = self.array.oindex[selected, :]
        except READ_ERRORS:
            # The normal batch failed: retry independent inner chunks, never individual floats.
            result = np.zeros((len(indices), self.shape[1]), dtype=self.dtype)
            k = self.chunk_shape[0]
            for chunk in np.unique(selected // k):
                rows = positions[selected // k == chunk]
                start, stop = int(chunk) * k, min((int(chunk) + 1) * k, self.shape[0])
                try:
                    block = self.array[start:stop]
                    result[rows] = block[indices[rows] - start]
                except READ_ERRORS as exc:
                    ok[rows] = False
                    issues.append(Issue(self.key, start, stop, f"unreadable data chunk: {type(exc).__name__}: {exc}"))
        else:
            if len(positions) == len(indices):
                return block, ok, issues
            result = np.zeros((len(indices), self.shape[1]), dtype=self.dtype)
            result[positions] = block
        return result, ok, issues


def metric_sources(path, entry):
    """Recover from damaged metric headers using independently declared decoding metadata."""
    meta = validate_entry(entry)
    issues = []
    if entry["state"] != "complete":
        issues.append(Issue("write state", 0, meta["windows"], f"write is {entry['state']}, not complete"))
    try:
        header = read_json(Path(path) / "zarr.json")
        if header.get("node_type") != "group" or header.get("attributes", {}).get("pspi") != meta:
            raise MetadataError("metric metadata differs from its completion declaration")
    except (MetadataError, AttributeError) as exc:
        issues.append(Issue("metric metadata", 0, meta["windows"], str(exc)))
    keys = MODES.get(meta.get("mode"), ())
    specs = entry.get("array_metadata", {})
    sources = {key: CheckedArray(Path(path) / key, key, meta, specs.get(key)) for key in keys}
    return meta, sources, issues
