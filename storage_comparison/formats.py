"""Experimental lossless stores. All public readers return channels x channels x time."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr
import zstandard
from zarr.codecs import BloscCodec, ZstdCodec


@dataclass(frozen=True)
class Variant:
    name: str
    codec: str = "zstd"
    packed: bool = False
    original_layout: bool = False
    sharded: bool = False
    blocks: bool = False


VARIANTS = {
    v.name: v
    for v in (
        Variant("original_zstd", original_layout=True),
        Variant("time_zstd"),
        Variant("time_shuffle", codec="shuffle"),
        Variant("time_bitshuffle", codec="bitshuffle"),
        Variant("packed_zstd", packed=True),
        Variant("packed_bitshuffle", codec="bitshuffle", packed=True),
        Variant("packed_sharded", codec="bitshuffle", packed=True, sharded=True),
        Variant("blocks_zstd", packed=True, blocks=True),
    )
}


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.tobytes(order="C") == b.tobytes(order="C")
    )


def symmetric_bits(data: np.ndarray) -> bool:
    bits = data.view(np.uint64)
    return bool(np.array_equal(bits, bits.swapaxes(0, 1)))


def _canonical_nan_only(data: np.ndarray) -> bool:
    bits = data.view(np.uint64)
    canonical = np.array(np.nan, dtype=data.dtype).view(np.uint64).item()
    return bool(np.all(bits == canonical))


def _noncanonical_nans(data: np.ndarray) -> bool:
    mask = np.isnan(data)
    canonical = np.array(np.nan, dtype=data.dtype).view(np.uint64).item()
    return bool(np.any(data.view(np.uint64)[mask] != canonical))


def _payload(data: np.ndarray, variant: Variant):
    if data.dtype != np.dtype("float64"):
        raise ValueError(f"This experiment requires float64, got {data.dtype}")
    if data.ndim != 3 or data.shape[0] != data.shape[1] or not data.shape[2]:
        raise ValueError(f"Expected a nonempty square channels x channels x time array: {data.shape}")
    n = data.shape[0]
    if variant.original_layout:
        return np.ascontiguousarray(data), "full", "ccw"
    time_first = np.moveaxis(data, 2, 0)
    if variant.packed and symmetric_bits(data):
        upper = np.triu_indices(n)
        return np.ascontiguousarray(time_first[:, upper[0], upper[1]]), "upper", "wp"
    return np.ascontiguousarray(time_first), "full", "wcc"


def write_store(
    path: Path,
    variant: Variant,
    data: np.ndarray,
    *,
    context: dict | None = None,
    chunk_windows: int = 8,
) -> dict:
    """Create a new experimental store; never overwrite an existing path."""
    if path.exists():
        raise FileExistsError(path)
    if chunk_windows < 1:
        raise ValueError("chunk_windows must be positive")
    payload, mode, layout = _payload(data, variant)
    span = min(chunk_windows, data.shape[2])
    meta = {
        "schema": 1,
        "variant": variant.name,
        "mode": mode,
        "layout": layout,
        "shape": list(data.shape),
        "payload_shape": list(payload.shape),
        "dtype": data.dtype.str,
        "context": context or {},
    }
    if variant.blocks:
        path.mkdir(parents=True)
        entries = []
        compressor = zstandard.ZstdCompressor(level=3)
        for start in range(0, data.shape[2], span):
            block = payload[start : start + span]
            filename = None
            if not _canonical_nan_only(block):
                filename = f"{start:08d}.npy.zst"
                buffer = io.BytesIO()
                np.save(buffer, block, allow_pickle=False)
                (path / filename).write_bytes(compressor.compress(buffer.getbuffer()))
            entries.append({"start": start, "stop": start + len(block), "file": filename})
        meta.update(blocks=entries, compression={"codec": "zstd", "level": 3})
        (path / "index.json").write_text(json.dumps(meta, separators=(",", ":")))
    else:
        if variant.original_layout:
            chunks = (*payload.shape[:2], 1)
        else:
            chunks = (span, *payload.shape[1:])
        shards = None
        if variant.sharded:
            shards = (span * min(4, math.ceil(data.shape[2] / span)), *payload.shape[1:])
        if variant.codec == "zstd":
            codec = ZstdCodec(level=0, checksum=False)
        else:
            codec = BloscCodec(
                cname="zstd", clevel=3, typesize=8,
                shuffle="bitshuffle" if variant.codec == "bitshuffle" else "shuffle",
            )
        # Empty canonical-NaN chunks can be omitted just as in the original stores.
        # Other NaN payloads must be explicitly written to preserve their bits.
        zarr.create_array(
            str(path), data=payload, chunks=chunks, shards=shards,
            compressors=[codec], filters=[], fill_value=np.nan,
            attributes={"storage_comparison": meta},
            config={"write_empty_chunks": _noncanonical_nans(payload)},
        )
    return meta


def read_metadata(path: Path) -> dict:
    index = path / "index.json"
    if index.is_file():
        return json.loads(index.read_text())
    return json.loads((path / "zarr.json").read_text())["attributes"]["storage_comparison"]


def read_store(path: Path, windows: Sequence[int] | None = None) -> np.ndarray:
    meta = read_metadata(path)
    n, _, count = meta["shape"]
    indices = np.arange(count, dtype=np.int64) if windows is None else np.asarray(windows, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
        raise IndexError("Window indices must be a one-dimensional sequence within the stored sample")
    if "blocks" in meta:
        payload = np.empty((len(indices), *meta["payload_shape"][1:]), dtype=meta["dtype"])
        decompressor = zstandard.ZstdDecompressor()
        for entry in meta["blocks"]:
            positions = np.flatnonzero((indices >= entry["start"]) & (indices < entry["stop"]))
            if not len(positions):
                continue
            if entry["file"] is None:
                payload[positions] = np.nan
            else:
                decoded = decompressor.decompress((path / entry["file"]).read_bytes())
                block = np.load(io.BytesIO(decoded), allow_pickle=False)
                payload[positions] = block[indices[positions] - entry["start"]]
    else:
        array = zarr.open_array(str(path), mode="r")
        if windows is None:
            payload = array[:]
        elif meta["layout"] == "ccw":
            payload = array.oindex[:, :, indices]
        elif meta["layout"] == "wp":
            payload = array.oindex[indices, :]
        else:
            payload = array.oindex[indices, :, :]
    if meta["mode"] == "upper":
        upper = np.triu_indices(n)
        result = np.empty((len(indices), n, n), dtype=meta["dtype"])
        result[:, upper[0], upper[1]] = payload
        result[:, upper[1], upper[0]] = payload
        return np.ascontiguousarray(np.moveaxis(result, 0, 2))
    if meta["layout"] == "ccw":
        return np.ascontiguousarray(payload)
    return np.ascontiguousarray(np.moveaxis(payload, 0, 2))


def filesystem_size(path: Path) -> dict:
    result = dict(stored_bytes=0, allocated_bytes=0, data_bytes=0, metadata_bytes=0, files=0, directories=0)
    for item in (path, *path.rglob("*")):
        info = item.stat()
        result["allocated_bytes"] += getattr(info, "st_blocks", 0) * 512
        if item.is_dir():
            result["directories"] += 1
        else:
            result["files"] += 1
            result["stored_bytes"] += info.st_size
            key = "metadata_bytes" if item.suffix == ".json" else "data_bytes"
            result[key] += info.st_size
    return result
