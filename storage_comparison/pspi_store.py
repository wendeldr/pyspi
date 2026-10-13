"""Reference reader/writer for compact pyspi results (prototype for the recommended format).

Layout (a Zarr v3 hierarchy; one directory per recording, one bounded unit per metric)::

    results.pspi/                 group, attrs: {"pspi_store": {...}}
      <metric>/                   group, attrs: {"pspi": {mode, n, windows, dtype, labels, ...}}
        upper/  or  full/         array (windows, values), ONE shard file, crc32c-checked inner chunks
        residual/                 only for near-(anti)symmetric matrices (exact XOR residual)

Structure is detected per array at write time (never from metric names):
  sym       bit-exactly symmetric                -> diagonal + upper triangle
  anti      bit-exactly sign-flipped mirror      -> diagonal + upper triangle
  sym_res   symmetric up to floating-point noise -> triangle + compact exact XOR residual
  anti_res  antisymmetric up to noise            -> triangle + compact exact XOR residual
  full      anything else (directed)             -> every entry

Reads return (n, n, windows) arrays, or (n, n) for results written as a single matrix.
``verify(path)`` (or ``python pspi_store.py verify <path>``) reads every checksummed chunk, e.g.
after a copy, and names the metrics that fail so only those directories need copying again.
``mean`` computes the NaN-ignoring time average directly from the stored triangle; it is
bit-identical to ``np.nanmean(load(metric), axis=2)``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import zarr
import zarr.codecs.numcodecs as zn
from zarr.codecs import Crc32cCodec, ZstdCodec

from packing import UINT, Packed, _negate_bits, pack, unpack

SCHEMA = "pspi-store/0.1"
TARGET_CHUNK_VALUES = 256 * 1024          # values per independently decoded, crc-checked chunk
MAX_SHARD_VALUES = 128 * 1024 * 1024      # cap per shard file (1 GiB of raw float64) for very large n
READ_BLOCK_VALUES = 4 * 1024 * 1024       # full loads unpack this many stored values at a time
MEAN_BLOCK_COLUMNS = 8192                 # time means reduce this many matrix entries at a time


def _codec_chain(codec: str, level: int | None):
    if codec == "pcodec":
        return zn.PCodec(level=8 if level is None else level), [Crc32cCodec()]
    if codec == "zstd":
        return "auto", [ZstdCodec(level=3 if level is None else level), Crc32cCodec()]
    raise ValueError(f"unknown codec {codec!r} (use 'pcodec' or 'zstd')")


def _residual_smaller(codec: str):
    """Keep the residual form only if it encodes smaller than the full matrix."""
    from codecs_lib import ALL
    enc = ALL["pcodec8" if codec == "pcodec" else "zstd3"][0]

    def ok(res: Packed, full: Packed) -> bool:
        return sum(len(enc(a)) for a in res.arrays().values()) < sum(len(enc(a)) for a in full.arrays().values())
    return ok


def write_metric(root, metric: str, values: np.ndarray, *, labels: Iterable[str] = (),
                 dtype="float64", codec: str = "pcodec", level: int | None = None,
                 attrs: Mapping | None = None, overwrite: bool = False,
                 target_chunk_values: int = TARGET_CHUNK_VALUES, max_shard_values: int = MAX_SHARD_VALUES,
                 float32_out_of_range: str = "fallback") -> dict:
    """Write one metric: ``values`` is (n, n) or (n, n, windows). Returns the stored metadata.

    With ``dtype="float32"``, an array whose finite values would overflow or fall below the
    float32 normal range is stored as float64 instead (``"fallback"``), rejected (``"error"``),
    or rounded anyway (``"allow"``). The decision is recorded as ``float32_fallback``.
    """
    values = np.asarray(values)
    single = values.ndim == 2
    if single:
        values = values[:, :, None]
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected (n, n) or (n, n, windows), got {values.shape}")
    dt = np.dtype(dtype)
    if dt not in UINT:
        raise ValueError("dtype must be float64 or float32")
    fallback = None
    if dt == np.float32 and values.dtype != np.float32:
        fallback = float32_range_problem(values)
        if fallback and float32_out_of_range == "fallback":
            dt = np.dtype("float64")
        elif fallback and float32_out_of_range == "error":
            raise ValueError(f"{metric}: {fallback}")
    data = np.ascontiguousarray(values, dtype=dt)
    p = pack(data, structure="exact", residual_ok=_residual_smaller(codec))
    root = zarr.open_group(str(root), mode="a", attributes=None)
    if "pspi_store" not in root.attrs:
        root.attrs["pspi_store"] = {"schema": SCHEMA}
    if metric in root:
        if not overwrite:
            raise FileExistsError(f"{metric} already stored")
        del root[metric]
    meta = dict(schema=SCHEMA, mode=p.mode, n=p.n, windows=p.windows, single_matrix=single,
                dtype=dt.str, source_dtype=values.dtype.str, requested_dtype=np.dtype(dtype).str,
                float32_fallback=fallback, codec=codec, arrays=sorted(p.arrays()),
                labels=list(labels), **(dict(attrs) if attrs else {}))
    g = root.create_group(metric, attributes={"pspi": meta})
    for key, arr in p.arrays().items():
        w, cols = arr.shape
        k = int(max(1, min(w, target_chunk_values // max(1, cols))))
        per_shard = max(1, min(math.ceil(w / k), max_shard_values // (k * cols)))   # chunks per shard file
        ser, comp = _codec_chain(codec, level)
        kwargs = dict(shape=arr.shape, dtype=arr.dtype, chunks=(k, cols), shards=(per_shard * k, cols),
                      compressors=comp, filters=[], fill_value=0, config={"write_empty_chunks": True})
        if ser != "auto":
            kwargs["serializer"] = ser
        g.create_array(key, **kwargs)[:] = arr
    return meta


def float32_range_problem(values: np.ndarray) -> str | None:
    """Reason a float32 cast would lose more than rounding (overflow or sub-normal-range values)."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    mag = np.abs(finite)
    info = np.finfo(np.float32)
    over = int(np.sum(mag > info.max))
    under = int(np.sum((mag > 0) & (mag < info.tiny)))
    if over or under:
        return f"{over} values overflow and {under} values underflow float32"
    return None


def write_results(path, results: Mapping[str, np.ndarray], *, labels: Mapping[str, Iterable[str]] | None = None,
                  recording: Mapping | None = None, **kwargs) -> None:
    """Write several metrics; ``recording`` holds shared metadata (channel names, window onsets, ...)."""
    if recording is not None:
        set_recording_info(path, recording)
    for metric, values in results.items():
        write_metric(path, metric, values, labels=(labels or {}).get(metric, ()), **kwargs)


def set_recording_info(path, info: Mapping) -> None:
    """Store JSON-serialisable recording metadata on the root group (merged with existing)."""
    root = zarr.open_group(str(path), mode="a")
    store_attrs = dict(root.attrs.get("pspi_store", {"schema": SCHEMA}))
    store_attrs["recording"] = {**store_attrs.get("recording", {}), **dict(info)}
    root.attrs["pspi_store"] = store_attrs


class Results:
    """Read access to a results store. Metadata reads are cheap; data is read per metric."""

    def __init__(self, path):
        self.path = Path(path)
        self._root = zarr.open_group(str(self.path), mode="r")

    @property
    def metrics(self) -> list[str]:
        return sorted(k for k, _ in self._root.groups())

    @property
    def recording(self) -> dict:
        return dict(self._root.attrs.get("pspi_store", {}).get("recording", {}))

    def info(self, metric: str) -> dict:
        return dict(self._root[metric].attrs["pspi"])

    def _arrays(self, metric, windows=None):
        g = self._root[metric]
        meta = g.attrs["pspi"]
        sel = slice(None) if windows is None else np.atleast_1d(np.asarray(windows, dtype=np.int64))
        out = {}
        for key in meta["arrays"]:
            a = g[key]
            if isinstance(sel, slice):       # block reads: a whole-shard read holds the compressed shard too
                out[key] = np.empty(a.shape, dtype=a.dtype)
                step = _block_windows(a)
                for s in range(0, a.shape[0], step):
                    out[key][s:s + step] = a[s:s + step]
            else:
                out[key] = a.oindex[sel, :]
        return meta, out

    def load(self, metric: str, windows=None) -> np.ndarray:
        """(n, n, windows) array (or (n, n) for a single stored matrix / scalar window index)."""
        if windows is None:
            data = self._load_all(metric)
        else:
            meta, arrays = self._arrays(metric, windows)
            w = len(next(iter(arrays.values())))
            data = unpack(Packed(meta["mode"], n=meta["n"], windows=w, dtype=meta["dtype"], **arrays))
        meta = self._root[metric].attrs["pspi"]
        scalar_window = windows is not None and np.ndim(windows) == 0
        if scalar_window or (meta["single_matrix"] and windows is None):
            return data[:, :, 0]
        return data

    def _load_all(self, metric):
        """Unpack window blocks straight into the output so peak memory stays near the result size."""
        g = self._root[metric]
        meta = g.attrs["pspi"]
        n, w = meta["n"], meta["windows"]
        stored = {key: g[key] for key in meta["arrays"]}
        step = _block_windows(stored[meta["arrays"][0]])
        out = np.empty((n, n, w), dtype=meta["dtype"])
        for s in range(0, w, step):
            block = {key: a[s:s + step] for key, a in stored.items()}
            out[:, :, s:s + step] = unpack(Packed(meta["mode"], n=n, windows=len(block[meta["arrays"][0]]),
                                                  dtype=meta["dtype"], **block))
        return out

    def mean(self, metric: str) -> np.ndarray:
        """NaN-ignoring mean over windows as float64 (n, n); bit-identical to nanmean(load(), axis=2)."""
        meta, arrays = self._arrays(metric)
        return _mean_from_packed(meta, arrays)


def verify(path) -> dict[str, str]:
    """Read every stored chunk of every metric; return {metric: error} for failures (empty if clean).

    Data chunks carry CRC32C checksums, so damaged bytes raise instead of decoding to wrong values.
    The small zarr.json metadata files are not checksummed; damage there usually fails to parse.
    """
    bad = {}
    for d in sorted(p for p in Path(path).iterdir() if p.is_dir()):
        try:
            g = zarr.open_group(str(d), mode="r")
            for key in g.attrs["pspi"]["arrays"]:
                a = g[key]
                step = _block_windows(a)
                for s in range(0, a.shape[0], step):
                    a[s:s + step]
        except Exception as e:                     # report and continue with the next metric
            bad[d.name] = f"{type(e).__name__}: {e}"
    return bad


def _block_windows(a) -> int:
    """Whole inner chunks' worth of windows holding about READ_BLOCK_VALUES stored values."""
    k = a.chunks[0]
    return k * max(1, READ_BLOCK_VALUES // (k * max(1, a.shape[1])))


def open_results(path) -> Results:
    return Results(path)


def _rows_nanmean(payload_wp: np.ndarray) -> np.ndarray:
    """nanmean over windows for each stored column, a column block at a time (same summation order
    as ``np.nanmean(x, axis=2)`` on the (n, n, W) array, so the result is bit-identical)."""
    out = np.empty(payload_wp.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in range(0, payload_wp.shape[1], MEAN_BLOCK_COLUMNS):
            block = np.ascontiguousarray(payload_wp[:, c:c + MEAN_BLOCK_COLUMNS].T).astype(np.float64, copy=False)
            out[c:c + MEAN_BLOCK_COLUMNS] = np.nanmean(block, axis=1)
    return out


def _mean_from_packed(meta, arrays):
    n, mode, dtype = meta["n"], meta["mode"], np.dtype(meta["dtype"])
    if mode == "full":
        return _rows_nanmean(arrays["full"]).reshape(n, n)
    iu, js = np.triu_indices(n), np.triu_indices(n, 1)
    out = np.empty((n, n))
    out[iu[0], iu[1]] = _rows_nanmean(arrays["upper"])
    up = np.ascontiguousarray(out[js[0], js[1]])
    if mode == "sym":
        out[js[1], js[0]] = up
    elif mode == "anti":
        out[js[1], js[0]] = _negate_bits(up, np.float64).view(np.float64)
    else:
        strict_cols = np.flatnonzero(iu[0] != iu[1])
        lower = np.empty(len(strict_cols))
        for c in range(0, len(strict_cols), MEAN_BLOCK_COLUMNS):
            strict = arrays["upper"][:, strict_cols[c:c + MEAN_BLOCK_COLUMNS]]
            ref = strict.view(UINT[dtype]) if mode == "sym_res" else _negate_bits(strict, dtype)
            lower[c:c + MEAN_BLOCK_COLUMNS] = _rows_nanmean((ref ^ arrays["residual"][:, c:c + MEAN_BLOCK_COLUMNS]).view(dtype))
        out[js[1], js[0]] = lower
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "verify":
        sys.exit("usage: pspi_store.py verify <store>")
    failures = verify(sys.argv[2])
    for metric, err in failures.items():
        print(f"FAILED {metric}: {err}")
    print(f"{len(failures)} damaged metric(s)" if failures else "all metrics verified")
    sys.exit(1 if failures else 0)
