"""On-disk containers for packed connectivity results.

Every container stores one patient as a directory with one bounded unit per metric, and
implements write_metric / read_metric / read_mean. ``read_metric`` returns (n, n, windows)
and ``read_mean`` returns the (n, n) NaN-ignoring time average, computed from the packed
payload without materialising the full matrix stack.
"""

from __future__ import annotations

import json
import math
import struct
import warnings
import zlib
from pathlib import Path

import blosc2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs.numcodecs as zn
import zstandard
from zarr.codecs import BloscCodec, Crc32cCodec, ZstdCodec

from codecs_lib import ALL
from packing import UINT, Packed, _negate_bits

blosc2.set_nthreads(1)

TARGET_CHUNK_VALUES = 256 * 1024


def chunk_windows(values_per_window: int, windows: int, target=TARGET_CHUNK_VALUES) -> int:
    return int(max(1, min(windows, target // max(1, values_per_window))))


def _meta(p: Packed, extra: dict) -> dict:
    return dict(schema="pspi-packed/1", mode=p.mode, n=p.n, windows=p.windows, dtype=p.dtype,
                arrays=sorted(p.arrays()), info=p.info, **extra)


def _nanmean_rows(payload_wp: np.ndarray) -> np.ndarray:
    """NaN-ignoring mean over windows of a (W, P) payload, bit-identical to nanmean over
    axis 2 of the contiguous (n, n, W) source (same per-row pairwise summation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.ascontiguousarray(payload_wp.T).astype(np.float64, copy=False), axis=1)


def mean_from_packed(meta: dict, arrays: dict) -> np.ndarray:
    n, mode = meta["n"], meta["mode"]
    if mode == "full":
        return _nanmean_rows(arrays["full"]).reshape(n, n)
    dtype = np.dtype(meta["dtype"])
    iu = np.triu_indices(n)
    js = np.triu_indices(n, 1)
    out = np.empty((n, n), dtype=np.float64)
    out[iu[0], iu[1]] = _nanmean_rows(arrays["upper"])
    if mode == "sym":
        out[js[1], js[0]] = out[js[0], js[1]]
        return out
    if mode == "anti":
        # Negation is exact and rounding is sign-symmetric, so the mean of the sign-flipped
        # series is the sign-flipped mean. Zero and NaN means follow the same keep-as-is rule
        # as the stored values (x + -x rounds to +0; all-NaN gives the same NaN).
        out[js[1], js[0]] = _negate_bits(np.ascontiguousarray(out[js[0], js[1]]), np.float64).view(np.float64)
        return out
    # residual modes: reconstruct the lower triangle values, then average.
    upper = arrays["upper"]
    strict = np.ascontiguousarray(upper[:, _strict_positions(n)])
    ut = UINT[dtype]
    ref = strict.view(ut).copy() if mode == "sym_res" else _negate_bits(strict, dtype)
    lower = (ref ^ arrays["residual"]).view(dtype)
    out[js[1], js[0]] = _nanmean_rows(lower)
    return out


_STRICT_CACHE: dict[int, np.ndarray] = {}


def _strict_positions(n: int) -> np.ndarray:
    """Columns of the triu_indices(n) payload that are strictly upper, in triu_indices(n, 1) order."""
    if n not in _STRICT_CACHE:
        iu = np.triu_indices(n)
        _STRICT_CACHE[n] = np.flatnonzero(iu[0] != iu[1])
    return _STRICT_CACHE[n]


def packed_from_arrays(meta: dict, arrays: dict) -> Packed:
    return Packed(meta["mode"], n=meta["n"], windows=len(next(iter(arrays.values()))), dtype=meta["dtype"],
                  upper=arrays.get("upper"), residual=arrays.get("residual"), full=arrays.get("full"))


# ---------------------------------------------------------------------------------------------
class Container:
    name: str
    suffix = ""

    def path(self, root: Path, metric: str) -> Path:
        return root / (metric + self.suffix)

    def write_metric(self, root: Path, metric: str, p: Packed, extra: dict) -> dict:
        raise NotImplementedError

    def read_arrays(self, root: Path, metric: str, windows=None) -> tuple[dict, dict]:
        raise NotImplementedError

    def read_metric(self, root: Path, metric: str, windows=None) -> np.ndarray:
        from packing import unpack
        meta, arrays = self.read_arrays(root, metric, windows)
        if meta.get("layout") == "nnw":
            return arrays["nnw"]
        return unpack(packed_from_arrays(meta, arrays))

    def read_mean(self, root: Path, metric: str) -> np.ndarray:
        meta, arrays = self.read_arrays(root, metric)
        if meta.get("layout") == "nnw":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(arrays["nnw"].astype(np.float64, copy=False), axis=2)
        return mean_from_packed(meta, arrays)


# ---------------------------------------------------------------------------------------------
class ZarrCurrent(Container):
    """Recreation of the existing layout: full (n, n, W) float64, (n, n, 1) chunks, Zstd level 0."""

    name = "zarr_current"

    def write_metric(self, root, metric, p, extra):
        from packing import unpack
        data = unpack(p)
        zarr.create_array(str(self.path(root, metric)), data=data, chunks=(p.n, p.n, 1),
                          compressors=[ZstdCodec(level=0, checksum=False)], filters=[], fill_value=np.nan,
                          attributes={"pspi": _meta(p, dict(extra, layout="nnw"))},
                          config={"write_empty_chunks": False})
        return {}

    def read_arrays(self, root, metric, windows=None):
        a = zarr.open_array(str(self.path(root, metric)), mode="r")
        meta = a.attrs["pspi"]
        data = a[:] if windows is None else a.oindex[:, :, np.asarray(windows)]
        return meta, {"nnw": data}


class ZarrPacked(Container):
    """Zarr v3 group per metric with packed arrays (W, P). Optional one-shard-per-array."""

    def __init__(self, codec="pcodec8", sharded=True, crc=True, target=TARGET_CHUNK_VALUES):
        self.codec, self.sharded, self.crc, self.target = codec, sharded, crc, target
        self.name = f"zarr_{'shard' if sharded else 'dir'}_{codec}" + ("" if target == TARGET_CHUNK_VALUES else f"_c{target // 1024}k")

    def _codecs(self, dtype):
        isize = np.dtype(dtype).itemsize
        if self.codec == "pcodec8":
            ser, comp = zn.PCodec(level=8), []
        elif self.codec == "bzstd_shuf":
            ser, comp = "auto", [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=isize)]
        elif self.codec == "bzstd_bitshuf":
            ser, comp = "auto", [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=isize)]
        elif self.codec == "zstd3":
            ser, comp = "auto", [ZstdCodec(level=3)]
        else:
            raise ValueError(self.codec)
        if self.crc:
            comp = [*comp, Crc32cCodec()]
        return ser, comp

    def write_metric(self, root, metric, p, extra):
        g = zarr.open_group(str(self.path(root, metric)), mode="w-",
                            attributes={"pspi": _meta(p, dict(extra, layout="wp", container=self.name))})
        for key, arr in p.arrays().items():
            w, cols = arr.shape
            k = chunk_windows(cols, w, self.target)
            shards = (math.ceil(w / k) * k, cols) if self.sharded else None
            ser, comp = self._codecs(arr.dtype)
            if ser == "auto":
                a = g.create_array(key, shape=arr.shape, dtype=arr.dtype, chunks=(k, cols), shards=shards,
                                   compressors=comp, filters=[], fill_value=0, config={"write_empty_chunks": True})
            else:
                a = g.create_array(key, shape=arr.shape, dtype=arr.dtype, chunks=(k, cols), shards=shards,
                                   serializer=ser, compressors=comp, filters=[], fill_value=0,
                                   config={"write_empty_chunks": True})
            a[:] = arr
        return {}

    def read_arrays(self, root, metric, windows=None):
        g = zarr.open_group(str(self.path(root, metric)), mode="r")
        meta = g.attrs["pspi"]
        arrays = {}
        for key in meta["arrays"]:
            a = g[key]
            arrays[key] = a[:] if windows is None else a.oindex[np.asarray(windows), :]
        return meta, arrays


class Framed(Container):
    """One self-describing file per metric: independent CRC32-checked blocks + JSON trailer.

    File: b"PSPZ0001" | blocks | index JSON | u64 index length | b"PSPZEND1".
    Block: b"BLK1" | u32 header length | header JSON | payload. Each block holds a window
    range of one payload array, compressed with a named codec. If the trailer is damaged, a
    reader can recover blocks by scanning for block markers.
    """

    suffix = ".pspz"

    def __init__(self, codec="pcodec8", target=TARGET_CHUNK_VALUES):
        self.codec, self.target = codec, target
        self.name = f"framed_{codec}"

    def write_metric(self, root, metric, p, extra):
        enc = ALL[self.codec][0]
        blocks = []
        with open(self.path(root, metric), "wb") as fh:
            fh.write(b"PSPZ0001")
            for key, arr in p.arrays().items():
                w, cols = arr.shape
                k = chunk_windows(cols, w, self.target)
                for s in range(0, w, k):
                    blob = enc(np.ascontiguousarray(arr[s:s + k]))
                    hdr = dict(array=key, start=s, stop=min(w, s + k), cols=cols, dtype=arr.dtype.str,
                               codec=self.codec, length=len(blob), crc32=zlib.crc32(blob))
                    hb = json.dumps(hdr, separators=(",", ":")).encode()
                    offset = fh.tell()
                    fh.write(b"BLK1" + struct.pack("<I", len(hb)) + hb)
                    hdr["offset"] = fh.tell()
                    fh.write(blob)
                    hdr["block_offset"] = offset
                    blocks.append(hdr)
            index = json.dumps(dict(_meta(p, dict(extra, layout="wp", container=self.name)), blocks=blocks)).encode()
            fh.write(index + struct.pack("<Q", len(index)) + b"PSPZEND1")
        return {}

    def read_arrays(self, root, metric, windows=None):
        dec = None
        with open(self.path(root, metric), "rb") as fh:
            fh.seek(-16, 2)
            tail = fh.read(16)
            if tail[8:] != b"PSPZEND1":
                raise IOError("trailer damaged")
            (length,) = struct.unpack("<Q", tail[:8])
            fh.seek(-16 - length, 2)
            meta = json.loads(fh.read(length))
            sel = None if windows is None else np.asarray(windows)
            arrays = {}
            for key in meta["arrays"]:
                blocks = [b for b in meta["blocks"] if b["array"] == key]
                w = meta["windows"]
                cols, dt = blocks[0]["cols"], np.dtype(blocks[0]["dtype"])
                idx = np.arange(w) if sel is None else sel
                out = np.empty((len(idx), cols), dtype=dt)
                for b in blocks:
                    pos = np.flatnonzero((idx >= b["start"]) & (idx < b["stop"]))
                    if not len(pos):
                        continue
                    fh.seek(b["offset"])
                    blob = fh.read(b["length"])
                    if zlib.crc32(blob) != b["crc32"]:
                        raise IOError(f"block crc mismatch {key}[{b['start']}:{b['stop']}]")
                    dec = ALL[b["codec"]][1]
                    block = dec(blob, dt, (b["stop"] - b["start"], cols))
                    out[pos] = block[idx[pos] - b["start"]]
                arrays[key] = out
        return meta, arrays


class ParquetLong(Container):
    """Parquet per metric: rows (window, i, j, value) for the stored triangle (or full matrix).

    Exact sym/anti matrices store i <= j rows; any other structure stores every (i, j).
    Value columns use BYTE_STREAM_SPLIT + Zstd. Directly queryable with DuckDB/Polars/R arrow.
    """

    suffix = ".parquet"

    def __init__(self, level=3, target=TARGET_CHUNK_VALUES):
        self.level, self.target = level, target
        self.name = "parquet_bss_zstd"

    def write_metric(self, root, metric, p, extra):
        from packing import unpack
        n, w = p.n, p.windows
        if p.mode in ("sym", "anti"):
            iu = np.triu_indices(n)
            vals = p.upper
            mode = p.mode
        else:
            iu = np.unravel_index(np.arange(n * n), (n, n))
            vals = p.full if p.full is not None else np.moveaxis(unpack(p), 2, 0).reshape(w, n * n)
            mode = "full"
        cols = len(iu[0])
        table = pa.table({
            "window": pa.array(np.repeat(np.arange(w, dtype=np.uint16), cols)),
            "i": pa.array(np.tile(iu[0].astype(np.uint16), w)),
            "j": pa.array(np.tile(iu[1].astype(np.uint16), w)),
            "value": pa.array(np.ascontiguousarray(vals).ravel()),
        })
        meta = _meta(p, dict(extra, layout="long", container=self.name))
        meta["mode"] = mode
        meta["arrays"] = ["upper"] if mode != "full" else ["full"]
        table = table.replace_schema_metadata({"pspi": json.dumps(meta)})
        rg = chunk_windows(cols, w, self.target) * cols
        pq.write_table(table, self.path(root, metric), compression="zstd", compression_level=self.level,
                       use_dictionary=["window", "i", "j"], use_byte_stream_split=["value"],
                       row_group_size=rg, write_statistics=["window"], write_page_checksum=True)
        return {}

    def read_arrays(self, root, metric, windows=None):
        path = self.path(root, metric)
        pf = pq.ParquetFile(path, page_checksum_verification=True)
        meta = json.loads(pf.schema_arrow.metadata[b"pspi"])
        filters = None if windows is None else [("window", "in", [int(x) for x in windows])]
        t = pq.read_table(path, columns=["value"], filters=filters, page_checksum_verification=True)
        v = t.column("value").to_numpy()
        w = meta["windows"] if windows is None else len(windows)
        key = meta["arrays"][0]
        return meta, {key: v.reshape(w, -1)}


class Blosc2Frame(Container):
    """One blosc2 .b2nd file per payload array (chunked frame, independent chunks)."""

    def __init__(self, filters=("SHUFFLE", "BYTEDELTA"), clevel=5, target=TARGET_CHUNK_VALUES):
        self.filters, self.clevel, self.target = filters, clevel, target
        self.name = "b2nd_zstd_" + "_".join(f.lower() for f in filters)

    def path(self, root, metric):
        return root / metric

    def write_metric(self, root, metric, p, extra):
        d = self.path(root, metric)
        d.mkdir()
        for key, arr in p.arrays().items():
            w, cols = arr.shape
            k = chunk_windows(cols, w, self.target)
            filters = [getattr(blosc2.Filter, f) for f in self.filters]
            cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=self.clevel, filters=filters,
                                     filters_meta=[0] * len(filters), typesize=arr.dtype.itemsize, nthreads=1)
            blosc2.asarray(np.ascontiguousarray(arr), urlpath=str(d / f"{key}.b2nd"), mode="w",
                           chunks=(k, cols), cparams=cparams)
        (d / "meta.json").write_text(json.dumps(_meta(p, dict(extra, layout="wp", container=self.name))))
        return {}

    def read_arrays(self, root, metric, windows=None):
        d = self.path(root, metric)
        meta = json.loads((d / "meta.json").read_text())
        arrays = {}
        for key in meta["arrays"]:
            a = blosc2.open(str(d / f"{key}.b2nd"), mode="r", dparams=blosc2.DParams(nthreads=1))
            arrays[key] = a[:] if windows is None else a[np.asarray(windows)]
        return meta, arrays


DEFAULT_CONTAINERS = (
    "zarr_current", "zarr_dir_pcodec8", "zarr_shard_pcodec8", "zarr_shard_bzstd_shuf", "zarr_shard_zstd3",
    "framed_pcodec8", "framed_bzstd_shuf", "parquet_bss_zstd", "b2nd_zstd_shuffle_bytedelta",
)


def all_containers():
    sweep = [ZarrPacked(codec, sharded=True, target=t * 1024)
             for codec in ("pcodec8", "zstd3") for t in (16, 64, 1024, 4096)]
    return {c.name: c for c in (
        ZarrCurrent(),
        ZarrPacked("pcodec8", sharded=False),
        ZarrPacked("pcodec8", sharded=True),
        ZarrPacked("bzstd_shuf", sharded=True),
        ZarrPacked("zstd3", sharded=True),
        Framed("pcodec8"),
        Framed("bzstd_shuf"),
        ParquetLong(),
        Blosc2Frame(),
        *sweep,
    )}
