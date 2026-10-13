"""Bytes-level codec registry for the in-memory screen. All codecs are single-threaded.

encode(arr) -> bytes ; decode(bytes, dtype, shape) -> arr. Float-only codecs (zfp, fpzip)
hand integer arrays (XOR residuals) to blosc-zstd-bitshuffle, recorded as such.
"""

from __future__ import annotations

import lzma
import threading

import blosc2
import fpzip
import numpy as np
import zfpy
import zstandard
from pcodec import ChunkConfig, standalone

blosc2.set_nthreads(1)


def _zstd(level):
    local = threading.local()          # zstandard contexts are not thread-safe: one per thread

    def ctx():
        if not hasattr(local, "c"):
            local.c, local.d = zstandard.ZstdCompressor(level=level), zstandard.ZstdDecompressor()
        return local.c, local.d
    return (lambda a: ctx()[0].compress(a.tobytes()),
            lambda b, dt, sh: np.frombuffer(ctx()[1].decompress(b), dtype=dt).reshape(sh))


def _blosc(codec, clevel, filters):
    def enc(a):
        return blosc2.compress2(np.ascontiguousarray(a), codec=codec, clevel=clevel, filters=filters,
                                filters_meta=[0] * len(filters), typesize=a.dtype.itemsize, nthreads=1)
    return enc, lambda b, dt, sh: np.frombuffer(blosc2.decompress2(b), dtype=dt).reshape(sh)


def _pcodec(level):
    cfg = ChunkConfig(compression_level=level)
    return (lambda a: standalone.simple_compress(np.ascontiguousarray(a).ravel(), cfg),
            lambda b, dt, sh: standalone.simple_decompress(b).reshape(sh))


_BITSHUFFLE_FALLBACK = _blosc(blosc2.Codec.ZSTD, 5, [blosc2.Filter.BITSHUFFLE])


def _float_only(enc, dec):
    def e(a):
        if a.dtype.kind == "f":
            return enc(a)
        return _BITSHUFFLE_FALLBACK[0](a)

    def d(b, dt, sh):
        if np.dtype(dt).kind == "f":
            return dec(b, dt, sh)
        return _BITSHUFFLE_FALLBACK[1](b, dt, sh)
    return e, d


def _zfp():
    return _float_only(lambda a: zfpy.compress_numpy(np.ascontiguousarray(a)),
                       lambda b, dt, sh: zfpy.decompress_numpy(b).reshape(sh))


def _fpzip():
    return _float_only(lambda a: fpzip.compress(np.ascontiguousarray(a), precision=0, order="C"),
                       lambda b, dt, sh: fpzip.decompress(b, order="C").reshape(sh))


def _lzma(preset):
    return (lambda a: lzma.compress(a.tobytes(), preset=preset),
            lambda b, dt, sh: np.frombuffer(lzma.decompress(b), dtype=dt).reshape(sh))


def _shuffled(inner):
    """Byte-shuffle in numpy, then a plain bytes codec (to test xz/zstd-high on shuffled data)."""
    enc, dec = inner

    def e(a):
        s = a.dtype.itemsize
        return enc(np.ascontiguousarray(a.reshape(-1).view(np.uint8).reshape(-1, s).T))

    def d(b, dt, sh):
        s = np.dtype(dt).itemsize
        raw = dec(b, np.uint8, (s, -1))
        return np.ascontiguousarray(raw.T).view(dt).reshape(sh)
    return e, d


FAST = {
    "zstd1": _zstd(1),
    "zstd3": _zstd(3),
    "lz4_shuf": _blosc(blosc2.Codec.LZ4, 5, [blosc2.Filter.SHUFFLE]),
    "bzstd_shuf": _blosc(blosc2.Codec.ZSTD, 5, [blosc2.Filter.SHUFFLE]),
    "bzstd_bitshuf": _blosc(blosc2.Codec.ZSTD, 5, [blosc2.Filter.BITSHUFFLE]),
    "bzstd_bytedelta": _blosc(blosc2.Codec.ZSTD, 5, [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]),
    "pcodec8": _pcodec(8),
    "zfp_rev": _zfp(),
    "fpzip": _fpzip(),
}

SLOW = {
    "zstd9_shuf": _shuffled(_zstd(9)),
    "zstd19_shuf": _shuffled(_zstd(19)),
    "xz6_shuf": _shuffled(_lzma(6)),
    "pcodec12": _pcodec(12),
    "bzstd9_bytedelta": _blosc(blosc2.Codec.ZSTD, 9, [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]),
}



# ---- further native (Rust / C / C++) candidates; imported lazily -------------------------------
def _openzl(successor):
    """Meta OpenZL (C): numeric input, byte-plane transpose, then ``successor`` per plane."""
    def run(a):
        import openzl.ext as zl
        c = zl.Compressor()
        succ = zl.graphs.Entropy()(c) if successor == "entropy" else zl.graphs.Zstd(compression_level=3)(c)
        graph = zl.nodes.TransposeSplit()(c, succ)        # build the graph before binding the context
        cctx = zl.CCtx()
        cctx.ref_compressor(c)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        cctx.select_starting_graph(c, graph)
        u = np.ascontiguousarray(a).reshape(-1).view(np.dtype(f"u{a.dtype.itemsize}"))
        return cctx.compress([zl.Input(zl.Type.Numeric, u)])

    def back(b, dt, sh):
        import openzl.ext as zl
        return np.array(zl.DCtx().decompress(b)[0].content.as_nparray()).view(dt).reshape(sh)
    return run, back


def _vortex():
    """SpiralDB Vortex (Rust): adaptive cascading encodings (ALP / ALP-RD for floats); file bytes."""
    import os
    import tempfile

    def run(a):
        import pyarrow as pa
        import vortex as vx
        vx.set_worker_threads(1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.vortex")
            vx.io.write(vx.compress(vx.array(pa.table({"v": np.ascontiguousarray(a).reshape(-1)}))), path)
            return open(path, "rb").read()

    def back(b, dt, sh):
        import vortex as vx
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.vortex")
            open(path, "wb").write(b)
            rd = vx.open(path).to_arrow()
            tb = rd.read_all() if hasattr(rd, "read_all") else rd
            return np.ascontiguousarray(tb.column("v").to_numpy()).view(dt).reshape(sh)
    return run, back


def _duckdb():
    """DuckDB (C++) native single-column table (ALP / ALP-RD for doubles); checkpointed file bytes."""
    import os
    import tempfile

    def run(a):
        import duckdb
        import pyarrow as pa
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.duckdb")
            con = duckdb.connect(path, config={"threads": 1})
            con.register("t_in", pa.table({"v": np.ascontiguousarray(a).reshape(-1)}))
            con.execute("CREATE TABLE t AS SELECT v FROM t_in")
            con.execute("CHECKPOINT")
            con.close()
            return open(path, "rb").read()

    def back(b, dt, sh):
        import duckdb
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.duckdb")
            open(path, "wb").write(b)
            con = duckdb.connect(path, read_only=True, config={"threads": 1})
            v = con.execute("SELECT v FROM t").fetchnumpy()["v"]
            con.close()
            return np.ascontiguousarray(v).view(dt).reshape(sh)
    return run, back


NATIVE = {
    "openzl_tsplit_zstd": _openzl("zstd"),
    "openzl_tsplit_entropy": _openzl("entropy"),
    "vortex": _vortex(),
    "duckdb_native": _duckdb(),
}

ALL = {**FAST, **SLOW, **NATIVE}
