"""Round-trip tests for pspi_store on synthetic pyspi-like outputs (no real data needed).

Run: .venv/bin/python test_pspi_store.py   (or pytest if installed)
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

from pspi_store import open_results, write_metric

RNG = np.random.default_rng(20261013)
QNAN = np.array([0x7FF8000000000123], dtype=np.uint64).view(np.float64)[0]   # non-canonical NaN payload


def fake(kind: str, n: int, w: int, dtype=np.float64, clean: bool = False) -> np.ndarray:
    """Generate (n, n, w) results with a given structural kind and awkward special values."""
    out = np.empty((n, n, w))
    for t in range(w):
        x = RNG.standard_normal((n, max(2, n) + 3))
        if kind == "sym":
            m = np.corrcoef(x) if n > 1 else np.ones((1, 1))
            m = np.triu(m) + np.triu(m, 1).T
        elif kind == "near_sym":       # inverse of SPD matrix: symmetric up to rounding noise
            m = np.linalg.inv(np.cov(x) + 1e-3 * np.eye(n)) if n > 1 else np.ones((1, 1))
        elif kind == "anti":           # sign-flipped mirror, zeros and NaNs copied unchanged
            u = np.triu(RNG.standard_normal((n, n)), 1)
            u[RNG.random((n, n)) < 0.05] = 0.0
            m = u - u.T
        elif kind == "phase_like":     # antisymmetric up to a constant offset on some pairs
            u = np.triu(RNG.uniform(-np.pi, np.pi, (n, n)), 1)
            m = u - u.T + np.where(RNG.random((n, n)) < 0.3, 2 * np.pi / 512, 0.0) * (np.tri(n, k=-1))
        else:                          # directed
            m = RNG.standard_normal((n, n))
        out[:, :, t] = m
    np.einsum("iit->it", out)[:] = np.nan
    if n > 2:
        out[0, 1, 0] = out[1, 0, 0] = np.nan                     # mirrored NaN
        out[0, 2, -1] = np.inf
        out[1, 2, 0] = -0.0
        if kind != "sym" and not clean:
            out[2, 0, 0] = QNAN                                   # unmirrored NaN payload
        out[0, n - 1, w // 2] = 5e-324                            # subnormal
        if clean:                                                 # specials mirrored per structure
            sign = -1.0 if kind == "anti" else 1.0
            out[2, 0, -1] = sign * np.inf
            out[2, 1, 0] = -0.0                                   # zeros are copied, never flipped
            out[0, n - 1, w // 2] = 1e-40                         # subnormal in float32 too
            out[n - 1, 0, w // 2] = sign * 1e-40
    return out.astype(dtype)


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def nanmean(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(d.astype(np.float64), axis=2)


def check(tmp: Path, kind: str, n: int, w: int, dtype, codec, clean=False, store="r.pspi", **write_kw):
    name = f"{kind}{'_clean' if clean else ''}_{n}_{w}_{np.dtype(dtype).name}_{codec}"
    data = fake(kind, n, w, dtype, clean)
    meta = write_metric(tmp / store, name, data, labels=["test"], dtype=dtype, codec=codec, **write_kw)
    r = open_results(tmp / store)
    back = r.load(name)
    assert same_bits(back, data), (name, meta["mode"])
    sel = [0, w - 1] if w > 1 else [0]
    assert same_bits(r.load(name, windows=sel), np.ascontiguousarray(data[:, :, sel])), name
    assert same_bits(r.load(name, windows=w // 2), np.ascontiguousarray(data[:, :, w // 2])), name
    assert same_bits(r.mean(name), nanmean(data)), (name, meta["mode"])
    return meta["mode"]


def test_all():
    tmp = Path(tempfile.mkdtemp(prefix="pspi_store_test_"))
    try:
        modes = {}
        for kind in ("sym", "near_sym", "anti", "phase_like", "directed"):
            for n, w in ((1, 3), (2, 1), (2, 4), (7, 5), (40, 33)):
                for dtype in (np.float64, np.float32):
                    for codec in ("pcodec", "zstd"):
                        modes[(kind, n, w, np.dtype(dtype).name, codec)] = check(tmp, kind, n, w, dtype, codec)
        for kind, expect in (("sym", "sym"), ("anti", "anti")):
            for dtype in (np.float64, np.float32):
                mode = check(tmp, kind, 9, 6, dtype, "pcodec", clean=True)
                assert mode == expect, (kind, mode)
                modes[(kind + "_clean", 9, 6, np.dtype(dtype).name, "pcodec")] = mode
        # blockwise full loads and means: tiny chunks and blocks so every path spans many blocks
        import pspi_store
        saved = pspi_store.READ_BLOCK_VALUES, pspi_store.MEAN_BLOCK_COLUMNS
        pspi_store.READ_BLOCK_VALUES, pspi_store.MEAN_BLOCK_COLUMNS = 64, 7
        try:
            for kind in ("sym", "near_sym", "anti", "phase_like", "directed"):
                for dtype in (np.float64, np.float32):
                    modes[(kind + "_blocks", 40, 33, np.dtype(dtype).name, "pcodec")] = check(
                        tmp, kind, 40, 33, dtype, "pcodec", store="blocks.pspi", target_chunk_values=200)
        finally:
            pspi_store.READ_BLOCK_VALUES, pspi_store.MEAN_BLOCK_COLUMNS = saved
        # bounded shard files: force several shard files per metric
        d = fake("sym", 12, 50)
        write_metric(tmp / "r.pspi", "multi_shard", d, target_chunk_values=200, max_shard_values=1000)
        shard_files = [f for f in (tmp / "r.pspi" / "multi_shard" / "upper").rglob("*") if f.is_file() and f.name != "zarr.json"]
        assert len(shard_files) > 1, shard_files
        r = open_results(tmp / "r.pspi")
        assert same_bits(r.load("multi_shard"), d) and same_bits(r.mean("multi_shard"), nanmean(d))
        # float32 range guard: in-range float64 is rounded; out-of-range arrays stay float64 and exact
        ok64 = RNG.standard_normal((6, 6, 4))
        write_metric(tmp / "r.pspi", "f32_ok", ok64, dtype="float32")
        r = open_results(tmp / "r.pspi")
        assert r.info("f32_ok")["dtype"] == "<f4" and same_bits(r.load("f32_ok"), ok64.astype(np.float32))
        tiny = ok64.copy()
        tiny[0, 1, 0] = 1e-40
        write_metric(tmp / "r.pspi", "f32_tiny", tiny, dtype="float32")
        info = open_results(tmp / "r.pspi").info("f32_tiny")
        assert info["dtype"] == "<f8" and info["float32_fallback"], info
        assert same_bits(open_results(tmp / "r.pspi").load("f32_tiny"), tiny)
        try:
            write_metric(tmp / "r.pspi", "f32_big", ok64 * 1e300, dtype="float32", float32_out_of_range="error")
            raise AssertionError("overflow not rejected")
        except ValueError:
            pass
        # recording-level metadata
        from pspi_store import set_recording_info
        set_recording_info(tmp / "r.pspi", {"channels": ["A1", "A2"], "window_onsets": [0, 1024]})
        assert open_results(tmp / "r.pspi").recording["channels"] == ["A1", "A2"]
        # single (n, n) matrix round trip
        m = fake("sym", 5, 1)[:, :, 0]
        write_metric(tmp / "r.pspi", "single", m)
        r = open_results(tmp / "r.pspi")
        assert same_bits(r.load("single"), m) and r.load("single").shape == (5, 5)
        # verify(): a flipped data byte names exactly the damaged metric
        from pspi_store import verify
        assert verify(tmp / "r.pspi") == {}
        shard = max((f for f in (tmp / "r.pspi" / "multi_shard" / "upper").rglob("*") if f.is_file() and f.name != "zarr.json"),
                    key=lambda f: f.stat().st_size)
        raw = bytearray(shard.read_bytes())
        raw[len(raw) // 2] ^= 0x10
        shard.write_bytes(bytes(raw))
        assert list(verify(tmp / "r.pspi")) == ["multi_shard"], verify(tmp / "r.pspi")
        # refusing silent overwrite
        try:
            write_metric(tmp / "r.pspi", "single", m)
            raise AssertionError("overwrite not refused")
        except FileExistsError:
            pass
        seen = {}
        for k, v in modes.items():
            seen.setdefault((k[0], k[1] > 2), set()).add(v)
        print("modes by kind (n>2):", {k: sorted(v) for k, v in seen.items() if k[1]})
        print(f"{len(modes) + 1} round trips bit-exact (values, subsets, scalar window, time mean)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_all()
