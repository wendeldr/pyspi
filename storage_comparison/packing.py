"""Generic matrix-structure packing for channels x channels x windows results.

Each array is inspected at write time; nothing depends on metric names. Exact modes
reconstruct every float bit pattern. The optional tolerance mode is a deliberate,
recorded precision choice for matrices that are mathematically symmetric or
antisymmetric but differ by floating-point noise.

Payload layout is window-major: ``upper`` has shape (windows, n*(n+1)/2) holding the
diagonal and strict upper triangle in ``np.triu_indices(n)`` order. ``full`` has shape
(windows, n*n) in C order of the original (row, column) matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UINT = {np.dtype("float64"): np.uint64, np.dtype("float32"): np.uint32}


def sign_bit(dtype) -> int:
    return 1 << (np.dtype(dtype).itemsize * 8 - 1)


@dataclass
class Packed:
    mode: str                    # full | sym | anti | sym_res | anti_res
    n: int
    windows: int
    dtype: str
    upper: np.ndarray | None = None      # (W, n(n+1)/2) float
    residual: np.ndarray | None = None   # (W, n(n-1)/2) uint, XOR against reference lower
    full: np.ndarray | None = None       # (W, n*n) float
    info: dict = field(default_factory=dict)

    def arrays(self):
        return {k: v for k, v in (("upper", self.upper), ("residual", self.residual), ("full", self.full)) if v is not None}


def _negate_bits(u: np.ndarray, dtype) -> np.ndarray:
    """Antisymmetric reference: flip the sign bit except for NaN and zeros, which are mirrored unchanged.

    This matches observed library behaviour (e.g. NaN pairs copied, +0.0 mirrored as +0.0).
    Any entry that does not follow the rule is captured by the residual, so the rule is
    only an efficiency choice, never a correctness assumption.
    """
    ut = UINT[np.dtype(dtype)]
    bits = u.view(ut)
    keep = np.isnan(u) | (u == 0)
    return np.where(keep, bits, bits ^ ut(sign_bit(dtype)))


def time_first(data: np.ndarray) -> np.ndarray:
    """(n, n, W) -> contiguous (W, n, n)."""
    return np.ascontiguousarray(np.moveaxis(data, 2, 0))


def _triangles(tf: np.ndarray):
    n = tf.shape[1]
    iu = np.triu_indices(n)
    js = np.triu_indices(n, 1)
    upper = np.ascontiguousarray(tf[:, iu[0], iu[1]])
    lower = np.ascontiguousarray(tf[:, js[1], js[0]])      # mirrored lower, strict-upper order
    strict = np.ascontiguousarray(tf[:, js[0], js[1]])
    return upper, strict, lower


def analyse(data: np.ndarray) -> dict:
    """Structure statistics used to pick a mode. ``data`` is (n, n, W)."""
    tf = time_first(data)
    ut = UINT[tf.dtype]
    upper, strict, lower = _triangles(tf)
    lb = lower.view(ut)
    sym_ref = strict.view(ut)
    anti_ref = _negate_bits(strict, tf.dtype)
    sym_match = float(np.mean(lb == sym_ref)) if lb.size else 1.0
    anti_match = float(np.mean(lb == anti_ref)) if lb.size else 1.0
    fin = np.isfinite(strict) & np.isfinite(lower)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        scale = np.maximum(np.abs(strict), np.abs(lower))
        rel_sym = np.where(scale > 0, np.abs(strict - lower) / scale, 0.0)
        rel_anti = np.where(scale > 0, np.abs(strict + lower) / scale, 0.0)
    nan_mismatch = int(np.sum(np.isnan(strict) != np.isnan(lower)))
    return dict(
        sym_match=sym_match, anti_match=anti_match,
        max_rel_asym=float(rel_sym[fin].max()) if fin.any() else 0.0,
        max_rel_antiasym=float(rel_anti[fin].max()) if fin.any() else 0.0,
        nan_mismatch=nan_mismatch,
    )


def pack(data: np.ndarray, *, structure: str = "exact", tolerance: float = 0.0, residual_ok=None) -> Packed:
    """Pack (n, n, W) data.

    structure:
      ``full``  -- no structure exploitation.
      ``exact`` -- sym/anti if bit-exact, else triangle + XOR residual when ``residual_ok(packed_res, packed_full)``
                   returns True (caller compares encoded sizes), else full.
      ``tol``   -- like exact, but if the maximum relative (anti)asymmetry over finite entries is
                   <= tolerance and NaN positions mirror, store the triangle only (upper values kept as-is).
    """
    if data.ndim != 3 or data.shape[0] != data.shape[1]:
        raise ValueError(f"expected (n, n, W), got {data.shape}")
    dtype = data.dtype
    if dtype not in UINT:
        raise ValueError(f"unsupported dtype {dtype}")
    n, _, w = data.shape
    tf = time_first(data)
    base = dict(n=n, windows=w, dtype=dtype.str)
    if structure == "full" or n == 1:
        return Packed("full", full=tf.reshape(w, n * n), **base)
    ut = UINT[dtype]
    upper, strict, lower = _triangles(tf)
    lb = lower.view(ut)
    sym_ref = strict.view(ut)
    if np.array_equal(lb, sym_ref):
        return Packed("sym", upper=upper, **base)
    anti_ref = _negate_bits(strict, dtype)
    if np.array_equal(lb, anti_ref):
        return Packed("anti", upper=upper, **base)
    stats = analyse(data)
    if structure == "tol" and stats["nan_mismatch"] == 0:
        if stats["max_rel_asym"] <= tolerance:
            return Packed("sym", upper=upper, info=dict(tolerance_applied=True, **stats), **base)
        if stats["max_rel_antiasym"] <= tolerance:
            return Packed("anti", upper=upper, info=dict(tolerance_applied=True, **stats), **base)
    use_anti = stats["anti_match"] > stats["sym_match"]
    ref = anti_ref if use_anti else sym_ref
    res = Packed("anti_res" if use_anti else "sym_res", upper=upper, residual=np.ascontiguousarray(lb ^ ref), info=stats, **base)
    full = Packed("full", full=tf.reshape(w, n * n), info=stats, **base)
    if residual_ok is None or residual_ok(res, full):
        return res
    return full


def unpack(p: Packed, windows=None) -> np.ndarray:
    """Return (n, n, W_selected) array. ``windows`` indexes the packed window axis."""
    sel = slice(None) if windows is None else np.asarray(windows)
    n = p.n
    dtype = np.dtype(p.dtype)
    if p.mode == "full":
        tf = p.full[sel].reshape(-1, n, n)
        return np.ascontiguousarray(np.moveaxis(tf, 0, 2))
    ut = UINT[dtype]
    upper = p.upper[sel]
    wsel = upper.shape[0]
    iu = np.triu_indices(n)
    js = np.triu_indices(n, 1)
    out = np.empty((wsel, n, n), dtype=dtype)
    out[:, iu[0], iu[1]] = upper
    strict = out[:, js[0], js[1]]
    if p.mode in ("sym", "sym_res"):
        lower_bits = strict.view(ut).copy()
    else:
        lower_bits = _negate_bits(strict, dtype)
    if p.mode.endswith("_res"):
        lower_bits ^= p.residual[sel]
    out[:, js[1], js[0]] = lower_bits.view(dtype)
    return np.ascontiguousarray(np.moveaxis(out, 0, 2))
