"""Generic matrix-structure packing for channels x channels x windows results.

Each array is inspected at write time; nothing depends on metric names. Every
mode reconstructs every float bit pattern. No tolerance-based packing is used.

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


def pack(data: np.ndarray, *, structure: str = "exact", residual_ok=None) -> Packed:
    """Pack (n, n, W) data.

    structure:
      ``full``  -- no structure exploitation.
      ``exact`` -- sym/anti if bit-exact, else triangle + XOR residual when ``residual_ok(packed_res, packed_full)``
                   returns True (caller compares encoded sizes), else full.
    """
    if structure not in ("exact", "full"):
        raise ValueError("structure must be exact or full")
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
    use_anti = np.count_nonzero(lb == anti_ref) > np.count_nonzero(lb == sym_ref)
    ref = anti_ref if use_anti else sym_ref
    res = Packed("anti_res" if use_anti else "sym_res", upper=upper, residual=np.ascontiguousarray(lb ^ ref), **base)
    full = Packed("full", full=tf.reshape(w, n * n), **base)
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
