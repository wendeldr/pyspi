#!/usr/bin/env python3
"""Shared synthetic data suite (D1-D15) for the pyspi calculation-correctness audit.

This module is deliberately independent of pyspi: it imports only numpy and
scipy so that pyspi (run in the project venv) and independent reference
implementations (run in a separate audit venv) load byte-identical arrays.

Layout
------
``<out>/<name>_T<T>.npz``  one file per dataset x length.  Each file stores
    ``x``     float64 array, shape (P, T) -- "ps" order (processes, observations),
              raw and un-normalised;
    truth arrays named per dataset (see ``datasets()`` and MANIFEST.json).
``<out>/MANIFEST.json``  name, T, P, seed, params, a one-line description and,
    for every array in the file (``x`` included), its sha256 over
    ``np.ascontiguousarray(v).tobytes()`` with dtype and shape (``arrays``);
    the legacy top-level ``x_sha256`` is kept (deterministic: sorted keys, no
    timestamps).  ``load()`` refuses to return data without a manifest and
    checks every array hash and the key set.
``<out>/CONVENTIONS.md``  hand-written conventions for downstream check authors.

Determinism
-----------
Every draw uses ``np.random.default_rng(seed)`` with
``seed = base_seed * 1_000_000 + T`` (never the global RNG).  Different lengths
are therefore independent draws, not prefixes of one another.  D3, D4 and D10
reuse D2's exact draw (D2's seed); D10 uses its own seed for the contamination.

Conventions
-----------
* VAR coefficients: ``x_t = sum_k A_k x_{t-k} + e_t``; ``A[i, j]`` is the effect
  of the past of process j on process i (row = target, column = source).
* Spectral matrix: ``S(f) = H(f) Sigma H(f)^H`` with
  ``H(f) = (I - sum_k A_k exp(-2 pi i f k))^{-1}``, f in cycles/sample.  This is
  the two-sided density ``S(f) = sum_h Gamma(h) exp(-2 pi i f h)`` with
  ``Gamma(h) = E[x_{t+h} x_t^T]``, so ``integral_{-1/2}^{1/2} S(f) df = Gamma(0)``.
* Delays (D6): ``x_k(t) = x_0(t - delay_k)`` (+ noise); x0 leads.

API: ``datasets()``, ``generate(name, T)``, ``write_all(out_dir)``,
``load(name, T, data_dir=None)``.
CLI: ``python make_data.py [--out DIR] [--list] [--selftest]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.linalg

LENGTHS = (64, 256, 1024, 4096)
DEFAULT_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_NAME = "MANIFEST.json"
FORMAT_VERSION = 2  # 2: per-array sha256/dtype/shape in "arrays"; load() always verifies
BURN_IN = 1000
N_FREQS = 513


# --------------------------------------------------------------------------- utils
def file_seed(base_seed: int, T: int) -> int:
    """Seed for one (dataset, length) file."""
    return int(base_seed) * 1_000_000 + int(T)


def x_sha256(x: np.ndarray) -> str:
    """sha256 of the C-contiguous little-endian float64 bytes of ``x``."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest()


def array_sha256(v: np.ndarray) -> str:
    """sha256 of ``np.ascontiguousarray(v).tobytes()`` (any dtype; bytes as stored)."""
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def array_record(v: np.ndarray) -> dict:
    """Manifest record of one array: sha256, dtype (with byte order) and shape."""
    v = np.asarray(v)
    return {"sha256": array_sha256(v), "dtype": v.dtype.str, "shape": list(v.shape)}


def _f(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def file_name(name: str, T: int) -> str:
    return f"{name}_T{int(T)}.npz"


def companion(A_list) -> np.ndarray:
    """Companion matrix of a VAR(p) with coefficient list [A1, ..., Ap]."""
    P, p = A_list[0].shape[0], len(A_list)
    F = np.zeros((P * p, P * p))
    F[:P, :] = np.hstack(A_list)
    if p > 1:
        F[P:, :-P] = np.eye(P * (p - 1))
    return F


def spectral_radius(A_list) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(companion(A_list)))))


def var_gamma0(A_list, Sigma) -> np.ndarray:
    """Stationary covariance Gamma(0) of a VAR(p) via the companion form (numpy only)."""
    P = Sigma.shape[0]
    F = companion(A_list)
    n = F.shape[0]
    Q = np.zeros((n, n))
    Q[:P, :P] = Sigma
    G = np.linalg.solve(np.eye(n * n) - np.kron(F, F), Q.reshape(-1)).reshape(n, n)
    return G[:P, :P]


def var_autocov(A_list, Sigma, hmax: int) -> np.ndarray:
    """Gamma(h) = E[x_{t+h} x_t^T] for h = 0..hmax via the companion form.

    With F the companion matrix and G its stationary covariance,
    Cov(z_{t+h}, z_t) = F^h G, whose top-left P x P block is Gamma(h).
    Returns shape (hmax + 1, P, P).  Gamma(-h) = Gamma(h)^T.
    """
    P = Sigma.shape[0]
    F = companion(A_list)
    n = F.shape[0]
    Q = np.zeros((n, n))
    Q[:P, :P] = Sigma
    G = np.linalg.solve(np.eye(n * n) - np.kron(F, F), Q.reshape(-1)).reshape(n, n)
    out = np.empty((hmax + 1, P, P))
    M = G
    for h in range(hmax + 1):
        out[h] = M[:P, :P]
        M = F @ M
    return out


def var_spectral_matrix(A_list, Sigma, freqs) -> np.ndarray:
    """Analytic S(f) = H(f) Sigma H(f)^H, H(f) = (I - sum_k A_k e^{-2 pi i f k})^{-1}.

    Returns complex128 array of shape (len(freqs), P, P).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    P = Sigma.shape[0]
    S = np.empty((freqs.size, P, P), dtype=np.complex128)
    for i, f in enumerate(freqs):
        M = np.eye(P, dtype=np.complex128)
        for k, A in enumerate(A_list, start=1):
            M = M - A * np.exp(-2j * np.pi * f * k)
        H = np.linalg.inv(M)
        S[i] = H @ Sigma @ H.conj().T
    return S


def simulate_var(A_list, Sigma, T, rng, burn=BURN_IN) -> np.ndarray:
    """Simulate a VAR(p) from zeros, discard ``burn`` samples, return (P, T)."""
    P, p = Sigma.shape[0], len(A_list)
    n = T + burn
    # Rows e[0:p] are drawn but never used (x[0:p] stays at the zero initial
    # condition).  They are kept deliberately: drawing (n - p, P) instead would
    # shift the RNG stream and change every stored VAR series.  Their effect is
    # in any case washed out by the burn-in.
    e = rng.standard_normal((n, P)) @ np.linalg.cholesky(Sigma).T
    x = np.zeros((n, P))
    for t in range(p, n):
        acc = e[t].copy()
        for k, A in enumerate(A_list, start=1):
            acc += A @ x[t - k]
        x[t] = acc
    return np.ascontiguousarray(x[burn:].T)


# --------------------------------------------------------------------- parameters
BASE_SEED = {
    "iid_gaussian": 1, "bivariate_gaussian": 2, "scaled_offset": 3, "trending": 4,
    "var2_bivariate": 51, "var2_chain": 52, "delayed_noise": 6,
    "narrowband_f0-0.1": 71, "narrowband_f0-0.25": 72, "narrowband_f0-0.4": 73,
    "logistic_maps": 81, "logistic_maps_bidir": 82, "logistic_maps_chaotic": 83,
    "logistic_maps_sugihara": 84, "nonlinear_dependence": 9,
    "contaminated": 10, "cointegrated": 11, "anm": 12, "sparse_precision": 13,
    "shared_envelope": 14, "gauss_ar1_pair": 15,
}

D2_RHO = (0.3, 0.8)
D3_SCALE = (0.2, 2.0, 5.0, 10.0)
D3_OFFSET = (0.0, 1.0, -2.0, 5.0)
D4_SLOPE = (0.0, 0.05, -0.02, 0.1)
VAR2 = {
    "var2_bivariate": dict(
        A1=[[0.5, 0.0], [0.5, 0.4]],
        A2=[[-0.3, 0.0], [-0.2, -0.2]],
        Sigma=[1.0, 0.8]),
    "var2_chain": dict(
        A1=[[0.5, 0.0, 0.0], [0.5, 0.4, 0.0], [0.0, 0.5, 0.3]],
        A2=[[-0.3, 0.0, 0.0], [-0.2, -0.2, 0.0], [0.0, -0.2, -0.1]],
        Sigma=[1.0, 0.8, 0.6]),
}
D6_DELAYS = (0, 1, 3, 5)
D6_NOISE = 0.1
D7_F0 = (0.1, 0.25, 0.4)
D7_PHASE = (0.0, np.pi / 4, -np.pi / 2)
D7_NOISE = 0.05
LOGISTIC = {
    "logistic_maps": dict(r_x=3.8, r_y=3.5, beta_xy=0.1, beta_yx=0.0),
    "logistic_maps_bidir": dict(r_x=3.8, r_y=3.5, beta_xy=0.1, beta_yx=0.02),
    # x (r=3.8, chaotic) drives y (r=3.5): beta_yx multiplies x in y's update, so x -> y only
    "logistic_maps_chaotic": dict(r_x=3.8, r_y=3.5, beta_xy=0.0, beta_yx=0.1),
    # Sugihara et al. 2012 Fig. 1 caption: X(t+1)=X[3.8-3.8X-0.02Y], Y(t+1)=Y[3.5-3.5Y-0.1X]
    "logistic_maps_sugihara": dict(r_x=3.8, r_y=3.5, beta_xy=0.02, beta_yx=0.1),
}
D10_FRAC, D10_MEAN, D10_STD = 0.15, (6.0, -6.0), 0.25
D13_P, D13_DIAG, D13_OFF = 5, 1.0, -0.4
D14_FC, D14_FENV, D14_ENV_AMP, D14_LAG = 0.2, 0.005, 0.5, np.pi / 3
D14_X3_FENV, D14_X3_PHASE_STEP = 0.0031, 0.3
D15_A = ((0.5, 0.3), (0.2, 0.5))
D15_SIGMA_E = ((1.0, 0.2), (0.2, 1.0))


# ----------------------------------------------------------------------- builders
def _d2_draw(T: int) -> np.ndarray:
    rng = np.random.default_rng(file_seed(BASE_SEED["bivariate_gaussian"], T))
    z = rng.standard_normal((4, T))
    x = np.empty_like(z)
    for k, rho in enumerate(D2_RHO):
        x[2 * k] = z[2 * k]
        x[2 * k + 1] = rho * z[2 * k] + np.sqrt(1.0 - rho ** 2) * z[2 * k + 1]
    return x


def _d2_cov() -> np.ndarray:
    C = np.eye(4)
    for k, rho in enumerate(D2_RHO):
        C[2 * k, 2 * k + 1] = C[2 * k + 1, 2 * k] = rho
    return C


def _b_iid(T, seed):
    return {"x": np.random.default_rng(seed).standard_normal((5, T))}


def _b_bivariate(T, seed):
    rho = _f(D2_RHO)
    return {"x": _d2_draw(T), "rho": rho, "true_mi_nats": -0.5 * np.log(1.0 - rho ** 2),
            "pairs": np.array([[0, 1], [2, 3]], dtype=np.int64), "cov_true": _d2_cov()}


def _b_scaled(T, seed):
    s, o = _f(D3_SCALE), _f(D3_OFFSET)
    x = s[:, None] * _d2_draw(T) + o[:, None]
    return {"x": x, "scale": s, "offset": o, "cov_true": np.outer(s, s) * _d2_cov(),
            "corr_true": _d2_cov()}


def _b_trending(T, seed):
    slope = _f(D4_SLOPE)
    return {"x": _d2_draw(T) + slope[:, None] * np.arange(T, dtype=np.float64)[None, :],
            "slope": slope, "cov_true_detrended": _d2_cov()}


def _make_var2(name):
    c = VAR2[name]

    def build(T, seed):
        A1, A2, Sigma = _f(c["A1"]), _f(c["A2"]), np.diag(_f(c["Sigma"]))
        rho = spectral_radius([A1, A2])
        if not rho < 1.0:
            raise RuntimeError(f"{name}: unstable VAR (spectral radius {rho})")
        x = simulate_var([A1, A2], Sigma, T, np.random.default_rng(seed))
        freqs = np.linspace(0.0, 0.5, N_FREQS)
        return {"x": x, "A1": A1, "A2": A2, "Sigma": Sigma, "spectral_radius": _f(rho),
                "freqs": freqs, "S_true": var_spectral_matrix([A1, A2], Sigma, freqs),
                "Gamma0": var_gamma0([A1, A2], Sigma)}
    return build


def _b_delayed(T, seed):
    rng = np.random.default_rng(seed)
    m = max(D6_DELAYS)
    base = rng.standard_normal(T + m)
    x = np.empty((len(D6_DELAYS), T))
    x[0] = base[m:]
    for k, d in enumerate(D6_DELAYS[1:], start=1):
        x[k] = base[m - d:m - d + T] + D6_NOISE * rng.standard_normal(T)
    return {"x": x, "delay": np.array(D6_DELAYS, dtype=np.int64)}


def _make_narrowband(f0):
    def build(T, seed):
        rng = np.random.default_rng(seed)
        t = np.arange(T, dtype=np.float64)
        x = np.stack([np.sin(2 * np.pi * f0 * t + ph) for ph in D7_PHASE])
        x = x + D7_NOISE * rng.standard_normal(x.shape)
        return {"x": x, "f0": _f(f0), "phase_shift": _f(D7_PHASE)}
    return build


def _make_logistic(name):
    c = LOGISTIC[name]

    def build(T, seed):
        rng = np.random.default_rng(seed)
        n = T + BURN_IN
        x, y = np.empty(n), np.empty(n)
        x[0], y[0] = rng.uniform(0.2, 0.8, size=2)
        for t in range(n - 1):
            x[t + 1] = x[t] * (c["r_x"] - c["r_x"] * x[t] - c["beta_xy"] * y[t])
            y[t + 1] = y[t] * (c["r_y"] - c["r_y"] * y[t] - c["beta_yx"] * x[t])
        out = {"x": np.stack([x[BURN_IN:], y[BURN_IN:]])}
        out.update({k: _f(v) for k, v in c.items()})
        return out
    return build


def _b_nonlinear(T, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, T)
    x1 = x0 ** 2 + 0.05 * rng.standard_normal(T)
    th = rng.uniform(0.0, 2 * np.pi, T)
    return {"x": np.stack([x0, x1, np.cos(th), np.sin(th)]), "theta": th}


def _b_contaminated(T, seed):
    x_clean = _d2_draw(T)[2:4].copy()
    rng = np.random.default_rng(seed)
    n_out = int(round(D10_FRAC * T))
    idx = np.sort(rng.choice(T, size=n_out, replace=False))
    mask = np.zeros(T, dtype=bool)
    mask[idx] = True
    mu = _f(D10_MEAN)
    x = x_clean.copy()
    x[:, idx] = mu[:, None] + D10_STD * rng.standard_normal((2, n_out))
    C = _d2_cov()[2:4, 2:4]
    w = n_out / T
    mix = (1 - w) * C + w * D10_STD ** 2 * np.eye(2) + w * (1 - w) * np.outer(mu, mu)
    return {"x": x, "x_clean": x_clean, "clean_cov_true": C, "outlier_mask": mask,
            "outlier_mean": mu, "outlier_std": _f(D10_STD), "mixture_cov_population": mix}


def _b_cointegrated(T, seed):
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.standard_normal(T))
    e = rng.standard_normal((2, T))
    walks = np.cumsum(rng.standard_normal((2, T)), axis=1)
    x = np.stack([w + e[0], 2 * w + e[1], walks[0], walks[1]])
    coint = np.zeros((4, 4), dtype=bool)
    coint[0, 1] = coint[1, 0] = True
    return {"x": x, "coint_vector": _f([2.0, -1.0, 0.0, 0.0]), "is_cointegrated": coint,
            "common_trend": w}


def _b_anm(T, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, T)
    x1 = x0 ** 3 + 0.2 * rng.standard_normal(T)
    x2 = rng.uniform(-1.0, 1.0, T)
    x3 = np.tanh(2 * x2) + 0.2 * rng.standard_normal(T)
    return {"x": np.stack([x0, x1, x2, x3]), "cause": np.array([0, 2], dtype=np.int64),
            "effect": np.array([1, 3], dtype=np.int64)}


def _d13_precision() -> np.ndarray:
    K = D13_DIAG * np.eye(D13_P)
    i = np.arange(D13_P - 1)
    K[i, i + 1] = K[i + 1, i] = D13_OFF
    return K


def _b_sparse_precision(T, seed):
    K = _d13_precision()
    if np.linalg.eigvalsh(K).min() <= 0:
        raise RuntimeError("D13 precision not positive definite")
    C = np.linalg.inv(K)
    C = 0.5 * (C + C.T)
    x = np.linalg.cholesky(C) @ np.random.default_rng(seed).standard_normal((D13_P, T))
    return {"x": x, "precision_true": K, "cov_true": C}


def _b_shared_envelope(T, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    phi0, psi, phi3 = rng.uniform(0.0, 2 * np.pi, size=3)
    env = 1.0 + D14_ENV_AMP * np.sin(2 * np.pi * D14_FENV * t)
    th0 = 2 * np.pi * D14_FC * t + phi0
    env3 = 1.0 + D14_ENV_AMP * np.sin(2 * np.pi * D14_X3_FENV * t + psi)
    walk = np.concatenate([[0.0], np.cumsum(D14_X3_PHASE_STEP * rng.standard_normal(T - 1))])
    th3 = 2 * np.pi * D14_FC * t + phi3 + walk
    theta = np.stack([th0, th0 + D14_LAG, th0, th3])
    envelope = np.stack([env, env, env, env3])
    x = envelope * np.sin(theta)
    x[2] = x[0].copy()
    # phase_offset_rel_x0[3] is a deliberate NaN sentinel: x3 has its own
    # random-walk phase, so no constant offset to x0 exists.  phase_offset_defined
    # marks which entries are meaningful (appended last so earlier arrays keep
    # their position in the npz).
    return {"x": x, "envelope": envelope, "phase": np.angle(np.exp(1j * theta)),
            "phase_offset_rel_x0": _f([0.0, D14_LAG, 0.0, np.nan]), "phi0": _f(phi0),
            "phase_offset_defined": np.array([True, True, True, False], dtype=bool)}


def _b_gauss_ar1(T, seed):
    A, SE = _f(D15_A), _f(D15_SIGMA_E)
    rad = float(np.max(np.abs(np.linalg.eigvals(A))))
    if not rad < 1.0:
        raise RuntimeError("D15 A not stable")
    x = simulate_var([A], SE, T, np.random.default_rng(seed))
    SX = scipy.linalg.solve_discrete_lyapunov(A, SE)
    SX = 0.5 * (SX + SX.T)
    return {"x": x, "A": A, "Sigma_E": SE, "Sigma_X": SX, "Sigma_XY": A @ SX,
            "spectral_radius": _f(rad)}


# ----------------------------------------------------------------------- registry
def _spec(did, P, description, params, builder, derived_from=None, notes=""):
    return dict(id=did, P=P, description=description, params=params, builder=builder,
                derived_from=derived_from, notes=notes, lengths=list(LENGTHS))


def _registry() -> "OrderedDict[str, dict]":
    R = OrderedDict()
    R["iid_gaussian"] = _spec("D1", 5, "5 iid N(0,1) processes (null behaviour, symmetry, bias)",
                              {}, _b_iid)
    R["bivariate_gaussian"] = _spec(
        "D2", 4, "Gaussian pairs (0,1) rho=0.3 and (2,3) rho=0.8, unit variance, zero mean",
        {"rho": list(D2_RHO), "pairs": [[0, 1], [2, 3]]}, _b_bivariate,
        notes="true_mi_nats = -0.5*log(1-rho^2) per pair")
    R["scaled_offset"] = _spec(
        "D3", 4, "D2 (same draw) times per-process scale plus offset",
        {"scale": list(D3_SCALE), "offset": list(D3_OFFSET)}, _b_scaled,
        derived_from="bivariate_gaussian", notes="x = scale*D2 + offset")
    R["trending"] = _spec(
        "D4", 4, "D2 (same draw) plus linear trends slope*t, t = 0..T-1",
        {"slope": list(D4_SLOPE)}, _b_trending, derived_from="bivariate_gaussian")
    for name, desc in (("var2_bivariate", "stable VAR(2), X->Y only (lag1 0.5, lag2 -0.2)"),
                       ("var2_chain", "stable VAR(2) chain X->Y->Z, no direct X->Z")):
        c = VAR2[name]
        R[name] = _spec("D5", len(c["Sigma"]), desc,
                        {"A1": c["A1"], "A2": c["A2"], "Sigma_diag": c["Sigma"],
                         "burn_in": BURN_IN, "n_freqs": N_FREQS}, _make_var2(name),
                        notes="x_t = A1 x_{t-1} + A2 x_{t-2} + e_t; A[i,j] = source j -> target i; "
                              "S_true(f) = H Sigma H^H, two-sided, integral over [-1/2,1/2] = Gamma0")
    R["delayed_noise"] = _spec(
        "D6", 4, "x0 white noise; x1,x2,x3 = x0 delayed by 1,3,5 samples + 0.1 noise",
        {"delay": list(D6_DELAYS), "noise_std": D6_NOISE}, _b_delayed,
        notes="x_k(t) = x0(t - delay_k) + noise; x0 leads; exact (non-circular) delays")
    for f0 in D7_F0:
        R[f"narrowband_f0-{f0:g}"] = _spec(
            "D7", 3, f"sinusoids at f0={f0:g} cycles/sample, phase shifts 0, +pi/4, -pi/2, +0.05 noise",
            {"f0": f0, "phase_shift": list(D7_PHASE), "noise_std": D7_NOISE}, _make_narrowband(f0),
            notes="x_k(t) = sin(2 pi f0 t + phase_shift_k) + noise, t = 0..T-1")
    for name, desc in (("logistic_maps", "coupled logistic maps (Sugihara et al. 2012, Eq. 1 of the main text), "
                                         "y->x only"),
                       ("logistic_maps_bidir", "coupled logistic maps, y->x (0.1) and x->y (0.02)"),
                       ("logistic_maps_chaotic", "coupled logistic maps, x->y only (0.1); the chaotic r=3.8 map "
                                                 "drives the r=3.5 map, so neither series is periodic"),
                       ("logistic_maps_sugihara", "coupled logistic maps with the Sugihara et al. 2012 Fig. 1 "
                                                  "caption parameters: x->y 0.1, y->x 0.02")):
        R[name] = _spec("D8", 2, desc, dict(LOGISTIC[name], burn_in=BURN_IN), _make_logistic(name),
                        notes="process 0 = x, process 1 = y; x(t+1)=x(t)[r_x - r_x x(t) - beta_xy y(t)], "
                              "y(t+1)=y(t)[r_y - r_y y(t) - beta_yx x(t)]; beta_xy>0 means y CAUSES x")
    R["nonlinear_dependence"] = _spec(
        "D9", 4, "x1 = x0^2 + 0.05 noise (Pearson ~0, dependent); (x2,x3) = (cos th, sin th) circle",
        {"x0": "uniform(-1,1)", "noise_std": 0.05, "theta": "uniform(0,2pi)"}, _b_nonlinear)
    R["contaminated"] = _spec(
        "D10", 2, "D2 pair (2,3) rho=0.8 with 15% samples replaced by cluster at (6,-6)",
        {"fraction": D10_FRAC, "outlier_mean": list(D10_MEAN), "outlier_std": D10_STD},
        _b_contaminated, derived_from="bivariate_gaussian",
        notes="x_clean = D2 rows 2,3 (same draw); contamination drawn from this dataset's seed; "
              "mixture_cov_population uses the realised outlier fraction")
    R["cointegrated"] = _spec(
        "D11", 4, "x0 = w + e1, x1 = 2w + e2 (cointegrated [2,-1]); x2, x3 independent random walks",
        {"coint_vector": [2.0, -1.0, 0.0, 0.0], "noise_std": 1.0}, _b_cointegrated,
        notes="2*x0 - x1 = 2 e1 - e2 is stationary; is_cointegrated[i,j] symmetric, False on diagonal")
    R["anm"] = _spec(
        "D12", 4, "additive noise models x0->x1 (cubic) and x2->x3 (tanh)",
        {"x1": "x0^3 + 0.2 N(0,1)", "x3": "tanh(2 x2) + 0.2 N(0,1)", "x0,x2": "uniform(-1,1)"}, _b_anm)
    R["sparse_precision"] = _spec(
        "D13", D13_P, "Gaussian with tridiagonal precision (diag 1.0, off-diag -0.4)",
        {"diag": D13_DIAG, "offdiag": D13_OFF}, _b_sparse_precision)
    R["shared_envelope"] = _spec(
        "D14", 4, "x0,x1 shared slow envelope at 0.2 c/s with pi/3 lag; x2 = copy of x0; "
                  "x3 independent phase-diffusing narrowband with own envelope "
                  "(phase_offset_rel_x0[3] = NaN: no fixed offset; see phase_offset_defined)",
        {"fc": D14_FC, "f_env": D14_FENV, "env_amp": D14_ENV_AMP, "lag": D14_LAG,
         "x3_f_env": D14_X3_FENV, "x3_phase_step_std": D14_X3_PHASE_STEP}, _b_shared_envelope,
        notes="x_k = envelope_k * sin(phase_k); phase = wrapped sin argument (analytic-signal "
              "phase is approx phase - pi/2); x2 is bit-identical to x0; "
              "phase_offset_rel_x0[3] is NaN by design (x3 has its own random-walk phase, no "
              "constant offset) and phase_offset_defined = [True, True, True, False] marks it")
    R["gauss_ar1_pair"] = _spec(
        "D15", 2, "2-node Gaussian AR(1) X(t+1) = A X(t) + E (Oizumi et al. 2016 style)",
        {"A": [list(r) for r in D15_A], "Sigma_E": [list(r) for r in D15_SIGMA_E],
         "burn_in": BURN_IN}, _b_gauss_ar1,
        notes="Sigma_X = stationary Cov(X) from solve_discrete_lyapunov; "
              "Sigma_XY = E[X(t+1) X(t)^T] = A Sigma_X (row: time t+1, column: time t); "
              "Cov(X(t+1) | X(t)) = Sigma_E")
    return R


_REGISTRY = _registry()


def datasets() -> "OrderedDict[str, dict]":
    """Ordered dataset specs (JSON-friendly; builder omitted)."""
    out = OrderedDict()
    for name, s in _REGISTRY.items():
        d = {k: v for k, v in s.items() if k != "builder"}
        d["name"] = name
        d["base_seed"] = BASE_SEED[name]
        out[name] = d
    return out


def _seed_for(name: str, T: int) -> int:
    src = _REGISTRY[name]["derived_from"] or name
    return file_seed(BASE_SEED[src], T)


# Truth arrays allowed to hold non-finite sentinels, mapped to the boolean array
# that marks the defined (finite) entries.  Everything else must be finite.
NONFINITE_ALLOWED = {("shared_envelope", "phase_offset_rel_x0"): "phase_offset_defined"}


def generate(name: str, T: int) -> dict:
    """Generate one dataset at length T; returns {'x': (P, T) float64, <truth>: ...}."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {list(_REGISTRY)}")
    spec = _REGISTRY[name]
    out = spec["builder"](int(T), file_seed(BASE_SEED[name], T))
    x = np.ascontiguousarray(out["x"], dtype=np.float64)
    if x.shape != (spec["P"], T) or not np.all(np.isfinite(x)):
        raise RuntimeError(f"{name} T={T}: bad x (shape {x.shape}, finite={np.isfinite(x).all()})")
    out["x"] = x
    for k, v in out.items():
        v = np.asarray(v)
        if v.dtype.kind not in "fc":
            continue  # bool / int arrays are always finite
        if (name, k) in NONFINITE_ALLOWED:
            bad = ~np.isfinite(v)
            ok = NONFINITE_ALLOWED[(name, k)]
            if not np.array_equal(bad, ~np.asarray(out[ok], dtype=bool)):
                raise RuntimeError(f"{name} T={T}: non-finite entries of {k!r} do not match ~{ok}")
        elif not np.all(np.isfinite(v)):
            raise RuntimeError(f"{name} T={T}: truth array {k!r} has non-finite values")
    return out


def _manifest_entry(name: str, T: int, arrays: dict) -> dict:
    s = _REGISTRY[name]
    e = {"file": file_name(name, T), "name": name, "id": s["id"], "T": int(T), "P": s["P"],
         "seed": _seed_for(name, T), "params": s["params"], "description": s["description"],
         "notes": s["notes"], "x_sha256": x_sha256(arrays["x"]),
         "arrays": {k: array_record(v) for k, v in arrays.items()},
         "truth": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                   for k, v in arrays.items() if k != "x"}}
    if s["derived_from"]:
        e["derived_from"] = s["derived_from"]
        e["own_seed"] = file_seed(BASE_SEED[name], T)
    return e


def write_all(out_dir=None) -> Path:
    """Write every dataset x length npz plus MANIFEST.json; returns manifest path.

    Stale ``*.npz`` files in ``out_dir`` that are not part of the regenerated
    manifest are removed (and reported), so the directory always matches it.
    """
    out = Path(out_dir) if out_dir is not None else DEFAULT_DIR
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, spec in _REGISTRY.items():
        for T in spec["lengths"]:
            arrays = generate(name, T)
            np.savez(out / file_name(name, T), **arrays)
            files.append(_manifest_entry(name, T, arrays))
    manifest = {"format_version": FORMAT_VERSION, "array_order": "ps (processes, observations)",
                "numpy_version": np.__version__, "seed_rule": "base_seed * 1_000_000 + T",
                "x_sha256_rule": "sha256 of C-contiguous little-endian float64 bytes of x",
                "array_sha256_rule": "per file, arrays[key].sha256 = sha256 of "
                                     "np.ascontiguousarray(v).tobytes() for every npz key (x "
                                     "included); dtype is numpy dtype.str (with byte order)",
                "files": files}
    path = out / MANIFEST_NAME
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    keep = {e["file"] for e in files}
    for stale in sorted(out.glob("*.npz")):
        if stale.name not in keep:
            stale.unlink()
            print(f"removed stale {stale}")
    return path


def load(name: str, T: int, data_dir=None) -> dict:
    """Load one written file as a dict of arrays, verified against MANIFEST.json.

    Always verifies (there is deliberately no opt-out): raises FileNotFoundError
    if the manifest is missing, KeyError if the file is not listed in it, and
    ValueError if the npz key set differs from the manifest's or any array's
    sha256 / dtype / shape (or the legacy x_sha256) does not match.
    """
    d = Path(data_dir) if data_dir is not None else DEFAULT_DIR
    fname = file_name(name, T)
    mpath = d / MANIFEST_NAME
    if not mpath.is_file():
        raise FileNotFoundError(f"{mpath} not found; cannot verify {fname} "
                                "(regenerate with make_data.py --out)")
    manifest = json.loads(mpath.read_text())
    entries = {e["file"]: e for e in manifest["files"]}
    e = entries.get(fname)
    if e is None:
        raise KeyError(f"{fname} missing from manifest {mpath}")
    if "arrays" not in e:
        raise ValueError(f"{mpath}: format_version {manifest.get('format_version')} has no "
                         "per-array hashes; regenerate with make_data.py --out")
    with np.load(d / fname, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if set(arrays) != set(e["arrays"]):
        raise ValueError(f"{fname}: npz keys {sorted(arrays)} != manifest keys {sorted(e['arrays'])}")
    for k, v in arrays.items():
        rec = array_record(v)
        if rec != e["arrays"][k]:
            diff = [f for f in ("sha256", "dtype", "shape") if rec[f] != e["arrays"][k][f]]
            raise ValueError(f"{fname}: array {k!r} {'/'.join(diff)} mismatch with manifest")
    if x_sha256(arrays["x"]) != e["x_sha256"]:
        raise ValueError(f"{fname}: x sha256 mismatch with manifest")
    return arrays


# ----------------------------------------------------------------------- selftest
def _var_ols(x: np.ndarray, p: int):
    P, T = x.shape
    Y = x[:, p:].T
    Z = np.hstack([x[:, p - k:T - k].T for k in range(1, p + 1)])
    B = np.linalg.lstsq(Z, Y, rcond=None)[0].T
    return [B[:, (k - 1) * P:k * P] for k in range(1, p + 1)]


def _expect_raise(fn, exc) -> str:
    """Run fn(); return '' if it raised ``exc``, else a description of what happened."""
    try:
        fn()
    except exc:
        return ""
    except Exception as err:  # noqa: BLE001 - report the wrong exception type
        return f"raised {type(err).__name__}: {err}"
    return "did not raise"


def _selftest_roundtrip(check) -> None:
    """write_all -> load round trip in a temp dir, plus tampering / fail-closed checks."""
    with tempfile.TemporaryDirectory(prefix="make_data_selftest_") as tmp:
        out = Path(tmp) / "data"
        out.mkdir()
        np.savez(out / "stale_T1.npz", x=np.zeros((1, 1)))
        write_all(out)
        check("write_all removes stale npz", not (out / "stale_T1.npz").exists(), "stale_T1.npz")
        files = json.loads((out / MANIFEST_NAME).read_text())["files"]
        n_ok, mism = 0, []
        for e in files:
            got = load(e["name"], e["T"], out)
            ref = generate(e["name"], e["T"])
            same = set(got) == set(ref) and all(
                got[k].dtype == np.asarray(ref[k]).dtype
                and np.array_equal(got[k], ref[k], equal_nan=got[k].dtype.kind in "fc")
                for k in ref)
            n_ok += same
            if not same:
                mism.append(e["file"])
        check("write_all -> load round trip (all hashes verified, arrays equal generate())",
              n_ok == len(files), f"{n_ok}/{len(files)} files" + (f"; mismatch {mism}" if mism else ""))

        name, T = "bivariate_gaussian", 64
        fname = file_name(name, T)
        tam = Path(tmp) / "tampered"
        tam.mkdir()
        shutil.copy(out / MANIFEST_NAME, tam / MANIFEST_NAME)
        with np.load(out / fname, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        arrs["rho"] = arrs["rho"].copy()
        arrs["rho"][0] += 1e-12  # one truth value; x untouched
        np.savez(tam / fname, **arrs)
        r = _expect_raise(lambda: load(name, T, tam), ValueError)
        check("tampered truth value raises ValueError", r == "", r or "rho[0] += 1e-12")
        arrs2 = dict(arrs, rho=load(name, T, out)["rho"], extra=np.zeros(1))
        np.savez(tam / fname, **arrs2)
        r = _expect_raise(lambda: load(name, T, tam), ValueError)
        check("extra npz key raises ValueError", r == "", r or "key 'extra'")
        del arrs2["extra"], arrs2["cov_true"]
        np.savez(tam / fname, **arrs2)
        r = _expect_raise(lambda: load(name, T, tam), ValueError)
        check("missing npz key raises ValueError", r == "", r or "key 'cov_true' removed")
        (tam / MANIFEST_NAME).unlink()
        shutil.copy(out / fname, tam / fname)
        r = _expect_raise(lambda: load(name, T, tam), FileNotFoundError)
        check("missing MANIFEST.json raises FileNotFoundError", r == "", r or "fail closed")


def selftest() -> bool:
    results = []

    def check(label, ok, detail):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")

    for name in ("var2_bivariate", "var2_chain"):
        d = generate(name, 4096)
        check(f"D5 {name} spectral radius", d["spectral_radius"] < 1, f"{float(d['spectral_radius']):.4f}")
        A1h, A2h = _var_ols(d["x"], 2)
        err = max(np.abs(A1h - d["A1"]).max(), np.abs(A2h - d["A2"]).max())
        check(f"D5 {name} OLS recovers A1/A2", err < 0.1, f"max abs err {err:.4f} (tol 0.1)")
        G = 2 * np.trapezoid(d["S_true"].real, d["freqs"], axis=0)
        rel = np.abs(G - d["Gamma0"]).max() / np.abs(d["Gamma0"]).max()
        check(f"D5 {name} integral S_true = Gamma0", rel < 1e-3, f"rel err {rel:.2e}")
        # S_true[f] = sum_h Gamma(h) e^{-2 pi i f h}, Gamma(h) = E[x_{t+h} x_t^T] from the
        # companion form (independent of H(f)).  A transposed (= conjugated) S_true must fail.
        H = 400
        Gam = var_autocov([d["A1"], d["A2"]], d["Sigma"], H)
        ph = np.exp(-2j * np.pi * np.outer(d["freqs"], np.arange(1, H + 1)))  # (F, H)
        S_ref = (Gam[0][None] + np.einsum("fh,hij->fij", ph, Gam[1:])
                 + np.einsum("fh,hji->fij", ph.conj(), Gam[1:]))
        scale = np.abs(S_ref).max()
        e_ok = np.abs(d["S_true"] - S_ref).max() / scale
        e_tr = np.abs(np.swapaxes(d["S_true"], 1, 2) - S_ref).max() / scale
        check(f"D5 {name} S_true = sum_h Gamma(h) e^(-2 pi i f h) (discriminating)",
              e_ok < 1e-10 and e_tr > 1e-2,
              f"rel err {e_ok:.2e} (tol 1e-10); transposed S_true rel err {e_tr:.2e} (must be > 1e-2)")
    for T in LENGTHS:
        d = generate("delayed_noise", T)
        x0 = d["x"][0]
        est = []
        for k in range(d["x"].shape[0]):
            cc = [np.dot(x0[:T - L], d["x"][k][L:]) for L in range(0, 11)]
            est.append(int(np.argmax(cc)))
        check(f"D6 delays T={T}", est == list(d["delay"]), f"estimated {est}")
    for f0 in D7_F0:
        for T in LENGTHS:
            d = generate(f"narrowband_f0-{f0:g}", T)
            fr = np.fft.rfftfreq(T)
            peaks = [float(fr[1:][np.argmax(np.abs(np.fft.rfft(r))[1:])]) for r in d["x"]]
            ok = all(abs(p - f0) <= 0.5 / T + 1e-12 for p in peaks)
            check(f"D7 f0={f0:g} T={T} FFT peak", ok, f"peaks {peaks} (bin 1/T={1 / T:.5f})")
            # Least squares x_k ~ a cos(w t) + b sin(w t) => sin(w t + phi), phi = atan2(a, b).
            t = np.arange(T, dtype=np.float64)
            Z = np.stack([np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)], axis=1)
            ab = np.linalg.lstsq(Z, d["x"].T, rcond=None)[0]  # (2, P)
            phi = np.arctan2(ab[0], ab[1])
            rel = np.angle(np.exp(1j * (phi - phi[0])))
            err = float(np.abs(np.angle(np.exp(1j * (rel - d["phase_shift"])))).max())
            check(f"D7 f0={f0:g} T={T} LS phase rel x0 = (0, +pi/4, -pi/2)", err < 0.05,
                  f"phases {np.round(rel, 4).tolist()}, max err {err:.4f} rad (tol 0.05)")
    for name in LOGISTIC:
        d = generate(name, 4096)
        x = d["x"]
        check(f"D8 {name} bounded", bool(((x > 0) & (x < 1)).all()), f"range [{x.min():.3f}, {x.max():.3f}]")
        X, Y = x[0], x[1]  # process 0 = x, process 1 = y; beta_xy multiplies y in x's update
        rx = np.abs(X[1:] - X[:-1] * (d["r_x"] - d["r_x"] * X[:-1] - d["beta_xy"] * Y[:-1])).max()
        ry = np.abs(Y[1:] - Y[:-1] * (d["r_y"] - d["r_y"] * Y[:-1] - d["beta_yx"] * X[:-1])).max()
        check(f"D8 {name} stored series satisfy both recurrences", max(rx, ry) < 1e-10,
              f"max residual x {rx:.1e}, y {ry:.1e} (tol 1e-10)")
        n_distinct = [len(np.unique(np.round(x[k, -200:], 12))) for k in range(2)]
        if name == "logistic_maps":
            check("D8 logistic_maps: undriven y (r=3.5) is a period-4 orbit (tied-data fixture)",
                  n_distinct[1] == 4, f"distinct values in last 200 samples: x {n_distinct[0]}, y {n_distinct[1]}")
        else:
            check(f"D8 {name}: both series aperiodic (> 150 distinct values in last 200 samples)",
                  min(n_distinct) > 150, f"distinct values: x {n_distinct[0]}, y {n_distinct[1]}")
    d = generate("cointegrated", 4096)
    r = d["coint_vector"] @ d["x"]
    growth = lambda s, k=1024: np.var(s[k:] - s[:-k]) / np.var(np.diff(s))  # Var(k-step)/Var(1-step)
    g_r, g_w = growth(r), min(growth(d["x"][k]) for k in range(4))
    vr, vw = np.var(r), min(np.var(d["x"][k]) for k in range(4))
    check("D11 cointegrating residual", vr < 0.05 * vw and g_r < 0.1 * g_w,
          f"var resid {vr:.2f} vs min walk {vw:.1f}; growth resid {g_r:.2f} vs min walk {g_w:.1f}")
    d = generate("sparse_precision", 4096)
    pd_ok = np.linalg.eigvalsh(d["precision_true"]).min() > 0
    check("D13 precision positive definite", pd_ok, f"min eig {np.linalg.eigvalsh(d['precision_true']).min():.4f}")
    err = np.abs(np.linalg.inv(np.cov(d["x"])) - d["precision_true"]).max()
    check("D13 sample precision T=4096", err < 0.2, f"max abs err {err:.4f} (tol 0.2)")
    d = generate("gauss_ar1_pair", 4096)
    A, SX, SE = d["A"], d["Sigma_X"], d["Sigma_E"]
    res = np.abs(SX - (A @ SX @ A.T + SE)).max()
    kron = np.linalg.solve(np.eye(4) - np.kron(A, A), SE.reshape(-1)).reshape(2, 2)
    check("D15 Lyapunov Sigma_X = A Sigma_X A^T + Sigma_E", res < 1e-10 and np.allclose(SX, kron, atol=1e-10),
          f"residual {res:.2e}, kron diff {np.abs(SX - kron).max():.2e}")
    check("D15 stability", d["spectral_radius"] < 1, f"spectral radius {float(d['spectral_radius']):.4f}")
    x = d["x"]
    C = x[:, 1:] @ x[:, :-1].T / (x.shape[1] - 1)  # sample E[x(t+1) x(t)^T]
    dn, dt = np.abs(C - d["Sigma_XY"]).max(), np.abs(C - d["Sigma_XY"].T).max()
    check("D15 sample E[x(t+1) x(t)^T] T=4096 closer to Sigma_XY than its transpose", dn < dt,
          f"max abs dist {dn:.4f} vs transpose {dt:.4f}")
    d = generate("bivariate_gaussian", 4096)
    c = np.corrcoef(d["x"])
    check("D2 sample rho T=4096", abs(c[0, 1] - 0.3) < 0.05 and abs(c[2, 3] - 0.8) < 0.05,
          f"rho01 {c[0, 1]:.3f}, rho23 {c[2, 3]:.3f}")
    d = generate("shared_envelope", 256)
    check("D14 x2 identical to x0", bool(np.array_equal(d["x"][0], d["x"][2])), "bitwise")
    same = all(x_sha256(generate(n, 256)["x"]) == x_sha256(generate(n, 256)["x"]) for n in _REGISTRY)
    check("determinism (regenerate T=256)", same, f"{len(_REGISTRY)} datasets")
    _selftest_roundtrip(check)
    n_fail = results.count(False)
    print(f"selftest: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


# ---------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help=f"output directory (default {DEFAULT_DIR})")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--selftest", action="store_true", help="run numpy/scipy self-checks")
    a = ap.parse_args(argv)
    if a.list:
        for name, s in datasets().items():
            print(f"{s['id']:<4} {name:<22} P={s['P']}  T={s['lengths']}  {s['description']}")
        return 0
    if a.selftest and not selftest():
        return 1
    if a.out is not None or not a.selftest:
        path = write_all(a.out)
        n = len(json.loads(path.read_text())["files"])
        print(f"wrote {n} files + {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
