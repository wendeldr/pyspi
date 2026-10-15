"""Probes for the phi family (PROJECT venv: pyspi.lib.phi_native, pyspi.statistics.infotheory).

Every function has the probe signature ``fn(x, run)`` (x = raw dataset (P, T) after optional process
selection; run = the manifest entry) and returns an array that pyspi_outputs.py stores under ``value``.
The unit (phitype, normalization) is read from ``run["unit_id"]`` (infotheory.phi.<star|Geo>_norm-<0|1>),
and the arguments are passed exactly as IntegratedInformation.bivariate passes them
(pyspi/statistics/infotheory.py:549-573: partition [1, 2], params {"tau": 1}, options
{type_of_phi, type_of_dist "Gauss", normalization}).

pop            phi_native on the POPULATION covariances of run["dataset"] (see _pop_cov):
               gauss_ar1_pair: Cov_X = Cov_Y = Sigma_X, Cov_XY = Sigma_XY.T (the dataset stores
               E[X(t+1)X(t)']; pyspi's cov_comp Cov_XY is E[X(t)X(t+1)']); var2_bivariate: lag-0/lag-1
               autocovariances of the VAR(2) truth.  star -> [phi, I]; Geo -> [phi].
nullpop        phi_native on two population models with no cross-coupling (A diagonal):
               (a) A = diag(0.5, 0.3), Sigma_E = I (fully disconnected: Phi* = Phi_G = 0 analytically);
               (b) A = diag(0.5, 0.3), Sigma_E = [[1, .5], [.5, 1]] (instantaneous noise correlation only).
               -> [phi_a, phi_b].
star_details   per pair i < j of the path-preprocessed data (pyspi Data: kernel raw, pipeline z-score):
               phi_comp with a spy on scipy.optimize.minimize inside phi_native -> (P, P, 4) array
               [phi, I, beta_opt, nit] (upper triangle; NaN elsewhere).  pop_details is the same on the
               gauss_ar1_pair population covariances -> (1, 1, 4).
geo_details    per pair i < j: phi_g_gauss -> (P, P, 5) [phi, A_p[0,0], A_p[1,1], max|offdiag A_p|,
               AL iterations] (iterations counted from the logdet calls of phi_g_gauss_al: 2 per
               iteration + 2 final).
unmirrored     Directed.multivariate (pyspi/base.py:85-94) on the pyspi Data object: every ordered pair
               computed, NO lower-triangle mirror -> (P, P).  A[i, j] = bivariate(data, i, j) on rows [i, j].
is_tilde       pyspi's own objective i_s_objective (phi_native.py:118-145) captured from inside
               phi_star_gauss on the gauss_ar1_pair population covariances and evaluated at
               beta in {0.5, 1, 2} -> [I_s(0.5), I_s(1), I_s(2)] (sign flipped back to I_s).
exact_delay    exact one-step-delay pair built from delayed_noise row 0: rows [u(t+1), u(t)] (so the
               present of row 1 is the past of row 0), phi_comp on (a) z-scored rows (pyspi Data) and
               (b) raw rows -> [Re a, Im a, Re b, Im b] (phi_native.logdet can return complex dtype).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")


def _unit(run):
    tail = run["unit_id"].split(".")[-1]           # star_norm-0
    phitype, norm = tail.split("_norm-")
    return phitype, int(norm)


def _opts(run):
    phitype, norm = _unit(run)
    return {"type_of_phi": phitype, "type_of_dist": "Gauss", "normalization": norm}


def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        d = Data(x, dim_order="ps", normalise=False, detrend=False)
    elif path == "pipeline":
        d = Data(x, dim_order="ps")
    elif path == "pipeline-detrend":
        d = Data(x, dim_order="ps", normalise=True, detrend=True)
    else:
        raise ValueError(path)
    return d


def _pop_cov(dataset="gauss_ar1_pair"):
    """Population (Cov_X, Cov_XY, Cov_Y) in phi_native's orientation (Cov_XY = E[X(t-1) X(t)']).

    gauss_ar1_pair: dataset truth (Sigma_XY stores E[X(t+1)X(t)'], so it is transposed).
    var2_bivariate: lag-0 / lag-1 autocovariances of the VAR(2) truth (A1, A2, Sigma) from the
    companion-form Lyapunov equation; s_t = [x_t; x_{t-1}], Var(s) = [[G0, G1], [G1', G0]] with
    G1 = E[x_t x_{t-1}'], so Cov_XY = E[x_{t-1} x_t'] = G1'.
    """
    import make_data
    d = make_data.load(dataset, 1024)
    if dataset == "gauss_ar1_pair":
        SX = np.asarray(d["Sigma_X"], dtype=np.float64)
        SXY = np.asarray(d["Sigma_XY"], dtype=np.float64)
        return SX, SXY.T.copy(), SX.copy()
    if dataset == "var2_bivariate":
        import scipy.linalg
        A1 = np.asarray(d["A1"], dtype=np.float64)
        A2 = np.asarray(d["A2"], dtype=np.float64)
        Sg = np.asarray(d["Sigma"], dtype=np.float64)
        n = A1.shape[0]
        F = np.block([[A1, A2], [np.eye(n), np.zeros((n, n))]])
        Q = np.zeros((2 * n, 2 * n))
        Q[:n, :n] = Sg
        V = scipy.linalg.solve_discrete_lyapunov(F, Q)
        V = 0.5 * (V + V.T)
        G0 = V[:n, :n]
        G1 = V[:n, n:]
        return G0.copy(), G1.T.copy(), G0.copy()
    raise ValueError(dataset)


def pop(x, run):
    from pyspi.lib import phi_native as pn
    phitype, norm = _unit(run)
    CX, CXY, CY = _pop_cov(run["dataset"])
    Z = np.array([1, 2])
    if phitype == "star":
        phi, I, beta = pn.phi_star_gauss(CX, CXY, CY, Z, beta_init=1.0, normalization=norm)
        return np.array([phi, I], dtype=np.float64)
    phi, _, _ = pn.phi_g_gauss(CX, CXY, CY, Z, method=None, normalization=norm)
    return np.array([phi], dtype=np.float64)


def _ar1_pop(A, SE):
    import scipy.linalg
    SX = scipy.linalg.solve_discrete_lyapunov(A, SE)
    SX = 0.5 * (SX + SX.T)
    return SX, (A @ SX).T.copy(), SX.copy()


def nullpop(x, run):
    from pyspi.lib import phi_native as pn
    phitype, norm = _unit(run)
    A = np.diag([0.5, 0.3])
    out = []
    for SE in (np.eye(2), np.array([[1.0, 0.5], [0.5, 1.0]])):
        CX, CXY, CY = _ar1_pop(A, SE)
        if phitype == "star":
            v = pn.phi_star_gauss(CX, CXY, CY, np.array([1, 2]), 1.0, norm)[0]
        else:
            v = pn.phi_g_gauss(CX, CXY, CY, np.array([1, 2]), None, norm)[0]
        out.append(v)
    return np.array(out, dtype=np.float64)


class _Spy:
    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, fun, x0, args=(), **kw):
        res = self.real(fun, x0, args=args, **kw)
        self.calls.append(dict(fun=fun, args=args, res=res))
        return res


def _star_one(pn, CX, CXY, CY, norm):
    real = pn.minimize
    spy = _Spy(real)
    pn.minimize = spy
    try:
        phi, I, beta = pn.phi_star_gauss(CX, CXY, CY, np.array([1, 2]), beta_init=1.0, normalization=norm)
    finally:
        pn.minimize = real
    nit = float(spy.calls[-1]["res"].nit) if spy.calls else np.nan
    return [np.real(phi), np.real(I), float(np.real(beta)), nit]


def star_details(x, run):
    from pyspi.lib import phi_native as pn
    _, norm = _unit(run)
    z = _data(x, run["path"]).to_numpy(squeeze=True)
    P = z.shape[0]
    out = np.full((P, P, 4), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            c = pn.cov_comp(z[[i, j]], 1, True)
            out[i, j] = _star_one(pn, c["Cov_X"], c["Cov_XY"], c["Cov_Y"], norm)
    return out


def pop_details(x, run):
    from pyspi.lib import phi_native as pn
    _, norm = _unit(run)
    CX, CXY, CY = _pop_cov(run["dataset"])
    return np.array(_star_one(pn, CX, CXY, CY, norm), dtype=np.float64).reshape(1, 1, 4)


def geo_details(x, run):
    from pyspi.lib import phi_native as pn
    _, norm = _unit(run)
    z = _data(x, run["path"]).to_numpy(squeeze=True)
    P = z.shape[0]
    out = np.full((P, P, 5), np.nan)
    real = pn.logdet
    for i in range(P):
        for j in range(i + 1, P):
            c = pn.cov_comp(z[[i, j]], 1, True)
            n = [0]

            def counting(X):
                n[0] += 1
                return real(X)
            pn.logdet = counting
            try:
                phi, CEp, Ap = pn.phi_g_gauss(c["Cov_X"], c["Cov_XY"], c["Cov_Y"], np.array([1, 2]),
                                              method=None, normalization=norm)
            finally:
                pn.logdet = real
            if Ap is None:
                out[i, j] = [np.real(phi), np.nan, np.nan, np.nan, np.nan]
                continue
            # calls: 2 per AL iteration + 2 in the final phi_G (+1 per h_gauss when normalization=1)
            iters = (n[0] - 2 - (2 if norm == 1 else 0)) / 2.0
            out[i, j] = [np.real(phi), Ap[0, 0], Ap[1, 1], max(abs(Ap[0, 1]), abs(Ap[1, 0])), iters]
    return out


def unmirrored(x, run):
    from pyspi.base import Directed
    from pyspi.statistics.infotheory import IntegratedInformation
    phitype, norm = _unit(run)
    spi = IntegratedInformation(phitype=phitype, delay=1, normalization=norm)
    data = _data(x, run["path"])
    return np.asarray(Directed.multivariate(spi, data), dtype=np.float64)


def is_tilde(x, run):
    from pyspi.lib import phi_native as pn
    CX, CXY, CY = _pop_cov()
    real = pn.minimize
    spy = _Spy(real)
    pn.minimize = spy
    try:
        pn.phi_star_gauss(CX, CXY, CY, np.array([1, 2]), beta_init=1.0, normalization=0)
    finally:
        pn.minimize = real
    call = spy.calls[-1]
    # the objective returns a shape-(1,) array for a shape-(1,) beta (as L-BFGS-B calls it)
    return np.array([-np.asarray(call["fun"](np.array([b]), *call["args"]), dtype=np.float64).reshape(-1)[0]
                     for b in (0.5, 1.0, 2.0)])


def exact_delay(x, run):
    from pyspi.lib import phi_native as pn
    u = np.asarray(x[0], dtype=np.float64)
    pair = np.stack([u[1:], u[:-1]])
    opts = _opts(run)
    out = []
    for path in ("pipeline", "kernel"):
        z = _data(pair, path).to_numpy(squeeze=True)
        try:
            v = complex(pn.phi_comp(z, np.array([1, 2]), {"tau": 1}, opts))
        except Exception:                      # recorded as NaN (bivariate's except does the same)
            v = complex(np.nan, np.nan)
        out += [v.real, v.imag]
    return np.array(out, dtype=np.float64)
