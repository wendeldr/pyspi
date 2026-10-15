"""Probes for the cdt_pairwise family (anm, cds, reci, igci): PROJECT venv only.

Each function is called by ``audit/checks/pyspi_outputs.py`` as ``fn(x, run)`` with the raw dataset
array ``x`` (P, T) (after optional process selection) and the manifest entry ``run``; it returns the
array stored under ``value`` in ``out/pyspi/<run_id>.npz``. The probes call pyspi / cdt / sklearn
exactly as ``pyspi/statistics/causal.py`` does, to expose intermediates the plain SPI path does not
return (fitted GP hyperparameters, residuals), to compute the pairwise decision matrix that cdt's
``predict_proba`` would use, or to feed pyspi an inline input that is not one of the shared datasets.

Preprocessing follows ``run["path"]`` the way ``pyspi_outputs.make_pyspi_data`` does:
kernel = Data(x, normalise=False, detrend=False); pipeline = Data(x) (z-score, ddof=1);
pipeline-detrend = Data(x, detrend=True).
"""
import sys
import warnings

import numpy as np

sys.path.insert(0, "/media/dan/Data/git/pyspi/tests")


def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _spi(identifier):
    from pyspi.statistics import causal
    cls = {"anm": causal.AdditiveNoiseModel, "cds": causal.ConditionalDistributionSimilarity,
           "reci": causal.RegressionErrorCausalInference,
           "igci": causal.InformationGeometricConditionalIndependence}[identifier]
    return cls()


def _matrix(identifier, data):
    """The pyspi matrix exactly as the plain SPI path computes it (harness.compute_spi: multivariate + NaN diagonal)."""
    from regression import harness
    return harness.compute_spi(_spi(identifier), data)


# ------------------------------------------------------------------------------------------- anm
def _gp(xc, yc, random_state=42):
    """The GP of pyspi's corrected_anm_score (causal.py:19), fitted on (T, 1) columns."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianProcessRegressor(random_state=random_state).fit(xc, yc)


def anm_theta(x, run):
    """(2, P, P): sklearn's fitted kernel_.theta = (log constant, log length_scale) for every ordered pair (i, j),
    GP of y = z[j] on x = z[i] as corrected_anm_score fits it (causal.py:19); NaN diagonal."""
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((2, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                out[:, i, j] = _gp(z[i], z[j]).kernel_.theta
    return out


def anm_collapse(x, run):
    """(3, P, P) per ordered pair (i, j): [fitted RBF length scale, std(r)/std(y) with r = predict(x) - y
    (causal.py:20-21), median of r / (-(alpha / constant) * y)] for sklearn's fit; NaN diagonal."""
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((3, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            gp = _gp(z[i], z[j])
            c, l = np.exp(gp.kernel_.theta)
            y = z[j].ravel()
            r = gp.predict(z[i]).reshape(-1, 1).ravel() - y
            out[:, i, j] = [l, r.std() / y.std(), np.median(r / (-(1e-10 / c) * y))]
    return out


def anm_random_state(x, run):
    """(P, P): max over random_state in {7, None} of |score(random_state) - score(42)|, with score =
    normalized_hsic(predict(x) - y, x) as in corrected_anm_score (causal.py:18-22). 0 means random_state has no effect."""
    from cdt.causality.pairwise.ANM import normalized_hsic
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            s = []
            for rs in (42, 7, None):
                gp = _gp(z[i], z[j], rs)
                s.append(normalized_hsic(gp.predict(z[i]).reshape(-1, 1) - z[j], z[i]))
            out[i, j] = max(abs(s[1] - s[0]), abs(s[2] - s[0]))
    return out


def anm_patch_global(x, run):
    """(3,): [cdt ANM.anm_score is pyspi's corrected_anm_score after importing pyspi.statistics.causal,
    cdt's own ANM().anm_score on two (T, 1) columns returns pyspi's value (patched behaviour),
    cdt's unpatched formula gp.predict(x) - y would broadcast to (T, T)] as 1.0/0.0."""
    from cdt.causality.pairwise import ANM
    from pyspi.statistics.causal import AdditiveNoiseModel
    z = _data(x, run["path"]).to_numpy()
    patched = ANM.anm_score is AdditiveNoiseModel.corrected_anm_score
    a = ANM().anm_score(z[0], z[1])
    b = AdditiveNoiseModel().bivariate(_data(x, run["path"]), i=0, j=1)
    gp = _gp(z[0], z[1])
    broadcast_shape = (gp.predict(z[0]) - z[1]).shape
    return np.array([float(patched), float(a == b), float(broadcast_shape == (z.shape[1], z.shape[1]))])


def anm_labels(x, run):
    """(3,): ['linear' in AdditiveNoiseModel.labels (causal.py:15), 'nonlinear' in the class labels,
    'nonlinear' in pyspi/config.yaml AdditiveNoiseModel labels]."""
    import yaml
    from pathlib import Path
    import pyspi
    from pyspi.statistics.causal import AdditiveNoiseModel
    cfg = yaml.safe_load((Path(pyspi.__file__).parent / "config.yaml").read_text())
    cfg_labels = cfg[".statistics.causal"]["AdditiveNoiseModel"]["labels"]
    lab = AdditiveNoiseModel.labels
    return np.array([float("linear" in lab), float("nonlinear" in lab), float("nonlinear" in cfg_labels)])


def pairdiff(x, run):
    """(P, P): D = A.T - A from the pyspi matrix A of run['identifier'] (the cdt predict_proba form
    score(j, i) - score(i, j) for the 'lower = causal' scores anm, cds, reci). D[i, j] > 0 is evidence for i -> j.
    For igci the matrix itself is returned (already antisymmetric, positive = i -> j)."""
    A = _matrix(run["identifier"], _data(x, run["path"]))
    if run["identifier"] == "igci":
        return A
    return A.T - A


# ------------------------------------------------------------------------------------------- cds
def spike_input(T=512, seed=2026):
    """Inline spike-heavy pair (mirrored in refs/cdt_pairwise.py): x is a tight bulk at 0 plus a few large
    excursions (two at each of +-1..+-6, so at most 12 samples on either side of the bulk): after cdt's trimmed
    re-standardisation one x-bin holds the bulk and every other bin has <= 12 samples, for any T; y is iid N(0, 1).
    Returns (2, T)."""
    rng = np.random.default_rng(seed)
    n_sp = 24
    x = 1e-3 * rng.standard_normal(T)
    levels = np.repeat(np.array([-6.0, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]), 2)
    pos = rng.choice(T, n_sp, replace=False)
    x[pos] = levels * (1 + 0.01 * rng.standard_normal(n_sp))
    y = rng.standard_normal(T)
    return np.vstack([x, y])


def cds_spike(x, run):
    """(2, 2) pyspi cds matrix on spike_input(T=run['T']) through run['path']; the dataset x is ignored."""
    return _matrix("cds", _data(spike_input(run["T"]), run["path"]))


# ------------------------------------------------------------------------------------------ igci
def cubic_pair(T, seed):
    """Inline deterministic invertible pair (mirrored in refs/cdt_pairwise.py): X ~ U(0, 1), Y = X^3."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, T)
    return np.vstack([u, u ** 3])


def igci_cubic(x, run):
    """(2, 2): mean over 20 seeds (1000..1019) of the pyspi igci matrix on cubic_pair(T, seed) through run['path'];
    the dataset x is ignored."""
    acc = np.zeros((2, 2))
    for s in range(20):
        acc += _matrix("igci", _data(cubic_pair(run["T"], 1000 + s), run["path"]))
    return acc / 20


def igci_raw_bivariate(x, run):
    """(2, 2): pyspi igci through the two-raw-array bivariate API, [[nan, bivariate(u, v)], [bivariate(v, u), nan]]
    for (u, v) = cubic_pair(1024, seed 0); parse_bivariate builds Data with add_process (base.py:33-36)."""
    u, v = cubic_pair(1024, 0)
    spi = _spi("igci")
    out = np.full((2, 2), np.nan)
    out[0, 1] = spi.bivariate(u.copy(), v.copy())
    out[1, 0] = spi.bivariate(v.copy(), u.copy())
    return out


def igci_labels(x, run):
    """(3,): ['unsigned' in the class labels (causal.py:60), issigned() (base.py Unsigned -> False),
    'unsigned' in config.yaml InformationGeometricConditionalIndependence labels]."""
    import yaml
    from pathlib import Path
    import pyspi
    from pyspi.statistics.causal import InformationGeometricConditionalIndependence as C
    cfg = yaml.safe_load((Path(pyspi.__file__).parent / "config.yaml").read_text())
    cl = cfg[".statistics.causal"]["InformationGeometricConditionalIndependence"]["labels"]
    return np.array([float("unsigned" in C.labels), float(C().issigned()), float("unsigned" in cl)])


def anm_replica_projectenv(x, run):
    """(P, P): the numpy/scipy replica _kernels_distance.anm_replica (imports neither sklearn nor cdt) evaluated in the
    PROJECT venv (scipy 1.17.1 LAPACK, pyspi's single-thread setting) at sklearn's fitted theta for each ordered pair.
    Separates the replica's arithmetic from the comparison venv's LAPACK build (scipy 1.18.1)."""
    sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
    import _kernels_distance as kd
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                c, l = np.exp(_gp(z[i], z[j]).kernel_.theta)
                out[i, j] = kd.anm_replica(z[i].ravel(), z[j].ravel(), c, l)
    return out
