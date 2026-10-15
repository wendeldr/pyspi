"""gwtau family probes (PROJECT venv): distance.gwtau (pyspi GromovWasserstainTau).

Every probe returns an array computed by pyspi's own code on rows that the reference side
(refs/gwtau.py, comparison venv) rebuilds with plain numpy.  Each derivation here is
mirrored in refs/gwtau.py (``derive`` / ``sinusoid_rows`` / ``UNEQUAL_SPECS``).  The probe
kind is the last field of the run_id.

  unmirrored   base.Directed.multivariate on the GromovWasserstainTau instance: every ordered
               pair (i, j) is computed by bivariate(i, j) and nothing is mirrored, so the lower
               triangle is pyspi's own A[j, i] computation (Undirected.multivariate, base.py:132-138,
               would overwrite it with a copy of the upper triangle).
  unequal-*    GromovWasserstainTau.gwtau(x_i[:n1], x_j[:n2]) for the (i, j, n1, n2) list in
               UNEQUAL_SPECS: the unequal-length lambda branch of wass_sorted (distance.py:286-300),
               which the Calculator cannot reach (Data stores equal-length rows).
  invar        rows (x0, -x0, x0 + 5, 3 x0 - 2, x1) through the run's path.
  perm*        rows jointly permuted in time with one permutation (PERM_SEED) through the run's path.
  rev          rows jointly reversed in time.
  sin-grid25   noise-free sin(2 pi 0.25 t + phi), phi = (0, pi/4, -pi/2), t = 0..T-1, run's path.
  sin-freq     noise-free rows sin(2 pi 0.1 t), sin(2 pi 0.4 t), sin(2 pi 0.1 t + pi/4),
               sin(2 pi 0.4 t - pi/2), run's path.
  sin-eeg      noise-free rows sin(2 pi f t + phi) for f = 10/256 (10 Hz alpha at fs = 256 Hz) and
               f = 0.2 (50 Hz line at fs = 250 Hz), phi = (0, pi/4, -pi/2) each, run's path.
  xfreq        the dataset narrowband_f0-0.1 rows (x) stacked over the narrowband_f0-0.4 rows at
               the same T (loaded with make_data), run's path.
  fs           sampling-rate probe (ignores x): the same continuous pair
               h0(s) = sin(2 pi s) + 0.5 sin(2 pi 3 s + 1), h1(s) = sin(2 pi 1.3 s + 0.4), s in seconds,
               over FS_DURATION seconds sampled at each rate in FS_RATES; returns pyspi's gwtau of the
               pair (bivariate on a 2-row Data through the run's path) at each rate, shape (len(FS_RATES),).

Paths: "pipeline" = Data(x) (Calculator default: z-score ddof=1), "kernel" =
Data(x, normalise=False, detrend=False), as in pyspi_outputs.make_pyspi_data.
"""
import numpy as np

PERM_SEED = 20261015
UNEQUAL_SPECS = ((0, 1, 37, 53), (1, 2, 100, 80), (0, 3, 256, 129), (3, 4, 200, 201),
                 (2, 4, 5, 7), (1, 3, 64, 64), (4, 0, 17, 255))
FS_DURATION = 16.0
FS_RATES = (16, 32, 64, 128, 256)


def _tag(run):
    return run["run_id"].split("|")[-1]


def _data(X, path):
    from pyspi.data import Data
    X = np.array(X, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(X, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(X, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(X, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _spi():
    from pyspi.statistics.distance import GromovWasserstainTau
    return GromovWasserstainTau()


def _matrix(X, path):
    """Calculator-style matrix: Undirected.multivariate, then NaN diagonal (calculator.py:329-332)."""
    S = np.array(_spi().multivariate(_data(X, path)), dtype=np.float64)
    np.fill_diagonal(S, np.nan)
    return S


def sinusoid_rows(kind, T):
    t = np.arange(T, dtype=np.float64)
    if kind == "sin-grid25":
        return np.vstack([np.sin(2 * np.pi * 0.25 * t + p) for p in (0.0, np.pi / 4, -np.pi / 2)])
    if kind == "sin-freq":
        return np.vstack([np.sin(2 * np.pi * 0.1 * t), np.sin(2 * np.pi * 0.4 * t),
                          np.sin(2 * np.pi * 0.1 * t + np.pi / 4), np.sin(2 * np.pi * 0.4 * t - np.pi / 2)])
    if kind == "sin-eeg":
        return np.vstack([np.sin(2 * np.pi * f * t + p) for f in (10 / 256, 0.2) for p in (0.0, np.pi / 4, -np.pi / 2)])
    raise ValueError(kind)


def fs_rows(fs):
    s = np.arange(int(round(FS_DURATION * fs)), dtype=np.float64) / fs
    return np.vstack([np.sin(2 * np.pi * 1.0 * s) + 0.5 * np.sin(2 * np.pi * 3.0 * s + 1.0),
                      np.sin(2 * np.pi * 1.3 * s + 0.4)])


def derive(kind, x, T=None):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[1] if T is None else T
    if kind == "invar":
        return np.vstack([x[0], -x[0], x[0] + 5.0, 3.0 * x[0] - 2.0, x[1]])
    if kind.startswith("perm"):
        return x[:, np.random.default_rng(PERM_SEED).permutation(T)]
    if kind == "rev":
        return x[:, ::-1]
    if kind.startswith("sin-"):
        return sinusoid_rows(kind, T)
    raise ValueError(kind)


def unmirrored(x, run):
    from pyspi.base import Directed
    S = np.array(Directed.multivariate(_spi(), _data(x, run["path"])), dtype=np.float64)
    np.fill_diagonal(S, np.nan)
    return S


def unequal(x, run):
    from pyspi.statistics.distance import GromovWasserstainTau
    x = np.asarray(x, dtype=np.float64)
    assert run["path"] == "kernel", "unequal-length probe is defined on raw rows"
    return np.array([GromovWasserstainTau.gwtau(x[i, :n1], x[j, :n2]) for i, j, n1, n2 in UNEQUAL_SPECS])


def derived(x, run):
    return _matrix(derive(_tag(run), x), run["path"])


def xfreq(x, run):
    import make_data   # audit/checks is on sys.path in pyspi_outputs.py
    other = np.asarray(make_data.load("narrowband_f0-0.4", run["T"])["x"], dtype=np.float64)
    return _matrix(np.vstack([np.asarray(x, dtype=np.float64), other]), run["path"])


def fs(x, run):
    out = []
    for r in FS_RATES:
        out.append(float(_spi().bivariate(_data(fs_rows(r), run["path"]), i=0, j=1)))
    return np.array(out)
