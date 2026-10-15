"""Probes for the ccm family (causal.ccm.E-None, causal.ccm.E-1, causal.ccm.E-10): PROJECT venv only.

Each function is called by ``audit/checks/pyspi_outputs.py`` as ``fn(x, run)`` with the raw dataset array
``x`` (P, T) (after optional process selection) and the manifest entry ``run``. Parameters that are not
config identifiers are read from the last ``|``-separated field of ``run["run_id"]`` (the tag), e.g.
``xmap-E2-Lfull`` or ``xmap-E10-Lfull-X10``. Preprocessing follows ``run["path"]`` exactly like
``pyspi_outputs.make_pyspi_data`` (kernel = raw, pipeline = z-score ddof=1, pipeline-detrend = detrend + z-score).

* ``kernel_xmap``: pyEDM.CCM called directly (the delegate, isolated from pyspi) with ``random=False`` and a
  single library size, i.e. one contiguous library of the first L valid rows (cppEDM CCM.cc L289-325); the
  DataFrame is built as pyspi builds it (causal.py:91-96). A[i, j] = LibMeans column "proc_i:proc_j"
  (columns=proc_i, target=proc_j), A[j, i] = column "proc_j:proc_i" of the same call; NaN diagonal.
* ``curve``: pyspi's own cached CCM curve ``data.ccm[E]`` (P, P, 22) after ``ConvergentCrossMapping.multivariate``.
* ``stats3``: the mean / max / diff matrices of the three pyspi SPIs computed on ONE Data object (they share
  the cached curve), stacked (3, P, P), NaN diagonal as in Calculator.compute (calculator.py:332).
* ``schedule``: spies on the pyEDM.CCM call pyspi makes (libSizes string, E) and returns the library-size
  schedule cppEDM expanded; the spy runs CCM with sample=1 (only the LibSize column is used).
* ``embed_choice``: E-None only. Spies on pyEDM.EmbedDimension and pyEDM.CCM; returns per process
  [E passed to CCM for the pair, rho(E = 1..10) from EmbedDimension]; CCM runs with sample=1.
* ``short_series``: runs ``multivariate`` on the first T_short and T_short + 1 samples; returns status codes.
* ``diagonal``: the raw ``multivariate`` matrix of a direct call (no Calculator diagonal fill).
* ``seeds``: pyEDM.CCM as pyspi calls it but averaged over seeds 1..16 (bias test of the random-library curve).
* ``aggregate``: the three pyspi statistics (mean, max, diff; raw ``multivariate`` output, no diagonal fill) and the
  cached curve ``data.ccm[E]`` from ONE Data object, as (P, P, 3 + 22): [..., 0:3] = statistics, [..., 3:] = curve.
"""
import sys
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")
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


def _tag(run):
    """'xmap-E10-Lfull-X10' -> {'E': '10', 'L': 'full', 'X': '10'}."""
    out = {}
    for tok in run["run_id"].split("|")[-1].split("-")[1:]:
        out[tok[0]] = tok[1:]
    return out


def _E(tag):
    return None if tag.get("E", "None") == "None" else int(tag["E"])


def _df(z):
    import pandas as pd
    M, N = z.shape
    return pd.DataFrame(np.concatenate([np.atleast_2d(np.arange(0, N)), z]).T,
                        columns=["index"] + [f"proc{p}" for p in range(M)])


def kernel_xmap(x, run):
    import pyEDM
    tag = _tag(run)
    E = int(tag["E"])
    z = _data(x, run["path"]).to_numpy(squeeze=True)
    P, N = z.shape
    L = N - (E - 1) if tag.get("L", "full") == "full" else int(tag["L"])
    X = int(tag.get("X", 0))
    df = _df(z)
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            r = pyEDM.CCM(dataFrame=df, E=E, columns=f"proc{i}", target=f"proc{j}", libSizes=str(L),
                          sample=1, random=False, exclusionRadius=X)
            assert list(r.columns) == ["LibSize", f"proc{i}:proc{j}", f"proc{j}:proc{i}"], list(r.columns)
            A[i, j] = r[f"proc{i}:proc{j}"].values[0]
            A[j, i] = r[f"proc{j}:proc{i}"].values[0]
    return A


def _ccm(statistic, E):
    from pyspi.statistics.causal import ConvergentCrossMapping
    return ConvergentCrossMapping(statistic=statistic, embedding_dimension=E)


def curve(x, run):
    E = _E(_tag(run))
    data = _data(x, run["path"])
    spi = _ccm("mean", E)
    spi.multivariate(data)
    return np.array(data.ccm[spi.key], dtype=np.float64)


def stats3(x, run):
    E = _E(_tag(run))
    data = _data(x, run["path"])
    out = []
    for st in ("mean", "max", "diff"):
        S = np.array(_ccm(st, E).multivariate(data), dtype=np.float64)
        np.fill_diagonal(S, np.nan)
        out.append(S)
    return np.stack(out)


class _Spy:
    """Stands in for the pyEDM module inside pyspi.statistics.causal; records the calls pyspi makes."""

    def __init__(self, sample1=True):
        import pyEDM
        self._real, self.ccm_calls, self.embed_calls, self._sample1 = pyEDM, [], [], sample1

    def CCM(self, **kw):
        res = self._real.CCM(**dict(kw, sample=1) if self._sample1 else kw)
        self.ccm_calls.append((dict(kw), res))
        return res

    def EmbedDimension(self, **kw):
        res = self._real.EmbedDimension(**kw)
        self.embed_calls.append((dict(kw), res))
        return res


def _with_spy(fn, sample1=True):
    from pyspi.statistics import causal
    spy = _Spy(sample1)
    real = causal.pyEDM
    causal.pyEDM = spy
    try:
        fn()
    finally:
        causal.pyEDM = real
    return spy


def schedule(x, run):
    """[E, lowerE, upperE, inc, n_sizes_cppEDM, last_size_cppEDM, n_kept, kept sizes (22)] for the first pair."""
    E = _E(_tag(run))
    data = _data(x, run["path"])
    spy = _with_spy(lambda: _ccm("mean", E).multivariate(data))
    kw, res = spy.ccm_calls[0]
    lo, hi, inc = (int(v) for v in kw["libSizes"].split())
    sizes = res["LibSize"].values.astype(float)
    kept = sizes[:22]
    return np.concatenate([[kw["E"], lo, hi, inc, len(sizes), sizes[-1], len(kept)],
                           np.pad(kept, (0, 22 - len(kept)), constant_values=np.nan)])


def embed_choice(x, run):
    """(P, 11): column 0 = E pyspi passed to pyEDM.CCM for the pair containing the process; 1..10 = rho(E)."""
    data = _data(x, run["path"])
    spy = _with_spy(lambda: _ccm("mean", None).multivariate(data))
    P = data.n_processes
    out = np.full((P, 11), np.nan)
    for p, (kw, res) in enumerate(spy.embed_calls):
        assert kw["columns"] == f"proc{p}" and kw["lib"] == kw["pred"], kw
        assert list(res["E"].astype(int)) == list(range(1, 11))
        out[p, 1:] = res["rho"].values
    for kw, _ in spy.ccm_calls:
        i, j = int(kw["columns"][4:]), int(kw["target"][4:])
        out[i, 0] = out[j, 0] = kw["E"]
    return out


def short_series(x, run):
    """[status(T_short), status(T_short + 1), finite fraction of the T_short + 1 result]; status 0 ok,
    1 RuntimeError 'increment 0 is invalid', 2 other error. T_short from the tag (e.g. 'short-E10-T60')."""
    tag = _tag(run)
    E, Ts = _E(tag), int(tag["T"])
    codes, fin = [], np.nan
    for T in (Ts, Ts + 1):
        try:
            S = np.array(_ccm("mean", E).multivariate(_data(x[:, :T], run["path"])), dtype=float)
            codes.append(0)
            off = ~np.eye(S.shape[0], dtype=bool)
            fin = float(np.isfinite(S[off]).mean())
        except Exception as exc:  # noqa: BLE001 - Calculator.compute does the same (calculator.py:336-339)
            codes.append(1 if "increment 0 is invalid" in str(exc) else 2)
    return np.array(codes + [fin], dtype=float)


def diagonal(x, run):
    E = _E(_tag(run))
    return np.array(_ccm("mean", E).multivariate(_data(x, run["path"])), dtype=np.float64)


def seeds(x, run):
    """Bias test for the random-library curve: pyEDM.CCM called exactly as pyspi calls it (causal.py:91-142: same
    DataFrame, E, libSizes string, sample=100) but with seeds 1..16 instead of 42. Returns (P, P) = mean over the 16
    seeds of the mean over library sizes (pyspi's ``mean`` statistic), NaN diagonal; tag e.g. 'seeds-E1-S16'."""
    import pyEDM
    tag = _tag(run)
    E, n_seeds = int(tag["E"]), int(tag.get("S", 16))
    z = _data(x, run["path"]).to_numpy(squeeze=True)
    P, N = z.shape
    upperE = int(np.floor((N - E - 1) / 10) * 10)
    lowerE = int(np.ceil(2 * E / 10) * 10)
    lib_sizes = f"{lowerE} {upperE} {int((upperE - lowerE) / 21)}"
    df = _df(z)
    A = np.zeros((P, P))
    for s in range(1, n_seeds + 1):
        for i in range(P):
            for j in range(i + 1, P):
                r = pyEDM.CCM(dataFrame=df, E=E, columns=f"proc{i}", target=f"proc{j}", libSizes=lib_sizes,
                              sample=100, seed=s)
                A[i, j] += r.iloc[:, 1].values[:22].mean() / n_seeds
                A[j, i] += r.iloc[:, 2].values[:22].mean() / n_seeds
    np.fill_diagonal(A, np.nan)
    return A


def aggregate(x, run):
    """(P, P, 25): [..., 0] mean, [..., 1] max, [..., 2] diff (ConvergentCrossMapping.multivariate on one Data
    object, so all three read the same cached curve), [..., 3:25] = data.ccm[E] (the curve they aggregate)."""
    E = _E(_tag(run))
    data = _data(x, run["path"])
    stats = []
    for st in ("mean", "max", "diff"):
        spi = _ccm(st, E)
        stats.append(np.array(spi.multivariate(data), dtype=np.float64))
    c = np.array(data.ccm[spi.key], dtype=np.float64)
    return np.concatenate([np.stack(stats, axis=-1), c], axis=-1)
