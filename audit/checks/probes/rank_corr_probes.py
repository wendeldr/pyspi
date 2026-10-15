"""Probes for the rank_corr family (basic.spearmanr, basic.kendalltau). PROJECT venv.

Each function is called by ``pyspi_outputs.py`` as ``fn(x, run)`` with ``x`` the raw
dataset array (P, T) and ``run`` the manifest entry; ``run["identifier"]`` names the SPI
(spearmanr, spearmanr-sq, kendalltau, kendalltau-sq). The reference script
``refs/rank_corr.py`` applies exactly the same input transformation (``tie_round``,
``split_realisations``) before computing its own value.

Functions
---------
ties_kernel      x rounded to 1 decimal, ``Data(normalise=False)``, Calculator-style matrix.
ties_pipeline    x rounded to 1 decimal, ``Data()`` default (z-score, ddof=1).
directed_kernel  ``Directed.multivariate`` (both ordered pairs computed, no Undirected mirror)
                 on the raw x; diagonal NaN.
directed_ties    the same on x rounded to 1 decimal.
labels_fresh     labels in a fresh interpreter: [default, squared, default-after-squared] instances
                 and the instances a minimal Calculator builds from a 2-SPI config.
call_count       number of ``bivariate`` calls made by one ``multivariate`` on x.
realisations     x split into R = 2 realisations (dim_order 'psr'); records what ``bivariate`` returns.
array_input      ``spi.bivariate(x0, x1)`` with raw arrays (parse_bivariate wrapping) vs Data input.
"""
import json
import subprocess
import sys
import textwrap

import numpy as np

TIE_DECIMALS = 1


def tie_round(x):
    """Identical rounding in both venvs: numpy round-half-to-even at 1 decimal."""
    return np.round(np.asarray(x, dtype=np.float64), TIE_DECIMALS)


def split_realisations(x, R=2):
    """(P, T) -> (P, T//R, R): realisation r is the r-th contiguous block of T//R samples."""
    P, T = x.shape
    n = T // R
    return np.stack([x[:, r * n:(r + 1) * n] for r in range(R)], axis=2)


def _spi(identifier):
    from pyspi.statistics.basic import KendallTau, SpearmanR
    cls = {"spearmanr": SpearmanR, "kendalltau": KendallTau}[identifier.replace("-sq", "")]
    return cls(squared=identifier.endswith("-sq"))


def _data(x, normalise):
    from pyspi.data import Data
    return Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps", normalise=normalise, detrend=False)


def _calc_matrix(spi, data):
    S = np.array(spi.multivariate(data), dtype=np.float64, copy=True)
    np.fill_diagonal(S, np.nan)
    return S


def ties_kernel(x, run):
    return _calc_matrix(_spi(run["identifier"]), _data(tie_round(x), normalise=False))


def ties_pipeline(x, run):
    return _calc_matrix(_spi(run["identifier"]), _data(tie_round(x), normalise=True))


def _directed(x, run):
    from pyspi.base import Directed
    spi = _spi(run["identifier"])
    A = np.array(Directed.multivariate(spi, _data(x, normalise=False)), dtype=np.float64, copy=True)
    np.fill_diagonal(A, np.nan)
    return A


def directed_kernel(x, run):
    return _directed(x, run)


def directed_ties(x, run):
    return _directed(tie_round(x), run)


_LABEL_SCRIPT = textwrap.dedent(r'''
    import json, os, sys, tempfile, contextlib, io
    from pyspi.statistics.basic import SpearmanR, KendallTau
    name = sys.argv[1]
    cls = {"SpearmanR": SpearmanR, "KendallTau": KendallTau}[name]
    out = {}
    out["fresh_default"] = list(cls().labels)
    out["squared"] = list(cls(squared=True).labels)
    out["default_after_squared"] = list(cls().labels)
    # Calculator path (fresh interpreter needed: class lists are mutated above, so run a 2nd process)
    print(json.dumps(out))
''')

_CALC_SCRIPT = textwrap.dedent(r'''
    import json, os, sys, tempfile, contextlib, io
    name = sys.argv[1]
    cfg = "." + "statistics.basic:\n  " + name + ":\n    configs:\n      - squared: True\n      - squared: False\n"
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=sys.argv[2]); os.write(fd, cfg.encode()); os.close(fd)
    from pyspi.calculator import Calculator
    with contextlib.redirect_stdout(io.StringIO()):
        calc = Calculator(configfile=path)
    os.remove(path)
    out = {k: list(v.labels) for k, v in calc.spis.items()}
    out["same_list_object"] = len({id(v.labels) for v in calc.spis.values()}) == 1
    print(json.dumps(out))
''')


def _label_code(labels):
    """[n 'signed', n 'unsigned', 'linear' in, 'nonlinear' in, len]"""
    return [labels.count("signed"), labels.count("unsigned"), int("linear" in labels),
            int("nonlinear" in labels), len(labels)]


def labels_fresh(x, run):
    """Rows: fresh default, squared, default after squared (fresh interpreter), then the
    Calculator-built '-sq' and plain instances (second fresh interpreter); last row
    [same_list_object, 0, 0, 0, 0]."""
    import tempfile
    name = {"spearmanr": "SpearmanR", "kendalltau": "KendallTau"}[run["identifier"].replace("-sq", "")]
    r1 = subprocess.run([sys.executable, "-c", _LABEL_SCRIPT, name], capture_output=True, text=True, check=True)
    d1 = json.loads(r1.stdout.strip().splitlines()[-1])
    tmpdir = tempfile.gettempdir()
    r2 = subprocess.run([sys.executable, "-c", _CALC_SCRIPT, name, tmpdir], capture_output=True, text=True, check=True)
    d2 = json.loads(r2.stdout.strip().splitlines()[-1])
    base = run["identifier"].replace("-sq", "")
    rows = [_label_code(d1["fresh_default"]), _label_code(d1["squared"]), _label_code(d1["default_after_squared"]),
            _label_code(d2[base + "-sq"]), _label_code(d2[base]), [int(d2["same_list_object"]), 0, 0, 0, 0]]
    return np.asarray(rows, dtype=np.float64)


def call_count(x, run):
    """[bivariate calls made by one multivariate, P, P(P-1)/2, max|A - A.T| of the output]"""
    spi = _spi(run["identifier"])
    orig = spi.bivariate
    n = [0]

    def counting(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    spi.bivariate = counting
    A = _calc_matrix(spi, _data(x, normalise=True))
    P = x.shape[0]
    asym = float(np.nanmax(np.abs(A - A.T)))
    return np.asarray([n[0], P, P * (P - 1) / 2, asym], dtype=np.float64)


def realisations(x, run):
    """Data with R = 2 realisations (dim_order 'psr', normalise=False), pair (0, 1).
    Returns [ndim of bivariate result, size of result, scalar value or NaN,
    multivariate raised (1/0), multivariate A[0,1] or NaN]."""
    from pyspi.data import Data
    spi = _spi(run["identifier"])
    x3 = split_realisations(np.asarray(x, dtype=np.float64))
    data = Data(x3, dim_order="psr", normalise=False, detrend=False)
    b = np.asarray(spi.bivariate(data, i=0, j=1))
    val = float(b) if b.size == 1 else np.nan
    try:
        A = spi.multivariate(data)
        raised, a01 = 0, float(A[0, 1])
    except Exception:
        raised, a01 = 1, np.nan
    return np.asarray([b.ndim, b.size, val, raised, a01], dtype=np.float64)


def array_input(x, run):
    """[bivariate(x0, x1) raw arrays, bivariate(Data(x, normalise=False), 0, 1),
    mean and std (ddof=1) of row 0 and row 1 of the Data parse_bivariate builds]."""
    from pyspi.data import Data
    spi = _spi(run["identifier"])
    x0, x1 = np.array(x[0], dtype=np.float64), np.array(x[1], dtype=np.float64)
    v_arr = float(spi.bivariate(x0.copy(), x1.copy()))
    v_data = float(spi.bivariate(_data(x[:2], normalise=False), i=0, j=1))
    d = Data()
    d.add_process(x0.copy())
    d.add_process(x1.copy())
    z = d.to_numpy(squeeze=True)
    return np.asarray([v_arr, v_data, z[0].mean(), z[0].std(ddof=1), z[1].mean(), z[1].std(ddof=1)],
                      dtype=np.float64)
