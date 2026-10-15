"""Probes for the spectral_coherence family (cohmag, phase, icoh); PROJECT venv only.

Each function is called as ``fn(x, run)`` by ``audit/checks/pyspi_outputs.py`` with the raw
dataset array ``x`` (P, T) and the manifest run dict; it returns the array stored as pyspi's
``value``.  Parameters that are not config identifiers travel in ``run["probe"]["kwargs"]``
(part of the cache key).  Every value returned here is computed by pyspi's own classes
(``pyspi/statistics/spectral.py``) on a ``pyspi.data.Data`` object built the way
``pyspi_outputs.make_pyspi_data`` builds it; nothing is recomputed outside pyspi except where a
docstring says so.

Functions
---------
custom_band   one SPI with parameters that are not config identifiers (e.g. fmin = fmax = 0.25,
              a single FFT bin) -> (P, P)
eeg_bands     the 11 retained EEG identifiers of one class, built from the band edges of
              ``pyspi/workflow/retained185.yaml`` (identifier strings asserted against
              ``audit/units.yaml`` eeg_identifiers), sharing one Data object as the Calculator
              does -> (11, P, P)
empty_max     ``statistic="max"`` on a band that holds no FFT bin: returns [code, n_bins]
              with code 1 = ValueError "zero-size array", 2 = another exception, 0 = no error
              (the exception is caught here because compare.py fails any run whose pyspi side
              raised; the reference judges the code)
truncated     a config identifier on ``x[:, :T_sub]`` (T_sub not 5-smooth exercises the
              library's n_fft = scipy.fftpack.next_fast_len rule and zero padding) -> (P, P)
constant_channel  a config identifier with one channel replaced by a constant (zero power after
              the constant detrend; kernel path, so no z-score NaN) -> (P, P)
grid          pyspi's own frequency vector ``data.spectral_mv["freq"]`` after computing the
              identifier on ``x[:, :T_sub]`` (or all of x) -> (n_bins,)
"""
from pathlib import Path

import numpy as np
import yaml

AUDIT = Path(__file__).resolve().parents[2]
CLASSES = {"cohmag": "CoherenceMagnitude", "phase": "CoherencePhase", "icoh": "ImaginaryCoherence"}

# Band edges of the retained EEG identifiers (pyspi/workflow/retained185.yaml:227-247, identical
# for CoherencePhase, CoherenceMagnitude and ImaginaryCoherence): Hz / 2048, in units.yaml order.
EEG_BANDS = [
    (1 / 2048, 4 / 2048), (1 / 2048, 70 / 2048), (1 / 2048, 250 / 2048), (4 / 2048, 8 / 2048),
    (8 / 2048, 12 / 2048), (12 / 2048, 30 / 2048), (30 / 2048, 70 / 2048), (70 / 2048, 150 / 2048),
    (70 / 2048, 250 / 2048), (150 / 2048, 250 / 2048), None,   # None = {"fs": 1} (full band)
]


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


def _cls(short):
    from pyspi.statistics import spectral
    return getattr(spectral, CLASSES[short])


def _unit_short(run):
    return run["unit_id"].split(".")[1]


def _compute(spi, data):
    S = np.array(spi.multivariate(data), dtype=np.float64, copy=True)
    np.fill_diagonal(S, np.nan)
    return S


def _identifier_params(identifier):
    doc = yaml.safe_load((AUDIT / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u.get("identifier_params", {}):
            return u["identifier_params"][identifier]
    raise KeyError(identifier)


def custom_band(x, run):
    kw = dict(run["probe"]["kwargs"])
    path = kw.pop("path", "kernel")
    spi = _cls(_unit_short(run))(**kw)
    return _compute(spi, _data(x, path))


def eeg_bands(x, run):
    kw = run["probe"].get("kwargs", {})
    short = _unit_short(run)
    doc = yaml.safe_load((AUDIT / "units.yaml").read_text())
    expected = next(u for u in doc["units"] if u["unit_id"] == run["unit_id"])["eeg_identifiers"]
    data = _data(x, kw.get("path", "pipeline-detrend"))
    out = []
    for band, ident in zip(EEG_BANDS, expected):
        spi = _cls(short)(fs=1) if band is None else _cls(short)(fmin=band[0], fmax=band[1])
        if spi.identifier != ident:
            raise AssertionError(f"constructed {spi.identifier} != units.yaml {ident}")
        out.append(_compute(spi, data))
    return np.stack(out)


def empty_max(x, run):
    kw = dict(run["probe"]["kwargs"])
    path = kw.pop("path", "kernel")
    data = _data(x, path)
    spi = _cls(_unit_short(run))(statistic="max", **kw)
    try:
        _compute(spi, data)
        code = 0.0
    except ValueError as exc:
        code = 1.0 if "zero-size array" in str(exc) else 2.0
    except Exception:
        code = 2.0
    freq = data.spectral_mv["freq"]
    n_bins = float(((freq >= kw["fmin"]) & (freq <= kw["fmax"])).sum())
    return np.array([code, n_bins])


def truncated(x, run):
    kw = run["probe"]["kwargs"]
    spi = _cls(_unit_short(run))(**_identifier_params(run["identifier"]))
    assert spi.identifier == run["identifier"], (spi.identifier, run["identifier"])
    return _compute(spi, _data(x[:, : int(kw["T_sub"])], kw.get("path", "kernel")))


def grid(x, run):
    kw = run["probe"].get("kwargs", {})
    T_sub = int(kw.get("T_sub", x.shape[1]))
    spi = _cls(_unit_short(run))(**_identifier_params(run["identifier"]))
    data = _data(x[:, :T_sub], kw.get("path", "kernel"))
    _compute(spi, data)
    return np.asarray(data.spectral_mv["freq"], dtype=np.float64)


def constant_channel(x, run):
    """Config identifier on a copy of x whose channel ``kwargs["channel"]`` is replaced by the
    constant ``kwargs["value"]`` (zero power after the library's constant detrend) -> (P, P)."""
    kw = run["probe"]["kwargs"]
    x = np.array(x, dtype=np.float64, copy=True)
    x[int(kw["channel"])] = float(kw["value"])
    spi = _cls(_unit_short(run))(**_identifier_params(run["identifier"]))
    return _compute(spi, _data(x, kw.get("path", "kernel")))
