"""pec family probes (PROJECT venv): pyspi PowerEnvelopeCorrelation on inputs or
parameters that the plain manifest SPI path cannot express.

Every function has the pyspi_outputs probe signature ``fn(x, run)`` and returns
the array stored under ``value``.  The SPI is constructed directly from
``pyspi.statistics.misc.PowerEnvelopeCorrelation`` with the config parameters of
``run["identifier"]`` (pyspi/config.yaml:1360-1391) and post-processed like
Calculator.compute (diagonal NaN), via tests/regression/harness.compute_spi.
The matching reference computations are in audit/checks/refs/pec.py.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))

from regression import harness  # noqa: E402

PARAMS = {  # identifier -> (orth, log, absolute); config.yaml:1360-1391
    "pec": (False, False, False),
    "pec_orth": (True, False, False),
    "pec_log": (False, True, False),
    "pec_orth_log": (True, True, False),
    "pec_orth_abs": (True, False, True),
    "pec_orth_log_abs": (True, True, True),
}
ORDER = ["pec", "pec_orth", "pec_log", "pec_orth_log", "pec_orth_abs", "pec_orth_log_abs"]

# constant-envelope probe (trace stage, records new_finding_candidates[0] of the non-orth units)
CONST_T = 256
CONST_M = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]


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


def _spi(orth, log, absolute):
    from pyspi.statistics.misc import PowerEnvelopeCorrelation
    return PowerEnvelopeCorrelation(orth=orth, log=log, absolute=absolute)


def _compute(x, path, orth, log, absolute):
    return harness.compute_spi(_spi(orth, log, absolute), _data(x, path))


def constant_envelope_signals(m, T=CONST_T, seed=5):
    """(1 + m a(t)) sin(2 pi 0.25 t) and (1 + m b(t)) sin(2 pi 0.25 t + 0.7);
    a, b = first T samples of a 45-point moving average ('valid') of (T + 44) standard
    normals each, drawn a first from one default_rng(seed) stream (trace-stage probe)."""
    rng = np.random.default_rng(seed)
    k = np.ones(45) / 45
    a = np.convolve(rng.standard_normal(T + 44), k, mode="valid")[:T]
    b = np.convolve(rng.standard_normal(T + 44), k, mode="valid")[:T]
    t = np.arange(T, dtype=np.float64)
    return np.stack([(1 + m * a) * np.sin(2 * np.pi * 0.25 * t),
                     (1 + m * b) * np.sin(2 * np.pi * 0.25 * t + 0.7)])


def constant_envelope(x, run):
    """pyspi value of pair (0,1) for each m in CONST_M (dataset x is NOT used; the
    narrowband_f0-0.25 dataset only names the carrier f0 = 0.25).  Path from run."""
    o, l, a = PARAMS[run["identifier"]]
    return np.array([_compute(constant_envelope_signals(m), run["path"], o, l, a)[0, 1] for m in CONST_M])


def proportional_channels(x, run):
    """Channels [a, 2a, a + 0.3 b, b], a, b = successive default_rng(0).standard_normal(T)
    draws (T = run["T"]); exactly proportional pair (0,1).  Dataset x is NOT used."""
    rng = np.random.default_rng(0)
    T = int(run["T"])
    a = rng.standard_normal(T)
    b = rng.standard_normal(T)
    o, l, ab = PARAMS[run["identifier"]]
    return _compute(np.stack([a, 2 * a, a + 0.3 * b, b]), run["path"], o, l, ab)


def absolute_nonorth(x, run):
    """PowerEnvelopeCorrelation(orth=False, log=<log of run['identifier']>, absolute=True): a parameter
    combination with no config identifier; tests whether `absolute` acts without orth."""
    _, log, _ = PARAMS[run["identifier"]]
    return _compute(x, run["path"], False, log, True)


def padding_T1031(x, run):
    """pyspi on x[:, :1031] (next_fast_len(1031) = 1080, so MNE pads the Hilbert FFT)."""
    o, l, a = PARAMS[run["identifier"]]
    return _compute(np.asarray(x)[:, :1031], run["path"], o, l, a)


def zero_channel(x, run):
    """All six identifiers on x with channel 3 set to exactly 0 (kernel path: the
    pipeline would reject a constant channel); returns (6, P, P) in ORDER."""
    x = np.array(x, dtype=np.float64, copy=True)
    x[3] = 0.0
    out = []
    for ident in ORDER:
        o, l, a = PARAMS[ident]
        with np.errstate(all="ignore"):
            out.append(_compute(x, run["path"], o, l, a))
    return np.stack(out)
