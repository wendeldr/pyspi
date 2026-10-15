"""Probes for the spectral_csm family, unit spectral.band_selection (PROJECT venv).

Instrumentation of pyspi's band selection without re-implementing it:

* ``band_masks`` seeds the per-Data spectral cache (spectral.py:65-67 /
  :108-111) with an array whose value at frequency bin k is k, and swaps the SPI
  instance's ``_statfn`` for a recorder.  ``multivariate()`` / ``bivariate()``
  then run their own ``freq_id`` lines (spectral.py:88 / :132) on the real
  ``Connectivity.frequencies`` grid, and the recorder sees exactly the bins they
  selected.
* ``psi_gd_spy`` wraps ``Connectivity.phase_slope_index`` / ``group_delay`` and
  the module function ``_inner_combination`` to record the forwarded kwargs and
  the number of bins PSI sums over.
* ``eeg_outcomes`` runs real SPI objects for the retained EEG bands (exact float
  edges from pyspi/workflow/retained185.yaml) and records the outcome class.
  An exception is caught and encoded (the Calculator turns it into an all-NaN
  table, calculator.py:301-303); codes: 0 ok, 1 IndexError, 2 ValueError,
  3 other exception.
* ``fft_counts`` counts Multitaper.fft calls per SPI on one Data object.
* ``bv_aggregation`` returns the library's per-bin directed coherence / DTF of
  every ordered 2-channel pair next to pyspi's band summaries.

Preprocessing and the ``n<int>`` truncation token follow spectral_csm_lib.prepare.
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np

from pyspi._spectral_compat import spectral_connectivity as sc

_spec = importlib.util.spec_from_file_location("spectral_csm_lib", Path(__file__).with_name("spectral_csm_lib.py"))
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

CONFIG_BANDS = [(0, 0.5), (0, 0.25), (0.25, 0.5)]
# retained185.yaml edges (exact floats, cycles/sample), in the order of that file
EEG_BANDS = [(0.00048828125, 0.001953125), (0.001953125, 0.00390625), (0.00390625, 0.005859375),
             (0.005859375, 0.0146484375), (0.0146484375, 0.0341796875), (0.0341796875, 0.0732421875),
             (0.0732421875, 0.1220703125), (0.00048828125, 0.0341796875), (0.0341796875, 0.1220703125),
             (0.00048828125, 0.1220703125)]
ALL_BANDS = CONFIG_BANDS + EEG_BANDS


def _data(z):
    import contextlib
    import io
    from pyspi.data import Data
    # z is already preprocessed by lib.prepare; keep it untouched (Data's progress prints are muted)
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(z.copy(), dim_order="ps", normalise=False, detrend=False)


def _code(exc):
    if exc is None:
        return 0
    if isinstance(exc, IndexError):
        return 1
    if isinstance(exc, ValueError):
        return 2
    return 3


def band_masks(x, run):
    """(2, n_bands, n_freq) 0/1 masks of the bins pyspi's reducers select:
    [0] = multivariate path (CoherenceMagnitude), [1] = bivariate path (DirectedCoherence)."""
    from pyspi.statistics.spectral import CoherenceMagnitude, DirectedCoherence
    z = lib.prepare(x, run)
    _, c = lib.connectivity(z)
    freq = np.asarray(c.frequencies)
    nf, P = len(freq), z.shape[0]
    ramp = np.arange(nf, dtype=np.float64)
    out = np.zeros((2, len(ALL_BANDS), nf))
    for b, (lo, hi) in enumerate(ALL_BANDS):
        for p, cls in enumerate((CoherenceMagnitude, DirectedCoherence)):
            spi = cls(fmin=lo, fmax=hi)
            seen = []

            def rec(a, axis=0, _seen=seen):
                _seen.append(np.array(a, dtype=np.float64))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return np.nanmean(a, axis=axis) if np.size(a) else np.full(np.shape(a)[1:], np.nan)

            spi._statfn = rec
            data = _data(z)
            if p == 0:
                data.spectral_mv = {"freq": freq, spi.key: np.broadcast_to(ramp[None, :, None, None], (1, nf, P, P)).copy()}
                spi.multivariate(data)
                sel = seen[0][:, 0, 1] if seen[0].ndim == 3 else seen[0]
            else:
                data.spectral_bv = {"freq": freq, (spi.measure, 0, 1): np.broadcast_to(ramp[None, :, None, None], (1, nf, 2, 2)).copy()}
                spi.bivariate(data, i=0, j=1)
                sel = seen[0]
            idx = np.asarray(sel, dtype=int)
            out[p, b, idx] = 1.0
    return out


def psi_gd_spy(x, run):
    """Per band of ALL_BANDS, 13 numbers:
    PSI: [n library calls, n kwargs of first call, fmin fwd, fmax fwd, resolution fwd,
          n bins entering _inner_combination (-1 if never reached), outcome code]
    GD:  [n library calls, n kwargs of first call, fmin fwd, fmax fwd, resolution fwd, outcome code]."""
    import spectral_connectivity.connectivity as scc
    from pyspi.statistics.spectral import PhaseSlopeIndex, GroupDelay
    z = lib.prepare(x, run)
    rows = []
    orig_psi, orig_gd, orig_inner = scc.Connectivity.phase_slope_index, scc.Connectivity.group_delay, scc._inner_combination
    try:
        for lo, hi in ALL_BANDS:
            calls, inner = {"psi": [], "gd": []}, []

            def psi_wrap(self, *a, **k):
                calls["psi"].append(dict(k))
                return orig_psi(self, *a, **k)

            def gd_wrap(self, *a, **k):
                calls["gd"].append(dict(k))
                return orig_gd(self, *a, **k)

            def inner_wrap(data, axis=-3):
                inner.append(data.shape[axis])
                return orig_inner(data, axis)

            scc.Connectivity.phase_slope_index, scc.Connectivity.group_delay = psi_wrap, gd_wrap
            scc._inner_combination = inner_wrap
            row = []
            for name, cls, kw in (("psi", PhaseSlopeIndex, {}), ("gd", GroupDelay, {"statistic": "delay"})):
                err = None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cls(fmin=lo, fmax=hi, **kw).multivariate(_data(z))
                except Exception as exc:  # recorded as outcome code
                    err = exc
                cl = calls[name]
                fw = cl[1] if len(cl) > 1 else {}
                foi = fw.get("frequencies_of_interest", [np.nan, np.nan])
                row += [len(cl), len(cl[0]) if cl else -1, foi[0], foi[1], fw.get("frequency_resolution", np.nan)]
                if name == "psi":
                    row.append(inner[-1] if inner else -1)
                row.append(_code(err))
            rows.append(row)
    finally:
        scc.Connectivity.phase_slope_index, scc.Connectivity.group_delay = orig_psi, orig_gd
        scc._inner_combination = orig_inner
    return np.array(rows, dtype=np.float64)


def eeg_outcomes(x, run):
    """(n_EEG_bands, 5, 3): for cohmag mean, cohmag max, psi, gd delay, dcoh mean
    -> [outcome code, n NaN off-diagonal, n finite off-diagonal]."""
    from pyspi.statistics.spectral import CoherenceMagnitude, PhaseSlopeIndex, GroupDelay, DirectedCoherence
    z = lib.prepare(x, run)
    P = z.shape[0]
    off = ~np.eye(P, dtype=bool)
    out = np.zeros((len(EEG_BANDS), 5, 3))
    for b, (lo, hi) in enumerate(EEG_BANDS):
        spis = [CoherenceMagnitude(fmin=lo, fmax=hi, statistic="mean"), CoherenceMagnitude(fmin=lo, fmax=hi, statistic="max"),
                PhaseSlopeIndex(fmin=lo, fmax=hi), GroupDelay(fmin=lo, fmax=hi, statistic="delay"),
                DirectedCoherence(fmin=lo, fmax=hi, statistic="mean")]
        for s, spi in enumerate(spis):
            err, A = None, np.full((P, P), np.nan)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    A = np.asarray(spi.multivariate(_data(z)), dtype=np.float64)
            except Exception as exc:
                err = exc
            out[b, s] = [_code(err), np.isnan(A[off]).sum(), np.isfinite(A[off]).sum()]
    return out


def fft_counts(x, run):
    """Multitaper.fft calls triggered by each SPI, all on ONE Data object, in this order:
    cohmag mean full, cohmag mean low, cohmag max full, icoh mean full, psi full, psi low,
    gd full, dcoh mean full (bivariate path over all ordered pairs)."""
    from pyspi.statistics.spectral import (CoherenceMagnitude, ImaginaryCoherence, PhaseSlopeIndex,
                                           GroupDelay, DirectedCoherence)
    z = lib.prepare(x, run)
    data = _data(z)
    count = [0]
    orig = sc.Multitaper.fft

    def wrapped(self, *a, **k):
        count[0] += 1
        return orig(self, *a, **k)

    sc.Multitaper.fft = wrapped
    seq = [CoherenceMagnitude(), CoherenceMagnitude(fmin=0, fmax=0.25), CoherenceMagnitude(statistic="max"),
           ImaginaryCoherence(), PhaseSlopeIndex(fmin=0, fmax=0.5), PhaseSlopeIndex(fmin=0, fmax=0.25),
           GroupDelay(fmin=0, fmax=0.5, statistic="delay"), DirectedCoherence()]
    out = []
    try:
        for spi in seq:
            before = count[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spi.multivariate(data)
            out.append(count[0] - before)
    finally:
        sc.Multitaper.fft = orig
    return np.array(out, dtype=np.float64)


def cohmag_means3(x, run):
    """pyspi CoherenceMagnitude mean for [fmin-0_fmax-0-25, fmin-0-25_fmax-0-5, fmin-0_fmax-0-5]
    on the run's path; shape (3, P, P)."""
    from pyspi.statistics.spectral import CoherenceMagnitude
    z = lib.prepare(x, run)
    out = []
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0, 0.5)):
        out.append(CoherenceMagnitude(fmin=lo, fmax=hi, statistic="mean").multivariate(_data(z)))
    return np.stack(out)


def bv_aggregation(x, run):
    """(2 n_freq + 12, P, P): library per-bin directed_coherence and directed_transfer_function of
    the 2-channel [i, j] Connectivity, entry [0, 1], for every ordered pair (i, j); then pyspi
    DirectedCoherence and DirectedTransferFunction .multivariate for the six config identifiers
    (mean full, mean low, mean high, max full, max low, max high)."""
    from pyspi.statistics.spectral import DirectedCoherence, DirectedTransferFunction
    z = lib.prepare(x, run)
    P = z.shape[0]
    _, c = lib.connectivity(z[:2])
    nf = len(c.frequencies)
    per = np.full((2, nf, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                _, cij = lib.connectivity(z[[i, j]])
                per[0, :, i, j] = cij.directed_coherence()[0, :, 0, 1]
                per[1, :, i, j] = cij.directed_transfer_function()[0, :, 0, 1]
    summ = []
    for cls in (DirectedCoherence, DirectedTransferFunction):
        for st in ("mean", "max"):
            for lo, hi in CONFIG_BANDS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    summ.append(np.asarray(cls(fmin=lo, fmax=hi, statistic=st).multivariate(_data(z)), dtype=np.float64))
    return np.concatenate([per[0], per[1], np.stack(summ)], axis=0)
