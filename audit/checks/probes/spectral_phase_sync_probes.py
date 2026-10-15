"""Probes for the spectral_phase_sync family (PROJECT venv; called by pyspi_outputs.py).

Each function has the signature ``fn(x, run)`` where ``x`` is the raw dataset
array (P, T) after optional process selection and ``run`` is the manifest entry.
They expose pyspi's own intermediates (the per-bin connectivity array that the
band reducer at pyspi/statistics/spectral.py:85-102 summarises, and the
frequency grid it masks), or compute pyspi SPIs on inputs the plain SPI path
cannot build (scaled raw input, fresh null seeds, EEG band edges).

The SPI object is always the pyspi class itself, built with its default
constructor (fs=1, fmin=0, fmax=0.5, statistic=mean) unless stated; the per-bin
array is what ``NonparametricSpectralMultivariate._get_cache`` returns
(spectral.py:64-83), i.e. ``getattr(Connectivity, measure)()`` of
spectral_connectivity 1.1.2, shape (1, n_bins, P, P) -> we return [0].
"""
import numpy as np

UNIT_CLASS = {
    "spectral.plv": "PhaseLockingValue",
    "spectral.pli": "PhaseLagIndex",
    "spectral.wpli": "WeightedPhaseLagIndex",
    "spectral.dspli": "DebiasedSquaredPhaseLagIndex",
    "spectral.dswpli": "DebiasedSquaredWeightedPhaseLagIndex",
    "spectral.ppc": "PairwisePhaseConsistency",
}

# EEG band edges: exact k/2048 cycles/sample vs the .3g-rounded values printed in
# the identifiers (audit/units.yaml eeg_identifiers of spectral.plv).
EEG_K = [(1, 4), (1, 70), (1, 250), (4, 8), (8, 12), (12, 30), (30, 70), (70, 150), (70, 250), (150, 250)]
EEG_ROUNDED = [(0.000488, 0.00195), (0.000488, 0.0342), (0.000488, 0.122), (0.00195, 0.00391),
               (0.00391, 0.00586), (0.00586, 0.0146), (0.0146, 0.0342), (0.0342, 0.0732),
               (0.0342, 0.122), (0.0732, 0.122)]


def _spi(unit_id, **kw):
    from pyspi.statistics import spectral
    return getattr(spectral, UNIT_CLASS[unit_id])(**kw)


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


def _perbin_raw(x, run):
    spi = _spi(run["unit_id"])
    res, freq = spi._get_cache(_data(x, run.get("path", "kernel")))
    return np.array(res[0], dtype=np.float64, copy=True), np.asarray(freq, dtype=np.float64)


def perbin(x, run):
    """pyspi's per-bin array (n_bins, P, P) with the diagonal set to NaN in every bin."""
    a, _ = _perbin_raw(x, run)
    i = np.arange(a.shape[1])
    a[:, i, i] = np.nan
    return a


def perbin_abs(x, run):
    """|per-bin array| (canonical magnitude form of the per-bin estimator), diagonal NaN."""
    return np.abs(perbin(x, run))


def freq_grid(x, run):
    """The frequency grid the band mask is applied to (Connectivity.frequencies)."""
    return _perbin_raw(x, run)[1]


def _interior(nb, T):
    """Bins at least 2W = 6/T (in bins: 2 * NW * n_fft / T) away from DC and Nyquist."""
    m = np.arange(nb)
    margin = 6
    return (m >= margin) & (m <= nb - 1 - margin)


def null_summary(x, run):
    """Scalar: mean of pyspi's per-bin value over the upper-triangle pairs and the
    interior bins (excluding the 6 bins nearest DC and Nyquist, where the taper
    coefficients of real white noise are not circular)."""
    a, f = _perbin_raw(x, run)
    P = a.shape[1]
    iu = np.triu_indices(P, 1)
    keep = _interior(a.shape[0], x.shape[1])
    return np.array(np.mean(a[keep][:, iu[0], iu[1]]))


def saturation(x, run):
    """Fraction of off-diagonal entries equal to 1.0 and the mean off-diagonal value of the
    full-band MAX identifier over 30 fresh iid N(0,1) (5, T) draws (seeds 10000..10029,
    numpy default_rng), pipeline path (Data(x) z-score).  x is ignored except for T.
    Returns array (30, 2)."""
    T = x.shape[1]
    out = np.empty((30, 2))
    for s in range(30):
        z = np.random.default_rng(10000 + s).standard_normal((5, T))
        spi = _spi(run["unit_id"], statistic="max")
        A = np.asarray(spi.multivariate(_data(z, "pipeline")), dtype=np.float64)
        off = ~np.eye(5, dtype=bool)
        out[s] = (np.mean(A[off] == 1.0), np.mean(A[off]))
    return out


SCALES = (1.0, 1e-6, 1e-8, 1e-10)


def scale_sweep(x, run):
    """Full-band mean SPI on Data(x * s, normalise=False) for s in SCALES -> (4, P, P)."""
    from pyspi.data import Data
    outs = []
    for s in SCALES:
        spi = _spi(run["unit_id"])
        d = Data(np.array(x, dtype=np.float64) * s, dim_order="ps", normalise=False, detrend=False)
        A = np.array(spi.multivariate(d), dtype=np.float64)
        np.fill_diagonal(A, np.nan)
        outs.append(A)
    return np.stack(outs)


def eeg_bands(x, run):
    """pyspi band-mean value at pair (0, 1) for the 10 EEG bands, exact k/2048 edges (row 0)
    and the .3g-rounded edges printed in the identifiers (row 1), on the first L samples
    of x for L in (512, 1024) -> (2 lengths, 2 edge variants, 10 bands); pipeline path."""
    out = np.full((2, 2, len(EEG_K)), np.nan)
    import warnings
    for li, L in enumerate((512, 1024)):
        xs = np.array(x[:, :L], dtype=np.float64)
        for vi in range(2):
            for b in range(len(EEG_K)):
                fmin, fmax = ((EEG_K[b][0] / 2048, EEG_K[b][1] / 2048) if vi == 0 else EEG_ROUNDED[b])
                spi = _spi(run["unit_id"], fmin=fmin, fmax=fmax)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    A = spi.multivariate(_data(xs, "pipeline"))
                out[li, vi, b] = A[0, 1]
    return out
