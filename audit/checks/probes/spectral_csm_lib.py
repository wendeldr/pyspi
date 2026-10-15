"""Probes for the spectral_csm family, unit spectral.multitaper_csm (PROJECT venv).

Each function has the probe signature ``fn(x, run)``: ``x`` is the raw dataset
array (P, T) after optional process selection, ``run`` the manifest entry.
They build exactly the objects pyspi builds in
``pyspi/statistics/spectral.py:64-83`` (``sc.Multitaper(z, sampling_frequency=1)``
then ``sc.Connectivity.from_multitaper(m)``, with ``sc`` imported through
``pyspi._spectral_compat``) and return intermediates of spectral_connectivity
1.1.2 that the plain SPI path does not expose.

Conventions shared with ``refs/spectral_csm.py``:

* Preprocessing follows ``run["path"]``: ``kernel`` = the raw array as given;
  ``pipeline`` = ``Data(x).to_numpy(squeeze=True)`` (z-score, ddof=1);
  ``pipeline-detrend`` = ``Data(x, detrend=True)`` (linear detrend, then z-score).
* A token ``n<int>`` in the last ``|``-field of the run_id truncates the series to
  its first ``<int>`` samples (used for odd / non-5-smooth lengths and T=512).
* Complex arrays are returned as ``np.stack([real, imag])`` because compare.py
  reads float64.
"""
import numpy as np

from pyspi._spectral_compat import spectral_connectivity as sc

# band edges used by band_average_csm; refs/spectral_csm.py uses the same list
CSM_BANDS = [(0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.49)]


def _tag_tokens(run):
    return run["run_id"].split("|")[-1].split("-")


def truncation(run):
    for tok in _tag_tokens(run):
        if tok.startswith("n") and tok[1:].isdigit():
            return int(tok[1:])
    return None


def prepare(x, run):
    """(P, T) float64 array as pyspi's Data would hand it to the spectral class."""
    x = np.asarray(x, dtype=np.float64)
    n = truncation(run)
    if n is not None:
        x = x[:, :n]
    path = run.get("path", "kernel")
    if path == "kernel":
        return x.copy()
    from pyspi.data import Data
    if path == "pipeline":
        d = Data(x.copy(), dim_order="ps")
    elif path == "pipeline-detrend":
        d = Data(x.copy(), dim_order="ps", normalise=True, detrend=True)
    else:
        raise ValueError(path)
    return np.asarray(d.to_numpy(squeeze=True), dtype=np.float64)


def connectivity(z):
    """The (Multitaper, Connectivity) pair pyspi builds (spectral.py:68-69)."""
    m = sc.Multitaper(np.transpose(z), sampling_frequency=1)
    return m, sc.Connectivity.from_multitaper(m)


def _ri(a):
    return np.stack([np.real(a), np.imag(a)])


def csm_twosided(x, run):
    """Library two-sided CSM, window 0: shape (2, n_fft, P, P) = [Re, Im]."""
    _, c = connectivity(prepare(x, run))
    return _ri(c._expectation_cross_spectral_matrix()[0])


def coherency_nonneg(x, run):
    """Library complex coherency on the non-negative grid (NaN diagonal):
    (2, (n_fft+1)//2, P, P) = [Re, Im] of Connectivity.coherency()[0]."""
    _, c = connectivity(prepare(x, run))
    return _ri(c.coherency()[0])


def frequencies(x, run):
    """Connectivity.frequencies (the grid every pyspi spectral SPI reduces over)."""
    _, c = connectivity(prepare(x, run))
    return np.asarray(c.frequencies, dtype=np.float64)


def grid_summary(x, run):
    """[n_fft, n_kept_bins, n_tapers, n_observations] from the library objects."""
    m, c = connectivity(prepare(x, run))
    return np.array([m.n_fft_samples, len(c.frequencies), m.tapers.shape[1], c.n_observations], dtype=np.float64)


def tapers(x, run):
    """Library tapers as used (Multitaper.tapers, sqrt(fs)=1 scaling), shape (K, T)."""
    m, _ = connectivity(prepare(x, run))
    return np.asarray(m.tapers, dtype=np.float64).T


def taper_eigenvalues(x, run):
    """Concentration eigenvalues from spectral_connectivity.transforms.dpss_windows(T, 3, 5)
    with is_low_bias=False (all 5 reported), the values its low-bias filter (> 0.9) sees."""
    import spectral_connectivity.transforms as tr
    T = prepare(x, run).shape[1]
    return np.asarray(tr.dpss_windows(T, 3, 5, is_low_bias=False)[1], dtype=np.float64)


def identities(x, run):
    """[max|S - S^H| over all bins, max|Im S_ij(0)|, mean over the n_fft two-sided bins of S_ii (i=0..P-1)]."""
    _, c = connectivity(prepare(x, run))
    S = c._expectation_cross_spectral_matrix()[0]
    herm = np.abs(S - np.conj(np.swapaxes(S, -1, -2))).max()
    im0 = np.abs(np.imag(S[0])).max()
    pw = np.real(np.diagonal(S, axis1=-2, axis2=-1)).mean(axis=0)
    return np.concatenate([[herm, im0], pw])


def _band_quantities(S, f):
    """Per band of CSM_BANDS (lo <= f < hi): [S_ii for all i] + [Re S_ij, Im S_ij for i<j]."""
    P = S.shape[-1]
    iu = [(i, j) for i in range(P) for j in range(i + 1, P)]
    out = []
    for lo, hi in CSM_BANDS:
        m = (f >= lo) & (f < hi)
        Sb = S[m].mean(axis=0)
        row = [Sb[i, i].real for i in range(P)]
        for i, j in iu:
            row += [Sb[i, j].real, Sb[i, j].imag]
        out.append(row)
    return np.array(out)


def band_average_csm(x, run):
    """Band-averaged library CSM on the non-negative grid, shape (n_bands, n_quantities)."""
    _, c = connectivity(prepare(x, run))
    S = c._expectation_cross_spectral_matrix()[0][: len(c.frequencies)]
    return _band_quantities(S, np.asarray(c.frequencies))


def phase_slope(x, run):
    """D[i, j] = slope / (2 pi) of numpy.unwrap(angle S_ij(f)) fitted (numpy.polyfit, deg 1)
    over 0 < f < 0.1 of the library non-negative grid; NaN diagonal."""
    _, c = connectivity(prepare(x, run))
    f = np.asarray(c.frequencies)
    S = c._expectation_cross_spectral_matrix()[0][: len(f)]
    m = (f > 0) & (f < 0.1)
    P = S.shape[-1]
    D = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                D[i, j] = np.polyfit(f[m], np.unwrap(np.angle(S[m, i, j])), 1)[0] / (2 * np.pi)
    return D


def lowfreq_fraction(x, run):
    """Per channel: sum of library power over f < 6/T divided by the sum over the non-negative grid."""
    z = prepare(x, run)
    _, c = connectivity(z)
    f = np.asarray(c.frequencies)
    S = c._expectation_cross_spectral_matrix()[0][: len(f)]
    p = np.real(np.diagonal(S, axis1=-2, axis2=-1))
    return p[f < 6.0 / z.shape[1]].sum(axis=0) / p.sum(axis=0)
