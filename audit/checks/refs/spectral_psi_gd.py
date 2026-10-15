#!/usr/bin/env python3
"""References for the spectral_psi_gd family (spectral.psi, spectral.gd).

Runs in the COMPARISON venv (no pyspi, no spectral_connectivity).  For every run
of ``manifest/spectral_psi_gd.json`` it writes ``out/ref/<run_id>.npz`` in the
contract of ``compare.py``.  The run's tag (last ``|`` field of the run_id)
selects the check:

PSI (``spectral.psi``)
  rep          independent literal replica of what pyspi computes: multitaper
               coherency (``_kernels_spectral``: scipy DPSS NW=3 K=5, constant
               detrend, numpy FFT, S_ij = X_i conj X_j), STRICT band
               fmin < f < fmax, decimation step ceil(((fmax-fmin)/50) n_fft),
               Im sum over ALL unordered pairs of kept bins (``psi_pyspi``).
               criterion tolerance rtol 1e-10, atol 1e-11 (|A| up to ~800).
  nolte        Nolte et al. 2008 eq. 3 (arXiv:0712.2352v1 p.2): adjacent-pair
               raw PSI from the same independent coherency over every in-band
               bin (delta_f = 1/n_fft).  criterion direction: expected_sign =
               sign(Nolte); extra keys: per-pair pyspi/Nolte ratio, Spearman
               rank correlation over pairs, Nolte on the decimated bins, and the
               jackknife-normalised Psi (eq. 6, leave-one-taper-out).
  mne          mne_connectivity.phase_slope_index (mode multitaper, NW=3,
               adjacent pairs, Nolte eq. 3) as external package; criterion
               direction against its sign; extra keys: MNE vs our psi_adjacent.
  dir          analytic direction on delayed_noise: expected sign(d_j - d_i).
  delaycurve   analytic pure-delay model |C|^2 * sum_{a<b} sin(2 pi tau (f_b - f_a))
               over pyspi's kept bins; tolerance rtol 0.1.
  m5           REVIEW.md M5 four-unit-coherency example: the probe runs the
               library's own ``_inner_combination``; the reference checks that
               it equals the all-pairs formula and differs in sign from the
               adjacent formula (qualitative).
  m5data       a dataset case found by search where pyspi (all pairs) and Nolte
               (adjacent, decimated bins = the same bins) differ in sign
               (qualitative pass_flag = at least one sign disagreement).
  embed        pair alone vs pair embedded with 5 iid channels (probe returns
               the pair's 2x2 block of the embedded run); reference value is
               pyspi's own pair-alone output (self-consistency, exact).
  perm         channel permutation: probe returns A(X[perm])[inv, inv]; the
               reference value is pyspi's own unpermuted output (exact).
  eeg          EEG band probe (pyspi class constructed with the EEG fmin/fmax):
               < 2 kept bins -> qualitative pass iff the probe recorded an
               IndexError and returned all-NaN; otherwise tolerance vs replica.

GD (``spectral.gd``)
  rep          independent replica of the library pipeline (``gd_pyspi``):
               Fisher z with coherence_bias(n_tapers=5) = 1/8, one-sided normal
               p, Benjamini-Hochberg 0.05 over the flattened (frequency x pair)
               array, largest contiguous significant run (first on ties), every
               s-th bin of it, >= 3 points, numpy.unwrap of the phase over the
               whole in-band grid, OLS slope, delay = slope / (2 pi), A[j,i] =
               -A[i,j]; NaN pattern must match.  tolerance rtol 1e-9 atol 1e-10.
  dir          delayed_noise: expected sign(d_j - d_i) (direction).
  delay        delayed_noise: |A[i,j] - (d_j - d_i)| <= 0.2 (tolerance, atol 0.2).
  gotman       delayed_noise: plain OLS of the unwrapped phase over every in-band
               bin (``_kernels_spectral.group_delay_fit``, no FDR/cluster/
               decimation), tolerance atol 0.1.
  null         iid_gaussian: qualitative pass iff every off-diagonal entry is NaN
               (no significant coherence => no delay); extra keys count the
               finite entries and the replica's count with step = 2W n_fft = 6.
  embed, perm  as for PSI (expected to FAIL for embed: BH pooled across pairs).
  eeg          EEG band probe: 0 in-band bins -> qualitative pass iff the probe
               recorded the ValueError; 1-2 bins -> all-NaN expected; else
               tolerance vs replica.
  popgd        var2_bivariate / var2_chain: population group delay from the true
               spectrum (make_data S_true) via _kernels_spectral.group_delay_fit;
               criterion direction (sign of the population delay).
  shares, spy  (both units) probe-side bookkeeping judged qualitatively: the
               returned matrix is a view of data.spectral_mv; the library method is
               called twice, the second time with the configured band.
  excl25       narrowband_f0-0.25: f = 0.25 is excluded from both half bands
               (strict endpoints); qualitative pass iff the prediction of the
               trace (all-NaN at T >= 1024) holds and 0.25 is not a kept bin.

Paths: kernel = raw x; pipeline = scipy.stats.zscore(ddof=1); pipeline-detrend
= scipy.signal.detrend (linear) then zscore(ddof=1) (pyspi/data.py:184-195).
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.signal  # noqa: E402
import scipy.stats  # noqa: E402

import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

FAMILY = "spectral_psi_gd"
CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCHANGE = CHECKS / "out" / "exchange" / FAMILY
SELF = "refs/spectral_psi_gd.py"


# ------------------------------------------------------------------ preprocessing
def preprocess(x, path):
    x = np.asarray(x, dtype=np.float64)
    if path == "kernel":
        return x.copy()
    if path == "pipeline-detrend":
        x = scipy.signal.detrend(x, axis=1)
    return scipy.stats.zscore(x, axis=1, ddof=1)


# ------------------------------------------------------------ coherency (shared)
def mt_coherency(x):
    """Independent multitaper coherency on the library's non-negative grid.
    Returns C (n_f, P, P) with NaN diagonal, freqs (n_f,), n_fft, n_tapers."""
    coeffs, freqs2 = ks.multitaper_fft(x, NW=3.0)          # (K, n_fft, P)
    S = ks.cross_spectral_matrix(coeffs, nonnegative=True)
    C = ks.coherency(S, nan_diagonal=True)
    return C, ks.nonneg(freqs2), freqs2.size, coeffs.shape[0], coeffs


def kept_bins(freqs, fmin, fmax):
    """Indices (into the band) and frequencies kept by the library's PSI
    decimation: strict band, every s-th bin, s = ceil(((fmax-fmin)/50) / df)."""
    m = ks.band_mask(freqs, fmin, fmax, inclusive=False)
    fb = freqs[m]
    step = ks.library_frequency_step(freqs, (fmax - fmin) / 50.0)
    return np.nonzero(m)[0][::step], fb[::step], step, int(m.sum())


# --------------------------------------------------------------------------- PSI
def psi_pyspi(C, freqs, fmin, fmax):
    """pyspi/SC PSI: all unordered pairs of the decimated strict-band bins."""
    step = ks.library_frequency_step(freqs, (fmax - fmin) / 50.0)
    A = ks.psi_all_pairs(C, freqs, fmin, fmax, step=step)
    np.fill_diagonal(A, np.nan)
    return A


def psi_nolte(C, freqs, fmin, fmax, step=1):
    """Nolte 2008 eq. 3, adjacent pairs, strict band, optional decimation."""
    A = ks.psi_adjacent(C, freqs, fmin, fmax, inclusive=False, step=step)
    np.fill_diagonal(A, np.nan)
    return A


def psi_nolte_jackknife(coeffs, freqs2, fmin, fmax):
    """Nolte 2008 eq. 6: Psi = Psi_tilde / std(Psi_tilde), std by the jackknife;
    the observations left out are the tapers (single-segment data)."""
    K = coeffs.shape[0]
    full = _psi_from_coeffs(coeffs, freqs2, fmin, fmax)
    loo = np.stack([_psi_from_coeffs(np.delete(coeffs, k, axis=0), freqs2, fmin, fmax) for k in range(K)])
    sd = np.sqrt((K - 1) / K * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = full / sd
    np.fill_diagonal(out, np.nan)
    return out


def _psi_from_coeffs(coeffs, freqs2, fmin, fmax):
    S = ks.cross_spectral_matrix(coeffs, nonnegative=True)
    C = ks.coherency(S, nan_diagonal=True)
    return psi_nolte(C, ks.nonneg(freqs2), fmin, fmax)


def all_pairs_scalar(phases, mags=None):
    """Im sum_{a<b} conj(c_a) c_b for one coherency vector."""
    c = np.exp(1j * np.asarray(phases)) * (1.0 if mags is None else np.asarray(mags))
    return float(np.imag(sum(np.conj(c[a]) * c[b] for a, b in itertools.combinations(range(c.size), 2))))


def adjacent_scalar(phases, mags=None):
    c = np.exp(1j * np.asarray(phases)) * (1.0 if mags is None else np.asarray(mags))
    return float(np.imag((np.conj(c[:-1]) * c[1:]).sum()))


def psi_closed_band(C, freqs, fmin, fmax):
    """Counterfactual: the same all-pairs PSI on the CLOSED band fmin <= f <= fmax
    (documented intervals); used only to show the strict rule is what pyspi runs."""
    step = ks.library_frequency_step(freqs, (fmax - fmin) / 50.0)
    Cb = np.asarray(C)[ks.band_mask(freqs, fmin, fmax, inclusive=True)][::step]
    prefix = np.cumsum(Cb, axis=0) - Cb
    A = np.imag((np.conj(prefix) * Cb).sum(axis=0))
    np.fill_diagonal(A, np.nan)
    return A


# ---------------------------------------------------------------------------- GD
def _largest_run(sig):
    """Boolean mask of the largest contiguous True run (first on ties)."""
    best_start, best_len, i, n = -1, 0, 0, sig.size
    while i < n:
        if sig[i]:
            j = i
            while j < n and sig[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    out = np.zeros(n, dtype=bool)
    if best_len:
        out[best_start:best_start + best_len] = True
    return out


def benjamini_hochberg(p, alpha=0.05):
    """BH step-up on the flattened array (Benjamini & Hochberg 1995): reject every
    p <= p_(k*) with k* the largest k such that p_(k) <= k alpha / N."""
    flat = np.sort(p.ravel())
    line = alpha * np.arange(1, flat.size + 1) / flat.size
    ok = np.nonzero(flat <= line)[0]
    thr = flat[ok.max()] if ok.size else -1.0
    return p <= thr


def gd_pyspi(C, freqs, fmin, fmax, n_tapers, alpha=0.05, min_group=3, step=None,
             unwrap="grid", return_detail=False, inclusive=False):
    """Replica of SC 1.1.2 group_delay as called by pyspi (see module doc).
    unwrap='grid' unwraps over the whole in-band grid (library); 'run' unwraps the
    kept points' contiguous run only (for the unwrap-before-masking question)."""
    m = ks.band_mask(freqs, fmin, fmax, inclusive=inclusive)
    fb = freqs[m]
    if fb.size == 0:
        raise ValueError("zero in-band bins (library: mstats.linregress np.amax of empty x)")
    if step is None:
        step = ks.library_frequency_step(freqs, (fmax - fmin) / 50.0)
    P = C.shape[-1]
    pairs = list(itertools.combinations(range(P), 2))
    Cp = np.stack([C[m][:, i, j] for i, j in pairs], axis=1)            # (nf, npairs)
    bias = 1.0 / (2 * n_tapers - 2)
    mag = np.abs(Cp)
    mag[mag >= 1] = 1 - np.finfo(float).eps
    z = (np.arctanh(mag) - bias) / np.sqrt(bias)
    p = 1 - scipy.stats.norm.cdf(z)
    sig = benjamini_hochberg(p, alpha)
    phase_grid = np.unwrap(np.angle(Cp), axis=0)
    A = np.full((P, P), np.nan)
    detail = []
    for k, (i, j) in enumerate(pairs):
        run = _largest_run(sig[:, k])
        idx = np.nonzero(run)[0][::step]
        if idx.size < min_group:
            detail.append(dict(pair=(i, j), n_kept=int(idx.size), run_len=int(run.sum())))
            continue
        if unwrap == "grid":
            y = phase_grid[idx, k]
        else:
            r = np.nonzero(run)[0]
            y_run = np.unwrap(np.angle(Cp[r, k]))
            y = y_run[np.searchsorted(r, idx)]
        xf = fb[idx]
        xc = xf - xf.mean()
        slope = float(xc @ (y - y.mean()) / (xc @ xc))
        A[i, j] = slope / (2 * np.pi)
        A[j, i] = -slope / (2 * np.pi)
        detail.append(dict(pair=(i, j), n_kept=int(idx.size), run_len=int(run.sum())))
    return (A, detail) if return_detail else A


# -------------------------------------------------------------------- utilities
def triu(A):
    i, j = np.triu_indices(A.shape[0], 1)
    return A[i, j]


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(scipy.stats.spearmanr(a[m], b[m]).statistic)


def load_pyspi(run_id):
    p = OUT_PY / f"{run_id}.npz"
    with np.load(p) as z:
        return np.asarray(z["value"], dtype=np.float64)


def load_index_entry(run_id):
    p = OUT_PY / "index" / f"{FAMILY}.json"
    return json.loads(p.read_text()).get(run_id, {}) if p.is_file() else {}


def load_exchange(run_id):
    p = EXCHANGE / (run_id.replace("|", "__").replace("/", "_") + ".json")
    return json.loads(p.read_text()) if p.is_file() else {}


def save(run_id, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in kw.items():
        out[k] = np.asarray(v)
    np.savez(OUT_REF / f"{run_id}.npz", **out)


# ======================================================================= driver
BANDS = {"full": (0.0, 0.5), "low": (0.0, 0.25), "high": (0.25, 0.5)}
DELAYS = np.array([0, 1, 3, 5], dtype=float)           # delayed_noise truth (CONVENTIONS.md D6)
D6_NOISE = 0.1
M5_ADJ, M5_ALL = -1.3070153772, 1.1167942658           # docs/storage_comparison/REVIEW.md:176-177
REP_TOL = {"spectral.psi": (1e-10, 1e-11), "spectral.gd": (1e-9, 1e-10)}
EXACT = (1e-10, 1e-12)


def _probe_module():
    import importlib.util
    p = CHECKS / "probes" / "spectral_psi_gd_probe.py"
    spec = importlib.util.spec_from_file_location("spectral_psi_gd_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lead_matrix(d):
    """A[i, j] = d_j - d_i (> 0 when row i leads column j), NaN diagonal."""
    M = d[None, :] - d[:, None]
    M = M.astype(float)
    np.fill_diagonal(M, np.nan)
    return M


def estimate(unit, x, fmin, fmax, **kw):
    C, f, n_fft, K, coeffs = mt_coherency(x)
    if unit == "spectral.psi":
        return psi_pyspi(C, f, fmin, fmax)
    return gd_pyspi(C, f, fmin, fmax, K, **kw)


def rep_extras(unit, x_raw, path, fmin, fmax):
    """Normalisation/detrend effects measured with the reference itself."""
    out = {}
    try:
        a_pipe = estimate(unit, preprocess(x_raw, "pipeline"), fmin, fmax)
        a_raw = estimate(unit, preprocess(x_raw, "kernel"), fmin, fmax)
        a_det = estimate(unit, preprocess(x_raw, "pipeline-detrend"), fmin, fmax)
        with np.errstate(invalid="ignore"):
            out["normalise_effect_max_abs"] = float(np.nanmax(np.abs(a_pipe - a_raw))) if np.isfinite(a_pipe - a_raw).any() else 0.0
            out["detrend_effect_max_abs"] = float(np.nanmax(np.abs(a_det - a_pipe))) if np.isfinite(a_det - a_pipe).any() else 0.0
            out["detrend_nan_pattern_changes"] = int((np.isnan(a_det) != np.isnan(a_pipe)).sum())
    except ValueError:
        pass
    return out


def handle(run):
    rid, unit, ds, T, path = run["run_id"], run["unit_id"], run["dataset"], int(run["T"]), run["path"]
    tag = rid.split("|")[-1]
    kind, _, bkey = tag.partition(":")
    x_raw = np.asarray(make_data.load(ds, T)["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x_raw = x_raw[list(run["processes"])]
    x = preprocess(x_raw, path)
    fmin, fmax = BANDS.get(bkey, (None, None))
    is_psi = unit == "spectral.psi"
    fn = "psi_pyspi" if is_psi else "gd_pyspi"
    det = f"{SELF}:{fn} (+ _kernels_spectral multitaper_fft/cross_spectral_matrix/coherency" + (
        "/psi_all_pairs)" if is_psi else "; scipy.stats.norm, hand BH, numpy OLS)")

    if kind in ("rep", "base", "alone"):
        C, f, n_fft, K, coeffs = mt_coherency(x)
        idx, fk, step, nband = kept_bins(f, fmin, fmax)
        extra = dict(n_band_bins=nband, step=step, n_kept=int(fk.size), n_tapers=K)
        if is_psi:
            V = psi_pyspi(C, f, fmin, fmax)
        else:
            V = gd_pyspi(C, f, fmin, fmax, K)
            Vrun = gd_pyspi(C, f, fmin, fmax, K, unwrap="run")
            dd = np.abs(V - Vrun)
            extra["n_finite"] = int(np.isfinite(V).sum())
            extra["unwrap_grid_vs_run_max_abs"] = float(np.nanmax(dd)) if np.isfinite(dd).any() else 0.0
        S_py = load_pyspi(rid)
        Vc = psi_closed_band(C, f, fmin, fmax) if is_psi else gd_pyspi(C, f, fmin, fmax, K, inclusive=True)
        dc = np.abs(S_py - Vc)
        extra["closed_band_variant_max_abs_diff"] = float(np.nanmax(dc)) if np.isfinite(dc).any() else 0.0
        extra["closed_band_variant_nan_pattern_changes"] = int((np.isnan(S_py) != np.isnan(Vc)).sum())
        if kind == "rep" and ds in ("trending", "scaled_offset"):
            extra.update(rep_extras(unit, x_raw, path, fmin, fmax))
        rt, at = REP_TOL[unit]
        save(rid, value=V, criterion="tolerance", rtol=rt, atol=at, reference_kind="independent-impl",
             reference_detail=det, notes=f"literal replica, strict band ({fmin}, {fmax}), {path} preprocessing", **extra)
        return

    if kind == "dir":
        save(rid, value=lead_matrix(DELAYS), criterion="direction", expected_sign=np.nan_to_num(np.sign(lead_matrix(DELAYS))),
             reference_kind="analytic", reference_detail="CONVENTIONS.md D6: x_k = x0 delayed by d_k; A[i,j] > 0 iff row i leads column j (sign(d_j - d_i))",
             notes="value = d_j - d_i (true lead of row over column, samples)")
        return

    if kind == "delay":
        save(rid, value=lead_matrix(DELAYS), criterion="tolerance", rtol=0.0, atol=0.2, reference_kind="analytic",
             reference_detail="CONVENTIONS.md D6 pure delays: group delay of pair (i, j) = d_j - d_i samples",
             notes="|A[i,j] - (d_j - d_i)| <= 0.2 samples")
        return

    if kind == "gotman":
        C, f, n_fft, K, coeffs = mt_coherency(x)
        m = ks.band_mask(f, fmin, fmax, inclusive=False)
        P = x.shape[0]
        V = np.full((P, P), np.nan)
        for i, j in itertools.combinations(range(P), 2):
            g = ks.group_delay_fit(np.angle(C[m][:, i, j]), f[m])
            V[i, j], V[j, i] = g["delay"], -g["delay"]
        save(rid, value=V, criterion="tolerance", rtol=0.0, atol=0.1, reference_kind="analytic",
             reference_detail="_kernels_spectral.group_delay_fit (Gotman 1983 OLS of unwrapped coherency phase over every in-band bin, no FDR/cluster/decimation)",
             notes="canonical plain group delay; valid where coherence is significant over the whole band (delayed_noise)")
        return

    if kind == "null":
        S = load_pyspi(rid)
        C, f, n_fft, K, coeffs = mt_coherency(x)
        Vr = gd_pyspi(C, f, fmin, fmax, K)
        step2w = int(np.ceil((6.0 / T) * n_fft))            # 2W = 2 NW / T, NW = 3
        V2w = gd_pyspi(C, f, fmin, fmax, K, step=step2w)
        off = ~np.eye(S.shape[0], dtype=bool)
        fin = np.isfinite(S[off])
        save(rid, value=np.zeros(()), criterion="qualitative", pass_flag=int(not fin.any()), reference_kind="analytic",
             reference_detail="null: independent white series have zero coherence, so no significant group and no delay (all NaN)",
             notes="pass iff every off-diagonal entry is NaN",
             n_finite_pyspi=int(fin.sum()), n_offdiag=int(off.sum()), n_finite_replica=int(np.isfinite(Vr[off]).sum()),
             n_finite_replica_step_2W=int(np.isfinite(V2w[off]).sum()), step_library=kept_bins(f, fmin, fmax)[2],
             step_2W=step2w, max_abs_finite=float(np.nanmax(np.abs(S[off]))) if fin.any() else 0.0)
        return

    if kind == "popgd":
        S = load_pyspi(rid)
        d = make_data.load(ds, 1024)
        Ct, ft = ks.coherency(d["S_true"], nan_diagonal=True), d["freqs"]
        mt_ = ks.band_mask(ft, fmin, fmax, inclusive=False)
        C, f, n_fft, K, coeffs = mt_coherency(x)
        ms = ks.band_mask(f, fmin, fmax, inclusive=False)
        P = x.shape[0]
        V, Pl = np.full((P, P), np.nan), np.full((P, P), np.nan)
        for i, j in itertools.combinations(range(P), 2):
            g = ks.group_delay_fit(np.angle(Ct[mt_][:, i, j]), ft[mt_])
            V[i, j], V[j, i] = g["delay"], -g["delay"]
            h = ks.group_delay_fit(np.angle(C[ms][:, i, j]), f[ms])
            Pl[i, j], Pl[j, i] = h["delay"], -h["delay"]
        absC = np.abs(Ct[mt_])
        save(rid, value=V, criterion="direction", expected_sign=np.nan_to_num(np.sign(V)), reference_kind="analytic",
             reference_detail="population group delay: _kernels_spectral.group_delay_fit on the true VAR coherency (make_data S_true, 1/1024 grid), band strict",
             notes="pass iff pyspi is finite on some pair and every finite pyspi sign equals the population sign",
             population_delay_triu=triu(V), plain_sample_ols_triu=triu(Pl), pyspi_triu=triu(S),
             population_abs_coherency_min=float(np.nanmin(absC)), population_abs_coherency_max=float(np.nanmax(absC)),
             fisher_z_min_significant_abs_coherency=float(np.tanh(0.125 + 1.6449 * np.sqrt(0.125))))
        return

    if kind == "shares":
        v = np.ravel(load_pyspi(rid))
        save(rid, value=np.zeros(()), criterion="qualitative", pass_flag=int(v[0] == 1 and v[2] == 1),
             reference_kind="analytic", reference_detail="spectral.py:91-101 returns adj_freq[0] (a view of the cache) and fill_diagonal writes into it; first insert keyed by self.measure (spectral.py:78-81)",
             notes="pass iff the returned matrix shares memory with data.spectral_mv and a caller mutation of a later result reappears on the next call",
             shares_memory=float(v[0]), first_mutation_visible_call2=float(v[1]), second_mutation_visible_call3=float(v[2]))
        return

    if kind == "spy":
        v = np.ravel(load_pyspi(rid))
        ok = v[0] == 2 and v[1] == 0 and v[2] == fmin and v[3] == fmax and abs(v[4] - (fmax - fmin) / 50) < 1e-15
        save(rid, value=np.zeros(()), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
             reference_detail="spectral.py:73-75: argument-less library call raises TypeError, then _get_statistic forwards [fmin, fmax] and (fmax-fmin)/50",
             notes="pass iff two library calls, the first without kwargs, the last with the configured band and resolution",
             n_calls=float(v[0]), first_call_n_kwargs=float(v[1]), last_fmin=float(v[2]), last_fmax=float(v[3]), last_resolution=float(v[4]))
        return

    if kind == "excl25":
        S = load_pyspi(rid)
        C, f, n_fft, K, coeffs = mt_coherency(x)
        m = ks.band_mask(f, fmin, fmax, inclusive=False)
        has025 = bool(np.any(np.isclose(f[m], 0.25)))
        Vr, detail = gd_pyspi(C, f, fmin, fmax, K, return_detail=True)
        off = ~np.eye(S.shape[0], dtype=bool)
        all_nan = bool(np.isnan(S[off]).all())
        expect_excluded = bkey in ("low", "high")
        ok = all_nan and (has025 != expect_excluded)
        save(rid, value=np.zeros(()), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
             reference_detail="strict band rule fmin < f < fmax (connectivity.py:1313-1335); narrowband_f0-0.25 coupling sits at f = 0.25",
             notes="pass iff pyspi is all-NaN and f = 0.25 is excluded from the half bands (included in the full band)",
             band_contains_025=int(has025), pyspi_all_nan=int(all_nan), replica_n_finite=int(np.isfinite(Vr[off]).sum()),
             max_run_len=max(d_["run_len"] for d_ in detail), max_n_kept=max(d_["n_kept"] for d_ in detail),
             step=kept_bins(f, fmin, fmax)[2])
        return

    if kind == "nolte":
        S = load_pyspi(rid)
        C, f, n_fft, K, coeffs = mt_coherency(x)
        step = kept_bins(f, fmin, fmax)[2]
        N = psi_nolte(C, f, fmin, fmax)
        Nd = psi_nolte(C, f, fmin, fmax, step=step)
        Nj = psi_nolte_jackknife(coeffs, np.fft.fftfreq(n_fft), fmin, fmax)
        a, n, nd, nj = triu(S), triu(N), triu(Nd), triu(Nj)
        extra = dict(pyspi_triu=a, nolte_triu=n, nolte_dec_triu=nd, nolte_jackknife_z_triu=nj,
                     ratio_pyspi_over_nolte_triu=a / n,
                     median_abs_ratio=float(np.median(np.abs(a / n))),
                     sign_agree_nolte=float(np.mean(np.sign(a) == np.sign(n))),
                     sign_agree_nolte_dec=float(np.mean(np.sign(a) == np.sign(nd))),
                     spearman_pyspi_nolte=spearman(a, n), spearman_pyspi_nolte_dec=spearman(a, nd),
                     n_kept=int(kept_bins(f, fmin, fmax)[1].size), step=step)
        if ds in ("var2_bivariate", "var2_chain") and T <= 1024:
            d = make_data.load(ds, 1024)
            Ct = ks.coherency(d["S_true"], nan_diagonal=True)
            sub = np.arange(0, 513, 1024 // T)[: T // 2]
            Cg, fg = Ct[sub], d["freqs"][sub]
            pn, pa = triu(psi_nolte(Cg, fg, fmin, fmax)), triu(psi_pyspi(Cg, fg, fmin, fmax))
            extra.update(population_nolte_triu=pn, population_allpairs_triu=pa,
                         sign_agree_pyspi_population=float(np.mean(np.sign(a) == np.sign(pn))),
                         sign_agree_nolte_population=float(np.mean(np.sign(n) == np.sign(pn))))
        save(rid, value=N, criterion="direction", expected_sign=np.nan_to_num(np.sign(N)), reference_kind="analytic",
             reference_detail=f"{SELF}:psi_nolte = Nolte et al. 2008 eq. 3 (arXiv:0712.2352v1 p.2), adjacent bins at 1/n_fft, strict band, same independent coherency; eq. 6 jackknife over tapers as extra key",
             notes="pass iff sign(pyspi) == sign(Nolte eq. 3) on every pair; ratio/rank keys quantify the variant", **extra)
        return

    if kind == "mne":
        import warnings
        import mne
        import mne_connectivity
        from mne_connectivity import phase_slope_index
        mne.set_log_level("ERROR")
        S = load_pyspi(rid)
        P = x.shape[0]
        iu = np.triu_indices(P, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            con = phase_slope_index(x[None], indices=(iu[0], iu[1]), sfreq=1.0, mode="multitaper", fmin=fmin, fmax=fmax,
                                    mt_bandwidth=6.0 / T, mt_low_bias=True, mt_adaptive=False, verbose=False)
        v = np.asarray(con.get_data())[:, 0]
        M = np.full((P, P), np.nan)
        M[iu], M[iu[1], iu[0]] = v, -v
        C, f, n_fft, K, coeffs = mt_coherency(x)
        N = psi_nolte(C, f, fmin, fmax)
        a = triu(S)
        save(rid, value=M, criterion="direction", expected_sign=np.nan_to_num(np.sign(M)), reference_kind="external-pkg",
             reference_detail=f"mne_connectivity=={mne_connectivity.__version__} phase_slope_index(mode='multitaper', mt_bandwidth=6/T, low_bias) (adjacent bins, Nolte eq. 3), mne=={mne.__version__}",
             notes="pass iff sign(pyspi) == sign(MNE PSI) on every pair",
             mne_triu=v, pyspi_triu=a, ratio_pyspi_over_mne_triu=a / v,
             mne_vs_ref_nolte_max_abs=float(np.nanmax(np.abs(M - N))), mne_vs_ref_nolte_max_rel=float(np.nanmax(np.abs(M - N) / np.abs(N))),
             sign_agree=float(np.mean(np.sign(a) == np.sign(v))), spearman_pyspi_mne=spearman(a, v))
        return

    if kind == "delaycurve":
        C, f, n_fft, K, coeffs = mt_coherency(x)
        idx, fk, step, nband = kept_bins(f, fmin, fmax)
        fb = f[ks.band_mask(f, fmin, fmax, inclusive=False)]
        rho2 = np.array([[1.0, 1 / 1.01, 1 / 1.01, 1 / 1.01]] * 4)
        rho2[1:, 1:] = 1 / 1.01 ** 2
        rho2 = np.where(np.eye(4, dtype=bool), np.nan, rho2)      # |C|^2 for (x0, x_k) and (x_j, x_k)
        L = lead_matrix(DELAYS)
        V, Nad = np.full((4, 4), np.nan), np.full((4, 4), np.nan)
        for i in range(4):
            for j in range(4):
                if i != j:
                    V[i, j] = rho2[i, j] * all_pairs_scalar(2 * np.pi * L[i, j] * fk)
                    Nad[i, j] = rho2[i, j] * adjacent_scalar(2 * np.pi * L[i, j] * fb)
        save(rid, value=V, criterion="tolerance", rtol=0.1, atol=1.0, reference_kind="analytic",
             reference_detail="pure-delay model C_ij(f) = rho_ij exp(i 2 pi f (d_j - d_i)), rho^2 = 1/(1+0.1^2)^(1 or 2); all-pairs sum over pyspi's kept bins",
             notes="magnitude follows the all-pairs curve, not Nolte's (adjacent analytic in nolte_analytic_triu)",
             allpairs_analytic_triu=triu(V), nolte_analytic_triu=triu(Nad), lead_triu=triu(L), n_kept=int(fk.size), step=step)
        return

    if kind == "m5":
        lib = float(np.ravel(load_pyspi(rid))[0])
        ph = _probe_module().M5_PHASES
        adj, allp = adjacent_scalar(ph), all_pairs_scalar(ph)
        ok = (abs(lib - allp) < 1e-9 and abs(adj - M5_ADJ) < 1e-9 and abs(allp - M5_ALL) < 1e-9
              and np.sign(lib) != np.sign(adj))
        save(rid, value=np.asarray(allp), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
             reference_detail="REVIEW.md M5 (docs/storage_comparison/REVIEW.md:174-177): four unit coherencies; Nolte eq. 3 adjacent vs all unordered pairs",
             notes="pass iff the library's _inner_combination equals the all-pairs formula and has the opposite sign of the adjacent formula",
             library_value=lib, allpairs_value=allp, adjacent_value=adj)
        return

    if kind == "longdelay":
        pm = _probe_module()
        base = np.asarray(make_data.load("delayed_noise", 4096)["x"][0], dtype=np.float64)
        noise = np.asarray(make_data.load("iid_gaussian", T)["x"], dtype=np.float64)
        xl = preprocess(pm.build_longdelay(base, noise, T), path)
        d = np.asarray(pm.LONG_DELAYS, dtype=float)
        L = lead_matrix(d)
        S = load_pyspi(rid)
        C, f, n_fft, K, coeffs = mt_coherency(xl)
        idx, fk, step, nband = kept_bins(f, fmin, fmax)
        V = psi_pyspi(C, f, fmin, fmax) if is_psi else gd_pyspi(C, f, fmin, fmax, K)
        extra = dict(lead_triu=triu(L), pyspi_triu=triu(S), replica_max_abs_diff=float(np.nanmax(np.abs(S - V))),
                     step=step, aliasing_limit_samples=float(n_fft / (2 * step)))
        if is_psi:
            N = psi_nolte(C, f, fmin, fmax)
            Nd = psi_nolte(C, f, fmin, fmax, step=step)
            extra.update(nolte_triu=triu(N), nolte_dec_triu=triu(Nd),
                         sign_agree_nolte_truth=float(np.mean(np.sign(triu(N)) == np.sign(triu(L)))),
                         sign_agree_pyspi_truth=float(np.mean(np.sign(triu(S)) == np.sign(triu(L)))))
        save(rid, value=V, criterion="direction", expected_sign=np.nan_to_num(np.sign(L)), reference_kind="analytic",
             reference_detail=f"probes/spectral_psi_gd_probe.py:build_longdelay delays {tuple(pm.LONG_DELAYS)}; expected sign(d_j - d_i); value = {fn} replica",
             notes="direction under long delays; PSI sign aliases when tau * step / n_fft is in (1/2, 1) mod 1", **extra)
        return

    if kind in ("embed", "perm"):
        src_tag = ("alone:" if kind == "embed" else "base:") + bkey
        src = "|".join(rid.split("|")[:-1] + [src_tag])
        V = load_pyspi(src)
        extra = {"source_run": src}
        if kind == "embed":
            S = load_pyspi(rid)
            noise = np.asarray(make_data.load("iid_gaussian", T)["x"], dtype=np.float64)
            xe = preprocess(np.vstack([x_raw, noise]), path)
            Ve = estimate(unit, xe, fmin, fmax)
            Va = estimate(unit, x, fmin, fmax)
            extra.update(pyspi_alone_01=float(V[0, 1]), pyspi_embedded_01=float(S[0, 1]),
                         replica_alone_01=float(Va[0, 1]), replica_embedded_01=float(Ve[0, 1]))
        save(rid, value=V, criterion="exact", rtol=EXACT[0], atol=EXACT[1], reference_kind="same-lineage",
             reference_detail=f"pyspi self-consistency: value = pyspi output of run {src}",
             notes=("bivariate invariance: pair alone vs pair + 5 iid_gaussian channels" if kind == "embed"
                    else "permutation equivariance: A(X[perm])[inv, inv] == A(X)"), **extra)
        return

    if kind == "eeg":
        pr = run["probe"]
        fmin, fmax = float(pr["fmin"]), float(pr["fmax"])
        ex = load_exchange(rid)
        S = load_pyspi(rid)
        C, f, n_fft, K, coeffs = mt_coherency(x)
        idx, fk, step, nband = kept_bins(f, fmin, fmax)
        err = ex.get("error") or ""
        extra = dict(n_band_bins=nband, n_kept=int(fk.size), step=step, pyspi_error=err or "none",
                     pyspi_all_nan=int(bool(np.isnan(S).all())), identifier=ex.get("identifier", ""))
        if is_psi and fk.size < 2:
            ok = "IndexError" in err and bool(np.isnan(S).all())
            save(rid, value=np.zeros(()), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                 reference_detail="bin count: kept bins k/n_fft with fmin < f < fmax, every s-th; < 2 kept bins -> SC _inner_combination IndexError -> Calculator all-NaN",
                 notes="pass iff pyspi raised IndexError and the matrix is all NaN", **extra)
        elif (not is_psi) and nband == 0:
            ok = "ValueError" in err and bool(np.isnan(S).all())
            save(rid, value=np.zeros(()), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                 reference_detail="bin count: zero in-band bins -> scipy mstats.linregress np.amax(empty) ValueError -> Calculator all-NaN",
                 notes="pass iff pyspi raised ValueError and the matrix is all NaN", **extra)
        else:
            V = psi_pyspi(C, f, fmin, fmax) if is_psi else gd_pyspi(C, f, fmin, fmax, K)
            extra["n_finite_replica"] = int(np.isfinite(V).sum())
            if not is_psi:                                   # same rule with the 2W = 6/T spacing
                extra["n_finite_replica_step_2W"] = int(np.isfinite(
                    gd_pyspi(C, f, fmin, fmax, K, step=int(np.ceil((6.0 / T) * n_fft)))).sum())
            rt, at = REP_TOL[unit]
            save(rid, value=V, criterion="tolerance", rtol=rt, atol=at, reference_kind="independent-impl",
                 reference_detail=det, notes=f"EEG band ({fmin}, {fmax}) via probe; replica", **extra)
        return

    raise KeyError(f"no handler for tag {tag!r} in {rid}")


def main():
    m = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_err = 0
    for run in m["runs"]:
        try:
            handle(run)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            import traceback
            print(f"REF ERROR {run['run_id']}: {exc!r}\n{traceback.format_exc()[-1500:]}", file=sys.stderr)
    print(f"refs/{FAMILY}.py: {n_ok} references written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
