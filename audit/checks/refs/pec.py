"""pec family references (COMPARISON venv; never imports mne / mne_connectivity / pyspi).

Units: misc.pec.orth-{False,True}_log-{False,True} (identifiers pec, pec_log, pec_orth,
pec_orth_abs, pec_orth_log, pec_orth_log_abs).  pyspi delegates to
mne_connectivity.envelope_correlation (0.9.0); every reference here is built from
numpy/scipy primitives:

* ``_kernels_basic_misc.envelope_correlation`` (Hipp et al. 2012 Nat Neurosci 15:884
  amplitude/log-power envelope correlation with pairwise orthogonalisation; analytic
  signal = scipy.signal.hilbert zero-padded to the next 5-smooth length and truncated,
  as mne does; scipy.signal.hilbert is the same primitive mne calls, allowed as a
  numpy/scipy building block).  Default evaluation order: the explicit real-arithmetic
  form (Im a_i Re a_j - Re a_i Im a_j)/|a_j|, which is exactly 0 for a bit-identical copy.
  Forming the same quantity through a complex product, Im(a_i * conj(a_j)), is NOT exactly
  0 on this machine (numpy 2.5.4: residue up to ~1e-16 at every sample, from the contracted
  complex multiply), and neither is mne's order a_i * (conj(a_j)/|a_j|) (envelope.py:147-148,
  168).  ``mne_order=True`` evaluates mne's order and is used only to show that pyspi's
  value for duplicated channels is that rounding residue; ``orth_complex_product`` below
  gives the complex-product form for comparison (extra keys only).
* ``directional``: the same orthogonalised correlation before symmetrisation, c_ij =
  corr_t(|Im(a_i conj a_j)|/|a_j| [log of its square], env_j), to test what `absolute` does.
* analytic population values for white bivariate Gaussian pairs with zero-lag
  correlation rho (analytic signals circular complex Gaussian with coherence rho):
  amplitude envelopes  (2F1(-1/2,-1/2;1;rho^2) - 1) / (4/pi - 1)  (Rayleigh envelope
  moments, Davenport & Root 1958 ch. 8); log-power  6 Li2(rho^2) / pi^2  (log of
  exponential variables: Var = pi^2/6, Cov = Li2(rho^2)); orthogonalised signed
  variants 0 (the orthogonal part of y = rho x + sqrt(1-rho^2) n is sqrt(1-rho^2)
  Im(n e^{-i arg x}), independent of x).  Standard errors from a numpy Monte Carlo of
  the same kernel on fresh draws (cached under out/exchange/pec/).

Writes out/ref/<run_id>.npz for every run of manifest/pec.json.
"""
import json
import sys
from pathlib import Path

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))

import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402
from scipy.special import hyp2f1, spence  # noqa: E402
from scipy.stats import norm  # noqa: E402

import make_data as md  # noqa: E402
import _kernels_basic_misc as km  # noqa: E402

OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / "pec"
SELF = "audit/checks/refs/pec.py"
KDET = "audit/checks/refs/_kernels_basic_misc.py:envelope_correlation"

PARAMS = {"pec": (False, False, False), "pec_orth": (True, False, False), "pec_log": (False, True, False),
          "pec_orth_log": (True, True, False), "pec_orth_abs": (True, False, True),
          "pec_orth_log_abs": (True, True, True)}
ORDER = ["pec", "pec_orth", "pec_log", "pec_orth_log", "pec_orth_abs", "pec_orth_log_abs"]
CONST_M = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]


# ------------------------------------------------------------------ helpers
def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)


def detrend_lstsq(x):
    x = np.asarray(x, float)
    T = x.shape[1]
    D = np.stack([np.arange(T, dtype=float), np.ones(T)], 1)
    coef, *_ = np.linalg.lstsq(D, x.T, rcond=None)
    return x - (D @ coef).T


def ref(X, ident, mne_order=False):
    o, l, a = PARAMS[ident]
    C = km.envelope_correlation(X, orth=o, log=l, absolute=a, mne_order=mne_order)
    C = np.array(C, float)
    np.fill_diagonal(C, np.nan)
    return C


def ref_unpadded(X, ident):
    saved = km._next_5smooth
    km._next_5smooth = lambda n: n
    try:
        return ref(X, ident)
    finally:
        km._next_5smooth = saved


def _zc(v):
    v = v - v.mean(-1, keepdims=True)
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(s == 0, 1.0, s)


def directional(X, log=False):
    """c[i, j] = corr_t(orth envelope of a_i w.r.t. a_j, envelope of a_j) (before symmetrisation)."""
    X = np.asarray(X, float)
    P, T = X.shape
    a = hilbert(X, N=km._next_5smooth(T), axis=-1)[:, :T]
    mag = np.abs(a)
    Ez = _zc(np.log(mag ** 2) if log else mag)
    C = np.empty((P, P))
    for i in range(P):
        O = np.abs(a[i].imag[None, :] * a.real - a[i].real[None, :] * a.imag) / mag
        O[i] = 1.0
        if log:
            with np.errstate(divide="ignore"):
                O = np.log(O ** 2)
            O[~np.isfinite(O).all(-1)] = 0.0
        C[i] = np.sum(_zc(O) * Ez, axis=-1)
    return C


def orth_complex_product(X, ident):
    """Same as ref(X, ident) for the orth identifiers, but the orthogonal part is formed as
    Im(a_i * conj(a_j)) / |a_j| through numpy's complex multiply (not the explicit real form
    of the kernel).  Used only to report what a complex-arithmetic implementation of the
    definitional order gives for a bit-identical pair."""
    o, l, a_ = PARAMS[ident]
    X = np.asarray(X, float)
    P, T = X.shape
    a = hilbert(X, N=km._next_5smooth(T), axis=-1)[:, :T]
    mag = np.abs(a)
    Ez = _zc(np.log(mag ** 2) if l else mag)
    C = np.empty((P, P))
    for i in range(P):
        O = np.abs((a[i][None, :] * a.conj()).imag) / mag
        O[i] = 1.0
        if l:
            with np.errstate(divide="ignore"):
                O = np.log(O ** 2)
            O[~np.isfinite(O).all(-1)] = 0.0
        C[i] = np.sum(_zc(O) * Ez, axis=-1)
    if a_:
        C = np.abs(C)
    C = (C + C.T) / 2
    np.fill_diagonal(C, np.nan)
    return C


def power_env_pearson(X):
    a = hilbert(np.asarray(X, float), N=km._next_5smooth(X.shape[1]), axis=-1)[:, :X.shape[1]]
    Ez = _zc(np.abs(a) ** 2)
    C = Ez @ Ez.T
    np.fill_diagonal(C, np.nan)
    return C


def maxabs(A, B):
    m = ~np.isnan(A) & ~np.isnan(B)
    return float(np.max(np.abs(A[m] - B[m]))) if m.any() else 0.0


def asym(S):
    m = ~np.isnan(S) & ~np.isnan(S.T)
    return float(np.max(np.abs(S - S.T)[m])) if m.any() else 0.0


def load_py(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float), np.asarray(z["input"], float)


def const_env_signals(m, T=256, seed=5):
    rng = np.random.default_rng(seed)
    k = np.ones(45) / 45
    a = np.convolve(rng.standard_normal(T + 44), k, mode="valid")[:T]
    b = np.convolve(rng.standard_normal(T + 44), k, mode="valid")[:T]
    t = np.arange(T, dtype=np.float64)
    return np.stack([(1 + m * a) * np.sin(2 * np.pi * 0.25 * t), (1 + m * b) * np.sin(2 * np.pi * 0.25 * t + 0.7)])


def analytic_white(ident, rho, orth_zero=True):
    """Population value for a jointly circular Gaussian analytic pair with |coherence| = rho.
    The orthogonalised signed value is 0 only when the coherence is real (zero-lag mixing,
    orth_zero=True); otherwise (and for the abs variants) there is no closed form here (None)."""
    o, l, a = PARAMS[ident]
    if o:
        return 0.0 if (not a and orth_zero) else None
    if l:
        return 6.0 * spence(1.0 - rho ** 2) / np.pi ** 2      # Li2(r^2) = spence(1 - r^2)
    return (hyp2f1(-0.5, -0.5, 1.0, rho ** 2) - 1.0) / (4.0 / np.pi - 1.0)


def mc(kind, T, nrep=400):
    """Monte Carlo of all six identifiers on fresh white draws, z-scored like the pipeline
    (returns mean, sd and the (nrep, P, P) samples).
    kind 'd2': 4 channels with the bivariate_gaussian structure (rho 0.3 on (0,1), 0.8 on
    (2,3)); kind 'iid': 5 iid channels; kind 'delayed': the delayed_noise construction
    (x_k = x0 delayed by d_k = 0, 1, 3, 5 plus 0.1 N(0,1), make_data._b_delayed).  Returns {ident: (mean (P,P), sd (P,P), samples (nrep,P,P))}."""
    EXCH.mkdir(parents=True, exist_ok=True)
    f = EXCH / f"mc_{kind}_T{T}_n{nrep}.npz"
    if f.is_file():
        with np.load(f) as z:
            if all(i + "_samples" in z.files for i in ORDER):
                return {i: (z[i + "_mean"], z[i + "_sd"], z[i + "_samples"]) for i in ORDER}
    rng = np.random.default_rng(20261014 + T + {"d2": 0, "iid": 7, "delayed": 13}[kind])
    acc = {i: [] for i in ORDER}
    for _ in range(nrep):
        if kind == "d2":
            z = rng.standard_normal((4, T))
            x = np.empty_like(z)
            for k, rho in enumerate((0.3, 0.8)):
                x[2 * k] = z[2 * k]
                x[2 * k + 1] = rho * z[2 * k] + np.sqrt(1 - rho ** 2) * z[2 * k + 1]
        elif kind == "delayed":
            base = rng.standard_normal(T + 5)
            x = np.empty((4, T))
            x[0] = base[5:]
            for k, d in enumerate((1, 3, 5), start=1):
                x[k] = base[5 - d:5 - d + T] + 0.1 * rng.standard_normal(T)
        else:
            x = rng.standard_normal((5, T))
        x = zscore(x)
        for i in ORDER:
            acc[i].append(ref(x, i))
    out = {}
    for i in ORDER:
        A = np.stack(acc[i])
        out[i] = (A.mean(0), A.std(0, ddof=1), A)
    np.savez(f, **{i + "_mean": out[i][0] for i in ORDER}, **{i + "_sd": out[i][1] for i in ORDER},
             **{i + "_samples": out[i][2] for i in ORDER})
    return out


def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{rid}.npz", **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in kw.items()})


# ------------------------------------------------------------------ per-run references
def run_kern(r, x, tag):
    ident = r["identifier"]
    o, l, a = PARAMS[ident]
    mne = tag.endswith("mneorder")
    V = ref(x, ident, mne_order=mne)
    extra = {}
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        extra["pyspi_max_asym"] = asym(S)
        extra["pyspi_min"] = float(np.nanmin(S))
        extra["pyspi_max"] = float(np.nanmax(S))
        if o:
            other = ref(x, ident, mne_order=not mne)
            extra["maxdiff_vs_other_order"] = maxabs(S, other)
            if r["dataset"] == "shared_envelope":
                extra["pyspi_02"] = float(S[0, 2])
                extra["ref_def_order_02"] = float(ref(x, ident)[0, 2])
                extra["ref_mne_order_02"] = float(ref(x, ident, mne_order=True)[0, 2])
                extra["ref_complex_product_order_02"] = float(orth_complex_product(x, ident)[0, 2])
                m = np.ones_like(S, bool)
                m[[0, 2], [2, 0]] = False
                extra["maxdiff_def_order_excl_02"] = maxabs(np.where(m, S, np.nan), ref(x, ident))
        elif not l:
            extra["maxdiff_vs_power_envelope"] = maxabs(S, power_env_pearson(x))
    order = "mne evaluation order a_i*(conj(a_j)/|a_j|) (mne_order=True)" if mne else \
        "definitional order in explicit real form (Im a_i Re a_j - Re a_i Im a_j)/|a_j| (mne_order=False)" \
        if o else "centred Pearson"
    save(r["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{KDET}(orth={o}, log={l}, absolute={a}, mne_order={mne}); scipy.signal.hilbert "
                          f"N=next 5-smooth >= T; numpy {np.__version__}",
         notes=f"kernel path (raw input); {order}", **extra)


def run_pipe(r, x, tag):
    """pipeline / pipeline-detrend: reference on our own z-score (ddof=1) [after lstsq detrend]."""
    ident = r["identifier"]
    xin = zscore(detrend_lstsq(x)) if r["path"] == "pipeline-detrend" else zscore(x)
    V = ref(xin, ident)
    extra = {}
    py = load_py(r["run_id"])
    if py is not None:
        S, pin = py
        extra["pyspi_max_asym"] = asym(S)
        extra["input_maxdiff_vs_pyspi_preproc"] = float(np.max(np.abs(pin - xin)))
        extra["pyspi_min"] = float(np.nanmin(S))
        if PARAMS[ident][0]:
            extra["maxdiff_vs_mne_order"] = maxabs(S, ref(xin, ident, mne_order=True))
        if r["dataset"] == "trending":
            clean = md.load("bivariate_gaussian", r["T"])["x"]
            extra["pyspi_23"] = float(S[2, 3])
            extra["clean_ref_23"] = float(ref(zscore(clean), ident)[2, 3])
            extra["detrended_ref_23"] = float(ref(zscore(detrend_lstsq(x)), ident)[2, 3])
            extra["raw_zscored_ref_23"] = float(ref(zscore(x), ident)[2, 3])
    pre = "lstsq linear detrend then z-score ddof=1" if r["path"] == "pipeline-detrend" else "z-score ddof=1"
    save(r["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{KDET} on numpy-preprocessed input ({pre})",
         notes=f"{r['dataset']} {r['path']}: estimator + preprocessing vs independent reference", **extra)


def run_invar(r, x, tag):
    """scaled_offset pipeline: reference computed on the UNSCALED, UNSHIFTED bivariate_gaussian draw."""
    ident = r["identifier"]
    base = md.load("bivariate_gaussian", r["T"])["x"]
    V = ref(zscore(base), ident)
    extra = {}
    py = load_py(r["run_id"])
    pb = load_py(f"{r['unit_id']}|bivariate_gaussian|{r['T']}|pipeline|{ident}-stat")
    if py is not None and pb is not None:
        extra["pyspi_scaled_vs_pyspi_base_maxdiff"] = maxabs(py[0], pb[0])
    save(r["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{KDET} on z-scored bivariate_gaussian (scaled_offset = scale*D2 + offset, same draw)",
         notes="pipeline invariance to positive scale and offset", **extra)


def run_offsetkern(r, x, tag):
    ident = r["identifier"]
    d = md.load("scaled_offset", r["T"])
    base = md.load("bivariate_gaussian", r["T"])["x"]
    V = ref(x, ident)
    scale = np.asarray(d["scale"])[:, None]
    extra = dict(ref_offset_effect_maxdiff=maxabs(V, ref(base, ident)),
                 ref_offset_only_0_1_m2_5_maxdiff=maxabs(ref(base + np.array([0.0, 1.0, -2.0, 5.0])[:, None], ident),
                                                         ref(base, ident)),
                 ref_scale_only_effect_maxdiff=maxabs(ref(scale * base, ident), ref(base, ident)))
    save(r["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{KDET} on raw scaled_offset (kernel path)",
         notes="kernel path: value depends on per-channel offset (Hilbert keeps the DC bin); extra keys give the "
               "reference-side offset-only and scale-only effects relative to bivariate_gaussian", **extra)


def run_stat(r, x, tag):
    ident = r["identifier"]
    T = r["T"]
    kind = {"bivariate_gaussian": "d2", "iid_gaussian": "iid", "delayed_noise": "delayed"}[r["dataset"]]
    mcs = mc(kind, T)
    mean, sd, samples = mcs[ident]
    P = mean.shape[0]
    V = np.zeros((P, P))
    rho = np.zeros((P, P))
    if kind == "d2":
        rho[0, 1] = rho[1, 0] = 0.3
        rho[2, 3] = rho[3, 2] = 0.8
    if kind == "delayed":
        # analytic signal of white noise: corr(z(t+d), z(t)) = delta_d + i g(d), g(d) = 2/(pi d) for odd d,
        # 0 for even d (discrete Hilbert kernel); x_k has variance 1.01, so |coherence(x0, x_k)| =
        # g(d_k)/sqrt(1.01) and every other pair has an even lag difference (2 or 4) -> 0.
        for k, d in enumerate((1, 3, 5), start=1):
            rho[0, k] = rho[k, 0] = 2.0 / (np.pi * d) / np.sqrt(1.01)
    o, l, a = PARAMS[ident]
    for i in range(P):
        for j in range(P):
            if i != j:
                av = analytic_white(ident, rho[i, j], orth_zero=(kind != "delayed" or rho[i, j] == 0))
                V[i, j] = mean[i, j] if av is None else av
    np.fill_diagonal(V, np.nan)
    m = P * (P - 1) // 2
    zstar = float(norm.ppf(1 - 0.025 / m))            # Bonferroni z over the m unique pairs (reported only)
    se = sd                                            # compare.py contract: pass if |S - V| <= 2 se = 2 MC sd
    extra = dict(z_star_bonferroni=zstar, n_pairs=m, mc_nrep=400)
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        zz = np.abs(S - V) / sd
        extra["max_z_raw"] = float(np.nanmax(zz))
        extra["n_pairs_beyond_2sd"] = int(np.sum(np.triu(np.nan_to_num(zz) > 2.0, 1)))
        extra["bonferroni_pass"] = int(float(np.nanmax(zz)) <= zstar)
        wi, wj = np.unravel_index(np.nanargmax(zz), zz.shape)
        dev = abs(S[wi, wj] - mean[wi, wj])
        extra["worst_pair"] = [int(min(wi, wj)), int(max(wi, wj))]
        extra["worst_pair_mc_two_sided_p"] = float(np.mean(np.abs(samples[:, wi, wj] - mean[wi, wj]) >= dev))
        extra["worst_pair_mc_fraction_above"] = float(np.mean(samples[:, wi, wj] >= S[wi, wj]))
        extra["pyspi_max_asym"] = asym(S)
    if kind == "d2":
        for (i, j), rv in (((0, 1), 0.3), ((2, 3), 0.8)):
            extra[f"value_{i}{j}"] = float(V[i, j])
            extra[f"mc_mean_{i}{j}"] = float(mean[i, j])
            extra[f"mc_sd_{i}{j}"] = float(sd[i, j])
            if py is not None:
                extra[f"pyspi_{i}{j}"] = float(py[0][i, j])
        if not o and not l:
            extra["power_env_rho2_01_23"] = [0.09, 0.64]
    elif kind == "delayed":
        for j in (1, 2, 3):
            extra[f"coherence_0{j}"] = float(rho[0, j])
            extra[f"value_0{j}"] = float(V[0, j])
            extra[f"mc_mean_0{j}"] = float(mean[0, j])
            extra[f"mc_sd_0{j}"] = float(sd[0, j])
            if py is not None:
                extra[f"pyspi_0{j}"] = float(py[0][0, j])
    else:
        off = ~np.eye(P, dtype=bool)
        extra["mc_mean_offdiag_mean"] = float(mean[off].mean())
        extra["mc_sd_offdiag_mean"] = float(sd[off].mean())
        if py is not None:
            extra["pyspi_offdiag_mean"] = float(np.nanmean(py[0]))
            extra["pyspi_offdiag_min"] = float(np.nanmin(py[0]))
    has_closed = analytic_white(ident, 0.3, orth_zero=(kind != "delayed")) is not None
    kindtxt = "analytic population value" if has_closed else \
        "Monte Carlo mean of the reference kernel on fresh draws (no closed form here)"
    save(r["run_id"], value=V, se=se, criterion="statistical",
         reference_kind="analytic" if has_closed else "independent-impl",
         reference_detail=f"{SELF}:analytic_white ((2F1(-1/2,-1/2;1;r^2)-1)/(4/pi-1); 6 Li2(r^2)/pi^2; 0 for "
                          f"orth) with se from {SELF}:mc ({kind}, T={T}, 400 reps, {KDET})",
         notes=f"{kindtxt}; pass if |pyspi - value| <= 2 MC sd per pair (compare.py rule, no multiplicity correction); "
               f"bonferroni_pass reports the family-wise z_star_bonferroni = {zstar:.3f} reading for information only", **extra)


def run_dir(r, x, tag):
    ident = r["identifier"]
    o = PARAMS[ident][0]
    V = ref(zscore(x), ident)
    sign = np.zeros_like(V)
    for i, j in ([(0, 1), (1, 2)] if o else [(0, 1), (0, 2), (1, 2)]):
        sign[i, j] = sign[j, i] = 1
    extra = {}
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        extra["pyspi_max_asym"] = asym(S)
        extra["pyspi_01"], extra["pyspi_02"], extra["pyspi_03"], extra["pyspi_12"] = (float(S[0, 1]), float(S[0, 2]),
                                                                                       float(S[0, 3]), float(S[1, 2]))
        extra["pyspi_01_equals_12_bitwise"] = bool(S[0, 1] == S[1, 2])
    save(r["run_id"], value=V, expected_sign=sign, criterion="direction", reference_kind="independent-impl",
         reference_detail=f"{KDET} on z-scored shared_envelope; expected sign + for the shared-envelope pairs",
         notes="undirected: both orientations carry the same value; x2 == x0 bitwise, x1 = x0 shifted by pi/3 in "
               "phase with the same envelope; (0,2) excluded for orth (degenerate, see dupchan runs)", **extra)


def run_constenv(r, x, tag):
    ident = r["identifier"]
    pipe = r["path"] == "pipeline"
    refs, cva, cvb = [], [], []
    for m in CONST_M:
        X = const_env_signals(m)
        Xin = zscore(X) if pipe else X
        refs.append(float(ref(Xin, ident)[0, 1]))
        a = np.abs(hilbert(Xin, N=km._next_5smooth(Xin.shape[1]), axis=-1))
        cv = a.std(1) / a.mean(1)
        cva.append(float(cv[0]))
        cvb.append(float(cv[1]))
    extra = dict(m_values=CONST_M, ref_centred=refs, env_cv_a=cva, env_cv_b=cvb)
    ok = 0
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        extra["pyspi_values"] = [float(v) for v in S]
        extra["absdiff_vs_centred"] = [float(abs(s - c)) for s, c in zip(S, refs)]
        ok = int(np.all(np.abs(S) <= 1.0))
    save(r["run_id"], value=np.array(refs), criterion="qualitative", pass_flag=ok, reference_kind="independent-impl",
         reference_detail=f"{SELF}:const_env_signals + {KDET} (centred Pearson); pass iff every pyspi value is a "
                          f"valid correlation (|r| <= 1)",
         notes=f"pair (0,1) of (1 + m a) sin(2 pi 0.25 t), (1 + m b) sin(2 pi 0.25 t + 0.7), T=256, "
               f"{'pipeline (z-scored)' if pipe else 'kernel (raw)'}; m = {CONST_M}", **extra)


def run_dupchan(r, x, tag):
    ident = r["identifier"]
    xin = zscore(x)
    Vd, Vm = ref(xin, ident), ref(xin, ident, mne_order=True)
    extra = dict(ref_def_order_02=float(Vd[0, 2]), ref_mne_order_02=float(Vm[0, 2]),
                 ref_complex_product_order_02=float(orth_complex_product(xin, ident)[0, 2]))
    ok = 0
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        extra["pyspi_02"] = float(S[0, 2])
        extra["maxdiff_vs_mne_order_full"] = maxabs(S, Vm)
        extra["input_x0_equals_x2_bitwise"] = bool(np.array_equal(py[1][0], py[1][2]))
        ok = int(abs(S[0, 2]) <= 1e-8)
    save(r["run_id"], value=Vd, criterion="qualitative", pass_flag=ok, reference_kind="independent-impl",
         reference_detail=f"{KDET} definitional order in explicit real form (0 for a bit-identical copy) and "
                          f"mne_order=True; {SELF}:orth_complex_product for the complex-multiply form",
         notes="x2 is bit-identical to x0: the orthogonal part is 0 in exact arithmetic, so the orthogonalised "
               "correlation is 0 (or undefined); pass iff |pyspi[0,2]| <= 1e-8", **extra)


def run_propchan(r, x, tag):
    ident = r["identifier"]
    rng = np.random.default_rng(0)
    T = int(r["T"])
    a = rng.standard_normal(T)
    b = rng.standard_normal(T)
    xin = zscore(np.stack([a, 2 * a, a + 0.3 * b, b]))
    Vd, Vm = ref(xin, ident), ref(xin, ident, mne_order=True)
    extra = dict(ref_def_order_01=float(Vd[0, 1]), ref_mne_order_01=float(Vm[0, 1]),
                 ref_complex_product_order_01=float(orth_complex_product(xin, ident)[0, 1]),
                 zscored_channels_bit_identical=bool(np.array_equal(xin[0], xin[1])))
    ok = 0
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        extra["pyspi_01"] = float(S[0, 1])
        extra["maxdiff_vs_mne_order_full"] = maxabs(S, Vm)
        m = np.ones_like(S, bool)
        m[[0, 1], [1, 0]] = False
        extra["maxdiff_def_order_excl_01"] = maxabs(np.where(m, S, np.nan), Vd)
        ok = int(abs(S[0, 1]) <= 1e-8)
    save(r["run_id"], value=Vd, criterion="qualitative", pass_flag=ok, reference_kind="independent-impl",
         reference_detail=f"{KDET} on z-scored [a, 2a, a+0.3b, b], a, b = default_rng(0) draws; pipeline path",
         notes="exactly proportional pair (0,1): orthogonalised correlation is 0 in exact arithmetic; pass iff "
               "|pyspi[0,1]| <= 1e-8", **extra)


def run_absnonorth(r, x, tag):
    ident = r["identifier"]
    o, l, _ = PARAMS[ident]
    V = km.envelope_correlation(x, orth=False, log=l, absolute=True)
    V = np.array(V, float)
    np.fill_diagonal(V, np.nan)
    extra = {}
    py = load_py(r["run_id"])
    pp = load_py(r["run_id"].replace("absnonorth", "kern"))
    if py is not None:
        extra["pyspi_min"] = float(np.nanmin(py[0]))
        if pp is not None:
            extra["bitwise_equal_to_absolute_false"] = bool(np.array_equal(py[0], pp[0], equal_nan=True))
    save(r["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{KDET}(orth=False, log={l}, absolute=True): absolute ignored without orth "
                          f"(envelope.py:184-188)",
         notes="PowerEnvelopeCorrelation(orth=False, absolute=True) via probe (no config identifier)", **extra)


def run_abssem(r, x, tag):
    ident = r["identifier"]
    l = PARAMS[ident][1]
    signed_id = "pec_orth_log" if l else "pec_orth"
    C = directional(x, log=l)
    Vabs = (np.abs(C) + np.abs(C).T) / 2
    Vsig = (C + C.T) / 2
    np.fill_diagonal(Vabs, np.nan)
    np.fill_diagonal(Vsig, np.nan)
    kern_abs = ref(x, ident)
    extra = dict(directional_vs_kernel_maxdiff=maxabs(Vabs, kern_abs),
                 n_pairs_opposite_sign=int(np.sum(np.triu((np.sign(C) * np.sign(C.T)) < 0, 1))))
    ok = 0
    py = load_py(r["run_id"])
    ps = load_py(r["run_id"].replace(f"{ident}-abssem", f"{signed_id}-kern").replace(r["unit_id"], r["unit_id"]))
    if py is not None and ps is not None:
        A, S = py[0], ps[0]
        gap = A - np.abs(S)
        extra["pyspi_abs_vs_mean_abs_directional_maxdiff"] = maxabs(A, Vabs)
        extra["pyspi_abs_vs_abs_of_signed_maxdiff"] = maxabs(A, np.abs(S))
        extra["pyspi_signed_vs_directional_mean_maxdiff"] = maxabs(S, Vsig)
        extra["min_gap_abs_minus_abs_signed"] = float(np.nanmin(gap))
        extra["n_pairs_strict_gap"] = int(np.sum(np.triu(np.nan_to_num(gap) > 1e-12, 1)))
        ok = int(maxabs(A, Vabs) <= 1e-12 and np.nanmin(gap) >= -1e-15)
    save(r["run_id"], value=Vabs, criterion="qualitative", pass_flag=ok, reference_kind="independent-impl",
         reference_detail=f"{SELF}:directional (c_ij before symmetrisation); abs variant = (|c_ij| + |c_ji|)/2",
         notes=f"pass iff pyspi {ident} == (|c_ij| + |c_ji|)/2 within 1e-12 and >= |{signed_id}| everywhere "
               f"(abs applied per direction before averaging)", **extra)


def run_pad(r, x, tag):
    ident = r["identifier"]
    xin = zscore(np.asarray(x)[:, :1031])
    V = ref(xin, ident)
    extra = dict(n_fft=km._next_5smooth(1031), ref_padded_vs_unpadded_maxdiff=maxabs(V, ref_unpadded(xin, ident)))
    py = load_py(r["run_id"])
    if py is not None:
        extra["pyspi_vs_unpadded_maxdiff"] = maxabs(py[0], ref_unpadded(xin, ident))
    save(r["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{KDET} with Hilbert FFT padded to next 5-smooth (1031 -> 1080) on z-scored "
                          f"iid_gaussian_T4096[:, :1031]",
         notes="T = 1031 is not 5-smooth; extra keys give the padded-vs-unpadded effect", **extra)


def run_zero(r, x, tag):
    X = np.array(x, float)
    X[3] = 0.0
    P = X.shape[0]
    V = np.full((len(ORDER), P, P), np.nan)
    extra = {}
    ok = 0
    py = load_py(r["run_id"])
    if py is not None:
        S = py[0]
        off = ~np.eye(P, dtype=bool)
        rc = np.zeros((P, P), bool)
        rc[3, :] = rc[:, 3] = True
        rc &= off
        for k, ident in enumerate(ORDER):
            extra[f"{ident}_zero_rowcol"] = ("all NaN" if np.isnan(S[k][rc]).all() else
                                             f"finite, max |.| = {np.nanmax(np.abs(S[k][rc])):.3g}")
            extra[f"{ident}_other_entries_finite"] = bool(np.isfinite(S[k][off & ~rc]).all())
        ok = int(all(np.isnan(S[k][rc]).all() for k in range(len(ORDER))))
    save(r["run_id"], value=V, criterion="qualitative", pass_flag=ok, reference_kind="analytic",
         reference_detail="Pearson correlation with a zero-variance (all-zero) series is undefined (NaN)",
         notes="kernel path, iid_gaussian_T256 with channel 3 set to 0 (the pipeline rejects constant channels); "
               "pass iff all six identifiers return NaN for the undefined row/column; order " + ",".join(ORDER),
         **extra)


DISPATCH = {"kern": run_kern, "kern-mneorder": run_kern, "stat": run_stat, "lagstat": run_stat, "dir": run_dir, "pipe": run_pipe,
            "detrend": run_pipe, "trendraw": run_pipe, "constenv": run_constenv, "invar": run_invar,
            "offsetkern": run_offsetkern, "pad1031": run_pad, "dupchan": run_dupchan, "propchan": run_propchan,
            "absnonorth": run_absnonorth, "abssem": run_abssem, "zerochan": run_zero}


def main():
    man = json.loads((CHECKS / "manifest" / "pec.json").read_text())
    n = 0
    for r in man["runs"]:
        tag = r["run_id"].split("|")[-1].split("-", 1)[1]
        x = np.asarray(md.load(r["dataset"], r["T"])["x"], float)
        if r.get("processes") is not None:
            x = x[list(r["processes"])]
        DISPATCH[tag](r, x, tag)
        n += 1
    print(f"refs/pec.py: wrote {n} reference files")


if __name__ == "__main__":
    main()
