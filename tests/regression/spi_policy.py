"""Per-SPI policy for the golden-value regression suite.

Everything here is matched against SPI identifiers with :func:`fnmatch.fnmatch`.
Keep the lists short and explain every entry: the default tolerance is meant to
catch any intentional change in a calculation, so an override is a documented
exception, not a convenience.
"""
from fnmatch import fnmatch

#: Seed set immediately before every SPI computation (after the statistics
#: modules are imported, because ``pyspi.statistics.misc`` reseeds on import).
SEED = 42

#: Wall-clock threshold (seconds, measured by the generator at the selected
#: size) above which an SPI test gets the ``slow`` marker.
SLOW_THRESHOLD_S = 2.0

#: Default comparison tolerances. Deterministic SPIs reproduce to ~1e-14 on the
#: same environment; 1e-6 relative absorbs BLAS/compiler drift between machines,
#: and 1e-8 absolute covers values that are pure rounding noise (for example
#: ``bary_euclidean_mean`` is about 1e-16).
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-8

#: (pattern, rtol, atol, reason). First match wins; the default applies otherwise.
#: The stochastic rows keep the default values; they are listed to document
#: that those SPIs match only because ``seed_all()`` runs right before them.
#: The robust covariance SPIs (``*MinCovDet``, ``*EllipticEnvelope``) are no
#: longer listed: pyspi fits FAST-MCD with ``basic.ROBUST_RANDOM_STATE`` (the
#: harness seed), so they reproduce their goldens without ``seed_all()``.
#:
#: The iterative optimisers (``*GraphicalLassoCV*``, ``phi_*``, ``gpfit_*``,
#: ``anm``) are not listed: seeded run-to-run and against the stored goldens
#: they were measured bit-for-bit reproducible on the generating machine (max
#: difference 0 for all 11 matching identifiers), so the default applies. If
#: another machine shows drift, re-add a row for that pattern with the measured
#: drift in its reason.
TOLERANCES = [
    ("bary*_sgddtw_*", 1e-6, 1e-8, "tslearn subgradient DBA shuffles with the global numpy RNG; reproducible only via seed_all()"),
]

#: SPIs whose output depends on the global numpy RNG. They are reproducible
#: only when ``seed_all()`` runs right before ``multivariate``, and are marked
#: ``stochastic``.
STOCHASTIC = [
    "bary*_sgddtw_*",
]

#: Excluded from the Calculator end-to-end comparison: a single seeded
#: Calculator pass consumes the global RNG in a different order than the
#: per-SPI seeding, and these SPIs then differ from their per-SPI goldens.
CALC_E2E_SKIP = [
    "bary*_sgddtw_*",
]

#: Safety net used only when no measured time is available for an identifier
#: (for example a golden set written by an older generator): these are marked
#: ``slow`` then. When ``meta.json`` records a time, the measurement decides,
#: so a fast implementation of a historically slow SPI still runs by default.
FORCE_SLOW = [
    "ccm_*",
    "*kraskov*DCE*",
    "hhg",
    "mgc*",
    "*sgddtw*",
    "softdtw*",
    "anm",
    "gpfit_*",
    "*MinCovDet*",
    "*EllipticEnvelope*",
    "*GraphicalLassoCV*",
    "sgc_parametric*",
    "cce_*",
    "di_*",
]

#: Representative SPIs locked under every (normalise, detrend) combination.
#: Entries are identifier prefixes resolved against the catalogue (an entry
#: matches when the identifier equals it or starts with it followed by ``_``).
#: At least two per statistics module, except wavelet (its six SPIs are all
#: ``psi_wavelet`` band variants, so one represents them); all are fast at the
#: small size.
PREPROC_SPIS = [
    # basic
    "cov_EmpiricalCovariance",
    "prec_EmpiricalCovariance",
    "xcorr_max_sig-True_fs-1_lag-0-125_bartlett-holm_alpha-0-05",
    "spearmanr",
    "kendalltau",
    # distance
    "pdist_euclidean",
    "dcorr",
    "dtw",
    "gwtau",
    "bary_euclidean_mean",
    # causal
    "igci",
    "reci",
    "anm",  # the only normalise-sensitive causal SPI; ~0.25 s per variant
    # infotheory
    "mi_gaussian",
    "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1",
    "phi_star_t-1_norm-0",
    # spectral
    "cohmag_multitaper_mean",
    "plv_multitaper_mean",
    "sgc_nonparametric_mean",
    # wavelet
    "psi_wavelet_mean",
    # misc
    "coint_johansen_max_eig_stat_order-1_ardiff-1",
    "lmfit_Ridge",
    "pec",
    "ids",
]

#: Preprocessing variants: (normalise, detrend) -> key suffix.
PREPROC_VARIANTS = {
    (True, False): "n1d0",   # Calculator default
    (True, True): "n1d1",    # the EEG workflow: detrend first, then z-score
    (False, False): "n0d0",
    (False, True): "n0d1",
}


def matches_any(identifier, patterns):
    return any(fnmatch(identifier, p) for p in patterns)


def tolerance_for(identifier):
    """Return ``(rtol, atol)`` for an identifier."""
    for pattern, rtol, atol, _reason in TOLERANCES:
        if fnmatch(identifier, pattern):
            return rtol, atol
    return DEFAULT_RTOL, DEFAULT_ATOL


def is_stochastic(identifier):
    return matches_any(identifier, STOCHASTIC)


def is_slow(identifier, time_s=None):
    """``slow`` if measured above the threshold; FORCE_SLOW decides only when
    no measurement is available."""
    if time_s is None:
        return matches_any(identifier, FORCE_SLOW)
    return time_s > SLOW_THRESHOLD_S


def preproc_key(identifier, normalise, detrend):
    return f"{identifier}::{PREPROC_VARIANTS[(bool(normalise), bool(detrend))]}"


def split_preproc_key(key):
    """Inverse of :func:`preproc_key`: returns ``(identifier, normalise, detrend)``."""
    identifier, suffix = key.rsplit("::", 1)
    for (normalise, detrend), s in PREPROC_VARIANTS.items():
        if s == suffix:
            return identifier, normalise, detrend
    raise KeyError(key)


def select_preproc_identifiers(identifiers):
    """Resolve PREPROC_SPIS entries against a catalogue.

    Each entry selects exactly one identifier: an exact match if present,
    otherwise the first catalogue identifier starting with ``entry + "_"``
    (spectral and wavelet entries have several frequency-band variants; only
    the first is locked under the preprocessing variants). Returns identifiers
    in PREPROC_SPIS order; entries with no match are reported by the caller.
    """
    identifiers = list(identifiers)
    chosen = []
    for entry in PREPROC_SPIS:
        if entry in identifiers:
            chosen.append(entry)
            continue
        for ident in identifiers:
            if ident.startswith(entry + "_"):
                chosen.append(ident)
                break
    return chosen
