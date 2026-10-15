#!/usr/bin/env python3
"""Independent information-theoretic reference kernels for the pyspi audit.

Pure numpy/scipy (cKDTree, digamma, slogdet).  No pyspi, no jpype/JIDT.
Units are nats unless a function says otherwise (JIDT's kernel estimators
report bits, so the box-kernel functions default to bits).  Arrays are
"time-major": a 1-D series has shape (T,); a multivariate sample has shape
(N, d) (rows = observations).  A 1-D input is treated as (N, 1).

JIDT conventions established from source (java/source/infodynamics/...; "J:" = that
tree) and mirrored here so comparisons are like-for-like.

CITATION VERSION.  Every JIDT file:line in this module (this note and the function
docstrings) is at the v1.6.1 TAG unless it says a9b6b3e.  The jar bundled with pyspi is
labelled 1.6.1 but is a later build: its entries are dated 2024-04-21 and its source matches
commits 3d1f3fec86 and a9b6b3e77c (2024-04-20); the exact build commit is not settled, and the
a9b6b3e line numbers stand in for it (audit/notes/it_kozachenko.md).  Identical at the tag and
at a9b6b3e, so the cited lines also hold for the jar: MatrixUtils, MathsUtils, EuclideanUtils,
MutualInfoCalculatorMultiVariateKraskov1, ConditionalMutualInfoCalculatorMultiVariateKraskov1,
TransferEntropyCalculatorKraskov, TransferEntropyCommon, KernelEstimatorMultiVariate,
KernelEstimatorTransferEntropy, MutualInfoCalculatorMultiVariateKernel,
TransferEntropyCalculatorKernel, MutualInfoCalculatorMultiVariateGaussian,
ConditionalMutualInfoCalculatorMultiVariateGaussian.  Changed after the tag (a9b6b3e line
numbers in brackets; the cited text is unchanged unless noted): MutualInfoMultiVariateCommon
:509-530 [617-641, plus the seeding lines 629-631]; ConditionalMutualInfoMultiVariateCommon
:493-510 [756-776, plus seeding]; MutualInfoCalculatorMultiVariateKraskov :79, :183-184 [same
lines], :313-314 [306-307], :528-529 [521-522]; ConditionalMutualInfoCalculatorMultiVariateKraskov
:101, :217-218 [same lines], :553 [546]; NearestNeighbourSearcher :418-424 [469-475];
EntropyCalculatorMultiVariateGaussian :126-127 [90-92, adds an originalObservations
reference], :210-219 [162-175, adds an isComputed early return], :216-217 [171-172];
TransferEntropyCalculatorViaCondMutualInfo :368-377 [375-384], :572-597 [599-625, the
addObservations call now also records observation-set ids];
EntropyCalculatorMultiVariateKozachenko :300-321 [87-108, rawData renamed observations],
:417-426 [198-207], :436-474 [217-255], and its noise code (:105-115, :204-212) moved to
EntropyCalculatorMultiVariateCommon a9b6b3e:200-211; KdTree :703-704 and :1076-1078 were
rewritten to track observation sets, with the same rule for a single set
[751-752, 1124-1127]; EntropyCalculatorMultiVariateKernel :158-169 [109-124, adds an
isComputed early return; the bits conversion :168 is a9b6b3e:122 and now assigns
lastAverage].  The change found to alter values on pyspi's path
is NOISE_SEED (KSG noise bullet); the estimators here were compared with the jar's outputs
in the it_gaussian, it_kraskov, it_kozachenko and it_kernel checks (audit/notes/it_*.md).

* Covariance uses N-1: J:utils/MatrixUtils.java:3370-3377 (covarianceTwoColumns
  divides by data.length-1); Gaussian entropy uses it via
  J:measures/continuous/gaussian/EntropyCalculatorMultiVariateGaussian.java:126-127
  and the analytic H = 0.5*(d(1+ln 2pi) + ln det C) at :210-219 (nats, no bias
  correction).  MI/CMI/TE are ratios of determinants, so the ddof cancels there.
* NORMALISE (default true for MI/CMI/TE and kernel calculators) is a per-column
  z-score with the N-1 std: MatrixUtils.java:2490-2503 (stdDevs), :2663-2672
  (normalise; a zero-std column is only centred).  MI/CMI normalise the embedded
  variables, then add noise: J:.../MutualInfoMultiVariateCommon.java:509-530,
  J:.../ConditionalMutualInfoMultiVariateCommon.java:493-510.
* KSG noise: KSG MI/CMI add N(0, 1e-8^2) jitter after normalising by default
  (J:.../kraskov/MutualInfoCalculatorMultiVariateKraskov.java:183-184,
  .../ConditionalMutualInfoCalculatorMultiVariateKraskov.java:217-218); the v1.6.1
  tag draws it from an unseeded java.util.Random (MutualInfoMultiVariateCommon.java:520).
  The bundled jar also honours a NOISE_SEED property (pyspi/statistics/infotheory.py:91
  sets "42"): each time the observations are finalised a new Random seeded with it
  supplies the jitter (MutualInfoMultiVariateCommon.java a9b6b3e:626-642,
  ConditionalMutualInfoMultiVariateCommon.java a9b6b3e:766-785; Kozachenko:
  EntropyCalculatorMultiVariateCommon.java a9b6b3e:200-211).  That stream CAN be
  reproduced: java.util.Random is a specified 48-bit LCG with a polar-method nextGaussian.
  The pure-Python ports refs/it_kraskov.py:JavaRandom and refs/it_kozachenko.py:JavaRandom
  match the JVM's nextGaussian to <= 8.9e-16 (20 001 draws, seeds 42, 43, 7;
  audit/notes/it_kraskov.md) and <= 4.4e-16 (audit/notes/it_kozachenko.md).  This module
  does not port it: ``noise_level``/``seed`` here add numpy default_rng normals, for
  quantifying the jitter effect only.  To mirror JIDT, add the ported stream to the inputs
  (refs/it_kraskov.py:jidt_prep fills rows in order, source/var1 columns, then dest/var2,
  then cond; refs/it_kozachenko.py:jitter fills row-major) and pass noise_level=0.
* KSG norm: max norm in every space (MutualInfoCalculatorMultiVariateKraskov.java:79,
  ConditionalMutualInfoCalculatorMultiVariateKraskov.java:101).  Marginal counts
  are STRICT (< eps): KdTree.java:1076-1078 (allowEqualToR false via
  countPointsStrictlyWithinR, NearestNeighbourSearcher.java:418-424).
* KSG1 average: psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N)
  (MutualInfoCalculatorMultiVariateKraskov1.java:80-103, ...Kraskov.java:528-529);
  psi(N) is NOT adjusted for the Theiler window (JIDT TODO at Kraskov1.java:100-102).
  CMI KSG1: psi(k) + <psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1)>
  (ConditionalMutualInfoCalculatorMultiVariateKraskov1.java:255, ...Kraskov.java:553);
  with an empty conditional n_z = N-1 (Kraskov1.java:237).
* Theiler / DYN_CORR_EXCL for KSG: a neighbour s of t is used only if
  |s - t| > W, i.e. |s - t| <= W is excluded (KdTree.java:703-704, :1076).
  W = 0 excludes only self.  For the box kernels the rule differs: neighbours
  with |s - t| < W are excluded (KernelEstimatorMultiVariate.java:771-797), and the
  self point is INCLUDED when no exclusion is set.
* Kozachenko-Leonenko (J:.../kozachenko/EntropyCalculatorMultiVariateKozachenko.java):
  k = 1, EUCLIDEAN distance (EuclideanUtils.java:69-110, despite the "max norm"
  often assumed), H = (d/N) sum ln(2 r_i) + [psi(N) - psi(1)]
  + ln((pi/4)^{d/2} / Gamma(d/2+1))   (lines 300-321, 417-426, 436-474),
  i.e. Kraskov 2004 eq. 20 with the Euclidean unit ball.  No normalisation; 1e-8
  jitter by default (lines 105-115, 204-212).  MathsUtils.java:33 uses a truncated
  Euler constant 0.5772156 but it cancels in psi(N)-psi(1) for N < 10000.
* Box kernel (J:.../kernel/KernelEstimatorMultiVariate.java): half-width per
  dimension w_d = width * std_d (N-1) when NORMALISE (lines 205-213, default on);
  the data are not rescaled.  Step kernel accepts |dx_d| <= w_d in every
  dimension (lines 959-966); p = count / (N * prod 2 w_d) (lines 737, 816).
  Entropy/MI/TE are averaged in BITS (EntropyCalculatorMultiVariateKernel.java:168,
  MutualInfoCalculatorMultiVariateKernel.java:197, TransferEntropyCalculatorKernel.java).
* Delay embedding: MatrixUtils.makeDelayEmbeddingVector(data, k, tau, start, n)
  (MatrixUtils.java:2212-2236) returns rows [x[t], x[t-tau], ..., x[t-(k-1)tau]]
  (most recent FIRST) for t = start .. start+n-1; the 2-arg form uses start=k-1,
  n=T-k+1 (MatrixUtils.java:2084-2092), so k=0 yields T+1 empty rows.
* TE embedding (J:.../TransferEntropyCalculatorViaCondMutualInfo.java:368-377,
  572-597): s = max((k-1) k_tau, (l-1) l_tau + delay - 1); for n = 0..T-s-2 with
  t = s+n: dest_past ends at t, dest_next = y[t+1], source_past ends at t+1-delay.
  The first s+1 samples are dropped.  TE Kraskov uses CMI KSG1
  (kraskov/TransferEntropyCalculatorKraskov.java:119).
* TIES / QUANTISED DATA: the KSG and box-kernel references are only meaningful on
  continuous inputs.  KSG: Java's N-1 z-score differs from numpy's at the ulp level,
  so strict "< eps" counts can flip at exact-tie boundaries (audit probes: up to
  0.05 nats on data rounded to 0.1; pre-z-scored input with NORMALISE=false matches
  JIDT to 1e-15), and JIDT's seeded 1e-8 jitter breaks ties differently again.  Box:
  this module implements the |dx| <= w compare-to-all rule (forceCompareToAll).  JIDT's
  default is a bin-indexed search (KernelEstimatorMultiVariate.java:198-220, 286-294,
  floor binning at :820), and the entropy calculator cannot force compare-to-all.  In
  theory a pair at exactly |dx| == w that rounding puts two bins apart could be missed;
  this is a theoretical limit only, not reproduced in the it_kernel check: on the k*0.1
  grid (k = 0..30, w = 0.1, NORMALISE=false), where rounding puts some neighbour
  differences just above and some just below 0.1, JIDT's bin search returned the
  compare-to-all value (run infotheory.est.kernel|iid_gaussian|64|kernel|tie-grid, 8.9e-16),
  and on var2_bivariate T=1024 the MI and TE calculators gave bit-identical values with and
  without FORCE_KERNEL_COMPARE_TO_ALL (run ...|var2_bivariate|1024|pipeline|force-all).  It
  could matter only for quantised data with NORMALISE off.

Run ``python _kernels_infotheory.py --selftest``.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

LN2 = math.log(2.0)


# ------------------------------------------------------------------ helpers
def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def _cov(X, ddof: int = 1) -> np.ndarray:
    """Sample covariance of (N, d) data; ddof=1 mirrors MatrixUtils.java:3376."""
    X = _as2d(X)
    Xc = X - X.mean(axis=0)
    return np.atleast_2d(Xc.T @ Xc / (X.shape[0] - ddof))


def zscore_jidt(X) -> np.ndarray:
    """Per-column (x - mean)/std with the N-1 std; zero-std columns only centred.

    Mirrors MatrixUtils.normalise (MatrixUtils.java:2663-2672) with stdDevs
    (MatrixUtils.java:2490-2503).  Shape (N, d) -> (N, d).
    """
    X = _as2d(X)
    mu = X.mean(axis=0)
    sd = np.sqrt(((X - mu) ** 2).sum(axis=0) / (X.shape[0] - 1))
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def embed(x, k: int, tau: int = 1, start: int | None = None, n: int | None = None) -> np.ndarray:
    """Delay embedding mirroring JIDT MatrixUtils.makeDelayEmbeddingVector.

    Row r (t = start + r) is [x[t], x[t-tau], ..., x[t-(k-1)*tau]] (most recent
    first).  Defaults start=(k-1)*tau, n=T-start drop the first (k-1)*tau samples;
    k=0 gives T+1 rows of width 0 (MatrixUtils.java:2084-2092, 2212-2236).
    x: (T,) -> (n, k).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.size
    start = (k - 1) * tau if start is None else start
    n = T - start if n is None else n
    if start < (k - 1) * tau or start + n > T:
        raise ValueError("embedding out of range")
    t = start + np.arange(n)
    if k == 0:
        return np.empty((n, 0))
    return x[t[:, None] - tau * np.arange(k)[None, :]]


# ------------------------------------------------------------------ Gaussian
def gaussian_entropy(cov) -> float:
    """H = 0.5 ln((2 pi e)^d det C) nats (Cover & Thomas 2006, Thm 8.4.1).

    Same expression as EntropyCalculatorMultiVariateGaussian.java:216-217.
    cov: (d, d).  Returns nan if C is not positive definite (JIDT throws).
    """
    C = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    d = C.shape[0]
    if d == 0:
        return 0.0
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        return float("nan")
    return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logdet)


def gaussian_joint_entropy(*arrays, ddof: int = 1) -> float:
    """Gaussian H of the column-concatenation of (N, d_i) arrays from the N-1 covariance."""
    arrs = [_as2d(a) for a in arrays if _as2d(a).shape[1] > 0]
    if not arrs:
        return 0.0
    return gaussian_entropy(_cov(np.hstack(arrs), ddof))


def gaussian_conditional_entropy(X, Y, ddof: int = 1) -> float:
    """H(X|Y) = H(X,Y) - H(Y), exactly as pyspi JIDTBase._compute_conditional_entropy
    (pyspi/statistics/infotheory.py:180-192).  X: (N, dx), Y: (N, dy) (dy may be 0)."""
    return gaussian_joint_entropy(X, Y, ddof=ddof) - gaussian_joint_entropy(Y, ddof=ddof)


def gaussian_mi(x, y) -> float:
    """I(X;Y) = 0.5 ln(det Sx det Sy / det Sxy) nats; 1-D: -0.5 ln(1 - rho^2).

    Cover & Thomas 2006 eq. 8.56; MutualInfoCalculatorMultiVariateGaussian.java:411-413.
    x: (N,) or (N, dx); y likewise.
    """
    x, y = _as2d(x), _as2d(y)
    return (gaussian_joint_entropy(x) + gaussian_joint_entropy(y)
            - gaussian_joint_entropy(x, y))


def gaussian_cmi(x, y, z) -> float:
    """I(X;Y|Z) = 0.5 ln(det Sxz det Syz / (det Sxyz det Sz)) nats.

    ConditionalMutualInfoCalculatorMultiVariateGaussian.java:458-466.
    """
    return (gaussian_joint_entropy(x, z) + gaussian_joint_entropy(y, z)
            - gaussian_joint_entropy(x, y, z) - gaussian_joint_entropy(z))


def gaussian_tlmi(x, y, lag: int = 1) -> float:
    """Time-lagged MI I(X_t; Y_{t+lag}) with pyspi's alignment src[:-1], targ[1:]
    (pyspi/statistics/infotheory.py:305-309, lag=1).  x, y: (T,).  lag >= 1
    (lag 0 would slice x[:-0] = empty; use gaussian_mi for zero lag)."""
    if int(lag) != lag or lag < 1:
        raise ValueError(f"gaussian_tlmi needs integer lag >= 1, got {lag!r}; use gaussian_mi for lag 0")
    x, y = np.ravel(x), np.ravel(y)
    return gaussian_mi(x[:-lag], y[lag:])


def te_embedding(src, targ, k=1, k_tau=1, l=1, l_tau=1, delay=1):
    """(source_past (N,l), dest_next (N,1), dest_past (N,k)), N = T - s - 1 with
    s = max((k-1)k_tau, (l-1)l_tau + delay - 1).  Mirrors
    TransferEntropyCalculatorViaCondMutualInfo.java:368-377 and 584-596."""
    src, targ = np.ravel(src), np.ravel(targ)
    T = targ.size
    s = max((k - 1) * k_tau, (l - 1) * l_tau + delay - 1)
    n = T - s - 1
    dest_past = embed(targ, k, k_tau, start=s, n=n)
    dest_next = embed(targ, 1, 1, start=s + 1, n=n)
    src_past = embed(src, l, l_tau, start=s + 1 - delay, n=n)
    return src_past, dest_next, dest_past


def gaussian_te(src, targ, k=1, k_tau=1, l=1, l_tau=1, delay=1) -> float:
    """Gaussian TE = I(X_past; Y_t | Y_past) (Schreiber 2000 eq. 4; Barnett, Barrett
    & Seth 2009: TE = GC/2) from covariances, with JIDT's TE embedding
    (see te_embedding).  Nats.  src, targ: (T,)."""
    sp, dn, dp = te_embedding(src, targ, k, k_tau, l, l_tau, delay)
    return gaussian_cmi(sp, dn, dp)


def gaussian_te_ols(src, targ, k=1, k_tau=1, l=1, l_tau=1, delay=1) -> float:
    """Same TE as 0.5 ln(RSS_restricted / RSS_full) from OLS with intercept
    (Granger form; Barnett et al. 2009 eq. 6).  Identical to gaussian_te up to
    round-off because the Schur complements of the centred covariance are RSS/(N-1)."""
    sp, dn, dp = te_embedding(src, targ, k, k_tau, l, l_tau, delay)
    y = dn[:, 0]

    def rss(Z):
        Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
        beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
        r = y - Z1 @ beta
        return float(r @ r)

    return 0.5 * math.log(rss(dp) / rss(np.hstack([dp, sp])))


def cce_embedding(src, targ, i: int):
    """pyspi CausalEntropy samples for term i (pyspi/statistics/infotheory.py:447-452):
    Yp = makeDelayEmbeddingVector(targ, i-1)[:-1], Xp = makeDelayEmbeddingVector(src, i),
    Yf = targ[i-1:].  Row r: Yf = y[t], Yp = [y[t-1..t-i+1]], Xp = [x[t..t-i+1]],
    t = r + i - 1 (first i-1 samples dropped).  Returns (Yf (N,1), Yp (N,i-1), Xp (N,i))."""
    src, targ = np.ravel(src), np.ravel(targ)
    Yp = embed(targ, i - 1, 1, start=i - 2, n=targ.size - i + 2)[:-1]
    Xp = embed(src, i)
    Yf = targ[i - 1:, None]
    return Yf, Yp, Xp


def gaussian_cce(src, targ, n: int = 5) -> float:
    """Causally conditioned entropy H(Y^n || X^n) = sum_{i=1}^n H(Y_t | Y_{t-1..t-i+1},
    X_{t..t-i+1}) (Kramer 1998 eq. 3.6; Quinn et al. 2011), each term on its own
    sliding-window sample as pyspi does (infotheory.py:439-454).  Nats."""
    total = 0.0
    for i in range(1, n + 1):
        Yf, Yp, Xp = cce_embedding(src, targ, i)
        total += gaussian_conditional_entropy(Yf, np.hstack([Yp, Xp]))
    return total


def di_block_term(targ, n: int = 5) -> float:
    """B = sum_i H(Y^(i))/i - sum_i H(Y_t | Y_{t-1..t-i+1}); pyspi DI - Massey DI = B.
    H(Y^(i)) uses makeDelayEmbeddingVector(targ, i) as infotheory.py:497; the
    conditional terms use the CCE sample of term i.  B >= 0 asymptotically
    (block entropy per symbol >= conditional entropy), = 0 for white Y."""
    targ = np.ravel(targ)
    b = 0.0
    for i in range(1, n + 1):
        b += gaussian_joint_entropy(embed(targ, i)) / i
        Yf, Yp, _ = cce_embedding(targ, targ, i)
        b -= gaussian_conditional_entropy(Yf, Yp)
    return b


def gaussian_di(src, targ, n: int = 5, form: str = "pyspi") -> float:
    """Directed information X -> Y.

    form="pyspi": sum_{i=1}^n H(Y^(i))/i - CCE, as DirectedInfo computes it
      (infotheory.py:487-510).  Not Massey's quantity: the first sum is a sum of
      block-entropy rates, not H(Y^n).
    form="massey": I(X^n -> Y^n) = sum_i I(X^i; Y_i | Y^{i-1}) (Massey 1990 eq. 6)
      = sum_i [H(Y_t|Y_{t-1..t-i+1}) - H(Y_t|Y_{t-1..t-i+1}, X_{t..t-i+1})],
      each term on the CCE sample of term i.  pyspi - massey = di_block_term.
    """
    cce = gaussian_cce(src, targ, n)
    if form == "pyspi":
        return sum(gaussian_joint_entropy(embed(np.ravel(targ), i)) / i
                   for i in range(1, n + 1)) - cce
    if form == "massey":
        s = 0.0
        for i in range(1, n + 1):
            Yf, Yp, _ = cce_embedding(src, targ, i)
            s += gaussian_conditional_entropy(Yf, Yp)
        return s - cce
    raise ValueError(form)


def gaussian_si(x, y, tau: int = 1) -> float:
    """Stochastic interaction SI = H(X_t|X_{t-tau}) + H(Y_t|Y_{t-tau}) - H(XY_t|XY_{t-tau})
    (Ay 2015 eq. 6; Barrett & Seth 2011 eq. 9).  Nats.  Same alignment as pyspi
    (x[tau:] | x[:-tau]; infotheory.py:526-534)."""
    x, y = _as2d(x), _as2d(y)
    xy = np.hstack([x, y])
    hj = gaussian_conditional_entropy(xy[tau:], xy[:-tau])
    hx = gaussian_conditional_entropy(x[tau:], x[:-tau])
    hy = gaussian_conditional_entropy(y[tau:], y[:-tau])
    return hx + hy - hj


def xme_embedding(src, targ, k: int):
    """pyspi CrossmapEntropy samples (infotheory.py:407-415): targ_future = targ[k:],
    src_past columns = src lags 1, 2, ..., k-1 (lag 1 only when k = 1), i.e.
    history_length=k uses k-1 lags for k >= 2 and k=1, k=2 coincide.
    Returns (Yf (T-k,1), Xpast (T-k, max(1,k-1)))."""
    src, targ = np.ravel(src), np.ravel(targ)
    T = src.size
    lags = [1] + list(range(2, k))
    Xp = np.stack([src[k - L:T - L] for L in lags], axis=1)
    return targ[k:, None], Xp


def gaussian_xme(src, targ, k: int = 10) -> float:
    """Cross-map entropy as pyspi computes it: H(Y_t | X_{t-1..t-(k-1)}) = H(Y,Xp) - H(Xp)
    (infotheory.py:405-423).  Gaussian (N-1 covariance), nats."""
    Yf, Xp = xme_embedding(src, targ, k)
    return gaussian_conditional_entropy(Yf, Xp)


# ------------------------------------------------------------------ Kozachenko
def entropy_kozachenko(X, norm: str = "euclidean", normalise: bool = False,
                       noise_level: float = 0.0, seed: int | None = None) -> float:
    """Kozachenko-Leonenko k=1 entropy in nats (Kozachenko & Leonenko 1987;
    Kraskov et al. 2004 eq. 20 with k=1).

    norm="euclidean" mirrors JIDT EntropyCalculatorMultiVariateKozachenko
    (EuclideanUtils.java:69-110; constant ln((pi/4)^{d/2}/Gamma(d/2+1)) at
    Kozachenko.java:436-474 combined with d*ln(2 r_i), lines 307-321, and
    psi(N) - psi(1), lines 417-426).  norm="max" uses the max-norm unit ball
    (ln c_d = d ln 2).  The two coincide for d = 1.  JIDT never normalises and
    adds 1e-8 jitter by default; here both are opt-in (normalise uses the N-1
    z-score; jitter is N(0, noise_level^2) from numpy default_rng(seed), NOT JIDT's
    seeded java.util.Random stream -- see the module note on KSG noise).
    Duplicate points give r_i = 0 -> -inf (as JIDT without jitter).  X: (N,) or (N, d).

    SCOPE: exact nearest neighbours only (cKDTree).  JIDT uses its exact search only up to
    2000 rows; above that it switches to computeMinEuclideanDistancesNaive
    (EuclideanUtils.java:30, :69-75; identical at a9b6b3e), whose early exit can return
    non-nearest distances for d >= 2 (d = 1 is immune).  So for N > 2000 and d >= 2 this
    function does NOT model JIDT/pyspi (audit/notes/it_kozachenko.md; same-lineage port of
    the naive routine: refs/it_kozachenko.py:nn_naive_jidt).
    """
    X = _as2d(X)
    if normalise:
        X = zscore_jidt(X)
    if noise_level:
        X = X + noise_level * np.random.default_rng(seed).standard_normal(X.shape)
    N, d = X.shape
    p = 2 if norm == "euclidean" else np.inf
    r = cKDTree(X).query(X, k=2, p=p)[0][:, 1]
    with np.errstate(divide="ignore"):
        sum_log = d * np.mean(np.log(2.0 * r))
    if norm == "euclidean":
        log_cd_over_2d = 0.5 * d * math.log(math.pi / 4.0) - gammaln(d / 2.0 + 1.0)
    else:
        log_cd_over_2d = 0.0
    return float(sum_log + digamma(N) - digamma(1) + log_cd_over_2d)


# ------------------------------------------------------------------ box kernels
def _box_widths(X, width, normalise):
    X = _as2d(X)
    if normalise:
        sd = np.sqrt(((X - X.mean(0)) ** 2).sum(0) / (X.shape[0] - 1))
        return width * sd
    return np.full(X.shape[1], float(width))


def _box_match(X, widths, rows):
    """Boolean (len(rows), N): |X[r]-X[s]| <= w_d for every d (KernelEstimatorMultiVariate.java:959-966)."""
    return np.all(np.abs(X[rows, None, :] - X[None, :, :]) <= widths, axis=2)


def _chunks(N, size=256):
    for a in range(0, N, size):
        yield np.arange(a, min(N, a + size))


def _excl_mask(rows, N, theiler):
    """True where s is USED: no exclusion if theiler==0, else |s-t| >= theiler (kernel rule)."""
    if theiler <= 0:
        return None
    return np.abs(rows[:, None] - np.arange(N)[None, :]) >= theiler


def entropy_box_kernel(X, width: float, bits: bool = True, normalise: bool = True) -> float:
    """Box-kernel entropy -<log p(x_t)>, p = count(|x_s - x_t|_d <= w_d, self included)
    / (N prod 2 w_d).  Mirrors EntropyCalculatorMultiVariateKernel.java:158-169 with
    KernelEstimatorMultiVariate (widths = width*std_{N-1} if normalise, lines 205-213;
    p at line 816).  Returns bits by default (JIDT's unit).  X: (N,) or (N, d).
    Exact |dx| == w ties: compare-to-all rule, see module note on quantised data."""
    X = _as2d(X)
    N = X.shape[0]
    w = _box_widths(X, width, normalise)
    vol = np.prod(2.0 * w)
    counts = np.concatenate([_box_match(X, w, r).sum(1) for r in _chunks(N)])
    h = -np.mean(np.log(counts / (N * vol)))
    return float(h / LN2 if bits else h)


def mi_box_kernel(x, y, width: float, bits: bool = True, normalise: bool = True,
                  theiler: int = 0) -> float:
    """Box-kernel MI <log p(x,y)/(p(x)p(y))>, separate estimators for x, y and joint
    with their own N-1-std widths (MutualInfoCalculatorMultiVariateKernel.java:105-200;
    normaliseData is a no-op there, lines 167-171).  theiler>0 mirrors the kernel
    DYN_CORR_EXCL: neighbours with |s-t| < theiler (and self) dropped and the
    denominator reduced accordingly (KernelEstimatorMultiVariate.java:771-797).
    A sample with zero joint count contributes 0 (line 186).  Bits by default."""
    x, y = _as2d(x), _as2d(y)
    J = np.hstack([x, y])
    N = J.shape[0]
    wx, wy = _box_widths(x, width, normalise), _box_widths(y, width, normalise)
    wj = np.concatenate([wx, wy])
    total = 0.0
    for rows in _chunks(N):
        mx, my = _box_match(x, wx, rows), _box_match(y, wy, rows)
        mj = mx & my & _box_match(J, wj, rows)
        use = _excl_mask(rows, N, theiler)
        denom = N if use is None else use.sum(1)
        if use is not None:
            mx, my, mj = mx & use, my & use, mj & use
        px = mx.sum(1) / denom / np.prod(2 * wx)
        py = my.sum(1) / denom / np.prod(2 * wy)
        pj = mj.sum(1) / denom / np.prod(2 * wj)
        ok = pj > 0
        total += np.sum(np.log(pj[ok] / (px[ok] * py[ok])))
    mi = total / N
    return float(mi / LN2 if bits else mi)


def te_box_kernel(src, targ, k: int = 1, width: float = 0.25, bits: bool = True,
                  normalise: bool = True) -> float:
    """Box-kernel TE (Schreiber 2000 eq. 4 with step kernels; Kaiser & Schreiber 2002)
    mirroring TransferEntropyCalculatorKernel.java:255-310 + KernelEstimatorTransferEntropy:
    N = T-k samples; dest_past rows [y[t]..y[t-k+1]] for t = k-1..T-2
    (TransferEntropyCommon.java:265-275), next = y[t+1], source = x[t] (one lag,
    no tau).  Widths: past column d = width*std_d, NEXT reuses the width of past
    column 0, source = width*std(source sample) (N-1 std).  Self included, |dx| <= w.
    local = ln[(c_nps/c_ps)/(c_np/c_p)] (0 if c_nps = 0).  Bits by default."""
    src, targ = np.ravel(src), np.ravel(targ)
    T = targ.size
    past = embed(targ, k, 1, start=k - 1, n=T - k)
    nxt = targ[k:]
    sv = src[k - 1:T - 1]
    wp = _box_widths(past, width, normalise)
    ws = _box_widths(sv, width, normalise)[0]
    wn = wp[0]
    N = T - k
    total = 0.0
    for rows in _chunks(N):
        mp = _box_match(past, wp, rows)
        mn = np.abs(nxt[rows, None] - nxt[None, :]) <= wn
        ms = np.abs(sv[rows, None] - sv[None, :]) <= ws
        cp, cnp, cps, cnps = mp.sum(1), (mp & mn).sum(1), (mp & ms).sum(1), (mp & mn & ms).sum(1)
        ok = cnps > 0
        total += np.sum(np.log((cnps[ok] / cps[ok]) / (cnp[ok] / cp[ok])))
    te = total / N
    return float(te / LN2 if bits else te)


# ------------------------------------------------------------------ KSG
def _prep(arrays, normalise, noise_level, seed):
    out = [zscore_jidt(a) if normalise else _as2d(a).copy() for a in arrays]
    if noise_level:
        rng = np.random.default_rng(seed)
        out = [a + noise_level * rng.standard_normal(a.shape) for a in out]
    return out


def _knn_eps(J, k, theiler):
    """Max-norm distance to the k-th neighbour of each row excluding |s-t| <= theiler
    (KdTree.java:703-704; theiler=0 excludes only self)."""
    N = J.shape[0]
    kq = min(N, k + 2 * theiler + 1)
    dist, idx = cKDTree(J).query(J, k=kq, p=np.inf)
    keep = np.abs(idx - np.arange(N)[:, None]) > theiler
    keep &= idx < N
    pos = np.argmax(np.cumsum(keep, axis=1) == k, axis=1)
    if not np.all(keep.sum(1) >= k):
        raise ValueError("too few samples outside the Theiler window")
    return dist[np.arange(N), pos]


def _count_strict(A, eps, theiler):
    """# s with ||A_s - A_t||_max < eps_t and |s-t| > theiler (KdTree.java:1076-1078)."""
    N = A.shape[0]
    r = np.nextafter(eps, 0.0)
    c = cKDTree(A).query_ball_point(A, r=r, p=np.inf, return_length=True).astype(np.int64)
    t = np.arange(N)
    for o in range(-theiler, theiler + 1):
        s = t + o
        ok = (s >= 0) & (s < N)
        d = np.max(np.abs(A[t[ok]] - A[s[ok]]), axis=1)
        c[t[ok]] -= (d <= r[ok]).astype(np.int64)
    c[eps <= 0] = 0
    return c


def mi_ksg1(x, y, k: int = 4, theiler: int = 0, normalise: bool = True,
            noise_level: float = 0.0, seed: int | None = None) -> float:
    """KSG algorithm 1 MI (Kraskov, Stoegbauer & Grassberger 2004 eq. 8), nats:
    psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N), max norm, strict marginal counts,
    Theiler exclusion |s-t| <= theiler, psi(N) unadjusted, N-1 z-scoring, optional
    jitter; mirrors MutualInfoCalculatorMultiVariateKraskov1.java:80-103 and
    MutualInfoCalculatorMultiVariateKraskov.java:313-314, 528-529.  x, y: (N,) or (N, d).
    Continuous inputs only for JIDT comparisons (tied data: see module note)."""
    x, y = _prep([x, y], normalise, noise_level, seed)
    N = x.shape[0]
    eps = _knn_eps(np.hstack([x, y]), k, theiler)
    nx, ny = _count_strict(x, eps, theiler), _count_strict(y, eps, theiler)
    return float(digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N))


def cmi_ksg1(x, y, z, k: int = 4, theiler: int = 0, normalise: bool = True,
             noise_level: float = 0.0, seed: int | None = None) -> float:
    """KSG1 conditional MI (Frenzel & Pompe 2007 eq. 2 with KSG algorithm 1 counts;
    JIDT ConditionalMutualInfoCalculatorMultiVariateKraskov1.java:75-260), nats:
    psi(k) + <psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1)>; eps from the joint (x,y,z)
    max-norm kNN, counts strict and Theiler-excluded; n_z = N-1 when z is empty
    (line 237), which makes cmi_ksg1(x, y, empty) == mi_ksg1(x, y)."""
    x, y, z = _prep([x, y, z], normalise, noise_level, seed)
    N = x.shape[0]
    eps = _knn_eps(np.hstack([x, y, z]), k, theiler)
    if z.shape[1] == 0:
        nxz, nyz = _count_strict(x, eps, theiler), _count_strict(y, eps, theiler)
        nz = np.full(N, N - 1)
    else:
        nxz = _count_strict(np.hstack([x, z]), eps, theiler)
        nyz = _count_strict(np.hstack([y, z]), eps, theiler)
        nz = _count_strict(z, eps, theiler)
    return float(digamma(k) + np.mean(digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1)))


def te_ksg(src, targ, k_hist=1, k_tau=1, l_hist=1, l_tau=1, knn=4, theiler=0,
           delay=1, normalise=True, noise_level=0.0, seed=None) -> float:
    """KSG TE = cmi_ksg1(source_past; dest_next | dest_past) on JIDT's TE embedding
    (te_embedding; TransferEntropyCalculatorKraskov.java:119 uses KSG1).  The
    z-scoring is applied to the embedded variables (ConditionalMutualInfoMultiVariateCommon
    .java:493-494).  Theiler indices are sample indices of the embedded series."""
    sp, dn, dp = te_embedding(src, targ, k_hist, k_tau, l_hist, l_tau, delay)
    return cmi_ksg1(sp, dn, dp, k=knn, theiler=theiler, normalise=normalise,
                    noise_level=noise_level, seed=seed)


def mi_ksg1_sklearn(x, y, k: int = 4, seed: int = 0) -> float:
    """SECOND OPINION (external, labelled): sklearn.feature_selection.mutual_info_regression
    (KSG algorithm 1, its own scaling and 1e-10 jitter).  1-D x, y only.  Nats."""
    from sklearn.feature_selection import mutual_info_regression
    return float(mutual_info_regression(np.ravel(x)[:, None], np.ravel(y), n_neighbors=k,
                                        random_state=seed)[0])


# ------------------------------------------------------------------ Theiler windows
def acf_pyspi(x) -> np.ndarray:
    """Biased ACF at lags 0..T-1 as pyspi/utils.py:47-60 (z-score, np.correlate full,
    divide by lag 0).  x: (T,) -> (T,)."""
    x = np.ravel(np.asarray(x, dtype=np.float64))
    x = (x - x.mean()) / x.std()
    a = np.correlate(x, x, mode="full")
    a = a / a[a.size // 2]
    return a[a.size // 2:]


def theiler_window_pyspi(x, y, raw: bool = False):
    """pyspi AUTO window: int(2 * dot(acf_x, acf_y)) over ALL lags 0..T-1, lag 0
    included (pyspi/statistics/infotheory.py:195-220).  Equals int(bartlett + 1)
    where bartlett = theiler_window_bartlett(x, y, max_lag=None)."""
    v = 2.0 * float(np.dot(acf_pyspi(x), acf_pyspi(y)))
    return v if raw else int(v)


def theiler_window_bartlett(x, y, max_lag: int | None = None) -> float:
    """Bartlett (1946) variance-inflation factor for a cross-correlation estimate,
    1 + 2 sum_{tau=1}^{max_lag} acf_x(tau) acf_y(tau) (Bartlett's formula; e.g.
    Pyper & Peterman 1998 eq. 1: N_eff = N / this).  max_lag=None uses all lags as pyspi."""
    ax, ay = acf_pyspi(x), acf_pyspi(y)
    m = ax.size - 1 if max_lag is None else max_lag
    return 1.0 + 2.0 * float(np.dot(ax[1:m + 1], ay[1:m + 1]))


# ------------------------------------------------------------------ selftest
def _var2_te_truth(A1, A2, Sigma, src, tgt):
    """Population TE(src->tgt) with k=l=2 for a bivariate VAR(2): 0.5 ln(v_r / Sigma_tt)."""
    from scipy.linalg import solve_discrete_lyapunov
    P = A1.shape[0]
    F = np.block([[A1, A2], [np.eye(P), np.zeros((P, P))]])
    Q = np.zeros((2 * P, 2 * P))
    Q[:P, :P] = Sigma
    S = solve_discrete_lyapunov(F, Q)
    C = F @ S  # Cov(s_{t+1}, s_t), s_t = [x_t, y_t, x_{t-1}, y_{t-1}]
    past = [tgt, tgt + P]
    c = C[tgt, past]
    v_r = (F @ S @ F.T + Q)[tgt, tgt] - c @ np.linalg.solve(S[np.ix_(past, past)], c)
    return 0.5 * math.log(v_r / Sigma[tgt, tgt])


def selftest() -> bool:
    sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
    import make_data  # noqa: E402

    results = []

    def check(name, ok, detail):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    T = 4096
    # (0) analytic Gaussian identities
    h1 = gaussian_entropy(np.eye(3))
    check("gaussian_entropy(I3) = 1.5 ln(2 pi e)", abs(h1 - 1.5 * math.log(2 * math.pi * math.e)) < 1e-12, f"{h1:.12f}")
    C = np.array([[1.0, 0.6], [0.6, 1.0]])
    hmi = 2 * gaussian_entropy([[1.0]]) - gaussian_entropy(C)
    check("Gaussian MI identity -0.5 ln(1-rho^2)", abs(hmi + 0.5 * math.log(1 - 0.36)) < 1e-12, f"{hmi:.12f}")

    # (1) gaussian_mi on bivariate_gaussian
    D = make_data.load("bivariate_gaussian", T)
    for (i, j), mi_true in zip(D["pairs"], D["true_mi_nats"]):
        v = gaussian_mi(D["x"][i], D["x"][j])
        check(f"(1) gaussian_mi pair ({i},{j})", abs(v - mi_true) < 0.02, f"{v:.4f} vs true {mi_true:.4f} (tol 0.02)")

    # (2) KSG1 MI vs truth and the sklearn second opinion; CMI checks
    for (i, j), mi_true in zip(D["pairs"], D["true_mi_nats"]):
        v = mi_ksg1(D["x"][i], D["x"][j], k=4)
        s = mi_ksg1_sklearn(D["x"][i], D["x"][j], k=4)
        check(f"(2) mi_ksg1 pair ({i},{j})", abs(v - mi_true) < 0.03, f"{v:.4f} vs true {mi_true:.4f} (tol 0.03)")
        check(f"(2) sklearn 2nd opinion pair ({i},{j})", abs(s - v) < 0.01, f"sklearn {s:.4f} vs mi_ksg1 {v:.4f} (tol 0.01)")
    xi, yi = D["x"][2], D["x"][3]
    a, b = mi_ksg1(xi, yi), cmi_ksg1(xi, yi, np.empty((T, 0)))
    check("(2) cmi_ksg1(x,y,empty) == mi_ksg1 (n_z = N-1)", abs(a - b) < 1e-12, f"{b:.12f} vs {a:.12f}")
    vw = mi_ksg1(xi, yi, theiler=5)
    check("(2) mi_ksg1 Theiler=5 ~ unchanged on serially iid samples (rho=0.8 pair)", abs(vw - a) < 0.02, f"{vw:.4f} vs {a:.4f} (tol 0.02)")
    # brute-force O(N^2) KSG1 with ties (rounded y) and Theiler W=3: strict counts, |s-t|<=W excluded
    rb = np.random.default_rng(3)
    bx = rb.standard_normal(300)
    by = np.round(0.6 * bx + rb.standard_normal(300), 1)
    zx, zy = zscore_jidt(bx), zscore_jidt(by)
    tt_ = np.arange(300)
    ex = np.abs(tt_[:, None] - tt_[None]) <= 3
    Dx, Dy = np.abs(zx - zx.T), np.abs(zy - zy.T)
    eb = np.sort(np.where(ex, np.inf, np.maximum(Dx, Dy)), axis=1)[:, 3]
    brute = digamma(4) + digamma(300) - np.mean(digamma(((Dx < eb[:, None]) & ~ex).sum(1) + 1)
                                                + digamma(((Dy < eb[:, None]) & ~ex).sum(1) + 1))
    fast = mi_ksg1(bx, by, k=4, theiler=3)
    check("(2) mi_ksg1 (cKDTree) == brute-force KSG1, ties + Theiler 3", abs(fast - brute) < 1e-12, f"{fast:.12f} vs {brute:.12f}")
    pre = mi_ksg1(zscore_jidt(bx), zscore_jidt(by), k=4, theiler=3, normalise=False)
    check("(2) tied data: pre-z-scored + normalise=False == normalise=True (JIDT-matching path)", pre == fast, f"{pre:.15f}")
    tj = mi_ksg1(bx, by, k=4, noise_level=1e-8, seed=5) - mi_ksg1(bx, by, k=4)
    print(f"INFO  tied data: 1e-8 jitter moves mi_ksg1 by {tj:+.3e} nats (KSG ties are estimator-ill-posed)")
    jit = mi_ksg1(xi, yi, noise_level=1e-8, seed=1)
    print(f"INFO  JIDT-style 1e-8 jitter effect on mi_ksg1: {jit - a:+.2e} nats")
    SP = make_data.load("sparse_precision", T)
    Ct = SP["cov_true"]
    cmi_true = (gaussian_entropy(Ct[np.ix_([0, 2], [0, 2])]) + gaussian_entropy(Ct[np.ix_([1, 2], [1, 2])])
                - gaussian_entropy(Ct[np.ix_([0, 1, 2], [0, 1, 2])]) - gaussian_entropy(Ct[np.ix_([2], [2])]))
    v = cmi_ksg1(SP["x"][0], SP["x"][1], SP["x"][2])
    g = gaussian_cmi(SP["x"][0], SP["x"][1], SP["x"][2])
    check("(2) cmi_ksg1 I(x0;x1|x2) sparse_precision", abs(v - cmi_true) < 0.03, f"{v:.4f} (gauss {g:.4f}) vs true {cmi_true:.4f} (tol 0.03)")
    v0 = cmi_ksg1(SP["x"][0], SP["x"][2], SP["x"][1])
    check("(2) cmi_ksg1 I(x0;x2|x1) ~ 0 (tridiagonal precision)", abs(v0) < 0.02, f"{v0:.4f} (tol 0.02)")

    # (3) Kozachenko
    Z = make_data.load("iid_gaussian", T)["x"]
    h_true = 0.5 * math.log(2 * math.pi * math.e)
    v = entropy_kozachenko(Z[0])
    check("(3) entropy_kozachenko iid N(0,1)", abs(v - h_true) < 0.05, f"{v:.4f} vs {h_true:.4f} (tol 0.05)")
    vm = entropy_kozachenko(Z[0], norm="max")
    check("(3) euclidean == max norm in 1-D", abs(v - vm) < 1e-12, f"{vm:.12f}")
    for nm in ("euclidean", "max"):
        v2 = entropy_kozachenko(Z[:2].T, norm=nm)
        check(f"(3) entropy_kozachenko 2-D {nm}", abs(v2 - 2 * h_true) < 0.08, f"{v2:.4f} vs {2 * h_true:.4f} (tol 0.08)")
    vj = entropy_kozachenko(Z[0], noise_level=1e-8, seed=1)
    vs, vn = entropy_kozachenko(Z[0] * 3 + 1), entropy_kozachenko(Z[0] * 3 + 1, normalise=True)
    sd1 = float(np.std(Z[0], ddof=1))
    check("(3) Kozachenko scale equivariance: H(3x+1)=H(x)+ln3; normalise -> H(x)-ln sd",
          abs(vs - v - math.log(3)) < 1e-9 and abs(vn - (v - math.log(sd1))) < 1e-9, f"{vs:.6f}, {vn:.6f}")
    print(f"INFO  Kozachenko: JIDT-style 1e-8 jitter effect {vj - v:+.2e} nats")

    # (4) box kernel on uniform data (analytic expectation with edge effect)
    w = 0.1
    u01 = np.random.default_rng(12345).uniform(0, 1, T)
    edge = lambda L: math.log2(L) + 2 * w * (1 - LN2) / (L * LN2)  # E[H_box] for U(0,L), w <= L/2
    v = entropy_box_kernel(u01, w, normalise=False)
    check("(4) entropy_box_kernel U(0,1), w=0.1 vs edge-corrected analytic", abs(v - edge(1.0)) < 0.02, f"{v:.4f} bits vs {edge(1.0):.4f} (tol 0.02)")
    check("(4) entropy_box_kernel U(0,1) vs true h = log2(1) = 0", abs(v) < 0.1, f"{v:.4f} bits (tol 0.1); brief's -log2(w)={-math.log2(w):.3f} is not this estimator's target")
    NL = make_data.load("nonlinear_dependence", T)["x"]
    v = entropy_box_kernel(NL[0], w, normalise=False)
    check("(4) entropy_box_kernel U(-1,1) dataset vs log2(2)+edge", abs(v - edge(2.0)) < 0.02, f"{v:.4f} bits vs {edge(2.0):.4f} (tol 0.02)")
    vb = mi_box_kernel(D["x"][2], D["x"][3], 0.25)
    vi = mi_box_kernel(Z[0], Z[1], 0.25)
    tb = D["true_mi_nats"][1] / LN2
    # iid tolerance 0.05 bits admits the O(1/(N q_joint)) positive bias of self-included counts
    q = np.round(3 * np.random.default_rng(4).standard_normal(400))  # integers: |dx| == w=1 is exact in FP
    dq = np.abs(q[:, None] - q[None])
    hin, hex_ = (-np.mean(np.log2(m.sum(1) / 800.0)) for m in (dq <= 1, dq < 1))
    hq = entropy_box_kernel(q, 1.0, normalise=False)
    check("(4) entropy_box_kernel integer data, w=1: |dx|==w pairs counted (inclusive, compare-to-all)",
          abs(hq - hin) < 1e-12 and abs(hq - hex_) > 0.1, f"{hq:.6f} bits; inclusive {hin:.6f}, strict {hex_:.6f}")
    check("(4) mi_box_kernel rho=0.8 within 0.15 bits of truth; iid |MI| < 0.05 bits", abs(vb - tb) < 0.15 and abs(vi) < 0.05,
          f"{vb:.4f} vs {tb:.4f} bits; iid {vi:.4f}")

    # (5) TE on var2_bivariate (x = process 0 drives y = process 1)
    V = make_data.load("var2_bivariate", T)
    x, y = V["x"][0], V["x"][1]
    txy, tyx = gaussian_te(x, y, k=2, l=2), gaussian_te(y, x, k=2, l=2)
    oxy, oyx = gaussian_te_ols(x, y, k=2, l=2), gaussian_te_ols(y, x, k=2, l=2)
    check("(5) gaussian_te k=l=2: TE(x->y)>0.05, TE(y->x)<0.01", txy > 0.05 and tyx < 0.01, f"{txy:.4f}, {tyx:.4f}")
    check("(5) OLS form == covariance form", max(abs(txy - oxy), abs(tyx - oyx)) < 1e-10, f"diff {max(abs(txy - oxy), abs(tyx - oyx)):.1e}")
    tt = _var2_te_truth(V["A1"], V["A2"], V["Sigma"], 0, 1)
    check("(5) gaussian_te(x->y) vs population VAR(2) truth", abs(txy - tt) < 0.02, f"{txy:.4f} vs {tt:.4f} (tol 0.02)")
    t1 = (gaussian_te(x, y), gaussian_te(y, x))
    print(f"INFO  gaussian_te k=l=1 (pyspi default): x->y {t1[0]:.4f}, y->x {t1[1]:.4f} (y->x non-zero: target under-embedded)")
    kxy, kyx = te_ksg(x, y, 2, 1, 2, 1), te_ksg(y, x, 2, 1, 2, 1)
    check("(5) te_ksg k=l=2 same ordering", kxy > 0.05 and kyx < 0.02 and kxy > kyx, f"{kxy:.4f}, {kyx:.4f}")
    bxy, byx = te_box_kernel(x, y, 1, 0.25), te_box_kernel(y, x, 1, 0.25)
    check("(5) te_box_kernel k=1 ordering", bxy > byx, f"{bxy:.4f} > {byx:.4f} bits")

    # (6) SI, CCE, DI on gauss_ar1_pair
    G = make_data.load("gauss_ar1_pair", T)
    gx, gy = G["x"][0], G["x"][1]
    SX, SXY, SE = G["Sigma_X"], G["Sigma_XY"], G["Sigma_E"]
    vv = [SX[i, i] - SXY[i, i] ** 2 / SX[i, i] for i in range(2)]
    si_true = 0.5 * math.log(vv[0] * vv[1] / np.linalg.det(SE))
    si = gaussian_si(gx, gy)
    check("(6) gaussian_si > 0 and within 0.02 of analytic", si > 0 and abs(si - si_true) < 0.02, f"{si:.4f} vs {si_true:.4f}")
    c1_true = 0.5 * math.log(2 * math.pi * math.e * (SX[1, 1] - SX[0, 1] ** 2 / SX[0, 0]))
    c1 = gaussian_cce(gx, gy, n=1)
    check("(6) gaussian_cce n=1 = H(Y_t|X_t) analytic", abs(c1 - c1_true) < 0.02, f"{c1:.4f} vs {c1_true:.4f}")
    dp, dm, B = gaussian_di(gx, gy, 5, "pyspi"), gaussian_di(gx, gy, 5, "massey"), di_block_term(gy, 5)
    check("(6) DI pyspi - massey == block term B", abs(dp - dm - B) < 1e-10, f"pyspi {dp:.4f}, massey {dm:.4f}, B {B:.4f}")
    check("(6) block term B > 0 for autocorrelated Y; ~0 for white Y", B > 0.05 and abs(di_block_term(Z[1], 5)) < 0.01,
          f"B_ar1 {B:.4f}, B_white {di_block_term(Z[1], 5):.4f}")

    # (7) embedding alignment on a ramp
    r = np.arange(20.0)
    E = embed(r, 3, 2)
    check("(7) embed(ramp,3,2): first row [4,2,0], T-(k-1)tau rows", E[0].tolist() == [4, 2, 0] and E.shape == (16, 3), f"{E[0].tolist()} {E.shape}")
    check("(7) embed(ramp,0) = T+1 empty rows (JIDT k=0)", embed(r, 0).shape == (21, 0), f"{embed(r, 0).shape}")
    sp, dn, dpst = te_embedding(r, r + 100, k=2, k_tau=1, l=3, l_tau=2, delay=1)
    ok = dn[0, 0] == 105 and dpst[0].tolist() == [104, 103] and sp[0].tolist() == [4, 2, 0] and dn.shape[0] == 15
    check("(7) TE embedding s=max((k-1)k_tau,(l-1)l_tau+delay-1)=4", ok, f"next {dn[0,0]}, past {dpst[0].tolist()}, src {sp[0].tolist()}, N {dn.shape[0]}")
    Yf, Yp, Xp = cce_embedding(r, r + 100, 3)
    check("(7) CCE term 3 alignment", Yf[0, 0] == 102 and Yp[0].tolist() == [101, 100] and Xp[0].tolist() == [2, 1, 0] and len(Yf) == len(Xp) == 18,
          f"Yf {Yf[0,0]}, Yp {Yp[0].tolist()}, Xp {Xp[0].tolist()}")
    Y1, X1 = xme_embedding(r, r + 100, 1)
    Y2, X2 = xme_embedding(r, r + 100, 2)
    Y4, X4 = xme_embedding(r, r + 100, 4)
    check("(7) xme lags: k=1 and k=2 both lag 1; k=4 uses lags 1..3", X1[0, 0] == Y1[0, 0] - 101 and X2.shape[1] == 1 and (Y4[0, 0] - 100 - X4[0]).tolist() == [1, 2, 3],
          f"k=4 lags {(Y4[0, 0] - 100 - X4[0]).tolist()}")
    tl2 = gaussian_tlmi(r ** 1.5, np.sin(r), lag=2) == gaussian_mi((r ** 1.5)[:-2], np.sin(r)[2:])
    try:
        gaussian_tlmi(gx, gy, lag=0)
        raised = False
    except ValueError as exc:
        raised = "lag >= 1" in str(exc)
    check("(7) gaussian_tlmi lag=2 alignment x[:-2], y[2:]; lag=0 rejected with clear error", tl2 and raised, f"lag2 {tl2}, lag0 raised {raised}")
    xa, xb = gaussian_xme(gx[1:], gy[1:], 1), gaussian_xme(gx, gy, 2)
    check("(7) gaussian_xme k=2 == k=1 minus first sample (pyspi off-by-one)", abs(xa - xb) < 1e-12, f"{xb:.6f} vs {xa:.6f}")

    # (8) Theiler windows
    pr, bt = theiler_window_pyspi(x, y, raw=True), theiler_window_bartlett(x, y)
    check("(8) pyspi raw window == Bartlett + 1 (lag-0 term double counted)", abs(pr - bt - 1.0) < 1e-9, f"pyspi {pr:.3f} -> int {int(pr)}, bartlett {bt:.3f}")
    rng = np.random.default_rng(7)
    e = rng.standard_normal(T + 500)
    ar = np.zeros_like(e)
    for t in range(1, e.size):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    ar = ar[500:]
    bta = theiler_window_bartlett(ar, ar, max_lag=50)
    check("(8) Bartlett factor AR(1) phi=0.5 self-pair = (1+phi^2)/(1-phi^2)", abs(bta - 5 / 3) < 0.15, f"{bta:.3f} vs {5 / 3:.3f} (tol 0.15)")
    return all(results)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = selftest()
        print("ALL PASS" if ok else "SOME CHECKS FAILED")
        sys.exit(0 if ok else 1)
    print(__doc__)
