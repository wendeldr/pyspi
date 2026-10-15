"""References for the cov_prec_robust family (COMPARISON venv; never imports sklearn or pyspi).

Units: basic.cov.MinCovDet, basic.cov.EllipticEnvelope, basic.prec.MinCovDet,
basic.prec.EllipticEnvelope. pyspi fits sklearn.covariance.MinCovDet / EllipticEnvelope with all
defaults on the (T, P) array it holds (pyspi/statistics/basic.py:31-44) and returns covariance_
or precision_ (basic.py:46-54). One handler per run_id tag ("<unit>|<dataset>|<T>|<path>|<tag>"):

state      same-lineage  from-scratch correction + reweighting (Croux-Haesbroeck factors, chi2_{p,.975}
                         hard rejection, MLE divisor) applied to sklearn's raw FAST-MCD estimates, which
                         are the only sklearn-derived input (probe fit_state); compares the whole fitted
                         state (support_, location_, covariance_, precision_, dist_, SPI output).
base       same-lineage  the plain SPI output vs the same from-scratch reweighting of its companion state.
rdet       external-pkg  robustbase::covMcd(nsamp="deterministic", initHsets = sklearn raw_support_)
                         (r/cov_prec_robust.R), its reweighted cov mapped to sklearn's conventions
                         (divisor n_w - 1 -> n_w; c(p,.975) also when every point is kept).
rrcov      external-pkg  rrcov::CovMcd with its own random FAST-MCD search (20 R seeds), h matched.
                         Tolerance = (sklearn 20-seed range) + (rrcov 20-seed range): two randomised
                         searches for the same h-subset optimum can differ by at most what each one's
                         own run-to-run variability allows when both target the same estimator.
                         Wide by construction: an agreement here is weak evidence, not value-level
                         agreement; within_sklearn_range_only records the stricter outcome with the
                         sklearn 20-seed range alone as the elementwise bound.
seeds      same-lineage  proposition "output does not depend on the global RNG seed" (spread <= 1e-12);
                         pyspi-only self-consistency (numpy summary of pyspi's own outputs).
range      qualitative   proposition "every off-diagonal value lies in [-1, 1]" (supplement 1.1.1/1.1.2).
eeqmcd     same-lineage  EllipticEnvelope SPI output == MinCovDet SPI output under the same seed, exact.
freshfull  same-lineage  (pyspi self-consistency) proposition "two fresh-process Calculator runs (default config, basic block) are identical".
freshbasic same-lineage  (pyspi self-consistency) the same with a config holding only the 8 robust SPIs (pyspi.statistics.misc not imported).
alias      same-lineage  (pyspi self-consistency) proposition "the returned matrix does not alias the cached sklearn attribute".
det        same-lineage  (pyspi self-consistency) proposition "two calls under harness.seed_all(42) are bit-identical".
rngstream  same-lineage  (pyspi self-consistency) proposition "output unchanged when another global-RNG SPI ran first".
statc      independent   null datasets: MCD output vs the MLE estimate of the same draw, SE = MC SD of the difference.
stat       analytic      target in the units of the path (z-units: Sigma_ij/(s_i s_j), Omega_ij s_i s_j with
                         s the ddof=1 SD pyspi divided by), SE from a Monte-Carlo of the same design run
                         through this file's from-scratch reweighted FAST-MCD; Bonferroni over the
                         P(P-1)/2 distinct entries (se scaled so that 2*se is the family-wise 5% bound).
pcorr      analytic      proposition "prec_* equals the partial correlation" (supplement 1.1.2).
rcorr      analytic      robust correlation C_ij/sqrt(C_ii C_jj) of the fit vs the clean correlation.
breakdown  qualitative   proposition "|pyspi - clean truth| < |EmpiricalCovariance - clean truth|".
singular   qualitative   proposition "pyspi signals singular input (NaN or error)".
sq         same-lineage  -sq identifier == square of the unsquared output (same seed), exact.
perm       same-lineage  processes reversed -> output is the permuted matrix (orientation).
dir        analytic      sign of the analytic target where |corr| or |partial corr| >= 0.2.
opt        independent   optimality: log det of sklearn's raw h-subset MLE covariance vs the best of 2000
                         from-scratch C-step runs (1000 random h-subsets, 1000 random (p+1)-subsets).

Run by run_checks.py; writes out/ref/<run_id>.npz.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal, stats

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))
import make_data  # noqa: E402

FAMILY = "cov_prec_robust"
OUTP = CHECKS / "out" / "pyspi"
OUTR = CHECKS / "out" / "ref"
EXCH = CHECKS / "out" / "exchange" / FAMILY
R_SCRIPT = CHECKS / "r" / f"{FAMILY}.R"
NSEEDS_R = 20
SELF = "refs/cov_prec_robust.py"


# ------------------------------------------------------------------ from-scratch MCD pieces
def cfactor(p, alpha):
    """Croux & Haesbroeck (1999) Eq. 4.2 consistency factor c(p, alpha) = alpha / F_{chi2_{p+2}}(chi2_{p,alpha})."""
    return alpha / stats.chi2.cdf(stats.chi2.ppf(alpha, p), p + 2)


def h_sklearn(n, p):
    return min(int(np.ceil(0.5 * (n + p + 1))), n)


def mle_cov(X):
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / X.shape[0]


def mahal2(X, loc, C):
    D = X - loc
    return np.einsum("ij,ij->i", D, np.linalg.solve(C, D.T).T)


def reweight(X, raw_loc, raw_cov, h):
    """Raw distances / c(p, h/n); keep d2 < chi2_{p}.isf(0.025); Sigma = c(p, .975) * MLE cov of kept rows."""
    n, p = X.shape
    d2 = mahal2(X, raw_loc, raw_cov) / cfactor(p, h / n)
    mask = d2 < stats.chi2.isf(0.025, p)
    loc = X[mask].mean(axis=0)
    C = mle_cov(X[mask]) * cfactor(p, 0.975)
    P = np.linalg.inv(C)
    D = X - loc
    dist = np.einsum("ij,ij->i", D @ P, D)
    return dict(mask=mask, d2raw=d2, loc=loc, cov=C, prec=P, dist=dist)


def csteps(X, idx, h, max_iter=200):
    """Concentration steps (Rousseeuw & Van Driessen 1999, Theorem 1) to a fixed point."""
    idx = np.sort(idx)
    ld = np.inf
    for _ in range(max_iter):
        sub = X[idx]
        loc = sub.mean(axis=0)
        C = mle_cov(sub)
        s, ld = np.linalg.slogdet(C)
        if s <= 0:
            return -np.inf, idx, loc, C
        new = np.sort(np.argpartition(mahal2(X, loc, C), h - 1)[:h])
        if np.array_equal(new, idx):
            break
        idx = new
    return ld, idx, loc, C


def start_h(X, h, rng):
    return rng.permutation(X.shape[0])[:h]


def start_p1(X, h, rng):
    n, p = X.shape
    J = rng.choice(n, p + 1, replace=False)
    while True:
        C = mle_cov(X[J])
        if np.linalg.matrix_rank(C) == p or len(J) >= n:
            break
        J = np.append(J, rng.choice(np.setdiff1d(np.arange(n), J), 1))
    return np.argpartition(mahal2(X, X[J].mean(axis=0), C), h - 1)[:h]


def search(X, h, rng, n_h, n_p1):
    res = []
    for k in range(n_h + n_p1):
        idx = start_h(X, h, rng) if k < n_h else start_p1(X, h, rng)
        res.append(csteps(X, idx, h))
    return res


def scratch_rmcd(X, rng, n_starts=10):
    n, p = X.shape
    h = h_sklearn(n, p)
    best = min(search(X, h, rng, n_starts, 0), key=lambda r: r[0])
    return reweight(X, best[2], best[3], h)


# ------------------------------------------------------------------ datasets / designs
def truth(ds, T):
    return make_data.load(ds, T)


def preprocess(x, path):
    if path == "kernel":
        return x, np.ones(x.shape[0])
    if path == "pipeline-detrend":
        x = signal.detrend(x, axis=1)
    s = x.std(axis=1, ddof=1)
    return stats.zscore(x, axis=1, ddof=1), s


def sigma_of(ds, T):
    d = truth(ds, T)
    if ds in ("bivariate_gaussian", "sparse_precision"):
        return np.asarray(d["cov_true"], float)
    if ds == "trending":
        return np.asarray(d["cov_true_detrended"], float)
    if ds == "contaminated":
        return np.asarray(d["clean_cov_true"], float)
    if ds == "iid_gaussian":
        return np.eye(5)
    if ds == "delayed_noise":
        return np.diag([1.0] + [1.0 + make_data.D6_NOISE ** 2] * 3)
    raise KeyError(ds)


def simulate(ds, T, rng):
    """A fresh draw of the dataset's generating design (constants read from make_data / the truth arrays)."""
    if ds in ("bivariate_gaussian", "trending", "contaminated"):
        z = rng.standard_normal((4, T))
        x = np.empty_like(z)
        for k, rho in enumerate(make_data.D2_RHO):
            x[2 * k] = z[2 * k]
            x[2 * k + 1] = rho * z[2 * k] + np.sqrt(1 - rho ** 2) * z[2 * k + 1]
        if ds == "trending":
            return x + np.asarray(make_data.D4_SLOPE, float)[:, None] * np.arange(T)[None, :]
        if ds == "contaminated":
            x = x[2:4].copy()
            n_out = int(round(make_data.D10_FRAC * T))
            idx = rng.choice(T, size=n_out, replace=False)
            x[:, idx] = np.asarray(make_data.D10_MEAN, float)[:, None] + make_data.D10_STD * rng.standard_normal((2, n_out))
        return x
    if ds == "sparse_precision":
        return np.linalg.cholesky(sigma_of(ds, T)) @ rng.standard_normal((5, T))
    if ds == "iid_gaussian":
        return rng.standard_normal((5, T))
    if ds == "delayed_noise":
        d = make_data.D6_DELAYS
        m = max(d)
        base = rng.standard_normal(T + m)
        x = np.empty((len(d), T))
        x[0] = base[m:]
        for k, dk in enumerate(d[1:], start=1):
            x[k] = base[m - dk:m - dk + T] + make_data.D6_NOISE * rng.standard_normal(T)
        return x
    raise KeyError(ds)


def targets(Sigma, s):
    """Analytic targets in the units of the preprocessed data (s = SD pyspi divided by; 1 on kernel)."""
    Om = np.linalg.inv(Sigma)
    d = np.sqrt(np.diag(Sigma))
    dp = np.sqrt(np.diag(Om))
    return dict(cov=Sigma / np.outer(s, s), prec=Om * np.outer(s, s), rcorr=Sigma / np.outer(d, d),
                pcorr=-Om / np.outer(dp, dp))


def mc(ds, T, path):
    """Monte-Carlo of the from-scratch reweighted FAST-MCD on fresh draws of the design (cached)."""
    reps = 200 if T <= 1024 else 100
    f = EXCH / f"mc2_{ds}_{T}_{path}_{reps}.npz"
    if f.is_file():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    rng = np.random.default_rng(int(hashlib.sha1(f"{ds}|{T}|{path}".encode()).hexdigest()[:8], 16))
    Sigma = sigma_of(ds, T)
    out = {k: [] for k in ("cov", "prec", "rcorr", "s", "emp", "empprec")}
    for _ in range(reps):
        x, s = preprocess(simulate(ds, T, rng), path)
        r = scratch_rmcd(x.T, rng)
        C = r["cov"]
        d = np.sqrt(np.diag(C))
        out["cov"].append(C); out["prec"].append(r["prec"]); out["rcorr"].append(C / np.outer(d, d)); out["s"].append(s)
        E = mle_cov(x.T)
        out["emp"].append(E); out["empprec"].append(np.linalg.inv(E))
    res = {k: np.array(v) for k, v in out.items()}
    res["Sigma"] = Sigma
    EXCH.mkdir(parents=True, exist_ok=True)
    np.savez(f, **res)
    return res


def mc_error_stats(ds, T, path, q):
    """Per-entry SD and mean of (estimate - target) over the Monte-Carlo replicates."""
    m = mc(ds, T, path)
    errs = []
    for r in range(m["cov"].shape[0]):
        t = targets(m["Sigma"], m["s"][r])
        errs.append(m[q][r] - t[q])
    errs = np.array(errs)
    return errs.std(axis=0, ddof=1), errs.mean(axis=0), errs.shape[0]


def bonferroni_scale(P):
    m = P * (P - 1) // 2
    return stats.norm.ppf(1 - 0.05 / (2 * m)) / 2.0, m


# ------------------------------------------------------------------ pyspi side readers
def load_pyspi(rid):
    f = OUTP / f"{rid}.npz"
    if not f.is_file():
        return None
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def decode_state(v):
    T, P = int(v[0]), int(v[1])
    o = 4
    out = dict(T=T, P=P, h=int(v[2]), n_rew=int(v[3]))
    for name, size, shape in (("raw_support", T, (T,)), ("support", T, (T,)), ("raw_loc", P, (P,)),
                              ("raw_cov", P * P, (P, P)), ("loc", P, (P,)), ("cov", P * P, (P, P)),
                              ("prec", P * P, (P, P)), ("dist", T, (T,)), ("z", P * T, (P, T)),
                              ("S", P * P, (P, P))):
        out[name] = v[o:o + size].reshape(shape)
        o += size
    assert o == v.size, (o, v.size)
    out["raw_support"] = out["raw_support"].astype(bool)
    out["support"] = out["support"].astype(bool)
    return out


def encode_like(st, rw, kind):
    S = (rw["cov"] if kind == "cov" else rw["prec"]).copy()
    np.fill_diagonal(S, np.nan)
    parts = [np.array([st["T"], st["P"], st["h"], rw["mask"].sum()], float), st["raw_support"].astype(float),
             rw["mask"].astype(float), st["raw_loc"], st["raw_cov"].ravel(), rw["loc"], rw["cov"].ravel(),
             rw["prec"].ravel(), rw["dist"], st["z"].ravel(), S.ravel()]
    return np.concatenate([np.asarray(p, float).ravel() for p in parts])


def nandiag(M):
    M = np.array(M, float, copy=True)
    np.fill_diagonal(M, np.nan)
    return M


def offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def parse(rid):
    unit, ds, T, path, tag = rid.split("|")
    _, kind, est = unit.split(".")
    return dict(unit=unit, ds=ds, T=int(T), path=path, tag=tag, kind=kind, est=est)


def sym(M):
    M = np.asarray(M, float)
    m = ~np.isnan(M) & ~np.isnan(M.T)
    return float(np.abs(M - M.T)[m].max()) if m.any() else 0.0


# ------------------------------------------------------------------ R
def r_call(X, mode, h, init=None):
    EXCH.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(X.tobytes() + f"{mode}|{h}|{NSEEDS_R}".encode()
                       + (b"" if init is None else np.asarray(init, np.int64).tobytes())).hexdigest()[:16]
    fin, fout = EXCH / f"r_{key}_in.csv", EXCH / f"r_{key}_out.csv"
    if not fout.is_file():
        np.savetxt(fin, X, delimiter=",", fmt="%.17g")
        args = ["Rscript", str(R_SCRIPT), str(fin), str(fout), mode, str(h)]
        if mode == "det":
            finit = EXCH / f"r_{key}_init.csv"
            np.savetxt(finit, np.asarray(init, np.int64), fmt="%d")
            args.append(str(finit))
        else:
            args.append(str(NSEEDS_R))
        env = dict(os.environ, R_LIBS_USER=os.environ.get("R_LIBS_USER", "/home/dan/R/audit-lib"))
        subprocess.run(args, check=True, env=env, capture_output=True, text=True)
    import csv
    acc = {}
    with open(fout) as fh:
        for row in csv.DictReader(fh):
            acc.setdefault((row["key"], int(row["k"])), []).append((int(row["i"]), int(row["j"]), float(row["value"])))
    out = {}
    for (k, s), cells in acc.items():
        ni = max(c[0] for c in cells); nj = max(c[1] for c in cells)
        M = np.full((ni, nj), np.nan)
        for i, j, v in cells:
            M[i - 1, j - 1] = v
        out[(k, s)] = M if nj > 1 else M[:, 0] if ni > 1 else M[0, 0]
    return out


def r_mapped(cov_r, sum_w, n, p):
    """robustbase reweighted cov -> sklearn convention: divisor sum_w-1 -> sum_w; c(p,.975) applied even if sum_w == n."""
    C = np.asarray(cov_r, float) * (sum_w - 1) / sum_w
    if sum_w == n:
        C = C * cfactor(p, 0.975)
    return C


# ------------------------------------------------------------------ handlers
def h_state(rid, run, ps):
    st = decode_state(ps["value"])
    X = st["z"].T
    rw = reweight(X, st["raw_loc"], st["raw_cov"], st["h"])
    ref = encode_like(st, rw, run["kind"])
    # raw support is a C-step fixed point, and raw_covariance_ is the MLE of the raw support
    fixed = np.array_equal(np.sort(np.argpartition(mahal2(X, st["raw_loc"], st["raw_cov"]), st["h"] - 1)[:st["h"]]),
                           np.where(st["raw_support"])[0])
    rawmle = float(np.abs(mle_cov(X[st["raw_support"]]) - st["raw_cov"]).max())
    I = np.eye(st["P"])
    return dict(value=ref, criterion="tolerance", rtol=1e-9, atol=1e-10, reference_kind="same-lineage",
                reference_detail=f"{SELF}:reweight (Croux-Haesbroeck 1999 Eq. 4.2 factors, chi2_(p,0.975) rejection, MLE "
                                 "divisor, numpy.linalg.inv) on sklearn raw_location_/raw_covariance_",
                notes="whole fitted state compared; sklearn raw estimates are the only shared input",
                n_reweighted_pyspi=st["n_rew"], n_reweighted_ref=int(rw["mask"].sum()), h=st["h"],
                raw_support_cstep_fixed_point=int(fixed), raw_cov_vs_support_mle_max_abs=rawmle,
                sym_cov=sym(st["cov"]), sym_prec=sym(st["prec"]),
                inverse_residual=float(np.abs(st["prec"] @ st["cov"] - I).max()),
                cond_cov=float(np.linalg.cond(st["cov"])),
                diag_cov_min=float(np.diag(st["cov"]).min()), diag_cov_max=float(np.diag(st["cov"]).max()))


def companion(run, tag, unit=None, path=None):
    rid = f"{unit or run['unit']}|{run['ds']}|{run['T']}|{path or run['path']}|{tag}"
    return rid, load_pyspi(rid)


def h_base(rid, run, ps):
    crid, c = companion(run, "state")
    st = decode_state(c["value"])
    rw = reweight(st["z"].T, st["raw_loc"], st["raw_cov"], st["h"])
    M = nandiag(rw["cov"] if run["kind"] == "cov" else rw["prec"])
    return dict(value=M, criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"{SELF}:reweight on the raw estimates of {crid}",
                notes="SPI output vs from-scratch reweighting of the companion fit_state probe (same seed, same fit)",
                spi_equals_state_output=int(np.array_equal(ps["value"], st["S"], equal_nan=True)),
                sym_observed=sym(ps["value"]))


def h_rdet(rid, run, ps):
    crid, c = companion(run, "state")
    st = decode_state(c["value"])
    X = np.asarray(ps["input"], float).T
    n, p = X.shape
    init = np.where(st["raw_support"])[0] + 1
    r = r_call(X, "det", st["h"], init)
    sw = float(r[("sum_w", 0)])
    C = r_mapped(r[("cov", 0)], sw, n, p)
    M = nandiag(C if run["kind"] == "cov" else np.linalg.inv(C))
    raw_expected = st["raw_cov"] * cfactor(p, st["h"] / n) * st["h"] / (st["h"] - 1)
    rw = np.asarray(r[("raw_weights", 0)]).astype(bool)
    rwm = reweight(X, st["raw_loc"], st["raw_cov"], st["h"])
    mine = rwm["mask"]
    q = stats.chi2.isf(0.025, p)
    dis = rw != st["support"]
    band = (rwm["d2raw"] >= q) & (rwm["d2raw"] < q * st["h"] / (st["h"] - 1))
    return dict(value=M, criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="external-pkg",
                reference_detail="r/cov_prec_robust.R det: robustbase==0.99.2 covMcd(nsamp='deterministic', initHsets="
                                 "sklearn raw_support_, alpha s.t. h.alpha.n == sklearn h, use.correction=FALSE); cov * "
                                 "(n_w-1)/n_w; precision = numpy.linalg.inv",
                notes="robustbase raw.cov = c(p,h/n) * cov(H) (divisor h-1); reweighted = cov.wt (divisor n_w-1) * "
                      "c(p,.975) only if n_w != n. sklearn raw_covariance_ = MLE of H, uncorrected (factor only on dist_); "
                      "covariance_ = MLE (divisor n_w) * c(p,.975) always",
                r_best_equals_init=int(r[("best_equals_init", 0)]), r_n_csteps=int(r[("n_csteps", 0)]),
                r_sum_w=sw, sklearn_n_reweighted=st["n_rew"],
                r_raw_weights_equal_sklearn_support=int(np.array_equal(rw, st["support"])),
                r_raw_weights_equal_scratch_mask=int(np.array_equal(rw, mine)),
                n_mask_disagreements=int(dis.sum()), n_points_in_divisor_band=int(band.sum()),
                disagreements_all_in_divisor_band=int(bool(np.all(band[dis]))),
                r_raw_cov_vs_sklearn_raw_conv_max_rel=float(np.abs(r[("raw_cov", 0)] / raw_expected - 1).max()),
                sym_observed=sym(ps["value"]))


def seed_stack(arr, P):
    out = arr[:, :P * P].reshape(-1, P, P)
    return out, arr[:, P * P], arr[:, P * P + 1]


def h_rrcov(rid, run, ps):
    X = np.asarray(ps["input"], float).T
    n, p = X.shape
    h = h_sklearn(n, p)
    r = r_call(X, "own", h)
    mats = []
    for k in range(1, NSEEDS_R + 1):
        C = r_mapped(r[("cov", k)], float(r[("sum_w", k)]), n, p)
        mats.append(C if run["kind"] == "cov" else np.linalg.inv(C))
    mats = np.array(mats)
    crid, c = companion(run, "seeds")
    sk, _, _ = seed_stack(c["value"], p)
    off = ~np.eye(p, dtype=bool)
    rng_sk = (np.nanmax(sk, 0) - np.nanmin(sk, 0))
    rng_r = mats.max(0) - mats.min(0)
    tol = rng_sk + rng_r
    V = nandiag(mats[0])
    diff = np.abs(ps["value"] - V)
    r_ld = np.array([r_raw_logdet(r, k, h) for k in range(1, NSEEDS_R + 1)])
    _, sk_ld, _ = seed_stack(c["value"], p)
    srid, sc = companion(run, "state")
    sk42 = float(np.linalg.slogdet(decode_state(sc["value"])["raw_cov"])[1]) if sc is not None else np.nan
    dC = r_mapped(r[("default_cov", 0)], float(r[("default_sum_w", 0)]), n, p)
    dV = dC if run["kind"] == "cov" else np.linalg.inv(dC)
    return dict(value=V, criterion="tolerance", rtol=0.0, atol=float(tol[off].max()) + 1e-12,
                reference_kind="external-pkg",
                reference_detail="r/cov_prec_robust.R own: rrcov==1.7.5 CovMcd (robustbase 0.99.2 FAST-MCD, nsamp=500, "
                                 "alpha s.t. h == sklearn h, use.correction=FALSE), set.seed(1); cov*(n_w-1)/n_w",
                notes=f"atol = max_ij(sklearn 20-seed range [{crid}] + rrcov 20-seed range); value = rrcov seed 1",
                sklearn_seed_range_max=float(rng_sk[off].max()), rrcov_seed_range_max=float(rng_r[off].max()),
                elementwise_within=int(bool((diff[off] <= tol[off] + 1e-12).all())),
                within_sklearn_range_only=int(bool((diff[off] <= rng_sk[off] + 1e-12).all())),
                n_offdiag_outside_sklearn_range=int((diff[off] > rng_sk[off] + 1e-12).sum()),
                max_abs_diff=float(diff[off].max()),
                rrcov_median_vs_pyspi_max_abs=float(np.abs(np.median(mats, 0) - ps["value"])[off].max()),
                rrcov_default_h=int(r[("default_h", 0)]),
                rrcov_default_conventions_max_abs_diff=float(np.abs(dV - ps["value"])[off].max()),
                rrcov_default_cnp2=float(np.prod(np.atleast_1d(r[("default_cnp2", 0)]))),
                rrcov_sum_w_seed1=float(r[("sum_w", 1)]),
                rrcov_raw_logdet_min=float(r_ld.min()), rrcov_raw_logdet_max=float(r_ld.max()),
                sklearn_seeds_raw_logdet_min=float(sk_ld.min()), sklearn_seeds_raw_logdet_max=float(sk_ld.max()),
                sklearn_seed42_raw_logdet=sk42)


def r_raw_logdet(r, k, h):
    """log det of the MLE covariance of rrcov's raw h-subset: raw.cov / (c(p,h/n) * small-sample) * (h-1)/h."""
    cn = np.atleast_1d(r[("raw_cnp2", k)])
    C = np.asarray(r[("raw_cov", k)], float) / float(np.prod(cn)) * (h - 1) / h
    return float(np.linalg.slogdet(C)[1])


def h_seeds(rid, run, ps):
    arr = ps["value"]
    P = int(round(np.sqrt(arr.shape[1] - 2)))
    out, ld, hh = seed_stack(arr, P)
    off = ~np.eye(P, dtype=bool)
    spread = (np.nanmax(out, 0) - np.nanmin(out, 0))[off]
    nd = len(np.unique(np.round(ld, 9)))
    common = dict(max_spread=float(spread.max()), median_spread=float(np.median(spread)),
                  n_distinct_raw_logdet=nd, n_seeds=int(arr.shape[0]),
                  max_abs_offdiag=float(np.nanmax(np.abs(out))), min_offdiag=float(np.nanmin(out)),
                  max_offdiag=float(np.nanmax(out)), sym_observed=max(sym(o) for o in out))
    if run["tag"] == "seeds":
        return dict(value=spread, criterion="qualitative", pass_flag=int(spread.max() <= 1e-12),
                    reference_kind="same-lineage", reference_detail=f"{SELF}:h_seeds (pyspi self-consistency: numpy summary of pyspi's own outputs under 20 np.random.seed draws)",
                    notes="proposition: output does not depend on the global RNG seed (max off-diagonal spread <= 1e-12)",
                    **common)
    frac = float((np.abs(out[:, off]) > 1).any(axis=1).mean())
    return dict(value=np.abs(out[:, off]).max(axis=1), criterion="qualitative",
                pass_flag=int(np.nanmax(np.abs(out)) <= 1.0), reference_kind="literature-only",
                reference_detail="supplement 1.1.1 (cov 'within [-1, 1]') / 1.1.2 (prec 'partial correlation ... within [-1, 1]')",
                notes="proposition: every off-diagonal value of every one of 20 seeds lies in [-1, 1] as documented",
                frac_seeds_exceeding_1=frac, **common)


def h_eeqmcd(rid, run, ps):
    crid, c = companion(run, "base", unit=f"basic.{run['kind']}.MinCovDet")
    return dict(value=c["value"], criterion="exact", rtol=0.0, atol=0.0, reference_kind="same-lineage",
                reference_detail=f"pyspi {run['kind']}_MinCovDet output {crid} (sklearn MinCovDet under seed_all(42))",
                notes="EllipticEnvelope.fit = MinCovDet.fit + offset_ (sklearn/covariance/_elliptic_envelope.py:183-184)",
                bitwise_equal=int(np.array_equal(ps["value"], c["value"], equal_nan=True)))


def h_det(rid, run, ps):
    A = ps["value"]
    if run["tag"] == "det":
        ok = np.array_equal(A[0], A[1], equal_nan=True)
        return dict(value=np.nanmax(np.abs(A[1] - A[0])), criterion="qualitative", pass_flag=int(ok),
                    reference_kind="same-lineage", reference_detail=f"{SELF}:h_det (pyspi self-consistency: numpy array_equal of pyspi's own outputs)",
                    notes="proposition: two calls each preceded by harness.seed_all(42) give bit-identical output",
                    max_abs_call2_minus_call1=float(np.nanmax(np.abs(A[1] - A[0]))))
    ok = np.array_equal(A[0], A[2], equal_nan=True)
    return dict(value=np.nanmax(np.abs(A[2] - A[0])), criterion="qualitative", pass_flag=int(ok),
                reference_kind="same-lineage", reference_detail=f"{SELF}:h_det (pyspi self-consistency: numpy array_equal of pyspi's own outputs)",
                notes="proposition: output unchanged when the other robust estimator's SPI consumed the global RNG first "
                      "(seed_all(42) once, as in one Calculator pass)",
                max_abs_after_other_spi=float(np.nanmax(np.abs(A[2] - A[0]))),
                max_rel_after_other_spi=float(np.nanmax(np.abs(A[2] - A[0]) / np.abs(A[0]))))


def h_fresh(rid, run, ps):
    A = ps["value"]
    ok = np.array_equal(A[0], A[1], equal_nan=True)
    full = run["tag"] == "freshfull"
    return dict(value=np.nanmax(np.abs(A[1] - A[0])), criterion="qualitative", pass_flag=int(ok),
                reference_kind="same-lineage", reference_detail=f"{SELF}:h_fresh (pyspi self-consistency: numpy array_equal of two fresh-process pyspi outputs)",
                notes=("proposition: two fresh-process Calculator runs give bit-identical output. "
                       + ("Default config.yaml (imports pyspi.statistics.misc -> np.random.seed(1717)), basic block computed"
                          if full else "config with only the 8 robust SPIs (misc never imported; global RNG seeded from OS entropy)")),
                max_abs_process2_minus_process1=float(np.nanmax(np.abs(A[1] - A[0]))))


def h_alias(rid, run, ps):
    f = ps["value"].astype(bool)
    names = ["returned_is_cached_attribute", "cached_diag_all_nan_after_call", "other_attribute_diag_finite",
             "write_through_to_cache", "next_call_returns_edited_value"]
    return dict(value=ps["value"], criterion="qualitative", pass_flag=int(not f[0] and not f[3]),
                reference_kind="same-lineage", reference_detail=f"{SELF}:h_alias (pyspi self-consistency: flags from probe aliasing)",
                notes="proposition: the returned matrix is independent of the cached sklearn fit (no aliasing); "
                      "Calculator copies the matrix into its DataFrame, so a fail here is value-neutral there",
                **{n: int(v) for n, v in zip(names, f)})


def path_scale(run):
    x = np.asarray(truth(run["ds"], run["T"])["x"], float)
    _, s = preprocess(x, run["path"])
    return s


def h_stat(rid, run, ps):
    s = path_scale(run)
    q = run["kind"]
    t = targets(sigma_of(run["ds"], run["T"]), s)
    se, bias, reps = mc_error_stats(run["ds"], run["T"], run["path"], q)
    P = t[q].shape[0]
    k, m = bonferroni_scale(P)
    off = ~np.eye(P, dtype=bool)
    V = nandiag(t[q])
    return dict(value=V, criterion="statistical", se=se * k, reference_kind="analytic",
                reference_detail=f"{run['ds']} truth ({'Sigma/(s_i s_j)' if q == 'cov' else 'Omega*s_i s_j'}, s = ddof=1 SD "
                                 f"on path {run['path']}); SE: {SELF}:mc ({reps} draws, from-scratch reweighted FAST-MCD)",
                notes=f"se column = MC SD x {k:.3f} (Bonferroni over {m} distinct entries, family-wise 5%)",
                mc_reps=reps, mc_se_max=float(se[off].max()), mc_bias_max_abs=float(np.abs(bias[off]).max()),
                mc_bias_over_se_max=float(np.abs(bias[off] / se[off]).max()),
                max_abs_vs_target=float(np.nanmax(np.abs(ps["value"] - V))),
                max_z_raw=float(np.nanmax(np.abs(ps["value"] - V)[off] / se[off])),
                target_offdiag_max_abs=float(np.abs(t[q][off]).max()), sym_observed=sym(ps["value"]))


def h_statc(rid, run, ps):
    """Conditional null check: MCD estimate vs the empirical (MLE) estimate of the SAME draw, SE = MC SD of the
    difference (removes the draw's own sampling error, which both estimators share)."""
    q = run["kind"]
    m = mc(run["ds"], run["T"], run["path"])
    diff = m[q] - (m["emp"] if q == "cov" else m["empprec"])
    se = diff.std(axis=0, ddof=1)
    X = np.asarray(ps["input"], float).T
    E = mle_cov(X)
    V = nandiag(E if q == "cov" else np.linalg.inv(E))
    P = X.shape[1]
    k, mm = bonferroni_scale(P)
    off = ~np.eye(P, dtype=bool)
    return dict(value=V, criterion="statistical", se=se * k, reference_kind="independent-impl",
                reference_detail=f"numpy MLE {'covariance' if q == 'cov' else 'precision'} of pyspi's input (same draw); SE: "
                                 f"{SELF}:mc SD of (from-scratch reweighted MCD - MLE) over {diff.shape[0]} draws",
                notes=f"conditional on the draw; se column = MC SD x {k:.3f} (Bonferroni over {mm} entries)",
                mc_diff_se_max=float(se[off].max()), mc_diff_mean_max_abs=float(np.abs(diff.mean(0))[off].max()),
                empirical_offdiag_max_abs=float(np.abs(V[off]).max()),
                max_z_raw=float(np.nanmax(np.abs(ps["value"] - V)[off] / se[off])))


def h_pcorr(rid, run, ps):
    s = path_scale(run)
    t = targets(sigma_of(run["ds"], run["T"]), s)
    se, _, reps = mc_error_stats(run["ds"], run["T"], run["path"], "prec")
    P = s.size
    k, m = bonferroni_scale(P)
    V = nandiag(t["pcorr"])
    i, j = (2, 3) if run["ds"] == "bivariate_gaussian" else (0, 1)
    return dict(value=V, criterion="statistical", se=se * k, reference_kind="analytic",
                reference_detail="supplement 1.1.2: precision 'equivalent to the partial correlation'; partial correlation "
                                 "-Omega_ij/sqrt(Omega_ii Omega_jj) of the truth (Whittaker 1990 ch. 5)",
                notes="proposition: prec_* output equals the documented partial correlation; se = SD of the prec estimate",
                pyspi_entry=float(ps["value"][i, j]), partial_corr_entry=float(V[i, j]),
                omega_z_entry=float(t["prec"][i, j]), pair=f"({i},{j})")


def h_rcorr(rid, run, ps):
    Sigma = sigma_of(run["ds"], run["T"])
    t = targets(Sigma, np.ones(Sigma.shape[0]))
    se, bias, reps = mc_error_stats(run["ds"], run["T"], run["path"], "rcorr")
    k, m = bonferroni_scale(Sigma.shape[0])
    V = nandiag(t["rcorr"])
    return dict(value=V, criterion="statistical", se=se * k, reference_kind="analytic",
                reference_detail=f"clean_cov_true correlation (0.8); SE: {SELF}:mc ({reps} draws of the contaminated design)",
                notes="robust correlation C01/sqrt(C00 C11) of pyspi's cached covariance_ (probe robust_corr)",
                pyspi_rcorr=float(ps["value"][0, 1]), mc_se=float(se[0, 1]), mc_bias=float(bias[0, 1]))


def h_breakdown(rid, run, ps):
    d = truth(run["ds"], run["T"])
    C0 = np.asarray(d["clean_cov_true"], float)
    X = np.asarray(ps["input"], float).T
    E = mle_cov(X)                      # sklearn EmpiricalCovariance convention (np.cov bias=1)
    if run["kind"] == "cov":
        tv, ev = C0[0, 1], E[0, 1]
    else:
        tv, ev = np.linalg.inv(C0)[0, 1], np.linalg.inv(E)[0, 1]
    pv = float(ps["value"][0, 1])
    ep, ee = abs(pv - tv), abs(ev - tv)
    return dict(value=np.array(tv), criterion="qualitative", pass_flag=int(ep < ee), reference_kind="analytic",
                reference_detail="contaminated clean_cov_true (kernel path, raw units) vs numpy MLE covariance "
                                 "(EmpiricalCovariance convention, sklearn/covariance/_empirical_covariance.py:107)",
                notes="proposition: the robust estimate is closer to the clean truth than the empirical estimate",
                truth_entry=float(tv), pyspi_entry=pv, empirical_entry=float(ev), err_pyspi=float(ep),
                err_empirical=float(ee), err_ratio_pyspi_over_empirical=float(ep / ee))


def h_singular(rid, run, ps):
    X = np.asarray(ps["input"], float)
    rank = int(np.linalg.matrix_rank(X))
    V = ps["value"]
    flagged = bool(np.isnan(offdiag(V)).any())
    return dict(value=np.array(float(rank)), criterion="qualitative", pass_flag=int(flagged),
                reference_kind="independent-impl", reference_detail=f"{SELF}:h_singular (numpy matrix_rank of pyspi's input)",
                notes="proposition: pyspi signals a singular scatter (x2 == x0 makes every h-subset scatter singular) with "
                      "NaN or an error; a finite output is a pseudo-inverse",
                input_rank=rank, n_processes=int(X.shape[0]), prec_0_2=float(V[0, 2]),
                max_abs_offdiag=float(np.nanmax(np.abs(V))), sym_observed=sym(V))


def h_sq(rid, run, ps):
    crid, c = companion(run, "rdet")
    return dict(value=np.square(c["value"]), criterion="exact", reference_kind="same-lineage",
                reference_detail=f"square of pyspi unsquared output {crid} (same seed)",
                notes="-sq identifier = np.square(unsquared) (pyspi/statistics/basic.py:51-52)",
                bitwise_equal=int(np.array_equal(ps["value"], np.square(c["value"]), equal_nan=True)))


def h_perm(rid, run, ps):
    crid, c = companion(run, "rdet")
    perm = [3, 2, 1, 0]
    V = c["value"][np.ix_(perm, perm)]
    return dict(value=V, criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"pyspi output {crid} with rows/columns permuted [3,2,1,0]",
                notes="orientation: A[i, j] is the (i, j) entry of the matrix fitted on processes in the given order",
                bitwise_equal=int(np.array_equal(ps["value"], V, equal_nan=True)), sym_observed=sym(ps["value"]))


def h_dir(rid, run, ps):
    s = path_scale(run)
    t = targets(sigma_of(run["ds"], run["T"]), s)
    strength = t["rcorr"] if run["kind"] == "cov" else t["pcorr"]
    sign = np.where(np.abs(strength) >= 0.2, np.sign(t[run["kind"]]), 0.0)
    np.fill_diagonal(sign, 0)
    return dict(value=nandiag(t[run["kind"]]), criterion="direction", expected_sign=sign, reference_kind="analytic",
                reference_detail=f"sign of the {run['ds']} analytic {'covariance' if run['kind'] == 'cov' else 'precision'} "
                                 "where |corr| (cov) or |partial corr| (prec) >= 0.2",
                notes="undirected: both orientations carry the same expected sign", n_signed_entries=int((sign != 0).sum()),
                sym_observed=sym(ps["value"]))


def h_opt(rid, run, ps):
    st = decode_state(ps["value"])
    X = st["z"].T
    h = st["h"]
    sk_ld = float(np.linalg.slogdet(st["raw_cov"])[1])
    rng = np.random.default_rng(int(hashlib.sha1(rid.encode()).hexdigest()[:8], 16))
    res = search(X, h, rng, 1000, 1000)
    lds = np.array([r[0] for r in res])
    bs = int(np.argmin(lds))
    scratch_best = float(lds[bs])
    r = r_call(X, "own", h)
    r_ld = np.array([r_raw_logdet(r, k, h) for k in range(1, NSEEDS_R + 1)])
    bld = min(scratch_best, float(r_ld.min()))   # best of two independent searches (2000 from-scratch + 20 x rrcov)
    tol = 1e-8
    extra = {}
    crid, c = companion(run, "seeds")
    if c is not None:
        out, sld, _ = seed_stack(c["value"], st["P"])
        at = sld <= bld + tol
        off = ~np.eye(st["P"], dtype=bool)
        extra = dict(n_seeds_at_best=int(at.sum()), n_seeds=int(sld.size),
                     max_seed_gap=float((sld - bld).max()), median_seed_gap=float(np.median(sld - bld)),
                     spread_among_seeds_at_best=float((out[at].max(0) - out[at].min(0))[off].max()) if at.sum() > 1 else 0.0,
                     spread_all_seeds=float((out.max(0) - out.min(0))[off].max()))
    return dict(value=np.array(bld), criterion="optimality", pass_flag=int(sk_ld <= bld + tol),
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:search (2000 C-step runs to a fixed point; Rousseeuw & Van Driessen 1999 Thm 1) "
                                 "and r/cov_prec_robust.R own (rrcov==1.7.5 CovMcd, 20 R seeds); best of both",
                notes="pass: log det of sklearn's raw h-subset MLE covariance (seed_all(42) fit) <= best found + 1e-8",
                sklearn_raw_logdet=sk_ld, best_logdet=bld, gap=sk_ld - bld, scratch_best_gap=scratch_best - bld,
                frac_scratch_starts_at_best=float(np.mean(lds <= bld + tol)),
                n_distinct_local_optima=int(len(np.unique(np.round(lds, 9)))),
                rrcov_n_seeds_at_best=int((r_ld <= bld + tol).sum()), rrcov_max_gap=float(r_ld.max() - bld), **extra)


HANDLERS = dict(state=h_state, base=h_base, rdet=h_rdet, rrcov=h_rrcov, seeds=h_seeds, range=h_seeds,
                eeqmcd=h_eeqmcd, alias=h_alias, freshfull=h_fresh, freshbasic=h_fresh, det=h_det, rngstream=h_det, stat=h_stat, statc=h_statc, pcorr=h_pcorr, rcorr=h_rcorr,
                breakdown=h_breakdown, singular=h_singular, sq=h_sq, perm=h_perm, dir=h_dir, opt=h_opt)


def save(rid, d):
    OUTR.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for k, v in d.items():
        arrs[k] = np.array(v) if not isinstance(v, str) else np.array(v)
    np.savez(OUTR / f"{rid}.npz", **arrs)


def main():
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_skip = 0
    errors = []
    for r in man["runs"]:
        rid = r["run_id"]
        ps = load_pyspi(rid)
        if ps is None:
            n_skip += 1
            continue
        run = parse(rid)
        try:
            save(rid, HANDLERS[run["tag"]](rid, run, ps))
            n_ok += 1
        except Exception as exc:  # report, keep going
            errors.append(f"{rid}: {exc!r}")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    print(f"{FAMILY} refs: {n_ok} written, {n_skip} without pyspi output, {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
