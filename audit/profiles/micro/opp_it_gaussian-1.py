"""Micro-benchmark for opportunity it_gaussian-1 (Workflow 3, group infotheory).

Avenue: compute the pairwise Gaussian SPIs in closed form from ONE (lagged) covariance
matrix per Data in numpy, instead of one JIDT call per ordered pair through JPype.
Baseline = today's pyspi call (harness.compute_spi, seeded, fresh Data per repeat).
Candidate = the numpy closed forms below (they live only in this script, never in pyspi/).
Agreement with the baseline says only that the candidate reproduces today's output.

Also records component timings that back the JNI/JPype-conversion statements in the
it_gaussian note: one JIDT Gaussian MI call (1-D arrays) and the 2-D JArray conversion
that _compute_conditional_entropy / xme / cce use (T x d numpy -> double[][]), against
JPype's buffer-based ``JArray.of`` (same doubles; the entropy is compared bitwise).

Also times the candidates alone at 118x1024 (shapes["118x1024_candidate_only"], with the quiet
summary.json walls of each unit for comparison) and the 2-D conversion cost per array shape.

Run: .venv/bin/python audit/profiles/micro/opp_it_gaussian-1.py   (~1-2 minutes)
     .venv/bin/python audit/profiles/micro/opp_it_gaussian-1.py --split118   (~2-3 minutes)
``--split118`` (Workflow 4c) runs only 118x1024 and merges shapes["118x1024_split"] into the existing JSON
(other entries untouched, the run logged under ``merged_runs``): Data construction timed alone; per identifier
(SPLIT_CASES, which adds te_symbolic_k-1 to CASES) the candidate math timed alone on one prebuilt Data (the
like-for-like figure against the summary.json walls, which leave Data construction out: profile_units.py times
compute_spi only), and one fresh one-shot pyspi call on a fresh Data (after an untimed 8x1024 warm-up call of the
identifier for JVM class loading) with its diff against the math-only candidate.
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import numpy as np  # noqa: E402

import _bench as B  # noqa: E402

LN2PIE = np.log(2 * np.pi * np.e)


def _lagged_cov(z, k):
    """ddof=1 covariance of [L_0..L_{k-1}, next] blocks; L_m = z[:, k-1-m : T-1-m], next = z[:, k:]."""
    P, T = z.shape
    rows = [z[:, k - 1 - m: T - 1 - m] for m in range(k)] + [z[:, k:]]
    return np.cov(np.vstack(rows), ddof=1)


def _logdet(C, idx):
    """Batched log-determinants of C[idx_b, idx_b] for an (n_pairs, d) index array."""
    sub = C[idx[:, :, None], idx[:, None, :]]
    return np.linalg.slogdet(sub)[1]


def te_gauss(z, k):
    """Gaussian TE i->j with k target lags (tau 1), 1 source lag, delay 1 (JIDT layout), nats."""
    P = z.shape[0]
    C = _lagged_cov(z, k)
    ii, jj = np.where(~np.eye(P, dtype=bool))
    past = np.stack([m * P + jj for m in range(k)], axis=1)
    src = ii[:, None]
    nxt = (k * P + jj)[:, None]
    te = 0.5 * (_logdet(C, np.hstack([past, src])) + _logdet(C, np.hstack([nxt, past]))
                - _logdet(C, past) - _logdet(C, np.hstack([nxt, past, src])))
    S = np.full((P, P), np.nan)
    S[ii, jj] = te
    return S


def pair_stats(z):
    C = np.cov(z, ddof=1)
    v = np.diag(C)
    det2 = np.outer(v, v) - C ** 2
    return v, det2


def mi_gauss(z):
    v, det2 = pair_stats(z)
    S = 0.5 * np.log(np.outer(v, v) / det2)
    np.fill_diagonal(S, np.nan)
    return S


def je_gauss(z):
    v, det2 = pair_stats(z)
    S = LN2PIE + 0.5 * np.log(det2)
    np.fill_diagonal(S, np.nan)
    return S


def ce_gauss(z):
    v, det2 = pair_stats(z)
    S = LN2PIE + 0.5 * np.log(det2) - 0.5 * (LN2PIE + np.log(v))[:, None]
    np.fill_diagonal(S, np.nan)
    return S


def tlmi_gauss(z):
    C = _lagged_cov(z, 1)          # rows: z[:, :-1] (block 0), z[:, 1:] (block 1)
    P = z.shape[0]
    vs = np.diag(C)[:P]            # source variance on samples 0..T-2
    vt = np.diag(C)[P:]            # target variance on samples 1..T-1
    c = C[:P, P:]                  # cov(src_i[:-1], targ_j[1:])
    S = 0.5 * np.log(np.outer(vs, vt) / (np.outer(vs, vt) - c ** 2))
    np.fill_diagonal(S, np.nan)
    return S


CASES = [  # (identifier, candidate, unit_id)
    ("mi_gaussian", mi_gauss, "infotheory.mi.gaussian"),
    ("je_gaussian", je_gauss, "infotheory.je.gaussian"),
    ("ce_gaussian", ce_gauss, "infotheory.ce.gaussian"),
    ("tlmi_gaussian", tlmi_gauss, "infotheory.tlmi.gaussian"),
    ("gc_gaussian_k-1_kt-1_l-1_lt-1", lambda z: te_gauss(z, 1), "infotheory.te.gaussian_k-1_kt-1_l-1_lt-1"),
    ("te_symbolic_k-10_kt-1_l-1_lt-1", lambda z: te_gauss(z, 10), "infotheory.te.symbolic"),
]


def components():
    """Per-call JIDT/JPype costs at T=1024 (min of 200 calls each)."""
    import jpype as jp
    B.jvm()
    pkg = jp.JPackage("infodynamics.measures.continuous").gaussian
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(1024), rng.standard_normal(1024)
    X4 = rng.standard_normal((1023, 4))
    mi = pkg.MutualInfoCalculatorMultiVariateGaussian()
    ent = pkg.EntropyCalculatorMultiVariateGaussian()

    def best(fn, n=200):
        w = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); w.append(time.perf_counter() - t0)
        return min(w)

    def mi_call():
        mi.initialise(1, 1)
        mi.setObservations(jp.JArray(jp.JDouble)(x), jp.JArray(jp.JDouble)(y))
        return mi.computeAverageLocalOfObservations()

    def ent4_call():
        ent.initialise(4)
        ent.setObservations(jp.JArray(jp.JDouble, 2)(X4))
        return ent.computeAverageLocalOfObservations()

    j2 = jp.JArray(jp.JDouble, 2)(X4)

    def ent4_prebuilt():
        ent.initialise(4)
        ent.setObservations(j2)
        return ent.computeAverageLocalOfObservations()

    def ent4_of():
        ent.initialise(4)
        ent.setObservations(jp.JArray.of(X4))
        return ent.computeAverageLocalOfObservations()

    h_base, h_of = ent4_call(), ent4_of()
    # 2-D conversion cost by shape (the T x d arrays the composite/kernel/Kozachenko paths pass:
    # si 1023x1/1023x2/1023x4, je 1024x2, xme_k1 1023x1/1023x2, xme_k10 1014x9/1014x10)
    conv = {}
    for shp in ((1023, 1), (1024, 2), (1023, 4), (1014, 9), (1014, 10)):
        A = rng.standard_normal(shp)
        conv[f"{shp[0]}x{shp[1]}"] = dict(
            jarray_2d_s=best(lambda: jp.JArray(jp.JDouble, 2)(A), 50),
            jarray_of_s=best(lambda: jp.JArray.of(A), 50))
    return dict(
        conversion_2d_by_shape=conv,
        jarray_of_2d_1023x4_s=best(lambda: jp.JArray.of(X4), 50),
        jidt_gaussian_entropy_4d_call_jarray_of_s=best(ent4_of, 50),
        jarray_of_entropy_bit_identical=bool(h_base == h_of),
        jidt_gaussian_mi_call_1d_T1024_s=best(mi_call),
        jarray_1d_T1024_s=best(lambda: jp.JArray(jp.JDouble)(x)),
        jarray_2d_1023x4_s=best(lambda: jp.JArray(jp.JDouble, 2)(X4), 50),
        jidt_gaussian_entropy_4d_call_incl_conversion_s=best(ent4_call, 50),
        jidt_gaussian_entropy_4d_call_prebuilt_array_s=best(ent4_prebuilt, 50),
        numpy_cov_1023x4_s=best(lambda: np.cov(X4.T, ddof=1)),
    )


def main():
    shapes = {}
    for shape in ("8x1024", "32x1024"):
        raw, make = B.data(shape)
        per = {}
        for ident, cand, unit in CASES:
            S_base, t_base = B.baseline(ident, make, repeats=3)
            t_cand, S_cand, _ = B.timeit(lambda: cand(make().to_numpy(squeeze=True)), repeats=5)
            per[ident] = B.result(S_base, t_base, S_cand, t_cand, unit_id=unit)
            print(shape, ident, {k: per[ident][k] for k in ("baseline_s", "candidate_s", "speedup", "max_abs_diff", "max_rel_diff", "bit_identical")}, flush=True)
        shapes[shape] = per
    # Candidate-only timing at 118x1024 (the pyspi baseline there costs seconds to minutes per
    # identifier; compare with the quiet walls in audit/profiles/summary.json instead).
    import json
    summ = json.load(open(B.REPO / "audit" / "profiles" / "summary.json"))["units"]
    raw, make = B.data("118x1024")
    t_data, _, _ = B.timeit(lambda: make(), repeats=5)
    cand118 = dict(data_construction_s=t_data)
    for ident, cand, unit in CASES:
        t_cand, _, _ = B.timeit(lambda: cand(make().to_numpy(squeeze=True)), repeats=3)
        modes = summ[unit]["modes"]
        cand118[ident] = dict(unit_id=unit, candidate_s=t_cand, summary_118={
            m: dict(window_s=modes[m].get("projected_window_s_118"), measure_mode=modes[m].get("measure_mode"))
            for m in modes})
        print("118x1024", ident, cand118[ident], flush=True)
    shapes["118x1024_candidate_only"] = cand118
    comp = components()
    print(comp)
    B.write("it_gaussian-1", shapes, notes=(
        "Candidate = numpy closed forms from one ddof=1 (lagged) covariance per Data (np.cov + batched slogdet); "
        "baseline = harness.compute_spi on the same Data. Both timings include Data construction (detrend + z-score). "
        "te_symbolic_k-10 is computed as Gaussian TE with 10 target lags (what pyspi runs today, known bug 1). "
        "Agreement means the candidate reproduces today's output, nothing more."),
        components=comp, threads=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                                      NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS"),
                                      jidt_num_threads="n/a (Gaussian calculators are single-threaded)"))


SPLIT_CASES = CASES + [
    ("te_symbolic_k-1_kt-1_l-1_lt-1", lambda z: te_gauss(z, 1), "infotheory.te.symbolic"),
]


def split118():
    """118x1024 only: Data construction, candidate math and a one-shot baseline per identifier; merged."""
    import json
    summ = json.load(open(B.REPO / "audit" / "profiles" / "summary.json"))["units"]
    load_start = os.getloadavg()
    B.jvm()
    _, make8 = B.data("8x1024")
    raw, make = B.data("118x1024")
    t_data, d, _ = B.timeit(make, repeats=5)
    z = d.to_numpy(squeeze=True)
    out = dict(data_construction_s=t_data)
    for ident, cand, unit in SPLIT_CASES:
        t_math, S_math, _ = B.timeit(lambda: cand(z), repeats=5)
        sp = B.spi(ident); B.h.seed_all(B.SPI_SEED); B.h.compute_spi(sp, make8())   # untimed warm-up
        dd = make(); sp = B.spi(ident); B.h.seed_all(B.SPI_SEED)
        t_base, S_base, _ = B.timeit(lambda: B.h.compute_spi(sp, dd), repeats=1)
        modes = summ[unit]["modes"]
        r = dict(unit_id=unit, candidate_math_s=t_math, candidate_incl_data_s=t_math + t_data,
                 baseline_oneshot_s=t_base, speedup_math=t_base / t_math,
                 summary_118={m: dict(window_s=modes[m].get("projected_window_s_118"),
                                      measure_mode=modes[m].get("measure_mode")) for m in modes},
                 **B.diff(S_base, S_math))
        out[ident] = r
        print("118x1024_split", ident, {k: r[k] for k in ("candidate_math_s", "baseline_oneshot_s", "max_abs_diff", "max_rel_diff", "nan_mask_equal")}, flush=True)
    path = B.MICRO / "opp_it_gaussian-1.json"
    doc = json.loads(path.read_text())
    doc["shapes"]["118x1024_split"] = out
    doc.setdefault("merged_runs", []).append(dict(
        shapes=["118x1024_split"], argv=sys.argv[1:], written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        loadavg_start=list(load_start), loadavg_end=list(os.getloadavg()),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        note=("Workflow 4c: candidate math alone (one prebuilt Data, min of 5) and Data construction alone "
              "(min of 5); candidate_incl_data_s = their sum. The summary.json basis times compute_spi on a "
              "prebuilt Data, so candidate_math_s is the like-for-like cost. Baseline = one fresh one-shot call "
              "per identifier (shared machine, noisy); diff = baseline vs math-only candidate.")))
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    if "--split118" in sys.argv:
        split118()
    else:
        main()
