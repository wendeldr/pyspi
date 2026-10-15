"""spectral_csm-1: build the taper-level CSM only on the non-negative FFT bins (undirected measures).

The library (spectral_connectivity 1.1.2, connectivity.py:195-220) forms the per-taper outer
products X_i conj(X_j) for all n_fft two-sided bins, (1,1,5,n_fft,N,N) complex128, averages
over tapers and only afterwards drops the negative bins (_non_negative_frequencies). Candidate A
slices the Fourier coefficients to the (n_fft+1)//2 non-negative bins first and then runs the
same numpy operations (matmul, mean over (1,2), sqrt-normalisation; x/|x| for plv). Candidate B
(cohmag only) replaces the per-taper tensor by one batched matmul over the taper axis
(sum_k X_k^T conj(X_k) / K), which reorders the taper sum. Baselines: pyspi cohmag / plv
(representative identifiers) through harness.compute_spi. Peak traced numpy memory per call is
measured separately with tracemalloc. Candidate code lives only here, never in pyspi/.
Run: .venv/bin/python audit/profiles/micro/opp_spectral_csm-1.py [shapes...]
     .venv/bin/python audit/profiles/micro/opp_spectral_csm-1.py --add ppc,wpli,dswpli [shapes...]
``--add <measures>`` (added in Workflow 4c) runs only the listed phase-sync measures and merges
them into the existing JSON under ``<identifier>__sliced`` (the cohmag and plv entries, the other
keys and the notes are kept; the run is logged under ``merged_runs``). Its candidate calls the
library's own undecorated method (``__wrapped__`` below ``_asnumpy`` and
``_non_negative_frequencies``) on a Connectivity built from the Fourier coefficients sliced to the
non-negative bins; ppc is the library's formula on the undecorated plv. Its timing is paired: each
repeat runs the baseline and then the candidate on fresh Data, so both see the same machine state.
"""
import json
import os
import sys
import time
import tracemalloc
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402  (sets OMP_NUM_THREADS=1)
import numpy as np  # noqa: E402
from pyspi._spectral_compat import spectral_connectivity as sc  # noqa: E402

COH, PLV = "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5", "plv_multitaper_mean_fs-1_fmin-0_fmax-0-5"


def fourier(data):
    z = np.transpose(data.to_numpy(squeeze=True))
    m = sc.Multitaper(z, sampling_frequency=1)
    return m.fft(), m.frequencies


def band_mean(adj_freq, freq):                       # pyspi spectral.py:87-101, fmin 0, fmax 0.5
    nh = (freq.size + 1) // 2
    f = freq[:nh]
    fid = np.where((f >= 0) * (f <= 0.5))[0]
    adj = np.nanmean(adj_freq[0, fid, :, :], axis=0)
    np.fill_diagonal(adj, np.nan)
    return adj


def ctrans(x):
    return x.swapaxes(-1, -2).conjugate()


def cohmag_sliced(data):                             # candidate A, linear measure
    fc, freq = fourier(data)
    nh = (fc.shape[-2] + 1) // 2
    fc = np.ascontiguousarray(fc[..., :nh, :])
    power = np.mean(fc * fc.conjugate(), axis=(1, 2)).real
    norm = np.sqrt(power[..., :, np.newaxis] * power[..., np.newaxis, :])
    norm[norm == 0] = np.nan
    x = fc[..., np.newaxis]
    csm = np.mean(np.matmul(x, ctrans(x), dtype=np.complex128), axis=(1, 2))
    coh = csm / norm
    n = fc.shape[-1]
    coh[..., np.arange(n), np.arange(n)] = np.nan
    return band_mean(np.abs(coh) ** 2, freq)


def cohmag_gemm(data):                               # candidate B, taper sum inside one matmul
    fc, freq = fourier(data)
    nh = (fc.shape[-2] + 1) // 2
    X = fc[0, 0, :, :nh, :]                          # (K, F, N)
    K = X.shape[0]
    Xf = np.ascontiguousarray(X.transpose(1, 2, 0))  # (F, N, K)
    csm = np.matmul(Xf, Xf.conj().transpose(0, 2, 1)) / K
    power = np.real(np.einsum("fii->fi", csm))
    norm = np.sqrt(power[:, :, None] * power[:, None, :])
    norm[norm == 0] = np.nan
    coh = csm / norm
    n = X.shape[-1]
    coh[:, np.arange(n), np.arange(n)] = np.nan
    return band_mean((np.abs(coh) ** 2)[None], freq)


def plv_sliced(data):                                # candidate A, nonlinear (per-taper) measure
    fc, freq = fourier(data)
    nh = (fc.shape[-2] + 1) // 2
    x = np.ascontiguousarray(fc[..., :nh, :])[..., np.newaxis]
    c = np.matmul(x, ctrans(x), dtype=np.complex128)
    plv = np.abs(np.mean(c / np.abs(c), axis=(1, 2)))
    return band_mean(plv, freq)


def peak_mb(fn, *a):
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(*a)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2**20


def baseline_call(ident, make):
    s = B.spi(ident)
    B.h.seed_all(B.SPI_SEED)
    return B.h.compute_spi(s, make())


def baseline_walls(ident, make, reps):
    """B.baseline with the per-repeat walls kept (memory-bound 118x1024 calls vary 2-7x run to run)."""
    baseline_call(ident, make)                       # untimed warm-up
    t, S, walls = B.timeit(lambda: baseline_call(ident, make), repeats=reps)
    return S, t, walls


def run(shape):
    _, make = B.data(shape)
    reps = 3                                         # min of 3 at every shape (was 1 at 118x1024)
    out = {}
    for ident, cands in ((COH, {"sliced": cohmag_sliced, "gemm": cohmag_gemm}), (PLV, {"sliced": plv_sliced})):
        S_base, t_base, w_base = baseline_walls(ident, make, reps)
        rss_base = peak_mb(baseline_call, ident, make)
        for name, fn in cands.items():
            fn(make())                               # warm-up
            t_c, S_c, w_c = B.timeit(lambda: fn(make()), repeats=reps)
            out[f"{ident}__{name}"] = B.result(S_base, t_base, S_c, t_c,
                                                baseline_walls_s=w_base, candidate_walls_s=w_c,
                                                speedup_median=float(np.median(w_base) / np.median(w_c)),
                                                peak_traced_mb_baseline=rss_base,
                                                peak_traced_mb_candidate=peak_mb(fn, make()))
        print(shape, ident, {k: (v["speedup"], v["bit_identical"], v["max_abs_diff"]) for k, v in out.items() if k.startswith(ident)}, flush=True)
    return out


# --- --add mode (Workflow 4c): phase-sync measures through the library's undecorated methods ---
_C = sc.Connectivity
_RAW = {"plv": _C._phase_locking_value.__wrapped__.__wrapped__,
        "wpli": _C.weighted_phase_lag_index.__wrapped__.__wrapped__,
        "dswpli": _C.debiased_squared_weighted_phase_lag_index.__wrapped__.__wrapped__}


def ident_of(measure):
    return f"{measure}_multitaper_mean_fs-1_fmin-0_fmax-0-5"


def phase_sync_sliced(measure, data):                # candidate A for a phase-sync measure
    z = np.transpose(data.to_numpy(squeeze=True))
    m = sc.Multitaper(z, sampling_frequency=1)
    fc = m.fft()
    nh = (fc.shape[-2] + 1) // 2
    conn = _C(fourier_coefficients=np.ascontiguousarray(fc[..., :nh, :]),
              frequencies=m.frequencies[:nh], time=m.time)
    if measure == "ppc":                             # connectivity.py:690-696 on the undecorated plv
        n = conn.n_observations
        plv_sum = _RAW["plv"](conn) * n
        r = ((plv_sum * plv_sum.conjugate() - n) / (n ** 2 - n)).real
    elif measure == "plv":
        r = np.abs(_RAW["plv"](conn))
    else:
        r = _RAW[measure](conn)
    return band_mean(r, m.frequencies)


def run_added(shape, measures, reps=3):
    _, make = B.data(shape)
    _, mk_small = B.data("8x256")
    out = {}
    for meas in measures:
        ident = ident_of(meas)
        baseline_call(ident, mk_small)               # warm-up
        phase_sync_sliced(meas, mk_small())
        w_b, w_c, S_b, S_c = [], [], None, None
        for _ in range(reps):                        # paired: baseline then candidate, fresh Data each
            t, S_b, _w = B.timeit(lambda: baseline_call(ident, make), repeats=1)
            w_b.append(t)
            t, S_c, _w = B.timeit(lambda: phase_sync_sliced(meas, make()), repeats=1)
            w_c.append(t)
        out[f"{ident}__sliced"] = B.result(S_b, min(w_b), S_c, min(w_c),
                                          baseline_walls_s=w_b, candidate_walls_s=w_c,
                                          speedup_median=float(np.median(w_b) / np.median(w_c)),
                                          paired_ratios=[b / c for b, c in zip(w_b, w_c)],
                                          protocol="paired, workflow 4c --add")
        r = out[f"{ident}__sliced"]
        print(shape, ident, round(r["speedup"], 3), round(r["speedup_median"], 3), r["bit_identical"], r["nan_mask_equal"], flush=True)
    return out


def merge_added(measures, shapes):
    path = B.MICRO / "opp_spectral_csm-1.json"
    doc = json.loads(path.read_text())
    load_start = os.getloadavg()
    for shape in shapes:
        doc["shapes"].setdefault(shape, {}).update(run_added(shape, measures))
    gold = {}
    _, mk7 = B.data("7x100")
    for meas in measures:
        g = B.golden(ident_of(meas))
        if g is not None:
            gold[f"{meas}_sliced_vs_golden"] = B.diff(g, phase_sync_sliced(meas, mk7()))
    doc["shapes"].setdefault("7x100_golden", {}).update(gold)
    doc.setdefault("merged_runs", []).append(dict(
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), measures=list(measures), shapes=list(shapes),
        golden_keys=sorted(gold), loadavg_start=list(load_start), loadavg_end=list(os.getloadavg()),
        note=("--add mode: paired protocol (each repeat = baseline then candidate on fresh Data), 3 repeats; "
              "baseline_s/candidate_s are the minima, speedup = min/min, speedup_median = median/median, "
              "paired_ratios = per-repeat baseline/candidate. Candidate = the library's undecorated method on "
              "Fourier coefficients sliced to the non-negative bins. Run while other audit agents shared the "
              "machine (loadavg recorded), so walls are noisier than a quiet run; agreement with the baseline "
              "means the candidate reproduces current output, nothing more.")))
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)
    return path


if __name__ == "__main__" and "--add" in sys.argv:
    i = sys.argv.index("--add")
    meas = sys.argv[i + 1].split(",")
    assert all(m in ("plv", "ppc", "wpli", "dswpli") for m in meas), meas
    print(merge_added(meas, [a for a in sys.argv[1:] if a not in ("--add", sys.argv[i + 1])] or ["32x1024", "118x1024"]))
elif __name__ == "__main__":
    shapes = sys.argv[1:] or ["32x1024", "118x1024"]
    res = {s: run(s) for s in shapes}
    gold = B.golden(COH)
    if gold is not None:
        _, mk7 = B.data("7x100")
        res["7x100_golden"] = {"cohmag_sliced_vs_golden": B.diff(gold, cohmag_sliced(mk7())),
                               "plv_sliced_vs_golden": B.diff(B.golden(PLV), plv_sliced(mk7()))}
    p = B.write("spectral_csm-1", res, notes=(
        "Timings: min of 3 repeats at every shape (baseline_s, candidate_s, speedup); the per-repeat walls "
        "and the ratio of medians (speedup_median) are kept because memory-bound 118x1024 baselines vary "
        "2-7x run to run. Candidate A (sliced) = same numpy ops on the non-negative bins only; candidate B (gemm) = "
        "taper sum inside one batched matmul. Baseline = pyspi through harness.compute_spi on a fresh "
        "Data (includes Multitaper/DPSS/FFT, as do the candidates). peak_traced_mb_* = tracemalloc peak "
        "of numpy allocations during one call (not RSS). Agreement with the baseline means the "
        "candidate reproduces current output, nothing more."),
        threads=dict(OMP_NUM_THREADS=1, NUMBA_NUM_THREADS="n/a (no numba code)", JIDT="n/a"))
    print(p)
