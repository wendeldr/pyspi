"""cdt_pairwise-1: reci as one batched closed-form least-squares fit over all ordered pairs.

Baseline: pyspi causal.reci (cdt RECI.b_fit_score per ordered pair: minmax_scale, PolynomialFeatures(3)
with the x and x^2 columns zeroed, LinearRegression, in-sample mean_squared_error).
Candidate (lives only here): per-channel min-max scaling once, then for every pair the OLS of y on
[1, x^3] in closed form from one P x P cross-moment matrix: MSE[i, j] = var(y_j) - cov(x_i^3, y_j)^2 / var(x_i^3)
(population moments). Different float operation order from scipy lstsq, so within-tolerance is expected.
Run: OMP_NUM_THREADS=1 .venv/bin/python audit/profiles/micro/opp_cdt_pairwise-1.py
     OMP_NUM_THREADS=1 .venv/bin/python audit/profiles/micro/opp_cdt_pairwise-1.py --split118
``--split118`` (Workflow 4c) runs only 118x1024 and merges shapes["118x1024_split"] into the existing JSON
(other entries untouched, the run logged under ``merged_runs``): Data construction timed alone, the candidate
math timed alone on a prebuilt Data (the like-for-like figure against the summary.json walls, which leave Data
construction out: profile_units.py times compute_spi only), the candidate including Data construction, and a
fresh one-shot baseline with its diff against the math-only candidate.
"""
import sys
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import numpy as np
import _bench as B


def candidate(d):
    z = d.to_numpy(squeeze=True)
    lo, hi = z.min(axis=1, keepdims=True), z.max(axis=1, keepdims=True)
    y = (z - lo) / (hi - lo)
    x3 = y ** 3
    yc, xc = y - y.mean(axis=1, keepdims=True), x3 - x3.mean(axis=1, keepdims=True)
    T = z.shape[1]
    cxy = (xc @ yc.T) / T                       # cov(x_i^3, y_j)
    vx, vy = (xc * xc).sum(axis=1) / T, (yc * yc).sum(axis=1) / T
    A = vy[None, :] - cxy ** 2 / vx[:, None]
    np.fill_diagonal(A, np.nan)
    return A


def one_shot_baseline(make):
    d = make(); s = B.spi("reci"); B.h.seed_all(B.SPI_SEED)
    t, S, _ = B.timeit(lambda: B.h.compute_spi(s, d), repeats=1)
    return S, t


def split118():
    """118x1024 only: Data construction, candidate math and one-shot baseline timed separately; merged."""
    import json, time
    summ = json.load(open(B.REPO / "audit" / "profiles" / "summary.json"))["units"]["causal.reci"]["modes"]
    load_start = B.os.getloadavg()
    raw, make = B.data("118x1024")
    t_data, d, _ = B.timeit(make, repeats=5)
    t_math, S_math, _ = B.timeit(lambda: candidate(d), repeats=5)
    t_incl, _, _ = B.timeit(lambda: candidate(make()), repeats=5)
    S_base, t_base = one_shot_baseline(make)
    r = dict(data_construction_s=t_data, candidate_math_s=t_math, candidate_incl_data_s=t_incl,
             baseline_oneshot_s=t_base, speedup_math=t_base / t_math, speedup_incl_data=t_base / t_incl,
             summary_118={m: dict(window_s=summ[m].get("projected_window_s_118"),
                                  measure_mode=summ[m].get("measure_mode")) for m in summ},
             **B.diff(S_base, S_math))
    print("118x1024_split", r, flush=True)
    path = B.MICRO / "opp_cdt_pairwise-1.json"
    doc = json.loads(path.read_text())
    doc["shapes"]["118x1024_split"] = r
    doc.setdefault("merged_runs", []).append(dict(
        shapes=["118x1024_split"], argv=sys.argv[1:], written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        loadavg_start=list(load_start), loadavg_end=list(B.os.getloadavg()),
        omp_num_threads=B.os.environ.get("OMP_NUM_THREADS"),
        note=("Workflow 4c: candidate math alone (prebuilt Data, min of 5) vs Data construction alone (min of 5); "
              "the summary.json basis times compute_spi on a prebuilt Data, so candidate_math_s is the like-for-like "
              "cost. Baseline = one fresh one-shot call; diff = baseline vs math-only candidate.")))
    tmp = path.with_suffix(f".tmp-{B.os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    B.os.replace(tmp, path)


if "--split118" in sys.argv:
    split118()
    sys.exit(0)

shapes = {}
for shape in ("7x100", "8x1024", "32x1024", "118x1024"):
    raw, make = B.data(shape)
    S_base, t_base = B.baseline("reci", make) if shape != "118x1024" else one_shot_baseline(make)
    t_cand, S_cand, _ = B.timeit(lambda: candidate(make()), repeats=5)
    shapes[shape] = B.result(S_base, t_base, S_cand, t_cand,
                             baseline_repeats=3 if shape != "118x1024" else 1)
    print(shape, {k: shapes[shape][k] for k in ("baseline_s", "candidate_s", "speedup", "max_abs_diff", "max_rel_diff", "nan_mask_equal")})
g = B.golden("reci")
if g is not None:
    _, make = B.data("7x100")
    shapes["7x100"]["golden_vs_candidate"] = B.diff(g, candidate(make()))
print(B.write("cdt_pairwise-1", shapes,
              notes="reci batched closed-form OLS on [1, x^3] after per-channel min-max scaling; candidate_s includes "
                    "building a fresh Data (as the baseline's compute_spi does not), so the speedup is a lower bound. "
                    "118x1024 baseline is one timed call without the helper's warm-up. Agreement means the candidate "
                    "matches the baseline output, not that either value is mathematically right.",
              threads=dict(OMP_NUM_THREADS=B.os.environ.get("OMP_NUM_THREADS"), NUMBA_NUM_THREADS="n/a (no numba)", jidt="n/a")))
