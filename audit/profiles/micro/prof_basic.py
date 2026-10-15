"""Extra cProfile runs for the basic group (Workflow 3) at 118x1024, plus their index.

The 8x1024 profiles of the cov/prec units are dominated by fixed sklearn input validation, so the
group's hotspot claims at 118 channels rest on four extra profiles of the representative identifiers:
cov_EmpiricalCovariance, cov_GraphicalLasso, cov_GraphicalLassoCV and cov_MinCovDet.

Usage:
  .venv/bin/python audit/profiles/micro/prof_basic.py run <out_dir> [PxT]
      one untimed warm-up call at the same shape, then cProfile around one seeded
      harness.compute_spi call on a fresh Data; writes <out_dir>/prof_<unit>__<PxT>.pstats.
      Also refits sklearn GraphicalLassoCV on the same window with warnings recorded (pyspi
      silences them in basic.py:Estimators._from_cache) and prints the convergence result.
  .venv/bin/python audit/profiles/micro/prof_basic.py index [<repro_dir>]
      writes audit/profiles/micro/prof_basic__index.json from the pstats kept in this directory
      (opaque_share computed by profile_units.analyse_profile, the same definition as the
      <unit>__8x1024__profile.json files) and, if given, the frame shares of a reproduction run.

OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1 (no numba or JIDT code on these paths). Walls are inflated by
profiling; quote shares of the compute_spi frame, not walls.
"""
import json
import os
import sys
import time
import cProfile
import pstats
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402

UNITS = {"basic.cov.EmpiricalCovariance": "cov_EmpiricalCovariance",
         "basic.cov.GraphicalLasso": "cov_GraphicalLasso",
         "basic.cov.GraphicalLassoCV": "cov_GraphicalLassoCV",
         "basic.cov.MinCovDet": "cov_MinCovDet"}
# (file suffix, function) frames quoted in the records; share = cumtime / compute_spi cumtime
FRAMES = [("scipy/linalg/_decomp.py", "eigh"), ("scipy/linalg/_basic.py", "pinvh"),
          ("numpy/lib/_function_base_impl.py", "cov"), ("sklearn/utils/validation.py", "check_array"),
          ("sklearn/covariance/_robust_covariance.py", "_c_step"),
          ("sklearn/covariance/_empirical_covariance.py", "empirical_covariance"),
          ("sklearn/covariance/_graph_lasso.py", "graphical_lasso_path"),
          ("sklearn/covariance/_graph_lasso.py", "_graphical_lasso")]


def shares(path):
    st = pstats.Stats(str(path)).stats
    total = [v[3] for k, v in st.items() if k[2] == "compute_spi"][0]
    out = {"compute_spi_cum_s": round(total, 4)}
    for suffix, fn in FRAMES:
        for (f, _l, func), (_cc, nc, tt, ct, _c) in st.items():
            if func == fn and f.endswith(suffix):
                out[f"{suffix}:{fn}"] = dict(ncalls=nc, cum_pct=round(100 * ct / total, 1),
                                             self_pct=round(100 * tt / total, 1))
    return out


def run(out_dir, shape):
    os.makedirs(out_dir, exist_ok=True)
    raw, make = B.data(shape)
    for unit, ident in UNITS.items():
        s = B.spi(ident); B.h.seed_all(B.SPI_SEED); B.h.compute_spi(s, make())  # warm-up
        d = make(); sp = B.spi(ident); B.h.seed_all(B.SPI_SEED)
        pr = cProfile.Profile(); t0 = time.perf_counter()
        pr.enable(); B.h.compute_spi(sp, d); pr.disable()
        out = os.path.join(out_dir, f"prof_{unit}__{shape}.pstats")
        pr.dump_stats(out)
        print(json.dumps(dict(unit=unit, ident=ident, shape=shape,
                              profiled_wall_s=round(time.perf_counter() - t0, 3), pstats=out)), flush=True)
    from sklearn.covariance import GraphicalLassoCV
    z = make().to_numpy(squeeze=True).T
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = GraphicalLassoCV().fit(z)
    conv = dict(shape=shape, alpha_=float(m.alpha_), n_iter_=int(m.n_iter_),
                warnings=sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in w}))
    with open(os.path.join(out_dir, f"glassocv_convergence__{shape}.json"), "w") as f:
        json.dump(conv, f, indent=1)
    print(json.dumps(conv))


def index(repro_dir=None, shape="118x1024"):
    profs = {}
    for unit, ident in UNITS.items():
        p = B.MICRO / f"prof_{unit}__{shape}.pstats"
        a = B.pu.analyse_profile(str(p), str(B.REPO / "pyspi"))
        e = dict(unit_id=unit, identifier=ident, shape=shape,
                 pstats=f"audit/profiles/micro/{p.name}", opaque_share=round(a["opaque_share"], 4),
                 frames=shares(p))
        if repro_dir:
            q = os.path.join(repro_dir, p.name)
            if os.path.exists(q):
                e["reproduction_frames"] = shares(q)
        profs[f"{unit}__{shape}"] = e
    doc = dict(schema=1, script="audit/profiles/micro/prof_basic.py", written_at=time.strftime("%Y-%m-%d"),
               threads=dict(OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", JIDT="n/a"),
               notes="cProfile around one seeded harness.compute_spi call of the representative identifier on a "
                     "fresh Data, after one untimed warm-up at the same shape. The kept pstats were written by the "
                     "basic group's analyst with the same call sequence (scratch script); 'reproduction_frames' are "
                     "the shares from a later 'run' of this script into a scratch directory, kept to show the shares "
                     "are reproducible. opaque_share = tottime of builtin ('~') entries over total tottime "
                     "(profile_units.analyse_profile), a lower bound: compiled sklearn/scipy work is hidden in "
                     "the self time of Python frames (e.g. the Cython cd_fast solver inside _graphical_lasso).",
               profiles=profs)
    if repro_dir:
        q = os.path.join(repro_dir, f"glassocv_convergence__{shape}.json")
        if os.path.exists(q):
            doc["graphical_lasso_cv_refit"] = json.load(open(q))
    path = B.MICRO / "prof_basic__index.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "118x1024")
    else:
        index(sys.argv[2] if len(sys.argv) > 2 else None)
