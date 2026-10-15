"""Micro-benchmark for opportunity it_kraskov-1 (Workflow 3, group infotheory).

Avenue: pin JIDT's NUM_THREADS to 1 on the KSG calculators (pyspi never sets it, so JIDT uses
Runtime.availableProcessors(), 64 on this machine) and get throughput from process-level
parallelism instead (xc-window-parallel). Baseline = today's pyspi call (default NUM_THREADS).
Candidate = the same SPI object with ``_calc.setProperty("NUM_THREADS", "1")`` before the call;
for the auto-embedding units the property is copied into the internal AIS calculators by JIDT
(TransferEntropyCalculatorViaCondMutualInfo copies every stored property). Nothing in pyspi/
is changed. Agreement with the baseline says only that the candidate reproduces today's output.

Machine was shared with other audit groups while this ran (OMP_NUM_THREADS=1 everywhere);
the default-thread baseline can only lose from that contention, so the speedups are an upper
estimate of the single-process effect and say nothing about throughput under window parallelism.

Run: .venv/bin/python audit/profiles/micro/opp_it_kraskov-1.py   (~5 minutes on a shared machine)
     .venv/bin/python audit/profiles/micro/opp_it_kraskov-1.py --only 3x1024   (~2-3 minutes)
``--only <shape>`` runs only the cases at that shape and merges them into the existing JSON
(the other shapes, ``earlier_runs`` and ``run_variation_note`` are kept; the run is logged
under ``merged_runs``). The two 3x1024 cases were added after the first three runs to measure
the auto-embedding units at T=1024 (3 channels = 6 ordered pairs; the units are infeasible above 8x1024).
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402

CASES = [  # (shape, identifier, unit_id)
    ("8x256", "mi_kraskov_NN-4", "infotheory.mi.kraskov"),
    ("8x1024", "mi_kraskov_NN-4", "infotheory.mi.kraskov"),
    ("8x1024", "tlmi_kraskov_NN-4", "infotheory.tlmi.kraskov"),
    ("8x1024", "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1", "infotheory.te.kraskov_k-1_kt-1_l-1_lt-1"),
    ("8x1024", "mi_kraskov_NN-4_DCE", "infotheory.mi.kraskov_DCE"),
    ("3x256", "te_kraskov_NN-4_k-max-10_tau-max-4", "infotheory.te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4"),
    ("3x1024", "te_kraskov_NN-4_k-max-10_tau-max-4", "infotheory.te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4"),
    ("3x1024", "te_kraskov_NN-4_DCE_k-max-10_tau-max-4", "infotheory.te.kraskov_DCE_auto-MAX_CORR_AIS_kmax-10_taumax-4"),
]


def one_thread(identifier, make):
    sp = B.spi(identifier)
    sp._calc.setProperty("NUM_THREADS", "1")
    B.h.seed_all(B.SPI_SEED)
    return B.h.compute_spi(sp, make())


def cpu_wall(fn):
    """(process CPU seconds incl. JVM threads, wall seconds) of one call."""
    c0, w0 = time.process_time(), time.perf_counter()
    fn()
    return time.process_time() - c0, time.perf_counter() - w0


def default_threads(identifier, make):
    sp = B.spi(identifier)
    B.h.seed_all(B.SPI_SEED)
    return B.h.compute_spi(sp, make())


def main():
    import jpype as jp
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    B.jvm()
    ncpu = int(jp.JClass("java.lang.Runtime").getRuntime().availableProcessors())
    shapes = {}
    load_start = os.getloadavg()
    for shape, ident, unit in CASES:
        if only is not None and shape != only:
            continue
        raw, make = B.data(shape)
        S_base, t_base = B.baseline(ident, make, repeats=3)
        one_thread(ident, make)  # warm-up
        t_cand, S_cand, walls = B.timeit(lambda: one_thread(ident, make), repeats=3)
        cpu_b, wall_b = cpu_wall(lambda: default_threads(ident, make))
        cpu_c, wall_c = cpu_wall(lambda: one_thread(ident, make))
        r = B.result(S_base, t_base, S_cand, t_cand, unit_id=unit,
                     baseline_cpu_s=cpu_b, baseline_cpu_call_wall_s=wall_b,
                     candidate_cpu_s=cpu_c, candidate_cpu_call_wall_s=wall_c,
                     baseline_num_threads=f"default (availableProcessors = {ncpu})",
                     candidate_num_threads=1, candidate_walls_s=walls)
        shapes.setdefault(shape, {})[ident] = r
        print(shape, ident, {k: r[k] for k in ("baseline_s", "candidate_s", "speedup", "max_abs_diff", "bit_identical", "baseline_cpu_s", "candidate_cpu_s")}, flush=True)
    if only is not None:
        merge(shapes, ncpu, load_start)
        return
    B.write("it_kraskov-1", shapes, notes=(
        "Baseline = harness.compute_spi with JIDT's default NUM_THREADS; candidate = same SPI with "
        "NUM_THREADS=1 set on its calculator. Both include Data construction. Run while other audit "
        "groups shared the 64-core machine. The 3x256 auto-embedding case is a small labelled shape "
        "(the unit is infeasible above 8x1024). baseline_cpu_s / candidate_cpu_s = process CPU time "
        "(all threads, JVM included) of one extra call each, i.e. the cores a call occupies."),
        threads=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                     NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS"),
                     jidt_available_processors=ncpu,
                     jidt_num_threads_baseline="default (unset by pyspi)",
                     jidt_num_threads_candidate=1))


MERGE_NOTES = {("3x1024",): (
    "3x1024 = 3 channels x T=1024 (6 ordered pairs), the T=1024 shape for the two auto-embedding units "
    "(infeasible above 8x1024); same protocol as the other shapes (baseline = min of 3 default-thread calls "
    "after a warm-up, candidate = min of 3 NUM_THREADS=1 calls, candidate_walls_s = all three). Compare with "
    "the quiet default from summary.json (8x1024 fresh wall / 56 pairs x 6).")}


def merge(shapes, ncpu, load_start):
    """Add ``shapes`` to the existing opp_it_kraskov-1.json without touching its other entries."""
    import json
    path = B.MICRO / "opp_it_kraskov-1.json"
    doc = json.loads(path.read_text())
    for shape, cases in shapes.items():
        doc["shapes"].setdefault(shape, {}).update(cases)
    doc.setdefault("merged_runs", []).append(dict(
        shapes=sorted(shapes), argv=sys.argv[1:], written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        loadavg_start=list(load_start), loadavg_end=list(os.getloadavg()), jidt_available_processors=ncpu,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"), numba_num_threads=os.environ.get("NUMBA_NUM_THREADS"),
        note=MERGE_NOTES.get(tuple(sorted(shapes)), "")))
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
