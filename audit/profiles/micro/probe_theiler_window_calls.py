"""Probe (not an opportunity benchmark): un-profiled wall of every JIDTBase._set_theiler_window call.

Premise checked: whether the large self time of ``_set_theiler_window`` seen in two cProfile runs
(mi_kraskov_DCE 8x1024: 0.171 s self, 3.3%; te_kraskov_DCE_auto 8x256: 0.299 s self, 0.72%) is a
steady per-pair cost or a property of those profiled runs. The method is wrapped in-process with a
perf_counter timer (``pyspi.utils.acf`` too, to split the window build from the rest); nothing in
pyspi/ is edited, no candidate code runs and no outputs are compared.

Each case runs in its own fresh subprocess (new JVM), three processes per case, two seeded
``harness.compute_spi`` calls per process on a fresh ``Data`` each (call 0 = first call of that SPI in
the process, call 1 = a second call). In each call the first ``_set_theiler_window`` call builds
``data.theiler`` (P(P-1) acf calls); every later one only reads W[i, j] and calls JIDT ``setProperty``.

The ``precall`` case repeats te_kraskov_DCE_auto at 8x256 with one extra, trivial Java call
(``self._calc.getProperty(DYN_CORR_EXCL)``) timed separately just before each ``_set_theiler_window``
call (``pre_*`` fields). It asks whether a slow call is ``_set_theiler_window``'s own cost or a wait
that lands on whichever Java call comes first after the previous pair's JIDT work.

Run: .venv/bin/python audit/profiles/micro/probe_theiler_window_calls.py   (~10 minutes; writes the .json)
The committed JSON is run2; ``earlier_runs`` (run1, before the precall case and ``rest_n_over_10ms``
existed) and ``run_variation_note`` were added to it by hand after run2.
"""
import json
import os
import subprocess
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [  # (identifier, shape): the five DCE profiles in audit/profiles/
    ("mi_kraskov_NN-4_DCE", "8x1024"),
    ("tlmi_kraskov_NN-4_DCE", "8x1024"),
    ("te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1", "8x1024"),
    ("te_kraskov_NN-4_DCE_k-2_kt-1_l-1_lt-1", "8x1024"),
    ("te_kraskov_NN-4_DCE_k-max-10_tau-max-4", "8x256"),
    ("te_kraskov_NN-4_DCE_k-max-10_tau-max-4", "8x256", "precall"),
]
PROCESSES, CALLS = 3, 2


def worker(identifier, shape, variant=""):
    sys.path.insert(0, HERE)
    import numpy as np
    import _bench as B
    from pyspi import utils
    from pyspi.statistics import infotheory as it
    B.jvm()
    theiler_t, acf_t, pre_t = [], [], []
    orig_set, orig_acf = it.JIDTBase._set_theiler_window, utils.acf

    def set_wrapped(self, data, i, j):
        if variant == "precall":
            t0 = time.perf_counter()
            self._calc.getProperty(self._DYN_CORR_EXCL_PROP_NAME)
            pre_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = orig_set(self, data, i, j)
        theiler_t.append(time.perf_counter() - t0)
        return r

    def acf_wrapped(*a, **k):
        t0 = time.perf_counter()
        r = orig_acf(*a, **k)
        acf_t.append(time.perf_counter() - t0)
        return r
    it.JIDTBase._set_theiler_window, utils.acf = set_wrapped, acf_wrapped
    raw, make = B.data(shape)
    out = []
    for call in range(CALLS):
        theiler_t.clear(); acf_t.clear(); pre_t.clear()
        sp = B.spi(identifier)
        d = make()
        B.h.seed_all(B.SPI_SEED)
        t0 = time.perf_counter()
        B.h.compute_spi(sp, d)
        wall = time.perf_counter() - t0
        a, rest = np.array(theiler_t), np.array(theiler_t[1:])
        out.append(dict(call=call, n_calls=int(a.size), n_acf=len(acf_t), compute_spi_wall_s=wall,
                        sum_s=float(a.sum()), first_ms=float(a[0] * 1e3), acf_in_first_ms=float(sum(acf_t) * 1e3),
                        first_minus_acf_ms=float((a[0] - sum(acf_t)) * 1e3),
                        rest_median_ms=float(np.median(rest) * 1e3), rest_p95_ms=float(np.percentile(rest, 95) * 1e3),
                        rest_max_ms=float(rest.max() * 1e3), rest_sum_ms=float(rest.sum() * 1e3),
                        rest_n_over_10ms=int((rest > 0.010).sum())))
        if pre_t:
            pre = np.array(pre_t)
            out[-1].update(pre_median_ms=float(np.median(pre) * 1e3), pre_p95_ms=float(np.percentile(pre, 95) * 1e3),
                           pre_max_ms=float(pre.max() * 1e3), pre_sum_ms=float(pre.sum() * 1e3),
                           pre_n_over_10ms=int((pre > 0.010).sum()))
    print("RESULT " + json.dumps(out), flush=True)


def main():
    res = {}
    load_start = os.getloadavg()
    for ident, shape, *variant in CASES:
        variant = variant[0] if variant else ""
        runs = []
        for p in range(PROCESSES):
            cp = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", ident, shape, variant],
                                capture_output=True, text=True, timeout=900)
            line = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
            if cp.returncode != 0 or not line:
                raise RuntimeError(f"{ident} {shape} process {p}: rc={cp.returncode}\n{cp.stderr[-2000:]}")
            calls = json.loads(line[-1][len("RESULT "):])
            runs.append(dict(process=p, calls=calls))
            for c in calls:
                print(ident, shape, f"proc{p}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
        res["|".join(x for x in (ident, shape, variant) if x)] = dict(identifier=ident, shape=shape, variant=variant, runs=runs)
    doc = dict(schema=1, probe="theiler_window_calls", script="audit/profiles/micro/probe_theiler_window_calls.py",
               cases=res, processes_per_case=PROCESSES, calls_per_process=CALLS,
               env=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                        NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS"), cpu_count=os.cpu_count()),
               loadavg_start=list(load_start), loadavg_end=list(os.getloadavg()),
               written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               note=("Wall of each JIDTBase._set_theiler_window call without cProfile. first_ms = the call that builds "
                     "data.theiler (acf_in_first_ms of it is utils.acf); rest_* = the other P(P-1)-1 calls, which read "
                     "W[i, j] and call JIDT setProperty; rest_n_over_10ms counts calls above 10 ms. pre_* (precall case "
                     "only) = a trivial Java getProperty call timed just before each _set_theiler_window call. A timing of "
                     "the premise only; no candidate code, no output comparison."))
    path = os.path.join(HERE, "probe_theiler_window_calls.json")
    tmp = path + f".tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        main()
