"""Bounded, non-production probes of remaining JIDT optimization opportunities.

Run with one BLAS thread. No config or estimator implementation is modified.
Reuses the audit's independent entropy reference and Java noise convention.
Writes completed cases immediately; existing receipts are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "audit/checks"), str(ROOT / "audit/checks/refs")]

import jpype as jp
import numpy as np
import scipy
from scipy.signal import lfilter

import _kernels_infotheory as reference
from it_kozachenko import jitter
from pyspi.data import Data
from pyspi.utils import is_jpype_jvm_available

# Keep these historical Java probes reproducible after the production port.
# Other shared dependencies are unchanged by the kernel refactor.
BASELINE_COMMIT = "7578137"
baseline_source = subprocess.check_output(
    ["git", "show", f"{BASELINE_COMMIT}:pyspi/statistics/infotheory.py"], cwd=ROOT)
baseline = types.ModuleType("pyspi.statistics._remaining_java_baseline")
sys.modules[baseline.__name__] = baseline
exec(compile(baseline_source, f"{BASELINE_COMMIT}:infotheory.py", "exec"), baseline.__dict__)
JointEntropy, ConditionalEntropy, MutualInfo, TransferEntropy = (
    getattr(baseline, name) for name in
    ("JointEntropy", "ConditionalEntropy", "MutualInfo", "TransferEntropy"))


def signals(channels, samples=1024):
    rng = np.random.default_rng(20261028)
    x = lfilter([1.], [1., -.7], rng.normal(size=(channels, samples)), axis=1)
    if channels > 1:
        x[1, 1:] += .5 * x[0, :-1]
    return (x-x.mean(1, keepdims=True))/x.std(1, ddof=1, keepdims=True)


def timed(fn, repeats=3):
    seconds = []
    for _ in range(repeats):
        t = perf_counter()
        value = fn()
        seconds.append(perf_counter()-t)
    return value, {"median_seconds": float(np.median(seconds)), "seconds": seconds}


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    finite = np.isfinite(a) & np.isfinite(b)
    return {"max_absolute_difference": float(np.max(abs(a[finite]-b[finite]))) if finite.any() else None,
            "allclose": bool(np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)),
            "finite_masks_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(b)))}


class PackedBoxes:
    """One alignment and width; no exclusions, constants or missing data support."""
    def __init__(self, x, width):
        self.n = x.shape[1]
        assert self.n % 64 == 0
        self.widths = width * x.std(1, ddof=1)
        self.bits = np.stack([np.packbits(abs(row[:, None]-row) <= w, axis=1).view(np.uint64)
                              for row, w in zip(x, self.widths, strict=True)])
        self.marginal = np.bitwise_count(self.bits).sum(2)

    def pair_counts(self, i, j):
        return np.bitwise_count(self.bits[i] & self.bits[j]).sum(1)


def packed_simple_family(x):
    """JE/CE width .5 and MI width .25, including each width's preprocessing."""
    p, n = x.shape
    je, ce, mi = (np.full((p, p), np.nan) for _ in range(3))
    box = PackedBoxes(x, .5)
    marginal_h = -np.mean(np.log2(box.marginal / (n * 2 * box.widths[:, None])), axis=1)
    for i in range(p):
        for j in range(i+1, p):
            value = -np.mean(np.log2(box.pair_counts(i,j)/(n*4*box.widths[i]*box.widths[j])))
            je[i,j] = je[j,i] = value
            ce[i,j], ce[j,i] = value-marginal_h[i], value-marginal_h[j]
    memory = box.bits.nbytes + box.marginal.nbytes
    del box
    box = PackedBoxes(x, .25)
    for i in range(p):
        for j in range(i+1, p):
            value = np.mean(np.log2(box.pair_counts(i,j)*n/(box.marginal[i]*box.marginal[j])))
            mi[i,j] = mi[j,i] = value
    return (je, ce, mi), memory


def kozachenko_pair_family(x):
    """Exact k=1 Euclidean trees, audit's seeded jitter; JE/CE sharing."""
    p = len(x)
    je, ce = (np.full((p, p), np.nan) for _ in range(2))
    h = [reference.entropy_kozachenko(jitter(row[:, None])) for row in x]
    for i in range(p):
        for j in range(i+1, p):
            # Existing column-major pair traversal fills JE's unordered cache
            # from (j,i) first. Preserve that seeded-noise layout for comparison.
            value = reference.entropy_kozachenko(jitter(x[[j,i]].T))
            je[i,j] = je[j,i] = value
            ce[i,j], ce[j,i] = value-h[i], value-h[j]
    return je, ce


def java_family(x, kind):
    data = Data(x, dim_order="ps", normalise=False)
    spis = [JointEntropy(estimator=kind), ConditionalEntropy(estimator=kind)]
    if kind == "kernel":
        spis.append(MutualInfo(estimator=kind, kernel_width=.25))
    return tuple(spi.multivariate(data) for spi in spis)


def choose_ksg_embedding(x, exclusion=None):
    calc = jp.JClass("infodynamics.measures.continuous.kraskov.ActiveInfoStorageCalculatorKraskov")(1)
    for name, value in {"k":4, "NUM_THREADS":1, "NOISE_SEED":42, "BIAS_CORRECTION":"false",
                        "AUTO_EMBED_METHOD":"MAX_CORR_AIS", "AUTO_EMBED_K_SEARCH_MAX":10,
                        "AUTO_EMBED_TAU_SEARCH_MAX":4}.items():
        calc.setProperty(name, str(value))
    if exclusion is not None:
        calc.setProperty("DYN_CORR_EXCL", str(exclusion))
    calc.initialise()
    calc.setObservations(jp.JArray.of(x))
    return int(str(calc.getProperty("k_HISTORY"))), int(str(calc.getProperty("TAU")))


def auto_ksg(x, cached=False, dce=False):
    data = Data(x, dim_order="ps", normalise=False)
    settings = {"dyn_corr_excl":"AUTO"} if dce else {}
    baseline = TransferEntropy(estimator="kraskov", auto_embed_method="MAX_CORR_AIS",
                               k_search_max=10, tau_search_max=4, **settings)
    baseline._calc.setProperty("NUM_THREADS", "1")
    out = np.full((len(x),len(x)), np.nan)
    embeddings = np.full((len(x),len(x),4), -1)
    cache = {}
    for i in range(len(x)):
        for j in range(len(x)):
            if i==j: continue
            if cached:
                if dce:
                    baseline._set_theiler_window(data,i,j)
                    exclusion = int(str(baseline._calc.getProperty("DYN_CORR_EXCL")))
                else:
                    exclusion = None
                for c in (i,j):
                    if (c,exclusion) not in cache:
                        cache[c,exclusion] = choose_ksg_embedding(x[c], exclusion)
                l, lt = cache[i,exclusion]
                k, kt = cache[j,exclusion]
                calc = TransferEntropy(estimator="kraskov", k_history=k,k_tau=kt,l_history=l,l_tau=lt, **settings)
                calc._calc.setProperty("NUM_THREADS", "1")
            else:
                calc = baseline
            out[i,j] = calc.bivariate(data,i=i,j=j)
            embeddings[i,j] = [int(str(calc._calc.getProperty(s))) for s in ("k_HISTORY","k_TAU","l_HISTORY","l_TAU")]
    return out, embeddings, len(cache) if cached else 2*len(x)*(len(x)-1), list(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=("kernel","kozachenko","ksg","ksg_dce",
                                          "ksg_dce_heterogeneous","kozachenko_dimensions"), required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    t = perf_counter()
    assert is_jpype_jvm_available()
    startup = perf_counter()-t
    record = {"case":args.case,"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
              "baseline_commit":BASELINE_COMMIT,"baseline_infotheory_sha256":hashlib.sha256(baseline_source).hexdigest(),
              "java":str(jp.JClass("java.lang.System").getProperty("java.version")),
              "java_available_processors":int(jp.JClass("java.lang.Runtime").getRuntime().availableProcessors()),
              "jvm_startup_seconds":startup,"load_average_start":list(os.getloadavg()),
              "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__),ROOT/'pyspi/statistics/infotheory.py',ROOT/'pyspi/lib/jidt/infodynamics.jar',
                     ROOT/'audit/checks/refs/_kernels_infotheory.py',ROOT/'audit/checks/refs/it_kozachenko.py')},
              "limits":"Investigation prototypes, not production validation. Synthetic AR(1) with one lagged cross-channel link; fresh family cache, 1024 samples. Imports/JVM/preprocessing/output excluded. Large candidate runs do not have a new matched Java baseline."}
    if args.case in ("kernel","kozachenko"):
        x = signals(8)
        fn = (lambda x: packed_simple_family(x)[0]) if args.case=="kernel" else kozachenko_pair_family
        java_family(signals(3,128),args.case) # JIT warm-up separate from receipts
        fn(signals(3,128))
        baseline, bt = timed(lambda: java_family(x,args.case))
        candidate, ct = timed(lambda: fn(x))
        record['small'] = {"shape":[8,1024],"baseline":bt,"candidate":ct,"comparison":compare(baseline,candidate)}
        record['small']['speedup'] = bt['median_seconds']/ct['median_seconds']
        args.out.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record['small']),flush=True)
        assert record['small']['comparison']['allclose']
        large_x = signals(118)
        _, ct = timed(lambda: fn(large_x))
        record['large_candidate'] = {"shape":[118,1024],**ct,"seconds_600":ct['median_seconds']*600,
                                     "seconds_41400":ct['median_seconds']*41400}
        if args.case=="kernel":
            record['large_candidate']['one_width_retained_array_mib'] = 118*(1024*128+1024*8)/2**20
    elif args.case == "kozachenko_dimensions":
        spi = JointEntropy(estimator="kozachenko")
        record['dimensions'] = []
        for dim in (1,2,4,10):
            x = signals(dim).T
            spi._entropy_of(x)
            reference.entropy_kozachenko(jitter(x))
            baseline, bt = timed(lambda:spi._entropy_of(x))
            candidate, ct = timed(lambda:reference.entropy_kozachenko(jitter(x)))
            record['dimensions'].append({"dimensions":dim,"samples":1024,"baseline":bt,"candidate":ct,
                                         "comparison":compare(baseline,candidate),
                                         "speedup":bt['median_seconds']/ct['median_seconds']})
    else:
        # New cache experiment; both paths use one Java thread to isolate sharing.
        x = signals(4)
        dce = args.case.startswith("ksg_dce")
        if args.case == "ksg_dce_heterogeneous":
            rng = np.random.default_rng(20261029)
            x = np.array([lfilter([1.],[1.,-a],rng.normal(size=256)) for a in (.2,.5,.8,.95)])
            x = (x-x.mean(1,keepdims=True))/x.std(1,ddof=1,keepdims=True)
        auto_ksg(signals(2,128),cached=True,dce=dce)
        baseline, bt = timed(lambda:auto_ksg(x,dce=dce), repeats=1)
        candidate, ct = timed(lambda:auto_ksg(x,cached=True,dce=dce), repeats=1)
        record['small'] = {"shape":list(x.shape),"baseline":bt,"candidate":ct,
                           "comparison":compare(baseline[0],candidate[0]),
                           "embeddings_equal":bool(np.array_equal(baseline[1],candidate[1])),
                           "baseline_searches":baseline[2],"candidate_searches":candidate[2],
                           "candidate_cache_keys":candidate[3],
                           "candidate_embeddings":candidate[1].tolist(),
                           "speedup":bt['median_seconds']/ct['median_seconds']}
    record['load_average_end'] = list(os.getloadavg())
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k in ('small','large_candidate','dimensions')}),flush=True)


if __name__ == '__main__':
    main()
