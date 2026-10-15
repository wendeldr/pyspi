# Probe (not an opportunity benchmark): unprofiled share of mne_connectivity combine('mean') in the
# pec call path (pyspi/statistics/misc.py:168-174) at 118x1024; cProfile inflates combine's
# per-connection np.mean calls, so pec-1's Amdahl bound is taken from these wall shares.
# Run: OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python audit/profiles/micro/probe_pec_combine.py
import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, mne_connectivity as mnec
rng = np.random.default_rng(0); P, T = 118, 1024
x = rng.standard_normal((P, T)); x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
z = x[None]  # (1 epoch, P, T) as np.moveaxis(data.to_numpy(), 2, 0)
def best(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r
out = {"P": P, "T": T, "repeats": 5, "stat": "min",
       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
       "mne_connectivity": mnec.__version__, "variants": {}}
for orth in (False, "pairwise"):
    for log in (False, True):
        t_env, con = best(lambda: mnec.envelope_correlation(z, orthogonalize=orth, log=log, absolute=False))
        t_comb, c2 = best(lambda: con.combine("mean"))
        t_get, _ = best(lambda: c2.get_data(output="dense"))
        tot = t_env + t_comb + t_get
        out["variants"][f"orth-{orth}_log-{log}"] = {"envelope_s": t_env, "combine_s": t_comb, "get_data_s": t_get,
            "combine_share": t_comb / tot, "amdahl_bound_x": 1 / (1 - t_comb / tot)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_pec_combine.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
