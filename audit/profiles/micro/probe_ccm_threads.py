# Probe (not an opportunity benchmark): CPU/wall ratio of one pyEDM.CCM call, the premise of ccm-1.
# Run: OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python audit/profiles/micro/probe_ccm_threads.py
import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, pyEDM
rng = np.random.default_rng(0); N = 256
z = rng.standard_normal((2, N)); z = (z - z.mean(1, keepdims=True)) / z.std(1, keepdims=True)
df = pd.DataFrame(np.concatenate([np.arange(N)[None], z]).T, columns=["index", "proc0", "proc1"])
E = 10; upperE = int(np.floor((N - E - 1) / 10) * 10); lowerE = int(np.ceil(2 * E / 10) * 10); inc = int((upperE - lowerE) / 21)
w0 = time.perf_counter(); c0 = time.process_time()
pyEDM.CCM(dataFrame=df, E=E, columns="proc0", target="proc1", libSizes=f"{lowerE} {upperE} {inc}", sample=100, seed=42)
w = time.perf_counter() - w0; c = time.process_time() - c0
out = {"T": N, "E": E, "wall_s": w, "cpu_s": c, "cpu_over_wall": c / w,
       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
       "pyEDM": "1.15.2.0",
       "note": "one pyEDM.CCM call as issued by pyspi/statistics/causal.py:134-142 (sample=100, seed=42, 22-size libSizes); "
               "cpu/wall ~2 means cppEDM runs the two cross-map directions in two threads (cppEDM CCM.cc:80-90, CCM_THREADED)"}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_ccm_threads.json"), "w"), indent=1)
print(json.dumps(out))
