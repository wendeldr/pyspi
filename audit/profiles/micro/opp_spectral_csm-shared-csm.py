"""spectral_csm-shared-csm: one Multitaper FFT and one coherency per Data, shared by measures and bands.

pyspi caches spectral results per measure (spectral.py:46-50, 64-83) and per band for psi/gd, so
each of cohmag, phase, every psi band and every gd band rebuilds the Multitaper (DPSS + FFT) and the
full two-sided taper-level CSM; psi additionally evaluates coherency twice per band (its failing
default-argument call, connectivity.py:1086-1088, runs before the TypeError fallback). Candidate:
one Multitaper + Connectivity per Data, coherency evaluated once and memoised on the instance
(instance attribute shadows the method), then the SAME library methods (coherence_magnitude,
coherence_phase, phase_slope_index, group_delay) and the same pyspi post-processing per identifier.
Baseline: the pyspi identifiers in sequence on one Data (the calculator's situation).
Candidate code lives only here, never in pyspi/.
Run: .venv/bin/python audit/profiles/micro/opp_spectral_csm-shared-csm.py [shapes...]
     .venv/bin/python audit/profiles/micro/opp_spectral_csm-shared-csm.py --paired [shapes...]
``--paired`` (added in Workflow 4c; default shapes 8x1024 32x1024 118x1024) times psi, gd and phase
per unit in the same run, 3 repeats, and merges the result into the existing JSON under the
top-level key ``paired`` (the ``shapes`` entries and notes are kept; the run is logged under
``merged_runs``). It also times a candidate for the first-insert key alone (phase's first build
stored under the tuple key the later identifiers look up, spectral.py:81) and the wall of gd's
per-pair callback drivers inside the pyspi gd calls; see merge_paired's note.
"""
import sys
import time
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402  (sets OMP_NUM_THREADS=1)
import numpy as np  # noqa: E402
from pyspi._spectral_compat import spectral_connectivity as sc  # noqa: E402

BANDS = [("0", "0-5", 0.0, 0.5), ("0", "0-25", 0.0, 0.25), ("0-25", "0-5", 0.25, 0.5)]
IDS = (["cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5", "phase_multitaper_mean_fs-1_fmin-0_fmax-0-5"]
       + [f"psi_multitaper_mean_fs-1_fmin-{a}_fmax-{b}" for a, b, _, _ in BANDS]
       + [f"gd_multitaper_delay_fs-1_fmin-{a}_fmax-{b}" for a, b, _, _ in BANDS])


def baseline(make):
    data, out = make(), {}
    t0 = time.perf_counter()
    for ident in IDS:
        s = B.spi(ident)
        B.h.seed_all(B.SPI_SEED)
        out[ident] = B.h.compute_spi(s, data)
    return time.perf_counter() - t0, out


def band_stat(adj_freq, freq, fmin, fmax, fn):      # pyspi spectral.py:87-101 (mean/max path)
    fid = np.where((freq >= fmin) * (freq <= fmax))[0]
    adj = fn(adj_freq[0, fid, :, :], axis=0)
    np.fill_diagonal(adj, np.nan)
    return adj


def candidate(make):
    data = make()
    t0 = time.perf_counter()
    z = np.transpose(data.to_numpy(squeeze=True))
    conn = sc.Connectivity.from_multitaper(sc.Multitaper(z, sampling_frequency=1))
    coh = conn.coherency()
    conn.coherency = lambda: coh                    # memoised: every method below reuses it
    seg = {"fourier_and_coherency_once": time.perf_counter() - t0}
    freq = conn.frequencies
    out = {IDS[0]: band_stat(conn.coherence_magnitude(), freq, 0, 0.5, np.nanmean),
           IDS[1]: band_stat(conn.coherence_phase(), freq, 0, 0.5, np.nanmean)}
    for (a, b, lo, hi) in BANDS:
        kw = dict(frequencies_of_interest=[lo, hi], frequency_resolution=(hi - lo) / 50)
        t1 = time.perf_counter()
        p = conn.phase_slope_index(**kw)[0]
        np.fill_diagonal(p, np.nan)
        out[f"psi_multitaper_mean_fs-1_fmin-{a}_fmax-{b}"] = p
        t2 = time.perf_counter()
        g = conn.group_delay(**kw)[0][0]
        np.fill_diagonal(g, np.nan)
        out[f"gd_multitaper_delay_fs-1_fmin-{a}_fmax-{b}"] = g
        seg[f"psi_band_{a}_{b}"] = t2 - t1
        seg[f"gd_band_{a}_{b}"] = time.perf_counter() - t2
    candidate.segments = seg
    return time.perf_counter() - t0, out


def run(shape):
    _, make = B.data(shape)
    reps = 1 if shape.startswith("118") else 3
    baseline(B.data("8x256")[1]); candidate(B.data("8x256")[1])          # warm-up
    tb = min(baseline(make)[0] for _ in range(reps - 1)) if reps > 1 else None
    t_b, S_b = baseline(make)
    tb = t_b if tb is None else min(tb, t_b)
    runs = []
    for _ in range(reps):
        r = candidate(make)
        runs.append((r[0], r[1], dict(candidate.segments)))
    tc, S_c, segs = min(runs, key=lambda r: r[0])
    per_id = {i: B.diff(S_b[i], S_c[i]) for i in IDS}
    res = dict(baseline_s=tb, candidate_s=tc, speedup=tb / tc, identifiers=IDS, per_identifier=per_id,
               all_bit_identical=all(v["bit_identical"] for v in per_id.values()),
               all_nan_mask_equal=all(v["nan_mask_equal"] for v in per_id.values()),
               candidate_segments_s=segs)
    print(shape, round(tb, 3), round(tc, 3), round(tb / tc, 2), res["all_bit_identical"], res["all_nan_mask_equal"], flush=True)
    return res


# --- --paired mode (Workflow 4c): same-run, per-unit figures for psi, gd and phase ---
PHASE_IDS = [f"phase_multitaper_mean_fs-1_fmin-{a}_fmax-{b}" for a, b, _, _ in BANDS]
PSI_IDS = [f"psi_multitaper_mean_fs-1_fmin-{a}_fmax-{b}" for a, b, _, _ in BANDS]
GD_IDS = [f"gd_multitaper_delay_fs-1_fmin-{a}_fmax-{b}" for a, b, _, _ in BANDS]


class CallbackTimer:
    """Wall of gd's two per-pair callback drivers (np.ma.apply_along_axis -> linregress at
    connectivity.py:965-967; np.apply_along_axis -> _find_largest_independent_group at :1476),
    measured by wrapping the two numpy entry points (one extra Python call per band, no profiler)."""

    def __enter__(self):
        self.s = {"ma_apply_along_axis": 0.0, "np_apply_along_axis": 0.0}
        self._orig = (np.ma.apply_along_axis, np.apply_along_axis)
        self._depth = 0

        def wrap(name, fn):
            def w(*a, **k):
                self._depth += 1
                t0 = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    self._depth -= 1
                    if self._depth == 0:
                        self.s[name] += time.perf_counter() - t0
            return w
        np.ma.apply_along_axis = wrap("ma_apply_along_axis", self._orig[0])
        np.apply_along_axis = wrap("np_apply_along_axis", self._orig[1])
        return self

    def __exit__(self, *exc):
        np.ma.apply_along_axis, np.apply_along_axis = self._orig

    def take(self):
        v = sum(self.s.values())
        self.s = {k: 0.0 for k in self.s}
        return v


def seq(ids, data, timer=None, alias_first=False):
    """pyspi identifiers in sequence on one Data: (walls, outputs, callback walls).
    alias_first: after the first call, also store its cached array under the tuple key the
    later identifiers look up (what keying the first insert by self.key at spectral.py:81 would do)."""
    walls, out, cbs = [], {}, []
    for i, ident in enumerate(ids):
        s = B.spi(ident)
        B.h.seed_all(B.SPI_SEED)
        if timer is not None:
            timer.take()
        t0 = time.perf_counter()
        out[ident] = B.h.compute_spi(s, data)
        walls.append(time.perf_counter() - t0)
        cbs.append(timer.take() if timer is not None else None)
        if alias_first and i == 0:
            data.spectral_mv[s.key] = data.spectral_mv[s.measure]
    return walls, out, cbs


def shared_all(data, timer):
    """One Multitaper + one memoised coherency; phase, psi and gd bands read it (segment walls)."""
    t0 = time.perf_counter()
    z = np.transpose(data.to_numpy(squeeze=True))
    conn = sc.Connectivity.from_multitaper(sc.Multitaper(z, sampling_frequency=1))
    coh = conn.coherency()
    conn.coherency = lambda: coh
    seg, out = {"build": time.perf_counter() - t0}, {}
    freq = conn.frequencies
    t1 = time.perf_counter()
    ph = conn.coherence_phase()
    for ident, (_, _, lo, hi) in zip(PHASE_IDS, BANDS):
        out[ident] = band_stat(ph, freq, lo, hi, np.nanmean)
    seg["phase_3_bands"] = time.perf_counter() - t1
    seg["psi"], seg["gd"], seg["gd_callbacks"] = [], [], []
    for ident_p, ident_g, (_, _, lo, hi) in zip(PSI_IDS, GD_IDS, BANDS):
        kw = dict(frequencies_of_interest=[lo, hi], frequency_resolution=(hi - lo) / 50)
        t1 = time.perf_counter()
        p = np.array(conn.phase_slope_index(**kw)[0], dtype=float, copy=True)
        np.fill_diagonal(p, np.nan)
        out[ident_p] = p
        seg["psi"].append(time.perf_counter() - t1)
        timer.take()
        t1 = time.perf_counter()
        g = np.array(conn.group_delay(**kw)[0][0], dtype=float, copy=True)
        np.fill_diagonal(g, np.nan)
        out[ident_g] = g
        seg["gd"].append(time.perf_counter() - t1)
        seg["gd_callbacks"].append(timer.take())
    return seg, out


def paired(shape, reps=3):
    _, make = B.data(shape)
    _, mk_small = B.data("8x256")
    with CallbackTimer() as tm:
        for ids in (PSI_IDS, GD_IDS, PHASE_IDS):     # warm-up
            seq(ids, mk_small(), tm)
        seq(PHASE_IDS, mk_small(), tm, alias_first=True)
        shared_all(mk_small(), tm)
        R, diffs = [], {}
        for rep in range(reps):
            r = {}
            r["psi_base"], S_psi, _ = seq(PSI_IDS, make(), tm)
            r["gd_base"], S_gd, r["gd_base_callbacks"] = seq(GD_IDS, make(), tm)
            r["phase_base"], S_ph, _ = seq(PHASE_IDS, make(), tm)
            r["phase_first_insert_fixed"], S_fix, _ = seq(PHASE_IDS, make(), tm, alias_first=True)
            r["shared"], S_sh = shared_all(make(), tm)
            if rep == 0:
                for i in PSI_IDS:
                    diffs[f"{i}__shared"] = B.diff(S_psi[i], S_sh[i])
                for i in GD_IDS:
                    d = B.diff(S_gd[i], S_sh[i])
                    d["all_nan"] = bool(np.all(np.isnan(np.asarray(S_gd[i], dtype=float)[~np.eye(len(S_gd[i]), dtype=bool)])))
                    diffs[f"{i}__shared"] = d
                for i in PHASE_IDS:
                    diffs[f"{i}__shared"] = B.diff(S_ph[i], S_sh[i])
                    diffs[f"{i}__first_insert_fixed"] = B.diff(S_ph[i], S_fix[i])
            sh = r["shared"]
            r["ratios"] = dict(
                psi_later_consumer=sum(r["psi_base"]) / sum(sh["psi"]),
                psi_first_consumer=sum(r["psi_base"]) / (sh["build"] + sum(sh["psi"])),
                gd_later_consumer=sum(r["gd_base"]) / sum(sh["gd"]),
                gd_callback_share_base=sum(r["gd_base_callbacks"]) / sum(r["gd_base"]),
                gd_callback_amdahl_base=1 / (1 - sum(r["gd_base_callbacks"]) / sum(r["gd_base"])),
                phase_first_insert_fixed=sum(r["phase_base"]) / sum(r["phase_first_insert_fixed"]),
                phase_shared_first_consumer=sum(r["phase_base"]) / (sh["build"] + sh["phase_3_bands"]),
                phase_shared_later_consumer=sum(r["phase_base"]) / sh["phase_3_bands"])
            print(shape, rep, {k: round(v, 3) for k, v in r["ratios"].items()}, flush=True)
            R.append(r)
    keys = R[0]["ratios"].keys()
    summ = {k: [min(r["ratios"][k] for r in R), max(r["ratios"][k] for r in R)] for k in keys}
    return dict(reps=R, ratio_range=summ, diffs=diffs,
                all_bit_identical=all(v["bit_identical"] for v in diffs.values()),
                all_nan_mask_equal=all(v["nan_mask_equal"] for v in diffs.values()))


def merge_paired(shapes):
    import json
    import os
    path = B.MICRO / "opp_spectral_csm-shared-csm.json"
    doc = json.loads(path.read_text())
    load_start = os.getloadavg()
    for shape in shapes:
        doc.setdefault("paired", {})[shape] = paired(shape)
    doc.setdefault("merged_runs", []).append(dict(
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), mode="--paired", shapes=list(shapes),
        loadavg_start=list(load_start), loadavg_end=list(os.getloadavg()),
        note=("--paired mode (Workflow 4c), results under the top-level key 'paired' (the 'shapes' entries are "
              "kept). Each repeat, on a fresh Data each: psi's 3 config bands in sequence (pyspi), gd's 3 bands "
              "(pyspi, with the per-pair callback drivers timed by wrapping np.ma.apply_along_axis and "
              "np.apply_along_axis, no profiler), phase's 3 bands (pyspi: first build, first-insert duplicate, "
              "read), phase's 3 bands with the first insert also stored under the tuple key (candidate for the "
              "key fix at spectral.py:81), then one shared Multitaper + coherency read by phase, psi and gd "
              "(segment walls). ratios: psi_later_consumer = psi bands / psi segments without the build; "
              "psi_first_consumer adds the build; gd_later_consumer likewise; gd_callback_amdahl_base = "
              "1/(1 - callback share of the pyspi gd bands); phase_first_insert_fixed = pyspi phase bands / "
              "key-fixed bands; phase_shared_first_consumer = pyspi phase bands / (build + phase from shared). "
              "Run while other audit agents shared the machine (loadavg recorded). Agreement with the baseline "
              "means the candidate reproduces current output, nothing more.")))
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)
    return path


if __name__ == "__main__" and "--paired" in sys.argv:
    print(merge_paired([a for a in sys.argv[1:] if a != "--paired"] or ["8x1024", "32x1024", "118x1024"]))
elif __name__ == "__main__":
    shapes = sys.argv[1:] or ["32x1024", "118x1024"]
    res = {s: run(s) for s in shapes}
    p = B.write("spectral_csm-shared-csm", res, notes=(
        "Baseline = 8 pyspi identifiers (cohmag, phase full band; psi and gd x 3 config bands) in sequence on "
        "one Data through harness.compute_spi; candidate = one Multitaper/Connectivity and one coherency "
        "per Data feeding the same library methods. candidate_segments_s = wall of each candidate segment "
        "(Multitaper + Connectivity + the one coherency; then each psi and gd band reading it) in the fastest "
        "candidate repeat. Timings are single passes at 118x1024 (min of 3 below). "
        "Agreement means the candidate reproduces the current output, nothing more."),
        threads=dict(OMP_NUM_THREADS=1, NUMBA_NUM_THREADS="n/a (no numba code)", JIDT="n/a"))
    print(p)
