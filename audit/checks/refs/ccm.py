"""References for the ccm family (causal.ccm.E-None, causal.ccm.E-1, causal.ccm.E-10): COMPARISON venv.

Writes ``audit/checks/out/ref/<run_id>.npz`` for every run of ``manifest/ccm.json`` (contract in ``compare.py``).
Nothing here imports pyEDM / cppEDM (the library pyspi delegates to); the building blocks are numpy and the
self-tested kernel ``_kernels_distance`` (simplex_crossmap / simplex_predictions / ccm_skill_curve, written from
Sugihara & May 1990 and Sugihara et al. 2012), so every numeric reference is ``independent-impl`` or ``analytic``.

Run kinds (the last ``|`` field of the run_id is the tag; probes live in ``probes/ccm_probes.py``):

* ``xmap-E{E}-L{L}[-X{X}]`` (kernel path, probe ``kernel_xmap``: pyEDM.CCM with random=False and one library
  size): value[i, j] = kd.simplex_crossmap(x_i, x_j, E, lib = first L valid rows) (contiguous library, cppEDM
  CCM.cc L289-325); exact at 1e-10 / 1e-12. With ``X`` (exclusion radius) the reference is ``XMap`` below
  (vectorised form of the kernel arithmetic plus the cppEDM rule |pred row - lib row| <= X excluded,
  EDM_Neighbors.cc L173-178), validated against the kernel at X = 0 in the same run.
* ``tied-E{E}-Lfull`` (logistic_maps, the period-4 tied-data fixture): qualitative. The tie-free column must
  match the kernel exactly and the tied column must carry tied prediction rows; the pyEDM - kernel gap of the
  tied column is recorded (it depends on std::sort order among equal distances, see the kernel docstring).
* ``affine-E{E}-Lfull``: pyEDM.CCM on raw scaled_offset (processes 2, 3: scale 5 and 10, offsets -2 and 5) vs
  the kernel on the same draw of bivariate_gaussian (analytic invariance; tolerance 1e-10 / 1e-12).
* ``embed-ENone``: (P, 11) = [E pyspi passed to pyEDM.CCM, rho(E = 1..10) of pyEDM.EmbedDimension]. Reference:
  column 0 = 10 (the code's rule embed_df.max()["E"], the maximum of the E column; known bug 2), columns 1..10 =
  from-scratch simplex forecast (Tp = 1) with cppEDM's EmbedDimension conventions found here: lib = pred =
  "10 N-10" (1-based) -> prediction rows 9..N-11 (0-based) forecasting rows 10..N-10, library rows
  9 + (E-1) .. N-12 (partial-embedding rows at the library start removed, and the last row removed because its
  Tp = 1 target is outside the library, EDM_Neighbors.cc "reach exceeding grasp"). Exact.
* ``sched-E{E}``: the library-size schedule pyspi asks for and cppEDM expands (analytic; causal.py:127-131,
  cppEDM Parameter.cc L198-291): lowerE = ceil(2E/10)*10, upperE = floor((T-E-1)/10)*10,
  inc = int((upperE-lowerE)/21), sizes lowerE, lowerE+inc, ... <= upperE, first 22 kept. Exact.
* ``short-E{E}-T{T}``: qualitative; predicted status from the same schedule (inc < 1 -> cppEDM throws).
* ``diag-E{E}``: qualitative; a direct multivariate call returns exactly 0 on the diagonal.
* ``stat-{mean|max|diff}`` (plain pyspi SPI, sample = 100, seed = 42, pyspi's library grid) and ``stats3-E{E}``
  (probe: the three statistics on one Data object, stacked (3, P, P)): statistical. The reference draws 500 own
  random libraries per library size (numpy RNG, without replacement, the same library for both directions as
  cppEDM does: one std::default_random_engine(seed) per direction, CCM.cc L203), computes rho in both
  directions with ``XMap`` on pyspi's grid, and then bootstraps pyspi's estimator: 4000 resamples of 100 of the
  500 draws (with replacement, independently per library size) give the distribution of mean_L, max_L and
  mean_L(diff) of a 100-library curve. value = bootstrap mean; se = bootstrap sd * sqrt(1 + 100/500) (pyspi's
  own sampling spread plus the reference's). Pass if |pyspi - value| <= 2 se.
* ``stat-mean-docE`` (E-None): as ``stat-mean`` but with the DOCUMENTED E = max over the pair of the
  argmax-rho E of the simplex forecast table (supplement 1.3.5 p.9); expected to fail (known bug 2).
* ``curve-E{E}``: pyspi's cached curve data.ccm[E] (P, P, 22) vs the 500-draw mean per library size;
  qualitative with a pre-declared rule: max |z| <= 4 over the 2 x 22 entries and chi-square(44) p > 0.001,
  se per entry = sd_L * sqrt(1/100 + 1/500).
* ``dir-diff``: direction; expected sign of diff from the known coupling, using the CCM convention (the
  EFFECT's embedding cross-maps the CAUSE): A[i, j] = rho(x_j | M_{x_i}), so a cause j -> i makes A[i, j] large
  and diff[i, j] > 0. logistic_maps_chaotic / _sugihara / var2_bivariate (cause 0 -> effect 1): diff[1, 0] > 0;
  logistic_maps_bidir (y -> x stronger): diff[0, 1] > 0. ``dir-diff-swap`` runs the same data with
  processes = [1, 0] (y first), so the expected cell moves to diff[0, 1]: both orientations of the stored matrix.
* ``static-mean`` (E-1, nonlinear_dependence processes 0, 1: x1 = x0^2 + noise, iid samples): qualitative with the
  pre-declared rule A[0, 1] > 0.7 and |A[1, 0]| < 0.2 (E = 1 scores static functional dependence).
* ``seeds-E{E}-S16`` (probe ``seeds``): bias test. pyEDM.CCM called as pyspi calls it but with seeds 1..16; the
  mean over library sizes averaged over the 16 seeds (1600 libraries per size) vs 2000 own draws per size;
  statistical with the analytic se sqrt(sum_L var_L / 22^2 * (1/1600 + 1/2000)).
* ``agg-E{E}`` (probe ``aggregate``: (P, P, 25) = mean, max, diff and the 22-entry cached curve data.ccm[E] from
  one Data object): exact (analytic). value = [nanmean(c, 2), nanmax(c, 2), nanmean(c - c^T, 2), c] with c read
  from pyspi's own output, i.e. the aggregation step causal.py:152-163 recomputed in numpy (the curve itself is
  checked by the ``curve`` and ``stat`` runs); extra keys: curve length and max|diff - (mean - mean^T)|.
* ``eqE10-mean`` (E-None): pyspi self-consistency (same-lineage): ccm_E-None_mean bitwise equal to the E-10
  ``stat-mean`` run on the same data (known bug 2).

pyspi-side inputs: pipeline runs z-score with ddof = 1 (data.py:193-195); CCM is invariant to the per-process
affine map (kNN sets and exp(-d/d1) weights are scale-free, Pearson rho is affine invariant), which the
``affine`` runs confirm on the delegate. Bootstrap draws are cached in ``out/exchange/ccm/``.
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_distance as kd  # noqa: E402
from scipy.signal import detrend  # noqa: E402
from scipy.stats import chi2, zscore  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / "ccm"
MANIFEST = CHECKS / "manifest" / "ccm.json"
ME = "audit/checks/refs/ccm.py"
KD = "audit/checks/refs/_kernels_distance.py"
N_DRAWS, N_PYSPI, N_BOOT = 500, 100, 4000
# cause -> effect pairs (CONVENTIONS.md D5/D8): process 0 = x, process 1 = y
CAUSE_EFFECT = {"logistic_maps_chaotic": (0, 1), "logistic_maps_sugihara": (0, 1), "var2_bivariate": (0, 1),
                "logistic_maps_bidir": (1, 0)}


# ------------------------------------------------------------------------------------------------ helpers
def raw(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
    return x if run.get("processes") is None else x[list(run["processes"])]


def prep(x, path):
    """pyspi Data.set_data (data.py:184-195): optional scipy.signal.detrend, then zscore(ddof=1)."""
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return x
    if path == "pipeline-detrend":
        x = detrend(x, axis=1)
    return zscore(x, axis=1, ddof=1)


def pyspi_value(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64), np.asarray(z["input"], dtype=np.float64)


def tag_of(rid):
    t = rid.split("|")[-1]
    kind, *rest = t.split("-")
    return kind, {tok[0]: tok[1:] for tok in rest if tok and tok[0] in "ELXTS"}, rest


def write(rid, value, criterion, kind, detail, notes="", **extra):
    d = dict(value=np.asarray(value, dtype=np.float64), criterion=np.array(criterion),
             reference_kind=np.array(kind), reference_detail=np.array(detail), notes=np.array(notes))
    for k, v in extra.items():
        d[k] = np.asarray(v)
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{rid}.npz", **d)


def schedule(T, E):
    """pyspi causal.py:127-131 + cppEDM Parameter.cc L198-291 expansion; (lo, hi, inc, all sizes, kept sizes)."""
    hi = int(np.floor((T - E - 1) / 10) * 10)
    lo = int(np.ceil(2 * E / 10) * 10)
    inc = int((hi - lo) / 21)
    sizes = list(range(lo, hi + 1, inc)) if inc >= 1 else []
    return lo, hi, inc, sizes, sizes[:22]


def E_of_identifier(ident):
    e = ident.split("_")[1][2:]
    return None if e == "None" else int(e)


# ------------------------------------------------------------------------------------- vectorised cross-map
class XMap:
    """Simplex cross-map of y from the delay embedding of x (tau = -1, Tp = 0), the arithmetic of
    kd.simplex_predictions (coordinate-wise squared differences summed in order j = 0..E-1, sqrt; the E+1 nearest
    library rows with the prediction row itself excluded; u = exp(-d/d1), or [d == 0] if d1 = 0, floored at 1e-6;
    Pearson rho with 0 for zero variance) with the full distance matrix precomputed once. ``excl`` > 0 also drops
    library rows with |pred row - lib row| <= excl (cppEDM exclusionRadius, EDM_Neighbors.cc L173-178)."""

    def __init__(self, x, y, E, excl=0):
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        self.E, self.k, self.excl = E, E + 1, int(excl)
        self.rows = np.arange(E - 1, len(x))
        emb = np.stack([x[self.rows - j] for j in range(E)], 1)
        m = len(self.rows)
        ss = np.zeros((m, m))
        for j in range(E):
            ss += (emb[None, :, j] - emb[:, None, j]) ** 2      # [a, b] = (lib b - pred a)^2 as in the kernel
        self.D = np.sqrt(ss)
        self.yobs = y[self.rows]
        self.m = m

    def rho(self, lib, stable=False, return_tied=False):
        lib = np.sort(np.asarray(lib))
        Dsub = self.D[:, lib].copy()
        Dsub[lib, np.arange(len(lib))] = np.inf                  # leave-one-out
        if self.excl:
            Dsub[np.abs(np.arange(self.m)[:, None] - lib[None, :]) <= self.excl] = np.inf
        k = self.k
        if stable:
            o = np.argsort(Dsub, axis=1, kind="stable")[:, :k + 1]
        else:
            part = np.argpartition(Dsub, k, axis=1)[:, :k + 1]
            dp = np.take_along_axis(Dsub, part, 1)
            o = np.take_along_axis(part, np.argsort(dp, axis=1, kind="stable"), 1)
        d = np.take_along_axis(Dsub, o, 1)
        tied = d[:, k - 1] == d[:, k]
        dk = d[:, :k]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(dk[:, :1] > 0, np.exp(-dk / dk[:, :1]), (dk == 0).astype(float))
        w = np.maximum(w, 1e-6)
        yl = self.yobs[lib][o[:, :k]]
        pred = (w * yl).sum(1) / w.sum(1)
        r = kd._pearson0(self.yobs, pred)
        return (r, int(tied.sum())) if return_tied else r


def validate_xmap(x, y, E, rng, n=3):
    """max |XMap - kd.simplex_crossmap| over n random libraries (and the full library)."""
    X = XMap(x, y, E)
    libs = [np.arange(X.m)] + [rng.choice(X.m, min(X.m, 60), replace=False) for _ in range(n)]
    return max(abs(X.rho(l) - kd.simplex_crossmap(x, y, E, lib_indices=X.rows[np.sort(l)])) for l in libs)


# ---------------------------------------------------------------------------------- bootstrap of pyspi's CCM
def draws(z, E, T, key):
    """R (N_DRAWS, n_sizes, 2): rho of 'proc0:proc1' (x0 xmap x1 = A[0, 1]) and 'proc1:proc0' (A[1, 0]) for
    N_DRAWS random libraries per kept library size, one library shared by both directions. Cached."""
    EXCH.mkdir(parents=True, exist_ok=True)
    f = EXCH / f"draws_{key}.npz"
    sizes = schedule(T, E)[4]
    if f.is_file():
        with np.load(f) as c:
            if list(c["sizes"]) == sizes and np.array_equal(c["z"], z):
                return c["R"], sizes, float(c["validate"]), float(c["seconds"])
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    t0 = time.time()
    X01, X10 = XMap(z[0], z[1], E), XMap(z[1], z[0], E)
    val = max(validate_xmap(z[0], z[1], E, rng), validate_xmap(z[1], z[0], E, rng))
    R = np.empty((N_DRAWS, len(sizes), 2))
    for li, L in enumerate(sizes):
        for s in range(N_DRAWS):
            lib = rng.choice(X01.m, L, replace=False)
            R[s, li, 0], R[s, li, 1] = X01.rho(lib), X10.rho(lib)
    dt = time.time() - t0
    np.savez(f, R=R, sizes=np.array(sizes), z=z, validate=val, seconds=dt)
    return R, sizes, val, dt


def boot_stats(R, seed):
    """Distribution of pyspi's three statistics for a curve averaged over N_PYSPI libraries per size."""
    rng = np.random.default_rng(seed)
    nL = R.shape[1]
    out = {"mean": np.empty((N_BOOT, 2)), "max": np.empty((N_BOOT, 2)), "diff": np.empty((N_BOOT, 2))}
    for b0 in range(0, N_BOOT, 500):
        nb = min(500, N_BOOT - b0)
        idx = rng.integers(0, R.shape[0], (nb, nL, N_PYSPI))
        c = np.stack([R[idx[:, l, :], l, :].mean(1) for l in range(nL)], 1)   # (nb, nL, 2)
        out["mean"][b0:b0 + nb] = c.mean(1)
        out["max"][b0:b0 + nb] = c.max(1)
        dd = (c[:, :, 0] - c[:, :, 1]).mean(1)
        out["diff"][b0:b0 + nb] = np.stack([dd, -dd], 1)
    return out


def to_matrix(v01, v10):
    A = np.full((2, 2), np.nan)
    A[0, 1], A[1, 0] = v01, v10
    return A


def stat_reference(run, E, key_extra=""):
    x = prep(raw(run), run["path"])
    assert x.shape[0] == 2, "ccm statistical references are written for P = 2"
    key = f"{run['dataset']}_T{run['T']}_E{E}_{run['path']}_{'-'.join(map(str, run.get('processes') or []))}"
    R, sizes, val, dt = draws(x, E, run["T"], key)
    bs = boot_stats(R, zlib.crc32((key + key_extra).encode()))
    val_m = {s: (to_matrix(*bs[s].mean(0)), to_matrix(*bs[s].std(0, ddof=1))) for s in bs}
    return x, R, sizes, val, dt, val_m


# ----------------------------------------------------------------------------------------------- run kinds
def run_xmap(run, rid, p):
    E, Ls, X = int(p["E"]), p.get("L", "full"), int(p.get("X", 0))
    z = prep(raw(run), run["path"])
    P, N = z.shape
    L = N - (E - 1) if Ls == "full" else int(Ls)
    valid = np.arange(E - 1, N)[:L]
    V, ties = np.full((P, P), np.nan), np.zeros((P, P))
    S, _ = pyspi_value(rid)
    if X == 0:
        for i in range(P):
            for j in range(P):
                if i != j:
                    V[i, j], ties[i, j] = kd.simplex_crossmap(z[i], z[j], E, lib_indices=valid, return_tied=True)
        detail = f"{KD}:simplex_crossmap (contiguous library of the first L = {L} valid rows, E = {E})"
        extra = {}
    else:
        V0 = np.full((P, P), np.nan)
        for i in range(P):
            for j in range(P):
                if i != j:
                    V[i, j], ties[i, j] = XMap(z[i], z[j], E, excl=X).rho(np.arange(L), stable=True, return_tied=True)
                    V0[i, j] = XMap(z[i], z[j], E).rho(np.arange(L), stable=True)
        chk = max(abs(V0[i, j] - kd.simplex_crossmap(z[i], z[j], E, lib_indices=valid))
                  for i in range(P) for j in range(P) if i != j)
        detail = f"{ME}:XMap(excl={X}) (kernel arithmetic + cppEDM exclusionRadius rule), validated vs {KD}:simplex_crossmap at excl=0"
        extra = dict(xmap_excl0_vs_kernel_max_abs=chk, rho_excl0_01=V0[0, 1], rho_excl0_10=V0[1, 0],
                     rho_excl_01=V[0, 1], rho_excl_10=V[1, 0], gap_01=V0[0, 1] - V[0, 1], gap_10=V0[1, 0] - V[1, 0])
    notes = (f"A[i,j] = pyEDM.CCM(columns=proc_i, target=proc_j, E={E}, libSizes='{L}', random=False"
             f"{', exclusionRadius=%d' % X if X else ''}) column 'proc_i:proc_j' = rho(x_j | M_x_i); tied prediction "
             f"rows per entry {ties[~np.isnan(V)].astype(int).tolist()}")
    write(rid, V, "exact", "independent-impl", detail, notes, rtol=1e-10, atol=1e-12,
          n_tied_rows_total=ties[~np.isnan(V)].sum(), max_abs_vs_pyEDM=np.nanmax(np.abs(S - V)), **extra)


def run_tied(run, rid, p):
    E = int(p["E"])
    z = prep(raw(run), run["path"])
    N = z.shape[1]
    valid = np.arange(E - 1, N)
    S, _ = pyspi_value(rid)
    r01, t01 = kd.simplex_crossmap(z[0], z[1], E, lib_indices=valid, return_tied=True)
    r10, t10 = kd.simplex_crossmap(z[1], z[0], E, lib_indices=valid, return_tied=True)
    ok_free = abs(S[0, 1] - r01) <= 1e-12 + 1e-10 * abs(r01) and t01 == 0
    ok = bool(ok_free and t10 > 0)
    write(rid, to_matrix(r01, r10), "qualitative", "independent-impl",
          f"{KD}:simplex_crossmap (stable-argsort E+1 neighbours, full contiguous library)",
          f"logistic_maps (y period-4, y -> x): tie-free entry A[0,1] (embed x) must match exactly and A[1,0] (embed the "
          f"period-4 y) has ties at the (E+1)-th distance on {t10} of {N - E + 1} rows, so pyEDM's value depends on "
          f"std::sort order and no order-independent reference reproduces it; pass = tie-free exact AND tied rows > 0",
          pass_flag=int(ok), pyEDM_01=S[0, 1], kernel_01=r01, tied_rows_01=t01, pyEDM_10=S[1, 0], kernel_10=r10,
          tied_rows_10=t10, tied_gap_10=S[1, 0] - r10)


def run_affine(run, rid, p):
    E = int(p["E"])
    base = dict(run, dataset="bivariate_gaussian")
    z = raw(base)
    N = z.shape[1]
    valid = np.arange(E - 1, N)
    V = to_matrix(kd.simplex_crossmap(z[0], z[1], E, lib_indices=valid),
                  kd.simplex_crossmap(z[1], z[0], E, lib_indices=valid))
    write(rid, V, "tolerance", "analytic",
          f"analytic invariance; value = {KD}:simplex_crossmap on raw bivariate_gaussian processes {run['processes']} "
          f"(same draw as scaled_offset before scale/offset), full library, E = {E}",
          "pyspi side: pyEDM.CCM on raw scaled_offset (scale 5, 10; offset -2, 5); positive per-process affine maps "
          "leave kNN sets, exp(-d/d1) weights and Pearson rho unchanged", rtol=1e-10, atol=1e-12)


def embed_table(z, N):
    """rho(E = 1..10) of pyEDM.EmbedDimension(lib = pred = '10 N-10', Tp = 1) (conventions in the module doc)."""
    lo, hi = 10, N - 10
    out = []
    for E in range(1, 11):
        t, pred, tied = kd.simplex_predictions(z, z, E, Tp=1, lib_indices=np.arange(lo - 1 + E - 1, hi - 1))
        m = (t >= lo) & (t <= hi)
        out.append((kd._pearson0(z[t[m]], pred[m]), int(tied[m].sum())))
    return np.array([o[0] for o in out]), sum(o[1] for o in out)


def run_embed(run, rid, p):
    z = prep(raw(run), run["path"])
    P, N = z.shape
    V = np.full((P, 11), np.nan)
    argmax, ties = [], 0
    for q in range(P):
        r, t = embed_table(z[q], N)
        V[q, 1:], V[q, 0] = r, 10.0
        argmax.append(int(np.argmax(r)) + 1)
        ties += t
    S, _ = pyspi_value(rid)
    write(rid, V, "exact", "independent-impl",
          f"{ME}:embed_table via {KD}:simplex_predictions (Tp=1, lib rows 9+(E-1)..N-12, pred rows 9..N-11); column 0 "
          f"analytic: the code's rule embed_df.max()['E'] = 10 (known bug 2)",
          f"rows = processes; col 0 = E pyspi passed to pyEDM.CCM, cols 1..10 = EmbedDimension rho(E); argmax-rho E per "
          f"process {argmax}, documented E for the pair = max = {max(argmax)}; tied rows in the table {ties}",
          rtol=1e-10, atol=1e-12, argmax_rho_E=np.array(argmax), documented_E=max(argmax), pyspi_E=S[0, 0])


def run_sched(run, rid, p):
    E = 10 if p["E"] == "None" else int(p["E"])
    lo, hi, inc, sizes, kept = schedule(run["T"], E)
    V = np.concatenate([[E, lo, hi, inc, len(sizes), sizes[-1], len(kept)],
                        np.pad(np.array(kept, float), (0, 22 - len(kept)), constant_values=np.nan)])
    write(rid, V, "exact", "analytic",
          "causal.py:127-131 (lowerE, upperE, inc = int((upperE-lowerE)/21)) + cppEDM Parameter.cc L198-291 expansion "
          "start..stop step inc; causal.py:143-144 keeps the first 22; E-None uses E = 10 (known bug 2)",
          f"[E, lowerE, upperE, inc, n sizes cppEDM, last size cppEDM, n kept, kept sizes]; T = {run['T']}: kept "
          f"{kept[0]}..{kept[-1]} of {sizes[0]}..{sizes[-1]} ({len(sizes)} sizes); documented: 21 sizes between the bounds")


def run_short(run, rid, p):
    E = 10 if p["E"] == "None" else int(p["E"])
    Ts = int(p["T"])
    pred = [1 if schedule(T, E)[2] < 1 else 0 for T in (Ts, Ts + 1)]
    S, _ = pyspi_value(rid)
    ok = bool(list(S[:2].astype(int)) == pred and S[2] == 1.0)
    write(rid, np.array(pred + [1.0], float), "qualitative", "analytic",
          "causal.py:127-131 schedule: inc = int((upperE-lowerE)/21) < 1 makes cppEDM Parameters::Validate throw",
          f"status codes for T = {Ts}, {Ts + 1} (0 ok, 1 RuntimeError 'increment 0 is invalid', 2 other); predicted "
          f"{pred}; observed {S[:2].astype(int).tolist()}; finite fraction at T = {Ts + 1}: {S[2]}. Calculator.compute "
          f"turns the error into an all-NaN matrix (calculator.py:336-339, warning 354-355)",
          pass_flag=int(ok), predicted_status=np.array(pred), observed_status=S[:2])


def run_diag(run, rid, p):
    S, _ = pyspi_value(rid)
    P = S.shape[0]
    off = ~np.eye(P, dtype=bool)
    ok = bool(np.all(np.diag(S) == 0.0) and np.all(np.isfinite(S[off])))
    write(rid, np.zeros(P), "qualitative", "analytic",
          "causal.py:117 ccmf = np.zeros((M, M, 22)); only i != j filled (causal.py:143-144)",
          f"direct ConvergentCrossMapping.multivariate (no Calculator): diagonal {np.diag(S).tolist()}; "
          f"Calculator.compute sets it to NaN (calculator.py:332)", pass_flag=int(ok))


def run_stat(run, rid, p, statistic, E, layer=None, key_extra=""):
    x, R, sizes, val, dt, vm = stat_reference(run, E, key_extra)
    S, xin = pyspi_value(rid)
    stats = ("mean", "max", "diff") if layer else (statistic,)
    V = np.stack([vm[s][0] for s in stats]) if layer else vm[statistic][0]
    SD = np.stack([vm[s][1] for s in stats]) if layer else vm[statistic][1]
    se = SD * np.sqrt(1 + N_PYSPI / N_DRAWS)
    with np.errstate(invalid="ignore"):
        z = np.abs(S - V) / se
    extra = dict(n_draws=N_DRAWS, n_boot=N_BOOT, xmap_vs_kernel_max_abs=val, draw_seconds=dt,
                 lib_sizes=f"{sizes[0]}..{sizes[-1]} step {sizes[1] - sizes[0]} ({len(sizes)})")
    if not run.get("probe") and xin.shape == x.shape:
        extra["input_max_abs_vs_pyspi"] = float(np.max(np.abs(xin - x)))
    flat = lambda A: A[~np.isnan(A)].round(5).tolist()  # noqa: E731
    notes = (f"stat(s) {stats} at E = {E}: pyspi {flat(S)} vs bootstrap mean {flat(V)} (se {flat(se)}, |z| "
             f"{flat(z)}); orientation A[i,j] = rho(x_j | M_x_i)")
    write(rid, V, "statistical", "independent-impl",
          f"{ME}:stat_reference ({N_DRAWS} own random libraries per size on pyspi's grid, XMap = {KD}:simplex_crossmap "
          f"arithmetic, bootstrap of 100-library curves)", notes, se=se, **extra)


def run_doc_e(run, rid, p):
    z = prep(raw(run), run["path"])
    Edoc = max(int(np.argmax(embed_table(z[q], z.shape[1])[0])) + 1 for q in range(2))
    x, R, sizes, val, dt, vm = stat_reference(run, Edoc)
    S, _ = pyspi_value(rid)
    V, se = vm["mean"][0], vm["mean"][1] * np.sqrt(1 + N_PYSPI / N_DRAWS)
    _, _, _, _, _, vm10 = stat_reference(run, 10)
    write(rid, V, "statistical", "independent-impl",
          f"{ME}:run_doc_e (documented E = max of the per-process argmax-rho E, embed_table; then stat_reference)",
          f"documented rule gives E = {Edoc}; pyspi uses E = 10 (known bug 2), so this run is EXPECTED to fail; "
          f"reference at E = 10 for contrast {vm10['mean'][0][~np.isnan(vm10['mean'][0])].round(5).tolist()}",
          se=se, documented_E=Edoc, xmap_vs_kernel_max_abs=val)


def run_curve(run, rid, p):
    E = 10 if p["E"] == "None" else int(p["E"])
    x = prep(raw(run), run["path"])
    key = f"{run['dataset']}_T{run['T']}_E{E}_{run['path']}_{'-'.join(map(str, run.get('processes') or []))}"
    R, sizes, val, dt = draws(x, E, run["T"], key)
    S, _ = pyspi_value(rid)
    V = np.zeros((2, 2, len(sizes)))
    se = np.zeros_like(V)
    V[0, 1], V[1, 0] = R[:, :, 0].mean(0), R[:, :, 1].mean(0)
    sd = R.std(0, ddof=1)
    se[0, 1], se[1, 0] = sd[:, 0] * np.sqrt(1 / N_PYSPI + 1 / N_DRAWS), sd[:, 1] * np.sqrt(1 / N_PYSPI + 1 / N_DRAWS)
    zz = np.concatenate([(S[0, 1] - V[0, 1]) / se[0, 1], (S[1, 0] - V[1, 0]) / se[1, 0]])
    c2 = float((zz ** 2).sum())
    pval = float(chi2.sf(c2, zz.size))
    ok = bool(np.abs(zz).max() <= 4 and pval > 1e-3 and np.all(np.diag(S[:, :, 0]) == 0))
    write(rid, V, "qualitative", "independent-impl",
          f"{ME}:draws ({N_DRAWS} own random libraries per size, XMap = {KD}:simplex_crossmap arithmetic)",
          f"pyspi data.ccm[{E}] (P,P,22) vs per-size mean; rule: max|z| <= 4 and chi2({zz.size}) p > 0.001; "
          f"max|z| {np.abs(zz).max():.3f}, frac |z|<=2 {np.mean(np.abs(zz) <= 2):.3f}, chi2 {c2:.1f} p {pval:.3g}; "
          f"curve 01 pyspi first/last {S[0, 1, 0]:.4f}/{S[0, 1, -1]:.4f}, 10 {S[1, 0, 0]:.4f}/{S[1, 0, -1]:.4f}",
          pass_flag=int(ok), max_abs_z=float(np.abs(zz).max()), chi2=c2, chi2_p=pval,
          frac_within_2se=float(np.mean(np.abs(zz) <= 2)), xmap_vs_kernel_max_abs=val,
          pyspi_curve_01_first=S[0, 1, 0], pyspi_curve_01_last=S[0, 1, -1],
          pyspi_curve_10_first=S[1, 0, 0], pyspi_curve_10_last=S[1, 0, -1])


def run_dir(run, rid, p):
    c, e = CAUSE_EFFECT[run["dataset"]]
    procs = run.get("processes")
    if procs is not None:                    # swapped orientation: matrix index = position in ``processes``
        c, e = list(procs).index(c), list(procs).index(e)
    sign = np.zeros((2, 2))
    sign[e, c], sign[c, e] = 1, -1          # cause c -> effect e: A[e, c] = rho(x_c | M_x_e) large, diff[e, c] > 0
    S, _ = pyspi_value(rid)
    write(rid, np.where(np.eye(2, dtype=bool), np.nan, sign), "direction", "analytic",
          "Sugihara et al. 2012 criterion on the known coupling (CONVENTIONS.md D5/D8): the effect's shadow manifold "
          "cross-maps the cause",
          f"cause {c} -> effect {e}: expected diff[{e},{c}] > 0 (A[i,j] = rho(x_j | M_x_i), so large A[i,j] means "
          f"j -> i); processes {procs}; pyspi diff[{e},{c}] = {S[e, c]:+.5f}", expected_sign=sign,
          pyspi_diff_effect_cause=S[e, c])


def run_static(run, rid, p):
    """E-1 on nonlinear_dependence (x1 = x0^2 + noise, iid samples, no dynamics): pre-declared qualitative rule."""
    S, _ = pyspi_value(rid)
    ok = bool(S[0, 1] > 0.7 and abs(S[1, 0]) < 0.2)
    write(rid, np.where(np.eye(2, dtype=bool), np.nan, [[0, 1], [0, 0]]), "qualitative", "analytic",
          "construction of nonlinear_dependence (CONVENTIONS.md D9): x1 is a function of x0 (plus 0.05 noise), x0 is not "
          "a function of x1 (sign lost); no temporal structure",
          f"pre-declared rule: A[0,1] = rho(x1 | x0) > 0.7 and |A[1,0]| = |rho(x0 | x1)| < 0.2; pyspi A[0,1] = "
          f"{S[0, 1]:.5f}, A[1,0] = {S[1, 0]:.5f}. Under the CCM reading this large asymmetric score says 1 -> 0 although "
          f"the samples are iid (E = 1 scores static functional dependence)", pass_flag=int(ok),
          pyspi_A01=S[0, 1], pyspi_A10=S[1, 0])


def run_eq_e10(run, rid, p):
    """E-None vs pyspi's own E-10 output on the same data (known bug 2; pyspi self-consistency, same-lineage)."""
    S, _ = pyspi_value(rid)
    other = rid.replace("causal.ccm.E-None", "causal.ccm.E-10").replace("eqE10-mean", "stat-mean")
    S10, _ = pyspi_value(other)
    eq = bool(np.array_equal(S, S10, equal_nan=True))
    write(rid, S10, "qualitative", "same-lineage",
          f"pyspi ccm_E-10_mean on the same data ({other})",
          f"ccm_E-None_mean bitwise equal to ccm_E-10_mean: {eq} (known bug 2: embed_df.max()['E'] = 10)",
          pass_flag=int(eq), max_abs_vs_E10=float(np.nanmax(np.abs(S - S10))))


def run_seeds(run, rid, p):
    """Bias test: pyEDM curve mean over 16 seeds x 100 libraries vs N_SEEDS_REF own draws (analytic se)."""
    E, n_seeds, n_ref = int(p["E"]), int(p.get("S", 16)), 2000
    x = prep(raw(run), run["path"])
    T = run["T"]
    sizes = schedule(T, E)[4]
    key = f"seeds_{run['dataset']}_T{T}_E{E}_{run['path']}_{'-'.join(map(str, run.get('processes') or []))}_n{n_ref}"
    f = EXCH / f"draws_{key}.npz"
    if f.is_file():
        with np.load(f) as c:
            R = c["R"]
    else:
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        X01, X10 = XMap(x[0], x[1], E), XMap(x[1], x[0], E)
        R = np.empty((n_ref, len(sizes), 2))
        for li, L in enumerate(sizes):
            for s_ in range(n_ref):
                lib = rng.choice(X01.m, L, replace=False)
                R[s_, li, 0], R[s_, li, 1] = X01.rho(lib), X10.rho(lib)
        EXCH.mkdir(parents=True, exist_ok=True)
        np.savez(f, R=R)
    S, _ = pyspi_value(rid)
    nL = len(sizes)
    m = R.mean(0).mean(0)                                    # (2,) mean over L of per-L means
    var_single = (R.var(0, ddof=1)).sum(0) / nL ** 2         # var of mean_L for ONE library per size
    se = np.sqrt(var_single * (1 / (100 * n_seeds) + 1 / n_ref))
    V, SE = to_matrix(*m), to_matrix(*se)
    z = np.abs(S - V) / SE
    write(rid, V, "statistical", "independent-impl",
          f"{ME}:run_seeds ({n_ref} own random libraries per size on pyspi's grid, XMap = {KD}:simplex_crossmap "
          f"arithmetic; analytic se of the mean over L, libraries independent across sizes)",
          f"pyEDM.CCM as pyspi calls it, seeds 1..{n_seeds} (16 x 100 libraries per size): mean_L "
          f"{S[~np.isnan(S)].round(5).tolist()} vs reference {V[~np.isnan(V)].round(5).tolist()} (se "
          f"{SE[~np.isnan(SE)].round(5).tolist()}, |z| {z[~np.isnan(z)].round(3).tolist()}); tests for a systematic "
          f"bias of the random-library curve that seed 42 alone cannot separate from sampling noise",
          se=SE, n_seeds=n_seeds, n_ref_draws=n_ref)


def run_agg(run, rid, p):
    S, _ = pyspi_value(rid)
    c = S[..., 3:]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(c, axis=2)
        V = np.concatenate([np.stack([mean, np.nanmax(c, axis=2),
                                      np.nanmean(c - np.transpose(c, (1, 0, 2)), axis=2)], axis=-1), c], axis=-1)
    d = S[..., 0:3] - V[..., 0:3]
    diff_vs_meandiff = float(np.max(np.abs(S[..., 2] - (mean - mean.T))))
    write(rid, V, "exact", "analytic",
          f"{ME}:run_agg (numpy nanmean / nanmax / nanmean(c - c^T) over the 22 library sizes of pyspi's cached "
          f"curve; causal.py:152-163)",
          f"P = {S.shape[0]}, curve length {c.shape[2]}; max|stat - recomputed| mean {np.max(np.abs(d[..., 0])):.3g}, "
          f"max {np.max(np.abs(d[..., 1])):.3g}, diff {np.max(np.abs(d[..., 2])):.3g}; max|diff - (mean - mean^T)| "
          f"{diff_vs_meandiff:.3g}; max|diff + diff^T| {float(np.max(np.abs(S[..., 2] + S[..., 2].T))):.3g}",
          rtol=1e-10, atol=1e-12, curve_length=c.shape[2], agg_mean_max_abs=float(np.max(np.abs(d[..., 0]))),
          agg_max_max_abs=float(np.max(np.abs(d[..., 1]))), agg_diff_max_abs=float(np.max(np.abs(d[..., 2]))),
          diff_vs_mean_minus_meanT=diff_vs_meandiff)


def dispatch(run):
    rid = run["run_id"]
    kind, p, rest = tag_of(rid)
    if kind == "xmap":
        return run_xmap(run, rid, p)
    if kind == "tied":
        return run_tied(run, rid, p)
    if kind == "affine":
        return run_affine(run, rid, p)
    if kind == "embed":
        return run_embed(run, rid, p)
    if kind == "sched":
        return run_sched(run, rid, p)
    if kind == "short":
        return run_short(run, rid, p)
    if kind == "diag":
        return run_diag(run, rid, p)
    if kind == "stats3":
        E = 10 if p["E"] == "None" else int(p["E"])
        return run_stat(run, rid, p, None, E, layer=True)
    if kind == "curve":
        return run_curve(run, rid, p)
    if kind == "dir":
        return run_dir(run, rid, p)
    if kind == "seeds":
        return run_seeds(run, rid, p)
    if kind == "static":
        return run_static(run, rid, p)
    if kind == "agg":
        return run_agg(run, rid, p)
    if kind == "eqE10":
        return run_eq_e10(run, rid, p)
    if kind == "stat":
        statistic = rest[0]
        if rest[1:] == ["docE"]:
            return run_doc_e(run, rid, p)
        E = E_of_identifier(run["identifier"])
        return run_stat(run, rid, p, statistic, 10 if E is None else E)
    raise ValueError(f"unknown run kind {kind!r} in {rid}")


def main():
    runs = json.loads(MANIFEST.read_text())["runs"]
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in runs:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        if not (OUT_PY / f"{rid}.npz").is_file():
            print(f"SKIP {rid}: no pyspi output")
            continue
        try:
            dispatch(run)
            n_ok += 1
        except Exception as exc:  # report and continue
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}")
            n_err += 1
    print(f"ccm references: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
