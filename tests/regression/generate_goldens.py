"""Generate (or check) the golden values for the SPI regression suite.

Usage::

    .venv/bin/python tests/regression/generate_goldens.py --size small|full|all
        [--check] [--repeat N] [--spis GLOB] [--phi] [--preproc]
        [--out DIR] [--configfile PATH]

* ``--size``      golden set(s) of per-SPI values to produce (``all`` = small + full).
* ``--preproc``   the preprocessing-variant set (``spi_policy.PREPROC_SPIS`` x
                  ``spi_policy.PREPROC_VARIANTS`` on the distorted small window).
* ``--phi``       ``phi_native.npz`` for the ``pyspi.lib.phi_native`` unit test.
* ``--check``     recompute and compare against the existing goldens with the
                  ``spi_policy`` tolerances, and compare each SPI's catalogue
                  fields (module, cls, params, deps, labels) with the stored
                  meta; writes nothing; exit 1 on any drift.
* ``--repeat N``  compute every SPI N times (same seeding); keep the minimum time
                  and report SPIs whose repeats disagree (determinism audit).
* ``--spis GLOB`` restrict to identifiers matching the fnmatch pattern. When
                  writing, the result is merged into an existing golden set;
                  stored SPIs no longer in the catalogue are dropped.
* ``--out DIR``   golden root directory (default ``tests/regression/goldens``).

Every SPI is computed exactly as the regression tests do it: a fresh ``Data``
from the dataset spec, ``seed_all`` immediately before ``multivariate``, and the
diagonal set to NaN like ``Calculator.compute``. An SPI that raises is recorded
with status ``error`` and an all-NaN array, which is what the Calculator stores.
"""
import argparse
import datetime as _dt
import fnmatch
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from regression import harness, spi_policy  # noqa: E402

PHI_FILE = "phi_native.npz"
PHI_SEED = 42
PHI_Z = [1, 2]
PHI_PARAMS = {"tau": 1}
PHI_TYPES = ("star", "Geo")
PHI_NORMS = (0, 1)


def log(msg=""):
    print(msg, flush=True)


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def nan_offdiag(S):
    mask = np.isnan(S)
    np.fill_diagonal(mask, False)
    return int(mask.sum())


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def run_one(entry, raw, spec, repeat):
    """Compute one SPI ``repeat`` times. Returns ``(value, record, reproducible_msg)``.

    ``record`` holds ``time_s`` (minimum over repeats), ``status``, ``error`` and
    ``nan_offdiag`` of the first repeat. ``reproducible_msg`` is ``None`` when all
    repeats agree with the first under the policy tolerance.
    """
    ident = entry["identifier"]
    p = raw.shape[0]
    results, times = [], []
    for _ in range(max(1, repeat)):
        spi = harness.build_spi(entry)
        data = harness.make_data(raw, spec)
        harness.seed_all(spi_policy.SEED)
        t0 = time.perf_counter()
        try:
            S = harness.compute_spi(spi, data)
            err = None
        except Exception as exc:  # recorded, like Calculator.compute
            S = np.full((p, p), np.nan)
            err = repr(exc)
        times.append(time.perf_counter() - t0)
        results.append((S, err))

    S0, err0 = results[0]
    status = "error" if err0 is not None else harness.status_of(S0)
    record = dict(time_s=round(min(times), 6), status=status, error=err0,
                  nan_offdiag=nan_offdiag(S0))

    problem = None
    for k, (Sk, errk) in enumerate(results[1:], start=2):
        if (errk is None) != (err0 is None):
            problem = f"repeat {k}: error state differs ({err0!r} vs {errk!r})"
            break
        try:
            harness.assert_spi_allclose(Sk, S0, ident)
        except AssertionError:
            mabs, mrel = harness.max_differences(Sk, S0)
            problem = f"repeat {k}: max_abs={mabs:.3g} max_rel={mrel:.3g}"
            break
    return S0, record, problem


def warm_up(by_id, idents, raw, spec):
    """Compute selected SPIs once, untimed, so first-call costs are not measured.

    numba JIT (hyppo, tslearn) and lazy imports make the first call of a code
    path far slower than the rest (``dtw`` takes ~4 s cold vs ~5 ms warm at the
    small size). Warmed: every selected distance SPI (numba-heavy, all cheap)
    and the first selected SPI of every other class unless it is in FORCE_SLOW.
    """
    seen, chosen = set(), []
    for ident in idents:
        entry = by_id[ident]
        key = (entry["module"], entry["cls"])
        if entry["module"] == ".statistics.distance" or (
                key not in seen and not spi_policy.matches_any(ident, spi_policy.FORCE_SLOW)):
            chosen.append(entry)
        seen.add(key)
    t0 = time.perf_counter()
    for entry in chosen:
        try:
            harness.seed_all(spi_policy.SEED)
            harness.compute_spi(harness.build_spi(entry), harness.make_data(raw, spec))
        except Exception:
            pass  # the timed run records the error
    log(f"warm-up: {len(chosen)} SPIs in {time.perf_counter() - t0:.1f}s")


def compute_set(jobs, repeat):
    """``jobs`` = list of ``(key, entry, raw, spec)``. Returns values, records, problems."""
    values, records, problems = {}, {}, {}
    width = max((len(k) for k, *_ in jobs), default=10)
    for i, (key, entry, raw, spec) in enumerate(jobs, start=1):
        S, rec, problem = run_one(entry, raw, spec, repeat)
        values[key] = S
        records[key] = rec
        flag = "" if problem is None else "  NON-REPRODUCIBLE"
        err = f"  {rec['error'][:120]}" if rec["error"] else ""
        log(f"[{i:3d}/{len(jobs)}] {key:<{width}}  {rec['status']:<11} {rec['time_s']:9.3f}s{flag}{err}")
        if problem is not None:
            problems[key] = problem
    return values, records, problems


# --------------------------------------------------------------------------- #
# Golden sets
# --------------------------------------------------------------------------- #
CATALOGUE_FIELDS = ("module", "cls", "params", "labels", "deps")


def catalogue_fields(entry):
    return {k: entry[k] for k in CATALOGUE_FIELDS}


def catalogue_mismatches(stored, current):
    """``[(field, stored_value, current_value)]`` for one SPI's catalogue fields.

    Labels compare as sets (their order is not meaningful), matching
    ``test_spi_catalogue``.
    """
    out = []
    for field in CATALOGUE_FIELDS:
        old, new = stored.get(field), current.get(field)
        if field == "labels":
            old, new = sorted(set(old or [])), sorted(set(new or []))
        if old != new:
            out.append((field, old, new))
    return out


def base_meta(name, dataset, cat, configfile):
    entries, _ = cat
    return {
        "schema": 1,
        "name": name,
        "seed": spi_policy.SEED,
        "generated": utc_now(),
        "dataset": dataset,
        "configfile": "config.yaml" if configfile is None else str(configfile),
        "environment": harness.environment_info(),
        "calculator_order": [e["identifier"] for e in entries],
        "duplicate_identifiers": harness.duplicate_identifiers(entries),
        "spis": {},
    }


def select(identifiers, pattern):
    return [i for i in identifiers if pattern is None or fnmatch.fnmatch(i, pattern)]


def build_size_set(size, cat, args):
    entries, by_id = cat
    spec = harness.size_spec(size)
    raw = harness.load_raw(spec)
    order = list(by_id)  # Calculator order, last-wins on collisions
    idents = select(order, args.spis)
    log(f"\n=== golden set '{size}': {raw.shape[0]} processes x {raw.shape[1]} observations, "
        f"{len(idents)} SPIs, repeat={args.repeat} ===")
    warm_up(by_id, idents, raw, spec)
    values, records, problems = compute_set([(i, by_id[i], raw, spec) for i in idents], args.repeat)
    meta = base_meta(size, spec, cat, args.configfile)
    for ident in idents:
        meta["spis"][ident] = {**catalogue_fields(by_id[ident]), **records[ident]}
    return values, meta, problems


def build_preproc_set(cat, args):
    entries, by_id = cat
    idents = spi_policy.select_preproc_identifiers(list(by_id))
    unmatched = [e for e in spi_policy.PREPROC_SPIS
                 if not any(i == e or i.startswith(e + "_") for i in by_id)]
    if unmatched:
        log(f"WARNING: PREPROC_SPIS entries with no catalogue match: {unmatched}")
    idents = select(idents, args.spis)
    jobs = []
    for (normalise, detrend) in spi_policy.PREPROC_VARIANTS:
        spec = harness.preproc_spec(normalise, detrend)
        raw = harness.load_raw(spec)
        for ident in idents:
            jobs.append((spi_policy.preproc_key(ident, normalise, detrend), by_id[ident], raw, spec))
    base = harness.preproc_spec(True, False)
    log(f"\n=== golden set 'preproc': {len(idents)} SPIs x {len(spi_policy.PREPROC_VARIANTS)} "
        f"variants, repeat={args.repeat} ===")
    warm_up(by_id, idents, harness.load_raw(base), base)
    values, records, problems = compute_set(jobs, args.repeat)

    # meta["dataset"] keeps the Calculator-default flags (normalise=True,
    # detrend=False) so it stays a valid harness spec; each key's actual
    # flags are in meta["variants"][key] and take precedence.
    meta = base_meta("preproc", dict(base), cat, args.configfile)
    meta["variants"] = {}
    for key, _entry, _raw, spec in jobs:
        ident = spi_policy.split_preproc_key(key)[0]
        meta["variants"][key] = {"identifier": ident, "normalise": spec["normalise"],
                                 "detrend": spec["detrend"], **records[key]}
    for ident in idents:
        keys = [k for k, *_ in jobs if spi_policy.split_preproc_key(k)[0] == ident]
        default = records[spi_policy.preproc_key(ident, True, False)]
        meta["spis"][ident] = {
            **catalogue_fields(by_id[ident]),
            # time_s: slowest variant (drives the slow marker); status/error/
            # nan_offdiag: the Calculator-default variant (n1d0).
            "time_s": max(records[k]["time_s"] for k in keys),
            "status": default["status"], "error": default["error"],
            "nan_offdiag": default["nan_offdiag"],
        }
    return values, meta, problems


def _key_identifier(key):
    return spi_policy.split_preproc_key(key)[0] if "::" in key else key


def merge_into_existing(name, values, meta, partial):
    """With ``--spis`` only some keys were recomputed: keep the rest of the set.

    Stored keys whose identifier is no longer in the current
    ``calculator_order`` are dropped (and listed), so an SPI removed from the
    config cannot keep a passing test on a stale golden.
    """
    if not partial or not harness.golden_exists(name):
        return values, meta
    old_values = harness.load_values(name)
    with open(harness.golden_dir(name) / "meta.json") as f:
        old_meta = json.load(f)
    current = set(meta["calculator_order"])
    stale_keys = sorted(k for k in old_values if _key_identifier(k) not in current)
    stale_ids = sorted(i for i in old_meta.get("spis", {}) if i not in current)
    if stale_keys or stale_ids:
        log(f"dropped from '{name}' (identifier no longer in the catalogue): "
            f"spis={stale_ids} keys={stale_keys}")
    merged_values = {k: v for k, v in old_values.items() if k not in stale_keys}
    merged_values.update(values)
    merged_meta = dict(meta)
    merged_meta["spis"] = {**{i: r for i, r in old_meta.get("spis", {}).items() if i in current},
                           **meta["spis"]}
    if "variants" in meta:
        merged_meta["variants"] = {**{k: r for k, r in old_meta.get("variants", {}).items()
                                      if _key_identifier(k) in current},
                                   **meta["variants"]}
    log(f"merged {len(values)} recomputed keys into existing '{name}' ({len(merged_values)} keys)")
    return merged_values, merged_meta


def check_set(name, values, meta, partial):
    """Compare a fresh computation with the stored set. Returns True when it passes."""
    if not harness.golden_exists(name):
        log(f"CHECK '{name}': no golden set at {harness.golden_dir(name)}")
        return False
    gold = harness.load_values(name)
    with open(harness.golden_dir(name) / "meta.json") as f:
        gmeta = json.load(f)
    gstatus = {**{k: v.get("status") for k, v in gmeta.get("spis", {}).items()},
               **{k: v.get("status") for k, v in gmeta.get("variants", {}).items()}}
    nstatus = {**{k: v["status"] for k, v in meta["spis"].items()},
               **{k: v["status"] for k, v in meta.get("variants", {}).items()}}

    missing = sorted(set(values) - set(gold))          # computed, not in golden
    extra = [] if partial else sorted(set(gold) - set(values))  # in golden, not computed
    failures, status_changes, rows = [], [], []

    # Catalogue fields per identifier: what test_spi_catalogue and build_spi
    # rely on must match the stored meta, not only the values.
    catalogue_changes = []
    for ident, rec in meta["spis"].items():
        stored = gmeta.get("spis", {}).get(ident)
        if stored is None:
            continue  # reported as MISSING via the values
        for field, old, new in catalogue_mismatches(stored, rec):
            catalogue_changes.append((ident, field, old, new))

    for key in values:
        if key not in gold:
            continue
        ident = _key_identifier(key)
        mabs, mrel = harness.max_differences(values[key], gold[key])
        ok = True
        try:
            harness.assert_spi_allclose(values[key], gold[key], ident)
        except AssertionError as exc:
            ok = False
            failures.append((key, str(exc).strip().splitlines()[0]))
        if gstatus.get(key) != nstatus.get(key):
            status_changes.append((key, gstatus.get(key), nstatus.get(key)))
            ok = False
        rows.append((key, mabs, mrel, ok))

    log(f"\n--- check '{name}': {len(rows)} keys compared ---")
    width = max((len(r[0]) for r in rows), default=10)
    for key, mabs, mrel, ok in rows:
        log(f"  {key:<{width}}  max_abs={mabs:<10.3g} max_rel={mrel:<10.3g} {'ok' if ok else 'FAIL'}")
    for key, old, new in status_changes:
        log(f"  STATUS CHANGE {key}: {old} -> {new}")
    for ident, field, old, new in catalogue_changes:
        log(f"  CATALOGUE CHANGE {ident}.{field}: {old!r} -> {new!r}")
    for key, msg in failures:
        log(f"  FAIL {key}: {msg}")
    if missing:
        log(f"  MISSING from golden (computed now, not stored): {missing}")
    if extra:
        log(f"  EXTRA in golden (stored, not computed now): {extra}")
    if not partial:
        old_order = gmeta.get("calculator_order")
        if old_order != meta["calculator_order"]:
            log("  calculator_order differs from the golden meta")
            failures.append(("calculator_order", "differs"))
    passed = not (failures or status_changes or catalogue_changes or missing or extra)
    log(f"--- check '{name}': {'PASS' if passed else 'FAIL'} ---")
    return passed


def summarise(name, meta, problems, records_key="spis", values=None):
    recs = meta[records_key] if records_key == "spis" else meta["variants"]
    counts = {}
    for r in recs.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    log(f"\n=== summary '{name}' ===")
    log("status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad = sorted(k for k, r in recs.items() if r["status"] in ("all_nan", "error"))
    if bad:
        banner = "!!! " if len(bad) > 5 else ""
        log(f"{banner}all_nan/error ({len(bad)}): " + ", ".join(f"{k}[{recs[k]['status']}]" for k in bad))
    partial = sorted(k for k, r in recs.items() if r["status"] == "partial_nan")
    if partial:
        log(f"partial_nan ({len(partial)}): {', '.join(partial)}")
    if values is not None:
        # status only counts NaN; an off-diagonal full of +/-inf is recorded as
        # "ok", so flag non-finite snapshots here (assert_spi_allclose locks them).
        inf = sorted(k for k, v in values.items() if np.isinf(np.asarray(v)).any())
        if inf:
            log(f"non-finite (inf) entries ({len(inf)}): "
                + ", ".join(f"{k}[{int(np.isinf(np.asarray(values[k])).sum())}]" for k in inf))
    timed = sorted(((r["time_s"], k) for k, r in meta["spis"].items()), reverse=True)
    slow = [(k, t) for t, k in timed if spi_policy.is_slow(k, t)]
    log(f"slow set ({len(slow)}; > {spi_policy.SLOW_THRESHOLD_S}s measured; marked slow): "
        + ", ".join(f"{k}={t:.2f}s" for k, t in slow))
    over = [(k, t) for t, k in timed if t > spi_policy.SLOW_THRESHOLD_S]
    log(f"measured > {spi_policy.SLOW_THRESHOLD_S}s ({len(over)}): "
        + ", ".join(f"{k}={t:.2f}s" for k, t in over))
    stoch = [k for k in meta["spis"] if spi_policy.is_stochastic(k)]
    log(f"stochastic set ({len(stoch)}): {', '.join(stoch)}")
    log(f"total SPI compute time (min over repeats): {sum(t for t, _ in timed):.1f}s")
    if problems:
        log(f"DETERMINISM: {len(problems)} non-reproducible key(s):")
        for k, msg in problems.items():
            log(f"  {k}: {msg}")
    else:
        log("DETERMINISM: all repeats agree under policy tolerances")


# --------------------------------------------------------------------------- #
# phi_native
# --------------------------------------------------------------------------- #
def phi_values():
    from pyspi.lib.phi_native import phi_comp
    np.random.seed(PHI_SEED)
    X = np.random.randn(2, 100)
    Z = np.array(PHI_Z)
    out = {"X": X, "Z": Z}
    for t in PHI_TYPES:
        for n in PHI_NORMS:
            options = {"type_of_phi": t, "type_of_dist": "Gauss", "normalization": n}
            out[f"phi_{t}_norm{n}"] = np.float64(phi_comp(X, Z, dict(PHI_PARAMS), options))
    desc = {"seed": PHI_SEED, "X": "np.random.seed(seed); np.random.randn(2, 100)", "Z": PHI_Z,
            "params": PHI_PARAMS, "type_of_dist": "Gauss",
            "keys": {f"phi_{t}_norm{n}": {"type_of_phi": t, "normalization": n}
                     for t in PHI_TYPES for n in PHI_NORMS},
            "generated": utc_now()}
    out["params_json"] = np.array(json.dumps(desc, sort_keys=True))
    return out


def do_phi(check):
    path = harness.GOLDEN_DIR / PHI_FILE
    vals = phi_values()
    for k in sorted(vals):
        if k.startswith("phi_"):
            log(f"{k} = {float(vals[k])!r}")
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **vals)
        log(f"wrote {path}")
        return True
    if not path.is_file():
        log(f"CHECK phi: missing {path}")
        return False
    with np.load(path) as z:
        gold = {k: z[k] for k in z.files}
    ok = set(gold) == set(vals)
    if not ok:
        log(f"CHECK phi: keys differ {sorted(set(gold) ^ set(vals))}")
    for k in sorted(set(gold) & set(vals)):
        if k == "params_json":
            continue
        # inputs must match exactly; phi values use the phi_* policy tolerance
        rtol, atol = spi_policy.tolerance_for(k) if k.startswith("phi_") else (0.0, 0.0)
        good = np.allclose(vals[k], gold[k], rtol=rtol, atol=atol)
        mabs, mrel = harness.max_differences(np.atleast_1d(vals[k]), np.atleast_1d(gold[k]))
        log(f"  phi {k}: max_abs={mabs:.3g} max_rel={mrel:.3g} {'ok' if good else 'FAIL'}")
        ok &= bool(good)
    log(f"--- check phi: {'PASS' if ok else 'FAIL'} ---")
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", choices=["small", "full", "all"], default=None)
    ap.add_argument("--check", action="store_true", help="compare with existing goldens; write nothing")
    ap.add_argument("--repeat", type=int, default=1, help="computations per SPI (min time kept)")
    ap.add_argument("--spis", default=None, metavar="GLOB", help="fnmatch filter on identifiers")
    ap.add_argument("--phi", action="store_true", help="phi_native.npz")
    ap.add_argument("--preproc", action="store_true", help="preprocessing-variant set")
    ap.add_argument("--out", default=None, metavar="DIR", help="golden root (default tests/regression/goldens)")
    ap.add_argument("--configfile", default=None, metavar="PATH", help="SPI config (default pyspi/config.yaml)")
    args = ap.parse_args(argv)
    if args.size is None and not (args.phi or args.preproc):
        ap.error("nothing to do: give --size, --preproc and/or --phi")
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    t_start = time.perf_counter()
    if args.out is not None:
        # golden_dir()/golden_exists()/load_values()/save_golden() all resolve
        # through this module attribute at call time.
        harness.GOLDEN_DIR = Path(args.out).resolve()
    warnings.simplefilter("ignore")  # SPI warnings would drown the progress lines

    ok = True
    if args.phi:
        ok &= do_phi(args.check)

    sizes = [] if args.size is None else (["small", "full"] if args.size == "all" else [args.size])
    if sizes or args.preproc:
        java = harness.ensure_jvm()
        log(f"JVM available: {java}")
        harness.import_statistics_modules()
        cat = harness.catalogue(args.configfile, harness.optional_dependencies())
        dups = harness.duplicate_identifiers(cat[0])
        log(f"catalogue: {len(cat[0])} entries, {len(cat[1])} unique identifiers, duplicates={dups}")
        partial = args.spis is not None

        # Fail fast on an empty --spis selection: it would make --check pass
        # vacuously and a write merely re-stamp the set.
        candidates = {s: list(cat[1]) for s in sizes}
        if args.preproc:
            candidates["preproc"] = spi_policy.select_preproc_identifiers(list(cat[1]))
        empty = [name for name, ids in candidates.items() if not select(ids, args.spis)]
        if empty:
            raise SystemExit(f"--spis {args.spis!r} matched no identifier in set(s) {empty}")

        builds = [(s, lambda s=s: build_size_set(s, cat, args), "spis") for s in sizes]
        if args.preproc:
            builds.append(("preproc", lambda: build_preproc_set(cat, args), "variants"))
        for name, build, rkey in builds:
            t0 = time.perf_counter()
            values, meta, problems = build()
            summarise(name, meta, problems, rkey, values)
            log(f"wall time '{name}': {time.perf_counter() - t0:.1f}s")
            if args.check:
                ok &= check_set(name, values, meta, partial)
            else:
                values, meta = merge_into_existing(name, values, meta, partial)
                harness.save_golden(name, values, meta)
                log(f"wrote {harness.golden_dir(name)} ({len(values)} keys)")

    log(f"\ntotal wall time: {time.perf_counter() - t_start:.1f}s")
    if args.check:
        log("CHECK RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
