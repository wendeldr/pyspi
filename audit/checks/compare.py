"""Step 3 of the numeric checks: compare pyspi outputs with the references.

Runs under either interpreter (numpy only). For every ``run_id`` that has both
``out/pyspi/<run_id>.npz`` (from ``pyspi_outputs.py``) and
``out/ref/<run_id>.npz`` (from a ``refs/<family>.py`` script, an R or an
Octave script) it applies the criterion the reference declared and writes the
outcome to ``out/results/<family>.json`` (one file per family, safe under
concurrent family runs); ``out/results.json`` is the merged view rebuilt after
every run. A reference may also declare ``criterion = "qualitative"`` with
``pass_flag`` when the reference side judged the outcome itself (e.g. an
expected IndexError, a determinism check, or an R/Octave result it converted).

Reference npz contract (written by the reference side)::

    value            ndarray, same shape as pyspi's matrix (NaN diagonal) or a
                     scalar array; the quantity the reference computed
    criterion        "exact" | "tolerance" | "statistical" | "direction" | "optimality" | "qualitative"
    rtol, atol       floats for exact/tolerance (exact defaults 1e-10 / 1e-12)
    se               ndarray for "statistical": pass if |pyspi - value| <= 2*se
    expected_sign    ndarray of {-1, 0, +1} for "direction": sign(pyspi) must match where nonzero
    pass_flag        0/1 for "optimality"/"qualitative": the reference decided itself
    reference_kind   "analytic" | "independent-impl" | "external-pkg" | "same-lineage" | "literature-only"
    reference_detail str: script, function, package version or paper equation used
    notes            str (optional); any extra keys are copied into the result as strings

Run::

    python audit/checks/compare.py [--family F ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parent
OUT = CHECKS / "out"
RESULTS_DIR = OUT / "results"      # one file per family (safe under concurrent family runs)
RESULTS = OUT / "results.json"     # merged view, rebuilt after every run


def _read_json(path, attempts=20):
    """Read a JSON file that another family's run may be replacing concurrently.

    Writers use ``_write_json`` (temp file + ``os.replace``), so a reader only ever sees a
    complete file; the retry covers the window between ``exists()`` and ``read_text()`` on an
    older checkout and any file still being written by a pre-atomic writer.
    """
    import time
    for i in range(attempts):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if i == attempts - 1:
                raise
            time.sleep(0.25)


def _write_json(path, obj, **kw):
    """Atomic replace so a concurrent reader never sees a half-written file."""
    import os
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, **kw) + "\n")
    os.replace(tmp, path)


def load_index():
    merged = {}
    d = OUT / "pyspi" / "index"
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            merged.update(_read_json(f))
    return merged


def load_results():
    merged = {}
    if RESULTS_DIR.is_dir():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            merged.update(_read_json(f))
    return merged


def max_differences(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    both = ~(np.isnan(a) & np.isnan(e)) & ~(a == e)
    if not both.any():
        return 0.0, 0.0
    diff = np.abs(a[both] - e[both])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.abs(e[both])
    rel = rel[np.isfinite(rel)]
    return float(np.nanmax(diff)) if diff.size else 0.0, float(rel.max()) if rel.size else 0.0


def _scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = z[key]
    return v.item() if v.shape == () else v


def compare_one(rid, pyspi_npz, ref_npz):
    with np.load(pyspi_npz) as zp:
        S = np.asarray(zp["value"], dtype=np.float64)
    res = dict(run_id=rid)
    with np.load(ref_npz, allow_pickle=False) as zr:
        crit = str(_scalar(zr, "criterion", "tolerance"))
        res["criterion"] = crit
        res["reference_kind"] = str(_scalar(zr, "reference_kind", "independent-impl"))
        res["reference_detail"] = str(_scalar(zr, "reference_detail", ""))
        res["notes"] = str(_scalar(zr, "notes", ""))
        for k in zr.files:
            if k not in ("value", "criterion", "rtol", "atol", "se", "expected_sign", "pass_flag",
                         "reference_kind", "reference_detail", "notes"):
                v = zr[k]
                res[f"ref_{k}"] = v.item() if v.shape == () else v.tolist() if v.size <= 16 else f"array{v.shape}"
        if crit in ("optimality", "qualitative"):
            res["pass"] = bool(_scalar(zr, "pass_flag", 0))
            return res
        V = np.asarray(zr["value"], dtype=np.float64)
        if V.shape == () and S.shape != ():
            V = np.full(S.shape, float(V)); np.fill_diagonal(V, np.nan)
        if V.shape != S.shape:
            res.update(**{"pass": False, "error": f"shape mismatch pyspi {S.shape} vs ref {V.shape}"})
            return res
        res["nan_mask_equal"] = bool(np.array_equal(np.isnan(S), np.isnan(V)))
        res["max_abs"], res["max_rel"] = max_differences(S, V)
        if crit == "direction":
            sign = np.asarray(zr["expected_sign"], dtype=np.float64)
            mask = (sign != 0) & ~np.isnan(S)
            agree = np.sign(S[mask]) == np.sign(sign[mask])
            res["direction_agreement"] = float(agree.mean()) if agree.size else float("nan")
            res["pass"] = bool(agree.all()) if agree.size else False
            return res
        if crit == "statistical":
            se = np.asarray(zr["se"], dtype=np.float64)
            m = ~np.isnan(S) & ~np.isnan(V)
            z = np.abs(S[m] - V[m]) / np.where(se[m] > 0, se[m], np.inf)
            res["max_z"] = float(z.max()) if z.size else 0.0
            res["pass"] = bool(res["nan_mask_equal"] and (z <= 2.0).all())
            return res
        rtol = float(_scalar(zr, "rtol", 1e-10 if crit == "exact" else 1e-6))
        atol = float(_scalar(zr, "atol", 1e-12 if crit == "exact" else 1e-8))
        res["rtol"], res["atol"] = rtol, atol
        m = ~np.isnan(S) & ~np.isnan(V)
        close = np.allclose(S[m], V[m], rtol=rtol, atol=atol) if m.any() else True
        res["pass"] = bool(res["nan_mask_equal"] and close)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append")
    a = ap.parse_args()
    index = load_index()
    families = sorted({rec.get("family") for rec in index.values()})
    if a.family:
        families = [f for f in families if f in a.family]
    n_pass = n_fail = n_missing = 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for fam in families:
        results = {}
        for rid, rec in sorted(index.items()):
            if rec.get("family") != fam:
                continue
            p = OUT / "pyspi" / f"{rid}.npz"
            r = OUT / "ref" / f"{rid}.npz"
            base = {k: rec.get(k) for k in ("family", "unit_id", "identifier", "dataset", "T", "path")}
            if rec.get("status") != "ok":
                results[rid] = dict(base, run_id=rid, **{"pass": False}, error=f"pyspi side: {rec.get('error')}")
                n_fail += 1
                continue
            if not r.is_file():
                results[rid] = dict(base, run_id=rid, **{"pass": None}, error="no reference output")
                n_missing += 1
                continue
            try:
                out = compare_one(rid, p, r)
            except Exception as exc:
                out = {"run_id": rid, "pass": False, "error": f"compare: {exc!r}"}
            results[rid] = dict(base, **out)
            n_pass += bool(out.get("pass"))
            n_fail += not out.get("pass")
        _write_json(RESULTS_DIR / f"{fam}.json", results, default=str)
    _write_json(RESULTS, load_results(), default=str)
    print(f"compare: {n_pass} pass, {n_fail} fail, {n_missing} without reference -> {RESULTS_DIR}/ (merged: {RESULTS})")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
