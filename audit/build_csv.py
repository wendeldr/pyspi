"""Assemble ``audit/metrics_audit.csv`` and ``audit/identifier_map.csv``.

Joins, per unit: identity from ``units.yaml``; correctness and optimisation
fields from ``records/<unit_id>.yaml``; closeness numbers from
``checks/out/results.json`` (summarised per unit); timings from
``profiles/summary.json`` (if present). Opportunities are flattened: a unit
with N opportunities yields N rows in ``metrics_audit.csv`` whose identity and
correctness columns repeat and whose ``opportunity_*`` columns differ; a unit
with no opportunity yields one row with empty opportunity columns.

Cross-cutting opportunities (``profiles/crosscutting.yaml``, ``rows[]`` with
``unit_id: family:<name>``, ``title``, ``scope`` unit-id globs, optional
``projected_patient_s`` and ``opportunities[]``) become rows with
``module = cross-cutting``. A cross-cutting row without an explicit
``projected_patient_s`` is split by the tier of the units its scope covers (one
row per tier, ``unit_id`` suffixed ``/retained``, ``/eeg-fc-family`` or
``/other`` when the scope spans more than one tier), each with the summed basis
of its own units only. ``identifiers_covered`` / ``n_identifiers`` /
``eeg_identifiers`` of a cross-cutting row list the identifiers of its units;
``rank_units`` lists the units (a unit row's own ``unit_id``).

Ranking (README "Optimisation columns", rules 1-5): the basis of a unit is its
family-warm projection when present, else fresh, over its configured identifiers;
rows in the EEG selection (tier 0) rank on the retained identifiers where the
per-identifier family-warm increments separate them, or on the basis the record's
``eeg_basis`` names (``fresh`` when the family-warm run let identifiers outside the
EEG selection fill a cache the unit reads), and ``basis_scope`` /
``basis_scope_note`` say which (``unit_basis``, ``_cross_unit_warm``). An opportunity with
``eeg_applicable: false`` on a retained unit leaves tier 0 and ranks on the
configured identifiers. The per-window avenues of each unit are stacked: an avenue
comes after the ones it names in ``depends_on`` / ``subsumed_by`` on that unit
(transitively), and other avenues go in order of their lower-bound speedup
(largest first, ``speedup_kind: projection`` last); ``independent_of`` records that
two such avenues act on different parts of the cost (different factors, or disjoint
frames), so each keeps its own k on the residual, or, for disjoint frames, the
lower bound of its ``residual_speedup`` when it gives one (the same seconds off a
smaller residual). ``rank_basis_s`` is the
unit's basis divided by the lower-bound speedups of every avenue stacked before it,
so the savings of one unit's rows never sum past its basis;
``projected_patient_s_after`` divides ``rank_basis_s`` by the lower bound of
``expected_speedup`` (first number followed by ``x``/``×``), or of
``residual_speedup`` when a ``depends_on`` prerequisite or an ``independent_of``
avenue is stacked before it on that unit, or of a matching ``scope_speedup`` entry of a cross-cutting opportunity;
it equals ``rank_basis_s`` where a ``subsumed_by`` avenue applies.
``priority_rank`` orders per-window rows before ``speedup_kind: throughput`` rows,
then by tier (retained EEG selection, then EEG-FC families, then the rest), puts
``projection`` rows after the other per-window rows of their tier, then orders by
patient seconds saved (``rank_basis_s - projected_patient_s_after``), output risk
and effort; ``rank_rule`` states which rule applied.

Usage: ``python audit/build_csv.py [--check]`` (``--check`` fails if the CSVs are stale).
"""
import argparse
import csv
import fnmatch
import io
import json
import re
import sys
from pathlib import Path

import yaml

AUDIT = Path(__file__).resolve().parent

IDENTITY = ["unit_id", "module", "pyspi_class", "method_or_estimator", "hyperparameters", "source",
            "identifiers_covered", "n_identifiers", "representative", "family", "pair_basis",
            "in_eeg_selection", "eeg_identifiers", "status"]
CORRECTNESS = ["metric_name_standard", "canonical_ref", "canonical_definition", "documented_definition",
               "delegate_lib", "implemented_as", "implementation_trace", "deviation_from_documented",
               "deviation_from_canonical", "units_and_range", "direction_convention", "symmetry_expected",
               "check_method", "check_reference", "check_data", "agreement_criterion", "closeness",
               "closeness_measured", "verdict", "score_0_10", "score_rubric", "score_rubric_basis",
               "score_rubric_note", "documentation_status", "verdict_adjusted", "score_adjusted",
               "score_adjustment_rules", "score_adjustment_reason", "identifier_exceptions", "what_changed", "already_documented",
               "new_finding", "definition_under_review", "check_cost", "symmetry_observed", "direction_observed"]
TIMING = ["time_s_7x100_golden", "t_7x100_s", "machine_factor", "t_8x256_s", "t_8x1024_s", "t_32x1024_s",
          "t_118x1024_s", "measure_mode", "t_cold_s", "t_per_pair_ms", "t_shared_family_s", "t_incremental_s",
          "complexity_P_T", "P_exponent_measured", "T_exponent_measured", "peak_rss_mb", "hotspot_1",
          "hotspot_2", "hotspot_owner", "opaque_share", "profile_file", "projected_window_s_118",
          "projected_patient_s", "projected_patient_s_family_warm", "priority_rank"]
OPPORTUNITY = ["opportunity_id", "opportunity", "opp_type", "expected_speedup", "speedup_basis",
               "projected_patient_s_after", "output_risk", "risk_evidence", "effort", "depends_on",
               "subsumed_by", "independent_of", "residual_speedup", "eeg_applicable", "speedup_kind",
               "rank_basis_s", "basis_scope", "basis_scope_note", "rank_units", "rank_rule"]
COLUMNS = IDENTITY + CORRECTNESS + TIMING + OPPORTUNITY


def _s(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return " | ".join(_s(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def load_results():
    d = AUDIT / "checks" / "out" / "results"
    merged = {}
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            merged.update(json.loads(f.read_text()))
    elif (AUDIT / "checks" / "out" / "results.json").is_file():
        merged = json.loads((AUDIT / "checks" / "out" / "results.json").read_text())
    if not merged:
        return {}
    per_unit = {}
    for rid, r in merged.items():
        u = r.get("unit_id")
        if not u:
            continue
        per_unit.setdefault(u, []).append(r)
    out = {}
    for u, rs in per_unit.items():
        parts = []
        for r in sorted(rs, key=lambda r: r["run_id"]):
            tag = f"{r.get('dataset')}/T{r.get('T')}/{r.get('path')}"
            if r.get("pass") is None:
                parts.append(f"{tag}: no reference")
            elif "max_abs" in r:
                parts.append(f"{tag}: {r.get('criterion')} max_abs={r['max_abs']:.3g} max_rel={r['max_rel']:.3g} "
                             f"{'PASS' if r['pass'] else 'FAIL'}")
            else:
                parts.append(f"{tag}: {r.get('criterion')} {'PASS' if r.get('pass') else 'FAIL'}")
        out[u] = "; ".join(parts)
    return out


SHAPE_COLS = {"7x100": "t_7x100_s", "8x256": "t_8x256_s", "8x1024": "t_8x1024_s",
              "32x1024": "t_32x1024_s", "118x1024": "t_118x1024_s"}


def _fresh_record(unit_id, shape):
    p = AUDIT / "profiles" / f"{unit_id}__{shape}__fresh.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _warm_record(unit_id, shape):
    p = AUDIT / "profiles" / f"{unit_id}__{shape}__family-warm.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def load_summary():
    """Flatten ``profiles/summary.json`` (``units[uid].modes[fresh|family-warm]``) into
    one dict of CSV timing columns per unit; fresh mode supplies the headline numbers,
    family-warm supplies the per-variant increment at the largest measured shape."""
    p = AUDIT / "profiles" / "summary.json"
    if not p.is_file():
        return {}
    doc = json.loads(p.read_text())
    out = {}
    for uid, u in (doc.get("units") or {}).items():
        modes = u.get("modes") or {}
        fresh = modes.get("fresh") or {}
        warm = modes.get("family-warm") or {}
        row = {}
        measured = fresh.get("measured") or {}
        for shape, col in SHAPE_COLS.items():
            row[col] = measured.get(shape)
        row["measure_mode"] = fresh.get("measure_mode")
        row["machine_factor"] = fresh.get("machine_factor")
        row["time_s_7x100_golden"] = fresh.get("time_s_7x100_golden", warm.get("time_s_7x100_golden"))
        row["P_exponent_measured"] = fresh.get("P_exponent")
        row["T_exponent_measured"] = fresh.get("T_exponent")
        row["projected_window_s_118"] = fresh.get("projected_window_s_118")
        row["projected_patient_s"] = fresh.get("projected_patient_s")
        row["projected_patient_s_family_warm"] = warm.get("projected_patient_s")
        if row["measure_mode"] is None and warm.get("measure_mode"):
            row["measure_mode"] = f"family-warm:{warm['measure_mode']}"
        wm = warm.get("measured") or {}
        for shape in ("118x1024", "32x1024", "8x1024"):
            if wm.get(shape) is not None:
                row["t_incremental_s"] = f"{wm[shape]:.4g} @{shape}"
                break
        for shape in ("118x1024", "32x1024", "8x1024"):
            rec = _fresh_record(uid, shape)
            if rec and rec.get("status") == "ok":
                row["t_cold_s"] = rec.get("t_cold_s")
                row["t_per_pair_ms"] = rec.get("t_per_pair_ms")
                row["peak_rss_mb"] = rec.get("peak_rss_mb")
                break
        for shape in ("118x1024", "32x1024", "8x1024"):
            rec = _warm_record(uid, shape)
            if rec and rec.get("status") == "ok" and rec.get("first_identifier_increment_s") is not None:
                row["t_shared_family_s"] = (f"{rec['first_identifier_increment_s']:.4g} @{shape} "
                                            f"({rec.get('first_identifier')})")
                break
        out[uid] = {k: v for k, v in row.items() if v is not None}
    return out


def build():
    units = yaml.safe_load((AUDIT / "units.yaml").read_text())["units"]
    results = load_results()
    summary = load_summary()
    doc = _summary_doc()
    retained = {i for u in units for i in u["eeg_identifiers"]}
    ubases = {}
    rows, id_rows = [], []
    for u in units:
        rec_path = AUDIT / "records" / f"{u['unit_id']}.yaml"
        rec = yaml.safe_load(rec_path.read_text()) if rec_path.is_file() else {}
        c = rec.get("correctness") or {}
        o = rec.get("optimisation") or {}
        t = summary.get(u["unit_id"], {}) if isinstance(summary, dict) else {}
        base = dict(
            unit_id=u["unit_id"], module=u["module"], pyspi_class=u["pyspi_class"],
            method_or_estimator=u["method_or_estimator"], hyperparameters=_s(u["hyperparameters"]),
            source=u["source"], identifiers_covered=_s(u["identifiers"]), n_identifiers=len(u["identifiers"]),
            representative=_s(u["representative"]), family=u["family"], pair_basis=u["pair_basis"],
            in_eeg_selection=_s(u["in_eeg_selection"]), eeg_identifiers=_s(u["eeg_identifiers"]),
            status=_s(rec.get("status", "")),
        )
        for k in CORRECTNESS:
            if k == "closeness_measured":
                base[k] = results.get(u["unit_id"], "")
            else:
                base[k] = _s(c.get(k))
        for k in TIMING:
            base[k] = _s(o.get(k, t.get(k)))
        ub = ubases[u["unit_id"]] = unit_basis(u, summary, doc, o, retained)
        opps = o.get("opportunities") or []
        if not opps:
            r = dict(base, **{k: "" for k in OPPORTUNITY})
            if o.get("no_opportunity_reason"):
                r["opportunity"] = _s(o["no_opportunity_reason"])
            rows.append(r)
        for opp in opps:
            r = dict(base, **{k: _s(opp.get(k)) for k in OPPORTUNITY})
            tier = _opp_tier(u, opp)
            plan = "eeg" if tier == 0 else "cfg"
            r["_opp"], r["_covered"], r["_tier"], r["_plan"] = opp, [u["unit_id"]], tier, plan
            r["_units"] = {u["unit_id"]: ub[plan]}
            r["_eeg_na"] = tier != _tier(u)
            r["basis_scope"], r["basis_scope_note"] = ((SCOPE_CONFIGURED, EEG_NA_NOTE) if r["_eeg_na"]
                                                       else (ub["scope"], ub["note"]))
            rows.append(r)
        for ident in u["identifiers"]:
            id_rows.append(dict(identifier=ident, unit_id=u["unit_id"], module=u["module"], source=u["source"],
                                in_config=_s(u["source"] == "config"), in_extra=_s(u["source"] == "extra"),
                                in_eeg_selection=_s(ident in u["eeg_identifiers"]),
                                params=_s(u["identifier_params"].get(ident))))
        for ident in u["eeg_identifiers"]:
            if ident not in u["identifiers"]:
                id_rows.append(dict(identifier=ident, unit_id=u["unit_id"], module=u["module"], source="eeg-band-variant",
                                    in_config="false", in_extra="false", in_eeg_selection="true", params=""))
    rows += crosscutting_rows(summary, ubases)
    ranking_basis(rows, ubases)
    rank(rows)
    for r in rows:
        for k in [k for k in r if k.startswith("_")]:
            del r[k]
    return rows, id_rows


EEG_FC_FAMILIES = {"spectral_csm", "spectral_coherence", "spectral_phase_sync", "spectral_psi_gd",
                   "spectral_directed", "sgc", "pec", "it_gaussian", "it_kraskov", "it_kernel",
                   "it_kozachenko"}
RISK_ORDER = ["bit-identical", "within-tolerance", "rng-stream", "semantic"]
EFFORT_ORDER = ["S", "M", "L", "XL"]
_SPEEDUP = re.compile(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:[-–]\s*\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*)?[x×]", re.I)


def _speedup_lower(text):
    m = _SPEEDUP.search(str(text or ""))
    return float(m.group(1)) if m else None


TIER_PART = {0: "retained", 1: "eeg-fc-family", 2: "other"}
SCOPE_RETAINED = "retained-identifiers"
SCOPE_RETAINED_APPROX = "retained-identifiers-approx"
SCOPE_APPROX = "configured-identifiers-approx"
SCOPE_CONFIGURED = "configured-identifiers"
APPROX_SCOPES = (SCOPE_APPROX, SCOPE_RETAINED_APPROX)
EEG_BASIS = ("family-warm", "fresh", "retained-increments")
XWARM_FACTOR = 1.5
SHAPES_DESC = ("118x1024", "32x1024", "8x1024", "8x256", "7x100")
EEG_NA_NOTE = ("eeg_applicable: false (the avenue gives no gain on the EEG selection), so the row ranks outside tier 0 "
               "on the unit's configured identifiers")


def _summary_doc():
    p = AUDIT / "profiles" / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _warm_increments(uid):
    """Per-identifier family-warm increments at 118x1024 ({identifier: min_s}) and the family measurement
    order, from ``profiles/<unit_id>__118x1024__family-warm.json``; (None, []) when absent or not ok."""
    p = AUDIT / "profiles" / f"{uid}__118x1024__family-warm.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None, []
    if d.get("status") != "ok":
        return None, []
    inc = {k: v.get("min_s") for k, v in (d.get("increments_s") or {}).items() if isinstance(v, dict)}
    return inc, list(d.get("family_identifiers") or [])


def _cross_unit_warm(u, doc, retained):
    """Cross-unit cache check of a unit in the EEG selection (README "Optimisation columns", rule 1). At the
    largest shape where both its fresh and its family-warm wall were measured, returns (shape, fresh_s,
    warm_s, before) when the fresh wall exceeds ``XWARM_FACTOR`` times the family-warm one and the
    family-warm run at that shape timed identifiers outside the EEG selection (``before``, not in
    ``retained``) ahead of the unit's first identifier: its family-warm increments may then read a cache
    those identifiers filled, which an EEG run does not run. Returns None otherwise."""
    modes = (((doc.get("units") or {}).get(u["unit_id"]) or {}).get("modes") or {})
    fm = (modes.get("fresh") or {}).get("measured") or {}
    wm = (modes.get("family-warm") or {}).get("measured") or {}
    shape = next((s for s in SHAPES_DESC if fm.get(s) is not None and wm.get(s) is not None), None)
    if shape is None or not fm[shape] > XWARM_FACTOR * wm[shape]:
        return None
    try:
        d = json.loads((AUDIT / "profiles" / f"{u['unit_id']}__{shape}__family-warm.json").read_text())
    except (OSError, ValueError):
        return None
    fam = list(d.get("family_identifiers") or [])
    pos = [fam.index(i) for i in u["identifiers"] if i in fam]
    if d.get("status") != "ok" or not pos:
        return None
    before = [i for i in fam[:min(pos)] if i not in retained]
    return (shape, fm[shape], wm[shape], before) if before else None


def unit_basis(u, summary, doc, opt, retained=frozenset()):
    """Ranking basis of one unit (README "Optimisation columns", rule 1). ``cfg``: the family-warm projection
    over the unit's configured identifiers, else fresh. ``eeg``: what rows in the EEG selection (tier 0)
    rank on. It equals ``cfg`` when the configured identifiers are the retained ones; it is the sum of the
    retained identifiers' family-warm increments at 118x1024 times the windows per patient when that run
    timed every retained identifier before the non-retained ones; otherwise it stays ``cfg`` with
    ``scope`` configured-identifiers-approx. The record's ``optimisation.eeg_basis`` then decides
    cross-unit cache cases (``_cross_unit_warm``) and band-variant cases: ``fresh`` ranks on the fresh
    projection, ``retained-increments`` on the retained identifiers' increments although EEG identifiers
    were never timed (scope retained-identifiers-approx), ``family-warm`` keeps the family-warm figure;
    a unit that ``_cross_unit_warm`` flags and whose record names no ``eeg_basis`` gets scope
    configured-identifiers-approx. ``note`` gives the counted reason, followed by the record's
    ``optimisation.basis_scope_note`` when there is one; ``how`` is ``retained`` or ``fresh`` when ``eeg``
    is not ``cfg``. Units outside the EEG selection get scope configured-identifiers and no note."""
    uid = u["unit_id"]
    t = summary.get(uid, {}) if isinstance(summary, dict) else {}
    cfg = t.get("projected_patient_s_family_warm", t.get("projected_patient_s"))
    out = dict(cfg=cfg, eeg=cfg, tier0=bool(u.get("in_eeg_selection")), scope=SCOPE_CONFIGURED, note="", how=None)
    if not out["tier0"] or cfg is None:
        return out
    ids, eeg = list(u["identifiers"]), list(u["eeg_identifiers"])
    ret = [i for i in ids if i in eeg]
    nonret = [i for i in ids if i not in eeg]
    untimed = [i for i in eeg if i not in ids]
    inc, order = _warm_increments(uid)
    warm = (((doc.get("units") or {}).get(uid) or {}).get("modes") or {}).get("family-warm") or {}
    w118 = (warm.get("measured") or {}).get("118x1024")
    windows = doc.get("windows_per_patient") or 600
    timed = (inc is not None and w118 is not None and t.get("projected_patient_s_family_warm") is not None
             and all(inc.get(i) is not None for i in ids)
             and abs(sum(inc[i] for i in ids) - w118) <= 1e-9 * max(1.0, w118)
             and abs(w118 * windows - cfg) <= 1e-6 * max(1.0, cfg))
    if not nonret and not untimed:
        out.update(scope=SCOPE_RETAINED, note=f"the {len(ids)} configured identifier(s) are the unit's retained identifiers")
    elif untimed:
        share = f", {100 * sum(inc[i] for i in nonret) / w118:.0f}% of the basis" if timed and nonret else ""
        out.update(scope=SCOPE_APPROX, note=(
            f"{len(untimed)} of {len(eeg)} EEG identifiers never timed (EEG band variants outside the configuration); "
            f"the basis sums the {len(ids)} configured identifiers, {len(nonret)} of them outside the EEG selection{share}"))
    elif timed and all(i in order for i in ids) and max(order.index(i) for i in ret) < min(order.index(i) for i in nonret):
        out.update(eeg=sum(inc[i] for i in ret) * windows, scope=SCOPE_RETAINED, how="retained", note=(
            "sum of the retained identifiers' family-warm increments at 118x1024 ("
            + ", ".join(f"{i} {inc[i]:.3g} s" for i in ret) + f") x {windows} windows; "
            + ", ".join(f"{i} ({inc[i]:.3g} s)" for i in nonret)
            + " not retained, timed after them in the same run and left out"))
    else:
        why = ("measured after it in the family-warm run, so its increment may read its cache ("
               + ", ".join(f"{i} {inc[i]:.3g} s" for i in ids) + ")") if timed else \
            "no per-identifier family-warm increments at 118x1024"
        out.update(scope=SCOPE_APPROX, note=(
            f"{', '.join(nonret)} not retained, but the retained {', '.join(ret)} cannot be timed apart from it: "
            f"{why}; the configured sum is kept"))
    mode = str((opt or {}).get("eeg_basis") or "").strip()
    xw = _cross_unit_warm(u, doc, retained)
    xnote = (f"the family-warm run at {xw[0]} timed {', '.join(xw[3])} (outside the EEG selection) before this unit and its "
             f"fresh wall there is {xw[1]:.3g} s against family-warm {xw[2]:.3g} s, so its family-warm increments may read "
             f"a cache those identifiers filled") if xw else ""
    add = lambda s: out.update(note=f"{out['note']}; {s}" if out["note"] else s)
    if mode and mode not in EEG_BASIS:
        add(f"eeg_basis {mode!r} is not one of {', '.join(EEG_BASIS)} and is ignored")
        mode = ""
    if mode == "fresh" and t.get("projected_patient_s") is not None:
        out.update(eeg=t["projected_patient_s"], how="fresh",
                   scope=SCOPE_RETAINED if not nonret and not untimed else SCOPE_APPROX)
        add(f"eeg_basis fresh: ranked on the fresh projection ({t.get('projected_window_s_118', 0):.3g} s per window, "
            f"{t['projected_patient_s']:.4g} s per patient) in place of the family-warm {cfg:.3g} s per patient"
            + (f"; {xnote}" if xnote else ""))
    elif mode == "retained-increments":
        if timed and ret and all(i in order for i in ids) and (
                not nonret or max(order.index(i) for i in ret) < min(order.index(i) for i in nonret)):
            out.update(eeg=sum(inc[i] for i in ret) * windows, how="retained",
                       scope=SCOPE_RETAINED_APPROX if untimed else SCOPE_RETAINED, note=(
                           "eeg_basis retained-increments: ranked on the retained identifiers' family-warm increments at "
                           "118x1024 (" + ", ".join(f"{i} {inc[i]:.3g} s" for i in ret) + f") x {windows} windows"
                           + ("; " + ", ".join(f"{i} ({inc[i]:.3g} s)" for i in nonret)
                              + " not retained, timed after them and left out" if nonret else "")
                           + (f"; {len(untimed)} of {len(eeg)} EEG identifiers never timed (EEG band variants outside the "
                              "configuration) and left out" if untimed else "")))
        else:
            add("eeg_basis retained-increments not applied: the family-warm run at 118x1024 did not time every retained "
                "identifier before the non-retained ones")
    elif xw:
        if mode == "family-warm":
            add(f"eeg_basis family-warm: the family-warm figure is kept although {xnote}")
        else:
            out.update(scope=SCOPE_APPROX)
            add(f"{xnote}; the record names no eeg_basis, so the family-warm figure is kept as an approximation")
    extra = str((opt or {}).get("basis_scope_note") or "").strip()
    if extra:
        out["note"] = f"{out['note']}; {extra}" if out["note"] else extra
    return out


def _opp_tier(u, opp):
    """Tier of a unit's opportunity row (README rule 2): the unit's tier, except that ``eeg_applicable: false``
    on a unit in the EEG selection moves the row to the tier the unit would have outside it."""
    t = _tier(u)
    if t == 0 and opp.get("eeg_applicable") is False:
        return 1 if u.get("family") in EEG_FC_FAMILIES else 2
    return t


def _k_for(opp, uid):
    """Lower-bound speedup of ``opp`` on unit ``uid`` and its source: the first ``scope_speedup`` entry whose
    ``scope`` globs match the unit (cross-cutting opportunities), else ``expected_speedup`` (source None)."""
    for ent in opp.get("scope_speedup") or []:
        if any(fnmatch.fnmatch(uid, g) for g in _ids(ent.get("scope"))):
            return _speedup_lower(ent.get("expected_speedup")), "scope_speedup"
    return _speedup_lower(opp.get("expected_speedup")), None


def _ids(v):
    if not v:
        return []
    return [v] if isinstance(v, str) else [x for x in v if x]


def _kind(opp):
    """``throughput`` (wall-clock on more cores, no per-window CPU saving) or ``projection`` (a per-window
    saving projected for hardware or software that is not installed here; no code ran) when the record
    says so, else ``per-window``."""
    k = str(opp.get("speedup_kind") or "").strip()
    return k if k in ("throughput", "projection") else "per-window"


def _closure(uid, start, index, self_id):
    """Ids reachable from ``start`` (then through ``depends_on``) that are per-window opportunities
    applying to unit ``uid``: an opportunity of that unit, or a cross-cutting one whose scope covers it."""
    out, todo = [], list(start)
    while todo:
        d = todo.pop(0)
        p = index.get((uid, d))
        if d == self_id or d in out or p is None or _kind(p) == "throughput":
            continue
        out.append(d)
        todo += _ids(p.get("depends_on"))
    return out


def _fmt(x):
    return "" if x is None else f"{x:.6g}"


def _effort_key(effort):
    return EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else 9


def _unit_plan(uid, b, avs, index):
    """Stack the per-window avenues ``avs`` ({id: opp}) of unit ``uid`` (basis ``b``; README ranking rule 4).
    An avenue is placed after everything it names in ``depends_on`` / ``subsumed_by`` on this unit
    (transitively); among the avenues that are free to go next, the one with the largest lower-bound
    speedup goes first (``projection`` avenues after the others, then risk, effort, id). Each avenue is
    ranked on the residual left by every avenue placed before it; a subsumed avenue saves nothing and
    does not shrink the residual of later ones. A ``subsumed_by`` entry counts only when that avenue is in
    ``avs`` (this plan): one with ``eeg_applicable: false`` is left out of the EEG plan, so there the
    avenue it subsumes keeps its own k. Its k is the lower bound of ``residual_speedup`` when one
    of its ``depends_on`` prerequisites is stacked here, else of a matching ``scope_speedup`` entry, else
    of ``expected_speedup``. Returns {id: dict(resid, after, deps, subs, stacked, indep, eff, ksrc)}."""
    info = {}
    for oid, opp in avs.items():
        k, ksrc = _k_for(opp, uid)
        deps = _closure(uid, _ids(opp.get("depends_on")), index, oid)
        subs = [d for d in _ids(opp.get("subsumed_by")) if d in avs and _closure(uid, [d], index, oid)]
        req = deps + [d for d in _closure(uid, subs, index, oid) if d not in deps]
        if opp.get("residual_speedup") and any(d in avs for d in deps):
            k, ksrc = _speedup_lower(opp.get("residual_speedup")), "residual_speedup"
        k = k if k and k > 0 else None
        info[oid] = dict(k=k, ksrc=ksrc if k else None, deps=deps, subs=subs, req=req, opp=opp,
                         eff=1.0 if subs or k is None else k)
    order, left = [], set(avs)
    while left:
        ready = [o for o in left if all(p in order or p not in avs for p in info[o]["req"])] or sorted(left)
        o = min(ready, key=lambda o: (_kind(info[o]["opp"]) == "projection", -info[o]["eff"],
                                      _risk_key(info[o]["opp"].get("output_risk")),
                                      _effort_key(info[o]["opp"].get("effort")), o))
        order.append(o)
        left.discard(o)
    out, f = {}, 1.0
    for i, o in enumerate(order):
        x = info[o]
        resid = b / f
        after = resid if x["subs"] else (resid / x["k"] if x["k"] else None)
        stacked = [p for p in order[:i] if p not in x["req"] and info[p]["eff"] > 1.0]
        indep = [p for p in stacked if p in _ids(x["opp"].get("independent_of"))
                 or o in _ids(info[p]["opp"].get("independent_of"))]
        if (indep and not x["subs"] and x["ksrc"] != "residual_speedup"
                and x["opp"].get("residual_speedup")):
            # disjoint frames: the seconds the avenue saves are a larger share of the residual
            k = _speedup_lower(x["opp"].get("residual_speedup"))
            if k and k > 0:
                x.update(k=k, ksrc="residual_speedup_indep", eff=k)
                after = resid / k
        out[o] = dict(resid=resid, after=after, deps=[d for d in x["deps"] if d in avs], subs=x["subs"],
                      stacked=[p for p in stacked if p not in indep], indep=indep, eff=x["eff"],
                      ksrc=None if x["subs"] else x["ksrc"])
        f *= x["eff"]
    return out


def ranking_basis(rows, ubases):
    """Set ``speedup_kind``, ``rank_basis_s``, ``projected_patient_s_after``, ``rank_units`` and ``rank_rule``
    on opportunity rows (README "Optimisation columns", ranking rules 3-4). Per covered unit the row takes
    its residual and after-figure from that unit's stacking (``_unit_plan``): rows in tier 0 from the
    unit's EEG-selection plan (basis ``ubases[unit]["eeg"]``, avenues without ``eeg_applicable: false``),
    every other row from its configured plan (basis ``cfg``, every avenue that applies to the unit).
    Throughput avenues are not stacked and are never prerequisites (a throughput row is ranked on the
    residual after the per-window avenues it names in ``depends_on``, if any). Unit sums give the row's
    figures."""
    index, avs = {}, {}
    for r in rows:
        opp = r.get("_opp")
        if opp is None:
            continue
        oid = opp["opportunity_id"]
        for uid in r.get("_covered") or []:
            index.setdefault((uid, oid), opp)
            if _kind(opp) != "throughput" and uid in ubases:
                avs.setdefault(uid, {}).setdefault(oid, opp)
    plans = {}
    for uid, a in avs.items():
        ub = ubases[uid]
        if ub["cfg"] is not None:
            plans[("cfg", uid)] = _unit_plan(uid, ub["cfg"], a, index)
        if ub["tier0"] and ub["eeg"] is not None:
            plans[("eeg", uid)] = _unit_plan(uid, ub["eeg"], {o: p for o, p in a.items()
                                                              if p.get("eeg_applicable") is not False}, index)
    for r in rows:
        opp = r.get("_opp")
        if opp is None:
            continue
        oid, kind = opp["opportunity_id"], _kind(opp)
        k = _speedup_lower(opp.get("expected_speedup"))
        k = k if k and k > 0 else None
        units = r.get("_units")
        if units is None:
            units = {None: r.get("_explicit")}
        total, after, missing, n = None, 0.0, False, 0
        seen = {"subs": {}, "deps": {}, "stacked": {}, "indep": {}}
        ksrc = {}
        plan = r.get("_plan", "cfg")
        for uid, b in units.items():
            if b is None:
                continue
            n += 1
            pl = plans.get((plan, uid), {})
            p = pl.get(oid) if kind != "throughput" else None
            if p is None:
                deps = _closure(uid, _ids(opp.get("depends_on")), index, oid) if uid else []
                deps = [d for d in deps if d in pl]
                f = 1.0
                for d in deps:
                    f *= pl[d]["eff"]
                p = dict(resid=b / f, after=(b / f / k) if k else None, deps=deps, subs=[], stacked=[], indep=[],
                         ksrc=None)
            total = (total or 0.0) + p["resid"]
            if p["after"] is None:
                missing = True
            else:
                after += p["after"]
            for key in seen:
                for d in p[key]:
                    seen[key][d] = seen[key].get(d, 0) + 1
            if p.get("ksrc"):
                ksrc.setdefault(p["ksrc"], []).append(uid)
        after = None if total is None or missing else after
        order = _ids(opp.get("subsumed_by"))
        seen["subs"] = {d: seen["subs"][d] for d in sorted(seen["subs"], key=order.index)}
        r["speedup_kind"] = kind
        r["_rank_basis"], r["_after"] = total, after
        r["rank_basis_s"], r["projected_patient_s_after"] = _fmt(total), _fmt(after)
        r["rank_units"] = _s(r.get("_covered") or [])
        if kind == "throughput":
            rule = ["throughput: wall-clock on more cores, no per-window CPU saving; ranked after every per-window row"]
        elif kind == "projection":
            rule = ["projection: per-window saving projected for hardware or software not installed here (no code ran); "
                    "ranked after the other per-window rows of its tier"]
        else:
            rule = ["per-window"]
        ids = lambda d: ", ".join(f"{x}" + ("" if m == n else f" (on {m} of {n} units)") for x, m in d.items())
        if seen["subs"]:
            rule.append(f"subsumed by {ids(seen['subs'])}: no saving there")
        if seen["deps"]:
            rule.append(f"residual after depends_on {ids(seen['deps'])}")
        if seen["indep"]:
            rule.append(f"stacked after {ids(seen['indep'])}: recorded as independent (independent_of), "
                        + ("disjoint frames" if "residual_speedup_indep" in ksrc else "its own k applied to the residual"))
        if seen["stacked"]:
            rule.append(f"stacked after {ids(seen['stacked'])}: same-unit avenues with no link and a larger lower-bound speedup")
        if not any(seen.values()):
            rule.append("on today's cost")
        on = lambda us: "" if len(us) == n else f" (on {len(us)} of {n} units)"
        if "residual_speedup" in ksrc:
            rule.append(f"k = lower bound of residual_speedup {opp.get('residual_speedup')} on the residual after its depends_on"
                        + on(ksrc["residual_speedup"]))
        if "residual_speedup_indep" in ksrc:
            rule.append(f"k = lower bound of residual_speedup {opp.get('residual_speedup')} on the residual after that avenue"
                        + on(ksrc["residual_speedup_indep"]))
        if "scope_speedup" in ksrc:
            us = ksrc["scope_speedup"]
            figs = sorted({str(e.get("expected_speedup")) for e in opp.get("scope_speedup") or []
                           if any(fnmatch.fnmatch(u, g) for u in us for g in _ids(e.get("scope")))})
            rule.append(f"k from scope_speedup ({', '.join(figs)}) on {', '.join(us)}" + on(us))
        if r.get("_eeg_na"):
            rule.append("eeg_applicable: false, so ranked outside tier 0 on the configured identifiers")
        r["rank_rule"] = "; ".join(rule)


def _tier(u):
    if u.get("in_eeg_selection"):
        return 0
    return 1 if u.get("family") in EEG_FC_FAMILIES else 2


def _risk_key(risk):
    risk = str(risk or "")
    base = risk[len("unverified-"):] if risk.startswith("unverified-") else risk
    i = RISK_ORDER.index(base) if base in RISK_ORDER else len(RISK_ORDER)
    return (1 if risk.startswith("unverified-") else 0, i)


def rank(rows):
    """Write ``priority_rank`` (1..N) on opportunity rows; rows without one stay blank. Per-window rows
    come before throughput rows; then tier, projection rows after the other per-window rows of the tier,
    patient seconds saved on ``rank_basis_s``, risk, effort."""
    def saved(r):
        b, a = r.get("_rank_basis"), r.get("_after")
        return 0.0 if b is None or a is None else b - a
    opp_rows = [r for r in rows if r.get("opportunity_id")]
    opp_rows.sort(key=lambda r: (1 if r.get("speedup_kind") == "throughput" else 0,
                                 r.get("_tier", 2), 1 if r.get("speedup_kind") == "projection" else 0,
                                 -saved(r), _risk_key(r.get("output_risk")), _effort_key(r.get("effort")),
                                 r["unit_id"], r["opportunity_id"]))
    for i, r in enumerate(opp_rows, 1):
        r["priority_rank"] = str(i)


def _part_scope(us, ubases, plan):
    """``basis_scope`` / ``basis_scope_note`` of a cross-cutting part from its units' scopes."""
    if plan != "eeg":
        return SCOPE_CONFIGURED, ""
    approx = [u for u in us if ubases[u]["scope"] in APPROX_SCOPES]
    cut = [u for u in us if ubases[u].get("how") == "retained"]
    fresh = [u for u in us if ubases[u].get("how") == "fresh"]
    tail = (f"; restricted to the retained identifiers on {', '.join(cut)}" if cut else "") + \
        (f"; fresh projection (eeg_basis fresh) on {', '.join(fresh)}" if fresh else "")
    if not approx:
        return SCOPE_RETAINED, f"every unit's basis covers its retained identifiers ({len(us)} units){tail}"
    return SCOPE_APPROX, (f"approximate on {len(approx)} of {len(us)} units ({', '.join(approx)}; "
                          f"see those units' basis_scope_note){tail}")


def crosscutting_rows(summary, ubases):
    """Rows for ``profiles/crosscutting.yaml`` (absent file -> no rows). A row with an explicit
    ``projected_patient_s`` keeps it (tier 0 when ``eeg_relevant``, else 2) and takes ``basis_scope`` /
    ``basis_scope_note`` from the row. Otherwise the covered units are grouped by tier (``_tier``; every
    group tier 2 when ``eeg_relevant`` is false) and each group becomes its own row whose basis sums that
    group's units only, each unit on the basis of ``unit_basis`` for that tier (``eeg`` in tier 0, else
    ``cfg``). ``identifiers_covered`` / ``n_identifiers`` / ``eeg_identifiers`` list the identifiers of the
    row's units; ``rank_units`` (set in ``ranking_basis``) lists the units."""
    p = AUDIT / "profiles" / "crosscutting.yaml"
    if not p.is_file():
        return []
    doc = yaml.safe_load(p.read_text()) or {}
    by_id = {u["unit_id"]: u for u in yaml.safe_load((AUDIT / "units.yaml").read_text())["units"]}
    out = []
    for c in doc.get("rows") or []:
        scope = c.get("scope") or []
        covered = sorted({u for u in by_id for g in scope if fnmatch.fnmatch(u, g)})
        relevant = c.get("eeg_relevant", True)
        explicit = c.get("projected_patient_s")
        if explicit is not None or not covered:
            parts = [(c["unit_id"], 0 if relevant else 2, covered, None)]
        else:
            groups = {}
            for u in covered:
                groups.setdefault(_tier(by_id[u]) if relevant else 2, []).append(u)
            parts = [(c["unit_id"] if len(groups) == 1 else f"{c['unit_id']}/{TIER_PART[t]}", t, us,
                      {u: ubases[u]["eeg" if t == 0 else "cfg"] for u in us}) for t, us in sorted(groups.items())]
        for row_id, tier, us, ubasis in parts:
            if ubasis is None:
                basis = explicit
            else:
                vals = [v for v in ubasis.values() if v is not None]
                basis = sum(vals) if vals else None
            base = {k: "" for k in COLUMNS}
            idents = [i for u in us for i in by_id[u]["identifiers"]]
            eeg_idents = [i for u in us for i in by_id[u]["eeg_identifiers"]]
            base.update(unit_id=row_id, module="cross-cutting", family=c["unit_id"].split(":", 1)[-1],
                        method_or_estimator=_s(c.get("title")), identifiers_covered=_s(idents),
                        n_identifiers=len(idents), eeg_identifiers=_s(eeg_idents),
                        in_eeg_selection=_s(tier == 0), status="profiled",
                        projected_patient_s=_s(basis), implemented_as=_s(c.get("evidence")))
            for k in ("complexity_P_T", "hotspot_1", "hotspot_2", "hotspot_owner", "opaque_share", "profile_file"):
                base[k] = _s(c.get(k))
            plan = "eeg" if tier == 0 and ubasis is not None else "cfg"
            if ubasis is None:
                scope, snote = _s(c.get("basis_scope")), _s(c.get("basis_scope_note"))
            else:
                scope, snote = _part_scope(us, ubases, plan)
            opps = c.get("opportunities") or []
            if not opps:
                out.append(base)
            for opp in opps:
                r = dict(base, **{k: _s(opp.get(k)) for k in OPPORTUNITY})
                r["_opp"], r["_units"], r["_covered"], r["_tier"], r["_plan"] = opp, ubasis, us, tier, plan
                r["basis_scope"], r["basis_scope_note"] = scope, snote
                r["_explicit"] = basis if ubasis is None else None
                out.append(r)
    return out


def to_csv(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in columns})
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows, id_rows = build()
    csv1 = to_csv(rows, COLUMNS)
    csv2 = to_csv(id_rows, ["identifier", "unit_id", "module", "source", "in_config", "in_extra",
                            "in_eeg_selection", "params"])
    p1, p2 = AUDIT / "metrics_audit.csv", AUDIT / "identifier_map.csv"
    if a.check:
        stale = [p for p, c in ((p1, csv1), (p2, csv2)) if not p.is_file() or p.read_text() != c]
        if stale:
            print("stale: " + ", ".join(str(p) for p in stale), file=sys.stderr)
            sys.exit(1)
        print("CSVs up to date")
        return
    p1.write_text(csv1)
    p2.write_text(csv2)
    print(f"wrote {p1} ({len(rows)} rows) and {p2} ({len(id_rows)} rows)")


if __name__ == "__main__":
    main()
