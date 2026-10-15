"""Validate ``audit/records/*.yaml`` against ``records/SCHEMA.md`` and ``units.yaml``.

Usage: ``python audit/validate_records.py [--stage traced|checked|profiled] [unit_id ...]``
Exit status non-zero when any record is missing, malformed, or lacks a required
field for the stage it claims.
"""
import argparse
import sys
from pathlib import Path

import yaml

AUDIT = Path(__file__).resolve().parent
RECORDS = AUDIT / "records"

TRACED = ["metric_name_standard", "canonical_ref", "canonical_definition", "documented_definition",
          "delegate_lib", "implemented_as", "implementation_trace", "deviation_from_documented",
          "deviation_from_canonical", "units_and_range", "direction_convention", "symmetry_expected",
          "already_documented", "new_finding_candidates", "proposed_checks"]
CHECKED = ["check_method", "check_reference", "check_data", "agreement_criterion", "closeness",
           "verdict", "score_0_10", "what_changed", "new_finding", "definition_under_review", "check_cost",
           "symmetry_observed", "direction_observed", "candidate_resolutions", "checks_run"]
RESOLUTIONS = {"confirmed", "refuted", "inconclusive"}
PROFILED = ["complexity_P_T", "hotspot_1", "hotspot_owner", "profile_file", "opportunities"]
VERDICTS = {"matches", "variant", "related-quantity", "wrong", "degenerate", "unverified"}
METHODS = {"analytic", "independent-impl", "external-pkg", "same-lineage", "literature-only", "unverifiable"}
STAGES = ["traced", "checked", "profiled"]
SCORE_COLS = ["score_rubric", "score_rubric_basis", "score_rubric_note", "documentation_status",
              "verdict_adjusted", "score_adjusted", "score_adjustment_rules", "score_adjustment_reason",
              "identifier_exceptions"]
BASES = {"anchor", "nearest-anchor", "none"}
DOC_STATUS = {"described", "metadata-conflict", "misdescribed", "silent"}
RULES = {"R1", "R2", "R3", "R5"}
# verdict word -> allowed score_adjusted range (README glossary R6)
VERDICT_RANGE = {"matches": (9, 10), "variant": (5, 8), "related-quantity": (2, 5), "wrong": (1, 3),
                 "degenerate": (0, 0)}
BANNED = ("golden values are correct", "goldens are verified", "verified golden", "known-good golden")


def _score_columns(name, c):
    """Score columns (README "Score columns"): original score_0_10 kept for provenance, the rubric
    reading, and the adjusted score; enums and the invariants between them."""
    errs = [f"{name}: correctness.{k} missing" for k in SCORE_COLS if k not in c]
    if errs:
        return errs
    s0, sr, sa = c.get("score_0_10"), c.get("score_rubric"), c.get("score_adjusted")
    basis = c.get("score_rubric_basis")
    note = (c.get("score_rubric_note") or "").strip()
    rules = (c.get("score_adjustment_rules") or "").strip()
    reason = (c.get("score_adjustment_reason") or "").strip()
    if basis not in BASES:
        errs.append(f"{name}: score_rubric_basis {basis!r} not in {sorted(BASES)}")
    if c.get("documentation_status") not in DOC_STATUS:
        errs.append(f"{name}: documentation_status {c.get('documentation_status')!r} not in {sorted(DOC_STATUS)}")
    if c.get("verdict_adjusted") not in VERDICTS:
        errs.append(f"{name}: verdict_adjusted {c.get('verdict_adjusted')!r} not in {sorted(VERDICTS)}")
    for k, v in (("score_rubric", sr), ("score_adjusted", sa)):
        if (s0 is None) != (v is None):
            errs.append(f"{name}: {k} must be filled exactly when score_0_10 is")
        elif v is not None and not (isinstance(v, int) and 0 <= v <= 10):
            errs.append(f"{name}: {k} must be an integer 0-10")
    if (basis == "none") != (sr is None):
        errs.append(f"{name}: score_rubric_basis is none iff score_rubric is null")
    if basis == "nearest-anchor" and not note:
        errs.append(f"{name}: nearest-anchor needs score_rubric_note")
    if sr != s0 and not note:
        errs.append(f"{name}: score_rubric differs from score_0_10, so score_rubric_note must say why")
    ids = [r.strip() for r in rules.split(";") if r.strip()]
    if any(r not in RULES for r in ids):
        errs.append(f"{name}: score_adjustment_rules {rules!r} may only name {sorted(RULES)}")
    if (sa == sr) != (not ids):
        errs.append(f"{name}: score_adjusted differs from score_rubric exactly when score_adjustment_rules is non-empty")
    if ids and not all(reason.find(r + ":") >= 0 for r in ids):
        errs.append(f"{name}: score_adjustment_reason needs one '<rule>: <sentence>' per applied rule")
    if not ids and reason:
        errs.append(f"{name}: score_adjustment_reason must be empty when no rule applies")
    if c.get("verdict_adjusted") != c.get("verdict") and "R1" not in ids:
        errs.append(f"{name}: verdict_adjusted differs from verdict only under rule R1")
    for vk, sk, v_, sc in (("verdict", "score_rubric", c.get("verdict"), sr),
                           ("verdict_adjusted", "score_adjusted", c.get("verdict_adjusted"), sa)):
        rng = VERDICT_RANGE.get(v_)
        if rng and isinstance(sc, int) and not rng[0] <= sc <= rng[1]:
            errs.append(f"{name}: {vk} {v_!r} needs {sk} in {rng[0]}-{rng[1]}, got {sc}")
    return errs


def validate(path, unit_ids, stage_required=None):
    errs = []
    try:
        rec = yaml.safe_load(path.read_text())
    except Exception as exc:
        return [f"{path.name}: YAML error {exc}"]
    if not isinstance(rec, dict):
        return [f"{path.name}: not a mapping"]
    uid = rec.get("unit_id")
    if uid != path.stem:
        errs.append(f"{path.name}: unit_id {uid!r} != file stem")
    if uid not in unit_ids:
        errs.append(f"{path.name}: unit_id not in units.yaml")
    status = rec.get("status")
    if status not in STAGES:
        errs.append(f"{path.name}: status must be one of {STAGES}")
    stage = stage_required or status
    c = rec.get("correctness") or {}
    need = TRACED + (CHECKED if stage in ("checked", "profiled") else [])
    for k in need:
        v = c.get(k, None)
        if k == "candidate_resolutions" and isinstance(v, list):
            continue   # may be empty when new_finding_candidates is empty; checked below
        if v is None or v == "" or (isinstance(v, list) and not v and k in ("implementation_trace", "proposed_checks")):
            errs.append(f"{path.name}: correctness.{k} missing")
    if stage in ("checked", "profiled"):
        if c.get("verdict") not in VERDICTS:
            errs.append(f"{path.name}: verdict {c.get('verdict')!r} not in {sorted(VERDICTS)}")
        if c.get("check_method") not in METHODS:
            errs.append(f"{path.name}: check_method {c.get('check_method')!r} invalid")
        s = c.get("score_0_10")
        if c.get("verdict") != "unverified" and not (isinstance(s, (int, float)) and 0 <= s <= 10):
            errs.append(f"{path.name}: score_0_10 must be 0-10 unless verdict is unverified")
        errs += _score_columns(path.name, c)
        if c.get("check_method") == "same-lineage" and isinstance(s, (int, float)) and s > 6:
            errs.append(f"{path.name}: same-lineage checks cap the score at 6")
        if c.get("verdict") == "matches" and c.get("check_method") in ("same-lineage", "literature-only", "unverifiable"):
            errs.append(f"{path.name}: verdict 'matches' needs an analytic, independent-impl or external-pkg check_method")
        cands = c.get("new_finding_candidates") or []
        res = c.get("candidate_resolutions")
        if isinstance(res, list):
            if len(res) != len(cands):
                errs.append(f"{path.name}: candidate_resolutions has {len(res)} entries for {len(cands)} new_finding_candidates")
            for i, r in enumerate(res):
                if not isinstance(r, dict) or r.get("outcome") not in RESOLUTIONS or not r.get("evidence"):
                    errs.append(f"{path.name}: candidate_resolutions[{i}] needs outcome in {sorted(RESOLUTIONS)} and evidence")
            if c.get("new_finding") is True and not any(isinstance(r, dict) and r.get("outcome") == "confirmed" for r in res):
                errs.append(f"{path.name}: new_finding is true but no candidate_resolutions outcome is confirmed")
        elif cands:
            errs.append(f"{path.name}: candidate_resolutions must be a list (one entry per new_finding_candidates item)")
        cr = c.get("checks_run")
        if not isinstance(cr, list) or not cr:
            errs.append(f"{path.name}: checks_run must be a non-empty list")
        else:
            for i, r in enumerate(cr):
                if not isinstance(r, dict) or not r.get("run_ids") or not r.get("reference") or not r.get("outcome"):
                    errs.append(f"{path.name}: checks_run[{i}] needs run_ids, reference, outcome")
    if stage == "profiled":
        o = rec.get("optimisation") or {}
        for k in PROFILED:
            if o.get(k) in (None, ""):
                errs.append(f"{path.name}: optimisation.{k} missing")
        for opp in o.get("opportunities") or []:
            for k in ("opportunity_id", "opportunity", "opp_type", "expected_speedup", "speedup_basis", "output_risk", "effort"):
                if opp.get(k) in (None, ""):
                    errs.append(f"{path.name}: opportunity {opp.get('opportunity_id')!r} missing {k}")
    text = path.read_text().lower()
    for b in BANNED:
        if b in text:
            errs.append(f"{path.name}: wording rule: contains {b!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("unit_ids", nargs="*")
    a = ap.parse_args()
    units = yaml.safe_load((AUDIT / "units.yaml").read_text())["units"]
    unit_ids = {u["unit_id"] for u in units}
    targets = a.unit_ids or sorted(unit_ids)
    errs, present = [], 0
    for uid in targets:
        p = RECORDS / f"{uid}.yaml"
        if not p.is_file():
            errs.append(f"{uid}: record missing")
            continue
        present += 1
        errs += validate(p, unit_ids, a.stage)
    for e in errs:
        print(e)
    print(f"{present}/{len(targets)} records present, {len(errs)} problems")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
