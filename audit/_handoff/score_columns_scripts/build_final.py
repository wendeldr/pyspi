"""Build final_entries.json for apply.py from the score-columns workflow output.

Usage: python build_final.py <workflow_output.json> <out final_entries.json> <out report.json>
Takes each batch's check-stage entries (classify entries if a check is missing), applies the
consistency pass corrections, maps the workflow's old field names onto the adopted columns
(score_0_10 -> score_rubric, score_0_10_basis -> score_rubric_basis, score_0_10_note -> score_rubric_note,
verdict -> verdict_adjusted), keeps every record's original score_0_10 and verdict untouched, and checks
the README invariants. Problems go to the report; nothing is silently fixed except score_rubric_note
being required where score_rubric differs from the original (listed in report.needs_note).
"""
import json, sys
from pathlib import Path
import yaml

REC = Path('/media/dan/Data/git/pyspi/audit/records')
RANGE = {'matches': (9, 10), 'variant': (5, 8), 'related-quantity': (2, 5), 'wrong': (1, 3), 'degenerate': (0, 0)}
RULES = {'R1', 'R2', 'R3', 'R5'}


def main():
    wf = json.load(open(sys.argv[1]))
    res = wf.get('result', wf)
    by_uid, src = {}, {}
    for b in res['batches']:
        ents = (b.get('check') or {}).get('entries') or b['classify']['entries']
        for e in ents:
            by_uid[e['unit_id']] = dict(e)
            src[e['unit_id']] = b['key']
    cons = res.get('consistency') or {}
    for e in cons.get('entries', []):
        by_uid[e['unit_id']] = dict(e)
    records = {p.stem: yaml.safe_load(p.read_text()) for p in REC.glob('*.yaml')}
    report = {'missing': sorted(set(records) - set(by_uid)), 'extra': sorted(set(by_uid) - set(records)),
              'problems': [], 'needs_note': [], 'rubric_differs': [], 'adjusted_differs': [], 'verdict_adjusted_differs': [],
              'concerns': [], 'changes': [], 'consistency_changes': cons.get('changes', [])}
    for b in res['batches']:
        report['concerns'] += [dict(c, batch=b['key']) for c in ((b.get('check') or {}).get('concerns') or [])]
        report['changes'] += [dict(c, batch=b['key']) for c in ((b.get('check') or {}).get('changes') or [])]
    report['concerns'] += [dict(c, batch='consistency') for c in cons.get('concerns', [])]
    out = {}
    for uid, e in sorted(by_uid.items()):
        if uid not in records:
            continue
        c = records[uid]['correctness']
        f = {'score_rubric': e['score_0_10'], 'score_rubric_basis': e['score_0_10_basis'],
             'score_rubric_note': e['score_0_10_note'], 'documentation_status': e['documentation_status'],
             'verdict_adjusted': e['verdict'], 'score_adjusted': e['score_adjusted'],
             'score_adjustment_rules': e['score_adjustment_rules'], 'score_adjustment_reason': e['score_adjustment_reason'],
             'identifier_exceptions': e['identifier_exceptions']}
        out[uid] = f
        p = lambda m: report['problems'].append(f'{uid}: {m}')
        rules = [r for r in f['score_adjustment_rules'].split(';') if r]
        if set(rules) - RULES: p(f'unknown rules {rules}')
        if (f['score_adjusted'] != f['score_rubric']) != bool(rules): p('adjusted != rubric iff rules non-empty fails')
        for r in rules:
            if f'{r}:' not in f['score_adjustment_reason']: p(f'no reason for {r}')
        if f['score_rubric_basis'] == 'nearest-anchor' and not f['score_rubric_note'].strip(): p('nearest-anchor without note')
        if (f['score_rubric_basis'] == 'none') != (f['score_rubric'] is None): p('basis none iff score null fails')
        if f['verdict_adjusted'] != c['verdict'] and 'R1' not in rules: p(f"verdict_adjusted {f['verdict_adjusted']} != verdict {c['verdict']} without R1")
        for v, s, lab in ((c['verdict'], f['score_rubric'], 'verdict/score_rubric'), (f['verdict_adjusted'], f['score_adjusted'], 'verdict_adjusted/score_adjusted')):
            if v in RANGE and s is not None and not RANGE[v][0] <= s <= RANGE[v][1]: p(f'{lab} out of range: {v} {s}')
        if f['score_rubric'] != c['score_0_10']:
            report['rubric_differs'].append(f"{uid}: {c['score_0_10']} -> {f['score_rubric']} ({f['score_rubric_basis']}) note={f['score_rubric_note'][:160]!r}")
            if not f['score_rubric_note'].strip(): report['needs_note'].append(uid)
        if f['score_adjusted'] != f['score_rubric']:
            report['adjusted_differs'].append(f"{uid}: {f['score_rubric']} -> {f['score_adjusted']} [{f['score_adjustment_rules']}]")
        if f['verdict_adjusted'] != c['verdict']:
            report['verdict_adjusted_differs'].append(f"{uid}: {c['verdict']} -> {f['verdict_adjusted']}")
    json.dump(out, open(sys.argv[2], 'w'), indent=1, ensure_ascii=False)
    json.dump(report, open(sys.argv[3], 'w'), indent=1, ensure_ascii=False)
    print(f"entries {len(out)}/{len(records)}; missing {len(report['missing'])}; extra {len(report['extra'])}; problems {len(report['problems'])}; "
          f"needs_note {len(report['needs_note'])}; rubric_differs {len(report['rubric_differs'])}; adjusted_differs {len(report['adjusted_differs'])}; "
          f"concerns {len(report['concerns'])}")


if __name__ == '__main__':
    main()
