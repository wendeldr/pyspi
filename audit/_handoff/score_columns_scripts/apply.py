"""Insert the score columns into audit/records/*.yaml after the `  score_0_10:` line.

Usage: python apply.py final_entries.json [--dry-run]
final_entries.json: {unit_id: {score_rubric, score_rubric_basis, score_rubric_note, documentation_status,
                    verdict_adjusted, score_adjusted, score_adjustment_rules, score_adjustment_reason,
                    identifier_exceptions}}
Targeted text edit: existing lines for these keys are replaced; everything else is left byte-identical,
which is checked by re-loading the YAML and comparing all other keys.
"""
import json, sys
from pathlib import Path
import yaml

REC = Path('/media/dan/Data/git/pyspi/audit/records')
KEYS = ["score_rubric", "score_rubric_basis", "score_rubric_note", "documentation_status", "verdict_adjusted",
        "score_adjusted", "score_adjustment_rules", "score_adjustment_reason", "identifier_exceptions"]


def strip_keys(lines):
    out, skip = [], False
    for ln in lines:
        if ln.startswith('  ') and not ln.startswith('   '):
            key = ln[2:].split(':', 1)[0]
            skip = key in KEYS
        elif not ln.startswith('    ') and ln.strip():
            skip = False
        if not skip:
            out.append(ln)
    return out


def block(entry):
    d = {k: entry[k] for k in KEYS}
    text = yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=160, default_flow_style=False)
    return ['  ' + ln if ln else ln for ln in text.splitlines()]


def main():
    entries = json.load(open(sys.argv[1]))
    dry = '--dry-run' in sys.argv
    changed = 0
    for uid, e in sorted(entries.items()):
        p = REC / f'{uid}.yaml'
        old_text = p.read_text()
        before = yaml.safe_load(old_text)
        lines = strip_keys(old_text.split('\n'))
        idx = [i for i, ln in enumerate(lines) if ln.startswith('  score_0_10:')]
        assert len(idx) == 1, (uid, idx)
        new_lines = lines[:idx[0] + 1] + block(e) + lines[idx[0] + 1:]
        new_text = '\n'.join(new_lines)
        after = yaml.safe_load(new_text)
        bc, ac = dict(before['correctness']), dict(after['correctness'])
        for k in KEYS:
            bc.pop(k, None)
            assert ac.pop(k) == e[k], (uid, k)
        assert bc == ac, uid
        assert {k: v for k, v in before.items() if k != 'correctness'} == {k: v for k, v in after.items() if k != 'correctness'}, uid
        assert list(after['correctness']).index('score_rubric') == list(after['correctness']).index('score_0_10') + 1
        if new_text != old_text:
            changed += 1
            if not dry:
                p.write_text(new_text)
    print(f'{"would change" if dry else "changed"} {changed} of {len(entries)} records')


if __name__ == '__main__':
    main()
