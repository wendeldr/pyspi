"""Check downstream graph transformations against independent saved-array formulas."""
import argparse
import json
from pathlib import Path

import numpy as np

from pyspi.workflow import open_run
from pyspi.workflow.derived import iter_derived_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for entry in json.loads(args.profile.read_text())['runs']:
        if not entry['group'].endswith('-normalized'):
            continue
        run = open_run(entry['path'])
        count = min(4, run.recording['windows']['count'])
        assert count >= 4
        cases = [('pdist_cosine', transform, change)
                 for transform in ('pearson', 'absolute_pearson')
                 for change in (False, True)]
        relative = [m for m in run.metrics if m.startswith('histre_')]
        assert len(relative) == 1, relative
        cases.extend([(relative[0], 'identity', True), ('pec_orth_abs', 'identity', True)])
        for metric, transform, change in cases:
            raw, available = run.load(metric, np.arange(count), return_availability=True)
            for selection in (np.arange(count), np.array([0, 2, 3])):
                wanted = raw[:, :, selection].astype(np.float64)
                good = available[:, :, selection] & np.isfinite(wanted)
                good &= (run.statuses(metric)[selection] == 'success')[None, None, :]
                if transform != 'identity':
                    wanted = 1-wanted
                    if transform == 'absolute_pearson':
                        wanted = np.abs(wanted)
                if change:
                    expected = np.full_like(wanted, np.nan)
                    valid = np.zeros_like(good)
                    for t in range(1, len(selection)):
                        if selection[t]-selection[t-1] == 1:
                            valid[:, :, t] = good[:, :, t] & good[:, :, t-1]
                            expected[:, :, t] = np.where(valid[:, :, t],
                                                        np.abs(wanted[:, :, t]-wanted[:, :, t-1]), np.nan)
                else:
                    expected, valid = np.where(good, wanted, np.nan), good
                pieces = list(iter_derived_windows(run, metric, transform=transform,
                              absolute_change=change, windows=selection, batch_windows=2))
                np.testing.assert_array_equal(np.concatenate([p.window_indices for p in pieces]), selection)
                np.testing.assert_array_equal(np.concatenate([p.values for p in pieces], axis=2), expected)
                np.testing.assert_array_equal(np.concatenate([p.valid for p in pieces], axis=2), valid)
                assert all(p.provenance['recording']['reference'] == run.recording['reference']
                           and p.provenance['temporal_summary'] is None for p in pieces)
                rows.append({'patient': entry['patient_id'], 'group': entry['group'], 'metric': metric,
                             'transform': transform, 'absolute_change': change,
                             'windows': selection.tolist(), 'valid_values': int(valid.sum())})
            np.testing.assert_array_equal(run.load(metric, np.arange(count)), raw)
    args.out.write_text(json.dumps({'profile': str(args.profile),
        'checks': 'Independent saved-array formulas match exactly across two-window batch boundaries and selection gaps; raw arrays unchanged; reference provenance retained. No EEG calculations or temporal averages performed.',
        'cases': rows}, indent=2)+'\n')
    print(len(rows), 'derived integration cases passed')


if __name__ == '__main__':
    main()
