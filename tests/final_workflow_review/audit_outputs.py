"""Read every saved payload of a grouped profile and audit its output contracts.

Reads one metric/batch at a time. Clinical inputs and results are never modified.
The receipt distinguishes failed attempts, successful numerical NaNs, reference
unavailability, infinities and logged precision overflows.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import warnings

import numpy as np

from pyspi.storage import open_results
from pyspi.workflow.experiment import load_experiment, _reports
from pyspi.workflow._run import planned_windows
from pyspi.workflow._output import open_run, read_json, child_path


def audit(profile, config, bounded=False):
    report = read_json(profile)
    expected = {}
    for group, check in _reports(load_experiment(config)):
        for entry in check.recordings:
            expected[(entry['patient_id'], entry['recording_id'], group)] = (
                entry, check.configuration, [m.identifier for m in check.metrics])
    actual = {(r['patient_id'], r['recording_id'], r['group']): r for r in report['runs']}
    assert actual.keys() == expected.keys(), 'profile does not cover every configured group/recording'
    audited = []
    for key, record in actual.items():
        entry, cfg, metrics = expected[key]
        path = Path(record['path'])
        run = open_run(path)
        manifest = read_json(path/'run.json')
        assert manifest['state'] == 'complete'
        assert run.metrics == metrics
        assert manifest['channels'] == entry['output_channels']
        assert run.recording['reference'] == entry['reference']
        assert run.recording['preprocessing'] == {'detrend': cfg.detrend, 'zscore': cfg.zscore}
        assert manifest['reference_available'] == entry['reference_available']
        plan = planned_windows(entry, cfg).as_dict(include_ranges=False)
        if bounded:
            assert 0 < manifest['windows']['count'] <= plan['count']
            plan['count'] = manifest['windows']['count']
        else:
            assert manifest['contract']['max_windows'] is None
        assert manifest['windows'] == plan
        n = len(manifest['channels'])
        good = np.asarray(manifest['reference_available'], bool)
        eligible = good[:, None] & good[None, :] & ~np.eye(n, dtype=bool)
        totals = {m: Counter() for m in metrics}
        warning_counts, errors = Counter(), Counter()
        next_window = 0
        for batch in manifest['batches']:
            assert batch['state'] == 'complete' and batch['start'] == next_window
            next_window = batch['stop']
            root = child_path(path, batch['path'])
            events = read_json(root/'reports.json')
            indexed = {(e['window'], e['report']['identifier']): e for e in events}
            assert len(indexed) == len(events) == (batch['stop']-batch['start'])*len(metrics)
            store = open_results(root/'results.zarr')
            assert store.metrics == metrics
            assert store.recording['channels'] == manifest['channels']
            for metric in metrics:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    values, available = store.load(metric, return_availability=True)
                assert values.shape == (n, n, batch['stop']-batch['start'])
                assert values.dtype == np.float32
                assert available.all(), 'missing/corrupt stored payload'
                assert np.isnan(values[np.arange(n), np.arange(n)]).all()
                assert np.isnan(values[~good]).all() and np.isnan(values[:, ~good]).all()
                for local, window in enumerate(range(batch['start'], batch['stop'])):
                    event = indexed[(window, metric)]
                    status = event['report']['status']
                    assert status in ('success', 'failed')
                    totals[metric][status+'_windows'] += 1
                    data = values[:, :, local][eligible]
                    totals[metric][status+'_finite_pairs'] += int(np.isfinite(data).sum())
                    totals[metric][status+'_nan_pairs'] += int(np.isnan(data).sum())
                    totals[metric][status+'_infinite_pairs'] += int(np.isinf(data).sum())
                    if status == 'failed':
                        errors[json.dumps(event['report']['error'], sort_keys=True)] += 1
                        assert np.isnan(data).all()
                    for warning in event['report']['warnings']:
                        warning_counts[json.dumps(warning, sort_keys=True)] += 1
                    overflow = event.get('output_conversion')
                    if overflow:
                        assert overflow['kind'] == 'finite_overflow'
                        assert len(overflow['positions']) == len(overflow['original_values']) == overflow['count']
                        for position, original in zip(overflow['positions'], overflow['original_values']):
                            assert np.isfinite(original)
                            saved = values[position[0], position[1], local]
                            assert np.isinf(saved) and np.sign(saved) == np.sign(original)
                        totals[metric]['logged_overflow_values'] += overflow['count']
        assert next_window == manifest['windows']['count']
        audited.append({'patient': key[0], 'recording': key[1], 'group': key[2], 'path': str(path),
                        'channels': n, 'windows': next_window, 'metrics': dict(totals),
                        'warnings': dict(warning_counts), 'errors': dict(errors),
                        'reference_unavailable_contacts': int((~good).sum()),
                        'stored_bytes': sum(p.stat().st_size for p in path.rglob('*') if p.is_file())})
        print(key, next_window, len(metrics), 'audited', flush=True)
    return {'profile': str(profile), 'configuration': str(config), 'bounded_trial': bounded,
            'checks': 'All declared payloads read; full coverage, metadata contacts/reference/preprocessing/window plan, float32 dtype, storage availability, diagonal/unavailable-contact NaNs, attempt statuses and recoverable overflow values checked. Numerical NaNs and errors counted for follow-up, not assumed correct.',
            'runs': audited}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.profile, args.config, args.bounded)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
