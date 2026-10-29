"""Cross-check and summarize completed full-patient profiles and payload audits.

Run from the repository root after audit_outputs.py has read every saved payload.
This checks receipt consistency and provenance; it does not replace that audit.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PATIENTS = ('001', '006', '111')
CONFIGS = {
    'car-normalized': 'reviewed.yaml',
    'car-amplitude': 'amplitude_sensitivity.yaml',
    'local-normalized': 'additional_candidates.yaml',
    'local-amplitude': 'log_scale.yaml',
    'shaft-normalized': 'shaft_candidates.yaml',
    'shaft-amplitude': 'log_scale.yaml',
}
COUNTS = dict(zip(CONFIGS, (201, 13, 15, 1, 8, 1)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    records = []
    sources = None
    inventories = None
    for patient in PATIENTS:
        profile_path = HERE / f'{patient}-full-profile.json'
        audit_path = HERE / f'{patient}-full-audit.json'
        profile = json.loads(profile_path.read_text())
        audit = json.loads(audit_path.read_text())
        assert audit['bounded_trial'] is False
        assert (ROOT / audit['profile']).resolve() == profile_path
        assert (ROOT / audit['configuration']).resolve() == Path(profile['configuration'])
        assert digest(Path(profile['configuration'])) == profile['config_sha256']
        for path, expected in profile['source_sha256'].items():
            assert digest(ROOT / path) == expected, path
        for group, filename in CONFIGS.items():
            assert digest(ROOT / 'pyspi/workflow' / filename) == profile['metric_config_sha256'][group]
        if sources is None:
            sources = profile['source_sha256']
        assert profile['source_sha256'] == sources
        runs = {r['group']: r for r in profile['runs']}
        audits = {r['group']: r for r in audit['runs']}
        assert len(runs) == len(profile['runs']) == len(audits) == len(audit['runs']) == 6
        assert runs.keys() == audits.keys() == CONFIGS.keys()
        current_inventory = {g: list(a['metrics']) for g, a in audits.items()}
        if inventories is None:
            inventories = current_inventory
        assert current_inventory == inventories
        groups = []
        for group, r in runs.items():
            a = audits[group]
            session = r['session']
            assert session['status'] == 'complete' and not session['resume']
            assert r['patient_id'] == a['patient'] == patient
            assert r['recording_id'] == a['recording'] == 'baseline'
            assert r['path'] == a['path']
            assert r['channels'] == a['channels']
            assert r['windows']['count'] == a['windows'] == profile['recording_windows']
            assert r['windows']['length_samples'] == r['windows']['step_samples'] == 1024
            assert len(a['metrics']) == r['metrics'] == COUNTS[group]
            assert not a['errors']
            assert a['reference_unavailable_contacts'] == 0
            for metric, counts in a['metrics'].items():
                assert counts['success_windows'] == a['windows'], (patient, group, metric)
                assert counts.get('failed_windows', 0) == 0
                assert counts.get('success_infinite_pairs', 0) == 0
                assert counts.get('logged_overflow_values', 0) == 0
            seconds = session['elapsed_seconds'] / a['windows']
            groups.append({
                'group': group, 'outputs': len(a['metrics']), 'windows': a['windows'],
                'successful_attempts': len(a['metrics']) * a['windows'],
                'elapsed_seconds': session['elapsed_seconds'], 'seconds_per_window': seconds,
                'seconds_600': seconds * 600, 'seconds_41400': seconds * 41400,
                'stored_bytes': a['stored_bytes'],
                'metrics_with_successful_nan_pairs': {
                    m: c for m, c in a['metrics'].items() if c.get('success_nan_pairs', 0)},
                'warning_events': sum(a['warnings'].values()),
            })
        assert sum(g['outputs'] for g in groups) == 239
        records.append({
            'patient': patient, 'channels': profile['runs'][0]['channels'],
            'windows': profile['recording_windows'], 'groups': groups,
            'successful_attempts': sum(g['successful_attempts'] for g in groups),
            'stored_bytes': sum(g['stored_bytes'] for g in groups),
            'elapsed_seconds': profile['elapsed_seconds'],
            'seconds_per_window': profile['end_to_end_seconds_per_recording_window'],
            'seconds_600': profile['end_to_end_seconds_600'],
            'seconds_41400': profile['end_to_end_seconds_41400'],
            'process_tree_memory': profile['process_tree_memory'],
            'preparation_cache': profile['preparation_cache'],
            'parent_jvm_started': profile['parent_jvm_started'],
            'profile_sha256': digest(profile_path), 'audit_sha256': digest(audit_path),
        })
    return {
        'checks': 'Three unbounded full-patient payload audits; six identical metric inventories; 239 outputs; complete successful attempts; no missing reference contacts, failed attempts, infinite outputs or precision overflows; current production sources and configs match recorded hashes. Successful numerical NaNs and warnings remain explicit for scientific review.',
        'limits': 'Payload checks are performed by audit_outputs.py. These totals summarize its receipts. Elapsed times include preparation, setup, computation and writing, but exclude the subsequent payload audit. Projections repeat each patient size separately, not the mixed cohort. Two workers only for 80/118-contact main CAR; 249-contact jobs are serial. Sampled memory and host load limit comparisons.',
        'source_sha256': sources, 'patients': records,
    }


if __name__ == '__main__':
    result = summarize()
    (HERE / 'full_patient_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified', len(result['patients']), 'patients;',
          sum(p['successful_attempts'] for p in result['patients']), 'successful attempts')
