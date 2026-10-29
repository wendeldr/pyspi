"""Compare every stored window/metric of two bounded grouped execution profiles."""
import argparse
import json
from pathlib import Path

import numpy as np
from pyspi.workflow import open_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    first, second = (json.loads(p.read_text()) for p in (args.first, args.second))
    left = {(r['patient_id'], r['recording_id'], r['group']): r['path'] for r in first['runs']}
    right = {(r['patient_id'], r['recording_id'], r['group']): r['path'] for r in second['runs']}
    assert left.keys() == right.keys()
    results = []
    for key in left:
        a, b = open_run(left[key]), open_run(right[key])
        assert a.metrics == b.metrics
        for field in ('channels', 'reference', 'preprocessing', 'windows'):
            assert a.recording[field] == b.recording[field]
        for metric in a.metrics:
            np.testing.assert_equal(a.statuses(metric), b.statuses(metric))
            x, available_x = a.load(metric, return_availability=True)
            y, available_y = b.load(metric, return_availability=True)
            np.testing.assert_equal(available_x, available_y)
            np.testing.assert_equal(np.isfinite(x), np.isfinite(y))
            assert x.dtype == y.dtype == np.float32
            np.testing.assert_allclose(x, y, rtol=5e-6, atol=5e-7)
            valid = np.isfinite(x) & np.isfinite(y)
            difference = abs(x[valid].astype(float)-y[valid].astype(float))
            results.append({'patient_id': key[0], 'group': key[2], 'metric': metric,
                'finite_values': int(valid.sum()), 'different_finite_values': int(np.count_nonzero(difference)),
                'max_absolute_difference': float(difference.max()) if len(difference) else None})
    args.out.write_text(json.dumps({'first': str(args.first), 'second': str(args.second), 'metrics': results,
        'criteria': 'Identical axes, reference/preprocessing, statuses, availability, finite masks and float32 dtypes; values agree within rtol=5e-6, atol=5e-7. All windows in both bounded profiles compared.'}, indent=2)+'\n')
    print({'outputs': len(results), 'different_outputs': sum(r['different_finite_values'] > 0 for r in results),
           'largest_difference': max((r['max_absolute_difference'] or 0) for r in results)})


if __name__ == '__main__':
    main()
