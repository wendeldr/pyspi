"""Descriptive RE/JS redundancy across three cached patients/references.

Reads authoritative contact metadata only; does not modify source datasets or
re-extract EEG. Three sparse cached windows are a bounded pilot, not evidence
that every recording or temporal summary behaves identically.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import detrend
from scipy.stats import spearmanr, pearsonr
from seeg_dataset import patient_contacts

from pyspi.data import Data
from pyspi.statistics.distance import HistogramRelativeEntropy, HistogramJensenShannon
from pyspi.workflow._adaptive_reference import resolve_reference
from pyspi.workflow._reference import apply_shaft_reference, reference_plan


def compare(a, b):
    i, j = np.triu_indices(len(a), 1)
    x, y = a[i, j], b[i, j]
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    prediction = np.column_stack([x, np.ones(len(x))]) @ np.linalg.lstsq(
        np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    return {'pairs': len(x), 'spearman': float(spearmanr(x, y).statistic),
            'pearson': float(pearsonr(x, y).statistic),
            'linear_residual_fraction_of_js_sd': float(np.std(y-prediction)/np.std(y))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('/tmp/pyspi-candidate-pilot'))
    parser.add_argument('--dataset', type=Path, default=Path('/home/dan/data/IEEG/connectivity'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('receipt must be outside the clinical dataset')
    receipt = {'limits': 'Descriptive 3-patient x 3-reference x 3-sparse-window pilot, not predictive validation or exhaustive redundancy evidence. Equal-weight temporal means over cached windows. Same approved filtering and detrend-then-zscore preprocessing; custom local differs from the top_features source.',
               'patients': {}, 'source_sha256': {}}
    for pid in ('001', '006', '111'):
        path = args.cache/f'{pid}-inputs.npz'
        receipt['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = np.load(path)
        names = saved['channel_names'].tolist()
        assert names == patient_contacts(args.dataset/f'{pid}.zarr', patient_id=pid)
        record = {'channels': len(names), 'window_indices': saved['windows'].tolist(), 'references': {}}
        for method in ('car', 'adaptive_local', 'shaft_average'):
            request = {'method': method}
            if method == 'shaft_average': request['shafts'] = 'metadata'
            cfg = SimpleNamespace(reference=request, dataset_directory=args.dataset)
            reference = resolve_reference(cfg, SimpleNamespace(patient_id=pid))
            plan = reference_plan(names, reference)
            re, js = [], []
            for raw in saved['base']:
                x = raw-raw.mean(axis=0) if method == 'car' else apply_shaft_reference(raw, names, reference, plan)
                x = detrend(x, axis=1)
                x /= x.std(axis=1, ddof=1, keepdims=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    d = Data(x, normalise=False)
                re.append(HistogramRelativeEntropy().multivariate(d))
                js.append(HistogramJensenShannon().multivariate(d))
            record['references'][method] = {
                'reference_definition_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
                'per_window': [compare(a, b) for a, b in zip(re, js)],
                'temporal_mean': compare(np.mean(re, axis=0), np.mean(js, axis=0))}
        receipt['patients'][pid] = record
    for path in ('pyspi/_histogram.py', 'pyspi/workflow/_adaptive_reference.py',
                 'tests/final_workflow_review/assess_histogram_redundancy.py'):
        receipt['source_sha256'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({pid: {ref: value['temporal_mean'] for ref,value in patient['references'].items()}
                      for pid,patient in receipt['patients'].items()}, indent=2))


if __name__ == '__main__':
    main()
