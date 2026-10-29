"""Production-family comparison against the pre-goal implementation.

Run from repo root with one BLAS/OMP thread. Uses three cached, real 118x1024
EEG windows; does not read or write the clinical dataset. Includes Data creation,
cache ownership checks and diagnostics in each standalone/complete workload.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import types

import numpy as np
from scipy.signal import detrend
from pyspi.data import Data
from pyspi._envelope import EnvelopeWorkspace
from pyspi.statistics.basic import (FixedLagCorrelation, ThresholdEventExcess,
                                    LogScaleRatio, PartialCorrelation, Precision)
from pyspi.statistics.distance import HistogramRelativeEntropy, HistogramJensenShannon
from pyspi.statistics.misc import PowerEnvelopeCorrelation

BASELINE = '99ed77d'
SPECS = [(False, False, False), (True, False, False), (False, True, False),
         (True, True, False), (True, False, True), (True, True, True)]


def baseline_module(path):
    text = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], text=True)
    module = types.ModuleType('baseline_' + Path(path).stem)
    exec(compile(text, f'{BASELINE}:{path}', 'exec'), module.__dict__)
    return module, hashlib.sha256(text.encode()).hexdigest()


def make_data(x):
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(x.copy(), normalise=False)


def run_metrics(x, metrics):
    d = make_data(x)
    results = []
    for metric in metrics:
        results.append(metric.multivariate(d))
        if hasattr(metric, 'diagnostics'):
            metric.diagnostics(d)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/tmp/pyspi-candidate-pilot/001-inputs.npz'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() == args.input.resolve():
        parser.error('output must not overwrite input')
    base = np.load(args.input)['base']
    voltage = detrend(base-base.mean(axis=1, keepdims=True), axis=2)
    normalized = voltage/voltage.std(axis=2, ddof=1, keepdims=True)
    old_envelope, eh = baseline_module('pyspi/_envelope.py')
    old_histogram, hh = baseline_module('pyspi/_histogram.py')
    receipt = {'baseline_commit': BASELINE, 'shape': list(normalized.shape), 'threads': 1,
               'limits': 'Three real CAR windows from patient 001. Preparation, imports, JIT loading/compilation and output excluded. Includes Data construction and diagnostics. No throughput or whole-process memory claim; cohort totals are sequential projections without startup or contention.',
               'source_sha256': {str(args.input): hashlib.sha256(args.input.read_bytes()).hexdigest()},
               'baseline_sha256': {'envelope': eh, 'histogram': hh}, 'timings': {}, 'differences': {}}
    for path in ('pyspi/_envelope.py', 'pyspi/_envelope_kernel.py', 'pyspi/_histogram.py',
                 'pyspi/_lightweight.py', 'pyspi/statistics/basic.py', 'pyspi/statistics/distance.py',
                 'tests/final_workflow_review/benchmark_families.py'):
        receipt['source_sha256'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metrics = [FixedLagCorrelation(), ThresholdEventExcess(), PartialCorrelation(),
               HistogramRelativeEntropy(), HistogramJensenShannon()]
    envelope_metrics = [PowerEnvelopeCorrelation(orth=o, log=l, absolute=a) for o, l, a in SPECS]
    start = time.perf_counter()
    run_metrics(normalized[0, :3], metrics + envelope_metrics)
    receipt['numba_load_or_compile_plus_small_warmup_seconds'] = time.perf_counter()-start

    def measure(key, fn):
        times, results = [], []
        for k in range(3):
            start = time.perf_counter()
            results.append(fn(k))
            times.append(time.perf_counter()-start)
        median = float(np.median(times))
        receipt['timings'][key] = {'seconds': times, 'median_seconds': median,
                                   'x600_seconds': 600*median, 'x41400_hours': 41400*median/3600}
        print(f'{key}: {median:.6f}s', flush=True)
        return results

    def old_pec(k, specs):
        work = old_envelope.envelope_workspace(make_data(normalized[k]))
        results = []
        for orth, log, absolute in specs:
            results.append(work.result(orth, log, absolute))
            work.diagnostics(orth, log)
        return results

    for name, specs in [('amplitude_orth', [(True, False, True)]), ('complete_six', SPECS)]:
        old = measure(f'envelope_before_{name}', lambda k: old_pec(k, specs))
        new = measure(f'envelope_after_{name}', lambda k: run_metrics(normalized[k], [
            PowerEnvelopeCorrelation(orth=o, log=l, absolute=a) for o, l, a in specs]))
        differences = []
        for before, after in zip(old, new):
            for a, b in zip(before, after):
                np.testing.assert_allclose(a, b, atol=2e-11, rtol=1e-9)
                differences.append(float(np.nanmax(abs(a-b))))
        receipt['differences'][name] = max(differences)

    old = measure('histogram_before_RE', lambda k: old_histogram.histogram_relative_entropy(make_data(normalized[k]), 10, 1e-10)[0])
    new = measure('histogram_after_RE', lambda k: run_metrics(normalized[k], [HistogramRelativeEntropy()])[0])
    for a, b in zip(old, new):
        np.testing.assert_allclose(a, b, atol=2e-13)
    receipt['differences']['histogram_RE'] = max(float(np.nanmax(abs(a-b))) for a, b in zip(old, new))
    measure('histogram_after_RE_and_JS', lambda k: run_metrics(normalized[k], [HistogramRelativeEntropy(), HistogramJensenShannon()]))
    for metric in metrics:
        measure(metric.identifier, lambda k, metric=metric: run_metrics(normalized[k], [metric]))
    measure('log_scale_ratio_unnormalized', lambda k: run_metrics(voltage[k], [LogScaleRatio()]))
    measure('complete_five_normalized_candidates', lambda k: run_metrics(normalized[k], metrics))
    prepared = [make_data(x) for x in normalized]
    for d in prepared:
        Precision(estimator='LedoitWolf').multivariate(d)
        HistogramRelativeEntropy().multivariate(d)
    measure('increment_partial_and_JS_after_precision_and_RE', lambda k: (
        PartialCorrelation().multivariate(prepared[k]), PartialCorrelation().diagnostics(prepared[k]),
        HistogramJensenShannon().multivariate(prepared[k]), HistogramJensenShannon().diagnostics(prepared[k])))
    # Diagnostics must remain identical, including near-degenerate cases.
    rng = np.random.default_rng(101)
    statuses = []
    for mode in ('ordinary', 'proportional', 'almost_proportional', 'nonfinite', 'constant', 'zero'):
        x = rng.normal(size=(4, 129))
        if mode == 'proportional': x[1] = -3*x[0]
        if mode == 'almost_proportional': x[1] = x[0]+1e-8*x[1]
        if mode == 'nonfinite': x[1, 0] = np.nan
        if mode == 'constant': x[1] = 5.
        if mode == 'zero': x[1] = 0.
        a, b = old_envelope.EnvelopeWorkspace(x), EnvelopeWorkspace(x)
        for orth, log, absolute in SPECS:
            np.testing.assert_allclose(a.result(orth, log, absolute), b.result(orth, log, absolute), atol=2e-11, rtol=1e-9)
            for key, value in a.diagnostics(orth, log).items():
                np.testing.assert_equal(value, b.diagnostics(orth, log)[key])
        statuses.append(mode)
    receipt['matching_diagnostics_cases'] = statuses
    args.out.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
