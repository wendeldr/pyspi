"""Compare selected spectral family cache policies on real cached EEG windows.

One thread; no clinical dataset writes. Includes Calculator construction and
calculation, excludes filtering, imports and output storage. Baseline changes
only the optional phase-count cache policy, not any estimator formula.
"""
import ast
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.signal import detrend
import yaml

from pyspi import _spectral
from pyspi.calculator import Calculator
from pyspi.workflow._resources import ProcessTreeMemory


def main():
    source = subprocess.check_output(
        ['git', 'show', 'd19f42d:pyspi/_spectral.py'], text=True)
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'SpectralConnectivity')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_phase_statistics')
    namespace = dict(vars(_spectral))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<baseline>', 'exec'), namespace)
    before = namespace['_phase_statistics']
    after = _spectral.SpectralConnectivity._phase_statistics
    selected = yaml.safe_load(Path('pyspi/workflow/reviewed.yaml').read_text())
    selected = {k: v for k, v in selected.items() if k.endswith('.spectral')}
    results = []
    receipt = {
        'baseline_commit': 'd19f42d', 'threads': 1,
        'current_spectral_sha256': hashlib.sha256(Path('pyspi/_spectral.py').read_bytes()).hexdigest(),
        'baseline_spectral_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'limits': 'Three real windows per patient, CAR then detrend/sample-zscore. Complete selected spectral family including Calculator overhead. Excludes continuous preparation, import/startup, storage and other metric families. Process memory includes input windows and comparison arrays. No cohort throughput claim.',
        'runs': results,
    }
    with tempfile.TemporaryDirectory(prefix='pyspi-phase-bench-') as directory:
        config = Path(directory) / 'spectral.yaml'
        config.write_text(yaml.safe_dump(selected, sort_keys=False))
        for patient in ('001', '111'):
            input_path = Path(f'/tmp/pyspi-candidate-pilot/{patient}-inputs.npz')
            base = np.load(input_path)['base']
            voltage = detrend(base-base.mean(axis=1, keepdims=True), axis=2)
            values = voltage/voltage.std(axis=2, ddof=1, keepdims=True)
            del base, voltage
            for window, x in enumerate(values):
                reference = None
                for mode, cache, implementation in (
                    ('before256', 256, before), ('after256', 256, after),
                    ('after1024', 1024, after),
                ):
                    gc.collect()
                    _spectral.SpectralConnectivity._phase_statistics = implementation
                    with ProcessTreeMemory() as monitor, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        start = time.perf_counter()
                        calc = Calculator(dataset=x.copy(), normalise=False,
                                          configfile=str(config), spectral_cache_mib=cache)
                        calc.compute()
                        elapsed = time.perf_counter()-start
                    reports = calc.metric_reports
                    assert all(r.status == 'success' for r in reports.values())
                    output = calc.table.to_numpy(copy=True)
                    difference = None
                    if reference is None:
                        reference = output
                    else:
                        np.testing.assert_allclose(output, reference, rtol=1e-12, atol=1e-12)
                        difference = float(np.nanmax(abs(output-reference)))
                    row = {'patient': patient, 'channels': len(x), 'window': window,
                           'mode': mode, 'seconds': elapsed, 'outputs': len(calc.spis),
                           'max_absolute_difference': difference, 'memory': monitor.as_dict(),
                           'metric_seconds': {k: r.elapsed_seconds for k, r in reports.items()}}
                    results.append(row)
                    Path('tests/final_workflow_review/spectral_cache_timings.json').write_text(json.dumps(receipt, indent=2)+'\n')
                    print(patient, window, mode, round(elapsed, 3), difference, flush=True)
                    del calc, output, reports
                del reference
    _spectral.SpectralConnectivity._phase_statistics = after


if __name__ == '__main__':
    main()
