"""Optional adversarial audit: run with python tests/review_mutations.py.

Runs unmodified controls, then ten temporary runtime mutations in isolated
pytest processes. Logs and JSON go to tests/storage/.artifacts/test_review/.
Production files are never edited. This script is not part of normal collection.
"""
import os
from dataclasses import replace

import numpy as np


def pytest_runtest_call(item):
    mutant = os.environ.get("PYSPI_REVIEW_MUTANT")
    if not mutant:
        return
    def patch(obj, name, value):
        previous = getattr(obj, name)
        setattr(obj, name, value)
        item.addfinalizer(lambda: setattr(obj, name, previous))
    if mutant == "zero_covariance":
        from pyspi.statistics.basic import Covariance
        previous = Covariance.multivariate
        patch(Covariance, "multivariate", lambda self, data: previous(self, data) * 0)
    elif mutant == "transpose_directed_edges":
        from pyspi.statistics.spectral import DirectedCoherence
        previous = DirectedCoherence.multivariate
        patch(DirectedCoherence, "multivariate", lambda self, data: previous(self, data).T)
    elif mutant == "skip_detrending":
        import pyspi.data as data
        patch(data, "detrend", lambda value, axis: value)
    elif mutant == "ignore_wavelet_band":
        from pyspi.statistics.wavelet import mne
        previous = mne._get_cache
        def cache(self, data):
            tensor, indices = previous(self, data)
            return tensor, np.arange(tensor.shape[2])
        patch(mne, "_get_cache", cache)
    elif mutant == "drop_phase_angle":
        import pyspi.statistics.wavelet as wavelet
        patch(wavelet, "modify_stats", lambda statfn, modifier: statfn)
    elif mutant == "absolute_after_plv_mean":
        from pyspi.statistics.spectral import PhaseLockingValue
        previous = PhaseLockingValue.__init__
        def init(self, **kwargs):
            previous(self, **kwargs)
            self._statfn = lambda value, **options: np.abs(np.nanmean(value, **options))
        patch(PhaseLockingValue, "__init__", init)
    elif mutant == "paired_packing_order_change":
        from pyspi.storage import _packing
        original_pack, original_unpack = _packing.pack, _packing.unpack
        def pack(*args, **kwargs):
            result = original_pack(*args, **kwargs)
            return replace(result, upper=result.upper[:, ::-1].copy()) if result.upper is not None else result
        def unpack(result, *args, **kwargs):
            if result.upper is not None:
                result = replace(result, upper=result.upper[:, ::-1].copy())
            return original_unpack(result, *args, **kwargs)
        patch(_packing, "pack", pack)
        patch(_packing, "unpack", unpack)
    elif mutant == "reuse_first_mean_block":
        import pyspi.analysis as analysis
        def mean(payload):
            out = np.empty(payload.shape[1], dtype=np.float64)
            for start in range(0, payload.shape[1], analysis.MEAN_BLOCK_COLUMNS):
                width = min(analysis.MEAN_BLOCK_COLUMNS, payload.shape[1] - start)
                out[start:start + width] = np.nanmean(payload[:, :width], axis=0, dtype=np.float64)
            return out
        patch(analysis, "_rows_nanmean", mean)
    elif mutant == "swallow_recorded_error_changes":
        from regression import harness
        def compute(entry, data, seed, seed_fn=harness.seed_all):
            spi = harness.build_spi(entry)
            seed_fn(seed)
            try:
                return harness.compute_spi(spi, data)
            except Exception:
                if entry["status"] != "error":
                    raise
                return np.full((data.n_processes, data.n_processes), np.nan)
        patch(harness, "compute_recorded", compute)
    elif mutant == "discard_complex_comparison_component":
        from regression import harness
        patch(harness, "_real_array", lambda value, identifier: np.asarray(value).astype(np.float64))
    else:
        raise ValueError(mutant)


def main():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    root = Path(__file__).resolve().parents[1]
    artifacts = root / 'tests/storage/.artifacts/test_review/mutations'
    artifacts.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root/'tests/regression/goldens/small/meta.json').read_text())
    directed = next(name for name in meta['spis'] if name.startswith('dcoh_'))
    cases = {
     'zero_covariance': ('tests/regression/test_spi_goldens.py::test_spi_matches_golden[cov_EmpiricalCovariance]', False),
     'transpose_directed_edges': (f'tests/regression/test_spi_goldens.py::test_spi_matches_golden[{directed}]', False),
     'skip_detrending': ('tests/unit/test_data.py::test_detrend_then_zscore_order', False),
     'ignore_wavelet_band': ('tests/unit/test_numeric_boundaries.py::test_wavelet_nondegenerate_aggregation_and_backend_cache[CoherenceMagnitude-mean]', False),
     'drop_phase_angle': ('tests/unit/test_numeric_boundaries.py::test_wavelet_nondegenerate_aggregation_and_backend_cache[CoherencePhase-mean]', False),
     'absolute_after_plv_mean': ('tests/unit/test_numeric_boundaries.py::test_multitaper_aggregation_uses_selected_frequencies_and_preserves_orientation[PhaseLockingValue-mean]', False),
     'paired_packing_order_change': ('tests/storage/test_packing.py::test_packed_layout_and_decoder_have_independent_fixed_oracles[sym-float64]', True),
     'reuse_first_mean_block': ('tests/storage/test_api.py::test_analysis_reduction_crosses_column_blocks_and_remainders[float64-sym]', True),
     'swallow_recorded_error_changes': ('tests/unit/test_regression_guards.py::test_recorded_error_cannot_hide_another_failure_or_a_success', False),
     'discard_complex_comparison_component': ('tests/unit/test_regression_guards.py::test_comparator_rejects_output_mutations[complex]', False),
    }
    rows=[]
    controls=[]
    for storage in (False, True):
        env = dict(os.environ, PYTHONPATH=str(root/'tests'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        env.pop('PYSPI_REVIEW_MUTANT', None)
        args = [sys.executable, '-m', 'pytest', '-q']
        if storage:
            args += ['-c', 'tests/storage/pytest.ini']
        else:
            args += ['--require-java']
        args += list(dict.fromkeys(node for node, mode in cases.values() if mode == storage))
        result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=120)
        (artifacts/('control-storage.log' if storage else 'control-calculations.log')).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('Unmodified control failed; mutation findings would be invalid. See '+str(artifacts))
        controls.append(dict(storage=storage, passed=True))

    for name, (node, storage) in cases.items():
        env = dict(os.environ, PYTHONPATH=str(root/'tests'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYSPI_REVIEW_MUTANT=name)
        report=artifacts/(name+'.xml')
        args=[sys.executable, '-m', 'pytest', '-q', '-p', 'review_mutations']
        if storage:
            args += ['-c', 'tests/storage/pytest.ini']
        args += [node, '--junitxml='+str(report)]
        result=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,timeout=120)
        (artifacts/(name+'.log')).write_text(result.stdout+result.stderr)
        failed = ET.parse(report).getroot().findall('.//testcase/failure') if report.exists() else []
        errors = ET.parse(report).getroot().findall('.//testcase/error') if report.exists() else []
        detected = result.returncode==1 and bool(failed) and not errors
        rows.append(dict(mutation=name,test=node,returncode=result.returncode,
                         assertion_failures=len(failed),setup_errors=len(errors),detected=detected))
        print(name, 'DETECTED' if detected else 'INVALID OR SURVIVED',flush=True)
    summary=dict(method='Temporary pytest runtime monkeypatches; production source files unchanged.',
                 controls=controls,cases=rows,all_detected=all(row['detected'] for row in rows))
    (artifacts.parent/'mutations.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if summary['all_detected'] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
