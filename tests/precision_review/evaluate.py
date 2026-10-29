"""Evaluate output conversion on cached real EEG windows with current estimators.

Only raw per-window caches live in the local scratch directory; tracked receipts
contain aggregate errors and estimator identities, never patient matrices.
"""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pyspi.calculator import Calculator
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow._metrics import resolve_metrics
import yaml


def assess(values, dtype):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cast = values.astype(dtype)
        back = cast.astype(np.float64)
        finite = np.isfinite(values)
        comparable = finite & np.isfinite(back)
        error = np.abs(back[comparable] - values[comparable])
        nonzero = comparable & (values != 0)
        relative = np.abs((back[nonzero] - values[nonzero]) / values[nonzero])
        before_mean = np.nanmean(values, axis=2, dtype=np.float64)
        after_mean = np.nanmean(cast, axis=2, dtype=np.float64)
        good = np.isfinite(before_mean) & np.isfinite(after_mean)
        mean_error = np.abs(after_mean[good] - before_mean[good])
        scale = np.max(np.abs(values[finite])) if finite.any() else 0.
        return {"dtype": dtype, "finite_values": int(finite.sum()),
                "input_nan": int(np.isnan(values).sum()), "input_infinity": int(np.isinf(values).sum()),
                "finite_overflow": int((finite & np.isinf(back)).sum()),
                "underflow_to_zero": int((finite & (values != 0) & (back == 0)).sum()),
                "nonzero_sign_changes": int((comparable & (back != 0) & (values != 0) & (np.signbit(back) != np.signbit(values))).sum()),
                "maximum_absolute_value": float(scale),
                "maximum_absolute_error": float(error.max()) if len(error) else None,
                "maximum_relative_error": float(relative.max()) if len(relative) else None,
                "maximum_error_relative_to_metric_scale": float(error.max()/scale) if len(error) and scale else 0.,
                "maximum_time_mean_error": float(mean_error.max()) if len(mean_error) else None,
                "mean_finite_to_nonfinite": int((np.isfinite(before_mean) & ~np.isfinite(after_mean)).sum())}



def definition_check(values, spec):
    # Only established coefficient/angle bounds, not a new estimator review.
    bounded = {"SpearmanR", "KendallTau", "CrossCorrelation", "CoherenceMagnitude",
               "ImaginaryCoherence", "PhaseLockingValue", "PhaseLagIndex", "SignedPhaseLagIndex",
               "WeightedPhaseLagIndex", "SignedWeightedPhaseLagIndex", "PairwisePhaseConsistency",
               "DebiasedSquaredPhaseLagIndex", "DebiasedSquaredWeightedPhaseLagIndex", "PowerEnvelopeCorrelation"}
    limits = (-np.pi, np.pi) if spec.class_name == "CoherencePhase" else (-1., 1.) if spec.class_name in bounded else None
    if limits is None:
        return {"bounds_checked": False}
    finite = np.isfinite(values)
    outside = finite & ((values < limits[0]-1e-12) | (values > limits[1]+1e-12))
    return {"bounds_checked": True, "bounds": list(limits), "roundoff_allowance": 1e-12,
            "finite_input_violations": int(outside.sum()), "values_clipped": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('/tmp/pyspi-candidate-pilot'))
    parser.add_argument('--scratch', type=Path, default=ROOT/'.candidate-scratch/precision')
    parser.add_argument('--out', type=Path, default=ROOT/'tests/precision_review/results.json')
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    result = {"input_sha256": hashlib.sha256((args.inputs/'001-inputs.npz').read_bytes()).hexdigest(),
              "calculation_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted([*(ROOT/'pyspi/statistics').glob('*.py'), *(ROOT/'pyspi').glob('*.py')])},
              "candidate_window_axis": "Cached candidate arrays are window,row,column; moved to row,column,window before time means.",
              "design": "Three cached 118-channel, 1024-sample patient001 EEG windows, CAR, linear detrend; current reviewed and amplitude estimators. Existing three-patient/two-reference candidate outputs reused through documented ID mapping. Float64 time-mean accumulation before/after conversion. No cohort run.",
              "selections": {}, "candidates": {}, "limits": "Bounded sampled windows, not proof for every future patient. Estimator nonfinites are counted separately from conversion overflow. Raw outputs are local scratch only."}
    with np.load(args.inputs/'001-inputs.npz') as inputs:
        base=inputs['base'].copy()
        windows=inputs['windows'].tolist()
    base -= base.mean(axis=1, keepdims=True)
    for selection, normalized in [('reviewed', True), ('amplitude_sensitivity', False)]:
        config=ROOT/f'pyspi/workflow/{selection}.yaml'
        cache=args.scratch/f'{selection}.npz'
        reports=[]
        specs=resolve_metrics(config, sampling_rate_hz=2048)
        identifiers=[m.identifier for m in specs]
        arrays={}
        if cache.exists():
            with np.load(cache) as saved:
                arrays={k:saved[k] for k in saved.files if k in identifiers}
        missing=[m for m in specs if m.identifier not in arrays]
        if missing:
            # Resolve workflow-rate settings before constructing calculators.
            # Recalculate only missing identities, preserving completed evidence.
            document={}
            for m in missing:
                entry=document.setdefault(m.module,{}).setdefault(m.class_name,
                    {"dependencies":list(m.dependencies),"configs":[]})
                entry["configs"].append(m.parameters)
            normalized_config=args.scratch/f'{selection}-resolved.yaml'
            normalized_config.write_text(yaml.safe_dump(document,sort_keys=False))
            with redirect_stdout(sys.stderr):
                calc=Calculator(configfile=str(normalized_config), normalise=normalized, detrend=True)
                for w, values in enumerate(base):
                    calc.load_dataset(values.copy(), sampling_rate_hz=2048)
                    def collect(report, output):
                        print(f'{selection} window{windows[w]} {report.identifier} {report.status} {report.elapsed_seconds:.3f}s', file=sys.stderr, flush=True)
                        reports.append(report.as_dict())
                        if output is None:
                            output=np.full((base.shape[1],base.shape[1]),np.nan)
                        arrays.setdefault(report.identifier, np.full((*output.shape,len(base)),np.nan))[:,:,w]=output
                    calc.compute(on_metric_complete=collect)
            arrays={name:arrays[name] for name in identifiers}
            np.savez(cache, **arrays)
            reports_path=args.scratch/f'{selection}-reports.json'
            previous=json.loads(reports_path.read_text()) if reports_path.exists() else []
            replaced={m.identifier for m in missing}
            reports_path.write_text(json.dumps([r for r in previous if r['identifier'] in identifiers and r['identifier'] not in replaced]+reports,indent=2))
        result['selections'][selection]={"config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
            "windows":windows, "channels":base.shape[1], "metric_count":len(arrays),
            "source_definition_checks": {m.identifier:definition_check(arrays[m.identifier], m) for m in specs},
            "metrics":{name:{dt:assess(a,dt) for dt in ['float16','float32','float64']} for name,a in arrays.items()}}
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    names={}
    with (ROOT/'docs/metric_names.csv').open() as f:
        for row in csv.DictReader(f):
            if 'additional_candidate' in row['configs']:
                names[row['old_identifier']]=row['new_identifier']
    for patient in ('004','001','111'):
        for ref in ('car','laplacian'):
            with np.load(args.inputs/f'{patient}-{ref}-results.npz') as saved:
                result['candidates'][f'{patient}-{ref}']={names.get(name,name):{dt:assess(np.moveaxis(saved[name], 0, 2),dt) for dt in ['float16','float32','float64']} for name in saved.files}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    with numerical_threads(1):
        main()
