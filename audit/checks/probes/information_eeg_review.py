"""Label-blind EEG feasibility/runtime checks for approved information candidates.

No signal samples or EZ labels are written. Reuse the already-reviewed EDF
preprocessing, retained-contact CAR and deterministic contact/window selection.
"""

import contextlib
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/pyspi-mpl')
os.environ.setdefault('_MNE_FAKE_HOME_DIR','/tmp/pyspi-mne')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from pyspi.data import Data
from pyspi.statistics.infotheory import TransferEntropy
from spectral_window_smoothing import read_eeg


def metrics():
    result={f'GC{k}':TransferEntropy(k_history=k,l_history=k) for k in (1,4,16,32)}
    result.update({f'SYM{spacing}':TransferEntropy(estimator='symbolic',pattern_length=3,
                     pattern_spacing=spacing,prediction_shift=spacing) for spacing in (4,16)})
    return result


def describe(value,diagnostic,seconds):
    mask=~np.eye(len(value),dtype=bool); values=value[mask]; finite=values[np.isfinite(values)]
    reasons,counts=np.unique(diagnostic['reason'][mask],return_counts=True)
    result={'seconds':seconds,'finite_fraction':float(np.mean(np.isfinite(values))),
        'infinite_fraction':float(np.mean(np.isinf(values))),
        'quantiles':np.quantile(finite,[0,.25,.5,.75,1]).tolist() if len(finite) else None,
        'reason_counts':{str(reason):int(count) for reason,count in zip(reasons,counts)}}
    for key in ('n_observations','patterns_observed','tied_pattern_fraction','target_transition_entropy'):
        if key in diagnostic:
            result[key]=diagnostic[key].tolist() if isinstance(diagnostic[key],np.ndarray) else diagnostic[key]
    if 'joint_states_observed' in diagnostic:
        result['joint_states_quantiles']=np.quantile(diagnostic['joint_states_observed'][mask],[0,.5,1]).tolist()
    return result


def calculate(x,*,announce=False,gaussian_only=False):
    with contextlib.redirect_stdout(io.StringIO()):
        data=Data(x.copy(),normalise=True,detrend=True)
    result={}
    for name,spi in metrics().items():
        if gaussian_only and name.startswith('SYM'):
            continue
        if announce:
            print('Representative full window start',name,flush=True)
        start=time.perf_counter(); value=spi.multivariate(data); seconds=time.perf_counter()-start
        result[name]=describe(value,spi.diagnostics(data),seconds)
        if announce:
            print('Representative full window',name,seconds,result[name]['finite_fraction'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recheck-gaussian',action='store_true',
        help='Recheck changed numerical guards only; preserve the initial pilot and symbolic results.')
    args=parser.parse_args()
    root=Path('/home/dan/data/IEEG/connectivity')
    destination=ROOT/('audit/briefing/information_eeg_conditioning_review.json' if args.recheck_gaussian
                     else 'audit/briefing/information_eeg_review.json')
    sources=['pyspi/_gaussian_history.py','pyspi/_symbolic.py','pyspi/statistics/infotheory.py']
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        'source':str(root),'patients':[],
        'protocol':'Full retained-contact CAR, notch/bandpass/resample to 2048Hz; detrend and ddof=1 normalize each 0.5s nonoverlapping window. Twelve label-blind contacts and twelve evenly spaced windows per patient; additionally one full 118-contact middle window in patient 001.',
        'limits':'Feasibility and elapsed-time pilot, not clinical validation. Timings share Data within a window; includes cache population, excludes preprocessing, diagnostics and I/O. No EZ labels used.'}
    if args.recheck_gaussian:
        report['reason']='Changed reciprocal-condition guard after long-history covariance failures; only Gaussian candidates repeated. Symbolic pilot was already complete and is unchanged.'
    for patient in ('001','041','096'):
        print('Preparing',patient,flush=True)
        raw,metadata=read_eeg(root/f'{patient}.zarr',contacts=118 if patient=='001' else 12)
        patient_report={'metadata':metadata,'windows':[]}
        if patient=='001':
            start=(raw.shape[1]//1024//2)*1024
            patient_report['full_window']={'window_index':int(start//1024),'n_channels':len(raw),
                                           'metrics':calculate(raw[:,start:start+1024],announce=True,
                                                               gaussian_only=args.recheck_gaussian)}
            selected=np.sort(np.random.default_rng(82500+int(patient)).choice(len(raw),12,replace=False))
            raw=raw[selected]
            patient_report['pilot_contact_indices']=selected.tolist()
        for start in np.linspace(0,raw.shape[1]//1024-1,12,dtype=int)*1024:
            values=calculate(raw[:,start:start+1024],gaussian_only=args.recheck_gaussian)
            patient_report['windows'].append({'window_index':int(start//1024),'metrics':values})
            print(patient,int(start//1024),{name:round(result['seconds'],4) for name,result in values.items()},flush=True)
        report['patients'].append(patient_report)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        del raw


if __name__=='__main__':
    main()
