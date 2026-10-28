"""Checkpointed kernel-family benchmark; historical Java vs native shared work.

Baseline is the committed pre-refactor wrapper at 7578137. No full-size Java
repeat: compare at eight channels, measure native at 118, and distinguish them.
Raw EEG windows remain in /tmp; receipts contain hashes and timing only.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jpype
import numpy as np
from scipy.signal import lfilter
from pyspi.data import Data
from pyspi.statistics import infotheory as native
from pyspi._moments import moment_workspace

CASES=[('JointEntropy',{},'je'),('ConditionalEntropy',{},'ce'),
       ('MutualInfo',{'kernel_width':.25},'mi'),('TimeLaggedMutualInfo',{'kernel_width':.25},'tlmi'),
       ('TransferEntropy',{'kernel_width':.25},'te'),('StochasticInteraction',{},'si'),
       ('CrossmapEntropy',{'history_length':1},'xme1'),('CrossmapEntropy',{'history_length':10},'xme10'),
       ('CausalEntropy',{},'cce'),('DirectedInfo',{},'di')]


def baseline():
    source=subprocess.check_output(['git','show','7578137:pyspi/statistics/infotheory.py'],cwd=ROOT)
    module=types.ModuleType('pyspi.statistics._kernel_benchmark_baseline')
    sys.modules[module.__name__]=module
    exec(compile(source,'7578137:pyspi/statistics/infotheory.py','exec'),module.__dict__)
    return module


def family(module,x,record,path,key):
    with redirect_stdout(io.StringIO()):
        data=Data(x.copy(),dim_order='ps',normalise=False)
    matrices={};stages={}
    record[key]={'stages':stages}
    for cls,params,label in CASES:
        spi=getattr(module,cls)(estimator='kernel',**params)
        start=perf_counter();value=spi.multivariate(data);elapsed=perf_counter()-start
        matrices[label]=value
        stages[label]={'seconds':elapsed,'finite_off_diagonal':int(np.isfinite(value).sum())}
        record[key]['total_seconds']=sum(v['seconds'] for v in stages.values())
        record[key]['retained_array_mib']=moment_workspace(data).arrays.nbytes/2**20
        path.write_text(json.dumps(record,indent=2)+'\n')
        print(key,label,round(elapsed,5),flush=True)
    if module is native:
        record[key]['diagnostics']={}
        for cls,params,label in CASES:
            diag=getattr(module,cls)(estimator='kernel',**params).diagnostics(data)
            mask=~np.eye(len(x),dtype=bool)
            record[key]['diagnostics'][label]={
                'reasons':{str(v):int(n) for v,n in zip(*np.unique(diag['reason'][mask],return_counts=True))},
                'mean_joint_self_only_fraction':float(np.nanmean(diag['joint_self_only_fraction'][mask]))}
    return matrices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('synthetic_small','synthetic_large','eeg_small','eeg_large'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--eeg-window',type=Path,default=Path('/tmp/pyspi-kernel-eeg-window.npy'))
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    record={'case':args.case,'baseline_commit':'7578137','versions':{'numpy':np.__version__},
            'load_average_start':list(os.getloadavg()),'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in ('pyspi/statistics/infotheory.py','pyspi/_kernel_information.py','tests/benchmark_kernel.py')},
            'limits':'Single pass per path; calculation stages include actual shared increments. Imports, data preprocessing and I/O excluded. Large native runs have no newly measured matching Java baseline.'}
    if args.case.startswith('eeg'):
        if not args.eeg_window.exists():
            from pyspi.workflow import load_config
            from pyspi.workflow._prepare import prepare_recording
            cfg=load_config(ROOT/'tests/workflow/resource_profiles/configs/bounded_serial.yaml')
            start=perf_counter()
            with redirect_stdout(io.StringIO()):
                prepared=prepare_recording(cfg.recordings[0],cfg)
                d=Data(prepared.data[:,:1024].copy(),dim_order='ps',normalise=True,detrend=True)
            np.save(args.eeg_window,d.to_numpy(squeeze=True))
            metadata={'source':str(cfg.recordings[0].path),'channels':prepared.channel_names,'window':0,
                      'configuration_sha256':hashlib.sha256((ROOT/'tests/workflow/resource_profiles/configs/bounded_serial.yaml').read_bytes()).hexdigest(),
                      'preparation_seconds':perf_counter()-start,'settings':'CAR, 2048 Hz, 0.5 seconds, detrend, ddof=1 z-score'}
            args.eeg_window.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
            del prepared,d
        x=np.load(args.eeg_window)
        record['input_metadata']=json.loads(args.eeg_window.with_suffix('.json').read_text())
    else:
        x=lfilter([1.],[1.,-.7],np.random.default_rng(20261028).normal(size=(118,1024)),axis=1)
        x[1,1:]+=.5*x[0,:-1]
        x=(x-x.mean(1,keepdims=True))/x.std(1,ddof=1,keepdims=True)
    if args.case.endswith('small'): x=x[:8]
    record['shape']=list(x.shape);record['input_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
    if args.case.endswith('small'):
        from pyspi.utils import is_jpype_jvm_available
        start=perf_counter();assert is_jpype_jvm_available();record['jvm_startup_seconds']=perf_counter()-start
        old=baseline()
        with redirect_stdout(io.StringIO()):
            warm=Data(x[:2,:64],dim_order='ps',normalise=False)
        for cls,params,_ in CASES: getattr(old,cls)(estimator='kernel',**params).multivariate(warm)
        expected=family(old,x,record,args.out,'java')
    actual=family(native,x,record,args.out,'native')
    if args.case.endswith('small'):
        record['comparison']={}
        for key in actual:
            a,b=actual[key],expected[key];finite=np.isfinite(a)&np.isfinite(b)
            record['comparison'][key]={'allclose':bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True)),
                'finite_masks_equal':bool(np.array_equal(np.isfinite(a),np.isfinite(b))),
                'max_absolute_difference':float(np.max(abs(a[finite]-b[finite])))}
    elapsed=record['native']['total_seconds']
    record['native'].update(seconds_600=elapsed*600,seconds_41400=elapsed*41400)
    record.update(process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  jvm_started=jpype.isJVMStarted(),load_average_end=list(os.getloadavg()))
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    if 'comparison' in record: assert all(v['allclose'] and v['finite_masks_equal'] for v in record['comparison'].values())
    print('complete',args.case,elapsed,flush=True)


if __name__=='__main__': main()
