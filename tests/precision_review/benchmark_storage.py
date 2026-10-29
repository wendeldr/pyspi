"""Matched fresh-process benchmark of the actual bounded workflow writer.

Uses local raw results from evaluate.py. Each child writes/reads 3 real windows,
including conversion diagnostics, checks exact converted bits, then removes only
its own temporary output. No estimator calculation is timed or repeated.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np


def trial(args):
    from pyspi.workflow._metrics import resolve_metrics
    from pyspi.workflow._output import Journal, RUN_SCHEMA, open_run
    from pyspi.workflow._run import _store_values, _commit_batch
    arrays={}
    with np.load(args.scratch/f'{args.selection}.npz') as source:
        arrays={key:source[key] for key in source.files}
    metrics=resolve_metrics(ROOT/f'pyspi/workflow/{args.selection}.yaml', sampling_rate_hz=2048)
    n,_,windows=next(iter(arrays.values())).shape
    dtypes={name:np.dtype(args.dtype).str for name in arrays}
    before_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    with tempfile.TemporaryDirectory(prefix='output-bench-',dir=args.scratch) as directory:
        path=Path(directory)
        batches=[{'start':i,'stop':min(i+args.batch,windows),'state':'pending'} for i in range(0,windows,args.batch)]
        manifest={'schema':RUN_SCHEMA,'state':'complete','channels':[f'C{i}' for i in range(n)],
                  'contract':{'metrics':[m.as_dict() for m in metrics]},'dtypes':dtypes,
                  'windows':{'count':windows},'batches':batches}
        session={'output_seconds':0.}
        started=perf_counter()
        for batch in batches:
            journal=Journal(path/'batches'/f"{batch['start']:08d}"/'attempts.jsonl')
            buffers={}
            try:
                for w in range(batch['start'],batch['stop']):
                    for m in metrics:
                        report={'identifier':m.identifier,'status':'success','elapsed_seconds':0.,'warnings':[],'error':None}
                        event=SimpleNamespace(identifier=m.identifier,status='success',as_dict=lambda report=report:report)
                        conversion=_store_values(buffers,event,arrays[m.identifier][:,:,w],np.arange(n),n,
                                                 batch['stop']-batch['start'],w-batch['start'],dtypes)
                        journal.record(w,event,output_conversion=conversion)
                _commit_batch(path,manifest,batch,buffers,journal,metrics,session)
            finally:
                journal.close()
        write_seconds=perf_counter()-started
        files=[p for p in path.rglob('*') if p.is_file()]
        size=sum(p.stat().st_size for p in files)
        log_size=sum(p.stat().st_size for p in files if p.name in ('reports.json','attempts.jsonl'))
        output_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        reader=open_run(path)
        read_seconds=0.
        for name,values in arrays.items():
            started=perf_counter();actual=reader.load(name);read_seconds+=perf_counter()-started
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                expected=values.astype(args.dtype)
            bits=f'u{actual.dtype.itemsize}'
            np.testing.assert_array_equal(actual.view(bits),expected.view(bits))
        return {'selection':args.selection,'dtype':args.dtype,'batch_windows':args.batch,'channels':n,
                'windows':windows,'metrics':len(metrics),'write_seconds':write_seconds,'read_seconds':read_seconds,
                'stored_bytes':size,'diagnostic_bytes':log_size,'files':len(files),
                'output_peak_rss_mib':output_peak,'rss_before_conversion_mib':before_rss,
                'whole_process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'exact_converted_roundtrip':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch',type=Path,default=ROOT/'.candidate-scratch/precision')
    p.add_argument('--out',type=Path,default=ROOT/'tests/precision_review/storage_timings.json')
    p.add_argument('--child',action='store_true')
    p.add_argument('--selection',default='reviewed');p.add_argument('--dtype',default='float32')
    p.add_argument('--batch',type=int,default=3)
    args=p.parse_args()
    if args.child:
        print(json.dumps(trial(args)));return
    result={'design':'Fresh process per trial; actual conversion, overflow logs, durable batch commits and shared reader. Three real 118-channel windows per selection. Two dtype-order reversals at batch3; one bounded batch1 comparison. Filesystem cache uncontrolled, no cohort projection. Peak RSS includes imports/source arrays and readback; output peak also reported separately.', 'trials':[]}
    for selection in ['reviewed','amplitude_sensitivity']:
        cases=[(3,dtype) for dtypes in [('float64','float32','float16'),('float16','float32','float64')] for dtype in dtypes]
        cases += [(1,'float64'),(1,'float32'),(1,'float16')]
        for batch,dtype in cases:
            command=[sys.executable,__file__,'--child','--scratch',str(args.scratch),'--selection',selection,'--dtype',dtype,'--batch',str(batch)]
            proc=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
            if proc.returncode:
                sys.stderr.write(proc.stderr)
                proc.check_returncode()
            row=json.loads(proc.stdout.strip().splitlines()[-1]);result['trials'].append(row)
            print(selection,dtype,batch,row['write_seconds'],row['stored_bytes'],flush=True)
            args.out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
