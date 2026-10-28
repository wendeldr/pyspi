"""Production information-family timings against the committed Java wrappers.

Use --native-only for additional candidates without repeating old fits. The
full comparison includes library-only measures; the retained scope follows the
original run and its approved replacement. Timings exclude preprocessing/I/O.
"""

import argparse
import contextlib
import datetime
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pickle
import subprocess
import sys
import time
import tracemalloc
import types

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
startup=time.perf_counter()
import numpy as np
from scipy.stats import zscore
from threadpoolctl import threadpool_info, threadpool_limits
from pyspi.data import Data
from pyspi.statistics import infotheory
IMPORT_SECONDS=time.perf_counter()-startup


FULL=['JE','CE','CCE','XME1','XME10','DI','SI','MI','TLMI','GCauto','GC1','SYM4','SYM16']
RETAINED=['MI','TLMI','GC1','SYM4','SYM16']
CHANGED={'DI','XME10','SYM4','SYM16'}


def new_data(x):
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(x.copy(),normalise=False,detrend=False)


def make(module,name,legacy=False):
    if name in ('JE','CE','CCE','DI','SI','MI','TLMI'):
        cls={'JE':'JointEntropy','CE':'ConditionalEntropy','CCE':'CausalEntropy','DI':'DirectedInfo',
             'SI':'StochasticInteraction','MI':'MutualInfo','TLMI':'TimeLaggedMutualInfo'}[name]
        return getattr(module,cls)()
    if name.startswith('XME'):
        return module.CrossmapEntropy(history_length=int(name[3:]))
    if name=='GCauto':
        return module.TransferEntropy(auto_embed_method='MAX_CORR_AIS',k_search_max=10,tau_search_max=2)
    if name.startswith('GC'):
        order=int(name[2:]); return module.TransferEntropy(k_history=order,l_history=order)
    spacing=int(name[3:])
    if legacy:
        return module.TransferEntropy(estimator='symbolic',k_history=1 if spacing==4 else 10,l_history=1)
    return module.TransferEntropy(estimator='symbolic',pattern_length=3,pattern_spacing=spacing,prediction_shift=spacing)


def run(module,names,data,*,legacy=False,announce=False,checkpoint=None,identity=None):
    outputs,seconds={},{}
    if checkpoint is not None and checkpoint.exists():
        with checkpoint.open('rb') as stream:
            saved=pickle.load(stream)
        if saved['identity']!=identity:
            raise ValueError(f'Incompatible benchmark checkpoint: {checkpoint}')
        data,outputs,seconds=saved['data'],saved['outputs'],saved['seconds']
        print('Resuming completed baseline stages:',list(seconds),flush=True)
    for name in names:
        if name in outputs:
            continue
        spi=make(module,name,legacy)
        if announce:
            print('baseline' if legacy else 'native','start',name,flush=True)
        start=time.perf_counter(); outputs[name]=spi.multivariate(data); seconds[name]=time.perf_counter()-start
        if announce:
            print('baseline' if legacy else 'native',name,seconds[name],flush=True)
        if checkpoint is not None:
            # Preserve completed expensive stages AND their shared Data cache.
            # The local synthetic checkpoint is written outside timed regions.
            temporary=checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(dict(identity=identity,data=data,outputs=outputs,seconds=seconds),stream)
            temporary.replace(checkpoint)
    return outputs,seconds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='092f6e7')
    parser.add_argument('--scope',choices=['full','retained','candidates'],default='full')
    parser.add_argument('--native-only',action='store_true')
    parser.add_argument('--warm-regression',action='store_true')
    parser.add_argument('--granger-orders',type=int,nargs='*',default=[],
                        help='Additional native orders in the retained workload only.')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--channels',type=int,default=118)
    parser.add_argument('--samples',type=int,default=1024)
    parser.add_argument('--out',type=Path,default=ROOT/'tests/information_family_results.json')
    args=parser.parse_args()
    if args.scope=='candidates' and not args.native_only:
        parser.error('Candidate histories have no selected historical counterpart; use --native-only.')
    if args.warm_regression and not args.native_only:
        parser.error('--warm-regression measures the native increment only.')
    if args.granger_orders and args.scope!='retained':
        parser.error('--granger-orders applies to the retained scope.')
    revision=subprocess.check_output(['git','rev-parse',args.baseline],cwd=ROOT,text=True).strip()
    methods={'native':infotheory}; java_seconds=0.
    if not args.native_only:
        from pyspi.utils import is_jpype_jvm_available
        start=time.perf_counter(); assert is_jpype_jvm_available(); java_seconds=time.perf_counter()-start
        module=types.ModuleType('information_baseline')
        source=subprocess.check_output(['git','show',f'{revision}:pyspi/statistics/infotheory.py'],cwd=ROOT,text=True)
        exec(compile(source,'<committed Java information baseline>','exec'),module.__dict__)
        methods={'baseline':module,**methods}
    names=FULL if args.scope=='full' else RETAINED
    if args.scope=='candidates':
        names=['MI','TLMI','GC1','GC4','GC16','GC32','SYM4','SYM16']
    workloads={method: (['JE','CE']+names if method=='baseline' and args.scope=='retained' else names)
               for method in methods}
    if args.granger_orders:
        workloads['native']=names[:3]+[f'GC{k}' for k in args.granger_orders]+names[3:]
    native_names=workloads['native'].copy()
    rng=np.random.default_rng(1501)
    x=rng.normal(size=(args.channels,args.samples))
    x[1,1:]+=.7*x[0,:-1]
    x=zscore(x,axis=1,ddof=1)
    timing={method:[] for method in methods}; outputs={}; warm=[]; workload_caches=[]
    load_start=os.getloadavg()
    with threadpool_limits(limits=1):
        for method,module in methods.items():
            run(module,workloads[method],new_data(x[:3,:100]),legacy=method=='baseline')
        for repeat in range(args.repeats):
            order=list(methods) if repeat%2==0 else list(reversed(methods))
            for method in order:
                d=new_data(x); gc.collect()
                if args.warm_regression:
                    from pyspi.statistics.misc import LinearModel
                    from pyspi.statistics.causal import RegressionErrorCausalInference
                    start=time.perf_counter()
                    for spi in (LinearModel('Lasso',alpha=.1),LinearModel('ElasticNet'),LinearModel('BayesianRidge'),RegressionErrorCausalInference()):
                        spi.multivariate(d)
                    warm.append(time.perf_counter()-start)
                checkpoint=None
                identity=dict(revision=revision,shape=list(x.shape),seed=1501,names=workloads[method],repeat=repeat)
                if method=='baseline':
                    checkpoint=Path('/tmp')/f'{args.out.stem}-baseline-stage-{repeat}.pickle'
                outputs[method],row=run(methods[method],workloads[method],d,legacy=method=='baseline',
                                       announce=True,checkpoint=checkpoint,identity=identity)
                timing[method].append(row)
                if method=='native':
                    cache=d._calculation_cache
                    workload_caches.append(dict(snapshot_bytes=cache.source.nbytes,
                        derived_cache_bytes=cache.families['moments'].arrays.nbytes))
                # A completed costly Java pass remains available if a later
                # comparison finds a native issue; fixing it must not require
                # repeating all historical fits merely to recover the oracle.
                if method=='baseline':
                    cached=Path('/tmp')/(args.out.stem+'-baseline')
                    np.savez_compressed(str(cached)+'.npz',**outputs[method])
                    Path(str(cached)+'.json').write_text(json.dumps({
                        'baseline_commit':revision,'shape':list(x.shape),'seed':1501,
                        'scope':args.scope,'seconds':timing[method]},indent=2)+'\n')
        differences={}
        if 'baseline' in outputs:
            for name in native_names:
                if name not in outputs['baseline']:
                    continue
                a,b=outputs['baseline'][name],outputs['native'][name]
                if name not in CHANGED:
                    np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-11)
                differences[name]=float(np.nanmax(abs(a-b)))
        d=new_data(x)
        # Memory is a separate cold-family pass; no baseline or startup memory
        # is charged to these Python/NumPy allocation figures.
        tracemalloc.start(); run(infotheory,native_names,d)
        retained,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        cache=d._calculation_cache
        median={method:float(np.median([sum(row.values()) for row in rows])) for method,rows in timing.items()}
        stages={method:{name:float(np.median([row[name] for row in rows])) for name in workloads[method]}
                for method,rows in timing.items()}
        pools=threadpool_info()
    sources=['pyspi/_moments.py','pyspi/_gaussian.py','pyspi/_gaussian_history.py','pyspi/_symbolic.py',
             'pyspi/_information_history.py','pyspi/statistics/infotheory.py']
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline_commit':revision,
        'shape':list(x.shape),'seed':1501,'scope':args.scope,'repeats':args.repeats,'python':platform.python_version(),
        'numpy':np.__version__,'threadpools':pools,'import_seconds':IMPORT_SECONDS,'java_startup_seconds':java_seconds,
        'load_start':load_start,'load_end':os.getloadavg(),'seconds':timing,'total_median_seconds':median,
        'incremental_median_seconds':stages,'regression_warmup_seconds':warm,'max_absolute_difference':differences,
        'actual_native_workload_retained_cache':workload_caches,
        'projected_seconds':{method:{str(n):total*n for n in (1,600,41400)} for method,total in median.items()},
        'native_memory':{'tracemalloc_retained_bytes':retained,'tracemalloc_peak_bytes':peak,
                         'snapshot_bytes':cache.source.nbytes,'derived_cache_bytes':cache.families['moments'].arrays.nbytes},
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        'definition_changes':{'DI':'Correct conditional-information sum including present source.',
            'XME10':'Ten source lags, not nine.','SYM4/SYM16':'Genuine m=3 ordinal TE at spacing/shift 4/4 and 16/16; old variants executed Gaussian k=1 and k=10.'},
        'protocol':{'timed':'Fresh Data/cache, metric matrix calls in listed order; preprocessing, imports, startup and I/O excluded.',
                    'sharing':'Stage times are actual increments after the preceding family stages; warm-regression is recorded separately.',
                    'memory':'Separate cold native pass. Tracemalloc is not process RSS or Java/native-library heap accounting.',
                    'limits':'Synthetic window on this computer. Serial extrapolations are not measured cohort throughput. A one-repeat baseline has no estimated timing dispersion.'}}
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
