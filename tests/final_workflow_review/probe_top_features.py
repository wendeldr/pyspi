"""Bounded source comparison and timing probes, never a cohort extraction.

Run from the repository root with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1. Reads existing input caches; writes only the chosen receipt.
Prototype timings are not production inclusion guarantees.
"""
import argparse,ast,contextlib,io,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import detrend,hilbert
from sklearn.covariance import LedoitWolf
from numba import njit
from pyspi.data import Data
from pyspi.statistics.distance import HistogramRelativeEntropy
from pyspi._envelope import EnvelopeWorkspace
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=Path('/run/media/dan/DansFlash/ram_needs_rma_git/top_features'))
parser.add_argument('--input',type=Path,default=Path('/tmp/pyspi-candidate-pilot/001-inputs.npz'))
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
root=args.source
if args.out.resolve().is_relative_to(root.resolve()) or args.out.resolve()==args.input.resolve():
    parser.error('output receipt must not overwrite external source/input files')
pilot=root/'pilot_relative_entropy.py'; extended=root/'further_feature_exploration/scripts/extract.py'
ns={'np':np,'njit':njit}
selected={'relative_entropy','_hist_kernel','_js_kernel','hist_measures','_orth_kernel','orth_envelope'}
for path in (pilot,extended):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in selected]
 for node in nodes:
  for dec in node.decorator_list:
   if isinstance(dec,ast.Call):
    for kw in dec.keywords:
     if kw.arg=='cache':kw.value=ast.Constant(False)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
cache=args.input
raw=np.load(cache)['base']; referenced=raw-raw.mean(axis=1,keepdims=True)
x=detrend(referenced,axis=2); z=x/x.std(axis=2,ddof=1,keepdims=True)
receipt={'shape':list(z[0].shape),'windows':3,'threads':1,'design':'Three cached real EEG windows from patient 001, CAR, detrend then sample zscore. Candidate kernel-only probes, not production or complete workflow timings. Imports, CAR and detrending/zscore excluded. Exact source functions extracted via AST without running source scripts.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (pilot,extended,cache)},'timings':{},'relative_entropy_comparison':[]}
def measure(key,fn):
 times=[]
 for k in range(3):
  start=time.perf_counter(); out=fn(k); times.append(time.perf_counter()-start)
 mid=float(np.median(times));receipt['timings'][key]={'seconds':times,'median_seconds':mid,'x600_seconds':mid*600,'x41400_hours':mid*41400/3600}
 return out
for k in range(3):
 with contextlib.redirect_stdout(io.StringIO()): d=Data(z[k],normalise=False)
 a=HistogramRelativeEntropy().multivariate(d); b=ns['relative_entropy'](z[k]);mask=~np.eye(len(a),dtype=bool)
 receipt['relative_entropy_comparison'].append(float(np.max(abs(a[mask]-b[mask]))))
start=time.perf_counter();ns['hist_measures'](z[0,:3]);ns['orth_envelope'](hilbert(z[0,:3]));receipt['jit_warmup_seconds']=time.perf_counter()-start
measure('top_histogram_RE_plus_JS',lambda k: ns['hist_measures'](z[k]))
def current_re(k):
 with contextlib.redirect_stdout(io.StringIO()):d=Data(z[k],normalise=False)
 return HistogramRelativeEntropy().multivariate(d)
measure('current_relative_entropy_including_Data',current_re)
probs=[ns['_hist_kernel'](v,10) for v in z]
measure('JS_increment_after_histograms',lambda k: ns['_js_kernel'](probs[k]))
def lag(k):
 a=z[k,:,:-20].copy();b=z[k,:,20:].copy();a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
 return a@b.T/(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:])
measure('fixed_lag_20_samples',lag)
def event(k):
 e=(abs(z[k])>3).astype(float);p=e.mean(1);return e@e.T/e.shape[1]-p[:,None]*p[None,:]
measure('event_excess',event)
def logratio(k):
 s=np.log(x[k].std(1));return s[None,:]-s[:,None]
measure('log_scale_ratio_from_detrended_voltage',logratio)
def partial(k):
 p=LedoitWolf().fit(z[k].T).precision_;s=np.sqrt(np.diag(p));return -p/(s[:,None]*s[None,:])
measure('LedoitWolf_partial_standalone',partial)
ps=[LedoitWolf().fit(v.T).precision_ for v in z]
measure('partial_increment_after_fit',lambda k: -ps[k]/np.sqrt(np.diag(ps[k])[:,None]*np.diag(ps[k])[None,:]))
old=measure('top_orthogonal_envelope_amplitude_only',lambda k: ns['orth_envelope'](hilbert(z[k])))
new=measure('current_orthogonal_envelope_also_computes_log',lambda k: EnvelopeWorkspace(z[k]).result(True,False,True))
mask=~np.eye(len(new),dtype=bool)
receipt['orthogonal_envelope_max_difference_last_window']=float(np.nanmax(abs(old[mask]-new[mask])))
receipt['orthogonal_envelope_limits']='Source kernel omits production scale/degeneracy safeguards and does not compute log variants. Potential optimization, not a validated replacement or like-for-like family speedup.'
receipt['relative_entropy_limits']='Same nonconstant finite off-diagonal estimator; production uses NaN diagonals and diagnosed NaNs for constants/nonfinite channels. Whole source filtering/reference/preprocessing pipeline is different.'
args.out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
