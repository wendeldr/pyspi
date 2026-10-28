"""Independent box counts, alignments, limits and native integration."""
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from pyspi.data import Data
from pyspi._kernel_information import workspace, col, KernelWorkspace
from pyspi.statistics import infotheory as info

spec = importlib.util.spec_from_file_location("kernel_reference",
    Path(__file__).resolve().parents[2]/"audit/checks/refs/_kernels_infotheory.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def data(x):
    return Data(np.array(x,copy=True),dim_order="ps",normalise=False)


def h(x,w):
    return ref.entropy_box_kernel(x,w) if x.shape[1] else 0.


def conditional(y,x,w):
    return h(np.column_stack([y,x]),w)-h(x,w)


def expected(x,measure,w,i,j):
    t=x.shape[1]
    if measure=='je': return h(x[[i,j]].T,w)
    if measure=='ce': return h(x[[i,j]].T,w)-h(x[i,:,None],w)
    if measure=='mi': return ref.mi_box_kernel(x[i],x[j],w)
    if measure=='tlmi': return ref.mi_box_kernel(x[i,:-1],x[j,1:],w)
    if measure=='te': return ref.te_box_kernel(x[i],x[j],k=3,width=w)
    if measure=='xme':
        return conditional(x[j,3:,None],np.column_stack([x[i,3-l:t-l] for l in (1,2,3)]),w)
    if measure=='si':
        return sum(conditional(x[c,2:,None],x[c,:-2,None],w) for c in (i,j))-conditional(x[[i,j],2:].T,x[[i,j],:-2].T,w)
    causal=target=0.
    for m in range(1,4):
        yp=np.column_stack([x[j,m-1-l:t-l] for l in range(1,m)]) if m>1 else np.empty((t,0))
        xp=np.column_stack([x[i,m-1-l:t-l] for l in range(m)])
        y=x[j,m-1:,None]
        causal+=conditional(y,np.column_stack([yp,xp]),w)
        target+=conditional(y,yp,w)
    return causal if measure=='cce' else target-causal


CASES=[(info.JointEntropy,{},'je'),(info.ConditionalEntropy,{},'ce'),
       (info.MutualInfo,{},'mi'),(info.TimeLaggedMutualInfo,{},'tlmi'),
       (info.TransferEntropy,{'k_history':3},'te'),
       (info.CrossmapEntropy,{'history_length':3},'xme'),
       (info.CausalEntropy,{'n':3},'cce'),(info.DirectedInfo,{'n':3},'di'),
       (info.StochasticInteraction,{'delay':2},'si')]


@pytest.mark.parametrize('cls,parameters,measure',CASES)
@pytest.mark.parametrize('width',[.25,.5])
def test_all_measures_against_independent_counts(cls,parameters,measure,width):
    x=np.random.default_rng(211).normal(size=(3,79));x[1,1:]+=.7*x[0,:-1]
    d=data(x);spi=cls(estimator='kernel',kernel_width=width,**parameters)
    actual=spi.multivariate(d)
    for i in range(3):
        for j in range(3):
            if i==j: continue
            value=expected(x,measure,width,i,j)
            assert actual[i,j]==pytest.approx(value,abs=1e-12)
            assert spi.bivariate(d,i=i,j=j)==pytest.approx(value,abs=1e-12)
    p=[2,0,1]
    np.testing.assert_allclose(spi.multivariate(data(x[p])),actual[np.ix_(p,p)],atol=1e-12)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(spi)).multivariate(d),actual,atol=1e-12)


@pytest.mark.parametrize('cls,lag',[(info.MutualInfo,0),(info.TimeLaggedMutualInfo,1)])
@pytest.mark.parametrize('exclusion',[0,1,4,65,'AUTO'])
def test_mi_exclusions(cls,lag,exclusion):
    x=np.random.default_rng(212).normal(size=(2,67))
    spi=cls(estimator='kernel',kernel_width=.25,dyn_corr_excl=exclusion)
    w=ref.theiler_window_pyspi(x[0],x[1]) if exclusion=='AUTO' else exclusion
    n=67-lag
    if np.any((abs(np.arange(n)[:,None]-np.arange(n))>=w).sum(1)==0):
        assert np.isnan(spi.bivariate(data(x)))
        assert spi.diagnostics(data(x))['reason'][0,1]=='no_eligible_neighbors'
        return
    value=ref.mi_box_kernel(x[0,:67-lag],x[1,lag:],.25,theiler=w)
    assert spi.bivariate(data(x))==pytest.approx(value,abs=1e-12)


@pytest.mark.parametrize('exclusion',[0,1,5,60])
def test_te_exclusion_and_target_next_bandwidth(exclusion):
    x=np.random.default_rng(213).normal(size=(2,67));x[1,-1]=100
    k=3;n=64;w=.5
    past=np.column_stack([x[1,k-l:67-l] for l in range(1,k+1)])
    nxt=x[1,k:];src=x[0,k-1:-1]
    mp=np.all(abs(past[:,None,:]-past[None,:,:])<=w*past.std(0,ddof=1),axis=2)
    mn=abs(nxt[:,None]-nxt)<=w*past[:,0].std(ddof=1)
    ms=abs(src[:,None]-src)<=w*src.std(ddof=1)
    use=abs(np.arange(n)[:,None]-np.arange(n))>=exclusion
    cp=(mp&use).sum(1);cnp=(mp&mn&use).sum(1);cps=(mp&ms&use).sum(1);cj=(mp&mn&ms&use).sum(1)
    ok=cj>0
    value=np.log2((cj[ok]/cps[ok])/(cnp[ok]/cp[ok])).sum()/n
    spi=info.TransferEntropy(estimator='kernel',k_history=k,dyn_corr_excl=exclusion)
    if np.any(use.sum(1)==0):
        assert np.isnan(spi.bivariate(data(x)))
        assert spi.diagnostics(data(x))['reason'][0,1]=='no_eligible_neighbors'
        return
    assert spi.bivariate(data(x))==pytest.approx(value,abs=1e-12)


def test_packed_padding_boundaries_streaming_and_cache_limit(monkeypatch):
    x=np.random.default_rng(214).normal(size=(3,129));d=data(x)
    spi=info.TransferEntropy(estimator='kernel',k_history=3,dyn_corr_excl=4)
    full=spi.multivariate(d)
    streamed=data(x)
    work=workspace(streamed,.5)
    work.arrays.max_bytes=800
    monkeypatch.setattr(KernelWorkspace,'max_mask_bytes',1)
    monkeypatch.setattr(KernelWorkspace,'block_bytes',2048)
    np.testing.assert_allclose(spi.multivariate(streamed),full,atol=1e-12)
    assert work.arrays.nbytes<=800
    tie=np.tile([0.,1.,2.,3.],17)[:65]
    work=workspace(data(np.array([tie,tie+1])),1/tie.std(ddof=1))
    counts=work.counts(((col(0),),),0,len(tie))[0]
    np.testing.assert_array_equal(counts,(abs(tie[:,None]-tie)<=1).sum(1))


def test_invalid_pairs_diagnostics_mutation_and_ownership():
    x=np.random.default_rng(215).normal(size=(4,80));x[2]=1
    d=data(x);d.to_numpy()[3,5,0]=np.nan
    spi=info.MutualInfo(estimator='kernel')
    actual=spi.multivariate(d);diag=spi.diagnostics(d)
    assert np.isfinite(actual[0,1]) and np.isnan(actual[0,2:]).all()
    assert diag['reason'][0,2]=='zero_bandwidth' and diag['reason'][0,3]=='nonfinite_input'
    actual[0,1]=999;diag['reason'][:]='changed'
    assert spi.multivariate(d)[0,1]!=999 and spi.diagnostics(d)['reason'][0,1]=='estimated'
    before=spi.multivariate(d)[0,1]
    d.to_numpy()[1,:,0]=x[0]**2
    assert spi.multivariate(d)[0,1]!=before
    fresh=data(x)
    fresh.to_numpy()[:]=d.to_numpy()
    np.testing.assert_allclose(spi.multivariate(d),spi.multivariate(fresh),equal_nan=True)
    empty=info.MutualInfo(estimator='kernel',dyn_corr_excl=80)
    assert empty.diagnostics(data(x))['reason'][0,1]=='no_eligible_neighbors'
    assert info.TransferEntropy(estimator='kernel',k_history=80).diagnostics(data(x))['reason'][0,1]=='insufficient_observations'


def test_ce_di_reuse_joint_and_causal_matrices(monkeypatch):
    d=data(np.random.default_rng(216).normal(size=(3,90)))
    info.JointEntropy(estimator='kernel').multivariate(d)
    info.CausalEntropy(estimator='kernel',n=3).multivariate(d)
    work=workspace(d,.5);original=work.pair
    def forbidden(*args,**kwargs): raise AssertionError('recomputed a cached pair')
    monkeypatch.setattr(work,'pair',forbidden)
    assert np.isfinite(info.ConditionalEntropy(estimator='kernel').multivariate(d)[0,1])
    assert np.isfinite(info.DirectedInfo(estimator='kernel',n=3).multivariate(d)[0,1])
    monkeypatch.setattr(work,'pair',original)


def test_non_linear_and_lagged_models_and_sparse_neighborhood_diagnostic():
    x=np.random.default_rng(217).normal(size=(3,512));x[1]=x[0]**2+.05*x[1]
    spi=info.MutualInfo(estimator='kernel',kernel_width=.25)
    result=spi.multivariate(data(x));assert result[0,1]>result[0,2]+.5
    x[1,1:]=x[0,:-1]+.1*x[2,1:]
    te=info.TransferEntropy(estimator='kernel',kernel_width=.25).multivariate(data(x))
    assert te[0,1]>te[1,0]+.5
    d=info.CrossmapEntropy(estimator='kernel',history_length=10).diagnostics(data(x[:,:90]))
    assert d['joint_self_only_fraction'][0,2]>.9


@pytest.mark.parametrize('params',[{'kernel_width':0},{'kernel_width':float('nan')},
                                 {'kernel_width':True},{'dyn_corr_excl':-1},{'dyn_corr_excl':.5}])
def test_invalid_kernel_settings(params):
    with pytest.raises(ValueError): info.MutualInfo(estimator='kernel',**params)


def test_ignored_kernel_settings_are_rejected():
    with pytest.raises(ValueError,match='automatic embedding'):
        info.TransferEntropy(estimator='kernel',auto_embed_method='MAX_CORR_AIS',k_search_max=3)
    with pytest.raises(ValueError,match='single source lag'):
        info.TransferEntropy(estimator='kernel',l_history=2)
    with pytest.raises(ValueError,match='Temporal exclusion'):
        info.JointEntropy(estimator='kernel',dyn_corr_excl=2)


def test_extreme_units_do_not_change_mi():
    x=np.random.default_rng(218).normal(size=(3,81));spi=info.MutualInfo(estimator='kernel')
    baseline=spi.multivariate(data(x))
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(spi.multivariate(data(x*scale)),baseline,atol=1e-12)


def test_all_kernel_calculations_and_dependency_resolution_without_jvm(tmp_path):
    script='''
import jpype,numpy as np,yaml,sys
from pathlib import Path
from pyspi.data import Data
from pyspi.calculator import Calculator
from pyspi.workflow._metrics import resolve_metrics
from pyspi.statistics import infotheory as info
names=('JointEntropy','ConditionalEntropy','MutualInfo','TimeLaggedMutualInfo','TransferEntropy',
       'CrossmapEntropy','CausalEntropy','DirectedInfo','StochasticInteraction')
path=Path(sys.argv[1])
path.write_text(yaml.safe_dump({'.statistics.infotheory':{name:{'dependencies':['java'],
    'configs':[{'estimator':'kernel'}]} for name in names}}))
assert not jpype.isJVMStarted()
metrics=resolve_metrics(path)
assert len(metrics)==9 and all(not m.dependencies for m in metrics)
calc=Calculator(configfile=path,dataset=np.random.default_rng(219).normal(size=(3,67)))
calc.compute()
assert np.isfinite(calc.table.to_numpy()).sum()==9*6
assert not jpype.isJVMStarted()
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'kernel.yaml')],check=True,capture_output=True,text=True)


def test_unavailable_warning_can_be_recorded_by_workflow():
    from pyspi.calculation import _capture_metric_warnings
    spi=info.MutualInfo(estimator='kernel',dyn_corr_excl=30)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        with _capture_metric_warnings() as captured:
            spi.multivariate(data(np.random.default_rng(220).normal(size=(3,40))))
        assert any('no_eligible_neighbors' in w.message for w in captured)


def test_entropy_preserves_extreme_amplitude_units():
    x=np.random.default_rng(221).normal(size=(2,73))
    for cls,dimension in ((info.JointEntropy,2),(info.ConditionalEntropy,1)):
        spi=cls(estimator='kernel')
        baseline=spi.bivariate(data(x))
        for scale in (1e-200,1e200):
            assert spi.bivariate(data(x*scale)) == pytest.approx(
                baseline+dimension*np.log2(scale),abs=1e-12)
