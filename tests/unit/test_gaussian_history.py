"""Independent OLS/determinant checks of shared Gaussian history calculations."""

import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pytest

from pyspi._moments import moment_workspace
from pyspi.data import Data
from pyspi.statistics.infotheory import (TransferEntropy, CausalEntropy, DirectedInfo,
                                        CrossmapEntropy, StochasticInteraction, MutualInfo)


spec = importlib.util.spec_from_file_location("audit_information_reference",
    Path(__file__).resolve().parents[2]/"audit/checks/refs/_kernels_infotheory.py")
REFERENCE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REFERENCE)


def data(x):
    return Data(np.array(x, copy=True), normalise=False, detrend=False)


def conditional_entropy(y, predictors):
    y = y-y.mean()
    predictors = predictors-predictors.mean(axis=0)
    scale = predictors.std(axis=0)
    predictors = np.divide(predictors, scale, out=np.zeros_like(predictors), where=scale > 0)
    residual = y-predictors @ np.linalg.lstsq(predictors, y, rcond=None)[0]
    return .5*np.log(2*np.pi*np.e*(residual @ residual)/(len(y)-1))


def model(name):
    if name == "gc1":
        return TransferEntropy(), lambda a,b: REFERENCE.gaussian_te_ols(a,b)
    if name == "gc4":
        return TransferEntropy(k_history=4,l_history=4), lambda a,b: REFERENCE.gaussian_te_ols(a,b,k=4,l=4)
    if name == "mixed":
        return TransferEntropy(k_history=3,k_tau=2,l_history=2,l_tau=3), lambda a,b: REFERENCE.gaussian_te_ols(a,b,k=3,k_tau=2,l=2,l_tau=3)
    if name in ("xme1", "xme10"):
        k = int(name[3:])
        return CrossmapEntropy(history_length=k), lambda a,b: conditional_entropy(b[k:], np.column_stack([a[k-lag:len(a)-lag] for lag in range(1,k+1)]))
    if name == "cce":
        return CausalEntropy(), REFERENCE.gaussian_cce
    if name == "di":
        return DirectedInfo(), lambda a,b: REFERENCE.gaussian_di(a,b,form="massey")
    if name == "si":
        return StochasticInteraction(), REFERENCE.gaussian_si
    if name == "si4":
        return StochasticInteraction(delay=4), lambda a,b: REFERENCE.gaussian_si(a,b,tau=4)
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["gc1","gc4","mixed","xme1","xme10","cce","di","si","si4"])
@pytest.mark.parametrize("units", ["normalized","volts"])
def test_history_calculations_match_independent_references(name, units):
    x = np.random.default_rng(1201).normal(size=(4, 240))
    x[1, 1:] += .7*x[0, :-1]
    x[2] += .4*x[0]
    if units == "normalized":
        x = (x-x.mean(axis=1,keepdims=True))/x.std(axis=1,ddof=1,keepdims=True)
    else:
        x = x*np.array([1e-6,3e-5,2e-6,8e-6])[:,None]+1e-4
    spi, reference = model(name)
    expected = np.full((len(x),len(x)),np.nan)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                expected[i,j] = reference(x[i],x[j])
    d=data(x); actual=spi.multivariate(d)
    np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=5e-12)
    np.testing.assert_allclose(spi.bivariate(d,i=0,j=1),expected[0,1],atol=5e-12)
    order=[2,0,3,1]
    np.testing.assert_allclose(spi.multivariate(data(x[order])),actual[np.ix_(order,order)],atol=5e-12)
    if name.startswith("si"):
        np.testing.assert_array_equal(actual,actual.T)


def test_granger_longer_history_finds_lagged_source_and_retains_half_log_units():
    rng=np.random.default_rng(1202)
    x=rng.normal(size=(2,4000))
    x[1,4:] = .8*x[0,:-4]+.5*x[1,4:]
    short=TransferEntropy().multivariate(data(x))
    longer=TransferEntropy(k_history=4,l_history=4).multivariate(data(x))
    assert short[0,1] < .002 and longer[1,0] < .003
    assert longer[0,1] == pytest.approx(.5*np.log(1+(.8/.5)**2),rel=.08)


@pytest.mark.parametrize('order',[16,32])
def test_candidate_histories_against_independent_ols(order):
    x=np.random.default_rng(1208).normal(size=(3,1024))
    x[1,order:]+=.6*x[0,:-order]
    actual=TransferEntropy(k_history=order,l_history=order).multivariate(data(x))
    for i,j in ((0,1),(1,0),(0,2)):
        expected=REFERENCE.gaussian_te_ols(x[i],x[j],k=order,l=order)
        assert actual[i,j]==pytest.approx(expected,abs=1e-12)


def test_filtered_history_uses_observation_space_for_ill_conditioned_covariance():
    from scipy.signal import butter, sosfiltfilt
    # Smooth observations can have moderate Cholesky diagonal entries despite
    # an unreliable covariance solve. A direct standardized OLS reference
    # checks both nested models, without regularizing the estimator.
    raw=np.random.default_rng(7731).normal(size=(6,1024))[3:]
    x=sosfiltfilt(butter(5,200,fs=2048,output='sos'),raw)
    k=8
    actual=TransferEntropy(k_history=k,l_history=k).multivariate(data(x))
    for i,j in ((0,1),(1,0),(0,2)):
        y=x[j,k:]-x[j,k:].mean()
        own=np.column_stack([x[j,k-lag:-lag] for lag in range(1,k+1)])
        src=np.column_stack([x[i,k-lag:-lag] for lag in range(1,k+1)])
        for predictors in (own,src):
            predictors-=predictors.mean(axis=0)
            predictors/=np.linalg.norm(predictors,axis=0)
        errors=[]
        for predictors in (own,np.column_stack([own,src])):
            residual=y-predictors@np.linalg.lstsq(predictors,y,rcond=None)[0]
            errors.append(residual@residual)
        expected=.5*np.log(errors[0]/errors[1])
        assert actual[i,j]==pytest.approx(expected,abs=1e-9)


def test_directed_information_excludes_target_only_autocorrelation_and_includes_present_source():
    rng=np.random.default_rng(1203)
    x=rng.normal(size=(2,6000))
    for t in range(1,x.shape[1]):
        x[1,t] += .9*x[1,t-1]
    independent=DirectedInfo().multivariate(data(x))
    assert independent[0,1] < .01
    x[1]=.8*x[0]+.2*rng.normal(size=x.shape[1])
    contemporaneous=DirectedInfo().multivariate(data(x))
    past_only=TransferEntropy().multivariate(data(x))
    assert contemporaneous[0,1] > 5 and past_only[0,1] < .005
    np.testing.assert_allclose(DirectedInfo(n=1).multivariate(data(x)),MutualInfo().multivariate(data(x)),atol=2e-13)


def test_xme_uses_the_last_requested_lag():
    rng=np.random.default_rng(1204)
    x=rng.normal(size=(2,1500)); x[1,10:]=x[0,:-10]+.1*x[1,10:]
    nine=CrossmapEntropy(history_length=9).multivariate(data(x))[0,1]
    ten=CrossmapEntropy(history_length=10).multivariate(data(x))[0,1]
    assert nine-ten > 2


def _select_reference(x,kmax,taumax):
    best=-np.inf; chosen=(1,1)
    for k in range(1,kmax+1):
        for tau in range(1,(1 if k==1 else taumax)+1):
            start=1+(k-1)*tau
            past=np.column_stack([x[start-lag:len(x)-lag] for lag in 1+tau*np.arange(k)])
            score=REFERENCE.gaussian_mi(past,x[start:])-k/(2*(len(x)-start))
            if score>best:
                best,chosen=score,(k,tau)
    return chosen


def test_automatic_embedding_matches_independent_search_and_ols():
    rng=np.random.default_rng(1205)
    x=rng.normal(size=(4,400))
    for t in range(4,x.shape[1]):
        x[0,t]+=.7*x[0,t-2]
        x[1,t]+=.4*x[0,t-1]+.4*x[1,t-3]
        x[2,t]+=.8*x[2,t-1]
    spi=TransferEntropy(auto_embed_method="MAX_CORR_AIS",k_search_max=4,tau_search_max=2)
    d=data(x); actual=spi.multivariate(d); diagnostic=spi.diagnostics(d)
    selections=[_select_reference(row,4,2) for row in x]
    np.testing.assert_array_equal(diagnostic["history_length"],[k for k,tau in selections])
    np.testing.assert_array_equal(diagnostic["history_spacing"],[tau for k,tau in selections])
    for i,(l,lt) in enumerate(selections):
        for j,(k,kt) in enumerate(selections):
            if i!=j:
                expected=REFERENCE.gaussian_te_ols(x[i],x[j],k=k,k_tau=kt,l=l,l_tau=lt)
                assert actual[i,j] == pytest.approx(expected,abs=1e-12)
    assert sum(key[0]=="gaussian_auto_embedding" for key in moment_workspace(d).arrays.entries)==1
    np.testing.assert_allclose(pickle.loads(pickle.dumps(spi)).multivariate(d),actual)


def test_shared_predictions_ownership_mutation_and_bounded_fallback():
    x=np.random.default_rng(1206).normal(size=(4,160)); d=data(x)
    TransferEntropy().multivariate(d)
    work=moment_workspace(d)
    own=work.arrays.entries[("gaussian_self_prediction",1,(1,))]
    StochasticInteraction().multivariate(d)
    assert work.arrays.entries[("gaussian_self_prediction",1,(1,))] is own
    CausalEntropy().multivariate(d)
    predictions={key:id(value) for key,value in work.arrays.entries.items() if key[0]=="gaussian_prediction"}
    spi=DirectedInfo(); before=spi.multivariate(d)
    assert predictions=={key:id(value) for key,value in work.arrays.entries.items() if key[0]=="gaussian_prediction"}
    copied=spi.multivariate(d); copied[0,1]=99
    assert spi.multivariate(d)[0,1]!=99
    diagnostic=spi.diagnostics(d); diagnostic["reason"][:]='changed'
    assert spi.diagnostics(d)["reason"][0,1]=='estimated'
    tiny=data(x); small=moment_workspace(tiny); small.arrays.max_bytes=2048; small.block_bytes=8192
    np.testing.assert_allclose(spi.multivariate(tiny),before,atol=2e-12)
    assert small.arrays.nbytes<=2048
    d.to_numpy()[1,:,0] += 2*x[0]
    assert not np.allclose(spi.multivariate(d),before,equal_nan=True)


@pytest.mark.parametrize("name",["gc1","gc4","xme10","cce","di","si"])
def test_history_degeneracy_and_input_units(name):
    rng=np.random.default_rng(1207)
    x=rng.normal(size=(4,100)); x[2]=x[0]; x[3]=0.
    spi,_=model(name); d=data(x)
    actual=spi.multivariate(d)
    assert np.isfinite(actual[0,1])
    if name.startswith('gc'):
        assert abs(actual[0,2])<1e-12
        assert abs(actual[0,3]) < 1e-14 and abs(actual[3,0]) < 1e-14
    if name=='di':
        assert np.isposinf(actual[0,2]) and actual[0,3]==0.
    if name in ('cce','si'):
        assert np.isnan(actual[0,2])
    factors=np.array([1e-200,1e200,2e-6,1.])
    scaled=spi.multivariate(data(x*factors[:,None]))
    expected=actual.copy()
    if name in ('xme10','cce'):
        expected += (5 if name=='cce' else 1)*np.log(factors)[None,:]
    np.testing.assert_allclose(scaled,expected,atol=5e-11)
    d.to_numpy()[1,50,0]=np.nan
    invalid=spi.multivariate(d)
    assert np.isnan(invalid[1]).all() and np.isnan(invalid[:,1]).all()
    assert np.isnan(spi.multivariate(data(x[:,:1]))).all()


@pytest.mark.parametrize("cls,settings",[(TransferEntropy,{'k_history':0}),
    (TransferEntropy,{'l_tau':1.5}),(TransferEntropy,{'auto_embed_method':'other','k_search_max':3,'tau_search_max':1}),
    (CausalEntropy,{'n':True}),(DirectedInfo,{'n':0}),(CrossmapEntropy,{'history_length':-1}),
    (StochasticInteraction,{'delay':0})])
def test_invalid_history_parameters_fail_before_calculation(cls,settings):
    with pytest.raises(ValueError):
        cls(**settings)
