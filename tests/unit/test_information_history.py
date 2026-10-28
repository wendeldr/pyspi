"""Retained Java entropy estimators with corrected, shared history definitions."""

import jpype as jp
import numpy as np
import pytest

from pyspi._information_history import history
from pyspi._moments import moment_workspace
from pyspi.data import Data
from pyspi.statistics.infotheory import (CausalEntropy, DirectedInfo, CrossmapEntropy,
                                        JointEntropy, ConditionalEntropy, TransferEntropy,
                                        MutualInfo, StochasticInteraction)


pytestmark=pytest.mark.java


def data(x):
    return Data(np.array(x,copy=True),normalise=False,detrend=False)


def entropy(estimator, observations):
    if observations.shape[1]==0:
        return 0.
    package=jp.JPackage('infodynamics.measures.continuous')
    calc=(package.kernel.EntropyCalculatorMultiVariateKernel() if estimator=='kernel'
          else package.kozachenko.EntropyCalculatorMultiVariateKozachenko())
    calc.setProperty('NOISE_SEED','42')
    if estimator=='kernel':
        calc.setProperty('KERNEL_WIDTH','0.5')
    calc.initialise(observations.shape[1])
    calc.setObservations(jp.JArray(jp.JDouble,2)(observations))
    return float(calc.computeAverageLocalOfObservations())


def references(x,estimator,n):
    # Original JIDT embeddings independently establish sample and column order.
    utilities=jp.JPackage('infodynamics.utils').MatrixUtils
    xjava,yjava=[jp.JArray(jp.JDouble)(row) for row in x]
    causal,target=0.,0.
    for m in range(1,n+1):
        yp=np.asarray(utilities.makeDelayEmbeddingVector(yjava,m-1))[:-1]
        xp=np.asarray(utilities.makeDelayEmbeddingVector(xjava,m))
        yf=x[1,m-1:,None]
        cond=np.column_stack([yp,xp])
        causal+=entropy(estimator,np.column_stack([yf,cond]))-entropy(estimator,cond)
        target+=entropy(estimator,np.column_stack([yf,yp]))-entropy(estimator,yp)
    return causal,target-causal


@pytest.mark.parametrize('estimator',['kernel','kozachenko'])
@pytest.mark.parametrize('n',[1,3,5])
def test_corrected_causal_and_directed_entropy_against_java_embeddings(estimator,n):
    x=np.random.default_rng(1301).normal(size=(2,100)); x[1,1:]+=.5*x[0,:-1]
    expected_cce,expected_di=references(x,estimator,n)
    d=data(x)
    assert CausalEntropy(estimator=estimator,n=n).bivariate(d)==pytest.approx(expected_cce,abs=1e-12)
    assert DirectedInfo(estimator=estimator,n=n).bivariate(d)==pytest.approx(expected_di,abs=1e-12)


@pytest.mark.parametrize('estimator',['kernel','kozachenko'])
@pytest.mark.parametrize('k',[1,3,10])
def test_xme_exact_lag_count_for_each_retained_estimator(estimator,k):
    x=np.random.default_rng(1302).normal(size=(2,180)); x[1,k:]+=.7*x[0,:-k]
    source=np.column_stack([x[0,k-lag:len(x[0])-lag] for lag in range(1,k+1)])
    target=x[1,k:,None]
    expected=entropy(estimator,np.column_stack([source,target]))-entropy(estimator,source)
    assert CrossmapEntropy(estimator=estimator,history_length=k).bivariate(data(x))==pytest.approx(expected,abs=1e-12)


def test_histories_and_causal_results_shared_with_directed_information(monkeypatch):
    d=data(np.random.default_rng(1303).normal(size=(3,80)))
    work=moment_workspace(d)
    embedding=history(work,3,(0,1,2,3))
    np.testing.assert_array_equal(embedding[0,0],work.source[0,[3,2,1,0]])
    cce=CausalEntropy(estimator='kernel',n=3)
    cce.multivariate(d)
    calls=[]
    original=DirectedInfo._compute_conditional_entropy
    def counted(self,x,y):
        calls.append(y.shape[1])
        return original(self,x,y)
    monkeypatch.setattr(DirectedInfo,'_compute_conditional_entropy',counted)
    spi=DirectedInfo(estimator='kernel',n=3)
    before=spi.multivariate(d)
    assert calls==[0,1,2]*3  # One target-only sum per channel; CCE already cached.
    assert history(work,3,(0,1,2,3)) is embedding
    d.to_numpy()[1,:,0]+=2*d.to_numpy()[0,:,0]
    assert not np.allclose(spi.multivariate(d),before,equal_nan=True)


@pytest.mark.parametrize('cls',[JointEntropy,ConditionalEntropy])
def test_retained_entropy_caches_follow_data_mutation(cls):
    x=np.random.default_rng(1304).normal(size=(3,90)); d=data(x)
    spi=cls(estimator='kernel'); original=spi.multivariate(d)
    d.to_numpy()[1,:,0]+=x[0]
    updated=spi.multivariate(d)
    np.testing.assert_array_equal(updated,spi.multivariate(data(d.to_numpy(squeeze=True))))
    assert not np.allclose(updated,original,equal_nan=True)


def test_theiler_cache_reuses_channel_acfs_and_follows_mutation(monkeypatch):
    from pyspi import utils
    x=np.random.default_rng(1306).normal(size=(4,100)); d=data(x)
    calls=[]
    original=utils.acf
    def counted(row):
        calls.append(1)
        return original(row)
    monkeypatch.setattr(utils,'acf',counted)
    spi=MutualInfo(estimator='kraskov',dyn_corr_excl='AUTO')
    for i,j in ((0,1),(1,2),(2,0)):
        spi._set_theiler_window(d,i,j)
        expected=int(2*np.dot(original(x[i]),original(x[j])))
        assert int(str(spi._calc.getProperty('DYN_CORR_EXCL')))==expected
    assert len(calls)==4
    d.to_numpy()[1,:,0]=np.cumsum(x[0])
    spi._set_theiler_window(d,0,1)
    assert len(calls)==8
    expected=int(2*np.dot(original(x[0]),original(np.cumsum(x[0]))))
    assert int(str(spi._calc.getProperty('DYN_CORR_EXCL')))==expected


def test_stochastic_interaction_reuses_directed_information_target_terms(monkeypatch):
    d=data(np.random.default_rng(1307).normal(size=(3,80)))
    DirectedInfo(estimator='kernel',n=2).multivariate(d)
    calls=[]
    original=StochasticInteraction._compute_conditional_entropy
    def counted(self,x,y):
        calls.append(y.shape[1])
        return original(self,x,y)
    monkeypatch.setattr(StochasticInteraction,'_compute_conditional_entropy',counted)
    StochasticInteraction(estimator='kernel').multivariate(d)
    assert calls==[2]*6  # Joint fits only; all three target terms are cached.


@pytest.mark.parametrize('estimator',['kernel','kraskov'])
def test_java_transfer_deepcopy_restores_settings_and_outputs(estimator):
    import copy
    spi=TransferEntropy(estimator=estimator,k_history=3,l_history=2)
    cloned=copy.deepcopy(spi)
    d=data(np.random.default_rng(1308).normal(size=(2,120)))
    assert cloned._calc is not spi._calc
    assert cloned.bivariate(d)==pytest.approx(spi.bivariate(d),abs=1e-12)


def test_non_gaussian_transfer_entropy_pickle_keeps_embedding_settings():
    import pickle
    spi=TransferEntropy(estimator='kraskov',k_history=3,k_tau=2,l_history=2,l_tau=3)
    restored=pickle.loads(pickle.dumps(spi))
    for prop,value in {'k_HISTORY':'3','k_TAU':'2','l_HISTORY':'2','l_TAU':'3'}.items():
        assert restored._calc.getProperty(prop)==value
