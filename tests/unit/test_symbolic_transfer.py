"""Independent categorical counts for ordinal transfer entropy."""

from collections import Counter
import math
import pickle

import numpy as np
import pytest

from pyspi._moments import moment_workspace
from pyspi.data import Data
from pyspi.statistics.infotheory import TransferEntropy


def data(x):
    return Data(np.array(x,copy=True),normalise=False,detrend=False)


def reference(source,target,length,spacing,shift):
    def patterns(x):
        result=[]
        for t in range((length-1)*spacing,len(x)):
            values=[x[t-(length-1-i)*spacing] for i in range(length)]
            result.append(tuple(sorted(range(length),key=lambda i:(values[i],i))))
        return result
    x,y=patterns(source),patterns(target)
    observations=list(zip(x[:-shift],y[:-shift],y[shift:]))
    xyz=Counter(observations)
    xy=Counter((a,b) for a,b,c in observations)
    yz=Counter((b,c) for a,b,c in observations)
    yp=Counter(b for a,b,c in observations)
    return sum(count/len(observations)*math.log(count*yp[b]/(xy[a,b]*yz[b,c]))
               for (a,b,c),count in xyz.items())


@pytest.mark.parametrize('length,spacing,shift',[(2,1,1),(3,1,1),(3,4,4),(3,16,16),(3,4,8),(5,2,2)])
def test_symbolic_matches_independent_counts_and_alignment(length,spacing,shift):
    x=np.random.default_rng(1401).normal(size=(4,220))
    x[1,shift:]=x[0,:-shift]+.3*x[1,shift:]
    x[3]=np.round(x[3],1)  # Exercise time-ordered ties as well as continuous data.
    spi=TransferEntropy(estimator='symbolic',pattern_length=length,pattern_spacing=spacing,prediction_shift=shift)
    d=data(x); actual=spi.multivariate(d)
    expected=np.full(actual.shape,np.nan)
    for i in range(len(x)):
        for j in range(len(x)):
            if i!=j:
                expected[i,j]=reference(x[i],x[j],length,spacing,shift)
    np.testing.assert_allclose(actual,expected,atol=4e-15)
    order=[2,0,3,1]
    np.testing.assert_array_equal(spi.multivariate(data(x[order])),actual[np.ix_(order,order)])
    assert spi.bivariate(d,i=0,j=1)==actual[0,1]
    np.testing.assert_allclose(spi.multivariate(data(np.exp(x))),actual,atol=1e-15)
    assert np.nanmin(actual)>=0


def test_dense_and_sparse_paths_and_shared_symbolization():
    x=np.random.default_rng(1402).normal(size=(3,200)); d=data(x)
    first=TransferEntropy(estimator='symbolic',pattern_spacing=4,prediction_shift=4)
    second=TransferEntropy(estimator='symbolic',pattern_spacing=4,prediction_shift=8)
    expected=first.multivariate(d)
    work=moment_workspace(d); symbols=work.arrays.entries[('ordinal_symbols',3,4)]
    second.multivariate(d)
    assert work.arrays.entries[('ordinal_symbols',3,4)] is symbols
    tiny=data(x); small=moment_workspace(tiny); small.block_bytes=256; small.arrays.max_bytes=2048
    np.testing.assert_allclose(first.multivariate(tiny),expected,atol=2e-15)
    assert small.arrays.nbytes<=2048


def test_symbolic_direction_bias_and_degeneracy_are_visible():
    rng=np.random.default_rng(1403); x=rng.normal(size=(3,2000))
    x[1,4:]=x[0,:-4]
    spi=TransferEntropy(estimator='symbolic',pattern_spacing=4,prediction_shift=4)
    d=data(x); actual=spi.multivariate(d)
    assert actual[0,1] > actual[1,0]+.5
    assert actual[0,1] > actual[2,1]+.5
    diagnostic=spi.diagnostics(d)
    assert diagnostic['n_observations']==1988
    np.testing.assert_array_equal(diagnostic['patterns_observed'],6)
    np.testing.assert_array_equal(diagnostic['tied_pattern_fraction'],0.)
    assert 0<diagnostic['joint_states_observed'][0,1]<=216
    diagnostic['reason'][:]='changed'
    assert spi.diagnostics(d)['reason'][0,1]=='estimated'
    x[2]=1.
    constant=spi.multivariate(data(x))
    np.testing.assert_allclose(constant[2,:2],0.,atol=2e-15)
    np.testing.assert_allclose(constant[:2,2],0.,atol=2e-15)
    assert spi.diagnostics(data(x))['tied_pattern_fraction'][2]==1.
    d.to_numpy()[2,100,0]=np.nan
    invalid=spi.multivariate(d)
    assert np.isnan(invalid[2]).all() and np.isnan(invalid[:,2]).all()
    assert invalid[0,1]==actual[0,1]
    assert np.isnan(spi.multivariate(data(x[:,:8]))).all()


def test_symbolic_configuration_is_explicit_and_serializable():
    spi=TransferEntropy(estimator='symbolic',pattern_length=3,pattern_spacing=16,prediction_shift=16)
    assert spi.identifier=='te_symbolic_m-3_tau-16_h-16'
    assert not spi.requires_java(estimator='symbolic') and spi._calc is None
    d=data(np.random.default_rng(1404).normal(size=(2,100)))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(spi)).multivariate(d),spi.multivariate(d))
    for kwargs in ({'k_history':10},{'pattern_length':1},{'pattern_spacing':0},{'prediction_shift':True}):
        with pytest.raises(ValueError):
            TransferEntropy(estimator='symbolic',**kwargs)
    with pytest.raises(ValueError,match='apply only'):
        TransferEntropy(pattern_length=3)
