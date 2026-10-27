"""Cross-correlation definitions, independent numerical references, and caches."""

from unittest.mock import patch
import warnings

import numpy as np
import pytest
from scipy.linalg import toeplitz
from statsmodels.stats.multitest import multipletests

from pyspi._xcorr import (SUMMARY_KEYS, bartlett_variance,
                         holm_mask, summarize)
from pyspi.data import Data
from pyspi.statistics.basic import CrossCorrelation, CrossCorrelationUnavailableWarning


def data(x):
    return Data(np.asarray(x), normalise=False, detrend=False)


def direct_curve(x, y, lag):
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.dot(x, x)*np.dot(y, y))
    n = len(x)
    return np.array([sum(x[t]*y[t-h] for t in range(n) if 0 <= t-h < n)
                     / denominator for h in range(-lag, lag+1)])


@pytest.mark.parametrize("shape", [(2, 31), (5, 128), (3, 257)])
@pytest.mark.parametrize("statistic,squared", SUMMARY_KEYS)
def test_matches_independent_direct_sums(shape, statistic, squared):
    x = np.random.default_rng(701).normal(size=shape)*np.arange(1, shape[0]+1)[:, None] + 30
    expected = np.full((shape[0], shape[0]), np.nan)
    for i in range(shape[0]):
        for j in range(i+1, shape[0]):
            r = direct_curve(x[i], x[j], 7)
            if squared: r = r*r
            expected[i,j] = expected[j,i] = r[np.argmax(abs(r))] if statistic == "peak" else getattr(np, statistic)(r)
    spi = CrossCorrelation(statistic=statistic, squared=squared, sigonly=False, fs=56)
    actual = spi.multivariate(data(x))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    for i in range(shape[0]):
        for j in range(i+1, shape[0]):
            np.testing.assert_allclose(spi.bivariate(data(x),i=j,j=i), expected[i,j], atol=1e-14)


def test_operation_order_and_opposite_sign_peak_ties():
    curves = np.array([[-.9,.2],[-.9,.9],[.4,.8],[0.,0.]])
    result,count,ambiguous = summarize(curves,np.ones_like(curves,dtype=bool))
    np.testing.assert_allclose(result[:,0],[-.35,.2,-.9,-.9,.425,.81,.04])
    assert np.isnan(result[3,1]) and ambiguous[1]
    assert result[2,2] == .4  # min is numerical minimum, not a negative-only gate.
    assert result[3,3] == 0
    assert not ambiguous[0] and (count == 2).all()


@pytest.mark.parametrize("sigonly", [False,True])
@pytest.mark.parametrize("statistic,squared", SUMMARY_KEYS)
def test_channel_order_and_additional_channels_do_not_change_pair(sigonly,statistic,squared):
    x = np.random.default_rng(702).normal(size=(4,1024))
    x[1,17:] += 2*x[0,:-17]
    spi = CrossCorrelation(statistic=statistic,squared=squared,sigonly=sigonly,fs=2048)
    with np.errstate(invalid="ignore"):
        a = spi.multivariate(data(x))
        permutation = [3,1,0,2]
        b = spi.multivariate(data(x[permutation]))
        c = spi.multivariate(data(x[:2]))
    np.testing.assert_allclose(b,a[np.ix_(permutation,permutation)],atol=1e-14)
    np.testing.assert_allclose(c[0,1],a[0,1],atol=1e-14)


@pytest.mark.parametrize("fs,seconds,lag",[(2048,.125,256),(1000,.125,125),(1,.125,0),(8,.125,1),(2048,0,0)])
def test_explicit_time_limit_and_inclusive_native_lags(fs,seconds,lag):
    x = np.random.default_rng(703).normal(size=(2,1024))
    d = data(x);spi = CrossCorrelation(fs=fs,max_lag_seconds=seconds,sigonly=False)
    spi.multivariate(d)
    assert spi.last_diagnostics['maximum_lag_samples'] == lag
    assert spi.last_diagnostics['selected_lag_count'][0,1] == 2*lag+1
    w,_,_ = spi._workspace(d)
    curve = next(w.blocks(lag))[2][0]
    np.testing.assert_allclose(curve,direct_curve(*x,lag),atol=1e-14)


def test_zero_lag_matches_pearson_and_is_offset_scale_invariant():
    rng=np.random.default_rng(704);x=rng.normal(size=(3,128));x[1]=x[0]
    spi=CrossCorrelation(sigonly=False,max_lag_seconds=0)
    expected=np.corrcoef(x);np.fill_diagonal(expected,np.nan)
    np.testing.assert_allclose(spi.multivariate(data(x)),expected,atol=1e-14)
    np.testing.assert_allclose(spi.multivariate(data(x*[[2],[.1],[5]]+[[40],[-20],[5]])),expected,atol=1e-13)


@pytest.mark.parametrize("scale",[1e-300,1e300])
def test_normalization_handles_extreme_amplitudes(scale):
    x=np.random.default_rng(705).normal(size=(2,64))
    spi=CrossCorrelation(sigonly=False,fs=128)
    np.testing.assert_allclose(spi.multivariate(data(x*scale)),spi.multivariate(data(x)),atol=1e-14)


@pytest.mark.parametrize("ax,ay",[(0.,0.),(.8,.5),(-.7,.6)])
def test_bartlett_variance_matches_covariance_matrix_product(ax,ay):
    n=31;overlap=np.arange(1,n+1)
    acf_x=ax**np.arange(n);acf_y=ay**np.arange(n)
    expected=np.array([np.sum(toeplitz(acf_x[:m])*toeplitz(acf_y[:m]))/n**2 for m in overlap])
    np.testing.assert_allclose(bartlett_variance(acf_x,acf_y,overlap,n),expected,atol=1e-15)
    if ax == ay == 0:np.testing.assert_array_equal(expected,overlap/n**2)


def test_holm_matches_independent_library_and_is_row_local():
    p=np.random.default_rng(706).uniform(size=(6,19))**4;p[0,0]=np.nan
    actual=holm_mask(p,.05)
    for i,row in enumerate(p):
        expected=multipletests(np.nan_to_num(row,nan=1),alpha=.05,method='holm')[0]
        np.testing.assert_array_equal(actual[i],expected)
        np.testing.assert_array_equal(holm_mask(row[None,:],.05)[0],expected)


def test_significance_uses_signed_curve_before_squaring_and_keeps_delayed_peaks():
    rng=np.random.default_rng(707);x=rng.normal(size=(2,1024));x[1,23:]=-.9*x[0,:-23]+.1*x[1,23:]
    d=data(x)
    a=CrossCorrelation(statistic='min',sigonly=True,fs=2048)
    b=CrossCorrelation(statistic='max',squared=True,sigonly=True,fs=2048)
    assert abs(np.corrcoef(x)[0,1]) < .1
    assert a.multivariate(d)[0,1] < -.85
    assert b.multivariate(d)[0,1] > .85**2
    np.testing.assert_array_equal(a.last_diagnostics['selected_lag_count'],b.last_diagnostics['selected_lag_count'])
    assert a.last_diagnostics['selected_lag_count'][0,1] > 0


def test_no_qualifying_lags_are_distinguished_from_undefined_input():
    x=np.random.default_rng(708).normal(size=(3,128));x[2]=0
    spi=CrossCorrelation(sigonly=True,fs=128,alpha=1e-30)
    with pytest.warns(CrossCorrelationUnavailableWarning,match='no_significant_lags'):
        result=spi.multivariate(data(x))
    assert np.isnan(result).all()
    assert spi.last_diagnostics['status'][0,1]=='no_significant_lags'
    assert spi.last_diagnostics['status'][0,2]=='undefined_input'


def test_fft_and_selection_reused_across_summaries_and_scalar_api():
    import pyspi._xcorr as backend
    x=np.random.default_rng(709).normal(size=(4,128));d=data(x)
    with patch.object(backend,'rfft',wraps=backend.rfft) as fft:
        for statistic,squared in SUMMARY_KEYS:
            spi=CrossCorrelation(statistic=statistic,squared=squared,sigonly=False,fs=128)
            matrix=spi.multivariate(d)
            assert spi.bivariate(d,i=1,j=0)==matrix[0,1]
        assert fft.call_count == 1
    previous = spi._workspace(d)[0]
    d.to_numpy()[0,:,0]*=-1
    assert spi._workspace(d)[0] is not previous
    np.testing.assert_allclose(spi.multivariate(d), spi.multivariate(data(d.to_numpy()[:, :, 0].copy())), atol=1e-14)


def test_significance_selection_is_shared_and_settings_do_not_collide():
    import pyspi._xcorr as backend
    d = data(np.random.default_rng(713).normal(size=(3, 128)))
    with warnings.catch_warnings(), patch.object(backend, "holm_mask", wraps=backend.holm_mask) as holm:
        warnings.simplefilter("ignore", CrossCorrelationUnavailableWarning)
        CrossCorrelation(fs=128).multivariate(d)
        calls = holm.call_count
        assert calls > 0
        for statistic, squared in SUMMARY_KEYS:
            CrossCorrelation(statistic=statistic, squared=squared, fs=128).multivariate(d)
        assert holm.call_count == calls
        CrossCorrelation(fs=128, alpha=.01).multivariate(d)
        assert holm.call_count > calls
    a = CrossCorrelation(sigonly=False, fs=128)
    b = CrossCorrelation(sigonly=False, fs=128, max_lag_seconds=.25)
    a.multivariate(d); b.multivariate(d)
    assert a.identifier != b.identifier
    assert a.last_diagnostics['selected_lag_count'][0, 1] == 33
    assert b.last_diagnostics['selected_lag_count'][0, 1] == 65


def test_insufficient_observations_and_nonfinite_input_have_explicit_reasons():
    spi = CrossCorrelation()
    with pytest.warns(CrossCorrelationUnavailableWarning, match="undefined_inference"):
        result = spi.multivariate(data([[0., 1.], [1., 0.]]))
    assert np.isnan(result).all()
    assert spi.last_diagnostics['status'][0, 1] == 'undefined_inference'
    d = data(np.random.default_rng(714).normal(size=(3, 32)))
    d.to_numpy()[1, 0, 0] = np.inf
    spi = CrossCorrelation(sigonly=False)
    with pytest.warns(CrossCorrelationUnavailableWarning, match="undefined_input"):
        result = spi.multivariate(d)
    assert np.isfinite(result[0, 2]) and np.isnan(result[0, 1])


def test_cache_eviction_and_small_batches_preserve_values():
    x=np.random.default_rng(710).normal(size=(5,128));d=data(x)
    spi=CrossCorrelation(sigonly=False,fs=128)
    expected=spi.multivariate(d)
    work,lag,alpha=spi._workspace(data(x))
    work.arrays.max_bytes=32;work.block_bytes=1
    actual=work.all_pairs(lag,alpha)[0][SUMMARY_KEYS.index(('max',False))]
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    assert work.arrays.nbytes <= 32


def test_diagnostics_and_results_do_not_expose_cached_arrays():
    d=data(np.random.default_rng(711).normal(size=(3,64)))
    spi=CrossCorrelation(sigonly=False,fs=128)
    expected=spi.multivariate(d)
    spi.last_diagnostics['status'][:]='changed';spi.last_diagnostics['selected_lag_count'][:]=99
    output=spi.multivariate(d);output[:]=99
    np.testing.assert_array_equal(spi.multivariate(d),expected)
    assert spi.last_diagnostics['status'][0,1]=='ok'
    assert spi.last_diagnostics['selected_lag_count'][0,1]==33


@pytest.mark.parametrize('kwargs',[{'fs':0},{'fs':np.inf},{'max_lag_seconds':-1},{'alpha':0},{'alpha':1},
                                   {'statistic':'median'},{'statistic':'peak','squared':True},
                                   {'sigonly':False,'alpha':.01}])
def test_invalid_definitions_rejected(kwargs):
    with pytest.raises(ValueError):CrossCorrelation(**kwargs)


def test_unsupported_data_and_lag_are_explicit():
    with pytest.raises(ValueError,match='shorter than'):
        CrossCorrelation(fs=2048).multivariate(data(np.ones((2,64))))
    with pytest.raises(ValueError,match='two observations'):
        CrossCorrelation().multivariate(data(np.ones((2,1))))
    with pytest.raises(ValueError,match='single realisation'):
        CrossCorrelation().multivariate(Data(np.ones((2,8,2)),dim_order='psr',normalise=False))


def test_calculator_records_no_significant_lags_as_a_successful_missing_summary(tmp_path):
    from pyspi.calculator import Calculator
    cfg=tmp_path/'xcorr.yaml'
    cfg.write_text('.statistics.basic:\n  CrossCorrelation:\n    configs:\n      - fs: 128\n        alpha: 1.0e-30\n')
    calc=Calculator(dataset=np.random.default_rng(712).normal(size=(2,128)),configfile=str(cfg))
    with pytest.warns(CrossCorrelationUnavailableWarning,match='no_significant_lags'):
        calc.compute()
    report=next(iter(calc.metric_reports.values()))
    assert report.status=='success' and report.error is None
    assert any('no_significant_lags' in warning.message for warning in report.warnings)
    assert np.isnan(calc.table.to_numpy()).all()
