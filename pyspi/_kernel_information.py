"""Shared box-kernel information estimates in bits.

Closed neighborhoods, sample-SD bandwidths, and the existing JIDT sample
alignments are preserved. Entropy includes self; MI/TE can exclude |dt| < W.
No jitter or bias correction. Packed neighborhoods share the bounded moment
workspace cache. Large windows stream query rows instead of allocating N x N
floating-point distances or unbounded packed matrices.
"""
from itertools import groupby
import warnings

import numpy as np

from pyspi._cache import cached_workspace
from pyspi._moments import moment_workspace


REASONS = np.array(["estimated", "diagonal", "insufficient_observations",
                    "nonfinite_input", "zero_bandwidth", "no_eligible_neighbors",
                    "numerical_failure"])


class Unavailable(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(str(REASONS[reason]))


class KernelUnavailableWarning(UserWarning):
    """A kernel estimate could not be computed for some channel pairs."""


def validate(width, exclusion):
    if (isinstance(width, (bool, str)) or not np.isscalar(width)
            or np.iscomplexobj(width) or not np.isfinite(width) or width <= 0):
        raise ValueError("kernel_width must be a finite positive number.")
    if exclusion is not None and exclusion != "AUTO":
        if isinstance(exclusion, (bool, np.bool_)) or not isinstance(exclusion, (int, np.integer)) or exclusion < 0:
            raise ValueError("dyn_corr_excl must be None, AUTO, or a nonnegative integer.")


def col(channel, lag=0, bandwidth_lag=None):
    return (channel, lag, lag if bandwidth_lag is None else bandwidth_lag)


class KernelWorkspace:
    block_bytes = 8 * 2**20
    max_mask_bytes = 16 * 2**20

    def __init__(self, work, width):
        self.work, self.source, self.arrays = work, work.source, work.arrays
        self.width = float(width)

    def key(self, *parts):
        return ("kernel_information", self.width, *parts)

    def column(self, descriptor, start, stop):
        channel, lag, bandwidth_lag = descriptor
        n = stop-start
        if n < 2 or start < max(lag, bandwidth_lag) or stop > self.source.shape[1]:
            raise Unavailable(2)
        def compute():
            values = self.source[channel, start-lag:stop-lag]
            bw_values = self.source[channel, start-bandwidth_lag:stop-bandwidth_lag]
            if not np.isfinite(values).all() or not np.isfinite(bw_values).all():
                raise Unavailable(3)
            if np.all(bw_values == bw_values[0]):
                raise Unavailable(4)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                sd = bw_values.std(ddof=1)
                bandwidth = self.width*sd
                span = np.max(values)-np.min(values)
            if np.isfinite(sd) and sd > np.finfo(float).tiny and np.isfinite(span) and bandwidth > 0:
                return values, np.array([bandwidth, np.log2(self.width)+np.log2(sd)])
            # Preserve differential-entropy units while avoiding absolute floors.
            scale = max(np.max(abs(values)), np.max(abs(bw_values)))
            scaled_sd = (bw_values/scale).std(ddof=1)
            if not np.isfinite(scaled_sd) or scaled_sd <= 0:
                raise Unavailable(6)
            return values/scale, np.array([self.width*scaled_sd,
                         np.log2(self.width)+np.log2(scaled_sd)+np.log2(scale)])
        return self.arrays.get(self.key("column", descriptor, start, stop), compute)

    def _pack(self, descriptor, start, stop, first, last):
        values, bandwidth = self.column(descriptor, start, stop)
        n = stop-start
        packed = np.zeros((last-first, (n+63)//64*8), dtype=np.uint8)
        rows = max(1, self.block_bytes//(n*8))
        for a in range(first, last, rows):
            b = min(a+rows, last)
            match = abs(values[a:b, None]-values[None, :]) <= bandwidth[0]
            bits = np.packbits(match, axis=1, bitorder="little")
            packed[a-first:b-first, :bits.shape[1]] = bits
        return packed.view(np.uint64)

    def _mask(self, columns, start, stop, first, last):
        columns = tuple(sorted(columns))
        n = stop-start
        full = first == 0 and last == n
        if len(columns) == 1:
            compute = lambda: self._pack(columns[0],start,stop,first,last)
            return self.arrays.get(self.key("mask",columns,start,stop),compute) if full else compute()
        # Histories of one channel are reused for every partner. Cross-channel
        # products are temporary, avoiding P^2 retained neighborhood matrices.
        def compute():
            masks = [self._mask(tuple(g),start,stop,first,last)
                     for _,g in groupby(columns,key=lambda c:c[0])]
            result = masks[0].copy()
            for mask in masks[1:]:
                result &= mask
            return result
        if len({c[0] for c in columns}) == 1:
            # Avoid recursing into the same single-channel group.
            def compute():
                result = self._mask((columns[0],),start,stop,first,last).copy()
                for c in columns[1:]:
                    result &= self._mask((c,),start,stop,first,last)
                return result
            return self.arrays.get(self.key("mask",columns,start,stop),compute) if full else compute()
        return compute()

    def counts(self, groups, start, stop, exclusion=0):
        output=[]
        for group in groups:
            group=tuple(sorted(group))
            compute=lambda group=group:self._counts((group,),start,stop,exclusion)[0]
            if len({c[0] for c in group})<=1:
                value=self.arrays.get(self.key("counts",group,start,stop,exclusion),compute)
            else:
                value=compute()
            output.append(value)
        return output

    def _counts(self, groups, start, stop, exclusion=0):
        n = stop-start
        if n < 2:
            raise Unavailable(2)
        if exclusion >= n:
            raise Unavailable(5)
        groups = tuple(tuple(sorted(g)) for g in groups)
        packed_bytes = n*((n+63)//64)*8
        step = n if packed_bytes <= self.max_mask_bytes else max(1,self.block_bytes//(n*8))
        out = [np.empty(n,dtype=np.int64) for _ in groups]
        for first in range(0,n,step):
            last = min(n,first+step)
            use = None
            if exclusion:
                def exclusion_mask():
                    mask = abs(np.arange(first,last)[:,None]-np.arange(n)) >= exclusion
                    bits = np.packbits(mask,axis=1,bitorder="little")
                    padded = np.zeros((last-first,(n+63)//64*8),dtype=np.uint8)
                    padded[:,:bits.shape[1]] = bits
                    return padded.view(np.uint64)
                use = (self.arrays.get(self.key("exclusion",n,exclusion),exclusion_mask)
                       if step == n else exclusion_mask())
            for group, result in zip(groups,out):
                if not group:
                    result[first:last] = n if use is None else np.bitwise_count(use).sum(1)
                    continue
                mask = self._mask(group,start,stop,first,last)
                if use is not None:
                    mask = mask & use
                result[first:last] = np.bitwise_count(mask).sum(1)
        return out

    def log_volume(self, columns, start, stop):
        return sum(1+self.column(c,start,stop)[1][1] for c in columns)

    @staticmethod
    def estimate(value, joint, exclusion=0):
        if not np.isfinite(value):
            raise Unavailable(6)
        return float(value), float(np.mean(joint==1)) if not exclusion else np.nan, float(np.mean(joint==0))

    def conditional(self, future, past, start, stop):
        columns = tuple(future)+tuple(past)
        volume = self.log_volume(future,start,stop)
        cp,cj = self.counts((past,columns),start,stop)
        return self.estimate(np.mean(np.log2(cp/cj))+volume,cj)

    def channel_conditional(self, channel, lags, start, stop):
        lags = tuple(lags)
        return self.arrays.get(self.key("channel_conditional",channel,lags,start,stop),
            lambda:np.asarray(self.conditional((col(channel),),tuple(col(channel,l) for l in lags),start,stop)))

    def exclusion(self, setting, i, j):
        if setting != "AUTO":
            return setting or 0
        from pyspi import utils
        def compute():
            acfs = np.array([utils.acf(row) for row in self.source])
            return 2*(acfs @ acfs.T)
        table = self.arrays.get(("information_auto_theiler",),compute)
        if not np.isfinite(table[i,j]):
            raise Unavailable(4 if np.isfinite(self.source[[i,j]]).all() else 3)
        return int(table[i,j])

    def pair(self, measure, i, j, *, dyn_corr_excl=None, k_history=1, history_length=10, n=5, delay=1):
        t = self.source.shape[1]
        if not np.isfinite(self.source[[i,j]]).all():
            raise Unavailable(3)
        if measure in ("je","ce"):
            future,past = ((col(i),col(j)),()) if measure=="je" else ((col(j),),(col(i),))
            return self.conditional(future,past,0,t)
        if measure in ("mi","tlmi"):
            lag = int(measure=="tlmi")
            a,b = (col(i,lag),),(col(j),)
            self.log_volume(a+b,lag,t)  # reject undefined zero-width densities
            exclusion = self.exclusion(dyn_corr_excl,i,j)
            ca,cb,cab,eligible = self.counts((a,b,a+b,()),lag,t,exclusion)
            if np.any(eligible==0):
                raise Unavailable(5)
            ok = cab>0
            value = np.log2(cab[ok].astype(float)*eligible[ok]/(ca[ok].astype(float)*cb[ok])).sum()/(t-lag)
            return self.estimate(value,cab,exclusion)
        if measure=="te":
            k = k_history
            past = tuple(col(j,l) for l in range(1,k+1))
            future = (col(j,0,1),)  # JIDT reuses the nearest target-past width
            source = (col(i,1),)
            self.log_volume(past+future+source,k,t)
            exclusion = self.exclusion(dyn_corr_excl,i,j)
            cp,cnp,cps,cnps,eligible = self.counts((past,past+future,past+source,past+future+source,()),k,t,exclusion)
            if np.any(eligible==0):
                raise Unavailable(5)
            ok = cnps>0
            value = np.log2((cnps[ok]/cps[ok])/(cnp[ok]/cp[ok])).sum()/(t-k)
            return self.estimate(value,cnps,exclusion)
        if measure=="xme":
            k = history_length
            return self.conditional((col(j),),tuple(col(i,l) for l in range(1,k+1)),k,t)
        if measure in ("cce","di"):
            terms = [self.conditional((col(j),),tuple(col(j,l) for l in range(1,m))+
                     tuple(col(i,l) for l in range(m)),m-1,t) for m in range(1,n+1)]
            value = sum(x[0] for x in terms)
            if measure=="di":
                value = sum(self.channel_conditional(j,range(1,m),m-1,t)[0] for m in range(1,n+1))-value
            return value,max(x[1] for x in terms),max(x[2] for x in terms)
        if measure=="si":
            joint = self.conditional((col(i),col(j)),(col(i,delay),col(j,delay)),delay,t)
            value = self.channel_conditional(i,(delay,),delay,t)[0]+self.channel_conditional(j,(delay,),delay,t)[0]-joint[0]
            return value,joint[1],joint[2]
        raise ValueError(f"Unknown kernel measure {measure}")

    def matrix(self, measure, **parameters):
        key = self.key("result",measure,tuple(sorted(parameters.items())))
        def compute():
            p,t = self.source.shape
            values = np.full((p,p),np.nan)
            reason = np.ones((p,p),dtype=np.uint8)
            self_only = np.full((p,p),np.nan)
            zero = np.full((p,p),np.nan)
            if measure == "cce":
                # Align one history length across all pairs before moving on.
                # Pair-first traversal thrashes a bounded cache at large P.
                values[:]=0.;reason[:]=0;self_only[:]=0.;zero[:]=0.
                for m in range(1,parameters.get("n",5)+1):
                    for i in range(p):
                        source=tuple(col(i,l) for l in range(m))
                        for j in range(p):
                            if i==j or reason[i,j]: continue
                            try:
                                target=tuple(col(j,l) for l in range(1,m))
                                value,single,empty=self.conditional((col(j),),target+source,m-1,t)
                                values[i,j]+=value
                                self_only[i,j]=max(self_only[i,j],single)
                                zero[i,j]=max(zero[i,j],empty)
                            except Unavailable as exc:
                                values[i,j]=np.nan;reason[i,j]=exc.reason
                for array in (values,self_only,zero): np.fill_diagonal(array,np.nan)
                np.fill_diagonal(reason,1)
                return values,reason,self_only,zero
            if measure == "ce":
                joint,jreason,jsingle,jzero = self.matrix("je",**parameters)
                for i in range(p):
                    try:
                        marginal = self.channel_conditional(i,(),0,t)[0]
                        values[i] = joint[i]-marginal
                        reason[i],self_only[i],zero[i] = jreason[i],jsingle[i],jzero[i]
                    except Unavailable as exc:
                        reason[i] = exc.reason
                np.fill_diagonal(reason,1)
                return values,reason,self_only,zero
            if measure == "di":
                causal,creason,csingle,czero = self.matrix("cce",**parameters)
                length = parameters.get("n",5)
                for j in range(p):
                    try:
                        target = sum(self.channel_conditional(j,range(1,m),m-1,t)[0] for m in range(1,length+1))
                        values[:,j] = target-causal[:,j]
                        reason[:,j],self_only[:,j],zero[:,j] = creason[:,j],csingle[:,j],czero[:,j]
                    except Unavailable as exc:
                        reason[:,j] = exc.reason
                np.fill_diagonal(reason,1)
                return values,reason,self_only,zero
            symmetric = measure in ("je","mi","si")
            for i in range(p):
                for j in range(i+1 if symmetric else 0,p):
                    if i==j: continue
                    try:
                        values[i,j],self_only[i,j],zero[i,j] = self.pair(measure,i,j,**parameters)
                        reason[i,j] = 0
                    except Unavailable as exc:
                        reason[i,j] = exc.reason
                    if symmetric:
                        values[j,i],reason[j,i],self_only[j,i],zero[j,i] = values[i,j],reason[i,j],self_only[i,j],zero[i,j]
            return values,reason,self_only,zero
        return self.arrays.get(key,compute)


def workspace(data,width):
    work = moment_workspace(data)
    return cached_workspace(data,("kernel_information",float(width)),lambda _:KernelWorkspace(work,width))


def kernel_result(data,measure,*,width=.5,**parameters):
    values,reason,_,_ = workspace(data,width).matrix(measure,**parameters)
    invalid=reason[reason>1]
    if invalid.size:
        counts=zip(*np.unique(invalid,return_counts=True))
        details=", ".join(f"{REASONS[r]}: {n}" for r,n in counts)
        warnings.warn(f"Kernel {measure} unavailable ordered pairs ({details}); see diagnostics().",
                      KernelUnavailableWarning,stacklevel=3)
    return values.copy()


def kernel_pair(data,measure,i,j,*,width=.5,**parameters):
    try:
        return workspace(data,width).pair(measure,i,j,**parameters)[0]
    except Unavailable as exc:
        warnings.warn(f"Kernel {measure} unavailable: {exc}; see diagnostics().",
                      KernelUnavailableWarning,stacklevel=3)
        return np.nan


def kernel_diagnostics(data,measure,*,width=.5,**parameters):
    _,reason,self_only,zero = workspace(data,width).matrix(measure,**parameters)
    return {"reason":REASONS[reason].copy(),"joint_self_only_fraction":self_only.copy(),
            "zero_joint_count_fraction":zero.copy()}


def kernel_entropy(observations,width):
    """Standalone entropy for callers without channel/alignment descriptors."""
    from pyspi._moments import MomentWorkspace
    x = np.asarray(observations,dtype=float)
    if not x.shape[1]:
        return 0.
    work = KernelWorkspace(MomentWorkspace(x.T[:,:,None]),width)
    try:
        return work.conditional(tuple(col(c) for c in range(x.shape[1])),(),0,len(x))[0]
    except Unavailable:
        return np.nan
