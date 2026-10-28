"""Aligned Gaussian prediction errors shared by temporal information measures.

Rows are sources; columns are targets. Covariances use unit-energy, centered
observations. Entropy units are restored from log sample variances. There is
no ridge or jitter: ill-conditioned moment solves fall back to least squares
on the observations, dropping only numerically redundant conditioning columns.
"""

import numpy as np
from scipy.linalg.lapack import dpocon

from pyspi._moments import moment_workspace


H_UNIT = .5*(1+np.log(2*np.pi))
# Codes 0..6 agree with _gaussian.REASONS; 7 is a valid rank-reduced fit.
REDUNDANT_CONDITIONING = 7


def _ill_conditioned(chol, covariance):
    """Estimate reciprocal 1-norm condition without a second factorization.

    Small Cholesky diagonal entries alone miss very ill-conditioned lag
    designs (including interpolated EEG). POCON reuses the Cholesky factor;
    the two-predictor case has a cheap exact formula for this condition norm.
    """
    norm = np.max(np.sum(abs(covariance), axis=-2), axis=-1)
    if chol.shape[-1] == 2:
        determinant = (chol[..., 0, 0]*chol[..., 1, 1])**2
        reciprocal = determinant/norm**2
    else:
        reciprocal = np.array([dpocon(c, a, uplo="L")[0] for c, a in zip(chol, norm)])
    return reciprocal < 1e-8


class AlignedHistory:
    def __init__(self, work, start, lags):
        self.work = work
        self.start = start
        self.lags = tuple(sorted(set(lags)))
        self.p, self.t = work.source.shape
        self.n = self.t-start
        self.key = ("gaussian_history_gram_channel_major", start, self.lags)

    def indices(self, channels, lags):
        channels = np.asarray(channels)
        offsets = np.array([self.lags.index(lag) for lag in lags], dtype=int)
        return channels[..., None]*len(self.lags)+offsets

    def observations(self, indices):
        indices = np.asarray(indices)
        flat = indices.ravel()
        result = np.empty((len(flat), self.n))
        for slot in np.unique(flat % len(self.lags)):
            mask = flat % len(self.lags) == slot
            lag = self.lags[slot]
            unit, _, _ = self.work.standardized(self.start-lag, self.t-lag)
            result[mask] = unit[flat[mask]//len(self.lags)]
        return result.reshape(indices.shape+(self.n,))

    def log_variance(self):
        return self.work.standardized(self.start, self.t)[1]

    def gram(self, indices, *, shared=False):
        size = self.p*len(self.lags)
        # The global Gram saves all repeated cross-products for a dense pair
        # workload. Larger requests stay in observation blocks instead of
        # allocating an unbounded (channels*lags)^2 matrix.
        allow_global = size*size*8 <= self.work.arrays.max_bytes//2
        if self.key in self.work.arrays.entries or (shared and allow_global):
            def compute():
                observations = self.observations(np.arange(size))
                return observations @ observations.T
            gram = self.work.arrays.get(self.key, compute)
            return gram[indices[..., :, None], indices[..., None, :]]
        observations = self.observations(indices)
        return observations @ np.swapaxes(observations, -1, -2)


def _fit(history, targets, predictors, *, shared=False):
    """Conditional normalized covariance, log determinant and diagnostic codes.

    Most work uses batched small covariance solves. Near-singular conditioning
    or tiny prediction errors use direct residuals, so no subtraction/clipping
    invents a residual variance. The sample rank also detects saturated fits.
    """
    count, dy = targets.shape
    dx = predictors.shape[1]
    result = np.full((count, dy, dy), np.nan)
    logdet = np.full(count, np.nan)
    reason = np.zeros(count, dtype=np.uint8)
    if history.n < 2:
        reason[:] = 4
        return result, logdet, reason
    dimension = dx+dy
    global_size = history.p*len(history.lags)
    dense = history.key in history.work.arrays.entries or (shared and global_size**2*8 <= history.work.arrays.max_bytes//2)
    extent = dimension if dense else max(dimension, history.n)
    step = max(1, min(512, history.work.block_bytes//(4*dimension*extent*8)))
    for first in range(0, count, step):
        end = min(first+step, count)
        y, x = targets[first:end], predictors[first:end]
        indices = np.concatenate([x, y], axis=1)
        matrix = history.gram(indices, shared=shared)
        finite = np.isfinite(matrix).all(axis=(1, 2))
        reason[first:end][~finite] = 3
        fallback = np.zeros(end-first, dtype=bool)
        cov = result[first:end]
        if dx == 0:
            cov[finite] = matrix[finite]
        elif dx == 1:
            xx = matrix[:, 0, 0]
            xy = matrix[:, :1, 1:]
            slope = np.divide(xy, xx[:, None, None], out=np.zeros_like(xy), where=xx[:, None, None] > 0)
            cov[finite] = (matrix[:, 1:, 1:]-np.swapaxes(xy, 1, 2) @ slope)[finite]
            reason[first:end][finite & (xx == 0)] = REDUNDANT_CONDITIONING
        else:
            selected = np.flatnonzero(finite)
            if len(selected):
                xx = matrix[selected, :dx, :dx]
                xy = matrix[selected, :dx, dx:]
                try:
                    chol = np.linalg.cholesky(xx)
                    projected = np.linalg.solve(chol, xy)
                    cov[selected] = matrix[selected, dx:, dx:]-np.swapaxes(projected, 1, 2) @ projected
                    fallback[selected] = _ill_conditioned(chol, xx)
                except np.linalg.LinAlgError:
                    # One singular pair must not invalidate its entire block.
                    for local, a, b in zip(selected, xx, xy):
                        try:
                            chol = np.linalg.cholesky(a)
                            projected = np.linalg.solve(chol, b)
                            cov[local] = matrix[local, dx:, dx:]-projected.T @ projected
                            fallback[local] = _ill_conditioned(chol[None], a[None])[0]
                        except np.linalg.LinAlgError:
                            fallback[local] = True
        if history.n <= dx+dy:
            fallback |= finite
        # Scalar tiny errors and joint nearly singular residual covariances
        # both need an observation-space check.
        if dy == 1:
            fallback |= finite & (cov[:, 0, 0] <= 1e-8)
        else:
            eigenvalues = np.linalg.eigvalsh(np.where(np.isfinite(cov), cov, 0.))
            fallback |= finite & (eigenvalues[:, 0] <= 1e-8)
        stable = finite & ~fallback
        if dy == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                logdet[first:end][stable] = np.log(cov[stable, 0, 0])
        else:
            _, logdet[first:end][stable] = np.linalg.slogdet(cov[stable])
        for local in np.flatnonzero(fallback):
            yy = history.observations(y[local]).T
            xx = history.observations(x[local]).T
            try:
                beta, _, rank, _ = np.linalg.lstsq(xx, yy, rcond=None)
                if history.n-rank-1 < dy:
                    reason[first+local] = 4
                    cov[local] = np.nan
                    continue
                residual = yy-xx @ beta
                singular = np.linalg.svd(residual, compute_uv=False)
                # A floating-point zero is distinguished from a resolvable
                # small residual, without a threshold in physical units.
                singular[singular <= 8*np.finfo(float).eps*max(1, dx)] = 0.
                cov[local] = residual.T @ residual
                if dy == 1 and singular[0] == 0:
                    cov[local, 0, 0] = 0.
                with np.errstate(divide="ignore"):
                    logdet[first+local] = 2*np.log(singular).sum()
                if rank < dx:
                    reason[first+local] = REDUNDANT_CONDITIONING
            except np.linalg.LinAlgError:
                reason[first+local] = 6
                cov[local] = np.nan
    return result, logdet, reason


def _self_prediction(work, start, lags):
    """One target-only model per channel, shared across every source pairing."""
    def compute():
        history = AlignedHistory(work, start, (0, *lags))
        channels = np.arange(history.p)
        _, logvar, reason = _fit(history, history.indices(channels, (0,)), history.indices(channels, lags))
        return logvar, reason
    return work.arrays.get(("gaussian_self_prediction", start, tuple(lags)), compute)


def _prediction(work, source_lags, target_lags, *, pairs=None):
    start = max((0, *source_lags, *target_lags))
    history = AlignedHistory(work, start, (0, *source_lags, *target_lags))
    if pairs is None:
        pairs = np.nonzero(~np.eye(history.p, dtype=bool))
    source, target = pairs
    targets = history.indices(target, (0,))
    predictors = np.concatenate([history.indices(target, target_lags),
                                 history.indices(source, source_lags)], axis=1)
    # A sparse auto-embedding group should not construct a dense all-channel
    # history Gram merely to evaluate a handful of selected pairs.
    shared = len(source)*(predictors.shape[1]+1)**2 >= (history.p*len(history.lags))**2/4
    _, logvar, reason = _fit(history, targets, predictors, shared=shared)
    return logvar, reason


def _prediction_matrix(work, source_lags, target_lags):
    def compute():
        if tuple(source_lags) == (0,) and not target_lags:
            from pyspi._gaussian import _pair_information
            mi, _, target_entropy, status = _pair_information(work)
            value, reason = -2*mi, status.copy()
            value[:, np.isneginf(target_entropy)] = -np.inf
            np.fill_diagonal(value, np.nan)
            np.fill_diagonal(reason, 5)
            return value, reason
        p = len(work.source)
        pairs = np.nonzero(~np.eye(p, dtype=bool))
        value = np.full((p, p), np.nan)
        reason = np.full((p, p), 5, dtype=np.uint8)
        value[pairs], reason[pairs] = _prediction(work, source_lags, target_lags, pairs=pairs)
        return value, reason
    return work.arrays.get(("gaussian_prediction", tuple(source_lags), tuple(target_lags)), compute)


def _information(restricted, full, reason):
    restricted, full = np.broadcast_arrays(restricted, full)
    reason = np.broadcast_to(reason, full.shape).copy()
    with np.errstate(invalid="ignore"):
        value = .5*(restricted-full)
    redundant_target = np.isneginf(restricted) & np.isneginf(full)
    value[redundant_target] = 0.
    reason[redundant_target] = REDUNDANT_CONDITIONING
    reason[np.isposinf(value)] = 2
    inconsistent = value < -1e-10
    reason[inconsistent] = 6
    value[inconsistent] = np.nan
    value[np.isfinite(value) & (value < 0)] = 0.
    return value, reason


def _entropy(work, start, log_fraction, reason):
    reason = reason.copy()
    if work.source.shape[1]-start < 2:
        return np.full_like(log_fraction, np.nan), np.full_like(reason, 4)
    _, logvar, _ = work.standardized(start)
    value = H_UNIT+.5*(log_fraction+logvar[None, :])
    reason[np.isneginf(log_fraction)] = 2
    reason[np.broadcast_to(np.isneginf(logvar), value.shape)] = 1
    value[~np.isfinite(value)] = np.nan
    np.fill_diagonal(value, np.nan)
    np.fill_diagonal(reason, 5)
    return value, reason


def _fixed_transfer(work, k, kt, l, lt):
    target_lags = tuple(1+kt*np.arange(k))
    source_lags = tuple(1+lt*np.arange(l))
    start = max(*target_lags, *source_lags)
    own, own_reason = _self_prediction(work, start, target_lags)
    full, reason = _prediction_matrix(work, source_lags, target_lags)
    invalid = np.isin(own_reason, [3, 4, 6])
    reason = reason.copy()
    reason[:, invalid] = own_reason[invalid]
    value, reason = _information(own[None, :], full, reason)
    np.fill_diagonal(value, np.nan)
    np.fill_diagonal(reason, 5)
    return value, reason


def _auto_embedding(work, kmax, taumax):
    def compute():
        p, t = work.source.shape
        best = np.full(p, -np.inf)
        selected = np.ones((p, 2), dtype=int)
        valid = np.zeros(p, dtype=bool)
        for k in range(1, kmax+1):
            for tau in range(1, (1 if k == 1 else taumax)+1):
                lags = tuple(1+tau*np.arange(k))
                start = max(lags)
                if t-start < 2:
                    continue
                residual, reason = _self_prediction(work, start, lags)
                # JIDT MAX_CORR_AIS: subtract the candidate's own finite-sample
                # mean AIS bias, then retain the first strictly best choice.
                score = -.5*residual-k/(2*(t-start))
                constant = np.isneginf(work.standardized(start)[1])
                score[constant] = -k/(2*(t-start))
                eligible = ~np.isin(reason, [3, 4, 6]) & ~np.isnan(score)
                choose = eligible & (score > best)
                selected[choose] = k, tau
                best[choose] = score[choose]
                valid |= eligible
        return selected, best, valid
    return work.arrays.get(("gaussian_auto_embedding", kmax, taumax), compute)


def _auto_transfer(work, kmax, taumax):
    selected, _, valid = _auto_embedding(work, kmax, taumax)
    p = len(selected)
    value = np.full((p, p), np.nan)
    reason = np.full((p, p), 4, dtype=np.uint8)
    groups = {tuple(pair): np.flatnonzero(np.all(selected == pair, axis=1) & valid)
              for pair in np.unique(selected, axis=0)}
    for (k, kt), targets in groups.items():
        for (l, lt), sources in groups.items():
            source, target = np.meshgrid(sources, targets, indexing="ij")
            mask = source != target
            source, target = source[mask], target[mask]
            if not len(source):
                continue
            tlags, slags = tuple(1+kt*np.arange(k)), tuple(1+lt*np.arange(l))
            start = max(*tlags, *slags)
            own, own_reason = _self_prediction(work, start, tlags)
            full, status = _prediction(work, slags, tlags, pairs=(source, target))
            invalid = np.isin(own_reason[target], [3, 4, 6])
            status[invalid] = own_reason[target][invalid]
            value[source, target], reason[source, target] = _information(own[target], full, status)
    bad = ~np.isfinite(work.source).all(axis=1)
    reason[bad[:, None] | bad[None, :]] = 3
    np.fill_diagonal(reason, 5)
    return value, reason


def _causal_information(work, n):
    def compute():
        p = len(work.source)
        causal, directed = np.zeros((p, p)), np.zeros((p, p))
        causal_reason = np.zeros((p, p), dtype=np.uint8)
        directed_reason = np.zeros((p, p), dtype=np.uint8)
        for m in range(1, n+1):
            slags, tlags = tuple(range(m)), tuple(range(1, m))
            full, status = _prediction_matrix(work, slags, tlags)
            own, own_status = _self_prediction(work, m-1, tlags)
            h, hs = _entropy(work, m-1, full, status)
            information, info_status = _information(own[None, :], full, status)
            invalid = np.isin(own_status, [3, 4, 6])
            information[:, invalid] = np.nan
            info_status[:, invalid] = own_status[invalid]
            causal += h
            directed += information
            # Undefined terms dominate a valid redundant-conditioning note.
            for total, term, values in ((causal_reason, hs, causal), (directed_reason, info_status, directed)):
                replace = (total == 0) | ((total == REDUNDANT_CONDITIONING) & (term != 0))
                total[replace] = term[replace]
                total[np.isnan(values) & ~np.isin(total, [1, 2, 3, 4, 5, 6])] = 6
        for matrix in (causal, directed):
            np.fill_diagonal(matrix, np.nan)
        for status in (causal_reason, directed_reason):
            np.fill_diagonal(status, 5)
        return causal, causal_reason, directed, directed_reason
    return work.arrays.get(("gaussian_causal_information", n), compute)


def _stochastic_interaction(work, delay):
    p = len(work.source)
    history = AlignedHistory(work, delay, (0, delay))
    i, j = np.triu_indices(p, 1)
    own, own_status = _self_prediction(work, delay, (delay,))
    channels = np.stack([i, j], axis=1)
    targets = history.indices(channels, (0,)).reshape(-1, 2)
    predictors = history.indices(channels, (delay,)).reshape(-1, 2)
    _, joint_logdet, status = _fit(history, targets, predictors, shared=True)
    with np.errstate(invalid="ignore"):
        estimate = .5*(own[i]+own[j]-joint_logdet)
    singular = np.isneginf(joint_logdet) | ~np.isfinite(own[i]+own[j])
    estimate[singular] = np.nan
    status[singular] = 2
    for ch in (i, j):
        invalid = np.isin(own_status[ch], [3, 4, 6])
        status[invalid] = own_status[ch][invalid]
    inconsistent = estimate < -1e-10
    estimate[inconsistent], status[inconsistent] = np.nan, 6
    estimate[np.isfinite(estimate) & (estimate < 0)] = 0.
    value = np.full((p, p), np.nan)
    reason = np.full((p, p), 5, dtype=np.uint8)
    value[i, j] = value[j, i] = estimate
    reason[i, j] = reason[j, i] = status
    return value, reason


def history_result(data, measure, **parameters):
    work = moment_workspace(data)
    def compute():
        if measure == "te":
            if parameters.get("auto_embed_method") is not None:
                return _auto_transfer(work, parameters["k_search_max"], parameters["tau_search_max"])
            return _fixed_transfer(work, parameters["k_history"], parameters["k_tau"],
                                   parameters["l_history"], parameters["l_tau"])
        if measure == "xme":
            k = parameters["history_length"]
            fraction, status = _prediction_matrix(work, tuple(range(1, k+1)), ())
            return _entropy(work, k, fraction, status)
        if measure in ("cce", "di"):
            values = _causal_information(work, parameters["n"])
            return values[:2] if measure == "cce" else values[2:]
        if measure == "si":
            return _stochastic_interaction(work, parameters["delay"])
        raise ValueError(f"Unknown Gaussian history calculation: {measure}")
    return work.arrays.get(("gaussian_history_result", measure, tuple(sorted(parameters.items()))), compute)


def embedding_diagnostics(data, kmax, taumax):
    selection, score, valid = _auto_embedding(moment_workspace(data), kmax, taumax)
    return {"history_length": selection[:, 0].copy(), "history_spacing": selection[:, 1].copy(),
            "embedding_score": score.copy(), "valid_embedding": valid.copy()}
