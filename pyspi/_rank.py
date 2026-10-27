"""Coefficient-only rank correlations with reusable channel preprocessing."""

from functools import cached_property

import numpy as np
from scipy import stats
from scipy.stats._stats import _kendall_dis


class SpearmanWorkspace:
    def __init__(self, values):
        self.values = values
        self.pairs = {}

    @cached_property
    def ranks(self):
        return stats.rankdata(self.values, axis=1, method="average")

    @staticmethod
    def _correlate(ranks):
        if ranks.shape[1] < 2:
            return np.full((ranks.shape[0], ranks.shape[0]), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.atleast_2d(np.corrcoef(ranks))
        return (result + result.T) * 0.5

    @cached_property
    def matrix(self):
        return self._correlate(self.ranks)

    def all_pairs(self):
        return self.matrix

    def pair(self, i, j):
        if "matrix" in self.__dict__:
            return self.matrix[i, j]
        key = tuple(sorted((i, j)))
        if key not in self.pairs:
            self.pairs[key] = self._correlate(self.ranks[list(key)])[0, 1]
        return self.pairs[key]


class KendallWorkspace:
    """Tau-b, counting discordances with SciPy's compiled Fenwick tree.

    The pinned SciPy kernel needs positive integer ranks and x sorted into tie
    groups. It delays insertion of a whole x-tie group, so y need not be sorted
    within that group. Only joint tie counting needs a per-pair secondary sort.
    See scipy/stats/_stats.pyx::_kendall_dis (SciPy 1.17.1).
    """

    def __init__(self, values):
        p, n = values.shape
        self.total = n * (n - 1) // 2
        self.order = np.argsort(values, axis=1, kind="stable")
        sorted_values = np.take_along_axis(values, self.order, axis=1)
        changes = np.ones((p, n), dtype=bool)
        changes[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
        self.sorted_ranks = changes.cumsum(axis=1, dtype=np.intp)
        self.ranks = np.empty((p, n), dtype=np.intp)
        np.put_along_axis(self.ranks, self.order, self.sorted_ranks, axis=1)
        self.ties = []
        for row in changes:
            counts = np.diff(np.r_[np.flatnonzero(row), n]).astype(np.int64)
            self.ties.append(int(np.sum(counts * (counts - 1) // 2)))
        self.valid = ~np.isnan(values).any(axis=1) & (np.array(self.ties) < self.total)
        self.matrix = np.full((p, p), np.nan)
        self.computed = np.zeros((p, p), dtype=bool)

    def pair(self, i, j):
        if not self.computed[i, j]:
            value = self._coefficient(i, j)
            self.matrix[i, j] = self.matrix[j, i] = value
            self.computed[i, j] = self.computed[j, i] = True
        return self.matrix[i, j]

    def _coefficient(self, i, j):
        if not (self.valid[i] and self.valid[j]):
            return np.nan
        x = self.sorted_ranks[i]
        y = self.ranks[j, self.order[i]]
        discordant = int(_kendall_dis(x, y))
        joint_ties = 0
        if self.ties[i] and self.ties[j]:
            # Equal (x, y) pairs must be adjacent for the run-length count.
            order = np.lexsort((y, x))
            xx, yy = x[order], y[order]
            boundaries = np.r_[True, (xx[1:] != xx[:-1]) | (yy[1:] != yy[:-1]), True]
            counts = np.diff(np.flatnonzero(boundaries)).astype(np.int64)
            joint_ties = int(np.sum(counts * (counts - 1) // 2))
        numerator = self.total - self.ties[i] - self.ties[j] + joint_ties - 2 * discordant
        tau = numerator / np.sqrt(self.total - self.ties[i]) / np.sqrt(self.total - self.ties[j])
        return np.clip(tau, -1., 1.)

    def all_pairs(self):
        for j in range(self.matrix.shape[0]):
            for i in range(j):
                self.pair(i, j)
        return self.matrix
