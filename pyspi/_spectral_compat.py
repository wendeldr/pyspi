"""NumPy 2 compatibility for the pinned spectral-connectivity 1.1.2.

That release uses ``np.in1d`` in canonical coherence and group delay. Give
only its connectivity module a NumPy namespace with the equivalent operation;
do not change NumPy itself, installed dependency files, estimators, or spectra.
The dependency is pinned because newer releases change frequency conventions.
Remove this adapter when that numerical migration has been validated.
"""

import numpy as np
import spectral_connectivity as spectral_connectivity
import spectral_connectivity.connectivity as _connectivity


class _ConnectivityNumpy:
    @staticmethod
    def in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
        # in1d flattens its first input; isin otherwise preserves its shape.
        return np.isin(
            np.asarray(ar1).ravel(), ar2,
            assume_unique=assume_unique, invert=invert, kind=kind,
        )

    def __getattr__(self, name):
        return getattr(np, name)


_connectivity.np = _ConnectivityNumpy()
