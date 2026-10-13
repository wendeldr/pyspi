"""Explain arrays whose structure deviates from exact symmetry/antisymmetry (read-only)."""
import json, sys
from pathlib import Path
import numpy as np, zarr
SRC = Path('/media/dan/Data/outputs/calculated_seeg_patient_data')
SIGN = np.uint64(1 << 63)
patient, nwin = sys.argv[1], int(sys.argv[2])
g = SRC / f'{patient}.zarr/connectivity/0.5s'
fams = ('prec', 'pdist_euclidean', 'pec', 'phase', 'psi', 'gd', 'pli', 'wpli')
for p in sorted(g.iterdir()):
    if not p.name.startswith(fams) or not (p / 'zarr.json').is_file(): continue
    a = zarr.open_array(str(p), mode='r'); n, _, w = a.shape; s = (w - nwin) // 2
    d = np.ascontiguousarray(a[:, :, s:s + nwin])
    iu = np.triu_indices(n, 1)
    U, L = d[iu[0], iu[1]], d.swapaxes(0, 1)[iu[0], iu[1]]
    ub, lb = U.view(np.uint64), L.view(np.uint64)
    x = ub ^ lb
    nanU, nanL = np.isnan(U), np.isnan(L)
    fin = np.isfinite(U) & np.isfinite(L)
    sym_eq = (x == 0); anti_eq = (x == SIGN)
    # values-level checks
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_sym = np.abs(U - L)[fin] / np.maximum(np.abs(U[fin]), np.abs(L[fin]))
        rel_anti = np.abs(U + L)[fin] / np.maximum(np.abs(U[fin]), np.abs(L[fin]))
    rs = np.nan_to_num(rel_sym, nan=0.0); ra = np.nan_to_num(rel_anti, nan=0.0)
    ulp = np.abs(ub[fin].astype(np.int64) - lb[fin].astype(np.int64)) if fin.any() else np.array([0])
    other = ~(sym_eq | anti_eq)
    print(f"{p.name[:46]:46s} nanU={nanU.mean():.3f} nanmismatch={(nanU != nanL).sum():5d} "
          f"sym={sym_eq.mean():.3f} anti={anti_eq.mean():.3f} other={other.mean():.4f} "
          f"maxrel_sym={rs.max() if rs.size else 0:.2e} maxrel_anti={ra.max() if ra.size else 0:.2e} "
          f"zeros={(U == 0).sum()}")
