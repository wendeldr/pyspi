"""Compare pyspi directed/undirected labels with bitwise symmetry, antisymmetry, and near-symmetry."""
import json, sys
from pathlib import Path
import numpy as np, zarr
SRC = Path('/media/dan/Data/outputs/calculated_seeg_patient_data')
patient, nwin = sys.argv[1], int(sys.argv[2])
g = SRC / f'{patient}.zarr/connectivity/0.5s'
out = []
SIGN = np.uint64(1 << 63)
for p in sorted(g.iterdir()):
    zj = p / 'zarr.json'
    if not zj.is_file(): continue
    meta = json.loads(zj.read_text())
    if len(meta.get('shape', [])) != 3: continue
    labels = meta['attributes'].get('pyspi_labels', [])
    a = zarr.open_array(str(p), mode='r'); n, _, w = a.shape; s = (w - nwin)//2
    d = np.ascontiguousarray(a[:, :, s:s+nwin]); b = d.view(np.uint64); bt = b.swapaxes(0, 1)
    iu = np.triu_indices(n, 1)
    u, l = b[iu[0], iu[1]], bt[iu[0], iu[1]]  # upper vs mirrored lower
    both_nan = np.isnan(d[iu[0], iu[1]]) & np.isnan(d.swapaxes(0,1)[iu[0], iu[1]])
    x = u ^ l
    sym = bool((x == 0).all()); anti = bool(((x == 0) | (x == SIGN)).all()) and not sym
    anti_strict = bool((x == SIGN)[~both_nan].mean()) if (~both_nan).any() else 0
    eq_frac = float((x == 0).mean())
    # near-symmetry: how many high bits of xor are zero (log2 of ulp diff)
    with np.errstate(divide='ignore'):
        lz = 64 - np.where(x == 0, 0, np.floor(np.log2(x.astype(np.float64))) + 1)
    du = d[iu[0], iu[1]]; dl = d.swapaxes(0,1)[iu[0], iu[1]]
    fin = np.isfinite(du) & np.isfinite(dl)
    rel = np.abs(du[fin]-dl[fin]) / np.maximum(np.abs(du[fin]), 1e-300) if fin.any() else np.array([0.])
    out.append(dict(metric=p.name, directed='directed' in labels, undirected='undirected' in labels,
                    sym=sym, anti=anti, sign_flip_frac=anti_strict, eq_frac=eq_frac,
                    xor_lz_median=float(np.median(lz)), rel_asym_median=float(np.median(rel)), rel_asym_max=float(rel.max())))
Path(__file__).with_name(f'symmetry_{patient}_{nwin}.json').write_text(json.dumps(out, indent=1))
import collections
fam = collections.defaultdict(list)
for r in out: fam[r['metric'].split('_')[0]].append(r)
print(f"{'family':14s} n undir dir sym anti eq_frac  xorLZmed relasym_med")
for k, v in fam.items():
    print(f"{k:14s} {len(v):2d} {sum(r['undirected'] for r in v):4d} {sum(r['directed'] for r in v):3d} {sum(r['sym'] for r in v):3d} {sum(r['anti'] for r in v):4d} "
          f"{np.mean([r['eq_frac'] for r in v]):7.3f} {np.median([r['xor_lz_median'] for r in v]):6.1f} {np.median([r['rel_asym_median'] for r in v]):10.2e}")
mism = [r['metric'] for r in out if r['undirected'] and not r['sym']]
print('undirected but not bitwise symmetric:', mism)
print('directed but symmetric:', [r['metric'] for r in out if r['directed'] and r['sym']])
print('antisymmetric:', [r['metric'] for r in out if r['anti']])
print('neither label:', [r['metric'] for r in out if not r['directed'] and not r['undirected']])
