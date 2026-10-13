"""Read-only probe of per-metric value properties that bound achievable lossless compression."""
import json, sys, time
from pathlib import Path
import numpy as np
import zarr, zstandard

SRC = Path('/media/dan/Data/outputs/calculated_seeg_patient_data')
patient = sys.argv[1]
nwin = int(sys.argv[2]) if len(sys.argv) > 2 else 64
group = SRC / f'{patient}.zarr' / 'connectivity' / '0.5s'
names = sorted(p.name for p in group.iterdir() if (p / 'zarr.json').is_file()
               and len(json.loads((p / 'zarr.json').read_text()).get('shape', [])) == 3)
cache = {}
rows = []
cz = zstandard.ZstdCompressor(level=3)
for name in names:
    a = zarr.open_array(str(group / name), mode='r')
    n, _, w = a.shape
    s = (w - nwin) // 2
    d = np.ascontiguousarray(a[:, :, s:s + nwin])
    cache[name] = d
    bits = d.view(np.uint64)
    finite = np.isfinite(d)
    f32 = d.astype(np.float32).astype(np.float64)
    f32_exact = bool(np.array_equal(f32.view(np.uint64)[finite], bits[finite]))
    # trailing zero mantissa bits (min over finite nonzero)
    nz = finite & (d != 0)
    m = bits[nz] & np.uint64((1 << 52) - 1)
    tz = 52
    if m.size:
        low = m & (~m + np.uint64(1))  # lowest set bit
        with np.errstate(divide='ignore'):
            tzs = np.where(m == 0, 52, np.log2(low.astype(np.float64)).astype(int))
        tz_med = float(np.median(tzs))
        tz_min = int(tzs.min())
    else:
        tz_med = tz_min = 52
    sym = bool(np.array_equal(bits, bits.swapaxes(0, 1)))
    offdiag = ~np.eye(n, dtype=bool)
    diag = d[np.arange(n), np.arange(n), :]
    uniq = np.unique(bits[finite]).size if finite.any() else 0
    comp = len(cz.compress(np.moveaxis(d, 2, 0).tobytes()))
    rows.append(dict(metric=name, n=n, w=nwin, symmetric=sym, f32_exact=f32_exact,
                     tz_min=tz_min, tz_med=tz_med, nan_frac=float(np.isnan(d).mean()),
                     inf=int(np.isinf(d).sum()), negzero=int(((bits == np.uint64(1 << 63))).sum()),
                     diag_all_nan=bool(np.isnan(diag).all()), diag_const=bool(np.unique(diag.view(np.uint64)).size <= 1),
                     unique_frac=uniq / max(1, finite.sum()), zstd3_ratio=d.nbytes / comp,
                     vmin=float(np.nanmin(d)) if finite.any() else None, vmax=float(np.nanmax(d)) if finite.any() else None))
# derived squares
derived = []
for name, d in cache.items():
    if '-sq' in name:
        base = name.replace('-sq', '', 1)
        if base in cache:
            sq = np.square(cache[base])
            derived.append(dict(sq=name, base=base, exact=bool(np.array_equal(sq.view(np.uint64), d.view(np.uint64))),
                                exact_finite=bool(np.array_equal(sq[np.isfinite(d)], d[np.isfinite(d)]))))
out = Path(__file__).parent / f'properties_{patient}_{nwin}.json'
out.write_text(json.dumps(dict(rows=rows, derived=derived), indent=1))
import collections
print('metrics', len(rows), 'symmetric', sum(r['symmetric'] for r in rows), 'f32_exact', sum(r['f32_exact'] for r in rows))
print('derived squares exact', [(x['sq'], x['exact']) for x in derived])
fam = collections.defaultdict(list)
for r in rows:
    fam[r['metric'].split('_')[0]].append(r)
for k, v in fam.items():
    print(f"{k:14s} n={len(v):3d} sym={sum(r['symmetric'] for r in v):3d} f32={sum(r['f32_exact'] for r in v):3d} "
          f"tzmed={np.median([r['tz_med'] for r in v]):5.1f} nan={np.mean([r['nan_frac'] for r in v]):.3f} "
          f"uniq={np.mean([r['unique_frac'] for r in v]):.3f} zr={np.mean([r['zstd3_ratio'] for r in v]):.2f} "
          f"diagnan={sum(r['diag_all_nan'] for r in v)} inf={sum(r['inf'] for r in v)} range=[{min((r['vmin'] or 0) for r in v):.3g},{max((r['vmax'] or 0) for r in v):.3g}]")
