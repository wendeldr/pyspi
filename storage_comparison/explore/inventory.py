"""Read-only cohort inventory: per patient channels, windows, metric count, bytes, allocated bytes, files."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
SRC = Path('/media/dan/Data/outputs/calculated_seeg_patient_data')
def one(store):
    g = store / 'connectivity' / '0.5s'
    shapes = []
    for p in g.iterdir():
        zj = p / 'zarr.json'
        if zj.is_file():
            m = json.loads(zj.read_text())
            if len(m.get('shape', [])) == 3:
                shapes.append(m['shape'])
    files = size = alloc = 0
    cfiles = csize = calloc = 0
    for root, dirs, fs in os.walk(store):
        conn = '/connectivity/' in root + '/'
        for f in fs:
            st = os.stat(os.path.join(root, f))
            files += 1; size += st.st_size; alloc += st.st_blocks * 512
            if conn:
                cfiles += 1; csize += st.st_size; calloc += st.st_blocks * 512
    n = shapes[0][0] if shapes else 0
    w = shapes[0][2] if shapes else 0
    return dict(patient=store.name, n=n, windows=w, metrics=len(shapes), uniform=all(s == shapes[0] for s in shapes),
                files=files, bytes=size, alloc=alloc, conn_files=cfiles, conn_bytes=csize, conn_alloc=calloc,
                raw_f64=sum(s[0]*s[1]*s[2]*8 for s in shapes))
stores = sorted(p for p in SRC.iterdir() if p.suffix == '.zarr')
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(one, stores))
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
tot = {k: sum(r[k] for r in rows) for k in ('files', 'bytes', 'alloc', 'conn_files', 'conn_bytes', 'conn_alloc', 'raw_f64')}
print(len(rows), 'stores', {k: (f'{v/2**30:.1f} GiB' if 'bytes' in k or 'alloc' in k or 'raw' in k else v) for k, v in tot.items()})
print('compression ratio raw/conn_bytes', tot['raw_f64'] / tot['conn_bytes'])
print('n range', min(r['n'] for r in rows), max(r['n'] for r in rows), 'windows', min(r['windows'] for r in rows), max(r['windows'] for r in rows), 'nonuniform', [r['patient'] for r in rows if not r['uniform']])
for r in rows:
    if r['patient'] in ('004.zarr', '105.zarr', '111.zarr'):
        print(r)
