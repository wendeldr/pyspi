"""Summarise screen.jsonl: totals per combination, per family, and per patient."""
import json, sys, collections
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
rows = [json.loads(l) for l in (run / 'screen.jsonl').read_text().splitlines()]
key = lambda r: (r['precision'], r['structure'], r['layout'], r['codec'])
agg = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    a = agg[key(r)]
    a['raw'] += r['raw_bytes']; a['bytes'] += r['bytes']; a['enc'] += r['enc_s']; a['dec'] += r['dec_s']
    a['pack'] += r['pack_s']; a['unpack'] += r['unpack_s']; a['n'] += 1
base = agg[('f64', 'full', 'wm', 'zstd3')]['bytes']
out = []
for k, a in agg.items():
    out.append(dict(precision=k[0], structure=k[1], layout=k[2], codec=k[3], arrays=a['n'], ratio=a['raw'] / a['bytes'],
                    vs_full_zstd3=a['bytes'] / base, enc_MBps=a['raw'] / a['enc'] / 1e6, dec_MBps=a['raw'] / a['dec'] / 1e6,
                    pack_MBps=a['raw'] / a['pack'] / 1e6, unpack_MBps=a['raw'] / a['unpack'] / 1e6))
out.sort(key=lambda x: (x['precision'], x['structure'], -x['ratio']))
print(f"{'prec':4s} {'struct':6s} {'lay':3s} {'codec':16s} {'ratio':>6s} {'vs_zstd3full':>12s} {'enc MB/s':>9s} {'dec MB/s':>9s}")
for o in out:
    print(f"{o['precision']:4s} {o['structure']:6s} {o['layout']:3s} {o['codec']:16s} {o['ratio']:6.3f} {o['vs_full_zstd3']:12.3f} {o['enc_MBps']:9.0f} {o['dec_MBps']:9.0f}")
(run / 'screen_summary.json').write_text(json.dumps(out, indent=1))
