"""Refresh only approved information-definition changes and catalogue fields.

Independent history/ordinal-reference tests must pass first. Unchanged Gaussian
formulas and retained non-Gaussian calculations keep their numerical goldens.
"""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]

import numpy as np
from regression import generate_goldens as generator, harness


RENAMED={
    'te_symbolic_k-1_kt-1_l-1_lt-1':'te_symbolic_m-3_tau-4_h-4',
    'te_symbolic_k-10_kt-1_l-1_lt-1':'te_symbolic_m-3_tau-16_h-16',
}
CHANGED=set(RENAMED.values()) | {'di_gaussian','di_kozachenko','di_kernel_W-0.5',
     'xme_gaussian_k10','xme_kozachenko_k10','xme_kernel_W-0.5_k10'}


def main():
    harness.ensure_jvm()
    entries,by_id=harness.catalogue()
    sources=['pyspi/_gaussian.py','pyspi/_gaussian_history.py','pyspi/_information_history.py',
             'pyspi/_symbolic.py','pyspi/statistics/infotheory.py','pyspi/config.yaml',
             'tests/unit/test_gaussian_history.py','tests/unit/test_information_history.py',
             'tests/unit/test_symbolic_transfer.py',str(Path(__file__).relative_to(ROOT))]
    provenance=dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Approved standard DI sum, exact XME k=10 lag count, genuine m=3 ordinal TE. Gaussian Java dependency and differential-entropy labels corrected.',
        validation='39 Gaussian-history and 20 retained-estimator cases passed; 9 independent ordinal-count/edge cases passed before refresh. Unchanged formulas retain their numerical arrays.',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    prepared={}
    for size in ('small','full','preproc'):
        old_meta=harness.load_meta(size); old_values=harness.load_values(size)
        meta=copy.deepcopy(old_meta); values={key:value.copy() for key,value in old_values.items()}
        meta['calculator_order']=[RENAMED.get(key,key) for key in meta['calculator_order']]
        assert meta['calculator_order']==[entry['identifier'] for entry in entries]
        numerical=[]; metadata=[]; removed=[]
        if size!='preproc':
            for key in RENAMED:
                assert key in values
                del values[key]; del meta['spis'][key]; removed.append(key)
            raw=harness.load_raw(meta['dataset'])
            for key in sorted(CHANGED):
                value,record,problem=generator.run_one(by_id[key],raw,meta['dataset'],1)
                assert problem is None and record['status']=='ok',(key,record,problem)
                values[key]=value
                meta['spis'][key]={**generator.catalogue_fields(by_id[key]),**record}
                numerical.append(key)
        for key,record in meta['spis'].items():
            fields=generator.catalogue_fields(by_id[key])
            if any(record.get(field)!=value for field,value in fields.items()):
                assert record['module']=='.statistics.infotheory',key
                for field,value in fields.items():
                    if record.get(field)!=value:
                        assert field=='deps' and ('gaussian' in key or 'symbolic' in key) or (
                            field=='labels' and key.startswith(('xme_','cce_','te_symbolic_'))),(key,field)
                record.update(fields); metadata.append(key)
        unrelated=set(old_values)-CHANGED-set(RENAMED)
        assert all(values[key].tobytes()==old_values[key].tobytes() for key in unrelated)
        assert all(meta['spis'][key]==record for key,record in old_meta['spis'].items()
                   if key not in CHANGED|set(RENAMED)|set(metadata))
        meta.setdefault('partial_regenerations',[]).append(dict(**provenance,
            updated=numerical,metadata_only=metadata,removed=removed,
            unrelated_arrays_preserved=len(unrelated)))
        prepared[size]=(values,meta)
    for size,(values,meta) in prepared.items():
        destination=harness.golden_dir(size)
        if size!='preproc':
            np.savez_compressed(destination/'values.npz',**values)
        (destination/'meta.json').write_text(json.dumps(meta,indent=1,sort_keys=True)+'\n')
        change=meta['partial_regenerations'][-1]
        print(size,'numeric:',change['updated'],'metadata:',change['metadata_only'],flush=True)


if __name__=='__main__':
    main()
