"""Bounded real EEG run into copied metadata; original stores are read-only."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
import zarr
from seeg_dataset import open_run
from pyspi.workflow import load_config, preflight, run
from pyspi.workflow._metrics import resolve_metrics
from pyspi.workflow._parallel import numerical_threads
from pyspi.workflow import _run


def hashes(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}


def main():
    scratch=ROOT/'.candidate-scratch/precision'
    source=Path('/home/dan/data/IEEG/connectivity/001.zarr')
    before=hashes(source/'patient_info')
    specs=resolve_metrics(ROOT/'pyspi/workflow/reviewed.yaml',sampling_rate_hz=2048)
    chosen=[]
    for cls in ['Covariance','SpearmanR','TransferEntropy','CoherenceMagnitude','PhaseSlopeIndex','PowerEnvelopeCorrelation']:
        choices=[m for m in specs if m.class_name==cls]
        if cls=='Covariance':choices=[m for m in choices if m.parameters.get('estimator')=='EmpiricalCovariance']
        if cls in ('CoherenceMagnitude','PhaseSlopeIndex'):choices=[m for m in choices if m.parameters.get('fmin')==30 and m.parameters.get('fmax')==70]
        chosen.append(choices[0])
    with tempfile.TemporaryDirectory(prefix='eeg-validation-',dir=scratch) as temp:
        temp=Path(temp);patient=temp/'dataset/001.zarr'
        zarr.open_group(patient,mode='w')
        shutil.copytree(source/'patient_info',patient/'patient_info')
        document={}
        for m in chosen:
            document.setdefault(m.module,{}).setdefault(m.class_name,{'configs':[],'dependencies':list(m.dependencies)})['configs'].append(m.parameters)
        metrics=temp/'metrics.yaml';metrics.write_text(yaml.safe_dump(document,sort_keys=False))
        template=load_config(ROOT/'examples/workflow/additional_candidates_car.yaml')
        cfg=replace(template, dataset_directory=temp/'dataset',output_directory=temp/'dataset',
                    metric_config=metrics,run_name='validation',output_dtype='float32',output_dtype_overrides={},
                    execution={**template.execution,'workers':1,'scratch_directory':str(scratch)},output_batch_mib=1)
        report=preflight(cfg)
        expected={m.identifier:[] for m in chosen}
        original=_run._store_values
        def capture(buffers,event,values,valid,n,count,slot,dtypes):
            assert values is not None, event.identifier
            matrix=np.full((n,n),np.nan,dtype=dtypes[event.identifier])
            with np.errstate(over='ignore',under='ignore'):
                matrix[np.ix_(valid,valid)]=values.astype(dtypes[event.identifier])
            expected[event.identifier].append(matrix)
            return original(buffers,event,values,valid,n,count,slot,dtypes)
        _run._store_values=capture
        try:
            started=perf_counter();[path]=run(cfg,max_windows=2);elapsed=perf_counter()-started
        finally:
            _run._store_values=original
        reader=open_run(path);checks={}
        for name,windows in expected.items():
            want=np.stack(windows,axis=2);got,available=reader.load(name,return_availability=True)
            np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32))
            assert available.all()
            with np.errstate(invalid='ignore'):
                np.testing.assert_array_equal(np.nanmean(got,axis=2,dtype=np.float64),np.nanmean(want,axis=2,dtype=np.float64))
            checks[name]={'shape':list(got.shape),'dtype':str(got.dtype),'exact_converted_bits':True,
                          'statuses':reader.statuses(name).tolist()}
        assert before==hashes(patient/'patient_info')==hashes(source/'patient_info')
        output={'patient':'001','channels':len(reader.recording['channels']),'windows':2,'target_rate_hz':2048,
                'metadata_order_matches':reader.recording['channels']==[c['name'] for c in report.recordings[0]['channels']],
                'original_and_copy_metadata_unchanged':True,'raw_per_window_output':True,
                'elapsed_seconds':elapsed,'metrics':checks,'path_relative_to_patient':str(path.relative_to(patient)),
                'limits':'Bounded 2-window 6-metric integration, including full-record preparation; not a cohort timing projection.'}
        (ROOT/'tests/precision_review/eeg_integration.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':
    with numerical_threads(1):main()
