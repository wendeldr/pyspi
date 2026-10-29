"""Measure private scratch reference construction, including final file sync."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
import types

import numpy as np
from pyspi.workflow.experiment import load_experiment
from pyspi.workflow import preflight
from pyspi.workflow._prepare import prepare_recording
from pyspi.workflow._shared_prepare import _reference_array
from pyspi.workflow._resources import ProcessTreeMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    old_source = subprocess.check_output(['git','show','d19f42d:pyspi/workflow/_shared_prepare.py'],text=True)
    old = types.ModuleType('pyspi.workflow._io_baseline')
    old.__package__ = 'pyspi.workflow'
    exec(compile(old_source, 'd19f42d:pyspi/workflow/_shared_prepare.py','exec'),old.__dict__)
    experiment = load_experiment('examples/workflow/final_experiment.yaml')
    cfg = experiment.groups[0][1]
    cfg = replace(cfg, recordings=cfg.recordings[:1], reference={'method':'none'})
    cfg = preflight(cfg).configuration
    prepared = prepare_recording(cfg.recordings[0],cfg,bands=())
    with TemporaryDirectory(prefix='pyspi-io-benchmark-',dir=cfg.execution['scratch_directory']) as directory:
        source = Path(directory)/'input.npy'
        np.save(source,prepared.data,allow_pickle=False)
        prepared.data = np.load(source,mmap_mode='r')
        results = {'before': [], 'after': []}
        memory = {'before': [], 'after': []}
        comparisons = []
        for repetition in range(3):
            output = {}
            for kind in (('before','after') if repetition%2 == 0 else ('after','before')):
                function = old._reference_array if kind == 'before' else _reference_array
                filename = f'{kind}-{repetition}.npy'
                with ProcessTreeMemory() as monitor:
                    start = time.perf_counter()
                    values = function(prepared.data, prepared.channel_names, {'method':'car'}, directory, filename)
                    seconds = time.perf_counter()-start
                results[kind].append(seconds)
                memory[kind].append(monitor.as_dict())
                output[kind] = values
                print(kind, repetition, seconds, flush=True)
            identical = all(np.array_equal(output['before'][i],output['after'][i]) for i in range(len(prepared.data)))
            assert identical
            comparisons.append(identical)
            output.clear()
            del values
            for kind in ('before','after'):(Path(directory)/f'{kind}-{repetition}.npy').unlink()
        receipt = {'shape': list(prepared.data.shape), 'seconds': results,
                   'median_seconds': {k:float(np.median(v)) for k,v in results.items()},
                   'process_memory':memory,'bit_identical_repetitions':comparisons,
                   'baseline_commit':'d19f42d', 'baseline_source_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
                   'source_sha256':hashlib.sha256(Path('pyspi/workflow/_shared_prepare.py').read_bytes()).hexdigest(),
                   'limits':'Three alternating repetitions on the same prepared patient-001 recording; one CAR reference array including final flush, no metric/output writes. Working source is disk-backed. Process memory includes interpreter/input mappings. Linux dirty page cache also affects system availability.'}
        args.out.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':main()
