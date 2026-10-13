"""Bounded, read-only comparison of existing connectivity results."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import re
import resource
import statistics
import sys
import threading
import time
import uuid
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numcodecs.blosc
import numpy as np
import psutil
import zarr

from formats import VARIANTS, filesystem_size, read_store, same_bits, symmetric_bits, write_store

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/media/dan/Data/outputs/calculated_seeg_patient_data')


class MemorySampler:
    """Sample absolute process RSS; allocator retention is documented in reports."""

    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def metric_directories(group: Path):
    result = []
    for directory in sorted(group.iterdir()):
        metadata = directory / 'zarr.json'
        if metadata.is_file():
            meta = json.loads(metadata.read_text())
            if meta.get('node_type') == 'array' and len(meta.get('shape', [])) == 3:
                result.append((directory, meta))
    return result


def window_slice(total: int, count: int, start: str):
    count = min(count, total)
    if start == 'middle':
        first = (total - count) // 2
    elif start == 'end':
        first = total - count
    elif start == 'start':
        first = 0
    else:
        first = int(start)
    if first < 0 or first + count > total:
        raise ValueError(f'Window selection {first}:{first + count} exceeds {total} windows')
    return first, first + count


def existing_sample_size(directory: Path, metadata: dict, start: int, stop: int):
    chunks = metadata.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape')
    key = metadata.get('chunk_key_encoding', {})
    if chunks != [*metadata['shape'][:2], 1] or key.get('name') != 'default':
        return None
    separator = key.get('configuration', {}).get('separator', '/')
    files = [directory / 'zarr.json']
    files.extend(directory / separator.join(('c', '0', '0', str(w))) for w in range(start, stop))
    present = [p for p in files if p.is_file()]
    return {'stored_bytes': sum(p.stat().st_size for p in present), 'files': len(present)}


def mean(data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(data, axis=2)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['variant']].append(row)
    result = {}
    for variant, group in sorted(groups.items()):
        total_keys = ('stored_bytes', 'allocated_bytes', 'data_bytes', 'metadata_bytes',
                      'files', 'directories', 'write_seconds', 'read_seconds',
                      'subset_read_seconds', 'mean_seconds', 'raw_bytes')
        item = {key: sum(row[key] for row in group) for key in total_keys}
        item.update(metrics=len(group), packed=sum(row['mode'] == 'upper' for row in group),
                    rss_peak_bytes=max(row['rss_peak_bytes'] for row in group),
                    all_bitwise_valid=all(row['bitwise_valid'] for row in group))
        result[variant] = item
    baseline = result.get('original_zstd', {}).get('stored_bytes')
    if baseline:
        for item in result.values():
            item['fraction_of_baseline'] = item['stored_bytes'] / baseline
            item['reduction_percent'] = 100 * (1 - item['fraction_of_baseline'])
    return result


def write_reports(report_dir: Path, rows: list, sources: list, environment: dict, elapsed: float):
    summary = {
        'complete': True, 'elapsed_seconds': elapsed, 'environment': environment,
        'source_sample': {
            'metric_samples': len(sources),
            'raw_bytes': sum(row['raw_bytes'] for row in sources),
            'source_read_seconds': sum(row['source_read_seconds'] for row in sources),
            'exactly_symmetric': sum(row['exactly_symmetric'] for row in sources),
            'existing_selected_bytes': sum((row['existing_sample'] or {}).get('stored_bytes', 0) for row in sources),
        },
        'by_variant': aggregate(rows),
        'by_patient': {patient: aggregate([r for r in rows if r['patient'] == patient])
                       for patient in sorted({r['patient'] for r in rows})},
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary['process_peak_rss_bytes'] = peak if sys.platform == 'darwin' else peak * 1024
    (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (report_dir / 'sources.json').write_text(json.dumps(sources, indent=2))
    with (report_dir / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Lossless storage comparison results', '',
        f"Measured {len(sources)} patient/metric samples in {elapsed:.1f} seconds.", '',
        '| Representation | Stored MiB | Reduction | Files | Write s | Cached read s | Packed samples |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for variant, item in summary['by_variant'].items():
        reduction = f"{item['reduction_percent']:.1f}%" if 'reduction_percent' in item else 'n/a'
        lines.append(f"| {variant} | {item['stored_bytes'] / 2**20:.2f} | {reduction} | {item['files']} | "
                     f"{item['write_seconds']:.2f} | {item['read_seconds']:.2f} | {item['packed']} |")
    lines.extend([
        '', 'All completed results passed bit-for-bit reconstruction, subset selection, and time-average checks.', '',
        'Times sum individual metric operations. Reads are medians after an explicit warm-up. '
        'Writes are buffered and do not include fsync. They are not cold disk throughput or durable write measurements.', '',
        'The baseline recreates the current layout and codec on the same sample. '
        'Actual selected source chunk sizes are also recorded in sources.json; metadata and codec versions can differ.', '',
        'Per-operation RSS is sampled every 10 ms in a shared process and includes loaded source data and '
        'memory retained by allocators. It is not an isolated estimate of each format\'s memory requirement. '
        'The process-wide OS peak is recorded separately.', '',
        'The sample does not measure a full-cohort conversion or establish identical savings for other windows/patients.',
    ])
    (report_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--patients', nargs='+', default=['004', '105', '111'])
    parser.add_argument('--metrics', nargs='+', help='Exact saved metric directory names; default all')
    parser.add_argument('--windows', type=int, default=16)
    parser.add_argument('--start', default='middle', help='middle, start, end, or an integer source window index')
    parser.add_argument('--chunk-windows', type=int, default=8)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if min(args.windows, args.chunk_windows, args.repeats) < 1:
        parser.error('windows, chunk-windows and repeats must be positive')
    run_name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:6]
    if run_name in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', run_name):
        parser.error('run-name must be a simple directory name')
    source = args.source.resolve()
    output = HERE / 'runs' / run_name
    report_dir = HERE / 'reports' / run_name
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        parser.error('Source and output trees must be separate')
    if output.exists() or report_dir.exists():
        parser.error('Run name already exists; choose a new name to preserve previous results')
    inventory = []
    for patient in args.patients:
        if Path(patient).name != patient or patient in ('.', '..'):
            parser.error('patients must be simple store names')
        name = patient if patient.endswith('.zarr') else patient + '.zarr'
        group = source / name / 'connectivity' / '0.5s'
        entries = metric_directories(group)
        if args.metrics:
            known = {p.name for p, _ in entries}
            missing = set(args.metrics) - known
            if missing:
                parser.error(f'{name}: unknown metrics {sorted(missing)}')
            entries = [(p, m) for p, m in entries if p.name in args.metrics]
        if not entries:
            parser.error(f'{name}: no metric arrays found')
        for path, meta in entries:
            if meta['shape'][0] != meta['shape'][1] or meta['data_type'] != 'float64':
                parser.error(f'Unexpected source matrix shape or dtype: {path}')
            first, stop = window_slice(meta['shape'][2], args.windows, args.start)
            inventory.append((name, path, meta, first, stop))
    output.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    numcodecs.blosc.set_nthreads(1)
    zarr.config.set({'async.concurrency': 4, 'threading.max_workers': 4})
    environment = {
        'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
        'logical_cpus': os.cpu_count(), 'physical_cpus': psutil.cpu_count(logical=False),
        'memory_bytes': psutil.virtual_memory().total,
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'zarr', 'numcodecs', 'zstandard', 'psutil')},
        'blosc_threads': 1, 'zarr_io_concurrency': 4, 'zarr_thread_workers': 4,
        'reads': 'explicitly warmed; median of repeats', 'writes': 'buffered; no fsync',
        'rss': '10 ms sampled absolute shared-process RSS; allocator retention included',
        'source': str(source), 'data_output': str(output),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (report_dir / 'environment.json').write_text(json.dumps(environment, indent=2))
    rows, sources = [], []
    started = time.perf_counter()
    rng = random.Random(20261013)
    total = len(inventory)
    print(f'Run {run_name}: {total} patient/metric samples x {len(args.variants)} representations', flush=True)
    with (report_dir / 'progress.jsonl').open('w') as progress:
        for number, (patient, path, metadata, first, stop) in enumerate(inventory, 1):
            before = (path / 'zarr.json').read_bytes()
            read_started = time.perf_counter()
            array = zarr.open_array(str(path), mode='r')
            data = np.ascontiguousarray(array[:, :, first:stop])
            read_seconds = time.perf_counter() - read_started
            source_record = {
                'patient': patient, 'metric': path.name, 'shape': list(data.shape),
                'start': first, 'stop': stop, 'raw_bytes': data.nbytes,
                'source_read_seconds': read_seconds, 'exactly_symmetric': symmetric_bits(data),
                'nan_values': int(np.isnan(data).sum()), 'inf_values': int(np.isinf(data).sum()),
                'existing_sample': existing_sample_size(path, metadata, first, stop),
            }
            sources.append(source_record)
            context = {'patient': patient, 'metric': path.name, 'source_windows': list(range(first, stop)),
                       'channel_order': 'unchanged source indices', 'source_attributes': metadata.get('attributes', {})}
            expected_mean = mean(data)
            indices = sorted({0, data.shape[2] // 2, data.shape[2] - 1})
            order = list(args.variants)
            rng.shuffle(order)
            for name in order:
                destination = output / patient / name / path.name
                with MemorySampler() as memory:
                    mark = time.perf_counter()
                    encoded = write_store(destination, VARIANTS[name], data, context=context, chunk_windows=args.chunk_windows)
                    write_seconds = time.perf_counter() - mark
                    decoded = read_store(destination)
                    if not same_bits(data, decoded):
                        raise AssertionError(f'Full bitwise roundtrip failed: {patient}/{path.name}/{name}')
                    del decoded
                    read_times, mean_times = [], []
                    for _ in range(args.repeats):
                        mark = time.perf_counter()
                        decoded = read_store(destination)
                        read_times.append(time.perf_counter() - mark)
                        mark = time.perf_counter()
                        actual_mean = mean(decoded)
                        mean_times.append(time.perf_counter() - mark)
                        if not same_bits(actual_mean, expected_mean):
                            raise AssertionError(f'Time-average mismatch: {patient}/{path.name}/{name}')
                        del decoded
                    subset = read_store(destination, indices)
                    if not same_bits(subset, data[:, :, indices]):
                        raise AssertionError(f'Subset roundtrip failed: {patient}/{path.name}/{name}')
                    del subset
                    subset_times = []
                    for _ in range(args.repeats):
                        mark = time.perf_counter()
                        subset = read_store(destination, indices)
                        subset_times.append(time.perf_counter() - mark)
                        del subset
                row = {
                    'patient': patient, 'metric': path.name, 'variant': name,
                    'channels': data.shape[0], 'windows': data.shape[2], 'raw_bytes': data.nbytes,
                    'mode': encoded['mode'], 'bitwise_valid': True,
                    'write_seconds': write_seconds, 'read_seconds': statistics.median(read_times),
                    'mean_seconds': statistics.median(mean_times),
                    'subset_read_seconds': statistics.median(subset_times),
                    'rss_peak_bytes': memory.peak, **filesystem_size(destination),
                }
                rows.append(row)
                progress.write(json.dumps(row) + '\n')
                progress.flush()
            if before != (path / 'zarr.json').read_bytes():
                raise AssertionError(f'Source metadata changed: {path}')
            if number == 1 or number % 10 == 0 or number == total:
                print(f'{number}/{total} samples validated; elapsed {time.perf_counter() - started:.1f}s; '
                      f'{patient}/{path.name}', flush=True)
    elapsed = time.perf_counter() - started
    summary = write_reports(report_dir, rows, sources, environment, elapsed)
    for name, item in summary['by_variant'].items():
        print(f"{name:20s} {item['stored_bytes'] / 2**20:9.2f} MiB  "
              f"{item.get('reduction_percent', 0):6.1f}% smaller  {item['files']:6d} files", flush=True)
    print(f'Report: {report_dir / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
