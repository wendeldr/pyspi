"""Named homogeneous metric groups with shared continuous preparation."""

from dataclasses import dataclass, replace
from pathlib import Path

from ._config import (Configuration, WorkflowError, configuration_from_mapping,
                      fields, positive_number, read_yaml, safe_id)
from ._preflight import preflight
from ._shared_prepare import SharedPreparation, preparation_key


@dataclass(frozen=True)
class Experiment:
    source: Path
    groups: tuple[tuple[str, Configuration], ...]
    preparation_cache_mib: float


def load_experiment(path):
    path = Path(path).resolve()
    document = read_yaml(path)
    if not isinstance(document, dict):
        raise WorkflowError('experiment must be a mapping')
    groups = document.get('groups')
    if not isinstance(groups, dict) or not groups:
        raise WorkflowError('experiment.groups must be a nonempty mapping of named metric groups')
    sharing = fields(document.get('sharing', {}), {'preparation_cache_mib'}, 'sharing')
    cache = sharing.get('preparation_cache_mib', 8192)
    if cache != 0 or isinstance(cache, bool):
        cache = float(positive_number(cache, 'sharing.preparation_cache_mib'))
    defaults = {key: value for key, value in document.items() if key not in ('groups', 'sharing')}
    result = []
    for name, overrides in groups.items():
        safe_id(name, 'group name')
        fields(overrides, {'metrics', 'reference', 'preprocessing', 'output_dtype',
                          'output_dtype_overrides', 'execution'}, f'groups.{name}')
        combined = {**defaults, **overrides}
        for section in ('preprocessing', 'execution', 'output_dtype_overrides'):
            if section in overrides:
                # Validate mappings before merging to retain useful typo errors.
                if not isinstance(overrides[section], dict) or not isinstance(defaults.get(section, {}), dict):
                    raise WorkflowError(f'{section} must be a mapping')
                combined[section] = {**defaults.get(section, {}), **overrides[section]}
        combined['run_name'] = f"{defaults.get('run_name', 'experiment')}-{name}"
        result.append((name, configuration_from_mapping(combined, path)))
    return Experiment(path, tuple(result), float(cache))


def _reports(experiment):
    return [(name, preflight(cfg)) for name, cfg in experiment.groups]


def preflight_experiment(config):
    experiment = config if isinstance(config, Experiment) else load_experiment(config)
    reports = _reports(experiment)
    return {'schema': 'pyspi-experiment-preflight/1', 'configuration': str(experiment.source),
            'preparation_cache_mib': experiment.preparation_cache_mib,
            'groups': {name: report.as_dict() for name, report in reports},
            'output_count_per_recording': sum(len(report.metrics) for _, report in reports)}


def _output_path(cfg, recording, entry):
    if cfg.dataset_directory is not None:
        from ._dataset import dataset_address
        return dataset_address(cfg, recording, entry)[1]
    if cfg.output_directory is None:
        raise WorkflowError('output_directory is required for execution')
    return cfg.output_directory/recording.patient_id/'calculations'/cfg.run_name


def run_experiment(config, *, resume=False, retry_failures=False, max_windows=None):
    """Run one recording's groups before moving to the next recording.

    Separate output contracts retain each group's reference, preprocessing and
    precision. Cache scope ends at the recording boundary and is never reused
    across invocations. Missing later groups are created during cohort resume.
    """
    from ._parallel import numerical_threads
    from ._run import _run
    from ._bands import required_bands
    experiment = config if isinstance(config, Experiment) else load_experiment(config)
    reports = _reports(experiment)  # validate every group before any output
    jobs, unions, paths = {}, {}, []
    for name, report in reports:
        cfg = report.configuration
        for recording, entry in zip(cfg.recordings, report.recordings):
            job = replace(cfg, recordings=(recording,), reference=entry['reference'])
            identity = (recording.patient_id, recording.recording_id, str(recording.path))
            path = _output_path(job, recording, entry)
            paths.append(path.resolve())
            jobs.setdefault(identity, []).append((name, job, path))
            key = preparation_key(recording, job)
            unions.setdefault(key, set()).update(required_bands(report.metrics, job))
    if len(set(paths)) != len(paths):
        raise WorkflowError('experiment groups map to duplicate output paths')
    if retry_failures and not resume:
        raise WorkflowError('retry_failures requires explicit resume')
    if resume and not any((path/'run.json').is_file() for path in paths):
        raise WorkflowError('no run to resume among experiment groups')
    # Fail before creating any group if a non-resume invocation would overwrite.
    if not resume and any((path/'run.json').is_file() for path in paths):
        raise WorkflowError('experiment output already exists; use --resume or a new run_name')
    unions = {key: tuple(sorted(value)) for key, value in unions.items()}
    results, stats = [], []
    for identity, group_jobs in jobs.items():
        cache = SharedPreparation(max_bytes=int(experiment.preparation_cache_mib*2**20), band_unions=unions)
        try:
            for name, cfg, path in group_jobs:
                exists = (path/'run.json').is_file()
                with numerical_threads(cfg.execution['threads_per_worker']):
                    output = _run(cfg, resume=resume and exists, retry_failures=retry_failures and exists,
                                  max_windows=max_windows, preparation=cache)
                results.extend({'group': name, 'patient_id': identity[0], 'recording_id': identity[1],
                                'path': str(p)} for p in output)
            stats.append({'patient_id': identity[0], 'recording_id': identity[1], **cache.stats})
        finally:
            cache.close()
    return {'schema': 'pyspi-experiment-run/1', 'configuration': str(experiment.source),
            'runs': results, 'preparation_cache': stats}
