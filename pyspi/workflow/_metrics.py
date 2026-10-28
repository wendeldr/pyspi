"""Resolve requested SPIs strictly, without constructing Data or computing SPIs."""

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import importlib
import inspect
import json
import random
import sys
import warnings

from ._config import WorkflowError, fields, positive_number, read_yaml
from pyspi._dependencies import spi_dependencies


@dataclass(frozen=True)
class Metric:
    identifier: str
    module: str
    class_name: str
    parameters: dict
    constructor_signature: str
    dependencies: tuple[str, ...]
    labels: tuple[str, ...]
    initialization_warnings: tuple[str, ...]

    def as_dict(self):
        return asdict(self)


def _entries(path):
    cfg = read_yaml(path)
    if not isinstance(cfg, dict) or not cfg:
        raise WorkflowError("metric configuration must be a nonempty module mapping")
    result = []
    for module, classes in cfg.items():
        if not isinstance(module, str) or not module.startswith((".statistics.", "pyspi.statistics.")):
            raise WorkflowError(f"metric module must be under pyspi.statistics: {module!r}")
        if not isinstance(classes, dict) or not classes:
            raise WorkflowError(f"{module}: expected a nonempty class mapping")
        for name, spec in classes.items():
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise WorkflowError(f"invalid SPI class name {name!r}")
            where = f"{module}.{name}"
            fields(spec, {"labels", "dependencies", "configs"}, where)
            labels = spec.get("labels")
            if labels is not None and (not isinstance(labels, list) or any(not isinstance(s, str) for s in labels)):
                raise WorkflowError(f"{where}: labels must be a list of strings or null")
            dependencies = spec.get("dependencies")
            dependencies = [] if dependencies is None else dependencies
            if not isinstance(dependencies, list) or any(not isinstance(s, str) for s in dependencies):
                raise WorkflowError(f"{where}: dependencies must be a list of strings or null")
            if set(dependencies) - {"java"}:
                raise WorkflowError(f"{where}: unknown optional dependencies {dependencies}")
            configs = spec.get("configs")
            configs = [{}] if configs is None else configs
            if not isinstance(configs, list) or not configs:
                raise WorkflowError(f"{where}: configs must be a nonempty list or null for defaults")
            for params in configs:
                if not isinstance(params, dict) or any(not isinstance(k, str) for k in params):
                    raise WorkflowError(f"{where}: each parameter configuration must be a string-keyed mapping")
                try:
                    json.dumps(params, allow_nan=False)
                except (TypeError, ValueError, RecursionError) as exc:
                    raise WorkflowError(f"{where}: parameters must be finite JSON-compatible values") from exc
                result.append((module, name, params, dependencies))
    return result


def _validate_parameters(cls, params, where):
    # Several wrappers accept **kwargs, then forward only named parent
    # arguments or ignore them. A typo must not become an unnoticed default.
    allowed = {}
    for parent in cls.__mro__:
        initializer = vars(parent).get("__init__")
        if initializer is not None and parent is not object:
            for name, parameter in inspect.signature(initializer).parameters.items():
                if name != "self" and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                    allowed.setdefault(name, parameter)
    unknown = params.keys() - allowed.keys()
    if unknown:
        raise WorkflowError(f"{where}: unknown parameters {sorted(unknown)}")
    try:
        inspect.signature(cls).bind(**params)
    except TypeError as exc:
        raise WorkflowError(f"{where}: invalid constructor parameters: {exc}") from exc
    for name, value in params.items():
        if type(allowed[name].default) is bool and type(value) is not bool:
            raise WorkflowError(f"{where}: {name} must be a boolean")
    # Validate backend names/choices that these constructors defer until data
    # fitting. This performs no fit and changes no metric implementation.
    from pyspi.statistics import basic
    if issubclass(cls, basic.Estimators):
        estimator = params.get("estimator", "EmpiricalCovariance")
        if not isinstance(estimator, str) or estimator.startswith("_") or not inspect.isclass(getattr(basic.cov, estimator, None)):
            raise WorkflowError(f"{where}: unknown covariance estimator {estimator!r}")
    if issubclass(cls, basic.CrossCorrelation) and params.get("statistic", "max") not in ("mean", "max", "min", "peak"):
        raise WorkflowError(f"{where}: cross-correlation statistic must be mean, max, min, or peak")
    if any(parent.__name__ == "PairwiseDistance" and parent.__module__ == "pyspi.statistics.distance"
           for parent in cls.__mro__):
        from sklearn.metrics import pairwise
        # This is the same registry used by the installed backend's parameter
        # validation, including its version-dependent SciPy aliases.
        metric = params.get("metric", "euclidean")
        if not isinstance(metric, str) or metric not in {*pairwise._VALID_METRICS, "precomputed"}:
            raise WorkflowError(f"{where}: unknown pairwise-distance metric {metric!r}")
    if any(parent.__name__ == "JIDTBase" and parent.__module__ == "pyspi.statistics.infotheory"
           for parent in cls.__mro__):
        if params.get("estimator", "gaussian") not in ("gaussian", "kernel", "kraskov", "kozachenko", "symbolic"):
            raise WorkflowError(f"{where}: unknown information-theory estimator")
    if any(parent.__name__ in ("NonparametricSpectral", "ParametricSpectral", "mne")
           and parent.__module__ in ("pyspi.statistics.spectral", "pyspi.statistics.wavelet")
           for parent in cls.__mro__):
        fs = positive_number(params.get("fs", 1), f"{where}.fs")
        fmin, fmax = params.get("fmin", 0), params.get("fmax")
        fmax = float(fs / 2) if fmax is None else fmax
        if (isinstance(fmin, bool) or not isinstance(fmin, (int, float)) or
            isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or
            not 0 <= fmin <= fmax <= float(fs / 2)):
            raise WorkflowError(f"{where}: frequency band must satisfy 0 <= fmin <= fmax <= fs/2")


def resolve_metrics(path, *, sampling_rate_hz=None) -> tuple[Metric, ...]:
    """Instantiate and describe every request; never fall back, skip, or fit.

    Java-dependent constructors require starting the existing JIDT JVM, whose
    availability is checked only when requested/inferred. Class label lists and
    existing Python/NumPy RNG state are restored after inspection.
    """
    entries = _entries(path)
    import numpy as np

    numpy_state, python_state = np.random.get_state(), random.getstate()
    label_lists = {}
    result, seen = [], set()
    java_checked = False

    def java_required():
        nonlocal java_checked
        if not java_checked:
            from pyspi.utils import is_jpype_jvm_available
            if not is_jpype_jvm_available():
                raise WorkflowError("requested metrics require an available Java/JIDT runtime")
            java_checked = True

    try:
        with redirect_stdout(sys.stderr):
            for module_name, name, params, dependencies in entries:
                where = f"{module_name}.{name}({params!r})"
                try:
                    module = importlib.import_module(module_name, "pyspi")
                    cls = getattr(module, name)
                    from pyspi.base import Directed
                    if not inspect.isclass(cls) or not issubclass(cls, Directed):
                        raise WorkflowError(f"{where}: requested class is not an SPI")
                    from pyspi.statistics.basic import CrossCorrelation
                    if issubclass(cls, CrossCorrelation) and sampling_rate_hz is not None:
                        rate = float(positive_number(sampling_rate_hz, "sampling_rate_hz"))
                        if "fs" in params and float(positive_number(params["fs"], f"{where}.fs")) != rate:
                            raise WorkflowError(f"{where}: fs differs from workflow sampling_rate_hz")
                        params = {**params, "fs": rate}
                    required = spi_dependencies(cls, params, dependencies)
                    if "java" in required:
                        java_required()
                    for parent in cls.__mro__:
                        labels = vars(parent).get("labels")
                        if isinstance(labels, list):
                            label_lists.setdefault(id(labels), (labels, labels.copy()))
                    _validate_parameters(cls, params, where)
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        spi = cls(**params)
                except WorkflowError:
                    raise
                except Exception as exc:
                    raise WorkflowError(f"{where}: cannot initialize requested metric: {type(exc).__name__}: {exc}") from exc
                identifier = spi.identifier
                if not isinstance(identifier, str) or not identifier or "/" in identifier:
                    raise WorkflowError(f"{where}: invalid metric identifier {identifier!r}")
                if identifier in seen:
                    raise WorkflowError(f"duplicate metric identifier {identifier!r}; explicitly remove or distinguish the colliding configurations")
                seen.add(identifier)
                result.append(Metric(identifier, cls.__module__, name, json.loads(json.dumps(params)),
                                     str(inspect.signature(cls)), tuple(required), tuple(spi.labels),
                                     tuple(str(w.message) for w in captured)))
    finally:
        for labels, original in label_lists.values():
            labels[:] = original
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return tuple(result)
