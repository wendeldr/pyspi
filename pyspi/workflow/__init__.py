"""Terminal EEG workflow with bounded window workers and read-only EDF preflight."""

from ._config import Configuration, WorkflowError, load_config
from ._preflight import PreflightReport, preflight
from ._windows import IncompleteWindowWarning, WindowPlan
from ._edf import SkippedChannelsWarning
from ._metadata import export_selection
from ._output import RunResults, open_run
from ._run import run

__all__ = ["Configuration", "WorkflowError", "load_config", "PreflightReport",
           "preflight", "IncompleteWindowWarning", "WindowPlan", "SkippedChannelsWarning",
           "export_selection", "RunResults", "open_run", "run"]
