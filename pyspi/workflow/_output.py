"""Shared dataset run reader and durable output utilities."""

import sys
from seeg_dataset import _runs as _implementation

sys.modules[__name__] = _implementation
