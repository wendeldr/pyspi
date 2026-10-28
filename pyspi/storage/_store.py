"""Shared storage implementation owned by seeg-dataset."""

import sys
from seeg_dataset.storage import _store as _implementation

sys.modules[__name__] = _implementation
