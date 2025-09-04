import io
import re
from typing import Iterator, Tuple, Dict, Any

import yaml


MODULE_HDR_RE = re.compile(r"^(\.[\w\.]+)\s*:\s*$")


def iter_module_blocks(path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (module_name, module_dict) by streaming a large YAML file.

    Assumes the top-level mapping keys are lines like: `.statistics.basic:`
    and that each module block is properly indented beneath its header.

    This avoids loading the full YAML into memory at once by extracting
    and parsing one module section at a time.
    """
    with open(path, "r", encoding="utf-8") as f:
        current_name = None
        current_lines = []

        def emit():
            nonlocal current_name, current_lines
            if current_name is None:
                return
            # Build a minimal YAML containing only this module
            text = f"{current_name}:\n" + "".join(current_lines)
            data = yaml.load(io.StringIO(text), Loader=yaml.FullLoader)
            if not isinstance(data, dict) or current_name not in data:
                return
            yield (current_name, data[current_name])

        for line in f:
            m = MODULE_HDR_RE.match(line)
            if m:
                # Emit the previous block if present
                for item in emit() or []:
                    yield item
                # Start a new block
                current_name = m.group(1)
                current_lines = []
            else:
                # Accumulate lines for the current module
                if current_name is not None:
                    current_lines.append(line)

        # Emit the final block at EOF
        for item in emit() or []:
            yield item

