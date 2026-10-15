"""Run the numeric checks end to end with the right interpreter for each step.

1. ``make_data.py --out``            (project venv; skipped when data/MANIFEST.json exists)
2. ``pyspi_outputs.py``              (project venv: computes pyspi's matrices)
3. ``refs/<family>.py``              (scratch comparison venv: independent references;
   a family script may itself call ``Rscript r/<family>.R`` or ``octave-cli octave/<family>.m``
   through subprocess, exchanging CSV files under ``out/exchange/<family>/``)
4. ``compare.py``                    (either venv; writes out/results/<family>.json + merged results.json)

Interpreters: the project venv is ``<repo>/.venv/bin/python``; the comparison
venv is read from ``$AUDIT_REF_PYTHON`` or ``audit/env/comparison-venv.path``.
Build it with ``uv venv --python 3.12 DIR && uv pip install --python DIR/bin/python -r audit/env/comparison-requirements.txt``.
R packages live in ``$R_LIBS_USER`` (default ``~/R/audit-lib``, see env/r-packages.txt).

Usage: ``python audit/checks/run_checks.py [--family F ...] [--skip-pyspi] [--skip-refs] [--force]``
Exit status is non-zero if any step fails or any comparison fails.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
AUDIT = CHECKS.parent
REPO = AUDIT.parent
PROJECT_PY = REPO / ".venv" / "bin" / "python"


def ref_python():
    env = os.environ.get("AUDIT_REF_PYTHON")
    if env:
        return Path(env)
    p = AUDIT / "env" / "comparison-venv.path"
    if p.is_file():
        return Path(p.read_text().strip()) / "bin" / "python"
    sys.exit("comparison venv not configured: set AUDIT_REF_PYTHON or write audit/env/comparison-venv.path")


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", default=None)
    ap.add_argument("--skip-pyspi", action="store_true")
    ap.add_argument("--skip-refs", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fam_args = sum((["--family", f] for f in (a.family or [])), [])
    failures = []
    if not (CHECKS / "data" / "MANIFEST.json").is_file():
        if run([PROJECT_PY, CHECKS / "make_data.py", "--out", CHECKS / "data"]):
            sys.exit("make_data failed")
    if not a.skip_pyspi:
        if run([PROJECT_PY, CHECKS / "pyspi_outputs.py"] + fam_args + (["--force"] if a.force else [])):
            failures.append("pyspi_outputs")
    if not a.skip_refs:
        rp = ref_python()
        env = dict(os.environ, R_LIBS_USER=os.environ.get("R_LIBS_USER", str(Path.home() / "R" / "audit-lib")))
        for script in sorted((CHECKS / "refs").glob("*.py")):
            fam = script.stem
            if fam.startswith("_") or (a.family and fam not in a.family):
                continue
            if run([rp, script], env=env):
                failures.append(f"refs/{script.name}")
        # r/<family>.R and octave/<family>.m are invoked by refs/<family>.py itself (subprocess with
        # R_LIBS_USER set, CSV exchange under out/exchange/<family>/), so the family script owns the
        # conversion of their output into the reference npz contract.
    if run([PROJECT_PY, CHECKS / "compare.py"] + fam_args):
        failures.append("compare")
    if failures:
        print("FAILED steps: " + ", ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("all check steps passed")


if __name__ == "__main__":
    main()
