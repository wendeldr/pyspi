#!/usr/bin/env python
"""Archive the throwaway probe scripts that audit files cite by scratchpad path.

Records, notes, the reference scripts ``audit/checks/refs/*.py`` and every other
``audit/**/*.py`` that mentions the scratchpad (``scratchpad`` or ``<scratch>`` in its
text; this script excluded) cite probe scripts written in the session scratchpad
(``<dir>/<file>``, ``scratchpad/<dir>/<file>`` or just ``scratchpad/<dir>/``). The
scratch folder is outside the repo (``audit/PATHS.md`` says where it is kept), so
this packs the scripts of every cited scratchpad directory, plus every directory
named after a family (``<family>``, ``<family>_check``, ``<family>_fix``...),
into ``audit/checks/probes/scratch_probes_<date>.tar.xz`` under the same relative
paths, with a manifest (path, bytes, sha256). Only scripts are kept (.py .R .r
.m .sh, not symlinks); data, logs, virtual environments, caches, directories named ``upstream``
(copies of third-party source), ``phitoolbox`` (pyspi's vendored PhiToolbox, which
``checks/refs/phi.py`` reads; re-extract it with
``git archive 76ab2bb^ pyspi/lib/PhiToolbox | tar -x -C <scratch>/phitoolbox``)
and any file above 150 KB are left out. The
scripts were throwaway probes: many read scratch-only paths and are not
runnable as archived; they are kept for provenance.

Usage::

    .venv/bin/python audit/checks/archive_probes.py --scratch <scratchpad dir> [--date 2026-10-15] [--dry-run]
"""
import argparse
import hashlib
import io
import os
import re
import sys
import tarfile
from pathlib import Path

import yaml

AUDIT = Path(__file__).resolve().parents[1]
SCRIPT_EXT = (".py", ".R", ".r", ".m", ".sh")
SKIP_DIRS = {"__pycache__", "upstream", "phitoolbox", ".venv", "venv", "audit-venv", "audit-env", "site-packages"}
MAX_BYTES = 150 * 1024


def cited_dirs(scratch, texts):
    out = set()
    for d in sorted(p.name for p in scratch.iterdir() if p.is_dir()):
        if d in SKIP_DIRS:
            continue
        rx = re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(d) + r"/([A-Za-z0-9_.-]+)")
        bare = re.compile(r"scratchpad/" + re.escape(d) + r"(?=[/`'\s)]|$)")
        if any((scratch / d / m.group(1)).exists() for t in texts for m in rx.finditer(t)) or \
                any(bare.search(t) for t in texts):
            out.add(d)
    return out


def sources():
    """Files scanned for scratchpad citations, by kind, each list sorted by path."""
    refs_dir = AUDIT / "checks" / "refs"
    me = Path(__file__).resolve()
    other_py = sorted(p for p in AUDIT.rglob("*.py")
                      if "__pycache__" not in p.parts and p.parent != refs_dir and p.resolve() != me
                      and re.search(r"scratchpad|<scratch>", p.read_text(errors="replace")))
    return {"records": sorted((AUDIT / "records").glob("*.yaml")),
            "notes": sorted((AUDIT / "notes").glob("*.md")),
            "checks/refs py": sorted(refs_dir.glob("*.py")),
            "other audit py citing the scratchpad": other_py}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scratch = Path(a.scratch)
    src = sources()
    texts = [p.read_text(errors="replace") for kind in src for p in src[kind]]
    print("scanned " + ", ".join(f"{len(v)} {k}" for k, v in src.items()))
    fams = {u["family"] for u in yaml.safe_load((AUDIT / "units.yaml").read_text())["units"]}
    dirs = cited_dirs(scratch, texts) | {p.name for p in scratch.iterdir() if p.is_dir() and
                                         any(p.name == f or p.name.startswith(f + "_") for f in fams)}
    files = []
    for d in sorted(dirs):
        for root, subdirs, names in os.walk(scratch / d):
            subdirs[:] = sorted(s for s in subdirs if s not in SKIP_DIRS)
            for n in sorted(names):
                p = Path(root) / n
                if p.suffix in SCRIPT_EXT and not p.is_symlink() and p.stat().st_size <= MAX_BYTES:
                    files.append(p)
    manifest = ["# path\tbytes\tsha256"]
    for p in files:
        data = p.read_bytes()
        manifest.append(f"{p.relative_to(scratch)}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    print(f"{len(dirs)} directories, {len(files)} scripts, {sum(p.stat().st_size for p in files)/1e6:.2f} MB")
    if a.dry_run:
        return 0
    out_dir = AUDIT / "checks" / "probes"
    tar_path = out_dir / f"scratch_probes_{a.date}.tar.xz"
    with tarfile.open(tar_path, "w:xz") as tar:
        for p in files:
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(scratch)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(p, "rb") as fh:
                tar.addfile(info, fh)
    (out_dir / f"scratch_probes_{a.date}.manifest.tsv").write_text("\n".join(manifest) + "\n")
    print(f"wrote {tar_path} ({tar_path.stat().st_size/1e6:.2f} MB) and its manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
