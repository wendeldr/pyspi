"""Resumable-transfer test: what an interrupted copy or one corrupted byte costs per format.

For each kept format directory of one patient, plus a single-file tar of the recommended store
(a stand-in for one monolithic file such as an HDF5 container):

1. full copy with ``rsync -a`` (source evicted from the page cache first)
2. interrupted copy: ``rsync`` gets SIGINT once about half the bytes have landed, then the same
   ``rsync -a`` is rerun; records the bytes the resume had to send again
3. silent corruption on the copy: one byte flipped in a destination file with its timestamps
   restored (bit rot / bad write). Checks whether plain ``rsync -a`` notices, whether reading the
   copy notices (format checksums), and what ``rsync -a --checksum`` re-sends to repair it.

Copies go to --dest on the same drive; only benchmark outputs are copied, never the original stores.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

from bench_containers import evict
from containers import all_containers

CORRUPT_METRIC = "cov_EmpiricalCovariance"


def files_under(p: Path):
    return sorted(f for f in p.rglob("*") if f.is_file())


def rsync(src: Path, dst: Path, *flags):
    """Run rsync src/ dst/ and return (seconds, files sent, bytes sent) from --stats."""
    t = time.perf_counter()
    out = subprocess.run(["rsync", "-a", "--stats", *flags, f"{src}/", f"{dst}/"], check=True,
                         capture_output=True, text=True).stdout
    subprocess.run(["sync"], check=True)
    s = time.perf_counter() - t
    files = int(re.search(r"Number of regular files transferred: ([\d,]+)", out).group(1).replace(",", ""))
    sent = int(re.search(r"Total transferred file size: ([\d,]+)", out).group(1).replace(",", ""))
    return s, files, sent


def interrupted_rsync(src: Path, dst: Path, at: float = 0.5):
    """Start rsync and send SIGINT once overall progress reaches ``at``; return bytes landed."""
    p = subprocess.Popen(["rsync", "-a", "--info=progress2", "--no-inc-recursive", f"{src}/", f"{dst}/"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    buf, landed = b"", 0
    while True:
        chunk = p.stdout.read1(4096)
        if not chunk:
            break
        buf = (buf + chunk)[-512:]
        m = re.findall(rb"([\d,]+)\s+(\d+)%", buf)
        if m and int(m[-1][1]) >= at * 100:
            landed = int(m[-1][0].replace(b",", b""))
            os.killpg(p.pid, signal.SIGINT)
            break
    p.stdout.close()
    p.wait()
    subprocess.run(["sync"], check=True)
    return landed


def corrupt_target(fmt_dir: Path, container: str) -> Path:
    """The data file holding the middle of CORRUPT_METRIC in this format."""
    if container == "monolithic_tar":
        return files_under(fmt_dir)[0]
    files = [f for f in files_under(fmt_dir / CORRUPT_METRIC) if f.name != "zarr.json"]
    return max(files, key=lambda f: f.stat().st_size) if "shard" in container else files[len(files) // 2]


def flip_byte_keep_times(path: Path):
    st = path.stat()
    with open(path, "r+b") as fh:
        fh.seek(st.st_size // 2)
        b = fh.read(1)
        fh.seek(st.st_size // 2)
        fh.write(bytes([b[0] ^ 0x10]))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def self_check(src: Path, dst: Path, container: str):
    """Read every metric of the copy; report errors raised and silently wrong metrics."""
    c = all_containers()[container]
    t = time.perf_counter()
    errors, silent = [], []
    for m in sorted(p.name for p in src.iterdir() if p.is_dir()):
        try:
            got = c.read_metric(dst, m)
        except Exception as e:                         # checksum or decode failure names the metric
            errors.append(f"{m}: {type(e).__name__}")
            continue
        if m == CORRUPT_METRIC and got.tobytes() != c.read_metric(src, m).tobytes():
            silent.append(m)
    return dict(check_s=time.perf_counter() - t, errors=errors, silent_wrong=silent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats-root", type=Path, required=True, help="runs/<run>/<patient> from bench --keep")
    ap.add_argument("--dest", type=Path, required=True, help="scratch directory for the copies")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fmts = sorted(p for p in args.formats_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    mono_src = next(p for p in fmts if p.name.startswith("zarr_shard_pcodec8@f64"))
    mono = args.formats_root.parent / f"_{args.formats_root.name}_tar" / f"monolithic_tar@{mono_src.name.split('@')[1]}"
    mono.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    with tarfile.open(mono / "store.tar", "w") as tf:
        tf.add(mono_src, arcname=mono_src.name)
    tar_s = time.perf_counter() - t
    rows = []
    for fmt in [*fmts, mono]:
        container = fmt.name.split("@")[0]
        files = files_under(fmt)
        size = sum(f.stat().st_size for f in files)
        row = dict(format=fmt.name, files=len(files), bytes=size)
        if container == "monolithic_tar":
            row["tar_create_s"] = tar_s
        dst = args.dest / fmt.name
        shutil.rmtree(dst, ignore_errors=True)
        # 1. full copy
        evict(files)
        row["copy_s"], _, _ = rsync(fmt, dst)
        shutil.rmtree(dst)
        # 2. interrupted at ~50 %, then resumed
        evict(files)
        row["landed_before_interrupt"] = interrupted_rsync(fmt, dst)
        row["resume_s"], row["resume_files_sent"], row["resume_bytes_sent"] = rsync(fmt, dst)
        row["resume_resent_fraction"] = (row["landed_before_interrupt"] + row["resume_bytes_sent"]) / size - 1
        _, row["after_resume_files_differing"], _ = rsync(fmt, dst, "--checksum", "--dry-run")
        # 3. one flipped byte in the copy, timestamps restored
        target = corrupt_target(dst, container)
        flip_byte_keep_times(target)
        row["corrupted_file_bytes"] = target.stat().st_size
        _, row["plain_rsync_files_sent"], _ = rsync(fmt, dst)
        if container != "monolithic_tar":
            row["self_check"] = self_check(fmt, dst, container)
        evict(files + files_under(dst))
        row["repair_checksum_s"], row["repair_files_sent"], row["repair_bytes_sent"] = rsync(fmt, dst, "--checksum")
        if container != "monolithic_tar" and row["self_check"]["errors"]:
            bad = fmt / CORRUPT_METRIC
            flip_byte_keep_times(target)                  # re-damage, then repair only the named metric
            row["targeted_repair_s"], _, row["targeted_repair_bytes"] = rsync(bad, dst / CORRUPT_METRIC, "--checksum")
        shutil.rmtree(dst)
        rows.append(row)
        print(json.dumps(row), flush=True)
    shutil.rmtree(mono.parent)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
