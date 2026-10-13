"""Cohort size projection for the candidate formats (an extrapolation, not a measured conversion).

Method A (primary): the in-memory screen measured every metric of three patients (n = 40, 131, 249)
on 64 windows. Its compression ratio vs raw float64, interpolated over channel count, gives the
n-dependence; a container factor (full conversion ratio / screen ratio, measured on the fully
converted patients) adds chunking and metadata overhead at full window counts.

Method B (cross-check): fraction of the current store size, interpolated over n from the full
conversions, with the 111 point taken from its metric subset and corrected by the same subset's
fraction on 105 (the subset over-represents directed metrics).

Usage: project_cohort.py --screen reports/screen_fast_w64 --full reports/containers_full \
                         [--subset reports/containers_111_subset]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from analyze_containers import INVENTORY, source_bytes

VARIANTS = {   # container variant -> matching screen option (precision, structure, layout, codec)
    "zarr_shard_pcodec8@f64:exact": ("f64", "exact", "wm", "pcodec8"),
    "zarr_shard_zstd3@f64:exact": ("f64", "exact", "wm", "zstd3"),
    "zarr_shard_pcodec8@f32:exact": ("f32", "exact", "wm", "pcodec8"),
    "zarr_shard_zstd3@f32:exact": ("f32", "exact", "wm", "zstd3"),
}


def rows_of(run):
    return [r for r in map(json.loads, (Path(run) / "containers.jsonl").read_text().splitlines()) if "metric" in r]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--subset", type=Path)
    args = ap.parse_args()
    inv = {r["patient"].removesuffix(".zarr"): r for r in json.loads(INVENTORY.read_text())}
    total_raw = sum(r["raw_f64"] for r in inv.values())
    total_src = sum(r["conn_bytes"] for r in inv.values())
    total_files = sum(r["conn_files"] for r in inv.values())

    screen = collections.defaultdict(lambda: [0, 0])
    for line in (args.screen / "screen.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["precision"], r["structure"], r["layout"], r["codec"])
        if key in VARIANTS.values():
            s = screen[(key, r["patient"])]
            s[0] += r["raw_bytes"]
            s[1] += r["bytes"]
    full = collections.defaultdict(lambda: collections.Counter())
    for r in rows_of(args.full):
        a = full[(r["variant"], r["patient"])]
        a["metrics"] += 1
        for k in ("raw_f64_bytes", "bytes", "files"):
            a[k] += r[k]
        a["source"] += source_bytes(r["patient"], r["metric"])
    complete = {k: v for k, v in full.items() if v["metrics"] == inv[k[1]]["metrics"]}

    out = {}
    print(f"Cohort: {len(inv)} recordings, raw float64 {total_raw / 2**30:.1f} GiB, "
          f"current store {total_src / 2**30:.1f} GiB in {total_files:,} files\n")
    print("| Variant | Method A est. GiB | vs current | Method B est. GiB | vs current | Est. files | "
          "Screen ratio vs raw (n=40/131/249) | Container factor |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for variant, key in VARIANTS.items():
        pts = sorted((inv[p]["n"], raw / b) for (k, p), (raw, b) in screen.items() if k == key)
        if not pts:
            continue
        ns, ratios = map(np.array, zip(*pts))
        conv = sorted((inv[p]["n"], (a["raw_f64_bytes"] / a["bytes"]) / float(np.interp(inv[p]["n"], ns, ratios)),
                       a["files"] / a["metrics"], a["bytes"] / a["source"])
                      for (v, p), a in complete.items() if v == variant)
        if not conv:
            print(f"| {variant} | (no full conversion) | | | | | {', '.join(f'{x:.2f}' for x in ratios)} | |")
            continue
        cn, cf, fpm, frac = map(np.array, zip(*conv))
        est_a = sum(r["raw_f64"] / (np.interp(r["n"], ns, ratios) * np.interp(r["n"], cn, cf)) for r in inv.values())
        files = sum(np.interp(r["n"], cn, fpm) * r["metrics"] for r in inv.values())
        est_b = None
        if args.subset is not None and variant in {r["variant"] for r in rows_of(args.subset)}:
            sub = [r for r in rows_of(args.subset) if r["variant"] == variant]
            mets = {r["metric"] for r in sub}
            p_sub = sub[0]["patient"]
            ref = [r for r in rows_of(args.full) if r["variant"] == variant and r["patient"] == "105" and r["metric"] in mets]
            if ref and ("105" in [p for (_, p) in complete if _ == variant]):
                f_sub = sum(r["bytes"] for r in sub) / sum(source_bytes(p_sub, r["metric"]) for r in sub)
                f_ref = sum(r["bytes"] for r in ref) / sum(source_bytes("105", r["metric"]) for r in ref)
                f_105 = float(frac[list(cn).index(inv["105"]["n"])])
                bn, bf = list(cn) + [inv[p_sub]["n"]], list(frac) + [f_105 * f_sub / f_ref]
                order = np.argsort(bn)
                bn, bf = np.array(bn)[order], np.array(bf)[order]
                est_b = sum(np.interp(r["n"], bn, bf) * r["conn_bytes"] for r in inv.values())
        out[variant] = dict(method_a_bytes=est_a, method_b_bytes=est_b, est_files=files,
                            screen_points=dict(zip(map(int, ns), map(float, ratios))),
                            container_factor=dict(zip(map(int, cn), map(float, cf))))
        b_txt = f"{est_b / 2**30:.0f} | {est_b / total_src:.3f}" if est_b else "– | –"
        print(f"| {variant} | {est_a / 2**30:.0f} | {est_a / total_src:.3f} | {b_txt} | {files:,.0f} | "
              f"{' / '.join(f'{x:.2f}' for x in ratios)} | {', '.join(f'n={int(a)}: {b:.3f}' for a, b in zip(cn, cf))} |")
    (args.full / "cohort_projection.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
