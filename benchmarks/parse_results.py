import argparse
import csv
import json
import math
import sys
from statistics import mean, stdev
from typing import Dict, List, Any, Iterable


def aggregate_by_identifier(results: Iterable[Dict[str, Any]]):
    groups: Dict[str, Dict[str, Any]] = {}

    for rec in results:
        ident = rec.get("identifier") or "<unknown>"
        g = groups.setdefault(
            ident,
            {
                "times": [],
                "ok": 0,
                "failed": 0,
                "timeout": 0,
                "skipped": 0,
                "attempts": 0,
            },
        )
        attempts = rec.get("attempts", [])
        for a in attempts:
            g["attempts"] += 1
            status = a.get("status")
            if status == "ok":
                g["ok"] += 1
                times = a.get("times_sec") or []
                # Only include non-empty timing lists
                for t in times:
                    if t is not None and math.isfinite(t):
                        g["times"].append(float(t))
            elif status == "failed":
                g["failed"] += 1
            elif status == "timeout":
                g["timeout"] += 1
            else:
                # count other statuses as skipped
                g["skipped"] += 1
    return groups


def summarize(times: List[float]):
    n = len(times)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95": float("nan"),
        }
    m = mean(times)
    if n == 1:
        return {"n": 1, "mean": m, "std": 0.0, "ci95": 0.0}
    s = stdev(times)
    # Normal approximation for 95% CI of the mean
    z = 1.96
    se = s / math.sqrt(n)
    ci = z * se
    return {"n": n, "mean": m, "std": s, "ci95": ci}


def fmt_secs(x: float) -> str:
    if x != x:  # NaN
        return "NaN"
    if x < 0.001:
        return f"{x*1e6:.1f} µs"
    if x < 1.0:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize one or more benchmark run JSONs by identifier")
    ap.add_argument("json_files", nargs="+", help="Path(s) to raw_timings_*.json produced by bench_runner")
    ap.add_argument(
        "--sort",
        choices=["id", "mean"],
        default="id",
        help="Sort output by identifier or mean time",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path (use '-' for stdout). Contains raw numeric seconds.",
    )
    args = ap.parse_args(argv)

    all_results: List[Dict[str, Any]] = []
    for jf in args.json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        res = data.get("results", [])
        all_results.extend(res)

    groups = aggregate_by_identifier(all_results)

    rows = []
    for ident, g in groups.items():
        summ = summarize(g["times"])
        rows.append(
            {
                "identifier": ident,
                "n": summ["n"],
                "mean": summ["mean"],
                "std": summ["std"],
                "ci95": summ["ci95"],
                "ok": g["ok"],
                "failed": g["failed"],
                "timeout": g["timeout"],
            }
        )

    if args.sort == "mean":
        rows.sort(key=lambda r: (math.inf if r["mean"] != r["mean"] else r["mean"]))
    else:
        rows.sort(key=lambda r: r["identifier"])

    # Print readable table to stdout
    print(
        f"{'Identifier':40s}  {'N':>4s}  {'Mean':>12s}  {'Std':>12s}  {'95% CI ±':>12s}  {'OK':>4s}  {'Fail':>4s}  {'TO':>4s}"
    )
    for r in rows:
        ident = r["identifier"][:40]
        n = r["n"]
        m = fmt_secs(r["mean"]) if n > 0 else "NaN"
        s = fmt_secs(r["std"]) if n > 1 else ("0" if n == 1 else "NaN")
        ci = fmt_secs(r["ci95"]) if n > 1 else ("0" if n == 1 else "NaN")
        print(
            f"{ident:40s}  {n:4d}  {m:>12s}  {s:>12s}  {ci:>12s}  {r['ok']:4d}  {r['failed']:4d}  {r['timeout']:4d}"
        )

    # Optional CSV output
    if args.csv:
        if args.csv == "-":
            writer = csv.writer(sys.stdout)
            close_fp = False
            fp = None
        else:
            fp = open(args.csv, "w", newline="", encoding="utf-8")
            writer = csv.writer(fp)
            close_fp = True
        try:
            writer.writerow(["identifier", "n", "mean_sec", "std_sec", "ci95_sec", "ok", "failed", "timeout"])
            for r in rows:
                writer.writerow([
                    r["identifier"],
                    r["n"],
                    ("" if r["n"] == 0 else f"{r['mean']:.9f}"),
                    ("" if r["n"] <= 1 else f"{r['std']:.9f}"),
                    ("" if r["n"] <= 1 else f"{r['ci95']:.9f}"),
                    r["ok"],
                    r["failed"],
                    r["timeout"],
                ])
        finally:
            if close_fp and fp is not None:
                fp.close()


if __name__ == "__main__":
    main()
