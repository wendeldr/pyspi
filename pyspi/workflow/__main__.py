"""Serial EEG calculation, read-only preflight, and electrode-order export."""

import argparse
import json
import sys

from . import WorkflowError, preflight, run, export_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="validate configuration and EDF headers without computing connectivity")
    check.add_argument("config", help="workflow YAML path")
    check.add_argument("--json", action="store_true", help="print machine-readable report, including every [start, stop) sample range")
    execute = commands.add_parser("run", help="prepare EDFs and calculate serially into bounded result batches")
    execute.add_argument("config", help="workflow YAML path")
    execute.add_argument("--resume", action="store_true", help="retain committed batches and recompute unfinished batches")
    execute.add_argument("--retry-failures", action="store_true", help="with --resume, explicitly retry recorded failures")
    execute.add_argument("--max-windows", type=int, help="explicitly limit output for a bounded trial (continuous preparation still uses the recording)")
    export = commands.add_parser("export-selection", help="read legacy Zarr electrode order and match actual EDF headers")
    export.add_argument("store")
    export.add_argument("edf")
    export.add_argument("destination", help="new standalone metadata JSON file")
    export.add_argument("--patient-id")
    export.add_argument("--recording-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            paths = run(args.config, resume=args.resume, retry_failures=args.retry_failures,
                        max_windows=args.max_windows)
            print(json.dumps({"runs": [str(p) for p in paths]}, indent=2))
            return 0
        if args.command == "export-selection":
            export_selection(args.store, args.edf, args.destination,
                             patient_id=args.patient_id, recording_id=args.recording_id)
            print(args.destination)
            return 0
        report = preflight(args.config)
    except (WorkflowError, OSError) as exc:
        label = "Preflight error" if args.command == "preflight" else "Workflow error"
        print(f"{label}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; committed batches and diagnostic journals retained", file=sys.stderr)
        return 130
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, allow_nan=False))
    else:
        print(f"{len(report.recordings)} recording(s), {len(report.metrics)} metric(s), target {float(report.sampling_rate_hz):g} Hz")
        for recording in report.recordings:
            windows = recording["windows"]
            native = ", ".join(f"{rate:g}" for rate in sorted({c["native_sampling_rate_hz"] for c in recording["channels"]}))
            description = (f"{windows.count} full windows, {windows.length_samples} samples, step {windows.step_samples}"
                           if windows is not None else "requires resampling; window ranges deferred")
            print(f"{recording['patient_id']}/{recording['recording_id']}: {len(recording['channels'])} channels, native {native} Hz; {description}")
        print("Ready" if report.ready else "Preparation required: resample mismatched inputs before calculation")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
