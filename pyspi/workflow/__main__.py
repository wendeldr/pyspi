"""Terminal preflight; exit 0 ready, 1 resampling needed, 2 invalid input."""

import argparse
import json
import sys

from . import WorkflowError, preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="validate configuration and EDF headers without computing connectivity")
    check.add_argument("config", help="workflow YAML path")
    check.add_argument("--json", action="store_true", help="print machine-readable report, including every [start, stop) sample range")
    args = parser.parse_args(argv)
    try:
        report = preflight(args.config)
    except WorkflowError as exc:
        print(f"Preflight error: {exc}", file=sys.stderr)
        return 2
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
