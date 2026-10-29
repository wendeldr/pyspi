"""EEG calculation, runtime estimates, resource planning, read-only preflight and electrode export."""

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
    execute = commands.add_parser("run", help="prepare EDFs and calculate with configured window workers")
    execute.add_argument("config", help="workflow YAML path")
    execute.add_argument("--resume", action="store_true", help="retain committed batches and recompute unfinished batches")
    execute.add_argument("--retry-failures", action="store_true", help="with --resume, explicitly retry recorded failures")
    execute.add_argument("--max-windows", type=int, help="explicitly limit output for a bounded trial (continuous preparation still uses the recording)")
    group_check = commands.add_parser("preflight-experiment", help="validate every named metric group")
    group_check.add_argument("config")
    grouped = commands.add_parser("run-experiment", help="run named groups with bounded shared preparation")
    grouped.add_argument("config")
    grouped.add_argument("--resume", action="store_true")
    grouped.add_argument("--retry-failures", action="store_true")
    grouped.add_argument("--max-windows", type=int)
    export = commands.add_parser("export-selection", help="read legacy Zarr electrode order and match actual EDF headers")
    export.add_argument("store")
    export.add_argument("edf")
    export.add_argument("destination", help="new standalone metadata JSON file")
    export.add_argument("--patient-id")
    export.add_argument("--recording-id")
    estimate = commands.add_parser("estimate-resources", help="estimate worker limits from measured process peaks; no calculations")
    estimate.add_argument("--memory-mib", type=float, required=True, help="memory available to this run, including parent and workers")
    estimate.add_argument("--worker-peak-mib", type=float, required=True, help="measured worker peak RSS with the intended config/input shape")
    estimate.add_argument("--parent-peak-mib", type=float, required=True, help="measured parent peak RSS, including preparation and output")
    estimate.add_argument("--cpus", type=int, help="CPU allocation; defaults to current affinity")
    estimate.add_argument("--threads-per-worker", type=int, default=1)
    estimate.add_argument("--reserve-fraction", type=float, default=.2)
    measure = commands.add_parser("profile", help="run a bounded workflow and record process-tree memory and end-to-end timing")
    measure.add_argument("config")
    measure.add_argument("--max-windows", type=int, required=True)
    measure.add_argument("--report", required=True, help="JSON measurement receipt")
    experiment_measure = commands.add_parser("profile-experiment", help="profile complete grouped execution and process-tree memory")
    experiment_measure.add_argument("config")
    experiment_measure.add_argument("--max-windows", type=int, required=True)
    experiment_measure.add_argument("--report", required=True)
    guess = commands.add_parser("estimate", help="estimate run time and output size from the configuration; "
                                "measures a calibration beside it when none is current")
    guess.add_argument("config", help="workflow YAML path")
    guess.add_argument("--json", action="store_true", help="print the machine-readable estimate")
    build = commands.add_parser("calibrate", help="measure per-window cost across channel counts and save "
                                "<config>.calibration.json for estimate")
    build.add_argument("config", help="workflow YAML path")
    build.add_argument("--channels", help="comma-separated channel counts to time, e.g. 8,32,all "
                       "(default: 2,4,8,...,128 below the recording's count, then its full count)")
    build.add_argument("--windows-per-point", type=int, default=2, help="windows timed per channel count")
    build.add_argument("--synthetic", action="store_true", help="time AR(1) noise instead of the recording "
                       "(no preparation estimate; data-dependent estimators are misrepresented)")
    build.add_argument("--budget-seconds", type=float, default=600.,
                       help="stop climbing the channel ladder when the next count would exceed this; "
                       "larger recordings are then extrapolated")
    args = parser.parse_args(argv)
    try:
        if args.command == "profile-experiment":
            from ._profile import profile_experiment
            result = profile_experiment(args.config, max_windows=args.max_windows, report_path=args.report)
            print(json.dumps({key: result[key] for key in ("recording_windows", "elapsed_seconds", "process_tree_memory")}, indent=2))
            return 0
        if args.command in ("preflight-experiment", "run-experiment"):
            from .experiment import preflight_experiment, run_experiment
            result = (preflight_experiment(args.config) if args.command == "preflight-experiment" else
                      run_experiment(args.config, resume=args.resume, retry_failures=args.retry_failures,
                                     max_windows=args.max_windows))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.command == "estimate":
            from ._estimate import estimate, format_estimate
            result = estimate(args.config)
            print(json.dumps(result, indent=2, allow_nan=False) if args.json else format_estimate(result))
            return 0
        if args.command == "calibrate":
            from ._calibrate import calibrate
            channels = None
            if args.channels:
                total = max(sum(r["reference_available"]) for r in preflight(args.config).recordings)
                channels = [total if item.strip() == "all" else int(item) for item in args.channels.split(",")]
            result = calibrate(args.config, channels=channels, windows_per_point=args.windows_per_point,
                               synthetic=args.synthetic, budget_seconds=args.budget_seconds)
            print(json.dumps({"calibration": str(result["configuration"]) + ".calibration.json",
                              "channels": [p["channels"] for p in result["points"]],
                              "window_seconds": [p["window_seconds"] for p in result["points"]],
                              "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
            return 0
        if args.command == "profile":
            from ._profile import profile
            result = profile(args.config, max_windows=args.max_windows, report_path=args.report)
            print(json.dumps({key: result[key] for key in ("windows", "elapsed_seconds", "process_tree_memory")}, indent=2))
            return 0
        if args.command == "estimate-resources":
            from ._resources import estimate_workers
            try:
                result = estimate_workers(memory_mib=args.memory_mib, worker_peak_mib=args.worker_peak_mib,
                    parent_peak_mib=args.parent_peak_mib, cpus=args.cpus,
                    threads_per_worker=args.threads_per_worker, reserve_fraction=args.reserve_fraction)
            except ValueError as exc:
                raise WorkflowError(str(exc)) from exc
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
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
    except (WorkflowError, OSError, ValueError) as exc:
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
