"""Run every configured window and group, retaining a whole-process profile."""
import argparse
from pathlib import Path

from pyspi.workflow._profile import profile_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = profile_experiment(args.config, max_windows=None,
                                report_path=args.report, resume=args.resume)
    print('Completed', result['recording_windows'], 'recording windows in',
          result['elapsed_seconds'], 'seconds', flush=True)
