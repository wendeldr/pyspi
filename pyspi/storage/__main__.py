"""Run ``python -m pyspi.storage verify PATH``."""

import sys

from . import verify


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "verify":
        sys.exit("usage: python -m pyspi.storage verify <store>")
    failures = verify(sys.argv[2])
    for metric, error in failures.items():
        print(f"FAILED {metric}: {error}")
    print(f"{len(failures)} damaged metric(s)" if failures else "all metrics verified")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
