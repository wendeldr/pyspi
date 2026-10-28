"""Refresh only directed spectral and PEC goldens, preserving other provenance.

Run after independent formula/integration checks. Generation uses the normal
regression harness in a temporary copy; unrelated arrays are checked byte-for-
byte before replacing the three affected golden sets.
"""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]

import numpy as np
from regression import generate_goldens as generator, harness, spi_policy


PREFIXES = ("dcoh_", "pdcoh_", "gpdcoh_", "dtf_", "ddtf_", "sgc_", "pec")


def affected(identifier):
    if "::" in identifier:
        identifier = spi_policy.split_preproc_key(identifier)[0]
    return identifier.startswith(PREFIXES)


def main():
    golden_root = ROOT/"tests/regression/goldens"
    names = ("small", "full", "preproc")
    previous = {name: json.loads((golden_root/name/"meta.json").read_text()) for name in names}
    commands = [*[['--size', 'all', '--spis', pattern] for pattern in
                  ('[dgp]*coh_*', '*dtf_*', 'sgc_*', 'pec*')],
                ['--preproc', '--spis', 'sgc_nonparametric*'],
                ['--preproc', '--spis', 'pec*']]
    sources = ("pyspi/_spectral.py", "pyspi/_wilson.py", "pyspi/_directed.py", "pyspi/_envelope.py",
               "pyspi/statistics/spectral.py", "pyspi/statistics/misc.py", "pyspi/calculator.py",
               "audit/checks/probes/refresh_spectral_pec_goldens.py")
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    with tempfile.TemporaryDirectory(prefix="pyspi-reviewed-goldens-", dir="/tmp") as temporary:
        for name in names:
            shutil.copytree(golden_root/name, Path(temporary)/name)
        for command in commands:
            assert generator.main([*command, '--repeat', '1', '--out', temporary]) == 0
        prepared = {}
        for name in names:
            generated = json.loads((Path(temporary)/name/"meta.json").read_text())
            with np.load(golden_root/name/"values.npz") as old, np.load(Path(temporary)/name/"values.npz") as new:
                assert set(old.files) == set(new.files), name
                unrelated = [key for key in old.files if not affected(key)]
                assert all(old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
                           and old[key].tobytes() == new[key].tobytes() for key in unrelated), name
                identifiers = sorted(key for key in new.files if affected(key))
            metadata = copy.deepcopy(previous[name])
            removed_catalogue_entries = sorted(set(metadata["calculator_order"])-set(generated["calculator_order"]))
            # The preprocessing set predates the approved removal of phase max;
            # none of its 96 numerical arrays uses that removed definition.
            assert all(item.startswith("phase_multitaper_max_") for item in removed_catalogue_entries)
            assert [item for item in metadata["calculator_order"] if item not in removed_catalogue_entries] == generated["calculator_order"]
            metadata["calculator_order"] = generated["calculator_order"]
            for field in ("spis", "variants"):
                if field not in generated:
                    continue
                assert set(metadata[field]) == set(generated[field])
                for identifier, entry in generated[field].items():
                    if affected(identifier):
                        metadata[field][identifier] = entry
            record = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                command=".venv/bin/python -B audit/checks/probes/refresh_spectral_pec_goldens.py",
                generator_arguments=[[*command, '--repeat', '1'] for command in commands], identifiers=identifiers,
                removed_obsolete_catalogue_order_entries=removed_catalogue_entries,
                reason="Approved directed formulas, source-first orientation, symmetric Wilson fits, native directed grid, pair-local failures and corrected PEC definitions/numerics.",
                environment=generated["environment"], source_sha256=hashes,
                unrelated_arrays_verified_byte_identical=len(unrelated))
            metadata.setdefault("partial_regenerations", []).append(record)
            prepared[name] = metadata
        for name in names:
            shutil.copyfile(Path(temporary)/name/"values.npz", golden_root/name/"values.npz")
            (golden_root/name/"meta.json").write_text(json.dumps(prepared[name], indent=1, sort_keys=True)+"\n")
            print(f"Updated {name}: {len(prepared[name]['partial_regenerations'][-1]['identifiers'])} affected arrays; unrelated arrays and metadata preserved.", flush=True)


if __name__ == "__main__":
    main()
