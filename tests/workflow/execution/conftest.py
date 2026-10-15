import numpy as np
import pytest


@pytest.fixture(autouse=True)
def prohibit_connectivity():
    """Execution tests deliberately exercise Data and Calculator."""


@pytest.fixture
def signal_edf(edf):
    def write(*, rate=256, records=20, labels=("EEG A1", "EEG B2", "ECG unused"), name="signals.EDF"):
        rates = [rate] * len(labels) if isinstance(rate, int) else rate
        values = []
        for index, r in enumerate(rates):
            t = np.arange(r * records) / r
            values.append(np.rint(6000*np.sin(2*np.pi*10*t + index*.35) +
                                  2000*np.sin(2*np.pi*60*t) + 500*t))
        return edf(labels=labels, samples=tuple(rates), records=records, values=values, name=name)
    return write


@pytest.fixture
def execution_config(workflow, signal_edf, tmp_path):
    def write(*, source=None, **changes):
        config = dict(sampling_rate_hz=256, bandpass={"low_hz": 2, "high_hz": 80},
                      reference="car", output_directory=str(tmp_path / "output"),
                      output_batch_mib=.0001)
        config.update(changes)
        return workflow(source if source is not None else signal_edf(), **config)
    return write
