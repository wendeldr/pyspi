import pytest

from pyspi.workflow._resources import estimate_workers


def test_resource_estimate_limits_memory_and_cpu_independently():
    small = estimate_workers(memory_mib=8192, worker_peak_mib=1500, parent_peak_mib=1000, cpus=32)
    assert small["workers_upper_bound"] == 3
    large = estimate_workers(memory_mib=256*1024, worker_peak_mib=1500, parent_peak_mib=1000,
                             cpus=32, threads_per_worker=2)
    assert large["workers_upper_bound"] == 16
    impossible = estimate_workers(memory_mib=1024, worker_peak_mib=1500, parent_peak_mib=1000, cpus=32)
    assert impossible["workers_upper_bound"] == 0


@pytest.mark.parametrize("change", [{"memory_mib": 0}, {"worker_peak_mib": float("nan")},
    {"parent_peak_mib": -1}, {"cpus": True}, {"threads_per_worker": 0}, {"reserve_fraction": 1}])
def test_resource_estimate_rejects_invalid_limits(change):
    values = dict(memory_mib=8192, worker_peak_mib=1500, parent_peak_mib=1000, cpus=8)
    values.update(change)
    with pytest.raises(ValueError):
        estimate_workers(**values)
