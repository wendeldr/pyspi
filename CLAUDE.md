# Working in pyspi

## Scope is the whole package

"Metrics", "SPIs" or "calculations" means every SPI class in `pyspi/statistics/`
and every value its constructor accepts, unless a specific subset is named.
The YAML configs (`pyspi/config.yaml`, `fast_config.yaml`, `sonnet_config.yaml`,
`fabfour_config.yaml`, `pyspi/workflow/*.yaml`, `tests/regression/extra_config.yaml`)
are selections for particular runs, not the definition of what is supported.

- Enumerate SPIs by walking the classes in `pyspi.statistics.*` and their
  accepted options; use the configs only as additional cases.
- If a check covers only configured SPIs, say so and extend it before calling
  the work done.
