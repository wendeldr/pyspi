"""Resolve estimator-dependent requirements within a configured SPI class."""


def spi_dependencies(cls, parameters=None, declared=None):
    required = list(dict.fromkeys(declared or ()))
    requires_java = getattr(cls, "requires_java", None)
    if requires_java is not None:
        required = [dep for dep in required if dep != "java"]
        if requires_java(**(parameters or {})):
            required.append("java")
    return required
