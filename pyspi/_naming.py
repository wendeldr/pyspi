"""SPI identifiers in one fixed, filename-safe grammar.

    <measure>[_<variant>...][_<parameter>...]

- Only ``a-z``, ``0-9``, ``_`` and ``-`` appear, so an identifier can be used
  unchanged as a file name or a table column.
- ``_`` separates fields. ``-`` only joins two numbers into a range, and ``p``
  is the decimal point inside a number (``0p05`` is 0.05).
- The first field is the measure. Variant fields hold no digits (transforms
  such as ``sq``, then method or estimator, test, statistic); a digit in an
  option's value is spelled out (``l1`` -> ``lone``). Parameter fields
  are ``<name><number or range><unit>``, e.g. ``band0030-0070hz``,
  ``maxlag0125ms`` or ``history01``. A number without a unit counts samples.
- Numbers are zero-padded to a per-parameter minimum width so identifiers in
  one family line up and sort numerically; a wider value is written in full.
"""

from decimal import Decimal
import re

_WORD = re.compile(r"[a-z]+")
_PARAMETER = re.compile(r"([a-z]+?)(\d[\dp]*(?:-\d[\dp]*)?)([a-z]*)")
_NUMBER = re.compile(r"\d+(?:p\d+)?")
_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def word(value):
    """A variant field: ``'EmpiricalCovariance'`` -> ``'empiricalcovariance'``.

    Digits are spelled out (``'l1'`` -> ``'lone'``, ``'MI1'`` -> ``'mione'``) so a
    variant can never be read as a parameter, and distinct values stay distinct.
    """
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    text = re.sub(r"\d", lambda digit: _DIGITS[int(digit.group())], text)
    if not _WORD.fullmatch(text):
        raise ValueError(f"{value!r} cannot be written as a variant word.")
    return text


def number(value, width=1):
    """Exact, filename-safe decimal: ``number(30, 4)`` -> ``'0030'``, ``number(0.05)`` -> ``'0p05'``."""
    if isinstance(value, bool):
        raise TypeError("Booleans are written as variant words, not numbers.")
    exact = Decimal(repr(float(value)))
    if not exact.is_finite() or exact < 0:
        raise ValueError(f"Identifier numbers must be finite and nonnegative, not {value!r}.")
    whole, _, fraction = format(exact, "f").partition(".")
    fraction = fraction.rstrip("0")
    return whole.zfill(width) + (f"p{fraction}" if fraction else "")


def parameter(name, value, width=2, unit=""):
    """A parameter field: ``parameter('history', 1)`` -> ``'history01'``."""
    return f"{word(name)}{number(value, width)}{unit}"


def frequency_band(fmin, fmax, fs):
    """Band in Hz, or in cycles per sample when ``fs`` is the default of 1 (rate unknown)."""
    if float(fs) == 1:
        return f"band{number(fmin)}-{number(fmax)}cyc"
    return f"band{number(fmin, 4)}-{number(fmax, 4)}hz"


def duration(name, seconds):
    """A time parameter in milliseconds: ``duration('maxlag', 0.125)`` -> ``'maxlag0125ms'``."""
    return f"{word(name)}{number(Decimal(repr(float(seconds))) * 1000, 4)}ms"


def compose(measure, *fields):
    """Join fields with ``_`` after checking each one against the grammar."""
    if not _WORD.fullmatch(measure):
        raise ValueError(f"Measure {measure!r} must be lowercase letters.")
    for field in fields:
        if not (_WORD.fullmatch(field) or _PARAMETER.fullmatch(field)):
            raise ValueError(f"Identifier field {field!r} does not fit the naming grammar.")
    return "_".join((measure,) + tuple(fields))


def _value(text):
    return float(text.replace("p", "."))


def parse(identifier):
    """Split an identifier into its measure, variant words and parameters.

    Parameters map to ``(value, unit)``; a range value is a ``(low, high)`` tuple.
    """
    measure, *fields = identifier.split("_")
    if not _WORD.fullmatch(measure):
        raise ValueError(f"{identifier!r} does not start with a measure word.")
    variants, parameters = [], {}
    for field in fields:
        if _WORD.fullmatch(field):
            variants.append(field)
            continue
        match = _PARAMETER.fullmatch(field)
        if match is None:
            raise ValueError(f"{identifier!r}: field {field!r} does not fit the naming grammar.")
        name, value, unit = match.groups()
        low, _, high = value.partition("-")
        if not _NUMBER.fullmatch(low) or (high and not _NUMBER.fullmatch(high)):
            raise ValueError(f"{identifier!r}: field {field!r} has a malformed number.")
        parameters[name] = ((_value(low), _value(high)) if high else _value(low), unit)
    return measure, variants, parameters
