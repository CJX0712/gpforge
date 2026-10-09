"""Typed error hierarchy (E100–E500) so failures are diagnosable, not opaque."""

from __future__ import annotations


class GpForgeError(Exception):
    """Base class for all gpforge errors."""


class ConfigError(GpForgeError):
    """Invalid configuration or environment variable."""


class DataError(GpForgeError):
    """Dataset shape / value problem (e.g. non-finite features, mismatched dims)."""


class ModelError(GpForgeError):
    """Model construction or fitting failure (e.g. non-PSD kernel matrix)."""


class MethodError(GpForgeError):
    """Algorithmic-method misuse (e.g. predict before fit, unknown kernel)."""


class EvalError(GpForgeError):
    """Metric / evaluation failure (e.g. empty arrays, bad quantiles)."""
