"""Deprecated, this was renamed to math."""
import warnings
warnings.warn(
    'srctools.vec module was deprecated, import srctools.math instead.',
    DeprecationWarning,
    stacklevel=2,
)
del warnings
from srctools.math import *  # noqa
