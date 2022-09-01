"""Deprecated, this was renamed to srctools.keyvalues."""
from typing import TYPE_CHECKING
import warnings


__all__ = ['KeyValError', 'NoKeyError', 'Property']

warnings.warn(
    'srctools.property_parser module was deprecated, import srctools.keyvalues instead.',
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from srctools.keyvalues import (
        KeyValError as KeyValError, Keyvalues as Property, NoKeyError as NoKeyError,
    )
else:
    from srctools import keyvalues
    def __getattr__(name: str) -> object:
        if name == 'Property':
            warnings.warn(
                'srctools.property_parser.Property is renamed to srctools.keyvalues.Keyvalues',
                DeprecationWarning,
                stacklevel=2,
            )
            return keyvalues.Keyvalues
        elif name in ('KeyValError', 'NoKeyError'):
            warnings.warn(
                f'srctools.property_parser.{name} is renamed to srctools.keyvalues.{name}',
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(keyvalues, name)
        raise AttributeError(name)
