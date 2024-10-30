"""Deprecated, this was renamed to math."""
from typing import TYPE_CHECKING
import warnings


__all__ = [
    'Vec', 'Angle', 'Matrix', 'Vec_tuple',
    'parse_vec_str', 'to_matrix', 'lerp',
]

warnings.warn(
    'srctools.vec module was deprecated, import srctools.math instead.',
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from srctools.math import (
        Angle as Angle, Matrix as Matrix, Vec as Vec, Vec_tuple as Vec_tuple,
        lerp as lerp, parse_vec_str as parse_vec_str, to_matrix as to_matrix,
    )
else:
    from srctools import math

    def __getattr__(name: str) -> object:
        if name in __all__:
            warnings.warn(
                f'srctools.vec.{name} is renamed to srctools.math.{name}',
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(math, name)
        raise AttributeError(name)
