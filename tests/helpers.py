"""Helpers for performing tests."""
from typing import Callable, Generator, Iterable, Iterator, Optional, Tuple, Type, TypeVar, Union
from typing_extensions import TypeAlias
import builtins
import itertools
import math

from dirty_equals import DirtyEquals
import pytest

from srctools import math as vec_mod


# These are for testing uses only.
# isort: off
# noinspection PyProtectedMember
from srctools.math import (
    Py_Vec, Py_FrozenVec, Cy_Vec, Cy_FrozenVec, 
    Py_Angle, Py_FrozenAngle, Cy_Angle, Cy_FrozenAngle,
    Py_Matrix, Py_FrozenMatrix, Cy_Matrix, Cy_FrozenMatrix,
    Py_parse_vec_str, Cy_parse_vec_str,
)
# isort: on


__all__ = [
    'iter_vec', 'ExactType', 'py_c_vec', 'vec_mod', 'parameterize_cython',
    'VALID_NUMS', 'VALID_ZERONUMS', 'EPSILON', 'PyCVec',
    'Py_Vec', 'Py_FrozenVec', 'Cy_Vec', 'Cy_FrozenVec', 
    'Py_Angle', 'Py_FrozenAngle', 'Cy_Angle', 'Cy_FrozenAngle',
    'Py_Matrix', 'Py_FrozenMatrix', 'Cy_Matrix', 'Cy_FrozenMatrix', 
    'Py_parse_vec_str', 'Cy_parse_vec_str',
    'VecClass', 'AngleClass', 'MatrixClass',
    'assert_vec', 'assert_ang', 'assert_rot',
    'frozen_thawed_vec', 'frozen_thawed_angle', 'frozen_thawed_matrix',
]


VALID_NUMS = [
    # 10e38 is the max single value, make sure we use double-precision.
    30, 1.5, 0.2827, 2.3464545636e47,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = [*VALID_NUMS, 0, -0, 2.535047750982637e-175]

# In SMD files the maximum precision is this, so it should be a good reference.
EPSILON = 1e-6

PyCVec: TypeAlias = Tuple[
    Type[Py_Vec], Type[Py_Angle], Type[Py_Matrix],
    Callable[..., Tuple[float, float, float]
]]
T = TypeVar('T')
VecClass: TypeAlias = Union[Type[Py_Vec], Type[Py_FrozenVec]]
AngleClass: TypeAlias = Union[Type[Py_Angle], Type[Py_FrozenAngle]]
MatrixClass: TypeAlias = Union[Type[Py_Matrix], Type[Py_FrozenMatrix]]


def iter_vec(nums: Iterable[T]) -> Iterator[Tuple[T, T, T]]:
    return itertools.product(nums, nums, nums)


class ExactType(DirtyEquals[object]):
    """Proxy object which verifies both value and types match."""
    def __init__(self, val: object) -> None:
        super().__init__(val)
        self.compare = val

    def equals(self, other: object) -> bool:
        if isinstance(other, ExactType):
            other = other.compare
        return type(self.compare) is type(other) and self.compare == other


def assert_ang(
    ang: vec_mod.AngleBase,
    pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0,
    msg: object = '',
    tol: float = EPSILON,
    type: Optional[type] = None,
):
    """Asserts that an Angle is equal to the provided angles."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    assert builtins.type(ang).__name__ in ('Angle', 'FrozenAngle'), ang
    if type is not None:
        assert builtins.type(ang) is type, f'{builtins.type(ang)} != {type}: {msg}'

    pitch %= 360
    yaw %= 360
    roll %= 360

    # Ignore slight variations, and handle one being ~359.99
    if not (
        math.isclose(ang.pitch, pitch, abs_tol=tol) or
        math.isclose(ang.pitch - pitch, 360.0, abs_tol=tol)
    ):
        failed = 'pitch'
    elif not (
        math.isclose(ang.yaw, yaw, abs_tol=tol) or
        math.isclose(ang.yaw - yaw, 360.0, abs_tol=tol)
    ):
        failed = 'yaw'
    elif not (
        math.isclose(ang.roll, roll, abs_tol=tol) or
        math.isclose(ang.roll - roll, 360.0, abs_tol=tol)
    ):
        failed = 'roll'
    else:
        # Success!
        return

    new_msg = (
        f"Angle({ang.pitch:.10g}, {ang.yaw:.10g}, {ang.roll:.10g}).{failed} "
        f"!= ({pitch:.10g}, {yaw:.10g}, {roll:.10g})"
    )
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


def assert_vec(
    vec: vec_mod.VecBase,
    x: float = 0.0, y: float = 0.0, z: float = 0.0,
    msg: object = '',
    tol: float = EPSILON,
    type: Optional[type] = None,
) -> None:
    """Asserts that Vec is equal to (x,y,z)."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    assert builtins.type(vec).__name__ in ('Vec', 'FrozenVec'), vec
    if type is not None:
        assert builtins.type(vec) is type, f'{builtins.type(vec)} != {type}: {msg}'

    if not math.isclose(vec.x, x, abs_tol=tol):
        failed = 'x'
    elif not math.isclose(vec.y, y, abs_tol=tol):
        failed = 'y'
    elif not math.isclose(vec.z, z, abs_tol=tol):
        failed = 'z'
    else:
        # Success!
        return

    new_msg = f"{vec!r}.{failed} != ({x}, {y}, {z})"
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


def assert_rot(
    rot: vec_mod.MatrixBase, exp_rot: vec_mod.MatrixBase,
    msg: object = '', type: Optional[type] = None,
) -> None:
    """Asserts that the two rotations are the same."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    assert builtins.type(rot).__name__ in ('Matrix', 'FrozenMatrix'), rot
    if type is not None:
        assert builtins.type(rot) is type, f'{builtins.type(rot)} != {type}: {msg}'

    for x, y in itertools.product(range(3), range(3)):
        if not math.isclose(rot[x, y], exp_rot[x, y], abs_tol=EPSILON):
            break
    else:
        # Success!
        return

    new_msg = f'{rot} != {exp_rot}\nAxis: {x},{y}'
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


ATTRIBUTES = [
    'Vec', 'VecBase', 'FrozenVec',
    'Angle', 'AngleBase', 'FrozenAngle',
    'Matrix', 'MatrixBase', 'FrozenMatrix',
    'parse_vec_str', 'to_matrix',
]
if Py_Vec is Cy_Vec:
    parms = ['Python']
    print('No _vec! ')
else:
    parms = ['Python', 'Cython']


@pytest.fixture(params=parms)
def py_c_vec(request) -> Generator[None, None, None]:
    """Run the test twice, for the Python and C versions."""
    originals = [getattr(vec_mod, name) for name in ATTRIBUTES]
    prefix = request.param[:2] + '_'  # Python -> Py_
    try:
        for name in ATTRIBUTES:
            setattr(vec_mod, name, getattr(vec_mod, prefix + name))
        yield None
    finally:
        for name, orig in zip(ATTRIBUTES, originals):
            setattr(vec_mod, name, orig)


def parameterize_cython(param: str, py_vers: object, cy_vers: object):
    """If the Cython version is available, parameterize the test function."""
    if py_vers is cy_vers:
        return pytest.mark.parametrize(param, [py_vers], ids=['Python'])
    else:
        return pytest.mark.parametrize(param, [py_vers, cy_vers], ids=['Python', 'Cython'])


@pytest.fixture(params=['Vec', 'FrozenVec'])
def frozen_thawed_vec(py_c_vec, request) -> VecClass:
    """Support testing both mutable and immutable vectors."""
    return getattr(vec_mod, request.param)


@pytest.fixture(params=['Angle', 'FrozenAngle'])
def frozen_thawed_angle(py_c_vec, request) -> AngleClass:
    """Support testing both mutable and immutable angles."""
    return getattr(vec_mod, request.param)


@pytest.fixture(params=['Matrix', 'FrozenMatrix'])
def frozen_thawed_matrix(py_c_vec, request) -> MatrixClass:
    """Support testing both mutable and immutable matrices."""
    return getattr(vec_mod, request.param)
