"""Helpers for performing tests."""
from typing import Callable, Iterable, Iterator, Tuple, Type
import itertools
import math

import pytest

from srctools import math as vec_mod
from srctools.math import (
    Cy_Angle, Cy_Matrix, Cy_parse_vec_str, Cy_Vec, Py_Angle, Py_Matrix, Py_parse_vec_str,
    Py_Vec,
)


VALID_NUMS = [
    # 10e38 is the max single value, make sure we use double-precision.
    30, 1.5, 0.2827, 2.3464545636e47,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = VALID_NUMS + [0, -0]

# In SMD files the maximum precision is this, so it should be a good reference.
EPSILON = 1e-6

PyCVec = Tuple[Type[Py_Vec], Type[Py_Angle], Type[Py_Matrix], Callable[..., Tuple[float, float, float]]]


def iter_vec(nums: Iterable[float]) -> Iterator[Tuple[float, float, float]]:
    for x in nums:
        for y in nums:
            for z in nums:
                yield x, y, z


class ExactType:
    """Proxy object which verifies both value and types match."""
    def __init__(self, val: object) -> None:
        self.value = val

    def __repr__(self) -> str:
        return f'{self.value!r}'

    def __eq__(self, other) -> bool:
        if isinstance(other, ExactType):
            other = other.value
        return type(self.value) is type(other) and self.value == other


def assert_ang(ang, pitch=0, yaw=0, roll=0, msg='', tol=EPSILON):
    """Asserts that an Angle is equal to the provided angles."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

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

    new_msg = "Angle({:.10g}, {:.10g}, {:.10g}).{} != ({:.10g}, {:.10g}, {:.10g})".format(ang.pitch, ang.yaw, ang.roll, failed, pitch, yaw, roll)
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


def assert_vec(vec, x, y, z, msg='', tol=EPSILON):
    """Asserts that Vec is equal to (x,y,z)."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    assert type(vec).__name__ == 'Vec'

    if not math.isclose(vec.x, x, abs_tol=tol):
        failed = 'x'
    elif not math.isclose(vec.y, y, abs_tol=tol):
        failed = 'y'
    elif not math.isclose(vec.z, z, abs_tol=tol):
        failed = 'z'
    else:
        # Success!
        return

    new_msg = "{!r}.{} != ({}, {}, {})".format(vec, failed, x, y, z)
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


def assert_rot(rot, exp_rot, msg=''):
    """Asserts that the two rotations are the same."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

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


if Py_Vec is Cy_Vec:
    parms = [(Py_Vec, Py_Angle, Py_Matrix, Py_parse_vec_str)]
    names = ['Python']
    print('No _vec! ')
else:
    parms = [(Py_Vec, Py_Angle, Py_Matrix, Py_parse_vec_str),
             (Cy_Vec, Cy_Angle, Cy_Matrix, Cy_parse_vec_str)]
    names = ['Python', 'Cython']


@pytest.fixture(params=parms, ids=names)
def py_c_vec(request):
    """Run the test twice, for the Python and C versions."""
    orig_vec = vec_mod.Vec
    orig_Matrix = vec_mod.Matrix
    orig_Angle = vec_mod.Angle
    orig_parse = vec_mod.parse_vec_str

    try:
        (
            vec_mod.Vec,
            vec_mod.Matrix,
            vec_mod.Angle,
            vec_mod.parse_vec_str,
        ) = request.param
        yield request.param
    finally:
        vec_mod.Vec = orig_vec
        vec_mod.Matrix = orig_Matrix
        vec_mod.Angle = orig_Angle
        vec_mod.parse_vec_str = orig_parse


def parameterize_cython(param: str, py_vers, cy_vers):
    """If the Cython version is available, parameterize the test function."""
    if py_vers is cy_vers:
        return pytest.mark.parametrize(param, [py_vers], ids=['Python'])
    else:
        return pytest.mark.parametrize(param, [py_vers, cy_vers], ids=['Python', 'Cython'])
