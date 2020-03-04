"""Helpers for performing tests."""
import itertools

import srctools
import pytest
import math

VALID_NUMS = [
    # 10e38 is the max single value, make sure we use double-precision.
    30, 1.5, 0.2827, 2.3464545636e47,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = VALID_NUMS + [0, -0]

# In SMD files the maximum precision is this, so it should be a good reference.
EPSILON = 1e-6


def iter_vec(nums):
    for x in nums:
        for y in nums:
            for z in nums:
                yield x, y, z


def assert_ang(ang, pitch=0, yaw=0, roll=0, msg=''):
    """Asserts that an Angle is equal to the provided angles."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    pitch %= 360
    yaw %= 360
    roll %= 360

    # Ignore slight variations
    if not math.isclose(ang.pitch, pitch, abs_tol=EPSILON):
        failed = 'pitch'
    elif not math.isclose(ang.yaw, yaw, abs_tol=EPSILON):
        failed = 'yaw'
    elif not math.isclose(ang.roll, roll, abs_tol=EPSILON):
        failed = 'roll'
    else:
        # Success!
        return

    new_msg = "{!r}.{} != ({:g}, {:g}, {:g})".format(ang, failed, pitch, yaw, roll)
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

    for row, col in itertools.product('abc', 'abc'):
        pos = row + col
        if not math.isclose(
            getattr(rot, pos),
            getattr(exp_rot, pos),
            abs_tol=EPSILON,
        ):
            break
    else:
        # Success!
        return

    new_msg = '{} != {}\nAxis: {}'.format(rot, exp_rot, pos)
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


@pytest.fixture(params=[srctools.Vec])
def py_c_vec(request):
    """Run the test twice, for the Python and C versions."""
    orig_vec = srctools.Vec
    srctools.Vec = request.param
    try:
        yield request.param
    finally:
        srctools.Vec = orig_vec
