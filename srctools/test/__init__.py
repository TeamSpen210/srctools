"""Helpers for performing tests."""
from srctools import Vec, Angle
import pytest

VALID_NUMS = [
    1, 1.5, 0.2827, 2346.45,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = VALID_NUMS + [0, -0]


def iter_vec(nums):
    for x in nums:
        for y in nums:
            for z in nums:
                yield x, y, z


def assert_ang(ang: Angle, pitch, yaw, roll, msg=''):
    """Asserts that an Angle is equal to the provided angles.."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    pitch %= 360
    yaw %= 360
    roll %= 360

    # Ignore slight variations
    if not ang.pitch == pytest.approx(pitch):
        pass
    elif not ang.yaw == pytest.approx(yaw):
        pass
    elif not ang.roll == pytest.approx(roll):
        pass
    else:
        # Success!
        return

    new_msg = "{!r} != ({:g}, {:g}, {:g})".format(ang, pitch, yaw, roll)
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


def assert_vec(vec, x, y, z, msg=''):
    """Asserts that Vec is equal to (x,y,z)."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    # Ignore slight variations
    if not vec.x == pytest.approx(x):
        pass
    elif not vec.y == pytest.approx(y):
        pass
    elif not vec.z == pytest.approx(z):
        pass
    else:
        # Success!
        return

    new_msg = "{!r} != ({:g}, {:g}, {:g})".format(vec, x, y, z)
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)
