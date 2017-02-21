from srctools import Vec, Angle, RotationMatrix
from srctools.test.test_vec import assert_vec
import pytest


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


def test_matrix_conversion():
    values = [0, -90, 90, 270, 37, 28]
    for p in values:
        for y in values:
            for r in values:
                ang = Angle(p, y, r)
                mat = RotationMatrix()
                mat.rotate_by_angle(ang)

                v = Vec(50)
                v2 = v * mat
                v.rotate(p, y, r)
                assert_vec(v, v2.x, v2.y, v2.z, (ang, mat))

                new_ang = mat.to_angle()
                assert_ang(new_ang, p, y, r)
