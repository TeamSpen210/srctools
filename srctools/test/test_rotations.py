"""Test rotations in srctools.vec."""
from srctools.test import *
from srctools import Vec, Rotation, Angle


def test_vec_identities() -> None:
    """Check that vectors in the same axis as the rotation don't get spun."""
    for ang in range(0, 360, 13):
        # Check the two constructors match.
        assert Rotation.from_pitch(ang) == Rotation.from_angle(Angle(pitch=ang))
        assert Rotation.from_yaw(ang) == Rotation.from_angle(Angle(yaw=ang))
        assert Rotation.from_roll(ang) == Rotation.from_angle(Angle(roll=ang))
        
        # Various magnitudes to test
        for mag in (-250, -1, 0, 1, 250):
            assert_vec(Vec(y=mag) @ Rotation.from_pitch(ang), 0, mag, 0)
            assert_vec(Vec(z=mag) @ Rotation.from_yaw(ang), 0, 0, mag)
            assert_vec(Vec(x=mag) @ Rotation.from_roll(ang), mag, 0, 0)


def test_vec_basic_yaw() -> None:
    """Check each direction rotates appropriately in yaw."""
    assert_vec(Vec(200, 0, 0) @ Rotation.from_yaw(0), 200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Rotation.from_yaw(0), 0, 150, 0)
    
    assert_vec(Vec(200, 0, 0) @ Rotation.from_yaw(90), 0, 200, 0)
    assert_vec(Vec(0, 150, 0) @ Rotation.from_yaw(90), -150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) @ Rotation.from_yaw(180), -200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Rotation.from_yaw(180), 0, -150, 0)
    
    assert_vec(Vec(200, 0, 0) @ Rotation.from_yaw(270), 0, -200, 0)
    assert_vec(Vec(0, 150, 0) @ Rotation.from_yaw(270), 150, 0, 0)


def test_vec_basic_pitch():
    """Check each direction rotates appropriately in pitch."""
    assert_vec(Vec(200, 0, 0) @ Rotation.from_pitch(0), 200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Rotation.from_pitch(0), 0, 0, 150)
    
    assert_vec(Vec(200, 0, 0) @ Rotation.from_pitch(90), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) @ Rotation.from_pitch(90), 150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) @ Rotation.from_pitch(180), -200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Rotation.from_pitch(180), 0, 0, -150)
    
    assert_vec(Vec(200, 0, 0) @ Rotation.from_pitch(270), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) @ Rotation.from_pitch(270), -150, 0, 0)


def test_ang_matrix_roundtrip():
    """Check converting to and from a Matrix does not change values."""
    for p, y, r in iter_vec(range(0, 360, 90)):
        vert = Vec(x=1).rotate(p, y, r).z
        if vert < 0.99 or vert > 0.99:
            # If nearly vertical, gimbal lock prevents roundtrips.
            continue
        mat = Rotation.from_angle(Angle(p, y, r))
        assert_ang(mat.to_angle(), p, y, r)


def test_matrix_roundtrip_pitch():
    """Check converting to and from a Matrix does not change values."""
    # We can't directly check the resulted value, some of these produce
    # gimbal lock and can't be recovered.
    # So instead check the rotation matrix is the same.
    for pitch in range(0, 360, 45):
        old_ang = Angle(pitch, 0, 0)
        new_ang = Rotation.from_pitch(pitch).to_angle()
        assert_rot(
            Rotation.from_angle(old_ang),
            Rotation.from_angle(new_ang),
            (old_ang, new_ang),
        )


def test_matrix_roundtrip_yaw():
    """Check converting to and from a Matrix does not change values."""
    for yaw in range(0, 360, 45):
        mat = Rotation.from_yaw(yaw)
        assert_ang(mat.to_angle(), 0, yaw, 0)


def test_matrix_roundtrip_roll():
    """Check converting to and from a Matrix does not change values."""
    for roll in range(0, 360, 45):
        if roll in (90, -90):
            # Don't test gimbal lock.
            continue
        mat = Rotation.from_roll(roll)
        assert_ang(mat.to_angle(), 0, 0, roll)


def test_single_axis():
    """In each axis, two rotations should be the same as adding."""
    # Pitch gives gimbal lock and breaks recovery of the values.
    for axis in ('yaw', 'roll'):
        for ang1 in range(0, 360, 45):
            for ang2 in range(0, 360, 45):
                if ang1 + ang2 == 0:
                    # 0 gives a value around the 360-0 split,
                    # so it can round to the wrong side sometimes.
                    continue
                assert_ang(
                    Angle(**{axis: ang1}) @
                    Angle(**{axis: ang2}),
                    **{axis: (ang1 + ang2) % 360},
                    msg=(axis, ang1, ang2)
                )


def test_gen_check():
    """Do an exhaustive check on all rotation math using data from the engine."""
    X = Vec(x=1)
    Y = Vec(y=1)
    Z = Vec(z=1)
    with open('rotation.txt') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.startswith('|'):
                # Skip other junk in the log.
                continue

            (
                pit, yaw, roll,
                for_x, for_y, for_z,
                left_x, left_y, left_z,
                up_x, up_y, up_z
            ) = map(float, line[1:].split())

            mat = Rotation.from_angle(Angle(pit, yaw, roll))

            # Then check rotating vectors works correctly.
            assert_vec(abs(X @ mat), for_x, for_y, for_z)
            assert_vec(abs(Y @ mat), left_x, left_y, left_z)
            assert_vec(abs(Z @ mat), up_x, up_y, up_z)

            assert math.isclose(for_x, mat.a)
            assert math.isclose(-for_y, mat.b)
            assert math.isclose(for_z, mat.c)

            assert math.isclose(left_x, mat.d)
            assert math.isclose(-left_y, mat.e)
            assert math.isclose(left_z, mat.f)

            assert math.isclose(up_x, mat.g)
            assert math.isclose(-up_y, mat.h)
            assert math.isclose(up_z, mat.i)

