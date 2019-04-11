"""Test rotations in srctools.vec."""
from srctools.test import *
from srctools import Vec, Matrix, Angle


def test_vec_identities() -> None:
    """Check that vectors in the same axis as the rotation don't get spun."""
    for ang in range(0, 360, 13):
        # Check the two constructors match.
        assert Matrix.from_pitch(ang) == Matrix.from_angle(Angle(pitch=ang))
        assert Matrix.from_yaw(ang) == Matrix.from_angle(Angle(yaw=ang))
        assert Matrix.from_roll(ang) == Matrix.from_angle(Angle(roll=ang))
        
        # Various magnitudes to test
        for mag in (-250, -1, 0, 1, 250):
            assert_vec(Vec(y=mag) @ Matrix.from_pitch(ang), 0, mag, 0)
            assert_vec(Vec(z=mag) @ Matrix.from_yaw(ang), 0, 0, mag)
            assert_vec(Vec(x=mag) @ Matrix.from_roll(ang), mag, 0, 0)


def test_vec_basic_yaw() -> None:
    """Check each direction rotates appropriately in yaw."""
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(0), 200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(0),  0, 150, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(90), 0, -200, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(90), 150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(180), -200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(180), 0, -150, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(270), 0, 200, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(270), -150, 0, 0)


def test_vec_basic_pitch():
    """Check each direction rotates appropriately in pitch."""
    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(0), 200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(0), 0, 0, 150)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(90), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(90), 150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(180), -200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(180), 0, 0, -150)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(270), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(270), -150, 0, 0)


def test_ang_matrix_roundtrip():
    """Check converting to and from a Matrix does not change values."""
    for p, y, r in iter_vec(range(0, 360, 90)):
        vert = Vec(x=1).rotate(p, y, r).z
        if vert < 0.99 or vert > 0.99:
            # If nearly vertical, gimbal lock prevents roundtrips.
            continue
        mat = Matrix.from_angle(Angle(p, y, r))
        assert_ang(mat.to_angle(), p, y, r)


def test_matrix_roundtrip_pitch():
    """Check converting to and from a Matrix does not change values."""
    for pitch in range(0, 360, 45):
        if pitch in (90, -90):
            # Don't test gimbal lock.
            continue
        mat = Matrix.from_pitch(pitch)
        assert_ang(mat.to_angle(), pitch, 0, 0)


def test_matrix_roundtrip_yaw():
    """Check converting to and from a Matrix does not change values."""
    for yaw in range(0, 360, 45):
        mat = Matrix.from_yaw(yaw)
        assert_ang(mat.to_angle(), 0, yaw, 0)


def test_matrix_roundtrip_roll():
    """Check converting to and from a Matrix does not change values."""
    for roll in range(0, 360, 45):
        if roll in (90, -90):
            # Don't test gimbal lock.
            continue
        mat = Matrix.from_roll(roll)
        assert_ang(mat.to_angle(), 0, 0, roll)


def test_single_axis():
    """In each axis, two rotations should be the same as adding."""
    for axis in ('pitch', 'yaw', 'roll'):
        for ang1 in VALID_ZERONUMS:
            for ang2 in VALID_ZERONUMS:
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
    """Do an exhaustive check on all rotation math using data from the engine.

    For each 45 degree angle, an offset in 6 directions is computed as well
    as each 45 degree angle.
    """
    OFF = [
        Vec(x=64),
        Vec(x=-64),
        Vec(y=64),
        Vec(y=-64),
        Vec(z=64),
        Vec(z=-64),
    ]
    with open('rotation.txt') as f:
        for w_pitch in range(0, 360, 45):
            for w_yaw in range(0, 360, 45):
                for w_roll in range(0, 360, 45):
                    world = Angle(w_pitch, w_yaw, w_roll)

                    # First the world line, which should match us.
                    # world: pitch yaw roll
                    world_line = f.readline().split()
                    assert len(world_line) == 4

                    assert_ang(
                        world,
                        float(world_line[1]),
                        float(world_line[2]),
                        float(world_line[3]),
                    )
                    # 6 offsets at 64 from origin in order.
                    for off in OFF:
                        head, name, x, y, z = f.readline().split()
                        assert_vec(
                            round(off @ world),
                            round(float(x)),
                            round(float(y)),
                            round(float(z)),
                            msg='({}) @ {}'.format(off, world),
                        )
                    # Then all 512 local positions.
                    for l_pitch in range(0, 360, 45):
                        for l_yaw in range(0, 360, 45):
                            for l_roll in range(0, 360, 45):
                                local = Angle(l_pitch, l_yaw, l_roll)
                                rotated = world @ local
                                head, p, y, r = f.readline().split()
                                assert_ang(
                                    rotated,
                                    float(p), float(y), float(r),
                                    (world, local),
                                )
