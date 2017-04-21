"""Test rotations in srctools.vec."""
from srctools.test import *
from srctools import Vec, Quat, Angle


def test_vec_identities():
    """Check that vectors in the same axis as the rotation don't get spun."""
    for ang in range(0, 360, 13):
        # Check the two constructors match.
        assert Quat.from_pitch(ang) == Quat.from_angle(Angle(pitch=ang))
        assert Quat.from_yaw(ang) == Quat.from_angle(Angle(yaw=ang))
        assert Quat.from_roll(ang) == Quat.from_angle(Angle(roll=ang))
        
        # Various magnitudes to test
        for mag in (-250, -1, 0, 1, 250):
            assert_vec(Vec(y=mag) * Quat.from_pitch(ang), 0, mag, 0)
            assert_vec(Vec(z=mag) * Quat.from_yaw(ang), 0, 0, mag)
            assert_vec(Vec(x=mag) * Quat.from_roll(ang), mag, 0, 0)


def test_vec_basic_yaw():
    """Check each direction rotates appropriately in yaw."""
    assert_vec(Vec(200, 0, 0) * Quat.from_yaw(0), 200, 0, 0)
    assert_vec(Vec(0, 150, 0) * Quat.from_yaw(0),  0, 150, 0)
    
    assert_vec(Vec(200, 0, 0) * Quat.from_yaw(90), 0, -200, 0)
    assert_vec(Vec(0, 150, 0) * Quat.from_yaw(90), 150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) * Quat.from_yaw(180), -200, 0, 0)
    assert_vec(Vec(0, 150, 0) * Quat.from_yaw(180), 0, -150, 0)
    
    assert_vec(Vec(200, 0, 0) * Quat.from_yaw(270), 0, 200, 0)
    assert_vec(Vec(0, 150, 0) * Quat.from_yaw(270), -150, 0, 0)
    

def test_vec_basic_pitch():
    """Check each direction rotates appropriately in pitch."""
    assert_vec(Vec(200, 0, 0) * Quat.from_pitch(0), 200, 0, 0)
    assert_vec(Vec(0, 0, 150) * Quat.from_pitch(0), 0, 0, 150)
    
    assert_vec(Vec(200, 0, 0) * Quat.from_pitch(90), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) * Quat.from_pitch(90), 150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) * Quat.from_pitch(180), -200, 0, 0)
    assert_vec(Vec(0, 0, 150) * Quat.from_pitch(180), 0, 0, -150)
    
    assert_vec(Vec(200, 0, 0) * Quat.from_pitch(270), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) * Quat.from_pitch(270), -150, 0, 0)


def test_ang_quat_roundtrip():
    """Check converting to and from a quaternion does not change values."""
    for p, y, r in iter_vec(range(0, 360, 90)):
        vert = Vec(x=1).rotate(p, y, r).z
        if vert < 0.99 or vert > 0.99:
            # If nearly vertical, gimbal lock prevents roundtrips.
            continue
        quat = Quat.from_angle(Angle(p, y, r))
        assert_ang(quat.to_angle(), p, y, r)
