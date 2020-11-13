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
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(0), 0, 150, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(90), 0, 200, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(90), -150, 0, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(180), -200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(180), 0, -150, 0)
    
    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(270), 0, -200, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(270), 150, 0, 0)


def test_vec_basic_pitch() -> None:
    """Check each direction rotates appropriately in pitch."""
    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(0), 200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(0), 0, 0, 150)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(90), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(90), 150, 0, 0)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(180), -200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(180), 0, 0, -150)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(270), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(270), -150, 0, 0)


def test_vec_basic_roll():
    """Check each direction rotates appropriately in roll."""
    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(0), 0, 200, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(0), 0, 0, 150)

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(90), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(90), 0, -150, 0)

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(180), 0, -200, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(180), 0, 0, -150)

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(270), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(270), 0, 150, 0)


def test_ang_matrix_roundtrip():
    """Check converting to and from a Matrix does not change values."""
    for p, y, r in iter_vec(range(0, 360, 90)):
        vert = Vec(x=1).rotate(p, y, r).z
        if vert < 0.99 or vert > 0.99:
            # If nearly vertical, gimbal lock prevents roundtrips.
            continue
        mat = Matrix.from_angle(Angle(p, y, r))
        assert_ang(mat.to_angle(), p, y, r)


def test_to_angle_roundtrip():
    """Check Vec.to_angle() roundtrips."""
    for x, y, z in iter_vec((-1, 0, 1)):
        if x == y == z == 0:
            continue
        norm = Vec(x, y, z).norm()
        ang = norm.to_angle()
        assert_vec(Vec(x=1).rotate(*ang), norm.x, norm.y, norm.z, ang)


def test_matrix_roundtrip_pitch():
    """Check converting to and from a Matrix does not change values."""
    # We can't directly check the resulted value, some of these produce
    # gimbal lock and can't be recovered.
    # So instead check the rotation matrix is the same.
    for pitch in range(0, 360, 45):
        old_ang = Angle(pitch, 0, 0)
        new_ang = Matrix.from_pitch(pitch).to_angle()
        assert_rot(
            Matrix.from_angle(old_ang),
            Matrix.from_angle(new_ang),
            (old_ang, new_ang),
        )


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


def old_mat_mul(
    self,
    a: float, b: float, c: float,
    d: float, e: float, f: float,
    g: float, h: float, i: float,
) -> None:
    """Code from an earlier version of Vec, that does rotation.

    This just does the matrix multiplication.
    """
    x, y, z = self.x, self.y, self.z

    self.x = (x * a) + (y * b) + (z * c)
    self.y = (x * d) + (y * e) + (z * f)
    self.z = (x * g) + (y * h) + (z * i)


def old_rotate(
    self,
    pitch: float=0.0,
    yaw: float=0.0,
    roll: float=0.0
) -> 'Vec':
    """Code from an earlier version of Vec, that does rotation."""
    # pitch is in the y axis
    # yaw is the z axis
    # roll is the x axis

    rad_pitch = math.radians(pitch)
    rad_yaw = math.radians(yaw)
    rad_roll = math.radians(roll)
    cos_p = math.cos(rad_pitch)
    cos_y = math.cos(rad_yaw)
    cos_r = math.cos(rad_roll)

    sin_p = math.sin(rad_pitch)
    sin_y = math.sin(rad_yaw)
    sin_r = math.sin(rad_roll)

    # Need to do transformations in roll, pitch, yaw order
    old_mat_mul(  # Roll = X
        self,
        1, 0, 0,
        0, cos_r, -sin_r,
        0, sin_r, cos_r,
    )

    old_mat_mul(  # Pitch = Y
        self,
        cos_p, 0, sin_p,
        0, 1, 0,
        -sin_p, 0, cos_p,
    )

    old_mat_mul(  # Yaw = Z
        self,
        cos_y, -sin_y, 0,
        sin_y, cos_y, 0,
        0, 0, 1,
    )
    return self


def test_old_rotation() -> None:
    """Verify that the code matches the results from the earlier Vec.rotate code."""
    for pitch in range(0, 360, 15):
        for yaw in range(0, 360, 15):
            for roll in range(0, 360, 15):
                ang = Angle(pitch, yaw, roll)
                mat = Matrix.from_angle(ang)

                # Construct a matrix directly from 3 vector rotations.
                old_mat = Matrix()
                old_mat.aa, old_mat.ab, old_mat.ac = old_rotate(Vec(x=1), pitch, yaw, roll)
                old_mat.ba, old_mat.bb, old_mat.bc = old_rotate(Vec(y=1), pitch, yaw, roll)
                old_mat.ca, old_mat.cb, old_mat.cc = old_rotate(Vec(z=1), pitch, yaw, roll)

                assert_rot(mat, old_mat, ang)
                old = old_rotate(Vec(128, 0, 0), pitch, yaw, roll)

                by_ang = Vec(128, 0, 0) @ ang
                by_mat = Vec(128, 0, 0) @ mat
                assert_vec(by_ang, old.x, old.y, old.z, ang, tol=1e-1)
                assert_vec(by_mat, old.x, old.y, old.z, ang, tol=1e-1)


def test_gen_check() -> None:
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

            mat = Matrix.from_angle(Angle(pit, yaw, roll))

            # Then check rotating vectors works correctly.
            # The engine actually gave us a right vector, so we need to flip that.
            assert_vec(X @ mat, for_x, for_y, for_z)
            assert_vec(Y @ mat, -left_x, -left_y, -left_z)
            assert_vec(Z @ mat, up_x, up_y, up_z)

            assert math.isclose(for_x, mat.aa, abs_tol=EPSILON)
            assert math.isclose(for_y, mat.ab, abs_tol=EPSILON)
            assert math.isclose(for_z, mat.ac, abs_tol=EPSILON)

            assert math.isclose(-left_x, mat.ba, abs_tol=EPSILON)
            assert math.isclose(-left_y, mat.bb, abs_tol=EPSILON)
            assert math.isclose(-left_z, mat.bc, abs_tol=EPSILON)

            assert math.isclose(up_x, mat.ca, abs_tol=EPSILON)
            assert math.isclose(up_y, mat.cb, abs_tol=EPSILON)
            assert math.isclose(up_z, mat.cc, abs_tol=EPSILON)

            # Also test Matrix.from_basis().
            x = Vec(for_x, for_y, for_z)
            y = -Vec(left_x, left_y, left_z)
            z = Vec(up_x, up_y, up_z)
            assert_rot(Matrix.from_basis(x=x, y=y, z=z), mat)
            assert_rot(Matrix.from_basis(x=x, y=y), mat)
            assert_rot(Matrix.from_basis(y=y, z=z), mat)
            assert_rot(Matrix.from_basis(x=x, z=z), mat)
