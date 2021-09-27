"""Test rotations in srctools.vec."""
import copy
import pickle
from pathlib import Path
from typing import NamedTuple, List

import pytest

from srctools.test import *
from srctools import Vec


class RotationData(NamedTuple):
    """Result of the rotation_data fixture."""
    angle: Tuple[float, float, float]

    for_x: float
    for_y: float
    for_z: float

    left_x: float
    left_y: float
    left_z: float

    up_x: float
    up_y: float
    up_z: float


# TODO: pytest-datadir doesn't have session-scope fixture.
@pytest.fixture(scope='session')
def rotation_data() -> List[RotationData]:
    """Parse the rotation data dumped from the engine, used to check our math."""
    data = []
    with (Path(__file__).with_suffix('') / 'rotation.txt').open() as f:
        for line in f:
            if not line.startswith('|'):
                # Skip other junk in the log.
                continue
            (
                pit, yaw, roll,
                for_x, for_y, for_z,
                right_x, right_y, right_z,
                up_x, up_y, up_z
            ) = map(float, line[1:].split())
            # The engine actually gave us a right vector, so we need to flip that.
            left_x, left_y, left_z = -right_x, -right_y, -right_z
            data.append(RotationData(
                (pit, yaw, roll),
                for_x, for_y, for_z,
                left_x, left_y, left_z,
                up_x, up_y, up_z
            ))
    return data


def test_vec_identities(py_c_vec: PyCVec) -> None:
    """Check that vectors in the same axis as the rotation don't get spun."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for ang in range(0, 360, 13):
        # Check the two constructors match.
        assert_rot(Matrix.from_pitch(ang), Matrix.from_angle(Angle(pitch=ang)))
        assert_rot(Matrix.from_yaw(ang), Matrix.from_angle(Angle(yaw=ang)))
        assert_rot(Matrix.from_roll(ang), Matrix.from_angle(Angle(roll=ang)))

        # Various magnitudes to test
        for mag in (-250, -1, 0, 1, 250):
            assert_vec(Vec(y=mag) @ Matrix.from_pitch(ang), 0, mag, 0)
            assert_vec(Vec(z=mag) @ Matrix.from_yaw(ang), 0, 0, mag)
            assert_vec(Vec(x=mag) @ Matrix.from_roll(ang), mag, 0, 0)


def test_vec_basic_yaw(py_c_vec: PyCVec) -> None:
    """Check each direction rotates appropriately in yaw."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(0), 200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(0), 0, 150, 0)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(90), 0, 200, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(90), -150, 0, 0)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(180), -200, 0, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(180), 0, -150, 0)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_yaw(270), 0, -200, 0)
    assert_vec(Vec(0, 150, 0) @ Matrix.from_yaw(270), 150, 0, 0)


def test_vec_basic_pitch(py_c_vec: PyCVec) -> None:
    """Check each direction rotates appropriately in pitch."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(0), 200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(0), 0, 0, 150)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(90), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(90), 150, 0, 0)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(180), -200, 0, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(180), 0, 0, -150)

    assert_vec(Vec(200, 0, 0) @ Matrix.from_pitch(270), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_pitch(270), -150, 0, 0)


def test_vec_basic_roll(py_c_vec: PyCVec) -> None:
    """Check each direction rotates appropriately in roll."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(0), 0, 200, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(0), 0, 0, 150)

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(90), 0, 0, 200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(90), 0, -150, 0)

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(180), 0, -200, 0)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(180), 0, 0, -150)

    assert_vec(Vec(0, 200, 0) @ Matrix.from_roll(270), 0, 0, -200)
    assert_vec(Vec(0, 0, 150) @ Matrix.from_roll(270), 0, 150, 0)


def test_ang_matrix_roundtrip(py_c_vec: PyCVec) -> None:
    """Check converting to and from a Matrix does not change values."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for p, y, r in iter_vec(range(0, 360, 90)):
        vert = (Vec(x=1) @ Angle(p, y, r)).z
        if vert < 0.99 or vert > 0.99:
            # If nearly vertical, gimbal lock prevents roundtrips.
            continue
        mat = Matrix.from_angle(Angle(p, y, r))
        assert_ang(mat.to_angle(), p, y, r)


def test_to_angle_roundtrip(py_c_vec: PyCVec) -> None:
    """Check Vec.to_angle() roundtrips."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for x, y, z in iter_vec((-1, 0, 1)):
        if x == y == z == 0:
            continue
        norm = Vec(x, y, z).norm()
        ang = norm.to_angle()
        assert_vec(Vec(x=1) @ ang, norm.x, norm.y, norm.z, ang)


def test_matrix_roundtrip_pitch(py_c_vec: PyCVec) -> None:
    """Check converting to and from a Matrix does not change values."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

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


def test_matrix_roundtrip_yaw(py_c_vec: PyCVec) -> None:
    """Check converting to and from a Matrix does not change values."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for yaw in range(0, 360, 45):
        mat = Matrix.from_yaw(yaw)
        assert_ang(mat.to_angle(), 0, yaw, 0)


def test_matrix_roundtrip_roll(py_c_vec: PyCVec) -> None:
    """Check converting to and from a Matrix does not change values."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for roll in range(0, 360, 45):
        if roll in (90, -90):
            # Don't test gimbal lock.
            continue
        mat = Matrix.from_roll(roll)
        assert_ang(mat.to_angle(), 0, 0, roll)


def test_single_axis(py_c_vec: PyCVec) -> None:
    """In each axis, two rotations should be the same as adding."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

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


def test_axis_angle(py_c_vec: PyCVec) -> None:
    """Test Matrix.axis_angle() computes the 6 basis vectors correctly."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    def test(axis, equiv_ang: Py_Angle):
        for ang in range(0, 360, 15):
            assert_rot(Matrix.axis_angle(axis, ang), Matrix.from_angle(ang * equiv_ang), f'{axis} * {ang} != {equiv_ang}')
            # Inverse axis = reversed rotation.
            assert_rot(Matrix.axis_angle(-axis, ang), Matrix.from_angle(-ang * equiv_ang), f'{-axis} * {ang} != {equiv_ang}')
    test(Vec(1, 0, 0), Angle(0, 0, 1))
    test(Vec(0, 1, 0), Angle(1, 0, 0))
    test(Vec(0, 0, 1), Angle(0, 1, 0))


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


def test_old_rotation(py_c_vec: PyCVec) -> None:
    """Verify that the code matches the results from the earlier Vec.rotate code."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for pitch in range(0, 360, 15):
        for yaw in range(0, 360, 15):
            for roll in range(0, 360, 15):
                ang = Angle(pitch, yaw, roll)
                mat = Matrix.from_angle(ang)

                # Construct a matrix directly from 3 vector rotations.
                old_mat = Matrix()
                old_mat[0, 0], old_mat[0, 1], old_mat[0, 2] = old_rotate(Vec(x=1), pitch, yaw, roll)
                old_mat[1, 0], old_mat[1, 1], old_mat[1, 2] = old_rotate(Vec(y=1), pitch, yaw, roll)
                old_mat[2, 0], old_mat[2, 1], old_mat[2, 2] = old_rotate(Vec(z=1), pitch, yaw, roll)

                assert_rot(mat, old_mat, ang)
                old = old_rotate(Vec(128, 0, 0), pitch, yaw, roll)

                by_ang = Vec(128, 0, 0) @ ang
                by_mat = Vec(128, 0, 0) @ mat
                assert_vec(by_ang, old.x, old.y, old.z, ang, tol=1e-1)
                assert_vec(by_mat, old.x, old.y, old.z, ang, tol=1e-1)


# noinspection PyArgumentList
def test_bad_from_basis(py_c_vec: PyCVec) -> None:
    """Test invalid arguments to Matrix.from_basis()"""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    v = Vec(0, 1, 0)
    with pytest.raises(TypeError):
        Matrix.from_basis()
    with pytest.raises(TypeError):
        Matrix.from_basis(x=v)
    with pytest.raises(TypeError):
        Matrix.from_basis(y=v)
    with pytest.raises(TypeError):
        Matrix.from_basis(z=v)


def test_rotating_vectors(
    py_c_vec: PyCVec,
    rotation_data: List[RotationData],
) -> None:
    """Test our rotation code with engine rotation data."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    X = Vec(x=1)
    Y = Vec(y=1)
    Z = Vec(z=1)

    for data in rotation_data:
        mat = Matrix.from_angle(*data.angle)
        assert_rot(mat, Matrix.from_angle(Angle(data.angle)))

        # Check rotating vectors works correctly.
        assert_vec(X @ mat, data.for_x, data.for_y, data.for_z)
        assert_vec(Y @ mat, data.left_x, data.left_y, data.left_z)
        assert_vec(Z @ mat, data.up_x, data.up_y, data.up_z)


def test_matmul_direct(py_c_vec: PyCVec) -> None:
    """Test that directly calling the magic methods produces the right results.

    Normally __rmatmul__ isn't going to be called, so it may be incorrect.
    """
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    vec = Vec(34, 72, -10)
    ang = Angle(10, 30, 70)
    ang2 = Angle(56, -10, 25)
    mat = Matrix.from_angle(ang)
    mat2 = Matrix.from_angle(ang2)

    assert vec.__matmul__(mat) == mat.__rmatmul__(vec) == vec @ mat, vec @ mat
    assert vec.__matmul__(ang) == ang.__rmatmul__(vec) == vec @ ang, vec @ ang
    assert ang.__matmul__(ang2) == ang2.__rmatmul__(ang) == ang @ ang2, ang @ ang2
    assert ang.__matmul__(mat) == mat.__rmatmul__(ang) == ang @ mat, ang @ mat
    assert mat.__matmul__(mat2) == mat2.__rmatmul__(mat) == mat @ mat2, mat @ mat2


def test_inplace_rotation(py_c_vec: PyCVec) -> None:
    """Test inplace rotation operates correctly."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    vec = Vec(34, 72, -10)
    ang = Angle(10, 30, 70)
    ang2 = Angle(56, -10, 25)
    mat = Matrix.from_angle(ang)
    mat2 = Matrix.from_angle(ang2)

    v = vec.copy()
    v @= ang
    assert v == vec @ ang

    a = ang.copy()
    a @= ang2
    assert a == ang @ ang2

    a = ang.copy()
    a @= mat2
    assert a == ang @ mat2

    m = mat.copy()
    m @= mat2
    assert m == mat @ mat2


def test_matrix_getters(
    py_c_vec: PyCVec,
    rotation_data: List[RotationData],
) -> None:
    """Test functions which return the basis vectors for the matrix."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    for data in rotation_data:
        mat = Matrix.from_angle(*data.angle)

        assert_vec(mat.forward(), data.for_x, data.for_y, data.for_z)
        assert_vec(mat.left(), data.left_x, data.left_y, data.left_z)
        assert_vec(mat.up(), data.up_x, data.up_y, data.up_z)


@pytest.mark.parametrize('mag', [-5.0, 1.0, -1.0, 0.0, 12.45, -28.37])
def test_matrix_getters_with_mag(
    py_c_vec: PyCVec,
    rotation_data: List[RotationData],
    mag: float,
) -> None:
    """Test computing the basis vector with a magnitude."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    for data in rotation_data:
        mat = Matrix.from_angle(*data.angle)

        assert_vec(mat.forward(mag), mag * data.for_x, mag * data.for_y, mag * data.for_z, tol=1e-3)
        assert_vec(mat.left(mag), mag * data.left_x, mag * data.left_y, mag * data.left_z, tol=1e-3)
        assert_vec(mat.up(mag), mag * data.up_x, mag * data.up_y, mag * data.up_z, tol=1e-3)


def test_rotating_vec_tuples(
    py_c_vec: PyCVec,
    rotation_data: List[RotationData],
) -> None:
    """Test rotation is permitted with 3-tuples"""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for data in rotation_data:
        ang = Angle(data.angle)
        mat = Matrix.from_angle(ang)

        assert_vec((1, 0, 0) @ mat, data.for_x, data.for_y, data.for_z)
        assert_vec((0.0, 1.0, 0.0) @ mat, data.left_x, data.left_y, data.left_z)
        assert_vec((0.0, 0.0, 1.0) @ mat, data.up_x, data.up_y, data.up_z)

        assert_vec((1, 0, 0) @ ang, data.for_x, data.for_y, data.for_z)
        assert_vec((0.0, 1.0, 0.0) @ ang, data.left_x, data.left_y, data.left_z)
        assert_vec((0.0, 0.0, 1.0) @ ang, data.up_x, data.up_y, data.up_z)


def test_rotated_matrix_data(
    py_c_vec: PyCVec,
    rotation_data: List[RotationData],
) -> None:
    """Test our rotation code with engine rotation data."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    X = Vec(x=1)
    Y = Vec(y=1)
    Z = Vec(z=1)

    for data in rotation_data:
        mat = Matrix.from_angle(*data.angle)
        assert_rot(mat, Matrix.from_angle(Angle(data.angle)))

        assert math.isclose(data.for_x, mat[0, 0], abs_tol=EPSILON)
        assert math.isclose(data.for_y, mat[0, 1], abs_tol=EPSILON)
        assert math.isclose(data.for_z, mat[0, 2], abs_tol=EPSILON)

        assert math.isclose(data.left_x, mat[1, 0], abs_tol=EPSILON)
        assert math.isclose(data.left_y, mat[1, 1], abs_tol=EPSILON)
        assert math.isclose(data.left_z, mat[1, 2], abs_tol=EPSILON)

        assert math.isclose(data.up_x, mat[2, 0], abs_tol=EPSILON)
        assert math.isclose(data.up_y, mat[2, 1], abs_tol=EPSILON)
        assert math.isclose(data.up_z, mat[2, 2], abs_tol=EPSILON)


def test_from_basis_w_engine_data(py_c_vec: PyCVec, rotation_data) -> None:
    """Test matrix.from_basis() reproduces the matrix."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    for data in rotation_data:
        mat = Matrix.from_angle(*data.angle)

        x = Vec(data.for_x, data.for_y, data.for_z)
        y = Vec(data.left_x, data.left_y, data.left_z)
        z = Vec(data.up_x, data.up_y, data.up_z)
        assert_rot(Matrix.from_basis(x=x, y=y, z=z), mat)
        assert_rot(Matrix.from_basis(x=x, y=y), mat)
        assert_rot(Matrix.from_basis(y=y, z=z), mat)
        assert_rot(Matrix.from_basis(x=x, z=z), mat)

        # Angle.from_basis() == Matrix.from_basis().to_angle().
        assert_ang(Angle.from_basis(x=x, y=y, z=z), *Matrix.from_basis(x=x, y=y, z=z).to_angle())
        assert_ang(Angle.from_basis(x=x, y=y), *Matrix.from_basis(x=x, y=y).to_angle())
        assert_ang(Angle.from_basis(y=y, z=z), *Matrix.from_basis(y=y, z=z).to_angle())
        assert_ang(Angle.from_basis(x=x, z=z), *Matrix.from_basis(x=x, z=z).to_angle())


def test_vec_cross_w_engine_data(py_c_vec: PyCVec, rotation_data) -> None:
    """Test Vec.cross() with engine rotation data."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    for data in rotation_data:
        x = Vec(data.for_x, data.for_y, data.for_z)
        y = Vec(data.left_x, data.left_y, data.left_z)
        z = Vec(data.up_x, data.up_y, data.up_z)

        assert_vec(Vec.cross(x, y), data.up_x, data.up_y, data.up_z, tol=1e-5)
        assert_vec(Vec.cross(y, z), data.for_x, data.for_y, data.for_z, tol=1e-5)
        assert_vec(Vec.cross(x, z), -data.left_x, -data.left_y, -data.left_z, tol=1e-5)

        assert_vec(Vec.cross(y, x), -data.up_x, -data.up_y, -data.up_z, tol=1e-5)
        assert_vec(Vec.cross(z, y), -data.for_x, -data.for_y, -data.for_z, tol=1e-5)
        assert_vec(Vec.cross(z, x), data.left_x, data.left_y, data.left_z, tol=1e-5)

        assert_vec(Vec.cross(x, x), 0, 0, 0)
        assert_vec(Vec.cross(y, y), 0, 0, 0)
        assert_vec(Vec.cross(z, z), 0, 0, 0)


def test_copy_pickle(py_c_vec: PyCVec) -> None:
    """Test pickling, unpickling and copying Matrixes."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    vec_mod.Matrix = Matrix

    # Some random rotation, so all points are different.
    test_data = (38, 42, 63)

    orig = Matrix.from_angle(Angle(*test_data))

    cpy_meth = orig.copy()

    assert orig is not cpy_meth  # Must be a new object.
    assert cpy_meth is not orig.copy()  # Cannot be cached
    assert type(orig) is type(cpy_meth)
    assert orig == cpy_meth  # Numbers must be exactly identical!

    cpy = copy.copy(orig)

    assert orig is not cpy
    assert cpy_meth is not copy.copy(orig)
    assert orig == cpy

    dcpy = copy.deepcopy(orig)

    assert orig is not dcpy
    assert orig == dcpy

    pick = pickle.dumps(orig)
    thaw = pickle.loads(pick)

    assert orig is not thaw
    assert orig == thaw

    # Ensure both produce the same pickle - so they can be interchanged.
    # Copy over the floats, since calculations are going to be slightly different
    # due to optimisation etc. That's tested elsewhere to ensure accuracy, but
    # we need exact binary identity.
    cy_mat = Cy_Matrix()
    py_mat = Py_Matrix()
    for x in (0, 1, 2):
        for y in (0, 1, 2):
            cy_mat[x, y] = py_mat[x, y] = orig[x, y]

    assert pickle.dumps(cy_mat) == pickle.dumps(py_mat) == pick
