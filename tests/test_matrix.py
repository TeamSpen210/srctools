"""Tests the Matrix/FrozenMatrix class in srctools.math."""
import copy
import pickle

import pytest

from helpers import *


def test_matrix_constructor(py_c_vec: PyCVec, frozen_thawed_matrix: MatrixClass) -> None:
    """Test constructing matrices."""
    Matrix = frozen_thawed_matrix
    mat = Matrix()
    assert type(mat) is Matrix
    for x, y in itertools.product(range(3), range(3)):
        assert mat[x, y] == (1.0 if x == y else 0.0)

    mat_mutable = vec_mod.Matrix.from_angle(34.0, 29.0, 86.0)
    mat_frozen = vec_mod.FrozenMatrix.from_angle(34.0, 29.0, 86.0)
    copy_mutable = Matrix(mat_mutable)
    copy_frozen = Matrix(mat_frozen)

    assert type(copy_mutable) is Matrix
    assert type(copy_frozen) is Matrix

    assert copy_mutable is not mat_mutable

    for x, y in itertools.product(range(3), range(3)):
        assert copy_mutable[x, y] == mat_mutable[x, y]
        assert copy_frozen[x, y] == mat_frozen[x, y]

    # FrozenMatrix should not make a copy.
    # TODO: Cython doesn't let you override tp_new for this yet.
    if Matrix is vec_mod.Py_FrozenMatrix:
        assert mat_frozen is copy_frozen


def test_as_matrix(py_c_vec: PyCVec) -> None:
    """Test the as_matrix() method. """
    Matrix = vec_mod.Matrix
    FrozenMatrix = vec_mod.FrozenMatrix
    Angle = vec_mod.Angle
    FrozenAngle = vec_mod.FrozenAngle

    assert_rot(vec_mod.to_matrix(None), Matrix())
    ang = Angle(12, 34, -15)
    rot = Matrix.from_angle(ang)
    assert_rot(vec_mod.to_matrix(Matrix.from_angle(ang)), rot)
    assert_rot(vec_mod.to_matrix(FrozenMatrix.from_angle(ang)), rot)
    assert_rot(vec_mod.to_matrix(Angle(ang)), rot)
    assert_rot(vec_mod.to_matrix(FrozenAngle(ang)), rot)
    assert_rot(vec_mod.to_matrix((ang.pitch, ang.yaw, ang.roll)), rot)


# noinspection PyArgumentList
def test_bad_from_basis(
    py_c_vec: PyCVec,
    frozen_thawed_vec: VecClass,
    frozen_thawed_matrix: MatrixClass,
) -> None:
    """Test invalid arguments to Matrix.from_basis()"""
    Vec = frozen_thawed_vec
    Matrix = frozen_thawed_matrix

    v = Vec(0, 1, 0)
    with pytest.raises(TypeError):
        Matrix.from_basis()
    with pytest.raises(TypeError):
        Matrix.from_basis(x=v)
    with pytest.raises(TypeError):
        Matrix.from_basis(y=v)
    with pytest.raises(TypeError):
        Matrix.from_basis(z=v)


def test_matrix_no_iteration(py_c_vec: PyCVec) -> None:
    """Test that Matrix cannot be iterated, as this is rather useless."""
    Matrix = vec_mod.Matrix
    with pytest.raises(TypeError):
        iter(Matrix())
    with pytest.raises(TypeError):
        list(Matrix())


def text_invalid_from_angle(py_c_vec: PyCVec, frozen_thawed_matrix: MatrixClass) -> None:
    """Test invalid parameters passed to Matrix.from_angle()."""
    Matrix = frozen_thawed_matrix
    with pytest.raises(TypeError):
        Matrix.from_angle()
    with pytest.raises(TypeError):
        Matrix.from_angle(45.0)
    with pytest.raises(TypeError):
        Matrix.from_angle(45.0, 30.0)
    with pytest.raises(TypeError):
        Matrix.from_angle(roll=30.0)
    with pytest.raises(TypeError):
        Matrix.from_angle(yaw=30.0)
    with pytest.raises(TypeError):
        Matrix.from_angle(pitch=30.0, roll=12.5)


def test_copy(py_c_vec: PyCVec) -> None:
    """Test copying Matrixes."""
    Matrix = vec_mod.Matrix
    FrozenMatrix = vec_mod.FrozenMatrix

    # Some random rotation, so all points are different.
    test_data = (38, 42, 63)

    orig = Matrix.from_angle(*test_data)

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

    frozen = FrozenMatrix.from_angle(*test_data)
    # Copying FrozenMatrix does nothing.
    assert frozen is frozen.copy()
    assert frozen is copy.copy(frozen)
    assert frozen is copy.deepcopy(frozen)


def test_from_str(py_c_vec: PyCVec, frozen_thawed_matrix: MatrixClass) -> None:
    """Test the functionality of Matrix.from_angstr()."""
    Matrix = frozen_thawed_matrix
    for pit, yaw, rol in iter_vec(VALID_ZERONUMS):
        rot = Matrix.from_angle(pit, yaw, rol)
        assert_rot(Matrix.from_angstr(f'{pit} {yaw} {rol}'), rot)
        assert_rot(Matrix.from_angstr(f'<{pit} {yaw} {rol}>'), rot)
        # {x y z}
        assert_rot(Matrix.from_angstr(f'{{{pit} {yaw} {rol}}}'), rot)
        assert_rot(Matrix.from_angstr(f'({pit} {yaw} {rol})'), rot)
        assert_rot(Matrix.from_angstr(f'[{pit} {yaw} {rol}]'), rot)


def test_pickle(py_c_vec: PyCVec, frozen_thawed_matrix: MatrixClass) -> None:
    """Test pickling and unpickling matrices."""
    Matrix = frozen_thawed_matrix
    test_data = (38, 42, 63)

    orig = Matrix.from_angle(*test_data)

    pick = pickle.dumps(orig)
    thaw = pickle.loads(pick)

    assert orig is not thaw
    assert orig == thaw

    # Ensure we test the right frozen vs mutable class.
    CyMatrix = getattr(vec_mod, 'Cy_' + Matrix.__name__)
    PyMatrix = getattr(vec_mod, 'Py_' + Matrix.__name__)

    # Ensure both produce the same pickle - so they can be interchanged.
    # Copy over the floats, since calculations are going to be slightly different
    # due to optimisation etc. That's tested elsewhere to ensure accuracy, but
    # we need exact binary identity.
    data = [
        orig[x, y]
        for x in (0, 1, 2)
        for y in (0, 1, 2)
    ]
    cy_mat = CyMatrix._from_raw(*data)
    py_mat = PyMatrix._from_raw(*data)

    assert pickle.dumps(cy_mat) == pickle.dumps(py_mat)


def test_thaw_freezing(py_c_vec: PyCVec) -> None:
    """Test methods to convert between frozen <> mutable."""
    Matrix = vec_mod.Matrix
    FrozenMatrix = vec_mod.FrozenMatrix
    Angle = vec_mod.Angle
    # Other way around is not provided.
    with pytest.raises(AttributeError):
        Matrix.thaw()  # noqa
    with pytest.raises(AttributeError):
        FrozenMatrix.freeze()  # noqa

    ang = Angle(12.5, 34.3, -15.8)
    mut = Matrix.from_angle(ang)
    froze = mut.freeze()
    thaw = froze.thaw()

    assert_rot(mut, Matrix.from_angle(ang), type=Matrix)
    assert_rot(froze, Matrix.from_angle(ang), type=FrozenMatrix)
    assert_rot(thaw, Matrix.from_angle(ang), type=Matrix)
    # Test calling it on a temporary, in case this is optimised.
    assert_rot(Matrix.from_angle(ang).freeze(), Matrix.from_angle(ang), type=FrozenMatrix)
    assert_rot(FrozenMatrix.from_angle(ang).thaw(), Matrix.from_angle(ang), type=Matrix)



def test_matrix_inverse_known(py_c_vec: PyCVec) -> None:
    """Test the matrix inverse() method with a known inverse. """
    Matrix = vec_mod.Matrix

    # Test for matrix with known inverse
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = ( 1.0, -3.0,  7.0)
    mat[1, 0], mat[1, 1], mat[1, 2] = (-1.0,  4.0, -7.0)
    mat[2, 0], mat[2, 1], mat[2, 2] = (-1.0,  3.0, -6.0)

    correct = Matrix()
    correct[0, 0], correct[0, 1], correct[0, 2] = (-3.0,  3.0, -7.0)
    correct[1, 0], correct[1, 1], correct[1, 2] = ( 1.0,  1.0,  0.0)
    correct[2, 0], correct[2, 1], correct[2, 2] = ( 1.0,  0.0,  1.0)

    mat = mat.inverse()

    assert mat is not None

    for x, y in itertools.product(range(3), range(3)):
        assert abs(mat[x, y] - correct[x, y]) < 0.001

def test_matrix_inverse_fail(py_c_vec: PyCVec) -> None:
    """Test the matrix inverse() method for known failure. """
    Matrix = vec_mod.Matrix

    # Test for expected failure
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = (1, 2, 3)
    mat[1, 0], mat[1, 1], mat[1, 2] = (0, 0, 0)
    mat[2, 0], mat[2, 1], mat[2, 2] = (2, 4, 6)

    with pytest.raises(ArithmeticError):
        mat.inverse()
