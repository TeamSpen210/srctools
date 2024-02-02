from fractions import Fraction
import copy
import math
import pickle

from dirty_equals import IsFloat
import pytest

from helpers import *


VALID_NUMS = [
    # Make sure we use double-precision with the precise small value.
    30, 1.5, 0.2827, 282.34645456362782677821,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = [*VALID_NUMS, 0, -0]


def test_construction(frozen_thawed_angle: AngleClass) -> None:
    """Check the Angle() and FrozenAngle() constructors."""
    Angle = frozen_thawed_angle

    for pit, yaw, rol in iter_vec(VALID_ZERONUMS):
        assert_ang(Angle(pit, yaw, rol), pit, yaw, rol, type=Angle)
        assert_ang(Angle(pit, yaw), pit, yaw, 0, type=Angle)
        assert_ang(Angle(pit), pit, 0, 0, type=Angle)
        assert_ang(Angle(), 0, 0, 0, type=Angle)

        assert_ang(Angle([pit, yaw, rol]), pit, yaw, rol, type=Angle)
        assert_ang(Angle([pit, yaw], roll=rol), pit, yaw, rol, type=Angle)
        assert_ang(Angle([pit], yaw=yaw, roll=rol), pit, yaw, rol, type=Angle)
        assert_ang(Angle([pit]), pit, 0, 0, type=Angle)
        assert_ang(Angle([pit, yaw]), pit, yaw, 0, type=Angle)
        assert_ang(Angle([pit, yaw, rol]), pit, yaw, rol, type=Angle)


def test_copy(py_c_vec) -> None:
    """Test calling Angle() on an existing vec merely copies."""
    Angle = vec_mod.Angle
    FrozenAngle = vec_mod.FrozenAngle
    for pit, yaw, rol in iter_vec(VALID_ZERONUMS):
        # Test this does nothing (except copy).
        ang = Angle(pit, yaw, rol)
        ang2 = Angle(ang)
        assert_ang(ang2, pit, yaw, rol, type=Angle)
        assert ang is not ang2

        ang3 = Angle.copy(ang)
        assert_ang(ang3, pit, yaw, rol, type=Angle)
        assert ang is not ang3

        # Test doing the same with FrozenVec does not copy.
        fang = FrozenAngle(pit, yaw, rol)
        fang2 = FrozenAngle(ang)
        assert_ang(fang2, pit, yaw, rol, type=FrozenAngle)
        assert fang2 is not ang

        assert fang.copy() is fang

        # Ensure this doesn't mistakenly return the existing one.
        assert Angle(fang) is not fang
        # FrozenAngle should not make a copy.
        # TODO: Cython doesn't let you override tp_new for this yet.
        if FrozenAngle is vec_mod.Py_FrozenAngle:
            assert FrozenAngle(fang) is fang


def test_from_str(frozen_thawed_angle: AngleClass) -> None:
    """Test the functionality of Angle.from_str()."""
    Angle = frozen_thawed_angle
    for pit, yaw, rol in iter_vec(VALID_ZERONUMS):
        assert_ang(Angle.from_str(f'{pit} {yaw} {rol}'), pit, yaw, rol, type=Angle)
        assert_ang(Angle.from_str(f'<{pit} {yaw} {rol}>'), pit, yaw, rol, type=Angle)
        # {x y z}
        assert_ang(Angle.from_str(f'{{{pit} {yaw} {rol}}}'), pit, yaw, rol, type=Angle)
        assert_ang(Angle.from_str(f'({pit} {yaw} {rol})'), pit, yaw, rol, type=Angle)
        assert_ang(Angle.from_str(f'[{pit} {yaw} {rol}]'), pit, yaw, rol, type=Angle)

        # Test converting a converted Angle
        orig = Angle(pit, yaw, rol)
        new = Angle.from_str(Angle(pit, yaw, rol))
        assert_ang(new, pit, yaw, rol, type=Angle)
        assert orig is not new  # It must be a copy

        # Check as_tuple() makes an equivalent tuple - this is deprecated.
        with pytest.deprecated_call():
            tup = orig.as_tuple()

        # Flip to work arond the coercion.
        pit %= 360.0
        yaw %= 360.0
        rol %= 360.0

        assert isinstance(tup, tuple)
        assert (pit, yaw, rol) == tup
        assert hash((pit, yaw, rol)) == hash(tup)
        # Bypass subclass functions.
        assert tuple.__getitem__(tup, 0) == pit
        assert tuple.__getitem__(tup, 1) == yaw
        assert tuple.__getitem__(tup, 2) == rol

    # Check failures in Angle.from_str()
    # Note - does not pass through unchanged, they're converted to floats!
    for val in VALID_ZERONUMS:
        test_val = val % 360.0
        assert test_val == Angle.from_str('', pitch=val).pitch
        assert test_val == Angle.from_str('blah 4 2', yaw=val).yaw
        assert test_val == Angle.from_str('2 hi 2', pitch=val).pitch
        assert test_val == Angle.from_str('2 6 gh', roll=val).roll
        assert test_val == Angle.from_str('1.2 3.4', pitch=val).pitch
        assert test_val == Angle.from_str('34.5 38.4 -23 -38', roll=val).roll


def test_with_axes(frozen_thawed_angle: AngleClass) -> None:
    """Test the with_axes() constructor."""
    Angle = frozen_thawed_angle

    for axis, u, v in [
        ('pitch', 'yaw', 'roll'),
        ('yaw', 'pitch', 'roll'),
        ('roll', 'pitch', 'yaw'),
    ]:
        for num in VALID_ZERONUMS:
            test_num = num % 360

            ang = Angle.with_axes(axis, num)
            assert ang[axis] == test_num
            assert getattr(ang, axis) == test_num
            # Other axes are zero.
            assert ang[u] == 0
            assert ang[v] == 0
            assert getattr(ang, u) == 0
            assert getattr(ang, v) == 0

    for a, b, c in iter_vec(('pitch', 'yaw', 'roll')):
        if a == b or b == c or a == c:
            continue
        for x, y, z in iter_vec(VALID_ZERONUMS):
            ang = Angle.with_axes(a, x, b, y, c, z)

            x %= 360
            y %= 360
            z %= 360

            assert ang[a] == x
            assert ang[b] == y
            assert ang[c] == z

            assert getattr(ang, a) == x
            assert getattr(ang, b) == y
            assert getattr(ang, c) == z


def test_with_axes_conv(frozen_thawed_angle: AngleClass) -> None:
    """Test with_axes() converts values properly."""
    Angle = frozen_thawed_angle
    ang = Angle.with_axes('yaw', 8, 'roll', -45, 'pitch', 32)
    assert ang.pitch == IsFloat(exactly=32.0)
    assert ang.yaw == IsFloat(exactly=8.0)
    assert ang.roll == IsFloat(exactly=315.0)
    ang = Angle.with_axes('roll', Fraction(38, 25), 'pitch', Fraction(6876, 12), 'yaw', Fraction(-237, 16))
    assert ang.pitch == IsFloat(exactly=213.0)
    assert ang.yaw == IsFloat(exactly=345.1875)
    assert ang.roll == IsFloat(exactly=1.52)


def test_thaw_freezing(py_c_vec: PyCVec) -> None:
    """Test methods to convert between frozen <> mutable."""
    Angle = vec_mod.Angle
    FrozenAngle = vec_mod.FrozenAngle
    # Other way around is not provided.
    with pytest.raises(AttributeError):
        Angle.thaw()  # noqa
    with pytest.raises(AttributeError):
        FrozenAngle.freeze()  # noqa

    for p, y, r in iter_vec(VALID_ZERONUMS):
        mut = Angle(p, y, r)
        froze = mut.freeze()
        thaw = froze.thaw()

        assert_ang(mut, p, y, r, type=Angle)
        assert_ang(froze, p, y, r, type=FrozenAngle)
        assert_ang(thaw, p, y, r, type=Angle)
        # Test calling it on a temporary, in case this is optimised.
        assert_ang(Angle(p, y, r).freeze(), p, y, r, type=FrozenAngle)
        assert_ang(FrozenAngle(p, y, r).thaw(), p, y, r, type=Angle)


def test_angle_hash(py_c_vec: PyCVec) -> None:
    """Test hashing frozen angles."""
    Angle = vec_mod.Angle
    FrozenAngle = vec_mod.FrozenAngle

    with pytest.raises(TypeError):
        hash(Angle())

    for pitch, yaw, roll in iter_vec([0.0, 13, 25.8277, 128.474, 278.93]):
        expected = hash((round(pitch, 6), round(yaw, 6), round(roll, 6)))
        assert hash(FrozenAngle(pitch, yaw, roll)) == expected
        assert hash(FrozenAngle(pitch + 0.0000001, yaw + 0.0000001, roll + 0.0000001)) == expected


@pytest.mark.parametrize('axis, index, u, v, u_ax, v_ax', [
    ('pitch', 0, 'yaw', 'roll', 1, 2), ('yaw', 1, 'pitch', 'roll', 0, 2), ('roll', 2, 'pitch', 'yaw', 0, 1),
], ids=['pitch', 'yaw', 'roll'])
def test_attrs(py_c_vec, axis: str, index: int, u: str, v: str, u_ax: int, v_ax: int) -> None:
    """Test the pitch/yaw/roll attributes and item access."""
    Angle = vec_mod.Angle
    ang = Angle()
    # Should be constant.
    assert len(ang) == 3

    def check(targ: float, other: float):
        """Check all the indexes are correct."""
        assert math.isclose(getattr(ang, axis), targ), f'ang.{axis} != {targ}, other: {other}'
        assert math.isclose(getattr(ang, u), other), f'ang.{u} != {other}, targ={targ}'
        assert math.isclose(getattr(ang, v), other), f'ang.{v} != {other}, targ={targ}'

        assert math.isclose(ang[index], targ),  f'ang[{index}] != {targ}, other: {other}'
        assert math.isclose(ang[axis], targ), f'ang[{axis!r}] != {targ}, other: {other}'
        assert math.isclose(ang[u_ax], other),  f'ang[{u_ax!r}] != {other}, targ={targ}'
        assert math.isclose(ang[v_ax], other), f'ang[{v_ax!r}] != {other}, targ={targ}'
        assert math.isclose(ang[u], other),  f'ang[{u!r}] != {other}, targ={targ}'
        assert math.isclose(ang[v], other), f'[{v!r}] != {other}, targ={targ}'

    nums = [
        (0, 0.0),
        (38.29, 38.29),
        (-89.0, 271.0),
        (360.0, 0.0),
        (361.49, 1.49),
        (-725.87, 354.13),
    ]

    for oth_set, oth_read in nums:
        ang.pitch = ang.yaw = ang.roll = oth_set
        check(oth_read, oth_read)
        for x_set, x_read in nums:
            setattr(ang, axis, x_set)
            check(x_read, oth_read)


def test_iteration(py_c_vec: PyCVec, frozen_thawed_angle: AngleClass) -> None:
    """Test vector iteration."""
    Angle = frozen_thawed_angle
    v = Angle(45.0, 50, 65)
    it = iter(v)
    assert iter(it) is iter(it)

    assert next(it) == 45.0
    assert next(it) == 50.0
    assert next(it) == 65.0
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_rev_iteration(py_c_vec: PyCVec, frozen_thawed_angle: AngleClass) -> None:
    """Test reversed iteration."""
    Angle = frozen_thawed_angle
    v = Angle(45.0, 50, 65)
    it = reversed(v)
    assert iter(it) is iter(it)

    assert next(it) == 65.0
    assert next(it) == 50.0
    assert next(it) == 45.0
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_multiply(frozen_thawed_angle: AngleClass):
    """Test multiplying angles to scale them."""
    Angle = frozen_thawed_angle

    # Doesn't implement float(pit), and no other operators...
    obj = object()
    mutable = Angle is vec_mod.Angle

    for pit, yaw, rol in iter_vec(VALID_ZERONUMS):
        pit %= 360.0
        yaw %= 360.0
        rol %= 360.0
        for num in VALID_ZERONUMS:
            targ = Angle(pit, yaw, rol)
            fpit, fyaw, frol = (
                (num * pit) % 360.0,
                (num * yaw) % 360.0,
                (num * rol) % 360.0,
            )
            rpit, ryaw, rrol = (
                (pit * num) % 360,
                (yaw * num) % 360,
                (rol * num) % 360,
            )

            # Check forward and reverse fails.
            try:
                targ * obj  # noqa
            except TypeError:
                pass
            else:
                pytest.fail('Angle * scalar succeeded.')

            try:
                obj * targ  # noqa
            except TypeError:
                pass
            else:
                pytest.fail('scalar * Angle succeeded.')

            try:
                targ *= obj  # noqa
            except TypeError:
                pass
            else:
                pytest.fail('Angle *= scalar succeeded.')

            assert_ang(targ * num, rpit, ryaw, rrol)
            assert_ang(num * targ, fpit, fyaw, frol)

            # Ensure they haven't modified the original
            assert_ang(targ, pit, yaw, rol)

            res = targ
            res *= num
            assert_ang(
                res,
                rpit, ryaw, rrol,
                f'Return value for ({pit} {yaw} {rol}) *= {num}',
            )
            # Check that the original was or wasn't modified.
            if mutable:
                assert targ is res
                assert_ang(
                    targ,
                    rpit, ryaw, rrol,
                    f'Original for ({pit} {yaw} {rol}) *= {num}',
                )
            else:
                assert targ is not res
                assert_ang(
                    targ,
                    pit, yaw, rol,
                    f'Original for ({pit} {yaw} {rol}) *= {num}',
                )


def test_equality(py_c_vec, frozen_thawed_angle: AngleClass) -> None:
    """Test equality checks on Angles."""
    Angle = frozen_thawed_angle

    def test(p1: float, y1: float, r1: float, p2: float, y2: float, r2: float) -> None:
        """Check an Angle pair for incorrect comparisons."""
        ang1 = Angle(p1, y1, r1)
        ang2 = Angle(p2, y2, r2)

        equal = (abs(p1 - p2) % 360.0) < 1e-6 and (abs(y1 - y2) % 360.0) < 1e-6 and (abs(r1 - r2) % 360.0) < 1e-6

        comp = f'({p1} {y1} {r1}) ? ({p2} {y2} {r2})'

        assert (ang1 == ang2)         == equal, comp + ' ang == ang'
        assert (ang1 == (p2, y2, r2)) == equal, comp + ' ang == tup'
        assert ((p1, y1, r1) == ang2) == equal, comp + ' tup == ang'

        assert (ang1 != ang2)         != equal, comp + ' ang != ang'
        assert (ang1 != (p2, y2, r2)) != equal, comp + ' ang != tup'
        assert ((p1, y1, r1) != ang2) != equal, comp + ' tup != ang'

    # Test the absolute accuracy.
    values = [*VALID_ZERONUMS, 38.0, (38.0 + 1.1e6), (38.0 + 1e7)]

    for num in values:
        for num2 in values:
            # Test the whole comparison, then each axis pair seperately
            test(num, num, num, num2, num2, num2)
            test(0, num, num, num2, num2, num2)
            test(num, 0, num, num, num2, num2)
            test(num, num, 0, num2, num2, num2)
            test(num, num, num, 0, num2, num2)
            test(num, num, num, num, 0, num2)
            test(num, num, num, num, num, 0)

        # Test 360 wraps work.
        test(num, 0.0, 5.0 + num, num + 360.0, 0.0, num - 355.0)


def test_copy_mod(py_c_vec) -> None:
    """Test copying Angles and FrozenAngles."""
    Angle = vec_mod.Angle
    FrozenAngle = vec_mod.FrozenAngle

    test_data = 38.0, 257.125, 0.0

    orig = Angle(test_data)

    cpy_meth = orig.copy()

    assert orig is not cpy_meth  # Must be a new object.
    assert cpy_meth is not orig.copy()  # Cannot be cached
    assert orig == cpy_meth  # Numbers must be exactly identical!

    cpy = copy.copy(orig)

    assert orig is not cpy
    assert cpy_meth is not copy.copy(orig)
    assert orig == cpy

    dcpy = copy.deepcopy(orig)

    assert orig is not dcpy
    assert orig == dcpy

    frozen = FrozenAngle(test_data)
    # Copying FrozenAngle does nothing.
    assert frozen is frozen.copy()
    assert frozen is copy.copy(frozen)
    assert frozen is copy.deepcopy(frozen)


def test_pickle(frozen_thawed_angle: AngleClass) -> None:
    """Test pickling and unpickling works."""
    Angle = frozen_thawed_angle
    test_data = 38.0, 257.125, 0.0

    orig = Angle(test_data)
    pick = pickle.dumps(orig)
    thaw = pickle.loads(pick)

    assert orig is not thaw
    assert orig == thaw

    # Ensure both produce the same pickle - so they can be interchanged.
    cy_pick = pickle.dumps(getattr(vec_mod, 'Cy_' + Angle.__name__)(test_data))
    py_pick = pickle.dumps(getattr(vec_mod, 'Py_' + Angle.__name__)(test_data))

    assert cy_pick == py_pick == pick
