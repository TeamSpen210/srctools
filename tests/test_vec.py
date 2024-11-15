"""Test the Vector object."""
from typing import Union
from typing_extensions import Literal, TypeAlias
from fractions import Fraction
from pathlib import Path
from random import Random
import copy
import inspect
import math
import operator as op
import pickle
import re

from dirty_equals import IsFloat
import pytest

from helpers import *
from srctools import Vec_tuple, math as vec_mod


try:
    # noinspection PyProtectedMember
    from srctools import _math as cy_math_mod
except ImportError:
    cy_math_mod = None


# Reuse these context managers.
raises_typeerror = pytest.raises(TypeError)
raises_keyerror = pytest.raises(KeyError)
raises_zero_div = pytest.raises(ZeroDivisionError)
Axis: TypeAlias = Literal["x", "y", "z"]


@pytest.mark.parametrize('cls', ['Vec', 'FrozenVec', 'Matrix', 'Angle', 'FrozenAngle'])
def test_matching_apis(cls: str) -> None:
    """Check each class pair has the same methods."""
    py_type = getattr(vec_mod, 'Py_' + cls)
    cy_type = getattr(vec_mod, 'Cy_' + cls)
    if py_type is cy_type:
        pytest.fail(f'No Cython version of {cls}!')

    # Skip private attributes - not part of the API.
    # For dunders, we'd like to match but C slots make that complicated - mapping vs sequence etc.
    # So check that via our other tests.
    # Use inspect() to include parent classes properly.
    py_attrs = {name for name, _ in inspect.getmembers(py_type) if not name.startswith('_')}
    cy_attrs = {name for name, _ in inspect.getmembers(cy_type) if not name.startswith('_')}
    assert cy_attrs == py_attrs


@pytest.mark.parametrize('cls_name', ['AngleBase', 'VecBase', 'MatrixBase'])
def test_noconstruct_base(py_c_vec, cls_name: str) -> None:
    """Test the internal base classes cannot be instantiated."""
    cls = getattr(vec_mod, cls_name)
    with pytest.raises(TypeError):
        cls()


@parameterize_cython('lerp_func', vec_mod.Py_lerp, vec_mod.Cy_lerp)
def test_scalar_lerp(lerp_func) -> None:
    """Test the lerp function."""
    assert lerp_func(-4.0, -4.0, 10, 50.0, 80.0) == pytest.approx(50.0)
    assert lerp_func(10.0, -4.0, 10, 50.0, 80.0) == pytest.approx(80.0)
    assert lerp_func(2.0, 0.0, 10.0, 50.0, 40.0) == pytest.approx(48.0)
    assert lerp_func(5.0, 0.0, 10.0, 50.0, 40.0) == pytest.approx(45.0)
    assert lerp_func(-10, 0, 10, 8, 9) == pytest.approx(7.0)
    assert lerp_func(15, 0, 10, 8, 9) == pytest.approx(9.5)

    with raises_zero_div:
        lerp_func(30.0, 45.0, 45.0, 80, 90)  # In is equal


def test_construction(py_c_vec, frozen_thawed_vec):
    """Check various parts of the constructor.

    This tests Vec(), Vec.from_str() and parse_vec_str().
    """
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_ZERONUMS):
        assert_vec(Vec(x, y, z), x, y, z)
        assert_vec(Vec(x, y), x, y, 0)
        assert_vec(Vec(x), x, 0, 0)
        assert_vec(Vec(), 0, 0, 0)

        assert_vec(Vec([x, y, z]), x, y, z)
        assert_vec(Vec([x, y], z=z), x, y, z)
        assert_vec(Vec([x], y=y, z=z), x, y, z)
        assert_vec(Vec([x]), x, 0, 0)
        assert_vec(Vec([]), 0, 0, 0)
        assert_vec(Vec([x, y]), x, y, 0)
        assert_vec(Vec([x, y, z]), x, y, z)


def test_vec_copying(py_c_vec):
    """Test calling Vec() on an existing vec merely copies."""
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec

    for x, y, z in iter_vec(VALID_ZERONUMS):
        # Test this does nothing (except copy).
        v = Vec(x, y, z)
        v2 = Vec(v)
        assert_vec(v2, x, y, z)
        assert v is not v2

        v3 = Vec.copy(v)
        assert_vec(v3, x, y, z)
        assert v is not v3

        # Test doing the same with FrozenVec does not copy.
        fv = FrozenVec(x, y, z)
        fv2 = FrozenVec(v)
        assert_vec(fv2, x, y, z)
        assert fv2 is not v

        assert fv.copy() is fv

        # Ensure this doesn't mistakenly return the existing one.
        assert Vec(fv) is not fv
        # FrozenVec should not make a copy.
        # TODO: Cython doesn't let you override tp_new for this yet.
        if FrozenVec is vec_mod.Py_FrozenVec:
            assert FrozenVec(fv) is fv


def test_vec_from_str(py_c_vec, frozen_thawed_vec: VecClass) -> None:
    """Test the functionality of Vec.from_str() and parse_vec_str()."""
    parse_vec_str = vec_mod.parse_vec_str
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_ZERONUMS):
        # Test Vec.from_str()
        assert_vec(Vec.from_str(f'{x} {y} {z}'), x, y, z)
        assert_vec(Vec.from_str(f'<{x} {y} {z}>'), x, y, z)
        assert_vec(Vec.from_str(f'{{{x} {y} {z}}}'), x, y, z)
        assert_vec(Vec.from_str(f'({x} {y} {z})'), x, y, z)
        assert_vec(Vec.from_str(f'[{x} {y} {z}]'), x, y, z)

        # And parse_vec_str
        v = Vec(x, y, z)
        assert_vec(v, *parse_vec_str(f'{x} {y} {z}'))
        assert_vec(v, *parse_vec_str(f'<{x} {y} {z}>'))

        assert_vec(v, *parse_vec_str(f'{{{x} {y} {z}}}'))
        assert_vec(v, *parse_vec_str(f'({x} {y} {z})'))
        assert_vec(v, *parse_vec_str(f'[{x} {y} {z}]'))

        parse_res = parse_vec_str(f'{x} {y} {z}')
        assert isinstance(parse_res, tuple)
        assert parse_res == (x, y, z)

        # Test converting a converted Vec
        orig = Vec(x, y, z)
        new = Vec.from_str(Vec(x, y, z))
        assert_vec(new, x, y, z)
        assert orig is not new  # It must be a copy


def test_vec_as_tuple(frozen_thawed_vec: VecClass) -> None:
    """Test the functionality of Vec.as_tuple()."""
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_ZERONUMS):
        # Check as_tuple() makes an equivalent tuple
        orig = Vec(x, y, z)
        x = round(x, 6)
        y = round(y, 6)
        z = round(z, 6)
        with pytest.deprecated_call():
            tup = orig.as_tuple()
        assert isinstance(tup, tuple)
        assert (x, y, z) == tup
        assert hash((x, y, z)) == hash(tup)
        assert tup[0] == x
        assert tup[1] == y
        assert tup[2] == z
        # Bypass subclass functions.
        assert tuple.__getitem__(tup, 0) == x
        assert tuple.__getitem__(tup, 1) == y
        assert tuple.__getitem__(tup, 2) == z
        assert tup.x == x
        assert tup.y == y
        assert tup.z == z


def test_from_str_fails(py_c_vec, frozen_thawed_vec: VecClass) -> None:
    """Check failures in Vec.from_str()"""
    # Note - does not pass defaults through unchanged, they're converted to floats!
    parse_vec_str = vec_mod.parse_vec_str
    Vec = frozen_thawed_vec
    for val in VALID_ZERONUMS:
        assert val == parse_vec_str('', x=val)[0]
        assert val == parse_vec_str('blah 4 2', y=val)[1]
        assert val == parse_vec_str('2 hi 2', x=val)[0]
        assert val == parse_vec_str('2 6 gh', z=val)[2]
        assert val == parse_vec_str('1.2 3.4', x=val)[0]
        assert val == parse_vec_str('34.5 38.4 -23 -38', z=val)[2]

        assert val == Vec.from_str('', x=val).x
        assert val == Vec.from_str('blah 4 2', y=val).y
        assert val == Vec.from_str('2 hi 2', x=val).x
        assert val == Vec.from_str('2 6 gh', z=val).z
        assert val == Vec.from_str('1.2 3.4', x=val).x
        assert val == Vec.from_str('34.5 38.4 -23 -38', z=val).z


def test_thaw_freezing(py_c_vec: PyCVec):
    """Test methods to convert between frozen <> mutable."""
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec
    # Other way around is not provided.
    with pytest.raises(AttributeError):
        Vec.thaw()
    with pytest.raises(AttributeError):
        FrozenVec.freeze()

    for x, y, z in iter_vec(VALID_ZERONUMS):
        mut = Vec(x, y, z)
        froze = mut.freeze()
        thaw = froze.thaw()
        assert isinstance(froze, FrozenVec)
        assert isinstance(mut, Vec)
        assert isinstance(thaw, Vec)

        assert_vec(mut, x, y, z)
        assert_vec(froze, x, y, z)
        assert_vec(thaw, x, y, z)
        # Test calling it on a temporary, in case this is optimised.
        assert_vec(Vec(x, y, z).freeze(), x, y, z)
        assert_vec(FrozenVec(x, y, z).thaw(), x, y, z)


@pytest.mark.parametrize('value', [
    '0.4 2.5 3.9',
    '[0.4 2.5 3.9]',
    '[   0.4 2.5 3.9 ]',
])
def test_spaced_parsing(py_c_vec, value):
    """Test various edge cases regarding parsing."""
    parse_vec_str = vec_mod.parse_vec_str
    x, y, z = parse_vec_str(value, 1, 2, 3)
    assert x == 0.4
    assert y == 2.5
    assert z == 3.9


def test_parse_vec_passthrough(py_c_vec):
    """Test that non-floats can be given to parse_vec_str()."""
    parse_vec_str = vec_mod.parse_vec_str
    obj1, obj2, obj3 = object(), object(), object()
    assert parse_vec_str('1 2 3', obj1, obj2, obj3) == (1, 2, 3)
    assert parse_vec_str('fail', obj1, obj2, obj3) == (obj1, obj2, obj3)
    assert parse_vec_str(range, obj1, obj2, obj3) == (obj1, obj2, obj3)


def test_with_axes(frozen_thawed_vec: VecClass) -> None:
    """Test the with_axes() constructor."""
    Vec = frozen_thawed_vec
    for axis, u, v in ['xyz', 'yxz', 'zxy']:
        for num in VALID_ZERONUMS:
            vec = Vec.with_axes(axis, num)
            assert vec[axis] == num
            # Other axes are zero.
            assert vec[u] == 0
            assert vec[v] == 0

            vec2 = Vec.with_axes(axis, vec)
            assert vec2[axis] == num
            assert vec2[u] == 0
            assert vec2[v] == 0

    for a, b, c in iter_vec('xyz'):
        if a == b or b == c or a == c:
            continue
        for x, y, z in iter_vec(VALID_ZERONUMS):
            vec = Vec.with_axes(a, x, b, y, c, z)
            msg = f'{vec} = {a}={x}, {b}={y}, {c}={z}'
            assert vec[a] == x, msg
            assert vec[b] == y, msg
            assert vec[c] == z, msg

            vec2 = Vec.with_axes(a, vec, b, vec, c, vec)
            msg = f'{vec2} = {a}={x}, {b}={y}, {c}={z}'
            assert vec2[a] == x, msg
            assert vec2[b] == y, msg
            assert vec2[c] == z, msg


def test_with_axes_conv(frozen_thawed_vec: VecClass) -> None:
    """Test with_axes() converts values properly."""
    Vec = frozen_thawed_vec
    vec = Vec.with_axes('y', 8, 'z', -45, 'x', 32)
    assert vec.x == IsFloat(exactly=32.0)
    assert vec.y == IsFloat(exactly=8.0)
    assert vec.z == IsFloat(exactly=-45.0)
    vec = Vec.with_axes('z', Fraction(8, 2), 'x', Fraction(1, 4), 'y', Fraction(-23, 16))
    assert vec.x == IsFloat(exactly=0.25)
    assert vec.y == IsFloat(exactly=-1.4375)
    assert vec.z == IsFloat(exactly=4.0)


@pytest.mark.parametrize('clsname', ['Vec', 'FrozenVec', 'Angle', 'FrozenAngle'])
def test_vec_ang_stringification(py_c_vec, clsname: str) -> None:
    """Test string methods for both angles and vectors."""
    cls: Union[VecClass, AngleClass] = getattr(vec_mod, clsname)
    # Test:
    # Zeros, float in each pos, rounding down to int, up to int, format with more decimal places,
    # format with less.
    obj = cls(0.0, 0.0, 0.0)
    assert str(obj) == '0 0 0'
    assert repr(obj) == f'{clsname}(0, 0, 0)'
    assert obj.join() == '0, 0, 0'
    assert obj.join(':') == '0:0:0'
    assert format(obj) == str(obj)
    assert format(obj, '.2%') == '0.00% 0.00% 0.00%'

    # Test each axis individually, to ensure no mixups are made.
    obj = cls(3.14, 0.0, 0.0)
    assert str(obj) == '3.14 0 0'
    assert repr(obj) == f'{clsname}(3.14, 0, 0)'
    assert obj.join() == '3.14, 0, 0'
    assert obj.join(':') == '3.14:0:0'
    assert format(obj) == str(obj)
    assert format(obj, '.2%') == '314.00% 0.00% 0.00%'

    obj = cls(0.0, 3.14, 0.0)
    assert str(obj) == '0 3.14 0'
    assert repr(obj) == f'{clsname}(0, 3.14, 0)'
    assert obj.join() == '0, 3.14, 0'
    assert obj.join(':') == '0:3.14:0'
    assert format(obj) == str(obj)
    assert format(obj, '.2%') == '0.00% 314.00% 0.00%'

    obj = cls(0.0, 0.0, 3.14)
    assert str(obj) == '0 0 3.14'
    assert repr(obj) == f'{clsname}(0, 0, 3.14)'
    assert obj.join() == '0, 0, 3.14'
    assert obj.join(':') == '0:0:3.14'
    assert format(obj) == str(obj)
    assert format(obj, '.2%') == '0.00% 0.00% 314.00%'

    # Now test a value that will round up both to zero and an integer.
    # -0.001 rounds to 360.
    if clsname.endswith('Angle'):
        obj = cls(-0.00000012, 37.9999999, 162.99999999)
        assert str(obj) == '360 38 163'
        assert repr(obj) == f'{clsname}(360, 38, 163)'
        assert obj.join() == '360, 38, 163'
        assert obj.join(':') == '360:38:163'
        assert format(obj) == str(obj)
        assert format(obj, '.2%') == '36000.00% 3800.00% 16300.00%'
        assert format(obj, '.7f') == '359.9999999 37.9999999 163'  # Enough precision to not round.
    else:
        obj = cls(-0.00000012, 37.9999999, 162.99999999)
        assert str(obj) == '-0 38 163'
        assert repr(obj) == f'{clsname}(-0, 38, 163)'
        assert obj.join() == '-0, 38, 163'
        assert obj.join(':') == '-0:38:163'
        assert format(obj) == str(obj)
        assert format(obj, '.2%') == '-0.00% 3800.00% 16300.00%'
        assert format(obj, '.7f') == '-0.0000001 37.9999999 163'

    # And the same but rounding down.
    obj = cls(0.0000000012, 36.000000001, 68.000000012)
    assert str(obj) == '0 36 68'
    assert repr(obj) == f'{clsname}(0, 36, 68)'
    assert obj.join() == '0, 36, 68'
    assert obj.join(':') == '0:36:68'
    assert format(obj) == str(obj)
    assert format(obj, '.2%') == '0.00% 3600.00% 6800.00%'
    assert format(obj, '.9f') == '0.000000001 36.000000001 68.000000012'

    # Test -0.0 gets the negative stripped.
    neg_zero = 0.0/-1.0
    assert repr(neg_zero) == '-0.0'
    obj = cls(neg_zero, neg_zero, neg_zero)
    assert str(obj) == '0 0 0'
    assert repr(obj) == f'{clsname}(0, 0, 0)'
    assert obj.join() == '0, 0, 0'
    assert obj.join(':') == '0:0:0'
    assert format(obj) == str(obj)
    assert format(obj, '.2%') == '0.00% 0.00% 0.00%'
    assert format(obj, '.9f') == '0 0 0'


def test_unary_ops(frozen_thawed_vec: VecClass) -> None:
    """Test -vec and +vec."""
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_NUMS):
        assert_vec(-Vec(x, y, z), -x, -y, -z, type=Vec)
        assert_vec(+Vec(x, y, z), +x, +y, +z, type=Vec)


def test_mag(frozen_thawed_vec: VecClass) -> None:
    """Test magnitude methods."""
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_ZERONUMS):
        vec = Vec(x, y, z)
        mag = vec.mag()
        length = vec.len()
        assert mag == length, "Exact equality, should be identical."
        assert mag == pytest.approx(math.sqrt(x**2 + y**2 + z**2))

        mag_sq = vec.mag_sq()
        len_sq = vec.len_sq()
        assert mag_sq == len_sq, "Exact equality, should be identical."
        assert len_sq == pytest.approx(x**2 + y**2 + z**2)

        if mag == 0:
            # Vec(0, 0, 0).norm() = 0, 0, 0
            # Not ZeroDivisionError
            assert_vec(vec.norm(), 0, 0, 0, type=Vec)
        else:
            assert_vec(vec.norm(), x/length, y/length, z/length, vec, type=Vec)


def test_contains(frozen_thawed_vec: VecClass) -> None:
    # Match to list.__contains__
    Vec = frozen_thawed_vec
    for num in VALID_NUMS:
        for x, y, z in iter_vec(VALID_NUMS):
            assert (num in Vec(x, y, z)) == (abs(num - x) < 1e-6 or abs(num - y) < 1e-6 or abs(num - z) < 1e-6)


def test_iteration(frozen_thawed_vec: VecClass) -> None:
    """Test vector iteration."""
    Vec = frozen_thawed_vec
    v = Vec(45.0, 50, 65)
    it = iter(v)
    assert iter(it) is iter(it)

    assert next(it) == 45.0
    assert next(it) == 50.0
    assert next(it) == 65.0
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_rev_iteration(frozen_thawed_vec: VecClass) -> None:
    """Test reversed iteration."""
    Vec = frozen_thawed_vec
    v = Vec(45.0, 50, 65)
    it = reversed(v)
    assert iter(it) is iter(it)

    assert next(it) == 65.0
    assert next(it) == 50.0
    assert next(it) == 45.0
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_vec_lerp(frozen_thawed_vec: VecClass) -> None:
    """Test the vector lerp function."""
    Vec = frozen_thawed_vec
    assert_vec(
        Vec.lerp(14.0, 10.0, 20.0, Vec(20.0, -30.0, 8.0), Vec(40.0, -40.0, 8.0)),
        28.0, -34.0, 8.0,
    )
    assert_vec(
        Vec.lerp(15.0, 10.0, 20.0, Vec(8.0, 20.0, -30.0), Vec(8.0, 40.0, -40.0)),
        8.0, 30.0, -35.0,
    )
    assert_vec(
        Vec.lerp(16.0, 10.0, 20.0, Vec(-30.0, 8.0, 20.0), Vec(-40.0, 8.0, 40.0)),
        -36.0, 8.0, 32.0,
    )

    with raises_zero_div:
        Vec.lerp(48.4, -64.0, -64.0, Vec(), Vec())


def test_vec_clamped_invalid(frozen_thawed_vec: VecClass) -> None:
    """Test Vec.clamped()."""
    Vec = frozen_thawed_vec
    vec = Vec(48, 23, 284)
    with pytest.raises(TypeError, match='either 2 positional arguments or 1-2 keyword arguments'):
        vec.clamped(vec, vec, mins=vec, maxs=vec)
    with pytest.raises(TypeError, match='missing either'):
        vec.clamped()
    with pytest.raises(TypeError, match='missing 1 required positional argument'):
        vec.clamped(vec)
    with pytest.raises(TypeError, match='takes 2 positional arguments but 3 were given'):
        vec.clamped(vec, vec, vec)

    # Unchanged FrozenVec returns self.
    fvec = vec_mod.FrozenVec(30, 38, 87)
    assert fvec.clamped(Vec(-80, -80, -80), Vec(120, 120, 120)) is fvec


@pytest.mark.parametrize("axis, u, v", [
    ("x", "y", "z"),
    ("y", "x", "z"),
    ("z", "x", "y"),
])
def test_vec_clamped_axis(frozen_thawed_vec: VecClass, axis: Axis, u: Axis, v: Axis) -> None:
    """Test each axis is independent and behaves correctly."""
    Vec = frozen_thawed_vec
    vec = Vec.with_axes(axis, 400, u, 500, v, 800)
    unchanged = {axis: 400, u: 500, v: 800}
    # Unchanged, positional + kw.
    assert_vec(
        vec.clamped(
            Vec.with_axes(axis, 200, u, 300, v, 678),
            Vec.with_axes(axis, 900, u, 800, v, 1200),
        ),
        **unchanged,
        type=Vec,
    )
    assert_vec(vec, **unchanged)  # clamped() must not modify self!
    assert_vec(
        vec.clamped(
            mins=Vec.with_axes(axis, 200, u, 300, v, 678),
            maxs=Vec.with_axes(axis, 900, u, 800, v, 1200),
        ),
        **unchanged,
        type=Vec,
    )
    assert_vec(vec, **unchanged)
    # Uses mins, positional, kw, both kv.
    assert_vec(
        vec.clamped(
            Vec.with_axes(axis, 448, u, 300, v, 678),
            Vec.with_axes(axis, 900, u, 800, v, 1200),
        ),
        **{axis: 448, u: 500, v: 800},
        type=Vec,
    )
    assert_vec(vec, **unchanged)
    assert_vec(
        vec.clamped(
            mins=Vec.with_axes(axis, 448, u, 300, v, 678),
        ),
        **{axis: 448, u: 500, v: 800},
        type=Vec,
    )
    assert_vec(vec, **unchanged)
    assert_vec(
        vec.clamped(
            mins=Vec.with_axes(axis, 448, u, 300, v, 678),
            maxs=Vec.with_axes(axis, 900, u, 800, v, 1200),
        ),
        **{axis: 448, u: 500, v: 800},
        type=Vec,
    )
    # Uses maxes, positional, kw, both kv.
    assert_vec(
        vec.clamped(
            Vec.with_axes(axis, 200, u, 300, v, 678),
            Vec.with_axes(axis, 321, u, 800, v, 1200),
        ),
        **{axis: 321, u: 500, v: 800},
        type=Vec,
    )
    assert_vec(vec, **unchanged)
    assert_vec(
        vec.clamped(
            maxs=Vec.with_axes(axis, 321, u, 800, v, 1200),
        ),
        **{axis: 321, u: 500, v: 800},
        type=Vec,
    )
    assert_vec(vec, **unchanged)
    assert_vec(
        vec.clamped(
            mins=Vec.with_axes(axis, 200, u, 300, v, 678),
            maxs=Vec.with_axes(axis, 321, u, 800, v, 1200),
        ),
        **{axis: 321, u: 500, v: 800},
        type=Vec,
    )
    assert_vec(vec, **unchanged)


@pytest.mark.slow
def test_scalar(frozen_thawed_vec: VecClass) -> None:
    """Check that Vec() + 5, -5, etc does the correct thing.

    For +, -, *, /, // and % calling with a scalar should perform the
    operation on x, y, and z
    """
    Vec = frozen_thawed_vec
    operators = [
        ('+', op.add, op.iadd, VALID_ZERONUMS),
        ('-', op.sub, op.isub, VALID_ZERONUMS),
        ('*', op.mul, op.imul, VALID_ZERONUMS),
        ('//', op.floordiv, op.ifloordiv, VALID_NUMS),
        ('/', op.truediv, op.itruediv, VALID_NUMS),
        ('%', op.mod, op.imod, VALID_NUMS),
    ]

    # Doesn't implement float(x), and no other operators..
    obj = object()
    mutable = Vec is vec_mod.Vec

    for op_name, op_func, op_ifunc, domain in operators:
        for x, y, z in iter_vec(domain):
            for num in domain:
                targ = Vec(x, y, z)
                rx, ry, rz = (
                    op_func(x, num),
                    op_func(y, num),
                    op_func(z, num),
                )

                # Check forward and reverse fails.
                try:
                    op_func(targ, obj)
                except TypeError:
                    pass
                else:
                    pytest.fail('Vec ' + op_name + 'Scalar succeeded.')
                try:
                    op_func(obj, targ)
                except TypeError:
                    pass
                else:
                    pytest.fail('Scalar ' + op_name + ' Vec succeeded.')
                try:
                    op_ifunc(targ, obj)
                except TypeError:
                    pass
                else:
                    pytest.fail('Vec ' + op_name + '= scalar succeeded.')

                assert_vec(
                    op_func(targ, num),
                    rx, ry, rz,
                    'Forward ' + op_name,
                )

                assert_vec(
                    op_func(num, targ),
                    op_func(num, x),
                    op_func(num, y),
                    op_func(num, z),
                    'Reversed ' + op_name,
                )

                # Ensure they haven't modified the original
                assert_vec(targ, x, y, z)

                res = op_ifunc(targ, num)
                assert_vec(
                    res,
                    rx, ry, rz,
                    f'Return value for ({x} {y} {z}) {op_name}= {num}',
                )
                # Check that the original was or wasn't modified.
                if mutable:
                    assert targ is res
                    assert_vec(
                        targ,
                        rx, ry, rz,
                        f'Original for ({x} {y} {z}) {op_name}= {num}',
                    )
                else:
                    assert targ is not res
                    assert_vec(
                        targ,
                        x, y, z,
                        f'Original for ({x} {y} {z}) {op_name}= {num}',
                    )


@pytest.mark.parametrize('axis, index, u, v, u_ax, v_ax', [
    ('x', 0, 'y', 'z', 1, 2), ('y', 1, 'x', 'z', 0, 2), ('z', 2, 'x', 'y', 0, 1),
], ids='xyz')
def test_vec_props(frozen_thawed_vec: VecClass, axis: str, index: int, u: str, v: str, u_ax: int, v_ax: int) -> None:
    """Test the X/Y/Z attributes and item access."""
    Vec = frozen_thawed_vec

    for other in VALID_ZERONUMS:
        for targ in VALID_ZERONUMS:
            vec = Vec(**{axis: targ, u: other, v: other})

            # Should be constant.
            assert len(vec) == 3
            # Check attribute access
            assert getattr(vec, axis) == targ, (vec, targ, other)
            assert getattr(vec, u) == other, (vec, targ, other)
            assert getattr(vec, v) == other, (vec, targ, other)

            # And getitem access.
            assert vec[index] == targ, (vec, targ, other)
            assert vec[axis] == targ, (vec, targ, other)
            assert vec[u_ax] == other, (vec, targ, other)
            assert vec[v_ax] == other, (vec, targ, other)
            assert vec[u] == other, (vec, targ, other)
            assert vec[v] == other, (vec, targ, other)


def test_vec_to_vec(frozen_thawed_vec: VecClass):
    """Check that Vec() +/- Vec() does the correct thing.

    For +, -, two Vectors apply the operations to all values.
    Dot and cross products do something different.
    """
    Vec = frozen_thawed_vec
    mutable = Vec is vec_mod.Vec
    operators = [
        ('+', op.add, op.iadd),
        ('-', op.sub, op.isub),
    ]

    def test(x1, y1, z1, x2, y2, z2):
        """Check a Vec pair for addition and subtraction."""
        vec1 = Vec(x1, y1, z1)
        vec2 = Vec(x2, y2, z2)
        vec_tup_1 = pytest.deprecated_call(Vec_tuple, x1, y1, z1)
        vec_tup_2 = pytest.deprecated_call(Vec_tuple, x2, y2, z2)

        # These are direct methods, so no inheritence and iop to deal with.

        # Commutative
        assert vec1.dot(vec2) == pytest.approx(x1*x2 + y1*y2 + z1*z2)
        assert vec2.dot(vec1) == pytest.approx(x1*x2 + y1*y2 + z1*z2)
        assert_vec(
            vec1.cross(vec2),
            y1*z2-z1*y2,
            z1*x2-x1*z2,
            x1*y2-y1*x2,
            type=Vec,
        )
        # Ensure they haven't modified the originals
        assert_vec(vec1, x1, y1, z1, type=Vec)
        assert_vec(vec2, x2, y2, z2, type=Vec)


        # Addition and subtraction
        for op_name, op_func, op_ifunc in operators:
            result = (
                op_func(x1, x2),
                op_func(y1, y2),
                op_func(z1, z2),
            )
            assert_vec(
                op_func(vec1, vec2),
                *result,
                type=Vec,
                msg=f'Vec({x1} {y1} {z1}) {op_name} Vec({x2} {y2} {z2})'
            )
            # Ensure they haven't modified the originals
            assert_vec(vec1, x1, y1, z1)
            assert_vec(vec2, x2, y2, z2)

            assert_vec(
                op_func(vec1, vec_tup_2),
                *result,
                type=Vec,
                msg=f'Vec({x1} {y1} {z1}) {op_name} Vec_tuple({x2} {y2} {z2})'
            )
            assert_vec(vec1, x1, y1, z1)

            assert_vec(
                op_func(vec_tup_1, vec2),
                *result,
                type=Vec,
                msg=f'Vec_tuple({x1} {y1} {z1}) {op_name} Vec({x2} {y2} {z2})'
            )

            assert_vec(vec2, x2, y2, z2)

            new_vec1 = Vec(x1, y1, z1)
            assert_vec(
                op_ifunc(new_vec1, vec2),
                *result,
                msg=f'Return val: ({x1} {y1} {z1}) {op_name}= ({x2} {y2} {z2})',
                type=Vec,
            )
            if mutable:
                # Check it modifies the original object too.
                assert_vec(
                    new_vec1,
                    *result,
                    msg=f'Original: {Vec}({x1} {y1} {z1}) {op_name}= {Vec}({x2} {y2} {z2})'
                )
            else:
                # Check it did not modify the original.
                assert_vec(
                    new_vec1,
                    x1, y1, z1,
                    msg=f'Original: {Vec}({x1} {y1} {z1}) {op_name}= {Vec}({x2} {y2} {z2})'
                )

            new_vec1 = Vec(x1, y1, z1)
            assert_vec(
                op_ifunc(new_vec1, tuple(vec2)),
                *result,
                msg=f'Return val: ({x1} {y1} {z1}) {op_name}= tuple({x2} {y2} {z2})'
            )
            if mutable:
                # Check it modifies the original object too.
                assert_vec(
                    new_vec1,
                    *result,
                    msg=f'Original: {Vec}({x1} {y1} {z1}) {op_name}= tuple({x2} {y2} {z2})'
                )
            else:
                # Check it did not modify the original.
                assert_vec(
                    new_vec1,
                    x1, y1, z1,
                    msg=f'Original: {Vec}({x1} {y1} {z1}) {op_name}= tuple({x2} {y2} {z2})'
                )

    for num in VALID_ZERONUMS:
        for num2 in VALID_ZERONUMS:
            # Test the whole value, then each axis individually
            test(num, num, num, num2, num2, num2)
            test(0, num, num, num2, num2, num2)
            test(num, 0, num, num, num2, num2)
            test(num, num, 0, num2, num2, num2)
            test(num, num, num, 0, num2, num2)
            test(num, num, num, num, 0, num2)
            test(num, num, num, num, num, 0)


@pytest.mark.parametrize('op_func', [op.add, op.sub])
def test_vec_to_vec_types(py_c_vec: PyCVec, op_func) -> None:
    """Verify the correct types are returned when using differing types."""
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec

    assert type(op_func(Vec(1, 2, 3), Vec(1, 2, 3))) is Vec
    assert type(op_func(Vec(1, 2, 3), FrozenVec(1, 2, 3))) is Vec
    assert type(op_func(FrozenVec(1, 2, 3), Vec(1, 2, 3))) is FrozenVec
    assert type(op_func(FrozenVec(1, 2, 3), FrozenVec(1, 2, 3))) is FrozenVec


def test_scalar_zero(py_c_vec: PyCVec):
    """Check zero behaviour with division ops."""
    Vec = vec_mod.Vec
    for x, y, z in iter_vec(VALID_NUMS):
        vec = Vec(x, y, z)
        assert_vec(0 / vec, 0, 0, 0)
        assert_vec(0 // vec, 0, 0, 0)
        assert_vec(0 % vec, 0, 0, 0)
        assert_vec(0.0 / vec, 0, 0, 0)
        assert_vec(0.0 // vec, 0, 0, 0)
        assert_vec(0.0 % vec, 0, 0, 0)

        # We don't need to check divmod(0, vec) -
        # that always falls back to % and /.

        with raises_zero_div:
            vec / 0
        with raises_zero_div:
            vec // 0
        with raises_zero_div:
            vec % 0
        with raises_zero_div:
            divmod(vec, 0)
        with raises_zero_div:
            vec / 0.0
        with raises_zero_div:
            vec // 0.0
        with raises_zero_div:
            vec % 0.0
        with raises_zero_div:
            divmod(vec, 0.0)

        with raises_zero_div:
            vec /= 0
        with raises_zero_div:
            vec //= 0
        with raises_zero_div:
            vec %= 0
        with raises_zero_div:
            vec /= 0.0
        with raises_zero_div:
            vec //= 0.0
        with raises_zero_div:
            vec %= 0.0


def test_divmod_vec_scalar(frozen_thawed_vec):
    """Test divmod(vec, scalar)."""
    Vec = frozen_thawed_vec

    for x, y, z in iter_vec(VALID_ZERONUMS):
        for num in VALID_NUMS:
            div, mod = divmod(Vec(x, y, z), num)
            assert_vec(div, x // num, y // num, z // num, type=Vec)
            assert_vec(mod, x % num, y % num, z % num, type=Vec)


def test_divmod_scalar_vec(frozen_thawed_vec):
    """Test divmod(scalar, vec)."""
    Vec = frozen_thawed_vec

    for x, y, z in iter_vec(VALID_NUMS):
        for num in VALID_ZERONUMS:
            div, mod = divmod(num, Vec(x, y, z))
            assert_vec(div, num // x, num // y, num // z, type=Vec)
            assert_vec(mod, num % x, num % y, num % z, type=Vec)


@pytest.mark.parametrize('name, func', [
    ('*', op.mul),
    ('/', op.truediv),
    ('//', op.floordiv),
    ('%', op.mod),
    ('*=', op.imul),
    ('/=', op.itruediv),
    ('//=', op.ifloordiv),
    ('%=', op.imod),
    ('divmod', divmod),
])
def test_vector_mult_fail(frozen_thawed_vec, name, func):
    """Test *, /, //, %, divmod always fails between vectors."""
    Vec = frozen_thawed_vec

    try:
        for num in VALID_ZERONUMS:
            for num2 in VALID_NUMS:
                # Test the whole value, then each axis individually
                with raises_typeerror:
                    func(Vec(num, num, num), Vec(num2, num2, num2))

                with raises_typeerror:
                    func(Vec(0, num, num), Vec(num2, num2, num2))
                with raises_typeerror:
                    func(Vec(num, 0, num), Vec(num2, num2, num2))
                with raises_typeerror:
                    func(Vec(num, num, 0), Vec(num2, num2, num2))
                with raises_typeerror:
                    func(Vec(num, num, num), Vec(0, num2, num2))
                with raises_typeerror:
                    func(Vec(num, num, num), Vec(num2, 0, num2))
                with raises_typeerror:
                    func(Vec(num, num, num), Vec(num2, num2, 0))
    except AssertionError as exc:
        raise AssertionError(f'Expected TypError from vec {name} vec') from exc


def test_order(py_c_vec) -> None:
    """Test ordering operations (>, <, <=, >=, ==)."""
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec

    def cmp_eq(a: float, b: float) -> bool:
        return abs(a - b) < 1e-6

    def cmp_ne(a: float, b: float) -> bool:
        return abs(a - b) >= 1e-6

    def cmp_le(a: float, b: float) -> bool:
        return (a - b) <= 1e-6

    def cmp_gt(a: float, b: float) -> bool:
        return (a - b) > 1e-6

    def cmp_lt(a: float, b: float) -> bool:
        return (b - a) > 1e-6

    def cmp_ge(a: float, b: float) -> bool:
        return (b - a) <= 1e-6

    comp_ops = [
        (op.eq, cmp_eq),
        (op.ne, cmp_ne),
        (op.le, cmp_le),
        (op.lt, cmp_lt),
        (op.ge, cmp_ge),
        (op.gt, cmp_gt),
    ]

    def test(x1, y1, z1, x2, y2, z2):
        """Check a Vec pair for incorrect comparisons."""
        vec1 = Vec(x1, y1, z1)
        vec2 = Vec(x2, y2, z2)
        fvec1 = FrozenVec(x1, y1, z1)
        fvec2 = FrozenVec(x2, y2, z2)
        vec2_tup = pytest.deprecated_call(Vec_tuple, x2, y2, z2)
        for op_func, float_func in comp_ops:
            if float_func is cmp_ne:
                # special-case - != uses or, not and
                corr_result = float_func(x1, x2) or float_func(y1, y2) or float_func(z1, z2)
            else:
                corr_result = float_func(x1, x2) and float_func(y1, y2) and float_func(z1, z2)
            for left, right in [
                (vec1, vec2),
                (fvec1, vec2),
                (vec1, fvec2),
                (fvec1, fvec2),
                (vec1, vec2_tup),
                (fvec1, vec2_tup),
                (vec1, (x2, y2, z2)),
                (fvec1, (x2, y2, z2))
            ]:
                # Assert rewriting doesn't work in this nested function?
                if (res := op_func(vec1, vec2)) is not corr_result:
                    pytest.fail(
                        f'Incorrect {float_func.__name__} comparison for '
                        f'{type(left)}({x1} {y1} {z1}) {op_func.__name__} {type(right)}({x2} {y2} {z2}) = {res}'
                    )

    for num in VALID_ZERONUMS:
        for num2 in VALID_ZERONUMS:
            # Test the whole comparison, then each axis pair seperately
            test(num, num, num, num2, num2, num2)
            test(0, num, num, num2, num2, num2)
            test(num, 0, num, num, num2, num2)
            test(num, num, 0, num2, num2, num2)
            test(num, num, num, 0, num2, num2)
            test(num, num, num, num, 0, num2)
            test(num, num, num, num, num, 0)
            if abs(num - num2) > 1e-6:
                for op_func, float_func in comp_ops:
                    if op_func(num, num2) is not float_func(num, num2):
                        pytest.fail(f'{op_func}, {float_func}, {num}, {num2}')


def test_binop_fail(frozen_thawed_vec) -> None:
    """Test binary operations with invalid operands."""
    Vec = frozen_thawed_vec

    vec = Vec()
    operations = [
        op.add, op.iadd,
        op.sub, op.isub,
        op.truediv, op.itruediv,
        op.floordiv, op.ifloordiv,
        op.mul, op.imul,
        op.lt, op.gt,
        op.le, op.ge,

        divmod,
        op.concat, op.iconcat,
    ]
    for fail_object in [None, 'string', ..., staticmethod, tuple, Vec]:
        assert vec != fail_object
        assert fail_object != vec

        assert not vec == fail_object
        assert not fail_object == vec
        for operation in operations:
            pytest.raises(TypeError, operation, vec, fail_object)
            pytest.raises(TypeError, operation, fail_object, vec)


def test_axis(frozen_thawed_vec) -> None:
    """Test the Vec.axis() function."""
    Vec = frozen_thawed_vec
    handler = pytest.raises(ValueError, match='not an on-axis vector')

    for num in VALID_NUMS:
        assert Vec(num, 0, 0).axis() == 'x', num
        assert Vec(0, num, 0).axis() == 'y', num
        assert Vec(0, 0, num).axis() == 'z', num

        assert Vec(num, 1e-8, -0.0000008).axis() == 'x', num
        assert Vec(-1e-8, num, -0.0000008).axis() == 'y', num
        assert Vec(-0.000000878, 0.0000003782, num).axis() == 'z', num

        with handler:
            Vec(num, num, 0).axis()

        with handler:
            Vec(num, 0, num).axis()

        with handler:
            Vec(0, num, num).axis()

        with handler:
            Vec(num, num, num).axis()

        with handler:
            Vec(-num, num, num).axis()

        with handler:
            Vec(num, -num, num).axis()

        with handler:
            Vec(num, num, -num).axis()

    with handler:
        Vec().axis()


def test_other_axes(frozen_thawed_vec) -> None:
    """Test Vec.other_axes()."""
    Vec = frozen_thawed_vec

    bad_args = ['p', '', 0, 1, 2, False, Vec(2, 3, 5)]
    for x, y, z in iter_vec(VALID_NUMS):
        vec = Vec(x, y, z)
        assert vec.other_axes('x') == (y, z)
        assert vec.other_axes('y') == (x, z)
        assert vec.other_axes('z') == (x, y)
        # Test some bad args.
        for invalid in bad_args:
            with raises_keyerror: vec.other_axes(invalid)


def test_abs(frozen_thawed_vec) -> None:
    """Test the function of abs(Vec)."""
    Vec = frozen_thawed_vec

    for x, y, z in iter_vec(VALID_ZERONUMS):
        assert_vec(abs(Vec(x, y, z)), abs(x), abs(y), abs(z))


def test_bool(frozen_thawed_vec) -> None:
    """Test bool() applied to Vec."""
    Vec = frozen_thawed_vec

    # Empty vector is False
    assert not Vec(0, 0, 0)
    assert not Vec(-0, -0, -0)
    for val in VALID_NUMS:
        # Any number in any axis makes it True.
        assert Vec(val, -0, 0)
        assert Vec(0, val, 0)
        assert Vec(-0, 0, val)
        assert Vec(0, val, val)
        assert Vec(val, -0, val)
        assert Vec(val, val, 0)
        assert Vec(val, val, val)


def test_hash(py_c_vec) -> None:
    """Test hashing and dict key use for FrozenVec."""
    FrozenVec = vec_mod.FrozenVec
    Vec = vec_mod.Vec

    with pytest.raises(TypeError):
        hash(Vec())

    for x, y, z in iter_vec(VALID_NUMS):
        # Must match tuples.
        assert hash(FrozenVec(x, y, z)) == hash((x, y, z))
    test_dict = {
        FrozenVec(4.0, 5.8, 9.6): 'a',
        (12.8, -2.3, 12.0): 'b',
    }
    assert test_dict[4.0, 5.8, 9.6] == 'a'
    assert test_dict[FrozenVec(12.8, -2.3, 12.0)] == 'b'
    assert test_dict[Vec(12.8, -2.3, 12).freeze()] == 'b'


def test_iter_line(frozen_thawed_vec) -> None:
    """Test Vec.iter_line()"""
    Vec = frozen_thawed_vec
    for pos, x in zip(Vec(4, 5.82, -6.35).iter_line(Vec(10, 5.82, -6.35), 1), range(4, 11)):
        assert_vec(pos, x, 5.82, -6.35, type=Vec)
    for pos, y in zip(Vec(-4.36, 10.82, -6.35).iter_line(Vec(-4.36, 5.82, -6.35), 1), range(10, 4, -1)):
        assert_vec(pos, -4.36, y + 0.82, -6.35, type=Vec)
    for pos, z in zip(Vec(3.78, 12.98, -5.65).iter_line(Vec(3.78, 12.98, 6.35), 1), range(-6, 7)):
        assert_vec(pos, 3.78, 12.98, z + 0.35, type=Vec)


def test_iter_grid(frozen_thawed_vec):
    """Test Vec.iter_grid()."""
    Vec = frozen_thawed_vec
    it = Vec.iter_grid(Vec(35, 59.99999, 90), Vec(40, 70, 110.001), 5)

    assert_vec(next(it), 35, 60, 90, type=Vec)
    assert_vec(next(it), 35, 60, 95, type=Vec)
    assert_vec(next(it), 35, 60, 100, type=Vec)
    assert_vec(next(it), 35, 60, 105, type=Vec)
    assert_vec(next(it), 35, 60, 110, type=Vec)

    assert_vec(next(it), 35, 65, 90, type=Vec)
    assert_vec(next(it), 35, 65, 95, type=Vec)
    assert_vec(next(it), 35, 65, 100, type=Vec)
    assert_vec(next(it), 35, 65, 105, type=Vec)
    assert_vec(next(it), 35, 65, 110, type=Vec)

    assert_vec(next(it), 35, 70, 90, type=Vec)
    assert_vec(next(it), 35, 70, 95, type=Vec)
    assert_vec(next(it), 35, 70, 100, type=Vec)
    assert_vec(next(it), 35, 70, 105, type=Vec)
    assert_vec(next(it), 35, 70, 110, type=Vec)

    assert_vec(next(it), 40, 60, 90, type=Vec)
    assert_vec(next(it), 40, 60, 95, type=Vec)
    assert_vec(next(it), 40, 60, 100, type=Vec)
    assert_vec(next(it), 40, 60, 105, type=Vec)
    assert_vec(next(it), 40, 60, 110, type=Vec)

    assert_vec(next(it), 40, 65, 90, type=Vec)
    assert_vec(next(it), 40, 65, 95, type=Vec)
    assert_vec(next(it), 40, 65, 100, type=Vec)
    assert_vec(next(it), 40, 65, 105, type=Vec)
    assert_vec(next(it), 40, 65, 110, type=Vec)

    assert_vec(next(it), 40, 70, 90, type=Vec)
    assert_vec(next(it), 40, 70, 95, type=Vec)
    assert_vec(next(it), 40, 70, 100, type=Vec)
    assert_vec(next(it), 40, 70, 105, type=Vec)
    assert_vec(next(it), 40, 70, 110, type=Vec)

    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)

    assert list(Vec.iter_grid(Vec(35, 40, 20), Vec(35, 40, 19))) == []
    assert list(Vec.iter_grid(Vec(35, 40, 20), Vec(35, 40, 20))) == [Vec(35, 40, 20)]


# Various keys similar to correct values, in order to test edge cases in logic.
INVALID_KEYS = [
    '4',
    '',
    -1,
    3, 4,
    3.0, 4.0,
    bool,
    slice(0, 1),
    None,

    # Overflow checks - won't fit into 1 byte!
    2 ** 256,
    -2 ** 256,
    '♞',
]


def test_getitem(py_c_vec):
    """Test vec[x] with various args."""
    Vec = vec_mod.Vec
    v = Vec(1.5, 3.5, -8.7)

    assert v[0] == 1.5
    assert v[1] == 3.5
    assert v[2] == -8.7
    assert v['x'] == 1.5
    assert v['y'] == 3.5
    assert v['z'] == -8.7

    v[1] = 67.9

    assert v[0] == 1.5
    assert v[1] == 67.9
    assert v[2] == -8.7
    assert v['x'] == 1.5
    assert v['y'] == 67.9
    assert v['z'] == -8.7

    v[0] = -12.9

    assert v[0] == -12.9
    assert v[1] == 67.9
    assert v[2] == -8.7
    assert v['x'] == -12.9
    assert v['y'] == 67.9
    assert v['z'] == -8.7

    v[2] = 0

    assert v[0] == -12.9
    assert v[1] == 67.9
    assert v[2] == 0
    assert v['x'] == -12.9
    assert v['y'] == 67.9
    assert v['z'] == 0

    v.x = v.y = v.z = 0

    for invalid in INVALID_KEYS:
        try:
            res = v[invalid]
        except KeyError:
            pass
        else:
            pytest.fail(f"Key succeeded: {invalid!r} -> {res!r}")


def test_setitem(py_c_vec) -> None:
    """Test vec[x]=y with various args."""
    Vec = vec_mod.Vec

    for ind, axis in enumerate('xyz'):
        vec1 = Vec()
        vec1[axis] = 20.3
        assert vec1[axis] == 20.3, axis
        assert vec1.other_axes(axis) == (0.0, 0.0), axis

        vec1[axis] = Fraction(15, 12)
        assert vec1[axis] == IsFloat(exactly=1.25)

        vec2 = Vec()
        vec2[ind] = 1.25
        assert_vec(vec1, vec2.x, vec2.y, vec2.z, axis)

    vec = Vec()
    for invalid in INVALID_KEYS:
        with pytest.raises(KeyError):
            vec[invalid] = 8.0
        assert_vec(vec, 0, 0, 0, 'Invalid key set something!')

    with pytest.raises(TypeError):
        vec['x'] = 'test'
    with pytest.raises(TypeError):
        vec['z'] = []
    assert_vec(vec, 0, 0, 0, 'Invalid number was stored!')


def test_vec_constants(frozen_thawed_vec) -> None:
    """Check some of the constants assigned to Vec."""
    Vec = frozen_thawed_vec

    assert Vec.N == Vec.north == Vec(y=1)
    assert Vec.S == Vec.south == Vec(y=-1)
    assert Vec.E == Vec.east == Vec(x=1)
    assert Vec.W == Vec.west == Vec(x=-1)

    assert Vec.T == Vec.top == Vec(z=1)
    assert Vec.B == Vec.bottom == Vec(z=-1)

    assert set(Vec.INV_AXIS['x']) == {'y', 'z'}
    assert set(Vec.INV_AXIS['y']) == {'x', 'z'}
    assert set(Vec.INV_AXIS['z']) == {'x', 'y'}

    assert Vec.INV_AXIS['x', 'y'] == 'z'
    assert Vec.INV_AXIS['y', 'z'] == 'x'
    assert Vec.INV_AXIS['x', 'z'] == 'y'

    assert Vec.INV_AXIS['y', 'x'] == 'z'
    assert Vec.INV_AXIS['z', 'y'] == 'x'
    assert Vec.INV_AXIS['z', 'x'] == 'y'

    assert len(Vec.INV_AXIS) == 9, 'Extra values!'

# Copied from CPython's round() tests.
ROUND_VALS = [
    (1.0, 1.0),
    (10.0, 10.0),
    (1000000000.0, 1000000000.0),
    (1e20, 1e20),

    (-1.0, -1.0),
    (-10.0, -10.0),
    (-1000000000.0, -1000000000.0),
    (-1e20, -1e20),

    (0.1, 0.0),
    (1.1, 1.0),
    (10.1, 10.0),
    (1000000000.1, 1000000000.0),

    (-1.1, -1.0),
    (-10.1, -10.0),
    (-1000000000.1, -1000000000.0),

    (0.9, 1.0),
    (9.9, 10.0),
    (999999999.9, 1000000000.0),

    (-0.9, -1.0),
    (-9.9, -10.0),
    (-999999999.9, -1000000000.0),

    # Even/odd rounding behaviour..
    (5.5, 6),
    (6.5, 6),
    (-5.5, -6),
    (-6.5, -6),

    (5e15 - 1, 5e15 - 1),
    (5e15, 5e15),
    (5e15 + 1, 5e15 + 1),
    (5e15 + 2, 5e15 + 2),
    (5e15 + 3, 5e15 + 3),
]


def test_round(frozen_thawed_vec):
    """Test round(Vec)."""
    Vec = frozen_thawed_vec

    for from_val, to_val in ROUND_VALS:
        orig = Vec(from_val, from_val, from_val)
        rnd = round(orig)
        assert_vec(rnd, to_val, to_val, to_val, type=Vec)
        assert orig is not rnd

    # Check it doesn't mix up orders..
    for val in VALID_NUMS:
        assert_vec(round(Vec(val, 0, 0)), round(val), 0, 0, type=Vec)
        assert_vec(round(Vec(0, val, 0)), 0, round(val), 0, type=Vec)
        assert_vec(round(Vec(0, 0, val)), 0, 0, round(val), type=Vec)

MINMAX_VALUES = [
    (0, 0),
    (1, 0),
    (-5, -5),
    (0.3, 0.4),
    (-0.3, -0.2),
]
MINMAX_VALUES += [(b, a) for a, b in MINMAX_VALUES]


def test_minmax(py_c_vec):
    """Test Vec.min() and Vec.max()."""
    Vec = vec_mod.Vec

    vec_a = Vec()
    vec_b = Vec()

    for a, b in MINMAX_VALUES:
        max_val = max(a, b)
        min_val = min(a, b)
        for axis in 'xyz':
            vec_a.x = vec_a.y = vec_a.z = 0
            vec_b.x = vec_b.y = vec_b.z = 0

            vec_a[axis] = a
            vec_b[axis] = b
            assert vec_a.min(vec_b) is None, (a, b, axis, min_val)
            assert vec_a[axis] == min_val, (a, b, axis, min_val)

            vec_a[axis] = a
            vec_b[axis] = b
            assert vec_a.max(vec_b) is None, (a, b, axis, max_val)
            assert vec_a[axis] == max_val, (a, b, axis, max_val)


def test_mut_copy(py_c_vec):
    """Test copying Vectors."""
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec

    test_data = 1.5, 0.2827, 2.3464545636e47

    orig = Vec(test_data)

    cpy_meth = orig.copy()

    assert orig is not cpy_meth  # Must be a new object.
    assert cpy_meth is not orig.copy()  # Cannot be cached
    assert orig == cpy_meth  # Numbers must be exactly identical!

    cpy = copy.copy(orig)

    assert orig is not cpy
    assert cpy_meth is not copy.copy(orig)
    assert orig == cpy
    assert type(cpy) == Vec

    dcpy = copy.deepcopy(orig)

    assert orig is not dcpy
    assert orig == dcpy
    assert type(dcpy) == Vec

    frozen = FrozenVec(test_data)
    # Copying FrozenVec does nothing.
    assert frozen is frozen.copy()
    assert frozen is copy.copy(frozen)
    assert frozen is copy.deepcopy(frozen)


def test_pickle(frozen_thawed_vec):
    """Test pickling and unpickling Vectors."""
    Vec = frozen_thawed_vec

    test_data = 1.5, 0.2827, 2.3464545636e47
    orig = Vec(test_data)
    pick = pickle.dumps(orig)
    thaw = pickle.loads(pick)

    assert orig is not thaw
    assert orig == thaw
    assert type(thaw) == Vec

    # Ensure both produce the same pickle - so they can be interchanged.
    cy_pick = pickle.dumps(getattr(vec_mod, 'Cy_' + Vec.__name__)(test_data))
    py_pick = pickle.dumps(getattr(vec_mod, 'Py_' + Vec.__name__)(test_data))

    assert cy_pick == py_pick == pick


def test_bbox(frozen_thawed_vec: VecClass) -> None:
    """Test the functionality of Vec.bbox()."""
    Vec = frozen_thawed_vec

    # No arguments
    with raises_typeerror:
        Vec.bbox()

    # Non-iterable
    with raises_typeerror:
        Vec.bbox(None)

    # Starting with non-vector.
    with raises_typeerror:
        Vec.bbox(None, Vec())

    # Containing non-vector.
    with raises_typeerror:
        Vec.bbox(Vec(), None)

    # Empty iterable.
    with pytest.raises(ValueError, match=re.compile('empty', re.IGNORECASE)):
        Vec.bbox([])

    # Iterable starting with non-vector.
    with raises_typeerror:
        Vec.bbox([None])

    # Iterable containing non-vector.
    with raises_typeerror:
        Vec.bbox([Vec(), None])

    def test(*values):
        """Test these values work."""
        min_x = min(v.x for v in values)
        min_y = min(v.y for v in values)
        min_z = min(v.z for v in values)

        max_x = max(v.x for v in values)
        max_y = max(v.y for v in values)
        max_z = max(v.z for v in values)

        # Check passing a single iterable.
        bb_min, bb_max = Vec.bbox(values)
        assert_vec(bb_min, min_x, min_y, min_z, values)
        assert_vec(bb_max, max_x, max_y, max_z, values)

        # Check passing each value individually.
        bb_min, bb_max = Vec.bbox(*values)
        assert_vec(bb_min, min_x, min_y, min_z, values)
        assert_vec(bb_max, max_x, max_y, max_z, values)

    test(Vec(0.0, 0.0, 0.0))
    test(Vec(1.0, -1.0, 0.0), Vec(2.4, -2.4, 5.5))
    test(Vec(2.3, 4.5, 5.6), Vec(-3.4, 4.8, -2.3), Vec(-2.3, 8.2, 3.4))
    # Extreme double values.
    test(Vec(2.346436e47, -4.345e49, 3.59e50), Vec(-7.54e50, 3.45e127, -1.23e140))


# noinspection PyDeprecation
def test_vmf_rotation(datadir: Path, py_c_vec: PyCVec):
    """Complex test.

    Use a compiled map to check the functionality of Vec.rotate().
    """
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec
    Angle = vec_mod.Angle
    from srctools.bsp import BSP

    vmf = BSP(datadir / 'rot_main.bsp').ents

    for ent in vmf.entities:
        if ent['classname'] != 'info_target':
            continue
        angle_str = ent['angles']
        angles = Angle.from_str(angle_str)
        local_vec = Vec(
            float(ent['local_x']),
            float(ent['local_y']),
            float(ent['local_z']),
        )
        x, y, z = round(Vec.from_str(ent['origin']) / 128, 3)

        msg = f'{local_vec} @ {angles} => ({x}, {y}, {z})'

        assert_vec(Vec(local_vec) @ angles, x, y, z, msg, tol=1e-3, type=Vec)
        assert_vec(FrozenVec(local_vec) @ angles, x, y, z, msg, tol=1e-3, type=FrozenVec)
        # Since these two are deprecated, FrozenVec doesn't have them.
        with pytest.deprecated_call(match=r'vec @ Angle\.from_str()'):
            assert_vec(Vec(local_vec).rotate_by_str(angle_str), x, y, z, msg, tol=1e-3, type=Vec)
        with pytest.deprecated_call(match='vec @ Angle'):
            assert_vec(Vec(local_vec).rotate(*angles), x, y, z, msg, tol=1e-3, type=Vec)


def test_cross_product_axes(frozen_thawed_vec: VecClass):
    """Check all the cross product identities."""
    Vec = frozen_thawed_vec

    assert_vec(Vec.cross(Vec(x=1), Vec(y=1)), 0, 0, 1)
    assert_vec(Vec.cross(Vec(x=1), Vec(z=1)), 0, -1, 0)
    assert_vec(Vec.cross(Vec(y=1), Vec(z=1)), 1, 0, 0)
    assert_vec(Vec.cross(Vec(y=1), Vec(x=1)), 0, 0, -1)
    assert_vec(Vec.cross(Vec(z=1), Vec(x=1)), 0, 1, 0)
    assert_vec(Vec.cross(Vec(z=1), Vec(y=1)), -1, 0, 0)


def test_cross_types(py_c_vec) -> None:
    """Test mixing types gives the right classes."""
    Vec = vec_mod.Vec
    FrozenVec = vec_mod.FrozenVec
    assert type(Vec(1, 2, 3).cross(Vec(1, 2, 3))) is Vec
    assert type(Vec(1, 2, 3).cross(FrozenVec(1, 2, 3))) is Vec
    assert type(FrozenVec(1, 2, 3).cross(Vec(1, 2, 3))) is FrozenVec
    assert type(FrozenVec(1, 2, 3).cross(FrozenVec(1, 2, 3))) is FrozenVec

    assert type(Vec.cross(Vec(1, 2, 3), Vec(1, 2, 3))) is Vec
    assert type(Vec.cross(Vec(1, 2, 3), FrozenVec(1, 2, 3))) is Vec
    assert type(Vec.cross(FrozenVec(1, 2, 3), Vec(1, 2, 3))) is Vec
    assert type(Vec.cross(FrozenVec(1, 2, 3), FrozenVec(1, 2, 3))) is Vec

    assert type(FrozenVec.cross(Vec(1, 2, 3), Vec(1, 2, 3))) is FrozenVec
    assert type(FrozenVec.cross(Vec(1, 2, 3), FrozenVec(1, 2, 3))) is FrozenVec
    assert type(FrozenVec.cross(FrozenVec(1, 2, 3), Vec(1, 2, 3))) is FrozenVec
    assert type(FrozenVec.cross(FrozenVec(1, 2, 3), FrozenVec(1, 2, 3))) is FrozenVec


@pytest.mark.parametrize('x, xvalid', [
    (-64, False),
    (-32, True),
    (-10, True),
    (240, True),
    (300, False),
])
@pytest.mark.parametrize('y, yvalid', [
    (-64, False),
    (-48, True),
    (100, True),
    (310, True),
    (400, False),
])
@pytest.mark.parametrize('z, zvalid', [
    (-256, False),
    (-128, True),
    (-64, True),
    (23, True),
    (49, False),
])
def test_vec_in_bbox(
    frozen_thawed_vec: VecClass,
    x: float, y: float, z: float,
    xvalid: bool, yvalid: bool, zvalid: bool,
) -> None:
    """"Test Vec.in_bbox(a, b)"""
    Vec = frozen_thawed_vec
    a = Vec(-32, -48, -128)
    b = Vec(240, 310, 23)
    valid = xvalid and yvalid and zvalid
    assert Vec(x, y, z).in_bbox(a, b) is valid
    assert Vec(x, y, z).in_bbox(Vec(b.x, a.y, a.z), Vec(a.x, b.y, b.z)) is valid
    assert Vec(x, y, z).in_bbox(Vec(a.x, b.y, a.z), Vec(b.x, a.y, b.z)) is valid
    assert Vec(x, y, z).in_bbox(Vec(a.x, a.y, b.z), Vec(b.x, b.y, a.z)) is valid
    assert Vec(x, y, z).in_bbox(b, a) is valid


@pytest.mark.parametrize('zoff', [-2.0, -1.1, -0.9, +0.9, +1.1, +2.0])
@pytest.mark.parametrize('yoff', [-2.0, -1.1, -0.9, +0.9, +1.1, +2.0])
@pytest.mark.parametrize('xoff', [-2.0, -1.1, -0.9, +0.9, +1.1, +2.0])
def test_bbox_intersect(frozen_thawed_vec: VecClass, xoff: float, yoff: float, zoff: float) -> None:
    """Test Vec.bbox_intersect()."""
    Vec = frozen_thawed_vec
    rand = Random(2368)  # Ensure reproducibility.
    pos_a = Vec(rand.randint(-100, 100), rand.randint(-100, 100), rand.randint(-100, 100))
    dims_a = Vec(rand.randint(10, 20), rand.randint(10, 20), rand.randint(10, 20))
    dims_b = Vec(rand.randint(10, 20), rand.randint(10, 20), rand.randint(10, 20))

    # Compute offset, so -1 = touching, 0 = centered, etc.
    pos_b = pos_a + Vec(
        xoff * (dims_a.x + dims_b.x),
        yoff * (dims_a.y + dims_b.y),
        zoff * (dims_a.z + dims_b.z),
    )
    intersect = (-1 <= xoff <= +1) and (-1 <= yoff <= +1) and (-1 <= zoff <= +1)
    a1 = pos_a - dims_a
    a2 = pos_a + dims_a
    b1 = pos_b - dims_b
    b2 = pos_b + dims_b
    # Commutative.
    assert Vec.bbox_intersect(a1, a2, b1, b2) is intersect
    assert Vec.bbox_intersect(b1, b2, a1, a2) is intersect
