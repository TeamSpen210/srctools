"""Test the Vector object."""
import pickle
import copy
import operator as op
from pathlib import Path
from random import Random
import inspect

from helpers import *
from srctools import Vec_tuple, math as vec_mod

# Reuse these context managers.
raises_typeerror = pytest.raises(TypeError)
raises_valueerror = pytest.raises(ValueError)
raises_keyerror = pytest.raises(KeyError)
raises_zero_div = pytest.raises(ZeroDivisionError)
VecClass = Type[vec_mod.VecBase]

@pytest.fixture(params=['Vec', 'FrozenVec'])
def frozen_thawed_vec(py_c_vec, request) -> VecClass:
    """Support testing both mutable and immutable vectors."""
    yield getattr(vec_mod, request.param)


@pytest.mark.parametrize('cls', ['Vec', 'Matrix', 'Angle'])
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


def test_vec_copying(py_c_vec, frozen_thawed_vec):
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
        fv2 = FrozenVec(fv)
        assert_vec(fv2, x, y, z)
        assert v is not v3

        assert fv.copy() is fv


def test_vec_from_str(py_c_vec, frozen_thawed_vec):
    """Test the functionality of Vec.from_str()."""
    parse_vec_str = vec_mod.parse_vec_str
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_ZERONUMS):
        # Test Vec.from_str()
        assert_vec(Vec.from_str('{} {} {}'.format(x, y, z)), x, y, z)
        assert_vec(Vec.from_str('<{} {} {}>'.format(x, y, z)), x, y, z)
        # {x y z}
        assert_vec(Vec.from_str('{{{} {} {}}}'.format(x, y, z)), x, y, z)
        assert_vec(Vec.from_str('({} {} {})'.format(x, y, z)), x, y, z)
        assert_vec(Vec.from_str('[{} {} {}]'.format(x, y, z)), x, y, z)

        # And parse_vec_str
        v = Vec(x, y, z)
        assert_vec(v, *parse_vec_str('{} {} {}'.format(x, y, z)))
        assert_vec(v, *parse_vec_str('<{} {} {}>'.format(x, y, z)))

        assert_vec(v, *parse_vec_str('{{{} {} {}}}'.format(x, y, z)))
        assert_vec(v, *parse_vec_str('({} {} {})'.format(x, y, z)))
        assert_vec(v, *parse_vec_str('[{} {} {}]'.format(x, y, z)))

        parse_res = parse_vec_str('{} {} {}'.format(x, y, z))
        assert isinstance(parse_res, tuple)
        assert parse_res == (x, y, z)

        # Test converting a converted Vec
        orig = Vec(x, y, z)
        new = Vec.from_str(Vec(x, y, z))
        assert_vec(new, x, y, z)
        assert orig is not new  # It must be a copy


def test_vec_as_tuple(frozen_thawed_vec):
    """Test the functionality of Vec.from_str() and parse_vec_str()."""
    parse_vec_str = vec_mod.parse_vec_str
    Vec = frozen_thawed_vec
    for x, y, z in iter_vec(VALID_ZERONUMS):
        # Check as_tuple() makes an equivalent tuple
        orig = Vec(x, y, z)
        tup = orig.as_tuple()
        assert isinstance(tup, tuple)
        assert (x, y, z) == tup
        assert hash((x, y, z)) == hash(tup)
        # Bypass subclass functions.
        assert tuple.__getitem__(tup, 0) == x
        assert tuple.__getitem__(tup, 1) == y
        assert tuple.__getitem__(tup, 2) == z
        assert tup.x == x
        assert tup.y == y
        assert tup.z == z

def test_from_str_fails(py_c_vec, frozen_thawed_vec):
    """Check failures in Vec.from_str()"""
    # Note - does not pass through unchanged, they're converted to floats!
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


def test_with_axes(frozen_thawed_vec: VecClass):
    """Test the with_axes() constructor."""
    Vec = frozen_thawed_vec
    for axis, u, v in ['xyz', 'yxz', 'zxy']:
        for num in VALID_ZERONUMS:
            vec = Vec.with_axes(axis, num)
            assert vec[axis] == num
            # Other axes are zero.
            assert vec[u] == 0
            assert vec[v] == 0

    for a, b, c in iter_vec('xyz'):
        if a == b or b == c or a == c:
            continue
        for x, y, z in iter_vec(VALID_ZERONUMS):
            vec = Vec.with_axes(a, x, b, y, c, z)
            msg = f'{vec} = {a}={x}, {b}={y}, {c}={z}'
            assert vec[a] == x, msg
            assert vec[b] == y, msg
            assert vec[c] == z, msg


def test_vec_stringification(frozen_thawed_vec: VecClass):
    """Test the various string methods."""
    Vec = frozen_thawed_vec
    # Add on the edge case where '.0' needs to be kept.
    for x, y, z in iter_vec(VALID_NUMS + [-210.048]):
        v = Vec(x, y, z)
        assert str(v) == f'{x:g} {y:g} {z:g}'
        assert repr(v) == f'Vec({x:g}, {y:g}, {z:g})'
        assert v.join() == f'{x:g}, {y:g}, {z:g}'
        assert v.join(' : ') == f'{x:g} : {y:g} : {z:g}'
        assert format(v) == f'{x:g} {y:g} {z:g}'
        assert format(v, '.02f') == f'{x:.02f} {y:.02f} {z:.02f}'


def test_unary_ops(py_c_vec: PyCVec):
    """Test -vec and +vec."""
    Vec = vec_mod.Vec
    for x, y, z in iter_vec(VALID_NUMS):
        assert_vec(-Vec(x, y, z), -x, -y, -z)
        assert_vec(+Vec(x, y, z), +x, +y, +z)


def test_mag(py_c_vec: PyCVec):
    """Test magnitude methods."""
    Vec = vec_mod.Vec
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

        if x == y == z == 0:
            # Vec(0, 0, 0).norm() = 0, 0, 0
            # Not ZeroDivisionError
            assert_vec(vec.norm(), 0, 0, 0)
        else:
            assert_vec(vec.norm(), x/length, y/length, z/length, vec)


def test_contains(py_c_vec: PyCVec):
    # Match to list.__contains__
    Vec = vec_mod.Vec
    for num in VALID_NUMS:
        for x, y, z in iter_vec(VALID_NUMS):
            assert (num in Vec(x, y, z)) == (num in [x, y, z])


def test_iteration(frozen_thawed_vec: VecClass):
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


def test_rev_iteration(frozen_thawed_vec: VecClass):
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


def test_vec_lerp(frozen_thawed_vec: VecClass):
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


def test_scalar(frozen_thawed_vec: VecClass):
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
                with pytest.raises(TypeError):
                    op_func(targ, obj)
                    pytest.fail('Vec ' + op_name + 'Scalar succeeded.')
                with pytest.raises(TypeError):
                    op_func(obj, targ)
                    pytest.fail('Scalar ' + op_name + ' Vec succeeded.')
                with pytest.raises(TypeError):
                    op_ifunc(targ, obj)
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


def test_vec_to_vec(py_c_vec: PyCVec):
    """Check that Vec() +/- Vec() does the correct thing.

    For +, -, two Vectors apply the operations to all values.
    Dot and cross products do something different.
    """
    Vec = vec_mod.Vec
    operators = [
        ('+', op.add, op.iadd),
        ('-', op.sub, op.isub),
    ]

    def test(x1, y1, z1, x2, y2, z2):
        """Check a Vec pair for addition and subtraction."""
        vec1 = Vec(x1, y1, z1)
        vec2 = Vec(x2, y2, z2)

        # These are direct methods, so no inheritence and iop to deal with.

        # Commutative
        assert vec1.dot(vec2) == (x1*x2 + y1*y2 + z1*z2)
        assert vec2.dot(vec1) == (x1*x2 + y1*y2 + z1*z2)
        assert_vec(
            vec1.cross(vec2),
            y1*z2-z1*y2,
            z1*x2-x1*z2,
            x1*y2-y1*x2,
        )
        # Ensure they haven't modified the originals
        assert_vec(vec1, x1, y1, z1)
        assert_vec(vec2, x2, y2, z2)

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
                msg='Vec({} {} {}) {} Vec({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
            )
            # Ensure they haven't modified the originals
            assert_vec(vec1, x1, y1, z1)
            assert_vec(vec2, x2, y2, z2)

            assert_vec(
                op_func(vec1, Vec_tuple(x2, y2, z2)),
                *result,
                msg='Vec({} {} {}) {} Vec_tuple({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
            )
            assert_vec(vec1, x1, y1, z1)

            assert_vec(
                op_func(Vec_tuple(x1, y1, z1), vec2),
                *result,
                msg='Vec_tuple({} {} {}) {} Vec({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
            )

            assert_vec(vec2, x2, y2, z2)

            new_vec1 = Vec(x1, y1, z1)
            assert_vec(
                op_ifunc(new_vec1, vec2),
                *result,
                msg='Return val: ({} {} {}) {}= ({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
            )
            # Check it modifies the original object too.
            assert_vec(
                new_vec1,
                *result,
                msg='Original: ({} {} {}) {}= ({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
            )

            new_vec1 = Vec(x1, y1, z1)
            assert_vec(
                op_ifunc(new_vec1, tuple(vec2)),
                *result,
                msg='Return val: ({} {} {}) {}= tuple({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
            )
            # Check it modifies the original object too.
            assert_vec(
                new_vec1,
                *result,
                msg='Original: ({} {} {}) {}= tuple({} {} {})'.format(
                    x1, y1, z1, op_name, x2, y2, z2,
                )
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

        with raises_zero_div: vec / 0
        with raises_zero_div: vec // 0
        with raises_zero_div: vec % 0
        with raises_zero_div: divmod(vec, 0)
        with raises_zero_div: vec / 0.0
        with raises_zero_div: vec // 0.0
        with raises_zero_div: vec % 0.0
        with raises_zero_div: divmod(vec, 0.0)

        with raises_zero_div: vec /= 0
        with raises_zero_div: vec //= 0
        with raises_zero_div: vec %= 0
        with raises_zero_div: vec /= 0.0
        with raises_zero_div: vec //= 0.0
        with raises_zero_div: vec %= 0.0


def test_divmod_vec_scalar(py_c_vec):
    """Test divmod(vec, scalar)."""
    Vec = vec_mod.Vec

    for x, y, z in iter_vec(VALID_ZERONUMS):
        for num in VALID_NUMS:
            div, mod = divmod(Vec(x, y, z), num)
            assert_vec(div, x // num, y // num, z // num)
            assert_vec(mod, x % num, y % num, z % num)


def test_divmod_scalar_vec(py_c_vec):
    """Test divmod(scalar, vec)."""
    Vec = vec_mod.Vec

    for x, y, z in iter_vec(VALID_NUMS):
        for num in VALID_ZERONUMS:
            div, mod = divmod(num, Vec(x, y, z))
            assert_vec(div, num // x, num // y, num // z)
            assert_vec(mod, num % x, num % y, num % z)


def test_vector_mult_fail(py_c_vec):
    """Test *, /, //, %, divmod always fails between vectors."""
    Vec = vec_mod.Vec

    funcs = [
        ('*', op.mul),
        ('/', op.truediv),
        ('//', op.floordiv),
        ('%', op.mod),
        ('*=', op.imul),
        ('/=', op.itruediv),
        ('//=', op.ifloordiv),
        ('%=', op.imod),
        ('divmod', divmod),
    ]
    for name, func in funcs:
        msg = 'Expected TypError from vec {} vec'.format(name)
        for num in VALID_ZERONUMS:
            for num2 in VALID_NUMS:
                # Test the whole value, then each axis individually
                with raises_typeerror:
                    divmod(Vec(num, num, num), Vec(num2, num2, num2))
                    pytest.fail(msg)

                with raises_typeerror:
                    divmod(Vec(0, num, num), Vec(num2, num2, num2))
                    pytest.fail(msg)
                with raises_typeerror:
                    divmod(Vec(num, 0, num), Vec(num2, num2, num2))
                    pytest.fail(msg)
                with raises_typeerror:
                    divmod(Vec(num, num, 0), Vec(num2, num2, num2))
                    pytest.fail(msg)
                with raises_typeerror:
                    divmod(Vec(num, num, num), Vec(0, num2, num2))
                    pytest.fail(msg)
                with raises_typeerror:
                    divmod(Vec(num, num, num), Vec(num2, 0, num2))
                    pytest.fail(msg)
                with raises_typeerror:
                    divmod(Vec(num, num, num), Vec(num2, num2, 0))
                    pytest.fail(msg)


def test_order(py_c_vec):
    """Test ordering operations (>, <, <=, >=, ==)."""
    Vec = vec_mod.Vec

    comp_ops = [op.eq, op.le, op.lt, op.ge, op.gt, op.ne]

    def test(x1, y1, z1, x2, y2, z2):
        """Check a Vec pair for incorrect comparisons."""
        vec1 = Vec(x1, y1, z1)
        vec2 = Vec(x2, y2, z2)
        for op_func in comp_ops:
            if op_func is op.ne:
                # special-case - != uses or, not and
                corr_result = x1 != x2 or y1 != y2 or z1 != z2
            else:
                corr_result = op_func(x1, x2) and op_func(y1, y2) and op_func(z1, z2)
            comp = (
                'Incorrect {{}} comparison for '
                '({} {} {}) {} ({} {} {})'.format(
                    x1, y1, z1, op_func.__name__, x2, y2, z2
                )
            )
            assert corr_result == op_func(vec1, vec2), comp.format('Vec')
            assert corr_result == op_func(vec1, Vec_tuple(x2, y2, z2)), comp.format('Vec_tuple')
            assert corr_result == op_func(vec1, (x2, y2, z2)), comp.format('tuple')

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


def test_binop_fail(py_c_vec):
    """Test binary operations with invalid operands."""
    Vec = vec_mod.Vec

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


def test_axis(py_c_vec):
    """Test the Vec.axis() function."""
    Vec = vec_mod.Vec

    for num in VALID_NUMS:
        assert Vec(num, 0, 0).axis() == 'x', num
        assert Vec(0, num, 0).axis() == 'y', num
        assert Vec(0, 0, num).axis() == 'z', num

        assert Vec(num, 1e-8, -0.0000008).axis() == 'x', num
        assert Vec(-1e-8, num, -0.0000008).axis() == 'y', num
        assert Vec(-0.000000878, 0.0000003782, num).axis() == 'z', num

        with raises_valueerror:
            Vec(num, num, 0).axis()

        with raises_valueerror:
            Vec(num, 0, num).axis()

        with raises_valueerror:
            Vec(0, num, num).axis()

        with raises_valueerror:
            Vec(num, num, num).axis()

        with raises_valueerror:
            Vec(-num, num, num).axis()

        with raises_valueerror:
            Vec(num, -num, num).axis()

        with raises_valueerror:
            Vec(num, num, -num).axis()

    with raises_valueerror:
        Vec().axis()


def test_other_axes(py_c_vec):
    """Test Vec.other_axes()."""
    Vec = vec_mod.Vec

    bad_args = ['p', '', 0, 1, 2, False, Vec(2, 3, 5)]
    for x, y, z in iter_vec(VALID_NUMS):
        vec = Vec(x, y, z)
        assert vec.other_axes('x') == (y, z)
        assert vec.other_axes('y') == (x, z)
        assert vec.other_axes('z') == (x, y)
        # Test some bad args.
        for invalid in bad_args:
            with raises_keyerror: vec.other_axes(invalid)


def test_abs(py_c_vec):
    """Test the function of abs(Vec)."""
    Vec = vec_mod.Vec

    for x, y, z in iter_vec(VALID_ZERONUMS):
        assert_vec(abs(Vec(x, y, z)), abs(x), abs(y), abs(z))


def test_bool(py_c_vec):
    """Test bool() applied to Vec."""
    Vec = vec_mod.Vec

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


def test_iter_line(py_c_vec):
    """Test Vec.iter_line()"""
    Vec = vec_mod.Vec
    for pos, x in zip(Vec(4, 5.82, -6.35).iter_line(Vec(10, 5.82, -6.35), 1), range(4, 11)):
        assert_vec(pos, x, 5.82, -6.35)
    for pos, y in zip(Vec(-4.36, 10.82, -6.35).iter_line(Vec(-4.36, 5.82, -6.35), 1), range(10, 4, -1)):
        assert_vec(pos, -4.36, y + 0.82, -6.35)
    for pos, z in zip(Vec(3.78, 12.98, -5.65).iter_line(Vec(3.78, 12.98, 6.35), 1), range(-6, 7)):
        assert_vec(pos, 3.78, 12.98, z + 0.35)


def test_iter_grid(py_c_vec):
    """Test Vec.iter_grid()."""
    Vec = vec_mod.Vec
    it = Vec.iter_grid(Vec(35, 59.99999, 90), Vec(40, 70, 110.001), 5)

    assert_vec(next(it), 35, 60, 90)
    assert_vec(next(it), 35, 60, 95)
    assert_vec(next(it), 35, 60, 100)
    assert_vec(next(it), 35, 60, 105)
    assert_vec(next(it), 35, 60, 110)

    assert_vec(next(it), 35, 65, 90)
    assert_vec(next(it), 35, 65, 95)
    assert_vec(next(it), 35, 65, 100)
    assert_vec(next(it), 35, 65, 105)
    assert_vec(next(it), 35, 65, 110)

    assert_vec(next(it), 35, 70, 90)
    assert_vec(next(it), 35, 70, 95)
    assert_vec(next(it), 35, 70, 100)
    assert_vec(next(it), 35, 70, 105)
    assert_vec(next(it), 35, 70, 110)

    assert_vec(next(it), 40, 60, 90)
    assert_vec(next(it), 40, 60, 95)
    assert_vec(next(it), 40, 60, 100)
    assert_vec(next(it), 40, 60, 105)
    assert_vec(next(it), 40, 60, 110)

    assert_vec(next(it), 40, 65, 90)
    assert_vec(next(it), 40, 65, 95)
    assert_vec(next(it), 40, 65, 100)
    assert_vec(next(it), 40, 65, 105)
    assert_vec(next(it), 40, 65, 110)

    assert_vec(next(it), 40, 70, 90)
    assert_vec(next(it), 40, 70, 95)
    assert_vec(next(it), 40, 70, 100)
    assert_vec(next(it), 40, 70, 105)
    assert_vec(next(it), 40, 70, 110)

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


def test_setitem(py_c_vec):
    """Test vec[x]=y with various args."""
    Vec = vec_mod.Vec

    for ind, axis in enumerate('xyz'):
        vec1 = Vec()
        vec1[axis] = 20.3
        assert vec1[axis] == 20.3, axis
        assert vec1.other_axes(axis) == (0.0, 0.0), axis

        vec2 = Vec()
        vec2[ind] = 20.3
        assert_vec(vec1, vec2.x, vec2.y, vec2.z, axis)

    vec = Vec()
    for invalid in INVALID_KEYS:
        try:
            vec[invalid] = 8
        except KeyError:
            pass
        else:
            pytest.fail(f"Key succeeded: {invalid!r}")
        assert_vec(vec, 0, 0, 0, 'Invalid key set something!')


def test_vec_constants(py_c_vec):
    """Check some of the constants assigned to Vec."""
    Vec = vec_mod.Vec

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


def test_round(py_c_vec):
    """Test round(Vec)."""
    Vec = vec_mod.Vec

    for from_val, to_val in ROUND_VALS:
        assert_vec(round(Vec(from_val, from_val, from_val)), to_val, to_val, to_val)

    # Check it doesn't mix up orders..
    for val in VALID_NUMS:
        assert_vec(round(Vec(val, 0, 0)), round(val), 0, 0)
        assert_vec(round(Vec(0, val, 0)), 0, round(val), 0)
        assert_vec(round(Vec(0, 0, val)), 0, 0, round(val))

MINMAX_VALUES = [
    (0, 0),
    (1, 0),
    (-5, -5),
    (0.3, 0.4),
    (-0.3, -0.2),
]
MINMAX_VALUES += [(b, a) for a,b in MINMAX_VALUES]


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


def test_copy_pickle(py_c_vec):
    """Test pickling and unpickling and copying Vectors."""
    Vec = vec_mod.Vec

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

    dcpy = copy.deepcopy(orig)

    assert orig is not dcpy
    assert orig == dcpy

    pick = pickle.dumps(orig)
    thaw = pickle.loads(pick)

    assert orig is not thaw
    assert orig == thaw

    # Ensure both produce the same pickle - so they can be interchanged.
    cy_pick = pickle.dumps(Cy_Vec(test_data))
    py_pick = pickle.dumps(Py_Vec(test_data))

    assert cy_pick == py_pick == pick


def test_bbox(py_c_vec: PyCVec):
    """Test the functionality of Vec.bbox()."""
    Vec = vec_mod.Vec

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
    with raises_valueerror:
        Vec.bbox([])

    # Iterable starting with non-vector.
    with raises_typeerror:
        Vec.bbox([None])

    # Iterable containing non-vector.
    with raises_typeerror:
        Vec.bbox([Vec(), None])

    def test(*values):
        """Test these values work."""
        min_x = min([v.x for v in values])
        min_y = min([v.y for v in values])
        min_z = min([v.z for v in values])

        max_x = max([v.x for v in values])
        max_y = max([v.y for v in values])
        max_z = max([v.z for v in values])

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


def test_vmf_rotation(datadir: Path, py_c_vec: PyCVec):
    """Complex test.

    Use a compiled map to check the functionality of Vec.rotate().
    """
    Vec = vec_mod.Vec
    Matrix = vec_mod.Matrix
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

        msg = '{} @ {} => ({}, {}, {})'.format(local_vec, angles, x, y, z)

        assert_vec(Vec(local_vec).rotate_by_str(angle_str), x, y, z, msg, tol=1e-3)
        assert_vec(Vec(local_vec).rotate(*angles), x, y, z, msg, tol=1e-3)
        assert_vec(Vec(local_vec) @ angles, x, y, z, msg, tol=1e-3)


def test_cross_product_axes(frozen_thawed_vec: VecClass):
    """Check all the cross product identities."""
    Vec = frozen_thawed_vec

    assert_vec(Vec.cross(Vec(x=1), Vec(y=1)), 0, 0, 1)
    assert_vec(Vec.cross(Vec(x=1), Vec(z=1)), 0, -1, 0)
    assert_vec(Vec.cross(Vec(y=1), Vec(z=1)), 1, 0, 0)
    assert_vec(Vec.cross(Vec(y=1), Vec(x=1)), 0, 0, -1)
    assert_vec(Vec.cross(Vec(z=1), Vec(x=1)), 0, 1, 0)
    assert_vec(Vec.cross(Vec(z=1), Vec(y=1)), -1, 0, 0)


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
