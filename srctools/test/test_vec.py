"""Test the Vector object."""
import math
import pickle
import copy

import pytest
import operator as op
import srctools
from srctools import Vec_tuple
from srctools.vec import Py_Vec, Cy_Vec
from typing import Type

try:
    from importlib.resources import path as import_file_path
except ImportError:
    from importlib_resources import path as import_file_path


Vec = ...  # type: Type[srctools.Vec]

VALID_NUMS = [
    # 10e38 is the max single value, make sure we use double-precision.
    30, 1.5, 0.2827, 2.3464545636e47,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = VALID_NUMS + [0, -0]

# Reuse these context managers.
raises_typeerror = pytest.raises(TypeError)
raises_valueerror = pytest.raises(ValueError)
raises_keyerror = pytest.raises(KeyError)
raises_zero_div = pytest.raises(ZeroDivisionError)

if Py_Vec is Cy_Vec:
    parms = [Py_Vec]
    names = ['Python']
    print('No _vec! ')
else:
    parms = [Py_Vec, Cy_Vec]
    names = ['Python', 'Cython']


@pytest.fixture(params=parms, ids=names)
def py_c_vec(request):
    """Run the test twice, for the Python and C versions."""
    global Vec
    orig_vec = srctools.Vec
    Vec = request.param
    yield None
    Vec = orig_vec


def iter_vec(nums):
    for x in nums:
        for y in nums:
            for z in nums:
                yield x, y, z


def assert_vec(vec, x, y, z, msg=''):
    """Asserts that Vec is equal to (x,y,z)."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    assert type(vec).__name__ == 'Vec'

    if not math.isclose(vec.x, x):
        failed = 'x'
    elif not math.isclose(vec.y, y):
        failed = 'y'
    elif not math.isclose(vec.z, z):
        failed = 'z'
    else:
        # Success!
        return

    new_msg = "{!r}.{} != ({}, {}, {})".format(vec, failed, x, y, z)
    if msg:
        new_msg += ': ' + str(msg)
    pytest.fail(new_msg)


def test_construction(py_c_vec):
    """Check various parts of the constructor - Vec(), Vec.from_str()."""
    for x, y, z in iter_vec(VALID_ZERONUMS):
        assert_vec(Vec(x, y, z), x, y, z)
        assert_vec(Vec(x, y), x, y, 0)
        assert_vec(Vec(x), x, 0, 0)
        assert_vec(Vec(), 0, 0, 0)

        assert_vec(Vec([x, y, z]), x, y, z)
        assert_vec(Vec([x, y], z=z), x, y, z)
        assert_vec(Vec([x], y=y, z=z), x, y, z)
        assert_vec(Vec([x]), x, 0, 0)
        assert_vec(Vec([x, y]), x, y, 0)
        assert_vec(Vec([x, y, z]), x, y, z)

        # Test this does nothing (except copy).
        v = Vec(x, y, z)
        v2 = Vec(v)
        assert_vec(v2, x, y, z)
        assert v is not v2

        v3 = Vec.copy(v)
        assert_vec(v3, x, y, z)
        assert v is not v3

        # Test Vec.from_str()
        assert_vec(Vec.from_str('{} {} {}'.format(x, y, z)), x, y, z)
        assert_vec(Vec.from_str('<{} {} {}>'.format(x, y, z)), x, y, z)
        # {x y z}
        assert_vec(Vec.from_str('{{{} {} {}}}'.format(x, y, z)), x, y, z)
        assert_vec(Vec.from_str('({} {} {})'.format(x, y, z)), x, y, z)
        assert_vec(Vec.from_str('[{} {} {}]'.format(x, y, z)), x, y, z)

        # Test converting a converted Vec
        orig = Vec(x, y, z)
        new = Vec.from_str(Vec(x, y, z))
        assert_vec(new, x, y, z)
        assert orig is not new  # It must be a copy

        # Check as_tuple() makes an equivalent tuple
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

    # Check failures in Vec.from_str()
    # Note - does not pass through unchanged, they're converted to floats!
    for val in VALID_ZERONUMS:
        assert val == Vec.from_str('', x=val).x
        assert val == Vec.from_str('blah 4 2', y=val).y
        assert val == Vec.from_str('2 hi 2', x=val).x
        assert val == Vec.from_str('2 6 gh', z=val).z
        assert val == Vec.from_str('1.2 3.4', x=val).x
        assert val == Vec.from_str('34.5 38.4 -23 -38', z=val).z


def test_with_axes(py_c_vec):
    """Test the with_axes() constructor."""
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
            msg = '{} = {}={}, {}={}, {}={}'.format(vec, a, x, b, y, c, z)
            assert vec[a] == x, msg
            assert vec[b] == y, msg
            assert vec[c] == z, msg



def test_unary_ops(py_c_vec):
    """Test -vec and +vec."""
    for x, y, z in iter_vec(VALID_NUMS):
        assert_vec(-Vec(x, y, z), -x, -y, -z)
        assert_vec(+Vec(x, y, z), +x, +y, +z)


def test_mag(py_c_vec):
    """Test magnitude methods."""
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


def test_contains(py_c_vec):
    # Match to list.__contains__
    for num in VALID_NUMS:
        for x, y, z in iter_vec(VALID_NUMS):
            assert (num in Vec(x, y, z)) == (num in [x, y, z])


def test_scalar(py_c_vec):
    """Check that Vec() + 5, -5, etc does the correct thing.

    For +, -, *, /, // and % calling with a scalar should perform the
    operation on x, y, and z
    """
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
                with pytest.raises(TypeError, message='forward ' + op_name):
                    op_func(targ, obj)
                with pytest.raises(TypeError, message='backward ' + op_name):
                    op_func(obj, targ)
                with pytest.raises(TypeError, message='inplace ' + op_name):
                    op_ifunc(targ, obj)

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

                assert_vec(
                    op_ifunc(targ, num),
                    rx, ry, rz,
                    'Return value for ({} {} {}) {}= {}'.format(
                        x, y, z, op_name, num,
                    ),
                )
                # Check that the original was modified..
                assert_vec(
                    targ,
                    rx, ry, rz,
                    'Original for ({} {} {}) {}= {}'.format(
                        x, y, z, op_name, num,
                    ),
                )


def test_vec_to_vec(py_c_vec):
    """Check that Vec() +/- Vec() does the correct thing.

    For +, -, two Vectors apply the operations to all values.
    Dot and cross products do something different.
    """
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


def test_scalar_zero(py_c_vec):
    """Check zero behaviour with division ops."""
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
    for x, y, z in iter_vec(VALID_ZERONUMS):
        for num in VALID_NUMS:
            div, mod = divmod(Vec(x, y, z), num)
            assert_vec(div, x // num, y // num, z // num)
            assert_vec(mod, x % num, y % num, z % num)


def test_divmod_scalar_vec(py_c_vec):
    """Test divmod(scalar, vec)."""
    for x, y, z in iter_vec(VALID_NUMS):
        for num in VALID_ZERONUMS:
            div, mod = divmod(num, Vec(x, y, z))
            assert_vec(div, num // x, num // y, num // z)
            assert_vec(mod, num % x, num % y, num % z)


def test_vector_mult_fail(py_c_vec):
    """Test *, /, //, %, divmod always fails between vectors."""
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
        raises = pytest.raises(
            TypeError,
            message='Expected TypError from vec {} vec'.format(name),
        )
        for num in VALID_ZERONUMS:
            for num2 in VALID_NUMS:
                # Test the whole value, then each axis individually
                with raises:
                    divmod(Vec(num, num, num), Vec(num2, num2, num2))

                with raises:
                    divmod(Vec(0, num, num), Vec(num2, num2, num2))
                with raises:
                    divmod(Vec(num, 0, num), Vec(num2, num2, num2))
                with raises:
                    divmod(Vec(num, num, 0), Vec(num2, num2, num2))
                with raises:
                    divmod(Vec(num, num, num), Vec(0, num2, num2))
                with raises:
                    divmod(Vec(num, num, num), Vec(num2, 0, num2))
                with raises:
                    divmod(Vec(num, num, num), Vec(num2, num2, 0))


def test_order(py_c_vec):
    """Test ordering operations (>, <, <=, >=, ==)."""
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
    for num in VALID_NUMS:
        assert Vec(num, 0, 0).axis() == 'x', num
        assert Vec(0, num, 0).axis() == 'y', num
        assert Vec(0, 0, num).axis() == 'z', num

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
    for x, y, z in iter_vec(VALID_ZERONUMS):
        assert_vec(abs(Vec(x, y, z)), abs(x), abs(y), abs(z))


def test_bool(py_c_vec):
    """Test bool() applied to Vec."""
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


def test_len(py_c_vec):
    """Test len(Vec)."""
    # len(Vec) is the number of non-zero axes.

    assert len(Vec(0, 0, 0)) == 0
    assert len(Vec(-0, -0, -0)) == 0

    for val in VALID_NUMS:
        assert len(Vec(val, 0, -0)) == 1
        assert len(Vec(0, val, 0)) == 1
        assert len(Vec(0, -0, val)) == 1
        assert len(Vec(0, val, val)) == 2
        assert len(Vec(val, 0, val)) == 2
        assert len(Vec(val, val, -0)) == 2
        assert len(Vec(val, val, val)) == 3

INVALID_KEYS = [
    '4',
    '',
    -1,
    4,
    4.0,
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
        with raises_keyerror:
            v[invalid]


def test_setitem(py_c_vec):
    """Test vec[x]=y with various args."""
    for ind, axis in enumerate('xyz'):
        vec1 = Vec()
        vec1[axis] = 20.3
        assert vec1[axis] == 20.3, axis
        assert vec1.other_axes(axis) == (0.0, 0.0), axis

        vec2 = Vec()
        vec2[ind] = 20.3
        assert_vec(vec1, vec2.x, vec2.y, vec2.z, axis)

    vec = Vec()
    for invalid in ['4', '', -1, 4, 4.0, bool, slice(0, 1), Vec(2,3,4)]:
        with pytest.raises(KeyError, message=repr(invalid)):
            vec[invalid] = 8
        assert_vec(vec, 0, 0, 0, 'Invalid key set something!')


def test_vec_constants(py_c_vec):
    """Check some of the constants assigned to Vec."""
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
    for from_val, to_val in ROUND_VALS:
        assert round(Vec(from_val, from_val, from_val)) == Vec(to_val, to_val, to_val)

    # Check it doesn't mix up orders..
    for val in VALID_NUMS:
        assert round(Vec(val, 0, 0)) == Vec(round(val), 0, 0)
        assert round(Vec(0, val, 0)) == Vec(0, round(val), 0)
        assert round(Vec(0, 0, val)) == Vec(0, 0, round(val))

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

    test_data = 1.5, 0.2827, 2.3464545636e47

    orig = Vec(test_data)

    cpy_meth = orig.copy()

    assert orig is not cpy_meth  # Must be a new object.
    assert cpy_meth is not orig.copy()  # Cannot be cached
    assert orig == cpy_meth # Numbers must be exactly identical!

    cpy = copy.copy(orig)

    assert orig is not cpy
    assert cpy_meth is not copy.copy(orig)
    assert orig == cpy

    dcpy = copy.deepcopy(orig)

    assert orig is not dcpy
    assert orig == dcpy

    pick = pickle.loads(pickle.dumps(orig))

    assert orig is not pick
    assert orig == pick

    # TODO: Check both c/python version produce the same data.


def test_bbox(py_c_vec):
    """Test the functionality of Vec.bbox()."""

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


def test_vmf_rotation(py_c_vec):
    """Complex test.

    Use a compiled map to check the functionality of Vec.rotate().
    """
    from srctools.bsp import BSP
    import srctools.test

    with import_file_path(srctools.test, 'rot_main.bsp') as bsp_path:
        bsp = BSP(bsp_path)
        vmf = bsp.read_ent_data()
    del bsp

    for ent in vmf.entities:
        if ent['classname'] != 'info_target':
            continue
        angle_str = ent['angles']
        angles = Vec.from_str(angle_str)
        local_vec = Vec(
            float(ent['local_x']),
            float(ent['local_y']),
            float(ent['local_z']),
        )
        x, y, z = round(Vec.from_str(ent['origin']) / 128, 3)

        msg = '{} @ {} => ({}, {}, {})'.format(local_vec, angles, x, y, z)

        assert_vec(Vec(local_vec).rotate_by_str(angle_str), x, y, z, msg)
        assert_vec(Vec(local_vec).rotate(*angles), x, y, z, msg)

