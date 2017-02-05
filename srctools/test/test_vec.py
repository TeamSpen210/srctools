"""Test the Vector object."""
import pytest
import operator as op

from srctools import Vec_tuple, Vec


VALID_NUMS = [
    1, 1.5, 0.2827, 2346.45,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = VALID_NUMS + [0, -0]

def iter_vec(nums):
    for x in nums:
        for y in nums:
            for z in nums:
                yield x, y, z


def assert_vec(vec, x, y, z, msg=''):
    """Asserts that Vec is equal to (x,y,z)."""
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    # Ignore slight variations
    if not vec.x == pytest.approx(x):
        failed = 'x'
    elif not vec.y == pytest.approx(y):
        failed = 'y'
    elif not vec.z == pytest.approx(z):
        failed = 'z'
    else:
        # Success!
        return

    new_msg = "{!r} != ({}, {}, {})".format(vec, failed, x, y, z)
    if msg:
        new_msg += ': ' + msg
    pytest.fail(new_msg)


def test_construction():
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
        # Check copying keeps the same values..
        assert_vec(Vec(x, y, z).copy(), x, y, z)

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

    # Check failures in Vec.from_str()
    # Note - does not pass through unchanged, they're converted to floats!
    for val in VALID_ZERONUMS:
        assert val == Vec.from_str('', x=val).x
        assert val == Vec.from_str('blah 4 2', y=val).y
        assert val == Vec.from_str('2 hi 2', x=val).x
        assert val == Vec.from_str('2 6 gh', z=val).z
        assert val == Vec.from_str('1.2 3.4', x=val).x
        assert val == Vec.from_str('34.5 38.4 -23 -38', z=val).z


def test_scalar():
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

    for op_name, op_func, op_ifunc, domain in operators:
        for x, y, z in iter_vec(domain):
            for num in domain:
                targ = Vec(x, y, z)
                rx, ry, rz = (
                    op_func(x, num),
                    op_func(y, num),
                    op_func(z, num),
                )

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


def test_vec_plus_or_minus_vec():
    """Check that Vec() +/- Vec() does the correct thing.

    For +, -, two Vectors apply the operations to all values.
    """
    operators = [
        ('+', op.add, op.iadd),
        ('-', op.sub, op.isub),
    ]

    def test(x1, y1, z1, x2, y2, z2):
        """Check a Vec pair for addition and subtraction."""
        vec1 = Vec(x1, y1, z1)
        vec2 = Vec(x2, y2, z2)
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


def test_scalar_zero():
    """Check zero behaviour with division ops."""
    for x, y, z in iter_vec(VALID_NUMS):
        assert_vec(0 / Vec(x, y, z), 0, 0, 0)
        assert_vec(0 // Vec(x, y, z), 0, 0, 0)
        assert_vec(0 % Vec(x, y, z), 0, 0, 0)


def test_order():
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
            assert op_func(vec1, vec2) == corr_result, comp.format('Vec')
            assert op_func(vec1, Vec_tuple(x2, y2, z2)) == corr_result, comp.format('Vec_tuple')
            assert op_func(vec1, (x2, y2, z2)) == corr_result, comp.format('tuple')
            # Bare numbers compare magnitude..
            assert op_func(vec1, x2) == op_func(vec1.mag(), x2), comp.format('x')
            assert op_func(vec1, y2) == op_func(vec1.mag(), y2), comp.format('y')
            assert op_func(vec1, z2) == op_func(vec1.mag(), z2), comp.format('z')

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


def test_axis():
    """Test the Vec.axis() function."""
    assert Vec(1, 0, 0).axis() == 'x'
    assert Vec(-1, 0, 0).axis() == 'x'
    assert Vec(0, 1, 0).axis() == 'y'
    assert Vec(0, -1, 0).axis() == 'y'
    assert Vec(0, 0, 1).axis() == 'z'
    assert Vec(0, 0, -1).axis() == 'z'


def test_abs():
    """Test the function of abs(Vec)."""
    for x, y, z in iter_vec(VALID_ZERONUMS):
        assert_vec(abs(Vec(x, y, z)), abs(x), abs(y), abs(z))


def test_bool():
    """Test bool() applied to Vec."""
    # Empty vector is False
    assert not Vec(0, 0, 0)
    assert not Vec(-0, -0, -0)
    for val in VALID_NUMS:
        # Any number in any axis makes it True.
        assert Vec(val, 0, 0)
        assert Vec(0, val, 0)
        assert Vec(0, 0, val)
        assert Vec(0, val, val)
        assert Vec(val, 0, val)
        assert Vec(val, val, 0)
        assert Vec(val, val, val)


def test_vec_constants():
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
