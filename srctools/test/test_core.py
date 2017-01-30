"""Test functionality in srctools.__init__."""
from nose.tools import *  # assert_equal, assert_is, etc
import operator

from srctools import EmptyMapping
import srctools


class FalseObject:
    """Test object which is always False."""
    def __bool__(self):
        return False


class TrueObject:
    """Test object which is always True."""
    def __bool__(self):
        return True

true_vals = [1, 1.0, True, 'test', [2], (-1, ), TrueObject(), object()]
false_vals = [0, 0.0, False, '', [], (), FalseObject()]

ints = [
    ('0', 0),
    ('-0', -0),
    ('1', 1),
    ('12352343783', 12352343783),
    ('24', 24),
    ('-4784', -4784),
    ('28', 28),
    (1, 1),
    (-2, -2),
    (3783738378, 3783738378),
    (-23527, -23527),
]

floats = [
    ('0.0', 0.0),
    ('-0.0', -0.0),
    ('-4.5', -4.5),
    ('4.5', 4.5),
    ('1.2', 1.2),
    ('12352343783.189', 12352343783.189),
    ('24.278', 24.278),
    ('-4784.214', -4784.214),
    ('28.32', 28.32),
    (1.35, 1.35),
    (-2.26767, -2.26767),
    (338378.3246, 338378.234),
    (-23527.9573, -23527.9573),
]

false_strings = ['0', 'false', 'no', 'faLse', 'False', 'No', 'NO', 'nO']
true_strings = ['1', 'true', 'yes', 'True', 'trUe', 'Yes', 'yEs', 'yeS']

non_ints = ['-23894.0', '', 'hello', '5j', '6.2', '0.2', '6.9', None, object()]
non_floats = ['5j', '', 'hello', '6.2.5', '4F', '100-', None, object(), float]

# We want to pass through all object types unchanged as defaults.
def_vals = [
    1, 0, True, False, None, object(),
    TrueObject(), FalseObject(), 456.9,
    -4758.97
]


def check_empty_iterable(obj, name, item: object='x'):
    """Check the given object is iterable, and is empty."""
    try:
        iterator = iter(obj)
    except TypeError:
        raise AssertionError(name + ' is not iterable!')
    else:
        assert_not_in(item, obj)
        assert_raises(StopIteration, next, iterator)
        assert_raises(StopIteration, next, iterator)


def test_bool_as_int():
    for val in true_vals:
        assert_equal(srctools.bool_as_int(val), '1', repr(val))
    for val in false_vals:
        assert_equal(srctools.bool_as_int(val), '0', repr(val))


def test_conv_int():
    for string, result in ints:
        assert_equal(srctools.conv_int(string), result)


def test_conv_int_fails_on_float():
    # Check that float values fail
    marker = object()
    for string, result in floats:
        if isinstance(string, str):  # We don't want to check float-rounding
            assert_is(
                srctools.conv_int(string, marker),
                marker,
                msg=string,
            )


def test_conv_int_fails_on_str():
    for string in non_ints:
        assert_equal(srctools.conv_int(string), 0)
        for default in def_vals:
            # Check all default values pass through unchanged
            assert_is(srctools.conv_int(string, default), default)


def test_conv_bool():
    for val in true_strings:
        assert_true(srctools.conv_bool(val))
    for val in false_strings:
        assert_false(srctools.conv_bool(val))

    # Check that bools pass through
    assert_true(srctools.conv_bool(True))
    assert_false(srctools.conv_bool(False))

    # None passes through the default
    for val in def_vals:
        assert_is(srctools.conv_bool(None, val), val)


def test_conv_float():
    # Float should convert integers too
    for string, result in ints:
        assert_equal(srctools.conv_float(string), float(result))
        assert_equal(srctools.conv_float(string), result)

    for string in non_floats:
        assert_equal(srctools.conv_float(string), 0)
        for default in def_vals:
            # Check all default values pass through unchanged
            assert_is(srctools.conv_float(string, default), default)


def test_EmptyMapping():
    marker = object()
    
    # It should be possible to 'construct' an instance..
    assert_is(EmptyMapping(), EmptyMapping)

    # Must be passable to dict()
    assert_equal(dict(EmptyMapping), {})

    # EmptyMapping['x'] raises
    assert_raises(KeyError, operator.getitem, EmptyMapping, 'x')
    assert_raises(KeyError, operator.delitem, EmptyMapping, 'x')
    EmptyMapping['x'] = 4  # Shouldn't fail
    assert_not_in('x', EmptyMapping) # but it's a no-op

    # Check it's all empty
    check_empty_iterable(EmptyMapping, 'EmptyMapping')
    check_empty_iterable(EmptyMapping.keys(), 'keys()')
    check_empty_iterable(EmptyMapping.values(), 'values()')
    check_empty_iterable(EmptyMapping.items(), 'items()', item=('x', 'y'))

    # Dict methods 
    assert_is(EmptyMapping.get('x'), None)
    assert_is(EmptyMapping.setdefault('x'), None)
    assert_is(EmptyMapping.get('x', marker), marker)
    assert_is(EmptyMapping.setdefault('x', marker), marker)
    assert_is(EmptyMapping.pop('x', marker), marker)
    assert_raises(KeyError, EmptyMapping.popitem)
    assert_raises(KeyError, EmptyMapping.pop, 'x')
    assert_false(EmptyMapping)
    assert_equal(len(EmptyMapping), 0)
    EmptyMapping.update({1: 23, 'test': 34,})
    EmptyMapping.update(other=5, a=1, b=3)
    # Can't give more than one item..
    assert_raises(TypeError, lambda: EmptyMapping.update({3: 4}, {1: 2}))

    # Check it's registered in ABCs.
    from collections import abc
    assert_is_instance(EmptyMapping, abc.Container)
    assert_is_instance(EmptyMapping, abc.Sized)
    assert_is_instance(EmptyMapping, abc.Mapping)
    assert_is_instance(EmptyMapping, abc.MutableMapping)


def test_quote_escape():
    """Test escaping various quotes"""
    assert_equal(
        srctools.escape_quote_split('abcdef'),
        ['abcdef'],
    )
    # No escapes, equivalent to str.split
    assert_equal(
        srctools.escape_quote_split('"abcd"ef""  " test'),
        '"abcd"ef""  " test'.split('"'),
    )

    assert_equal(
        srctools.escape_quote_split(r'"abcd"ef\""  " test'),
        ['', 'abcd', 'ef"', '  ', ' test'],
    )
    # Check double-quotes next to others, and real quotes
    assert_equal(
        srctools.escape_quote_split(r'"test\"\"" blah"'),
        ['', 'test""', ' blah', ''],
    )
