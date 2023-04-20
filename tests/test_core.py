"""Test functionality in srctools.__init__."""
from typing import Any

import pytest

from srctools import EmptyMapping
import srctools


class FalseObject:
    """Test object which is always False."""
    def __bool__(self) -> bool:
        return False


class TrueObject:
    """Test object which is always True."""
    def __bool__(self) -> bool:
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


def check_empty_iterable(obj: Any, name: str, item: object='x') -> None:
    """Check the given object is iterable, and is empty."""
    try:
        iterator = iter(obj)
    except TypeError as exc:
        raise AssertionError(f'{name} is not iterable!') from exc
    else:
        assert item not in obj
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)


def test_bool_as_int() -> None:
    """Test result of srctools.bool_as_int."""
    for val in true_vals:
        assert srctools.bool_as_int(val) == '1', repr(val)
    for val in false_vals:
        assert srctools.bool_as_int(val) == '0', repr(val)


def test_conv_int() -> None:
    for st_int, result in ints:
        assert srctools.conv_int(st_int) == result, st_int

    # Check that float values fail
    marker = object()
    for st_float, result in floats:
        if isinstance(st_float, str):  # We don't want to check float-rounding
            assert srctools.conv_int(st_float, marker) is marker, repr(st_float)

    # Check non-integers return the default.
    for st_obj in non_ints:
        assert srctools.conv_int(st_obj) == 0
        for default in def_vals:
            # Check all default values pass through unchanged
            assert srctools.conv_int(st_obj, default) is default, repr(st_obj)  # type: ignore


def test_conv_bool() -> None:
    """Test srctools.conv_bool()"""
    for val in true_strings:
        assert srctools.conv_bool(val)
    for val in false_strings:
        assert not srctools.conv_bool(val)

    # Check that bools pass through
    assert srctools.conv_bool(True)
    assert not srctools.conv_bool(False)

    # None passes through the default
    for val in def_vals:
        assert srctools.conv_bool(None, val) is val


def test_conv_float() -> None:
    # Float should convert integers too
    for string, result in ints:
        assert srctools.conv_float(string) == float(result)
        assert srctools.conv_float(string) == result

    for string in non_floats:
        # Default default value
        assert srctools.conv_float(string) == 0.0
        for default in def_vals:
            # Check all default values pass through unchanged
            assert srctools.conv_float(string, default) is default


# noinspection PyStatementEffect, PyCallingNonCallable
def test_EmptyMapping() -> None:
    marker = object()
    
    # It should be possible to 'construct' an instance..
    assert EmptyMapping() is EmptyMapping  # type: ignore

    # Must be passable to dict()
    assert dict(EmptyMapping) == {}

    # EmptyMapping['x'] raises in various forms.
    assert 'x' not in EmptyMapping
    with pytest.raises(KeyError):
        EmptyMapping['x']
    with pytest.raises(KeyError):
        del EmptyMapping['x']

    EmptyMapping['x'] = 4  # Shouldn't fail

    assert 'x' not in EmptyMapping  # but it's a no-op
    with pytest.raises(KeyError):
        EmptyMapping['x']

    # Check it's all empty
    check_empty_iterable(EmptyMapping, 'EmptyMapping')

    # Dict methods 
    assert EmptyMapping.get('x') is None
    assert EmptyMapping.setdefault('x') is None

    assert EmptyMapping.get('x', marker) is marker
    assert EmptyMapping.setdefault('x', marker) is marker
    assert EmptyMapping.pop('x', marker) is marker

    with pytest.raises(KeyError):
        EmptyMapping.popitem()
    with pytest.raises(KeyError):
        EmptyMapping.pop('x')

    assert not EmptyMapping

    assert len(EmptyMapping) == 0

    # Should work, but do nothing and return None.
    assert EmptyMapping.update({1: 23, 'test': 34, }) is None
    assert EmptyMapping.update(other=5, a=1, b=3) is None

    # Can't give more than one mapping as a positional argument,
    # though.
    with pytest.raises(TypeError):
        EmptyMapping.update({3: 4}, {1: 2})

    # Check it's registered in ABCs.
    from collections import abc
    assert isinstance(EmptyMapping, abc.Container)
    assert isinstance(EmptyMapping, abc.Sized)
    assert isinstance(EmptyMapping, abc.Mapping)
    assert isinstance(EmptyMapping, abc.MutableMapping)


def test_EmptyMapping_keys() -> None:
    """Test EmptyMapping.keys() works."""
    assert len(EmptyMapping.keys()) == 0
    assert object() not in EmptyMapping.keys()
    check_empty_iterable(EmptyMapping.keys(), 'keys()')

    # Check it's registered in ABCs.
    from collections import abc
    assert isinstance(EmptyMapping.keys(), abc.MappingView)
    assert isinstance(EmptyMapping.items(), abc.Set)
    assert isinstance(EmptyMapping.keys(), abc.KeysView)


def test_EmptyMapping_values() -> None:
    """Test EmptyMapping.values() works. This isn't a Set."""
    check_empty_iterable(EmptyMapping.values(), 'values()')

    # Check it's registered in ABCs.
    from collections import abc
    assert isinstance(EmptyMapping.values(), abc.MappingView)
    assert isinstance(EmptyMapping.values(), abc.ValuesView)


def test_EmptyMapping_items() -> None:
    """Test EmptyMapping.items() works."""
    check_empty_iterable(EmptyMapping.items(), 'items()', item=('x', 'y'))

    # Check it's registered in ABCs.
    from collections import abc
    assert isinstance(EmptyMapping.items(), abc.MappingView)
    assert isinstance(EmptyMapping.items(), abc.Set)
    assert isinstance(EmptyMapping.items(), abc.ItemsView)


@pytest.mark.parametrize(
    'view',
    [EmptyMapping.keys(), EmptyMapping.items()],
    ids=['keys', 'items'],
)
def test_EmptyMapping_set_ops(view) -> None:
    """Test EmptyMapping.keys() and items() support set ops."""
    empty: set[object] = set()
    # Ensure it's valid as an items() tuple.
    full = {('key', 1), ('key2', 4)}

    assert empty == view
    assert not (full == view)
    assert view == empty
    assert not (view == full)
    
    assert full != view
    assert not (empty != view)
    assert view != full
    assert not (view != empty)

    assert empty >= view
    assert full >= view
    assert view >= empty
    assert not (view >= full)

    assert not (empty > view)
    assert full > view
    assert not (view > empty)
    assert not (view > full)
    
    assert not (view < empty)
    assert view < full
    assert not (empty < view)
    assert not (full < view)
    
    assert view <= empty
    assert view <= full
    assert empty <= view
    assert not (full <= view)
    
    assert view.isdisjoint(full)
    assert view.isdisjoint(empty)

    assert (view | empty) == (empty | view) == empty
    assert (view | full) == (full | view) == full

    assert (view & empty) == (empty & view) == empty
    assert (view & full) == (full & view) == empty

    assert view - empty == empty
    assert view - full == empty
    assert empty - view == empty
    assert full - view == full

    assert (view ^ empty) == (empty ^ view) == empty
    assert (view ^ full) == (full ^ view) == full


def test_quote_escape() -> None:
    """Test escaping various quotes."""
    assert srctools.escape_quote_split('abcdef') ==['abcdef']
    # No escapes, equivalent to str.split
    assert (
        srctools.escape_quote_split('"abcd"ef""  " test') ==
        '"abcd"ef""  " test'.split('"')
    )

    assert (
        srctools.escape_quote_split(r'"abcd"ef\""  " test') ==
        ['', 'abcd', 'ef"', '  ', ' test']
    )
    # Check double-quotes next to others, and real quotes
    assert (
        srctools.escape_quote_split(r'"test\"\"" blah"') ==
        ['', 'test""', ' blah', '']
    )
