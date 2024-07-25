"""Test functionality in srctools.__init__."""
from typing import Any
from pathlib import Path

from dirty_equals import IsList
import pytest

from helpers import ExactType
from srctools import (
    EmptyMapping, _cy_conv_bool, _cy_conv_float, _cy_conv_int, _py_conv_bool,
    _py_conv_float, _py_conv_int,
)
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
non_bools = ['', 'noe', 'tru', 'fals', None, object()]

# We want to pass through all object types unchanged as defaults.
def_vals = [
    1, 0, True, False, None, object(),
    TrueObject(), FalseObject(), 456.9,
    -4758.97
]


def check_empty_iterable(obj: Any, name: str, item: object = 'x') -> None:
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


def test_exacttype_helper() -> None:
    """Check this helper does the right thing."""
    assert 45 == ExactType(45)
    assert 45 != ExactType(45.0)
    assert 48 != ExactType(45)
    truth = True
    assert truth != ExactType(1)


def test_bool_as_int() -> None:
    """Test result of srctools.bool_as_int."""
    for val in true_vals:
        assert srctools.bool_as_int(val) == '1', repr(val)
    for val in false_vals:
        assert srctools.bool_as_int(val) == '0', repr(val)


@pytest.mark.parametrize(
    'func',
    [_py_conv_int, _cy_conv_int],
    ids=['Python', 'Cython'],
)
def test_conv_int(func) -> None:
    """Test srctools.conv_int()."""
    for st_int, result in ints:
        assert func(st_int) == result, st_int

    # Check that float values fail
    marker = object()
    for st_float, result in floats:
        if isinstance(st_float, str):  # We don't want to check float-rounding
            assert func(st_float, marker) is marker, repr(st_float)

    # Check non-integers return the default.
    for st_obj in non_ints:
        assert func(st_obj) == 0
        for default in def_vals:
            # Check all default values pass through unchanged
            assert func(st_obj, default) is default, repr(st_obj)  # type: ignore


@pytest.mark.parametrize(
    'func',
    [_py_conv_bool, _cy_conv_bool],
    ids=['Python', 'Cython'],
)
def test_conv_bool(func) -> None:
    """Test srctools.conv_bool()."""
    for val in true_strings:
        assert func(val) is True
    for val in false_strings:
        assert func(val) is False

    # Check that bools pass through
    assert func(True) is True
    assert func(False) is False

    # None passes through the default
    for val in non_bools:
        # Check default value.
        assert func(val) is False
        # Check all default values pass through unchanged
        for default in def_vals:
            assert func(val, default) is default


@pytest.mark.parametrize(
    'func',
    [_py_conv_float, _cy_conv_float],
    ids=['Python', 'Cython'],
)
def test_conv_float(func) -> None:
    """Test srctools.conv_float()."""
    # Float should convert integers too
    for string, result in ints:
        assert func(string) == float(result)
        assert func(string) == result

    for string in non_floats:
        # Default default value
        assert func(string) == 0.0
        for default in def_vals:
            # Check all default values pass through unchanged
            assert func(string, default) is default


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
    assert srctools.escape_quote_split('abcdef') == ['abcdef']
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


def test_atomic_writer_bytes(tmp_path: Path) -> None:
    """Test the atomic writer functionality, with bytes."""
    filename = tmp_path / 'filename.txt'
    filename.write_bytes(b'existing_data')
    writer = srctools.AtomicWriter(filename, is_bytes=True)
    assert filename.read_bytes() == b'existing_data'
    assert list(tmp_path.iterdir()) == [filename]  # No other files yet.
    with writer as dest:
        assert filename.read_bytes() == b'existing_data'  # Still not altered.
        # Tempfile appears.
        assert list(tmp_path.iterdir()) == IsList(filename, Path(dest.name), check_order=False)
        dest.write(b'the new data 1234')
        assert filename.read_bytes() == b'existing_data'
    assert list(tmp_path.iterdir()) == [filename]   # Overwritten.
    assert filename.read_bytes() == b'the new data 1234'


def test_atomic_writer_text(tmp_path: Path) -> None:
    """Test the atomic writer functionality, with text."""
    filename = tmp_path / 'filename.txt'
    filename.write_text('existing_data', encoding='utf16')
    writer = srctools.AtomicWriter(filename, is_bytes=False, encoding='utf16')
    assert filename.read_text('utf16') == 'existing_data'
    assert list(tmp_path.iterdir()) == [filename]  # No other files yet.
    with writer as dest:
        assert filename.read_text('utf16') == 'existing_data'  # Still not altered.
        # Tempfile appears.
        assert list(tmp_path.iterdir()) == IsList(filename, Path(dest.name), check_order=False)
        dest.write('the new data 1234')
        assert filename.read_text('utf16') == 'existing_data'
    assert list(tmp_path.iterdir()) == [filename]  # Overwritten.
    assert filename.read_text('utf16') == 'the new data 1234'


def test_atomic_writer_fails(tmp_path: Path) -> None:
    """Test that it cleans up on failure."""
    filename = tmp_path / 'filename.txt'
    filename.write_bytes(b'existing_data')
    writer = srctools.AtomicWriter(filename, is_bytes=True)
    assert filename.read_bytes() == b'existing_data'
    assert list(tmp_path.iterdir()) == [filename]  # No other files yet.
    with pytest.raises(ZeroDivisionError):
        with writer as dest:
            dest.write(b'some partial new data')
            # File was created.
            assert list(tmp_path.iterdir()) == IsList(filename, Path(dest.name), check_order=False)
            raise ZeroDivisionError
    assert list(tmp_path.iterdir()) == [filename]   # Temp file is removed.
    # And data wasn't changed.
    assert filename.read_bytes() == b'existing_data'
