"""Test tricky type definitions in the package root."""
from typing import Union
from typing_extensions import assert_type
from io import BufferedWriter, TextIOWrapper

import srctools


def test_conversions() -> None:
    """Check conversions + default interactions."""
    data: list[str] = ['a', 'b', 'c']
    assert_type(srctools.conv_int('45'), int)
    assert_type(srctools.conv_int('45', 32), int)
    assert_type(srctools.conv_int('45', data), Union[int, list[str]])

    assert_type(srctools.conv_float('45.25'), float)
    assert_type(srctools.conv_float('45.25', 32.25), float)
    assert_type(srctools.conv_float('45.25', data), Union[float, list[str]])

    assert_type(srctools.conv_bool('0'), bool)
    assert_type(srctools.conv_bool('true', data), Union[bool, list[str]])


def test_atomicwriter() -> None:
    with srctools.AtomicWriter('blah') as f1:
        assert_type(f1, TextIOWrapper)
        f1.write('Unicode')
        f1.write(b'Bytes')  # type: ignore

    with srctools.AtomicWriter('blah', is_bytes=False) as f2:
        assert_type(f2, TextIOWrapper)
        f2.write('Unicode')
        f2.write(b'Bytes')  # type: ignore

    with srctools.AtomicWriter('blah', is_bytes=True) as f3:
        assert_type(f3, BufferedWriter)
        f3.write(b'Bytes')
        f3.write('Unicode')  # type: ignore


def test_emptymapping() -> None:
    """Test some default parameters."""
    mapping: dict[str, int] = {'abc': 123}
    assert_type(srctools.EmptyMapping.get('key'), None)
    assert_type(srctools.EmptyMapping.get(45, mapping), dict[str, int])
    assert_type(srctools.EmptyMapping.setdefault('key'), None)
    assert_type(srctools.EmptyMapping.setdefault(45, mapping), dict[str, int])
