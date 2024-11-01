"""Test the datamodel exchange implementation."""
from typing import Callable, Set, Tuple, Type, cast
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4
import array
import collections

from dirty_equals import IsFloat, IsInt
import pytest

from helpers import *
from srctools import Angle, FrozenAngle, FrozenMatrix, FrozenVec, Keyvalues, Matrix, Vec
from srctools.dmx import (
    TYPE_CONVERT, Attribute, Color, Element, Quaternion, Time, ValueType, Vec2, Vec4,
    deduce_type,
)
from srctools.tokenizer import TokenSyntaxError


def assert_tree(tree1: Element, tree2: Element) -> None:
    """Checks both trees are identical, recursively."""
    return _assert_tree_elem(tree1.name, tree1, tree2, set())


def export(elem: Element, version: str, unicode: str = 'ascii') -> bytes:
    """Export a element and return the result, doing both text/binary in one for parameterisation."""
    buf = BytesIO()
    if version.startswith('binary_'):
        elem.export_binary(buf, int(version[-1]), unicode=unicode)
    else:
        elem.export_kv2(buf, flat='flat' in version, unicode=unicode, cull_uuid='cull' in version)
    return buf.getvalue()

EXPORT_VALS = [
    'binary_v1', 'binary_v2', 'binary_v3', 'binary_v4', 'binary_v5',
    'text_indent', 'text_flat', 'text_indent_cull', 'text_flat_cull'
]


def _assert_tree_elem(path: str, tree1: Element, tree2: Element, checked: Set[UUID]) -> None:
    """Checks two elements are the same."""
    if tree1 is None or tree2 is None:
        # Old code stored none for NULL elements
        pytest.fail(f'{path}: {tree1!r} <-> {tree2}')

    # Don't re-check (infinite loop).
    if tree1.uuid in checked:
        return
    checked.add(tree1.uuid)
    if tree1.uuid != tree2.uuid:
        pytest.fail(f'{path}: UUID {tree1.uuid.hex} != {tree2.uuid.hex}')
    if tree1.name != tree2.name:
        pytest.fail(f'{path}: name {tree1.name} != {tree2.name}')
    if tree1.type != tree2.type:
        pytest.fail(f'{path}: type {tree1.type} != {tree2.type}')

    for key in tree1.keys() | tree2.keys():
        attr_path = f'{path}.{key}'

        try:
            attr1 = tree1[key]
        except KeyError:
            return pytest.fail(f'{attr_path}: {key} not in LHS')
        try:
            attr2 = tree2[key]
        except KeyError:
            return pytest.fail(f'{attr_path}: {key} not in RHS')

        if attr1.type is not attr2.type:
            pytest.fail(f'{attr_path}: type {attr1.type} != {attr2.type}')
        assert attr1.name == attr2.name, attr_path

        if attr1.type is ValueType.ELEMENT:
            if attr1.is_array:
                assert len(attr1) == len(attr2), f'Mismatched len for {attr_path}'
                for i, elem1 in enumerate(attr1.iter_elem()):
                    _assert_tree_elem(f'{attr_path}[{i}]', elem1, attr2[i].val_elem, checked)
            else:
                _assert_tree_elem(attr_path, attr1.val_elem, attr2.val_elem, checked)
        elif attr1.is_array:
            assert len(attr1) == len(attr2), f'Mismatched len for {attr_path}'
            for i in range(len(attr1)):
                assert attr1[i].val_str == attr2[i].val_str, f'{attr_path}[{i}]: {attr1._value[i]} != {attr2._value[i]}'
        else:
            assert attr1.val_str == attr2.val_str, f'{attr_path}: {attr1._value} != {attr2._value}'


# ExactType() enforces an exact type check + value check.


def test_attr_val_int() -> None:
    """Test integer-type values."""
    elem = Attribute.int('Name', 45)
    assert elem.val_int == IsInt(exactly=45)
    assert elem.val_str == '45'
    assert elem.val_float == IsFloat(exactly=45.0)
    assert elem.val_time == ExactType(Time(45.0))

    assert elem.val_vec2 == Vec2(45.0, 45.0)
    assert elem.val_vec3 == ExactType(FrozenVec(45.0, 45.0, 45.0))
    assert elem.val_vec4 == Vec4(45.0, 45.0, 45.0, 45.0)
    assert elem.val_color == Color(45, 45, 45, 255)
    assert Attribute.int('min-clamp', -1).val_color == Color(0, 0, 0, 255)
    assert Attribute.int('max-clamp', 260).val_color == Color(255, 255, 255, 255)

    assert Attribute.int('Blah', 45).val_bool is True
    assert Attribute.int('Blah', 0).val_bool is False
    assert Attribute.int('Blah', -2).val_bool is True


def test_attr_array_int() -> None:
    """Test integer-type values in arrays."""
    elem: Attribute[int] = Attribute.array('Name', ValueType.INT)
    elem.append(-123)
    elem.append(45)
    assert list(elem.iter_int()) == [IsInt(exactly=-123), IsInt(exactly=45)]
    assert list(elem.iter_str()) == ['-123', '45']
    assert list(elem.iter_float()) == [IsFloat(exactly=-123.0), IsFloat(exactly=45.0)]
    assert list(elem.iter_time()) == [Time(-123.0), Time(45.0)]

    assert list(elem.iter_vec2()) == [Vec2(-123.0, -123.0), Vec2(45.0, 45.0)]
    assert list(elem.iter_vec3()) == [ExactType(FrozenVec(-123.0, -123.0, -123.0)), FrozenVec(45.0, 45.0, 45.0)]
    assert list(elem.iter_vec4()) == [Vec4(-123.0, -123.0, -123.0, -123.0), Vec4(45.0, 45.0, 45.0, 45.0)]
    assert list(elem.iter_color()) == [Color(0, 0, 0, 255), Color(45, 45, 45, 255)]

    elem[0] = 45
    elem[1] = 0
    elem.append(-2)
    assert list(elem.iter_bool()) == [True, False, True]


def test_attr_val_float() -> None:
    """Test float-type values."""
    elem = Attribute.float('Name', 32.25)
    assert elem.val_int == IsInt(exactly=32)
    assert Attribute.float('Name', -32.25).val_int == IsInt(exactly=-32)
    assert elem.val_str == '32.25'
    assert elem.val_float == IsFloat(exactly=32.25)
    assert elem.val_time == ExactType(Time(32.25))

    assert elem.val_vec2 == Vec2(32.25, 32.25)
    assert elem.val_vec3 == ExactType(FrozenVec(32.25, 32.25, 32.25))
    assert elem.val_vec4 == Vec4(32.25, 32.25, 32.25, 32.25)

    assert Attribute.float('Blah', 32.25).val_bool is True
    assert Attribute.float('Blah', 0.0).val_bool is False
    assert Attribute.float('Blah', -12.8).val_bool is True


def test_attr_array_float() -> None:
    """Test float-type values in arrays."""
    elem = Attribute.array('Name', ValueType.FLOAT)
    elem.append(-32.25)
    elem.append(32.25)

    assert list(elem.iter_int()) == [-32, 32]
    assert list(elem.iter_str()) == ['-32.25', '32.25']
    assert list(elem.iter_float()) == [-32.25, 32.25]
    assert list(elem.iter_time()) == [Time(-32.25), Time(32.25)]

    assert list(elem.iter_vec2()) == [Vec2(-32.25, -32.25), Vec2(32.25, 32.25)]
    assert list(elem.iter_vec3()) == [ExactType(FrozenVec(-32.25, -32.25, -32.25)), FrozenVec(32.25, 32.25, 32.25)]
    assert list(elem.iter_vec4()) == [Vec4(-32.25, -32.25, -32.25, -32.25), Vec4(32.25, 32.25, 32.25, 32.25)]

    elem[0] = 32.25
    elem[1] = 0.0
    elem.append(-12.8)
    assert list(elem.iter_bool()) == [True, False, True]


def test_attr_val_str() -> None:
    """Test string-type values."""
    assert Attribute.string('', '45').val_str == '45'
    assert Attribute.string('', '').val_str == ''
    assert Attribute.string('', 'testing str\ning').val_str == 'testing str\ning'

    assert Attribute.string('Name', '45').val_int == IsInt(exactly=45)
    assert Attribute.string('Name', '-45').val_int == IsInt(exactly=-45)
    assert Attribute.string('Name', '0').val_int == IsInt(exactly=0)

    assert Attribute.string('', '45').val_float == IsFloat(exactly=45.0)
    assert Attribute.string('', '45.0').val_float == IsFloat(exactly=45.0)
    assert Attribute.string('', '45.375').val_float == IsFloat(exactly=45.375)
    assert Attribute.string('', '-45.375').val_float == IsFloat(exactly=-45.375)
    assert Attribute.string('', '.25').val_float == IsFloat(exactly=0.25)
    assert Attribute.string('', '0').val_float == IsFloat(exactly=0.0)

    assert Attribute.string('', '1').val_bool is True
    assert Attribute.string('', '0').val_bool is False
    assert Attribute.string('', 'yEs').val_bool is True
    assert Attribute.string('', 'No').val_bool is False
    assert Attribute.string('', 'tRue').val_bool is True
    assert Attribute.string('', 'faLse').val_bool is False

    assert Attribute.string('', '4.8 290').val_vec2 == Vec2(4.8, 290.0)
    assert Attribute.string('', '4.8 -12.385 384').val_vec3 == ExactType(FrozenVec(4.8, -12.385, 384))
    assert Attribute.string('', '4.8 -12.385 284').val_ang == FrozenAngle(4.8, -12.385, 284)


def test_attr_val_bool() -> None:
    """Test bool value conversions."""
    truth = Attribute.bool('truth', True)
    false = Attribute.bool('false', False)

    assert truth.val_bool is True
    assert truth.val_int == IsInt(exactly=1)
    assert truth.val_float == IsFloat(exactly=1.0)
    assert truth.val_str == '1'
    assert truth.val_bytes == b'\x01'

    assert false.val_bool is False
    assert false.val_int == IsInt(exactly=0)
    assert false.val_float == IsFloat(exactly=0.0)
    assert false.val_str == '0'
    assert false.val_bytes == b'\x00'


def test_attr_array_bool() -> None:
    """Test bool value conversions."""
    elem = Attribute.array('boolean', ValueType.BOOL)
    elem.append(False)
    elem.append(True)

    assert list(elem.iter_bool()) == [False, True]
    assert list(elem.iter_int()) == [0, 1]
    assert list(elem.iter_float()) == [0.0, 1.0]
    assert list(elem.iter_str()) == ['0', '1']
    assert list(elem.iter_bytes()) == [b'\x00', b'\x01']


def test_attr_val_time() -> None:
    """Test time value conversions."""
    elem = Attribute.time('Time', 32.25)
    assert elem.val_int == IsInt(exactly=32)
    assert Attribute.time('Time', -32.25).val_int == IsInt(exactly=-32)
    assert elem.val_str == '32.25'
    assert elem.val_float == IsFloat(exactly=32.25)
    assert elem.val_time == ExactType(Time(32.25))

    assert Attribute.time('Blah', 32.25).val_bool is True
    assert Attribute.time('Blah', 0.0).val_bool is False
    # Negative is false.
    assert Attribute.time('Blah', -12.8).val_bool is False


def test_attr_val_color() -> None:
    """Test color value conversions."""
    elem = Attribute.color('RGB', 240, 128, 64)
    assert elem.val_color == ExactType(Color(240, 128, 64, 255))
    assert Attribute.color('RGB').val_color == ExactType(Color(0, 0, 0, 255))

    assert elem.val_str == '240 128 64 255'
    assert elem.val_vec3 == ExactType(FrozenVec(240.0, 128.0, 64.0))
    assert elem.val_vec4 == ExactType(Vec4(240.0, 128.0, 64.0, 255.0))


def test_attr_val_vector_2() -> None:
    """Test 2d vector conversions."""
    elem = Attribute.vec2('2D', 4.5, 38.2)
    assert elem.val_vec2 == Vec2(4.5, 38.2)
    assert elem.val_str == '4.5 38.2'
    assert Attribute.vec2('No zeros', 5.0, -2.0).val_str == '5 -2'
    assert elem.val_vec3 == ExactType(FrozenVec(4.5, 38.2, 0.0))
    assert elem.val_vec4 == Vec4(4.5, 38.2, 0.0, 0.0)

    assert elem.val_bool is True
    assert Attribute.vec2('True', 3.4, 0.0).val_bool is True
    assert Attribute.vec2('True', 0.0, 3.4).val_bool is True
    assert Attribute.vec2('True', 0.0, 0.0).val_bool is False


def test_attr_array_vector_2() -> None:
    """Test 2d vector array conversions."""
    elem = Attribute.array('2D', ValueType.VEC2)
    elem.append(Vec2(4.5, 38.2))
    elem.append(Vec2(5.0, -2.0))
    assert list(elem.iter_vec2()) == [Vec2(4.5, 38.2), Vec2(5.0, -2.0)]
    assert list(elem.iter_str()) == ['4.5 38.2', '5 -2']
    assert list(elem.iter_vec3()) == [
        ExactType(FrozenVec(4.5, 38.2, 0.0)),
        ExactType(FrozenVec(5.0, -2.0, 0.0)),
    ]
    assert list(elem.iter_vec4()) == [
        ExactType(Vec4(4.5, 38.2, 0.0, 0.0)),
        ExactType(Vec4(5.0, -2.0, 0.0, 0.0)),
    ]

    elem.append(Vec2(3.4, 0.0))
    elem.append(Vec2(0.0, -3.4))
    elem.append(Vec2(0.0, 0.0))
    assert list(elem.iter_bool()) == [True, True, True, True, False]


def test_attr_val_vector_3() -> None:
    """Test 3d vector conversions."""
    elem = Attribute.vec3('3D', 4.5, -12.6, 38.2)
    assert elem.val_vec3 == ExactType(FrozenVec(4.5, -12.6, 38.2))
    assert elem.val_str == '4.5 -12.6 38.2'
    assert Attribute.vec3('No zeros', 5.0, 15.0, -2.0).val_str == '5 15 -2'
    assert elem.val_vec2 == Vec2(4.5, -12.6)
    assert elem.val_vec4 == Vec4(4.5, -12.6, 38.2, 0.0)
    assert Attribute.vec3('RGB', 82, 96.4, 112).val_color == Color(82, 96, 112, 255)
    assert elem.val_ang == ExactType(FrozenAngle(4.5, -12.6, 38.2))

    assert elem.val_bool is True
    assert Attribute.vec3('True', 3.4, 0.0, 0.0).val_bool is True
    assert Attribute.vec3('True', 0.0, 3.4, 0.0).val_bool is True
    assert Attribute.vec3('True', 0.0, 0.0, 3.4).val_bool is True
    assert Attribute.vec3('Fals', 0.0, 0.0, 0.0).val_bool is False


# TODO: Remaining value types.


def test_attr_eq() -> None:
    """Test that attributes can be compared to each other."""
    # We need to check both __eq__ and __ne__.
    assert Attribute.string('test', 'blah') == Attribute.string('test', 'blah')
    assert not Attribute.string('test', 'blah') != Attribute.string('test', 'blah')

    # Names are case-insensitive.
    assert Attribute.float('caseDiff', 3.5) == Attribute.float('CAsediff', 3.5)
    assert not Attribute.float('caseDiff', 3.5) != Attribute.float('CAsediff', 3.5)
    # Types must be the same.
    assert Attribute.string('name', '43') != Attribute.int('name', 43)
    assert not Attribute.string('name', '43') == Attribute.int('name', 43)
    # Array != scalar.
    arr1: Attribute[bool] = Attribute.array('NAM', ValueType.BOOL)
    arr1.append(True)
    assert Attribute.bool('NAM', True) != arr1
    assert not Attribute.bool('NAM', True) == arr1
    arr1.append(False)

    arr2: Attribute[bool] = Attribute.array('NAM', ValueType.BOOL)
    arr2.append(True)
    arr2.append(False)
    assert arr1 == arr2
    assert not arr1 != arr2


def test_attr_append() -> None:
    """Test appending values to an attribute array."""
    arr = Attribute.array('arr', ValueType.STRING)
    assert len(arr) == 0

    arr.append('value')
    assert len(arr) == 1
    assert arr[0].val_str == 'value'

    arr.append(FrozenVec(1, 2, 3))
    assert len(arr) == 2
    assert arr[0].val_str == 'value'
    assert arr[1].val_str == '1 2 3'

    arr.append(45.2)
    assert len(arr) == 3
    assert arr[0].val_str == 'value'
    assert arr[1].val_str == '1 2 3'
    assert arr[2].val_str == '45.2'

    assert list(arr.iter_str()) == ['value', '1 2 3', '45.2']
    arr.clear_array()
    assert len(arr) == 0
    assert list(arr.iter_str()) == []
    assert list(arr.iter_quat()) == []


def test_attr_extend() -> None:
    """Test extending attribute arrays with iterables."""
    arr = Attribute.array('arr', ValueType.FLOAT)
    assert len(arr) == 0
    arr.extend([1, 2.0, 3])
    assert len(arr) == 3
    assert arr[0].val_float == 1.0
    assert arr[1].val_float == 2.0
    assert arr[2].val_float == 3.0

    # Test a pure iterator
    arr.extend(x for x in [1.3, 4.2, '8.9'])
    assert len(arr) == 6
    assert arr[0].val_float == 1.0
    assert arr[1].val_float == 2.0
    assert arr[2].val_float == 3.0
    assert arr[3].val_float == 1.3
    assert arr[4].val_float == 4.2
    assert arr[5].val_float == 8.9
    assert list(arr.iter_str()) == ['1', '2', '3', '1.3', '4.2', '8.9']


@pytest.mark.parametrize('typ, attr, value, binary, text', [
    (ValueType.INT, 'val_int',  0, b'\0\0\0\0', '0'),
    (ValueType.INT, 'val_int',  308823027, bytes.fromhex('f3 43 68 12'), '308823027'),
    (ValueType.INT, 'val_int',  -282637363, bytes.fromhex('cd 4b 27 ef'), '-282637363'),
    (ValueType.BOOL, 'val_bool', False, b'\x00', '0'),
    (ValueType.BOOL, 'val_bool', True, b'\x01', '1'),
    (ValueType.FLOAT, 'val_float', 345.125, b'\x00\x90\xacC', '345.125'),
    (ValueType.FLOAT, 'val_float', -2048.453125, b'@\x07\x00\xc5', '-2048.453125'),
    (ValueType.FLOAT, 'val_float', -2048.0, b'\0\0\0\xc5', '-2048'),
    (ValueType.COLOUR, 'val_color', Color(0, 0, 0, 255), bytes([0, 0, 0, 255]), '0 0 0 255'),
    (ValueType.COLOUR, 'val_color', Color(192, 64, 192, 32), bytes([192, 64, 192, 32]), '192 64 192 32'),
    (ValueType.BINARY, 'val_bin', b'\x34\xFF\x20\x3D', b'\x34\xFF\x20\x3D', '34 FF 20 3D'),
    (ValueType.TIME, 'val_time', Time(60.5), bytes.fromhex('48 3b 09 00'), '60.5'),
    (
        ValueType.VEC2, 'val_vec2',
        Vec2(36.5, -12.75),
        bytes.fromhex('00 00 12 42  00 00 4c c1'),
        '36.5 -12.75',
    ), (
        ValueType.VEC3, 'val_vec3',
        FrozenVec(36.5, 0.125, -12.75),
        bytes.fromhex('00 00 12 42 00 00 00 3e 00 00 4c c1'),
        '36.5 0.125 -12.75',
    ), (
        ValueType.VEC4, 'val_vec4',
        Vec4(384.0, 36.5, 0.125, -12.75),
        bytes.fromhex('00 00 c0 43 00 00 12 42 00 00 00 3e 00 00 4c c1'),
        '384 36.5 0.125 -12.75',
    ), (
        ValueType.QUATERNION, 'val_quat',
        Quaternion(384.0, 36.5, 0.125, -12.75),
        bytes.fromhex('00 00 c0 43 00 00 12 42 00 00 00 3e 00 00 4c c1'),
        '384 36.5 0.125 -12.75',
    ), (
        ValueType.MATRIX, 'val_mat', Matrix(),
        b'1000' b'0100' b'0010' b'0001'.replace(b'0', b'\x00\x00\x00\x00').replace(b'1', b'\x00\x00\x80?'),
        '1.0 0.0 0.0 0.0\n'
        '0.0 1.0 0.0 0.0\n'
        '0.0 0.0 1.0 0.0\n'
        '0.0 0.0 0.0 1.0',
    ),
])
def test_binary_text_conversion(typ: ValueType, attr: str, value, binary: bytes, text: str) -> None:
    """Test the required binary and text conversions."""
    assert TYPE_CONVERT[typ, ValueType.BINARY](value) == binary
    assert TYPE_CONVERT[ValueType.BINARY, typ](binary) == value
    assert Attribute('', typ, value).val_bytes == binary
    assert getattr(Attribute.binary('', binary), attr) == value

    assert TYPE_CONVERT[typ, ValueType.STRING](value) == text
    assert TYPE_CONVERT[ValueType.STRING, typ](text) == value
    assert Attribute('', typ, value).val_str == text
    assert getattr(Attribute.string('', text), attr) == value


@pytest.mark.parametrize('typ, iterable, expected', [
    (ValueType.INTEGER, [1, 2, 8.0], [1, 2, 8]),
    (ValueType.FLOAT, [2.4, 9.0, 23], [2.4, 9.0, 23.0]),
    (ValueType.BOOL, [False, True, 1.0], [False, True, True]),
    (ValueType.STR, ['', 'a', 'spam'], ['', 'a', 'spam']),
    (ValueType.BIN,
     [b'abcde', bytearray(b'fghjik'), array.array('B', [0x68, 0x65, 0x6c, 0x6c, 0x6f])],
     [b'abcde', b'fghjik', b'hello'],
     ),
    (ValueType.COLOR,
     [Color(255, 0, 255, 128), (25, 38, 123), (127, 180, 255, 192)],
     [Color(255, 0, 255, 128), Color(25, 38, 123, 255), Color(127, 180, 255, 192)],
     ),
    (ValueType.TIME, [Time(0.0), 360.5], [Time(0.0), Time(360.5)]),
    (ValueType.VEC2,
     [Vec2(1.0, 2.0), (2.0, 3.0), [5, 8.0]],
     [Vec2(1.0, 2.0), Vec2(2.0, 3.0), Vec2(5.0, 8.0)],
     ),
    (ValueType.VEC3,
     [FrozenVec(2.0, 3.0, 4.0), Vec(4.5, 6.8, 9.2), (1.0, 2, 3.0), range(3)],
     [FrozenVec(2.0, 3.0, 4.0), FrozenVec(4.5, 6.8, 9.2), FrozenVec(1.0, 2.0, 3.0), FrozenVec(0.0, 1.0, 2.0)]
     ),
    (ValueType.VEC4,
     [Vec4(5.0, 2.8, 9, 12), (4, 5, 28, 12.0), range(4, 8)],
     [Vec4(5.0, 2.8, 9.0, 12.0), Vec4(4.0, 5.0, 28.0, 12.0), Vec4(4.0, 5.0, 6.0, 7.0)],
     ),
    (ValueType.ANGLE,
     [Angle(45.0, 20.0, -22.5), Matrix.from_pitch(45.0), Matrix.from_yaw(15.0), (90.0, 45.0, 0.0)],
     [FrozenAngle(45.0, 20.0, 337.5), FrozenAngle(45.0, 0.0, 0.0),
      FrozenAngle(0.0, 15.0, 0.0), FrozenAngle(90.0, 45.0, 0.0)],
     ),
    (ValueType.QUATERNION,
     [Quaternion(1.0, 2.0, 3.0, 4.0), (8.0, -3.0, 0.5, 12.8)],
     [Quaternion(1.0, 2.0, 3.0, 4.0), Quaternion(8.0, -3.0, 0.5, 12.8)],
     ),
    (ValueType.MATRIX,
    [Angle(45.0, 20.0, -22.5), Matrix.from_pitch(45.0), FrozenMatrix.from_yaw(45.0),
     FrozenAngle(90.0, 45.0, 0.0)],
    [FrozenMatrix.from_angle(45.0, 20.0, -22.5), FrozenMatrix.from_pitch(45.0),
     FrozenMatrix.from_yaw(45.0), FrozenMatrix.from_angle(90.0, 45.0, 0.0)],
     ),
], ids=[
    'int', 'float', 'bool', 'str', 'bytes', 'time', 'color',
    'vec2', 'vec3', 'vec4', 'angle', 'quaternion', 'matrix',
])
def test_attr_array_constructor(typ: ValueType, iterable: list, expected: list) -> None:
    """Test constructing an array attribute."""
    expected_type = type(expected[0])
    attr = Attribute.array('some_array', typ, iterable)
    assert attr.is_array
    assert len(attr) == len(expected)
    for i, (actual, expect) in enumerate(zip(attr._value, expected)):
        assert type(actual) is expected_type, repr(actual)
        if expected_type is FrozenMatrix:  # Matrix comparisons are difficult.
            actual = actual.to_angle()
            expect = expect.to_angle()
        assert actual == expect, f'attr[{i}]: {actual!r} != {expect!r}'


def test_attr_array_elem_constructor() -> None:
    """Test constructing an Element array attribute."""
    elem1 = Element('Elem1', 'DMElement')
    elem2 = Element('Elem2', 'DMElement')
    attr = Attribute.array('elements', ValueType.ELEMENT, [elem1, elem2, elem1])
    assert attr.is_array
    assert len(attr) == 3
    assert attr[0].val_elem is elem1
    assert attr[1].val_elem is elem2
    assert attr[2].val_elem is elem1


deduce_type_tests = [
    (5, ValueType.INT, 5),
    (5.0, ValueType.FLOAT, 5.0),
    (False, ValueType.BOOL, False),
    (True, ValueType.BOOL, True),
    ('test', ValueType.STR, 'test'),
    (b'test', ValueType.BIN, b'test'),

    (Color(25, 48, 255, 255), ValueType.COLOR, Color(25, 48, 255, 255)),
    (Color(25.0, 48.2, 255.4, 255.9), ValueType.COLOR, Color(25, 48, 255, 255)),  # type: ignore

    (Vec2(1, 4.0), ValueType.VEC2, Vec2(1.0, 4.0)),
    (Vec(1.8, 4.0, 8), ValueType.VEC3, FrozenVec(1.8, 4.0, 8.0)),
    (FrozenVec(1.8, 4.0, 8), ValueType.VEC3, FrozenVec(1.8, 4.0, 8.0)),
    (Vec4(4, 8, 23, 29.8), ValueType.VEC4, Vec4(4.0, 8.0, 23.0, 29.8)),

    (FrozenAngle(45.0, 92.6, 23.0), ValueType.ANGLE, FrozenAngle(45.0, 92.6, 23.0)),
    (Angle(45.0, 92.6, 23.0), ValueType.ANGLE, FrozenAngle(45.0, 92.6, 23.0)),
    (Quaternion(4, 8, 23, 29.8), ValueType.QUATERNION, Quaternion(4.0, 8.0, 23.0, 29.8)),
]


@pytest.mark.parametrize('input, val_type, output', deduce_type_tests)
def test_deduce_type_basic(input, val_type, output) -> None:
    """Test type deduction behaves correctly."""
    [test_type, test_val] = deduce_type(input)
    assert test_type is val_type
    assert type(test_val) is type(output)
    assert test_val == output


@pytest.mark.parametrize('input, val_type, output', [
    # Add the above tests here too.
    ([inp, inp, inp], val_typ, [out, out, out])
    for inp, val_typ, out in
    deduce_type_tests
] + [
    # Numbers can be mixed.
    ([1.0, 4], ValueType.FLOAT, [1.0, 4.0]),
    ([1, 4.0, False, True], ValueType.FLOAT, [1.0, 4.0, 0.0, 1.0]),
    ([1, 4, False, True], ValueType.INT, [1, 4, 0, 1]),
    ([False, True, True, False], ValueType.BOOL, [False, True, True, False]),
    # Frozen/mutable classes can be mixed.
    (
        [Angle(3.0, 4.0, 5.0), FrozenAngle(4.0, 3.0, 6.0)],
        ValueType.ANGLE,
        [FrozenAngle(3.0, 4.0, 5.0), FrozenAngle(4.0, 3.0, 6.0)]
    ),
    (
        [Vec(3.0, 4.0, 5.0), FrozenVec(4.0, 3.0, 6.0)],
        ValueType.VEC3,
        [FrozenVec(3.0, 4.0, 5.0), FrozenVec(4.0, 3.0, 6.0)]
    ),

    # Tuples and other sequences work
    ((Vec2(4, 5), Vec2(6.0, 7.0)), ValueType.VEC2, [Vec2(4.0, 5.0), Vec2(6.0, 7.0)]),
    (collections.deque([1.0, 2.0, 3.0]), ValueType.FLOAT, [1.0, 2.0, 3.0]),
    (range(5), ValueType.INT, [0, 1, 2, 3, 4]),
])
def test_deduce_type_array(input, val_type, output) -> None:
    """Test array deduction, and some special cases."""
    [test_type, test_arr] = deduce_type(input)
    assert test_type is val_type
    assert len(input) == len(test_arr), repr(test_arr)
    for i, (test, out) in enumerate(zip(test_arr, output)):
        assert type(test) is type(out), f'{i}: {test!r} != {out!r}'
        assert test == out, f'{i}: {test!r} != {out!r}'


def test_deduce_type_adv() -> None:
    """Test specific overrides and behaviour."""
    # Print means any successful result ends up in the console.
    with pytest.raises(TypeError):  # Other types.
        print(deduce_type(...))
    with pytest.raises(TypeError):
        print(deduce_type([...]))

    # Empty result.
    with pytest.raises(TypeError):
        print(deduce_type([]))
    with pytest.raises(TypeError):
        print(deduce_type(()))
    with pytest.raises(TypeError):
        print(deduce_type(range(0)))


def test_special_attr_name() -> None:
    """Test the special behaviour of the "name" attribute.

    This just maps to a 'name' key.
    """
    elem = Element('MyName', 'DMElement')
    assert elem.name == 'MyName'
    assert 'name' in elem
    assert elem['name'].type is ValueType.STRING
    assert elem['name'].val_string == 'MyName'
    elem['name'].val_color = Color(192, 128, 64)
    assert elem.name == '192 128 64 255'
    elem.name = 'Third'
    assert elem['name'].val_string == 'Third'


def test_special_attr_id() -> None:
    """Test the special behaviour of the "id" attribute.

    This attribute can co-exist with a key of the same name.
    """
    uuid_a = uuid4()
    uuid_b = uuid4()
    elem = Element('MyName', 'DMElement', uuid_a)
    assert elem.uuid == uuid_a
    assert 'id' not in elem
    elem['id'] = uuid_b.bytes
    assert elem.uuid == uuid_a
    assert elem['id'].val_bytes == uuid_b.bytes


# TODO: We need to find a sample of legacy binary, v1 and v3 to verify implementation
@pytest.mark.parametrize('filename', [
    'keyvalues2',
    'binary_v2',  # HL2's dmxconvert
    'binary_v4',  # L4D2's dmxconvert
    'binary_v5',  # P2+'s dmxconvert
])
def test_parse(datadir: Path, filename: str) -> None:
    """Test parsing all the format types."""
    with (datadir / f'{filename}.dmx').open('rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)
    assert fmt_name == 'dmx'

    verify_sample(root)


def verify_sample(root: Element) -> None:
    """Verify this DMX matches the sample files."""
    # string <-> double <-> float conversions are not quite accurate.
    a = cast(Callable[[float], float], pytest.approx)

    assert root.name == 'Root_Name'
    assert root.type == 'DmeRootElement'
    assert root.uuid == UUID('b66a2ce3-d686-4dbf-85df-07c6b275bebb')
    assert len(root) == 4, root
    assert sorted(root.keys()) == ['arrays', 'name', 'recursive', 'scalars']

    # First, the recursive element tree.
    recur_attr = root['recursive']
    assert recur_attr.name == 'recurSive'
    recur = recur_attr.val_elem
    assert recur.name == 'Recurse1'
    assert recur.type == 'RecurseElement'
    assert len(recur) == 3, recur
    assert recur['leaf1'].name == 'Leaf1'
    assert recur['leaf2'].name == 'Leaf2'

    leaf_1 = recur['leaf1'].val_elem
    assert leaf_1.name == 'FirstLeaf'
    assert leaf_1.type == 'RecurseElement'
    assert leaf_1.uuid == UUID('1bed12dc-53a7-4a15-864b-51572c2ab0b1')

    leaf_2 = recur['leaf2'].val_elem
    assert leaf_2.name == 'SecondLeaf'
    assert leaf_2.type == 'RecurseElement'
    assert leaf_2.uuid == UUID('437fa198-1782-4737-ae86-175e8f76dc4f')

    # This must loop!
    assert leaf_2['recurse'].val_elem is recur

    # Then scalars.
    scalars = root['scalars'].val_elem
    assert scalars.type == 'TypeHolder'
    assert scalars.uuid == UUID('e1db21c6-d046-46ca-9452-25c667b70ede')
    assert scalars.name == 'ScalarValues'

    assert len(scalars) == 22
    assert scalars['neg_integer'].val_int == -1230552801
    assert scalars['pos_integer'].val_int == 296703200

    assert scalars['neg_float'].val_float == a(-16211.59325)
    assert scalars['pos_float'].val_float == a(22097.83875)

    # The UUID has a unique type, and can coexist with a key of the same name.
    assert scalars['id'].val_bytes == (
        b'\x5c\x81\x48\xee\x76\x78\x46\x1b'
        b'\xb5\xc5\xf3\xd0\xe1\x42\x7c\x01'
    )

    assert scalars['truth'].val_bool is True
    assert scalars['falsity'].val_bool is False

    assert scalars['string'].val_string == "string \n \t \v \b \r \f \a \\ ? \' \""

    assert scalars['red'].val_color == Color(240, 32, 32, 255)
    assert scalars['blue'].val_color == Color(32, 240, 32, 255)
    assert scalars['green'].val_color == Color(32, 32, 240, 255)
    assert scalars['white'].val_color == Color(255, 255, 255, 255)
    assert scalars['half'].val_color == Color(0, 0, 0, 128)

    assert scalars['vec2'].val_vec2 == Vec2(a(348.275), a(-389.935))
    assert scalars['vec3'].val_vec3 == FrozenVec(128.25, -1048.5, 182.125)
    assert scalars['vec4'].val_vec4 == Vec4(128.25, -1048.5, 182.125, -81.5)

    assert scalars['up'].val_ang == FrozenAngle(270, 0, 0)
    assert scalars['dn'].val_ang == FrozenAngle(90, 0, 0)
    assert scalars['somedir'].val_ang == FrozenAngle(291, 311.125, 45.0)
    assert scalars['quat'].val_quat == Quaternion(a(0.267261), a(0.534522), a(0.801784), 0.0)

    assert scalars['hex'].val_bin == (
        b'\x92\x26\xd0\xc7\x12\xec\x39\xe9\xd1\x63\x45\x19\xd1\xbd\x0f\x4a\x76'
        b'\x1c\x6f\x81\xaf\xb5\x78\x5b\x4b\x9c\x85\x29\x12\x74\xff\x26\xcd\x37'
        b'\x0e\xb1\x73\x18\xfa\x32\x6d\x22\xef\xf4\xd8\xb8\xf9\xd4\x1e\x6b\xee'
    )

    # And finally arrays.
    arrays = root['arrays'].val_elem
    assert arrays.type == 'TypeHolder'
    assert arrays.uuid == UUID('2b95889f-5041-436e-9350-813abcf504b0')
    assert arrays.name == 'ArrayValues'
    assert len(arrays) == 13

    arr_string = arrays['strings']
    assert len(arr_string) == 3
    assert arr_string[0].val_str == 'regular Value'
    assert arr_string[1].val_str == '\n \t \v \b \r \f \a'
    assert arr_string[2].val_str == '\\ ? \' "'

    arr_int = arrays['integers']
    assert len(arr_int) == 5
    assert arr_int[0].val_int == 1
    assert arr_int[1].val_int == 2
    assert arr_int[2].val_int == 35
    assert arr_int[3].val_int == -39
    assert arr_int[4].val_int == 0

    arr_float = arrays['floating']
    assert len(arr_float) == 10
    assert arr_float[0].val_float == a(-10291.15325)
    assert arr_float[1].val_float == a(-55646.21366)
    assert arr_float[2].val_float == a(78545.15227)
    assert arr_float[3].val_float == a(-95302.8789)
    assert arr_float[4].val_float == a(-45690.04457)
    assert arr_float[5].val_float == a(-55299.05236)
    assert arr_float[6].val_float == a(96178.44015)
    assert arr_float[7].val_float == a(58708.2978495)
    assert arr_float[8].val_float == a(-49957.20355)
    assert arr_float[9].val_float == a(23980.82395)

    arr_bool = arrays['logical']
    assert len(arr_bool) == 8
    assert list(arr_bool.iter_bool()) == [
        False, False, True, False,
        False, True, True, True,
    ]
    assert list(arrays['colors'].iter_colour()) == [
        Color(255, 0, 0, 255),
        Color(0, 0, 255, 128),
        Color(0, 240, 0, 0),
    ]
    assert list(arrays['2ds'].iter_vec2()) == [
        Vec2(x, y)
        for x in [-1, 0, 1]
        for y in [-1, 0, 1]
    ]
    assert list(arrays['3ds'].iter_vec3()) == [
        FrozenVec(34.0, -348.25, 128.125),
        FrozenVec(0.0, 0.0, 0.0),
        FrozenVec(0.9, 0.8, 0.5),
    ]
    assert list(arrays['4dimensional'].iter_vec4()) == [
        Vec4(0.0, 0.0, 0.0, 0.0),
        Vec4(0.0, 0.0, 0.0, 1.0),
        Vec4(0.0, 2.0, 0, 48.0),
        Vec4(-28.0, 380.0, -39.0, 39.0),
    ]
    assert list(arrays['directions'].iter_angle()) == [
        FrozenAngle(0, 0, 0), FrozenAngle(45, 0, 90),
        FrozenAngle(0, 135, 180), FrozenAngle(45, 31, 321),
        FrozenAngle(94, 165, 23),
    ]
    assert len(arrays['quaternions']) == 2
    assert arrays['quaternions'][0].val_quat == Quaternion(a(0.267261), a(-0.801784), a(0.534522), 0.0)
    assert arrays['quaternions'][1].val_quat == Quaternion(a(-0.534522), a(0.267261), 0.0, a(-0.801784))

    assert len(arrays['hexes']) == 3
    assert list(arrays['hexes'].iter_bin()) == [
        bytes([0, 1, 2, 3, 4, 5]),
        bytes([0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F]),
        b'\xFF\xEE\xDD\xCC\xBB\xAA',
    ]

    arr_elem = arrays['elements']
    assert len(arr_elem) == 4
    arr_elem_1 = arr_elem[0].val
    assert arr_elem_1.name == 'A'
    assert arr_elem_1.type == 'FirstElem'
    assert arr_elem_1.uuid == UUID('96d00e24-4c16-41ec-ab2f-e2a24f3a6ad2')
    assert arr_elem_1['a'].val_int == 1
    assert arr_elem_1['b'].val_int == 2

    arr_elem_2 = arr_elem[1].val
    assert arr_elem_2.name == 'B'
    assert arr_elem_2.type == 'SecondElem'
    assert arr_elem_2.uuid == UUID('ef7272f0-7c48-4d2f-aefd-036e30a2da15')
    assert arr_elem_2['a'].val_float == 3.0
    assert arr_elem_2['b'].val_float == 4.0

    assert arr_elem[2].val is arr_elem_2
    assert arr_elem[3].val is arr_elem_1


def test_parse_binaryv3(datadir: Path) -> None:
    """Test parsing of binary version 3.

    We don't have a DMXconvert that produces this, so instead compare an existing
    file to the text version.
    """
    with (datadir / 'tf_movies.dmx').open('rb') as f:
        root_bin, fmt_name, fmt_version = Element.parse(f)
    with (datadir / 'tf_movies_text.dmx').open('rb') as f:
        root_txt, fmt_name, fmt_version = Element.parse(f)
    assert_tree(root_bin, root_txt)


def test_parse_long_header(datadir: Path) -> None:
    """Test parsing a DMX with a long header, requiring additional reads to complete."""
    with (datadir / 'kv2_long_header.dmx').open('rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)
    assert len(fmt_name) == 205
    assert fmt_version == 123456789
    assert root.uuid == UUID('09ab03ae-93a2-455e-8a71-f6cc12374fa7')
    assert root.name == "LongHeaders"
    assert root['value'].val_int == 42


@pytest.mark.parametrize('version', [2, 4, 5])
def test_export_bin_roundtrip(datadir: Path, version: int) -> None:
    """Test exporting binary types roundtrip."""
    with (datadir / 'binary_v5.dmx').open('rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)

    buf = BytesIO()
    root.export_binary(buf, version, fmt_name, fmt_version)
    buf.seek(0)
    rnd_root, rnd_name, rnd_ver = Element.parse(buf)

    assert rnd_name == fmt_name
    assert rnd_ver == fmt_version
    # Check when parsed it matches the assertions above.
    verify_sample(rnd_root)


@pytest.mark.parametrize('flat', [False, True], ids=['indented', 'flat'])
def test_export_kv2_roundtrip(datadir: Path, flat: bool) -> None:
    """Test exporting keyvalues2 roundtrip."""
    with (datadir / 'keyvalues2.dmx').open('rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)

    buf = BytesIO()
    root.export_kv2(buf, fmt_name, fmt_version, flat=flat)
    buf.seek(0)

    rnd_root, rnd_name, rnd_ver = Element.parse(buf)
    assert rnd_name == fmt_name
    assert rnd_ver == fmt_version
    verify_sample(rnd_root)


@pytest.mark.parametrize('version', EXPORT_VALS)
def test_ext_roundtrip_unicode(version: str) -> None:
    """Test the 'silent' extension doesn't affect ASCII only files,

    but allows roundtrip of unicode.
    """
    ascii_text = ''.join(map(chr, range(128)))
    root = Element('name', 'DMERoot')
    root['key'] = attr = Attribute.string('key', ascii_text)

    orig = export(root, version, 'ascii')
    silent = export(root, version, 'silent')

    assert orig == silent

    attr.val_str = unicode_text = 'Ascii ╒══╕ text and some 🤐♝♛🥌♚♝'
    silent = export(root, version, 'silent')
    explicit = export(root, version, 'format')

    # No flags, fails. UnicodeError from binary, TokenSyntaxError from text.
    exc: Tuple[Type[Exception], ...] = (UnicodeError, TokenSyntaxError)
    with pytest.raises(exc):
        Element.parse(BytesIO(silent))

    # Format flag detected.
    no_flag = Element.parse(BytesIO(explicit))

    # When informed, it can detect it.
    flagged_silent = Element.parse(BytesIO(silent), unicode=True)

    # Doesn't matter either way.
    flagged_explicit = Element.parse(BytesIO(explicit), unicode=True)

    for name, (elem, fmt_name, fmt_ver) in [
        ('unicode=False, unicode_', no_flag),
        ('unicode=True, binary', flagged_silent),
        ('unicode=True, unicode_binary', flagged_explicit),
    ]:
        new_attr = elem['key']
        assert new_attr.val_str == unicode_text, f'{name}: {new_attr.val_str!r} != {unicode_text!r}'


@pytest.mark.parametrize('version', EXPORT_VALS)
def test_export_regression(version: str, datadir: Path, file_regression) -> None:
    """Test regressions in the export results."""
    with (datadir / 'binary_v5.dmx').open('rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)
    file_regression.check(export(root, version), extension='.dmx', binary=True)


def test_kv1_to_dmx() -> None:
    """Test converting KV1 property trees into DMX works."""
    tree1 = Keyvalues('rOOt', [
        Keyvalues('child1', [Keyvalues('key', 'value')]),
        Keyvalues('child2', [
            Keyvalues('key', 'value'),
            Keyvalues('key2', '45'),
        ]),
    ])
    elem1 = Element.from_kv1(tree1)
    assert elem1.type == 'DmElement'
    assert elem1.name == 'rOOt'

    subkey: Attribute[Element] = elem1['subkeys']
    assert subkey.type is ValueType.ELEMENT
    assert subkey.is_array
    [child1, child2] = subkey.iter_elem()
    assert child1.type == 'DmElement'
    assert child1.name == 'child1'
    assert child1['key'].type is ValueType.STRING
    assert child1['key'].val_str == 'value'

    assert child2['key'].type is ValueType.STRING
    assert child2['key'].val_str == 'value'
    assert child2['key2'].type is ValueType.STRING
    assert child2['key2'].val_str == '45'


def test_kv1_to_dmx_dupleafs() -> None:
    """Test converting KV1 trees with duplicate keys."""
    tree = Keyvalues('Root', [
        Keyvalues('Key1', 'blah'),
        Keyvalues('key2', 'another'),
        Keyvalues('key1', 'value'),
    ])
    root = Element.from_kv1(tree)
    assert root.type == 'DmElement'
    assert root.name == 'Root'
    subkeys: Attribute[Element] = root['subkeys']
    assert subkeys.type is ValueType.ELEMENT
    assert subkeys.is_array
    [k1, k2, k3] = subkeys.iter_elem()
    assert k1.name == 'Key1'
    assert k1.type == 'DmElementLeaf'
    assert k1['value'].type is ValueType.STRING
    assert k1['value'].val_str == 'blah'

    assert k2.name == 'key2'
    assert k2.type == 'DmElementLeaf'
    assert k2['value'].type is ValueType.STRING
    assert k2['value'].val_str == 'another'

    assert k3.name == 'key1'
    assert k3.type == 'DmElementLeaf'
    assert k3['value'].type is ValueType.STRING
    assert k3['value'].val_str == 'value'


def test_kv1_to_dmx_leaf_and_blocks() -> None:
    """If both leafs and blocks, we upgrade to an element per attribute."""
    tree = Keyvalues('blah', [
        Keyvalues('a_leaf', 'result'),
        Keyvalues('block', []),
    ])
    root = Element.from_kv1(tree)
    assert root.type == 'DmElement'
    assert root.name == 'blah'
    subkeys: Attribute[Element] = root['subkeys']
    assert subkeys.type is ValueType.ELEMENT
    assert subkeys.is_array
    [e1, e2] = subkeys.iter_elem()
    assert e1.type == 'DmElementLeaf'
    assert e1['value'].val_str == 'result'
    assert e2.type == 'DmElement'
    assert len(e2) == 1
    assert list(e2.keys()) == ['name']


def test_dmx_to_kv1_roundtrip() -> None:
    """Test we can smuggle KV1 trees in DMX elements."""
    from test_keyvalues import assert_tree as assert_prop, parse_result
    elem = Element.from_kv1(parse_result)
    roundtrip = elem.to_kv1()
    assert_prop(roundtrip, parse_result)
