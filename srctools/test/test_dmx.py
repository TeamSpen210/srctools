"""Test the datamodel exchange implementation."""
from typing import Callable, cast, Set
from uuid import UUID

import pytest

from srctools import Matrix, Angle
from srctools.dmx import (
    Element, Attribute, ValueType, Vec2, Vec3, Vec4, AngleTup, Color,
    Quaternion, deduce_type, TYPE_CONVERT, Time,
)


def assert_tree(tree1: Element, tree2: Element) -> None:
    """Checks both trees are identical, recursively."""
    return _assert_tree_elem(tree1.name, tree1, tree2, set())


def _assert_tree_elem(path: str, tree1: Element, tree2: Element, checked: Set[UUID]) -> None:
    """Checks two elements are the same."""
    if tree1 is None:
        if tree2 is None:
            return  # Matches.
        pytest.fail(f'{path}: NULL != {tree2}')
    elif tree2 is None:
        pytest.fail(f'{path}: {tree1} != NULL')

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

        # Allow one to be NULL, but the other to be a missing element.
        try:
            attr1 = tree1[key]
        except KeyError:
            attr1 = None
        try:
            attr2 = tree2[key]
        except KeyError:
            attr2 = None
        if attr1 is None:
            if attr2 is None:
                raise AssertionError(f'{key} not in {tree1.keys()} and {tree2.keys()}, but in union?')
            elif attr2.type is ValueType.ELEMENT or attr2.val_elem is None:
                continue
            raise AssertionError(f'{attr_path}: NULL != {attr2.type}')
        elif attr2 is None:
            if attr1.type is ValueType.ELEMENT and attr1.val_elem is None:
                continue
            raise AssertionError(f'{attr_path}: {attr1.type} != NULL')

        assert attr1.type is attr2.type, attr_path
        assert attr1.name == attr2.name, attr_path
        if attr1.type is ValueType.ELEMENT:
            if attr1.is_array:
                assert len(attr1) == len(attr2), f'Mismatched len for {attr_path}'
                for i, elem1 in enumerate(attr1):
                    _assert_tree_elem(f'{attr_path}[{i}]', elem1, attr2[i].val_elem, checked)
            else:
                _assert_tree_elem(attr_path, attr1.val_elem, attr2.val_elem, checked)
        elif attr1.is_array:
            assert len(attr1) == len(attr2), f'Mismatched len for {attr_path}'
            for i in range(len(attr1)):
                assert attr1[i].val_str == attr2[i].val_str, f'{attr_path}[{i}]: {attr1._value[i]} != {attr2._value[i]}'
        else:
            assert attr1.val_str == attr2.val_str, f'{attr_path}: {attr1._value} != {attr2._value}'


def test_attr_val_int() -> None:
    """Test integer-type values."""
    elem = Attribute.int('Name', 45)
    assert elem.val_int == 45
    assert elem.val_str == '45'
    assert elem.val_float == 45.0
    assert elem.val_time == 45.0

    assert elem.val_vec2 == Vec2(45.0, 45.0)
    assert elem.val_vec3 == Vec3(45.0, 45.0, 45.0)
    assert elem.val_vec4 == Vec4(45.0, 45.0, 45.0, 45.0)
    assert elem.val_color == Color(45, 45, 45, 255)

    assert Attribute.int('Blah', 45).val_bool is True
    assert Attribute.int('Blah', 0).val_bool is False
    assert Attribute.int('Blah', -2).val_bool is True


def test_attr_val_float() -> None:
    """Test float-type values."""
    elem = Attribute.float('Name', 32.25)
    assert elem.val_int == 32
    assert Attribute.float('Name', -32.25).val_int == -32
    assert elem.val_str == '32.25'
    assert elem.val_float == 32.25
    assert elem.val_time == 32.25

    assert elem.val_vec2 == Vec2(32.25, 32.25)
    assert elem.val_vec3 == Vec3(32.25, 32.25, 32.25)
    assert elem.val_vec4 == Vec4(32.25, 32.25, 32.25, 32.25)

    assert Attribute.float('Blah', 32.25).val_bool is True
    assert Attribute.float('Blah', 0.0).val_bool is False
    assert Attribute.float('Blah', -12.8).val_bool is True


def test_attr_val_str() -> None:
    """Test string-type values."""
    assert Attribute.string('', '45').val_str == '45'
    assert Attribute.string('', '').val_str == ''
    assert Attribute.string('', 'testing str\ning').val_str == 'testing str\ning'

    assert Attribute.string('Name', '45').val_int == 45
    assert Attribute.string('Name', '-45').val_int == -45
    assert Attribute.string('Name', '0').val_int == 0

    assert Attribute.string('', '45').val_float == 45.0
    assert Attribute.string('', '45.0').val_float == 45.0
    assert Attribute.string('', '45.375').val_float == 45.375
    assert Attribute.string('', '-45.375').val_float == -45.375
    assert Attribute.string('', '.25').val_float == 0.25
    assert Attribute.string('', '0').val_float == 0.0

    assert Attribute.string('', '1').val_bool is True
    assert Attribute.string('', '0').val_bool is False
    assert Attribute.string('', 'yEs').val_bool is True
    assert Attribute.string('', 'No').val_bool is False
    assert Attribute.string('', 'tRue').val_bool is True
    assert Attribute.string('', 'faLse').val_bool is False

    assert Attribute.string('', '4.8 290').val_vec2 == Vec2(4.8, 290.0)
    assert Attribute.string('', '4.8 -12.385 384').val_vec3 == Vec3(4.8, -12.385, 384)
    assert Attribute.string('', '4.8 -12.385 284').val_ang == AngleTup(4.8, -12.385, 284)


def test_attr_val_bool() -> None:
    """Test bool value conversions."""
    truth = Attribute.bool('truth', True)
    false = Attribute.bool('false', False)

    assert truth.val_bool is True
    assert truth.val_int == 1
    assert truth.val_float == 1.0
    assert truth.val_str == '1'
    assert truth.val_bytes == b'\x01'

    assert false.val_bool is False
    assert false.val_int == 0
    assert false.val_float == 0.0
    assert false.val_str == '0'
    assert false.val_bytes == b'\x00'


def test_attr_val_time() -> None:
    """Test time value conversions."""
    elem = Attribute.float('Time', 32.25)
    assert elem.val_int == 32
    assert Attribute.float('Time', -32.25).val_int == -32
    assert elem.val_str == '32.25'
    assert elem.val_float == 32.25
    assert elem.val_time == 32.25

    assert Attribute.time('Blah', 32.25).val_bool is True
    assert Attribute.time('Blah', 0.0).val_bool is False
    # Negative is false.
    assert Attribute.time('Blah', -12.8).val_bool is False


def test_attr_val_color() -> None:
    """Test color value conversions."""
    elem = Attribute.color('RGB', 240, 128, 64)
    assert elem.val_color == Color(240, 128, 64, 255)
    assert Attribute.color('RGB').val_color == Color(0, 0, 0, 255)

    assert elem.val_str == '240 128 64 255'
    assert elem.val_vec3 == Vec3(240.0, 128.0, 64.0)
    assert elem.val_vec4 == Vec4(240.0, 128.0, 64.0, 255.0)


def test_attr_val_vector_2() -> None:
    """Test 2d vector conversions."""
    elem = Attribute.vec2('2D', 4.5, 38.2)
    assert elem.val_vec2 == Vec2(4.5, 38.2)
    assert elem.val_str == '4.5 38.2'
    assert Attribute.vec2('No zeros', 5.0, -2.0).val_str == '5 -2'
    assert elem.val_vec3 == Vec3(4.5, 38.2, 0.0)
    assert elem.val_vec4 == Vec4(4.5, 38.2, 0.0, 0.0)

    assert elem.val_bool is True
    assert Attribute.vec2('True', 3.4, 0.0).val_bool is True
    assert Attribute.vec2('True', 0.0, 3.4).val_bool is True
    assert Attribute.vec2('True', 0.0, 0.0).val_bool is False


def test_attr_val_vector_3() -> None:
    """Test 3d vector conversions."""
    elem = Attribute.vec3('3D', 4.5, -12.6, 38.2)
    assert elem.val_vec3 == Vec3(4.5, -12.6, 38.2)
    assert elem.val_str == '4.5 -12.6 38.2'
    assert Attribute.vec3('No zeros', 5.0, 15.0, -2.0).val_str == '5 15 -2'
    assert elem.val_vec2 == Vec2(4.5, -12.6)
    assert elem.val_vec4 == Vec4(4.5, -12.6, 38.2, 0.0)
    assert Attribute.vec3('RGB', 82, 96.4, 112).val_color == Color(82, 96, 112, 255)
    assert elem.val_ang == AngleTup(4.5, -12.6, 38.2)

    assert elem.val_bool is True
    assert Attribute.vec3('True', 3.4, 0.0, 0.0).val_bool is True
    assert Attribute.vec3('True', 0.0, 3.4, 0.0).val_bool is True
    assert Attribute.vec3('True', 0.0, 0.0, 3.4).val_bool is True
    assert Attribute.vec3('Fals', 0.0, 0.0, 0.0).val_bool is False


@pytest.mark.parametrize(['attr', 'typ'], [
    pytest.param(attr, typ, id=attr)
    for attr, typ in [
        ('val_str', str),
        ('val_string', str),
        ('val_bin', bytes),
        ('val_binary', bytes),
        ('val_bytes', bytes),
        ('val_time', float),  # Time isn't actually a type.
        ('val_int', int),
        ('val_bool', bool),
        ('val_float', float),
        ('val_ang', AngleTup),
        ('val_angle', AngleTup),
        ('val_color', Color),
        ('val_vec2', Vec2),
        ('val_vec3', Vec3),
        ('val_vec4', Vec4),
        ('val_quat', Quaternion),
        ('val_quaternion', Quaternion),
        ('val_mat', Matrix),
        ('val_matrix', Matrix),
]])
@pytest.mark.parametrize('attribute', [
    Attribute.int('', 45),
    Attribute.float('', 48.9),
    Attribute.time('', Time(60.5)),
    Attribute.vec2('', 3, 4),
    Attribute.vec3('', 4, 5, 6),
    Attribute.vec4('', 5, 6, 7),
    Attribute.angle('', 45.0, 90.0, 0.0),
    Attribute.color('', 255, 128, 64),
    Attribute.quaternion('', 0.0, 0.0, 0.0, 1.0),
], ids=lambda attr: attr.type.name.lower())
def test_attr_conv_types(attribute: Attribute, attr: str, typ: type) -> None:
    """Check all the conversions either fail or produce the right type.

    We don't test string/bytes since valid values are different for different dests.
    """
    try:
        result = getattr(attribute, attr)
    except ValueError:
        # Conversion failed, that's fine for all types except for string/binary.
        if typ is str or typ is bytes:
            return pytest.fail('This conversion is required.')
        else:
            return pytest.xfail('Conversion not defined.')

    assert type(result) is typ, result


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
    (ValueType.TIME, 'val_time', 60.5, b'\0\0rB', '60.5'),
    (ValueType.VEC2, 'val_vec2', Vec2(36.5, -12.75), bytes.fromhex('00 00 12 42  00 00 4c c1'), '36.5 -12.75'),
    (ValueType.VEC3, 'val_vec3', Vec3(36.5, 0.125, -12.75), bytes.fromhex('00 00 12 42 00 00 00 3e 00 00 4c c1'), '36.5 0.125 -12.75'),
    (ValueType.VEC4, 'val_vec4', Vec4(384.0, 36.5, 0.125, -12.75), bytes.fromhex('00 00 c0 43 00 00 12 42 00 00 00 3e 00 00 4c c1'), '384 36.5 0.125 -12.75'),
    (ValueType.QUATERNION, 'val_quat', Quaternion(384.0, 36.5, 0.125, -12.75), bytes.fromhex('00 00 c0 43 00 00 12 42 00 00 00 3e 00 00 4c c1'), '384 36.5 0.125 -12.75'),
    (
        ValueType.MATRIX, 'val_mat', Matrix(),
        (b'1000' b'0100' b'0010' b'0001').replace(b'0', b'\x00\x00\x00\x00').replace(b'1', b'\x00\x00\x80?'),
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


deduce_type_tests = [
    (5, ValueType.INT, 5),
    (5.0, ValueType.FLOAT, 5.0),
    (False, ValueType.BOOL, False),
    (True, ValueType.BOOL, True),
    ('test', ValueType.STR, 'test'),
    (b'test', ValueType.BIN, b'test'),

    (Color(25, 48, 255, 255), ValueType.COLOR, Color(25, 48, 255, 255)),
    (Color(25.0, 48.2, 255.4, 255.9), ValueType.COLOR, Color(25, 48, 255, 255)),

    (Vec2(1, 4.0), ValueType.VEC2, Vec2(1.0, 4.0)),
    (Vec3(1.8, 4.0, 8), ValueType.VEC3, Vec3(1.8, 4.0, 8.0)),
    (Vec4(4, 8, 23, 29.8), ValueType.VEC4, Vec4(4.0, 8.0, 23.0, 29.8)),

    (AngleTup(45.0, 92.6, 23.0), ValueType.ANGLE, AngleTup(45.0, 92.6, 23.0)),
    (Angle(45.0, 92.6, 23.0), ValueType.ANGLE, AngleTup(45.0, 92.6, 23.0)),
    (Quaternion(4, 8, 23, 29.8), ValueType.QUATERNION, Quaternion(4.0, 8.0, 23.0, 29.8)),

    # Iterable testing.
    ((3, 4), ValueType.VEC2, Vec2(3.0, 4.0)),
    ((3, 4, 5), ValueType.VEC3, Vec3(3.0, 4.0, 5.0)),
    (range(4), ValueType.VEC4, Vec4(0.0, 1.0, 2.0, 3.0)),
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
    # Angle/AngleTup can be mixed.
    ([Angle(3.0, 4.0, 5.0), AngleTup(4.0, 3.0, 6.0)], ValueType.ANGLE, [AngleTup(3.0, 4.0, 5.0), AngleTup(4.0, 3.0, 6.0)]),

    # A list of lists is an iterable.
    ([[4, 5], Vec2(6.0, 7.0)], ValueType.VEC2, [Vec2(4.0, 5.0), Vec2(6.0, 7.0)])
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

    with pytest.raises(TypeError):
        print(deduce_type([]))
    # Iterable with wrong size.
    with pytest.raises(TypeError):
        print(deduce_type(()))
    with pytest.raises(TypeError):
        print(deduce_type((1, )))
    with pytest.raises(TypeError):
        print(deduce_type((1, 2, 3, 4, 5)))
    with pytest.raises(TypeError):
        print(deduce_type(range(0)))
    with pytest.raises(TypeError):
        print(deduce_type(range(1)))
    with pytest.raises(TypeError):
        print(deduce_type(range(5)))


@pytest.mark.parametrize('filename', [
    'keyvalues2',
    'binary_v2',  # HL2's dmxconvert
    'binary_v4',  # L4D2's dmxconvert
    'binary_v5',  # P2+'s dmxconvert
])
def test_parse(filename: str) -> None:
    """Test parsing all the format types."""
    # string <-> double <-> float conversions are not quite accurate.
    a = cast(Callable[[float], float], pytest.approx)

    with open(f'dmx_samples/{filename}.dmx', 'rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)
    assert fmt_name == 'dmx'
    assert fmt_version == 4

    assert root.name == 'Root_Name'
    assert root.type == 'DmeRootElement'
    assert root.uuid == UUID('b66a2ce3-d686-4dbf-85df-07c6b275bebb')
    assert len(root) == 3
    assert sorted(root.keys()) == ['arrays', 'recursive', 'scalars']

    # First, the recursive element tree.
    recur_attr = root['recursive']
    assert recur_attr.name == 'recurSive'
    recur = recur_attr.val_elem
    assert recur.name == 'Recurse1'
    assert recur.type == 'RecurseElement'
    assert len(recur) == 2
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

    assert len(scalars) == 19
    assert scalars['neg_integer'].val_int == -1230552801
    assert scalars['pos_integer'].val_int == 296703200

    assert scalars['neg_float'].val_float == a(-16211.59325)
    assert scalars['pos_float'].val_float == a(22097.83875)

    assert scalars['truth'].val_bool is True
    assert scalars['falsity'].val_bool is False

    assert scalars['red'].val_color == Color(240, 32, 32, 255)
    assert scalars['blue'].val_color == Color(32, 240, 32, 255)
    assert scalars['green'].val_color == Color(32, 32, 240, 255)
    assert scalars['white'].val_color == Color(255, 255, 255, 255)
    assert scalars['half'].val_color == Color(0, 0, 0, 128)

    assert scalars['vec2'].val_vec2 == Vec2(a(348.275), a(-389.935))
    assert scalars['vec3'].val_vec3 == Vec3(a(128.25), a(-1048.5), a(16382.1902))
    assert scalars['vec4'].val_vec4 == Vec4(a(128.25), a(-1048.5), a(16382.1902), a(-389.935))

    assert scalars['up'].val_ang == AngleTup(-90, 0, 0)
    assert scalars['dn'].val_ang == AngleTup(90, 0, 0)
    assert scalars['somedir'].val_ang == AngleTup(a(291), a(-48.9), a(45.0))
    assert scalars['quat'].val_quat == Quaternion(a(0.267261), a(0.534522), a(0.801784), 0.0)

    assert scalars['hex'].val_bin == (
        b'\x92&\xd0\xc7\x12\xec9\xe9\xd1cE\x19\xd1\xbd\x0f'
        b'Jv\x1co\x81\xaf\xb5x[K\x9c\x85)\x12t\xff&\xcd7'
        b'\x0e\xb1s\x18\xfa2m"\xef\xf4\xd8\xb8\xf9\xd4\x1ek'
    )

    # And finally arrays.
    arrays = root['arrays'].val_elem
    assert arrays.type == 'TypeHolder'
    assert arrays.uuid == UUID('2b95889f-5041-436e-9350-813abcf504b0')
    assert arrays.name == 'ArrayValues'
    assert len(arrays) == 11

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
    assert list(arr_bool) == [
        False, False, True, False,
        False, True, True, True,
    ]
    assert list(arrays['colors']) == [
        Color(255, 0, 0, 255),
        Color(0, 0, 255, 128),
        Color(0, 240, 0, 0),
    ]
    assert list(arrays['2ds']) == [
        Vec2(x, y)
        for x in [-1, 0, 1]
        for y in [-1, 0, 1]
    ]
    assert list(arrays['3ds']) == [
        Vec3(a(34.0), a(-348.25), a(128.125)),
        Vec3(0.0, 0.0, 0.0),
        Vec3(a(0.9), a(0.8), a(0.5)),
    ]
    assert list(arrays['4dimensional']) == [
        Vec4(0.0, 0.0, 0.0, 0.0),
        Vec4(0.0, 0.0, 0.0, 1.0),
        Vec4(0.0, 2.0, 0, 48.0),
        Vec4(-28.0, 380.0, -39.0, 39.0),
    ]
    assert list(arrays['directions']) == [
        AngleTup(0, 0, 0), AngleTup(45, 0, 90),
        AngleTup(0, 135, 180), AngleTup(45, 31, -39),
        AngleTup(94, 165, 23),
    ]
    assert len(arrays['quaternions']) == 2
    assert arrays['quaternions'][0].val_quat == Quaternion(a(0.267261), a(-0.801784), a(0.534522), 0.0)
    assert arrays['quaternions'][1].val_quat == Quaternion(a(-0.534522), a(0.267261), 0.0, a(-0.801784))

    assert len(arrays['hexes']) == 3
    assert list(arrays['hexes']) == [
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


def test_parse_binaryv3() -> None:
    """Test parsing of binary version 3.

    We don't have a DMXconvert that produces this, so instead compare an existing
    file to the text version.
    """
    with open(f'dmx_samples/tf_movies.dmx', 'rb') as f:
        root_bin, fmt_name, fmt_version = Element.parse(f)
    with open(f'dmx_samples/tf_movies_text.dmx', 'rb') as f:
        root_txt, fmt_name, fmt_version = Element.parse(f)
    assert_tree(root_bin, root_txt)
