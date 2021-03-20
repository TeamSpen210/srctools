"""Test the datamodel exchange implementation."""
from uuid import UUID

import pytest

from srctools import Matrix, Angle
from srctools.dmx import (
    Element, Attribute, ValueType, Vec2, Vec3, Vec4, AngleTup, Color,
    Quaternion, deduce_type,
)


def test_attr_val_int() -> None:
    """Test integer-type values."""
    elem = Attribute.int('Name', 45)
    assert elem.val_int == 45
    assert elem.val_str == '45'
    assert elem.val_float == 45.0

    assert elem.val_vec2 == Vec2(45.0, 45.0)
    assert elem.val_vec3 == Vec3(45.0, 45.0, 45.0)
    assert elem.val_vec4 == Vec4(45.0, 45.0, 45.0, 45.0)

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


@pytest.mark.parametrize(['attr', 'typ'], [
    pytest.param(attr, typ, id=attr)
    for attr, typ in [
        ('val_str', str),
        ('val_string', str),
        ('val_bin', bytes),
        ('val_binary', bytes),
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
    Attribute.vec2('', 3, 4),
    Attribute.vec3('', 4, 5, 6),
    Attribute.vec4('', 5, 6, 7),
    Attribute.angle('', 45.0, 90.0, 0.0),
    Attribute.color('', 255, 128, 64),
    Attribute.quaternion('', 0.0, 0.0, 0.0, 1.0),
], ids=lambda attr: attr.type.name.lower())
def test_attr_conv_types(attribute: Attribute, attr: str, typ: type) -> None:
    """Check all the conversions either fail or produce the right result.

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
])
def test_parse(filename: str) -> None:
    """Test parsing all the format types."""
    with open(f'dmx_samples/{filename}.dmx', 'rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)
    assert fmt_name == 'generic'
    assert fmt_version == 25

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

    assert len(scalars) == 18
    assert scalars['neg_integer'].val_int == -1230552801
    assert scalars['pos_integer'].val_int == 296703200

    assert scalars['neg_float'].val_float == -16211.593677226905
    assert scalars['pos_float'].val_float == 22097.838726432383

    assert scalars['truth'].val_bool is True
    assert scalars['falsity'].val_bool is False

    assert scalars['red'].val_color == Color(240, 32, 32, 255)
    assert scalars['blue'].val_color == Color(32, 240, 32, 255)
    assert scalars['green'].val_color == Color(32, 32, 240, 255)
    assert scalars['white'].val_color == Color(255, 255, 255, 255)
    assert scalars['half'].val_color == Color(0, 0, 0, 128)

    assert scalars['vec2'].val_vec2 == Vec2(348.275, -389.935)
    assert scalars['vec3'].val_vec3 == Vec3(128.25, -1048.5, 16382.1902)
    assert scalars['vec4'].val_vec4 == Vec4(128.25, -1048.5, 16382.1902, -389.935)

    assert scalars['up'].val_ang == AngleTup(-90, 0, 0)
    assert scalars['dn'].val_ang == AngleTup(90, 0, 0)
    assert scalars['somedir'].val_ang == AngleTup(291, -48.9, 45.0)
    assert scalars['quat'].val_quat == Quaternion(3.9, -2.3, 3.4, 0.0)

    # And finally arrays.
    arrays = root['arrays'].val_elem
    assert arrays.type == 'TypeHolder'
    assert arrays.uuid == UUID('2b95889f-5041-436e-9350-813abcf504b0')
    assert arrays.name == 'ArrayValues'
    assert len(arrays) == 10

    arr_int = arrays['integers']
    assert len(arr_int) == 5
    assert arr_int[0].val_int == 1
    assert arr_int[1].val_int == 2
    assert arr_int[2].val_int == 35
    assert arr_int[3].val_int == -39
    assert arr_int[4].val_int == 0

    arr_float = arrays['floating']
    assert len(arr_float) == 10
    assert arr_float[0].val_float == -10291.153564142704
    assert arr_float[1].val_float == -55646.21366998823
    assert arr_float[2].val_float == 78545.15227150527
    assert arr_float[3].val_float == -95302.87890687701
    assert arr_float[4].val_float == -45690.04457919854
    assert arr_float[5].val_float == -55299.052361444636
    assert arr_float[6].val_float == 96178.44015134772
    assert arr_float[7].val_float == 58708.297849454975
    assert arr_float[8].val_float == -49957.20355861797
    assert arr_float[9].val_float == 23980.82395184

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
        Vec3(34.0, -348.25, 128.125),
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.9, 0.8, 0.5),
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
    assert arrays['quaternions'][0].val_quat == Quaternion(0.1, -0.9, 2.3, 0.0)
    assert arrays['quaternions'][1].val_quat == Quaternion(5.0, 1.5, -4.0, 1.0)

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
