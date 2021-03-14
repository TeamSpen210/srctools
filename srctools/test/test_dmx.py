"""Test the datamodel exchange implementation."""
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
