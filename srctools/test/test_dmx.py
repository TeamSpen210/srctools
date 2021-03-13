"""Test the datamodel exchange implementation."""
import pytest

from srctools import Matrix
from srctools.dmx import (
    Element, Attribute, ValueType, Vec2, Vec3, Vec4, AngleTup, Color,
    Quaternion,
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
