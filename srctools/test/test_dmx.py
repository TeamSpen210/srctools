"""Test the datamodel exchange implementation."""
import pytest

from srctools import Matrix
from srctools.dmx import (
    Element, ValueType, Vec2, Vec3, Vec4, AngleTup, Color,
    Quaternion,
)


def test_val_int() -> None:
    """Test integer-type values."""
    elem = Element.int('Name', 45)
    assert elem.val_int == 45
    assert elem.val_str == '45'
    assert elem.val_float == 45.0

    assert elem.val_vec2 == Vec2(45.0, 45.0)
    assert elem.val_vec3 == Vec3(45.0, 45.0, 45.0)
    assert elem.val_vec4 == Vec4(45.0, 45.0, 45.0, 45.0)

    assert Element.int('Blah', 45).val_bool is True
    assert Element.int('Blah', 0).val_bool is False
    assert Element.int('Blah', -2).val_bool is True


def test_val_float() -> None:
    """Test float-type values."""
    elem = Element.float('Name', 32.25)
    assert elem.val_int == 32
    assert Element.float('Name', -32.25).val_int == -32
    assert elem.val_str == '32.25'
    assert elem.val_float == 32.25

    assert elem.val_vec2 == Vec2(32.25, 32.25)
    assert elem.val_vec3 == Vec3(32.25, 32.25, 32.25)
    assert elem.val_vec4 == Vec4(32.25, 32.25, 32.25, 32.25)

    assert Element.float('Blah', 32.25).val_bool is True
    assert Element.float('Blah', 0.0).val_bool is False
    assert Element.float('Blah', -12.8).val_bool is True


def test_val_str() -> None:
    """Test string-type values."""
    assert Element.string('', '45').val_str == '45'
    assert Element.string('', '').val_str == ''
    assert Element.string('', 'testing str\ning').val_str == 'testing str\ning'

    assert Element.string('Name', '45').val_int == 45
    assert Element.string('Name', '-45').val_int == -45
    assert Element.string('Name', '0').val_int == 0

    assert Element.string('', '45').val_float == 45.0
    assert Element.string('', '45.0').val_float == 45.0
    assert Element.string('', '45.375').val_float == 45.375
    assert Element.string('', '-45.375').val_float == -45.375
    assert Element.string('', '.25').val_float == 0.25
    assert Element.string('', '0').val_float == 0.0

    assert Element.string('', '1').val_bool is True
    assert Element.string('', '0').val_bool is False
    assert Element.string('', 'yEs').val_bool is True
    assert Element.string('', 'No').val_bool is False
    assert Element.string('', 'tRue').val_bool is True
    assert Element.string('', 'faLse').val_bool is False


def test_val_void() -> None:
    """Test void values have a default result."""
    elem = Element.void('Name')
    assert elem.val_int == 0
    assert elem.val_float == 0.0
    assert elem.val_bool is False

    assert elem.val_vec2 == Vec2(0.0, 0.0)
    assert elem.val_vec3 == Vec3(0.0, 0.0, 0.0)
    assert elem.val_vec4 == Vec4(0.0, 0.0, 0.0, 0.0)

    assert elem.val_str == ''
    assert elem.val_string == ''

    assert elem.val_color == Color(0, 0, 0)
    assert elem.val_colour == Color(0, 0, 0)

    assert elem.val_angle == AngleTup(0.0, 0.0, 0.0)
    assert elem.val_ang == AngleTup(0.0, 0.0, 0.0)

    assert elem.val_quat == Quaternion(0.0, 0.0, 0.0, 1.0)
    assert elem.val_quaternion == Quaternion(0.0, 0.0, 0.0, 1.0)

    assert elem.val_mat == Matrix()
    assert elem.val_matrix == Matrix()


@pytest.mark.parametrize(['attr', 'typ'], [
    pytest.param(attr, typ, id=attr)
    for attr, typ in [
        ('val_str', str),
        ('val_string', str),
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
@pytest.mark.parametrize('element', [
    Element.void(''),
    Element.int('', 45),
    Element.float('', 48.9),
    Element.vec2('', 3, 4),
    Element.vec3('', 4, 5, 6),
    Element.vec4('', 5, 6, 7),
    Element.angle('', 45.0, 90.0, 0.0),
    Element.color('', 255, 128, 64),
], ids=lambda el: el.typ.name.lower())
def test_conv_types(element: Element, attr: str, typ: type) -> None:
    """Check all the conversions either fail or produce the right result.

    We don't test strings since valid values are different for different dests.
    """
    try:
        result = getattr(element, attr)
    except ValueError:
        # Conversion failed, that's fine.
        pytest.xfail()
        return

    assert type(result) is typ, result
