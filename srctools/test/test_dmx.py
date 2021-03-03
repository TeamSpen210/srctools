"""Test the datamodel exchange implementation."""
from srctools.dmx import Element, ValueType, Vec2, Vec3, Vec4


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
    assert elem.val_int == 32.25
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
    assert Element.string('', '45') == '45'
    assert Element.string('', '') == ''
    assert Element.string('', 'testing str\ning') == 'testing str\ning'

    assert Element.string('Name', '45').val_int == 45
    assert Element.string('Name', '-45').val_int == -45
    assert Element.string('Name', '0').val_int == 0

    assert Element.string('', '45') == 45.0
    assert Element.string('', '45.0') == 45.0
    assert Element.string('', '45.375') == 45.375
    assert Element.string('', '-45.375') == -45.375
    assert Element.string('', '.25') == 0.25
    assert Element.string('', '0') == 0.0

    assert Element.string('', '').val_bool is False
    assert Element.string('', '1').val_bool is True
    assert Element.string('', '0').val_bool is False
    assert Element.string('', 'yEs').val_bool is True
    assert Element.string('', 'No').val_bool is False
    assert Element.string('', 'tRue').val_bool is True
    assert Element.string('', 'faLse').val_bool is False

