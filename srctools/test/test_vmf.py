"""Tests for the VMF library."""
from srctools.vmf import Entity, VMF
from pytest import raises


def test_fixup_basic() -> None:
    """Test ent.fixup functionality."""
    obj = object()  # Arbitrary example object.

    ent = VMF().create_ent('any')
    assert len(ent.fixup) == 0
    assert list(ent.fixup) == []
    assert list(ent.fixup.keys()) == []
    assert list(ent.fixup.values()) == []
    assert list(ent.fixup.items()) == []

    ent.fixup['$test'] = 'hello'

    assert ent.fixup['$test'] == 'hello', 'Standard'
    assert ent.fixup['$teSt'] == 'hello', 'Case-insentive'
    assert ent.fixup['test'] == 'hello', 'No $ sign is allowed'

    assert ent.fixup['$notPresent'] == '', 'Defaults to ""'
    assert ent.fixup['$test', 'default'] == 'hello', 'Unused default.'
    assert ent.fixup['not_here', 'default'] == 'default', 'Used default.'
    assert ent.fixup['not_here', obj] is obj, 'Default can be anything.'

    assert ent.fixup.get('$notPresent') == '', 'Defaults to ""'
    assert ent.fixup.get('$test', 'default') == 'hello', 'Unused default.'
    assert ent.fixup.get('not_here', obj) is obj, 'Default can be anything.'

    ent.fixup['$VALUE'] = 42  # Integer, converted to string.
    assert ent.fixup['$value'] == '42'
    ent.fixup['$value'] = 45.75
    assert ent.fixup['$value'] == '45.75'
    ent.fixup['$true'] = True  # Special case, bools become 1/0.
    assert ent.fixup['true'] == '1'
    ent.fixup['$false'] = False
    assert ent.fixup['$false'] == '0'

    # Order not guaranteed.
    assert len(ent.fixup) == 4
    assert set(ent.fixup) == {'test', 'value', 'true', 'false'}
    assert set(ent.fixup.keys()) == {'test', 'value', 'true', 'false'}
    assert set(ent.fixup.values()) == {'hello', '1', '0', '45.75'}
    assert set(ent.fixup.items()) == {
        ('test', 'hello'),
        ('value', '45.75'),
        ('true', '1'),
        ('false', '0')
    }
    # Keys/values/items should have the same order.
    assert list(ent.fixup.items()) == list(zip(ent.fixup.keys(), ent.fixup.values()))

