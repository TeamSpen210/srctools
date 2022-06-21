"""Tests for the VMF library."""
import pytest

from srctools import Vec, Angle
from srctools.vmf import Entity, VMF, Output


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
    assert set(ent.fixup) == {'test', 'VALUE', 'true', 'false'}
    assert set(ent.fixup.keys()) == {'test', 'VALUE', 'true', 'false'}
    assert set(ent.fixup.values()) == {'hello', '1', '0', '45.75'}
    assert set(ent.fixup.items()) == {
        ('test', 'hello'),
        ('VALUE', '45.75'),
        ('true', '1'),
        ('false', '0')
    }
    # Keys/values/items should have the same order.
    assert list(ent.fixup.items()) == list(zip(ent.fixup.keys(), ent.fixup.values()))


def test_fixup_containment() -> None:
    """Check in operators in Entity.fixup."""
    ent = VMF().create_ent('any')
    ent.fixup['$test'] = 'hello'
    ent.fixup['VALUE'] = '45.75'
    ent.fixup['true'] = '1'
    ent.fixup['false'] = '0'
    # Invalid types should return False, not TypeError.
    assert '$test' in ent.fixup.keys()
    assert 'fAlSe' in ent.fixup.keys()
    assert '' not in ent.fixup.keys()
    assert ValueError not in ent.fixup.keys()

    assert '1' in ent.fixup.values()
    assert '45.75' in ent.fixup.values()
    assert '' not in ent.fixup.values()
    assert 'false' not in ent.fixup.values()
    assert ValueError not in ent.fixup.values()

    assert ('$true', '1') in ent.fixup.items()
    assert ('VaLuE', '45.75') in ent.fixup.items()
    assert ('$true', '0') not in ent.fixup.items()
    assert ('FaLse', object) not in ent.fixup.items()
    assert ('', 'test') not in ent.fixup.items()
    assert ('false', ) not in ent.fixup.items()
    assert ValueError not in ent.fixup.items()


def test_fixup_substitution() -> None:
    """Test Entity.fixup.substitute()."""
    ent = VMF().create_ent('any')
    ent.fixup['$var'] = 'out'

    assert ent.fixup.substitute('no var 1234') == 'no var 1234'
    assert ent.fixup.substitute('$var') == 'out'
    assert ent.fixup.substitute('prefix$varsuffix') == 'prefixoutsuffix'
    with pytest.raises(KeyError) as exc:
        ent.fixup.substitute('$notPRESent')
    assert '$notPRESent' in str(exc)

    assert ent.fixup.substitute('blah_$notPresent45:more', 'def') == 'blah_def:more'
    # Potential edge case - 1-long var.
    assert ent.fixup.substitute('blah$x:more', 'def') == 'blahdef:more'
    # Blank and fully numeric vars are not allowed.
    assert ent.fixup.substitute('blah$:more$45', 'def') == 'blah$:more$45'

    ent.fixup['$variable'] = 'long'
    # Value changed, should have remade the regex.
    assert ent.fixup.substitute('X=$variable') == 'X=long'

    # Check common prefixes don't break it.
    assert ent.fixup.substitute('_$var_and_$variable_') == '_out_and_long_'

    ent.fixup.update({'x': 'dunder'})
    assert ent.fixup.substitute('__$x__') == '__dunder__'

    # Ensure regex chars in the var are escaped.
    ent.fixup['$a_var_with*_[]_regex'] = 'ignored'
    assert ent.fixup.substitute('V = $a_var_with*_[]_regex') == 'V = ignored'


def test_fixup_substitution_invert() -> None:
    """Test the additional inverting behaviour."""
    ent = VMF().create_ent('any')
    ent.fixup['$true'] = '1'
    ent.fixup['$false'] = '0'
    ent.fixup['$nonbool'] = 'not_a_bool'

    # Without $, nothing happens.
    assert ent.fixup.substitute('!0') == '!0'
    assert ent.fixup.substitute('!0', allow_invert=True) == '!0'

    assert ent.fixup.substitute('$true $false') == '1 0'
    assert ent.fixup.substitute('$true $false', allow_invert=True) == '1 0'

    # If unspecified, the "!" is left intact.
    assert ent.fixup.substitute('A!$trueB_C!$falseD') == 'A!1B_C!0D'
    # When specified, it doesn't affect other values.
    assert ent.fixup.substitute('A!$trueB_C!$falseD', allow_invert=True) == 'A0B_C1D'

    # For non-boolean values, it is consumed but does nothing.
    assert ent.fixup.substitute('a_$nonbool_value') == 'a_not_a_bool_value'
    assert ent.fixup.substitute('a_!$nonbool_value', allow_invert=True) == 'a_not_a_bool_value'

    # If defaults are provided, those can be flipped too.
    assert ent.fixup.substitute('$missing !$flipped', '0', allow_invert=True) == '0 1'
    assert ent.fixup.substitute('$missing !$flipped', '1', allow_invert=True) == '1 0'


@pytest.mark.parametrize('first, second, expected', [
    (Output('OnTrigger', 'targ1', 'DoAnything', 'ignored'),
     Output('OnWhatever', 'dest', 'Trigger', '45'),
     Output('OnTrigger', 'dest', 'Trigger', '45')),

    (Output('OnIgnited', 'relay', 'Trigger', '42', delay=0.25),
     Output('OnTriggered', 'counter', 'SetValue', ''),
     Output('OnIgnited', 'counter', 'SetValue', '42', delay=0.25)),

    (Output('OnTrigger', 'targ1', 'DoAnything', '', inst_in='ignored', inst_out='output', delay=0.125),
     Output('OnWhatever', 'dest', 'Trigger', '42', inst_in='input', inst_out='ignored', delay=1.0),
     Output('OnTrigger', 'dest', 'Trigger', '42', inst_in='input', inst_out='output', delay=1.125)),

    (Output('OnSingle', 'relay', 'Trigger', only_once=True),
     Output('OnTrigger', 'target', 'FireUser1'),
     Output('OnSingle', 'target', 'FireUser1', only_once=True)),

    (Output('OnMulti', 'doit_once', 'Trigger'),
     Output('OnTrigger', 'target', 'FireUser1', only_once=True),
     Output('OnMulti', 'target', 'FireUser1', only_once=True)),

    (Output('OnMulti', 'multi', 'Trigger', times=9),
     Output('OnTrigger', 'target', 'FireUser1', times=4),
     Output('OnMulti', 'target', 'FireUser1', times=4)),

    (Output('OnFive', 'relay', 'Trigger', times=5),
     Output('OnTrigger', 'target', 'FireUser1', times=10),
     Output('OnFive', 'target', 'FireUser1', times=5)),
], ids=[
    'both_param', 'one_param',
    'instance',
    'first_once', 'second_once',
    'first_times', 'second_times',
])
def test_output_combine(first: Output, second: Output, expected: Output) -> None:
    """Test combining various outputs produces the right results."""
    result = Output.combine(first, second)
    assert result.output == expected.output
    assert result.input == expected.input
    assert result.inst_in == expected.inst_in
    assert result.inst_out == expected.inst_out
    assert result.params == expected.params
    assert pytest.approx(result.delay) == expected.delay
    assert result.times == expected.times
    assert result.comma_sep == expected.comma_sep


def test_regression(file_regression) -> None:
    """Generate a VMF to ensure code doesn't unintentionally alter output."""
    vmf = VMF()
    vmf.create_ent(
        'info_target',
        origin=Vec(24, 38, 1028),
        angles=Angle(10, 270, 0)
    ).add_out(
        Output('OnUser1', '!player', 'HolsterWeapon', delay=0.1),
    )
    file_regression.check(vmf.export(), extension='.vmf')
