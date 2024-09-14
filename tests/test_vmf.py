"""Tests for the VMF library."""
from typing import Optional
from enum import Enum

from dirty_equals import IsList
from pytest_regressions.file_regression import FileRegressionFixture
import pytest

from helpers import ExactType
from srctools import Angle, FrozenAngle, FrozenMatrix, FrozenVec, Keyvalues, Matrix, Vec
from srctools.vmf import (
    VMF, Axis, Entity, Output, Strata2DViewport, Strata3DViewport,
    StrataInstanceVisibility, conv_kv,
)


def assert_output(
    out: Output,
    output: str,
    target: str,
    inp: str,
    params: str = '',
    delay: float = 0.0,
    *,
    times: int = -1,
    inst_out: Optional[str] = None,
    inst_in: Optional[str] = None,
    comma_sep: bool = False,
) -> None:
    """Assert that the output matches the provided values."""
    if out.output != output:
        fail = 'output'
    elif out.target != target:
        fail = 'target'
    elif out.input != inp:
        fail = 'input'
    elif out.params != params:
        fail = 'params'
    elif out.delay != pytest.approx(delay, abs=0.005):
        fail = 'delay'
    elif out.times != times:
        fail = 'times'
    elif out.inst_in != inst_in:
        fail = 'inst_in'
    elif out.inst_out != inst_out:
        fail = 'inst_out'
    elif out.comma_sep != comma_sep:
        fail = 'comma_sep'
    else:
        return  # Success!
    pytest.fail(
        f'{out!r}.{fail} != Output(out={output!r}, targ={target!r}, in={inp!r}, '
        f'params={params!r}, delay={delay!r}, times={times!r}, i_out={inst_out!r}, '
        f'i_in={inst_in!r}, comma={comma_sep!r})'
    )


class KVEnum(Enum):
    """An enum with a value that can be converted to a KV."""
    PREFIX = 0
    SUFFIX = True
    NONE = 2.0


class StrClass(str):
    """A string subclass."""


class FloatClass(str):
    """A float subclass."""


class IntClass(str):
    """An integer subclass."""


def test_valid_kvs() -> None:
    """Test valid KV conversions."""
    assert conv_kv(4.7) == "4.7"
    assert conv_kv("four") == "four"
    assert conv_kv(8) == "8"
    assert conv_kv(False) == "0"
    assert conv_kv(True) == "1"
    assert conv_kv(Vec(1, 2.5, 3)) == "1 2.5 3"
    assert conv_kv(FrozenVec(1, 2.5, 3)) == "1 2.5 3"
    assert conv_kv(Angle(12.5, 90, 0.0)) == "12.5 90 0"
    assert conv_kv(FrozenAngle(12.5, 90, 0.0)) == "12.5 90 0"
    assert conv_kv(Matrix.from_roll(30)) == "0 0 30"
    assert conv_kv(FrozenMatrix.from_roll(30)) == "0 0 30"
    assert conv_kv(KVEnum.PREFIX) == ExactType("0")
    assert conv_kv(KVEnum.SUFFIX) == ExactType("1")
    assert conv_kv(KVEnum.NONE) == ExactType("2")
    # Check subclasses are valid, and produce the same types.
    assert conv_kv(StrClass("subclass")) == ExactType("subclass")
    assert conv_kv(FloatClass(4.7)) == ExactType("4.7")
    assert conv_kv(IntClass(8)) == ExactType("8")


def test_entkey_basic() -> None:
    """Test entity mapping functionality."""
    obj = object()  # Arbitrary example object.

    ent = VMF().create_ent('info_null')
    # If an assertion fails, include the internal state with --showlocals.
    internal_keys = ent._keys  # noqa
    assert len(ent) == 1
    assert list(ent) == ['classname']
    with pytest.warns(DeprecationWarning):
        assert list(ent.keys()) == ['classname']
    assert list(ent.values()) == ['info_null']
    assert list(ent.items()) == [('classname', 'info_null')]

    ent['target'] = 'the_target'

    assert ent['target'] == 'the_target'  # Standard
    assert ent['tARget'] == 'the_target'  # Case-insensitive

    assert ent['invalid'] == '', 'Defaults to ""'
    assert ent['target', '!picker'] == 'the_target'  # Unused default.
    assert ent['not_here', 'default'] == 'default'  # Used default.
    assert ent['not_here', obj] is obj  # Default can be anything.

    assert ent.get('invalid') == ''  # Defaults to ""
    assert ent.get('target', 'default') == 'the_target'  # Unused default.
    assert ent.get('not_here', obj) is obj  # Default can be anything.

    ent['health'] = 42  # Integer, converted to string.
    assert ent['health'] == '42'
    ent['Range'] = 45.75
    assert ent['Range'] == '45.75'
    ent['allowRespawn'] = True  # Special case, bools become 1/0.
    assert ent['allowrespawn'] == '1'
    ent['canKill'] = False
    assert ent['canKill'] == '0'
    ent['movedirection'] = Angle(0, 90, 0)  # Angles convert.
    assert ent['movedirection'] == '0 90 0'

    # Order not guaranteed.
    assert len(ent) == 7
    assert list(ent) == IsList(
        'classname', 'target', 'health', 'Range', 'allowRespawn', 'canKill', 'movedirection',
        check_order=False,
    )
    with pytest.warns(DeprecationWarning):
        keys = ent.keys()
    assert list(keys) == IsList(
        'classname', 'target', 'health', 'Range', 'allowRespawn', 'canKill', 'movedirection',
        check_order=False,
    )
    assert list(ent.values()) == IsList(
        'info_null', 'the_target', '42', '45.75', '1', '0', '0 90 0',
        check_order=False,
    )
    assert list(ent.items()) == IsList(
        ('classname', 'info_null'),
        ('target', 'the_target'),
        ('health', '42'),
        ('Range', '45.75'),
        ('allowRespawn', '1'),
        ('canKill', '0'),
        ('movedirection', '0 90 0'),
        check_order=False,
    )
    # Keys/values/items should have the same order.
    assert list(ent.fixup.items()) == list(zip(ent.fixup.keys(), ent.fixup.values()))


def test_parse_entity() -> None:
    """Test parsing entities from keyvalues."""
    vmf = VMF()
    ent = Entity.parse(vmf, Keyvalues('Entity', [
        Keyvalues('claSSname', 'info_target'),  # Check these are case-insensitive.
        Keyvalues('tarGetname', 'a_target'),
        Keyvalues('origin', '384 -63 278.125'),
        Keyvalues('sOmeValue', '48'),
        Keyvalues('multiplyDefined', 'first'),
        Keyvalues('multiplydefined', 'second'),
        Keyvalues('repLace04', '$pOs 1 2 3'),  # Fixup value.
        Keyvalues('replaceNo', 'this is a regular kv'),
        Keyvalues('replace01', '$var '),  # Special case, blank keyvalue.
    ]))
    assert list(ent.items()) == IsList(
        ('claSSname', 'info_target'),
        ('tarGetname', 'a_target'),
        ('origin', '384 -63 278.125'),
        ('sOmeValue', '48'),
        ('multiplyDefined', 'second'),
        ('replaceNo', 'this is a regular kv'),
        check_order=False,
    )
    assert list(ent.fixup.items()) == IsList(
        ('pOs', '1 2 3'),
        ('var', ''),
        check_order=False,
    )
    vmf.add_ent(ent)
    assert vmf.by_class['info_target'] == {ent}
    assert vmf.by_target['a_target'] == {ent}


def test_by_target() -> None:
    """Test the behaviour of the by_target lookup mechanism."""
    vmf = VMF()
    assert vmf.by_target == {None: {vmf.spawn}}
    ent1 = vmf.create_ent('info_target')
    assert vmf.by_target == {None: {ent1, vmf.spawn}}

    ent1['targetname'] = 'some_name'
    assert list(vmf.by_target) == IsList('some_name', None, check_order=False)
    assert list(vmf.by_target.keys()) == IsList('some_name', None, check_order=False)
    assert vmf.by_target['some_name'] == {ent1}
    assert vmf.by_target == {'some_name': {ent1}, None: {vmf.spawn}}

    ent2 = Entity(vmf, {'classname': 'info_target', 'targetname': 'some_name'})
    assert vmf.by_target == {'some_name': {ent1}, None: {vmf.spawn}}  # Not yet included.
    vmf.add_ent(ent2)
    assert vmf.by_target == {'some_name': {ent1, ent2}, None: {vmf.spawn}}

    ent1['targetname'] = 'another'
    assert vmf.by_target == {'some_name': {ent2}, 'another': {ent1}, None: {vmf.spawn}}
    del ent2['targetname']
    assert vmf.by_target == {'another': {ent1}, None: {vmf.spawn, ent2}}
    ent2['targetname'] = 'some_name'
    ent1.remove()
    assert vmf.by_target == {'some_name': {ent2}, None: {vmf.spawn}}


def test_by_class() -> None:
    """Test the behaviour of the by_class lookup mechanism."""
    vmf = VMF()
    assert vmf.by_class == {'worldspawn': {vmf.spawn}}

    ent1 = vmf.create_ent('info_target')
    assert vmf.by_class == {'worldspawn': {vmf.spawn}, 'info_target': {ent1}}
    assert list(vmf.by_class) == IsList('worldspawn', 'info_target', check_order=False)
    assert list(vmf.by_class.keys()) == IsList('worldspawn', 'info_target', check_order=False)
    assert vmf.by_class['info_target'] == {ent1}

    ent2 = Entity(vmf, {'classname': 'info_target'})
    assert vmf.by_class == {'worldspawn': {vmf.spawn}, 'info_target': {ent1}}  # Not yet included.
    vmf.add_ent(ent2)
    assert vmf.by_class == {'worldspawn': {vmf.spawn}, 'info_target': {ent1, ent2}}

    ent1['classname'] = 'math_counter'
    assert vmf.by_class == {'worldspawn': {vmf.spawn}, 'info_target': {ent2}, 'math_counter': {ent1}}
    with pytest.raises(KeyError):  # Not allowed.
        del ent2['classname']
    ent2.remove()
    assert vmf.by_class == {
        'worldspawn': {vmf.spawn},
        'math_counter': {ent1},
    }

    with pytest.raises(ValueError, match='worldspawn'):  # Not allowed to change this.
        vmf.spawn['classname'] = 'func_brush'
    assert vmf.spawn['classname'] == 'worldspawn'  # Change did not apply.


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

    assert ent.fixup['$test'] == 'hello'  # Standard
    assert ent.fixup['$teSt'] == 'hello'  # Case-insensitive
    assert ent.fixup['test'] == 'hello'  # No $ sign is allowed

    assert ent.fixup['$notPresent'] == ''  # Defaults to ""
    assert ent.fixup['$test', 'default'] == 'hello'  # Unused default.
    assert ent.fixup['not_here', 'default'] == 'default'  # Used default.
    assert ent.fixup['not_here', obj] is obj  # Default can be anything.

    assert ent.fixup.get('$notPresent') == ''  # Defaults to ""
    assert ent.fixup.get('$test', 'default') == 'hello'  # Unused default.
    assert ent.fixup.get('not_here', obj) is obj  # Default can be anything.

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


def test_output_parse() -> None:
    """Test parsing various forms of syntax."""
    # New separator is prioritised.
    assert_output(
        Output.parse(Keyvalues('OnOutput', 'the_target\x1bAnInput\x1bfunc(1,2,3)\x1b0.5\x1b5')),
        'OnOutput', 'the_target', 'AnInput', 'func(1,2,3)', 0.5, times=5, comma_sep=False,
    )
    # Old comma separator
    assert_output(
        Output.parse(Keyvalues('OnOutput', 'the_target,AnInput,param,0.5,5')),
        'OnOutput', 'the_target', 'AnInput', 'param', 0.5, times=5, comma_sep=True,
    )
    # Extension feature, if extra commas appear, assume it's params.
    assert_output(
        Output.parse(Keyvalues('OnOutput', 'the_target,AnInput,func(4, 6),0.5,5')),
        'OnOutput', 'the_target', 'AnInput', 'func(4, 6)', 0.5, times=5, comma_sep=True,
    )
    # Parsing of the instance parameters
    assert_output(
        Output.parse(Keyvalues(
            'instance:out_rl;OnTrigger',
            'the_target\x1binstance:thing;Ignite\x1bfunc(1,2,3)\x1b0.5\x1b5'
        )),
        'OnTrigger', 'the_target', 'Ignite', 'func(1,2,3)', 0.5, times=5, comma_sep=False,
        inst_out='out_rl', inst_in='thing',
    )


def test_blank_vmf(file_regression: FileRegressionFixture) -> None:
    """Test parsing a blank file produces a default VMF."""
    vmf = VMF.parse(Keyvalues.root())
    file_regression.check(vmf.export(), extension='.vmf')


def test_regression(file_regression: FileRegressionFixture) -> None:
    """Generate a VMF to ensure code doesn't unintentionally alter output."""
    vmf = VMF()
    vmf.create_ent(
        'info_target',
        origin=Vec(24, 38, 1028),
        angles=Angle(10, 270, 0),
        escaped_key='Some key with "quotes" in it.',
    ).add_out(
        Output('OnUser1', '!player', 'Holster"Weapon"', delay=0.1),
    )
    vmf.map_ver = 8192
    vmf.show_grid = True
    vmf.snap_grid = False
    vmf.hammer_build = 9999
    vmf.grid_spacing = 512
    vmf.strata_instance_vis = StrataInstanceVisibility.NORMAL
    vmf.strata_viewports = [
        Strata3DViewport(Vec(8, -3, 128), Angle(0, 90, 0)),
        Strata2DViewport('y', 128.384, -12.7467, 4.0),
        Strata2DViewport('x', 128.75, 36.25, 0.125),
        Strata2DViewport('z', 7181, -282.875, 1.0),
    ]

    vmf.add_brush(vmf.make_prism(
        Vec(128, 128, 128), Vec(256, 512, 1024),
        set_points=True,
    ))
    file_regression.check(vmf.export(), extension='.vmf')


@pytest.mark.parametrize('source', ['constructor', 'parse'])
def test_vmf_defaults(source: str) -> None:
    """Check a blank VMF produces sensible results."""
    if source == 'constructor':
        vmf = VMF()
    else:
        vmf = VMF.parse(Keyvalues.root())
    # Check default options.
    assert vmf.map_ver == 0
    assert vmf.show_grid is True
    assert vmf.show_3d_grid is False
    assert vmf.snap_grid is True
    assert vmf.show_logic_grid is False
    assert vmf.hammer_ver == 400
    assert vmf.hammer_build == 5304
    assert vmf.grid_spacing == 64
    assert vmf.active_cam == -1
    assert vmf.quickhide_count == 0


def test_map_info() -> None:
    """Test setting options via map_info."""
    # No warning with an empty map info.
    VMF(map_info={})

    vmf = pytest.deprecated_call(VMF,  map_info={'showgrid': '0'})
    assert vmf.show_grid is False

    vmf = pytest.deprecated_call(VMF,  map_info={'snaptogrid': '0'})
    assert vmf.snap_grid is False

    vmf = pytest.deprecated_call(VMF,  map_info={'show3dgrid': '1'})
    assert vmf.show_3d_grid is True

    vmf = pytest.deprecated_call(VMF,  map_info={'showlogicalgrid': '1'})
    assert vmf.show_logic_grid is True

    vmf = pytest.deprecated_call(VMF,  map_info={'gridspacing': '1024'})
    assert vmf.grid_spacing == 1024

    vmf = pytest.deprecated_call(VMF,  map_info={'active_cam': '84'})
    assert vmf.active_cam == 84

    vmf = pytest.deprecated_call(VMF,  map_info={'quickhide': '12'})
    assert vmf.quickhide_count == 12


def test_make_unique() -> None:
    """Test the entity.make_unique() method."""
    vmf = VMF()
    vmf.create_ent('info_target', targetname='alpha')
    vmf.create_ent('info_target', targetname='alpha1')

    vmf.create_ent('info_target', targetname='alpha2')

    alpha4 = vmf.create_ent('info_target', targetname='alpha4')
    alpha4.make_unique()  # Name clear, preserve current name.
    assert alpha4['targetname'] == 'alpha4'

    alpha3 = vmf.create_ent('info_target')
    alpha3.make_unique('alpha')
    assert alpha3['targetname'] == 'alpha3'

    # With overlaps, discard existing numeric suffix.
    alpha5 = vmf.create_ent('info_target', targetname='alpha2')
    alpha5.make_unique('beta')
    assert alpha5['targetname'] == 'alpha5'

    for i in range(6, 12):
        vmf.create_ent('info_target', targetname=f'alpha{i}')
    # Check double-digits work.
    alpha12 = vmf.create_ent('info_target').make_unique('alpha')
    assert alpha12['targetname'] == 'alpha12'


@pytest.mark.parametrize('position, axis, u, v', [
    (Vec(65536, 8.75, -12), 'x', 8.75, -12),
    (Vec(8.75, 65536,  -12), 'y', 8.75, -12),
    (Vec(8.75, -12, -65536), 'z', 8.75, -12),
    (Vec(0, -8.75, 12), 'x', -8.75, 12),
])
def test_strata_2d_viewport_position(position: Vec, axis: Axis, u: float, v: float) -> None:
    """The 2D viewport value encodes both a position and axis."""
    assert Strata2DViewport.from_vector(position, 8.0) == Strata2DViewport(axis, u, v, 8.0)


def test_deprecated_cordonsolid() -> None:
    """Solid.is_cordon used to be a cordon_solid attribute."""
    vmf = VMF()
    solid = vmf.make_prism(Vec(-128, -128, -128), Vec(+128, +128, +128)).solid
    with pytest.deprecated_call():
        assert solid.cordon_solid is None
    solid.is_cordon = True
    with pytest.deprecated_call():
        assert solid.cordon_solid == 1
    with pytest.deprecated_call():
        solid.cordon_solid = None
    assert solid.is_cordon is False
