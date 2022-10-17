"""Test the FGD module."""
from typing import Callable
import copy
import io

import pytest

from srctools import Vec
from srctools.fgd import *
from srctools.filesys import VirtualFileSystem


@pytest.mark.parametrize('name1', ['alpha', 'beta', 'gamma'])
@pytest.mark.parametrize('name2', ['alpha', 'beta', 'gamma'])
def test_autovisgroup_cmp(name1: str, name2: str) -> None:
    """Test the AutoVisgroup class is comparable based on name."""
    vis1 = AutoVisgroup(name1, 'SomeParent')
    vis2 = AutoVisgroup(name2, 'SomeParent')

    assert (vis1 == vis2) == (name1 == name2), 'Mismatch'
    assert (vis1 != vis2) == (name1 != name2), 'Mismatch'
    assert (vis1 < vis2) == (name1 < name2), 'Mismatch'
    assert (vis1 > vis2) == (name1 > name2), 'Mismatch'
    assert (vis1 <= vis2) == (name1 <= name2), 'Mismatch'
    assert (vis1 >= vis2) == (name1 >= name2), 'Mismatch'


def test_entity_parse() -> None:
    """Verify parsing an entity produces the correct results."""
    fsys = VirtualFileSystem({'test.fgd': """

@PointClass base(Base1, Base2, Base3) 
base(Base4, base_5)
sphere(radii) 
unknown(a, b, c)
line(240 180 50, targetname, target)
autovis(Auto, some, group)
halfgridsnap // Special, no args
appliesto(tag1, tag2, !tag3)
= some_entity: "The description for this prop, which is spread over " 
+ "multiple lines."
    [
    keyvalue1(string): "Name" : "default": "documentation"
    keyvalue2(int): "Hi": 4
    keyvalue1[tag](boolean): "Tagged Name": 0
    target(target_destination): "An ent"
    
    spawnflags(flags) : "Flags" = [
        1 : "A" : 0
        2 : "B" : 1
        4 : "C" : 0
        8 : "D value" : 0 [old, !good]
        8 : "E" : 1 [new]
    ]
    
    choicelist(choices) : "A Choice" : 0 : "Blahdy blah" = 
        [
        0: "First"
        1: "Second" [new]
        1: "Old second" [-oLd]
        2: "Third"
        "four": "Fourth"
        ]
"""})
    fgd = FGD()
    with fsys:
        fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False)
    ent = fgd['Some_ENtity']
    assert ent.type is EntityTypes.POINT
    assert ent.classname == 'some_entity'
    assert ent.desc == 'The description for this prop, which is spread over multiple lines.'
    assert ent.bases == ['Base1', 'Base2', 'Base3', 'Base4', 'base_5']

    assert ent.helpers == [
        HelperSphere(255, 255, 255, 'radii'),
        UnknownHelper('unknown', ['a', 'b', 'c']),
        HelperLine(240, 180, 50, 'targetname', 'target'),
        HelperHalfGridSnap(),
        HelperExtAppliesTo(['tag1', 'tag2', '!tag3'])
    ]

    assert ent.kv['keyvalue1', set()] == KeyValues(
        'keyvalue1',
        ValueTypes.STRING,
        'Name',
        'default',
        'documentation',
    )
    assert ent.kv['keyvalue1', {'tag'}] == KeyValues(
        'keyvalue1',
        ValueTypes.BOOL,
        'Tagged Name',
        '0',
        '',
    )
    assert ent.kv['keyvalue2'] == KeyValues(
        'keyvalue2',
        ValueTypes.INT,
        'Hi',
        '4',
        '',
    )

    assert ent.kv['spawnflags'] == KeyValues(
        'spawnflags',
        ValueTypes.SPAWNFLAGS,
        'Flags',
        val_list=[
            (1, 'A', False, frozenset()),
            (2, 'B', True, frozenset()),
            (4, 'C', False, frozenset()),
            (8, 'D value', False, frozenset({'OLD', '!GOOD'})),
            (8, 'E', True, frozenset({'NEW'})),
        ],
    )

    assert ent.kv['choicelist'] == KeyValues(
        'choicelist',
        ValueTypes.CHOICES,
        'A Choice',
        '0',
        'Blahdy blah',
        val_list=[
            ('0', 'First', frozenset()),
            ('1', 'Second', frozenset({'NEW'})),
            ('1', 'Old second', frozenset({'-OLD'})),
            ('2', 'Third', frozenset()),
            ('four', 'Fourth', frozenset()),
        ],
    )


@pytest.mark.parametrize('code, is_readonly, is_report', [
    ('(int): "None"', False, False),
    ('(int) readonly: "Readonly"', True, False),
    ('(*int): "Old Report"', False, True),
    ('(int) report: "New Report"', False, True),
    ('(*int) readonly: "Both old-style"', True, True),
    ('(int) readonly report: "Both new-style"', True, True),
    # 'report readonly' is not allowed.
    ('(*int) report: "Redundant"', False, True),
    ('(*int) readonly report: "Redundant + readonly"', True, True),
])
def test_parse_kv_flags(code, is_readonly, is_report) -> None:
    """Test the readonly and reportable flags can be set."""
    fsys = VirtualFileSystem({'test.fgd': f"""
    @PointClass = ent
        [
        keyvalue{code}: 1
        ]
    """})
    fgd = FGD()
    fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False)
    kv = fgd['ent'].kv['keyvalue']

    assert kv.readonly is is_readonly, kv
    assert kv.reportable is is_report, kv


def test_export_regressions(file_regression) -> None:
    """Generate a FGD file to ensure code doesn't unintentionally alter output."""
    fgd = FGD()
    base_origin = EntityDef(EntityTypes.BASE, 'Origin')
    base_angles = EntityDef(EntityTypes.BASE, 'Angles')

    ent = EntityDef(EntityTypes.NPC, 'npc_test')
    ent.bases = [base_origin, base_angles]

    fgd.entities = {
        # 'origin': base_origin,
        'angles': base_angles,
        'npc_test': ent,
    }
    base_origin.keyvalues['origin'] = {frozenset(): KeyValues(
        'origin',
        ValueTypes.VEC_ORIGIN,
        'Origin',
        '0 0 0',
    )}
    base_angles.keyvalues['angles'] = {frozenset(): KeyValues(
        'angles',
        ValueTypes.ANGLES,
        'Angles (Pitch Yaw Roll)',
        '0 0 0',
    )}

    ent.helpers = [
        HelperSphere(255.0, 128.0, 64.0, 'radius'),
        HelperModel('models/editor/a_prop.mdl'),
        UnknownHelper('extrahelper', ['1', '15', 'thirtytwo']),
        HelperSize(Vec(-16, -16, -16), Vec(16, 16, 16)),
    ]
    ent.desc = 'Entity description, extending beyond 1000 characters: ' + ', '.join(map(str, range(500))) + '. Done!'
    ent.keyvalues['test_kv'] = {frozenset(): KeyValues(
        'test_kv',
        ValueTypes.COLOR_255,
        'A test keyvalue',
        '255 255 128',
        'Help text for a keyvalue',
    )}

    # The two special types with value lists.
    ent.keyvalues['spawnflags'] = {frozenset(): KeyValues(
        'spawnflags',
        ValueTypes.SPAWNFLAGS,
        'Flags',
        val_list=[
            (1, 'A', False, frozenset()),
            (2, 'B', True, frozenset()),
            (4, 'C', False, frozenset()),
            (8, 'D value', False, frozenset({'OLD', '!GOOD'})),
            (8, 'E', True, frozenset({'NEW'})),
        ],
    )}

    ent.keyvalues['multichoice'] = {frozenset(): KeyValues(
        'multichoice',
        ValueTypes.CHOICES,
        'Multiple Choice',
        val_list=[
            ('-1', 'Loss', frozenset()),
            ('0', 'Draw', frozenset()),
            ('1', 'Win', frozenset()),
            ('bad', 'Very Bad', frozenset({'NEW'})),
        ],
    )}

    # Test exporting with blank defaults and description.
    ent.keyvalues['test_both'] = {frozenset(): KeyValues(
        'test_both',
        ValueTypes.STRING,
        'Both default and desc',
        'defaulted',
        'A description',
    )}
    ent.keyvalues['test_desc'] = {frozenset(): KeyValues(
        'test_desc',
        ValueTypes.STRING,
        'just desc',
        desc='A description',
    )}
    ent.keyvalues['test_def'] = {frozenset(): KeyValues(
        'test_def',
        ValueTypes.STRING,
        'Just default',
        default='defaulted',
    )}
    ent.keyvalues['test_neither'] = {frozenset(): KeyValues(
        'test_neither',
        ValueTypes.STRING,
        'Neither',
    )}
    # Special case, boolean must supply default.
    ent.keyvalues['test_bool_neither'] = {frozenset(): KeyValues(
        'test_bool_neither',
        ValueTypes.BOOL,
        'Neither',
    )}
    ent.keyvalues['test_bool_desc'] = {frozenset(): KeyValues(
        'test_bool_desc',
        ValueTypes.BOOL,
        'Neither',
        desc='A description',
    )}

    ent.inputs['Enable'] = {frozenset(): IODef('Enable')}
    ent.inputs['SetSkin'] = {frozenset({'since_L4D'}): IODef('SetSkin', ValueTypes.INT, 'Set the skin.')}

    ent.outputs['OnNoDesc'] = {frozenset(): IODef('OnNoDesc', ValueTypes.VOID)}
    ent.outputs['OnSomething'] = {frozenset({'alpha'}): IODef('OnSomething', ValueTypes.VOID)}

    # IO special case, boolean value type is named differently.
    ent.outputs['OnGetValue'] = {frozenset(): IODef(
        'OnGetValue',
        ValueTypes.BOOL,
        'Get some value',
    )}

    buf = io.StringIO()
    fgd.export(buf)
    file_regression.check(buf.getvalue(), extension='.fgd')


@pytest.mark.parametrize('func', [
    KeyValues.copy, copy.copy, copy.deepcopy,
], ids=['method', 'copy', 'deepcopy'])
def test_kv_copy(func: Callable[[KeyValues], KeyValues]) -> None:
    """Test copying of keyvalues objects."""
    test_kv = KeyValues(
        name='some_key',
        type=ValueTypes.TARG_DEST,
        disp_name='Some Key',
        default='!player',
        desc='Does something',
        readonly=False,
        reportable=True,
    )
    duplicate = func(test_kv)
    assert duplicate.name == test_kv.name
    assert duplicate.type is test_kv.type
    assert duplicate.disp_name == test_kv.disp_name
    assert duplicate.default == test_kv.default
    assert duplicate.desc == test_kv.desc
    assert duplicate.val_list is None
    assert not duplicate.readonly
    assert duplicate.reportable

    test_kv = KeyValues(
        name='another_key',
        type=ValueTypes.CHOICES,
        disp_name='Another Key',
        default='45',
        desc='Does something else',
        readonly=True,
        reportable=False,
        val_list=[
            ('43', 'Fourty-Three', frozenset()),
            ('44', 'Fourty-Four', frozenset(['a', 'b'])),
            ('45', 'Fourty-Five', frozenset('-engine')),
        ]
    )
    duplicate = func(test_kv)
    assert duplicate.type is test_kv.type
    assert duplicate.val_list is not test_kv.val_list
    assert duplicate.val_list == test_kv.val_list
    assert duplicate.readonly
    assert not duplicate.reportable


@pytest.mark.parametrize('func', [
    IODef.copy, copy.copy, copy.deepcopy,
], ids=['method', 'copy', 'deepcopy'])
def test_io_copy(func: Callable[[IODef], IODef]) -> None:
    """Test copying of IODef objects."""
    test_io = IODef(
        name='OnWhatever',
        type=ValueTypes.VOID,
        desc='Does something',
    )
    duplicate = func(test_io)
    assert duplicate.name == 'OnWhatever'
    assert duplicate.type is ValueTypes.VOID
    assert duplicate.desc == 'Does something'
