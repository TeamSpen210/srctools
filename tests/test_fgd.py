"""Test the FGD module."""
from typing import Callable
import copy
import io

import pytest

from srctools import Vec
from srctools.filesys import VirtualFileSystem


with pytest.deprecated_call(match=r'srctools\.fgd\.KeyValues is renamed to srctools\.fgd\.KVDef'):
    from srctools.fgd import *
    from srctools.fgd import Snippet


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

    assert ent.kv['keyvalue1', set()] == KVDef(
        'keyvalue1',
        ValueTypes.STRING,
        'Name',
        'default',
        'documentation',
    )
    assert ent.kv['keyvalue1', {'tag'}] == KVDef(
        'keyvalue1',
        ValueTypes.BOOL,
        'Tagged Name',
        '0',
        '',
    )
    assert ent.kv['keyvalue2'] == KVDef(
        'keyvalue2',
        ValueTypes.INT,
        'Hi',
        '4',
        '',
    )

    assert ent.kv['spawnflags'] == KVDef(
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

    assert ent.kv['choicelist'] == KVDef(
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


def test_snippet_desc() -> None:
    """Test snippet descriptions."""
    fgd = FGD()
    fsys = VirtualFileSystem({
        'snippets.fgd': """\
@snippet desc first_Desc = "Some text." +
    " Another line of description.\\n" +
    "And another."
@snippet description Another = "Some description"
@snippet description EntDesc = "This is an entity that does things."
""",
        'overlap.fgd': """\
@snippet desc first_desc = "Different text"
""",
        'ent_def.fgd': """\

@PointClass = test_entity: "First line. " + 
    #snippet EntDesc +
    " Last line."
    [
    keyvalue(string): "Has desc" : "..." : #snippet "anOther"
    input SomeInput(void): #snippet first_desc
    ]
"""
    })
    fgd.parse_file(fsys, fsys['snippets.fgd'])
    with pytest.raises(ValueError, match="^Two description snippets were defined"):
        fgd.parse_file(fsys, fsys['overlap.fgd'])
    assert fgd.snippet_desc == {
        'first_desc': Snippet(
            'first_Desc', 'snippets.fgd', 1,
            'Some text. Another line of description.\nAnd another.',
        ),
        'another': Snippet(
            'Another', 'snippets.fgd', 4,
            'Some description',
        ),
        'entdesc': Snippet(
            'EntDesc', 'snippets.fgd', 5,
            'This is an entity that does things.',
        ),
    }
    fgd.parse_file(fsys, fsys['ent_def.fgd'])

    ent = fgd['test_entity']
    assert ent.desc == 'First line. This is an entity that does things. Last line.'
    assert ent.kv['keyvalue'] == KVDef(
        name="keyvalue",
        disp_name="Has desc",
        default="...",
        type=ValueTypes.STRING,
        desc="Some description",
    )
    assert ent.inp['SomeInput'] == IODef(
        name="SomeInput",
        type=ValueTypes.VOID,
        desc='Some text. Another line of description.\nAnd another.',
    )


def test_snippet_choices() -> None:
    """Test parsing snippet choices."""
    fgd = FGD()
    fsys = VirtualFileSystem({'snippets.fgd': """\

@snippet choices TRInary = [
    -1: "EOF" [+srctools]
    0: "No"
    1: "Yes"
]

@PointClass = test_ent [
    keyvalue(choices): "KeyValue" : -1 : "desc" = #snippet trinary
]
    """})
    fgd.parse_file(fsys, fsys['snippets.fgd'])
    choices = [
        ('-1', 'EOF', frozenset({'+SRCTOOLS'})),
        ('0', 'No', frozenset()),
        ('1', 'Yes', frozenset()),
    ]
    assert fgd.snippet_choices == {
        'trinary': Snippet('TRInary', 'snippets.fgd', 2, choices)
    }
    kv = fgd['test_ent'].kv['keyvalue']
    assert kv == KVDef(
        name="keyvalue",
        disp_name="KeyValue",
        default="-1",
        type=ValueTypes.CHOICES,
        desc="desc",
        val_list=choices,
    )
    # It shouldn't be a shared list!
    assert kv.val_list is not fgd.snippet_choices['trinary'].value


def test_snippet_spawnflags() -> None:
    """Test parsing snippet spawnflags."""
    fgd = FGD()
    fsys = VirtualFileSystem({'snippets.fgd': """\

    @snippet flags Trigger = [
        1: "Clients (Players/Bots)" : 1 [TF2, CSGO, CSS, MESA]
        1: "Clients (Players)" : 1 [!TF2, !CSGO, !CSS, !MESA]
        2: "NPCs" : 0 [!ASW]
        2: "Marines and Aliens" : 0 [ASW]
        4: "func_pushable" : 0
        8: "VPhysics Objects" : 0
        8192: "Items (weapons, items, projectiles)" : 0 [MBase]
    ]

@PointClass = test_ent [
    spawnflags(flags) = [
        #snippet Trigger
        16: "Special Stuff": 1
    ]
]
    """})
    fgd.parse_file(fsys, fsys['snippets.fgd'])

    spawnflags = [
        (1, 'Clients (Players/Bots)', True, frozenset({'TF2', 'CSGO', 'CSS', 'MESA'})),
        (1, 'Clients (Players)', True, frozenset({'!TF2', '!CSGO', '!CSS', '!MESA'})),
        (2, 'NPCs', False, frozenset({'!ASW'})),
        (2, 'Marines and Aliens', False, frozenset({'ASW'})),
        (4, 'func_pushable', False, frozenset()),
        (8, 'VPhysics Objects', False, frozenset()),
        (8192, 'Items (weapons, items, projectiles)', False, frozenset({'MBASE'})),
    ]

    assert fgd.snippet_flags == {
        'trigger': Snippet('Trigger', 'snippets.fgd', 2, spawnflags)
    }
    kv = fgd['test_ent'].kv['spawnflags']
    assert kv == KVDef(
        name="spawnflags",
        disp_name="spawnflags",
        type=ValueTypes.SPAWNFLAGS,
        val_list=[
            *spawnflags,
            (16, "Special Stuff", True, frozenset()),
        ]
    )
    # It shouldn't be a shared list!
    assert kv.val_list is not fgd.snippet_flags['trigger'].value


def test_snippet_keyvalues() -> None:
    """Test parsing snippet keyvalues."""
    fgd = FGD()
    fsys = VirtualFileSystem({'snippets.fgd': """\

    @snippet keyvalue InvStartEnabled = start_enabled[-engine](choices) : "Start Enabled" : 0 : "Start it." = [
        0: "Yes"
        1: "No"
    ]
    """})
    fgd.parse_file(fsys, fsys['snippets.fgd'])
    assert fgd.snippet_keyvalue == {
        'invstartenabled': Snippet(
            'InvStartEnabled', 'snippets.fgd', 2,
            (frozenset(['-ENGINE']), KVDef(
                'start_enabled', ValueTypes.CHOICES,
                disp_name='Start Enabled',
                default='0',
                desc='Start it.',
                val_list=[
                    ('0', 'Yes', frozenset()),
                    ('1', 'No', frozenset()),
                ]
            ))
        )
    }


def test_snippet_io() -> None:
    """Test parsing snippet i/o."""
    fgd = FGD()
    fsys = VirtualFileSystem({'snippets.fgd': """\

    @snippet input uSer1 = FireUser1[+tag](void) : "Causes this entity's OnUser1 output to be fired." 
    @snippet output uSer1 = OnUser1[-tag](void) : "Fired in response to FireUser1 input."
    """})
    fgd.parse_file(fsys, fsys['snippets.fgd'])
    assert fgd.snippet_input == {
        'user1': Snippet(
            'uSer1', 'snippets.fgd', 2,
            (frozenset(['+TAG']), IODef(
                name='FireUser1',
                type=ValueTypes.VOID,
                desc="Causes this entity's OnUser1 output to be fired.",
            ))
        )
    }
    assert fgd.snippet_output == {
        'user1': Snippet(
            'uSer1', 'snippets.fgd', 3,
            (frozenset(['-TAG']), IODef(
                name='OnUser1',
                type=ValueTypes.VOID,
                desc="Fired in response to FireUser1 input.",
            ))
        )
    }


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
    base_origin.keyvalues['origin'] = {frozenset(): KVDef(
        'origin',
        ValueTypes.VEC_ORIGIN,
        'Origin',
        '0 0 0',
    )}
    base_angles.keyvalues['angles'] = {frozenset(): KVDef(
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
    ent.keyvalues['test_kv'] = {frozenset(): KVDef(
        'test_kv',
        ValueTypes.COLOR_255,
        'A test keyvalue',
        '255 255 128',
        'Help text for a keyvalue',
    )}

    # The two special types with value lists.
    ent.keyvalues['spawnflags'] = {frozenset(): KVDef(
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

    ent.keyvalues['multichoice'] = {frozenset(): KVDef(
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
    ent.keyvalues['test_both'] = {frozenset(): KVDef(
        'test_both',
        ValueTypes.STRING,
        'Both default and desc',
        'defaulted',
        'A description',
    )}
    ent.keyvalues['test_desc'] = {frozenset(): KVDef(
        'test_desc',
        ValueTypes.STRING,
        'just desc',
        desc='A description',
    )}
    ent.keyvalues['test_def'] = {frozenset(): KVDef(
        'test_def',
        ValueTypes.STRING,
        'Just default',
        default='defaulted',
    )}
    ent.keyvalues['test_neither'] = {frozenset(): KVDef(
        'test_neither',
        ValueTypes.STRING,
        'Neither',
    )}
    # Special case, boolean must supply default.
    ent.keyvalues['test_bool_neither'] = {frozenset(): KVDef(
        'test_bool_neither',
        ValueTypes.BOOL,
        'Neither',
    )}
    ent.keyvalues['test_bool_desc'] = {frozenset(): KVDef(
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
    KVDef.copy, copy.copy, copy.deepcopy,
], ids=['method', 'copy', 'deepcopy'])
def test_kv_copy(func: Callable[[KVDef], KVDef]) -> None:
    """Test copying of keyvalues objects."""
    test_kv = KVDef(
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

    test_kv = KVDef(
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
