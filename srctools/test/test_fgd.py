"""Test the FGD module."""
from srctools.filesys import VirtualFileSystem
from srctools.fgd import *
import pytest


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
line(240 180 50, targetname, target)
autovis(Auto, some, group)
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
        HelperLine(240, 180, 50, 'targetname', 'target'),
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
