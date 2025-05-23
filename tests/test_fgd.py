"""Test the FGD module."""
from typing import Any
from collections.abc import Callable, Generator
import copy
import io
import re

from pytest_regressions.file_regression import FileRegressionFixture
import pytest

from srctools import fgd as fgd_mod
from srctools.const import FileType
from srctools.math import Vec
from srctools.fgd import (
    FGD, AutoVisgroup, EntityDef, EntityTypes, FGDParseError, HelperExtAppliesTo,
    HelperHalfGridSnap, HelperLine, HelperModel, HelperSize, HelperSphere, IODef, KVDef,
    Resource, Snippet, UnknownHelper, ValueTypes,
)
from srctools.filesys import VirtualFileSystem
# noinspection PyProtectedMember
from srctools.tokenizer import Cy_Tokenizer, Py_Tokenizer, TokenSyntaxError


if Cy_Tokenizer is not None:
    parms = [Cy_Tokenizer, Py_Tokenizer]
    ids = ['Cython tokenizer', 'Python tokenizer']
else:
    pytest.fail('No _tokenizer')
    parms = [Py_Tokenizer]
    ids = ['Python tokenizer']


@pytest.fixture(params=parms, ids=ids)
def py_c_token(request: Any) -> Generator[None, None, None]:
    """Run the test twice, for the Python and C versions of Tokenizer."""
    orig_tok = fgd_mod.Tokenizer
    try:
        fgd_mod.Tokenizer = request.param
        yield None
    finally:
        fgd_mod.Tokenizer = orig_tok


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


def test_deprecated_kvdef() -> None:
    """Test that this produces a warning when imported."""
    with pytest.deprecated_call(match=r'srctools\.fgd\.KeyValues is renamed to srctools\.fgd\.KVDef'):
        from srctools.fgd import KeyValues  # noqa


def test_entity_lookup_keyvalues() -> None:
    """Test the shorthand KV lookup functionality."""
    kv_a = KVDef('keyvalue1', ValueTypes.STRING, 'Name')
    kv_b = KVDef('keyvalue1', ValueTypes.STRING, 'Tag 1A')
    kv_c = KVDef('keyvalue1', ValueTypes.STRING, 'Tag 2A')
    kv_a2 = KVDef('keyvalue1', ValueTypes.STRING, 'Name 2')
    kv_b2 = KVDef('keyvalue1', ValueTypes.STRING, 'Tag 1B')
    kv_c2 = KVDef('keyvalue1', ValueTypes.STRING, 'Tag 2B')

    ent = EntityDef(EntityTypes.POINT)
    ent.keyvalues['keyvalue1'] = {
        frozenset(): kv_a,
        frozenset({'TAG1'}): kv_b,
        frozenset({'TAG1', '-TAG2'}): kv_c,
    }
    assert ent.kv['keyvalue1'] is kv_a
    assert ent.kv['keyvALUe1', ()] is kv_a
    assert ent.kv['keyvalue1', []] is kv_a
    assert ent.kv['kEYvalue1', ['tag1']] is kv_c  # Longer, takes precedence.
    assert ent.kv['kEYvalue1', 'tag1'] is kv_c  # Edge case, don't iter direct strings.
    assert ent.kv['keyvalue1', ['tag1', 'tag2']] is kv_b

    with pytest.raises(KeyError):
        # noinspection PyStatementEffect
        ent.kv['missing']

    # Test setting values.
    ent.kv['keyvalue1'] = kv_a2
    assert len(ent.keyvalues['keyvalue1']) == 3
    assert ent.keyvalues['keyvalue1'][frozenset()] is kv_a2
    ent.kv['keyvalue1', 'tag80'] = kv_b2
    assert ent.keyvalues['keyvalue1'][frozenset({'TAG80'})] is kv_b2
    ent.kv['keyvalue1', ('-tag2', 'tag1')] = kv_c2
    assert ent.keyvalues['keyvalue1'][frozenset({'TAG1', '-TAG2'})] is kv_c2

    ent.kv['keyvalue1'] = kv_a
    ent.kv['keyvalue1', ('-exclude', )] = kv_a2
    del ent.kv['keyvalue1', ()]  # Removes only base
    with pytest.raises(KeyError):
        # Untagged ignores the tags.
        # noinspection PyStatementEffect
        ent.kv['keyvalue1']
    with pytest.raises(KeyError):
        # noinspection PyStatementEffect
        ent.kv['keyvalue1', ('exclude', )]
    print(ent.keyvalues)

    assert ent.kv['keyvalue1', 'tag1'] is kv_c2  # Still present.
    # Empty tag means something else could match.
    assert ent.kv['keyvalue1', ()] is kv_a2
    del ent.kv['keyvalue1', 'tag1']
    with pytest.raises(KeyError):
        # noinspection PyStatementEffect
        ent.kv['keyvalue1', ('tag1', 'tag2', 'exclude')]
    # Doesn't remove non-equal but matching
    assert ent.keyvalues['keyvalue1'][frozenset({'TAG1', '-TAG2'})] is kv_c2
    del ent.kv['keyvalue1']  # Removes the whole thing.
    assert 'keyvalue1' not in ent.keyvalues


def test_entity_lookup_io() -> None:
    """Test the shorthand IO lookup functionality.

    Implementation is shared with KVs, so we just need a smoke test for IO.
    """
    ent = EntityDef(EntityTypes.POINT)
    ent.inputs['trigger'] = {
        frozenset(): IODef('Trigger', ValueTypes.VOID),
        frozenset({'pass'}): IODef('Trigger', ValueTypes.STRING),
    }
    assert ent.inp['TrIgger'].type is ValueTypes.VOID
    assert ent.inp['TriGGer', 'pass'].type is ValueTypes.STRING
    ent.outputs['ontrigger'] = {
        frozenset(): IODef('OnTrigger', ValueTypes.VOID),
        frozenset({'pass'}): IODef('OnTrigger', ValueTypes.STRING),
    }
    assert ent.out['onTrIgger'].type is ValueTypes.VOID
    assert ent.out['onTrIgger', ('pass', )].type is ValueTypes.STRING


def test_entity_parse(py_c_token) -> None:
    """Verify parsing an entity produces the correct results."""
    fsys = VirtualFileSystem({'test.fgd': """

@PointClass base(Base1, Base2, Base3) 
base(Base4, base_5)
sphere(radii) 
unknown[+a, -b](a, b, c)
line[+b](240 180 50, targetname, target)
autovis(Auto, some, group)
halfgridsnap // Special, no args
appliesto(tag1, tag2, !tag3)
= some_entity: "The description for this prop, which is spread over " 
+ "multiple lines."
    [
    keyvalue1(string): "Name" : "default": "documentation"
    keyVAlue2(int): "Hi": 4
    keyvalue1[tag](boolean): "Tagged Name": 0
    target(target_destination): "An ent"
    
    spawnflags(flags) : "Flags" = [
        2^0 : "A" : 0
        ^1 : "[2]  B" : 1
        4 : "[48] C" : 0
        8 : "D value" : 0 [old, !good]
        8 : "E" : 1 [new]
        ^16 : "16bit" : 0
    ]
    
    input Trigger(void): "Trigger the entity."
    output OnTrigger(void): "Handle triggering."
    output OnTrigger[adv](float): "Handle triggering, with value. " + 
        "Second line."
    
    choicelist(choices) : "A Choice" : 0 : "Blahdy blah. "
    + "Another line." = 
        [
        0: "First"
        1: "Second" [new]
        1: "Old second" [-oLd]
        2: "Third"
        "four": "Fourth"
        ]
    ]
@SolidClass = multiline: "Another description "
 + "with the plus at the start" 
    [
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
        UnknownHelper('unknown', ['a', 'b', 'c'], tags=frozenset({'+A', '-B'})),
        HelperLine(240, 180, 50, 'targetname', 'target', tags=frozenset({'+B'})),
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
        'keyVAlue2',
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
            (4, '[48] C', False, frozenset()),
            (8, 'D value', False, frozenset({'OLD', '!GOOD'})),
            (8, 'E', True, frozenset({'NEW'})),
            (2**16, '16bit', False, frozenset()),
        ],
    )

    assert ent.kv['choicelist'] == KVDef(
        'choicelist',
        ValueTypes.CHOICES,
        'A Choice',
        '0',
        'Blahdy blah. Another line.',
        val_list=[
            ('0', 'First', frozenset()),
            ('1', 'Second', frozenset({'NEW'})),
            ('1', 'Old second', frozenset({'-OLD'})),
            ('2', 'Third', frozenset()),
            ('four', 'Fourth', frozenset()),
        ],
    )

    assert ent.inp['trigger'] == IODef(
        'Trigger',
        ValueTypes.VOID,
        'Trigger the entity.'
    )
    assert ent.out['onTrigger', set()] == IODef(
        'OnTrigger',
        ValueTypes.VOID,
        'Handle triggering.'
    )
    assert ent.out['onTrigger', {'adv'}] == IODef(
        'OnTrigger',
        ValueTypes.FLOAT,
        'Handle triggering, with value. Second line.'
    )

    assert fgd['multiline'].desc == 'Another description with the plus at the start'


def test_entity_extend(py_c_token) -> None:
    """Verify handling ExtendClass."""
    fsys = VirtualFileSystem({'test.fgd': """

@PointClass base(Base1, Base2, Base3)
sphere(radii)
= some_entity: "Some description" 
    [
    keyvalue1(string): "Name" : "default": "documentation"
    keyVAlue2(int): "Hi": 4
    input Trigger(void): "Trigger the entity."
    output OnTrigger(void): "Handle triggering."

    ]

@ExtendClass base(Base4, base_5)
unknown(a, b, c)
= some_entity: "New description" 
    [
    keyvalue3(string): "Name3" : "default3": "documentation3"
    input Trigger2(void): "Trigger the entity."
    output OnTrigger2(void): "Handle triggering."
    ]

"""})
    fgd = FGD()
    fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False, eval_extensions=True)
    ent = fgd['Some_ENtity']
    assert ent.type is EntityTypes.POINT
    assert ent.classname == 'some_entity'
    assert ent.desc == 'New description'
    assert ent.bases == ['Base1', 'Base2', 'Base3', 'Base4', 'base_5']

    assert ent.helpers == [
        HelperSphere(255, 255, 255, 'radii'),
        UnknownHelper('unknown', ['a', 'b', 'c']),
    ]

    assert ent.kv['keyvalue1'] == KVDef(
        'keyvalue1',
        ValueTypes.STRING,
        'Name',
        'default',
        'documentation',
    )
    assert ent.kv['keyvalue2'] == KVDef(
        'keyVAlue2',
        ValueTypes.INT,
        'Hi',
        '4',
        '',
    )
    assert ent.kv['keyvalue3'] == KVDef(
        'keyvalue3',
        ValueTypes.STRING,
        'Name3',
        'default3',
        'documentation3',
    )

    assert ent.inp['trigger'] == IODef(
        'Trigger',
        ValueTypes.VOID,
        'Trigger the entity.'
    )
    assert ent.out['onTrigger'] == IODef(
        'OnTrigger',
        ValueTypes.VOID,
        'Handle triggering.'
    )

    assert ent.inp['trigger2'] == IODef(
        'Trigger2',
        ValueTypes.VOID,
        'Trigger the entity.'
    )
    assert ent.out['onTrigger2'] == IODef(
        'OnTrigger2',
        ValueTypes.VOID,
        'Handle triggering.'
    )


def test_entity_ignore_unknown_valuetype(py_c_token) -> None:
    """Verify handling unknown ValueTypes."""
    fsys = VirtualFileSystem({'test.fgd': """

@PointClass
= some_entity
    [
    keyvalue1(super_string): "Name" : "default": "documentation"
    input Trigger(ultra_void): "Trigger the entity."
    output OnTrigger(int1024): "Handle triggering."
    ]
"""})

    # Make sure this still counts as an error normally
    fgd = FGD()
    with pytest.raises(TokenSyntaxError):
        fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False, eval_extensions=True, ignore_unknown_valuetype=False)

    # Parse it ignoring any unknowns
    fgd = FGD()
    with pytest.warns(SyntaxWarning):
        fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False, eval_extensions=True, ignore_unknown_valuetype=True)

    ent = fgd['some_entity']
    assert ent.type is EntityTypes.POINT
    assert ent.classname == 'some_entity'

    kv = ent.kv['keyvalue1']
    inp = ent.inp['Trigger']
    out = ent.out['OnTrigger']

    assert kv == KVDef(
        'keyvalue1',
        'super_string',
        'Name',
        'default',
        'documentation',
    )
    assert kv.type is ValueTypes.STRING
    assert kv.custom_type == 'super_string'

    assert inp == IODef(
        'Trigger',
        'ultra_void',
        'Trigger the entity.'
    )
    assert inp.type is ValueTypes.STRING
    assert inp.custom_type == 'ultra_void'

    assert out == IODef(
        'OnTrigger',
        'int1024',
        'Handle triggering.'
    )
    assert out.type is ValueTypes.STRING
    assert out.custom_type == 'int1024'


@pytest.mark.parametrize('code, is_readonly, is_report, editor_only', [
    ('(int): "None"', False, False, False),
    ('(int) readonly: "Readonly"', True, False, False),
    ('(*int): "Old Report"', False, True, False),
    ('(int) report: "New Report"', False, True, False),
    ('(*int) readonly: "Both old-style"', True, True, False),
    ('(int) readonly report: "Both new-style"', True, True, False),
    ('(*int) report: "Redundant"', False, True, False),
    ('(*int) readonly report: "Redundant + readonly"', True, True, False),
    # Following aren't allowed by Valve.
    ('(int) report readonly: "Extension - reverse order "', True, True, False),
    ('(int) editor: "Editor only"', False, False, True),
    ('(int) editor report: "Editor + report"', False, True, True),
    ('(int) report editor: "Editor + report"', False, True, True),
    ('(*int) editor: "Editor + report"', False, True, True),
    ('(int) editor readonly: "Editor + Readonly"', True, False, True),
    ('(int) readonly editor: "Editor + Readonly"', True, False, True),
    ('(int) editor report readonly: "All"', True, True, True),
    ('(int) readonly editor report: "All"', True, True, True),
])
def test_parse_kv_flags(
    py_c_token, code: str,
    is_readonly: bool, is_report: bool, editor_only: bool,
) -> None:
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
    assert kv.editor_only is editor_only, kv


def test_parse_helpers_simple(py_c_token, file_regression: FileRegressionFixture) -> None:
    """Test parsing simple helper types."""
    fsys = VirtualFileSystem({'test.fgd': """@PointClass
    base(Base1, Base2, Base3)
    halfgridsnap // Special, no args
    size(32 64 128)
    size[+A](-64 -48 0, 64 48 32)
    color[+A](128 255 0)
    sphere()
    sphere[+A](radii)
    sphere[+A](radii, 128 192 64)
    line(240 180 50, targetname, target)
    line[+A](12 180 50, targetname target, parentname, next)
    frustum()
    frustum[+A](light_fov)
    frustum(light_fov, light_near)
    frustum(light_fov, 450, light_far)
    frustum(97, light_near, light_far, bright)
    frustum(light_fov, light_near, 3200, bright, +6.78)
    frustum(light_fov, light_near, light_far, 128 64 0)
    frustum(light_fov, light_near, light_far, 32, 96, 255, -2.5)
    cylinder(24 48 96, targetname, start)
    cylinder(24 48 96, targetname, start start_size)
    cylinder(24 48 96, targetname, start start_size, target, end)
    cylinder(24 48 96, targetname, start start_size, target, end, end_size)
    origin[+A](centroid)
    axis  // No args, no tags in mid
    axis[+A](hinge)
    vecline[+A]("axis")
    sidelist[+A](decalled)
    wirebox[+A](mins maxes)
    sweptplayerhull[+A]()
    obb[+A](orient_mins, orient_maxes)
    sphere  // No args, no tags at end
    = some_entity [ ]
    """})
    fgd = FGD()
    fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False)
    ent = fgd['some_entity']
    tags = frozenset({'+A'})
    assert ent.helpers == [
        fgd_mod.HelperHalfGridSnap(),
        fgd_mod.HelperSize(Vec(-16, -32, -64), Vec(16, 32, 64)),
        fgd_mod.HelperSize(Vec(-64, -48, 0), Vec(64, 48, 32), tags=tags),
        fgd_mod.HelperRenderColor(128, 255, 0, tags=tags),
        fgd_mod.HelperSphere(255, 255, 255, "radius"),
        fgd_mod.HelperSphere(255, 255, 255, "radii", tags=tags),
        fgd_mod.HelperSphere(128, 192, 64, "radii", tags=tags),
        fgd_mod.HelperLine(240, 180, 50, "targetname", "target", None, None),
        fgd_mod.HelperLine(12, 180, 50, "targetname", "target", "parentname", "next", tags=tags),
        fgd_mod.HelperFrustum("_fov", "_nearplane", "_farplane", "_light", 1.0),
        fgd_mod.HelperFrustum("light_fov", "_nearplane", "_farplane", "_light", 1.0, tags=tags),
        fgd_mod.HelperFrustum("light_fov", "light_near", "_farplane", "_light", 1.0),
        fgd_mod.HelperFrustum("light_fov", 450, "light_far", "_light", 1.0),
        fgd_mod.HelperFrustum(97, "light_near", "light_far", "bright", 1.0),
        fgd_mod.HelperFrustum("light_fov", "light_near", 3200, "bright", 6.78),
        fgd_mod.HelperFrustum("light_fov", "light_near", "light_far", (128, 64, 0), 1.0),
        fgd_mod.HelperFrustum("light_fov", "light_near", "light_far", (32, 96, 255), -2.5),
        fgd_mod.HelperCylinder(24, 48, 96, "targetname", "start"),
        fgd_mod.HelperCylinder(24, 48, 96, "targetname", "start", start_radius="start_size"),
        fgd_mod.HelperCylinder(24, 48, 96, "targetname", "start", "target", "end", "start_size"),
        fgd_mod.HelperCylinder(24, 48, 96, "targetname", "start", "target", "end", "start_size", "end_size"),
        fgd_mod.HelperOrigin("centroid", tags=tags),
        fgd_mod.HelperAxis("axis"),
        fgd_mod.HelperAxis("hinge", tags=tags),
        fgd_mod.HelperVecLine("axis", tags=tags),
        fgd_mod.HelperBrushSides("decalled", tags=tags),
        fgd_mod.HelperBoundingBox("mins", "maxes", tags=tags),
        fgd_mod.HelperSweptPlayerHull(tags=tags),
        fgd_mod.HelperOrientedBBox("orient_mins", "orient_maxes", tags=tags),
        fgd_mod.HelperSphere(255, 255, 255, "radius"),
    ]
    file_regression.check(fgd.export(), basename="parse_helpers_simple", extension='.fgd')


def test_parse_helpers_icon(py_c_token, file_regression: FileRegressionFixture) -> None:
    """Test parsing sprite/model helper types."""
    fsys = VirtualFileSystem({'test.fgd': """@PointClass
    sprite[+B]()
    sprite[+C](editor/env_fire)
    iconsprite[+B]()
    iconsprite[+C]("editor/env_blood")
    studio[+B]  // No args in the middle
    studio[+C](models/error.mdl)
    studioprop[+B]()
    studioprop[+C]("models/props/metal_box.mdl")
    lightprop[+C]("models/editor/spotlight.mdl")
    lightprop[+B]  // No args but tags at the end.
    = some_entity [ ]
    """})
    fgd = FGD()
    fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False)
    ent = fgd['some_entity']
    tag_b = frozenset({'+B'})
    tag_c = frozenset({'+C'})
    assert ent.helpers == [
        fgd_mod.HelperEnvSprite(None, tags=tag_b),
        fgd_mod.HelperEnvSprite("editor/env_fire", tags=tag_c),
        fgd_mod.HelperSprite(None, tags=tag_b),
        fgd_mod.HelperSprite("editor/env_blood", tags=tag_c),
        fgd_mod.HelperModel(None, tags=tag_b),
        fgd_mod.HelperModel("models/error.mdl", tags=tag_c),
        fgd_mod.HelperModelProp(None, tags=tag_b),
        fgd_mod.HelperModelProp("models/props/metal_box.mdl", tags=tag_c),
        fgd_mod.HelperModelLight("models/editor/spotlight.mdl", tags=tag_c),
        fgd_mod.HelperModelLight(None, tags=tag_b),
    ]
    file_regression.check(fgd.export(), basename="parse_helpers_icon", extension='.fgd')


def test_parse_helpers_special(py_c_token, file_regression: FileRegressionFixture) -> None:
    """Test parsing specialised helper types."""
    fsys = VirtualFileSystem({'test.fgd': """@PointClass
    instance[+D]()
    decal[+D]()
    overlay[+D]()
    overlay_transition[+D]()
    light[+D]()
    lightcone[+D]()
    lightcone(inner, outer, color, -1.5) 
    keyframe[+D]()
    keyframe[+D](revo_key)
    animator[+D]()
    quadbounds[+D]()
    worldtext[+D]()
    catapult[+D]()
    = some_entity [ ]
    """})
    fgd = FGD()
    fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False)
    ent = fgd['some_entity']
    tag_c = frozenset({'+D'})
    assert ent.helpers == [
        fgd_mod.HelperInstance(tags=tag_c),
        fgd_mod.HelperDecal(tags=tag_c),
        fgd_mod.HelperOverlay(tags=tag_c),
        fgd_mod.HelperOverlayTransition(tags=tag_c),
        fgd_mod.HelperLight(tags=tag_c),
        fgd_mod.HelperLightSpot("_inner_cone", "_cone", "_light", 1.0, tags=tag_c),
        fgd_mod.HelperLightSpot("inner", "outer", "color", -1.5),
        fgd_mod.HelperRope(None, tags=tag_c),
        fgd_mod.HelperRope("revo_key", tags=tag_c),
        fgd_mod.HelperTrack(tags=tag_c),
        fgd_mod.HelperBreakableSurf(tags=tag_c),
        fgd_mod.HelperWorldText(tags=tag_c),
        fgd_mod.HelperExtCatapult(tags=tag_c),
    ]

    file_regression.check(fgd.export(), basename="parse_helpers_special", extension='.fgd')


def test_parse_helpers_extension(py_c_token, file_regression: FileRegressionFixture) -> None:
    """Test parsing extra helper types."""
    fsys = VirtualFileSystem({'test.fgd': """@PointClass
    appliesto(tag1, +tag2, -tag3, !tag4)
    orderby(a, c, b)
    autovis(Auto, some, group)
    autovis(Spaced Parent, spaced cHIld)
    unknown[-E](a+b, c  d 4+3-25, e)
    = some_entity [
        a(int) : "First"
        b(string) : "Third"
        c(string) : "Second"
    ]
    """})
    fgd = FGD()
    fgd.parse_file(fsys, fsys['test.fgd'], eval_bases=False)
    ent = fgd['some_entity']
    assert ent.helpers == [
        fgd_mod.HelperExtAppliesTo(["tag1", "+tag2", "-tag3", "!tag4"]),
        fgd_mod.HelperExtOrderBy(["a", "c", "b"]),
        fgd_mod.UnknownHelper("unknown", ["a+b", "c", "d", "4+3-25", "e"], tags=frozenset({'-E'})),
    ]
    assert fgd.auto_visgroups == {
        "some": fgd_mod.AutoVisgroup("some", "Auto", {"some_entity"}),
        "group": fgd_mod.AutoVisgroup("group", "some", {"some_entity"}),
        "spaced parent": fgd_mod.AutoVisgroup("Spaced Parent", "Auto", {"some_entity"}),
        "spaced child": fgd_mod.AutoVisgroup("spaced cHIld", "Spaced Parent", {"some_entity"}),
    }

    file_regression.check(fgd.export(), basename="parse_helpers_ext", extension='.fgd')


def test_illegal_helpers_size(py_c_token) -> None:
    """The size helper is special, and is parsed specially."""
    fsys = VirtualFileSystem({
        'no_args.fgd': """@PointClass
        size() 
        = illegal []
        """,
        'two_coords.fgd': """@PointClass
        size(1 2) 
        = illegal []
        """,
        'early_comma.fgd': """@PointClass
        size(1, 2 3, 4 5 6) 
        = illegal []
        """,
        'no_comma.fgd': """@PointClass
        size(1 2 3 4 5 6) 
        = illegal []
        """,
        'extra_coords.fgd': """@PointClass
        size(1 2 3, 4 5 6, 7 8 9) 
        = illegal []
        """,
    })
    for file in fsys:
        with pytest.raises(TokenSyntaxError):
            FGD().parse_file(fsys, file)


def test_illegal_helpers_tags(py_c_token) -> None:
    """Most extension helpers don't allow tags."""
    fsys = VirtualFileSystem({
        'early_tag.fgd': """@PointClass
        [a, b]sphere()
        = illegal []
        """,
        'double_tag.fgd': """@PointClass
        sphere[a][b]()
        = illegal []
        """,
        'appliesto.fgd': """@PointClass
        appliesto[a](tag1, +tag2, -tag3, !tag4)
        = illegal []
        """,
        'orderby.fgd': """@PointClass
        orderby[b](a, c, b)
        = illegal []
        """,
        'autovis.fgd': """@PointClass
        autovis[c](Auto, some, group)
        = illegal []
        """,
    })
    for file in fsys:
        with pytest.raises(TokenSyntaxError):
            FGD().parse_file(fsys, file)


def test_snippet_desc(py_c_token) -> None:
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
    with pytest.raises(ValueError, match=r"^Two description snippets were defined"):
        fgd.parse_file(fsys, fsys['overlap.fgd'])
    assert fgd.snippet_desc == {
        'first_desc': [Snippet(
            'first_Desc', 'snippets.fgd', 1,
            'Some text. Another line of description.\nAnd another.',
        )],
        'another': [Snippet(
            'Another', 'snippets.fgd', 4,
            'Some description',
        )],
        'entdesc': [Snippet(
            'EntDesc', 'snippets.fgd', 5,
            'This is an entity that does things.',
        )],
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


def test_snippet_choices(py_c_token) -> None:
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
        'trinary': [Snippet('TRInary', 'snippets.fgd', 2, choices)]
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
    [snip] = fgd.snippet_choices['trinary']
    assert kv.val_list is not snip.value


def test_snippet_spawnflags(py_c_token) -> None:
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
        'trigger': [Snippet('Trigger', 'snippets.fgd', 2, spawnflags)]
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
    [snip] = fgd.snippet_flags['trigger']
    assert kv.val_list is not snip.value


def test_snippet_keyvalues(py_c_token) -> None:
    """Test parsing snippet keyvalues."""
    fgd = FGD()
    fsys = VirtualFileSystem({'snippets.fgd': """\

    @snippet keyvalue InvStartEnabled = start_enabled[-engine](choices) : "Start Enabled" : 0 : "Start it." = [
        0: "Yes"
        1: "No"
    ]
    
    @snippet keyvalue KVBlock = [
        start_open(boolean) : "Start Open": 1
        height(int) : "Height" : 48
    ]
    
    @PointClass = some_ent [
        #snippet keyvalue InvStartEnabled
    ]
    """})
    fgd.parse_file(fsys, fsys['snippets.fgd'])
    assert fgd.snippet_keyvalue == {
        'invstartenabled': [Snippet(
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
        )],
        'kvblock': [
            Snippet(
                'KVBlock', 'snippets.fgd', 8,
                (frozenset(), KVDef(
                    'start_open', ValueTypes.BOOL,
                    disp_name="Start Open", default='1'
                ))
            ),
            Snippet(
                'KVBlock', 'snippets.fgd', 9,
                (frozenset(), KVDef(
                    'height', ValueTypes.INT,
                    disp_name="Height", default='48'
                ))
            ),
        ]
    }
    # Check it was included, but is not shared.
    snip_kv = fgd.snippet_keyvalue['invstartenabled'][0].value[1]
    ent_kv = fgd.entities['some_ent'].kv['start_enabled', {'-engine'}]
    assert ent_kv == snip_kv
    assert ent_kv is not snip_kv
    assert ent_kv.val_list is not snip_kv.val_list


def test_snippet_io(py_c_token) -> None:
    """Test parsing snippet i/o."""
    fgd = FGD()
    fsys = VirtualFileSystem({'snippets.fgd': """\

    @snippet input uSer1 = FireUser1[+tag](void) : "Causes this entity's OnUser1 output to be fired." 
    @snippet output uSer1 = OnUser1[-tag](void) : "Fired in response to FireUser1 input."
    
    @PointClass = some_ent [
        #snippet input User1
        #snippet output User1
    ]
    """})
    fgd.parse_file(fsys, fsys['snippets.fgd'])
    assert fgd.snippet_input == {
        'user1': [Snippet(
            'uSer1', 'snippets.fgd', 2,
            (frozenset(['+TAG']), IODef(
                name='FireUser1',
                type=ValueTypes.VOID,
                desc="Causes this entity's OnUser1 output to be fired.",
            ))
        )]
    }
    assert fgd.snippet_output == {
        'user1': [Snippet(
            'uSer1', 'snippets.fgd', 3,
            (frozenset(['-TAG']), IODef(
                name='OnUser1',
                type=ValueTypes.VOID,
                desc="Fired in response to FireUser1 input.",
            ))
        )]
    }
    # Check they were included correctly, but are not shared (since these are mutable).
    [tags, snip_in] = fgd.snippet_input['user1'][0].value
    [tags, snip_out] = fgd.snippet_output['user1'][0].value
    ent = fgd.entities['some_ent']
    ent_inp = ent.inp['fireuser1', {'tag'}]
    ent_out = ent.out['onuser1', ()]
    assert ent_inp == snip_in
    assert ent_inp is not snip_in, 'Shared!'
    assert ent_out == snip_out
    assert ent_out is not snip_out, 'Shared!'


def test_snippet_no_dup(py_c_token) -> None:
    """Test duplicating snippets is not allowed."""
    fgd = FGD()
    fsys = VirtualFileSystem({
        'snip_1.fgd': """\

        @snippet input InpBlock = FirstInput(void) : "Does something."
        
        @snippet desc a_desc = "Some description"
        @snippet input InpBlock = SecondInput(void) : "I/O appends to existing."
        @snippet desc replace a_desc = "Overwrites existing."
        """,
        'snip_2.fgd': """\
        @snippet desc a_desc = "Will error, no replace command."
        """,
        'snip_3.fgd': """\
        @snippet desc replace missing = "Will error, first does not exist."
        """,
    })
    fgd.parse_file(fsys, fsys['snip_1.fgd'])
    assert fgd.snippet_input == {
        'inpblock': [
            Snippet(
                'InpBlock', 'snip_1.fgd', 2,
                (frozenset(), IODef(
                    'FirstInput', ValueTypes.VOID,
                    "Does something."
                ))
            ), Snippet(
                'InpBlock', 'snip_1.fgd', 5,
                (frozenset(), IODef(
                    'SecondInput', ValueTypes.VOID,
                    "I/O appends to existing."
                ))
            )
        ]
    }
    assert fgd.snippet_desc == {
        'a_desc': [Snippet(
            'a_desc', 'snip_1.fgd', 6,
            "Overwrites existing."
        )]
    }
    with pytest.raises(ValueError, match=re.escape(
        'Two description snippets were defined with the name "a_desc":\n'
        '- snip_1.fgd:6\n'
        '- snip_2.fgd:1'
    )):
        fgd.parse_file(fsys, fsys['snip_2.fgd'])
    with pytest.raises(ValueError, match=re.escape(
        'Tried to replace description snippet with name "missing", but none exist!'
    )):
        fgd.parse_file(fsys, fsys['snip_3.fgd'])


PARSE_EOF = {
    'entity.fgd': """\
@PointClass = some_entity: "Some description" 
[
keyvalue1(string): "Name" : "default": "documentation"
    """,
    'entity_desc.fgd': """\
@PointClass = some_entity: 
    """,
    'choices.fgd': """
@PointClass = some_entity: "Some description" 
[
somechoices(choices): "Name" : 0 : "doc" = [
"value1": "friendly"
    """,
    'spawnflags.fgd': """
@PointClass = some_entity: "Some description" 
[
spawnflags(flags) = [
1: "enable"
2: "ignite"
    """
}


@pytest.mark.parametrize('filename', PARSE_EOF)
def test_parse_eof(py_c_token, filename: str) -> None:
    """Test missing the ending bracket correctly causes errors."""
    fgd = FGD()
    fsys = VirtualFileSystem(PARSE_EOF)
    with pytest.raises(FGDParseError) as err:
        fgd.parse_file(fsys, fsys[filename])
    assert 'ended unexpectedly' in err.value.mess
    assert err.value.file == filename
    # Don't particularly care where the line number is, just that it is set.
    assert err.value.line_num is not None


@pytest.mark.parametrize('custom_syntax', [False, True], ids=['vanilla', 'custom'])
def test_export_regressions(file_regression: FileRegressionFixture, custom_syntax: bool) -> None:
    """Generate a FGD file to ensure code doesn't unintentionally alter output."""
    fgd = FGD()
    base_origin = EntityDef(EntityTypes.BASE, 'Origin')
    base_angles = EntityDef(EntityTypes.BASE, 'Angles')

    ent = EntityDef(EntityTypes.NPC, 'npc_test')
    ent.bases = [base_origin, base_angles, 'MissingBase']

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
        HelperExtAppliesTo(['-episodic']),
        UnknownHelper('extrahelper', ['1', '15', 'thirtytwo']),
        HelperSize(Vec(-16, -16, -16), Vec(16, 16, 16)),
    ]
    ent.desc = 'Entity description, extending beyond 1000 characters: ' + ', '.join(map(str, range(500))) + '. Done!'
    # Only newlines can be escaped in the original parser.
    ent.desc += '\n escapes: tab=\t, newline=\n, "quoted", bell=\a'
    ent.keyvalues['test_kv'] = {frozenset(): KVDef(
        'test_kv',
        ValueTypes.COLOR_255,
        'A test keyvalue',
        '255 255 128',
        'Help text for a keyvalue',
    )}
    ent.keyvalues['custom_kv'] = {frozenset(): KVDef(
        'custom_Kv',
        'super_string',
        'Custom Key Value',
        'super duper',
        'A keyvalue with a custom valuetype',
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
    ent.outputs['SetLocale'] = {frozenset(): IODef('SetLocale', 'locale', 'Set with a custom value type.')}

    ent.outputs['OnNoDesc'] = {frozenset(): IODef('OnNoDesc', ValueTypes.VOID)}
    ent.outputs['OnSomething'] = {frozenset({'alpha'}): IODef('OnSomething', ValueTypes.VOID)}
    ent.outputs['OnCustom'] = {frozenset(): IODef('OnCustom', 'bitfield', 'Uses custom value type.')}

    # IO special case, boolean value type is named differently.
    ent.outputs['OnGetValue'] = {frozenset(): IODef(
        'OnGetValue',
        ValueTypes.BOOL,
        'Get some value',
    )}

    ent.resources = [
        Resource.mdl('models/error.mdl', frozenset({'!hl2'})),
        Resource.mat('tools/toolsnodraw'),
        Resource.snd('Default.Null'),
        Resource('testing/test.nut', FileType.VSCRIPT_SQUIRREL),
        Resource('func_button_sounds', FileType.ENTCLASS_FUNC),
    ]

    buf = io.StringIO()
    fgd.export(buf, custom_syntax=custom_syntax)
    file_res = buf.getvalue()
    direct_res = fgd.export(custom_syntax=custom_syntax)
    # Check fgd.export() produces a string directly in the same way.
    assert file_res == direct_res

    file_regression.check(buf.getvalue(), extension='.fgd')


@pytest.mark.parametrize('label', [False, True], ids=['none', 'added'])
def test_export_spawnflag_label(file_regression: FileRegressionFixture, label: bool) -> None:
    """Test both options for labelling spawnflags."""
    fgd = FGD()
    fgd.entities['demo_ent'] = ent = EntityDef(EntityTypes.ROPES, 'demo_ent')
    ent.keyvalues['spawnflags'] = {frozenset(): KVDef(
        'spawnflags',
        ValueTypes.SPAWNFLAGS,
        'Flags',
        val_list=[
            (1, 'A', False, frozenset()),
            (2, 'B', True, frozenset()),
            (4, 'C', False, frozenset()),
            (8, 'D', False, frozenset()),
            (8, 'E', True, frozenset({'NEW'})),
        ],
    )}

    result = fgd.export(label_spawnflags=label)
    assert ('"[4] C"' in result) is label
    file_regression.check(result, extension='.fgd')


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

    test_kv = KVDef(
        name='custom_key',
        type='special_str',
        disp_name='Custom Keyvalue',
    )
    duplicate = func(test_kv)
    assert duplicate.type is ValueTypes.STRING
    assert duplicate.custom_type == 'special_str'


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

    test_kv = IODef(
        name='OnCustomType',
        type='special_str',
        desc='Outputs a special value',
    )
    duplicate = func(test_kv)
    assert duplicate.name == 'OnCustomType'
    assert duplicate.type is ValueTypes.STRING
    assert duplicate.custom_type == 'special_str'
    assert duplicate.desc == 'Outputs a special value'


def test_resource_shorthand() -> None:
    """Test the shorthand resource methods."""
    assert Resource.mat(
        'some_material.vmt', frozenset({"A", "-B"})
    ) == Resource('some_material.vmt', FileType.MATERIAL, frozenset({"A", "-B"}))

    assert Resource.snd(
        'sound/error.wav', frozenset({"A", "-B"})
    ) == Resource('sound/error.wav', FileType.GAME_SOUND, frozenset({"A", "-B"}))

    assert Resource.weapon_script(
        'scripts/death.txt', frozenset({"A", "-B"})
    ) == Resource('scripts/death.txt', FileType.WEAPON_SCRIPT, frozenset({"A", "-B"}))

    assert Resource.mdl(
        'models/noskin.mdl', frozenset({"A", "-B"})
    ) == Resource('models/noskin.mdl', FileType.MODEL, frozenset({"A", "-B"}))

    assert Resource.mdl(
        'models/skinnable.mdl', frozenset({"A", "-B"}), 0,
    ) == Resource('models/skinnable.mdl#0', FileType.MODEL, frozenset({"A", "-B"}))

    assert Resource.mdl(
        'models/skinnable.mdl', frozenset({"A", "-B"}), [],
    ) == Resource('models/skinnable.mdl', FileType.MODEL, frozenset({"A", "-B"}))

    assert Resource.mdl(
        'models/skinnable.mdl', frozenset({"A", "-B"}), [0, 4, 8]
    ) == Resource('models/skinnable.mdl#0,4,8', FileType.MODEL, frozenset({"A", "-B"}))
