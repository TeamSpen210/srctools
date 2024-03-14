"""Test the resource functions implemented for specific entities."""
from typing import Dict, Iterable, List, Mapping, Union
from operator import itemgetter

import pytest

from srctools import EmptyMapping
from srctools.const import FileType
from srctools.fgd import EntityDef, Resource, ResourceCtx
from srctools.filesys import VirtualFileSystem
from srctools.vmf import VMF, ValidKVs


TAGS_EZ2 = frozenset({"hl2", "episodic", "mapbase", "entropyzero2"})

# Additional resources EZ2 precaches for any prop.
RES_EZ2_PROP = [
    Resource.snd('Metal.SawbladeStick'),
    Resource.snd('PropaneTank.Burst'),
    Resource.snd('Weapon_FlareGun.Burn'),
    Resource.part('env_fire_large'),
    Resource.part('env_fire_large_smoke'),
    Resource.part('env_fire_medium'),
    Resource.part('env_fire_medium_smoke'),
    Resource.part('env_fire_small'),
    Resource.part('env_fire_small_smoke'),
    Resource.part('env_fire_tiny'),
    Resource.part('env_fire_tiny_smoke'),
    Resource.mdl('models/weapons/flare.mdl'),
]


def check_entity(
    *resources: Resource,
    classname: str,
    mapname__: str = '',
    tags__: Iterable[str] = (),
    filesys__: Mapping[str, Union[str, bytes]] = EmptyMapping,
    **keyvalues: ValidKVs,
) -> None:
    """Check this entity produces the specified resources."""
    vmf = VMF()
    ent = vmf.create_ent(classname, **keyvalues)
    expected = {(res.type, res.filename) for res in resources}
    ctx = ResourceCtx(
        tags=tags__,
        mapname=mapname__,
        fsys=VirtualFileSystem(filesys__),
    )
    ent_def = EntityDef.engine_def(classname)
    actual = set(ent_def.get_resources(ctx, ent=ent, on_error=LookupError))
    # Sort each so that PyCharm shows diffs correctly.
    assert sorted(actual, key=itemgetter(1)) == sorted(expected, key=itemgetter(1))


def test_basic_ent() -> None:
    """Test a basic entity, no base resources or a function."""
    check_entity(
        Resource("AlyxEmp.Charge", FileType.GAME_SOUND),
        Resource("AlyxEmp.Discharge", FileType.GAME_SOUND),
        Resource("materials/effects/laser1.vmt", FileType.MATERIAL),
        Resource("AlyxEmp.Stop", FileType.GAME_SOUND),
        classname='env_alyxemp',
    )


@pytest.mark.xfail
def test_asw_emitter() -> None:
    """Test handling the ASW-specific effect system."""
    raise NotImplementedError


@pytest.mark.xfail
def test_asw_snow_volume() -> None:
    """Test handling a snowing effect using the ASW effect system."""
    raise NotImplementedError


@pytest.mark.xfail
def test_asw_spawner() -> None:
    """Test special keyvalues on asw_spawner."""
    raise NotImplementedError


def test_color_correction() -> None:
    """Color correction packs the correction filename."""
    check_entity(
        classname='color_correction',
    )
    check_entity(
        Resource('correction/correct.raw'),
        classname='color_correction',
        filename='correction/correct.raw',
    )


def test_func_button() -> None:
    """Test regular func_button sound packing."""
    check_entity(classname='func_button')
    check_entity(
        Resource('Buttons.snd4', FileType.GAME_SOUND),
        Resource('Buttons.snd12', FileType.GAME_SOUND),
        Resource('Buttons.snd26', FileType.GAME_SOUND),
        classname='func_button',
        sounds=4,
        locked_sound='26',
        unlocked_sound=12,
    )


def test_func_button_timed() -> None:
    """This variant has a single sound."""
    check_entity(classname='func_button_timed')
    check_entity(
        Resource('Buttons.snd8', FileType.GAME_SOUND),
        classname='func_button_timed',
        locked_sound=8,
        # Rest aren't used.
        sounds=23,
        unlocked_sound=5,
    )


def test_momentary_rot_button() -> None:
    """If a flag is set, the sounds keyvalue is not used."""
    check_entity(classname='momentary_rot_button')
    check_entity(
        Resource('Buttons.snd8', FileType.GAME_SOUND),
        Resource('Buttons.snd34', FileType.GAME_SOUND),
        classname='momentary_rot_button',
        sounds=16,
        locked_sound='8',
        unlocked_sound=34,
    )
    check_entity(
        Resource('Buttons.snd16', FileType.GAME_SOUND),
        Resource('Buttons.snd8', FileType.GAME_SOUND),
        Resource('Buttons.snd34', FileType.GAME_SOUND),
        classname='momentary_rot_button',
        spawnflags=1024,  # SF_BUTTON_USE_ACTIVATES
        sounds=16,
        locked_sound='8',
        unlocked_sound=34,
    )


@pytest.mark.xfail
def test_base_plat_train() -> None:
    raise NotImplementedError


# TODO: These should be based on the material...
BREAKABLE = [
    Resource.snd('Breakable.Ceiling'),
    Resource.snd('Breakable.Computer'),
    Resource.snd('Breakable.Concrete'),
    Resource.snd('Breakable.Crate'),
    Resource.snd('Breakable.Flesh'),
    Resource.snd('Breakable.Glass'),
    Resource.snd('Breakable.MatConcrete'),
    Resource.snd('Breakable.MatFlesh'),
    Resource.snd('Breakable.MatGlass'),
    Resource.snd('Breakable.MatMetal'),
    Resource.snd('Breakable.MatWood'),
    Resource.snd('Breakable.Metal'),
]


def test_func_breakable_materials() -> None:
    """Test breakable chunk lookups for breakable brushes."""
    check_entity(
        *BREAKABLE,
        Resource('WoodChunks', FileType.BREAKABLE_CHUNK),
        classname='func_breakable',
        material=1,
    )
    check_entity(
        *BREAKABLE,
        # Not sure which is generated, so specify both.
        Resource('ConcreteChunks', FileType.BREAKABLE_CHUNK),
        Resource('CeilingTile', FileType.BREAKABLE_CHUNK),
        classname='func_breakable',
        material=4,
    )
    check_entity(  # Changed in Portal 2.
        *BREAKABLE,
        Resource('WebGibs', FileType.BREAKABLE_CHUNK),
        classname='func_breakable',
        material=9,
    )
    check_entity(  # Changed in Portal 2.
        *BREAKABLE,
        Resource('MetalPanelChunks', FileType.BREAKABLE_CHUNK),
        classname='func_breakable',
        material=9,
        tags__=['P2'],
    )


def test_func_breakable_spawns() -> None:
    """Test the builtin spawn table for breakable brushes."""
    check_entity(
        *BREAKABLE,
        Resource('WoodChunks', FileType.BREAKABLE_CHUNK),
        Resource.snd('HealthKit.Touch'),
        Resource.mdl('models/items/healthkit.mdl'),
        classname='func_breakable',
        material=1,
        spawnobject=2,
    )
    check_entity(
        *BREAKABLE,
        Resource('WoodChunks', FileType.BREAKABLE_CHUNK),
        classname='func_breakable',
        material=1,
        spawnobject=34,
    )
    # TODO: Mesa and HL1 resources, so we can check their spawn table.


def test_func_breakable_surf() -> None:
    """Breakable surf has two sets of special resources loaded."""
    check_entity(
        *BREAKABLE,
        Resource.mdl("models/brokenglass_piece.mdl"),
        Resource.mat("materials/models/brokenglass/glassbroken_solid.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_01a.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_01b.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_01c.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_01d.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_02a.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_02b.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_02c.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_02d.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_03a.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_03b.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_03c.vmt"),
        Resource.mat("materials/models/brokenglass/glassbroken_03d.vmt"),
        Resource.mat("materials/effects/fleck_glass1.vmt"),
        Resource.mat("materials/effects/fleck_glass2.vmt"),
        classname='func_breakable_surf',
        material=0,
        surfacetype=0,
    )
    check_entity(
        *BREAKABLE,
        Resource.mdl("models/brokenglass_piece.mdl"),
        Resource.mat("materials/models/brokentile/tilebroken_01a.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_01b.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_01c.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_01d.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_02a.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_02b.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_02c.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_02d.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_03a.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_03b.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_03c.vmt"),
        Resource.mat("materials/models/brokentile/tilebroken_03d.vmt"),
        Resource.mat("materials/effects/fleck_tile1.vmt"),
        Resource.mat("materials/effects/fleck_tile2.vmt"),
        classname='func_breakable_surf',
        material=5,
        surfacetype=1,
    )


@pytest.mark.parametrize('cls', ['move_rope', 'keyframe_rope'])
def test_sprite_ropes(cls: str) -> None:
    """Test both sprite rope entities."""
    check_entity(
        Resource('materials/cable/rope_shadowdepth.vmt', FileType.MATERIAL),
        Resource('materials/cable/cable.vmt', FileType.MATERIAL),
        classname=cls,
    )
    check_entity(
        Resource('materials/cable/rope_shadowdepth.vmt', FileType.MATERIAL),
        # Custom is done via FGD.
        classname=cls,
        ropematerial='materials/something_custom.vmt',
    )
    check_entity(
        Resource('materials/cable/rope_shadowdepth.vmt', FileType.MATERIAL),
        Resource('materials/cable/cable.vmt', FileType.MATERIAL),
        classname=cls,
        ropeshader=0,
    )
    check_entity(
        Resource('materials/cable/rope_shadowdepth.vmt', FileType.MATERIAL),
        Resource('materials/cable/rope.vmt', FileType.MATERIAL),
        classname=cls,
        ropeshader=1,
    )
    check_entity(
        Resource('materials/cable/rope_shadowdepth.vmt', FileType.MATERIAL),
        Resource('materials/cable/chain.vmt', FileType.MATERIAL),
        classname=cls,
        ropeshader=2,
    )


def test_env_break_shooter() -> None:
    """Test break shooter model."""
    check_entity(
        Resource('MetalChunks', FileType.BREAKABLE_CHUNK),
        classname='env_break_shooter',
        model='MetalChunks',
        modeltype=0,
    )
    check_entity(
        Resource('models/some_gib/gib1.mdl', FileType.MODEL),
        classname='env_break_shooter',
        model='models/some_gib/gib1.mdl',
        modeltype=1,
    )
    check_entity(
        classname='env_break_shooter',
        model='some_template',
        modeltype=2,
    )


def test_env_fire() -> None:
    """Fires have a few different resource combos."""
    check_entity(  # Plasma
        Resource('Fire.Plasma', FileType.GAME_SOUND),
        Resource("materials/sprites/plasma1.vmt", FileType.MATERIAL),  # These two from the _plasma ent.
        Resource("materials/sprites/fire_floor.vmt", FileType.MATERIAL),
        classname='env_fire',
        firetype=1,
    )
    # Natural, smoking.
    check_entity(
        Resource('Fire.Plasma', FileType.GAME_SOUND),
        Resource('env_fire_tiny_smoke', FileType.PARTICLE_SYSTEM),
        Resource('env_fire_small_smoke', FileType.PARTICLE_SYSTEM),
        Resource('env_fire_medium_smoke', FileType.PARTICLE_SYSTEM),
        Resource('env_fire_large_smoke', FileType.PARTICLE_SYSTEM),
        classname='env_fire',
        firetype=0,
        spawnflags=4,  # 2 = Smoking, 4 = start on (does nothing)
    )
    # Natural, smokeless
    check_entity(
        Resource('Fire.Plasma', FileType.GAME_SOUND),
        Resource('env_fire_tiny', FileType.PARTICLE_SYSTEM),
        Resource('env_fire_small', FileType.PARTICLE_SYSTEM),
        Resource('env_fire_medium', FileType.PARTICLE_SYSTEM),
        Resource('env_fire_large', FileType.PARTICLE_SYSTEM),
        classname='env_fire',
        firetype=0,
        spawnflags=4 | 2,
    )


def test_env_headcrabcanister() -> None:
    """Headcrab canisters use different resources depending on spawnflags."""
    common = [
        Resource.mdl("models/props_combine/headcrabcannister01b.mdl"),
    ]
    common_crab = [
        *BASE_NPC,
        Resource.snd("NPC_HeadCrab.Gib"),
        Resource.snd("NPC_Headcrab.BurrowIn"),
        Resource.snd("NPC_Headcrab.BurrowOut"),
    ]
    SF_NO_IMPACT_SOUND = 0x1
    SF_NO_LAUNCH_SOUND = 0x2
    SF_START_IMPACTED = 0x1000
    SF_REMOVE_ON_IMPACT = 0x40000
    SF_NO_IMPACT_EFFECTS = 0x80000
    # Without START_IMPACTED, we have a smoke trail and fly through the air.
    check_entity(
        Resource.mdl("models/props_combine/headcrabcannister01a.mdl"),
        Resource.mdl("models/props_combine/headcrabcannister01a_skybox.mdl"),
        Resource.mat("materials/sprites/smoke.vmt"),
        classname='env_headcrabcanister',
        headcrabtype=0,
        spawnflags=(
            SF_NO_IMPACT_SOUND | SF_NO_LAUNCH_SOUND |
            SF_NO_IMPACT_EFFECTS | SF_REMOVE_ON_IMPACT
        ),
    )
    # Without remove on impact, we spawn headcrabs.
    check_entity(
        *common,
        Resource.mdl("models/props_combine/headcrabcannister01b.mdl"),
        Resource.snd("HeadcrabCanister.AfterLanding"),
        Resource.snd("HeadcrabCanister.Open"),
        # npc_headcrab
        *common_crab,
        Resource.mdl("models/headcrabclassic.mdl"),
        Resource.snd("NPC_HeadCrab.Idle"),
        Resource.snd("NPC_HeadCrab.Alert"),
        Resource.snd("NPC_HeadCrab.Pain"),
        Resource.snd("NPC_HeadCrab.Die"),
        Resource.snd("NPC_HeadCrab.Attack"),
        Resource.snd("NPC_HeadCrab.Bite"),
        classname='env_headcrabcanister',
        headcrabtype=0,  # Normal
        spawnflags=(
            SF_NO_IMPACT_SOUND | SF_NO_LAUNCH_SOUND |
            SF_NO_IMPACT_EFFECTS | SF_START_IMPACTED
        ),
    )
    check_entity(
        *common,
        Resource.mdl("models/props_combine/headcrabcannister01b.mdl"),
        Resource.snd("HeadcrabCanister.AfterLanding"),
        Resource.snd("HeadcrabCanister.Open"),
        # npc_headcrab_fast
        *common_crab,
        Resource.mdl("models/headcrab.mdl"),
        Resource.snd("NPC_FastHeadCrab.Idle"),
        Resource.snd("NPC_FastHeadCrab.Alert"),
        Resource.snd("NPC_FastHeadCrab.Pain"),
        Resource.snd("NPC_FastHeadCrab.Die"),
        Resource.snd("NPC_FastHeadCrab.Attack"),
        Resource.snd("NPC_FastHeadCrab.Bite"),
        classname='env_headcrabcanister',
        headcrabtype=1,  # Fast
        spawnflags=(
            SF_NO_IMPACT_SOUND | SF_NO_LAUNCH_SOUND |
            SF_NO_IMPACT_EFFECTS | SF_START_IMPACTED
        ),
    )
    check_entity(
        *common,
        Resource.mdl("models/props_combine/headcrabcannister01b.mdl"),
        Resource.snd("HeadcrabCanister.AfterLanding"),
        Resource.snd("HeadcrabCanister.Open"),
        # npc_headcrab_poison
        *common_crab,
        Resource.mdl("models/headcrabblack.mdl"),
        Resource.snd("NPC_BlackHeadcrab.Telegraph"),
        Resource.snd("NPC_BlackHeadcrab.Attack"),
        Resource.snd("NPC_BlackHeadcrab.Bite"),
        Resource.snd("NPC_BlackHeadcrab.Threat"),
        Resource.snd("NPC_BlackHeadcrab.Alert"),
        Resource.snd("NPC_BlackHeadcrab.Idle"),
        Resource.snd("NPC_BlackHeadcrab.Talk"),
        Resource.snd("NPC_BlackHeadcrab.AlertVoice"),
        Resource.snd("NPC_BlackHeadcrab.Pain"),
        Resource.snd("NPC_BlackHeadcrab.Die"),
        Resource.snd("NPC_BlackHeadcrab.Impact"),
        Resource.snd("NPC_BlackHeadcrab.ImpactAngry"),
        Resource.snd("NPC_BlackHeadcrab.FootstepWalk"),
        Resource.snd("NPC_BlackHeadcrab.Footstep"),
        classname='env_headcrabcanister',
        headcrabtype=2,  # Poison
        spawnflags=(
            SF_NO_IMPACT_SOUND | SF_NO_LAUNCH_SOUND |
            SF_NO_IMPACT_EFFECTS | SF_START_IMPACTED
        ),
    )
    # Without no_impact_sound, we play those.
    check_entity(
        Resource.mat("materials/sprites/smoke.vmt"),
        Resource.mdl("models/props_combine/headcrabcannister01a.mdl"),
        Resource.mdl("models/props_combine/headcrabcannister01a_skybox.mdl"),
        Resource.snd("HeadcrabCanister.Explosion"),
        Resource.snd("HeadcrabCanister.IncomingSound"),
        Resource.snd("HeadcrabCanister.SkyboxExplosion"),
        classname='env_headcrabcanister',
        headcrabtype=0,  # Normal
        spawnflags=SF_NO_LAUNCH_SOUND | SF_NO_IMPACT_EFFECTS | SF_REMOVE_ON_IMPACT,
    )
    # Without no_launch_sound, we play that.
    check_entity(
        Resource.mat("materials/sprites/smoke.vmt"),
        Resource.mdl("models/props_combine/headcrabcannister01a.mdl"),
        Resource.mdl("models/props_combine/headcrabcannister01a_skybox.mdl"),
        Resource.snd("HeadcrabCanister.LaunchSound"),
        classname='env_headcrabcanister',
        headcrabtype=0,  # Normal
        spawnflags=SF_NO_IMPACT_SOUND | SF_NO_IMPACT_EFFECTS | SF_REMOVE_ON_IMPACT,
    )
    # With impact effects, we spawn an ar2explosion.
    check_entity(
        Resource.mdl("models/props_combine/headcrabcannister01a.mdl"),
        Resource.mdl("models/props_combine/headcrabcannister01a_skybox.mdl"),
        Resource.mat("materials/sprites/smoke.vmt"),
        Resource.mat("materials/particle/particle_noisesphere.vmt"),
        Resource.snd("HeadcrabCanister.LaunchSound"),
        classname='env_headcrabcanister',
        headcrabtype=0,  # Normal
        spawnflags=SF_NO_IMPACT_SOUND | SF_REMOVE_ON_IMPACT,
    )
    # Everything together.
    check_entity(
        Resource.mdl("models/props_combine/headcrabcannister01a.mdl"),
        Resource.mdl("models/props_combine/headcrabcannister01a_skybox.mdl"),
        Resource.mdl("models/props_combine/headcrabcannister01b.mdl"),
        Resource.mat("materials/particle/particle_noisesphere.vmt"),
        Resource.mat("materials/sprites/smoke.vmt"),
        Resource.snd("HeadcrabCanister.LaunchSound"),
        Resource.snd("HeadcrabCanister.Explosion"),
        Resource.snd("HeadcrabCanister.IncomingSound"),
        Resource.snd("HeadcrabCanister.SkyboxExplosion"),
        Resource.snd("HeadcrabCanister.AfterLanding"),
        Resource.snd("HeadcrabCanister.Open"),
        # npc_headcrab_fast
        *common_crab,
        Resource.mdl("models/headcrab.mdl"),
        Resource.snd("NPC_FastHeadCrab.Idle"),
        Resource.snd("NPC_FastHeadCrab.Alert"),
        Resource.snd("NPC_FastHeadCrab.Pain"),
        Resource.snd("NPC_FastHeadCrab.Die"),
        Resource.snd("NPC_FastHeadCrab.Attack"),
        Resource.snd("NPC_FastHeadCrab.Bite"),
        classname='env_headcrabcanister',
        headcrabtype=1,  # Fast
        spawnflags=0,
    )


@pytest.mark.xfail
def test_env_shooter() -> None:
    raise NotImplementedError


def test_env_smokestack() -> None:
    """Smokestacks have unique behaviour, loading resources in numeric order."""
    sample = b'''UnlitGeneric
    {
    $basetexture white
    }
'''
    keys_partial = [
        'materials/sprites/some_smoky_mat.vmt',
        'materials/sprites/some_smoky_mat1.vmt',
        'materials/sprites/some_smoky_mat2.vmt',
        'materials/sprites/some_smoky_mat4.vmt',
        'materials/sprites/some_smoky_mat5.vmt',
        'materials/sprites/some_smoky_mat6.vmt',
        'materials/sprites/some_smoky_mat7.vmt',
        'materials/sprites/some_smoky_mat8.vmt',
        'materials/sprites/some_smoky_mat9.vmt',
    ]
    fsys_partial = dict.fromkeys(keys_partial, sample)
    fsys_complete = dict.fromkeys([
        *keys_partial,
        'materials/sprites/some_smoky_mat3.vmt',
    ], sample)
    # In base HL2, this doesn't occur.
    check_entity(
        Resource.mat('materials/particle/SmokeStack.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat.vmt'),
        classname='env_smokestack',
        smokematerial='sprites/some_smoky_mat.vmt',
        filesys__=fsys_complete,
        tags__=['hl2'],
    )

    # In episodic, they're included up to 8 mats.
    check_entity(
        Resource.mat('materials/particle/SmokeStack.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat1.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat2.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat3.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat4.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat5.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat6.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat7.vmt'),
        classname='env_smokestack',
        smokematerial='sprites/some_smoky_mat.vmt',
        filesys__=fsys_complete,
        tags__=['hl2', 'episodic'],
    )
    # If one is missing, it stops.
    check_entity(
        Resource.mat('materials/particle/SmokeStack.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat1.vmt'),
        Resource.mat('materials/sprites/some_smoky_mat2.vmt'),
        classname='env_smokestack',
        smokematerial='sprites/some_smoky_mat.vmt',
        filesys__=fsys_partial,
        tags__=['hl2', 'episodic'],
    )


ammo_crate_models_hl2 = {
    0: "models/items/ammocrate_pistol.mdl",
    1: "models/items/ammocrate_smg1.mdl",
    2: "models/items/ammocrate_ar2.mdl",
    3: "models/items/ammocrate_rockets.mdl",
    4: "models/items/ammocrate_buckshot.mdl",
    # 5: "models/items/ammocrate_grenade.mdl",
    6: "models/items/ammocrate_smg1.mdl",
    7: "models/items/ammocrate_smg1.mdl",
    8: "models/items/ammocrate_ar2.mdl",
    9: "models/items/ammocrate_smg2.mdl",
}

ammo_crate_models_mbase = {
    0: "models/items/ammocrate_pistol.mdl",
    1: "models/items/ammocrate_smg1.mdl",
    2: "models/items/ammocrate_ar2.mdl",
    3: "models/items/ammocrate_rockets.mdl",
    4: "models/items/ammocrate_buckshot.mdl",
    # 5: "models/items/ammocrate_grenade.mdl",
    6: "models/items/ammocrate_357.mdl",
    7: "models/items/ammocrate_xbow.mdl",
    8: "models/items/ammocrate_ar2alt.mdl",
    9: "models/items/ammocrate_smg2.mdl",
    # 10: "models/items/ammocrate_slam.mdl",
    11: "models/items/ammocrate_empty.mdl",
}


@pytest.mark.parametrize('ammo, tags', [
    (ammo_crate_models_hl2, []),
    (ammo_crate_models_mbase, ['mapbase']),
], ids=['hl2', 'mapbase'])
def test_item_ammo_crate(ammo: Dict[int, str], tags: List[str]) -> None:
    """Test ammo crate models."""
    for i, ammo_mdl in ammo.items():
        check_entity(
            Resource.snd('AmmoCrate.Open'),
            Resource.snd('AmmoCrate.Close'),
            Resource.mdl(ammo_mdl),
            classname='item_ammo_crate',
            ammotype=i,
            tags__=tags,
        )
    # Grenades include the entity.
    check_entity(
        Resource.snd('AmmoCrate.Open'),
        Resource.snd('AmmoCrate.Close'),
        Resource.mdl('models/items/ammocrate_grenade.mdl'),
        # npc_weapon_frag:
        Resource.snd('WeaponFrag.Throw'),
        Resource.snd('WeaponFrag.Roll'),
        Resource.mdl("models/Weapons/w_grenade.mdl"),
        Resource.mat("materials/sprites/redglow1.vmt"),
        Resource.mat("materials/sprites/bluelaser1.vmt"),
        Resource.snd("Grenade.Blip"),
        Resource.snd("BaseGrenade.Explode"),
        Resource.weapon_script("scripts/weapon_frag.txt"),
        classname='item_ammo_crate',
        ammotype=5,
        tags__=tags,
    )


def test_item_ammo_crate_mbase_extras() -> None:
    """Test the SLAM, only available in Mapbase."""
    # The SLAM gives the weapon.
    check_entity(
        Resource.snd('AmmoCrate.Open'),
        Resource.snd('AmmoCrate.Close'),
        Resource.mdl('models/items/ammocrate_slam.mdl'),
        # weapon_slam:
        Resource.snd("Weapon_SLAM.ThrowMode"),
        Resource.snd("Weapon_SLAM.TripMineMode"),
        Resource.snd("Weapon_SLAM.SatchelDetonate"),
        Resource.snd("Weapon_SLAM.TripMineAttach"),
        Resource.snd("Weapon_SLAM.SatchelThrow"),
        Resource.snd("Weapon_SLAM.SatchelAttach"),
        Resource.weapon_script("scripts/weapon_slam.txt"),
        # npc_tripmine
        Resource.mdl("models/Weapons/w_slam.mdl"),
        Resource.snd("TripmineGrenade.Charge"),
        Resource.snd("TripmineGrenade.PowerUp"),
        Resource.snd("TripmineGrenade.StopSound"),
        Resource.snd("TripmineGrenade.Activate"),
        Resource.snd("TripmineGrenade.ShootRope"),
        Resource.snd("TripmineGrenade.Hook"),
        # npc_satchel
        Resource.mdl("models/Weapons/w_slam.mdl"),
        Resource.snd("SatchelCharge.Pickup"),
        Resource.snd("SatchelCharge.Bounce"),
        Resource.snd("SatchelCharge.Slide"),
        classname='item_ammo_crate',
        ammotype=10,
        tags__=['mapbase'],
    )


def test_item_item_crate() -> None:
    """Item crates can produce an arbitary entity, and swap models."""
    # No classname = don't spawn anything.
    check_entity(
        Resource.snd('PropaneTank.Burst'),  # From prop_physics.
        Resource.mdl('models/items/item_item_crate.mdl'),
        classname='item_item_crate',
    )
    check_entity(
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/items/item_item_crate.mdl'),
        Resource.mdl('models/items/boxmrounds.mdl'),
        classname='item_item_crate',
        itemclass='item_ammo_smg1',
    )
    check_entity(
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/items/item_item_crate.mdl'),
        classname='item_item_crate',
        itemclass='item_ammo_smg1',
        cratetype=1,  # Mapbase - spawn template, so not spawned by class.
    )
    check_entity(
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/items/item_beacon_crate.mdl'),
        classname='item_item_crate',
        crateappearance=1,  # Beacon version.
        cratetype=1,
    )
    check_entity(
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/my_custom_crate.mdl'),
        classname='item_item_crate',
        crateappearance=2,  # Mapbase, custom model.
        model='models/my_custom_crate.mdl',
        cratetype=1,
    )
    # In EZ2, the variant is passed along.
    check_entity(
        *RES_EZ2_PROP,
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/items/item_item_crate.mdl'),
        classname='item_item_crate',
        tags__=TAGS_EZ2,
    )
    check_entity(
        *RES_EZ2_PROP,
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/items/item_item_crate.mdl'),
        Resource.snd('HealthKit.Touch'),
        Resource.mdl('models/items/healthkit.mdl#0'),
        classname='item_item_crate',
        ezvariant=0,
        itemclass='item_healthkit',
        tags__=TAGS_EZ2,
    )
    check_entity(
        *RES_EZ2_PROP,
        Resource.snd('PropaneTank.Burst'),
        Resource.mdl('models/items/item_item_crate.mdl'),
        Resource.snd('HealthKit_Arbeit.Touch'),
        Resource.mdl('models/items/arbeit/healthkit.mdl#0'),
        classname='item_item_crate',
        ezvariant=4,
        itemclass='item_healthkit',
        tags__=TAGS_EZ2,
    )


def test_item_teamflag() -> None:
    check_entity(classname='item_teamflag')
    check_entity(
        Resource.mat('materials/vgui/flags/my_flag_red.vmt'),
        Resource.mat('materials/vgui/flags/my_flag_blue.vmt'),
        Resource.mat('materials/effects/trail/the_trail_red.vmt'),
        Resource.mat('materials/effects/trail/the_trail_blue.vmt'),
        classname='item_teamflag',
        flag_icon='flags/my_flag',
        flag_trail='trail/the_trail',
    )
    # Test putting in parent folders.
    check_entity(
        Resource.mat('materials/not_vgui/my_flag_red.vmt'),
        Resource.mat('materials/not_vgui/my_flag_blue.vmt'),
        Resource.mat('materials/non_effects/the_trail_red.vmt'),
        Resource.mat('materials/non_effects/the_trail_blue.vmt'),
        classname='item_teamflag',
        flag_icon='../not_vgui/my_flag',
        flag_trail='../non_effects/the_trail',
    )


def test_item_healthkit() -> None:
    """Entropy Zero 2 has several variants for this item."""
    # Regular HL2.
    check_entity(
        Resource.snd('HealthKit.Touch'),
        Resource.mdl('models/items/healthkit.mdl'),
        classname='item_healthkit',
        tags__=['hl2'],
    )
    check_entity(
        Resource.snd('HealthKit.Touch'),
        Resource.mdl('models/items/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=0,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Xen.Touch'),
        Resource.mdl('models/items/xen/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=1,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Rad.Touch'),
        Resource.mdl('models/items/arbeit/healthkit.mdl#1'),
        classname='item_healthkit',
        ezvariant=2,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Temporal.Touch'),
        Resource.mdl('models/items/temporal/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=3,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Arbeit.Touch'),
        Resource.mdl('models/items/arbeit/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=4,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Blood.Touch'),
        Resource.mdl('models/items/blood/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=5,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Athenaeum.Touch'),
        Resource.mdl('models/items/athenaeum/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=6,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthKit_Ash.Touch'),
        Resource.mdl('models/items/ash/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=7,
        tags__=TAGS_EZ2,
    )


def test_item_healthvial() -> None:
    """Entropy Zero 2 has several variants for this item."""
    # Regular HL2.
    check_entity(
        Resource.snd('HealthVial.Touch'),
        Resource.mdl('models/healthvial.mdl'),
        classname='item_healthvial',
        tags__=['hl2'],
    )
    check_entity(
        Resource.snd('HealthVial.Touch'),
        Resource.mdl('models/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=0,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Xen.Touch'),
        Resource.mdl('models/items/xen/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=1,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Rad.Touch'),
        Resource.mdl('models/items/arbeit/healthvial.mdl#1'),
        classname='item_healthvial',
        ezvariant=2,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Temporal.Touch'),
        Resource.mdl('models/items/temporal/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=3,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Arbeit.Touch'),
        Resource.mdl('models/items/arbeit/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=4,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Blood.Touch'),
        Resource.mdl('models/items/blood/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=5,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Athenaeum.Touch'),
        Resource.mdl('models/items/athenaeum/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=6,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('HealthVial_Ash.Touch'),
        Resource.mdl('models/items/ash/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=7,
        tags__=TAGS_EZ2,
    )


def test_item_battery() -> None:
    """Entropy Zero 2 has several variants for this item."""
    # Regular HL2.
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/battery.mdl'),
        classname='item_battery',
        tags__=['hl2'],
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/battery.mdl#0'),
        classname='item_battery',
        ezvariant=0,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/xen/battery.mdl#0'),
        classname='item_battery',
        ezvariant=1,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/arbeit/battery.mdl#1'),
        classname='item_battery',
        ezvariant=2,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/temporal/battery.mdl#0'),
        classname='item_battery',
        ezvariant=3,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/arbeit/battery.mdl#0'),
        classname='item_battery',
        ezvariant=4,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/blood/battery.mdl#0'),
        classname='item_battery',
        ezvariant=5,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/athenaeum/battery.mdl#0'),
        classname='item_battery',
        ezvariant=6,
        tags__=TAGS_EZ2,
    )
    check_entity(
        Resource.snd('ItemBattery.Touch'),
        Resource.mdl('models/items/ash/battery.mdl#0'),
        classname='item_battery',
        ezvariant=7,
        tags__=TAGS_EZ2,
    )


# Common base NPC sounds:
BASE_NPC = [
    Resource.snd('AI_BaseNPC.BodyDrop_Heavy'),
    Resource.snd('AI_BaseNPC.BodyDrop_Light'),
    Resource.snd('AI_BaseNPC.SentenceStop'),
    Resource.snd('AI_BaseNPC.SwishSound'),
]


def test_npcs() -> None:
    """All NPCs have the additionalequipment keyvalue, causing them to spawn with specific weapons."""
    check_entity(
        *BASE_NPC,
        Resource.mdl("models/breen.mdl"),
        classname='npc_breen',
    )

    check_entity(
        *BASE_NPC,
        Resource.mdl("models/breen.mdl"),
        Resource.snd("Weapon_StunStick.Activate"),
        Resource.snd("Weapon_StunStick.Deactivate"),
        Resource.weapon_script("scripts/weapon_stunstick.txt"),
        classname='npc_breen',
        additionalequipment='weapon_stunstick',
    )
    # With Entropy Zero, there's temporal resources available on anything.
    check_entity(
        *BASE_NPC,
        Resource.mdl("models/breen.mdl"),
        Resource.snd('NPC_TemporalHeadcrab.Vanish'),
        Resource.snd('NPC_TemporalHeadcrab.Appear'),
        Resource.snd('ShadowCrab_Vanish'),
        Resource.snd('ShadowCrab_Appear'),
        classname='npc_breen',
        ezvariant=3,
    )


RES_ANT_COMMON = [
    *BASE_NPC,
    Resource.snd("NPC_Antlion.RunOverByVehicle"),
    Resource.snd("NPC_Antlion.MeleeAttack"),
    Resource.snd("NPC_Antlion.Footstep"),
    Resource.snd("NPC_Antlion.BurrowIn"),
    Resource.snd("NPC_Antlion.BurrowOut"),
    Resource.snd("NPC_Antlion.FootstepSoft"),
    Resource.snd("NPC_Antlion.FootstepHeavy"),
    Resource.snd("NPC_Antlion.MeleeAttackSingle"),
    Resource.snd("NPC_Antlion.MeleeAttackDouble"),
    Resource.snd("NPC_Antlion.Distracted"),
    Resource.snd("NPC_Antlion.Idle"),
    Resource.snd("NPC_Antlion.Pain"),
    Resource.snd("NPC_Antlion.Land"),
    Resource.snd("NPC_Antlion.WingsOpen"),
    Resource.snd("NPC_Antlion.LoopingAgitated"),
    Resource.snd("NPC_Antlion.Distracted"),
]
RES_ANT_EPISODIC = [
    Resource.snd("NPC_Antlion.TrappedMetal"),
    Resource.snd("NPC_Antlion.ZappedFlip"),
    Resource.snd("NPC_Antlion.MeleeAttack_Muffled"),
]
RES_ANT_REGULAR = [
    *RES_ANT_COMMON,
    Resource.part("AntlionGib"),
    Resource.mdl("models/gibs/antlion_gib_large_1.mdl"),
    Resource.mdl("models/gibs/antlion_gib_large_2.mdl"),
    Resource.mdl("models/gibs/antlion_gib_large_3.mdl"),
    Resource.mdl("models/gibs/antlion_gib_medium_1.mdl"),
    Resource.mdl("models/gibs/antlion_gib_medium_2.mdl"),
    Resource.mdl("models/gibs/antlion_gib_medium_3.mdl"),
    Resource.mdl("models/gibs/antlion_gib_small_1.mdl"),
    Resource.mdl("models/gibs/antlion_gib_small_2.mdl"),
    Resource.mdl("models/gibs/antlion_gib_small_3.mdl"),
]
RES_ANT_WORKER = [
    Resource.part("blood_impact_antlion_worker_01"),
    Resource.part("antlion_gib_02"),
    Resource.part("blood_impact_yellow_01"),
    Resource.snd("NPC_Antlion.PoisonBurstScream"),
    Resource.snd("NPC_Antlion.PoisonBurstScreamSubmerged"),
    Resource.snd("NPC_Antlion.PoisonBurstExplode"),
    Resource.snd("NPC_Antlion.PoisonShoot"),
    Resource.snd("NPC_Antlion.PoisonBall"),
    # grenade_spit:
    Resource.mdl("models/spitball_large.mdl"),
    Resource.mdl("models/spitball_medium.mdl"),
    Resource.mdl("models/spitball_small.mdl"),
    Resource.snd("BaseGrenade.Explode"),
    Resource.snd("GrenadeSpit.Hit"),
    Resource.part("antlion_spit"),
    Resource.part("antlion_spit_player"),
]


def test_npc_antlion_regular() -> None:
    """Antlions have the worker variant in episodic, and EZ2 adds a few more."""
    check_entity(  # Regular antlion.
        *RES_ANT_REGULAR,
        Resource.mdl("models/antlion.mdl"),
        Resource.part('blood_impact_antlion_01'),
        classname='npc_antlion',
        tags__=["hl2"],
        spawnflags=0,
    )
    check_entity(  # Episodic additions.
        *RES_ANT_REGULAR, *RES_ANT_EPISODIC,
        Resource.mdl("models/antlion.mdl"),
        Resource.part('blood_impact_antlion_01'),
        classname='npc_antlion',
        tags__=["hl2", "episodic"],
        spawnflags=0,
    )
    check_entity(
        *RES_ANT_REGULAR, *RES_ANT_EPISODIC,
        Resource.mdl("models/antlion.mdl"),
        Resource.part('blood_impact_antlion_01'),
        classname='npc_antlion',
        tags__=TAGS_EZ2,
        spawnflags=0,
        ezvariant=0,  # Still normal.
    )
    check_entity(
        *RES_ANT_REGULAR, *RES_ANT_EPISODIC,
        Resource.mdl("models/antlion_xen.mdl"),
        Resource.part('blood_impact_antlion_01'),
        classname='npc_antlion',
        tags__=TAGS_EZ2,
        spawnflags=0,
        ezvariant=1,  # Xen
    )
    check_entity(
        *RES_ANT_REGULAR, *RES_ANT_EPISODIC,
        Resource.mdl("models/antlion_blue.mdl"),
        Resource.part('blood_impact_blue_01'),
        classname='npc_antlion',
        tags__=TAGS_EZ2,
        spawnflags=0,
        ezvariant=2,  # Radiation
    )
    check_entity(
        *RES_ANT_REGULAR, *RES_ANT_EPISODIC,
        Resource.mdl("models/bloodlion.mdl"),
        Resource.part('blood_impact_antlion_01'),
        classname='npc_antlion',
        tags__=TAGS_EZ2,
        spawnflags=0,
        ezvariant=5,  # Bloodlion
    )


def test_npc_antlion_worker() -> None:
    """Antlions have the worker variant in episodic, and EZ2 adds a few more."""
    # Now, workers.
    check_entity(
        *RES_ANT_COMMON, *RES_ANT_EPISODIC, *RES_ANT_WORKER,
        Resource.mdl("models/antlion_worker.mdl"),
        classname='npc_antlion',
        tags__=["hl2", "episodic"],
        spawnflags=(1 << 18) | 3,  # 3 = irrelevant.
    )
    # EZ variant 0 doesn't change the antlion, adds some grenade resources though.
    check_entity(
        *RES_ANT_COMMON, *RES_ANT_EPISODIC, *RES_ANT_WORKER,
        Resource.mdl("models/antlion_worker.mdl"),
        Resource.part("ExplosionCore"),  # Not actually used, but loaded via grenade base class.
        Resource.part("ExplosionEmbers"),
        Resource.part("ExplosionFlash"),
        classname='npc_antlion',
        tags__=TAGS_EZ2,
        spawnflags=(1 << 18) | 3,
        ezvariant=0,
    )
    check_entity(
        *RES_ANT_COMMON, *RES_ANT_EPISODIC, *RES_ANT_WORKER,
        Resource.mdl("models/bloodlion_worker.mdl"),
        Resource.part("ExplosionCore"),
        Resource.part("ExplosionEmbers"),
        Resource.part("ExplosionFlash"),
        classname='npc_antlion',
        tags__=TAGS_EZ2,
        spawnflags=(1 << 18) | 3,
        ezvariant=5,  # Bloodlion
    )


@pytest.mark.xfail
def test_npc_antlionguard() -> None:
    """Quite complicated, episodic specific sounds, and EZ2 has variants."""
    raise NotImplementedError


def test_npc_antlion_template_maker() -> None:
    """The template maker can force spawned antlions to be workers."""
    check_entity(classname='npc_antlion_template_maker')
    check_entity(  # Includes spore effects.
        Resource.mat("materials/particle/fire.vmt"),
        classname='npc_antlion_template_maker',
        createspores='1',
        tags__=["hl2", "episodic"],
    )
    # We need to include the additional worker resources.
    check_entity(
        *RES_ANT_WORKER,
        Resource.mdl("models/antlion_worker.mdl"),
        classname='npc_antlion_template_maker',
        workerspawnrate=0.25,
        tags__=["hl2", "episodic"],
    )
    # Both at once
    check_entity(
        *RES_ANT_WORKER,
        Resource.mdl("models/antlion_worker.mdl"),
        Resource.mat("materials/particle/fire.vmt"),
        classname='npc_antlion_template_maker',
        createspores='1',
        workerspawnrate=0.25,
        tags__=["hl2", "episodic"],
    )
    # In Entropy Zero, add in the bloodlion variant.
    check_entity(
        *RES_ANT_WORKER,
        Resource.mdl("models/antlion_worker.mdl"),
        Resource.mdl("models/bloodlion_worker.mdl"),
        Resource.part("ExplosionCore"),   # EZ2 adds these extra particles too.
        Resource.part("ExplosionEmbers"),
        Resource.part("ExplosionFlash"),
        classname='npc_antlion_template_maker',
        workerspawnrate=0.25,
        tags__=TAGS_EZ2,
    )


# npc_turret_floor
TURRET_RES_EZ2 = [
    *BASE_NPC,
    Resource.mdl("models/combine_turrets/citizen_turret.mdl"),
    Resource.mdl("models/combine_turrets/floor_turret.mdl"),
    Resource("MetalChunks", FileType.BREAKABLE_CHUNK),
    Resource.mat("materials/effects/laser1.vmt"),
    Resource.mat("materials/sprites/glow1.vmt"),
    Resource.snd("NPC_FloorTurret.AlarmPing"),
    Resource.snd("NPC_FloorTurret.Retire"),
    Resource.snd("NPC_FloorTurret.Deploy"),
    Resource.snd("NPC_FloorTurret.Move"),
    Resource.snd("NPC_Combine.WeaponBash"),
    Resource.snd("NPC_FloorTurret.Activate"),
    Resource.snd("NPC_FloorTurret.Alert"),
    Resource.snd("NPC_FloorTurret.ShotSounds"),
    Resource.snd("NPC_FloorTurret.Die"),
    Resource.snd("NPC_FloorTurret.Retract"),
    Resource.snd("NPC_FloorTurret.Alarm"),
    Resource.snd("NPC_FloorTurret.Ping"),
    Resource.snd("NPC_FloorTurret.DryFire"),
    Resource.snd("NPC_FloorTurret.Destruct"),
    Resource.part("explosion_turret_break"),  # Episodic
]


def test_npc_arbeit_turret_floor() -> None:
    check_entity(
        *TURRET_RES_EZ2,
        Resource.mdl("models/props/turret_01.mdl"),
        Resource.snd("NPC_ArbeitTurret.DryFire"),
        classname='npc_arbeit_turret_floor',
        tags__=TAGS_EZ2,  # Should always be present.
    )

    check_entity(
        *TURRET_RES_EZ2,
        Resource.mdl("models/props/turret_01.mdl"),
        Resource.snd("NPC_ArbeitTurret.DryFire"),
        classname='npc_arbeit_turret_floor',
        ezvariant=0,  # Normal
        tags__=TAGS_EZ2,
    )
    check_entity(
        *TURRET_RES_EZ2,
        Resource.mdl("models/props/hackedturret_01.mdl"),
        Resource.snd("NPC_ArbeitTurret.DryFire"),
        classname='npc_arbeit_turret_floor',
        ezvariant=0,  # Normal
        spawnflags=0x200 | 0x20,  # Citizen modified, auto active (unrelated)
        tags__=TAGS_EZ2,
    )
    check_entity(
        *TURRET_RES_EZ2,
        Resource.mdl("models/props/glowturret_01.mdl"),
        Resource.mat("materials/cable/goocable.vmt"),
        Resource.snd("NPC_ArbeitTurret.DryFire"),
        classname='npc_arbeit_turret_floor',
        ezvariant=2,  # Radiation
        tags__=TAGS_EZ2,
    )
    check_entity(
        *TURRET_RES_EZ2,
        Resource.mdl("models/props/camoturret_01.mdl"),
        Resource.mdl("models/props/camoturret_02.mdl"),
        Resource.snd("NPC_ArbeitTurret.DryFire"),
        classname='npc_arbeit_turret_floor',
        ezvariant=4,  # Arbeit
        tags__=TAGS_EZ2,
    )


@pytest.mark.xfail
def test_npc_bullsquid() -> None:
    raise NotImplementedError


RES_COMBINE_MINE = [
    Resource.mdl("models/props_combine/combine_mine01.mdl"),
    Resource.snd("NPC_CombineMine.Hop"),
    Resource.snd("NPC_CombineMine.FlipOver"),
    Resource.snd("NPC_CombineMine.TurnOn"),
    Resource.snd("NPC_CombineMine.TurnOff"),
    Resource.snd("NPC_CombineMine.OpenHooks"),
    Resource.snd("NPC_CombineMine.CloseHooks"),
    Resource.snd("NPC_CombineMine.ActiveLoop"),
    Resource.mat("materials/sprites/glow01.vmt"),
]


RES_SCANNER_CITY = [
    Resource.mdl("models/combine_scanner.mdl"),
    Resource.mdl("models/gibs/scanner_gib01.mdl"),
    Resource.mdl("models/gibs/scanner_gib02.mdl"),
    # Gib 3 does not exist!
    Resource.mdl("models/gibs/scanner_gib04.mdl"),
    Resource.mdl("models/gibs/scanner_gib05.mdl"),
    Resource.mat("materials/sprites/light_glow03.vmt"),
    Resource.mat("materials/sprites/glow_test02.vmt"),
    Resource.snd("NPC_CScanner.Shoot"),
    Resource.snd("NPC_CScanner.Alert"),
    Resource.snd("NPC_CScanner.Die"),
    Resource.snd("NPC_CScanner.Combat"),
    Resource.snd("NPC_CScanner.Idle"),
    Resource.snd("NPC_CScanner.Pain"),
    Resource.snd("NPC_CScanner.TakePhoto"),
    Resource.snd("NPC_CScanner.AttackFlash"),
    Resource.snd("NPC_CScanner.DiveBombFlyby"),
    Resource.snd("NPC_CScanner.DiveBomb"),
    Resource.snd("NPC_CScanner.DeployMine"),
    Resource.snd("NPC_CScanner.FlyLoop"),
    *RES_COMBINE_MINE,
    *BASE_NPC,
]

RES_SCANNER_CLAW = [
    Resource.mdl("models/shield_scanner.mdl"),
    Resource.mdl("models/gibs/Shield_Scanner_Gib1.mdl"),
    Resource.mdl("models/gibs/Shield_Scanner_Gib2.mdl"),
    Resource.mdl("models/gibs/Shield_Scanner_Gib3.mdl"),
    Resource.mdl("models/gibs/Shield_Scanner_Gib4.mdl"),
    Resource.mdl("models/gibs/Shield_Scanner_Gib5.mdl"),
    Resource.mdl("models/gibs/Shield_Scanner_Gib6.mdl"),
    Resource.mat("materials/sprites/light_glow03.vmt"),
    Resource.mat("materials/sprites/glow_test02.vmt"),
    Resource.snd("NPC_SScanner.Shoot"),
    Resource.snd("NPC_SScanner.Alert"),
    Resource.snd("NPC_SScanner.Die"),
    Resource.snd("NPC_SScanner.Combat"),
    Resource.snd("NPC_SScanner.Idle"),
    Resource.snd("NPC_SScanner.Pain"),
    Resource.snd("NPC_SScanner.TakePhoto"),
    Resource.snd("NPC_SScanner.AttackFlash"),
    Resource.snd("NPC_SScanner.DiveBombFlyby"),
    Resource.snd("NPC_SScanner.DiveBomb"),
    Resource.snd("NPC_SScanner.DeployMine"),
    Resource.snd("NPC_SScanner.FlyLoop"),
    *RES_COMBINE_MINE,
    *BASE_NPC,
]


def test_npc_cscanner() -> None:
    """The city scanner becomes a claw scanner in certain maps."""
    check_entity(
        *RES_SCANNER_CITY,
        classname='npc_cscanner',
    )
    check_entity(
        *RES_SCANNER_CLAW,
        classname='npc_cscanner',
        mapname__='d3_c17_rebellion',
    )
    # This episodic classname always is the claw variant.
    check_entity(
        *RES_SCANNER_CLAW,
        classname='npc_clawscanner',
        tags__=['hl2', 'episodic'],
    )


@pytest.mark.xfail
def test_npc_citizen() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_hl2_gamerules() -> None:
    """In Mapbase, the default citizen can be overridden using the gamerules entity."""
    raise NotImplementedError


def test_npc_combinedropship() -> None:
    """The dropship can spawn with a variety of cargos."""
    common = [
        *BASE_NPC,
        Resource.mdl("models/combine_dropship.mdl"),
        Resource.snd("NPC_CombineDropship.RotorLoop"),
        Resource.snd("NPC_CombineDropship.FireLoop"),
        Resource.snd("NPC_CombineDropship.NearRotorLoop"),
        Resource.snd("NPC_CombineDropship.OnGroundRotorLoop"),
        Resource.snd("NPC_CombineDropship.DescendingWarningLoop"),
        Resource.snd("NPC_CombineDropship.NearRotorLoop"),
    ]
    # 0 = no crate. 2 = roller hopper, adds nothing. -2 picks up an APC in the map, so no new resources.
    for i in [0, 2, -2]:
        check_entity(
            *common,
            classname='npc_combinedropship',
            tags__=['hl2'],
            cratetype=i,
        )

    check_entity(
        *common,
        Resource.mdl("models/combine_dropship_container.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_01.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_02.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_03.mdl"),
        Resource.mdl("models/gibs/hgibs.mdl"),
        classname='npc_combinedropship',
        tags__=['hl2'],
        cratetype=1,  # Soldier Crate
    )
    check_entity(
        *common,
        Resource.mdl("models/combine_strider.mdl"),
        Resource.snd("NPC_Strider.StriderBusterExplode"),
        Resource.snd("explode_5"),
        Resource.snd("DoSpark"),
        Resource.snd("NPC_Strider.Charge"),
        Resource.snd("NPC_Strider.RagdollDetach"),
        Resource.snd("NPC_Strider.Whoosh"),
        Resource.snd("NPC_Strider.Creak"),
        Resource.snd("NPC_Strider.Alert"),
        Resource.snd("NPC_Strider.Pain"),
        Resource.snd("NPC_Strider.Death"),
        Resource.snd("NPC_Strider.FireMinigun"),
        Resource.snd("NPC_Strider.Shoot"),
        Resource.snd("NPC_Strider.OpenHatch"),
        Resource.snd("NPC_Strider.Footstep"),
        Resource.snd("NPC_Strider.Skewer"),
        Resource.snd("NPC_Strider.Hunt"),
        Resource.mat("materials/effects/water_highlight.vmt"),
        Resource.mat("materials/sprites/physbeam.vmt"),
        Resource.mat("materials/sprites/bluelaser1.vmt"),
        Resource.mat("materials/effects/blueblacklargebeam.vmt"),
        Resource.mat("materials/effects/strider_pinch_dudv.vmt"),
        Resource.mat("materials/effects/blueblackflash.vmt"),
        Resource.mat("materials/effects/strider_bulge_dudv.vmt"),
        Resource.mat("materials/effects/strider_muzzle.vmt"),
        Resource.mdl("models/chefhat.mdl"),
        # Concussiveblast:
        Resource.mat("materials/sprites/lgtning.vmt"),
        Resource.mat("materials/effects/blueflare1.vmt"),
        Resource.mat("materials/particle/particle_smokegrenade.vmt"),
        Resource.mat("materials/particle/particle_noisesphere.vmt"),
        classname='npc_combinedropship',
        tags__=['hl2'],
        cratetype=-1,  # Strider
    )
    check_entity(
        *common,
        Resource.mdl("models/buggy.mdl"),
        classname='npc_combinedropship',
        tags__=['hl2'],
        cratetype=-3,  # Jeep, just a visual prop.
    )


def test_npc_combinegunship() -> None:
    """The gunship can swap to a helicopter, like in Lost Coast."""
    common = [
        *BASE_NPC,
        Resource("MetalChunks", FileType.BREAKABLE_CHUNK),
        Resource.mat("materials/sprites/lgtning.vmt"),
        Resource.mat("materials/effects/ar2ground2.vmt"),
        Resource.mat("materials/effects/blueblackflash.vmt"),
        Resource.snd("NPC_CombineGunship.SearchPing"),
        Resource.snd("NPC_CombineGunship.PatrolPing"),
        Resource.snd("NPC_Strider.Charge"),
        Resource.snd("NPC_Strider.Shoot"),
        Resource.snd("NPC_CombineGunship.SeeEnemy"),
        Resource.snd("NPC_CombineGunship.CannonStartSound"),
        Resource.snd("NPC_CombineGunship.Explode"),
        Resource.snd("NPC_CombineGunship.Pain"),
        Resource.snd("NPC_CombineGunship.CannonStopSound"),
        Resource.snd("NPC_CombineGunship.DyingSound"),
        Resource.snd("NPC_CombineGunship.CannonSound"),
        Resource.snd("NPC_CombineGunship.RotorSound"),
        Resource.snd("NPC_CombineGunship.ExhaustSound"),
        Resource.snd("NPC_CombineGunship.RotorBlastSound"),
    ]

    check_entity(
        *common,
        Resource.mdl('models/gunship.mdl'),
        classname='npc_combinegunship',
        tags__=['hl2'],
    )

    # Episodic adds the citadel energy core.
    check_entity(
        *common,
        Resource.mdl("models/gunship.mdl"),
        Resource.mat("materials/sprites/physbeam.vmt"),
        Resource.mat("materials/effects/strider_muzzle.vmt"),
        Resource.mat("materials/effects/combinemuzzle2.vmt"),
        Resource.mat("materials/effects/combinemuzzle2_dark.vmt"),
        classname='npc_combinegunship',
        tags__=['hl2', 'episodic'],
    )
    check_entity(
        *common,
        Resource.mdl("models/combine_helicopter.mdl"),
        Resource.mdl("models/combine_helicopter_broken.mdl"),
        # helicopter_chunk:
        Resource.mdl("models/gibs/helicopter_brokenpiece_01.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_02.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_03.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_04_cockpit.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_05_tailfan.mdl"),
        Resource.mdl("models/gibs/helicopter_brokenpiece_06_body.mdl"),
        Resource.snd("BaseExplosionEffect.Sound"),
        Resource.snd("NPC_AttackHelicopter.Crash"),
        # shared by env_smoketrail, env_fire_trail, ar2explosion
        Resource.mat("materials/particle/particle_smokegrenade.vmt"),
        Resource.mat("materials/particle/particle_noisesphere.vmt"),
        # env_smoketrail:
        Resource.mat("materials/sprites/flamelet1.vmt"),
        Resource.mat("materials/sprites/flamelet2.vmt"),
        Resource.mat("materials/sprites/flamelet3.vmt"),
        Resource.mat("materials/sprites/flamelet4.vmt"),
        Resource.mat("materials/sprites/flamelet5.vmt"),
        classname='npc_combinegunship',
        spawnflags=8192,
        tags__=['hl2'],
    )


@pytest.mark.xfail
def test_npc_egg() -> None:
    raise NotImplementedError


def test_npc_maker() -> None:
    """npc_maker spawns another NPC, potentially with equipment."""
    check_entity(classname='npc_maker')  # No class, nothing happens.
    check_entity(
        *BASE_NPC,
        Resource.mdl("models/eli.mdl"),
        classname='npc_maker',
        npctype='npc_eli',
    )
    check_entity(
        *BASE_NPC,
        Resource.mdl("models/eli.mdl"),
        Resource.weapon_script("scripts/weapon_rpg.txt"),
        Resource.snd("Missile.Ignite"),
        Resource.snd("Missile.Accelerate"),
        Resource.mdl("models/weapons/w_missile.mdl"),
        Resource.mdl("models/weapons/w_missile_launch.mdl"),
        Resource.mdl("models/weapons/w_missile_closed.mdl"),
        Resource.mat("materials/effects/laser1_noz.vmt"),
        Resource.mat("materials/sprites/redglow1.vmt"),
        Resource.mat("materials/effects/muzzleflash1.vmt"),
        Resource.mat("materials/effects/muzzleflash2.vmt"),
        Resource.mat("materials/effects/muzzleflash3.vmt"),
        Resource.mat("materials/effects/muzzleflash4.vmt"),
        Resource.mat("materials/sprites/flamelet1.vmt"),
        Resource.mat("materials/sprites/flamelet2.vmt"),
        Resource.mat("materials/sprites/flamelet3.vmt"),
        Resource.mat("materials/sprites/flamelet4.vmt"),
        classname='npc_maker',
        npctype='npc_eli',
        additionalequipment='weapon_rpg',
    )


@pytest.mark.xfail
def test_npc_metropolice() -> None:
    raise NotImplementedError


@pytest.mark.parametrize('classname', ['npc_zassassin', 'monster_gonome'])
def test_npc_zassassin(classname: str) -> None:
    """The Zassassin / Plan B has some variants."""
    common = [
        *BASE_NPC,
        Resource.snd("Gonome.Idle"),
        Resource.snd("Gonome.Pain"),
        Resource.snd("Gonome.Alert"),
        Resource.snd("Gonome.Die"),
        Resource.snd("Gonome.Attack"),
        Resource.snd("Gonome.Bite"),
        Resource.snd("Gonome.Growl"),
        Resource.snd("Gonome.FoundEnemy"),
        Resource.snd("Gonome.RetreatMode"),
        Resource.snd("Gonome.BerserkMode"),
        Resource.snd("Gonome.RunFootstepLeft"),
        Resource.snd("Gonome.RunFootstepRight"),
        Resource.snd("Gonome.FootstepLeft"),
        Resource.snd("Gonome.FootstepRight"),
        Resource.snd("Gonome.JumpLand"),
        Resource.snd("Gonome.Eat"),
        Resource.snd("Gonome.BeginSpawnCrab"),
        Resource.snd("Gonome.EndSpawnCrab"),
        Resource.snd("npc_zassassin.kickburst"),
        Resource.part("glownome_explode"),
        # squidspit
        Resource.snd("NPC_BigMomma.SpitTouch1"),
        Resource.snd("NPC_BigMomma.SpitHit1"),
        Resource.snd("NPC_BigMomma.SpitHit2"),
        Resource.snd("Zombie.AttackHit"),
        Resource.snd("Zombie.AttackMiss"),
    ]

    check_entity(
        *common,
        Resource.mdl("models/gonome.mdl"),
        Resource.mat("materials/sprites/gonomespit.vmt"),
        classname=classname,
        ezvariant=0,  # Normal
        tags__=TAGS_EZ2,
    )
    check_entity(
        *common,
        Resource.mdl("models/xonome.mdl"),
        Resource.mat("materials/sprites/gonomespit.vmt"),
        classname=classname,
        ezvariant=1,  # Xen
        tags__=TAGS_EZ2,
    )
    check_entity(
        *common,
        Resource.mdl("models/glownome.mdl"),
        Resource.mat("materials/sprites/glownomespit.vmt"),
        Resource.mat("materials/cable/goocable.vmt"),
        Resource.part("blood_impact_blue_01"),
        classname=classname,
        ezvariant=2,  # Radiation
        tags__=TAGS_EZ2,
    )


def test_point_entity_replace() -> None:
    """This entity may spawn in another entity."""
    check_entity(
        classname='point_entity_replace',
        replacementtype=0,   # Entity name, so not packed.
        replacemententity='npc_strider',
    )
    check_entity(
        Resource.mat("materials/sprites/light_glow02_add_noz.vmt"),
        classname='point_entity_replace',
        replacementtype=1,  # Classname
        replacemententity='env_lightglow',
    )


def test_skybox_swapper() -> None:
    """Skybox swapper packs the other sides of the skybox."""
    check_entity(classname='skybox_swapper')
    check_entity(
        Resource.mat('materials/skybox/sky_purplert.vmt'),
        Resource.mat('materials/skybox/sky_purplebk.vmt'),
        Resource.mat('materials/skybox/sky_purplelf.vmt'),
        Resource.mat('materials/skybox/sky_purpleft.vmt'),
        Resource.mat('materials/skybox/sky_purpleup.vmt'),
        Resource.mat('materials/skybox/sky_purpledn.vmt'),
        classname='skybox_swapper',
        skyboxname='sky_purple',
    )


@pytest.mark.xfail
def test_prop_door_rotating() -> None:
    """Prop_door_rotating uses a special $keyvalues block in the model to specify sounds."""
    raise NotImplementedError


@pytest.mark.parametrize('cls', ['move_rope', 'keyframe_rope'])
def test_sprite_rope(cls: str) -> None:
    """move_rope/keyframe_rope can fall back to a hardocded list of materials."""
    check_entity(
        Resource.mat("materials/cable/rope_shadowdepth.vmt"),
        Resource.mat('materials/cable/cable.vmt'),
        classname=cls,
    )
    check_entity(
        Resource.mat("materials/cable/rope_shadowdepth.vmt"),
        classname=cls,
        ropematerial='custom/some_rope',  # Done by generic FGD packing.
    )
    check_entity(
        Resource.mat("materials/cable/rope_shadowdepth.vmt"),
        Resource.mat('materials/cable/cable.vmt'),
        classname=cls,
        ropematerial='custom/some_rope',  # Ignored.
        ropeshader=0,
    )
    check_entity(
        Resource.mat("materials/cable/rope_shadowdepth.vmt"),
        Resource.mat('materials/cable/rope.vmt'),
        classname=cls,
        ropematerial='custom/some_rope',  # Ignored.
        ropeshader=1,
    )
    check_entity(
        Resource.mat("materials/cable/rope_shadowdepth.vmt"),
        Resource.mat('materials/cable/chain.vmt'),
        classname=cls,
        ropeshader=2,
    )


@pytest.mark.xfail
def test_team_control_point() -> None:
    raise NotImplementedError


def test_weapon_scripts() -> None:
    """Test the handler which adds weapon scripts for every weapon_ class."""
    check_entity(
        Resource.snd("Weapon_StunStick.Activate"),
        Resource.snd("Weapon_StunStick.Deactivate"),
        Resource.weapon_script("scripts/weapon_stunstick.txt"),
        classname='weapon_stunstick',
    )
    check_entity(
        Resource.weapon_script('scripts/weapon_pulsepistol.txt'),
        classname='weapon_pulsepistol',
    )
    # Black Mesa uses DMX scripts in a different location.
    check_entity(
        Resource.weapon_script("scripts/weapon_crowbar.txt"),
        classname='weapon_crowbar',
        tags__=['hl2'],
    )
    check_entity(
        Resource.weapon_script("scripts/gameplay/weapons/weapon_crowbar.dmx"),
        classname='weapon_crowbar',
        tags__=['mesa'],
    )
