"""Test the resource functions implemented for specific entities."""
from typing import Dict, Generator, Iterable, List, Mapping, Union
from operator import itemgetter

import pytest

from srctools import EmptyMapping
from srctools.const import FileType
from srctools.fgd import FGD, Resource, ResourceCtx
from srctools.filesys import VirtualFileSystem
from srctools.vmf import VMF, Entity, ValidKVs


fgd: FGD


@pytest.fixture(scope='module', autouse=True)
def fgd_db() -> Generator:
    """Get the FGD database, cached for the module."""
    global fgd
    fgd = FGD.engine_dbase()
    yield
    del fgd  # Delete from memory
    

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
        fgd=fgd,
        tags=tags__,
        mapname=mapname__,
        fsys=VirtualFileSystem(filesys__),
    )
    ent_def = fgd[classname]
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



@pytest.mark.xfail
def test_base_plat_train() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_func_breakable() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_func_breakable_surf() -> None:
    raise NotImplementedError


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


@pytest.mark.xfail
def test_env_headcrabcanister() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_env_shooter() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_env_smokestack() -> None:
    raise NotImplementedError


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
    4:"models/items/ammocrate_buckshot.mdl",
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
        classname='item_ammo_crate',
        ammotype=5,
        tags__=tags,
    )

    # And Mapbase adds the SLAM which does the same:
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


@pytest.mark.xfail
def test_item_teamflag() -> None:
    raise NotImplementedError


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
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthKit_Xen.Touch'),
        Resource.mdl('models/items/xen/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=1,
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthKit_Rad.Touch'),
        Resource.mdl('models/items/arbeit/healthkit.mdl#1'),
        classname='item_healthkit',
        ezvariant=2,
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthKit_Temporal.Touch'),
        Resource.mdl('models/items/temporal/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=3,
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthKit_Arbeit.Touch'),
        Resource.mdl('models/items/arbeit/healthkit.mdl#0'),
        classname='item_healthkit',
        ezvariant=4,
        tags__=['entropyzero2', 'hl2'],
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
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthVial_Xen.Touch'),
        Resource.mdl('models/items/xen/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=1,
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthVial_Rad.Touch'),
        Resource.mdl('models/items/arbeit/healthvial.mdl#1'),
        classname='item_healthvial',
        ezvariant=2,
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthVial_Temporal.Touch'),
        Resource.mdl('models/items/temporal/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=3,
        tags__=['entropyzero2', 'hl2'],
    )
    check_entity(
        Resource.snd('HealthVial_Arbeit.Touch'),
        Resource.mdl('models/items/arbeit/healthvial.mdl#0'),
        classname='item_healthvial',
        ezvariant=4,
        tags__=['entropyzero2', 'hl2'],
    )


@pytest.mark.xfail
def test_NPCs() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_antlion() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_antlionguard() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_antlion_template_maker() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_arbeit_turret_floor() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_bullsquid() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_cscanner() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_clawscanner() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_citizen() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_combinedropship() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_combinegunship() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_egg() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_maker() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_metropolice() -> None:
    raise NotImplementedError


@pytest.mark.xfail
def test_npc_zassassin() -> None:
    raise NotImplementedError


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
def test_team_control_point() -> None:
    raise NotImplementedError
