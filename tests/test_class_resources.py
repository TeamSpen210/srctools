"""Test the resource functions implemented for specific entities."""
from typing import Dict, Generator, Iterable

import pytest

from srctools.const import FileType
from srctools.fgd import FGD, Resource, ResourceCtx
from srctools.vmf import Entity, VMF, ValidKVs


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
    )
    ent_def = fgd[classname]
    actual = set(ent_def.get_resources(ctx, ent=ent, on_error=LookupError))
    assert actual == expected


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


# TODO: base_plat_train
# TODO: func_breakable
# TODO: func_breakable_surf


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


# TODO: env_headcrabcanister
# TODO: env_shooter
# TODO: env_smokestack
# TODO: item_ammo_crate
# TODO: item_teamflag
# TODO: item_healthkit
# TODO: item_healthvial
# TODO: NPCs
# TODO: npc_antlion
# TODO: npc_antlionguard
# TODO: npc_antlion_template_maker
# TODO: npc_arbeit_turret_floor
# TODO: npc_bullsquid
# TODO: npc_cscanner
# TODO: npc_clawscanner
# TODO: npc_citizen
# TODO: npc_combinedropship
# TODO: npc_combinegunship
# TODO: npc_egg
# TODO: npc_maker
# TODO: npc_metropolice
# TODO: npc_zassassin
# TODO: point_entity_replace
# TODO: skybox_swapper
# TODO: team_control_point
