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
# TODO: move_rope & keyframe_rope
# TODO: env_break_shooter
# TODO: env_fire
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
