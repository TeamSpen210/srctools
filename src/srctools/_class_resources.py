"""This module defines functions which list resources for various entities with special functionality."""
from typing import Callable, Dict, Final, Iterator, Mapping, Sequence, TypeVar, Union
from typing_extensions import TypeAlias
from enum import IntEnum
import itertools

from . import KeyValError, Keyvalues, NoKeyError, conv_bool, conv_float, conv_int
from .fgd import Resource, ResourceCtx
from .mdl import Model
from .packlist import FileType
from .vmf import VMF, Entity


ResGen: TypeAlias = Iterator[Union[Resource, Entity]]
ClassFunc: TypeAlias = Callable[[ResourceCtx, Entity], ResGen]
ClassFuncT = TypeVar('ClassFuncT', bound=ClassFunc)
CLASS_FUNCS: Dict[str, ClassFunc] = {}
# Dummy VMF, we create entities from this to pass back to recursively get resources for.
_blank_vmf = VMF(preserve_ids=False)

FLAG_NPC_START_EFFICENT = 1 << 4
FLAG_NPC_DROP_HEALTHKIT = 1 << 3


def cls_func(func: ClassFuncT) -> ClassFuncT:
    """Save a function to do special checks for a classname."""
    name = func.__name__
    if name in CLASS_FUNCS:
        raise ValueError(f'Class function already defined for "{name}"!')
    CLASS_FUNCS[name] = func
    return func


def button_sound(index: Union[int, str]) -> Resource:
    """Return the resource matching the hardcoded set of sounds in button ents."""
    if index:
        return Resource.snd(f'Buttons.snd{conv_int(index):d}')
    else:  # Not set, skip
        return Resource.snd('')


# Entropy Zero 2 variant constants.
EZ_VARIANT_DEFAULT: Final = 0
EZ_VARIANT_XEN: Final = 1
EZ_VARIANT_RAD: Final = 2
EZ_VARIANT_TEMPORAL: Final = 3
EZ_VARIANT_ARBEIT: Final = 4
EZ_VARIANT_BLOOD: Final = 5
EZ_VARIANT_ATHENAEUM: Final = 6
EZ_VARIANT_ASH: Final = 7


# TODO: We need to parse vehicle scripts.


@cls_func
def asw_emitter(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Complicated thing, probably can't fully process here."""
    template = ent['template']
    if template and template != 'None':
        yield Resource(f'resource/particletemplates/{template}.ptm')

    # TODO: Read the following keys from the file:
    # - "material"
    # - "glowmaterial"
    # - "collisionsound"
    # - "collisiondecal"


@cls_func
def asw_snow_volume(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Another clientside entity that spawns in emitters."""
    # TODO: Dependencies of this...
    snow_type = conv_int(ent['snowtype'])
    if snow_type == 0:
        yield Resource('resource/particletemplates/snow2.ptm')
    else:
        yield Resource('resource/particletemplates/snow3.ptm')
        if snow_type == 1:
            yield Resource('resource/particletemplates/snowclouds.ptm')


ASW_SPAWNER_ORDER = [
    'asw_drone',
    'asw_buzzer',
    'asw_parasite',
    'asw_shieldbug',
    'asw_grub',
    'asw_drone_jumper',
    'asw_harvester',
    'asw_parasite_defanged',
    'asw_queen',
    'asw_boomer',
    'asw_ranger',
    'asw_mortarbug',
    'asw_shaman',
]


@cls_func
def asw_spawner(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """The spawner spawns from an indexed list."""
    try:
        classname = ASW_SPAWNER_ORDER[int(ent['AlienClass'])]
    except (IndexError, ValueError, TypeError):
        return
    spawner_flags = conv_int(ent['spawnflags'])
    ent_flags = 1 << 2  # SF_NPC_FALL_TO_GROUND
    if spawner_flags & 4:  # ASW_SF_NEVER_SLEEP
        ent_flags |= 1 << 10  # SF_NPC_ALWAYSTHINK
    if conv_bool(ent['longrange']):
        ent_flags |= 1 << 8  # SF_NPC_LONG_RANGE
    yield _blank_vmf.create_ent(
        classname,
        spawnflags=ent_flags,
        startburrowed=ent['startburrowed'],
    )


@cls_func
def func_button_sounds(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Pack the legacy sound indexes."""
    yield button_sound(ent['sounds'])
    yield button_sound(ent['locked_sound'])
    yield button_sound(ent['unlocked_sound'])
    # TODO locked and unlocked sentences in HL1.
    # locked_sentence -> ["NA", "ND", "NF", "NFIRE", "NCHEM", "NRAD", "NCON", "NH", "NG"]
    # unlocked_sentence -> ["EA", "ED", "EF", "EFIRE", "ECHEM", "ERAD", "ECON", "EH"]


@cls_func
def func_button_timed(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This only has one sound?"""
    yield button_sound(ent['locked_sound'])


@cls_func
def momentary_rot_button(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Inherits from func_button, but doesn't always use 'sounds'."""
    if conv_int(ent['spawnflags']) & 1024:  # USE_ACTIVATES
        yield button_sound(ent['sounds'])
    yield button_sound(ent['locked_sound'])
    yield button_sound(ent['unlocked_sound'])


@cls_func
def color_correction(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Pack the color correction file."""
    yield Resource(ent['filename'], FileType.GENERIC)


# Index->sound lists for CBasePlatTrain in HL:Source.
HL1_PLAT_MOVE: Final[Sequence[str]] = [
    'Plat.DefaultMoving', 'Plat.BigElev1', 'Plat.BigElev2', 'Plat.TechElev1', 'Plat.TechElev2',
    'Plat.TechElev3', 'Plat.FreightElev1', 'Plat.FreightElev2', 'Plat.HeavyElev', 'Plat.RackElev',
    'Plat.RailElev', 'Plat.SqueakElev', 'Plat.OddElev1', 'Plat.OddElev2',
]
HL1_PLAT_STOP: Final[Sequence[str]] = [
    "Plat.DefaultArrive", "Plat.BigElevStop1", "Plat.BigElevStop2", "Plat.FreightElevStop",
    "Plat.HeavyElevStop", "Plat.RackStop", "Plat.RailStop", "Plat.SqueakStop", "Plat.QuickStop",
]


MATERIAL_GIB_TYPES: Final[Mapping[int, str]] = {
    0: 'GlassChunks',
    1: 'WoodChunks',
    2: 'MetalChunks',
    3: 'FleshGibs',
    # 4: Cinderblock -> "ConcreteChunks" or "CinderBlocks"
    5: 'CeilingTile',
    6: 'ComputerGibs',
    7: 'GlassChunks',  # Unbreakable Glass
    8: 'ConcreteChunks',  # Rocks
    # 9 = Web (episodic) or Metal Panel (P2)
}

# Classnames spawned by func_breakable.
BREAKABLE_SPAWNS: Mapping[int, str] = {
    1: "item_battery",
    2: "item_healthkit",
    3: "item_ammo_pistol",
    4: "item_ammo_pistol_large",
    5: "item_ammo_smg1",
    6: "item_ammo_smg1_large",
    7: "item_ammo_ar2",
    8: "item_ammo_ar2_large",
    9: "item_box_buckshot",
    10: "item_flare_round",
    11: "item_box_flare_rounds",
    12: "item_ml_grenade",
    13: "item_smg1_grenade",
    14: "item_box_sniper_rounds",
    # 15: "unused1",
    16: "weapon_stunstick",
    # 17: "weapon_ar1",
    18: "weapon_ar2",
    # 19: "unused2",
    20: "weapon_ml",
    21: "weapon_smg1",
    22: "weapon_smg2",
    23: "weapon_slam",
    24: "weapon_shotgun",
    # 25: "weapon_molotov",
    26: "item_dynamic_resupply",

    # Black Mesa:
    27: "item_ammo_glock",
    28: "item_ammo_mp5",
    29: "item_ammo_357",
    30: "item_ammo_crossbow",
    31: "item_ammo_shotgun",
    32: "item_ammo_energy",
    33: "item_grenade_mp5",
    34: "item_grenade_rpg",
    35: "item_weapon_357",
    36: "item_weapon_crossbow",
    37: "item_weapon_crowbar",
    38: "item_weapon_frag",
    39: "item_weapon_glock",
    40: "item_weapon_gluon",
    41: "item_weapon_hivehand",
    42: "item_weapon_mp5",
    43: "item_weapon_rpg",
    44: "item_weapon_satchel",
    45: "item_weapon_shotgun",
    46: "item_weapon_snark",
    47: "item_weapon_tau",
    48: "item_weapon_tripmine",
    49: "item_syringe",
    50: "item_ammo_box",
    51: "prop_soda",
}

# A different set in HL1.
BREAKABLE_SPAWNS_HL1: Mapping[int, str] = {
    1: "item_battery",
    2: "item_healthkit",
    3: "weapon_glock",
    4: "ammo_9mmclip",
    5: "weapon_mp5",
    6: "ammo_9mmAR",
    7: "ammo_ARgrenades",
    8: "weapon_shotgun",
    9: "ammo_buckshot",
    10: "weapon_crossbow",
    11: "ammo_crossbow",
    12: "weapon_357",
    13: "ammo_357",
    14: "weapon_rpg",
    15: "ammo_rpgclip",
    16: "ammo_gaussclip",
    17: "weapon_handgrenade",
    18: "weapon_tripmine",
    19: "weapon_satchel",
    20: "weapon_snark",
    21: "weapon_hornetgun",
}


def base_plat_train(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Check for HL1 train movement sounds."""
    if 'movesnd' in ent:
        try:
            sound = HL1_PLAT_MOVE[int(ent['movesnd'])]
        except (IndexError, TypeError, ValueError):
            pass
        else:
            yield Resource.snd(sound)
    if 'stopsnd' in ent:
        try:
            sound = HL1_PLAT_STOP[int(ent['stopsnd'])]
        except (IndexError, TypeError, ValueError):
            pass
        else:
            yield Resource.snd(sound)


@cls_func
def breakable_brush(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Breakable brushes are able to spawn specific entities."""
    mat_type = conv_int(ent['material'])
    if mat_type == 4:  # Cinderblocks - not clear which branch has what, include both.
        yield Resource('CeilingTile', FileType.BREAKABLE_CHUNK)
        yield Resource('ConcreteChunks', FileType.BREAKABLE_CHUNK)
    elif mat_type == 9:  # Web, P2 metal panel
        yield Resource('MetalPanelChunks' if 'P2' in ctx.tags else 'WebGibs', FileType.BREAKABLE_CHUNK)
    else:
        yield Resource(MATERIAL_GIB_TYPES.get(mat_type, 'WoodChunks'), FileType.BREAKABLE_CHUNK)
    object_ind = conv_int(ent['spawnobject'])
    spawns = BREAKABLE_SPAWNS_HL1 if 'HLS' in ctx.tags else BREAKABLE_SPAWNS
    # 27+ is Black Mesa exclusive.
    if object_ind < 27 or 'MESA' in ctx.tags:
        try:
            breakable_class = spawns[object_ind]
        except (KeyError, TypeError, ValueError):
            pass
        else:
            yield _blank_vmf.create_ent(breakable_class)


@cls_func
def func_breakable_surf(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Additional materials required for func_breakable_surf."""
    surf_type = conv_int(ent['surfacetype'])

    if surf_type == 1:  # Tile
        mat_type = 'tile'
    elif surf_type == 0:  # Glass
        mat_type = 'glass'
        yield Resource.mat('materials/models/brokenglass/glassbroken_solid.vmt')
    else:
        # Unknown
        return

    yield Resource.mat(f'materials/effects/fleck_{mat_type}1.vmt')
    yield Resource.mat(f'materials/effects/fleck_{mat_type}2.vmt')

    for num in '123':
        for letter in 'abcd':
            yield Resource.mat(
                f'materials/models/broken{mat_type}/'
                f'{mat_type}broken_0{num}{letter}.vmt'
            )


@cls_func
def sprite_rope(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Handles a legacy keyvalue for the material used on move_rope and keyframe_rope."""
    if 'ropeshader' in ent:
        old_shader_type = conv_int(ent['ropeshader'])
        if old_shader_type == 0:
            yield Resource.mat('materials/cable/cable.vmt')
        elif old_shader_type == 1:
            yield Resource.mat('materials/cable/rope.vmt')
        else:
            yield Resource.mat('materials/cable/chain.vmt')
    elif not ent['ropematerial']:  # If unset, default to this.
        yield Resource.mat('materials/cable/cable.vmt')


@cls_func
def env_break_shooter(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Special behaviour on the 'model' KV."""
    model_type = conv_int(ent['modeltype'])
    if model_type == 0:  # MODELTYPE_BREAKABLECHUNKS
        yield Resource(ent['model'], FileType.BREAKABLE_CHUNK)
    elif model_type == 1:  # MODELTYPE_MODEL
        yield Resource.mdl(ent['model'])
    # else: Template name, that does the resources.


@cls_func
def env_fire(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Two types of fire, with different resources."""
    fire_type = conv_int(ent['firetype'])
    if fire_type == 0:  # Natural
        flags = conv_int(ent['spawnflags'])
        if flags & 2:  # Smokeless?
            suffix = ''  # env_fire_small
        else:
            suffix = '_smoke'  # env_fire_medium_smoke
        for name in ['tiny', 'small', 'medium', 'large']:
            yield Resource.part(f'env_fire_{name}{suffix}')
    elif fire_type == 1:  # Plasma
        yield _blank_vmf.create_ent('_plasma')


@cls_func
def env_headcrabcanister(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Check if it spawns in skybox or not, and precache the headcrab."""
    flags = conv_int(ent['spawnflags'])
    if flags & 0x1 == 0:  # !SF_NO_IMPACT_SOUND
        yield Resource.snd('HeadcrabCanister.Explosion')
        yield Resource.snd('HeadcrabCanister.IncomingSound')
        yield Resource.snd('HeadcrabCanister.SkyboxExplosion')
    if flags & 0x2 == 0:  # !SF_NO_LAUNCH_SOUND
        yield Resource.snd('HeadcrabCanister.LaunchSound')
    if flags & 0x1000 == 0:  # !SF_START_IMPACTED, flying through the air.
        yield Resource.mat('materials/sprites/smoke.vmt')
        yield Resource.mdl("models/props_combine/headcrabcannister01a.mdl")
        yield Resource.mdl("models/props_combine/headcrabcannister01a_skybox.mdl")

    if flags & 0x80000 == 0:  # !SF_NO_IMPACT_EFFECTS
        yield _blank_vmf.create_ent('ar2explosion')
    if flags & 0x40000 == 0:  # if SF_REMOVE_ON_IMPACT, it'll never land or open.
        yield Resource.mdl("models/props_combine/headcrabcannister01b.mdl")
        yield Resource.snd("HeadcrabCanister.AfterLanding")
        yield Resource.snd("HeadcrabCanister.Open")
        # Also precache the appropriate headcrab's resources.
        try:
            headcrab = (
                'npc_headcrab',
                'npc_headcrab_fast',
                'npc_headcrab_poison',
            )[conv_int(ent['HeadcrabType'])]
        except IndexError:
            pass
        else:
            yield _blank_vmf.create_ent(headcrab)


SHOOTER_SOUNDS = [
    Resource.snd("Breakable.MatGlass"),
    Resource.snd("Breakable.MatWood"),
    Resource.snd("Breakable.MatMetal"),
    Resource.snd("Breakable.MatFlesh"),
    Resource.snd("Breakable.MatConcrete"),
]


@cls_func
def env_shooter(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """A hardcoded array of sounds to play."""
    try:
        yield SHOOTER_SOUNDS[conv_int(ent['shootsounds'])]
    except IndexError:
        pass

    # Valve does this same check.
    if ent['shootmodel'].casefold().endswith('.vmt'):
        yield Resource.mat(ent['shootmodel'])
    else:
        yield Resource.mdl(ent['shootmodel'])


@cls_func
def env_smokestack(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This tries using each numeric material that exists."""
    mat_base = ent['smokematerial'].casefold().replace('\\', '/')
    if not mat_base:
        return

    if mat_base.endswith('.vmt'):
        mat_base = mat_base[:-4]
    if not mat_base.startswith('materials/'):
        mat_base = 'materials/' + mat_base

    yield Resource.mat(mat_base + '.vmt')
    if 'EPISODIC' in ctx.tags:
        for i in range(1, 8):
            fname = f'{mat_base}{i}.vmt'
            if fname in ctx.fsys:
                yield Resource.mat(fname)
            else:
                break


# Mapbase adds additional models here.
# The first is Valve's, the second is the mapbase version.
AMMO_BOX_MDLS = [
    ("pistol.mdl", "pistol.mdl"),
    ("smg1.mdl", "smg1.mdl"),
    ("ar2.mdl", "ar2.mdl"),
    ("rockets.mdl", "rockets.mdl"),
    ("buckshot.mdl", "buckshot.mdl"),
    ("grenade.mdl", "grenade.mdl"),
    # Valve reused models for these three.
    ("smg1.mdl", "357.mdl"),
    ("smg1.mdl", "xbow.mdl"),
    ("ar2.mdl", "ar2alt.mdl"),

    ("smg2.mdl", "smg2.mdl"),
    # Two added by mapbase.
    ("", "slam.mdl"),
    ("", "empty.mdl"),
]


@cls_func
def item_ammo_crate(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Handle loading the specific ammo box type."""
    try:
        mdl_valve, mdl_mbase = AMMO_BOX_MDLS[int(ent['AmmoType'])]
    except (IndexError, TypeError, ValueError):
        return  # Invalid ammo type.
    model = mdl_mbase if 'MAPBASE' in ctx.tags else mdl_valve
    if model:
        yield Resource.mdl('models/items/ammocrate_' + model)

    if model == 'grenade.mdl':
        yield _blank_vmf.create_ent('weapon_frag')
    elif model == 'slam.mdl':
        yield _blank_vmf.create_ent('weapon_slam')


@cls_func
def item_item_crate(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Item crates can spawn another arbitary entity."""
    appearance = conv_int(ent['crateappearance'])
    if appearance == 0:  # Default
        yield Resource.mdl('models/items/item_item_crate.mdl')
    elif appearance == 1:  # Beacon
        yield Resource.mdl('models/items/item_beacon_crate.mdl')
    elif appearance == 2:  # Mapbase custom model
        yield Resource.mdl(ent['model'])
    if conv_int(ent['cratetype']) == 0 and ent['itemclass']:  # "Specific Item"
        spawned = _blank_vmf.create_ent(ent['itemclass'])
        if 'ezvariant' in ent and 'ENTROPYZERO2' in ctx.tags:
            spawned['ezvariant'] = ent['ezvariant']
        yield spawned


@cls_func
def item_teamflag(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This item has several special team-specific options."""
    for kvalue, prefix in [
        ('flag_icon', 'materials/vgui/'),
        ('flag_trail', 'materials/effects/')
    ]:
        value = ent[kvalue]
        if value:
            # Allow going up one directory - this is done by Valve in the icon.
            # Going up two outside materials/ doesn't make any sense.
            if value.startswith(('../', '..\\')):
                folder = f'materials/{value[3:]}'
            else:
                folder = prefix + value
            yield Resource.mat(folder + '_red.vmt')
            yield Resource.mat(folder + '_blue.vmt')


EZ_MODEL_FOLDERS = [
    # model folder, skin, sound folder
    ('', 0, ''),  # Normal
    ('xen/', 0, '_Xen'),
    ('arbeit/', 1, '_Rad'),
    ('temporal/', 0, '_Temporal'),
    ('arbeit/', 0, '_Arbeit'),
    ('blood/', 0, '_Blood'),
    ('athenaeum/', 0, '_Athenaeum'),
    ('ash/', 0, '_Ash'),
]


@cls_func
def item_ammo_ar2_altfire(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """AR2 energy ball ammo has multiple variants in EZ2."""
    if 'ezvariant' not in ent:
        return
    variant = conv_int(ent['ezvariant'])
    model, skin, snd = EZ_MODEL_FOLDERS[variant]

    yield Resource.mdl(f'models/items/{model}combine_rifle_ammo01.mdl#{skin}')


@cls_func
def item_battery(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Suit batteries have multiple variants in EZ2."""
    if 'ezvariant' not in ent:
        return
    variant = conv_int(ent['ezvariant'])
    model, skin, snd = EZ_MODEL_FOLDERS[variant]

    yield Resource.mdl(f'models/items/{model}battery.mdl#{skin}')


@cls_func
def item_healthkit(ctx: ResourceCtx, ent: Entity, kind: str = 'kit') -> ResGen:
    """Healthkits have multiple variants in EZ2."""
    if 'ezvariant' not in ent:
        return
    variant = conv_int(ent['ezvariant'])
    model, skin, snd = EZ_MODEL_FOLDERS[variant]

    if kind == 'vial' and model == '':
        # Special case, the regular model is not in items.
        yield Resource.mdl('models/healthvial.mdl#0')
    else:
        yield Resource.mdl(f'models/items/{model}health{kind}.mdl#{skin}')
    yield Resource.snd(f'Health{kind.title()}{snd}.Touch')


@cls_func
def item_healthvial(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Health vials also have multiple variants in EZ2."""
    return item_healthkit(ctx, ent, 'vial')


@cls_func
def base_npc(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Resources precached in CAI_BaseNPC & CBaseCombatCharacter."""
    spawnflags = conv_int(ent['spawnflags'])

    if conv_int(ent['ezvariant']) == EZ_VARIANT_TEMPORAL:
        yield Resource.snd('NPC_TemporalHeadcrab.Vanish')
        yield Resource.snd('NPC_TemporalHeadcrab.Appear')
        yield Resource.snd('ShadowCrab_Vanish')
        yield Resource.snd('ShadowCrab_Appear')
    equipment = ent['additionalequipment']
    if equipment not in ('', '0'):
        yield _blank_vmf.create_ent(equipment)
    if spawnflags & FLAG_NPC_DROP_HEALTHKIT:
        yield _blank_vmf.create_ent('item_healthkit')


ANT_WORKER_RESOURCES = [
    Resource.part("blood_impact_antlion_worker_01"),
    Resource.part("antlion_gib_02"),
    Resource.part("blood_impact_yellow_01"),
    Resource.snd("NPC_Antlion.PoisonBurstScream"),
    Resource.snd("NPC_Antlion.PoisonBurstScreamSubmerged"),
    Resource.snd("NPC_Antlion.PoisonBurstExplode"),
    Resource.snd("NPC_Antlion.PoisonShoot"),
    Resource.snd("NPC_Antlion.PoisonBall"),
]


@cls_func
def npc_antlion(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Antlions require different resources for the worker version."""
    ez_variant = conv_int(ent['ezvariant'])
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 18):  # Is worker?
        if ez_variant == EZ_VARIANT_BLOOD:
            yield Resource.mdl("models/bloodlion_worker.mdl")
        else:
            yield Resource.mdl("models/antlion_worker.mdl")
        yield from ANT_WORKER_RESOURCES
        yield _blank_vmf.create_ent('grenade_spit')
    else:  # Regular antlion.
        if ez_variant == EZ_VARIANT_RAD:
            yield Resource.mdl("models/antlion_blue.mdl")
            yield Resource.part("blood_impact_blue_01")
        elif ez_variant == EZ_VARIANT_XEN:
            yield Resource.mdl("models/antlion_xen.mdl")
            yield Resource.part("blood_impact_antlion_01")
        elif ez_variant == EZ_VARIANT_BLOOD:
            yield Resource.mdl("models/bloodlion.mdl")
            yield Resource.part("blood_impact_antlion_01")
        else:
            yield Resource.mdl("models/antlion.mdl")
            yield Resource.part("blood_impact_antlion_01")
        yield Resource.part("AntlionGib")
        for size, i in itertools.product(("small", "medium", "large"), (1, 2, 3)):
            yield Resource.mdl(f"models/gibs/antlion_gib_{size}_{i}.mdl")


@cls_func
def npc_antlionguard(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """In Entropy Zero, some alternate models are available."""
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 17):  # Inside Footsteps
        yield Resource.snd("NPC_AntlionGuard.Inside.StepLight")
        yield Resource.snd("NPC_AntlionGuard.Inside.StepHeavy")
    else:
        yield Resource.snd("NPC_AntlionGuard.StepLight")
        yield Resource.snd("NPC_AntlionGuard.StepHeavy")
    if 'ezvariant' in ent:  # Entropy Zero.
        variant = conv_int(ent['ezvaraiant'])
        if variant == EZ_VARIANT_XEN:
            yield Resource.mdl("models/antlion_guard_xen.mdl")
            yield Resource.part("xenpc_spawn")
        elif variant == EZ_VARIANT_RAD:
            yield Resource.mdl("models/antlion_guard_blue.mdl")
            yield Resource.part("blood_impact_blue_01")
        elif variant == EZ_VARIANT_BLOOD:
            yield Resource.mdl("models/bloodlion_guard.mdl")
        else:
            yield Resource.mdl("models/antlion_guard.mdl")
    else:  # Regular HL2.
        yield Resource.mdl("models/antlion_guard.mdl")


@cls_func
def npc_antlion_template_maker(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Depending on KVs this may or may not spawn workers."""
    # There will be an antlion present in the map, as the template NPC. So we don't need to add
    # those resources.
    if conv_float(ent['workerspawnrate']) > 0.0:
        # It randomly spawns worker antlions, so load that resource set.
        yield Resource.mdl("models/bloodlion_worker.mdl", frozenset(['entropyzero2']))
        yield Resource.mdl("models/antlion_worker.mdl")
        yield from ANT_WORKER_RESOURCES
        yield _blank_vmf.create_ent('grenade_spit')
    if conv_bool(ent['createspores']):
        yield _blank_vmf.create_ent('env_sporeexplosion')


@cls_func
def npc_arbeit_turret_floor(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Arbeit/Aperture turrets have EZ variants."""
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        yield Resource.mdl('models/props/glowturret_01.mdl')
        yield Resource.mat("materials/cable/goocable.vmt")
    elif variant == EZ_VARIANT_ARBEIT:
        yield Resource.mdl('models/props/camoturret_01.mdl')
        yield Resource.mdl('models/props/camoturret_02.mdl')
    elif conv_int(ent['spawnflags']) & 0x200:  # Citizen Modified
        yield Resource.mdl('models/props/hackedturret_01.mdl')
    else:
        yield Resource.mdl('models/props/turret_01.mdl')


@cls_func
def npc_bullsquid(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This has various EZ variants."""
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_XEN:
        yield Resource.mdl('models/bullsquid_xen.mdl')
        yield Resource.mdl('models/babysquid_xen.mdl')
        yield Resource.mdl('models/bullsquid_egg_xen.mdl')
        yield Resource.part('blood_impact_yellow_01')
    elif variant == EZ_VARIANT_RAD:
        yield Resource.mdl('models/bullsquid_rad.mdl')
        yield Resource.mdl('models/babysquid_rad.mdl')
        yield Resource.mdl('models/bullsquid_egg_rad.mdl')
        yield Resource.part('blood_impact_blue_01')
    else:
        yield Resource.mdl('models/bullsquid.mdl')
        yield Resource.mdl('models/babysquid.mdl')
        yield Resource.mdl('models/bullsquid_egg.mdl')
        yield Resource.part('blood_impact_yellow_01')


@cls_func
def combine_scanner(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Detect the kind of scanner (city or shield/claw), then pick the right resources."""
    if ent['classname'] == 'npc_clawscanner':  # Added in episodic, always the shield scanner.
        is_shield = True
    else:  # It checks the map name directly to determine this.
        is_shield = ctx.mapname.lower().startswith('d3_c17')
    if is_shield:
        yield Resource.mdl("models/shield_scanner.mdl")
        for i in range(1, 7):
            yield Resource.mdl(f"models/gibs/Shield_Scanner_Gib{i}.mdl")
        snd_prefix = 'NPC_SScanner.'
    else:
        yield Resource.mdl("models/combine_scanner.mdl")
        for i in [1, 2, 4, 5]:  # No gib 3!
            yield Resource.mdl(f"models/gibs/scanner_gib{i:02}.mdl")
        snd_prefix = 'NPC_CScanner.'

    for snd_name in [
        "Shoot", "Alert", "Die", "Combat", "Idle", "Pain", "TakePhoto", "AttackFlash",
        "DiveBombFlyby", "DiveBomb", "DeployMine", "FlyLoop",
    ]:
        yield Resource.snd(snd_prefix + snd_name)


CIT_HEADS = [
    "female_01.mdl",
    "female_02.mdl",
    "female_03.mdl",
    "female_04.mdl",
    "female_06.mdl",
    "female_07.mdl",
    "male_01.mdl",
    "male_02.mdl",
    "male_03.mdl",
    "male_04.mdl",
    "male_05.mdl",
    "male_06.mdl",
    "male_07.mdl",
    "male_08.mdl",
    "male_09.mdl",
]


class CitizenTypes(IntEnum):
    """npc_citizen type enum."""
    DEFAULT = 0
    DOWNTRODDEN = 1
    REFUGEE = 2
    REBEL = 3
    UNIQUE = 4
    EZ2_BRUTE = 5
    EZ2_LONGFALL = 6
    EZ2_ARCTIC = 7
    EZ2_ARBEIT = 8  # Pre-war Arbeit employees
    EZ2_ARBEIT_SEC = 9  # Pre-war Arbeit security guards

CIT_MAPNAMES = [
    ("trainstation", CitizenTypes.DOWNTRODDEN),
    ("canals", CitizenTypes.REFUGEE),
    ("town", CitizenTypes.REFUGEE),
    ("coast", CitizenTypes.REFUGEE),
    ("prison", CitizenTypes.DOWNTRODDEN),
    ("c17", CitizenTypes.REBEL),
    ("citadel", CitizenTypes.DOWNTRODDEN),
]

CIT_FOLDERS = {
    CitizenTypes.DOWNTRODDEN: 'Group01',
    CitizenTypes.REFUGEE: 'Group02',
    CitizenTypes.REBEL: 'Group03{}',
    CitizenTypes.EZ2_BRUTE: 'Group03b{}',
    CitizenTypes.EZ2_LONGFALL: 'Group03x{}',
    CitizenTypes.EZ2_ARCTIC: 'Group04{}',
    CitizenTypes.EZ2_ARBEIT: 'Group05',
    CitizenTypes.EZ2_ARBEIT_SEC: 'Group05b',
}


@cls_func
def npc_citizen(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Citizens have a complex set of precaching rules."""
    if ent['targetname'] == 'matt':
        # Special crowbar.
        yield Resource.mat("models/props_canal/mattpipe.mdl")

    spawnflags = conv_int(ent['spawnflags'])

    if spawnflags & FLAG_NPC_START_EFFICENT:
        yield Resource.mdl("models/humans/male_cheaple.mdl")

    try:
        cit_type = CitizenTypes(conv_int(ent['citizentype']))
    except ValueError:
        cit_type = CitizenTypes.DEFAULT

    if cit_type is CitizenTypes.DEFAULT:
        mapname = ctx.mapname.casefold()
        for group_name, poss_type in CIT_MAPNAMES:
            if group_name in mapname:
                cit_type = poss_type
                break
        else:
            cit_type = CitizenTypes.DOWNTRODDEN

    if 'ez2' in ctx.tags:
        can_be_medic = bool((1 << 17) & spawnflags)
    else:
        can_be_medic = cit_type is CitizenTypes.REBEL

    yield from citizen_resources(cit_type, can_be_medic)


def citizen_resources(cit_type: CitizenTypes, can_be_medic: bool) -> ResGen:
    """Yield the resources for a specific citizen type."""
    if 'EZ2' in cit_type.name:
        # Citizens may pull this out if disarmed.
        yield _blank_vmf.create_ent('weapon_css_glock')

    if cit_type is CitizenTypes.UNIQUE:  # Uses model in KVs directly.
        return

    folder = CIT_FOLDERS[cit_type]

    for head in CIT_HEADS:
        filename = f'models/humans/{folder}/{head}'
        yield Resource.mdl(filename.format(''))
        if can_be_medic:
            yield Resource.mdl(filename.format('m'))


@cls_func
def hl2_gamerules(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """In Mapbase, hl2_gamerules allows overriding the map-specific default citizen."""
    try:
        cit_type = CitizenTypes(conv_int(ent['DefaultCitizenType']))
    except ValueError:
        pass
    else:
        if cit_type is not CitizenTypes.DEFAULT and cit_type is not CitizenTypes.UNIQUE:
            yield from citizen_resources(cit_type, True)


@cls_func
def npc_combinedropship(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """The Combine Dropship may spawn with a variety of cargo types."""
    cargo_type = conv_int(ent['cratetype'])
    if cargo_type == -3:  # Spawns a prop_dynamic Jeep
        yield Resource.mdl("models/buggy.mdl")
    elif cargo_type == -1:  # Strider
        yield _blank_vmf.create_ent('npc_strider')
    elif cargo_type == 1:  # Soldiers in a container.
        yield _blank_vmf.create_ent('prop_dropship_container')
    # Other valid values:
    # -2 = Grabs the APC specified in KVs - that'll load its own resources.
    #  0 = Roller Hopper, does nothing
    #  2 = No cargo


@cls_func
def npc_combinegunship(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This has the ability to spawn as the helicopter instead."""
    if conv_int(ent['spawnflags']) & (1 << 13):
        yield Resource.mdl("models/combine_helicopter.mdl")
        yield Resource.mdl("models/combine_helicopter_broken.mdl")
        yield _blank_vmf.create_ent('helicopter_chunk')
    else:
        yield Resource.mdl("models/gunship.mdl")


@cls_func
def npc_egg(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """These are EZ2 bullsquid eggs, which spawn a specific EZ variant."""
    yield _blank_vmf.create_ent('npc_bullsquid', ezvariant=ent['ezvariant'])


@cls_func
def npc_maker(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """We spawn the NPC automatically."""
    to_spawn = ent['npctype']
    if to_spawn:
        yield _blank_vmf.create_ent(
            to_spawn,
            # Pass this along, it should then pack that too.
            additionalequipment=ent['additionalequipment'],
        )


@cls_func
def npc_metropolice(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """If a spawnflag is set, a cheap model is used."""
    if conv_int(ent['spawnflags']) & 16:
        yield Resource.mdl("models/police_cheaple.mdl")
    else:
        yield Resource.mdl("models/police.mdl")


@cls_func
def npc_zassassin(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Entropy Zero 2's "Plan B"/Gonome. """
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        yield Resource.mdl('models/glownome.mdl')
        yield Resource.part('blood_impact_blue_01')
        yield Resource.mat('materials/cable/goocable.vmt')
        yield Resource.mat('materials/sprites/glownomespit.vmt')
    else:
        yield Resource.mat('materials/sprites/gonomespit.vmt')
        if variant == EZ_VARIANT_XEN:
            yield Resource.mdl('models/xonome.mdl')
        else:
            yield Resource.mdl('models/gonome.mdl')


@cls_func
def point_entity_replace(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """In one specific mode, an entity is spawned by classname."""
    if conv_int(ent['replacementtype']) == 1:
        yield _blank_vmf.create_ent(ent['replacemententity'])


@cls_func
def prop_door_rotating(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Parse the special door_options block."""
    try:
        mdl = Model(ctx.fsys, ctx.fsys[ent['model']])
        kv = Keyvalues.parse(mdl.keyvalues, single_line=True).find_key('door_options')
    except (OSError, KeyValError, NoKeyError):
        return
    skin = kv.find_key(f'skin{ent["skin"]}', or_blank=True)
    hardware_key = kv.find_key(f'hardware{ent["hardware"]}', or_blank=True)
    defaults = kv.find_key('defaults', or_blank=True)
    for key in ['open', 'close', 'move']:
        try:
            yield Resource.snd(skin[key])
        except LookupError:
            yield Resource.snd(defaults[key, ''])
    for key in ['locked', 'unlocked']:
        try:
            yield Resource.snd(hardware_key[key])
        except LookupError:
            yield Resource.snd(defaults[key, ''])


@cls_func
def skybox_swapper(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This needs to pack a skybox."""
    sky_name = ent['skyboxname']
    if not sky_name:
        return
    for suffix in ['bk', 'dn', 'ft', 'lf', 'rt', 'up']:
        yield Resource.mat(f'materials/skybox/{sky_name}{suffix}.vmt')


@cls_func
def team_control_point(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Special '_locked' materials."""
    for kvalue in ['team_icon_0', 'team_icon_1', 'team_icon_2']:
        icon = ent[kvalue]
        if icon:
            yield Resource.mat(f'materials/{icon}.vmt')
            yield Resource.mat(f'materials/{icon}_locked.vmt')
