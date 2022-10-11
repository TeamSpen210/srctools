"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in keyvalues.
The only public values are CLASS_RESOURCES and ALT_NAMES, but those
should be imported from packlist instead.
"""
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, TypeVar, Union
from typing_extensions import Final, TypeAlias
import itertools

from .. import conv_bool, conv_int
from ..packlist import FileType, PackList
from ..fgd import ResourceCtx, Resource
from ..vmf import VMF, Entity, ValidKVs


#  For various entity classes, we know they require hardcoded files.
# List them here - classname -> [(file, type), ...]
# Additionally or instead you could have a function to call with the
# entity to do class-specific behaviour, yielding files to pack.


ResGen: TypeAlias = Iterator[Union[Resource, Entity]]
ClassFunc: TypeAlias = Callable[[ResourceCtx, Entity], ResGen]
ClassFuncT = TypeVar('ClassFuncT', bound=ClassFunc)
CLASS_FUNCS: Dict[str, ClassFunc] = {}
_blank_vmf = VMF(preserve_ids=False)


def res(
    cls: str,
    *items: Union[str, Tuple[str, FileType]],
    includes: str='',
    aliases: str='',
    func: Optional[ClassFunc] = None,
) -> None:
    """Add a class to class_resources, with each of the files it always uses.

    :param cls: The classname to register.
    :param includes: This adds the resources of the other ent to this one if we spawn another.
    :param aliases: This indicates additional classnames which are identical to ours.
    :param items: The items to pack.
    :param func: A function to call to do special additional packing checks.
    """
    pass


def cls_func(func: ClassFuncT) -> ClassFuncT:
    """Save a function to do special checks for a classname."""
    name = func.__name__
    if name in CLASS_FUNCS:
        raise ValueError(f'Class function already defined for "{name}"!')
    CLASS_FUNCS[name] = func
    return func


def button_sound(index: Union[int, str]) -> Resource:
    """Return the resource matching the hardcoded set of sounds in button ents."""
    return Resource(f'Buttons.snd{conv_int(index):d}', FileType.GAME_SOUND)


# Entropy Zero 2 variant constants.
EZ_VARIANT_DEFAULT: Final = 0
EZ_VARIANT_XEN: Final = 1
EZ_VARIANT_RAD: Final = 2
EZ_VARIANT_TEMPORAL: Final = 3
EZ_VARIANT_ARBEIT: Final = 4
EZ_VARIANT_BLOOD: Final = 5


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
            yield Resource(sound, FileType.GAME_SOUND)
    if 'stopsnd' in ent:
        try:
            sound = HL1_PLAT_STOP[int(ent['stopsnd'])]
        except (IndexError, TypeError, ValueError):
            pass
        else:
            yield Resource(sound, FileType.GAME_SOUND)


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
        except (IndexError, TypeError, ValueError):
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
        yield Resource('materials/models/brokenglass/glassbroken_solid.vmt', FileType.MATERIAL)
    else:
        # Unknown
        return

    yield Resource(f'materials/effects/fleck_{mat_type}1.vmt', FileType.MATERIAL)
    yield Resource(f'materials/effects/fleck_{mat_type}2.vmt', FileType.MATERIAL)

    for num in '123':
        for letter in 'abcd':
            yield Resource(
                f'materials/models/broken{mat_type}/{mat_type}broken_0{num}{letter}.vmt',
                FileType.MATERIAL,
            )


def sprite_rope(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Handles a legacy keyvalue for the material used on move_rope and keyframe_rope."""
    if 'ropeshader' in ent:
        old_shader_type = conv_int(ent['ropeshader'])
        if old_shader_type == 0:
            yield Resource('materials/cable/cable.vmt', FileType.MATERIAL)
        elif old_shader_type == 1:
            yield Resource('materials/cable/rope.vmt', FileType.MATERIAL)
        else:
            yield Resource('materials/cable/chain.vmt', FileType.MATERIAL)


@cls_func
def env_break_shooter(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Special behaviour on the 'model' KV."""
    model_type = conv_int(ent['modeltype'])
    if model_type == 0:  # MODELTYPE_BREAKABLECHUNKS
        yield Resource(ent['model'], FileType.BREAKABLE_CHUNK)
    elif model_type == 1:  # MODELTYPE_MODEL
        yield Resource(ent['model'], FileType.MODEL)
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
            yield Resource(f'env_fire_{name}{suffix}', FileType.PARTICLE_SYSTEM)
    elif fire_type == 1:  # Plasma
        yield _blank_vmf.create_ent('_plasma')


@cls_func
def env_headcrabcanister(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Check if it spawns in skybox or not, and precache the headcrab."""
    flags = conv_int(ent['spawnflags'])
    if flags & 0x1 == 0:  # !SF_NO_IMPACT_SOUND
        yield Resource('HeadcrabCanister.Explosion', FileType.GAME_SOUND)
        yield Resource('HeadcrabCanister.IncomingSound', FileType.GAME_SOUND)
        yield Resource('HeadcrabCanister.SkyboxExplosion', FileType.GAME_SOUND)
    if flags & 0x2 == 0:  # !SF_NO_LAUNCH_SOUND
        yield Resource('HeadcrabCanister.LaunchSound', FileType.GAME_SOUND)
    if flags & 0x1000 == 0:  # !SF_START_IMPACTED
        yield Resource('materials/sprites/smoke.vmt', FileType.MATERIAL)

    if flags & 0x80000 == 0:  # !SF_NO_IMPACT_EFFECTS
        yield Resource('particle/particle_noisesphere', FileType.MATERIAL)  # AR2 explosion
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
    Resource("Breakable.MatGlass", FileType.GAME_SOUND),
    Resource("Breakable.MatWood", FileType.GAME_SOUND),
    Resource("Breakable.MatMetal", FileType.GAME_SOUND),
    Resource("Breakable.MatFlesh", FileType.GAME_SOUND),
    Resource("Breakable.MatConcrete", FileType.GAME_SOUND),
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
        yield Resource(ent['shootmodel'], FileType.MATERIAL)
    else:
        yield Resource(ent['shootmodel'], FileType.MODEL)


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

    yield Resource(mat_base + '.vmt', FileType.MATERIAL)
    for i in itertools.count(1):
        fname = f'{mat_base}{i}.vmt'
        if fname in ctx.fsys:
            yield Resource(fname, FileType.MATERIAL)
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
    ("ar2.mdl",  "ar2alt.mdl"),

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
        yield Resource('models/items/ammocrate_' + model, FileType.MODEL)

    if model == 'grenade.mdl':
        yield _blank_vmf.create_ent('weapon_frag')
    elif model == 'slam.mdl':
        yield _blank_vmf.create_ent('weapon_slam')


@cls_func
def item_item_crate(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Item crates can spawn another arbitary entity."""
    appearance = conv_int(ent['crateappearance'])
    if appearance == 0:  # Default
        yield Resource('models/items/item_item_crate.mdl', FileType.MODEL)
    elif appearance == 1:  # Beacon
        yield Resource('models/items/item_beacon_crate.mdl', FileType.MODEL)
    # else: 2 = Mapbase custom model, that'll be packed automatically.
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
        value = prefix + ent[kvalue]
        if value != prefix:
            yield Resource(value + '.vmt', FileType.MATERIAL)
            yield Resource(value + '_red.vmt', FileType.MATERIAL)
            yield Resource(value + '_blue.vmt', FileType.MATERIAL)


EZ_HEALTH_FOLDERS = [
    # model folder, skin, sound folder
    ('', 0, ''),  # Normal
    ('xen/', 0, '_Xen'),
    ('arbeit/', 1, '_Rad'),
    ('temporal/', 0, '_Temporal'),
    ('arbeit/', 0, '_Arbeit'),
]


@cls_func
def item_healthkit(ctx: ResourceCtx, ent: Entity, kind: str='kit') -> ResGen:
    """Healthkits have multiple variants in EZ2."""
    if 'ezvariant' not in ent:
        return
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_BLOOD:  # Causes a segfault.
        ent['ezvariant'] = variant = EZ_VARIANT_DEFAULT
    model, skin, snd = EZ_HEALTH_FOLDERS[variant]

    yield Resource(f'models/items/{model}health{kind}.mdl#{skin}', FileType.MODEL)
    yield Resource(f'Health{kind.title()}{snd}.Touch', FileType.GAME_SOUND)


@cls_func
def item_healthvial(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Health vials also have multiple variants in EZ2."""
    return item_healthkit(ctx, ent, 'vial')


@cls_func
def base_npc(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Resources precached in CAI_BaseNPC."""
    if conv_int(ent['ezvariant']) == EZ_VARIANT_TEMPORAL:
        yield Resource('NPC_TemporalHeadcrab.Vanish', FileType.GAME_SOUND)
        yield Resource('NPC_TemporalHeadcrab.Appear', FileType.GAME_SOUND)
        yield Resource('ShadowCrab_Vanish', FileType.PARTICLE_SYSTEM)
        yield Resource('ShadowCrab_Appear', FileType.PARTICLE_SYSTEM)
    equipment = ent['additionalequipment']
    if equipment not in ('', '0'):
        yield _blank_vmf.create_ent(equipment)


@cls_func
def npc_antlion(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Antlions require different resources for the worker version."""
    ez_variant = conv_int(ent['ezvariant'])
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 18):  # Is worker?
        if ez_variant == EZ_VARIANT_BLOOD:
            yield Resource("models/bloodlion_worker.mdl", FileType.MODEL)
        else:
            yield Resource("models/antlion_worker.mdl", FileType.MODEL)
        yield Resource("blood_impact_antlion_worker_01", FileType.PARTICLE_SYSTEM)
        yield Resource("antlion_gib_02", FileType.PARTICLE_SYSTEM)
        yield Resource("blood_impact_yellow_01", FileType.PARTICLE_SYSTEM)

        yield _blank_vmf.create_ent('grenade_spit')
    else:  # Regular antlion.
        if ez_variant == EZ_VARIANT_RAD:
            yield Resource("models/antlion_blue.mdl", FileType.MODEL)
            yield Resource("blood_impact_blue_01", FileType.PARTICLE_SYSTEM)
        elif ez_variant == EZ_VARIANT_XEN:
            yield Resource("models/antlion_xen.mdl", FileType.MODEL)
            yield Resource("blood_impact_antlion_01", FileType.PARTICLE_SYSTEM)
        elif ez_variant == EZ_VARIANT_BLOOD:
            yield Resource("models/bloodlion.mdl", FileType.MODEL)
            yield Resource("blood_impact_antlion_01", FileType.PARTICLE_SYSTEM)
        else:
            yield Resource("models/antlion.mdl", FileType.MODEL)
            yield Resource("blood_impact_antlion_01", FileType.PARTICLE_SYSTEM)
        yield Resource("AntlionGib", FileType.PARTICLE_SYSTEM)


@cls_func
def npc_antlionguard(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """In Entropy Zero, some alternate models are available."""
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 17):  # Inside Footsteps
        yield Resource("NPC_AntlionGuard.Inside.StepLight", FileType.GAME_SOUND)
        yield Resource("NPC_AntlionGuard.Inside.StepHeavy", FileType.GAME_SOUND)
    else:
        yield Resource("NPC_AntlionGuard.StepLight", FileType.GAME_SOUND)
        yield Resource("NPC_AntlionGuard.StepHeavy", FileType.GAME_SOUND)
    if 'ezvariant' in ent:  # Entropy Zero.
        variant = conv_int(ent['ezvaraiant'])
        if variant == EZ_VARIANT_XEN:
            yield Resource("models/antlion_guard_xen.mdl", FileType.MODEL)
            yield Resource("xenpc_spawn", FileType.PARTICLE_SYSTEM)
        elif variant == EZ_VARIANT_RAD:
            yield Resource("models/antlion_guard_blue.mdl", FileType.MODEL)
            yield Resource("blood_impact_blue_01", FileType.PARTICLE_SYSTEM)
        elif variant == EZ_VARIANT_BLOOD:
            yield Resource("models/bloodlion_guard.mdl", FileType.MODEL)
        else:
            yield Resource("models/antlion_guard.mdl", FileType.MODEL)
    else:  # Regular HL2.
        yield Resource("models/antlion_guard.mdl", FileType.MODEL)


@cls_func
def npc_antlion_template_maker(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Depending on KVs this may or may not spawn workers."""
    # There will be an antlion present in the map, as the template
    # NPC. So we don't need to add those resources.
    if conv_int(ent['workerspawnrate']) > 0:
        # It randomly spawns worker antlions, so load that resource set.
        yield Resource("models/antlion_worker.mdl", FileType.MODEL)
        yield Resource("blood_impact_antlion_worker_01", FileType.PARTICLE)
        yield Resource("antlion_gib_02", FileType.PARTICLE)
        yield Resource("blood_impact_yellow_01", FileType.PARTICLE)

        yield _blank_vmf.create_ent('grenade_spit')
    if conv_bool(ent['createspores']):
        yield _blank_vmf.create_ent('env_sporeexplosion')


@cls_func
def npc_arbeit_turret_floor(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Arbeit/Aperture turrets have EZ variants."""
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        yield Resource('models/props/glowturret_01.mdl', FileType.MODEL)
    elif variant == EZ_VARIANT_ARBEIT:
        yield Resource('models/props/camoturret_01.mdl', FileType.MODEL)
        yield Resource('models/props/camoturret_02.mdl', FileType.MODEL)
    elif conv_int(ent['spawnflags']) & 0x200:  # Citizen Modified
        yield Resource('models/props/hackedturret_01.mdl', FileType.MODEL)
    else:
        yield Resource('models/props/turret_01.mdl', FileType.MODEL)


@cls_func
def npc_bullsquid(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This has various EZ variants."""
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_XEN:
        yield Resource('models/bullsquid_xen.mdl', FileType.MODEL)
        yield Resource('models/babysquid_xen.mdl', FileType.MODEL)
        yield Resource('models/bullsquid_egg_xen.mdl', FileType.MODEL)
        yield Resource('blood_impact_yellow_01', FileType.PARTICLE_SYSTEM)
    elif variant == EZ_VARIANT_RAD:
        yield Resource('models/bullsquid_rad.mdl', FileType.MODEL)
        yield Resource('models/babysquid_rad.mdl', FileType.MODEL)
        yield Resource('models/bullsquid_egg_rad.mdl', FileType.MODEL)
        yield Resource('blood_impact_blue_01', FileType.PARTICLE_SYSTEM)
    else:
        yield Resource('models/bullsquid.mdl', FileType.MODEL)
        yield Resource('models/babysquid.mdl', FileType.MODEL)
        yield Resource('models/bullsquid_egg.mdl', FileType.MODEL)
        yield Resource('blood_impact_yellow_01', FileType.PARTICLE_SYSTEM)


@cls_func
def combine_scanner(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Detect the kind of scanner (city or shield/claw), then pick the right resources."""
    if ent['classname'] == 'npc_clawscanner':  # Added in episodic, always the shield scanner.
        is_shield = True
    else:  # It checks the map name directly to determine this.
        is_shield = ctx.mapname.lower().startswith('d3_c17')
    if is_shield:
        yield Resource("models/shield_scanner.mdl", FileType.MODEL)
        for i in range(1, 7):
            yield Resource(f"models/gibs/Shield_Scanner_Gib{i}.mdl", FileType.MODEL)
        snd_prefix = 'NPC_SScanner.'
    else:
        yield Resource("models/combine_scanner.mdl", FileType.MODEL)
        for i in range(1, 6):
            yield Resource(f"models/gibs/scanner_gib{i:02}.mdl", FileType.MODEL)
        snd_prefix = 'NPC_CScanner.'

    for snd_name in [
        "Shoot", "Alert", "Die", "Combat", "Idle", "Pain", "TakePhoto", "AttackFlash",
        "DiveBombFlyby", "DiveBomb", "DeployMine", "FlyLoop",
    ]:
        yield Resource(snd_prefix + snd_name, FileType.GAME_SOUND)


CIT_HEADS = [
    "male_01.mdl",
    "male_02.mdl",
    "female_01.mdl",
    "male_03.mdl",
    "female_02.mdl",
    "male_04.mdl",
    "female_03.mdl",
    "male_05.mdl",
    "female_04.mdl",
    "male_06.mdl",
    "female_06.mdl",
    "male_07.mdl",
    "female_07.mdl",
    "male_08.mdl",
    "male_09.mdl",
]


@cls_func
def npc_citizen(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Cizizens have a complex set of precaching rules."""
    if ent['targetname'] == 'matt':
        # Special crowbar.
        yield Resource("models/props_canal/mattpipe.mdl", FileType.MODEL)

    cit_type = conv_int(ent['citizentype'])

    if cit_type == 0:  # Default
        # TODO: Pick via mapname:
        # { "trainstation",	CT_DOWNTRODDEN	},
        # { "canals",		CT_REFUGEE		},
        # { "town",			CT_REFUGEE		},
        # { "coast",		CT_REFUGEE		},
        # { "prison",		CT_DOWNTRODDEN	},
        # { "c17",			CT_REBEL		},
        # { "citadel",		CT_DOWNTRODDEN	},
        for head in CIT_HEADS:
            yield Resource('models/humans/group01/' + head, FileType.MODEL)
            yield Resource('models/humans/group02/' + head, FileType.MODEL)
            yield Resource('models/humans/group03/' + head, FileType.MODEL)
            yield Resource('models/humans/group03m/' + head, FileType.MODEL)
        return
    elif cit_type == 1:  # Downtrodden
        folder = 'group01'
    elif cit_type == 2:  # Refugee
        folder = 'group02'
    elif cit_type == 3:  # Rebel
        folder = 'group03'
        # The rebels have an additional set of models.
        for head in CIT_HEADS:
            yield Resource('models/humans/group03m/' + head, FileType.MODEL)
    elif cit_type == 4:  # Use model in KVs directly.
        return
    else:  # Invalid type?
        # TODO: Entropy Zero variants.
        return

    for head in CIT_HEADS:
        yield Resource(f'models/humans/{folder}/{head}', FileType.MODEL)


@cls_func
def npc_combinedropship(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """The Combine Dropship may spawn with a variety of cargo types."""
    cargo_type = conv_int(ent['cratetype'])
    if cargo_type == -3:  # Spawns a prop_dynamic Jeep
        yield Resource("models/buggy.mdl", FileType.MODEL)
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
        yield Resource("models/combine_helicopter.mdl", FileType.MODEL)
        yield Resource("models/combine_helicopter_broken.mdl", FileType.MODEL)
        yield _blank_vmf.create_ent('helicopter_chunk')
    else:
        yield Resource("models/gunship.mdl", FileType.MODEL)


@cls_func
def npc_egg(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """These are EZ2 bullsquid eggs, which spawn a specific EZ variant."""
    yield _blank_vmf.create_ent('npc_bullsquid', ezvariant=ent['ezvariant'])


@cls_func
def npc_maker(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """We spawn the NPC automatically."""
    # Pass this along, it should then pack that too.
    yield _blank_vmf.create_ent(ent['npctype'], additionalequipment=ent['additionalequipment'])


@cls_func
def npc_metropolice(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """If a spawnflag is set, a cheap model is used."""
    if conv_int(ent['spawnflags']) & 5:
        yield Resource("models/police_cheaple.mdl", FileType.MODEL)
    else:
        yield Resource("models/police.mdl", FileType.MODEL)


@cls_func
def npc_zassassin(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Entropy Zero 2's "Plan B"/Gonome. """
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        yield Resource('models/glownome.mdl', FileType.MODEL)
        yield Resource('blood_impact_blue_01', FileType.PARTICLE_SYSTEM)
        yield Resource('materials/cable/goocable.vmt', FileType.MATERIAL)
        yield Resource('materials/sprites/glownomespit.vmt', FileType.MATERIAL)
    else:
        yield Resource('materials/sprites/gonomespit.vmt', FileType.MATERIAL)
        if variant == EZ_VARIANT_XEN:
            yield Resource('models/xonome.mdl', FileType.MODEL)
        else:
            yield Resource('models/gonome.mdl', FileType.MODEL)


@cls_func
def point_entity_replace(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """In one specific mode, an entity is spawned by classname."""
    if conv_int(ent['replacementtype']) == 1:
        yield _blank_vmf.create_ent(ent['replacemententity'])


@cls_func
def skybox_swapper(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This needs to pack a skybox."""
    sky_name = ent['skyboxname']
    for suffix in ['bk', 'dn', 'ft', 'lf', 'rt', 'up']:
        yield Resource(
            f'materials/skybox/{sky_name}{suffix}.vmt',
            FileType.MATERIAL,
        )
        hdr_file = f'materials/skybox/{sky_name}{suffix}_hdr.vmt'
        if hdr_file in ctx.fsys:
            yield Resource(hdr_file, FileType.MATERIAL)


@cls_func
def team_control_point(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Special '_locked' materials."""
    for kvalue in ['team_icon_0', 'team_icon_1', 'team_icon_2']:
        icon = ent[kvalue]
        if icon:
            yield Resource(f'materials/{icon}.vmt', FileType.MATERIAL)
            yield Resource(f'materials/{icon}_locked.vmt', FileType.MATERIAL)


# TODO: Weapons are unusual, they don't directly specify the models.
# Instead, it's specified in the weapon script.
