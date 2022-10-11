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


ResGen: TypeAlias = Iterator[Union[Resource], Entity]
ClassFunc: TypeAlias = Callable[[Entity, 'ResourceCtx'], ResGen]
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


def mdl(path: str) -> Resource:
    """Convienence function."""
    return Resource(path, FileType.MODEL)


def mat(path: str) -> Resource:
    """Convienence function."""
    return Resource(path, FileType.MATERIAL)


def sound(path: str) -> Resource:
    """Convienence function."""
    return Resource(path, FileType.GAME_SOUND)


def part(path: str) -> Resource:
    """Convienence function."""
    return Resource(path, FileType.PARTICLE)


def choreo(path: str) -> Resource:
    """Convienence function."""
    return Resource(path, FileType.CHOREO)


def pack_ent_class(pack: PackList, clsname: str, **keys: ValidKVs) -> None:
    """Call to pack another entity class generically."""
    reslist = CLASS_RESOURCES[clsname]
    for fname, ftype in reslist:
        pack.pack_file(fname, ftype)
    try:
        cls_function = CLASS_FUNCS[clsname]
    except KeyError:
        pass
    else:
        # Create a dummy entity so we can call.
        cls_function(pack, Entity(_blank_vmf, keys={'classname': clsname, **keys}))


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


# In alphabetical order:


@cls_func
def asw_emitter(pack: PackList, ent: Entity) -> None:
    """Complicated thing, probably can't fully process here."""
    template = ent['template']
    if template and template != 'None':
        yield Resource(f'resource/particletemplates/{template}.ptm')

    # TODO: Read the following keys from the file:
    # - "material"
    # - "glowmaterial"
    # - "collisionsound"
    # - "collisiondecal"


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


res('base_boss')


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


res('cycler_blender')
res('cycler_flex')
res('cycler_weapon')


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
# TODO: Mapbase, EZ2 etc additions?


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
    try:
        breakable_class = BREAKABLE_SPAWNS[conv_int(ent['spawnobject'])]
    except (IndexError, TypeError, ValueError):
        pass
    else:
        yield _blank_vmf.create(breakable_class)


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

res('func_plat',
    sound('Plat.DefaultMoving'),
    sound('Plat.DefaultArrive'),
    func=base_plat_train,
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
    if conv_int(ent['modeltype']) == 1:  # MODELTYPE_MODEL
        yield Resource(ent['model'], FileType.MODEL)
    # Otherwise, a template name or a regular gib.


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
        if fname in pack.fsys:
            yield Resource(fname, FileType.MATERIAL)
        else:
            break



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

from srctools._class_resources import (
    asw_, item_, npcs,
)
