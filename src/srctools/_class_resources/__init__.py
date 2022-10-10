"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in keyvalues.
The only public values are CLASS_RESOURCES and ALT_NAMES, but those
should be imported from packlist instead.
"""
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
from typing_extensions import Final, TypeAlias
import itertools

from .. import conv_bool, conv_int
from ..packlist import FileType, PackList
from ..vmf import VMF, Entity, ValidKVs


#  For various entity classes, we know they require hardcoded files.
# List them here - classname -> [(file, type), ...]
# Additionally or instead you could have a function to call with the
# entity to do class-specific behaviour, yielding files to pack.

ClassFunc: TypeAlias = Callable[[PackList, Entity], object]
ClassFuncT = TypeVar('ClassFuncT', bound=ClassFunc)
ResourceTup: TypeAlias = Tuple[str, FileType]
CLASS_RESOURCES: Dict[str, Iterable[ResourceTup]] = {}
CLASS_FUNCS: Dict[str, ClassFunc] = {}
INCLUDES: Dict[str, List[str]] = {}
ALT_NAMES: Dict[str, str] = {}
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
    res_list: Iterable[Tuple[str, FileType]]
    if items:
        CLASS_RESOURCES[cls] = res_list = [
            (file, FileType.GENERIC) if isinstance(file, str) else file
            for file in items
        ]
    else:
        # Use a tuple here for empty ones, to save a bit of memory
        # with the many ents that don't use resources.
        CLASS_RESOURCES[cls] = res_list = ()
    if includes:
        INCLUDES[cls] = includes.split()
    for alt in aliases.split():
        ALT_NAMES[alt] = cls
        CLASS_RESOURCES[alt] = res_list
    if func is not None:
        if cls in CLASS_FUNCS:
            raise ValueError(f'Class function already defined for "{cls}"!')
        CLASS_FUNCS[cls] = func
        for alt in aliases.split():
            if alt in CLASS_FUNCS:
                raise ValueError(f'Class function already defined for "{alt}"!')
            CLASS_FUNCS[alt] = func


def cls_func(func: ClassFuncT) -> ClassFuncT:
    """Save a function to do special checks for a classname."""
    name = func.__name__
    if name in CLASS_FUNCS:
        raise ValueError(f'Class function already defined for "{name}"!')
    CLASS_FUNCS[name] = func
    # Ensure this is also defined.
    CLASS_RESOURCES.setdefault(name, ())
    return func


def _process_includes() -> None:
    """Apply the INCLUDES dict."""
    while INCLUDES:
        has_changed = False
        for cls in list(INCLUDES):
            resources = CLASS_RESOURCES[cls]
            includes = INCLUDES[cls]
            if not isinstance(resources, list):
                resources = CLASS_RESOURCES[cls] = list(resources)
            for inc_cls in includes[:]:
                if inc_cls not in INCLUDES:
                    try:
                        resources.extend(CLASS_RESOURCES[inc_cls])
                    except KeyError:
                        raise ValueError(f'{inc_cls} does not exist, but included by {cls}!') from None
                    # If this inherits from a class func, we must also have one.
                    if inc_cls in CLASS_FUNCS and cls not in CLASS_FUNCS:
                        raise ValueError(
                            f'{inc_cls} defines func, but included by '
                            f'{cls} which doesn\'t have one!'
                        )
                    includes.remove(inc_cls)
                    has_changed = True
            if not includes:
                del INCLUDES[cls]
                has_changed = True
            if not resources:  # Convert back to empty tuple.
                CLASS_RESOURCES[cls] = ()
        if not has_changed:
            raise ValueError('Circular loop in includes: {}'.format(sorted(INCLUDES)))
    # Copy over aliased class functions.
    for alias, cls in ALT_NAMES.items():
        try:
            CLASS_FUNCS[alias] = CLASS_FUNCS[cls]
        except KeyError:
            pass


def mdl(path: str) -> ResourceTup:
    """Convienence function."""
    return (path, FileType.MODEL)


def mat(path: str) -> ResourceTup:
    """Convienence function."""
    return (path, FileType.MATERIAL)


def sound(path: str) -> ResourceTup:
    """Convienence function."""
    return (path, FileType.GAME_SOUND)


def part(path: str) -> ResourceTup:
    """Convienence function."""
    return (path, FileType.PARTICLE)


def choreo(path: str) -> ResourceTup:
    """Convienence function."""
    return (path, FileType.CHOREO)


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


def pack_button_sound(pack: PackList, index: Union[int, str]) -> None:
    """Add the resource matching the hardcoded set of sounds in button ents."""
    pack.pack_soundscript(f'Buttons.snd{conv_int(index):d}')


# Entropy Zero 2 variant constants.
EZ_VARIANT_DEFAULT: Final = 0
EZ_VARIANT_XEN: Final = 1
EZ_VARIANT_RAD: Final = 2
EZ_VARIANT_TEMPORAL: Final = 3
EZ_VARIANT_ARBEIT: Final = 4
EZ_VARIANT_BLOOD: Final = 5


# TODO: We need to parse vehicle scripts.


# In alphabetical order:

res('base_boss')


@cls_func
def color_correction(pack: PackList, ent: Entity) -> None:
    """Pack the color correction file."""
    pack.pack_file(ent['filename'])


res('cycler_blender')
res('cycler_flex')
res('cycler_weapon')


def sprite_rope(pack: PackList, ent: Entity) -> None:
    """Handles a legacy keyvalue for the material used on move_rope and keyframe_rope."""
    if 'ropeshader' in ent:
        old_shader_type = conv_int(ent['ropeshader'])
        if old_shader_type == 0:
            pack.pack_file('materials/cable/cable.vmt', FileType.MATERIAL)
        elif old_shader_type == 1:
            pack.pack_file('materials/cable/rope.vmt', FileType.MATERIAL)
        else:
            pack.pack_file('materials/cable/chain.vmt', FileType.MATERIAL)


@cls_func
def env_break_shooter(pack: PackList, ent: Entity) -> None:
    """Special behaviour on the 'model' KV."""
    if conv_int(ent['modeltype']) == 1:  # MODELTYPE_MODEL
        pack.pack_file(ent['model'], FileType.MODEL)
    # Otherwise, a template name or a regular gib.


@cls_func
def env_fire(pack: PackList, ent: Entity) -> None:
    """Two types of fire, with different resources."""
    fire_type = conv_int(ent['firetype'])
    if fire_type == 0:  # Natural
        flags = conv_int(ent['spawnflags'])
        if flags & 2:  # Smokeless?
            suffix = ''  # env_fire_small
        else:
            suffix = '_smoke'  # env_fire_medium_smoke
        for name in ['tiny', 'small', 'medium', 'large']:
            pack.pack_particle(f'env_fire_{name}{suffix}')
    elif fire_type == 1:  # Plasma
        pack_ent_class(pack, '_plasma')


@cls_func
def env_headcrabcanister(pack: PackList, ent: Entity) -> None:
    """Check if it spawns in skybox or not, and precache the headcrab."""
    flags = conv_int(ent['spawnflags'])
    if flags & 0x1 == 0:  # !SF_NO_IMPACT_SOUND
        pack.pack_soundscript('HeadcrabCanister.Explosion')
        pack.pack_soundscript('HeadcrabCanister.IncomingSound')
        pack.pack_soundscript('HeadcrabCanister.SkyboxExplosion')
    if flags & 0x2 == 0:  # !SF_NO_LAUNCH_SOUND
        pack.pack_soundscript('HeadcrabCanister.LaunchSound')
    if flags & 0x1000 == 0:  # !SF_START_IMPACTED
        pack.pack_file('materials/sprites/smoke.vmt', FileType.MATERIAL)

    if flags & 0x80000 == 0:  # !SF_NO_IMPACT_EFFECTS
        pack.pack_file('particle/particle_noisesphere',
                       FileType.MATERIAL)  # AR2 explosion
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
        pack_ent_class(pack, headcrab)


@cls_func
def env_shooter(pack: PackList, ent: Entity) -> None:
    """A hardcoded array of sounds to play."""
    try:
        snd_name = (
            "Breakable.MatGlass",
            "Breakable.MatWood",
            "Breakable.MatMetal",
            "Breakable.MatFlesh",
            "Breakable.MatConcrete",
        )[conv_int(ent['shootsounds'])]
        pack.pack_soundscript(snd_name)
    except IndexError:
        pass

    # Valve does this same check.
    if ent['shootmodel'].casefold().endswith('.vmt'):
        pack.pack_file(ent['shootmodel'], FileType.MATERIAL)
    else:
        pack.pack_file(ent['shootmodel'], FileType.MODEL)


@cls_func
def env_smokestack(pack: PackList, ent: Entity) -> None:
    """This tries using each numeric material that exists."""
    mat_base = ent['smokematerial'].casefold().replace('\\', '/')
    if not mat_base:
        return

    if mat_base.endswith('.vmt'):
        mat_base = mat_base[:-4]
    if not mat_base.startswith('materials/'):
        mat_base = 'materials/' + mat_base

    pack.pack_file(mat_base + '.vmt', FileType.MATERIAL)
    for i in itertools.count(1):
        fname = f'{mat_base}{i}.vmt'
        if fname in pack.fsys:
            pack.pack_file(fname)
        else:
            break



@cls_func
def point_entity_replace(pack: PackList, ent: Entity) -> None:
    """In one specific mode, an entity is spawned by classname."""
    if conv_int(ent['replacementtype']) == 1:
        pack_ent_class(pack, ent['replacemententity'])


@cls_func
def skybox_swapper(pack: PackList, ent: Entity) -> None:
    """This needs to pack a skybox."""
    sky_name = ent['skyboxname']
    for suffix in ['bk', 'dn', 'ft', 'lf', 'rt', 'up']:
        pack.pack_file(
            f'materials/skybox/{sky_name}{suffix}.vmt',
            FileType.MATERIAL,
        )
        pack.pack_file(
            f'materials/skybox/{sky_name}{suffix}_hdr.vmt',
            FileType.MATERIAL,
            optional=True,
        )


@cls_func
def team_control_point(pack: PackList, ent: Entity) -> None:
    """Special '_locked' materials."""
    for kvalue in ['team_icon_0', 'team_icon_1', 'team_icon_2']:
        icon = ent[kvalue]
        if icon:
            pack.pack_file(f'materials/{icon}.vmt', FileType.MATERIAL)
            pack.pack_file(f'materials/{icon}_locked.vmt', FileType.MATERIAL)


from srctools._class_resources import (
    asw_, func_, item_, npcs, weapons,
)
