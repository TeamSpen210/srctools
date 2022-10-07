"""env_ entities."""
import itertools

from ..packlist import PackList
from ..vmf import Entity
from . import *

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

# Does the same.
res('env_rotorshooter', func=env_shooter)


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
        fname = '{}{}.vmt'.format(mat_base, i)
        if fname in pack.fsys:
            pack.pack_file(fname)
        else:
            break
