"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in keyvalues.
The only public values are CLASS_RESOURCES and ALT_NAMES, but those
should be imported from packlist instead.
"""
from typing import Callable, Tuple, Union, List, Dict, Iterable

from srctools.packlist import FileType, PackList
from srctools import Entity, conv_int

#  For various entity classes, we know they require hardcoded files.
# List them here - classname -> [(file, type), ...]
# Alternatively it's a function to call with the entity to do class-specific
# behaviour, yielding files to pack.

ClassFunc = Callable[[PackList, Entity], None]
CLASS_RESOURCES = {}  # type: Dict[str, Union[ClassFunc, Iterable[Tuple[str, FileType] ]]]
INCLUDES = {}  # type: Dict[str, List[str]]
ALT_NAMES = {}  # type: Dict[str, str]


def res(cls: str, *items: Union[str, Tuple[str, FileType]], includes: str='', aliases: str='') -> None:
    """Add a class to class_resources, with each of the files it always uses.

    includes adds the resources of the other ent to this one if we spawn another.
    aliases indicate additional classnames which are identical to ours.
    """
    if items:
        CLASS_RESOURCES[cls] = [
            (file, FileType.GENERIC) if isinstance(file, str) else file
            for file in items
        ]
    else:
        # Use a tuple here for empty ones, to save a bit of memory
        # with the many ents that don't use resources.
        CLASS_RESOURCES[cls] = ()
    if includes:
        INCLUDES[cls] = includes.split()
    if aliases:
        for alt in aliases.split():
            ALT_NAMES[alt] = cls
            CLASS_RESOURCES[alt] = CLASS_RESOURCES[cls]


def cls_func(func: ClassFunc) -> ClassFunc:
    """Save a function to do special checks for a classname."""
    CLASS_RESOURCES[func.__name__] = func
    return func


def _process_includes() -> None:
    """Apply the INCLUDES dict."""
    for cls in INCLUDES:
        if callable(CLASS_RESOURCES[cls]):
            raise ValueError('Class {} has include and function!'.format(cls))
    while INCLUDES:
        has_changed = False
        for cls in list(INCLUDES):
            resources = CLASS_RESOURCES[cls]
            includes = INCLUDES[cls]
            if not isinstance(resources, list):
                resources = CLASS_RESOURCES[cls] = list(resources)
            for inc_cls in includes[:]:
                if inc_cls not in INCLUDES:
                    resources.extend(CLASS_RESOURCES[inc_cls])
                    includes.remove(inc_cls)
                    has_changed = True
            if not includes:
                del INCLUDES[cls]
                has_changed = True
        if not has_changed:
            raise ValueError('Circular loop in includes: {}'.format(sorted(INCLUDES)))


def mdl(path: str) -> Tuple[str, FileType]:
    """Convienence function."""
    return (path, FileType.MODEL)


def mat(path: str) -> Tuple[str, FileType]:
    """Convienence function."""
    return (path, FileType.MATERIAL)


def sound(path: str) -> Tuple[str, FileType]:
    """Convienence function."""
    return (path, FileType.GAME_SOUND)


def part(path: str) -> Tuple[str, FileType]:
    """Convienence function."""
    return (path, FileType.PARTICLE)


def pack_button_sound(pack: PackList, index: Union[int, str]) -> None:
    """Add the resource matching the hardcoded set of sounds in button ents."""
    pack.pack_soundscript('Buttons.snd{:d}'.format(conv_int(index)))

# In alphabetical order:

res('_firesmoke', *[
    # env_fire_[tiny/small/medium/large][_smoke]
    part('env_fire_ ' + name + smoke)
    for name in ['tiny', 'small', 'medium', 'large']
    for smoke in ['', '_smoke']
])
res('_plasma',
    mat("materials/sprites/plasma1.vmt"),
    mat("materials/sprites/fire_floor.vmt"),
    )


res('aiscripted_schedule')

res('ambient_generic')  # Sound is a keyvalue
res('ambient_music')

# The actual explosion itself.
res('ar2explosion', mat("materials/particle/particle_noisesphere.vmt"))
res('assault_assaultpoint')
res('assault_rallypoint')


res('combine_mine',
    mdl('models/props_combine/combine_mine01.mdl'),
    sound('NPC_CombineMine.Hop'),
    sound('NPC_CombineMine.FlipOver'),
    sound('NPC_CombineMine.TurnOn'),
    sound('NPC_CombineMine.TurnOff'),
    sound('NPC_CombineMine.OpenHooks'),
    sound('NPC_CombineMine.CloseHooks'),
    sound('NPC_CombineMine.ActiveLoop'),
    mat('materials/sprites/glow01.vmt'),
    aliases='bounce_bomb combine_bouncemine'
    )

res('commentary_auto')
res('commentary_dummy')
res('commentary_zombie_spawner')

res('ent_watery_leech', mdl("models/leech.mdl"))

res('event_queue_saveload_proxy')


res('gibshooter',
    mdl('models/gibs/hgibs.mdl'),
    mdl('models/germanygibs.mdl'),
    )
res('grenade_helicopter',  # Bomb dropped by npc_helicopter
    mdl("models/combine_helicopter/helicopter_bomb01.mdl"),
    sound("ReallyLoudSpark"),
    sound("NPC_AttackHelicopterGrenade.Ping"),
    sound("NPC_AttackHelicopterGrenade.PingCaptured"),
    sound("NPC_AttackHelicopterGrenade.HardImpact"),
    )
res('hammer_updateignorelist')
res('helicopter_chunk',  # Broken bits of npc_helicopter
    mdl("models/gibs/helicopter_brokenpiece_01.mdl"),
    mdl("models/gibs/helicopter_brokenpiece_02.mdl"),
    mdl("models/gibs/helicopter_brokenpiece_03.mdl"),
    mdl("models/gibs/helicopter_brokenpiece_04_cockpit.mdl"),
    mdl("models/gibs/helicopter_brokenpiece_05_tailfan.mdl"),
    mdl("models/gibs/helicopter_brokenpiece_06_body.mdl"),
    sound('BaseExplosionEffect.Sound'),
    sound('NPC_AttackHelicopter.Crash'),
    includes='env_smoketrail env_fire_trail ar2explosion'
    )
res('hunter_flechette',
    mdl("models/weapons/hunter_flechette.mdl"),
    mat("materials/sprites/light_glow02_noz.vmt"),
    sound("NPC_Hunter.FlechetteNearmiss"),
    sound("NPC_Hunter.FlechetteHitBody"),
    sound("NPC_Hunter.FlechetteHitWorld"),
    sound("NPC_Hunter.FlechettePreExplode"),
    sound("NPC_Hunter.FlechetteExplode"),
    part("hunter_flechette_trail_striderbuster"),
    part("hunter_flechette_trail"),
    part("hunter_projectile_explosion_1"),
    )

res('info_constraint_anchor')
res('lookdoorthinker')


@cls_func
def move_rope(pack: PackList, ent: Entity) -> None:
    """Implement move_rope and keyframe_rope resources."""
    old_shader_type = conv_int(ent['RopeShader'])
    if old_shader_type == 0:
        pack.pack_file('materials/cable/cable.vmt', FileType.MATERIAL)
    elif old_shader_type == 1:
        pack.pack_file('materials/cable/rope.vmt', FileType.MATERIAL)
    else:
        pack.pack_file('materials/cable/chain.vmt', FileType.MATERIAL)
    pack.pack_file('materials/cable/rope_shadowdepth.vmt', FileType.MATERIAL)

# These classes are identical.
CLASS_RESOURCES['keyframe_rope'] = CLASS_RESOURCES['move_rope']
ALT_NAMES['keyframe_rope'] = 'move_rope'

res('phys_bone_follower')
res('physics_entity_solver')
res('physics_npc_solver')

res('point_energy_ball_launcher',
    includes='prop_energy_ball',
    )
res('point_futbol_shooter', 
    sound('World.Wheatley.fire'),
    includes='prop_exploding_futbol',
    )
res('point_hurt')

res('point_prop_use_target')
res('point_proximity_sensor')
res('point_push')
res('point_ragdollboogie', includes='env_ragdoll_boogie')
res('point_spotlight',
    'materials/sprites/light_glow03.vmt',
    'materials/sprites/glow_test02.vmt',
)

res('rope_anchor', mat("materials/cable/cable.vmt"))
res('rocket_turret_projectile',
    mdl('models/props_bts/rocket.mdl'),
    mat('materials/decals/scorchfade.vmt'),
    sound('NPC_FloorTurret.RocketFlyLoop'),
    )
res('spark_shower',
    mat('materials/sprites/glow01.vmt'),
    mat('materials/effects/yellowflare.vmt'),
    )
res('squadinsignia', "models/chefhat.mdl")  # Yeah.


@cls_func
def team_control_point(pack: PackList, ent: Entity) -> None:
    """Special '_locked' materials."""
    for kvalue in ['team_icon_0', 'team_icon_1', 'team_icon_2']:
        mat = ent[kvalue]
        if mat:
            pack.pack_file('materials/{}.vmt'.format(mat), FileType.MATERIAL)
            pack.pack_file('materials/{}_locked.vmt'.format(mat), FileType.MATERIAL)


res('vgui_screen',
    'materials/engine/writez.vmt',
    )
res('waterbullet', mdl('models/weapons/w_bullet.mdl'))
res('window_pane', mdl('models/brokenglass_piece.mdl'))


from srctools._class_resources import (
    ai_, asw_, env_, filters, func_,
    item_, logic, npcs, props, triggers,
)
# Now all of these have been done, apply 'includes' commands.
_process_includes()
