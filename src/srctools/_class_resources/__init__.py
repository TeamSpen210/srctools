"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in keyvalues.
The only public values are CLASS_RESOURCES and ALT_NAMES, but those
should be imported from packlist instead.
"""
from typing import Callable, TypeVar, Tuple, Union, List, Dict, Iterable

from srctools.packlist import FileType, PackList
from srctools import Entity, conv_int

#  For various entity classes, we know they require hardcoded files.
# List them here - classname -> [(file, type), ...]
# Additionally or instead you could have a function to call with the
# entity to do class-specific behaviour, yielding files to pack.

ClassFunc = Callable[[PackList, Entity], None]
ClassFuncT = TypeVar('ClassFuncT', bound=ClassFunc)
CLASS_RESOURCES: Dict[str, Iterable[Tuple[str, FileType]]] = {}
CLASS_FUNCS: Dict[str, ClassFunc] = {}
INCLUDES: Dict[str, List[str]] = {}
ALT_NAMES: Dict[str, str] = {}


def res(cls: str, *items: Union[str, Tuple[str, FileType]], includes: str='', aliases: str='') -> None:
    """Add a class to class_resources, with each of the files it always uses.

    includes adds the resources of the other ent to this one if we spawn another.
    aliases indicate additional classnames which are identical to ours.
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
    if aliases:
        for alt in aliases.split():
            ALT_NAMES[alt] = cls
            CLASS_RESOURCES[alt] = res_list


def cls_func(func: ClassFuncT) -> ClassFuncT:
    """Save a function to do special checks for a classname."""
    CLASS_FUNCS[func.__name__] = func
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
                    if inc_cls in CLASS_FUNCS:
                        raise ValueError(f'{inc_cls} defines func, but included by {cls}!')
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


def choreo(path: str) -> Tuple[str, FileType]:
    """Convienence function."""
    return (path, FileType.CHOREO)


def pack_ent_class(pack: PackList, clsname: str) -> None:
    """Call to pack another entity class."""
    reslist = CLASS_RESOURCES[clsname]
    if callable(reslist):
        raise ValueError("Can't pack \"{}\", has a custom function!".format(clsname))
    for fname, ftype in reslist:
        pack.pack_file(fname, ftype)


def pack_button_sound(pack: PackList, index: Union[int, str]) -> None:
    """Add the resource matching the hardcoded set of sounds in button ents."""
    pack.pack_soundscript(f'Buttons.snd{conv_int(index):d}')

# In alphabetical order:

res('_ballplayertoucher')
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
res('apc_missile', includes='rpg_missile')  # Inherits from this.

# The actual explosion itself.
res('ar2explosion', mat("materials/particle/particle_noisesphere.vmt"))
res('assault_assaultpoint')
res('assault_rallypoint')

res('bullseye_strider_focus', includes='npc_bullseye')  # Unchanged subclass.

res('challenge_mode_end_node',
    # Assumed based on console logs.
    mdl("models/props/stopwatch_finish_line.mdl"),
    *[sound("glados.dlc1_leaderboard{:02}".format(i)) for i in range(1, 24)]
    )
res('concussiveblast',
    mat('materials/sprites/lgtning.vmt'),
    mat('materials/effects/blueflare1.vmt'),
    mat("materials/particle/particle_smokegrenade.vmt"),
    mat("materials/particle/particle_noisesphere.vmt"),
    )
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

@cls_func
def color_correction(pack: PackList, ent: Entity) -> None:
    """Pack the color correction file."""
    pack.pack_file(ent['filename'])


@cls_func
def color_correction_volume(pack: PackList, ent: Entity) -> None:
    """Pack the color correction file for this too."""
    pack.pack_file(ent['filename'])


res('commentary_auto')
res('commentary_dummy')
res('commentary_zombie_spawner')
res('crossbow_bolt',
    mdl('models/crossbow_bolt.mdl'),
    mat('materials/sprites/light_glow02_noz.vmt'),
    )
res('cycler')
res('cycler_blender')
res('cycler_flex')
res('cycler_weapon')
res('cycler_wreckage')
res('ent_watery_leech', mdl("models/leech.mdl"))

res('event_queue_saveload_proxy')
res('fish')
res('floorturret_tipcontroller')

res('game_ui')
res('gib')
res('gibshooter',
    mdl('models/gibs/hgibs.mdl'),
    mdl('models/germanygibs.mdl'),
    )
res('grenade_ar2',  # Actually the SMG's grenade.
    mdl("models/Weapons/ar2_grenade.mdl"),
    )
res('grenade_beam',
    mdl('Models/weapons/flare.mdl'),  # Not visible, but loaded.
    mat('materials/sprites/laser.vmt'),
    sound("GrenadeBeam.HitSound"),
    )
res('grenade_beam_chaser')  # The back part of the glow following grenades.
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
res('info_camera_link')
res('info_darknessmode_lightsource')
res('info_hint')
res('info_ladder_dismount')
res('info_lighting_relative')
res('info_mass_center')
res('info_node')
res('info_node_hint')
res('info_node_air')
res('info_node_air_hint')
res('info_node_climb')
res('info_node_link')
res('info_node_link_controller')
res('info_node_link_filtered')
res('info_node_link_logic')
res('info_node_link_oneway')
res('info_null')
res('info_overlay_accessor')
res('info_particle_system')  # Particle packed by FGD database.
res('info_radial_link_controller')
res('info_remarkable')
res('info_snipertarget')
res('info_target')
res('info_target_gunshipcrash')
res('info_target_helicopter_crash')
res('info_teleport_destination')

res('light')
res('light_directional')
res('light_dynamic')
res('light_environment')
res('light_spot')
res('lookdoorthinker')

res('mortarshell',
    mat('materials/sprites/physbeam.vmt'),
    mat('materials/effects/ar2ground2.vmt'),
    sound('Weapon_Mortar.Impact'),
    )

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
CLASS_FUNCS['keyframe_rope'] = CLASS_FUNCS['move_rope']
ALT_NAMES['keyframe_rope'] = 'move_rope'

res('passtime_ball',
    mdl('models/passtime/ball/passtime_ball_halloween.mdl'),
    mdl('models/passtime/ball/passtime_ball.mdl'),
    mat("materials/passtime/passtime_balltrail_red.vmt"),
    mat("materials/passtime/passtime_balltrail_blu.vmt"),
    mat("materials/passtime/passtime_balltrail_unassigned.vmt"),
    sound('Passtime.BallSmack'),
    sound('Passtime.BallGet'),
    sound('Passtime.BallIdle'),
    sound('Passtime.BallHoming'),
    includes='_ballplayertoucher',
)

res('phys_bone_follower')
res('physics_cannister')  # All in KVs.
res('physics_entity_solver')
res('physics_npc_solver')

res('point_advanced_finder')
res('point_commentary_node',
    mdl('models/extras/info_speech.mdl'),
    includes='point_commentary_viewpoint',
    )
res('point_commentary_viewpoint', mat('materials/sprites/redglow1.vmt'))
res('point_energy_ball_launcher',
    includes='prop_energy_ball',
    )
res('point_entity_finder')
res('point_entity_replace')
res('point_flesh_effect_target')
res('point_futbol_shooter',
    sound('World.Wheatley.fire'),
    includes='prop_exploding_futbol',
    )
res('point_gamestats_counter')
res('point_message')
res('point_message_localized')
res('point_hurt')

res('point_prop_use_target')
res('point_proximity_sensor')
res('point_push')
res('point_ragdollboogie', includes='env_ragdoll_boogie')
res('point_spotlight',
    'materials/sprites/light_glow03.vmt',
    'materials/sprites/glow_test02.vmt',
)
res('point_teleport')
res('point_template')
res('point_tesla', sound("sprites/physbeam.vmt"))  # Default material
res('point_velocitysensor')
res('point_viewcontrol')
res('point_viewcontrol_multiplayer')
res('point_viewcontrol_node')
res('point_viewcontrol_survivor')
res('point_viewproxy')
res('point_weaponstrip')

res('raggib')
res('rope_anchor', mat("materials/cable/cable.vmt"))
res('rocket_turret_projectile',
    mdl('models/props_bts/rocket.mdl'),
    mat('materials/decals/scorchfade.vmt'),
    sound('NPC_FloorTurret.RocketFlyLoop'),
    )
res('rpg_missile',
    mdl("models/weapons/w_missile.mdl"),
    mdl("models/weapons/w_missile_launch.mdl"),
    mdl("models/weapons/w_missile_closed.mdl"),
    )

res('script_intro',
    mat('materials/scripted/intro_screenspaceeffect.vmt'),
    )


@cls_func
def skybox_swapper(pack: PackList, ent: Entity) -> None:
    """This needs to pack a skybox."""
    sky_name = ent['skyboxname']
    for suffix in ['bk', 'dn', 'ft', 'lf', 'rt', 'up']:
        pack.pack_file(
            'materials/skybox/{}{}.vmt'.format(sky_name, suffix),
            FileType.MATERIAL,
        )
        pack.pack_file(
            'materials/skybox/{}{}_hdr.vmt'.format(sky_name, suffix),
            FileType.MATERIAL,
            optional=True,
        )

res('soundent')
res('spraycan', sound("SprayCan.Paint"))
res('sparktrail', sound('DoSpark'))
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


res('test_effect', mat('materials/sprites/lgtning.vmt'), includes='env_beam')
res('test_proxytoggle')


@cls_func
def vgui_movie_display(pack: PackList, ent: Entity):
    """Mark the BIK movie as being used, though it can't be packed."""
    pack.pack_file(ent['MovieFilename'])


res('vgui_screen',
    'materials/engine/writez.vmt',
    )
res('waterbullet', mdl('models/weapons/w_bullet.mdl'))
res('window_pane', mdl('models/brokenglass_piece.mdl'))


from srctools._class_resources import (
    ai_, asw_, env_, filters, func_,
    item_, logic, npcs, props, triggers,
    weapons,
)
# Now all of these have been done, apply 'includes' commands.
_process_includes()
