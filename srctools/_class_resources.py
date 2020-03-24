"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in keyvalues.
"""
import itertools
from typing import Callable, Tuple, Union, List, Dict, Iterable

from srctools.packlist import FileType, PackList
from srctools import Entity, conv_int

__all__ = ['CLASS_RESOURCES', 'ALT_NAMES']

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

res('ai_ally_manager')
res('ai_battle_line')
res('ai_changehintgroup')
res('ai_changetarget')
res('ai_citizen_response_system')
res('ai_goal_actbusy')
res('ai_goal_actbusy_queue')
res('ai_goal_assault')
res('ai_goal_fear')
res('ai_goal_fightfromcover')
res('ai_goal_follow')
res('ai_goal_injured_follow')
res('ai_goal_lead')
res('ai_goal_lead_weapon')
res('ai_goal_operator')
res('ai_goal_police')
res('ai_goal_standoff')
res('ai_npc_eventresponsesystem')
res('ai_relationship')
res('ai_relationship_classify')
res('ai_script_conditions')
res('ai_sound')
res('ai_speechfilter')
res('ai_weaponmodifier')
res('aiscripted_schedule')

res('ambient_generic')  # Sound is a keyvalue
res('ambient_music')

# The actual explosion itself.
res('ar2explosion', mat("materials/particle/particle_noisesphere.vmt"))
res('assault_assaultpoint')
res('assault_rallypoint')

res('asw_camera_control')
res('asw_alien_goo',
    mdl('models/aliens/biomass/biomasshelix.mdl'),
    mdl('models/aliens/biomass/biomassl.mdl'),
    mdl('models/aliens/biomass/biomasss.mdl'),
    mdl('models/aliens/biomass/biomassu.mdl'),
    sound('ASWGoo.GooLoop'),
    sound('ASWGoo.GooScream'),
    sound('ASWGoo.GooDissolve'),
    sound('ASWFire.AcidBurn'),
    part('biomass_dissolve'),
    part('acid_touch'),
    part('grubsack_death'),
    includes='asw_grub',
    )
CLASS_RESOURCES['asw_ambient_generic'] = CLASS_RESOURCES['ambient_generic']  # Identical.
res('asw_ammo_autogun', mdl('models/swarm/ammo/ammoautogun.mdl'))

res('asw_ammo_drop', 
    mdl('models/items/Ammobag/AmmoBag.mdl' ),
    sound('ASW_Ammobag.DropImpact'),
    sound('ASW_Ammobag.Pickup_sml'),
    sound('ASW_Ammobag.Pickup_med'),
    sound('ASW_Ammobag.Pickup_lrg'),
    sound('ASW_Ammobag.Fail' ),
    part('ammo_satchel_take_sml'),
    part('ammo_satchel_take_med'),
    part('ammo_satchel_take_lrg'),
    )


res('asw_ammo_flamer', mdl('models/swarm/ammo/ammoflamer.mdl'))
res('asw_ammo_mining_laser', mdl('models/swarm/ammo/ammobattery.mdl'))
res('asw_ammo_pdw', mdl('models/swarm/ammo/ammopdw.mdl'))
res('asw_ammo_pistol', mdl('models/swarm/ammo/ammopistol.mdl'))
res('asw_ammo_railgun', mdl('models/swarm/ammo/ammorailgun.mdl'))  # Commented out
res('asw_ammo_rifle', mdl('models/swarm/ammo/ammoassaultrifle.mdl'))
res('asw_ammo_shotgun', mdl('models/swarm/ammo/ammoshotgun.mdl'))
res('asw_ammo_vindicator', mdl('models/swarm/ammo/ammovindicator.mdl'))

res('asw_barrel_explosive',
    mdl('models/swarm/Barrel/barrel.mdl'),
    part('explosion_barrel'),
    sound('ASWBarrel.Explode'),
    )

res('asw_barrel_radioactive',
    mdl('models/swarm/Barrel/barrel.mdl'),
    part('barrel_rad_gas_cloud'),
    part('barrel_rad_gas_jet'),
    sound('Misc.Geiger'),
    )

res('asw_bloodhound', mdl('models/swarmprops/Vehicles/BloodhoundMesh.mdl'))

res('asw_boomer',
    part('boomer_explode'),
    part('joint_goo'),
    mdl('models/aliens/boomer/boomerLegA.mdl'),
    mdl('models/aliens/boomer/boomerLegB.mdl'),
    mdl('models/aliens/boomer/boomerLegC.mdl'),
    mdl('models/aliens/boomer/boomer.mdl'),
    sound('ASW_Boomer.Death_Explode'),
    sound('ASW_Boomer.Death_Gib'),
    )

res('asw_broadcast_camera')
res('asw_buzzer',
    mdl('models/aliens/buzzer/buzzer.mdl'),
    sound('ASW_Buzzer.Attack'),
    sound('ASW_Buzzer.Death'),
    sound('ASW_Buzzer.Pain'),
    sound('ASW_Buzzer.Idle'),
    sound('ASW_Buzzer.OnFire'),

    part('buzzer_trail' ),
    part('buzzer_death' ),
    sound('ASWFire.BurningFlesh'),
    sound('ASWFire.StopBurning'),
    )
res('asw_client_corpse')

res('asw_colonist',
    mdl('models/swarm/Colonist/Male/MaleColonist.mdl'),
    sound('NPC_Citizen.FootstepLeft'),
    sound('NPC_Citizen.FootstepRight'),
    sound('NPC_Citizen.Die'),
    sound('MaleMarine.Pain'),
    )

res('asw_debrief_info')
res('asw_director_control')

res('asw_door',
    mdl('models/swarm/doors/swarm_singledoor.mdl'),
    mdl('models/swarm/doors/swarm_singledoor_flipped.mdl'),
    mdl('models/props/doors/heavy_doors/doorleft.mdl'),
    mdl('models/props/doors/heavy_doors/doorright.mdl'),
    sound('ASW_Door.Dented'),
    sound('ASW_Door.MeleeHit'),
    sound('ASW_Welder.WeldDeny'),
    )

res('asw_drone',
    sound('ASW_Drone.Land'),
    sound('ASW_Drone.Pain'),
    sound('ASW_Drone.Alert'),
    sound('ASW_Drone.Death'),
    sound('ASW_Drone.Attack'),
    sound('ASW_Drone.Swipe'),

    sound('ASW_Drone.GibSplatHeavy'),
    sound('ASW_Drone.GibSplat'),
    sound('ASW_Drone.GibSplatQuiet'),
    sound('ASW_Drone.DeathFireSizzle'),

    mdl('models/aliens/drone/ragdoll_tail.mdl'),
    mdl('models/aliens/drone/ragdoll_uparm.mdl'),
    mdl('models/aliens/drone/ragdoll_uparm_r.mdl'),
    mdl('models/aliens/drone/ragdoll_leg_r.mdl'),
    mdl('models/aliens/drone/ragdoll_leg.mdl'),
    mdl('models/aliens/drone/gib_torso.mdl'),
    mdl('models/aliens/drone/drone.mdl'),
    )
CLASS_RESOURCES['asw_drone_jumper'] = CLASS_RESOURCES['asw_drone']  # Identical
res('asw_drone_uber', includes='asw_drone')  # Seems the same resources.

res('asw_egg',
    part('egg_open'),
    part('egg_hatch'),
    part('egg_death'),

    mdl('models/aliens/egg/egggib_1.mdl'),
    mdl('models/aliens/egg/egggib_2.mdl'),
    mdl('models/aliens/egg/egggib_3.mdl'),
    mdl('models/aliens/egg/egg.mdl'),

    sound('ASW_Egg.Open'),
    sound('ASW_Egg.Gib'),
    sound('ASW_Parasite.EggBurst'),
    includes='asw_parasite',
    )


@cls_func
def asw_emitter(pack: PackList, ent: Entity):
    """Complicated thing, probably can't fully process here."""
    template = ent['template']
    if template and template != 'None':
        pack.pack_file('resource/particletemplates/{}.ptm'.format(template))

    # TODO: Read the following keys from the file:
    # - "material"
    # - "glowmaterial"
    # - "collisionsound"
    # - "collisiondecal"

    pack.pack_file('materials/effects/yellowflare.vmt', FileType.MATERIAL)


@cls_func
def asw_env_explosion(pack: PackList, ent: Entity):
    """Handle the no-sound spawnflag."""
    pack.pack_file("asw_env_explosion", FileType.PARTICLE_SYSTEM)
    if (conv_int(ent['spawnflags']) & 0x00000010) == 0:
        pack.pack_file('ASW_Explosion.Explosion_Default', FileType.GAME_SOUND)

res('asw_parasite',
    mdl('models/aliens/parasite/parasite.mdl'),
    sound('ASW_Parasite.Death'),
    sound('ASW_Parasite.Attack'),
    sound('ASW_Parasite.Idle'),
    sound('ASW_Parasite.Pain'),
    sound('ASW_Parasite.Attack'),
    )

CLASS_RESOURCES['asw_parasite_defanged'] = CLASS_RESOURCES['asw_parasite']  # Identical.

res('asw_grub',
    mdl('models/swarm/Grubs/Grub.mdl'),
    mdl('models/Swarm/Grubs/GrubGib1.mdl'),
    mdl('models/Swarm/Grubs/GrubGib2.mdl'),
    mdl('models/Swarm/Grubs/GrubGib3.mdl'),
    mdl('models/Swarm/Grubs/GrubGib4.mdl'),
    mdl('models/Swarm/Grubs/GrubGib5.mdl'),
    mdl('models/Swarm/Grubs/GrubGib6.mdl'),
    sound('ASW_Parasite.Death'),
    sound('ASW_Parasite.Attack'),
    sound('ASW_Parasite.Idle'),
    sound('NPC_AntlionGrub.Squash'),
    sound('ASW_Drone.GibSplatQuiet'),
    part('grub_death'),
    part('grub_death_fire'),
    )
res('asw_rope_anchor', mat("materials/cable/cable.vmt"))
res('asw_snow_volume')  # TODO: Uses an asw_emitter

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
res('env_airstrike_indoors')
res('env_airstrike_outdoors')
res('env_ambient_light')
res('env_alyxemp',
    mat('materials/effects/laser1.vmt'),
    sound('AlyxEmp.Charge'),
    sound('AlyxEmp.Discharge'),
    sound('AlyxEmp.Stop'),
    )
res('env_ar2explosion', includes="ar2explosion")

res('env_beam')
res('env_beverage', mdl('models/can.mdl'))
res('env_blood')
res('env_bubbles', mat('materials/sprites/bubble.vmt'))


@cls_func
def env_break_shooter(pack: PackList, ent: Entity):
    """Special behaviour on the 'model' KV."""
    if conv_int(ent['modeltype']) == 1:  # MODELTYPE_MODEL
        pack.pack_file(ent['model'], FileType.MODEL)
    # Otherwise, a template name or a regular gib.


res('env_citadel_energy_core',
    mat('materials/effects/strider_muzzle.vmt'),
    mat('materials/effects/combinemuzzle2.vmt'),
    mat('materials/effects/combinemuzzle2_dark.vmt'),
    )
res('env_credits',
    'scripts/credits.txt',  # Script file with all the credits.
    )
res('env_detail_controller')
res('env_dof_controller')
res('env_embers', mat('materials/particle/fire.vmt'))

res('env_dustpuff',
    mat("materials/particle/particle_smokegrenade.vmt"),
    mat("materials/particle/particle_noisesphere.vmt"),
    )
res('env_fire_trail',
    mat("materials/sprites/flamelet1.vmt"),
    mat("materials/sprites/flamelet2.vmt"),
    mat("materials/sprites/flamelet3.vmt"),
    mat("materials/sprites/flamelet4.vmt"),
    mat("materials/sprites/flamelet5.vmt"),
    mat("materials/particle/particle_smokegrenade.vmt"),
    mat("materials/particle/particle_noisesphere.vmt"),
    )


@cls_func
def env_fire(pack: PackList, ent: Entity) -> None:
    """Two types of fire, with different resources."""
    pack.pack_file('Fire.Plasma', FileType.GAME_SOUND)
    fire_type = conv_int(ent['firetype'])
    if fire_type == 0:  # Natural
        flags = conv_int(ent['spawnflags'])
        if flags & 2:  # Smokeless?
            suffix = ''  # env_fire_small
        else:
            suffix = '_smoke'  # env_fire_medium_smoke
        for name in ['tiny', 'small', 'medium', 'large']:
            pack.pack_file('env_fire_{}{}'.format(name, suffix), FileType.PARTICLE_SYSTEM)
    elif fire_type == 1:  # Plasma
        for args in CLASS_RESOURCES['_plasma']:
            pack.pack_file(*args)

res('env_firesensor')
res('env_firesource')
res('env_flare',
    mdl("models/weapons/flare.mdl"),
    sound("Weapon_FlareGun.Burn"),
    includes="_firesmoke",
    )
res('env_fire_trail',
    mat('materials/sprites/flamelet1.vmt'),
    mat('materials/sprites/flamelet2.vmt'),
    mat('materials/sprites/flamelet3.vmt'),
    mat('materials/sprites/flamelet4.vmt'),
    mat('materials/sprites/flamelet5.vmt'),
    mat('materials/particle/particle_smokegrenade.vmt'),
    mat('materials/particle/particle_noisesphere.vmt'),
    )
res('env_funnel', mat('materials/sprites/flare6.vmt'))

res('env_global')
res('env_global_light')
res('env_gunfire')


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
        pack.pack_file('particle/particle_noisesphere', FileType.MATERIAL)  # AR2 explosion

    # All three could be used, depending on exactly where the ent is and other
    # stuff we can't easily check.
    pack.pack_file('models/props_combine/headcrabcannister01a.mdl', FileType.MODEL)
    pack.pack_file('models/props_combine/headcrabcannister01b.mdl', FileType.MODEL)
    pack.pack_file('models/props_combine/headcrabcannister01a_skybox.mdl', FileType.MODEL)
    
    pack.pack_soundscript('HeadcrabCanister.AfterLanding')
    pack.pack_soundscript('HeadcrabCanister.Open')
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
        for args in CLASS_RESOURCES[headcrab]:
            pack.pack_file(*args)


res('env_laser')
res('env_lightglow', mat('materials/sprites/light_glow02_add_noz.vmt'))
res('env_lightrail_endpoint',
    mat('materials/effects/light_rail_endpoint.vmt'),
    mat('materials/effects/strider_muzzle.vmt'),
    mat('materials/effects/combinemuzzle2.vmt'),
    mat('materials/effects/combinemuzzle2_dark.vmt'),
    )
res('env_physexplosion')
res('env_physics_blocker')
res('env_physimpact')


res('env_portal_laser',
    mdl('models/props/laser_emitter.mdl'),
    sound('Flesh.LaserBurn'),
    sound('Laser.BeamLoop'),
    sound('Player.PainSmall'),
    part('laser_start_glow'),
    part('reflector_start_glow'),
    )

res('env_projectedtexture')  # Texture handled by generic FGD parsing.
res('env_rotorwash_emitter',
    mat('materials/effects/splashwake3.vmt'),  # Water ripples
    mat('materials/effects/splash1.vmt'),  # Over water
    mat('materials/effects/splash2.vmt'),
    mat("materials/particle/particle_smokegrenade.vmt"),  # Over ground
    mat("materials/particle/particle_noisesphere.vmt"),
    )


@cls_func
def env_rotorshooter(pack: PackList, ent: Entity):
    """Inherits from env_shooter, so it just does that."""
    env_shooter(pack, ent)

res('env_screeneffect',
    mat('materials/effects/stun.vmt'),
    mat('materials/effects/introblur.vmt'),
    )


@cls_func
def env_shooter(pack: PackList, ent: Entity):
    """A hardcoded array of sounds to play."""
    try:
        snd_name = (
            "Breakable.MatGlass", 
            "Breakable.MatWood", 
            "Breakable.MatMetal", 
            "Breakable.MatFlesh", 
            "Breakable.MatConcrete",
        )[conv_int(ent['spawnflags'])]
        pack.pack_soundscript(snd_name)
    except IndexError:
        pass

    # Valve does this same check.
    if ent['shootmodel'].casefold().endswith('.vmt'):
        pack.pack_file(ent['shootmodel'], FileType.MATERIAL)
    else:
        pack.pack_file(ent['shootmodel'], FileType.MODEL)


res('env_spark',
    mat('materials/sprites/glow01.vmt'),
    mat('materials/effects/yellowflare.vmt'),
    sound('DoSpark'),
    )
res('env_smoketrail',
    mat("materials/particle/particle_smokegrenade.vmt"),
    mat("materials/particle/particle_noisesphere.vmt"),
    )


@cls_func
def env_smokestack(pack: PackList, ent: Entity) -> None:
    """This tries using each numeric material that exists."""
    pack.pack_file('materials/particle/SmokeStack.vmt', FileType.MATERIAL)

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

res('env_starfield',
    mat('materials/effects/spark_noz.vmt'),
    )

res('env_steam',
    mat('materials/particle/particle_smokegrenade.vmt'),
    mat('materials/sprites/heatwave.vmt'),
    )
res('env_steamjet', includes='env_steam')
res('env_sun', mat('materials/sprites/light_glow02_add_noz.vmt'))
res('env_texturetoggle')

res('event_queue_saveload_proxy')

res('filter_activator_class')
res('filter_activator_classify')
res('filter_activator_context')
res('filter_activator_flag')
res('filter_activator_hintgroup')
res('filter_activator_infected_class')
res('filter_activator_involume')
res('filter_activator_keyfield')
res('filter_activator_mass_greater')
res('filter_activator_model')
res('filter_activator_name')
res('filter_activator_relationship')
res('filter_activator_squad')
res('filter_activator_surfacedata')
res('filter_activator_team')
res('filter_activator_tfteam')
res('filter_damage_class')
res('filter_base')
res('filter_blood_control')
res('filter_combineball_type')
res('filter_damage_mod')
res('filter_damage_transfer')
res('filter_damage_type')
res('filter_enemy')
res('filter_health')
res('filter_melee_damage')
res('filter_multi')
res('filter_player_held')
res('filter_redirect_inflictor')
res('filter_redirect_owner')
res('filter_redirect_weapon')
res('filter_tf_bot_has_tag')
res('filter_tf_class')
res('filter_tf_condition')
res('filter_tf_damaged_by_weapon_in_slot')
res('filter_tf_player_can_cap')


@cls_func
def func_breakable_surf(pack: PackList, ent: Entity):
    """Additional materials required for func_breakable_surf."""
    pack.pack_file('models/brokenglass_piece.mdl', FileType.MODEL)

    surf_type = conv_int(ent['surfacetype'])

    if surf_type == 1:  # Tile
        mat_type = 'tile'
    elif surf_type == 0:  # Glass
        mat_type = 'glass'
        pack.pack_file('materials/models/brokenglass/glassbroken_solid.vmt', FileType.MATERIAL)
    else:
        # Unknown
        return

    for num in '123':
        for letter in 'abcd':
            pack.pack_file(
                'materials/models/broken{0}/'
                '{0}broken_0{1}{2}.vmt'.format(mat_type, num, letter),
                FileType.MATERIAL,
            )

res('func_dust',
    'materials/particle/sparkles.vmt',
    )
res('func_movelinear')
res('func_portal_bumper')
res('func_portal_detector')
res('func_portal_orientation')
res('func_portalled')

res('func_tankchange', sound('FuncTrackChange.Blocking'))
res('func_recharge',
    sound('SuitRecharge.Deny'),
    sound('SuitRecharge.Start'),
    sound('SuitRecharge.ChargingLoop'),
    )
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

res('item_grubnugget',  # Antlion Grub Nugget
    mdl('models/grub_nugget_small.mdl'),
    mdl('models/grub_nugget_medium.mdl'),
    mdl('models/grub_nugget_large.mdl'),
    sound('GrubNugget.Touch'),
    sound('NPC_Antlion_Grub.Explode'),
    part('antlion_spit_player'),
    )
res('item_healthkit',
    mdl('models/healthkit.mdl'),
    sound('HealthKit.Touch'),
    )
res('item_healthvial',
    mdl('models/healthvial.mdl'),
    sound('HealthVial.Touch'),
    )
res('item_healthcharger',
    mdl('models/props_combine/health_charger001.mdl'),
    sound('WallHealth.Deny'),
    sound('WallHealth.Start'),
    sound('WallHealth.LoopingContinueCharge'),
    sound('WallHealth.Recharge'),
    )
res('item_suitcharger',
    mdl('models/props_combine/suit_charger001.mdl'),
    sound('WallHealth.Deny'),
    sound('WallHealth.Start'),
    sound('WallHealth.LoopingContinueCharge'),
    sound('WallHealth.Recharge'),
    )


@cls_func
def item_teamflag(pack: PackList, ent: Entity) -> None:
    """This item has several special team-specific options."""
    for kvalue, prefix in [
        ('flag_icon', 'materials/vgui/'),
        ('flag_trail', 'materials/effects/')
    ]:
        value = prefix + ent[kvalue]
        if value != prefix:
            pack.pack_file(value + '.vmt', FileType.MATERIAL)
            pack.pack_file(value + '_red.vmt', FileType.MATERIAL)
            pack.pack_file(value + '_blue.vmt', FileType.MATERIAL)


@cls_func
def npc_antlion(pack: PackList, ent: Entity):
    """Antlions require different resources for the worker version."""
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 18):  # Is worker?
        pack.pack_file("models/antlion_worker.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_worker_01", FileType.PARTICLE)
        pack.pack_file("antlion_gib_02", FileType.PARTICLE)
        pack.pack_file("blood_impact_yellow_01", FileType.PARTICLE)

        for fname, ftype in CLASS_RESOURCES['grenade_spit']:
            pack.pack_file(fname, ftype)
    else:
        pack.pack_file("models/antlion.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_01")
        pack.pack_file("AntlionGib", FileType.PARTICLE)

    for i in ('1', '2', '3'):
        for size in ('small', 'medium', 'large'):
            pack.pack_file("models/gibs/antlion_gib_{}_{}.mdl".format(size, i), FileType.MODEL)

    pack.pack_soundscript("NPC_Antlion.RunOverByVehicle")
    pack.pack_soundscript("NPC_Antlion.MeleeAttack")
    pack.pack_soundscript("NPC_Antlion.Footstep")
    pack.pack_soundscript("NPC_Antlion.BurrowIn")
    pack.pack_soundscript("NPC_Antlion.BurrowOut")
    pack.pack_soundscript("NPC_Antlion.FootstepSoft")
    pack.pack_soundscript("NPC_Antlion.FootstepHeavy")
    pack.pack_soundscript("NPC_Antlion.MeleeAttackSingle")
    pack.pack_soundscript("NPC_Antlion.MeleeAttackDouble")
    pack.pack_soundscript("NPC_Antlion.Distracted")
    pack.pack_soundscript("NPC_Antlion.Idle")
    pack.pack_soundscript("NPC_Antlion.Pain")
    pack.pack_soundscript("NPC_Antlion.Land")
    pack.pack_soundscript("NPC_Antlion.WingsOpen")
    pack.pack_soundscript("NPC_Antlion.LoopingAgitated")
    pack.pack_soundscript("NPC_Antlion.Distracted")

    # TODO: These are Episodic only..
    pack.pack_soundscript("NPC_Antlion.PoisonBurstScream")
    pack.pack_soundscript("NPC_Antlion.PoisonBurstScreamSubmerged")
    pack.pack_soundscript("NPC_Antlion.PoisonBurstExplode")
    pack.pack_soundscript("NPC_Antlion.MeleeAttack_Muffled")
    pack.pack_soundscript("NPC_Antlion.TrappedMetal")
    pack.pack_soundscript("NPC_Antlion.ZappedFlip")
    pack.pack_soundscript("NPC_Antlion.PoisonShoot")
    pack.pack_soundscript("NPC_Antlion.PoisonBall")

res('npc_antlion_grub',
    mdl("models/antlion_grub.mdl"),
    mdl("models/antlion_grub_squashed.mdl"),
    mat("materials/sprites/grubflare1.vmt"),
    sound("NPC_Antlion_Grub.Idle"),
    sound("NPC_Antlion_Grub.Alert"),
    sound("NPC_Antlion_Grub.Stimulated"),
    sound("NPC_Antlion_Grub.Die"),
    sound("NPC_Antlion_Grub.Squish"),
    part("GrubSquashBlood"),
    part("GrubBlood"),
    includes="item_grubnugget",
    )


@cls_func
def npc_antlion_template_maker(pack: PackList, ent: Entity):
    """Depending on KVs this may or may not spawn workers."""
    # There will be an antlion present in the map, as the template
    # NPC. So we don't need to add those resources.
    if conv_int(ent['workerspawnrate']) > 0:
        # It randomly spawns worker antlions, so load that resource set.
        pack.pack_file("models/antlion_worker.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_worker_01", FileType.PARTICLE)
        pack.pack_file("antlion_gib_02", FileType.PARTICLE)
        pack.pack_file("blood_impact_yellow_01", FileType.PARTICLE)

        for fname, ftype in CLASS_RESOURCES['grenade_spit']:
            pack.pack_file(fname, ftype)

res('npc_barnacle',
    mdl('models/barnacle.mdl'),
    mdl("models/gibs/hgibs.mdl"),
    mdl("models/gibs/hgibs_scapula.mdl"),
    mdl("models/gibs/hgibs_rib.mdl"),
    mdl("models/gibs/hgibs_spine.mdl"),
    sound("NPC_Barnacle.Digest"),
    sound("NPC_Barnacle.BreakNeck"),
    sound("NPC_Barnacle.Scream"),
    sound("NPC_Barnacle.PullPant"),
    sound("NPC_Barnacle.TongueStretch"),
    sound("NPC_Barnacle.FinalBite"),
    sound("NPC_Barnacle.Die"),
    includes='npc_barnacle_tongue_tip',
    )
res('npc_barnacle_tongue_tip', 'models/props_junk/rock001a.mdl')  # Random model it loads.
res('npc_combine_cannon',
    mdl('models/combine_soldier.mdl'),
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    sound('NPC_Combine_Cannon.FireBullet'),
    )
res('npc_zombie',
    mdl("models/zombie/classic.mdl"),
    mdl("models/zombie/classic_torso.mdl"),
    mdl("models/zombie/classic_legs.mdl"),
    sound("Zombie.FootstepRight"),
    sound("Zombie.FootstepLeft"),
    sound("Zombie.FootstepLeft"),
    sound("Zombie.ScuffRight"),
    sound("Zombie.ScuffLeft"),
    sound("Zombie.AttackHit"),
    sound("Zombie.AttackMiss"),
    sound("Zombie.Pain"),
    sound("Zombie.Die"),
    sound("Zombie.Alert"),
    sound("Zombie.Idle"),
    sound("Zombie.Attack"),
    sound("NPC_BaseZombie.Moan1"),
    sound("NPC_BaseZombie.Moan2"),
    sound("NPC_BaseZombie.Moan3"),
    sound("NPC_BaseZombie.Moan4"),
    )
# Actually an alias, but we don't want to swap these.
CLASS_RESOURCES['npc_zombie_torso'] = CLASS_RESOURCES['npc_zombie']

res('npc_fastzombie',
    mdl("models/zombie/fast.mdl"),
    # TODO - Episodic only:
        mdl("models/zombie/Fast_torso.mdl"),
        sound("NPC_FastZombie.CarEnter1"),
        sound("NPC_FastZombie.CarEnter2"),
        sound("NPC_FastZombie.CarEnter3"),
        sound("NPC_FastZombie.CarEnter4"),
        sound("NPC_FastZombie.CarScream"),
    mdl("models/gibs/fast_zombie_torso.mdl"),
    mdl("models/gibs/fast_zombie_legs.mdl"),
    sound("NPC_FastZombie.LeapAttack"),
    sound("NPC_FastZombie.FootstepRight"),
    sound("NPC_FastZombie.FootstepLeft"),
    sound("NPC_FastZombie.AttackHit"),
    sound("NPC_FastZombie.AttackMiss"),
    sound("NPC_FastZombie.LeapAttack"),
    sound("NPC_FastZombie.Attack"),
    sound("NPC_FastZombie.Idle"),
    sound("NPC_FastZombie.AlertFar"),
    sound("NPC_FastZombie.AlertNear"),
    sound("NPC_FastZombie.GallopLeft"),
    sound("NPC_FastZombie.GallopRight"),
    sound("NPC_FastZombie.Scream"),
    sound("NPC_FastZombie.RangeAttack"),
    sound("NPC_FastZombie.Frenzy"),
    sound("NPC_FastZombie.NoSound"),
    sound("NPC_FastZombie.Die"),
    sound("NPC_FastZombie.Gurgle"),
    sound("NPC_FastZombie.Moan1"),
    )
# Actually an alias, but we don't want to swap these.
CLASS_RESOURCES['npc_fastzombie_torso'] = CLASS_RESOURCES['npc_fastzombie']

res('npc_headcrab',
    mdl('models/headcrabclassic.mdl'),
    sound('NPC_HeadCrab.Gib'),
    sound('NPC_HeadCrab.Idle'),
    sound('NPC_HeadCrab.Alert'),
    sound('NPC_HeadCrab.Pain'),
    sound('NPC_HeadCrab.Die'),
    sound('NPC_HeadCrab.Attack'),
    sound('NPC_HeadCrab.Bite'),
    sound('NPC_Headcrab.BurrowIn'),
    sound('NPC_Headcrab.BurrowOut'),
    )

res('npc_headcrab_black',
    mdl('models/headcrabblack.mdl'),

    sound('NPC_BlackHeadcrab.Telegraph'),
    sound('NPC_BlackHeadcrab.Attack'),
    sound('NPC_BlackHeadcrab.Bite'),
    sound('NPC_BlackHeadcrab.Threat'),
    sound('NPC_BlackHeadcrab.Alert'),
    sound('NPC_BlackHeadcrab.Idle'),
    sound('NPC_BlackHeadcrab.Talk'),
    sound('NPC_BlackHeadcrab.AlertVoice'),
    sound('NPC_BlackHeadcrab.Pain'),
    sound('NPC_BlackHeadcrab.Die'),
    sound('NPC_BlackHeadcrab.Impact'),
    sound('NPC_BlackHeadcrab.ImpactAngry'),
    sound('NPC_BlackHeadcrab.FootstepWalk'),
    sound('NPC_BlackHeadcrab.Footstep'),

    sound('NPC_HeadCrab.Gib'),
    sound('NPC_Headcrab.BurrowIn'),
    sound('NPC_Headcrab.BurrowOut'),
    )

CLASS_RESOURCES['npc_headcrab_poison'] = CLASS_RESOURCES['npc_headcrab_black']  # Alias

res('npc_headcrab_fast',
    mdl('models/headcrab.mdl'),
    sound('NPC_FastHeadCrab.Idle'),
    sound('NPC_FastHeadCrab.Alert'),
    sound('NPC_FastHeadCrab.Pain'),
    sound('NPC_FastHeadCrab.Die'),
    sound('NPC_FastHeadCrab.Attack'),
    sound('NPC_FastHeadCrab.Bite'),

    sound('NPC_HeadCrab.Gib'),
    sound('NPC_Headcrab.BurrowIn'),
    sound('NPC_Headcrab.BurrowOut'),
    )

res('npc_heli_avoidbox')
res('npc_heli_avoidsphere')
res('npc_heli_nobomb')
res('npc_helicopter',
    mdl("models/combine_helicopter.mdl"),
    mdl("models/combine_helicopter_broken.mdl"),
    mat("materials/sprites/redglow1.vmt"),
    includes='helicopter_chunk grenade_helicopter',
    )


@cls_func
def move_rope(pack: PackList, ent: Entity):
    """Implement move_rope and keyframe_rope resources."""
    old_shader_type = conv_int(ent['RopeShader'])
    if old_shader_type == 0:
        yield 'materials/cable/cable.vmt'
    elif old_shader_type == 1:
        yield 'materials/cable/rope.vmt'
    else:
        yield 'materials/cable/chain.vmt'
    yield 'materials/cable/rope_shadowdepth.vmt'

# These classes are identical.
CLASS_RESOURCES['keyframe_rope'] = CLASS_RESOURCES['move_rope']
ALT_NAMES['keyframe_rope'] = 'move_rope'

res('phys_bone_follower')
res('physics_entity_solver')
res('physics_npc_solver')

res('npc_rocket_turret',
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    mdl('models/props_bts/rocket_sentry.mdl'),
    sound('NPC_RocketTurret.LockingBeep'),
    sound('NPC_FloorTurret.LockedBeep'),
    sound('NPC_FloorTurret.RocketFire'),
    includes='rocket_turret_projectile',
    )

res('npc_vehicledriver',
    'models/roller_vehicledriver.mdl',
    )

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

res('point_spotlight',
    'materials/sprites/light_glow03.vmt',
    'materials/sprites/glow_test02.vmt',
)

res('prop_floor_button')  # Model set in keyvalues.
res('prop_floor_ball_button', mdl('models/props/ball_button.mdl'))
res('prop_floor_cube_button', mdl('models/props/box_socket.mdl'))
res('prop_button', 
    mdl('models/props/switch001.mdl'),
    sound('Portal.button_down'),
    sound('Portal.button_up'),
    sound('Portal.button_locked'),
    sound('Portal.room1_TickTock'),
    )

res('prop_energy_ball',
    mdl('models/effects/combineball.mdl'),
    mat('materials/effects/eball_finite_life.vmt'),
    mat('materials/effects/eball_infinite_life.vmt'),
    sound('EnergyBall.Explosion'),
    sound('EnergyBall.Launch'),
    sound('EnergyBall.KillImpact'),
    sound('EnergyBall.Impact'),
    sound('EnergyBall.AmbientLoop'),
    )

res('prop_exploding_futbol',
    mdl('models/npcs/personality_sphere_angry.mdl'),
    part('bomb_trail'),
    )

res('prop_laser_catcher',
    part('laser_relay_powered'),
    sound('prop_laser_catcher.powerloop'),
    sound('prop_laser_catcher.poweroff'),
    sound('prop_laser_catcher.poweron'),
    )

res('prop_laser_relay', 
    mdl('models/props/laser_receptacle.mdl'),
    part('laser_relay_powered'),
    sound('prop_laser_catcher.powerloop'),
    sound('prop_laser_catcher.poweroff'),
    sound('prop_laser_catcher.poweron'),
    )

res('prop_linked_portal_door', mdl('models/props/portal_door.mdl'))

res('prop_monster_box',
    mdl('models/npcs/monsters/monster_a.mdl'),
    mdl('models/npcs/monsters/monster_a_box.mdl'),
    sound('DoSparkSmaller'),
    )

res('prop_under_button', 
    mdl('models/props_underground/underground_testchamber_button.mdl'),
    sound('Portal.button_down'),
    sound('Portal.button_up'),
    sound('Portal.button_locked'),
    sound('Portal.room1_TickTock'),
    )
res('prop_under_floor_button', mdl('models/props_underground/underground_floor_button.mdl'))

res('prop_wall_projector',
    mdl('models/props/wall_emitter.mdl'),
    sound('VFX.BridgeGlow'),
    sound('music.ctc_lbout'),
    # sound('music.mapname_lbout')
    sound('music.sp_all_maps_lbout'),
    part('projected_wall_impact'),
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

res('trigger_active_weapon_detect')
res('trigger_add_tf_player_condition')
res('trigger_apply_impulse')
res('trigger_asw_button_area')
res('trigger_asw_chance')
res('trigger_asw_computer_area')
res('trigger_asw_door_area')
res('trigger_asw_jump')
res('trigger_asw_marine_knockback')
res('trigger_asw_marine_position')
res('trigger_asw_random_target')
res('trigger_asw_supplies_chatter')
res('trigger_asw_synup_chatter')
res('trigger_auto_crouch')
res('trigger_autosave')
res('trigger_bomb_reset')
res('trigger_bot_tag')
res('trigger_capture_area')
res('trigger_catapult')
res('trigger_changelevel')
res('trigger_escape')
res('trigger_fall')
res('trigger_finale')
res('trigger_finale_dlc3')
res('trigger_gravity')
res('trigger_hierarchy')
res('trigger_hurt')
res('trigger_hurt_ghost')
res('trigger_ignite')
res('trigger_ignite_arrows')
res('trigger_impact')
res('trigger_look')
res('trigger_multiple')
res('trigger_once')
res('trigger_paint_cleanser')
res('trigger_passtime_ball')
res('trigger_physics_trap')
res('trigger_ping_detector')
res('trigger_player_respawn_override')
res('trigger_playermovement')
res('trigger_playerteam')

res(
    'trigger_portal_cleanser',
    part('cleanser_scanline'),
    sound('VFX.FizzlerLp'),
    sound('VFX.FizzlerDestroy'),
    sound('VFX.FizzlerStart'),
    sound('VFX.FizzlerVortexLp'),
    sound('Prop.Fizzled'),
    )

res('trigger_proximity')
res('trigger_push')
res('trigger_rd_vault_trigger')
res('trigger_remove')
res('trigger_remove_tf_player_condition')
res('trigger_rpgfire')
res('trigger_serverragdoll')
res('trigger_softbarrier')
res('trigger_soundoperator')
res('trigger_soundscape')
res('trigger_standoff')
res('trigger_stun')
res('trigger_teleport')
res('trigger_teleport_relative')
res('trigger_timer_door')
res('trigger_tonemap')
res('trigger_transition')
res('trigger_upgrade_laser_sight')
res('trigger_vphysics_motion')
res('trigger_waterydeath',
    mdl("models/leech.mdl"),
    sound("coast.leech_bites_loop"),
    sound("coast.leech_water_churn_loop"),
    )
res('trigger_weapon_dissolve')
res('trigger_weapon_strip')
res('trigger_wind')

res('vgui_screen',
    'materials/engine/writez.vmt',
    )
res('waterbullet', 'models/weapons/w_bullet.mdl')


# Now all of these have been done, apply 'includes' commands.
_process_includes()