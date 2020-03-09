"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in keyvalues.
"""
from typing import Iterator, Callable, Tuple, Union, List, Dict, Iterable

from srctools.packlist import FileType
from srctools import Entity, conv_int

__all__ = ['CLASS_RESOURCES', 'ALT_NAMES']

#  For various entity classes, we know they require hardcoded files.
# List them here - classname -> [(file, type), ...]
# Alternatively it's a function to call with the entity to do class-specific
# behaviour, yielding files to pack.

ClassFunc = Callable[[Entity], Iterator[Union[str, Tuple[str, FileType]]]]
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

res('ambient_generic')  # Sound is a keyvalue
res('ambient_music')

# The actual explosion itself.
res('ar2explosion', mat("materials/particle/particle_noisesphere.vmt"))
res('assault_assaultpoint')
res('assault_rallypoint')

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
def asw_emitter(ent: Entity):
    """Complicated thing, probably can't fully process here."""
    template = ent['template']
    if template and template != 'None':
        yield 'resource/particletemplates/{}.ptm'.format(template)

    # TODO: Read the following keys from the file:
    # - "material"
    # - "glowmaterial"
    # - "collisionsound"
    # - "collisiondecal"

    yield mat('materials/effects/yellowflare.vmt')

@cls_func
def asw_env_explosion(ent: Entity):
    """Handle the no-sound spawnflag."""
    yield part("asw_env_explosion")
    if (conv_int(ent['spawnflags']) & 0x00000010) == 0:
        yield sound('ASW_Explosion.Explosion_Default')

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

res('commentary_auto')
res('commentary_dummy')
res('commentary_zombie_spawner')

res('comp_choreo_sceneset')
res('comp_entity_finder')
res('comp_entity_mover')
res('comp_kv_setter')
res('comp_numeric_transition')

# Handled by srctools.bsp_transform.packing transforms.
res('comp_pack')
res('comp_pack_rename')
res('comp_pack_replace_soundscript')
res('comp_precache_model')
res('comp_precache_sound')

res('comp_propcombine_set')
res('comp_scriptvar_setter')
res('comp_trigger_coop')
res('comp_trigger_p2_goo')

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
def env_break_shooter(ent: Entity):
    """Special behaviour on the 'model' KV."""
    if conv_int(ent['modeltype']) == 1: # MODELTYPE_MODEL
        yield mdl(ent['model'])
    # Otherwise, a template name or a regular gib.

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
def env_fire(ent: Entity):
    """Two types of fire, with different resources."""
    yield sound('Fire.Plasma')
    fire_type = conv_int(ent['firetype'])
    if fire_type == 0:  # Natural
        flags = conv_int(ent['spawnflags'])
        if flags & 2:  # Smokeless?
            suffix = ''  # env_fire_small
        else:
            suffix = '_smoke'  # env_fire_medium_smoke
        for name in ['tiny', 'small', 'medium', 'large']:
            yield part('env_fire_{}{}'.format(name, suffix))
    elif fire_type == 1:  # Plasma
        yield from CLASS_RESOURCES['_plasma']

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
def env_headcrabcanister(ent: Entity):
    """Check if it spawns in skybox or not, and precache the headcrab."""
    flags = conv_int(ent['spawnflags'])
    if flags & 0x1 == 0:  # !SF_NO_IMPACT_SOUND
        yield sound('HeadcrabCanister.Explosion')
        yield sound('HeadcrabCanister.IncomingSound')
        yield sound('HeadcrabCanister.SkyboxExplosion')
    if flags & 0x2 == 0:  # !SF_NO_LAUNCH_SOUND
        yield sound('HeadcrabCanister.LaunchSound')
    if flags & 0x1000 == 0:  # !SF_START_IMPACTED
        yield mat('materials/sprites/smoke.vmt')

    if flags & 0x80000 == 0:  # !SF_NO_IMPACT_EFFECTS
        yield mat('particle/particle_noisesphere')  # AR2 explosion

    # All three could be used, depending on exactly where the ent is and other
    # stuff we can't easily check.
    yield mdl('models/props_combine/headcrabcannister01a.mdl')
    yield mdl('models/props_combine/headcrabcannister01b.mdl')
    yield mdl('models/props_combine/headcrabcannister01a_skybox.mdl')
    
    yield sound('HeadcrabCanister.AfterLanding')
    yield sound('HeadcrabCanister.Open')
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
        yield from CLASS_RESOURCES[headcrab]


res('env_laser')
res('env_lightglow', mat('materials/sprites/light_glow02_add_noz.vmt'))
res('env_lightrail_endpoint',
    mat('effects/light_rail_endpoint'),
    mat('effects/strider_muzzle'),
    mat('effects/combinemuzzle2'),
    mat('effects/combinemuzzle2_dark'),
    )
res('env_physexplosion')
res('env_physics_blocker')
res('env_physimpact')


@cls_func
def env_portal_laser(ent: Entity):
    """The P2 laser defaults to a model if not provided."""
    if not ent['model']:
        yield mdl('models/props/laser_emitter.mdl')
    yield sound('Flesh.LaserBurn')
    yield sound('Laser.BeamLoop')
    yield sound('Player.PainSmall')
    yield part('laser_start_glow')
    yield part('reflector_start_glow')

res('env_rotorwash_emitter',
    mat('materials/effects/splashwake3.vmt'),  # Water ripples
    mat('materials/effects/splash1.vmt'),  # Over water
    mat('materials/effects/splash2.vmt'),
    mat("materials/particle/particle_smokegrenade.vmt"),  # Over ground
    mat("materials/particle/particle_noisesphere.vmt"),
    )

res('env_screeneffect',
    'materials/effects/stun.vmt',
    'materials/effects/introblur.vmt',
    )


@cls_func
def env_shooter(ent: Entity):
    """A hardcoded array of sounds to play."""
    try:
        yield (
            "Breakable.MatGlass", 
            "Breakable.MatWood", 
            "Breakable.MatMetal", 
            "Breakable.MatFlesh", 
            "Breakable.MatConcrete",
        )[conv_int(ent['spawnflags'])]
    except IndexError:
        pass

res('env_spark',
    mat('materials/sprites/glow01.vmt'),
    mat('materials/effects/yellowflare.vmt'),
    sound('DoSpark'),
    )
res('env_smoketrail',
    mat("materials/particle/particle_smokegrenade.vmt"),
    mat("materials/particle/particle_noisesphere.vmt"),
    )
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

res('filter_activator_class')
res('filter_activator_classify')
res('filter_activator_context')
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
def func_breakable_surf(ent: Entity):
    """Additional materials required for func_breakable_surf."""
    yield 'models/brokenglass_piece.mdl'

    surf_type = conv_int(ent['surfacetype'])
    if surf_type == 1:  # Tile
        for num in '123':
            for letter in 'abcd':
                yield mat('materials/models/brokentile/tilebroken_0{}{}.vmt'.format(num, letter))
    elif surf_type == 0:  # Glass
        yield mat('materials/models/brokenglass/glassbroken_solid.vmt')
        for num in '123':
            for letter in 'abcd':
                yield mat('materials/models/brokenglass/'
                          'glassbroken_0{}{}.vmt'.format(num, letter))

res('func_dust',
    'materials/particle/sparkles.vmt',
    )

res('func_tankchange',
    ('FuncTrackChange.Blocking', FileType.GAME_SOUND),
    )

res('grenade_helicopter',  # Bomb dropped by npc_helicopter
    mdl("models/combine_helicopter/helicopter_bomb01.mdl"),
    sound("ReallyLoudSpark"),
    sound("NPC_AttackHelicopterGrenade.Ping"),
    sound("NPC_AttackHelicopterGrenade.PingCaptured"),
    sound("NPC_AttackHelicopterGrenade.HardImpact"),
    )

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
def item_teamflag(ent: Entity) -> Iterator[str]:
    """This item has several special team-specific options."""
    for kvalue, prefix in [
        ('flag_icon', 'materials/vgui/'),
        ('flag_trail', 'materials/effects/')
    ]:
        value = prefix + ent[kvalue]
        if value != prefix:
            yield value + '.vmt'
            yield value + '_red.vmt'
            yield value + '_blue.vmt'

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
    'models/combine_soldier.mdl',
    'materials/effects/bluelaser1.vmt',
    'materials/sprites/light_glow03.vmt',
    ('NPC_Combine_Cannon.FireBullet', FileType.GAME_SOUND),
    )

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
def move_rope(ent: Entity):
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
    'materials/effects/bluelaser1.vmt',
    'materials/sprites/light_glow03.vmt',
    'models/props_bts/rocket_sentry.mdl',
    ('NPC_RocketTurret.LockingBeep', FileType.GAME_SOUND),
    ('NPC_FloorTurret.LockedBeep', FileType.GAME_SOUND),
    ('NPC_FloorTurret.RocketFire', FileType.GAME_SOUND),
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

res('rocket_turret_projectile',
    'models/props_bts/rocket.mdl',
    'materials/decals/scorchfade.vmt',
    ('NPC_FloorTurret.RocketFlyLoop', FileType.GAME_SOUND),
    )

res('squadinsignia', "models/chefhat.mdl")  # Yeah.

@cls_func
def team_control_point(ent: Entity) -> Iterator[str]:
    """Special '_locked' materials."""
    for kvalue in ['team_icon_0', 'team_icon_1', 'team_icon_2']:
        mat = ent[kvalue]
        if mat:
            yield 'materials/{}.vmt'.format(mat)
            yield 'materials/{}_locked.vmt'.format(mat)

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
res('trigger_waterydeath')
res('trigger_weapon_dissolve')
res('trigger_weapon_strip')
res('trigger_wind')

res('vgui_screen',
    'materials/engine/writez.vmt',
    )


# Now all of these have been done, apply 'includes' commands.
_process_includes()