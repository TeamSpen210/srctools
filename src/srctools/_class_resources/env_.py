"""env_ entities."""
import itertools
from srctools._class_resources import *

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
res('env_entity_dissolver', mat('materials/sprites/blueglow1.vmt'))
res('env_embers', mat('materials/particle/fire.vmt'))
res('env_explosion', mat('materials/sprites/zerogxplode.vmt'))

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
    fire_type = conv_int(ent['firetype'])
    if fire_type == 0:  # Natural
        flags = conv_int(ent['spawnflags'])
        if flags & 2:  # Smokeless?
            suffix = ''  # env_fire_small
        else:
            suffix = '_smoke'  # env_fire_medium_smoke
        for name in ['tiny', 'small', 'medium', 'large']:
            pack.pack_file('env_fire_{}{}'.format(name, suffix),
                           FileType.PARTICLE_SYSTEM)
    elif fire_type == 1:  # Plasma
        for fname, ftype in CLASS_RESOURCES['_plasma']:
            pack.pack_file(fname, ftype)
res('env_fire', sound('Fire.Plasma'))

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
        pack.pack_file('particle/particle_noisesphere',
                       FileType.MATERIAL)  # AR2 explosion

    # All three could be used, depending on exactly where the ent is and other
    # stuff we can't easily check.
    pack.pack_file('models/props_combine/headcrabcannister01a.mdl',
                   FileType.MODEL)
    pack.pack_file('models/props_combine/headcrabcannister01b.mdl',
                   FileType.MODEL)
    pack.pack_file('models/props_combine/headcrabcannister01a_skybox.mdl',
                   FileType.MODEL)

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
res('env_laserdot', mat('materials/sprites/redglow1.vmt'))
res('env_lightglow', mat('materials/sprites/light_glow02_add_noz.vmt'))
res('env_lightrail_endpoint',
    mat('materials/effects/light_rail_endpoint.vmt'),
    mat('materials/effects/strider_muzzle.vmt'),
    mat('materials/effects/combinemuzzle2.vmt'),
    mat('materials/effects/combinemuzzle2_dark.vmt'),
    )
res('env_microphone')
res('env_movieexplosion', mat('materials/particle/particle_sphere.vmt'))
res('env_hudhint')
res('env_player_surface_trigger')
res('env_physexplosion')
res('env_physics_blocker')
res('env_physimpact')
res('env_physwire')
res('env_quadratic_beam')
res('env_rockettrail',
    mat('materials/effects/muzzleflash1.vmt'),
    mat('materials/effects/muzzleflash2.vmt'),
    mat('materials/effects/muzzleflash3.vmt'),
    mat('materials/effects/muzzleflash4.vmt'),
    mat('materials/sprites/flamelet1.vmt'),
    mat('materials/sprites/flamelet2.vmt'),
    mat('materials/sprites/flamelet3.vmt'),
    mat('materials/sprites/flamelet4.vmt'),
    )

res('env_portal_laser',
    mdl('models/props/laser_emitter.mdl'),
    sound('Flesh.LaserBurn'),
    sound('Laser.BeamLoop'),
    sound('Player.PainSmall'),
    part('laser_start_glow'),
    part('reflector_start_glow'),
    )

res('env_projectedtexture')  # Texture handled by generic FGD parsing.
res('env_ragdoll_boogie', sound('RagdollBoogie.Zap'))
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
res('env_splash',
    mat('materials/effects/splash2.vmt'),
    mat('materials/effects/splashwake1.vmt'),
    sound('Physics.WaterSplash'),
    part('slime_splash_01'),
    part('slime_splash_02'),
    part('slime_splash_03'),
    )
res('env_sprite', aliases='env_glow')
res('env_sprite_clientside')
res('env_sprite_oriented')
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
