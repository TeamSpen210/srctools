"""asw_ entities."""
from srctools._class_resources import *


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
