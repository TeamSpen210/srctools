"""asw_ entities."""
from . import *


# CASW_Base_Spawner
BASE_SPAWNER = [
    sound('Spawner.Horde'),
    sound('Spawner.AreaClear'),
]


res('asw_camera_control')
res('asw_alien',
    part('drone_death'),
    part('drone_shot'),
    part('freeze_statue_shatter'),
    # TODO: Also, particle gib effects in model??
    )
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
def asw_emitter(pack: PackList, ent: Entity) -> None:
    """Complicated thing, probably can't fully process here."""
    template = ent['template']
    if template and template != 'None':
        pack.pack_file(f'resource/particletemplates/{template}.ptm')

    # TODO: Read the following keys from the file:
    # - "material"
    # - "glowmaterial"
    # - "collisionsound"
    # - "collisiondecal"

res('asw_emitter', mat('materials/effects/yellowflare.vmt'))


@cls_func
def asw_env_explosion(pack: PackList, ent: Entity) -> None:
    """Handle the no-sound spawnflag."""
    if (conv_int(ent['spawnflags']) & 0x00000010) == 0:
        pack.pack_file('ASW_Explosion.Explosion_Default', FileType.GAME_SOUND)

res('asw_env_explosion', part('asw_env_explosion'))
res('asw_env_shake')
res('asw_env_spark',
    part('asw_env_sparks'),
    sound('DoSpark'),
    )
res('asw_equip_req')

res('asw_gamerules')
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
res('asw_grub_sac', includes='asw_alien_goo')

res('asw_harvester',
    mdl('models/swarm/harvester/Harvester.mdl'),
    mdl('models/aliens/harvester/harvester.mdl'),
    sound('ASW_Harvester.Death'),
    sound('ASW_Harvester.Pain'),
    sound('ASW_Harvester.Scared'),
    sound('ASW_Harvester.SpawnCritter'),
    sound('ASW_Harvester.Alert'),
    sound('ASW_Harvester.Sniffing'),
    includes='asw_parasite_defanged',
    )
res('asw_holdout_mode')
res('asw_holdout_spawner', *BASE_SPAWNER)
res('asw_holo_sentry', part('sentry_build_display'))
res('asw_holoscan', mat('materials/swarm/effects/greenlaser1.vmt'))
res('asw_hurt_nearest_marine')

res('asw_info_heal')
res('asw_info_message')
res('asw_intro_control')
res('asw_jukebox')
res('asw_mission_objective')
res('asw_marines_past_area')
res('asw_marker')
res('asw_mortarbug',
    mdl('models/aliens/mortar/mortar.mdl'),
    sound('ASW_MortarBug.Idle'),
    sound('ASW_MortarBug.Pain'),
    sound('ASW_MortarBug.Spit'),
    sound('ASW_MortarBug.OnFire'),
    sound('ASW_MortarBug.Death'),
    part('mortar_launch'),
    includes='asw_mortarbug_shell',
    )
res('asw_mortarbug_shell',
    mdl('models/swarm/MortarBugProjectile/MortarBugProjectile.mdl'),
    sound('ASW_Boomer_Grenade.Explode'),
    sound('ASW_Boomer_Projectile.Spawned'),
    sound('ASW_Boomer_Projectile.ImpactHard'),
    part('mortar_shell_aura'),
    )
res('asw_objective_countdown',
    sound('ASW.WarheadExplosion'),
    sound('ASW.WarheadExplosionLF'),
    )
res('asw_objective_destroy_goo', includes='asw_mission_objective')
res('asw_objective_dummy', includes='asw_mission_objective')
res('asw_objective_escape', includes='asw_mission_objective')
res('asw_objective_kill_aliens', includes='asw_mission_objective')
res('asw_objective_kill_eggs', includes='asw_mission_objective')
res('asw_objective_survive', includes='asw_mission_objective')
res('asw_objective_triggered', includes='asw_mission_objective')
res('asw_order_nearby_aliens')

res('asw_pickup_ammo_bag', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_ammo_satchel', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_autogun', mdl('models/weapons/autogun/autogun.mdl'))
res('asw_pickup_buff_grenade', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_chainsaw', mdl('models/weapons/Chainsaw/Chainsaw.mdl'))
res('asw_pickup_fire_extinguisher', mdl('models/swarm/FireExt/fireextpickup.mdl'))
res('asw_pickup_flamer', mdl('models/weapons/flamethrower/flamethrower.mdl'))
res('asw_pickup_flares', mdl('models/items/itembox/itemboxsmall.mdl'))
res('asw_pickup_flashlight', mdl('models/swarm/flashlight/flashlightpickup.mdl'))
res('asw_pickup_grenades', mdl('models/swarm/grenades/grenadebox.mdl'))
res('asw_pickup_heal_grenade', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_hornet_barrage', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_medical_satchel', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_medkit', mdl('models/items/personalMedkit/personalMedkit.mdl'))
res('asw_pickup_mines', mdl('models/items/itembox/itemboxsmall.mdl'))
res('asw_pickup_mining_laser', mdl('models/weapons/MiningLaser/MiningLaser.mdl'))
res('asw_pickup_pdw', mdl('models/weapons/pdw/pdw.mdl'))
res('asw_pickup_pistol', mdl('models/weapons/pistol/Pistol.mdl'))
res('asw_pickup_prifle', mdl('models/weapons/Prototype/prototyperifle.mdl'))
res('asw_pickup_railgun', mdl('models/weapons/Railgun/Railgun.mdl'))
res('asw_pickup_rifle', mdl('models/weapons/assaultrifle/assaultrifle.mdl'))
res('asw_pickup_sentry', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_sentry_cannon', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_sentry_flamer', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_sentry_freeze', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_shotgun', mdl('models/weapons/Shotgun/Shotgun.mdl'))
res('asw_pickup_stim', mdl('models/items/itembox/itemboxsmall.mdl'))
res('asw_pickup_t75', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_tesla_trap', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_vindicator', mdl('models/weapons/vindicator/vindicator.mdl'))
res('asw_pickup_welder', mdl('models/swarm/Welder/Welder.mdl'))

res('asw_parasite',
    mdl('models/aliens/parasite/parasite.mdl'),
    sound('ASW_Parasite.Death'),
    sound('ASW_Parasite.Attack'),
    sound('ASW_Parasite.Idle'),
    sound('ASW_Parasite.Pain'),
    sound('ASW_Parasite.Attack'),
    )

CLASS_RESOURCES['asw_parasite_defanged'] = CLASS_RESOURCES['asw_parasite']  # Identical.

res('asw_queen',
    mdl('models/swarm/Queen/queen.mdl'),
    sound('ASW_Queen.Death'),
    sound('ASW_Queen.Pain'),
    sound('ASW_Queen.PainBig'),
    sound('ASW_Queen.Slash'),
    sound('ASW_Queen.SlashShort'),
    sound('ASW_Queen.AttackWave'),
    sound('ASW_Queen.Spit'),
    sound('ASW_Queen.TentacleAttackStart'),
    includes='asw_alien',
    )
res('asw_queen_retreat_spot')
res('asw_ranger',
    mdl('models/aliens/rangerSpit/rangerspit.mdl'),
    mdl('models/aliens/mortar3/mortar3.mdl'),
    sound('Ranger.projectileImpactPlayer'),
    sound('Ranger.projectileImpactWorld'),
    sound('Ranger.GibSplatHeavy'),
    sound('ASW_Drone.DeathFireSizzle'),
    sound('ASW_Ranger_Projectile.Spawned'),
    part('ranger_projectile_main_trail'),
    part('ranger_projectile_hit'),
    includes='asw_alien',
    )
res('asw_remote_turret',
    mdl('models/swarm/SentryGun/remoteturret.mdl'),
    sound('ASW_Sentry.Fire'),
    sound('ASW_Sentry.Turn'),
    )
res('asw_rope_anchor', mat("materials/cable/cable.vmt"))
res('asw_scanner_noise')
res('asw_snow_volume')  # TODO: Uses an asw_emitter

res('asw_sentry_top',
    mdl('models/sentry_gun/machinegun_top.mdl'),
    mdl('models/sentry_gun/freeze_top.mdl'),
    mdl('models/sentry_gun/grenade_top.mdl'),
    mdl('models/sentry_gun/flame_top.mdl'),
    sound('ASW_Sentry.Fire'),
    sound('ASW_Sentry.Turn'),
    sound('ASW_Sentry.AmmoWarning'),
    sound('ASW_Sentry.OutOfAmmo'),
    sound('ASW_Sentry.Deploy'),
    sound('ASW_Sentry.CannonFire'),
    sound('ASW_Sentry.FlameLoop'),
    sound('ASW_Sentry.FlameStop'),
    sound('ASW_Sentry.IceLoop'),
    sound('ASW_Sentry.IceStop'),
    part('asw_flamethrower'),
    part('asw_freezer_spray'),
)
res('asw_sentry_top_cannon', includes='asw_sentry_top')
res('asw_sentry_top_flamer', includes='asw_sentry_top')
res('asw_sentry_top_icer', includes='asw_sentry_top')
res('asw_sentry_top_machinegun', includes='asw_sentry_top')
res('asw_shieldbug',
    mdl('models/aliens/Shieldbug/Shieldbug.mdl'),
    mdl('models/swarm/Shieldbug/Shieldbug.mdl'),
    mdl('models/aliens/shieldbug/gib_back_leg.mdl'),
    mdl('models/aliens/shieldbug/gib_leg_claw.mdl'),
    mdl('models/aliens/shieldbug/gib_leg_middle.mdl'),
    mdl('models/aliens/shieldbug/gib_leg_upper.mdl'),
    mdl('models/aliens/shieldbug/gib_leg_l.mdl'),
    mdl('models/aliens/shieldbug/gib_leg_r.mdl'),
    sound('ASW_Drone.Alert'),
    sound('ASW_Drone.Attack'),
    sound('ASW_Parasite.Death'),
    sound('ASW_Parasite.Idle'),
    sound('ASW_Parasite.Attack'),
    sound('ASW_ShieldBug.StepLight'),
    sound('ASW_ShieldBug.Pain'),
    sound('ASW_ShieldBug.Alert'),
    sound('ASW_ShieldBug.Death'),
    sound('ASW_ShieldBug.Attack'),
    sound('ASW_ShieldBug.Circle'),
    sound('ASW_ShieldBug.Idle'),
    part('shieldbug_brain_explode'),
    part('shieldbug_fountain'),
    part('shieldbug_body_explode'),
    includes='asw_alien',
)

res('asw_simple_drone',
    mdl('models/swarm/drone/Drone.mdl'),
    mdl('models/aliens/drone/drone.mdl'),
    sound('ASW_Drone.Land'),
    sound('ASW_Drone.Pain'),
    sound('ASW_Drone.Alert'),
    sound('ASW_Drone.Death'),
    sound('ASW_Drone.Attack'),
    sound('ASW_Drone.Swipe'),
    mdl('models/swarm/DroneGibs/dronepart01.mdl'),
    mdl('models/swarm/DroneGibs/dronepart20.mdl'),
    mdl('models/swarm/DroneGibs/dronepart29.mdl'),
    mdl('models/swarm/DroneGibs/dronepart31.mdl'),
    mdl('models/swarm/DroneGibs/dronepart32.mdl'),
    mdl('models/swarm/DroneGibs/dronepart44.mdl'),
    mdl('models/swarm/DroneGibs/dronepart45.mdl'),
    mdl('models/swarm/DroneGibs/dronepart47.mdl'),
    mdl('models/swarm/DroneGibs/dronepart49.mdl'),
    mdl('models/swarm/DroneGibs/dronepart50.mdl'),
    mdl('models/swarm/DroneGibs/dronepart53.mdl'),
    mdl('models/swarm/DroneGibs/dronepart54.mdl'),
    mdl('models/swarm/DroneGibs/dronepart56.mdl'),
    mdl('models/swarm/DroneGibs/dronepart57.mdl'),
    mdl('models/swarm/DroneGibs/dronepart58.mdl'),
    mdl('models/swarm/DroneGibs/dronepart59.mdl'),
    )
res('asw_spawn_group')


SPAWNER_ORDER = [
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
def asw_spawner(pack: PackList, ent: Entity) -> None:
    """The spawner spawns from an indexed list."""
    try:
        classname = SPAWNER_ORDER[int(ent['AlienClass'])]
    except (IndexError, ValueError, TypeError):
        pass
    else:
        pack_ent_class(pack, classname)

res('asw_spawner', *BASE_SPAWNER)
res('asw_stylincam')
res('asw_tech_marine_req')
res('asw_trigger_fall')
res('asw_prop_jeep',
    sound('PropJeep.AmmoClose'),
    sound('PropJeep.FireCannon'),
    sound('PropJeep.FireChargedCannon'),
    sound('PropJeep.AmmoOpen'),
    sound('Jeep.GaussCharge'),
    mat('materials/sprites/laserbeam.vmt'),
    )
res('asw_weapon_blink',
    mdl('models/swarm/Bayonet/bayonet.mdl'),
    sound('ASW_Weapon.BatteryCharged'),
    sound('ASW_Weapon.InsufficientBattery'),
    )
res('asw_weapon_jump_jet', includes='asw_weapon_blink')
