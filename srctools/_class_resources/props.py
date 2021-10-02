"""Prop entities."""
from srctools import Property, KeyValError, NoKeyError
from srctools._class_resources import *


res('prop_floor_button')  # Model set in keyvalues.
res('prop_floor_ball_button', mdl('models/props/ball_button.mdl'))
res('prop_floor_cube_button', mdl('models/props/box_socket.mdl'))
res('prop_combine_ball',
    mdl('models/effects/combineball.mdl'),
    mat('sprites/combineball_trail_black_1.vmt'),
    mat("materials/sprites/lgtning.vmt"),
    sound("NPC_CombineBall.Launch"),
    sound("NPC_CombineBall.KillImpact"),
    sound("NPC_CombineBall.HoldingInPhysCannon"),
    # TODO: Episodic version, and HL2-only version.
    sound("NPC_CombineBall_Episodic.Explosion"),
    sound("NPC_CombineBall_Episodic.WhizFlyby"),
    sound("NPC_CombineBall_Episodic.Impact"),
    # TODO: HL2-only:
    sound("NPC_CombineBall.Explosion"),
    sound("NPC_CombineBall.WhizFlyby"),
    sound("NPC_CombineBall.Impact"),
    )
res('prop_coreball', mdl('models/props_combine/coreball.mdl'))
res('prop_button',
    mdl('models/props/switch001.mdl'),
    sound('Portal.button_down'),
    sound('Portal.button_up'),
    sound('Portal.button_locked'),
    sound('Portal.room1_TickTock'),
    )

res('prop_dynamic',
    sound("Metal.SawbladeStick"),
    sound("PropaneTank.Burst"),
    aliases='dynamic_prop',
    )
# The same class, but don't let these swap to each other!
res('prop_dynamic_override', includes='prop_dynamic')
res('prop_dropship_container',
    mdl('models/combine_dropship_container.mdl'),
    mdl('models/gibs/helicopter_brokenpiece_01.mdl'),
    mdl('models/gibs/helicopter_brokenpiece_02.mdl'),
    mdl('models/gibs/helicopter_brokenpiece_03.mdl'),
    mdl('models/gibs/hgibs.mdl'),
    )
res('prop_energy_ball',
    mat('materials/effects/eball_finite_life.vmt'),
    mat('materials/effects/eball_infinite_life.vmt'),
    mat('decals/smscorch1model.vmt'),
    mat('decals/smscorch1_subrect.vmt'),
    sound('EnergyBall.Explosion'),
    sound('EnergyBall.Launch'),
    sound('EnergyBall.KillImpact'),
    sound('EnergyBall.Impact'),
    sound('EnergyBall.AmbientLoop'),
    includes='prop_combine_ball',
    )

res('prop_exploding_barrel',
    mdl('models/props/coop_cementplant/exloding_barrel/exploding_barrel.mdl'),
    mdl('models/props/coop_cementplant/exloding_barrel/exploding_barrel_top.mdl'),
    mdl('models/props/coop_cementplant/exloding_barrel/exploding_barrel_bottom.mdl'),
    sound('BaseGrenade.Explode'),
    sound('Inferno.Start_IncGrenade'),
    part('explosion_hegrenade_interior'),
    part('dust_burning_engine'),
    includes='prop_physics',
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
res('prop_physics', aliases='physics_prop')
# The same class, but don't let these swap to each other!
res('prop_physics_override', includes='prop_physics')

res('prop_testchamber_door',
    mdl('models/props/portal_door_combined.mdl'),
    sound('prop_portal_door.open'),
    sound('prop_portal_door.close'),
)

res('prop_thumper',
    mdl("models/props_combine/CombineThumper002.mdl"),
    sound("coast.thumper_hit"),
    sound("coast.thumper_ambient"),
    sound("coast.thumper_dust"),
    sound("coast.thumper_startup"),
    sound("coast.thumper_shutdown"),
    sound("coast.thumper_large_hit"),
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
    # TODO: sound('music.map_name_here_lbout')
    sound('music.sp_all_maps_lbout'),
    part('projected_wall_impact'),
    )
res('prop_ragdoll', aliases='physics_prop_ragdoll')
res('prop_scalable')
