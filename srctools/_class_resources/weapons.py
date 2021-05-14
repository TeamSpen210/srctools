"""All the weapon pickups.

Weapons are unusual, they don't directly specify the models.
Instead it's specified in the weapon script.
"""
from srctools._class_resources import *

res('weapon_357')
# res('weapon_adrenaline_spawn')
# res('weapon_ak47')
res('weapon_alyxgun')
# res('weapon_ammo_spawn')
res('weapon_annabelle')
res('weapon_ar2',
    includes='prop_combine_ball env_entity_dissolver',
    )
# res('weapon_aug')
# res('weapon_autoshotgun_spawn')
# res('weapon_awp')
# res('weapon_bizon')
res('weapon_bugbait',
    sound("Weapon_Bugbait.Splat"),
    includes="npc_grenade_bugbait",
    )
# res('weapon_c4')
# res('weapon_chainsaw_spawn')
res('weapon_citizenpackage')
res('weapon_citizensuitcase')
res('weapon_crossbow',
    mat('sprites/blueflare1.vmt'),
    mat('sprites/light_glow02_noz.vmt'),
    sound('Weapon_Crossbow.BoltHitBody'),
    sound('Weapon_Crossbow.BoltHitWorld'),
    sound('Weapon_Crossbow.BoltSkewer'),
    includes='crossbow_bolt',
    )
res('weapon_crowbar')
res('weapon_cubemap')
# res('weapon_cz75a')
# res('weapon_deagle')
# res('weapon_decoy')
# res('weapon_defibrillator_spawn')
# res('weapon_elite')
# res('weapon_famas')
# res('weapon_first_aid_kit')
# res('weapon_first_aid_kit_spawn')
# res('weapon_fiveseven')
# res('weapon_flashbang')
res('weapon_frag',
    sound('WeaponFrag.Throw'),
    sound('WeaponFrag.Roll'),
    includes='npc_grenade_frag',
    )
# res('weapon_g3sg1')
# res('weapon_galilar')
# res('weapon_gascan_spawn')
# res('weapon_glock')
# res('weapon_grenade_launcher')
# res('weapon_grenade_launcher_spawn')
# res('weapon_healthshot')
# res('weapon_hegrenade')
# res('weapon_hkp2000')
# res('weapon_hunting_rifle_spawn')
# res('weapon_incgrenade')
# res('weapon_item_spawn')
# res('weapon_knife')
# res('weapon_m249')
# res('weapon_m4a1')
# res('weapon_m4a1_silencer')
# res('weapon_mac10')
# res('weapon_mag7')
# res('weapon_melee_spawn')
# res('weapon_molotov')
# res('weapon_molotov_spawn')
# res('weapon_mp7')
# res('weapon_mp9')
# res('weapon_negev')
# res('weapon_nova')
# res('weapon_p250')
# res('weapon_p90')
# res('weapon_pain_pills_spawn')
res('weapon_physcannon',
    mat("materials/sprites/orangelight1.vmt"),
    mat("materials/sprites/glow04_noz.vmt"),
    mat("materials/sprites/orangeflare1.vmt"),
    mat("materials/sprites/orangecore1.vmt"),
    mat("materials/sprites/orangecore2.vmt"),

    mat("materials/sprites/lgtning_noz.vmt"),
    mat("materials/sprites/blueflare1_noz.vmt"),
    mat("materials/effects/fluttercore.vmt"),
    mdl("models/weapons/v_superphyscannon.mdl"),

    sound("Weapon_PhysCannon.HoldSound"),
    sound("Weapon_Physgun.Off"),
    sound("Weapon_MegaPhysCannon.DryFire"),
    sound("Weapon_MegaPhysCannon.Launch"),
    sound("Weapon_MegaPhysCannon.Pickup"),
    sound("Weapon_MegaPhysCannon.Drop"),
    sound("Weapon_MegaPhysCannon.HoldSound"),
    sound("Weapon_MegaPhysCannon.ChargeZap"),
    )
# res('weapon_pipe_bomb_spawn')
res('weapon_pistol')
# res('weapon_pistol_magnum_spawn')
# res('weapon_pistol_spawn')
# res('weapon_portalgun')
# res('weapon_pumpshotgun_spawn')
# res('weapon_revolver')
# res('weapon_rifle_ak47_spawn')
# res('weapon_rifle_desert_spawn')
# res('weapon_rifle_m60_spawn')
# res('weapon_rifle_spawn')
res('weapon_rpg',
    sound("Missile.Ignite"),
    sound("Missile.Accelerate"),
    mat("materials/effects/laser1_noz.vmt"),
    mat("materials/sprites/redglow1.vmt"),
    includes="rpg_missile",
    )
# res('weapon_sawedoff')
# res('weapon_scar20')
# res('weapon_scavenge_item_spawn')
# res('weapon_sg556')
res('weapon_shotgun')
# res('weapon_shotgun_chrome_spawn')
# res('weapon_shotgun_spas_spawn')
res('weapon_smg1', includes="grenade_ar2")
# res('weapon_smg_silenced_spawn')
# res('weapon_smg_spawn')
# res('weapon_smokegrenade')
# res('weapon_sniper_military_spawn')
# res('weapon_spawn')
# res('weapon_ssg08')
res('weapon_striderbuster',
    mdl("models/magnusson_device.mdl"),
    sound("Weapon_StriderBuster.StickToEntity"),
    sound("Weapon_StriderBuster.Detonate"),
    sound("Weapon_StriderBuster.Dud_Detonate"),
    sound("Weapon_StriderBuster.Ping"),

    mat("materials/sprites/orangeflare1.vmt"),
    mat("materials/sprites/lgtning.vmt"),
    part("striderbuster_attach"),
    part("striderbuster_attached_pulse"),
    part("striderbuster_explode_core"),
    part("striderbuster_explode_dummy_core"),
    part("striderbuster_break_flechette"),
    part("striderbuster_trail"),
    part("striderbuster_shotdown_trail"),
    part("striderbuster_break"),
    part("striderbuster_flechette_attached"),

    includes="env_citadel_energy_core sparktrail",
    aliases="prop_stickybomb",
    )
res('weapon_stunstick',
    sound("Weapon_StunStick.Activate"),
    sound("Weapon_StunStick.Deactivate"),
    )
# res('weapon_tagrenade')
# res('weapon_taser')
# res('weapon_tec9')
# res('weapon_ump45')
