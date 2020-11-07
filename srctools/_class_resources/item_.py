"""item_ entities."""
from srctools._class_resources import *

res('item_ammo_pistol',
    mdl('models/items/boxsrounds.mdl'),
    aliases='item_box_srounds',
    )
res('item_ammo_pistol_large',
    mdl('models/items/boxsrounds.mdl'),
    aliases='item_large_box_srounds',
    )

res('item_ammo_smg1',
    mdl('models/items/boxmrounds.mdl'),
    aliases='item_box_mrounds',
    )
res('item_ammo_smg1_large',
    mdl('models/items/boxmrounds.mdl'),
    aliases='item_large_box_mrounds',
    )

res('item_ammo_ar2',
    mdl('models/items/combine_rifle_cartridge01.mdl'),
    aliases='item_box_lrounds',
    )
res('item_ammo_ar2_large',
    mdl('models/items/combine_rifle_cartridge01.mdl'),
    aliases='item_large_box_lrounds',
    )

res('item_ammo_357', mdl('models/items/357ammo.mdl'))
res('item_ammo_ar2_large', mdl('models/items/357ammobox.mdl'))

res('item_ammo_crossbow', mdl('models/items/crossbowrounds.mdl'))

res('item_flare_round', mdl('models/items/flare.mdl'))
res('item_box_flare_rounds', mdl('models/items/boxflare.mdl'))

res('item_rpg_round',
    mdl('models/weapons/w_missile_closed.mdl'),
    aliases='item_ml_grenade',
    )
res('item_ammo_smg1_grenade',
    mdl('models/items/ar2_grenade.mdl'),  # Has nothing to do with AR2s...
    aliases='item_ar2_grenade',
    )
res('item_box_sniper_rounds', mdl('models/items/boxsniperrounds.mdl'))
res('item_box_buckshot', mdl('models/items/boxbuckshot.mdl'))
res('item_ammo_ar2_altfire',
    part('combineball'),
    mdl('models/items/combine_rifle_ammo01.mdl'),
    )
res('item_battery',
    mdl('models/items/battery.mdl'),
    sound('ItemBattery.Touch'),
    )

# Mapbase adds additional models here.
# The second is the mapbase version, the first is Valve's.
AMMO_BOX_MDLS = [
    ("pistol.mdl", "pistol.mdl"),
    ("smg1.mdl", "smg1.mdl"),
    ("ar2.mdl", "ar2.mdl"),
    ("rockets.mdl", "rockets.mdl"),
    ("buckshot.mdl", "buckshot.mdl"),
    ("grenade.mdl", "grenade.mdl"),
    # Valve reused models for these three.
    ("smg1.mdl", "357.mdl"),
    ("smg1.mdl", "xbow.mdl"),
    ("ar2.mdl",  "ar2alt.mdl"),

    ("smg2.mdl", "smg2.mdl"),
    # Two added by mapbase.
    ("", "slam.mdl"),
    ("", "empty.mdl"),
]


@cls_func
def item_ammo_crate(pack: PackList, ent: Entity) -> None:
    """Handle loading the specific ammo box type."""
    pack.pack_soundscript("AmmoCrate.Open")
    pack.pack_soundscript("AmmoCrate.Close")

    try:
        mdl_valve, mdl_mbase = AMMO_BOX_MDLS[int(ent['AmmoType'])]
    except (IndexError, TypeError, ValueError):
        pass
    else:
        # TODO: Only need to pack Mapbase files if it's being used.
        if mdl_valve:
            pack.pack_file('models/items/ammocrate_' + mdl_valve, FileType.MODEL)
        pack.pack_file('models/items/ammocrate_' + mdl_mbase, FileType.MODEL, optional=True)
        if mdl_valve == 'grenade.mdl':
            pack_ent_class(pack, 'weapon_frag')
        elif mdl_mbase == 'slam.mdl':
            pack_ent_class(pack, 'weapon_slam')


res('item_dynamic_resupply',
    # This just spawns a bunch of different ents.
    includes='''item_healthkit item_battery
    
    item_ammo_pistol		
    item_ammo_smg1			
    item_ammo_smg1_grenade
    item_ammo_ar2			
    item_box_buckshot		
    item_rpg_round			
    weapon_frag			
    item_ammo_357			
    item_ammo_crossbow		
    item_ammo_ar2_altfire
    '''
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
res('item_sodacan',
    mdl("models/can.mdl"),
    sound("ItemSoda.Bounce")
    )
res('item_suit', mdl('models/items/hevsuit.mdl'))
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
