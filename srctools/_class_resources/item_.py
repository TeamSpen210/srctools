"""item_ entities."""
from srctools._class_resources import *

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
