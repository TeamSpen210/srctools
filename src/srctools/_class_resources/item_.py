"""item_ entities."""
from ..packlist import FileType, PackList
from ..vmf import Entity
from . import *

res('item_flare_round', mdl('models/items/flare.mdl'))
res('item_box_flare_rounds', mdl('models/items/boxflare.mdl'))

res('item_rpg_round', aliases='item_ml_grenade')
res('item_box_sniper_rounds', mdl('models/items/boxsniperrounds.mdl'))

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

res('item_creature_crate', part('creaturecrate_stasisbreak'))  # EZ2


@cls_func
def item_item_crate(pack: PackList, ent: Entity) -> None:
    """Item crates can spawn another arbitary entity."""
    appearance = conv_int(ent['crateappearance'])
    if appearance == 0:  # Default
        pack.pack_file('models/items/item_item_crate.mdl', FileType.MODEL)
    elif appearance == 1:  # Beacon
        pack.pack_file('models/items/item_beacon_crate.mdl', FileType.MODEL)
    # else: 2 = Mapbase custom model, that'll be packed automatically.
    if conv_int(ent['cratetype']) == 0 and ent['itemclass']:  # "Specific Item"
        try:
            if 'ezvariant' in ent:  # Transfer this for accurate packing.
                pack_ent_class(pack, ent['itemclass'], ezvariant=ent['ezvariant'])
            else:
                pack_ent_class(pack, ent['itemclass'])
        except KeyError:  # Invalid classname.
            pass


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
