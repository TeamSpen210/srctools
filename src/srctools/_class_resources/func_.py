"""func_ entities."""
from typing import Mapping, Sequence

from . import *


# Index->sound lists for CBasePlatTrain in HL:Source.
HL1_PLAT_MOVE: Final[Sequence[str]] = [
    'Plat.DefaultMoving', 'Plat.BigElev1', 'Plat.BigElev2', 'Plat.TechElev1', 'Plat.TechElev2',
    'Plat.TechElev3', 'Plat.FreightElev1', 'Plat.FreightElev2', 'Plat.HeavyElev', 'Plat.RackElev',
    'Plat.RailElev', 'Plat.SqueakElev', 'Plat.OddElev1', 'Plat.OddElev2',
]
HL1_PLAT_STOP: Final[Sequence[str]] = [
    "Plat.DefaultArrive", "Plat.BigElevStop1", "Plat.BigElevStop2", "Plat.FreightElevStop",
    "Plat.HeavyElevStop", "Plat.RackStop", "Plat.RailStop", "Plat.SqueakStop", "Plat.QuickStop",
]


MATERIAL_GIB_TYPES: Final[Mapping[int, str]] = {
    0: 'GlassChunks',
    1: 'WoodChunks',
    2: 'MetalChunks',
    # 3 = Flesh
    4: 'ConcreteChunks',  # Cinderblock
    # 5 = Ceiling Tile
    # 6 = Computer
    7: 'GlassChunks',  # Unbreakable Glass
    8: 'ConcreteChunks',  # Rocks
    # 9 = Web
}

# Classnames spawned by func_breakable.
BREAKABLE_SPAWNS: Mapping[int, str] = {

    1: "item_battery",
    2: "item_healthkit",
    3: "item_ammo_pistol",
    4: "item_ammo_pistol_large",
    5: "item_ammo_smg1",
    6: "item_ammo_smg1_large",
    7: "item_ammo_ar2",
    8: "item_ammo_ar2_large",
    9: "item_box_buckshot",
    10: "item_flare_round",
    11: "item_box_flare_rounds",
    12: "item_ml_grenade",
    13: "item_smg1_grenade",
    14: "item_box_sniper_rounds",
    # 15: "unused1",
    16: "weapon_stunstick",
    # 17: "weapon_ar1",
    18: "weapon_ar2",
    # 19: "unused2",
    20: "weapon_ml",
    21: "weapon_smg1",
    22: "weapon_smg2",
    23: "weapon_slam",
    24: "weapon_shotgun",
    # 25: "weapon_molotov",
    26: "item_dynamic_resupply",

    # Black Mesa:
    27: "item_ammo_glock",
    28: "item_ammo_mp5",
    29: "item_ammo_357",
    30: "item_ammo_crossbow",
    31: "item_ammo_shotgun",
    32: "item_ammo_energy",
    33: "item_grenade_mp5",
    34: "item_grenade_rpg",
    35: "item_weapon_357",
    36: "item_weapon_crossbow",
    37: "item_weapon_crowbar",
    38: "item_weapon_frag",
    39: "item_weapon_glock",
    40: "item_weapon_gluon",
    41: "item_weapon_hivehand",
    42: "item_weapon_mp5",
    43: "item_weapon_rpg",
    44: "item_weapon_satchel",
    45: "item_weapon_shotgun",
    46: "item_weapon_snark",
    47: "item_weapon_tau",
    48: "item_weapon_tripmine",
    49: "item_syringe",
    50: "item_ammo_box",
    51: "prop_soda",
}
# TODO: Mapbase, EZ2 etc additions?


def base_plat_train(pack: PackList, ent: Entity) -> None:
    """Check for HL1 train movement sounds."""
    if 'movesnd' in ent:
        try:
            sound = HL1_PLAT_MOVE[int(ent['movesnd'])]
        except (IndexError, TypeError, ValueError):
            pass
        else:
            pack.pack_soundscript(sound)
    if 'stopsnd' in ent:
        try:
            sound = HL1_PLAT_STOP[int(ent['stopsnd'])]
        except (IndexError, TypeError, ValueError):
            pass
        else:
            pack.pack_soundscript(sound)


def breakable_brush(pack: PackList, ent: Entity) -> None:
    """Breakable brushes are able to spawn specific entities."""

    mat_type = conv_int(ent['material'])
    pack.pack_breakable_chunk(MATERIAL_GIB_TYPES.get(mat_type, 'WoodChunks'))
    try:
        breakable_class = BREAKABLE_SPAWNS[conv_int(ent['spawnobject'])]
    except (IndexError, TypeError, ValueError):
        pass
    else:
        pack_ent_class(pack, breakable_class)


@cls_func
def func_breakable_surf(pack: PackList, ent: Entity) -> None:
    """Additional materials required for func_breakable_surf."""
    surf_type = conv_int(ent['surfacetype'])

    if surf_type == 1:  # Tile
        mat_type = 'tile'
    elif surf_type == 0:  # Glass
        mat_type = 'glass'
        pack.pack_file('materials/models/brokenglass/glassbroken_solid.vmt', FileType.MATERIAL)
    else:
        # Unknown
        return

    pack.pack_file(f'materials/effects/fleck_{mat_type}1.vmt')
    pack.pack_file(f'materials/effects/fleck_{mat_type}2.vmt')

    for num in '123':
        for letter in 'abcd':
            pack.pack_file(
                f'materials/models/broken{mat_type}/{mat_type}broken_0{num}{letter}.vmt',
                FileType.MATERIAL,
            )



res('func_plat',
    sound('Plat.DefaultMoving'),
    sound('Plat.DefaultArrive'),
    func=base_plat_train,
    )
