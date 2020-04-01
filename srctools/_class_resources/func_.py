"""func_ entities."""
from srctools._class_resources import *

MATERIAL_GIB_TYPES = {
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
BREAKABLE_SPAWNS = [
    "",  # NULL
    "item_battery",
    "item_healthkit",
    "item_ammo_pistol",
    "item_ammo_pistol_large",
    "item_ammo_smg1",
    "item_ammo_smg1_large",
    "item_ammo_ar2",
    "item_ammo_ar2_large",
    "item_box_buckshot",
    "item_flare_round",
    "item_box_flare_rounds",
    "item_rpg_round",
    "",
    "item_box_sniper_rounds",
    "",
    "weapon_stunstick",
    "",
    "weapon_ar2",
    "",
    "weapon_rpg",
    "weapon_smg1",
    "",  # weapon_smg2
    "",  #weapon_slam
    "weapon_shotgun",
    "",  # weapon_molotov
    "item_dynamic_resupply",
]

res('func_areaportal')
res('func_areaportalwindow')
res('func_areaportal_oneway')


@cls_func
def func_breakable(pack: PackList, ent: Entity) -> None:
    """Packs a number of specific gibs/resources."""
    pack.pack_soundscript("Breakable.MatGlass")
    pack.pack_soundscript("Breakable.MatWood")
    pack.pack_soundscript("Breakable.MatMetal")
    pack.pack_soundscript("Breakable.MatFlesh")
    pack.pack_soundscript("Breakable.MatConcrete")
    pack.pack_soundscript("Breakable.Computer")
    pack.pack_soundscript("Breakable.Crate")
    pack.pack_soundscript("Breakable.Glass")
    pack.pack_soundscript("Breakable.Metal")
    pack.pack_soundscript("Breakable.Flesh")
    pack.pack_soundscript("Breakable.Concrete")
    pack.pack_soundscript("Breakable.Ceiling")

    mat_type = conv_int(ent['material'])
    pack.pack_breakable_chunk(MATERIAL_GIB_TYPES.get(mat_type, 'WoodChunks'))
    try:
        breakable_class = BREAKABLE_SPAWNS[conv_int(ent['spawnobject'])]
    except (IndexError, TypeError, ValueError):
        breakable_class = ''
    if breakable_class:
        pack_ent_class(pack, breakable_class)


@cls_func
def func_breakable_surf(pack: PackList, ent: Entity):
    """Additional materials required for func_breakable_surf."""
    # First pack the base func_breakable stuff.
    func_breakable(pack, ent)

    pack.pack_file('models/brokenglass_piece.mdl', FileType.MODEL)

    surf_type = conv_int(ent['surfacetype'])

    if surf_type == 1:  # Tile
        mat_type = 'tile'
    elif surf_type == 0:  # Glass
        mat_type = 'glass'
        pack.pack_file('materials/models/brokenglass/glassbroken_solid.vmt', FileType.MATERIAL)
    else:
        # Unknown
        return

    for num in '123':
        for letter in 'abcd':
            pack.pack_file(
                'materials/models/broken{0}/'
                '{0}broken_0{1}{2}.vmt'.format(mat_type, num, letter),
                FileType.MATERIAL,
            )

res('func_conveyor')
res('func_clip_vphysics')
res('func_dust', mat('materials/particle/sparkles.vmt'))
res('func_dustcloud', mat('materials/particle/sparkles.vmt'))
res('func_dustmotes', mat('materials/particle/sparkles.vmt'))
res('func_fish_pool')
res('func_healthcharger',
    sound("WallHealth.Deny"),
    sound("WallHealth.Start"),
    sound("WallHealth.LoopingContinueCharge"),
    sound("WallHealth.Recharge"),
    )
res('func_illusionary')
res('func_instance_io_proxy')
res('func_movelinear')
res('func_portal_bumper')
res('func_portal_detector')
res('func_portal_orientation')
res('func_portalled')
res('func_rotating', sound('DoorSound.Null'))

# Subclass of that.
@cls_func
def func_physbox(pack: PackList, ent: Entity) -> None:
    func_breakable(pack, ent)


res('func_tankchange', sound('FuncTrackChange.Blocking'))
res('func_recharge',
    sound('SuitRecharge.Deny'),
    sound('SuitRecharge.Start'),
    sound('SuitRecharge.ChargingLoop'),
    )
res('func_weight_button')
