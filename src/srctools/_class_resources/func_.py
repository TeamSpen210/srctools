"""func_ entities."""
from . import *


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

    mat_type = conv_int(ent['material'])
    pack.pack_breakable_chunk(MATERIAL_GIB_TYPES.get(mat_type, 'WoodChunks'))
    try:
        breakable_class = BREAKABLE_SPAWNS[conv_int(ent['spawnobject'])]
    except (IndexError, TypeError, ValueError):
        breakable_class = ''
    if breakable_class:
        pack_ent_class(pack, breakable_class)
res('func_breakable',
    sound("Breakable.MatGlass"),
    sound("Breakable.MatWood"),
    sound("Breakable.MatMetal"),
    sound("Breakable.MatFlesh"),
    sound("Breakable.MatConcrete"),
    sound("Breakable.Computer"),
    sound("Breakable.Crate"),
    sound("Breakable.Glass"),
    sound("Breakable.Metal"),
    sound("Breakable.Flesh"),
    sound("Breakable.Concrete"),
    sound("Breakable.Ceiling"),
    )


@cls_func
def func_breakable_surf(pack: PackList, ent: Entity):
    """Additional materials required for func_breakable_surf."""
    # First pack the base func_breakable stuff.
    func_breakable(pack, ent)

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
res('func_breakable_surf',
    mdl('models/brokenglass_piece.mdl'),
    includes='func_breakable',
    )

res('func_brush')

@cls_func
def func_button(pack: PackList, ent: Entity) -> None:
    """Pack the legacy sound indexes."""
    pack_button_sound(pack, ent['sounds'])
    pack_button_sound(pack, ent['locked_sound'])
    pack_button_sound(pack, ent['unlocked_sound'])
    # TODO locked and unlocked sentences in HL1.
    # locked_sentence -> ["NA", "ND", "NF", "NFIRE", "NCHEM", "NRAD", "NCON", "NH", "NG"]
    # unlocked_sentence -> ["EA", "ED", "EF", "EFIRE", "ECHEM", "ERAD", "ECON", "EH"]


res('func_conveyor')
res('func_clip_vphysics')


res('func_door',
    # Defaults if unspecified.
    sound('DoorSound.DefaultMove'),
    sound('DoorSound.DefaultArrive'),
    sound('DoorSound.DefaultLocked'),
    sound('DoorSound.Null'),
    # Todo: also locked and unlocked sentences in HL1.
    )
res('func_door_rotating',
    # Defaults if unspecified.
    sound('RotDoorSound.DefaultMove'),
    sound('RotDoorSound.DefaultArrive'),
    sound('RotDoorSound.DefaultLocked'),
    sound('DoorSound.Null'),
    )

res('func_dust', mat('materials/particle/sparkles.vmt'))
res('func_dustcloud', mat('materials/particle/sparkles.vmt'))
res('func_dustmotes', mat('materials/particle/sparkles.vmt'))
res('func_combine_ball_spawner', includes='prop_combine_ball')
res('func_fish_pool')
res('func_healthcharger',
    sound("WallHealth.Deny"),
    sound("WallHealth.Start"),
    sound("WallHealth.LoopingContinueCharge"),
    sound("WallHealth.Recharge"),
    )
res('func_illusionary')
res('func_instance_io_proxy')


@cls_func
def momentary_rot_button(pack: PackList, ent: Entity) -> None:
    """Inherits from func_button, but doesn't always use 'sounds'."""
    if conv_int(ent['spawnflags']) & 1024:  # USE_ACTIVATES
        pack_button_sound(pack, ent['sounds'])
    pack_button_sound(pack, ent['locked_sound'])
    pack_button_sound(pack, ent['unlocked_sound'])

res('func_movelinear', aliases='momentary_door')
res('func_noportal_volume')
res('func_null')
res('func_portal_bumper')
res('func_portal_detector')
res('func_portal_orientation')
res('func_portalled')


@cls_func
def func_pushable(pack: PackList, ent: Entity) -> None:
    """Subclass of func_breakable."""
    func_breakable(pack, ent)


res('func_rotating', sound('DoorSound.Null'))


# Subclass of that.
@cls_func
def func_physbox(pack: PackList, ent: Entity) -> None:
    """Subclass of func_breakable."""
    func_breakable(pack, ent)

res('func_precipitation',
    mat("materials/effects/fleck_ash1.vmt"),
    mat("materials/effects/fleck_ash2.vmt"),
    mat("materials/effects/fleck_ash3.vmt"),
    mat("materials/effects/ember_swirling001.vmt"),
    mat("materials/particle/rain.vmt"),
    mat("materials/particle/snow.vmt"),
    part("rain_storm"),
    part("rain_storm_screen"),
    part("rain_storm_outer"),
    part("rain"),
    part("rain_outer"),
    part("ash"),
    part("ash_outer"),
    part("snow"),
    part("snow_outer"),
    )


res('func_tank',
    sound('Func_Tank.BeginUse'),
    # Only if set to cannon, but that doesn't really matter too much.
    sound('NPC_Combine_Cannon.FireBullet'),
    )
res('func_tankpulselaser', includes='func_tank grenade_beam')
res('func_tanklaser', includes='func_tank')
res('func_tankrocket', includes='func_tank rpg_missile')
res('func_tankairboatgun',
    sound('Airboat.FireGunLoop'),
    sound('Airboat.FireGunRevDown'),
    includes='func_tank',
    )
res('func_tankapcrocket',
    sound('PropAPC.FireCannon'),
    includes='func_tank apc_missile',
    )
res('func_tankmortar', includes='func_tank mortarshell')
res('func_tankphyscannister', includes='func_tank')
res('func_tank_combine_cannon',
    mat('materials/effects/blueblacklargebeam.vmt'),
    part('Weapon_Combine_Ion_Cannon'),
    includes='func_tank',
    )

res('func_tankchange', sound('FuncTrackChange.Blocking'))
res('func_recharge',
    sound('SuitRecharge.Deny'),
    sound('SuitRecharge.Start'),
    sound('SuitRecharge.ChargingLoop'),
    )


@cls_func
def func_rot_button(pack: PackList, ent: Entity) -> None:
    """Inherits from func_button."""
    func_button(pack, ent)

res('func_water', includes='func_door')  # Same class.
res('func_water_analog', includes='func_movelinear')  # Also same class.
res('func_weight_button')
