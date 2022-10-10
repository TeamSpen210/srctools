"""asw_ entities."""
from . import *


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

res('asw_mission_objective')

res('asw_pickup_ammo_satchel', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_buff_grenade', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_heal_grenade', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_hornet_barrage', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_medical_satchel', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_railgun', mdl('models/weapons/Railgun/Railgun.mdl'))

res('asw_pickup_sentry_cannon', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_sentry_flamer', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))
res('asw_pickup_sentry_freeze', mdl('models/items/ItemBox/ItemBoxLarge.mdl'))

res('asw_pickup_t75', mdl('models/items/Mine/mine.mdl'))
res('asw_pickup_tesla_trap', mdl('models/items/Mine/mine.mdl'))

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
