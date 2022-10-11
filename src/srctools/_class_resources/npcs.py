"""NPC entities."""
from . import *
from . import _blank_vmf


@cls_func
def base_npc(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Resources precached in CAI_BaseNPC."""
    if conv_int(ent['ezvariant']) == EZ_VARIANT_TEMPORAL:
        yield Resource('NPC_TemporalHeadcrab.Vanish', FileType.GAME_SOUND)
        yield Resource('NPC_TemporalHeadcrab.Appear', FileType.GAME_SOUND)
        yield Resource('ShadowCrab_Vanish', FileType.PARTICLE_SYSTEM)
        yield Resource('ShadowCrab_Appear', FileType.PARTICLE_SYSTEM)
    equipment = ent['additionalequipment']
    if equipment not in ('', '0'):
        yield _blank_vmf.create_ent(equipment)


@cls_func
def npc_antlion(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Antlions require different resources for the worker version."""
    ez_variant = conv_int(ent['ezvariant'])
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 18):  # Is worker?
        if ez_variant == EZ_VARIANT_BLOOD:
            yield Resource("models/bloodlion_worker.mdl", FileType.MODEL)
        else:
            yield Resource("models/antlion_worker.mdl", FileType.MODEL)
        yield Resource("blood_impact_antlion_worker_01", FileType.PARTICLE_SYSTEM)
        yield Resource("antlion_gib_02", FileType.PARTICLE_SYSTEM)
        yield Resource("blood_impact_yellow_01", FileType.PARTICLE_SYSTEM)

        yield _blank_vmf.create_ent('grenade_spit')
    else:  # Regular antlion.
        if ez_variant == EZ_VARIANT_RAD:
            yield Resource("models/antlion_blue.mdl", FileType.MODEL)
            yield Resource("blood_impact_blue_01", FileType.PARTICLE_SYSTEM)
        elif ez_variant == EZ_VARIANT_XEN:
            yield Resource("models/antlion_xen.mdl", FileType.MODEL)
            yield Resource("blood_impact_antlion_01", FileType.PARTICLE_SYSTEM)
        elif ez_variant == EZ_VARIANT_BLOOD:
            yield Resource("models/bloodlion.mdl", FileType.MODEL)
            yield Resource("blood_impact_antlion_01", FileType.PARTICLE_SYSTEM)
        else:
            yield Resource("models/antlion.mdl", FileType.MODEL)
            yield Resource("blood_impact_antlion_01", FileType.PARTICLE_SYSTEM)
        yield Resource("AntlionGib", FileType.PARTICLE_SYSTEM)


@cls_func
def npc_antlionguard(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """In Entropy Zero, some alternate models are available."""
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 17):  # Inside Footsteps
        yield Resource("NPC_AntlionGuard.Inside.StepLight", FileType.GAME_SOUND)
        yield Resource("NPC_AntlionGuard.Inside.StepHeavy", FileType.GAME_SOUND)
    else:
        yield Resource("NPC_AntlionGuard.StepLight", FileType.GAME_SOUND)
        yield Resource("NPC_AntlionGuard.StepHeavy", FileType.GAME_SOUND)
    if 'ezvariant' in ent:  # Entropy Zero.
        variant = conv_int(ent['ezvaraiant'])
        if variant == EZ_VARIANT_XEN:
            yield Resource("models/antlion_guard_xen.mdl", FileType.MODEL)
            yield Resource("xenpc_spawn", FileType.PARTICLE_SYSTEM)
        elif variant == EZ_VARIANT_RAD:
            yield Resource("models/antlion_guard_blue.mdl", FileType.MODEL)
            yield Resource("blood_impact_blue_01", FileType.PARTICLE_SYSTEM)
        elif variant == EZ_VARIANT_BLOOD:
            yield Resource("models/bloodlion_guard.mdl", FileType.MODEL)
        else:
            yield Resource("models/antlion_guard.mdl", FileType.MODEL)
    else:  # Regular HL2.
        yield Resource("models/antlion_guard.mdl", FileType.MODEL)


@cls_func
def npc_antlion_template_maker(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Depending on KVs this may or may not spawn workers."""
    # There will be an antlion present in the map, as the template
    # NPC. So we don't need to add those resources.
    if conv_int(ent['workerspawnrate']) > 0:
        # It randomly spawns worker antlions, so load that resource set.
        yield Resource("models/antlion_worker.mdl", FileType.MODEL)
        yield Resource("blood_impact_antlion_worker_01", FileType.PARTICLE)
        yield Resource("antlion_gib_02", FileType.PARTICLE)
        yield Resource("blood_impact_yellow_01", FileType.PARTICLE)

        yield _blank_vmf.create_ent('grenade_spit')
    if conv_bool(ent['createspores']):
        yield _blank_vmf.create_ent('env_sporeexplosion')


@cls_func
def npc_arbeit_turret_floor(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Arbeit/Aperture turrets have EZ variants."""
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        yield Resource('models/props/glowturret_01.mdl', FileType.MODEL)
    elif variant == EZ_VARIANT_ARBEIT:
        yield Resource('models/props/camoturret_01.mdl', FileType.MODEL)
        yield Resource('models/props/camoturret_02.mdl', FileType.MODEL)
    elif conv_int(ent['spawnflags']) & 0x200:  # Citizen Modified
        yield Resource('models/props/hackedturret_01.mdl', FileType.MODEL)
    else:
        yield Resource('models/props/turret_01.mdl', FileType.MODEL)


@cls_func
def npc_bullsquid(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This has various EZ variants."""
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_XEN:
        yield Resource('models/bullsquid_xen.mdl', FileType.MODEL)
        yield Resource('models/babysquid_xen.mdl', FileType.MODEL)
        yield Resource('models/bullsquid_egg_xen.mdl', FileType.MODEL)
        yield Resource('blood_impact_yellow_01', FileType.PARTICLE_SYSTEM)
    elif variant == EZ_VARIANT_RAD:
        yield Resource('models/bullsquid_rad.mdl', FileType.MODEL)
        yield Resource('models/babysquid_rad.mdl', FileType.MODEL)
        yield Resource('models/bullsquid_egg_rad.mdl', FileType.MODEL)
        yield Resource('blood_impact_blue_01', FileType.PARTICLE_SYSTEM)
    else:
        yield Resource('models/bullsquid.mdl', FileType.MODEL)
        yield Resource('models/babysquid.mdl', FileType.MODEL)
        yield Resource('models/bullsquid_egg.mdl', FileType.MODEL)
        yield Resource('blood_impact_yellow_01', FileType.PARTICLE_SYSTEM)


@cls_func
def combine_scanner(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Detect the kind of scanner (city or shield/claw), then pick the right resources."""
    if ent['classname'] == 'npc_clawscanner':  # Added in episodic, always the shield scanner.
        is_shield = True
    else:  # It checks the map name directly to determine this.
        is_shield = ctx.mapname.lower().startswith('d3_c17')
    if is_shield:
        yield Resource("models/shield_scanner.mdl", FileType.MODEL)
        for i in range(1, 7):
            yield Resource(f"models/gibs/Shield_Scanner_Gib{i}.mdl", FileType.MODEL)
        snd_prefix = 'NPC_SScanner.'
    else:
        yield Resource("models/combine_scanner.mdl", FileType.MODEL)
        for i in range(1, 6):
            yield Resource(f"models/gibs/scanner_gib{i:02}.mdl", FileType.MODEL)
        snd_prefix = 'NPC_CScanner.'

    for snd_name in [
        "Shoot", "Alert", "Die", "Combat", "Idle", "Pain", "TakePhoto", "AttackFlash",
        "DiveBombFlyby", "DiveBomb", "DeployMine", "FlyLoop",
    ]:
        yield Resource(snd_prefix + snd_name, FileType.GAME_SOUND)


CIT_HEADS = [
    "male_01.mdl",
    "male_02.mdl",
    "female_01.mdl",
    "male_03.mdl",
    "female_02.mdl",
    "male_04.mdl",
    "female_03.mdl",
    "male_05.mdl",
    "female_04.mdl",
    "male_06.mdl",
    "female_06.mdl",
    "male_07.mdl",
    "female_07.mdl",
    "male_08.mdl",
    "male_09.mdl",
]


@cls_func
def npc_citizen(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Cizizens have a complex set of precaching rules."""
    if ent['targetname'] == 'matt':
        # Special crowbar.
        yield Resource("models/props_canal/mattpipe.mdl", FileType.MODEL)

    cit_type = conv_int(ent['citizentype'])

    if cit_type == 0:  # Default
        # TODO: Pick via mapname:
        # { "trainstation",	CT_DOWNTRODDEN	},
        # { "canals",		CT_REFUGEE		},
        # { "town",			CT_REFUGEE		},
        # { "coast",		CT_REFUGEE		},
        # { "prison",		CT_DOWNTRODDEN	},
        # { "c17",			CT_REBEL		},
        # { "citadel",		CT_DOWNTRODDEN	},
        for head in CIT_HEADS:
            yield Resource('models/humans/group01/' + head, FileType.MODEL)
            yield Resource('models/humans/group02/' + head, FileType.MODEL)
            yield Resource('models/humans/group03/' + head, FileType.MODEL)
            yield Resource('models/humans/group03m/' + head, FileType.MODEL)
        return
    elif cit_type == 1:  # Downtrodden
        folder = 'group01'
    elif cit_type == 2:  # Refugee
        folder = 'group02'
    elif cit_type == 3:  # Rebel
        folder = 'group03'
        # The rebels have an additional set of models.
        for head in CIT_HEADS:
            yield Resource('models/humans/group03m/' + head, FileType.MODEL)
    elif cit_type == 4:  # Use model in KVs directly.
        return
    else:  # Invalid type?
        # TODO: Entropy Zero variants.
        return

    for head in CIT_HEADS:
        yield Resource(f'models/humans/{folder}/{head}', FileType.MODEL)


@cls_func
def npc_combinedropship(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """The Combine Dropship may spawn with a variety of cargo types."""
    cargo_type = conv_int(ent['cratetype'])
    if cargo_type == -3:  # Spawns a prop_dynamic Jeep
        yield Resource("models/buggy.mdl", FileType.MODEL)
    elif cargo_type == -1:  # Strider
        yield _blank_vmf.create_ent('npc_strider')
    elif cargo_type == 1:  # Soldiers in a container.
        yield _blank_vmf.create_ent('prop_dropship_container')
    # Other valid values:
    # -2 = Grabs the APC specified in KVs - that'll load its own resources.
    #  0 = Roller Hopper, does nothing
    #  2 = No cargo


@cls_func
def npc_combinegunship(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """This has the ability to spawn as the helicopter instead."""
    if conv_int(ent['spawnflags']) & (1 << 13):
        yield Resource("models/combine_helicopter.mdl", FileType.MODEL)
        yield Resource("models/combine_helicopter_broken.mdl", FileType.MODEL)
        yield _blank_vmf.create_ent('helicopter_chunk')
    else:
        yield Resource("models/gunship.mdl", FileType.MODEL)


@cls_func
def npc_egg(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """These are EZ2 bullsquid eggs, which spawn a specific EZ variant."""
    yield _blank_vmf.create_ent('npc_bullsquid', ezvariant=ent['ezvariant'])


@cls_func
def npc_maker(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """We spawn the NPC automatically."""
    # Pass this along, it should then pack that too.
    yield _blank_vmf.create_ent(ent['npctype'], additionalequipment=ent['additionalequipment'])


@cls_func
def npc_metropolice(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """If a spawnflag is set, a cheap model is used."""
    if conv_int(ent['spawnflags']) & 5:
        yield Resource("models/police_cheaple.mdl", FileType.MODEL)
    else:
        yield Resource("models/police.mdl", FileType.MODEL)


@cls_func
def npc_zassassin(ctx: ResourceCtx, ent: Entity) -> ResGen:
    """Entropy Zero 2's "Plan B"/Gonome. """
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        yield Resource('models/glownome.mdl', FileType.MODEL)
        yield Resource('blood_impact_blue_01', FileType.PARTICLE_SYSTEM)
        yield Resource('materials/cable/goocable.vmt', FileType.MATERIAL)
        yield Resource('materials/sprites/glownomespit.vmt', FileType.MATERIAL)
    else:
        yield Resource('materials/sprites/gonomespit.vmt', FileType.MATERIAL)
        if variant == EZ_VARIANT_XEN:
            yield Resource('models/xonome.mdl', FileType.MODEL)
        else:
            yield Resource('models/gonome.mdl', FileType.MODEL)
