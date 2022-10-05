"""NPC entities."""
from . import *


def base_npc(pack: PackList, ent: Entity) -> None:
    """Resources precached in CAI_BaseNPC."""
    if conv_int(ent['ezvariant']) == EZ_VARIANT_TEMPORAL:
        pack.pack_soundscript('NPC_TemporalHeadcrab.Vanish')
        pack.pack_soundscript('NPC_TemporalHeadcrab.Appear')
        pack.pack_particle('ShadowCrab_Vanish')
        pack.pack_particle('ShadowCrab_Appear')
    equipment = ent['additionalequipment']
    if equipment not in ('', '0'):
        pack_ent_class(pack, equipment)


BASE_COMBINE = [
    mdl('models/Weapons/w_grenade.mdl'),
    sound('NPC_Combine.GrenadeLaunch'),
    sound('NPC_Combine.WeaponBash'),
    sound('Weapon_CombineGuard.Special1'),
    # TODO: Entropy Zero only.
    sound('NPC_Combine.Following'),
    sound('NPC_Combine.StopFollowing'),
]


@cls_func
def npc_antlion(pack: PackList, ent: Entity) -> None:
    """Antlions require different resources for the worker version."""
    base_npc(pack, ent)
    ez_variant = conv_int(ent['ezvariant'])
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 18):  # Is worker?
        if ez_variant == EZ_VARIANT_BLOOD:
            pack.pack_file("models/bloodlion_worker.mdl", FileType.MODEL)
        else:
            pack.pack_file("models/antlion_worker.mdl", FileType.MODEL)
        pack.pack_particle("blood_impact_antlion_worker_01")
        pack.pack_particle("antlion_gib_02")
        pack.pack_particle("blood_impact_yellow_01")

        pack_ent_class(pack, 'grenade_spit')
    else:
        if ez_variant == EZ_VARIANT_RAD:
            pack.pack_file("models/antlion_blue.mdl", FileType.MODEL)
            pack.pack_particle("blood_impact_blue_01")
        elif ez_variant == EZ_VARIANT_XEN:
            pack.pack_file("models/antlion_xen.mdl", FileType.MODEL)
            pack.pack_particle("blood_impact_antlion_01")
        elif ez_variant == EZ_VARIANT_BLOOD:
            pack.pack_file("models/bloodlion.mdl", FileType.MODEL)
            pack.pack_particle("blood_impact_antlion_01")
        else:
            pack.pack_file("models/antlion.mdl", FileType.MODEL)
            pack.pack_particle("blood_impact_antlion_01")
        pack.pack_particle("AntlionGib")
res('npc_antlion',
    *BASE_NPC,
    sound("NPC_Antlion.RunOverByVehicle"),
    sound("NPC_Antlion.MeleeAttack"),
    sound("NPC_Antlion.Footstep"),
    sound("NPC_Antlion.BurrowIn"),
    sound("NPC_Antlion.BurrowOut"),
    sound("NPC_Antlion.FootstepSoft"),
    sound("NPC_Antlion.FootstepHeavy"),
    sound("NPC_Antlion.MeleeAttackSingle"),
    sound("NPC_Antlion.MeleeAttackDouble"),
    sound("NPC_Antlion.Distracted"),
    sound("NPC_Antlion.Idle"),
    sound("NPC_Antlion.Pain"),
    sound("NPC_Antlion.Land"),
    sound("NPC_Antlion.WingsOpen"),
    sound("NPC_Antlion.LoopingAgitated"),
    sound("NPC_Antlion.Distracted"),

    # TODO: These are Episodic only..
    sound("NPC_Antlion.PoisonBurstScream"),
    sound("NPC_Antlion.PoisonBurstScreamSubmerged"),
    sound("NPC_Antlion.PoisonBurstExplode"),
    sound("NPC_Antlion.MeleeAttack_Muffled"),
    sound("NPC_Antlion.TrappedMetal"),
    sound("NPC_Antlion.ZappedFlip"),
    sound("NPC_Antlion.PoisonShoot"),
    sound("NPC_Antlion.PoisonBall"),
    # These aren't though.
    *[
        mdl("models/gibs/antlion_gib_{}_{}.mdl".format(size, i))
        for i in ('1', '2', '3')
        for size in ('small', 'medium', 'large')
    ],
)


@cls_func
def npc_antlionguard(pack: PackList, ent: Entity) -> None:
    """In Entropy Zero, some alternate models are available."""
    base_npc(pack, ent)
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 17):  # Inside Footsteps
        pack.pack_soundscript("NPC_AntlionGuard.Inside.StepLight")
        pack.pack_soundscript("NPC_AntlionGuard.Inside.StepHeavy")
    else:
        pack.pack_soundscript("NPC_AntlionGuard.StepLight")
        pack.pack_soundscript("NPC_AntlionGuard.StepHeavy")
    if 'ezvariant' in ent:  # Entropy Zero.
        variant = conv_int(ent['ezvaraiant'])
        if variant == EZ_VARIANT_XEN:
            pack.pack_file("models/antlion_guard_xen.mdl", FileType.MODEL)
            pack.pack_particle("xenpc_spawn")
        elif variant == EZ_VARIANT_RAD:
            pack.pack_file("models/antlion_guard_blue.mdl", FileType.MODEL)
            pack.pack_particle("blood_impact_blue_01")
        elif variant == EZ_VARIANT_BLOOD:
            pack.pack_file("models/bloodlion_guard.mdl", FileType.MODEL)
        else:
            pack.pack_file("models/antlion_guard.mdl", FileType.MODEL)
    else:  # Regular HL2.
        pack.pack_file("models/antlion_guard.mdl", FileType.MODEL)
res('npc_antlionguard',
    *BASE_NPC,
    mdl('NPC_AntlionGuard.Shove'),
    mdl('NPC_AntlionGuard.HitHard'),
    sound("NPC_AntlionGuard.Anger"),
    sound("NPC_AntlionGuard.Roar"),
    sound("NPC_AntlionGuard.Die"),
    sound("NPC_AntlionGuard.GrowlHigh"),
    sound("NPC_AntlionGuard.GrowlIdle"),
    sound("NPC_AntlionGuard.BreathSound"),
    sound("NPC_AntlionGuard.Confused"),
    sound("NPC_AntlionGuard.Fallover"),
    sound("NPC_AntlionGuard.FrustratedRoar"),
    part('blood_antlionguard_injured_light'),
    part('blood_antlionguard_injured_heavy'),
    # TODO: Episodic only.
    sound("NPC_AntlionGuard.NearStepLight"),
    sound("NPC_AntlionGuard.NearStepHeavy"),
    sound("NPC_AntlionGuard.FarStepLight"),
    sound("NPC_AntlionGuard.FarStepHeavy"),
    sound("NPC_AntlionGuard.BreatheLoop"),
    sound("NPC_AntlionGuard.ShellCrack"),
    sound("NPC_AntlionGuard.Pain_Roar"),
    mat("materials/sprites/grubflare1.vmt"),
    )


@cls_func
def npc_antlion_template_maker(pack: PackList, ent: Entity) -> None:
    """Depending on KVs this may or may not spawn workers."""
    # There will be an antlion present in the map, as the template
    # NPC. So we don't need to add those resources.
    if conv_int(ent['workerspawnrate']) > 0:
        # It randomly spawns worker antlions, so load that resource set.
        pack.pack_file("models/antlion_worker.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_worker_01", FileType.PARTICLE)
        pack.pack_file("antlion_gib_02", FileType.PARTICLE)
        pack.pack_file("blood_impact_yellow_01", FileType.PARTICLE)

        pack_ent_class(pack, 'grenade_spit')
    if conv_bool(ent['createspores']):
        pack_ent_class(pack, 'env_sporeexplosion')

res('npc_apcdriver', includes='npc_vehicledriver', func=base_npc)


@cls_func
def npc_arbeit_turret_floor(pack: PackList, ent: Entity) -> None:
    """Arbeit/Aperture turrets have EZ variants."""
    base_npc(pack, ent)
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        pack.pack_file('models/props/glowturret_01.mdl', FileType.MODEL)
    elif variant == EZ_VARIANT_ARBEIT:
        pack.pack_file('models/props/camoturret_01.mdl', FileType.MODEL)
        pack.pack_file('models/props/camoturret_02.mdl', FileType.MODEL)
    elif conv_int(ent['spawnflags']) & 0x200:  # Citizen Modified
        pack.pack_file('models/props/hackedturret_01.mdl', FileType.MODEL)
    else:
        pack.pack_file('models/props/turret_01.mdl', FileType.MODEL)

res('npc_arbeit_turret_floor',
    sound('NPC_ArbeitTurret.DryFire'),
    includes='npc_turret_floor',
    )

res('npc_badcop', mdl('models/bad_cop.mdl'), includes='npc_clonecop', func=base_npc)
res('npc_barnacle',
    *BASE_NPC,
    mdl('models/barnacle.mdl'),
    mdl("models/gibs/hgibs.mdl"),
    mdl("models/gibs/hgibs_scapula.mdl"),
    mdl("models/gibs/hgibs_rib.mdl"),
    mdl("models/gibs/hgibs_spine.mdl"),
    sound("NPC_Barnacle.Digest"),
    sound("NPC_Barnacle.BreakNeck"),
    sound("NPC_Barnacle.Scream"),
    sound("NPC_Barnacle.PullPant"),
    sound("NPC_Barnacle.TongueStretch"),
    sound("NPC_Barnacle.FinalBite"),
    sound("NPC_Barnacle.Die"),
    includes='npc_barnacle_tongue_tip',
    func=base_npc,
    )
res('npc_barney',
    *BASE_NPC,
    mdl("models/barney.mdl"),
    sound("NPC_Barney.FootstepLeft"),
    sound("NPC_Barney.FootstepRight"),
    sound("NPC_Barney.Die"),
    choreo("scenes/Expressions/BarneyIdle.vcd"),
    choreo("scenes/Expressions/BarneyAlert.vcd"),
    choreo("scenes/Expressions/BarneyCombat.vcd"),
    func=base_npc,
    )

@cls_func
def npc_bullsquid(pack: PackList, ent: Entity) -> None:
    """This has various EZ variants."""
    base_npc(pack, ent)
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_XEN:
        pack.pack_file('models/bullsquid_xen.mdl', FileType.MODEL)
        pack.pack_file('models/babysquid_xen.mdl', FileType.MODEL)
        pack.pack_file('models/bullsquid_egg_xen.mdl', FileType.MODEL)
        pack.pack_particle('blood_impact_yellow_01')
    elif variant == EZ_VARIANT_RAD:
        pack.pack_file('models/bullsquid_rad.mdl', FileType.MODEL)
        pack.pack_file('models/babysquid_rad.mdl', FileType.MODEL)
        pack.pack_file('models/bullsquid_egg_rad.mdl', FileType.MODEL)
        pack.pack_particle('blood_impact_blue_01')
    else:
        pack.pack_file('models/bullsquid.mdl', FileType.MODEL)
        pack.pack_file('models/babysquid.mdl', FileType.MODEL)
        pack.pack_file('models/bullsquid_egg.mdl', FileType.MODEL)
        pack.pack_particle('blood_impact_yellow_01')


res('npc_bullsquid',  # This is the EZ2/HLS variant, TODO: resources for Black mesa version?
    *BASE_NPC,
    mat('materials/sprites/greenspit1.vmt'),
    part('bullsquid_explode'),
    sound('NPC_Bullsquid.Idle'),
    sound('NPC_Bullsquid.Pain'),
    sound('NPC_Bullsquid.Alert'),
    sound('NPC_Bullsquid.Death'),
    sound('NPC_Bullsquid.Attack1'),
    sound('NPC_Bullsquid.FoundEnemy'),
    sound('NPC_Bullsquid.Growl'),
    sound('NPC_Bullsquid.TailWhi'),
    sound('NPC_Bullsquid.Bite'),
    sound('NPC_Bullsquid.Eat'),
    sound('NPC_Babysquid.Idle'),
    sound('NPC_Babysquid.Pain'),
    sound('NPC_Babysquid.Alert'),
    sound('NPC_Babysquid.Death'),
    sound('NPC_Babysquid.Attack1'),
    sound('NPC_Babysquid.FoundEnemy'),
    sound('NPC_Babysquid.Growl'),
    sound('NPC_Babysquid.TailWhip'),
    sound('NPC_Babysquid.Bite'),
    sound('NPC_Babysquid.Eat'),
    sound('NPC_Antlion.PoisonShoot'),
    sound('NPC_Antlion.PoisonBall'),
    sound('NPC_Bullsquid.Explode'),
    # The bullsquid might spawn an npc_egg, which then spawns bullsquid...
    sound('npc_bullsquid.egg_alert'),
    sound('npc_bullsquid.egg_hatch'),
    part('bullsquid_egg_hatch'),
    includes='grenade_spit',
    )
res('npc_cscanner',
    *BASE_NPC,
    # In HL2, the claw scanner variant is chosen if the map starts with d3_c17.
    # In episodic, npc_clawscanner is now available to force that specifically.
    # TODO: Check the BSP name, pack shield in that case.
    mdl("models/combine_scanner.mdl"),
    mdl("models/gibs/scanner_gib01.mdl"),
    mdl("models/gibs/scanner_gib02.mdl"),
    mdl("models/gibs/scanner_gib02.mdl"),
    mdl("models/gibs/scanner_gib04.mdl"),
    mdl("models/gibs/scanner_gib05.mdl"),
    mat("material/sprites/light_glow03.vmt"),
    sound("NPC_CScanner.Shoot"),
    sound("NPC_CScanner.Alert"),
    sound("NPC_CScanner.Die"),
    sound("NPC_CScanner.Combat"),
    sound("NPC_CScanner.Idle"),
    sound("NPC_CScanner.Pain"),
    sound("NPC_CScanner.TakePhoto"),
    sound("NPC_CScanner.AttackFlash"),
    sound("NPC_CScanner.DiveBombFlyby"),
    sound("NPC_CScanner.DiveBomb"),
    sound("NPC_CScanner.DeployMine"),
    sound("NPC_CScanner.FlyLoop"),
    func=base_npc,
    )

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
def npc_citizen(pack: PackList, ent: Entity) -> None:
    """Cizizens have a complex set of precaching rules."""
    base_npc(pack, ent)
    if ent['targetname'] == 'matt':
        # Special crowbar.
        pack.pack_file("models/props_canal/mattpipe.mdl", FileType.MODEL)

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
            pack.pack_file('models/humans/group01/' + head, FileType.MODEL)
            pack.pack_file('models/humans/group02/' + head, FileType.MODEL)
            pack.pack_file('models/humans/group03/' + head, FileType.MODEL)
            pack.pack_file('models/humans/group03m/' + head, FileType.MODEL)
        return
    elif cit_type == 1:  # Downtrodden
        folder = 'group01'
    elif cit_type == 2:  # Refugee
        folder = 'group02'
    elif cit_type == 3:  # Rebel
        folder = 'group03'
        # The rebels have an additional set of models.
        for head in CIT_HEADS:
            pack.pack_file('models/humans/group03m/' + head, FileType.MODEL)
    elif cit_type == 4:  # Use model in KVs directly.
        return
    else:  # Invalid type?
        # TODO: Entropy Zero variants.
        return

    for head in CIT_HEADS:
        pack.pack_file(f'models/humans/{folder}/{head}', FileType.MODEL)

res('npc_citizen',
    *BASE_NPC,
    sound("NPC_Citizen.FootstepLeft"),
    sound("NPC_Citizen.FootstepRight"),
    sound("NPC_Citizen.Die"),
    )

res('npc_clonecop',
    *BASE_NPC,
    mdl('models/clone_cop.mdl'),
    part('blood_spurt_synth_01'),
    part('blood_drip_synth_01'),
    part('blood_impact_blue_01'),  # Radiation/temporal only, but not really important.
    includes='item_ammo_ar2_altfire',
    func=base_npc,
    )


@cls_func
def npc_combinedropship(pack: PackList, ent: Entity) -> None:
    """The Combine Dropship may spawn with a variety of cargo types."""
    base_npc(pack, ent)
    cargo_type = conv_int(ent['cratetype'])
    if cargo_type == -3:  # Spawns a prop_dynamic Jeep
        pack.pack_file("models/buggy.mdl", FileType.MODEL)
    elif cargo_type == -1:  # Strider
        pack_ent_class(pack, 'npc_strider')
    elif cargo_type == 1:  # Soldiers in a container.
        pack_ent_class(pack, 'prop_dropship_container')
    # Other valid values:
    # -2 = Grabs the APC specified in KVs - that'll load its own resources.
    #  0 = Roller Hopper, does nothing
    #  2 = No cargo


@cls_func
def npc_combinegunship(pack: PackList, ent: Entity) -> None:
    """This has the ability to spawn as the helicopter instead."""
    base_npc(pack, ent)
    if conv_int(ent['spawnflags']) & (1 << 13):
        pack.pack_file("models/combine_helicopter.mdl", FileType.MODEL)
        pack.pack_file("models/combine_helicopter_broken.mdl", FileType.MODEL)
        pack_ent_class(pack, 'helicopter_chunk')
    else:
        pack.pack_file("models/gunship.mdl", FileType.MODEL)


res('npc_combine_cannon',
    *BASE_NPC,
    mdl('models/combine_soldier.mdl'),
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    sound('NPC_Combine_Cannon.FireBullet'),
    func=base_npc,
    )
res('npc_combine_s', *BASE_COMBINE)
res('npc_clawscanner',
    *BASE_NPC,
    mdl("models/shield_scanner.mdl"),
    mdl("models/gibs/Shield_Scanner_Gib1.mdl"),
    mdl("models/gibs/Shield_Scanner_Gib2.mdl"),
    mdl("models/gibs/Shield_Scanner_Gib3.mdl"),
    mdl("models/gibs/Shield_Scanner_Gib4.mdl"),
    mdl("models/gibs/Shield_Scanner_Gib5.mdl"),
    mdl("models/gibs/Shield_Scanner_Gib6.mdl"),
    mat("material/sprites/light_glow03.vmt"),

    sound("NPC_SScanner.Shoot"),
    sound("NPC_SScanner.Alert"),
    sound("NPC_SScanner.Die"),
    sound("NPC_SScanner.Combat"),
    sound("NPC_SScanner.Idle"),
    sound("NPC_SScanner.Pain"),
    sound("NPC_SScanner.TakePhoto"),
    sound("NPC_SScanner.AttackFlash"),
    sound("NPC_SScanner.DiveBombFlyby"),
    sound("NPC_SScanner.DiveBomb"),
    sound("NPC_SScanner.DeployMine"),
    sound("NPC_SScanner.FlyLoop"),
    includes="combine_mine",
    func=base_npc,
    )

@cls_func
def npc_egg(pack: PackList, ent: Entity) -> None:
    """These are EZ2 bullsquid eggs, which spawn a specific EZ variant."""
    pack_ent_class(pack, 'npc_bullsquid', ezvariant=ent['ezvariant'])

res('npc_egg',
    *BASE_NPC,
    mdl('models/eggs/bullsquid_egg.mdl'),
    sound('npc_bullsquid.egg_alert'),
    sound('npc_bullsquid.egg_hatch'),
    part('bullsquid_egg_hatch'),
    )
res('npc_grenade_bugbait',
    mdl("models/weapons/w_bugbait.mdl"),
    sound("GrenadeBugBait.Splat"),
    includes='grenade',
    )

res('npc_handgrenade', mdl('models/weapons/w_grenade.mdl'))

@cls_func
def npc_maker(pack: PackList, ent: Entity) -> None:
    """We spawn the NPC automatically."""
    try:
        pack_ent_class(pack, ent['npctype'])
    except ValueError:
        # Dependent on keyvalues.
        pass


@cls_func
def npc_metropolice(pack: PackList, ent: Entity) -> None:
    """If a spawnflag is set, a cheap model is used."""
    base_npc(pack, ent)
    if conv_int(ent['spawnflags']) & 5:
        pack.pack_file("models/police_cheaple.mdl", FileType.MODEL)
    else:
        pack.pack_file("models/police.mdl", FileType.MODEL)

res('npc_metropolice',
    *BASE_NPC, *BASE_COMBINE,
    sound("NPC_Metropolice.Shove"),
    sound("NPC_MetroPolice.WaterSpeech"),
    sound("NPC_MetroPolice.HidingSpeech"),
    # TODO: pack.pack_sentence_group("METROPOLICE"),
    # Entity precaches npc_handgrenade, but they actually spawn these.
    includes='npc_grenade_frag npc_manhack',
)

res('npc_rocket_turret',
    *BASE_NPC,
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    mdl('models/props_bts/rocket_sentry.mdl'),
    sound('NPC_RocketTurret.LockingBeep'),
    sound('NPC_FloorTurret.LockedBeep'),
    sound('NPC_FloorTurret.RocketFire'),
    includes='rocket_turret_projectile',
    func=base_npc,
    )


@cls_func
def npc_stalker(pack: PackList, ent: Entity) -> None:
    """Mapbase optionally allows blood particles."""
    base_npc(pack, ent)
    if conv_bool(ent['bleed']):  # Mapbase
        pack.pack_particle('blood_impact_synth_01')

res('npc_template_maker')  # This specially precaches, but the ent must exist in the map already.
res('npc_turret_ceiling',
    *BASE_NPC,
    mdl('models/combine_turrets/ceiling_turret.mdl'),
    mat('materials/sprites/glow1.vmt'),
    sound('NPC_CeilingTurret.Active'),
    sound('NPC_CeilingTurret.Alert'),
    sound('NPC_CeilingTurret.Deploy'),
    sound('NPC_CeilingTurret.Die'),
    sound('NPC_CeilingTurret.Move'),
    sound('NPC_CeilingTurret.Ping'),
    sound('NPC_CeilingTurret.Retire'),
    sound('NPC_CeilingTurret.ShotSounds'),
    sound('NPC_FloorTurret.DryFire'),
    func=base_npc,
    )

res('npc_turret_ground',
    *BASE_NPC,
    mdl('models/combine_turrets/ground_turret.mdl'),
    mat('materials/effects/bluelaser2.vmt'),
    sound('NPC_CeilingTurret.Deploy'),
    sound('NPC_FloorTurret.ShotSounds'),
    sound('NPC_FloorTurret.Die'),
    sound('NPC_FloorTurret.Ping'),
    sound('DoSpark'),
    func=base_npc,
    )

res('npc_turret_floor',
    *BASE_NPC,
    mdl('models/combine_turrets/floor_turret.mdl'),
    mdl('models/combine_turrets/citizen_turret.mdl'),
    mat('materials/effects/laser1.vmt'),  # Citizen only
    sound('NPC_FloorTurret.AlarmPing'),  # Citizen only
    sound('NPC_Combine.WeaponBash'),
    sound('NPC_FloorTurret.Activate'),
    sound('NPC_FloorTurret.Alarm'),
    sound('NPC_FloorTurret.Alert'),
    sound('NPC_FloorTurret.Deploy'),
    sound('NPC_FloorTurret.Destruct'),
    sound('NPC_FloorTurret.Die'),
    sound('NPC_FloorTurret.DryFire'),
    sound('NPC_FloorTurret.Move'),
    sound('NPC_FloorTurret.Ping'),
    sound('NPC_FloorTurret.Retire'),
    sound('NPC_FloorTurret.Retract'),
    sound('NPC_FloorTurret.ShotSounds'),
    part('explosion_turret_break'),
    func=base_npc,
    )

res('npc_turret_lab',
    *BASE_NPC,
    mdl('models/props_lab/labturret_npc.mdl'),
    mat('materials/sprites/glow1.vmt'),
    sound('NPC_LabTurret.Retire'),
    sound('NPC_LabTurret.Deploy'),
    sound('NPC_LabTurret.Move'),
    sound('NPC_LabTurret.Active'),
    sound('NPC_LabTurret.Alert'),
    sound('NPC_LabTurret.ShotSounds'),
    sound('NPC_LabTurret.Ping'),
    sound('NPC_LabTurret.Die'),
    sound('NPC_FloorTurret.DryFire'),
    func=base_npc,
    )


@cls_func
def npc_zassassin(pack: PackList, ent: Entity) -> None:
    """Entropy Zero 2's "Plan B"/Gonome. """
    base_npc(pack, ent)
    variant = conv_int(ent['ezvariant'])
    if variant == EZ_VARIANT_RAD:
        pack.pack_file('models/glownome.mdl', FileType.MODEL)
        pack.pack_particle('blood_impact_blue_01')
        pack.pack_file('materials/cable/goocable.vmt', FileType.MATERIAL)
        pack.pack_file('materials/sprites/glownomespit.vmt', FileType.MATERIAL)
    else:
        pack.pack_file('materials/sprites/gonomespit.vmt', FileType.MATERIAL)
        if variant == EZ_VARIANT_XEN:
            pack.pack_file('models/xonome.mdl', FileType.MODEL)
        else:
            pack.pack_file('models/gonome.mdl', FileType.MODEL)

res('npc_zassassin',
    *BASE_NPC,
    sound('Gonome.Idle'),
    sound('Gonome.Pain'),
    sound('Gonome.Alert'),
    sound('Gonome.Die'),
    sound('Gonome.Attack'),
    sound('Gonome.Bite'),
    sound('Gonome.Growl'),
    sound('Gonome.FoundEnem'),
    sound('Gonome.RetreatMod'),
    sound('Gonome.BerserkMod'),
    sound('Gonome.RunFootstepLeft'),
    sound('Gonome.RunFootstepRight'),
    sound('Gonome.FootstepLeft'),
    sound('Gonome.FootstepRight'),
    sound('Gonome.JumpLand'),
    sound('Gonome.Eat'),
    sound('Gonome.BeginSpawnCrab'),
    sound('Gonome.EndSpawnCrab'),
    part('glownome_explode'),
    sound('npc_zassassin.kickburst'),
    includes='squidspit',
    aliases='monster_gonome',
    )

res('npc_zombie',
    *BASE_NPC,
    mdl("models/zombie/classic.mdl"),
    mdl("models/zombie/classic_torso.mdl"),
    mdl("models/zombie/classic_legs.mdl"),
    sound("Zombie.FootstepRight"),
    sound("Zombie.FootstepLeft"),
    sound("Zombie.FootstepLeft"),
    sound("Zombie.ScuffRight"),
    sound("Zombie.ScuffLeft"),
    sound("Zombie.AttackHit"),
    sound("Zombie.AttackMiss"),
    sound("Zombie.Pain"),
    sound("Zombie.Die"),
    sound("Zombie.Alert"),
    sound("Zombie.Idle"),
    sound("Zombie.Attack"),
    sound("NPC_BaseZombie.Moan1"),
    sound("NPC_BaseZombie.Moan2"),
    sound("NPC_BaseZombie.Moan3"),
    sound("NPC_BaseZombie.Moan4"),
    func=base_npc,
    )
# Actually an alias, but we don't want to swap these.
res('npc_zombie_torso', includes='npc_zombie', func=base_npc)
res('npc_zombine',
    *BASE_NPC,
    mdl("models/zombie/zombie_soldier.mdl"),
    sound("Zombie.FootstepRight"),
    sound("Zombie.FootstepLeft"),
    sound("Zombine.ScuffRight"),
    sound("Zombine.ScuffLeft"),
    sound("Zombie.AttackHit"),
    sound("Zombie.AttackMiss"),
    sound("Zombine.Pain"),
    sound("Zombine.Die"),
    sound("Zombine.Alert"),
    sound("Zombine.Idle"),
    sound("Zombine.ReadyGrenade"),
    sound("ATV_engine_null"),
    sound("Zombine.Charge"),
    sound("Zombie.Attack"),
    includes='npc_zombie',
    func=base_npc,
    )
