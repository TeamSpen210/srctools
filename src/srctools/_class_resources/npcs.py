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


BASE_NPC = [
    sound("AI_BaseNPC.SwishSound"),
    sound("AI_BaseNPC.BodyDrop_Heavy"),
    sound("AI_BaseNPC.BodyDrop_Light"),
    sound("AI_BaseNPC.SentenceStop"),
]

BASE_COMBINE = [
    mdl('models/Weapons/w_grenade.mdl'),
    sound('NPC_Combine.GrenadeLaunch'),
    sound('NPC_Combine.WeaponBash'),
    sound('Weapon_CombineGuard.Special1'),
    # TODO: Entropy Zero only.
    sound('NPC_Combine.Following'),
    sound('NPC_Combine.StopFollowing'),
]


res('npc_advisor',
    *BASE_NPC,
    mdl("models/advisor.mdl"),
    mat("materials/sprites/lgtning.vmt"),
    mat("materials/sprites/greenglow1.vmt"),
    sound("NPC_Advisor.Blast"),
    sound("NPC_Advisor.Gib"),
    sound("NPC_Advisor.Idle"),
    sound("NPC_Advisor.Alert"),
    sound("NPC_Advisor.Die"),
    sound("NPC_Advisor.Pain"),
    sound("NPC_Advisor.ObjectChargeUp"),
    part("Advisor_Psychic_Beam"),
    part("advisor_object_charge"),
    )

res('npc_aliencrow',  # Entropy Zero 2
    *BASE_NPC,
    mdl("models/aflock.mdl"),

    sound("NPC_Boid.Hop"),
    sound("NPC_Boid.Squawk"),
    sound("NPC_Boid.Gib"),
    sound("NPC_Boid.Idle"),
    sound("NPC_Boid.Alert"),
    sound("NPC_Boid.Die"),
    sound("NPC_Boid.Pain"),
    sound("NPC_Boid.Flat"),
    aliases="npc_boid",
    )


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

res('npc_antlion_grub',
    *BASE_NPC,
    mdl("models/antlion_grub.mdl"),
    mdl("models/antlion_grub_squashed.mdl"),
    mat("materials/sprites/grubflare1.vmt"),
    sound("NPC_Antlion_Grub.Idle"),
    sound("NPC_Antlion_Grub.Alert"),
    sound("NPC_Antlion_Grub.Stimulated"),
    sound("NPC_Antlion_Grub.Die"),
    sound("NPC_Antlion_Grub.Squish"),
    part("GrubSquashBlood"),
    part("GrubBlood"),
    includes="item_grubnugget",
    func=base_npc,
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

res('npc_alyx',
    *BASE_NPC,
    mdl('models/alyx_emptool_prop.mdl'),
    sound('npc_alyx.die'),
    sound('DoSpark'),
    sound('npc_alyx.starthacking'),
    sound('npc_alyx.donehacking'),
    sound('npc_alyx.readytohack'),
    sound('npc_alyx.interruptedhacking'),
    sound('ep_01.al_dark_breathing01'),
    sound('Weapon_CombineGuard.Special1'),
    includes='env_alyxemp',
    func=base_npc,
    )
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
res('npc_barnacle_tongue_tip', 'models/props_junk/rock001a.mdl')  # Random model it loads.
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
res('npc_breen', *BASE_NPC, mdl("models/breen.mdl"))
res('npc_bullseye', *BASE_NPC)


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

res('npc_combine_camera',
    *BASE_NPC,
    mdl("models/combine_camera/combine_camera.mdl"),
    mat("materials/sprites/glow1.vmt"),
    mat("materials/sprites/light_glow03.vmt"),
    sound("NPC_CombineCamera.Move"),
    sound("NPC_CombineCamera.BecomeIdle"),
    sound("NPC_CombineCamera.Active"),
    sound("NPC_CombineCamera.Click"),
    sound("NPC_CombineCamera.Ping"),
    sound("NPC_CombineCamera.Angry"),
    sound("NPC_CombineCamera.Die"),
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

res('npc_combinedropship',
    *BASE_NPC,
    mdl("models/combine_dropship.mdl"),
    sound("NPC_CombineDropship.RotorLoop"),
    sound("NPC_CombineDropship.FireLoop"),
    sound("NPC_CombineDropship.NearRotorLoop"),
    sound("NPC_CombineDropship.OnGroundRotorLoop"),
    sound("NPC_CombineDropship.DescendingWarningLoop"),
    sound("NPC_CombineDropship.NearRotorLoop"),
    )


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

res('npc_combinegunship',
    *BASE_NPC,
    mat("materials/sprites/lgtning.vmt"),
    mat("materials/effects/ar2ground2"),
    mat("materials/effects/blueblackflash"),

    sound("NPC_CombineGunship.SearchPing"),
    sound("NPC_CombineGunship.PatrolPing"),
    sound("NPC_Strider.Charge"),
    sound("NPC_Strider.Shoot"),
    sound("NPC_CombineGunship.SeeEnemy"),
    sound("NPC_CombineGunship.CannonStartSound"),
    sound("NPC_CombineGunship.Explode"),
    sound("NPC_CombineGunship.Pain"),
    sound("NPC_CombineGunship.CannonStopSound"),

    sound("NPC_CombineGunship.DyingSound"),
    sound("NPC_CombineGunship.CannonSound"),
    sound("NPC_CombineGunship.RotorSound"),
    sound("NPC_CombineGunship.ExhaustSound"),
    sound("NPC_CombineGunship.RotorBlastSound"),

    # TODO: These two are Episodic only.
    mat("materials/sprites/physbeam.vmt"),
    includes='env_citadel_energy_core',
    )

res('npc_combine_cannon',
    *BASE_NPC,
    mdl('models/combine_soldier.mdl'),
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    sound('NPC_Combine_Cannon.FireBullet'),
    func=base_npc,
    )
res('npc_combine_s',
    *BASE_NPC, *BASE_COMBINE,
    mdl('models/combine_soldier.mdl'),
    # TODO: Manhacks added by Mapbase.
    # Entity precaches npc_handgrenade, but they actually spawn these.
    includes='npc_handgrenade npc_manhack item_healthvial weapon_frag item_ammo_ar2_altfire',
    func=base_npc,
    )
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
res('npc_cranedriver', includes='npc_vehicledriver', func=base_npc)
res('npc_crow',
    *BASE_NPC,
    mdl("models/crow.mdl"),
    sound("NPC_Crow.Hop"),
    sound("NPC_Crow.Squawk"),
    sound("NPC_Crow.Gib"),
    sound("NPC_Crow.Idle"),
    sound("NPC_Crow.Alert"),
    sound("NPC_Crow.Die"),
    sound("NPC_Crow.Pain"),
    sound("NPC_Crow.Flap"),
    func=base_npc,
    )
res('npc_dog',
    *BASE_NPC,
    mdl("models/dog.mdl"),
    sound("Weapon_PhysCannon.Launch"),
    mat("materials/sprites/orangelight1.vmt"),
    mat("materials/sprites/physcannon_bluelight2.vmt"),
    mat("materials/sprites/glow04_noz.vmt"),
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
res('npc_eli', *BASE_NPC, mdl("models/eli.mdl"), func=base_npc)
res('npc_enemyfinder', *BASE_NPC, mdl("models/player.mdl"), func=base_npc)
res('npc_enemyfinder_combinecannon', includes='npc_enemyfinder', func=base_npc)
res('npc_gman', *BASE_NPC, mdl('models/gman.mdl'), func=base_npc)
res('npc_grenade_bugbait',
    mdl("models/weapons/w_bugbait.mdl"),
    sound("GrenadeBugBait.Splat"),
    includes='grenade',
    )
res('npc_grenade_frag',
    mdl("models/Weapons/w_grenade.mdl"),
    mat('materials/sprites/redglow1.vmt'),
    mat('materials/sprites/bluelaser1.vmt'),
    sound("Grenade.Blip"),
    includes='grenade',
    )
res('npc_grenade_hopwire',  #  A primed EZ2 Xen Grenade.
    mdl('models/props_junk/metal_paintcan001b.mdl'),  # "Dense Ball Model"
    sound('NPC_Strider.Charge'),
    sound('NPC_Strider.Shoot'),
    sound('WeaponXenGrenade.Explode'),
    sound('WeaponXenGrenade.SpawnXenPC'),
    sound('WeaponXenGrenade.Blip'),
    sound('WeaponXenGrenade.Hop'),
    includes='vortex_controller grenade',
    )

res('npc_fastzombie',
    *BASE_NPC,
    mdl("models/zombie/fast.mdl"),
    # TODO - Episodic only:
        mdl("models/zombie/Fast_torso.mdl"),
        sound("NPC_FastZombie.CarEnter1"),
        sound("NPC_FastZombie.CarEnter2"),
        sound("NPC_FastZombie.CarEnter3"),
        sound("NPC_FastZombie.CarEnter4"),
        sound("NPC_FastZombie.CarScream"),
    mdl("models/gibs/fast_zombie_torso.mdl"),
    mdl("models/gibs/fast_zombie_legs.mdl"),
    sound("NPC_FastZombie.LeapAttack"),
    sound("NPC_FastZombie.FootstepRight"),
    sound("NPC_FastZombie.FootstepLeft"),
    sound("NPC_FastZombie.AttackHit"),
    sound("NPC_FastZombie.AttackMiss"),
    sound("NPC_FastZombie.LeapAttack"),
    sound("NPC_FastZombie.Attack"),
    sound("NPC_FastZombie.Idle"),
    sound("NPC_FastZombie.AlertFar"),
    sound("NPC_FastZombie.AlertNear"),
    sound("NPC_FastZombie.GallopLeft"),
    sound("NPC_FastZombie.GallopRight"),
    sound("NPC_FastZombie.Scream"),
    sound("NPC_FastZombie.RangeAttack"),
    sound("NPC_FastZombie.Frenzy"),
    sound("NPC_FastZombie.NoSound"),
    sound("NPC_FastZombie.Die"),
    sound("NPC_FastZombie.Gurgle"),
    sound("NPC_FastZombie.Moan1"),
    func=base_npc,
    )
# Actually an alias, but we don't want to swap these.
res(
    'npc_fastzombie_torso',
    includes='npc_fastzombie',
    func=base_npc,
)

res('npc_fisherman',
    *BASE_NPC,
    mdl("models/lostcoast/fisherman/fisherman.mdl"),
    sound("NPC_Fisherman.FootstepLeft"),
    sound("NPC_Fisherman.FootstepRight"),
    sound("NPC_Fisherman.Die"),
    choreo("scenes/Expressions/FishermanIdle.vcd"),
    choreo("scenes/Expressions/FishermanAlert.vcd"),
    choreo("scenes/Expressions/FishermanCombat.vcd"),
    func=base_npc,
    )

res('npc_headcrab',
    *BASE_NPC,
    mdl('models/headcrabclassic.mdl'),
    sound('NPC_HeadCrab.Gib'),
    sound('NPC_HeadCrab.Idle'),
    sound('NPC_HeadCrab.Alert'),
    sound('NPC_HeadCrab.Pain'),
    sound('NPC_HeadCrab.Die'),
    sound('NPC_HeadCrab.Attack'),
    sound('NPC_HeadCrab.Bite'),
    sound('NPC_Headcrab.BurrowIn'),
    sound('NPC_Headcrab.BurrowOut'),
    func=base_npc,
    )

res('npc_headcrab_black',
    *BASE_NPC,
    mdl('models/headcrabblack.mdl'),

    sound('NPC_BlackHeadcrab.Telegraph'),
    sound('NPC_BlackHeadcrab.Attack'),
    sound('NPC_BlackHeadcrab.Bite'),
    sound('NPC_BlackHeadcrab.Threat'),
    sound('NPC_BlackHeadcrab.Alert'),
    sound('NPC_BlackHeadcrab.Idle'),
    sound('NPC_BlackHeadcrab.Talk'),
    sound('NPC_BlackHeadcrab.AlertVoice'),
    sound('NPC_BlackHeadcrab.Pain'),
    sound('NPC_BlackHeadcrab.Die'),
    sound('NPC_BlackHeadcrab.Impact'),
    sound('NPC_BlackHeadcrab.ImpactAngry'),
    sound('NPC_BlackHeadcrab.FootstepWalk'),
    sound('NPC_BlackHeadcrab.Footstep'),

    sound('NPC_HeadCrab.Gib'),
    sound('NPC_Headcrab.BurrowIn'),
    sound('NPC_Headcrab.BurrowOut'),
    aliases='npc_headcrab_poison',
    func=base_npc,
)

res('npc_headcrab_fast',
    *BASE_NPC,
    mdl('models/headcrab.mdl'),
    sound('NPC_FastHeadCrab.Idle'),
    sound('NPC_FastHeadCrab.Alert'),
    sound('NPC_FastHeadCrab.Pain'),
    sound('NPC_FastHeadCrab.Die'),
    sound('NPC_FastHeadCrab.Attack'),
    sound('NPC_FastHeadCrab.Bite'),

    sound('NPC_HeadCrab.Gib'),
    sound('NPC_Headcrab.BurrowIn'),
    sound('NPC_Headcrab.BurrowOut'),
    func=base_npc,
    )

res('npc_heli_avoidbox')
res('npc_heli_avoidsphere')
res('npc_heli_nobomb')
res('npc_helicopter',
    *BASE_NPC,
    mdl("models/combine_helicopter.mdl"),
    mdl("models/combine_helicopter_broken.mdl"),
    mat("materials/sprites/redglow1.vmt"),
    includes='helicopter_chunk grenade_helicopter',
    func=base_npc,
    )
res('npc_helicoptersensor')
res('npc_handgrenade', mdl('models/weapons/w_grenade.mdl'))
res('npc_hunter',
    *BASE_NPC,
    mdl("models/hunter.mdl"),

    sound("NPC_Hunter.Idle"),
    sound("NPC_Hunter.Scan"),
    sound("NPC_Hunter.Alert"),
    sound("NPC_Hunter.Pain"),
    sound("NPC_Hunter.PreCharge"),
    sound("NPC_Hunter.Angry"),
    sound("NPC_Hunter.Death"),
    sound("NPC_Hunter.FireMinigun"),
    sound("NPC_Hunter.Footstep"),
    sound("NPC_Hunter.BackFootstep"),
    sound("NPC_Hunter.FlechetteVolleyWarn"),
    sound("NPC_Hunter.FlechetteShoot"),
    sound("NPC_Hunter.FlechetteShootLoop"),
    sound("NPC_Hunter.FlankAnnounce"),
    sound("NPC_Hunter.MeleeAnnounce"),
    sound("NPC_Hunter.MeleeHit"),
    sound("NPC_Hunter.TackleAnnounce"),
    sound("NPC_Hunter.TackleHit"),
    sound("NPC_Hunter.ChargeHitEnemy"),
    sound("NPC_Hunter.ChargeHitWorld"),
    sound("NPC_Hunter.FoundEnemy"),
    sound("NPC_Hunter.FoundEnemyAck"),
    sound("NPC_Hunter.DefendStrider"),
    sound("NPC_Hunter.HitByVehicle"),

    part("hunter_muzzle_flash"),
    part("blood_impact_synth_01"),
    part("blood_impact_synth_01_arc_parent"),
    part("blood_spurt_synth_01"),
    part("blood_drip_synth_01"),

    choreo("scenes/npc/hunter/hunter_scan.vcd"),
    choreo("scenes/npc/hunter/hunter_eyeclose.vcd"),
    choreo("scenes/npc/hunter/hunter_roar.vcd"),
    choreo("scenes/npc/hunter/hunter_pain.vcd"),
    choreo("scenes/npc/hunter/hunter_eyedarts_top.vcd"),
    choreo("scenes/npc/hunter/hunter_eyedarts_bottom.vcd"),

    mat("materials/effects/water_highlight.vmt"),

    includes="hunter_flechette sparktrail",
    func=base_npc,
    )
# This uses a template so the hunter has to be already in the map and will be analysed.
res('npc_hunter_maker', includes='npc_template_maker')

res('npc_kleiner', *BASE_NPC, mdl('models/kleiner.mdl'), func=base_npc)
res('npc_launcher',
    *BASE_NPC,
    mdl('models/player.mdl'),
    includes='grenade_homer grenade_pathfollower',
    func=base_npc,
    )

@cls_func
def npc_maker(pack: PackList, ent: Entity) -> None:
    """We spawn the NPC automatically."""
    try:
        pack_ent_class(pack, ent['npctype'])
    except ValueError:
        # Dependent on keyvalues.
        pass

res('npc_magnusson', *BASE_NPC, mdl('models/magnusson.mdl'), func=base_npc)
res('npc_manhack',
    *BASE_NPC,
    mdl("models/manhack.mdl"),
    mat("materials/sprites/glow1.vmt"),

    sound("NPC_Manhack.Die"),
    sound("NPC_Manhack.Bat"),
    sound("NPC_Manhack.Grind"),
    sound("NPC_Manhack.Slice"),
    sound("NPC_Manhack.EngineNoise"),
    sound("NPC_Manhack.Unpack"),
    sound("NPC_Manhack.ChargeAnnounce"),
    sound("NPC_Manhack.ChargeEnd"),
    sound("NPC_Manhack.Stunned"),
    sound("NPC_Manhack.EngineSound1"),
    sound("NPC_Manhack.EngineSound2"),
    sound("NPC_Manhack.BladeSound"),
    func=base_npc,
    )
res('npc_monk',
    *BASE_NPC,
    mdl("models/monk.mdl"),
    sound("NPC_Citizen.FootstepLeft"),
    sound("NPC_Citizen.FootstepRight"),
    func=base_npc,
    )
res('npc_mossman', *BASE_NPC, mdl('models/mossman.mdl'), func=base_npc)


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
res('npc_missiledefense',
    *BASE_NPC,
    mdl('models/missile_defense.mdl'),
    mdl('models/gibs/missile_defense_gibs.mdl'),
    sound('NPC_MissileDefense.Attack'),
    sound('NPC_MissileDefense.Reload'),
    sound('NPC_MissileDefense.Turn'),
    func=base_npc,
    )

res('npc_pigeon',
    *BASE_NPC,
    mdl("models/pigeon.mdl"),
    sound("NPC_Pigeon.Idle"),

    sound("NPC_Crow.Hop"),
    sound("NPC_Crow.Squawk"),
    sound("NPC_Crow.Gib"),
    sound("NPC_Crow.Pain"),
    sound("NPC_Crow.Die"),
    func=base_npc,
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
res('npc_rollermine',
    *BASE_NPC,
    mdl("models/roller.mdl"),
    mdl("models/roller_spikes.mdl"),
    mat("materials/sprites/bluelight1.vmt"),
    mat("materials/effects/rollerglow.vmt"),
    mat("materials/sprites/rollermine_shock.vmt"),
    mat("materials/sprites/rollermine_shock_yellow.vmt"),

    sound("NPC_RollerMine.Taunt"),
    sound("NPC_RollerMine.OpenSpikes"),
    sound("NPC_RollerMine.Warn"),
    sound("NPC_RollerMine.Shock"),
    sound("NPC_RollerMine.ExplodeChirp"),
    sound("NPC_RollerMine.Chirp"),
    sound("NPC_RollerMine.ChirpRespond"),
    sound("NPC_RollerMine.ExplodeChirpRespond"),
    sound("NPC_RollerMine.JoltVehicle"),
    sound("NPC_RollerMine.Tossed"),
    sound("NPC_RollerMine.Hurt"),
    sound("NPC_RollerMine.Roll"),
    sound("NPC_RollerMine.RollWithSpikes"),
    sound("NPC_RollerMine.Ping"),
    sound("NPC_RollerMine.Held"),
    sound("NPC_RollerMine.Reprogram"),

    # TODO: Episodic only
    sound("RagdollBoogie.Zap"),
    func=base_npc,
    )
res('npc_seagull',
    *BASE_NPC,
    mdl("models/seagull.mdl"),
    sound("NPC_Seagull.Idle"),
    sound("NPC_Seagull.Pain"),

    sound("NPC_Crow.Hop"),
    sound("NPC_Crow.Squawk"),
    sound("NPC_Crow.Gib"),
    sound("NPC_Crow.Flap"),
    func=base_npc,
    )

res('npc_sniper',
    mdl('models/combine_soldier.mdl'),
    mat('materials/sprites/light_glow03.vmt'),
    mat('materials/sprites/muzzleflash1.vmt'),
    mat('materials/effects/bluelaser1.vmt'),
    sound('NPC_Sniper.Die'),
    sound('NPC_Sniper.TargetDestroyed'),
    sound('NPC_Sniper.HearDange'),
    sound('NPC_Sniper.FireBullet'),
    sound('NPC_Sniper.Reload'),
    sound('NPC_Sniper.SonicBoom'),
    includes='sniperbullet',
    )


@cls_func
def npc_stalker(pack: PackList, ent: Entity) -> None:
    """Mapbase optionally allows blood particles."""
    base_npc(pack, ent)
    if conv_bool(ent['bleed']):  # Mapbase
        pack.pack_particle('blood_impact_synth_01')


res('npc_stalker',
    *BASE_NPC,
    mdl('models/stalker.mdl'),
    mat('materials/sprites/laser.vmt'),
    mat('materials/sprites/redglow1.vmt'),
    mat('materials/sprites/orangeglow1.vmt'),
    mat('materials/sprites/yellowglow1.vmt'),
    sound('NPC_Stalker.BurnFlesh'),
    sound('NPC_Stalker.BurnWall'),
    sound('NPC_Stalker.FootstepLeft'),
    sound('NPC_Stalker.FootstepRight'),
    sound('NPC_Stalker.Hit'),
    sound('NPC_Stalker.Ambient01'),
    sound('NPC_Stalker.Scream'),
    sound('NPC_Stalker.Pain'),
    sound('NPC_Stalker.Die'),
    )
res('npc_strider',
    *BASE_NPC,
    mdl("models/combine_strider.mdl"),
    sound("NPC_Strider.StriderBusterExplode"),
    sound("explode_5"),
    sound("NPC_Strider.Charge"),
    sound("NPC_Strider.RagdollDetach"),
    sound("NPC_Strider.Whoosh"),
    sound("NPC_Strider.Creak"),
    sound("NPC_Strider.Alert"),
    sound("NPC_Strider.Pain"),
    sound("NPC_Strider.Death"),
    sound("NPC_Strider.FireMinigun"),
    sound("NPC_Strider.Shoot"),
    sound("NPC_Strider.OpenHatch"),
    sound("NPC_Strider.Footstep"),
    sound("NPC_Strider.Skewer"),
    sound("NPC_Strider.Hunt"),
    mat("materials/effects/water_highlight.vmt"),
    mat("materials/sprites/physbeam.vmt"),
    mat("materials/sprites/bluelaser1"),
    mat("materials/effects/blueblacklargebeam"),
    mat("materials/effects/strider_pinch_dudv"),
    mat("materials/effects/blueblackflash"),
    mat("materials/effects/strider_bulge_dudv"),
    mat("materials/effects/strider_muzzle"),
    mdl("models/chefhat.mdl"),  # For some reason...
    includes="concussiveblast sparktrail",
    func=base_npc,
    )

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

res('npc_vehicledriver',
    *BASE_NPC,
    'models/roller_vehicledriver.mdl',
    func=base_npc,
    )
res('npc_vortigaunt',
    *BASE_NPC,
    mdl('models/vortigaunt.mdl'),  # Only if not set.
    mat('materials/sprites/lgtning.vmt'),
    mat('materials/sprites/vortring1.vmt'),
    mat('materials/sprites/light_glow02_add'),
    # EP2 only...
    mat('materials/effects/rollerglow.vmt'),
    sound('NPC_Vortigaunt.SuitOn'),
    sound('NPC_Vortigaunt.SuitCharge'),
    sound('NPC_Vortigaunt.ZapPowerup'),
    sound('NPC_Vortigaunt.ClawBeam'),
    sound('NPC_Vortigaunt.StartHealLoop'),
    sound('NPC_Vortigaunt.Swing'),
    sound('NPC_Vortigaunt.StartShootLoop'),
    sound('NPC_Vortigaunt.FootstepLeft'),
    sound('NPC_Vortigaunt.FootstepRight'),
    sound('NPC_Vortigaunt.DispelStart'),
    sound('NPC_Vortigaunt.DispelImpact'),
    sound('NPC_Vortigaunt.Explode'),

    part('vortigaunt_beam'),
    part('vortigaunt_beam_charge'),
    part('vortigaunt_hand_glow'),
    includes="vort_charge_token",
    func=base_npc,
    )

res('npc_wilson',
    *BASE_NPC,
    mdl('models/will_e.mdl'),
    mdl('models/will_e_damaged.mdl'),
    mat('materials/sprites/glow1.vmt'),
    sound('NPC_Wilson.Destruct'),
    sound('NPC_Combine.WeaponBash'),
    sound('RagdollBoogie.Zap'),
    part('explosion_turret_break'),
    choreo('scenes/npc/wilson/expression_idle.vcd'),
    choreo('scenes/npc/wilson/expression_alert.vcd'),
    choreo('scenes/npc/wilson/expression_combat.vcd'),
    choreo('scenes/npc/wilson/expression_dead.vcd'),
    choreo('scenes/npc/wilson/expression_scanning.vcd'),
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
