"""NPC entities."""
from srctools._class_resources import *


res('npc_advisor',
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


@cls_func
def npc_antlion(pack: PackList, ent: Entity):
    """Antlions require different resources for the worker version."""
    spawnflags = conv_int(ent['spawnflags'])
    if spawnflags & (1 << 18):  # Is worker?
        pack.pack_file("models/antlion_worker.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_worker_01", FileType.PARTICLE)
        pack.pack_file("antlion_gib_02", FileType.PARTICLE)
        pack.pack_file("blood_impact_yellow_01", FileType.PARTICLE)

        pack_ent_class(pack, 'grenade_spit')
    else:
        pack.pack_file("models/antlion.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_01")
        pack.pack_file("AntlionGib", FileType.PARTICLE)

    for i in ('1', '2', '3'):
        for size in ('small', 'medium', 'large'):
            pack.pack_file("models/gibs/antlion_gib_{}_{}.mdl".format(size, i), FileType.MODEL)

    pack.pack_soundscript("NPC_Antlion.RunOverByVehicle")
    pack.pack_soundscript("NPC_Antlion.MeleeAttack")
    pack.pack_soundscript("NPC_Antlion.Footstep")
    pack.pack_soundscript("NPC_Antlion.BurrowIn")
    pack.pack_soundscript("NPC_Antlion.BurrowOut")
    pack.pack_soundscript("NPC_Antlion.FootstepSoft")
    pack.pack_soundscript("NPC_Antlion.FootstepHeavy")
    pack.pack_soundscript("NPC_Antlion.MeleeAttackSingle")
    pack.pack_soundscript("NPC_Antlion.MeleeAttackDouble")
    pack.pack_soundscript("NPC_Antlion.Distracted")
    pack.pack_soundscript("NPC_Antlion.Idle")
    pack.pack_soundscript("NPC_Antlion.Pain")
    pack.pack_soundscript("NPC_Antlion.Land")
    pack.pack_soundscript("NPC_Antlion.WingsOpen")
    pack.pack_soundscript("NPC_Antlion.LoopingAgitated")
    pack.pack_soundscript("NPC_Antlion.Distracted")

    # TODO: These are Episodic only..
    pack.pack_soundscript("NPC_Antlion.PoisonBurstScream")
    pack.pack_soundscript("NPC_Antlion.PoisonBurstScreamSubmerged")
    pack.pack_soundscript("NPC_Antlion.PoisonBurstExplode")
    pack.pack_soundscript("NPC_Antlion.MeleeAttack_Muffled")
    pack.pack_soundscript("NPC_Antlion.TrappedMetal")
    pack.pack_soundscript("NPC_Antlion.ZappedFlip")
    pack.pack_soundscript("NPC_Antlion.PoisonShoot")
    pack.pack_soundscript("NPC_Antlion.PoisonBall")

res('npc_antlion_grub',
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
    )


@cls_func
def npc_antlion_template_maker(pack: PackList, ent: Entity):
    """Depending on KVs this may or may not spawn workers."""
    # There will be an antlion present in the map, as the template
    # NPC. So we don't need to add those resources.
    if conv_int(ent['workerspawnrate']) > 0:
        # It randomly spawns worker antlions, so load that resource set.
        pack.pack_file("models/antlion_worker.mdl", FileType.MODEL)
        pack.pack_file("blood_impact_antlion_worker_01", FileType.PARTICLE)
        pack.pack_file("antlion_gib_02", FileType.PARTICLE)
        pack.pack_file("blood_impact_yellow_01", FileType.PARTICLE)

        for fname, ftype in CLASS_RESOURCES['grenade_spit']:
            pack.pack_file(fname, ftype)
            
res('npc_alyx',
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
    )

res('npc_barnacle',
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
    )
res('npc_barnacle_tongue_tip', 'models/props_junk/rock001a.mdl')  # Random model it loads.
res('npc_barney',
    mdl("models/barney.mdl"),
    sound("NPC_Barney.FootstepLeft"),
    sound("NPC_Barney.FootstepRight"),
    sound("NPC_Barney.Die"),
    choreo("scenes/Expressions/BarneyIdle.vcd"),
    choreo("scenes/Expressions/BarneyAlert.vcd"),
    choreo("scenes/Expressions/BarneyCombat.vcd"),
    )
res('npc_breen', mdl("models/breen.mdl"))
res('npc_bullseye')
res('npc_cscanner',
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
    if ent['targetname'] == 'matt':
        # Special crowbar.
        pack.pack_file("models/props_canal/mattpipe.mdl", FileType.MODEL)

    pack.pack_soundscript("NPC_Citizen.FootstepLeft")
    pack.pack_soundscript("NPC_Citizen.FootstepRight")
    pack.pack_soundscript("NPC_Citizen.Die")

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
        return

    for head in CIT_HEADS:
        pack.pack_file('models/humans/{}/{}'.format(folder, head), FileType.MODEL)

res('npc_combine_camera',
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
    )


@cls_func
def npc_combinedropship(pack: PackList, ent: Entity) -> None:
    """The Combine Dropship may spawn with a variety of cargo types."""
    pack.pack_file("models/combine_dropship.mdl", FileType.MODEL)
    pack.pack_soundscript("NPC_CombineDropship.RotorLoop")
    pack.pack_soundscript("NPC_CombineDropship.FireLoop")
    pack.pack_soundscript("NPC_CombineDropship.NearRotorLoop")
    pack.pack_soundscript("NPC_CombineDropship.OnGroundRotorLoop")
    pack.pack_soundscript("NPC_CombineDropship.DescendingWarningLoop")
    pack.pack_soundscript("NPC_CombineDropship.NearRotorLoop")

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
    if conv_int(ent['spawnflags']) & (1 << 13):
        pack.pack_file("models/combine_helicopter.mdl", FileType.MODEL)
        pack.pack_file("models/combine_helicopter_broken.mdl", FileType.MODEL)
        pack_ent_class(pack, 'helicopter_chunk')
    else:
        pack.pack_file("models/gunship.mdl", FileType.MODEL)

    pack.pack_file("materials/sprites/lgtning.vmt", FileType.MATERIAL)
    pack.pack_file("materials/effects/ar2ground2", FileType.MATERIAL)
    pack.pack_file("materials/effects/blueblackflash", FileType.MATERIAL)

    pack.pack_soundscript("NPC_CombineGunship.SearchPing")
    pack.pack_soundscript("NPC_CombineGunship.PatrolPing")
    pack.pack_soundscript("NPC_Strider.Charge")
    pack.pack_soundscript("NPC_Strider.Shoot")
    pack.pack_soundscript("NPC_CombineGunship.SeeEnemy")
    pack.pack_soundscript("NPC_CombineGunship.CannonStartSound")
    pack.pack_soundscript("NPC_CombineGunship.Explode")
    pack.pack_soundscript("NPC_CombineGunship.Pain")
    pack.pack_soundscript("NPC_CombineGunship.CannonStopSound")

    pack.pack_soundscript("NPC_CombineGunship.DyingSound")
    pack.pack_soundscript("NPC_CombineGunship.CannonSound")
    pack.pack_soundscript("NPC_CombineGunship.RotorSound")
    pack.pack_soundscript("NPC_CombineGunship.ExhaustSound")
    pack.pack_soundscript("NPC_CombineGunship.RotorBlastSound")

    # TODO: if Episodic only
    pack_ent_class(pack, 'env_citadel_energy_core')
    pack.pack_file("materials/sprites/physbeam.vmt", FileType.MATERIAL)

res('npc_combine_cannon',
    mdl('models/combine_soldier.mdl'),
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    sound('NPC_Combine_Cannon.FireBullet'),
    )
res('npc_clawscanner',
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
    )
res('npc_crow',
    mdl("models/crow.mdl"),
    sound("NPC_Crow.Hop"),
    sound("NPC_Crow.Squawk"),
    sound("NPC_Crow.Gib"),
    sound("NPC_Crow.Idle"),
    sound("NPC_Crow.Alert"),
    sound("NPC_Crow.Die"),
    sound("NPC_Crow.Pain"),
    sound("NPC_Crow.Flap"),
    )
res('npc_dog',
    mdl("models/dog.mdl"),
    sound("Weapon_PhysCannon.Launch"),
    mat("materials/sprites/orangelight1.vmt"),
    mat("materials/sprites/physcannon_bluelight2.vmt"),
    mat("materials/sprites/glow04_noz.vmt"),
    )
res('npc_eli', mdl("models/eli.mdl"))
res('npc_grenade_bugbait',
    mdl("models/weapons/w_bugbait.mdl"),
    sound("GrenadeBugBait.Splat"),
    )
res('npc_grenade_frag',
    mdl("models/Weapons/w_grenade.mdl"),
    mat('materials/sprites/redglow1.vmt'),
    mat('materials/sprites/bluelaser1.vmt'),
    sound("Grenade.Blip"),
    )

res('npc_fastzombie',
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
    )
# Actually an alias, but we don't want to swap these.
CLASS_RESOURCES['npc_fastzombie_torso'] = CLASS_RESOURCES['npc_fastzombie']

res('npc_fisherman',
    mdl("models/lostcoast/fisherman/fisherman.mdl"),
    sound("NPC_Fisherman.FootstepLeft"),
    sound("NPC_Fisherman.FootstepRight"),
    sound("NPC_Fisherman.Die"),
    choreo("scenes/Expressions/FishermanIdle.vcd"),
    choreo("scenes/Expressions/FishermanAlert.vcd"),
    choreo("scenes/Expressions/FishermanCombat.vcd"),
    )

res('npc_headcrab',
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
    )

res('npc_headcrab_black',
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
    )

res('npc_headcrab_fast',
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
    )

res('npc_heli_avoidbox')
res('npc_heli_avoidsphere')
res('npc_heli_nobomb')
res('npc_helicopter',
    mdl("models/combine_helicopter.mdl"),
    mdl("models/combine_helicopter_broken.mdl"),
    mat("materials/sprites/redglow1.vmt"),
    includes='helicopter_chunk grenade_helicopter',
    )
res('npc_helicoptersensor')
res('npc_hunter',
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
    )
res('npc_manhack',
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
    )


@cls_func
def npc_metropolice(pack: PackList, ent: Entity) -> None:
    """If a spawnflag is set, a cheap model is used."""
    if conv_int(ent['spawnflags']) & 5:
        pack.pack_file("models/police_cheaple.mdl", FileType.MODEL)
    else:
        pack.pack_file("models/police.mdl", FileType.MODEL)
    pack.pack_soundscript("NPC_Metropolice.Shove")
    pack.pack_soundscript("NPC_MetroPolice.WaterSpeech")
    pack.pack_soundscript("NPC_MetroPolice.HidingSpeech")
    # TODO: pack.pack_sentence_group("METROPOLICE")

res('npc_pigeon',
    mdl("models/pigeon.mdl"),
    sound("NPC_Pigeon.Idle"),

    sound("NPC_Crow.Hop"),
    sound("NPC_Crow.Squawk"),
    sound("NPC_Crow.Gib"),
    sound("NPC_Crow.Pain"),
    sound("NPC_Crow.Die"),
    )

res('npc_rocket_turret',
    mat('materials/effects/bluelaser1.vmt'),
    mat('materials/sprites/light_glow03.vmt'),
    mdl('models/props_bts/rocket_sentry.mdl'),
    sound('NPC_RocketTurret.LockingBeep'),
    sound('NPC_FloorTurret.LockedBeep'),
    sound('NPC_FloorTurret.RocketFire'),
    includes='rocket_turret_projectile',
    )
res('npc_rollermine',
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
    )
res('npc_seagull',
    mdl("models/seagull.mdl"),
    sound("NPC_Seagull.Idle"),
    sound("NPC_Seagull.Pain"),

    sound("NPC_Crow.Hop"),
    sound("NPC_Crow.Squawk"),
    sound("NPC_Crow.Gib"),
    sound("NPC_Crow.Flap"),
    )
res('npc_strider',
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
    )

res('npc_turret_ceiling',
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
    )

res('npc_turret_ground',
    mdl('models/combine_turrets/ground_turret.mdl'),
    mat('materials/effects/bluelaser2.vmt'),
    sound('NPC_CeilingTurret.Deploy'),
    sound('NPC_FloorTurret.ShotSounds'),
    sound('NPC_FloorTurret.Die'),
    sound('NPC_FloorTurret.Ping'),
    sound('DoSpark'),
    )

res('npc_turret_floor',
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
    )

res('npc_turret_lab',
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
    )

res('npc_vehicledriver',
    'models/roller_vehicledriver.mdl',
    )

res('npc_zombie',
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
    )
# Actually an alias, but we don't want to swap these.
CLASS_RESOURCES['npc_zombie_torso'] = CLASS_RESOURCES['npc_zombie']
res('npc_zombine',
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
    )
