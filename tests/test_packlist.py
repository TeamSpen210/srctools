"""Test packlist logic."""
from dirty_equals import IsList

from srctools.vmf import VMF
from srctools.const import FileType
from srctools.filesys import FileSystemChain
from srctools.packlist import strip_extension, PackList


def test_strip_extensions() -> None:
    """Test extension stripping."""
    assert strip_extension('filename/file.txt') == 'filename/file'
    assert strip_extension('directory/../file') == 'directory/../file'
    assert strip_extension('directory/../file.extension') == 'directory/../file'
    assert strip_extension('directory.dotted/filename') == 'directory.dotted/filename'


# -------------------------------------------------------
# Test packing, by using various representative entities.
# -------------------------------------------------------

def packing_test(vmf: VMF) -> tuple[PackList, list[tuple[FileType, str]]]:
    """Pack ents in a VMF, then return the files for checking packing logic."""
    packlist = PackList(FileSystemChain())
    packlist.pack_from_ents(vmf, 'some_map')
    return packlist, [(file.type, file.filename) for file in packlist]


def test_valtype_basic() -> None:
    """Test npc_sniper, to check various basic types with no special behaviour."""
    vmf = VMF()
    vmf.create_ent(
        'npc_sniper',
        targetname='a_sniper',  # target_source
        angles='0 90 10',  # Angle
        radius=3.8,  # Float
        spawnflags=1024,  # Spawnflags
        beambrightness=15,  # Int
        shootzombiesinchest=True,  # Boolean
        beamcolor='255 100 125',  # Color255
        relationship='npc_alyx D_LI 9',  # String
        enemyfilter='some_filter',  # filterclass
        lightingorigin='sniper_light',  # target_destination
        velocity='0 0 50',  # Vector
    )
    packlist, files = packing_test(vmf)
    assert files == IsList(  # These are all npc_sniper default resources.
        (FileType.MATERIAL, 'materials/effects/bluelaser1.vmt'),
        (FileType.MATERIAL, 'materials/sprites/muzzleflash1.vmt'),
        (FileType.MATERIAL, 'materials/sprites/light_glow03.vmt'),
        (FileType.MODEL, 'models/combine_soldier.mdl'),
        check_order=False,
    )


def test_valtype_choreo() -> None:
    """Test logic_choreographed_scene, for the CHOREO types."""
    vmf = VMF()
    vmf.create_ent(
        'logic_choreographed_scene',
        scenefile='character/player/talk.vcd',
        resumescenefile='generic/swap.vcd',
    )
    packlist, files = packing_test(vmf)
    assert files == IsList(
        (FileType.CHOREO, 'character/player/talk.vcd'),
        (FileType.CHOREO, 'generic/swap.vcd'),
        check_order=False,
    )


def test_valtype_model_sound() -> None:
    """Test prop_physics, for the MODEL and SOUND types."""
    vmf = VMF()
    vmf.create_ent(
        'prop_physics',
        model="models/props/magic.mdl",
        puntsound="#)explosion.wav",
    )
    packlist, files = packing_test(vmf)
    assert (FileType.MODEL, "models/props/magic.mdl") in files, list(packlist)
    assert (FileType.RAW_SOUND, "sound/explosion.wav") in files, list(packlist)


def test_valtype_texture() -> None:
    """Test env_projectedtexture, for the TEXTURE type."""
    vmf = VMF()
    vmf.create_ent(
        'env_projectedtexture',
        texturename="effects/FLashlight001",
    )
    packlist, files = packing_test(vmf)
    assert (FileType.TEXTURE, "materials/effects/flashlight001.vtf") in files, list(packlist)


def test_valtype_material() -> None:
    """Test move_rope, for the TEXTURE type."""
    vmf = VMF()
    vmf.create_ent(
        'move_rope',
        ropematerial="cable/chain",
    )
    packlist, files = packing_test(vmf)
    assert (FileType.MATERIAL, "materials/cable/chain.vmt") in files, list(packlist)


def test_valtype_sprite() -> None:
    """Test env_beam, for the SPRITE type."""
    vmf = VMF()
    vmf.create_ent(
        'env_beam',
        rendercolor="0 200 200",
        BoltWidth="6",
        texture="folder/FANCY_beam.vmt",
    )
    vmf.create_ent(
        'env_beam',
        rendercolor="0 200 200",
        BoltWidth="6",
        texture="folder/LEGacy_sprite.spr",
    )
    packlist, files = packing_test(vmf)
    assert (FileType.MATERIAL, "materials/folder/fancy_beam.vmt") in files, list(packlist)
    assert (FileType.MATERIAL, "materials/sprites/folder/legacy_sprite.vmt") in files, list(packlist)


def test_valtype_decal() -> None:
    """Test infodecal, for the DECAL type."""
    vmf = VMF()
    vmf.create_ent(
        'infodecal',
        texture="decals/decal_crater001a",
    )
    packlist, files = packing_test(vmf)
    assert files == IsList(
        (FileType.MATERIAL, "materials/decals/decal_crater001a.vmt"),
        check_order=False,
    )
