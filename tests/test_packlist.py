"""Test packlist logic."""
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


def test_valtype_model() -> None:
    """Test prop_dynamic, for the MODEL type."""
    vmf = VMF()
    vmf.create_ent(
        'prop_dynamic',
        model="models/props/magic.mdl",
    )
    packlist, files = packing_test(vmf)
    assert (FileType.MODEL, "models/props/magic.mdl") in files, list(packlist)


def test_valtype_texture() -> None:
    """Test env_projectedtexture, for the TEXTURE type."""
    vmf = VMF()
    vmf.create_ent(
        'env_projectedtexture',
        texturename="effects/FLashlight001",
    )
    packlist, files = packing_test(vmf)
    assert (FileType.TEXTURE, "materials/effects/flashlight001.vtf") in files, list(packlist)


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
    assert (FileType.MATERIAL, "materials/decals/decal_crater001a.vmt") in files, list(packlist)
