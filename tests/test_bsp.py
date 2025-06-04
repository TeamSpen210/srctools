"""Test the BSP parser's functionality."""
from random import Random
import unittest.mock

import pytest

from srctools import Vec
from srctools.bsp import (
    BSP, BSP_LUMPS, VERSIONS, GameLump, GameVersion, Lump, runlength_decode,
    runlength_encode, PlaneType, Plane
)


def make_dummy() -> BSP:
    """Create a totally empty BSP object, so lump functions can be called."""
    with unittest.mock.patch('srctools.bsp.BSP.read'):
        bsp = BSP('<dummy>')
    # Set arbitary dummy values.
    bsp.version = VERSIONS.HL2
    bsp.game_ver = GameVersion.NORMAL
    bsp.lumps = {
        lump: Lump(lump, 1)
        for lump in BSP_LUMPS
    }
    bsp.game_lumps = {
        lump: GameLump(lump, 0, 1)
        for lump in [b'sprp', b'dprp']
    }
    return bsp


# Run-length encoding used by PVS/PAS lumps.
RLE_SAMPLES = [
    # Simple case.
    (bytes([0b11001101, 0, 4, 0b10110101]), bytes([0b11001101, 0, 0, 0, 0, 0b10110101])),
    # Edge case - over 255 zeros in a row.
    (bytes([0x28, 0, 255, 0, 18, 0x4E]), bytes([0x28, *[0] * (255+18), 0x4E])),
    # Check for the zero being at the end of the array when encoding.
    (bytes([0x34, 0, 12]), bytes([0x34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    # And at the start.
    (bytes([0, 9, 0x8D]), bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0x8D])),
]
RLE_IDS = [
    'simple', 'multibyte_zeros', 'zero_end', 'zero_start',
]


@pytest.mark.parametrize('compressed, uncompressed', RLE_SAMPLES, ids=RLE_IDS)
def test_pvs_runlength_decode(compressed: bytes, uncompressed: bytes) -> None:
    """Test run-length decoding produces the correct uncompressed data."""
    assert runlength_decode(compressed) == uncompressed


@pytest.mark.parametrize('compressed, uncompressed', RLE_SAMPLES, ids=RLE_IDS)
def test_pvs_runlength_encode(compressed: bytes, uncompressed: bytes) -> None:
    """Test run-length encoding produces the correct compressed data."""
    assert runlength_encode(uncompressed) == compressed


# Use some seeds to make this reproducible.
@pytest.mark.parametrize('seed', [5332, 750, 7678, 2713])
def test_pvs_runlength_roundtrip(seed: int) -> None:
    """Test RLE against a randomly generated set of bytes, by encoding then decoding data."""
    rand = Random(seed)
    # AND three random binary values together to make most of them zero.
    data = (
        rand.getrandbits(8192) & rand.getrandbits(8192) & rand.getrandbits(8192)
    ).to_bytes(1024, 'little')

    comp = runlength_encode(data)
    reconstruct = runlength_decode(comp)
    assert reconstruct == data


def test_plane_type_calculation() -> None:
    """Planes have a type field which will be lazily recalculated."""
    assert PlaneType.from_normal(Vec(1, 0, 0)) is PlaneType.X
    assert PlaneType.from_normal(Vec(-1, 0, 0)) is PlaneType.X

    assert PlaneType.from_normal(Vec(0, 1,0)) is PlaneType.Y
    assert PlaneType.from_normal(Vec(0, -1, 0)) is PlaneType.Y

    assert PlaneType.from_normal(Vec(0, 0, 1)) is PlaneType.Z
    assert PlaneType.from_normal(Vec(0, 0, -1)) is PlaneType.Z

    assert PlaneType.from_normal(Vec(-0.6, -0.5, 0.5)) is PlaneType.ANY_X
    assert PlaneType.from_normal(Vec(0.5, 0.6, -0.5)) is PlaneType.ANY_Y
    assert PlaneType.from_normal(Vec(0.2, 0.6, -0.7)) is PlaneType.ANY_Z


def test_plane_type_invalidation() -> None:
    """Planes have a type field which will be lazily recalculated."""
    plane = Plane(Vec(1, 0, 0), 32)
    assert plane.type is PlaneType.X
    # Mypy doesn't know the assignment invalidates narrowing.
    plane.normal = Vec(0, 1, 0)
    assert plane.type is PlaneType.Y  # type: ignore[comparison-overlap]
    plane.type = PlaneType.ANY_Z
    assert plane.type is PlaneType.ANY_Z
    plane.normal = Vec(0, -0.9, 0.1)
    assert plane.type is PlaneType.ANY_Y
