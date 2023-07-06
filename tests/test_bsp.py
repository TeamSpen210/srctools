"""Test the BSP parser's functionality."""
from random import Random
import unittest.mock

import pytest

from srctools.bsp import (
    BSP, BSP_LUMPS, VERSIONS, GameLump, GameVersion, Lump, _find_or_extend, _find_or_insert,
    runlength_decode, runlength_encode,
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


@pytest.mark.parametrize('original, item, index', [
    ([1, 2, 3, 4], 3, 2),
    ([1, 2, 4, 38], 12, 4),
    ([3, 4, 5, 4], 4, 3),
])
def test_find_or_insert(original: list, item, index: int) -> None:
    """Test the find-or-insert helper function correctly inserts values."""
    array = original.copy()
    finder = _find_or_insert(array, lambda x: -x)
    assert finder(item) == index
    assert finder(item) == index  # Doesn't repeat.
    assert array[index] == item  # And put it in that spot.
    # But the original is the same.
    assert array[:len(original)] == original


@pytest.mark.parametrize('original, subset, start', [
    (['a', 'b', 'c', 'd', 'e'], ['c', 'd'], 2),
    (['a', 'b', 'c'], ['j', 'k'], 3),
    (['a', 'c', 'e', 'f', 'g', 'c', 'e', 'a', 'k'], ['c', 'e', 'a'], 5),
])
def test_find_or_extend(original: list, subset: list, start: int) -> None:
    """Test the find-or-extend helper function correctly inserts a subset."""
    array = original.copy()
    finder = _find_or_extend(array, str.swapcase)

    assert finder(subset) == start
    assert finder(subset) == start  # Doesn't repeat.
    assert array[start: start + len(subset)] == subset  # And put it in that spot.
    # But the original is the same.
    assert array[:len(original)] == original


def test_find_or_insert_repeat() -> None:
    """Test repeatedly inserting is still valid."""
    lst = [4, 5, 8]
    finder = _find_or_insert(lst, lambda x: x**2)

    assert finder(5) == 1
    assert finder(12) == 3
    assert lst == [4, 5, 8, 12]
    assert finder(38) == 4
    assert lst == [4, 5, 8, 12, 38]
    assert finder(8) == 2
    assert finder(200) == 5
    assert lst == [4, 5, 8, 12, 38, 200]


def test_find_or_extend_repeat() -> None:
    """asset repeatedly extending is still valid."""
    lst = [4, 8, 20, 12, -50, 3, 4, 12]
    finder = _find_or_extend(lst, lambda x: x-4)
    assert finder([20, 12]) == 2
    assert finder([12, -50, 3]) == 3
    assert finder([4, 12]) == 6

    assert finder([14, 12]) == 8
    assert finder([14, 12]) == 8
    assert lst == [4, 8, 20, 12, -50, 3, 4, 12, 14, 12]
    assert finder([3, 4]) == 5
    assert finder([12, 14, 12]) == 7
    assert finder([1, 2, 3, 4, 8, 63]) == 10
    assert finder([3, 4, 8]) == 12

    assert finder([-50, 3, 4, 12, 14]) == 4
    assert finder([1, 2]) == 10
    assert finder([4, 8]) == 0
    assert finder([20, 12]) == 2
    assert lst == [4, 8, 20, 12, -50, 3, 4, 12, 14, 12, 1, 2, 3, 4, 8, 63]
    assert finder([14, 12, 2]) == 16


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
