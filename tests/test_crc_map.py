"""Test the special CRC mapping, used for choreo and subtitles."""
import pytest

# noinspection PyProtectedMember
from srctools._crc_map import ChecksumMap
from srctools.binformat import checksum


def test_direct_keys() -> None:
    mapping = ChecksumMap()
    assert len(mapping) == 0
    mapping['hi'] = 45
    assert mapping['hi'] == 45
    with pytest.raises(KeyError):
        mapping['missing']  # noqa


def test_checksum_keys() -> None:
    mapping = ChecksumMap()
    check = checksum('test key'.encode('utf8'))

    mapping[check] = 'result'
    # Present but key not known.
    assert len(mapping) == 1
    assert list(mapping.items()) == []

    found = mapping['test key']
    assert found == 'result'
    # Key now known.
    assert len(mapping) == 1
    assert list(mapping.items()) == [('test key', 'result')]
