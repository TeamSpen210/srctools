"""Test the VPK parser."""
from srctools import vpk


def test_ascii() -> None:
    """Test that VPKs only allow ascii filenames."""
    for i in range(128):
        assert vpk._check_is_ascii(chr(i) * 8)
    assert vpk._check_is_ascii('a_filename')
    assert not vpk._check_is_ascii('\x80'  * 4)
    # Special case, allow surrogate escape bytes too.
    for i in range(0xDC80, 0xDCFF + 1):
        assert vpk._check_is_ascii(chr(i) * 4)
