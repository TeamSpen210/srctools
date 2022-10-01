"""Test packlist logic."""
from srctools.packlist import strip_extension


def test_strip_extensions() -> None:
    """Test extension stripping."""
    assert strip_extension('filename/file.txt') == 'filename/file'
    assert strip_extension('directory/../file') == 'directory/../file'
    assert strip_extension('directory/../file.extension') == 'directory/../file'
    assert strip_extension('directory.dotted/filename') == 'directory.dotted/filename'
