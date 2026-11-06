"""Test the filesystem module"""

from pytest_datadir.plugin import LazyDataDir

from srctools.filesys import RawFileSystem


def test_raw_systems(lazy_datadir: LazyDataDir) -> None:
    fsys = RawFileSystem(lazy_datadir / 'raw_root')

    with fsys.open_str('somefolder/CHILDfolder/tarGETfile.txt') as file:
        assert file.read() == 'Hello World!\n'
