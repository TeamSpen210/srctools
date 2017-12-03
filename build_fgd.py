"""Compile the files in the fgd/ folder into a binary blob."""
from srctools import FGD
from srctools.filesys import RawFileSystem

from lzma import LZMAFile

fgd = FGD()

with RawFileSystem('fgd/') as fs:
    for file in fs:
        fgd.parse_file(fs, file)

with open('fgd.bfgd.lzma', 'wb') as f:
    with LZMAFile(f, mode='w') as cf:
        fgd.serialise(cf)
