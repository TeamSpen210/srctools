"""Common code for handling binary formats."""
from binascii import crc32
from struct import unpack, calcsize, Struct
from typing import IO, List
from srctools import Vec

ST_VEC = Struct('fff')


def struct_read(fmt: str, file: IO[bytes]) -> tuple:
    """Read a structure from the file."""
    return unpack(fmt, file.read(calcsize(fmt)))


def read_nullstr(file: IO[bytes], pos: int=None) -> str:
    """Read a null-terminated string from the file."""
    if pos is not None:
        if pos == 0:
            return ''
        file.seek(pos)

    text = []  # type: List[bytes]
    while True:
        char = file.read(1)
        if char == b'\0':
            return b''.join(text).decode('ascii')
        if not char:
            raise ValueError('Fell off end of file!')
        text.append(char)


def read_nullstr_array(file: IO[bytes], count: int) -> List[str]:
    """Read consecutive null-terminated strings from the file."""
    arr = [''] * count
    if not count:
        return arr

    for i in range(count):
        arr[i] = read_nullstr(file)
    return arr


def read_offset_array(file: IO[bytes], count: int) -> List[str]:
    """Read an array of offsets to null-terminated strings from the file."""
    cdmat_offsets = struct_read(str(count) + 'i', file)
    arr = [''] * count

    for ind, off in enumerate(cdmat_offsets):
        file.seek(off)
        arr[ind] = read_nullstr(file)
    return arr


def str_readvec(file: IO[bytes]) -> Vec:
    """Read a vector from a file."""
    return Vec(ST_VEC.unpack(file.read(ST_VEC.size)))


def checksum(data: bytes, prior=0) -> int:
    """Compute the VPK checksum for a file.

    Passing a previous computation to allow calculating
    repeatedly.
    """
    return crc32(data, prior)


EMPTY_CHECKSUM = checksum(b'')  # Checksum of empty bytes - 0.
