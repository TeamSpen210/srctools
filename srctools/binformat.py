"""Common code for handling binary formats."""
from binascii import crc32
from struct import unpack, calcsize, Struct
from typing import IO, List, Hashable, Dict, Tuple, Union
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


class DeferredWrites:
    """Several formats expect offsets to be written pointing to latter locations.

    This makes these easier to write, by initially writing null data, then returning to fill it later.
    The key can be any hashable type.
    """
    def __init__(self, file: IO[bytes]) -> None:
        self.file = file
        # Position to write to, and the struct format to use.
        self.loc: Dict[Hashable, Tuple[int, Struct]] = {}
        # Then the bytes to write there.
        self.data: Dict[Hashable, bytes] = {}

    def defer(self, key: Hashable, fmt: Union[str, Struct], write=False) -> None:
        """Mark that the given format data is going to be written here.

        If write is true, write null bytes.
        """
        if isinstance(fmt, str):
            fmt = Struct(fmt)
        self.loc[key] = (self.file.tell(), fmt)
        if write:
            self.file.write(bytes(fmt.size))

    def set_data(self, key: Hashable, *data: Union[int, str, bytes, float]):
        """Specify the data for the given key. Data is the same as pack()."""
        off, fmt = self.loc[key]
        self.data[key] = packed = fmt.pack(*data)
        assert len(packed) == fmt.size

    def pos_of(self, key: Hashable) -> int:
        """Return the previously marked offset with the given name."""
        off, fmt = self.loc[key]
        return off

    def write(self) -> None:
        """Write out all the data. All values should have been set."""
        prev_pos = self.file.tell()
        for key, (off, fmt) in self.loc.items():
            try:
                data = self.data.pop(key)
            except KeyError:
                raise ValueError(f'No data for key "{key}"!') from None
            self.file.seek(off)
            self.file.write(data)
        self.loc.clear()
        if self.data:
            raise ValueError(f'Data not specified for keys {list(self.loc)}!')
        self.file.seek(prev_pos)
