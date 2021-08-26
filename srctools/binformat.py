"""Common code for handling binary formats."""
from binascii import crc32
from struct import Struct
from typing import IO, List, Hashable, Union
from srctools import Vec

ST_VEC = Struct('fff')
SIZES = {
    fmt: Struct('<' + fmt).size
    for fmt in 'cbB?hHiIlLqQfd'
}
SIZE_CHAR = SIZES['b']
SIZE_SHORT = SIZES['h']
SIZE_INT = SIZES['i']
SIZE_LONG = SIZES['q']
SIZE_FLOAT = SIZES['f']
SIZE_DOUBLE = SIZES['d']

assert SIZE_CHAR == 1
assert SIZE_SHORT == 2
assert SIZE_INT == 4
assert SIZE_LONG == 8
assert SIZE_FLOAT == 4
assert SIZE_DOUBLE == 8


def struct_read(fmt: Union[Struct, str], file: IO[bytes]) -> tuple:
    """Read a structure from the file."""
    if not isinstance(fmt, Struct):
        fmt = Struct(fmt)
    return fmt.unpack(file.read(fmt.size))


def read_nullstr(file: IO[bytes], pos: int=None, encoding: str = 'ascii') -> str:
    """Read a null-terminated string from the file."""
    if pos is not None:
        if pos == 0:
            return ''
        file.seek(pos)

    text: list[bytes] = []
    while True:
        char = file.read(1)
        if char == b'\0':
            return b''.join(text).decode(encoding)
        if not char:
            raise ValueError('Fell off end of file!')
        text.append(char)


def read_nullstr_array(file: IO[bytes], count: int, encoding: str = 'ascii') -> List[str]:
    """Read consecutive null-terminated strings from the file."""
    arr = [''] * count
    if not count:
        return arr

    for i in range(count):
        arr[i] = read_nullstr(file, None, encoding)
    return arr


def read_offset_array(file: IO[bytes], count: int, encoding: str = 'ascii') -> List[str]:
    """Read an array of offsets to null-terminated strings from the file."""
    offsets = struct_read(str(count) + 'i', file)
    return [
        read_nullstr(file, off, encoding)
        for off in offsets
    ]


def read_array(fmt: str, data: bytes) -> List[int]:
    """Read a buffer containing a stream of integers."""
    if len(fmt) == 2:
        endianness, fmt = list(fmt)
    else:
        endianness = ''
    try:
        item_size = SIZES[fmt]
    except KeyError:
        raise ValueError(f'Unknown format character {fmt!r}!')
    count = len(data) // item_size
    return list(Struct(endianness + fmt * count).unpack_from(data))


def write_array(fmt: str, data: List[int]) -> bytes:
    """Build a packed array of integers."""
    if len(fmt) == 2:
        endianness = fmt[0]
        fmt = fmt[1]
    else:
        endianness = ''

    return Struct(endianness + fmt * len(data)).pack(*data)


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
        self.loc: dict[Hashable, tuple[int, Struct]] = {}
        # Then the bytes to write there.
        self.data: dict[Hashable, bytes] = {}

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
