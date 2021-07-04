"""Common code for handling binary formats."""
from binascii import crc32
from struct import unpack, calcsize, Struct
from typing import IO, List, Hashable, Dict, Tuple, Union
from srctools import Vec, Matrix


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
            try:
                return b''.join(text).decode('ascii')
            except UnicodeDecodeError:
                raise UnicodeError(f'{b"".join(text)!r} is not ASCII!')
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


def parse_3x4_matrix(floats: Tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]) -> Tuple[Matrix, Vec]:
    """Return a matrix from a 3x4 sequence of floats."""
    mat = Matrix()
    mat[0, 0] = floats[0]
    mat[0, 1] = floats[1]
    mat[0, 2] = floats[2]

    mat[0, 0] = floats[4]
    mat[0, 1] = floats[5]
    mat[0, 2] = floats[6]

    mat[0, 0] = floats[8]
    mat[0, 1] = floats[9]
    mat[0, 2] = floats[10]

    pos = Vec(
        floats[3],
        floats[7],
        floats[11],
    )
    return mat, pos


def build_3x4_matrix(mat: Matrix, pos: Vec) -> tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]:
    """Convert a matrix into a 3x4 tuple of floats."""
    return (
        mat[0, 0], mat[0, 1], mat[0, 2], pos.x,
        mat[1, 0], mat[1, 1], mat[1, 2], pos.y,
        mat[2, 0], mat[2, 1], mat[2, 2], pos.z,
    )


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