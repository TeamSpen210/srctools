"""
The binformat module :mod:`binformat` contains functionality for handling binary formats, \
esentially expanding on :external:mod:`struct`'s functionality.

"""
from typing import IO, Any, Collection, Dict, Hashable, List, Mapping, Optional, Tuple, Union
from typing_extensions import Final
from binascii import crc32
from struct import Struct
import lzma

from srctools.math import Vec


__all__ = [
    'SIZES',
    'SIZE_CHAR', 'SIZE_DOUBLE', 'SIZE_FLOAT', 'SIZE_INT', 'SIZE_LONG', 'SIZE_SHORT',
    'struct_read', 'read_nullstr', 'read_nullstr_array', 'read_offset_array',
    'read_array', 'write_array',
    'str_readvec', 'ST_VEC',
    'checksum', 'EMPTY_CHECKSUM',
    'DeferredWrites',
    'compress_lzma', 'decompress_lzma',
]

ST_VEC: Final = Struct('fff')
SIZES: Final[Mapping[str, int]] = {
    fmt: Struct('<' + fmt).size
    for fmt in 'cbB?hHiIlLqQfd'
}
SIZE_CHAR: Final = 1
SIZE_SHORT: Final = 2
SIZE_INT: Final = 4
SIZE_LONG: Final = 8
SIZE_FLOAT: Final = 4
SIZE_DOUBLE: Final = 8

assert SIZE_CHAR == SIZES['b']
assert SIZE_SHORT == SIZES['h']
assert SIZE_INT == SIZES['i']
assert SIZE_LONG == SIZES['q']
assert SIZE_FLOAT == SIZES['f']
assert SIZE_DOUBLE == SIZES['d']

LZMA_DIC_MIN: Final = (1 << 12)
ST_LZMA_SOURCE: Final = Struct('<4sIIbI')
# The options Source seems to be using.
LZMA_FILT: Final = {
    'id': lzma.FILTER_LZMA1,
    'dict_size': 1 << 24,
    'lc': 3,
    'lp': 0,
    'pb': 2,
}


def struct_read(fmt: Union[Struct, str], file: IO[bytes]) -> Tuple[Any, ...]:
    """Read a structure from the file, automatically computing the required number of bytes."""
    if not isinstance(fmt, Struct):
        fmt = Struct(fmt)
    return fmt.unpack(file.read(fmt.size))


def read_nullstr(file: IO[bytes], pos: Optional[int] = None, encoding: str = 'ascii') -> str:
    """Read a null-terminated string from the file.

    If the position is ``0``, this will instead immediately return an empty string. If set to any
    other value this will first seek to the location.
    """
    if pos is not None:
        if pos == 0:
            return ''
        file.seek(pos)

    text: List[bytes] = []
    while True:
        char = file.read(1)
        if char == b'\0':
            return b''.join(text).decode(encoding)
        if not char:
            raise ValueError('Fell off end of file!')
        text.append(char)


def read_nullstr_array(file: IO[bytes], count: int, encoding: str = 'ascii') -> List[str]:
    """Read the specified number of consecutive null-terminated strings from a file.

    If the count is zero, no reading will be performed at all.
    """
    arr = [''] * count
    if not count:
        return arr

    for i in range(count):
        arr[i] = read_nullstr(file, None, encoding)
    return arr


def read_offset_array(file: IO[bytes], count: int, encoding: str = 'ascii') -> List[str]:
    """Read an array of offsets to null-terminated strings from the file.

    This first reads the specified number of signed integers, then seeks to those locations and
    reads a null-terminated string from each.
    """
    offsets = struct_read('<{0}i'.format(str(count)), file)
    pos = file.tell()
    arr = [
        read_nullstr(file, off, encoding)
        for off in offsets
    ]
    file.seek(pos)
    return arr


def read_array(fmt: str, data: bytes) -> List[int]:
    """Read a buffer containing a stream of integers.

    The format string should be one of the integer format characters, optionally prefixed by an
     endianness indicator. As many integers as possible will then be read from the data.
    """
    if len(fmt) == 2:
        endianness = fmt[0]
        fmt = fmt[1]
    else:
        endianness = ''
    try:
        item_size = SIZES[fmt]
    except KeyError:
        raise ValueError(f'Unknown format character {fmt!r}!')
    count = len(data) // item_size
    return list(Struct(endianness + fmt * count).unpack_from(data))


def write_array(fmt: str, data: Collection[int]) -> bytes:
    """Build a packed array of integers.

    The format string should be one of the integer format characters, optionally prefixed by an
    endianness indicator. The integers in the data will then be packed into a bytes buffer and returned.
     """
    if len(fmt) == 2:
        endianness = fmt[0]
        fmt = fmt[1]
    else:
        endianness = ''

    return Struct(endianness + fmt * len(data)).pack(*data)


def str_readvec(file: IO[bytes]) -> Vec:
    """Shortcut to read a 3-float vector from a file."""
    return Vec(ST_VEC.unpack(file.read(ST_VEC.size)))


def checksum(data: bytes, prior: int = 0) -> int:
    """Compute the VPK checksum for a file (CRC32).

    Pass a previous computation to allow continuing a previous checksum.
    """
    return crc32(data, prior)


EMPTY_CHECKSUM: Final[int] = checksum(b'')
"""CRC32 checksum of an empty bytes buffer."""


class DeferredWrites:
    """Several formats require offsets or similar data to be written referring to data later in the file.

    Doing this in one pass would be quite complicated, but this class assists in writing such files.
    Initially null bytes are written in the slots, then the data is filled in at the end.

    To use this class, initialise it with the open and seekable file. Call :py:func:`defer()` when
    reaching the relevant parts of the file, passing a format string for the structure and a hashable
    key used to identify it later. Once the value has been determined, call :py:func:`set_data()`
    to store the data. When the file is written out, call :py:func:`write()` which will :external:py:meth:`~io.IOBase.seek`
    back and fill in the values.
    """
    def __init__(self, file: IO[bytes]) -> None:
        self.file = file
        # Position to write to, and the struct format to use.
        self.loc: Dict[Hashable, Tuple[int, Struct]] = {}
        # Then the bytes to write there.
        self.data: Dict[Hashable, bytes] = {}

    def defer(self, key: Hashable, fmt: Union[str, Struct], write: bool = False) -> None:
        """Mark that data of the specified format is going to be written here.

        :param key: Any hashable object, used to identify this location later.
        :param fmt: The structure of the data, either as an existing :external:py:class:`struct.Struct`
            instance, or a string in the same format.
        :param write: If true, write null bytes to occupy the space in the file. Set to false
            if you are doing this yourself.
        """
        if isinstance(fmt, str):
            fmt = Struct(fmt)
        self.loc[key] = (self.file.tell(), fmt)
        if write:
            self.file.write(bytes(fmt.size))

    def set_data(self, key: Hashable, *data: Union[int, str, bytes, float]) -> None:
        """Specify the data for the given key.

        :param key: Any hashable object, used to identify the location and format.
        :param data: The values passed to :external:py:func:`struct.pack` to build the data.
        """
        off, fmt = self.loc[key]
        self.data[key] = packed = fmt.pack(*data)
        assert len(packed) == fmt.size

    def pos_of(self, key: Hashable) -> int:
        """Return the previously marked offset with the given key.

        :param key: Any hashable object, used to match this call to the earlier :py:func:`defer` call.
        """
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


def decompress_lzma(data: bytes) -> bytes:
    """Decompress LZMA-encoded data, using the settings Source uses in BSP lumps."""
    # We have to decode Source's header, then build LZMA's header to
    #     allow it to be compressed.
    if data[:4] != b'LZMA':
        return data  # Not compressed.
    (sig, uncomp_size, comp_size, props, dict_size) = ST_LZMA_SOURCE.unpack_from(data)
    assert sig == b'LZMA'
    # Don't check against the expected size, some TF2 maps just have extra null data randomly.
    # real_comp_size = len(data) - ST_LZMA_SOURCE.size
    # if real_comp_size < comp_size:
    #     raise ValueError(
    #         f"File size doesn't match. Got {real_comp_size:,} bytes, expected {comp_size:,} bytes"
    #     )

    # Parse properties - Code from LZMA spec
    if props >= (9 * 5 * 5):
        raise ValueError("Incorrect LZMA properties")
    lc = props % 9
    props //= 9
    pb = props // 5
    lp = props % 5
    if dict_size < LZMA_DIC_MIN:
        dict_size = LZMA_DIC_MIN

    filt = {
        'id': lzma.FILTER_LZMA1,
        'dict_size': dict_size,
        'lc': lc,
        'lp': lp,
        'pb': pb,
    }
    decomp = lzma.LZMADecompressor(lzma.FORMAT_RAW, None, filters=[filt])
    # This technically leaves the decompressor in an incomplete state, it's expecting an EOF marker.
    # Valve doesn't include that, so just leave it like that.
    # Use a memoryview to avoid copying the whole buffer.
    res = decomp.decompress(memoryview(data)[ST_LZMA_SOURCE.size:])

    # In some cases it seems to have an extra null byte.
    if len(res) > uncomp_size:
        return res[:uncomp_size]
    # Sometimes the data is truncated to a single null byte??
    elif len(res) < uncomp_size and res != b'\x00':
        print(
            f'Incorrect decompressed size. Got {len(res):,} bytes, expected {uncomp_size:,} bytes.'
        )
    return res


def compress_lzma(data: bytes) -> bytes:
    """Compress data using the LZMA algorithm and the settings Source uses in BSP lumps."""
    # We have to convert the standard LZMA header into Source's version.
    comp_data = lzma.compress(data, lzma.FORMAT_RAW, filters=[LZMA_FILT])

    # From LZMA spec.
    props = (LZMA_FILT['pb'] * 5 + LZMA_FILT['lp']) * 9 + LZMA_FILT['lc']

    # Build up the header.
    return ST_LZMA_SOURCE.pack(
        b'LZMA',  # Signature
        len(data),
        len(comp_data),
        props, LZMA_FILT['dict_size'],  # Filter options encoded together.
    ) + comp_data
