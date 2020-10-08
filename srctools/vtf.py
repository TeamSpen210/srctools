"""Reads VTF image data into a PIL object.

After this is imported, the imghdr module can recoginise
VTF images (returning 'source_vtf').
"""
from array import array
from enum import Enum, Flag
from collections import namedtuple
import itertools
import math
import struct
import warnings

from srctools import Vec

from typing import (
    IO, Dict, List, Optional, Tuple, Iterable, Union,
    TYPE_CHECKING, Type, Collection,
)

# Only import while type checking, so these expensive libraries are only loaded
# if the user used them elsewhere.
if TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image
    import tkinter

# A little dance to import both the Cython and Python versions,
# and choose an appropriate unprefixed version.
# For type-checking purposes make it think the Cython version is the Python one.

# noinspection PyProtectedMember
from srctools import _py_vtf_readwrite as _py_format_funcs
_cy_format_funcs = _format_funcs = _py_format_funcs

if not TYPE_CHECKING:
    try:
        # noinspection PyUnresolvedReferences, PyProtectedMember
        from srctools import _cy_vtf_readwrite as _cy_format_funcs  # type: ignore
        _format_funcs = _cy_format_funcs  # type: ignore
    except ImportError:
        pass


# The _vtf_readwrite module contains save/load functions which
# convert the VTF data into a uniform 32-bit RGBA block, which we can then
# parse.
# That works for everything except RGBA16161616 (used for HDR cubemaps), which
# is 16-bit for each channel. We can't do much about that.

__all__ = [
    'VTF', 'Frame', 'FilterMode',
    'ResourceID', 'CubeSide', 'ImageFormats', 'VTFFlags',
]


class CubeSide(Enum):
    """The sides of a cubemap texture."""
    RIGHT = 0
    LEFT = 1
    BACK = 2
    FRONT = 3
    UP = 4
    DOWN = 5
    SPHERE = 6


CUBES_WITH_SPHERE: Collection[CubeSide] = list(CubeSide)
# Remove the Sphere type, for 7.5+
CUBES: Collection[CubeSide] = CUBES_WITH_SPHERE[:-1]

# One black, opaque pixel for creating blank images.
_BLANK_PIXEL = array('B', [0, 0, 0, 0xFF])

class ImageAlignment(namedtuple("ImageAlignment", 'r g b a size index')):
    """Raw image mode, pixel counts or object(), bytes per pixel."""
    # Force object-style comparisons, so formats with the same counts
    # compare different.
    __gt__ = object.__gt__
    __lt__ = object.__lt__
    __ge__ = object.__ge__
    __le__ = object.__le__
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__

_ind = -1


def f(r=0, g=0, b=0, a=0, *, l=0, size=0):
    """Helper function to construct ImageFormats."""
    global _ind
    if l:
        r = g = b = l
        size = l + a
    if not size:
        size = r + g + b + a
    _ind += 1

    return r, g, b, a, size, _ind


class ImageFormats(ImageAlignment, Enum):
    """All VTF image formats, with their data sizes in the value."""
    RGBA8888 = f(8, 8, 8, 8)
    ABGR8888 = f(8, 8, 8, 8)
    RGB888 = f(8, 8, 8, 0)
    BGR888 = f(8, 8, 8)
    RGB565 = f(5, 6, 5, 0)
    I8 = f(a=0, l=8)
    IA88 = f(a=8, l=8)
    P8 = f()  # Using a palette somehow - was never implemented by Valve.
    A8 = f(a=8)
    # Blue = alpha channel too
    RGB888_BLUESCREEN = f(8, 8, 8)
    BGR888_BLUESCREEN = f(8, 8, 8)
    ARGB8888 = f(8, 8, 8, 8)
    BGRA8888 = f(8, 8, 8, 8)
    DXT1 = f(size=64)
    DXT3 = f(size=128)
    DXT5 = f(size=128)
    BGRX8888 = f(8, 8, 8, 8)
    BGR565 = f(5, 6, 5)
    BGRX5551 = f(5, 5, 5, 1)
    BGRA4444 = f(4, 4, 4, 4)
    DXT1_ONEBITALPHA = f(size=64)
    BGRA5551 = f(5, 5, 5, 1)
    UV88 = f(size=16)
    UVWQ8888 = f(size=32)
    RGBA16161616F = f(16, 16, 16, 16)
    RGBA16161616 = f(16, 16, 16, 16)
    UVLX8888 = f(size=32)

    NONE = f()
    # These two aren't supported by VTEX & VTFEdit, but are by the engine.
    # They're useful for normal maps.
    ATI1N = f(size=64)
    ATI2N = f(size=128)

    @property
    def is_compressed(self) -> bool:
        """Checks if the format is compressed in 4x4 blocks."""
        return self.name.startswith('DXT') or self.name in ('ATI1N', 'ATI2N')

    def frame_size(self, width: int, height: int) -> int:
        """Compute the number of bytes needed for this image size."""
        if self.name == 'NONE':
            return 0
        if self.is_compressed:
            block_wid, mod = divmod(width, 4)
            if mod:
                block_wid += 1

            block_height, mod = divmod(height, 4)
            if mod:
                block_height += 1
            return self.size * block_wid * block_height // 8
        else:
            return self.size * width * height // 8

del f, _ind
# Initialise the internal mapping in the format modules.
_format_funcs.init(ImageFormats)
if _cy_format_funcs is not _py_format_funcs:
    _py_format_funcs.init(ImageFormats)


FORMAT_ORDER = {
    fmt.ind: fmt
    for fmt in ImageFormats.__members__.values()
    if fmt.name not in ('NONE', 'ATI1N', 'ATI2N')
}
# Since these are semi-"internal" formats, the position has changed
# in the enum. They're either 37 in 2013, or 34 in ASW+.
# They're backward because why not.
FORMAT_ORDER[34] = FORMAT_ORDER[37] = ImageFormats.ATI2N
FORMAT_ORDER[35] = FORMAT_ORDER[38] = ImageFormats.ATI1N


class VTFFlags(Flag):
    """The various image flags that may be set."""
    EMPTY = 0
    # Flags from the *.txt config file
    POINT_SAMPLE = 0x00000001
    TRILINEAR = 0x00000002
    CLAMP_S = 0x00000004
    CLAMP_T = 0x00000008
    ANISOTROPIC = 0x00000010
    HINT_DXT5 = 0x00000020
    PWL_CORRECTED = 0x00000040
    NORMAL = 0x00000080
    NO_MIP = 0x00000100
    NO_LOD = 0x00000200
    ALL_MIPS = 0x00000400
    PROCEDURAL = 0x00000800

    # These are automatically generated by vtex from the texture data.
    ONEBITALPHA = 0x00001000
    EIGHTBITALPHA = 0x00002000

    # Newer flags from the *.txt config file
    ENVMAP = 0x00004000
    RENDER_TARGET = 0x00008000
    DEPTH_RENDER_TARGET = 0x00010000
    NO_DEBUG_OVERRIDE = 0x00020000
    SINGLE_COPY = 0x00040000
    PRE_SRGB = 0x00080000

    NO_DEPTH_BUFFER = 0x00800000

    CLAMP_U = 0x02000000
    VERTEX_TEXTURE = 0x04000000
    SS_BUMP = 0x08000000
    BORDER = 0x20000000


class ResourceID(bytes, Enum):
    """For VTF format 7.3+, there is an extensible resource system."""
    # The two data parts in earlier versions.
    LOW_RES = b'\x01\0\0'  # The low-res thumbnail.
    HIGH_RES = b'\x30\0\0'  # The main image.

    # Used for particle spritesheets, decoded into .sheet_info
    PARTICLE_SHEET = b'\x10\0\0'
    # Cyclic Redundancy Checksum.
    CRC = b'CRC'

    # Allows forcing specific mipmaps to be used for 'medium' shader settings.
    LOD_SETTINGS = b'LOD'

    # 4 extra bytes of bitflags.
    EXTRA_FLAGS = b'TSO'

    # Block of keyvalues data.
    KEYVALUES = b'KVD'


class FilterMode(Enum):
    """The algorithm to use for generating mipmaps."""
    NEAREST = UPPER_LEFT = 0  # Just use the upper-left pixel.
    UPPER_RIGHT = 1  # Just use the upper-right pixel.
    LOWER_LEFT = 2  # Just use the lower-left pixel.
    LOWER_RIGHT = 3  # Just use the lower-right pixel.
    BILINEAR = AVERAGE = 4  # Average the four pixels

Resource = namedtuple('Resource', 'flags data')

Pixel = namedtuple('Pixel', 'r g b a')

_HEADER = struct.Struct(
    '<'    # Align
    'I'    # Header size
    'HH'   # Width, height
    'I'    # Flags
    'H'    # Frame count
    'H'    # First frame index
    '4x'
    'fff'  # Reflectivity vector
    '4x'
    'f'    # Bumpmap scale
    'i'    # High-res image format
    'B'    # Mipmap count
    'i'    # Low-res format (DXT1 usually)
    'BB'   # Low-res width, height
)


class Frame:
    """A single frame of a VTF. This should not be constructed independently.

    This is lazy, so it will only read from the file when actually used.
    """
    __slots__ = [
        'width',
        'height',
        '_data',
        '_fileinfo',
    ]
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        """Private constructor, creates a blank image of this size."""
        self.width = width
        self.height = height
        self._data = None  # type: Optional[array]
        self._fileinfo = None  # type: Optional[Tuple[IO[bytes], int, ImageFormats]]

    def load(self) -> None:
        """If the image has not been loaded, load it from the file stream."""
        if self._data is None:
            self._data = _BLANK_PIXEL * (self.width * self.height)

        if self._fileinfo is None:
            return

        stream, file_off, fmt = self._fileinfo
        self._fileinfo = None

        if getattr(stream, 'closed', False):
            warnings.warn(
                'VTF image frame read after stream was closed!\n'
                'If passing in a stream, close the VTF before closing '
                'the file.',
                ResourceWarning,
                source=stream,
            )

        stream.seek(file_off)
        data = stream.read(fmt.frame_size(self.width, self.height))
        _format_funcs.load(fmt, self._data, data, self.width, self.height)

    def clear(self) -> None:
        """This clears the contents of the frame.

        If the VTF is saved, this will be generated from the larger mipmaps.
        """
        self._data = self._fileinfo = None

    @overload
    def copy_from(self, source: 'Frame') -> None: ...
    @overload
    def copy_from(
        self,
        source: Union[bytes, bytearray, array, memoryview],
        format: ImageFormats = ImageFormats.RGBA8888,
    ) -> None: ...

    def copy_from(
        self,
        source: Union['Frame', bytes, bytearray, array, memoryview],
        format: ImageFormats = ImageFormats.RGBA8888,
    ) -> None:
        """Overwrite this frame with other data.

        The source can be another Frame, or any buffer with bytes-format data.
        """
        if isinstance(source, Frame):
            if self.width != source.width or self.height != source.height:
                raise ValueError("Tried copying from a frame of a different size!")
            source.load()
            if self._data is None:  # Duplicate the other array
                self._data = source._data[:]
            else: # Copy the other array onto us
                self._data[:] = source._data
        else:
            if self._data is None:
                self._data = _BLANK_PIXEL * (self.width * self.height)
            view = memoryview(source)
            # For efficiency, our functions assume the view is contiguous.
            # If it isn't, make a copy to force that.
            if not view.c_contiguous:
                view = view.tobytes()

            # We also have to verify format size.
            required_size = format.frame_size(self.width, self.height)
            if len(view) != required_size:
                raise ValueError(
                    f"Expected {required_size} bytes "
                    f"for {self.width}x{self.height} {format} image, "
                    f"got {len(view)} bytes!"
                )
            _format_funcs.load(format, self._data, view, self.width, self.height)

    def rescale_from(self, larger: 'Frame', filter: FilterMode=FilterMode.BILINEAR) -> None:
        """Regenerate this image from the provided frame, which is twice the size."""
        if 2 * self.width != larger.width or 2 * self.height != larger.height:
            raise ValueError("Larger image must be exactly twice the size!")
        if self._data is None:
            self._data = _BLANK_PIXEL * (self.width * self.height)
        if larger._data is not None:
            _format_funcs.scale_down(filter, self.width, self.height, larger._data, self._data)

    def __getitem__(self, item: Tuple[int, int]) -> Pixel:
        """Retrieve an individual pixel."""
        self.load()
        x, y = item
        if x > self.width or y > self.height:
            raise IndexError(item)
        off = x * self.width + y
        return Pixel._make(self._data[off: off + 4])

    def __setitem__(
        self,
        item: Tuple[int, int],
        data: Tuple[int, int, int, int],
    ) -> None:
        """Set an individual pixel."""
        self.load()

        x, y = item
        if x > self.width or y > self.height:
            raise IndexError(item)
        off = x * self.width + y
        [
            self._data[off],
            self._data[off + 1],
            self._data[off + 2],
            self._data[off + 3],
        ] = data

    def to_PIL(self) -> 'PIL_Image':
        """Convert the given frame into a PIL image.

        Requires Pillow to be installed.
        """
        self.load()

        from PIL.Image import frombuffer
        return frombuffer(
            'RGBA',
            (self.width, self.height),
            self._data,
            'raw',
            'RGBA',
            0,
            1,
        ).copy()

    def to_tkinter(self, tk: 'tkinter.Misc' = None) -> 'tkinter.PhotoImage':
        """Convert the given frame into a Tkinter PhotoImage."""
        self.load()

        import tkinter
        return tkinter.PhotoImage(
            master=tk,
            # Convert it to PPM format, which Tkinter understands natively.
            # That requires a bunch of data crunching, so the code is Cythonised
            # if possible.
            data=_format_funcs.ppm_convert(
                self._data,
                self.width,
                self.height,
            ),
        )


class VTF:
    """Valve Texture Format files, used in the Source Engine."""
    def __init__(
        self, 
        width: int,
        height: int,
        version: Tuple[int, int]=(7, 5),
        ref: Vec=Vec(0, 0, 0),
        frames: int=1,
        bump_scale: float=1.0,
        sheet_info: Iterable['SheetSequence']=(),
        flags: VTFFlags=VTFFlags.EMPTY,
        fmt: ImageFormats=ImageFormats.RGBA8888,
        thumb_fmt: ImageFormats=ImageFormats.DXT1,
        depth: int=1,
    ):
        """Load a VTF file."""
        if not ((7, 2) <= version <= (7, 5)):
            raise ValueError('Version must be between 7.2 and 7.5')
        if not math.log2(width).is_integer():
            raise ValueError("Width must be a power of 2!")
        if not math.log2(height).is_integer():
            raise ValueError("Height must be a power of 2!")
        if frames < 1:
            raise ValueError("Invalid frame count!")

        # If it's a cubemap, depth must be 1.
        if VTFFlags.ENVMAP in flags:
            if depth != 1:
                raise ValueError(
                    "Cubemaps must have a depth "
                    "of 1, not {!r}".format(depth)
                )
        elif depth < 1:
            raise ValueError("Depth must be positive!")

        self.width = width
        self.height = height
        self.depth = depth

        self.version = version
        self.reflectivity = ref
        self.bumpmap_scale = bump_scale
        self.resources = {}  # type: Dict[Union[ResourceID, bytes], Resource]
        self.sheet_info = list(sheet_info)
        self.flags = flags
        self.frame_count = frames
        self.high_format = fmt
        self.low_format = thumb_fmt

        # (frame, depth/cubemap, mipmap) -> frame
        self._frames = {}  # type: Dict[Tuple[int, Union[CubeSide, int], int], Frame]
        self._low_res = Frame(16, 16)

        if VTFFlags.ENVMAP in flags:
            if version[1] == 5:
                depth_iter = CUBES  # type: Iterable[Union[int, CubeSide]]
            else:
                depth_iter = CUBES_WITH_SPHERE
        else:
            depth_iter = range(depth)

        mip_count = 0
        for mip_count in itertools.count():
            for frame in range(frames):
                for cube_or_depth in depth_iter:
                    self._frames[frame, cube_or_depth, mip_count] = Frame(width, height)

            # Once either is 1 large, we have no more mipmaps.
            # Create the frame first, so we still create the final 1-large frame.
            if width <= 1 or height <= 1:
                break

            width >>= 1
            height >>= 1
        self.mipmap_count = mip_count

    @classmethod    
    def read(cls: 'Type[VTF]', file: IO[bytes]) -> 'VTF':
        """Read in a VTF file."""
        signature = file.read(4)
        if signature != b'VTF\0':
            raise ValueError('Bad file signature!')
        version_major, version_minor = struct.unpack('II', file.read(8))
        
        if version_major != 7 or not (0 <= version_minor <= 5):
            raise ValueError(
                "VTF version {}.{} is not "
                "between 7.0-7.5!".format(
                    version_major, version_minor,
                )
            )

        vtf = cls.__new__(cls)
        
        (
            header_size,
            width,
            height,
            flags,
            frame_count,
            first_frame_index,
            ref_r, ref_g, ref_b,
            bumpmap_scale,
            high_format,
            mipmap_count,
            low_format,
            low_width, low_height,
        ) = _HEADER.unpack(file.read(_HEADER.size))  # type: int, int, int, int, int, int, float, float, float, float, int, int, int, int, int

        vtf._frames = {}

        vtf.width = width
        vtf.height = height
        vtf.frame_count = frame_count
        vtf.mipmap_count = mipmap_count
        vtf.flags = VTFFlags(flags)
        vtf.reflectivity = Vec(ref_r, ref_g, ref_b)
        vtf.bumpmap_scale = bumpmap_scale
        vtf.format = fmt = FORMAT_ORDER[high_format]
        vtf.version = version_major, version_minor
        vtf.low_format = low_fmt = FORMAT_ORDER[low_format]

        if fmt is ImageFormats.NONE:
            raise ValueError('High-res format cannot be missing!')

        # For volumetric textures, multiple layers. (Cannot be used with faces.)
        vtf.depth = 1
        if version_minor >= 2:
            [vtf.depth] = struct.unpack('H', file.read(2))
        if vtf.depth <= 0:
            vtf.depth = 1

        low_res_offset = high_res_offset = None  # type: Optional[int]

        vtf.resources = {}

        # Read resources.
        if version_minor >= 3:
            [num_resources] = struct.unpack('<3xI8x', file.read(15))
            for i in range(num_resources):
                [res_id, res_flags, data] = struct.unpack('<3sBI', file.read(8))  # type: bytes, int, int
                if res_id in vtf.resources:
                    raise ValueError(
                        'Duplicate resource ID "{}"!'.format(res_id)
                    )

                # These do not go in the resources, it's only parsed as images.
                if res_id == ResourceID.LOW_RES:
                    low_res_offset = data
                elif res_id == ResourceID.HIGH_RES:
                    high_res_offset = data
                else:
                    try:
                        res_id = ResourceID(res_id)
                    except ValueError:
                        pass  # Custom.
                    vtf.resources[res_id] = Resource(res_flags, data)

            for res_id, (res_flags, data) in vtf.resources.items():
                if not res_flags & 0x02:
                    # There's actual data elsewhere in the file.
                    file.seek(data)
                    [size] = struct.unpack('I', file.read(4))
                    res_data = file.read(size)
                    vtf.resources[res_id] = Resource(res_flags, res_data)

            if low_res_offset is None and low_fmt is not ImageFormats.NONE:
                raise ValueError('Missing low-res thumbnail resource!')
            if high_res_offset is None:
                raise ValueError('Missing main image resource!')

            if ResourceID.PARTICLE_SHEET in vtf.resources:
                vtf.sheet_info = SheetSequence.from_resource(
                    vtf.resources[ResourceID.PARTICLE_SHEET].data
                )

        else:
            low_res_offset = header_size
            high_res_offset = low_res_offset + low_fmt.frame_size(low_width, low_height)

        # We don't implement these high-res formats.
        if fmt is ImageFormats.RGBA16161616 or fmt is ImageFormats.RGBA16161616F:
            return vtf

        vtf._low_res = Frame(low_width, low_height)
        if low_fmt is not ImageFormats.NONE:
            vtf._low_res._fileinfo = (file, low_res_offset, low_fmt)

        # If cubemaps are present, we iterate that for depth.
        # Otherwise it's the depth value.
        if VTFFlags.ENVMAP in vtf.flags:
            # For version 7.5, the spheremap is skipped.
            if version_minor == 5:
                depth_iter = CUBES  # type: Iterable[Union[int, CubeSide]]
            else:
                depth_iter = CUBES_WITH_SPHERE
        else:
            depth_iter = range(vtf.depth)

        for data_mipmap in reversed(range(mipmap_count)):
            mip_width = max(width >> data_mipmap, 1)
            mip_height = max(height >> data_mipmap, 1)
            for frame_ind in range(frame_count):
                for depth_or_cube in depth_iter:
                    frame = vtf._frames[
                        frame_ind,
                        depth_or_cube,
                        data_mipmap,
                    ] = Frame(mip_width, mip_height)
                    # noinspection PyProtectedMember
                    frame._fileinfo = (file, high_res_offset, fmt)
                    high_res_offset += fmt.frame_size(mip_width, mip_height)
        return vtf

    def __enter__(self) -> 'VTF':
        """The VTF file can be used as a context manager.

        This will close the streams if any frames still have them open.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the streams if any frames still have them open."""
        for frame in self._frames.values():
            frame._fileinfo = None

    def load(self) -> None:
        """Fully load all image frames from the VTF.

        This allows closing the file stream.
        """
        for frame in self._frames.values():
            frame.load()
        
    def __len__(self) -> int:
        """The length of a VTF is the number of image frames."""
        return len(self._frames)

    def get(
        self, *,
        frame: int = 0,
        depth: int = 0,
        side: CubeSide = None,
        mipmap: int = 0,
    ) -> Frame:
        """Get a specific image frame.

        If the texture is a cubemap, a side must be provided and depth must be 0.
        """
        if side is not None and depth != 0:
            raise TypeError('Side and depth are mutually exclusive!')
        if VTFFlags.ENVMAP in self.flags:
            if side is None:
                raise ValueError('Side must be provided for cubemaps!')
            depth_side = side
        else:
            depth_side = depth
        return self._frames[frame, depth_side, mipmap]


TexCoord = namedtuple('TexCoord', ['left', 'top', 'right', 'bottom'])


class SheetSequence:
    """VTFs may contain a number of sequences using different parts of the image."""
    MAX_COUNT = 64

    def __init__(
        self,
        frames: List[Tuple[float, TexCoord, TexCoord, TexCoord, TexCoord]],
        clamp: bool,
        duration: float,
    ):
        self.frames = frames
        self.clamp = clamp
        self.duration = duration

    @classmethod
    def from_resource(cls, data: bytes) -> List[Optional['SheetSequence']]:
        """Decode from the resource data."""
        (
            version,
            sequence_count,
        ) = struct.unpack_from('<II', data)
        offset = 8

        if version > 1:
            raise ValueError('Unknown version {}!'.format(version))

        sequences = [None] * SheetSequence.MAX_COUNT  # type: List[Optional['SheetSequence']]
        if sequence_count > SheetSequence.MAX_COUNT:
            raise ValueError('Cannot have more than {} sequences ({})!'.format(
                SheetSequence.MAX_COUNT,
                sequence_count
            ))

        for _ in range(sequence_count):
            (
                seq_num,
                clamp,
                frame_count,
                total_time,
            ) = struct.unpack_from('<Ixxx?If', data, offset)
            offset += 16
            # seq_num = _
            if not (0 <= seq_num < 64):
                raise ValueError('Invalid sequence number {}!'.format(seq_num))

            frames = []  # type: List[Tuple[float, TexCoord, TexCoord, TexCoord, TexCoord]]

            for frame_ind in range(frame_count):
                [duration] = struct.unpack_from('<f', data, offset)
                offset += 4

                if version == 0:
                    # Only one in the file, repeated 4 times.
                    tex_coord = TexCoord._make(struct.unpack_from('<4f', data, offset))
                    frames.append((duration, tex_coord, tex_coord, tex_coord, tex_coord))
                    offset += 16
                else:
                    frames.append((
                        duration,
                        TexCoord._make(struct.unpack_from('<4f', data, offset)),
                        TexCoord._make(struct.unpack_from('<4f', data, offset + 16)),
                        TexCoord._make(struct.unpack_from('<4f', data, offset + 32)),
                        TexCoord._make(struct.unpack_from('<4f', data, offset + 48)),
                    ))
                    offset += 64

            sequences[seq_num] = SheetSequence(frames, clamp, total_time)

        return list(filter(None, sequences))

# Add support for the imghdr module.


def test_vtf(h: bytes, f: IO[bytes]) -> Optional[str]:
    """Source Engine Valve Texture Format."""
    if h[:4] == b'VTF\0':
        try:
            version_major, version_minor = struct.unpack('II', h[4:12])
        except struct.error:
            return None
        if version_major == 7 and (0 <= version_minor <= 5):
            return 'source_vtf'
    return None

import imghdr
imghdr.test_vtf = test_vtf
imghdr.tests.append(test_vtf)
del imghdr, test_vtf
