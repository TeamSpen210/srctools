"""Reads VTF image data into a PIL object."""
from array import array
import math
import struct
from collections import namedtuple
from enum import Enum

from srctools import Vec

from typing import IO, Dict

# A little dance to import both the Cython and Python versions,
# and choose an appropriate unprefixed version.

# noinspection PyProtectedMember
from srctools import _py_vtf_readwrite as _py_format_funcs
try:
    # noinspection PyUnresolvedReferences, PyProtectedMember
    from srctools import _cy_vtf_readwrite as _cy_format_funcs # type: ignore
    _format_funcs = _cy_format_funcs  # type: ignore
except ImportError:
    # Type checker only reads this branch.
    _format_funcs = _py_format_funcs


# The _vtf_readwrite module contains load_FORMATNAME() functions which
# convert the VTF data into a uniform 32-bit RGBA block, which we can then
# parse.
# That works for everything except RGBA16161616 (used for HDR cubemaps), which
# is 16-bit for each channel. We can't do much about that.


class ImageAlignment(namedtuple("ImageAlignment", 'r g b a size')):
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


def f(r=0, g=0, b=0, a=0, *, l=0, size=0):
    """Helper function to construct ImageFormats."""
    if l:
        r = g = b = l
        size = l + a
    if not size:
        size = r + g + b + a

    return r, g, b, a, size


class ImageFormats(ImageAlignment, Enum):
    """All VTF image formats, with their data sizes in the value."""
    RGBA8888 = f(8, 8, 8, 8)
    ABGR8888 = f(8, 8, 8, 8)
    RGB888 = f(8, 8, 8, 0)
    BGR888 = f(8, 8, 8)
    RGB565 = f(5, 6, 5, 0)
    I8 = f(a=0, l=8)
    IA88 = f(a=8, l=8)
    P8 = f()  # Palletted, not used.
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

    def frame_size(self, width: int, height: int) -> int:
        """Compute the number of bytes needed for this image size."""
        if self.name in ('DXT1', 'DXT3', 'DXT5', 'DXT1_ONEBITALPHA'):
            block_wid, mod = divmod(width, 4)
            if mod:
                block_wid += 1

            block_height, mod = divmod(height, 4)
            if mod:
                block_height += 1
            return self.size * block_wid * block_height // 8
        else:
            return self.size * width * height // 8

del f


FORMAT_ORDER = dict(enumerate(ImageFormats))  # type: Dict[int, ImageFormats]
FORMAT_ORDER[-1] = ImageFormats.NONE


class ResourceID(bytes, Enum):
    """For VTF format 7.3+, there is an extensible resource system."""
    # The two data parts in earlier versions.
    LOW_RES = b'\x01\0\0'  # The low-res thumbnail.
    HIGH_RES = b'\x30\0\0'  # The main image.

    # Used for particle spritesheets.
    PARTICLE_SHEET = b'\x10\0\0'
    # Cyclic Redundancy Checksum.
    CRC = b'CRC'

    # Allows forcing specific mipmaps to be used for 'medium' shader settings.
    LOD_SETTINGS = b'LOD'

    # 4 extra bytes of bitflags.
    EXTRA_FLAGS = b'TSO'

    # Block of keyvalues data.
    KEYVALUES = b'KVD'


Resource = namedtuple('Resource', 'flags data')


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


def _blank_frame(width: int, height: int) -> array:
    """Construct a blank image of the desired size."""
    return _format_funcs.blank(width, height)


def _load_frame(
    fmt: ImageFormats,
    pixels: array,
    data: bytes,
    width: int,
    height: int,
) -> None:
    """Load in pixels from VTF data."""
    try:
        loader = getattr(_format_funcs, "load_" + fmt.name.casefold())
    except AttributeError:
        raise NotImplementedError(
            "Loading {} not implemented!".format(fmt.name)
        ) from None
    loader(pixels, data, width, height)


class VTF:
    """Valve Texture Format files, used in the Source Engine."""
    def __init__(
        self, 
        width: int,
        height: int,
        version=(7, 5),
        ref=Vec(0, 0, 0),
        frames=1,
        bump_scale=1.0,
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
        
        self.width = width
        self.height = height
        self.version = version
        self.reflectivity = ref
        self.bump_scale = bump_scale
        self.resources = {}
        
        self._frames = [
            _blank_frame(width, height)
            for _ in range(frames)
        ]

    @classmethod    
    def read(
        cls,
        file: IO[bytes],
        mipmap: int=0,
    ) -> 'VTF':
        """Read in a VTF file.

        If specified, mipmap will read in a shrunken image.
        """
        signature = file.read(4)
        if signature != b'VTF\0':
            raise ValueError('Bad file signature!')
        version_major, version_minor = struct.unpack('II', file.read(8))
        
        assert version_major == 7, version_major
        assert 0 <= version_minor <= 5, version_minor
        
        vtf = cls.__new__(cls)  # type: VTF
        
        (
            vtf._header_size,
            width,
            height,
            vtf.flags,
            frame_count,
            first_frame_index,
            ref_r, ref_g, ref_b,
            vtf.bumpmap_scale,
            high_format,
            mipmap_count,
            low_format,
            low_width, low_height,
        ) = _HEADER.unpack(file.read(_HEADER.size))

        vtf.width = max(width >> mipmap, 1)
        vtf.height = max(height >> mipmap, 1)
        
        vtf._frames = [
            _blank_frame(width, height)
            for _ in range(frame_count)
        ]
        
        vtf.reflectivity = Vec(ref_r, ref_g, ref_b)
        vtf.format = fmt = FORMAT_ORDER[high_format]
        vtf.version = version_major, version_minor
        vtf.low_format = low_fmt = FORMAT_ORDER[low_format]

        if fmt is ImageFormats.NONE:
            raise ValueError('High-res format cannot be missing!')

        # For volumetric textures, multiple layers. (Cannot be used with faces.)
        vtf.depth = 1
        if version_minor >= 2:
            [vtf.depth] = struct.unpack('H', file.read(2))

        low_res_offset = high_res_offset = None

        vtf.resources = {}

        # Read resources.
        if version_minor >= 3:
            [num_resources] = struct.unpack('<3xI8x', file.read(15))
            for i in range(num_resources):
                [res_id, flags, data] = struct.unpack('<3sBI', file.read(8))
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
                    vtf.resources[res_id] = Resource(flags, data)

            for res_id, (flags, data) in vtf.resources.items():
                if not flags & 0x02:
                    # There's actual data elsewhere in the file.
                    file.seek(data)
                    [size] = struct.unpack('I', file.read(4))
                    data = file.read(size)
                    vtf.resources[res_id] = Resource(flags, data)

            if low_res_offset is None and low_fmt is not ImageFormats.NONE:
                raise ValueError('Missing low-res thumbnail resource!')
            if high_res_offset is None:
                raise ValueError('Missing main image resource!')
        else:
            low_res_offset = vtf._header_size
            high_res_offset = low_res_offset + low_fmt.frame_size(low_width, low_height)

        # We don't implement these high-res formats.
        if fmt is ImageFormats.RGBA16161616 or fmt is ImageFormats.RGBA16161616F:
            return vtf

        vtf._low_res = _blank_frame(low_width, low_height)
        if low_fmt is not ImageFormats.NONE:
            file.seek(low_res_offset)
            _load_frame(
                low_fmt,
                vtf._low_res,
                file.read(low_fmt.frame_size(low_width, low_height)),
                low_width,
                low_height
            )

        file.seek(high_res_offset)
        for frame_ind in range(frame_count):
            for data_mipmap in reversed(range(mipmap_count)):
                mip_width = max(width >> data_mipmap, 1)
                mip_height = max(height >> data_mipmap, 1)
                mip_data = file.read(fmt.frame_size(mip_width, mip_height))
                if data_mipmap == mipmap:
                    _load_frame(
                        fmt,
                        vtf._frames[frame_ind],
                        mip_data,
                        mip_width,
                        mip_height,
                    )

        return vtf

    def to_PIL(self, frame: int):
        """Convert the given frame into a PIL image.

        Requires Pillow to be installed.
        """
        from PIL import Image
        return Image.frombuffer(
            'RGBA',
            (self.width, self.height),
            self._frames[frame],
            'raw',
            'RGBA',
            0,
            1,
        )

    def to_tkinter(self, frame: int, tk=None):
        """Convert the given frame into a Tkinter PhotoImage."""
        from tkinter import PhotoImage
        return PhotoImage(
            master=tk,
            data=_format_funcs.ppm_convert(
                self._frames[frame],
                self.width,
                self.height,
            ),
        )
