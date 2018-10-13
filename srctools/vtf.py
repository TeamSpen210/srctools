"""Reads VTF image data into a PIL object."""
from array import array
import math
import struct
from collections import namedtuple
from enum import Enum

from srctools import Vec

from typing import IO, List

# A little dance to import both the Cython and Python versions,
# and choose an appropriate unprefixed version.

# noinspection PyProtectedMember
from srctools import _vtf_readwrite as _Py_format_funcs
try:
    # noinspection PyUnresolvedReferences, PyProtectedMember
    from srctools import _vtf_readwrite_cython as _Cy_format_funcs # type: ignore
    _format_funcs = _Cy_format_funcs  # type: ignore
except ImportError:
    # Type checker only reads this branch.
    _format_funcs = _Py_format_funcs


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
    DXT1_ONEBITALPHA = f(a=1, size=4)
    BGRA5551 = f(5, 5, 5, 1)
    UV88 = f(size=16)
    UVWQ8888 = f(size=32)
    RGBA16161616F = f(16, 16, 16, 16)
    RGBA16161616 = f(16, 16, 16, 16)
    UVLX8888 = f(size=32)

    def frame_size(self, width: int, height: int) -> int:
        """Compute the number of bytes needed for this image size."""
        if self.name in ('DXT1', 'DXT3', 'DXT5', 'DXT1ONEBITALPHA'):
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


FORMAT_ORDER = list(ImageFormats)  # type: List[ImageFormats]


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
    'I'    # High-res image format
    'B'    # Mipmap count
    'I'    # Low-res format (DXT1 usually)
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
        assert 2 <= version_minor <= 5, version_minor
        
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
        vtf.low_formtat = low_fmt = FORMAT_ORDER[low_format]

        # For volumetric textures, multiple layers. (Cannot be used with faces.)
        vtf.depth = 1

        if version_minor >= 3:
            raise NotImplementedError()
        elif version_minor >= 2:
            [vtf.depth] = struct.unpack('H', file.read(2))

        # We don't implement this high-res format.
        if fmt is ImageFormats.RGBA16161616:
            return vtf

        # We always seek, there's an unknown amount of padding here.
        file.seek(vtf._header_size)

        vtf._low_res = _blank_frame(low_width, low_height)
        try:
            _load_frame(
                low_fmt,
                vtf._low_res,
                file.read(low_fmt.frame_size(low_width, low_height)),
                low_width,
                low_height
            )
        except NotImplementedError:
            # TODO: Implement all formats.
            vtf._low_res = None

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
