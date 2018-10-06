"""Reads VTF image data into a PIL object."""
import math
import struct

from srctools import Vec

from typing import IO, List

# A little dance to import both the Cython and Python versions,
# and choose an appropriate unprefixed version.

# noinspection PyProtectedMember
from srctools._vtf_frame import (
    ImageFrame as Py_ImageFrame,
    ImageFormats as _Py_ImageFormats,
)
try:
    # noinspection PyUnresolvedReferences, PyProtectedMember
    from srctools._vtf_frame_cython import (
        ImageFrame as Cy_ImageFrame,
        FORMATS as Cy_IMAGE_FORMATS,
        _FORMAT_ORDER
    )  # type: ignore
    ImageFrame = Cy_ImageFrame  # type: ignore
    _ImageFormats = Cy_IMAGE_FORMATS  # type: ignore
except ImportError:
    # Type checker only reads this branch.
    ImageFrame = Py_ImageFrame
    _ImageFormats = _Py_ImageFormats
    _FORMAT_ORDER = list(_ImageFormats)  # type: List[_ImageFormats]

FMT_RGBA8888 = _ImageFormats.RGBA8888
FMT_ABGR8888 = _ImageFormats.ABGR8888
FMT_RGB888 = _ImageFormats.RGB888
FMT_BGR888 = _ImageFormats.BGR888
FMT_RGB565 = _ImageFormats.RGB565
FMT_I8 = _ImageFormats.I8
FMT_IA88 = _ImageFormats.IA88
FMT_P8 = _ImageFormats.P8
FMT_A8 = _ImageFormats.A8
FMT_RGB888_BLUESCREEN = _ImageFormats.RGB888_BLUESCREEN
FMT_BGR888_BLUESCREEN = _ImageFormats.BGR888_BLUESCREEN
FMT_ARGB8888 = _ImageFormats.ARGB8888
FMT_BGRA8888 = _ImageFormats.BGRA8888
FMT_DXT1 = _ImageFormats.DXT1
FMT_DXT3 = _ImageFormats.DXT3
FMT_DXT5 = _ImageFormats.DXT5
FMT_BGRX8888 = _ImageFormats.BGRX8888
FMT_BGR565 = _ImageFormats.BGR565
FMT_BGRX5551 = _ImageFormats.BGRX5551
FMT_BGRA4444 = _ImageFormats.BGRA4444
FMT_DXT1_ONEBITALPHA = _ImageFormats.DXT1_ONEBITALPHA
FMT_BGRA5551 = _ImageFormats.BGRA5551
FMT_RGBA16161616F = _ImageFormats.RGBA16161616F

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
            ImageFrame(width, height)
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
        
        vtf = cls.__new__(cls)
        
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

        vtf.width = width >> mipmap
        vtf.height = height >> mipmap
        
        vtf._frames = [
            ImageFrame(width, height)
            for _ in range(frame_count)
        ]
        
        vtf.reflectivity = Vec(ref_r, ref_g, ref_b)
        vtf.format = fmt = _FORMAT_ORDER[high_format]
        vtf.version = version_major, version_minor
        
        if version_minor >= 3:
            raise NotImplementedError()
        elif version_minor == 2:
            [mipmap_depth] = struct.unpack('H', file.read(2))

        bytes_per_pixel = fmt.size
        
        for frame_ind in range(frame_count):
            for data_mipmap in reversed(range(mipmap_count)):
                mip_width = width >> data_mipmap
                mip_height = height >> data_mipmap
                mip_data = file.read(
                    bytes_per_pixel * mip_width * mip_height
                )
                if data_mipmap == mipmap:
                    vtf._frames[frame_ind]._load(fmt, mip_data)

        return vtf
