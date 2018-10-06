"""Reads VTF image data into a PIL object."""
import struct
from enum import Enum
from collections import namedtuple

from PIL import Image, ImageFile
from srctools import Vec

from typing import IO, Tuple, Union, List

# Raw image mode, pixel counts or object(), bytes per pixel. 
ImageAlignment = namedtuple("ImageAlignment", 'mode r g b a size')

def f(mode, r=0, g=0, b=0, a=0, *, l=0, size=0):
    """Helper function to construct ImageFormats."""
    if l:
        r = g = b = l
    if not size:
        size = r + g + b + a
        
    return (mode, r, g, b, a, size)

class ImageFormats(ImageAlignment, Enum):
    """All VTF image formats, with their data sizes in the value."""
    RGBA8888 = f('RGBA', 8, 8, 8, 8)
    ABGR8888 = f('ABGR', 8, 8, 8, 8)
    RGB888 = f('RGB', 8, 8, 8, 0)
    BGR888 = f('BGR', 8, 8, 8)
    RGB565 = f('RGB;16L', 5, 6, 5, 0)
    I8 = f('L', l=8, a=0)
    IA88 = f('LA', l=8, a=8)
    P8 = f('?') # Paletted, not used.
    A8 = f('a', a=8)
    # Blue = alpha channel too
    RGB888_BLUESCREEN = f('rgb', 8, 8, 8)
    BGR888_BLUESCREEN = f('bgr', 8, 8, 8)
    ARGB8888 = f('ARGB', 8, 8, 8, 8)
    BGRA8888 = f('BFRA', 8, 8, 8, 8)
    DXT1 = f('dxt1', size=4)
    DXT3 = f('dxt3', size=8)
    DXT5 = f('dxt5', size=8)
    BGRX8888 = f('bgr_', 8, 8, 8, 8)
    BGR565 = f('bgr', 5, 6, 5)
    BGRX5551 = f('bgr_', 5, 5, 5, 1)
    BGRA4444 = f('bgra', 4, 4, 4, 4)
    DXT1_ONEBITALPHA = f('dxt1', a=1, size=4)
    BGRA5551 = f('bgra', 5, 5, 5, 1)
    UV88 = f('?')
    UVWQ8888 = f('?')
    RGBA16161616F = f('rgba', 16, 16, 16, 16)
    RGBA16161616 = f('rgba', 16, 16, 16, 16)
    UVLX8888 = f('?')
    
    @property
    def mode(self):
        """Return the PIL image mode for this file format."""
        if self.name == 'RGBA16161616':
            return 'I' # 16-bit integer
        elif self.name == 'RGBA16161616F':
            return 'F'  # 16-bit floating point
        if self.name in ('A8', 'IA88'):
            return 'LA'
        elif self.name == 'I8':
            return 'L'
        elif self.a != 0 or 'A' in self.name or self.name in ('DXT3', 'DXT5'):
            return 'RGBA'
        else:
            return 'RGB'
        
    # Force object-style comparisons.
    __gt__ = object.__gt__
    __lt__ = object.__lt__
    __ge__ = object.__ge__
    __le__ = object.__le__
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__
    
del f

# Formats requiring specific code to decode them.
# These are all the compressed formats, plus some oddballs.
SPECIAL_FORMATS = {
    ImageFormats.DXT1,
    ImageFormats.DXT1_ONEBITALPHA,
    ImageFormats.DXT3,
    ImageFormats.DXT5,
    ImageFormats.P8,
    ImageFormats.UV88,
    ImageFormats.UVWQ8888,
    ImageFormats.UVLX8888,
}

for fmt in ImageFormats:
    if type(fmt.a) is object:
        assert fmt in SPECIAL_FORMATS, fmt
del fmt

FORMAT_INDEX = {
    ind: format
    for ind, format in
    enumerate(ImageFormats)
}

_HEADER = struct.Struct(
    '<'   # Align
    'I'   # header size
    'HH'  # width, height
    'I'   # flags
    'H'   # frame count
    'H'   # first frame index
    '4x'
    'fff' # reflectivity vector
    '4x'
    'f'   # bumpmap scale
    'I'   # high-res image format
    'B'   # mipmap count
    'I'   # low-res format (DXT1 usually)
    'BB'  # Low-res width, height
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
        if not ((7, 2) <= version_minor <= (7, 5)):
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
        
        img_mode = version.mode
        
        self.frames = [
            [Image.new(img_mode, (width, height))]
            for _ in range(frames)
        ]
           
    @classmethod    
    def read(cls, file: IO[bytes]) -> 'VTF':
        """Read in a VTF file."""
        signature = file.read(4)
        if signature != b'VTF\0':
            raise ValueError('Bad file signature!')
        version_major, version_minor = struct.unpack('II', file.read(8))
        
        assert version_major == 7, version_major
        assert 2 <= version_minor <= 5, version_minor
        
        vtf = cls.__new__(cls)
        
        (
            vtf._header_size,
            vtf.width, vtf.height,
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
        
        vtf._frames = []
        
        vtf.reflectivity = Vec(ref_r, ref_g, ref_b)
        vtf.format = fmt = FORMAT_INDEX[high_format]
        vtf.version = version_major, version_minor
        
        if version_minor >= 3:
            raise NotImplementedError()
        elif version_minor == 2:
            [mipmap_depth] = struct.unpack('H', file.read(2))
            
        vtf.frames = [None] * frame_count
        
        for frame_ind in range(frame_count):
            frame = vtf.frames[frame_ind] = [None] * mipmap_count
            if fmt in SPECIAL_FORMATS:
                continue
            for mipmap in reversed(range(mipmap_count)):
               frame[mipmap] = Image.frombytes(
                  'RGB',
                   (vtf.width>>mipmap, vtf.height>>mipmap),
                   file.read(3*(vtf.width>>mipmap)*(vtf.height>>mipmap)),
                  'raw',
                  fmt.mode,
               )
               if mipmap == 0:
                   frame[mipmap].show()
        
    def seek(self, frame: int) -> None:
        """Switch to the given frame, or raise EOFError if moved outside the file."""
        if frame < 0:
            raise ValueError('Negative frame')
        if frame > self.__frame_count:
            raise EOFError()
        self.__cur_frame = frame
        offset = (
            self.__img_start
            # + frame offset...
        )
        self.fp.seek(offset)
        print('Format =', self.__format)
        if self.__format in SPECIAL_FORMATS:
            raise NotImplementedError
            self.tile = [
                ("srcvtf", (0, 0) + self.size, offset, (
                    self.__format, 
                    offset,
                ))
            ]
        else:
            self.tile = [(
                "raw",
                (0, 0) + self.size,
                offset,
                (self.__format.order, 0, 1),
            )]

