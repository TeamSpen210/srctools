"""Reads VTF image data into a PIL object.

After this is imported, the imghdr module can recoginise
VTF images (returning 'source_vtf').
"""
from typing import (
    IO, TYPE_CHECKING, Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence,
    Tuple, Type, Union, overload,
)
from array import array
from collections import namedtuple
from enum import Enum, Flag
from io import BytesIO
import itertools
import math
import struct
import types
import warnings

from . import EmptyMapping, binformat
from .const import add_unknown
from .math import Vec


# Only import while type checking, so these expensive libraries are only loaded
# if the user used them elsewhere.
if TYPE_CHECKING:
    import tkinter

    from PIL.Image import Image as PIL_Image

# A little dance to import both the Cython and Python versions,
# and choose an appropriate unprefixed version.
# For type-checking purposes make it think the Cython version is the Python one.

# noinspection PyProtectedMember
from . import _py_vtf_readwrite as _py_format_funcs


_cy_format_funcs = _format_funcs = _py_format_funcs

if not TYPE_CHECKING:
    try:
        # noinspection PyUnresolvedReferences, PyProtectedMember
        from . import _cy_vtf_readwrite as _cy_format_funcs
        _format_funcs = _cy_format_funcs
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


CUBES_WITH_SPHERE: Sequence[CubeSide] = list(CubeSide)
# Remove the Sphere type, for 7.5+
CUBES: Sequence[CubeSide] = CUBES_WITH_SPHERE[:-1]

# One black, opaque pixel for creating blank images.
_BLANK_PIXEL = array('B', [0, 0, 0, 0xFF])


def _mk_fmt(
    r: int = 0, g: int = 0, b: int = 0,
    a: int = 0, *,
    grey: int = 0, size: int = 0,
) -> Tuple[int, int, int, int, int, int]:
    """Helper function to construct ImageFormats."""
    global _mk_fmt_ind
    if grey:
        r = g = b = grey
        size = grey + a
    if not size:
        size = r + g + b + a
    _mk_fmt_ind += 1

    return r, g, b, a, size, _mk_fmt_ind


_mk_fmt_ind = -1  # Incremented first time to 0


class ImageFormats(Enum):
    """All VTF image formats, with their data sizes in the value."""
    def __init__(self, r: int, g: int, b: int, a: int, size: int, ind: int) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.a = a
        self.size = size
        self.ind = ind

    RGBA8888 = _mk_fmt(8, 8, 8, 8)
    ABGR8888 = _mk_fmt(8, 8, 8, 8)
    RGB888 = _mk_fmt(8, 8, 8, 0)
    BGR888 = _mk_fmt(8, 8, 8)
    RGB565 = _mk_fmt(5, 6, 5, 0)
    I8 = _mk_fmt(a=0, grey=8)
    IA88 = _mk_fmt(a=8, grey=8)
    P8 = _mk_fmt()  # Using a palette somehow - was never implemented by Valve.
    A8 = _mk_fmt(a=8)
    # Blue = alpha channel too
    RGB888_BLUESCREEN = _mk_fmt(8, 8, 8)
    BGR888_BLUESCREEN = _mk_fmt(8, 8, 8)
    ARGB8888 = _mk_fmt(8, 8, 8, 8)
    BGRA8888 = _mk_fmt(8, 8, 8, 8)
    DXT1 = _mk_fmt(size=64)
    DXT3 = _mk_fmt(size=128)
    DXT5 = _mk_fmt(size=128)
    BGRX8888 = _mk_fmt(8, 8, 8, 8)
    BGR565 = _mk_fmt(5, 6, 5)
    BGRX5551 = _mk_fmt(5, 5, 5, 1)
    BGRA4444 = _mk_fmt(4, 4, 4, 4)
    DXT1_ONEBITALPHA = _mk_fmt(size=64)
    BGRA5551 = _mk_fmt(5, 5, 5, 1)
    UV88 = _mk_fmt(size=16)
    UVWQ8888 = _mk_fmt(size=32)
    RGBA16161616F = _mk_fmt(16, 16, 16, 16)
    RGBA16161616 = _mk_fmt(16, 16, 16, 16)
    UVLX8888 = _mk_fmt(size=32)

    NONE = _mk_fmt()
    # These two aren't supported by VTEX & VTFEdit, but are by the engine.
    # They're useful for normal maps.
    ATI1N = _mk_fmt(size=64)
    ATI2N = _mk_fmt(size=128)

    def __repr__(self) -> str:
        """Exclude RGB or A sizes if zero."""
        res = [f'<ImageFormats[{self.ind:02}] {self._name_}:']
        if self.r or self.g or self.b:
            res.append(f'r={self.r}, g={self.g}, b={self.b},')
        if self.a:
            res.append(f'a={self.a},')
        res.append(f'size={self.size}>')
        return ' '.join(res)

    @property
    def is_compressed(self) -> bool:
        """Checks if the format is compressed in 4x4 blocks."""
        return self.name.startswith('DXT') or self.name in ('ATI1N', 'ATI2N')

    def frame_size(self, width: int, height: int) -> int:
        """Compute the number of bytes needed for this image size."""
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

    def bin_value(self, asw: bool) -> int:
        """Return the enum value for the given format.

        This is tricky, since for ATIxN it is different in ASW+.
        """
        if self is ImageFormats.NONE:
            return -1
        elif self is ImageFormats.ATI1N:
            return 35 if asw else 38
        elif self is ImageFormats.ATI2N:
            return 34 if asw else 37
        else:
            return self.ind


del _mk_fmt, _mk_fmt_ind
# Initialise the internal mapping in the format modules.
_format_funcs.init(ImageFormats)
if _cy_format_funcs is not _py_format_funcs:
    _py_format_funcs.init(ImageFormats)


FORMAT_ORDER = {
    fmt.ind: fmt
    for fmt in ImageFormats.__members__.values()
    if fmt.name not in ('NONE', 'ATI1N', 'ATI2N')
}
FORMAT_ORDER[-1] = ImageFormats.NONE
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

    # Generate members for the remaining bits, so we can preserve flags we
    # don't recognise.
    add_unknown(locals())


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
    __slots__ = ['width', 'height', '_data', '_fileinfo']
    width: int
    height: int
    _data: Optional['array[int]']  # Only generic in stubs!
    _fileinfo: Optional[Tuple[IO[bytes], int, ImageFormats]]

    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        """Private constructor, creates a blank image of this size."""
        self.width = width
        self.height = height
        self._data = None
        self._fileinfo = None

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
        source: Union[bytes, bytearray, 'array[int]', memoryview],
        format: ImageFormats = ImageFormats.RGBA8888,
    ) -> None: ...

    def copy_from(
        self,
        source: Union['Frame', bytes, bytearray, 'array[int]', memoryview],
        format: ImageFormats = ImageFormats.RGBA8888,
    ) -> None:
        """Overwrite this frame with other data.

        The source can be another Frame, or any buffer with bytes-format data.
        """
        if isinstance(source, Frame):
            if self.width != source.width or self.height != source.height:
                raise ValueError("Tried copying from a frame of a different size!")
            source.load()
            assert source._data is not None
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
                view = memoryview(view.tobytes())

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
        """Regenerate this image from the next mipmap.

        The larger image must either have the same dimension, or exactly double.
        """
        if not (
            self.width == larger.width or 2 * self.width == larger.width
        ) or not (
            self.height == larger.height or 2 * self.height == larger.height
        ):
            raise ValueError(
                "Larger image must be exactly twice or the same size: "
                f"{larger.width}x{larger.height} -> {self.width}x{self.height}"
            )
        if self._data is None:
            self._data = _BLANK_PIXEL * (self.width * self.height)
        if larger._data is not None:
            _format_funcs.scale_down(filter, larger.width, larger.height, self.width, self.height, larger._data, self._data)

    def __getitem__(self, item: Tuple[int, int]) -> Pixel:
        """Retrieve an individual pixel."""
        self.load()
        assert self._data is not None

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
        assert self._data is not None

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
        assert self._data is not None

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

    def to_tkinter(
        self,
        tk: 'tkinter.Misc | None' = None,
        *,
        bg: Optional[Tuple[int, int, int]] = None,
    ) -> 'tkinter.PhotoImage':
        """Convert the given frame into a Tkinter PhotoImage.

        If bg is set, the image will be composited onto this background.
        Otherwise, alpha is ignored.
        """
        self.load()
        assert self._data is not None

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
                bg,
            ),
        )

    # TODO: wx has no type hints, so we can't import.
    def to_wx_image(self, bg: Optional[Tuple[int, int, int]]=None) -> Any:
        """Convert the given frame into a wxPython wx.Image.

        This requires wxPython to be installed.
        If bg is set, the image will be composited onto this background.
        Otherwise, alpha is ignored.
        """
        self.load()
        assert self._data is not None
        import wx  # type: ignore

        img = wx.Image(self.width, self.height)
        _format_funcs.alpha_flatten(self._data, img.GetDataBuffer(), self.width, self.height, bg)
        return img

    def to_wx_bitmap(self, bg: Optional[Tuple[int, int, int]]=None) -> Any:
        """Convert the given frame into a wxPython wx.Bitmap.

        This requires wxPython to be installed.
        If bg is set, the image will be composited onto this background.
        Otherwise, alpha is ignored.
        """
        self.load()
        assert self._data is not None
        import wx  # pyright: ignore

        img = wx.Bitmap(self.width, self.height)
        # Bitmap memory layout isn't public, so we have to write to a temporary
        # that it copies from.
        buf = bytearray(3 * self.width * self.height)
        _format_funcs.alpha_flatten(self._data, buf, self.width, self.height, bg)
        img.CopyFromBuffer(buf, wx.BitmapBufferFormat_RGB)
        return img


class VTF:
    """Valve Texture Format files, used in the Source Engine."""
    width: int
    height: int
    depth: int
    mipmap_count: int

    version: Tuple[int, int]
    reflectivity: Vec
    bumpmap_scale: float
    resources: Dict[Union[ResourceID, bytes], Resource]
    sheet_info: Dict[int, 'SheetSequence']
    flags: VTFFlags
    frame_count: int
    format: ImageFormats
    low_format: ImageFormats

    _frames: Dict[Tuple[int, Union[CubeSide, int], int], Frame]
    _low_res: Frame

    def __init__(
        self,
        width: int,
        height: int,
        version: Tuple[int, int]=(7, 5),
        ref: Vec=Vec(0, 0, 0),
        frames: int=1,
        bump_scale: float=1.0,
        sheet_info: Mapping[int, 'SheetSequence']=EmptyMapping,
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
        self.resources = {}
        self.sheet_info = dict(sheet_info)
        self.flags = flags
        self.frame_count = frames
        self.format = fmt
        self.low_format = thumb_fmt

        # (frame, depth/cubemap, mipmap) -> frame
        self._frames = {}
        self._low_res = Frame(16, 16)

        depth_iter: Iterable[Union[int, CubeSide]]
        if VTFFlags.ENVMAP in flags:
            if version[1] == 5:
                depth_iter = CUBES
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

        header_size: int
        low_width: int
        low_height: int
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
        ) = _HEADER.unpack(file.read(_HEADER.size))

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

        low_res_offset = -1
        high_res_offset = -1

        vtf.resources = {}
        vtf.sheet_info = {}

        # Read resources.
        if version_minor >= 3:
            [num_resources] = struct.unpack('<3xI8x', file.read(15))
            for i in range(num_resources):
                res_id: bytes
                res_flags: int
                data: int
                [res_id, res_flags, data] = struct.unpack('<3sBI', file.read(8))
                if res_id in vtf.resources:
                    raise ValueError(f'Duplicate resource ID {repr(res_id)[1:]}!')

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

            if ResourceID.PARTICLE_SHEET in vtf.resources:
                vtf.sheet_info = SheetSequence.from_resource(
                    vtf.resources.pop(ResourceID.PARTICLE_SHEET).data
                )

        else:
            low_res_offset = header_size
            high_res_offset = low_res_offset + low_fmt.frame_size(low_width, low_height)

        if high_res_offset < 0:
            raise ValueError('Missing main image resource!')

        # We don't implement these high-res formats.
        if fmt is ImageFormats.RGBA16161616 or fmt is ImageFormats.RGBA16161616F:
            return vtf

        vtf._low_res = Frame(low_width, low_height)
        if low_fmt is not ImageFormats.NONE:
            if low_res_offset < 0:
                raise ValueError('Missing low-res thumbnail resource!')
            vtf._low_res._fileinfo = (file, low_res_offset, low_fmt)

        # If cubemaps are present, we iterate that for depth.
        # Otherwise it's the depth value.
        depth_iter: Iterable[Union[int, CubeSide]]
        if VTFFlags.ENVMAP in vtf.flags:
            # For version 7.5, the spheremap is skipped.
            if version_minor == 5:
                depth_iter = CUBES
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

    def save(
        self,
        file: IO[bytes],
        version: Optional[Tuple[int, int]]=None,
        sheet_seq_version: int=1,
        asw_or_later: bool=True,
    ) -> None:
        """Write out the VTF file to this.

        If a version is specified, this overrides the one in the object.
        The particle system version needs to be specified here.
        If ATI1N or ATI2N used, whether the engine is ASW or later needs to
        be specified.
        """
        deferred = binformat.DeferredWrites(file)
        file.write(b'VTF\0')
        if version is None:
            version = self.version
        version_major, version_minor = version

        if version_major != 7 or not (0 <= version_minor <= 5):
            raise ValueError(
                "VTF version {}.{} is not "
                "between 7.0-7.5!".format(
                    version_major, version_minor,
                )
            )
        file.write(struct.pack('<II', version_major, version_minor))

        deferred.defer('header_size', '<I')
        file.write(_HEADER.pack(
            0,
            self.width,
            self.height,
            self.flags.value,
            self.frame_count,
            0,  # Todo: First frame index?
            *self.reflectivity,
            self.bumpmap_scale,
            self.format.bin_value(asw_or_later),
            self.mipmap_count,
            self.low_format.bin_value(asw_or_later),
            self._low_res.width, self._low_res.height,
        ))

        # For volumetric textures, multiple layers. (Cannot be used with faces.)
        if version_minor >= 2:
            file.write(struct.pack('<H', self.depth))
        elif self.depth > 1:
            raise ValueError('Cannot use volumetric textures with versions before 7.2!')

        # Read resources.
        if version_minor >= 3:
            res_count = len(self.resources) + 2  # low/high format are always present.
            if self.sheet_info:
                res_count += 1
            file.write(struct.pack('<3xI8x', res_count))
            for res_id, res in self.resources.items():
                if isinstance(res.data, bytes):
                    # It's later in the file.
                    file.write(struct.pack('<3sB', getattr(res_id, 'value', res_id), res.flags & ~0x02))
                    deferred.defer(('res', res_id), '<I', write=True)
                else:
                    # Just here.
                    file.write(struct.pack('<3sBI', res_id, res.flags | 0x02, res.data))

            # These are always present in the resource.
            file.write(struct.pack('<3sB', ResourceID.LOW_RES.value, 0))
            deferred.defer('low_res', '<I', write=True)
            file.write(struct.pack('<3sB', ResourceID.HIGH_RES.value, 0))
            deferred.defer('high_res', '<I', write=True)
            if self.sheet_info:
                file.write(struct.pack('<3sB', ResourceID.PARTICLE_SHEET.value, 0))
                deferred.defer('particle', '<I', write=True)
        else:
            file.write(bytes(15))  # Pad to 80 bytes.

        deferred.set_data('header_size', file.tell())
        if version_minor >= 3:
            # Write the data itself.
            for res_id, res in self.resources.items():
                if isinstance(res.data, bytes):
                    # There's actual data elsewhere in the file.
                    deferred.set_data(('res', res_id), file.tell())
                    file.write(struct.pack('<I', len(res.data)))
                    file.write(res.data)
            if self.sheet_info:
                particle_data = SheetSequence.make_data(self.sheet_info)
                deferred.set_data('particle', file.tell())
                file.write(struct.pack('<I', len(particle_data)))
                file.write(particle_data)

        self.compute_mipmaps()
        self._low_res.load()

        if version_minor >= 3:
            deferred.set_data('low_res', file.tell())
        if self.low_format is not ImageFormats.NONE:
            data = bytearray(self.low_format.frame_size(self._low_res.width, self._low_res.height))
            if self._low_res._data is not None:
                _format_funcs.save(self.low_format, self._low_res._data, data, self._low_res.width, self._low_res.height)
            file.write(data)

        # If cubemaps are present, we iterate that for depth.
        # Otherwise it's the depth value.
        depth_iter: Iterable[Union[int, CubeSide]]
        if VTFFlags.ENVMAP in self.flags:
            # For version 7.5, the spheremap is skipped.
            if version_minor == 5:
                depth_iter = CUBES
            else:
                depth_iter = CUBES_WITH_SPHERE
        else:
            depth_iter = range(self.depth)

        if version_minor >= 3:
            deferred.set_data('high_res', file.tell())
        for data_mipmap in reversed(range(self.mipmap_count)):
            for frame_ind in range(self.frame_count):
                for depth_or_cube in depth_iter:
                    frame = self._frames[
                        frame_ind,
                        depth_or_cube,
                        data_mipmap,
                    ]
                    frame.load()
                    data = bytearray(self.format.frame_size(frame.width, frame.height))
                    if frame._data is not None:
                        _format_funcs.save(self.format, frame._data, data, frame.width, frame.height)
                    file.write(data)
        deferred.write()

    def __enter__(self) -> 'VTF':
        """The VTF file can be used as a context manager.

        This will close the streams if any frames still have them open.
        """
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException], exc_val: BaseException, exc_tb: types.TracebackType,
    ) -> None:
        """Close the streams if any frames still have them open."""
        for frame in self._frames.values():
            frame._fileinfo = None

    def load(self) -> None:
        """Fully load all image frames from the VTF.

        This allows closing the file stream.
        """
        for frame in self._frames.values():
            frame.load()
        self._low_res.load()

    def clear_mipmaps(self, *, after: int = 0) -> None:
        """Erase the contents of all mipmaps smaller than the given size.

        When saved or compute_mipmaps() is called, these empty mipmaps will
        be recomputed from the largest mipmap.
        By default this clears all but the largest mipmap.
        """
        for (ind, depth_side, mipmap), frame in self._frames.items():
            if mipmap > after:
                frame.clear()
        self._low_res.clear()

    def compute_mipmaps(self, filter: FilterMode=FilterMode.BILINEAR) -> None:
        """Regenerate all mipmaps that have previously been cleared."""
        depth_iter: Collection[Union[int, CubeSide]]
        if VTFFlags.ENVMAP in self.flags:
            # For version 7.5, the spheremap is skipped.
            if self.version == (7, 5):
                depth_iter = CUBES
            else:
                depth_iter = CUBES_WITH_SPHERE
        else:
            depth_iter = range(self.depth)
        for frame_num in range(self.frame_count):
            for depth_side in depth_iter:
                # Force to blank if cleared, we can't load it from aynthing.
                self._frames[frame_num, depth_side, 0].load()
                for mipmap in range(1, self.mipmap_count):
                    frm = self._frames[frame_num, depth_side, mipmap]
                    if frm._data is None:
                        frm.rescale_from(
                            self._frames[frame_num, depth_side, mipmap - 1],
                            filter,
                        )

        # Also regenerate the low-res format.
        if self.low_format is not ImageFormats.NONE:
            for mipmap in range(self.mipmap_count):
                frame = self.get(mipmap=mipmap)
                if (
                    frame.width // 2 == self._low_res.width and
                    frame.height // 2 == self._low_res.height
                ):
                    self._low_res.rescale_from(frame, filter)

    def __len__(self) -> int:
        """The length of a VTF is the number of image frames."""
        return len(self._frames)

    def get(
        self, *,
        frame: int = 0,
        depth: int = 0,
        side: Optional[CubeSide] = None,
        mipmap: int = 0,
    ) -> Frame:
        """Get a specific image frame.

        If the texture is a cubemap, a side must be provided and depth must be 0.
        """
        if side is not None and depth != 0:
            raise TypeError('Side and depth are mutually exclusive!')
        depth_side: Union[int, CubeSide]
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
    def from_resource(cls, data: bytes) -> Dict[int, 'SheetSequence']:
        """Decode from the resource data."""
        (
            version,
            sequence_count,
        ) = struct.unpack_from('<II', data)
        offset = 8

        if version > 1:
            raise ValueError('Unknown version {}!'.format(version))

        sequences: Dict[int, SheetSequence] = {}
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
            if not (0 <= seq_num < SheetSequence.MAX_COUNT):
                raise ValueError('Invalid sequence number {}!'.format(seq_num))
            if seq_num in sequences:
                raise ValueError('Duplicate sequence number {}!'.format(seq_num))

            frames: List[Tuple[float, TexCoord, TexCoord, TexCoord, TexCoord]] = []

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

        return sequences

    @classmethod
    def make_data(cls, sequences: Mapping[int, 'SheetSequence'], version: int=0) -> bytes:
        """Write out the binary form of this."""
        file = BytesIO()

        if version > 1:
            raise ValueError('Unknown version {}!'.format(version))

        file.write(struct.pack('<II', version, len(sequences)))
        for seq_num, seq in sequences.items():
            file.write(struct.pack(
                '<Ixxx?If',
                seq_num,
                seq.clamp,
                len(seq.frames),
                seq.duration,
            ))
            for i, (duration, tex_a, tex_b, tex_c, tex_d) in enumerate(seq.frames):
                file.write(struct.pack('<f4f', duration, *tex_a))
                if version == 1: # We have an additional 3 coords.
                    file.write(struct.pack('<4f', *tex_b))
                    file.write(struct.pack('<4f', *tex_c))
                    file.write(struct.pack('<4f', *tex_d))

        return file.getvalue()
