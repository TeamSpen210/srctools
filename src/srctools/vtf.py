"""Reads and writes Valve's texture format, VTF.

This is designed to be used with the `Python Imaging Library`_ to do the editing of pixels or
saving/loading standard image files.

To compress to DXT formats, this uses the `libsquish`_ library. Currently, 16-bit HDR formats are
not supported, only metdata can be read.

.. _`Python Imaging Library`: https://pillow.readthedocs.io/en/stable/
.. _`libsquish`: https://sourceforge.net/projects/libsquish/
"""
from typing import TYPE_CHECKING, Any, Optional, Union, overload, ClassVar, final
from array import array
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum, Flag
from io import BytesIO
import itertools
import math
import struct
import types
import warnings

import attrs

from . import EmptyMapping, binformat, Keyvalues
from .const import add_unknown
from .math import AnyVec, FrozenVec, Vec
from .types import FileRSeek, FileWBinarySeek


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
LIBSQUISH_LICENSE = ""

if not TYPE_CHECKING:
    try:
        # noinspection PyUnresolvedReferences, PyProtectedMember
        from . import _cy_vtf_readwrite as _cy_format_funcs
    except ImportError:
        pass
    else:
        _format_funcs = _cy_format_funcs
        LIBSQUISH_LICENSE = _cy_format_funcs.LIBSQUISH_LICENSE


# The _vtf_readwrite module contains save/load functions which
# convert the VTF data into a uniform 32-bit RGBA block, which we can then
# parse.
# That works for everything except RGBA16161616 (used for HDR cubemaps), which
# is 16-bit for each channel. We can't do much about that.

__all__ = [
    'VTF', 'Frame', 'FilterMode', 'Pixel',
    'ResourceID', 'CubeSide', 'ImageFormats', 'VTFFlags',
    'Resource', 'SheetSequence', 'TexCoord', 'HotspotRect',
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
) -> tuple[int, int, int, int, int, int]:
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

        # Cache the format repr, since it's a little complicated to construct.
        # If sections aren't used, don't include them.
        repr_list = [f'<ImageFormats[{ind:02}] {self._name_}:']
        if r or g or b:
            repr_list.append(f' r={r}, g={g}, b={b},')
        if a:
            repr_list.append(f' a={a},')
        repr_list.append(f' size={size}>')
        self._repr = ''.join(repr_list)

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
    # Not implemented, these two don't fit in 8-bit RGBA.
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
        return self._repr

    @property
    def is_compressed(self) -> bool:
        """Checks if the format is compressed in 4x4 blocks."""
        return self.name.startswith('DXT') or self.name in ('ATI1N', 'ATI2N')

    @property
    def is_transparent(self) -> bool:
        """Checks if the format supports transparency."""
        return self in _FORMAT_TRANSPARENT

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
# Initialise the internal mapping in the format modules. For Cython, this validates that the enum
# order matches the C code.
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

# Formats which are transparent. Most with A bits are, then there's
# a few extra special cases in both directions.
_FORMAT_TRANSPARENT = {
    fmt for fmt in ImageFormats
    if fmt.a > 0
} | {
    ImageFormats.BGR888_BLUESCREEN, ImageFormats.RGB888_BLUESCREEN,
    ImageFormats.DXT1_ONEBITALPHA, ImageFormats.DXT5,
} - {ImageFormats.BGRX5551, ImageFormats.BGRX8888}


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
    """For VTF format 7.3+, there is an extensible resource system.

    Any 4-byte ID may be used. These are known IDs, some from Valve and some from elsewhere.
    """
    #: Valve ID. The low-res thumbnail. This is in a fixed position in earlier versions.
    LOW_RES = b'\x01\0\0'
    #: Valve ID. The main image. This is in a fixed position in earlier versions.
    HIGH_RES = b'\x30\0\0'

    #: Valve ID. Used for particle spritesheets, decoded into `~VTF.sheet_info`.
    PARTICLE_SHEET = b'\x10\0\0'
    #: A Cyclic Redundancy Checksum of the source image file.
    CRC = b'CRC'

    #: Allows forcing specific mipmaps to be used for 'medium' shader settings.
    LOD_SETTINGS = b'LOD'

    #: 4 extra bytes of bitflags.
    EXTRA_FLAGS = b'TSO'

    #: Defined by VTFLib, an arbitrary block of keyvalues data.
    KEYVALUES = b'KVD'

    #: Strata Source `extension <https://wiki.stratasource.org/modding/overview/vtf-hotspot-resource>`_,
    #: rectangular regions used to automatically retexture brush faces.
    STRATA_HOTSPOT = b"+\0\0"


class FilterMode(Enum):
    """The algorithm to use for generating mipmaps."""
    NEAREST = UPPER_LEFT = 0  #: Just use the upper-left pixel.
    UPPER_RIGHT = 1  #: Just use the upper-right pixel.
    LOWER_LEFT = 2  #: Just use the lower-left pixel.
    LOWER_RIGHT = 3  #: Just use the lower-right pixel.
    BILINEAR = AVERAGE = 4  #: Average the four pixels together.


@attrs.define
class Resource:
    """An arbitary resource contained in a VTF file.

    This can either be a 32-bit unsigned integer (stored right in the header), or a block of
    binary data.
    """
    flags: int
    data: Union[bytes, int]


# Used as a placeholder definition during saving for non-inline resources we generate directly.
_DUMMY_RESOURCE = Resource(0, b'')


@attrs.frozen
class Pixel:
    """Data structure to hold colour data retrieved from a frame."""
    r: int
    g: int
    b: int
    a: int

    def __iter__(self) -> Iterator[int]:
        yield self.r
        yield self.g
        yield self.b
        yield self.a


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
    _fileinfo: Optional[tuple[FileRSeek[bytes], int, ImageFormats]]

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
                'If passing in a stream, load the VTF before closing '
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

    def fill(self, r: int = 0, g: int = 0, b: int = 0, a: int = 255) -> None:
        """Fill the frame with the specified colour."""
        colour = array('B', [r, g, b, a])
        self._data = colour * (self.width * self.height)
        self._fileinfo = None

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
            else:  # Copy the other array onto us
                self._data[:] = source._data
            self._fileinfo = None
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
            self._fileinfo = None

    def rescale_from(self, larger: 'Frame', filter: FilterMode = FilterMode.BILINEAR) -> None:
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

    def __getitem__(self, item: tuple[int, int]) -> Pixel:
        """Retrieve an individual pixel at (x, y)."""
        self.load()
        assert self._data is not None

        x, y = item
        if x > self.width or y > self.height:
            raise IndexError(item)
        off = (y * self.width + x) * 4
        return Pixel(*self._data[off: off + 4])

    def __setitem__(
        self,
        item: tuple[int, int],
        data: Union[Pixel, tuple[int, int, int, int]],
    ) -> None:
        """Set an individual pixel at (x, y)."""
        self.load()
        assert self._data is not None

        x, y = item
        if x > self.width or y > self.height:
            raise IndexError(item)
        off = (y * self.width + x) * 4
        [
            self._data[off],
            self._data[off + 1],
            self._data[off + 2],
            self._data[off + 3],
        ] = data

    def __buffer__(self, flags: int) -> memoryview:
        """Allow access to the internal buffer of pixels."""
        # We don't need to check the flags, the memoryview itself knows that it's
        # C-contiguous and will deny F-contiguous requests.
        self.load()
        assert self._data is not None
        # Pyright thinks this is typing.cast?
        return memoryview(self._data).cast('B', (self.height, self.width, 4))  # pyright: ignore[reportUndefinedVariable]

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
            self._data,  # type: ignore  # frombuffer() incorrect.
            'raw',
            'RGBA',
            0,
            1,
        ).copy()

    def to_tkinter(
        self,
        tk: 'tkinter.Misc | None' = None,
        *,
        bg: Optional[tuple[int, int, int]] = None,
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
    def to_wx_image(self, bg: Optional[tuple[int, int, int]] = None) -> Any:
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

    def to_wx_bitmap(self, bg: Optional[tuple[int, int, int]] = None) -> Any:
        """Convert the given frame into a wxPython wx.Bitmap.

        This requires wxPython to be installed.
        If bg is set, the image will be composited onto this background.
        Otherwise, alpha is ignored.
        """
        self.load()
        assert self._data is not None
        import wx  # pyright: ignore

        img = wx.Bitmap(self.width, self.height)
        # The WX Bitmap memory layout isn't public, so we have to write to a
        # temporary that it copies from.
        buf = bytearray(3 * self.width * self.height)
        _format_funcs.alpha_flatten(self._data, buf, self.width, self.height, bg)
        img.CopyFromBuffer(buf, wx.BitmapBufferFormat_RGB)
        return img


@final
class VTF:
    """Valve Texture Format files, used in the Source Engine."""
    width: int  #: The width of the texture. This must be a power of two, but does not need to match the height.
    height: int  #: The height of the texture. This must be a power of two, but does not need to match the width.
    #: The "depth" of the texture, used to produce a volumetric texture that has data for a space.
    #: This is mutually exclusive with :py:attr:`VTFFlags.ENVMAP` - it must be 1 for cubemaps.
    depth: int
    mipmap_count: int  #: The total number of mipmaps in the image.

    #: The version number of the file. Supported versions vary from ``(7, 2) - (7, 5)``.
    version: tuple[int, int]
    #: An average of the colors in the texture, used to tint light bounced off surfaces.
    reflectivity: Vec
    #: Indicates how deep a heightmap/bumpmap ranges. Seemingly unused.
    bumpmap_scale: float
    flags: VTFFlags  #: Bitflags specifying behaviours and how the texture was compiled.
    frame_count: int  #: The number of frames, greater than one for an animated texture.
    first_frame_index: int  #: This field appears unused.
    format: ImageFormats  #: The image format to use for the main image.
    #: The image format to use for a small thumbnail, usually ≤ 16×16.
    #: This is usually :py:attr:`DXT1 <srctools.vtf.ImageFormats.DXT1>`.
    low_format: ImageFormats

    #: In version 7.3+, arbitrary resources may be stored in a VTF. :py:class:`ResourceID` specify
    #: known resources, but any 4-byte ID may be used. If you do use a custom resource, keep in
    #: mind this could break if future srctools versions parse this normally.
    resources: dict[Union[ResourceID, bytes], Resource]
    #: Textures used for particle system sprites may have this resource, defining subareas to
    #: randomly pick from when rendering.
    sheet_info: dict[int, 'SheetSequence']
    #: Strata Source adds the hotspot resource, defining regions used to automatically texture
    #: brushes.
    hotspot_info: Optional[list['HotspotRect']]
    #: Implementation-specific flags byte
    hotspot_flags: int

    _frames: dict[tuple[int, Union[CubeSide, int], int], Frame]
    _low_res: Frame

    def __init__(
        self,
        width: int,
        height: int,
        version: tuple[int, int] = (7, 5),
        *,
        ref: AnyVec = FrozenVec(0, 0, 0),
        frames: int = 1,
        bump_scale: float = 1.0,
        sheet_info: Mapping[int, 'SheetSequence'] = EmptyMapping,
        hotspot_info: Optional[list['HotspotRect']] = None,
        hotspot_flags: int = 0,
        flags: VTFFlags = VTFFlags.EMPTY,
        fmt: ImageFormats = ImageFormats.RGBA8888,
        thumb_fmt: ImageFormats = ImageFormats.DXT1,
        depth: int = 1,
    ) -> None:
        """Create a blank VTF file."""
        if not ((7, 2) <= version <= (7, 5)):
            raise ValueError(f'Version must be between 7.2 and 7.5! ({version!r})')
        if not math.log2(width).is_integer():
            raise ValueError(f"Width must be a power of 2! ({width!r}x{height!r})")
        if not math.log2(height).is_integer():
            raise ValueError(f"Height must be a power of 2! ({width!r}x{height!r})")
        if frames < 1:
            raise ValueError(f"Invalid frame count, must be positive! ({frames!r})")

        # If it's a cubemap, depth must be 1.
        if VTFFlags.ENVMAP in flags:
            if depth != 1:
                raise ValueError(f"Cubemaps must have a depth of 1, not {depth!r}")
        elif depth < 1:
            raise ValueError(f"Depth must be positive! ({depth!r})")

        self.width = width
        self.height = height
        self.depth = depth

        self.version = version
        self.reflectivity = Vec(ref)
        self.bumpmap_scale = bump_scale
        self.resources = {}
        self.sheet_info = dict(sheet_info)
        self.hotspot_info = hotspot_info
        self.hotspot_flags = hotspot_flags
        self.flags = flags
        self.frame_count = frames
        self.first_frame_index = 0  # Appears almost unused.
        self.format = fmt
        self.low_format = thumb_fmt

        # (frame, depth/cubemap, mipmap) -> frame
        self._frames = {}
        self._low_res = Frame(16, 16)

        depth_seq = self._depth_range()

        mip_count = 0
        for mip_count in itertools.count():
            for frame in range(frames):
                for cube_or_depth in depth_seq:
                    self._frames[frame, cube_or_depth, mip_count] = Frame(width, height)

            # Once either is 1 large, we have no more mipmaps.
            # Create the frame first, so we still create the final 1-large frame.
            if width <= 1 or height <= 1:
                break

            width >>= 1
            height >>= 1
        self.mipmap_count = mip_count

    @classmethod
    def read(cls: 'type[VTF]', file: FileRSeek[bytes], header_only: bool = False) -> 'VTF':
        """Read in a VTF file.

        :param file: The file to read from, must be seekable.
        :param header_only: If set, only read metadata, skip the frames entirely.
           If accessed the image data will be opaque black.
        """
        signature = file.read(4)
        if signature != b'VTF\0':
            raise ValueError('Bad file signature!')
        version_major, version_minor = struct.unpack('II', file.read(8))

        if version_major != 7 or not (0 <= version_minor <= 5):
            raise ValueError(
                f"VTF version {version_major}.{version_minor} is not between 7.0-7.5!"
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
        vtf.first_frame_index = first_frame_index
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

        depth_seq = vtf._depth_range()

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

            for res_id, resource in vtf.resources.items():
                if not resource.flags & 0x02:
                    # There's actual data elsewhere in the file.
                    offset = resource.data
                    assert isinstance(offset, int)
                    file.seek(offset)
                    [size] = struct.unpack('I', file.read(4))
                    resource.data = file.read(size)

            if ResourceID.PARTICLE_SHEET in vtf.resources:
                res_data = vtf.resources.pop(ResourceID.PARTICLE_SHEET).data
                if isinstance(res_data, int):
                    raise ValueError(f'Integer for particle data? {res_data!r}')
                vtf.sheet_info = SheetSequence.from_resource(res_data)
            if ResourceID.STRATA_HOTSPOT in vtf.resources:
                res_data = vtf.resources.pop(ResourceID.STRATA_HOTSPOT).data
                if isinstance(res_data, int):
                    raise ValueError(f'Integer for hotspot data? {res_data!r}')
                vtf.hotspot_info, vtf.hotspot_flags = HotspotRect.from_resource(res_data)

        else:
            low_res_offset = header_size
            high_res_offset = low_res_offset + low_fmt.frame_size(low_width, low_height)

        if high_res_offset < 0:
            raise ValueError('Missing main image resource!')

        # We don't implement these high-res formats, just return metadata.
        if fmt is ImageFormats.RGBA16161616 or fmt is ImageFormats.RGBA16161616F:
            header_only = True

        vtf._low_res = Frame(low_width, low_height)
        if low_fmt is not ImageFormats.NONE and not header_only:
            if low_res_offset < 0:
                raise ValueError('Missing low-res thumbnail resource!')
            vtf._low_res._fileinfo = (file, low_res_offset, low_fmt)

        for data_mipmap in reversed(range(mipmap_count)):
            mip_width = max(width >> data_mipmap, 1)
            mip_height = max(height >> data_mipmap, 1)
            for frame_ind in range(frame_count):
                for depth_or_cube in depth_seq:
                    frame = vtf._frames[
                        frame_ind,
                        depth_or_cube,
                        data_mipmap,
                    ] = Frame(mip_width, mip_height)
                    if not header_only:
                        # noinspection PyProtectedMember
                        frame._fileinfo = (file, high_res_offset, fmt)
                        high_res_offset += fmt.frame_size(mip_width, mip_height)
        return vtf

    def save(
        self,
        file: FileWBinarySeek,
        version: Optional[tuple[int, int]] = None,
        sheet_seq_version: int = 1,
        asw_or_later: bool = True,
        mip_filter: FilterMode = FilterMode.BILINEAR,
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
                f"VTF version {version_major}.{version_minor} is not between 7.0-7.5!"
            )
        file.write(struct.pack('<II', version_major, version_minor))

        deferred.defer('header_size', '<I')
        file.write(_HEADER.pack(
            0,
            self.width,
            self.height,
            self.flags.value,
            self.frame_count,
            self.first_frame_index,
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

        # Write the resource list. This is slightly complicated by the requirement to keep IDs in
        # ascending order.
        res_list: Optional[list[tuple[Union[ResourceID, bytes], Resource]]]
        if version_minor >= 3:
            # Add dummy definitions for the resources we handle, so they're sorted correctly.
            res_list = [
                *self.resources.items(),
                (ResourceID.LOW_RES, _DUMMY_RESOURCE),
                (ResourceID.HIGH_RES, _DUMMY_RESOURCE),
            ]
            if self.sheet_info:
                res_list.append((ResourceID.PARTICLE_SHEET, _DUMMY_RESOURCE))
            if self.hotspot_info is not None:
                res_list.append((ResourceID.STRATA_HOTSPOT, _DUMMY_RESOURCE))
            file.write(struct.pack('<3xI8x', len(res_list)))

            def res_key(tup: tuple[Union[ResourceID, bytes], Resource]) -> bytes:
                """Resources should be ordered in ascending order."""
                res_id = tup[0]
                if isinstance(res_id, ResourceID):
                    return res_id.value
                else:
                    return res_id

            res_list.sort(key=res_key)

            for res_id, res in res_list:
                raw_res_id = res_id.value if isinstance(res_id, ResourceID) else res_id
                if isinstance(res.data, bytes):
                    # It's later in the file.
                    file.write(struct.pack('<3sB', raw_res_id, res.flags & ~0x02))
                    deferred.defer(('res', res_id), '<I', write=True)
                else:
                    # Just here.
                    file.write(struct.pack('<3sBI', raw_res_id, res.flags | 0x02, res.data))
        else:
            file.write(bytes(15))  # Pad to 80 bytes.
            res_list = None

        deferred.set_data('header_size', file.tell())

        # Now for the main body.
        self.compute_mipmaps(mip_filter)
        self._low_res.load()

        if res_list is not None:  # IE version >= 7.3
            # Write the contents of resource blocks, for those that aren't inline.
            for res_id, res in res_list:
                if not isinstance(res.data, bytes):
                    continue  # Inline block.
                deferred.set_data(('res', res_id), file.tell())
                # Low/high res blocks omit the size.
                if res_id is ResourceID.LOW_RES:
                    self._write_lowres(file)
                elif res_id is ResourceID.HIGH_RES:
                    self._write_highres(file)
                elif res_id is ResourceID.PARTICLE_SHEET and self.sheet_info:
                    particle_data = SheetSequence.make_data(self.sheet_info, sheet_seq_version)
                    file.write(struct.pack('<I', len(particle_data)))
                    file.write(particle_data)
                elif res_id is ResourceID.STRATA_HOTSPOT and self.hotspot_info is not None:
                    hotspot_data = HotspotRect.build_resource(self.hotspot_info, self.hotspot_flags)
                    file.write(struct.pack('<I', len(hotspot_data)))
                    file.write(hotspot_data)
                else:  # Generic block.
                    file.write(struct.pack('<I', len(res.data)))
                    file.write(res.data)
        else:
            self._write_lowres(file)
            self._write_highres(file)

        deferred.write()

    def _write_lowres(self, file: FileWBinarySeek) -> None:
        """Create the low-res image data."""
        if self.low_format is not ImageFormats.NONE:
            data = bytearray(self.low_format.frame_size(self._low_res.width, self._low_res.height))
            if self._low_res._data is not None:
                _format_funcs.save(self.low_format, self._low_res._data, data, self._low_res.width, self._low_res.height)
            file.write(data)

    def _write_highres(self, file: FileWBinarySeek) -> None:
        """Write the high-res image data to this location."""
        depth_seq = self._depth_range()
        for data_mipmap in reversed(range(self.mipmap_count)):
            for frame_ind in range(self.frame_count):
                for depth_or_cube in depth_seq:
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

    def __enter__(self) -> 'VTF':
        """The VTF file can be used as a context manager.

        This will close the streams if any frames still have them open.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        """Close the streams if any frames still have them open."""
        self._low_res._fileinfo = None
        for frame in self._frames.values():
            frame._fileinfo = None

    def _depth_range(self) -> Sequence[Union[int, CubeSide]]:
        """Return the appropriate sequence for iterating over the _frames dict.

        Depending on the type of VTF, frames may either be per cubemap side, or per depth.
        """
        if VTFFlags.ENVMAP in self.flags:
            if self.version[1] >= 5:  # Spheremaps were removed in 7.5+
                return CUBES
            else:
                return CUBES_WITH_SPHERE
        else:
            return range(self.depth)

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

    def compute_mipmaps(self, filter: FilterMode = FilterMode.BILINEAR) -> None:
        """Regenerate all mipmaps that have previously been cleared."""
        depth_seq = self._depth_range()
        for frame_num in range(self.frame_count):
            for depth_side in depth_seq:
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
            side = CubeSide.FRONT if VTFFlags.ENVMAP in self.flags else None
            for mipmap in range(self.mipmap_count):
                frame = self.get(mipmap=mipmap, side=side)
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


@attrs.frozen
class TexCoord:
    """Sub-frame used for particle textures."""
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_binary(cls, buffer: bytes, offset: int) -> 'TexCoord':
        """Parse from the SheetSequence resource data."""
        data = struct.unpack_from('<4f', buffer, offset)
        return cls(*data)

    def to_binary(self) -> bytes:
        """Return the bytes form for the texture coordinate."""
        return struct.pack('<4f', self.left, self.top, self.right, self.bottom)


class SheetSequence:
    """VTFs may contain a number of sequences using different parts of the image."""
    MAX_COUNT = 64  #: Maximum possible number of sequences.

    def __init__(
        self,
        frames: list[tuple[float, TexCoord, TexCoord, TexCoord, TexCoord]],
        clamp: bool,
        duration: float,
    ) -> None:
        self.frames = frames
        self.clamp = clamp
        self.duration = duration

    @classmethod
    def from_resource(cls, data: bytes) -> dict[int, 'SheetSequence']:
        """Decode from the resource data."""
        (
            version,
            sequence_count,
        ) = struct.unpack_from('<II', data)
        offset = 8

        if version > 1:
            raise ValueError(f'Unknown version {version}!')

        sequences: dict[int, SheetSequence] = {}
        if sequence_count > SheetSequence.MAX_COUNT:
            raise ValueError(
                f'Cannot have more than {SheetSequence.MAX_COUNT} '
                f'sequences ({sequence_count})!'
            )

        for _ in range(sequence_count):
            (
                seq_num,
                clamp,
                frame_count,
                total_time,
            ) = struct.unpack_from('<Ixxx?If', data, offset)
            offset += 16
            if not (0 <= seq_num < SheetSequence.MAX_COUNT):
                raise ValueError(f'Invalid sequence number {seq_num}!')
            if seq_num in sequences:
                raise ValueError(f'Duplicate sequence number {seq_num}!')

            frames: list[tuple[float, TexCoord, TexCoord, TexCoord, TexCoord]] = []

            for frame_ind in range(frame_count):
                [duration] = struct.unpack_from('<f', data, offset)
                offset += 4

                if version == 0:
                    # Only one in the file, repeated 4 times.
                    tex_coord = TexCoord.from_binary(data, offset)
                    frames.append((duration, tex_coord, tex_coord, tex_coord, tex_coord))
                    offset += 16
                else:
                    frames.append((
                        duration,
                        TexCoord.from_binary(data, offset),
                        TexCoord.from_binary(data, offset + 16),
                        TexCoord.from_binary(data, offset + 32),
                        TexCoord.from_binary(data, offset + 48),
                    ))
                    offset += 64

            sequences[seq_num] = SheetSequence(frames, clamp, total_time)

        return sequences

    @classmethod
    def make_data(cls, sequences: Mapping[int, 'SheetSequence'], version: int = 0) -> bytes:
        """Write out the binary form of this."""
        file = BytesIO()

        if version > 1:
            raise ValueError(f'Unknown version {version}!')

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
                file.write(struct.pack('<f', duration))
                file.write(tex_a.to_binary())
                if version == 1:  # We have an additional 3 coords.
                    file.write(tex_b.to_binary())
                    file.write(tex_c.to_binary())
                    file.write(tex_d.to_binary())

        return file.getvalue()


@attrs.define
class HotspotRect:
    """A set of rectangular regions used to automatically retexture brushes.

    There are two methods to define this format. Hammer++ uses ``.rect`` keyvalues files, while
    Strata Source also allows a binary resource embedded in the VTF

    Only one version of the VTF format exists, ``v0x1``.
    """
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    #: Can the region be rotated randomly?
    random_rotation: bool = attrs.field(kw_only=True, default=False)
    #: Can the region be flipped horizontally?
    random_reflection: bool = attrs.field(kw_only=True, default=False)
    #: If enabled, this is an alternate region. If a modifier key is held, these regions are used
    # instead of the non-alternate ones.
    is_alternate: bool = attrs.field(kw_only=True, default=False)

    _ST_HEAD: ClassVar[struct.Struct] = struct.Struct('<BBH')
    _ST_RECT: ClassVar[struct.Struct] = struct.Struct('<B4H')

    @classmethod
    def from_resource(cls, data: bytes) -> tuple[list['HotspotRect'], int]:
        """Parse from the VTF resource data.

        This returns the list of regions, and an arbitrary implementation-specific flags byte.
        """
        if data[0] != 0x1:
            raise ValueError(f'Invalid hotspot version byte {data[0]:02X}, only 0x1 is valid.')
        (version, impl_flags, rect_count) = cls._ST_HEAD.unpack_from(data, 0)
        off = cls._ST_HEAD.size
        rects: list[HotspotRect] = []
        for _ in range(rect_count):
            (flags, min_x, min_y, max_x, max_y) = cls._ST_RECT.unpack_from(data, off)
            off += cls._ST_RECT.size
            rects.append(cls(
                random_rotation=flags & 0x1 != 0,
                random_reflection=flags & 0x2 != 0,
                is_alternate=flags & 0x4 != 0,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            ))

        return rects, impl_flags

    @classmethod
    def build_resource(cls, hotspots: Sequence['HotspotRect'], flags: int, version: int = 1) -> bytes:
        """Write out the VTF resource data.

        :param hotspots: The regions to write.
        :param flags: Implementation-specific flags.
        :param version: Format version, currently only ``0x1`` exists.
        """
        if version != 1:
            raise ValueError(f'Invalid hotspot version {version!r}, only 0x1 is valid.')
        buf = BytesIO()
        buf.write(cls._ST_HEAD.pack(1, flags, len(hotspots)))
        for rect in hotspots:
            buf.write(cls._ST_RECT.pack(
                (
                    0x1 * rect.random_rotation |
                    0x2 * rect.random_reflection |
                    0x4 * rect.is_alternate
                ),
                rect.min_x, rect.min_y, rect.max_x, rect.max_y,
            ))

        return buf.getvalue()

    @classmethod
    def parse_rect(cls, kv: Keyvalues) -> list['HotspotRect']:
        """Parse a ``.rect`` file keyvalues block."""
        if kv.is_root():  # Might have been passed a root KV containing this.
            kv = kv.find_block('Rectangles')
        rects = []
        for child in kv.find_all('rectangle'):
            mins = child['min']
            try:
                [a, b] = mins.split()
                min_x, min_y = int(a), int(b)
            except ValueError as exc:
                raise ValueError(f'Invalid mins value "{mins}"!') from exc
            maxs = child['max']
            try:
                [a, b] = maxs.split()
                max_x, max_y = int(a), int(b)
            except ValueError as exc:
                raise ValueError(f'Invalid maxes value "{maxs}"!') from exc
            rects.append(cls(
                random_rotation=child.bool('rotate'),
                random_reflection=child.bool('reflect'),
                is_alternate=child.bool('alt'),
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            ))
        return rects

    @classmethod
    def to_kv(cls, hotspots: Sequence['HotspotRect']) -> Keyvalues:
        """Rebuild a ``.rect`` keyvalues file."""
        root = Keyvalues('Rectangles', [])
        for rect in hotspots:
            kv = Keyvalues('rectangle', [
                Keyvalues('min', f'{rect.min_x} {rect.min_y}'),
                Keyvalues('max', f'{rect.max_x} {rect.max_y}'),
            ])
            if rect.random_rotation:
                kv.append(Keyvalues('rotate', '1'))
            if rect.random_reflection:
                kv.append(Keyvalues('reflect', '1'))
            if rect.is_alternate:
                kv.append(Keyvalues('alt', '1'))
            root.append(kv)
        return root
