"""Class implementing VTF image frames.

This is overwritten by a Cython version wherever possible.
"""
from enum import Enum
from collections import namedtuple
from typing import Dict, Callable

import array
import itertools


# Raw image mode, pixel counts or object(), bytes per pixel.
ImageAlignment = namedtuple("ImageAlignment", 'mode r g b a size')


def f(mode, r=0, g=0, b=0, a=0, *, l=0, size=0):
    """Helper function to construct ImageFormats."""
    if l:
        r = g = b = l
    if not size:
        size = r + g + b + a

    return mode, r, g, b, a, size


class ImageFormats(ImageAlignment, Enum):
    """All VTF image formats, with their data sizes in the value."""
    RGBA8888 = f('RGBA', 8, 8, 8, 8)
    ABGR8888 = f('ABGR', 8, 8, 8, 8)
    RGB888 = f('RGB', 8, 8, 8, 0)
    BGR888 = f('BGR', 8, 8, 8)
    RGB565 = f('RGB;16L', 5, 6, 5, 0)
    I8 = f('L', l=8, a=0)
    IA88 = f('LA', l=8, a=8)
    P8 = f('?')  # Palletted, not used.
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

    # Force object-style comparisons.
    __gt__ = object.__gt__
    __lt__ = object.__lt__
    __ge__ = object.__ge__
    __le__ = object.__le__
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__


del f


class ImageFrame:
    """Image frame. Cannot be created directly."""
    def __init__(
        self,
        width: int,
        height: int,
    ):
        """Create a blank image."""
        self.width = width
        self.height = height
        self._pixels = array.array('b', itertools.repeat(0, times=width*height*4))

    def _load(self, format: ImageFormats, data: bytes):
        """Load in pixels."""
        try:
            loader = LOADERS[format]
        except KeyError:
            raise NotImplementedError(
                "Loading {} not implemented!".format(format.name)
            )
        pixel_size = format.size
        area = self.width * self.height
        pixels = self._pixels
        for offset in range(area):
            loader(pixels, offset, data, pixel_size * offset)


# Format type -> function to decode.
# load(pixels, offset, data, data_off)
LoaderFunc = Callable[[array.array, int, bytes, int], None]
LOADERS = {}  # type: Dict[ImageFormats, LoaderFunc]


def loader(name):
    """Add via a decorator."""
    def adder(func):
        LOADERS[name] = func
        return func
    return adder


def loader_rgba(r_off, g_off, b_off, a_off=None):
    """Make the RGB loader functions."""
    if a_off is None:
        def loader(pixels, offset, data, data_off):
            pixels[offset] = data[data_off + r_off]
            pixels[offset + 1] = data[data_off + g_off]
            pixels[offset + 2] = data[data_off + b_off]
            pixels[offset + 3] = 255
    else:
        def loader(pixels, offset, data, data_off):
            pixels[offset] = data[data_off + r_off]
            pixels[offset + 1] = data[data_off + g_off]
            pixels[offset + 2] = data[data_off + b_off]
            pixels[offset + 3] = data[data_off + a_off]
    return loader


LOADERS[ImageFormats.RGBA8888] = loader_rgba(0, 1, 2, 3)
LOADERS[ImageFormats.ABGR8888] = loader_rgba(3, 2, 1, 0)
LOADERS[ImageFormats.RGB888] = loader_rgba(0, 1, 2)
LOADERS[ImageFormats.BGR888] = loader_rgba(2, 1, 0)
LOADERS[ImageFormats.ARGB8888] = loader_rgba(1, 2, 3, 0)
LOADERS[ImageFormats.BGRA8888] = loader_rgba(2, 1, 0, 3)
LOADERS[ImageFormats.BGRX8888] = loader_rgba(2, 1, 0)  # Size already skips.


@loader(ImageFormats.RGB565)
def load_rgb565(pixels, offset, data, data_off):
    """RGB format, packed into 2 bytes by dropping LSBs."""
    a = data[data_off]
    b = data[data_off + 1]
    pixels[offset] = a & 0b11111000
    pixels[offset+1] = (a & 0b00000111 << 5) + (b & 0b11100000 >> 3)
    pixels[offset+2] = b & 0b00011111 << 3


@loader(ImageFormats.BGR565)
def load_bgr565(pixels, offset, data, data_off):
    """RGB format, packed into 2 bytes by dropping LSBs."""
    a = data[data_off]
    b = data[data_off + 1]
    pixels[offset+2] = a & 0b11111000
    pixels[offset+1] = (a & 0b00000111 << 5) + (b & 0b11100000 >> 3)
    pixels[offset] = b & 0b00011111 << 3


@loader(ImageFormats.BGRA4444)
def load_bgra4444(pixels, offset, data, data_off):
    """BGRA format, only upper 4 bits."""
    a = data[data_off]
    b = data[data_off + 1]
    pixels[offset+2] = a & 0b00001111 << 4
    pixels[offset+1] = a & 0b11110000
    pixels[offset] = b & 0b00001111 << 4
    pixels[offset+3] = b & 0b11110000


@loader(ImageFormats.I8)
def load_I8(pixels, offset, data, data_off):
    """I8 format, R=G=B"""
    pixels[offset] = pixels[offset+1] = pixels[offset+2] = data[data_off]
    pixels[offset+3] = 255


@loader(ImageFormats.IA88)
def load_IA88(pixels, offset, data, data_off):
    """I8 format, R=G=B + A"""
    pixels[offset] = pixels[offset+1] = pixels[offset+2] = data[data_off]
    pixels[offset+3] = data[data_off+1]


# ImageFormats.P8 is not implemented by Valve either.


@loader(ImageFormats.A8)
def load_A8(pixels, offset, data, data_off):
    """Single alpha bytes."""
    pixels[offset] = pixels[offset+1] = pixels[offset+2] = 255
    pixels[offset+3] = data[data_off]

# @loader(ImageFormats.RGB888_BLUESCREEN)
# @loader(ImageFormats.BGR888_BLUESCREEN)
# @loader(ImageFormats.BGRX5551)
# @loader(ImageFormats.BGRA5551)
# @loader(ImageFormats.DXT1)
# @loader(ImageFormats.DXT3)
# @loader(ImageFormats.DXT5)
# @loader(ImageFormats.DXT1_ONEBITALPHA)


# Don't do the high-def 16-bit resolution.

# @loader(ImageFormats.RGBA16161616F)
# def load_16bit(pixels, offset, data, data_off):
#     """16-bit RGBA format - max resolution."""
#     pixels[offset] = data[data_off] << 8 + data[data_off+1]
#     pixels[offset + 1] = data[data_off+2] << 8 + data[data_off+3]
#     pixels[offset + 2] = data[data_off+4] << 8 + data[data_off+5]
#     pixels[offset + 3] = data[data_off+6] << 8 + data[data_off+7]
