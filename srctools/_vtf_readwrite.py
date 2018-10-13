"""Functions for reading/writing VTF data."""
import array
import itertools


def blank(width: int, height: int) -> array.array:
    """Construct a blank image of the desired size."""
    return array.array('B', itertools.repeat(0, times=width * height * 4))


def loader_rgba(mode: str):
    """Make the RGB loader functions."""
    r_off = mode.index('r')
    g_off = mode.index('g')
    b_off = mode.index('b')
    try:
        a_off = mode.index('a')
    except ValueError:
        def loader(pixels, data, width, height):
            for offset in range(width * height):
                pixels[4 * offset] = data[3 * offset + r_off]
                pixels[4 * offset + 1] = data[3 * offset + g_off]
                pixels[4 * offset + 2] = data[3 * offset + b_off]
                pixels[4 * offset + 3] = 255
    else:
        def loader(pixels, data, width, height):
            for offset in range(width * height):
                pixels[4 * offset] = data[4 * offset + r_off]
                pixels[4 * offset + 1] = data[4 * offset + g_off]
                pixels[4 * offset + 2] = data[4 * offset + b_off]
                pixels[4 * offset + 3] = data[4 * offset + a_off]
    return loader


load_rgba8888 = loader_rgba('rgba')
load_bgra8888 = loader_rgba('bgra')

load_argb8888 = loader_rgba('argb')
load_abgr8888 = loader_rgba('abgr')

load_rgb888 = loader_rgba('rgb')
load_bgr888 = loader_rgba('bgr')


# These semantically operate differently, but just have 4 channels.
load_uvlx8888 = loader_rgba('rgba')
load_uvwq8888 = loader_rgba('rgba')


def load_bgrx8888(pixels, data, width, height):
    """Strange - skip byte."""
    for offset in range(width * height):
        pixels[4 * offset] = data[4 * offset + 2]
        pixels[4 * offset + 1] = data[4 * offset + 1]
        pixels[4 * offset + 2] = data[4 * offset + 0]
        pixels[4 * offset + 3] = 255


def load_rgb565(pixels, data, width, height):
    """RGB format, packed into 2 bytes by dropping LSBs."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset] = (a & 0b00011111) << 3
        pixels[4 * offset + 1] = ((b & 0b00000111) << 5) | ((a & 0b11100000) >> 3)
        pixels[4 * offset + 2] = b & 0b11111000
        pixels[4 * offset + 3] = 255


def load_bgr565(pixels, data, width, height):
    """BGR format, packed into 2 bytes by dropping LSBs."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + 2] = (a & 0b00011111) << 3
        pixels[4 * offset + 1] = ((b & 0b00000111) << 5) | ((a & 0b11100000) >> 3)
        pixels[4 * offset] = b & 0b11111000
        pixels[4 * offset+3] = 255


def load_bgra4444(pixels, data, width, height):
    """BGRA format, only upper 4 bits. Bottom half is a copy of the top."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset+1] = (a & 0b11110000) | (a & 0b11110000) >> 4
        pixels[4 * offset+2] = (a & 0b00001111) | (a & 0b00001111) << 4
        pixels[4 * offset] = (b & 0b00001111) | (b & 0b00001111) << 4
        pixels[4 * offset+3] = (b & 0b11110000) | (b & 0b11110000) >> 4


def load_bgra5551(pixels, data, width, height):
    """BGRA format, 5 bits per color plus 1 bit of alpha."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset] = (b & 0b01111100) << 1
        pixels[4 * offset+1] = (a & 0b11100000) >> 2 | (b & 0b00000011) << 6
        pixels[4 * offset+2] = (a & 0b00011111) << 3
        pixels[4 * offset+3] = 255 if b & 0b10000000 else 0


def load_bgrx5551(pixels, data, width, height):
    """BGR format, 5 bits per color, alpha ignored."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset] = (b & 0b01111100) << 1
        pixels[4 * offset+1] = (a & 0b11100000) >> 2 | (b & 0b00000011) << 6
        pixels[4 * offset+2] = (a & 0b00011111) << 3
        pixels[4 * offset+3] = 255


def load_i8(pixels, data, width, height):
    """I8 format, R=G=B"""
    for offset in range(width * height):
        pixels[4*offset] = pixels[4*offset+1] = pixels[4*offset+2] = data[offset]
        pixels[4*offset+3] = 255


def load_ia88(pixels, data, width, height):
    """I8 format, R=G=B + A"""
    for offset in range(width * height):
        pixels[4*offset] = pixels[4*offset+1] = pixels[4*offset+2] = data[2*offset]
        pixels[4*offset+3] = data[2*offset+1]


# ImageFormats.P8 is not implemented by Valve either.

def load_a8(pixels, data, width, height):
    """Single alpha bytes."""
    for offset in range(width * height):
        pixels[4*offset] = pixels[4*offset+1] = pixels[4*offset+2] = 0
        pixels[4*offset+3] = data[offset]


def load_uv88(pixels, data, width, height):
    """UV-only, which is mapped to RG."""
    for offset in range(width * height):
        pixels[4*offset] = data[2*offset]
        pixels[4*offset+1] = data[2*offset+1]
        pixels[4*offset+2] = 0
        pixels[4*offset+3] = 255

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
