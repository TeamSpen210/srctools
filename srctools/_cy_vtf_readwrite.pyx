# cython: language_level=3, boundscheck=False, wraparound=False
"""Functions for reading/writing VTF data."""
from cpython cimport array
from libc.stdio cimport snprintf
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cython.parallel cimport prange, parallel

cdef extern from "squish.h" namespace "squish":
    ctypedef unsigned char u8;
    cdef enum:
        kDxt1 # Use DXT1 compression.
        kDxt3 # Use DXT3 compression.
        kDxt5 # Use DXT5 compression.
        kBc4  # Use BC4 compression.
        kBc5  # Use BC5 compression.
        kColourClusterFit # Use a slow but high quality colour compressor (the default).
        kColourRangeFit # Use a fast but low quality colour compressor.
        kWeightColourByAlpha # Weight the colour by alpha during cluster fit (disabled by default).
        kColourIterativeClusterFit # Use a very slow but very high quality colour compressor.
        kSourceBGRA # Source is BGRA rather than RGBA
        kForceOpaque # Force alpha to be opaque

    # void CompressImage(u8 *rgba, int width, int height, int pitch, void *blocks, int flags, float *metric);
    void CompressImage(u8 *rgba, int width, int height, void *blocks, int flags, float *metric) nogil;

    # void DecompressImage(u8 *rgba, int width, int height, int pitch, void *blocks, int flags );
    void DecompressImage(u8 *rgba, int width, int height, void *blocks, int flags ) nogil;

cdef object img_template = array.array('B')
ctypedef unsigned char byte
ctypedef unsigned int uint

cdef struct RGB:
    byte r
    byte g
    byte b

# Offsets for the colour channels.
DEF R = 0
DEF G = 1
DEF B = 2
DEF A = 3

# We specify all the arrays are C-contiguous, since we're the only one using
# these functions directly.

def blank(uint width, uint height):
    """Construct a blank image of the desired size."""
    return array.clone(img_template, 4 * width * height, zero=True)


def ppm_convert(const byte[::1] pixels, uint width, uint height):
    """Convert a frame into a PPM-format bytestring, for passing to tkinter."""
    cdef uint img_off, off
    cdef Py_ssize_t size = 3 * width * height

    # b'P6 65536 65536 255\n' is 19 characters long.
    # We shouldn't get a larger frame than that, it's already absurd.
    cdef byte *buffer = <byte *> malloc(size + 19)
    try:
        img_off = snprintf(<char *>buffer, 19, b'P6 %u %u 255\n', width, height)

        if img_off < 0: # If it does fail just produce a blank file.
            return b''

        for off in range(width * height):
            buffer[img_off + 3*off + R] = pixels[4*off]
            buffer[img_off + 3*off + G] = pixels[4*off+1]
            buffer[img_off + 3*off + B] = pixels[4*off+2]

        return buffer[:size+img_off]
    finally:
        free(buffer)


cdef inline byte upsample(byte bits, byte data) nogil:
    """Stretch bits worth of data to fill the byte.

    This is done by duplicating the MSB to fill the remaining space.
    """
    return data | (data >> bits)


cdef inline void decomp565(RGB *rgb, byte a, byte b) nogil:
    """Decompress 565-packed data into RGB triplets."""
    rgb.r = upsample(5, (a & 0b00011111) << 3)
    rgb.g = upsample(6, ((b & 0b00000111) << 5) | ((a & 0b11100000) >> 3))
    rgb.b = upsample(5, b & 0b11111000)


cdef inline (byte, byte) compress565(byte r, byte g, byte b) nogil:
    """Compress RGB triplets into 565-packed data."""
    return (
        (r & 0b11111000) | (g >> 5),
        (g << 5) & 0b11100000 | (b >> 3)
    )


# These semantically operate differently, but are implemented the same.
# They're a special case, since we can just copy across.
def load_rgba8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Parse RGBA-ordered 8888 pixels."""
    memcpy(&pixels[0], &data[0], 4 * width * height)

def save_rgba8888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate RGBA-ordered 8888 pixels."""
    memcpy(&data[0], &pixels[0], 4 * width * height)


def load_uvlx8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Parse UVLX data, copying them into RGBA respectively."""
    memcpy(&pixels[0], &data[0], 4 * width * height)


def save_uvlx8888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate UVLX data, by copying RGBA data in that order."""
    memcpy(&data[0], &pixels[0], 4 * width * height)


def load_uvwq8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Parse UVWQ data, copying them into RGBA respectively."""
    memcpy(&pixels[0], &data[0], 4 * width * height)


def save_uvwq8888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate UVWQ data, by copying RGBA data in that order."""
    memcpy(&data[0], &pixels[0], 4 * width * height)


def load_bgra8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load BGRA format images."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + A] = data[4 * offset + 3]


def save_bgra8888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate BGRA format images."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + B]
        data[4 * offset + 1] = pixels[4 * offset + G]
        data[4 * offset + 2] = pixels[4 * offset + R]
        data[4 * offset + 3] = pixels[4 * offset + A]


# This is totally the wrong order, but it's how it's actually ordered.
def load_argb8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """This is toally wrong - it's actually in GBAR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 3]
        pixels[4 * offset + G] = data[4 * offset + 0]
        pixels[4 * offset + B] = data[4 * offset + 1]
        pixels[4 * offset + A] = data[4 * offset + 2]


def load_abgr8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 3]
        pixels[4 * offset + G] = data[4 * offset + 2]
        pixels[4 * offset + B] = data[4 * offset + 1]
        pixels[4 * offset + A] = data[4 * offset + 0]


def load_rgb888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 0]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 2]
        pixels[4 * offset + A] = 255


def save_rgb888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate RGB-format data, discarding alpha."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[3 * offset + 0] = pixels[4 * offset + R]
        data[3 * offset + 1] = pixels[4 * offset + G]
        data[3 * offset + 2] = pixels[4 * offset + B]


def load_bgr888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 2]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 0]
        pixels[4 * offset + A] = 255


def save_bgr888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate BGR-format data, discarding alpha."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[3 * offset + 0] = pixels[4 * offset + B]
        data[3 * offset + 1] = pixels[4 * offset + G]
        data[3 * offset + 2] = pixels[4 * offset + R]


def load_bgrx8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Strange - skip byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + A] = 255


def save_bgrx8888(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate BGR-format data, with an extra ignored byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + B]
        data[4 * offset + 1] = pixels[4 * offset + G]
        data[4 * offset + 2] = pixels[4 * offset + R]
        data[4 * offset + 3] = 0


def load_rgb565(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """RGB format, packed into 2 bytes by dropping LSBs."""
    cdef Py_ssize_t offset
    cdef RGB col
    for offset in prange(width * height, nogil=True, schedule='static'):
        decomp565(&col, data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset + R] = col.r
        pixels[4 * offset + G] = col.g
        pixels[4 * offset + B] = col.b
        pixels[4 * offset + A] = 255


def save_rgb565(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate 565-format data, in RGB order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + R],
            pixels[4 * offset + G],
            pixels[4 * offset + B],
        )


def load_bgr565(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """BGR format, packed into 2 bytes by dropping LSBs."""
    cdef Py_ssize_t offset
    cdef RGB col
    for offset in prange(width * height, nogil=True, schedule='static'):
        decomp565(&col, data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset + R] = col.b
        pixels[4 * offset + G] = col.g
        pixels[4 * offset + B] = col.r
        pixels[4 * offset + A] = 255


def save_bgr565(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate 565-format data, in BGR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + B],
            pixels[4 * offset + G],
            pixels[4 * offset + R],
        )


def load_bgra4444(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """BGRA format, only upper 4 bits. Bottom half is a copy of the top."""
    cdef Py_ssize_t offset
    cdef byte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + B] = (a & 0b11110000) | (a & 0b11110000) >> 4
        pixels[4 * offset + G] = (a & 0b00001111) | (a & 0b00001111) << 4
        pixels[4 * offset + R] = (b & 0b00001111) | (b & 0b00001111) << 4
        pixels[4 * offset + A] = (b & 0b11110000) | (b & 0b11110000) >> 4


def load_bgra5551(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """BGRA format, 5 bits per color plus 1 bit of alpha."""
    cdef Py_ssize_t offset
    cdef byte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + R] = upsample(5, (b & 0b01111100) << 1)
        pixels[4 * offset + G] = upsample(5, (a & 0b11100000) >> 2 | (b & 0b00000011) << 6)
        pixels[4 * offset + B] = upsample(5, (a & 0b00011111) << 3)
        pixels[4 * offset + A] = 255 if b & 0b10000000 else 0


def load_bgrx5551(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """BGR format, 5 bits per color, alpha ignored."""
    cdef Py_ssize_t offset
    cdef byte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + R] = upsample(5, (b & 0b01111100) << 1)
        pixels[4 * offset + G] = upsample(5, (a & 0b11100000) >> 2 | (b & 0b00000011) << 6)
        pixels[4 * offset + B] = upsample(5, (a & 0b00011111) << 3)
        pixels[4 * offset + A] = 255


def load_i8(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """I8 format, R=G=B"""
    cdef Py_ssize_t offset
    cdef byte color
    for offset in prange(width * height, nogil=True, schedule='static'):
        color = data[offset]
        pixels[4*offset + R] = color
        pixels[4*offset + G] = color
        pixels[4*offset + B] = color
        pixels[4*offset + A] = 255


def save_i8(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Save in greyscale."""
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4*offset + R]
        g = pixels[4*offset + G]
        b = pixels[4*offset + B]
        data[offset] = (r + g + b) // 3


def load_ia88(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """I8 format, R=G=B + A"""
    cdef Py_ssize_t offset
    cdef byte color
    for offset in prange(width * height, nogil=True, schedule='static'):
        color = data[2*offset]
        pixels[4*offset + R] = color
        pixels[4*offset + G] = color
        pixels[4*offset + B] = color
        pixels[4*offset+3] = data[2*offset+1]


def save_ia88(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Save in greyscale, with alpha."""
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4*offset + R]
        g = pixels[4*offset + G]
        b = pixels[4*offset + B]
        data[2 * offset + 0] = (r + g + b) // 3
        data[2 * offset + 1] = pixels[4*offset + A]


# ImageFormats.P8 is not implemented by Valve either.

def load_a8(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Single alpha bytes."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + R] = 0
        pixels[4*offset + G] = 0
        pixels[4*offset + B] = 0
        pixels[4*offset+3] = data[offset]


def save_a8(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Save just the alpha channel."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[offset] = pixels[4*offset + A]


def load_uv88(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """UV-only, which is mapped to RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + R] = data[2*offset]
        pixels[4*offset + G] = 0
        pixels[4*offset + B] = 0
        pixels[4*offset + A] = data[2*offset+1]


def save_uv88(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate UV-format data, using RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset + 0] = pixels[4*offset + R]
        data[2*offset + 1] = pixels[4*offset + A]


def load_rgb888_bluescreen(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """RGB format, with 'bluescreen' mode for alpha.

    Pure blue pixels are transparent.
    """
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = data[3 * offset]
        g = data[3 * offset + 1]
        b = data[3 * offset + 2]
        if r == g == 0 and b == 255:
            pixels[4*offset] = pixels[4*offset+1] = 0
            pixels[4*offset+2] = pixels[4*offset+3] = 0
        else:
            pixels[4 * offset + R] = r
            pixels[4 * offset + G] = g
            pixels[4 * offset + B] = b
            pixels[4 * offset + A] = 255


def save_rgb888_bluescreen(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate RGB format, using pure blue for transparent pixels."""
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        if pixels[4*offset + A] < 128:
            data[3 * offset + 0] = 0
            data[3 * offset + 1] = 0
            data[3 * offset + 2] = 255
        else:
            data[3 * offset + 0] = pixels[4*offset + R]
            data[3 * offset + 1] = pixels[4*offset + G]
            data[3 * offset + 2] = pixels[4*offset + B]


def load_bgr888_bluescreen(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """BGR format, with 'bluescreen' mode for alpha.

    Pure blue pixels are transparent.
    """
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = data[3 * offset + 2]
        g = data[3 * offset + 1]
        b = data[3 * offset]
        if r == g == 0 and b == 255:
            pixels[4*offset] = pixels[4*offset+1] = 0
            pixels[4*offset+2] = pixels[4*offset+3] = 0
        else:
            pixels[4 * offset + R] = r
            pixels[4 * offset + G] = g
            pixels[4 * offset + B] = b
            pixels[4 * offset + A] = 255


def save_bgr888_bluescreen(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Generate BGR format, using pure blue for transparent pixels."""
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        if pixels[4*offset + A] < 128:
            data[3 * offset + 0] = 255
            data[3 * offset + 1] = 0
            data[3 * offset + 2] = 0
        else:
            data[3 * offset + 0] = pixels[4*offset + B]
            data[3 * offset + 1] = pixels[4*offset + G]
            data[3 * offset + 2] = pixels[4*offset + R]


def load_dxt1(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT1 data."""
    cdef Py_ssize_t offset

    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    with nogil:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt1)

        # Force back to 100% alpha.
        for offset in prange(width * height, schedule='static'):
            pixels[4 * offset + 3] = 255


def load_dxt1_onebitalpha(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    with nogil:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt1)


def load_dxt3(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT3 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    DecompressImage(&pixels[0], width, height, &data[0], kDxt3)


def save_dxt3(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Save compressed DXT3 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kDxt3, NULL)


def load_dxt5(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT5 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    DecompressImage(&pixels[0], width, height, &data[0], kDxt5)


def save_dxt5(const byte[::1] pixels, byte[::1] data, uint width, uint height):
    """Load compressed DXT5 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kDxt5, NULL)


def load_ati2n(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load 'ATI2N' format data, also known as BC5.

    This uses two copies of the DXT5 alpha block for data.
    """
    if width < 4 or height < 4:
        raise ValueError('ATI2N format must be 4x4 at minimum!')
    raise NotImplementedError


# Don't do the high-def 16-bit resolution.

# def load_rgba16161616f(pixels, offset, data, data_off):
#     """16-bit RGBA format - max resolution."""
#     pixels[offset] = data[data_off] << 8 + data[data_off+1]
#     pixels[offset + 1] = data[data_off+2] << 8 + data[data_off+3]
#     pixels[offset + 2] = data[data_off+4] << 8 + data[data_off+5]
#     pixels[offset + 3] = data[data_off+6] << 8 + data[data_off+7]
