# cython: language_level=3, boundscheck=False, wraparound=False
"""Functions for reading/writing VTF data."""
from libc.stdio cimport snprintf
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from cython.parallel cimport prange, parallel

cdef extern from "squish.h" namespace "squish":
    ctypedef unsigned char u8;
    cdef enum:
        kDxt1 # Use DXT1 compression.
        kDxt3 # Use DXT3 compression.
        kDxt5 # Use DXT5 compression.
        kBc4  # Use BC4 / ATI1n compression.
        kBc5  # Use BC5 / ATI2n compression.
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


def scale_down(
    filt: 'FilterMode',
    uint width, uint height,
    const byte[::1] src, byte[::1] dest,
) -> None:
    """Scale down the image to this smaller size."""
    cdef int filter_val = filt.value
    cdef Py_ssize_t x, y, pos_off, off, off2, channel
    if filter_val in (0, 1, 2, 3):  # Nearest-neighbour.
        # 0 = upper-left, 3 = lower-right
        pos_off = 8 * width if filt.value >= 2 else 0
        if filt.value & 1:
            pos_off += 4
        # for off in range(0, 4 * width * height, 4):
        for y in prange(height, nogil=True, schedule='static'):
            for x in range(width):
                off = 4 * (width * y + x)
                off2 = 8 * (2 * width * y + x)
                for channel in range(4):
                    dest[off + channel] = src[off2 + pos_off + channel]
    elif filter_val == 5:  # Bilinear
        for y in prange(height, nogil=True, schedule='static'):
            for x in range(width):
                off = 4 * (width * y + x)
                off2 = 8 * (2 * width * y + x)
                for channel in range(4):
                    dest[off + channel] = (
                        src[off2 + channel] +
                        src[off2 + channel + 4] +
                        src[off2 + 8 * width + channel] +
                        src[off2 + 8 * width + channel + 4]
                    ) // 4
    else:
        raise ValueError(f"Unknown filter {filt}")


cdef inline byte upsample(byte bits, byte data) nogil:
    """Stretch bits worth of data to fill the byte.

    This is done by duplicating the MSB to fill the remaining space.
    """
    return data | (data >> bits)


cdef inline RGB decomp565(byte a, byte b) nogil:
    """Decompress 565-packed data into RGB triplets."""
    return {
        'r': upsample(5, (a & 0b00011111) << 3),
        'g': upsample(6, ((b & 0b00000111) << 5) | ((a & 0b11100000) >> 3)),
        'b': upsample(5, b & 0b11111000),
    }


cdef inline (byte, byte) compress565(byte r, byte g, byte b) nogil:
    """Compress RGB triplets into 565-packed data."""
    return (
        (r & 0b11111000) | (g >> 5),
        (g << 5) & 0b11100000 | (b >> 3)
    )


# There's a few formats that just do RGBA. This is a special case, since we can just copy across.
# memcpy is going to be more efficient than manual code.
cdef bint load_copy(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Parse RGBA-ordered 8888 pixels."""
    memcpy(&pixels[0], &data[0], 4 * width * height)

cdef bint save_copy(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate RGBA-ordered 8888 pixels."""
    memcpy(&data[0], &pixels[0], 4 * width * height)


cdef bint load_bgra8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Load BGRA format images."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + A] = data[4 * offset + 3]


cdef bint save_bgra8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate BGRA format images."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + B]
        data[4 * offset + 1] = pixels[4 * offset + G]
        data[4 * offset + 2] = pixels[4 * offset + R]
        data[4 * offset + 3] = pixels[4 * offset + A]


# This is totally the wrong order, but it's how it's actually ordered.
cdef bint load_argb8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """This is toally wrong - it's actually in GBAR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 3]
        pixels[4 * offset + G] = data[4 * offset + 0]
        pixels[4 * offset + B] = data[4 * offset + 1]
        pixels[4 * offset + A] = data[4 * offset + 2]


cdef bint save_argb8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """This is toally wrong - it's actually in GBAR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + G]
        data[4 * offset + 1] = pixels[4 * offset + B]
        data[4 * offset + 2] = pixels[4 * offset + A]
        data[4 * offset + 3] = pixels[4 * offset + R]


cdef bint load_abgr8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 3]
        pixels[4 * offset + G] = data[4 * offset + 2]
        pixels[4 * offset + B] = data[4 * offset + 1]
        pixels[4 * offset + A] = data[4 * offset + 0]


cdef bint save_abgr8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate ABGR-ordered data."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + A]
        data[4 * offset + 1] = pixels[4 * offset + B]
        data[4 * offset + 2] = pixels[4 * offset + G]
        data[4 * offset + 3] = pixels[4 * offset + R]


cdef bint load_rgb888(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 0]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 2]
        pixels[4 * offset + A] = 255


cdef bint save_rgb888(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate RGB-format data, discarding alpha."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[3 * offset + 0] = pixels[4 * offset + R]
        data[3 * offset + 1] = pixels[4 * offset + G]
        data[3 * offset + 2] = pixels[4 * offset + B]


cdef bint load_bgr888(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 2]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 0]
        pixels[4 * offset + A] = 255


cdef bint save_bgr888(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate BGR-format data, discarding alpha."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[3 * offset + 0] = pixels[4 * offset + B]
        data[3 * offset + 1] = pixels[4 * offset + G]
        data[3 * offset + 2] = pixels[4 * offset + R]


cdef bint load_bgrx8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Strange - skip byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + A] = 255


cdef bint save_bgrx8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate BGR-format data, with an extra ignored byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + B]
        data[4 * offset + 1] = pixels[4 * offset + G]
        data[4 * offset + 2] = pixels[4 * offset + R]
        data[4 * offset + 3] = 0


cdef bint load_rgb565(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """RGB format, packed into 2 bytes by dropping LSBs."""
    cdef Py_ssize_t offset
    cdef RGB col
    for offset in prange(width * height, nogil=True, schedule='static'):
        col = decomp565(data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset + R] = col.r
        pixels[4 * offset + G] = col.g
        pixels[4 * offset + B] = col.b
        pixels[4 * offset + A] = 255


cdef bint save_rgb565(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate 565-format data, in RGB order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + R],
            pixels[4 * offset + G],
            pixels[4 * offset + B],
        )


cdef bint load_bgr565(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """BGR format, packed into 2 bytes by dropping LSBs."""
    cdef Py_ssize_t offset
    cdef RGB col
    for offset in prange(width * height, nogil=True, schedule='static'):
        col = decomp565(data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset + R] = col.b
        pixels[4 * offset + G] = col.g
        pixels[4 * offset + B] = col.r
        pixels[4 * offset + A] = 255


cdef bint save_bgr565(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate 565-format data, in BGR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + B],
            pixels[4 * offset + G],
            pixels[4 * offset + R],
        )


cdef bint load_bgra4444(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """BGRA format, only upper 4 bits. Bottom half is a copy of the top."""
    cdef Py_ssize_t offset
    cdef byte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + G] = (a & 0b11110000) | (a & 0b11110000) >> 4
        pixels[4 * offset + B] = (a & 0b00001111) | (a & 0b00001111) << 4
        pixels[4 * offset + R] = (b & 0b00001111) | (b & 0b00001111) << 4
        pixels[4 * offset + A] = (b & 0b11110000) | (b & 0b11110000) >> 4


cdef bint save_bgra4444(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate BGRA format images, using only 4 bits each."""
    cdef Py_ssize_t offset
    cdef byte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2 * offset + 0] = (pixels[4 * offset + B] & 0b11110000) | (pixels[4 * offset + G] >> 4)
        data[2 * offset + 1] = (pixels[4 * offset + R] & 0b11110000) | (pixels[4 * offset + A] >> 4)


cdef bint load_bgra5551(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
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


cdef bint save_bgra5551(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate BGRA format images, using 5 bits for color and 1 for alpha."""
    cdef Py_ssize_t offset
    cdef byte r, g, b, a
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4 * offset + R]
        g = pixels[4 * offset + G]
        b = pixels[4 * offset + B]
        a = pixels[4 * offset + A]
        #BBBBBGGG  GGRRRRRA
        data[2 * offset + 0] = (b & 0b11111000) | (g >> 5)
        data[2 * offset + 1] = ((g << 6) & 0b11000000) | ((r >> 2) & 0b00111110) | (a >> 7)


cdef bint load_bgrx5551(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
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


cdef bint save_bgrx5551(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate BGR format images, using 5 bits for color and 1 spare bit."""
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4 * offset + R]
        g = pixels[4 * offset + G]
        b = pixels[4 * offset + B]
        #BBBBBGGG  GGRRRRRX
        data[2 * offset + 0] = (b & 0b11111000) | (g >> 5)
        data[2 * offset + 1] = ((g << 6) & 0b11000000) | ((r >> 2) & 0b00111110)


cdef bint load_i8(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """I8 format, R=G=B"""
    cdef Py_ssize_t offset
    cdef byte color
    for offset in prange(width * height, nogil=True, schedule='static'):
        color = data[offset]
        pixels[4*offset + R] = color
        pixels[4*offset + G] = color
        pixels[4*offset + B] = color
        pixels[4*offset + A] = 255


cdef bint save_i8(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Save in greyscale."""
    cdef Py_ssize_t offset
    cdef byte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4*offset + R]
        g = pixels[4*offset + G]
        b = pixels[4*offset + B]
        data[offset] = (r + g + b) // 3


cdef bint load_ia88(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """I8 format, R=G=B + A"""
    cdef Py_ssize_t offset
    cdef byte color
    for offset in prange(width * height, nogil=True, schedule='static'):
        color = data[2*offset]
        pixels[4*offset + R] = color
        pixels[4*offset + G] = color
        pixels[4*offset + B] = color
        pixels[4*offset+3] = data[2*offset+1]


cdef bint save_ia88(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
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

cdef bint load_a8(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Single alpha bytes."""
    cdef Py_ssize_t offset
    # Set RGB to zero in bulk, instead of doing it in the loop.
    memset(&pixels[0], 0, width * height)
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + A] = data[offset]


cdef bint save_a8(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Save just the alpha channel."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[offset] = pixels[4*offset + A]


cdef bint load_uv88(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """UV-only, which is mapped to RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + R] = data[2*offset]
        pixels[4*offset + G] = data[2*offset+1]
        pixels[4*offset + B] = 0
        pixels[4*offset + A] = 255


cdef bint save_uv88(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Generate UV-format data, using RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset + 0] = pixels[4*offset + R]
        data[2*offset + 1] = pixels[4*offset + G]


cdef bint load_rgb888_bluescreen(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
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


cdef bint save_rgb888_bluescreen(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
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


cdef bint load_bgr888_bluescreen(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
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


cdef bint save_bgr888_bluescreen(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
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


cdef bint load_dxt1(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Load compressed DXT1 data."""
    cdef Py_ssize_t offset

    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    DecompressImage(&pixels[0], width, height, &data[0], kDxt1 | kForceOpaque)


cdef bint save_dxt1(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Save compressed DXT1 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kDxt1 | kForceOpaque, NULL)


cdef bint load_dxt1_alpha(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Load compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    with nogil:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt1)


cdef bint save_dxt1_alpha(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Save compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kDxt1, NULL)


cdef bint load_dxt3(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Load compressed DXT3 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    DecompressImage(&pixels[0], width, height, &data[0], kDxt3)


cdef bint save_dxt3(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Save compressed DXT3 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kDxt3, NULL)


cdef bint load_dxt5(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Load compressed DXT5 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    DecompressImage(&pixels[0], width, height, &data[0], kDxt5)


cdef bint save_dxt5(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Load compressed DXT5 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kDxt5, NULL)


cdef bint load_ati2n(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1:
    """Load 'ATI2N' format data, also known as BC5.

    This uses two copies of the DXT5 alpha block for data.
    """
    if width < 4 or height < 4:
        raise ValueError('ATI2N format must be 4x4 at minimum!')
    DecompressImage(&pixels[0], width, height, &data[0], kBc5)


cdef bint save_ati2n(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1:
    """Save 'ATI2N' format data, also known as BC5.

    This uses two copies of the DXT5 alpha block for data.
    """
    if width < 4 or height < 4:
        raise ValueError('ATI2N format must be 4x4 at minimum!')
    CompressImage(&pixels[0], width, height, &data[0], kBc5, NULL)

# Use a structure to match format names to the functions.
# This way they can all be cdef, and not have duplicate object conversion
# code.
ctypedef struct Format:
    char *name
    bint (*load)(byte[::1] pixels, const byte[::1] data, uint width, uint height) except 1
    bint (*save)(const byte[::1] pixels, byte[::1] data, uint width, uint height) except 1


cdef Format[30] FORMATS = [
    Format("RGBA8888", &load_copy, &save_copy),
    Format("ABGR8888", &load_abgr8888, &save_abgr8888),
    Format("RGB888", &load_rgb888, &save_rgb888),
    Format("BGR888", &load_bgr888, &save_bgr888),
    Format("RGB565", &load_rgb565, &save_rgb565),
    Format("I8", &load_i8, &save_i8),
    Format("IA88", &load_ia88, &save_ia88),
    Format("P8", NULL, NULL),  # Never implemented by Valve.
    Format("A8", &load_a8, &save_a8),
    Format("RGB888_BLUESCREEN", &load_rgb888_bluescreen, &save_rgb888_bluescreen),
    Format("BGR888_BLUESCREEN", &load_bgr888_bluescreen, &save_bgr888_bluescreen),
    Format("ARGB8888", &load_argb8888, &save_argb8888),
    Format("BGRA8888", &load_bgra8888, &save_bgra8888),
    Format("DXT1", &load_dxt1, &save_dxt1),
    Format("DXT3", &load_dxt3, &save_dxt3),
    Format("DXT5", &load_dxt5, &save_dxt5),
    Format("BGRX8888", &load_bgrx8888, &save_bgrx8888),
    Format("BGR565", &load_bgr565, &save_bgr565),
    Format("BGRX5551", &load_bgrx5551, &save_bgrx5551),
    Format("BGRA4444", &load_bgra4444, &save_bgra4444),
    Format("DXT1_ONEBITALPHA", &load_dxt1_alpha, &save_dxt1_alpha),
    Format("BGRA5551", &load_bgra5551, &save_bgra5551),
    Format("UV88", &load_uv88, &save_uv88),
    Format("UVWQ8888", &load_copy, &save_copy),
    # Don't do the high-def 16-bit resolutions.
    Format("RGBA16161616F", NULL, NULL),
    Format("RGBA16161616", NULL, NULL),
    Format("UVLX8888", &load_copy, &save_copy),
    # This doesn't match the actual engine struct, just the order of
    # the Python enum.
    Format("NONE", NULL, NULL),
    Format("ATI1N", NULL, NULL),
    Format("ATI2N", &load_ati2n, &save_ati2n),
]


def init(formats: 'srctools.vtf.ImageFormats') -> None:
    """Verify that the Python enum matches our array of functions."""
    cdef int index
    cdef const char *name
    for fmt in formats:
        index = fmt.index
        assert 0 <= index < (sizeof(FORMATS) // sizeof(Format))
        assert (<str ?>fmt.name).encode('ascii') == FORMATS[index].name, fmt.name


def load(object fmt: 'srctools.vtf.ImageFormats', byte[::1] pixels, const byte[::1] data, uint width, uint height) -> None:
    """Load pixels from data in the given format."""
    cdef int index = fmt.index
    # print("Index: ", index, "< ", (sizeof(FORMATS) // sizeof(Format)))
    if 0 <= index < (sizeof(FORMATS) // sizeof(Format)) and FORMATS[index].load != NULL:
        FORMATS[index].load(pixels, data, width, height)
    else:
        raise NotImplementedError(f"Loading {fmt.name} not implemented!")


def save(object fmt: 'srctools.vtf.ImageFormats', const byte[::1] pixels, byte[::1] data, uint width, uint height) -> None:
    cdef int index = fmt.index
    if 0 <= index < (sizeof(FORMATS) // sizeof(Format)) and FORMATS[index].save != NULL:
        FORMATS[index].save(pixels, data, width, height)
    else:
        raise NotImplementedError(f"Saving {fmt.name} not implemented!")

