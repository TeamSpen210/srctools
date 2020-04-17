# cython: language_level=3, boundscheck=False, wraparound=False
"""Functions for reading/writing VTF data."""
from cpython cimport array
from libc.stdio cimport snprintf
from libc.stdint cimport uint_least64_t
from libc.stdlib cimport abort, malloc, free
from cython.parallel cimport prange, parallel


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


# These semantically operate differently, but are implemented the same.
def load_rgba8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Parse RGBA-ordered 8888 pixels."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 0]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 2]
        pixels[4 * offset + A] = data[4 * offset + 3]


def load_uvlx8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Parse UVLX data, copying them into RGBA respectively."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 0]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 2]
        pixels[4 * offset + A] = data[4 * offset + 3]


def load_uvwq8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Parse UVWQ data, copying them into RGBA respectively."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 0]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 2]
        pixels[4 * offset + A] = data[4 * offset + 3]


def load_bgra8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + A] = data[4 * offset + 3]

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

def load_bgr888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 2]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 0]
        pixels[4 * offset + A] = 255

def load_bgrx8888(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Strange - skip byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + A] = 255


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


# ImageFormats.P8 is not implemented by Valve either.

def load_a8(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Single alpha bytes."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + R] = 0
        pixels[4*offset + G] = 0
        pixels[4*offset + B] = 0
        pixels[4*offset+3] = data[offset]


def load_uv88(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """UV-only, which is mapped to RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + R] = data[2*offset]
        pixels[4*offset + G] = 0
        pixels[4*offset + B] = 0
        pixels[4*offset + A] = data[2*offset+1]


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


def load_dxt1(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT1 data."""
    load_dxt1_impl(pixels, data, width, height, 255)


def load_dxt1_onebitalpha(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    load_dxt1_impl(pixels, data, width, height, 0)


# Colour table indexes.
DEF C0R = 0
DEF C0G = 1
DEF C0B = 2
DEF C0A = 3

DEF C1R = 4
DEF C1G = 5
DEF C1B = 6
DEF C1A = 7

DEF C2R = 8
DEF C2G = 9
DEF C2B = 10
DEF C2A = 11

DEF C3R = 12
DEF C3G = 13
DEF C3B = 14
DEF C3A = 15

cdef inline int load_dxt1_impl(
    byte[::1] pixels,
    const byte[::1] data,
    uint width,
    uint height,
    byte black_alpha,
) except 1:
    """Does the actual decompression."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')

    cdef uint block_wid, block_off
    cdef Py_ssize_t block_x, block_y  # OpenMP requires this to be signed.
    cdef byte *color_table
    cdef RGB c0, c1

    block_wid = width // 4
    if width % 4:
        block_wid += 1

    # Make each thread have their own local color_table and c0/c1.
    with nogil, parallel():
        c0 = [0, 0, 0]
        c1 = [0, 0, 0]
        color_table = <byte *>malloc(16)
        if color_table is NULL:
            with gil:
                raise MemoryError('Cannot allocate color_table!')
        try:
            # All but the 4th colour are opaque.
            color_table[C0A] = color_table[C1A] = color_table[C2A] = 255

            for block_y in prange(0, height //4, schedule='static'):
                for block_x in range(0, width // 4):
                    block_off = 8 * (block_wid * block_y + block_x)

                    # First, load the 2 colors.
                    decomp565(&c0, data[block_off], data[block_off+1])
                    decomp565(&c1, data[block_off+2], data[block_off+3])

                    color_table[C0B] = c0.r
                    color_table[C0G] = c0.g
                    color_table[C0R] = c0.b

                    color_table[C1B] = c1.r
                    color_table[C1G] = c1.g
                    color_table[C1R] = c1.b

                    # We store the lookup colors as bytes so we can directly copy them.

                    # Equivalent to 16-bit comparison...
                    if (
                        data[block_off] > data[block_off+2] or
                        data[block_off+1] > data[block_off+3]
                    ):
                        color_table[C2R] = (2*c0.b + c1.b) // 3
                        color_table[C2G] = (2*c0.g + c1.g) // 3
                        color_table[C2B] = (2*c0.r + c1.r) // 3

                        color_table[C3R] = (c0.b + 2*c1.b) // 3
                        color_table[C3G] = (c0.g + 2*c1.g) // 3
                        color_table[C3B] = (c0.r + 2*c1.r) // 3
                        color_table[C3A] = 255
                    else:
                        color_table[C2R] = (c0.b + c1.b) // 2
                        color_table[C2G] = (c0.g + c1.g) // 2
                        color_table[C2B] = (c0.r + c1.r) // 2

                        color_table[C3R] = 0
                        color_table[C3G] = 0
                        color_table[C3B] = 0
                        color_table[C3A] = black_alpha

                    dxt_color_table(
                        pixels, data, color_table,
                        block_off, block_wid,
                        block_x, block_y,
                        do_alpha=True,
                    )
        finally:
            free(color_table)


cdef inline void dxt_color_table(
    byte[::1] pixels,
    const byte[::1] data,
    byte *table,
    uint block_off,
    uint block_wid,
    uint block_x,
    uint block_y,
    bint do_alpha,
) nogil:
    """Decodes the actual colour table into pixels."""
    cdef unsigned int row, y
    cdef byte inp, off
    for y in range(4):
        inp = data[block_off + 4 + y]
        row = 16 * block_wid * (4 * block_y + y) + 16 * block_x

        off = 4 * ((inp & 0b11000000) >> 6)
        pixels[row + C3R] = table[off + 0]
        pixels[row + C3G] = table[off + 1]
        pixels[row + C3B] = table[off + 2]
        if do_alpha:
            pixels[row + C3A] = table[off + 3]

        off = 4 * ((inp & 0b00110000) >> 4)
        pixels[row + C2R] = table[off + 0]
        pixels[row + C2G] = table[off + 1]
        pixels[row + C2B] = table[off + 2]
        if do_alpha:
            pixels[row + C2A] = table[off + 3]

        off = 4 * ((inp & 0b00001100) >> 2)
        pixels[row + C1R] = table[off + 0]
        pixels[row + C1G] = table[off + 1]
        pixels[row + C1B] = table[off + 2]
        if do_alpha:
            pixels[row + C1A] = table[off + 3]

        off = 4 * (inp & 0b00000011)
        pixels[row + C0R] = table[off + 0]
        pixels[row + C0G] = table[off + 1]
        pixels[row + C0B] = table[off + 2]
        if do_alpha:
            pixels[row + C0A] = table[off + 3]


def load_dxt3(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT3 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')

    cdef uint block_wid, block_off, block_x, block_y
    cdef uint x, y, off, pos

    cdef byte[16] color_table

    cdef RGB c0, c1
    cdef byte inp

    # All colours are opaque.
    color_table[C0A] = color_table[C1A] = 255
    color_table[C2A] = color_table[C3A] = 255

    block_wid = width // 4
    if width % 4:
        block_wid += 1

    # for block_y in prange(0, height, 4, schedule='static', nogil=True):
    for block_y in range(0, height, 4):
        block_y //= 4
        for block_x in range(0, width, 4):
            block_x //= 4
            block_off = 16 * (block_wid * block_y + block_x)

            # First, load the 2 colors.
            decomp565(&c0, data[block_off + 8], data[block_off + 9])
            decomp565(&c1, data[block_off + 10], data[block_off + 11])

            color_table[C0R] = c0.b
            color_table[C0G] = c0.g
            color_table[C0B] = c0.r

            color_table[C1R] = c1.b
            color_table[C1G] = c1.g
            color_table[C1B] = c1.r

            color_table[C2R] = (2 * c0.b + c1.b) // 3
            color_table[C2G] = (2 * c0.g + c1.g) // 3
            color_table[C2B] = (2 * c0.r + c1.r) // 3

            color_table[C3R] = (c0.b + 2 * c1.b) // 3
            color_table[C3G] = (c0.g + 2 * c1.g) // 3
            color_table[C3B] = (c0.r + 2 * c1.r) // 3

            dxt_color_table(
                pixels, data, color_table,
                block_off + 8, block_wid,
                block_x, block_y,
                do_alpha=False,
            )
            # Now add on the real alpha values.
            for off in range(8):
                inp = data[block_off + off]
                y = off * 2 // 4
                x = off * 2 % 4
                pos = 16 * block_wid * (4 * block_y + y) + 4 * (4 * block_x  + x)
                pixels[pos + A] = inp & 0b00001111 | (inp & 0b00001111) << 4
                pixels[pos + A + 4] = inp & 0b11110000 | (inp & 0b11110000) >> 4


def load_dxt5(byte[::1] pixels, const byte[::1] data, uint width, uint height):
    """Load compressed DXT5 data."""
    if width < 4 or height < 4:
        raise ValueError('DXT format must be 4x4 at minimum!')

    cdef uint block_wid, block_off, block_x, block_y
    cdef uint x, y, i, off, pos

    cdef byte[16] color_table
    cdef byte[8] alpha

    cdef RGB c0, c1
    cdef byte inp

    cdef uint_least64_t lookup  # at least 48 bits!

    # All colours are opaque.
    color_table[C0A] = color_table[C1A] = 255
    color_table[C2A] = color_table[C3A] = 255

    block_wid = width // 4
    if width % 4:
        block_wid += 1

    for block_y in range(0, height, 4):
        block_y //= 4
        for block_x in range(0, width, 4):
            block_x //= 4
            block_off = 16 * (block_wid * block_y + block_x)

            alpha[0] = data[block_off]
            alpha[1] = data[block_off + 1]

            if alpha[0] >= alpha[1]:
                alpha[2] = (6*alpha[0] + 1*alpha[1]) // 7
                alpha[3] = (5*alpha[0] + 2*alpha[1]) // 7
                alpha[4] = (4*alpha[0] + 3*alpha[1]) // 7
                alpha[5] = (3*alpha[0] + 4*alpha[1]) // 7
                alpha[6] = (2*alpha[0] + 5*alpha[1]) // 7
                alpha[7] = (1*alpha[0] + 6*alpha[1]) // 7
            else:
                alpha[2] = (4*alpha[0] + 1*alpha[1]) // 5
                alpha[3] = (3*alpha[0] + 2*alpha[1]) // 5
                alpha[4] = (2*alpha[0] + 3*alpha[1]) // 5
                alpha[5] = (1*alpha[0] + 4*alpha[1]) // 5
                alpha[6] = 0
                alpha[7] = 255

            # Now, load the colour blocks.
            decomp565(&c0, data[block_off + 8], data[block_off + 9])
            decomp565(&c1, data[block_off + 10], data[block_off + 11])

            color_table[C0R] = c0.b
            color_table[C0G] = c0.g
            color_table[C0B] = c0.r

            color_table[C1R] = c1.b
            color_table[C1G] = c1.g
            color_table[C1B] = c1.r

            color_table[C2R] = (2 * c0.b + c1.b) // 3
            color_table[C2G] = (2 * c0.g + c1.g) // 3
            color_table[C2B] = (2 * c0.r + c1.r) // 3

            color_table[C3R] = (c0.b + 2 * c1.b) // 3
            color_table[C3G] = (c0.g + 2 * c1.g) // 3
            color_table[C3B] = (c0.r + 2 * c1.r) // 3

            dxt_color_table(
                pixels, data, color_table,
                block_off+8, block_wid,
                block_x, block_y,
                do_alpha=False,
            )
            # The alpha data is a 48-bit integer, where each 3 bits maps to an alpha
            # value. It's in little-endian order!
            # lookup = int.from_bytes(data[block_off + 2:8], 'little')
            lookup = 0
            for i in range(6):
                lookup |= data[block_off + 2 + 6 - i]
                lookup <<= 8

            for i in range(16):
                y = i // 4
                x = i % 4
                pos = 16 * block_wid * (4 * block_y + y) + 4 * (4 * block_x + x)
                pixels[pos + A] = alpha[(lookup >> (3*i)) & 0b111]

# Don't do the high-def 16-bit resolution.

# def load_rgba16161616f(pixels, offset, data, data_off):
#     """16-bit RGBA format - max resolution."""
#     pixels[offset] = data[data_off] << 8 + data[data_off+1]
#     pixels[offset + 1] = data[data_off+2] << 8 + data[data_off+3]
#     pixels[offset + 2] = data[data_off+4] << 8 + data[data_off+5]
#     pixels[offset + 3] = data[data_off+6] << 8 + data[data_off+7]
