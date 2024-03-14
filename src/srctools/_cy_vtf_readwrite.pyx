# cython: language_level=3, boundscheck=False, wraparound=False
"""Functions for reading/writing VTF data."""
from cpython.bytes cimport PyBytes_FromStringAndSize
from cython.parallel cimport parallel, prange
from libc.stdint cimport uint8_t as byte, uint_fast8_t as fastbyte, uint_fast16_t
from libc.stdio cimport snprintf
from libc.string cimport memcpy, memset, strcmp


LIBSQUISH_LICENSE = """\
Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the 
"Software"), to	deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to 
permit persons to whom the Software is furnished to do so, subject to 
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


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

ctypedef unsigned int uint

cdef struct RGB:
    fastbyte r
    fastbyte g
    fastbyte b

# Offsets for the colour channels.
# TODO: Swap to constants when Cython supports that.
cdef enum:
    R = 0
    G = 1
    B = 2
    A = 3

cdef char *EMPTY_BUFFER = b""


# We specify all the arrays are C-contiguous, since we're the only one using
# these functions directly.
def ppm_convert(const byte[::1] pixels, uint width, uint height, tuple bg or None):
    """Convert a frame into a PPM-format bytestring, for passing to tkinter."""
    cdef float r, g, b
    cdef float a, inv_a
    cdef Py_ssize_t off
    cdef Py_ssize_t size = 3 * width * height

    if size == 0:  # snprintf() wants to write a null terminator
        size = 1

    cdef const char * PPM_HEADER = b'P6 %u %u 255\n'
    cdef Py_ssize_t header_size = snprintf(EMPTY_BUFFER, 0, PPM_HEADER, width, height)
    assert header_size > 0, "Bad format string"
    # Allocate an uninitialised bytes object, that we can write to it.
    # That's allowed as long as we don't give anyone else access.
    cdef bytes result = PyBytes_FromStringAndSize(NULL, size + header_size)
    cdef byte *buffer = result

    snprintf(<char *>buffer, header_size + 1, PPM_HEADER, width, height)
    if bg is not None:
        if len(bg) != 3:
            raise ValueError('Background must be a 3-tuple!')
        r = bg[0]
        g = bg[1]
        b = bg[2]
        for off in prange(width * height, nogil=True, schedule='static'):
            a = pixels[4 * off + 3] / <float>255.0
            inv_a = <float>1.0 - a
            buffer[header_size + 3*off + R] = <byte> (pixels[4*off] * a + inv_a * r)
            buffer[header_size + 3*off + G] = <byte> (pixels[4*off + 1] * a + inv_a * g)
            buffer[header_size + 3*off + B] = <byte> (pixels[4*off + 2] * a + inv_a * b)
    else:
        for off in prange(width * height, nogil=True, schedule='static'):
            buffer[header_size + 3*off + R] = pixels[4*off]
            buffer[header_size + 3*off + G] = pixels[4*off + 1]
            buffer[header_size + 3*off + B] = pixels[4*off + 2]

    return result


def alpha_flatten(const byte[::1] pixels, byte[::1] buffer, uint width, uint height, tuple bg or None):
    """Flatten the image down to RGB, by removing the alpha channel.

    If bg is set, this is the background we composite into. Otherwise we
    just strip the alpha.
    """
    cdef float r, g, b
    cdef float a, inv_a
    cdef Py_ssize_t off
    cdef Py_ssize_t size = 3 * width * height

    if bg is not None:
        if len(bg) != 3:
            raise ValueError('Background must be a 3-tuple!')
        r = bg[0]
        g = bg[1]
        b = bg[2]
        for off in prange(width * height, nogil=True, schedule='static'):
            a = pixels[4*off + 3] / <float>255.0
            inv_a = <float>1.0 - a
            buffer[3*off + R] = <byte>(pixels[4*off] * a + inv_a * r)
            buffer[3*off + G] = <byte>(pixels[4*off + 1] * a + inv_a * g)
            buffer[3*off + B] = <byte>(pixels[4*off + 2] * a + inv_a * b)
    else:
        for off in prange(width * height, nogil=True, schedule='static'):
            buffer[3*off + R] = pixels[4*off]
            buffer[3*off + G] = pixels[4*off + 1]
            buffer[3*off + B] = pixels[4*off + 2]


def scale_down(
    filt: 'FilterMode',
    uint src_width, uint src_height,
    uint width, uint height,
    const byte[::1] src, byte[::1] dest,
) -> None:
    """Scale down the image to this smaller size.

    This is simplified for mipmap generation only:
    either dimension may be the same, or be scaled exactly half.
    """
    cdef int filter_val = filt.value
    cdef Py_ssize_t x, y, pos_off, off, off2, channel
    cdef Py_ssize_t vert_off, horiz_off, per_row, per_column

    # We allow the dimensions to remain the same.
    # So figure out the offsets we need to pick the right pixels.
    # per_row/column is the multiples needed to skip to the upper-left pixel.
    # horiz/vertical_off is the offset to the lower-right pixel in each dimension.
    if width != src_width:
        horiz_off, per_column = 4, 2
    else:
        horiz_off, per_column = 0, 1
    if height != src_height:
        vert_off, per_row = 4 * per_column * width, 2 * per_column * width
    else:
        vert_off, per_row = 0, per_column * width

    if filter_val == 4:  # Bilinear
        for y in prange(height, nogil=True, schedule='static'):
            for x in range(width):
                off = 4 * (width * y + x)
                off2 = 4 * (per_row * y + per_column * x)
                for channel in range(4):
                    dest[off + channel] = <byte>((
                        src[off2 + channel] +
                        src[off2 + channel + horiz_off] +
                        src[off2 + channel + vert_off] +
                        src[off2 + channel + vert_off + horiz_off]
                    ) // <uint_fast16_t>4)
        return

    # Otherwise, nearest-neighbour.
    elif filter_val == 0:  # upper-left
        pos_off = 0
    elif filter_val == 1:  # upper-right
        pos_off = horiz_off
    elif filter_val == 2:  # lower-left
        pos_off = vert_off
    elif filter_val == 3:  # lower-right
        pos_off = vert_off + horiz_off
    else:
        raise ValueError(f"Unknown filter {filt}")

    # for off in range(0, 4 * width * height, 4):
    for y in prange(height, nogil=True, schedule='static'):
        for x in range(width):
            off = 4 * (width * y + x)
            off2 = 4 * (per_row * y + per_column * x)
            for channel in range(4):
                dest[off + channel] = src[off2 + pos_off + channel]


cdef inline byte upsample(byte bits, byte data) noexcept nogil:
    """Stretch bits worth of data to fill the byte.

    This is done by duplicating the MSB to fill the remaining space.
    """
    return data | (data >> bits)


cdef inline RGB decomp565(byte a, byte b) noexcept nogil:
    """Decompress 565-packed data into RGB triplets."""
    return {
        'r': upsample(5, (a & 0b00011111) << 3),
        'g': upsample(6, ((b & 0b00000111) << 5) | ((a & 0b11100000) >> 3)),
        'b': upsample(5, b & 0b11111000),
    }


cdef inline (byte, byte) compress565(byte r, byte g, byte b) noexcept nogil:
    """Compress RGB triplets into 565-packed data."""
    return (
        (g << 3) & 0b11100000 | (b >> 3),
        (r & 0b11111000) | (g >> 5),
    )


# There's a few formats that just do RGBA. This is a special case, since we can just copy across.
# memcpy is going to be more efficient than manual code.
cdef bint load_copy(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Parse RGBA-ordered 8888 pixels."""
    memcpy(&pixels[0], &data[0], 4 * width * height)

cdef bint save_copy(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate RGBA-ordered 8888 pixels."""
    memcpy(&data[0], &pixels[0], 4 * width * height)


cdef bint load_bgra8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Load BGRA format images."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + A] = data[4 * offset + 3]


cdef bint save_bgra8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate BGRA format images."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + B]
        data[4 * offset + 1] = pixels[4 * offset + G]
        data[4 * offset + 2] = pixels[4 * offset + R]
        data[4 * offset + 3] = pixels[4 * offset + A]


# This is totally the wrong order, but it's how it's actually ordered.
cdef bint load_argb8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """This is toally wrong - it's actually in GBAR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 3]
        pixels[4 * offset + G] = data[4 * offset + 0]
        pixels[4 * offset + B] = data[4 * offset + 1]
        pixels[4 * offset + A] = data[4 * offset + 2]


cdef bint save_argb8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """This is toally wrong - it's actually in GBAR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + G]
        data[4 * offset + 1] = pixels[4 * offset + B]
        data[4 * offset + 2] = pixels[4 * offset + A]
        data[4 * offset + 3] = pixels[4 * offset + R]


cdef bint load_abgr8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 3]
        pixels[4 * offset + G] = data[4 * offset + 2]
        pixels[4 * offset + B] = data[4 * offset + 1]
        pixels[4 * offset + A] = data[4 * offset + 0]


cdef bint save_abgr8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate ABGR-ordered data."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + A]
        data[4 * offset + 1] = pixels[4 * offset + B]
        data[4 * offset + 2] = pixels[4 * offset + G]
        data[4 * offset + 3] = pixels[4 * offset + R]


cdef bint load_rgb888(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 0]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 2]
        pixels[4 * offset + A] = 255


cdef bint save_rgb888(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate RGB-format data, discarding alpha."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[3 * offset + 0] = pixels[4 * offset + R]
        data[3 * offset + 1] = pixels[4 * offset + G]
        data[3 * offset + 2] = pixels[4 * offset + B]


cdef bint load_bgr888(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[3 * offset + 2]
        pixels[4 * offset + G] = data[3 * offset + 1]
        pixels[4 * offset + B] = data[3 * offset + 0]
        pixels[4 * offset + A] = 255


cdef bint save_bgr888(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate BGR-format data, discarding alpha."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[3 * offset + 0] = pixels[4 * offset + B]
        data[3 * offset + 1] = pixels[4 * offset + G]
        data[3 * offset + 2] = pixels[4 * offset + R]


cdef bint load_bgrx8888(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Strange - skip byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4 * offset + R] = data[4 * offset + 2]
        pixels[4 * offset + G] = data[4 * offset + 1]
        pixels[4 * offset + B] = data[4 * offset + 0]
        pixels[4 * offset + A] = 255


cdef bint save_bgrx8888(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate BGR-format data, with an extra ignored byte."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[4 * offset + 0] = pixels[4 * offset + B]
        data[4 * offset + 1] = pixels[4 * offset + G]
        data[4 * offset + 2] = pixels[4 * offset + R]
        data[4 * offset + 3] = 0


cdef bint load_rgb565(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """RGB format, packed into 2 bytes by dropping LSBs."""
    cdef Py_ssize_t offset
    cdef RGB col
    for offset in prange(width * height, nogil=True, schedule='static'):
        col = decomp565(data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset + R] = col.r
        pixels[4 * offset + G] = col.g
        pixels[4 * offset + B] = col.b
        pixels[4 * offset + A] = 255


cdef bint save_rgb565(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate 565-format data, in RGB order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + R],
            pixels[4 * offset + G],
            pixels[4 * offset + B],
        )


cdef bint load_bgr565(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """BGR format, packed into 2 bytes by dropping LSBs."""
    cdef Py_ssize_t offset
    cdef RGB col
    for offset in prange(width * height, nogil=True, schedule='static'):
        col = decomp565(data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset + R] = col.b
        pixels[4 * offset + G] = col.g
        pixels[4 * offset + B] = col.r
        pixels[4 * offset + A] = 255


cdef bint save_bgr565(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate 565-format data, in BGR order."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + B],
            pixels[4 * offset + G],
            pixels[4 * offset + R],
        )


cdef bint load_bgra4444(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """BGRA format, only upper 4 bits. Bottom half is a copy of the top."""
    cdef Py_ssize_t offset
    cdef fastbyte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + G] = (a & 0b11110000) | (a & 0b11110000) >> 4
        pixels[4 * offset + B] = (a & 0b00001111) | (a & 0b00001111) << 4
        pixels[4 * offset + R] = (b & 0b00001111) | (b & 0b00001111) << 4
        pixels[4 * offset + A] = (b & 0b11110000) | (b & 0b11110000) >> 4


cdef bint save_bgra4444(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate BGRA format images, using only 4 bits each."""
    cdef Py_ssize_t offset
    cdef fastbyte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2 * offset + 0] = (pixels[4 * offset + G] & 0b11110000) | (pixels[4 * offset + B] >> 4)
        data[2 * offset + 1] = (pixels[4 * offset + A] & 0b11110000) | (pixels[4 * offset + R] >> 4)


cdef bint load_bgra5551(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """BGRA format, 5 bits per color plus 1 bit of alpha."""
    cdef Py_ssize_t offset
    cdef fastbyte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + R] = upsample(5, (b & 0b01111100) << 1)
        pixels[4 * offset + G] = upsample(5, (a & 0b11100000) >> 2 | (b & 0b00000011) << 6)
        pixels[4 * offset + B] = upsample(5, (a & 0b00011111) << 3)
        pixels[4 * offset + A] = 255 if b & 0b10000000 else 0


cdef bint save_bgra5551(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate BGRA format images, using 5 bits for color and 1 for alpha."""
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b, a
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4 * offset + R]
        g = pixels[4 * offset + G]
        b = pixels[4 * offset + B]
        a = pixels[4 * offset + A]
        #BBBBBGGG  GGRRRRRA
        data[2 * offset + 0] = ((g << 2) & 0b11100000) | (b >> 3)
        data[2 * offset + 1] = (a & 0b10000000) | ((r >> 1) & 0b01111100) | (g >> 6)


cdef bint load_bgrx5551(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """BGR format, 5 bits per color, alpha ignored."""
    cdef Py_ssize_t offset
    cdef fastbyte a, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset + R] = upsample(5, (b & 0b01111100) << 1)
        pixels[4 * offset + G] = upsample(5, (a & 0b11100000) >> 2 | (b & 0b00000011) << 6)
        pixels[4 * offset + B] = upsample(5, (a & 0b00011111) << 3)
        pixels[4 * offset + A] = 255


cdef bint save_bgrx5551(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate BGR format images, using 5 bits for color and 1 spare bit."""
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4 * offset + R]
        g = pixels[4 * offset + G]
        b = pixels[4 * offset + B]
        #BBBBBGGG  GGRRRRRX
        data[2 * offset + 0] = ((g << 2) & 0b11100000) | (b >> 3)
        data[2 * offset + 1] = ((r >> 1) & 0b01111100) | (g >> 6)


cdef bint load_i8(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """I8 format, R=G=B"""
    cdef Py_ssize_t offset
    cdef fastbyte color
    for offset in prange(width * height, nogil=True, schedule='static'):
        color = data[offset]
        pixels[4*offset + R] = color
        pixels[4*offset + G] = color
        pixels[4*offset + B] = color
        pixels[4*offset + A] = 255


cdef bint save_i8(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save in greyscale."""
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4*offset + R]
        g = pixels[4*offset + G]
        b = pixels[4*offset + B]
        # Only need to compute 3*255 / 3.
        data[offset] = <byte>((r + g + b) // <uint_fast16_t>3)


cdef bint load_ia88(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """I8 format, R=G=B + A"""
    cdef Py_ssize_t offset
    cdef fastbyte color
    for offset in prange(width * height, nogil=True, schedule='static'):
        color = data[2*offset]
        pixels[4*offset + R] = color
        pixels[4*offset + G] = color
        pixels[4*offset + B] = color
        pixels[4*offset+3] = data[2*offset+1]


cdef bint save_ia88(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save in greyscale, with alpha."""
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        r = pixels[4*offset + R]
        g = pixels[4*offset + G]
        b = pixels[4*offset + B]
        data[2 * offset + 0] = <byte>((r + g + b) // <uint_fast16_t>3)
        data[2 * offset + 1] = pixels[4*offset + A]


# ImageFormats.P8 is not implemented by Valve either.

cdef bint load_a8(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Single alpha bytes."""
    cdef Py_ssize_t offset
    # Set RGB to zero in bulk, instead of doing it in the loop.
    memset(&pixels[0], 0, width * height)
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + A] = data[offset]


cdef bint save_a8(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save just the alpha channel."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[offset] = pixels[4*offset + A]


cdef bint load_uv88(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """UV-only, which is mapped to RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        pixels[4*offset + R] = data[2*offset]
        pixels[4*offset + G] = data[2*offset+1]
        pixels[4*offset + B] = 0
        pixels[4*offset + A] = 255


cdef bint save_uv88(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate UV-format data, using RG."""
    cdef Py_ssize_t offset
    for offset in prange(width * height, nogil=True, schedule='static'):
        data[2*offset + 0] = pixels[4*offset + R]
        data[2*offset + 1] = pixels[4*offset + G]


cdef bint load_rgb888_bluescreen(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """RGB format, with 'bluescreen' mode for alpha.

    Pure blue pixels are transparent.
    """
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b
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


cdef bint save_rgb888_bluescreen(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Generate RGB format, using pure blue for transparent pixels."""
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b
    for offset in prange(width * height, nogil=True, schedule='static'):
        if pixels[4*offset + A] < 128:
            data[3 * offset + 0] = 0
            data[3 * offset + 1] = 0
            data[3 * offset + 2] = 255
        else:
            data[3 * offset + 0] = pixels[4*offset + R]
            data[3 * offset + 1] = pixels[4*offset + G]
            data[3 * offset + 2] = pixels[4*offset + B]


cdef bint load_bgr888_bluescreen(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """BGR format, with 'bluescreen' mode for alpha.

    Pure blue pixels are transparent.
    """
    cdef Py_ssize_t offset
    cdef fastbyte r, g, b
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


cdef bint save_bgr888_bluescreen(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
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


cdef bint load_dxt1(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Load compressed DXT1 data."""
    cdef Py_ssize_t offset
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in prange(width * height, nogil=True, schedule='static'):
            pixels[4 * offset] = 0
            pixels[4 * offset + 1] = 0
            pixels[4 * offset + 2] = 0
            pixels[4 * offset + 2] = 0xFF
    else:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt1 | kForceOpaque)


cdef bint save_dxt1(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save compressed DXT1 data."""
    if width >= 4 and height >= 4:
        # DXT format must be 4x4 at minimum. So just skip if not.
        CompressImage(&pixels[0], width, height, &data[0], kDxt1 | kForceOpaque, NULL)


cdef bint load_dxt1_alpha(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Load compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    cdef Py_ssize_t offset
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in prange(width * height, nogil=True, schedule='static'):
            pixels[4 * offset] = 0
            pixels[4 * offset + 1] = 0
            pixels[4 * offset + 2] = 0
            pixels[4 * offset + 2] = 0xFF
    else:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt1)


cdef bint save_dxt1_alpha(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    if width >= 4 and height >= 4:
        # DXT format must be 4x4 at minimum. So just skip if not.
        CompressImage(&pixels[0], width, height, &data[0], kDxt1, NULL)


cdef bint load_dxt3(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Load compressed DXT3 data."""
    cdef Py_ssize_t offset
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in range(0, 4 * width * height, 4):
            pixels[offset] = 0
            pixels[offset + 1] = 0
            pixels[offset + 2] = 0
            pixels[offset + 2] = 0xFF
    else:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt3)


cdef bint save_dxt3(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save compressed DXT3 data."""
    if width >= 4 and height >= 4:
        # DXT format must be 4x4 at minimum. So just skip if not.
        CompressImage(&pixels[0], width, height, &data[0], kDxt3, NULL)


cdef bint load_dxt5(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Load compressed DXT5 data."""
    cdef Py_ssize_t offset
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in prange(width * height, nogil=True, schedule='static'):
            pixels[4 * offset] = 0
            pixels[4 * offset + 1] = 0
            pixels[4 * offset + 2] = 0
            pixels[4 * offset + 2] = 0xFF
    else:
        DecompressImage(&pixels[0], width, height, &data[0], kDxt5)


cdef bint save_dxt5(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Load compressed DXT5 data."""
    if width >= 4 and height >= 4:
        # DXT format must be 4x4 at minimum. So just skip if not.
        CompressImage(&pixels[0], width, height, &data[0], kDxt5, NULL)


cdef bint load_ati2n(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil:
    """Load 'ATI2N' format data, also known as BC5.

    This uses two copies of the DXT5 alpha block for data.
    """
    cdef Py_ssize_t offset
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in prange(width * height, nogil=True, schedule='static'):
            pixels[4 * offset] = 0
            pixels[4 * offset + 1] = 0
            pixels[4 * offset + 2] = 0
            pixels[4 * offset + 2] = 0xFF
    else:
        DecompressImage(&pixels[0], width, height, &data[0], kBc5)


cdef bint save_ati2n(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil:
    """Save 'ATI2N' format data, also known as BC5.

    This uses two copies of the DXT5 alpha block for data.
    """
    if width >= 4 and height >= 4:
        # DXT format must be 4x4 at minimum. So just skip if not.
        CompressImage(&pixels[0], width, height, &data[0], kBc5, NULL)


# Functions for computing the required data size.
cdef Py_ssize_t size_8888(uint width, uint height) noexcept:
    return 4 * width * height

cdef Py_ssize_t size_888(uint width, uint height) noexcept:
    return 3 * width * height

cdef Py_ssize_t size_88(uint width, uint height) noexcept:
    return 2 * width * height

cdef Py_ssize_t size_8(uint width, uint height) noexcept:
    return 1 * width * height

cdef Py_ssize_t size_565(uint width, uint height) noexcept:
    return 2 * width * height

cdef Py_ssize_t size_5551(uint width, uint height) noexcept:
    return 2 * width * height

cdef Py_ssize_t size_4444(uint width, uint height) noexcept:
    return 2 * width * height

cdef Py_ssize_t size_dxt1(uint width, uint height) noexcept:
    return size_dxt_common(width, height, 8)

cdef Py_ssize_t size_dxt3(uint width, uint height) noexcept:
    return size_dxt_common(width, height, 16)

cdef Py_ssize_t size_dxt5(uint width, uint height) noexcept:
    return size_dxt_common(width, height, 16)

cdef Py_ssize_t size_ati1n(uint width, uint height) noexcept:
    return size_dxt_common(width, height, 8)

cdef Py_ssize_t size_ati2n(uint width, uint height) noexcept:
    return size_dxt_common(width, height, 16)


cdef Py_ssize_t size_dxt_common(uint width, uint height, uint per_block) noexcept:
    cdef int block_w, block_h
    block_w = width // 4
    if width % 4 != 0:
        block_w += 1
    block_h = height // 4
    if height % 4:
        block_h += 1
    return per_block * block_w * block_h


# Use a structure to match format names to the functions.
# This way they can all be cdef, and not have duplicate object conversion
# code.
ctypedef struct Format:
    char *name
    Py_ssize_t (*size)(uint width, uint height) noexcept
    bint (*load)(byte[::1] pixels, const byte[::1] data, uint width, uint height) noexcept nogil
    bint (*save)(const byte[::1] pixels, byte[::1] data, uint width, uint height) noexcept nogil


cdef Format[30] FORMATS
# Assign directly to each, so Cython doesn't write these to a temp array first
# in case an exception occurs.
FORMATS[ 0] = Format("RGBA8888", &size_8888, &load_copy, &save_copy)
FORMATS[ 1] = Format("ABGR8888", &size_8888, &load_abgr8888, &save_abgr8888)
FORMATS[ 2] = Format("RGB888", &size_888, &load_rgb888, &save_rgb888)
FORMATS[ 3] = Format("BGR888", &size_888, &load_bgr888, &save_bgr888)
FORMATS[ 4] = Format("RGB565", &size_565, &load_rgb565, &save_rgb565)
FORMATS[ 5] = Format("I8", &size_8, &load_i8, &save_i8)
FORMATS[ 6] = Format("IA88", &size_88, &load_ia88, &save_ia88)
FORMATS[ 7] = Format("P8", NULL, NULL, NULL)  # Never implemented by Valve.
FORMATS[ 8] = Format("A8",  &size_8, &load_a8, &save_a8)
FORMATS[ 9] = Format("RGB888_BLUESCREEN", &size_888, &load_rgb888_bluescreen, &save_rgb888_bluescreen)
FORMATS[10] = Format("BGR888_BLUESCREEN", &size_888, &load_bgr888_bluescreen, &save_bgr888_bluescreen)
FORMATS[11] = Format("ARGB8888", &size_8888, &load_argb8888, &save_argb8888)
FORMATS[12] = Format("BGRA8888", &size_8888, &load_bgra8888, &save_bgra8888)
FORMATS[13] = Format("DXT1", &size_dxt1, &load_dxt1, &save_dxt1)
FORMATS[14] = Format("DXT3", &size_dxt3, &load_dxt3, &save_dxt3)
FORMATS[15] = Format("DXT5", &size_dxt5, &load_dxt5, &save_dxt5)
FORMATS[16] = Format("BGRX8888", &size_8888, &load_bgrx8888, &save_bgrx8888)
FORMATS[17] = Format("BGR565", &size_565, &load_bgr565, &save_bgr565)
FORMATS[18] = Format("BGRX5551", &size_5551, &load_bgrx5551, &save_bgrx5551)
FORMATS[19] = Format("BGRA4444", &size_4444, &load_bgra4444, &save_bgra4444)
FORMATS[20] = Format("DXT1_ONEBITALPHA", &size_dxt1, &load_dxt1_alpha, &save_dxt1_alpha)
FORMATS[21] = Format("BGRA5551", &size_5551, &load_bgra5551, &save_bgra5551)
FORMATS[22] = Format("UV88", &size_88, &load_uv88, &save_uv88)
FORMATS[23] = Format("UVWQ8888", &size_8888, &load_copy, &save_copy)

# Don't do the high-def 16-bit resolutions.
FORMATS[24] = Format("RGBA16161616F", NULL, NULL, NULL)
FORMATS[25] = Format("RGBA16161616", NULL, NULL, NULL)

FORMATS[26] = Format("UVLX8888", &size_8888, &load_copy, &save_copy)

# This doesn't match the actual engine struct, just the order of
# the Python enum.
FORMATS[27] = Format("NONE", NULL, NULL, NULL)
FORMATS[28] = Format("ATI1N", &size_ati1n, NULL, NULL)
FORMATS[29] = Format("ATI2N", &size_ati2n, &load_ati2n, &save_ati2n)


def init(formats: 'srctools.vtf.ImageFormats') -> None:
    """Verify that the Python enum matches our array of functions."""
    cdef size_t index
    cdef bytes name
    for fmt in formats:
        index = fmt.ind
        assert 0 <= index < (sizeof(FORMATS) // sizeof(Format))
        assert strcmp((<str ?>fmt.name).encode('ascii'), FORMATS[index].name) == 0, f'{fmt} != {FORMATS[index].name.decode("ascii")}'
        if FORMATS[index].load != NULL or FORMATS[index].save != NULL:
            assert FORMATS[index].size != NULL, FORMATS[index].name.decode("ascii")


def load(object fmt: 'srctools.vtf.ImageFormats', byte[::1] pixels, const byte[::1] data, uint width, uint height) -> None:
    """Load pixels from data in the given format."""
    cdef Py_ssize_t data_size = 4 * width * height
    if data_size != len(pixels):
        raise BufferError(f"Incorrect pixel array size. Expected {data_size} bytes, got {len(pixels)} bytes.")

    cdef size_t index = fmt.ind

    # print("Index: ", index, "< ", (sizeof(FORMATS) // sizeof(Format)))
    if 0 <= index < (sizeof(FORMATS) // sizeof(Format)) and FORMATS[index].load != NULL:
        if FORMATS[index].size == NULL:
            raise RuntimeError(fmt)
        data_size = FORMATS[index].size(width, height)
        if data_size != len(data):
            raise BufferError(f"Incorrect data block size. Expected {data_size} bytes, got {len(data)} bytes.")

        with nogil:
            FORMATS[index].load(pixels, data, width, height)
    else:
        raise NotImplementedError(f"Loading {fmt.name} not implemented!")


def save(object fmt: 'srctools.vtf.ImageFormats', const byte[::1] pixels, byte[::1] data, uint width, uint height) -> None:
    cdef Py_ssize_t data_size = 4 * width * height
    if data_size != len(pixels):
        raise BufferError(f"Incorrect pixel array size. Expected {data_size} bytes, got {len(pixels)} bytes.")

    cdef size_t index = fmt.ind
    if 0 <= index < (sizeof(FORMATS) // sizeof(Format)) and FORMATS[index].save != NULL:
        if FORMATS[index].size == NULL:
            raise RuntimeError(fmt)
        data_size = FORMATS[index].size(width, height)
        if data_size != len(data):
            raise BufferError(f"Incorrect data block size. Expected {data_size} bytes, got {len(data)} bytes.")

        with nogil:
            FORMATS[index].save(pixels, data, width, height)
    else:
        raise NotImplementedError(f"Saving {fmt.name} not implemented!")
